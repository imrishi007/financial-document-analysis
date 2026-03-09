"""Extract embeddings from all pretrained modality encoders (V2).

Runs each Phase-4/5 model in inference mode, collects the penultimate-layer
representations for every (ticker, date) in the price dataset, and saves
them as a single ``fusion_embeddings.pt`` file.

V2 changes:
- Removed news modality completely
- Fixed surprise features (5-d, no leakage)
- Added macro MLP embeddings (32-d)
- Primary target: 60-day direction
- AMP + GPU optimization throughout

GPU optimization
----------------
* FinBERT is cast to **fp16** for doc extraction.
* ``torch.amp.autocast`` used where applicable.
* Large batch sizes fill 3-3.5 GB of GPU memory during extraction.
* ``torch.no_grad()`` throughout -- no gradient memory overhead.
"""

from __future__ import annotations

import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# ------------------------------------------------------------------
# Local imports
# ------------------------------------------------------------------
from src.data.graph_dataset import GraphSnapshotDataset, build_graph_snapshots
from src.data.graph_utils import load_graph, make_bidirectional
from src.data.preprocessing import fit_scaler
from src.data.price_dataset import (
    ENGINEERED_FEATURES,
    PriceWindowDataset,
    load_price_csv_dir,
    prepare_price_features,
)
from src.models.gat_model import GraphEnhancedModel
from src.models.price_model import PriceDirectionModel
from src.utils.gpu import setup_gpu, log_gpu_usage


# ======================================================================
# 1. Price embeddings  (256-d per sample)
# ======================================================================


def extract_price_embeddings(
    price_dataset: PriceWindowDataset,
    model_path: str | Path = "models/price_model_best.pt",
    device: str = "cuda",
    batch_size: int = 512,
) -> dict[tuple[str, str], np.ndarray]:
    """Run the trained PriceDirectionModel encoder (conv+BiLSTM, no head).

    Returns {(ticker, date_str): np.ndarray of shape (256,)}.
    """
    model = PriceDirectionModel(num_features=len(ENGINEERED_FEATURES))
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval().to(device)

    embeddings: dict[tuple[str, str], np.ndarray] = {}
    loader = DataLoader(
        price_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=4,
        persistent_workers=True,
    )

    with torch.no_grad():
        for batch in loader:
            x = batch["features"].to(device, non_blocking=True)
            tickers = batch["ticker"]
            dates = batch["date"]

            with torch.amp.autocast("cuda"):
                h = x.transpose(1, 2)
                h = model.conv(h)
                h = h.transpose(1, 2)
                h, _ = model.lstm(h)
                emb = h[:, -1, :].float().cpu().numpy()

            for i in range(len(tickers)):
                embeddings[(tickers[i], dates[i])] = emb[i]

    del model
    torch.cuda.empty_cache()
    return embeddings


# ======================================================================
# 2. GAT embeddings  (256-d per company per date)
# ======================================================================


def extract_gat_embeddings_v2(
    graph_dataset: GraphSnapshotDataset,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    tickers: list[str],
    model_path: str | Path = "models/graph_model_best.pt",
    device: str = "cuda",
    batch_size: int = 64,
) -> dict[tuple[str, str], np.ndarray]:
    """Extract GAT embeddings with proper date tracking."""
    model = GraphEnhancedModel(num_features=len(ENGINEERED_FEATURES))
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval().to(device)

    ei = edge_index.to(device)
    ew = edge_weight.to(device)

    embeddings: dict[tuple[str, str], np.ndarray] = {}

    with torch.no_grad():
        for si in range(len(graph_dataset)):
            sample = graph_dataset[si]
            x = sample["features"].to(device)
            m = sample["mask"]

            with torch.amp.autocast("cuda"):
                h = model.encode_price(x)
                h = model.node_proj(h)
                h1 = F.elu(model.gat1(h, ei, ew))
                h = model.gat_norm1(h + h1)
                h2 = F.elu(model.gat2(h, ei, ew))
                h = model.gat_norm2(h + h2)

            emb = h.float().cpu().numpy()
            date_str = graph_dataset._snapshots[si]["date"]

            for j in range(len(tickers)):
                if m[j]:
                    embeddings[(tickers[j], date_str)] = emb[j]

    del model
    torch.cuda.empty_cache()
    return embeddings


# ======================================================================
# 3. Document embeddings  (768-d per filing)
# ======================================================================


def extract_doc_embeddings(
    processed_dir: str | Path = "data/processed",
    model_path: str | Path = "models/document_model_best.pt",
    backbone: str = "ProsusAI/finbert",
    device: str = "cuda",
    chunk_batch_size: int = 64,
) -> dict[tuple[str, int], np.ndarray]:
    """Extract attention-pooled FinBERT embeddings for each 10-K filing.

    Returns {(ticker, year): np.ndarray of shape (768,)}.
    Uses fp16 for FinBERT to maximize GPU throughput (~1.5 GB peak).
    """
    from transformers import AutoTokenizer

    from src.data.document_dataset import DocumentChunkDataset
    from src.models.document_model import DocumentDirectionModel

    tokenizer = AutoTokenizer.from_pretrained(backbone)
    doc_ds = DocumentChunkDataset(processed_dir, tokenizer, max_chunks=64)

    model = DocumentDirectionModel(backbone_name=backbone)
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Cast encoder to fp16 for speed + memory savings
    model.encoder = model.encoder.half().to(device)
    model.pool = model.pool.to(device)

    embeddings: dict[tuple[str, int], np.ndarray] = {}

    with torch.no_grad():
        for idx in range(len(doc_ds)):
            sample = doc_ds[idx]
            ticker = sample["ticker"]
            year = sample["year"]
            n_chunks = sample["num_chunks"]
            ids = sample["input_ids"][:n_chunks]
            mask = sample["attention_mask"][:n_chunks]

            cls_list = []
            for start in range(0, n_chunks, chunk_batch_size):
                end = min(start + chunk_batch_size, n_chunks)
                mini_ids = ids[start:end].to(device)
                mini_mask = mask[start:end].to(device)

                with torch.amp.autocast("cuda"):
                    enc_out = model.encoder(
                        input_ids=mini_ids, attention_mask=mini_mask
                    )
                    cls_list.append(enc_out.last_hidden_state[:, 0, :].float())

                del mini_ids, mini_mask, enc_out

            chunk_cls = torch.cat(cls_list, dim=0)
            embedding = chunk_cls.mean(dim=0).cpu().numpy()
            embeddings[(ticker, year)] = embedding

    del model
    torch.cuda.empty_cache()
    log_gpu_usage("  After doc extraction: ")
    return embeddings


# ======================================================================
# 4. Macro MLP embeddings  (32-d per date)
# ======================================================================


def extract_macro_embeddings(
    macro_features_df: pd.DataFrame,
    model_path: str | Path = "models/macro_model_best.pt",
    device: str = "cuda",
) -> dict[str, np.ndarray]:
    """Extract macro embeddings for each date.

    If no trained macro model exists, returns raw features zero-padded to 32-d.

    Returns {date_str: np.ndarray of shape (32,)}.
    """
    from src.data.macro_features import MACRO_FEATURE_NAMES

    macro_features_df = macro_features_df.copy()
    macro_features_df.index = pd.to_datetime(macro_features_df.index)

    embeddings: dict[str, np.ndarray] = {}

    model_path = Path(model_path)
    if model_path.exists():
        from src.models.macro_model import MacroStateModel

        model = MacroStateModel(input_dim=12, hidden_dim=64, output_dim=32)
        ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval().to(device)

        with torch.no_grad():
            for date_idx in macro_features_df.index:
                date_str = str(date_idx.date())
                feat = macro_features_df.loc[date_idx, MACRO_FEATURE_NAMES].values
                if np.any(np.isnan(feat)):
                    continue
                x = torch.tensor(feat, dtype=torch.float32).unsqueeze(0).to(device)
                with torch.amp.autocast("cuda"):
                    emb = model.encode(x)
                embeddings[date_str] = emb.float().cpu().numpy()[0]

        del model
        torch.cuda.empty_cache()
    else:
        # No trained model -- use zero-padded raw features projected to 32-d
        for date_idx in macro_features_df.index:
            date_str = str(date_idx.date())
            feat = macro_features_df.loc[date_idx, MACRO_FEATURE_NAMES].values
            if np.any(np.isnan(feat)):
                continue
            padded = np.zeros(32, dtype=np.float32)
            padded[:12] = feat.astype(np.float32)
            embeddings[date_str] = padded

    return embeddings


# ======================================================================
# 5. Surprise features  (5-d per sample, NO LEAKAGE)
# ======================================================================


def build_surprise_features(
    targets_dir: str | Path = "data/targets",
    lookback_days: int = 90,
) -> dict[tuple[str, str], np.ndarray]:
    """Build FIXED surprise features: [surprise_pct, surprise_magnitude_normalized,
    recency_decay, trailing_3q_beat_rate, eps_revision_direction].

    NO is_beat or is_miss features -- these encode the target and cause leakage.

    For each (ticker, date), find earnings events within lookback_days.
    Returns {(ticker, date_str): np.array of shape (5,)}.
    """
    surp_df = pd.read_csv(Path(targets_dir) / "fundamental_surprise_targets.csv")
    surp_df["date"] = pd.to_datetime(surp_df["date"])

    events = surp_df[surp_df["has_earnings_event"] == 1].copy()
    events = events.sort_values(["ticker", "date"])

    ticker_events: dict[str, list[dict]] = defaultdict(list)
    for _, row in events.iterrows():
        surprise_pct = float(row.get("surprise_pct", 0.0))
        is_beat = 1.0 if row["surprise_id"] == 1 else 0.0
        ticker_events[row["ticker"]].append(
            {
                "date": row["date"],
                "surprise_pct": surprise_pct,
                "is_beat": is_beat,
            }
        )

    ticker_trailing: dict[str, list[dict]] = {}
    for ticker, evts in ticker_events.items():
        trailing = []
        all_pcts = []
        for i, evt in enumerate(evts):
            past_evts = evts[max(0, i - 3) : i]
            beat_rate = (
                sum(e["is_beat"] for e in past_evts) / len(past_evts)
                if past_evts
                else 0.5
            )
            all_pcts.append(evt["surprise_pct"])
            trailing_pcts = all_pcts[max(0, len(all_pcts) - 8) :]
            pct_mean = np.mean(trailing_pcts) if trailing_pcts else 0.0
            pct_std = np.std(trailing_pcts) if len(trailing_pcts) > 1 else 1.0

            trailing.append(
                {
                    **evt,
                    "beat_rate_3q": beat_rate,
                    "pct_mean": pct_mean,
                    "pct_std": max(pct_std, 1e-6),
                }
            )
        ticker_trailing[ticker] = trailing

    all_dates = surp_df[["ticker", "date"]].drop_duplicates()
    features: dict[tuple[str, str], np.ndarray] = {}

    for _, row in all_dates.iterrows():
        ticker = row["ticker"]
        dt = row["date"]
        date_str = str(dt.date())

        evts = ticker_trailing.get(ticker, [])
        best = None
        for evt in reversed(evts):
            if evt["date"] <= dt:
                days_since = (dt - evt["date"]).days
                if days_since <= lookback_days:
                    surprise_pct = float(evt["surprise_pct"])
                    surprise_mag_norm = (surprise_pct - evt["pct_mean"]) / evt[
                        "pct_std"
                    ]
                    recency_decay = float(np.exp(-days_since / 90.0))
                    beat_rate_3q = float(evt["beat_rate_3q"])
                    eps_revision = 0.0

                    best = np.array(
                        [
                            surprise_pct,
                            surprise_mag_norm,
                            recency_decay,
                            beat_rate_3q,
                            eps_revision,
                        ],
                        dtype=np.float32,
                    )
                break

        if best is not None:
            features[(ticker, date_str)] = best

    return features


# ======================================================================
# 6. Master extraction + alignment
# ======================================================================


def extract_all_embeddings(
    prices_dir: str | Path = "data/raw/prices",
    targets_dir: str | Path = "data/targets",
    processed_dir: str | Path = "data/processed",
    macro_csv: str | Path = "data/raw/macro/macro_prices.csv",
    save_path: str | Path = "data/embeddings/fusion_embeddings.pt",
    device: str = "cuda",
    verbose: bool = True,
) -> dict[str, Any]:
    """Extract embeddings from all pretrained models and save aligned dataset.

    V2: 4 modalities (no news), fixed surprise features, 60-day direction.
    """
    t0 = time.time()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    gpu_device = setup_gpu(verbose=verbose)
    device = str(gpu_device)

    # Load price data + normalize
    if verbose:
        print("Loading price data and normalizing...")
    prices = load_price_csv_dir(prices_dir)
    price_df = prepare_price_features(prices)

    targets_df = pd.read_csv(Path(targets_dir) / "direction_labels_multi_horizon.csv")
    targets_df["date"] = pd.to_datetime(targets_df["date"])
    price_df["date"] = pd.to_datetime(price_df["date"])

    train_mask = price_df["date"] <= "2022-12-31"
    scaler = fit_scaler(price_df.loc[train_mask], ENGINEERED_FEATURES)
    price_df_norm = scaler.transform(price_df.copy(), ENGINEERED_FEATURES)

    price_ds = PriceWindowDataset(
        price_df_norm,
        targets_df,
        window_size=60,
        feature_cols=ENGINEERED_FEATURES,
    )
    if verbose:
        print(f"  Price dataset: {len(price_ds)} samples")

    # A) Price embeddings
    if verbose:
        print("\n[1/5] Extracting price embeddings (batch=512)...")
    t1 = time.time()
    price_embs = extract_price_embeddings(price_ds, device=device, batch_size=512)
    if verbose:
        print(f"  Done -- {len(price_embs)} embeddings in {time.time()-t1:.1f}s")
        log_gpu_usage("  ")

    # B) GAT embeddings
    if verbose:
        print("\n[2/5] Extracting GAT embeddings...")
    t2 = time.time()
    graph = load_graph()
    ei, ew, et = make_bidirectional(
        graph["edge_index"], graph["edge_weight"], graph["edge_type"]
    )
    graph_tickers = graph["tickers"]

    gat_price_df, gat_targets_df, _ = build_graph_snapshots(
        prices_dir,
        targets_dir,
        graph_tickers,
        window_size=60,
    )
    gat_price_df["date"] = pd.to_datetime(gat_price_df["date"])
    gat_train_mask = gat_price_df["date"] <= "2022-12-31"
    gat_scaler = fit_scaler(gat_price_df.loc[gat_train_mask], ENGINEERED_FEATURES)
    gat_price_df = gat_scaler.transform(gat_price_df, ENGINEERED_FEATURES)

    graph_ds = GraphSnapshotDataset(
        gat_price_df,
        gat_targets_df,
        graph_tickers,
        window_size=60,
    )
    gat_embs = extract_gat_embeddings_v2(
        graph_ds,
        ei,
        ew,
        graph_tickers,
        device=device,
        batch_size=64,
    )
    if verbose:
        print(f"  Done -- {len(gat_embs)} embeddings in {time.time()-t2:.1f}s")

    # C) Document embeddings
    if verbose:
        print("\n[3/5] Extracting document embeddings (FinBERT fp16, chunks=64)...")
    t3 = time.time()
    doc_embs_raw = extract_doc_embeddings(
        processed_dir,
        device=device,
        chunk_batch_size=64,
    )
    if verbose:
        print(
            f"  Done -- {len(doc_embs_raw)} filing embeddings in {time.time()-t3:.1f}s"
        )

    doc_embs: dict[tuple[str, str], np.ndarray] = {}
    for ticker, date_str in price_embs:
        year = int(date_str[:4])
        filing_year = year - 1
        key = (ticker, filing_year)
        if key in doc_embs_raw:
            doc_embs[(ticker, date_str)] = doc_embs_raw[key]

    # D) Macro embeddings
    if verbose:
        print("\n[4/5] Extracting macro embeddings...")
    t4 = time.time()
    macro_path = Path(macro_csv)
    macro_embs: dict[str, np.ndarray] = {}
    if macro_path.exists():
        from src.data.macro_features import (
            load_macro_data,
            compute_macro_features,
            MacroFeatureScaler,
        )

        raw_macro = load_macro_data(macro_csv)
        macro_feat = compute_macro_features(raw_macro, lag_days=1)
        macro_scaler = MacroFeatureScaler()
        macro_scaler.fit(macro_feat, train_end="2022-12-31")
        macro_feat_norm = macro_scaler.transform(macro_feat)
        macro_embs = extract_macro_embeddings(macro_feat_norm, device=device)
    else:
        if verbose:
            print("  WARNING: No macro data found. Macro modality unavailable.")
    if verbose:
        print(f"  Done -- {len(macro_embs)} macro embeddings in {time.time()-t4:.1f}s")

    # E) Surprise features (FIXED -- no leakage)
    if verbose:
        print("\n[5/5] Building surprise features (FIXED, 5-d, no leakage)...")
    t5 = time.time()
    surprise_feats = build_surprise_features(targets_dir, lookback_days=90)
    if verbose:
        print(f"  Done -- {len(surprise_feats)} features in {time.time()-t5:.1f}s")

    # Align everything
    if verbose:
        print("\nAligning all modalities...")

    vol_df = pd.read_csv(Path(targets_dir) / "volatility_targets.csv")
    vol_df["date"] = pd.to_datetime(vol_df["date"])
    vol_lookup: dict[tuple[str, str], float] = {}
    for _, row in vol_df.iterrows():
        vol_lookup[(row["ticker"], str(row["date"].date()))] = float(
            row["realized_vol_20d_annualized"]
        )

    N = len(price_ds)
    PRICE_DIM, GAT_DIM, DOC_DIM, MACRO_DIM, SURP_DIM = 256, 256, 768, 32, 5

    all_price = np.zeros((N, PRICE_DIM), dtype=np.float32)
    all_gat = np.zeros((N, GAT_DIM), dtype=np.float32)
    all_doc = np.zeros((N, DOC_DIM), dtype=np.float32)
    all_macro = np.zeros((N, MACRO_DIM), dtype=np.float32)
    all_surprise = np.zeros((N, SURP_DIM), dtype=np.float32)
    all_mask = np.zeros((N, 4), dtype=np.float32)  # [price, gat, doc, macro]
    all_dir_label = np.full(N, -1, dtype=np.int64)
    all_vol_target = np.full(N, float("nan"), dtype=np.float32)
    all_tickers: list[str] = []
    all_dates: list[str] = []

    for i in range(N):
        sample = price_ds._samples[i]
        ticker = sample["ticker"]
        date_str = sample["date"]
        key = (ticker, date_str)

        all_tickers.append(ticker)
        all_dates.append(date_str)

        all_dir_label[i] = int(sample.get("direction_60d", -1))

        if key in vol_lookup:
            all_vol_target[i] = vol_lookup[key]

        if key in price_embs:
            all_price[i] = price_embs[key]
            all_mask[i, 0] = 1.0

        if key in gat_embs:
            all_gat[i] = gat_embs[key]
            all_mask[i, 1] = 1.0

        if key in doc_embs:
            all_doc[i] = doc_embs[key]
            all_mask[i, 2] = 1.0

        if date_str in macro_embs:
            all_macro[i] = macro_embs[date_str]
            all_mask[i, 3] = 1.0

        if key in surprise_feats:
            all_surprise[i] = surprise_feats[key]

    # Save
    data = {
        "price_emb": torch.from_numpy(all_price),
        "gat_emb": torch.from_numpy(all_gat),
        "doc_emb": torch.from_numpy(all_doc),
        "macro_emb": torch.from_numpy(all_macro),
        "surprise_feat": torch.from_numpy(all_surprise),
        "modality_mask": torch.from_numpy(all_mask),
        "direction_label": torch.from_numpy(all_dir_label),
        "volatility_target": torch.from_numpy(all_vol_target),
        "tickers": all_tickers,
        "dates": all_dates,
    }
    torch.save(data, save_path)

    summary = {
        "total_samples": N,
        "price_available": int(all_mask[:, 0].sum()),
        "gat_available": int(all_mask[:, 1].sum()),
        "doc_available": int(all_mask[:, 2].sum()),
        "macro_available": int(all_mask[:, 3].sum()),
        "surprise_available": int((all_surprise.sum(axis=1) != 0).sum()),
        "dir_labels_valid": int((all_dir_label >= 0).sum()),
        "vol_targets_valid": int((~np.isnan(all_vol_target)).sum()),
        "time_seconds": time.time() - t0,
        "save_path": str(save_path),
    }

    if verbose:
        print(f"\nExtraction complete -- {N} samples")
        for k, v in summary.items():
            if k != "save_path":
                print(f"  {k}: {v}")
        print(f"  Saved to {save_path}")
        log_gpu_usage("  Final: ")

    return summary
