"""Extract embeddings from all pretrained modality encoders.

Runs each Phase-4/5 model in inference mode, collects the penultimate-layer
representations for every (ticker, date) in the price dataset, and saves
them as a single ``fusion_embeddings.pt`` file that the fusion dataset
can load directly.

GPU optimisation
----------------
* FinBERT is cast to **fp16** (half precision) for doc / news extraction.
* ``torch.amp.autocast`` used where applicable.
* Large batch sizes fill 3–3.5 GB of GPU memory during extraction.
* ``torch.no_grad()`` throughout — no gradient memory overhead.
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
    loader = DataLoader(price_dataset, batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        for batch in loader:
            x = batch["features"].to(device)  # [B, T, F]
            tickers = batch["ticker"]  # list[str]
            dates = batch["date"]  # list[str]

            # Forward through conv + BiLSTM (skip head)
            h = x.transpose(1, 2)  # [B, F, T]
            h = model.conv(h)  # [B, 64, T]
            h = h.transpose(1, 2)  # [B, T, 64]
            h, _ = model.lstm(h)  # [B, T, 256]
            emb = h[:, -1, :].cpu().numpy()  # [B, 256]

            for i in range(len(tickers)):
                embeddings[(tickers[i], dates[i])] = emb[i]

    del model
    torch.cuda.empty_cache()
    return embeddings


# ======================================================================
# 2. GAT embeddings  (256-d per company per date)
# ======================================================================


def extract_gat_embeddings(
    graph_dataset: GraphSnapshotDataset,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    tickers: list[str],
    model_path: str | Path = "models/graph_model_best.pt",
    device: str = "cuda",
    batch_size: int = 64,
) -> dict[tuple[str, str], np.ndarray]:
    """Run the trained GraphEnhancedModel encoder (CNN-LSTM → GAT, no head).

    Returns {(ticker, date_str): np.ndarray of shape (256,)}.
    """
    model = GraphEnhancedModel(num_features=len(ENGINEERED_FEATURES))
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval().to(device)

    ei = edge_index.to(device)
    ew = edge_weight.to(device)

    embeddings: dict[tuple[str, str], np.ndarray] = {}
    loader = DataLoader(graph_dataset, batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        for batch in loader:
            features = batch["features"].to(device)  # [B, N, T, F]
            mask = batch["mask"]  # [B, N]
            B, N, T, Feat = features.shape

            for b in range(B):
                x = features[b]  # [N, T, F]
                m = mask[b]  # [N]

                # Encoder (conv + BiLSTM)
                h = model.encode_price(x)  # [N, 256]
                h = model.node_proj(h)  # [N, gat_in_dim]

                # GAT layer 1
                h1 = F.elu(model.gat1(h, ei, ew))
                h = model.gat_norm1(h + h1)

                # GAT layer 2
                h2 = F.elu(model.gat2(h, ei, ew))
                h = model.gat_norm2(h + h2)  # [N, 256]

                emb = h.cpu().numpy()
                date_str = graph_dataset._snapshots[
                    batch["__index__"][b] if "__index__" in batch else 0
                ]["date"]

                for j in range(N):
                    if m[j]:
                        embeddings[(tickers[j], date_str)] = emb[j]

    del model
    torch.cuda.empty_cache()
    return embeddings


# We need to patch the GraphSnapshotDataset to include index
def _graph_collate_with_idx(dataset, indices):
    """Collate that also returns snapshot indices for date lookup."""
    batch = [dataset[i] for i in indices]
    collated = {}
    for key in batch[0]:
        if isinstance(batch[0][key], torch.Tensor):
            collated[key] = torch.stack([b[key] for b in batch])
        else:
            collated[key] = [b[key] for b in batch]
    collated["__snap_indices__"] = indices
    return collated


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
        for snap_idx in range(0, len(graph_dataset), batch_size):
            end_idx = min(snap_idx + batch_size, len(graph_dataset))
            for si in range(snap_idx, end_idx):
                sample = graph_dataset[si]
                x = sample["features"].unsqueeze(0).to(device)  # [1, N, T, F]
                m = sample["mask"]  # [N]

                x = x.squeeze(0)  # [N, T, F]

                h = model.encode_price(x)
                h = model.node_proj(h)
                h1 = F.elu(model.gat1(h, ei, ew))
                h = model.gat_norm1(h + h1)
                h2 = F.elu(model.gat2(h, ei, ew))
                h = model.gat_norm2(h + h2)

                emb = h.cpu().numpy()
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
    Uses fp16 for FinBERT to maximise GPU throughput (~1.5 GB peak).
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
            ids = sample["input_ids"][:n_chunks]  # [C, S]
            mask = sample["attention_mask"][:n_chunks]  # [C, S]

            # Process chunks in large batches through FinBERT fp16
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

            chunk_cls = torch.cat(cls_list, dim=0)  # [C, 768]
            # Mean-pool across chunks (matches training approach)
            embedding = chunk_cls.mean(dim=0).cpu().numpy()  # (768,)
            embeddings[(ticker, year)] = embedding

    del model
    torch.cuda.empty_cache()
    return embeddings


# ======================================================================
# 4. News embeddings  (512-d per window)
# ======================================================================


def extract_news_embeddings(
    news_csv: str | Path = "data/raw/news/news_articles.csv",
    prices_dir: str | Path = "data/raw/prices",
    targets_dir: str | Path = "data/targets",
    model_path: str | Path = "models/news_model_best.pt",
    backbone: str = "ProsusAI/finbert",
    device: str = "cuda",
    batch_size: int = 16,
) -> dict[tuple[str, str], np.ndarray]:
    """Extract BiGRU temporal embeddings for each news window.

    Returns {(ticker, date_str): np.ndarray of shape (512,)}.
    Uses fp16 for FinBERT article encoding.
    """
    from transformers import AutoTokenizer

    from src.data.news_dataset import NewsWindowDataset, load_news_articles
    from src.models.news_model import NewsTemporalDirectionModel

    news_df = load_news_articles(news_csv)
    direction_df = pd.read_csv(Path(targets_dir) / "direction_labels_multi_horizon.csv")

    dir_dates = direction_df[["ticker", "date"]].copy()
    dir_dates["date"] = pd.to_datetime(dir_dates["date"])
    news_start = news_df["date"].min()
    anchors = dir_dates[dir_dates["date"] >= news_start].copy()

    tokenizer = AutoTokenizer.from_pretrained(backbone)
    news_ds = NewsWindowDataset(
        news_df,
        anchors,
        tokenizer,
        max_articles=16,
        max_length=128,
        window_days=7,
    )

    model = NewsTemporalDirectionModel(backbone_name=backbone)
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # fp16 FinBERT encoder
    model.encoder = model.encoder.half().to(device)
    model.temporal = model.temporal.to(device)

    embeddings: dict[tuple[str, str], np.ndarray] = {}

    with torch.no_grad():
        for idx in range(len(news_ds)):
            sample = news_ds[idx]
            ticker = sample["ticker"]
            date_str = sample["date"]
            n_articles = sample["num_articles"]

            if n_articles == 0:
                continue

            ids = sample["input_ids"][:n_articles].unsqueeze(0).to(device)  # [1, A, S]
            mask = sample["attention_mask"][:n_articles].unsqueeze(0).to(device)

            with torch.amp.autocast("cuda"):
                article_emb = model.encode_articles(ids, mask)  # [1, A, 768]

            article_emb = article_emb.float()
            temporal_out, _ = model.temporal(article_emb)  # [1, A, 512]
            emb = temporal_out[:, -1, :].cpu().numpy()  # [1, 512]
            embeddings[(ticker, date_str)] = emb[0]

    del model
    torch.cuda.empty_cache()
    return embeddings


# ======================================================================
# 5. Surprise features  (3-d per sample)
# ======================================================================


def build_surprise_features(
    targets_dir: str | Path = "data/targets",
    lookback_days: int = 90,
) -> dict[tuple[str, str], np.ndarray]:
    """Build surprise features: [is_beat, is_miss, recency].

    For each (ticker, date), find the most recent earnings event within
    lookback_days. Returns {(ticker, date_str): np.array([is_beat, is_miss,
    days_since / lookback_days])}.
    """
    surp_df = pd.read_csv(Path(targets_dir) / "fundamental_surprise_targets.csv")
    surp_df["date"] = pd.to_datetime(surp_df["date"])

    # Get only actual earnings events
    events = surp_df[surp_df["has_earnings_event"] == 1].copy()
    events = events.sort_values(["ticker", "date"])

    # Build per-ticker sorted event list
    ticker_events: dict[str, list[tuple]] = defaultdict(list)
    for _, row in events.iterrows():
        is_beat = 1.0 if row["surprise_id"] == 1 else 0.0
        is_miss = 1.0 if row["surprise_id"] == 0 else 0.0
        ticker_events[row["ticker"]].append((row["date"], is_beat, is_miss))

    # For each (ticker, date) in the price universe
    all_dates = surp_df[["ticker", "date"]].drop_duplicates()
    features: dict[tuple[str, str], np.ndarray] = {}

    for _, row in all_dates.iterrows():
        ticker = row["ticker"]
        dt = row["date"]
        date_str = str(dt.date())

        evts = ticker_events.get(ticker, [])
        best = None
        for evt_date, is_beat, is_miss in reversed(evts):
            if evt_date <= dt:
                days_since = (dt - evt_date).days
                if days_since <= lookback_days:
                    best = (is_beat, is_miss, days_since / lookback_days)
                break

        if best is not None:
            features[(ticker, date_str)] = np.array(best, dtype=np.float32)

    return features


# ======================================================================
# 6. Master extraction + alignment
# ======================================================================


def extract_all_embeddings(
    prices_dir: str | Path = "data/raw/prices",
    targets_dir: str | Path = "data/targets",
    processed_dir: str | Path = "data/processed",
    news_csv: str | Path = "data/raw/news/news_articles.csv",
    save_path: str | Path = "data/embeddings/fusion_embeddings.pt",
    device: str = "cuda",
    verbose: bool = True,
) -> dict[str, Any]:
    """Extract embeddings from all pretrained models and save aligned dataset.

    Returns a summary dict with counts and timing info.
    """
    t0 = time.time()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load price data + normalise (same pipeline as training)
    # ------------------------------------------------------------------
    if verbose:
        print("Loading price data and normalising...")
    prices = load_price_csv_dir(prices_dir)
    price_df = prepare_price_features(prices)

    targets_df = pd.read_csv(Path(targets_dir) / "direction_labels_multi_horizon.csv")
    targets_df["date"] = pd.to_datetime(targets_df["date"])
    price_df["date"] = pd.to_datetime(price_df["date"])

    # Fit z-score scaler on train dates (≤ 2022)
    train_mask = price_df["date"] <= "2022-12-31"
    scaler = fit_scaler(price_df.loc[train_mask], ENGINEERED_FEATURES)
    price_df_norm = scaler.transform(price_df.copy(), ENGINEERED_FEATURES)

    # Build PriceWindowDataset
    price_ds = PriceWindowDataset(
        price_df_norm,
        targets_df,
        window_size=60,
        feature_cols=ENGINEERED_FEATURES,
    )
    if verbose:
        print(f"  Price dataset: {len(price_ds)} samples")

    # ------------------------------------------------------------------
    # A) Price embeddings
    # ------------------------------------------------------------------
    if verbose:
        print("\n[1/5] Extracting price embeddings (batch=512)...")
    t1 = time.time()
    price_embs = extract_price_embeddings(price_ds, device=device, batch_size=512)
    if verbose:
        print(f"  Done — {len(price_embs)} embeddings in {time.time()-t1:.1f}s")

    # ------------------------------------------------------------------
    # B) GAT embeddings
    # ------------------------------------------------------------------
    if verbose:
        print("\n[2/5] Extracting GAT embeddings...")
    t2 = time.time()

    graph = load_graph()
    ei, ew, et = make_bidirectional(
        graph["edge_index"], graph["edge_weight"], graph["edge_type"]
    )
    graph_tickers = graph["tickers"]

    # Build graph dataset with same normalisation
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
        print(f"  Done — {len(gat_embs)} embeddings in {time.time()-t2:.1f}s")

    # ------------------------------------------------------------------
    # C) Document embeddings
    # ------------------------------------------------------------------
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
            f"  Done — {len(doc_embs_raw)} filing embeddings in {time.time()-t3:.1f}s"
        )

    # Map (ticker, year) → per-date availability
    # For a prediction date, use the filing from year-1
    doc_embs: dict[tuple[str, str], np.ndarray] = {}
    for ticker, date_str in price_embs:
        year = int(date_str[:4])
        filing_year = year - 1
        key = (ticker, filing_year)
        if key in doc_embs_raw:
            doc_embs[(ticker, date_str)] = doc_embs_raw[key]

    # ------------------------------------------------------------------
    # D) News embeddings
    # ------------------------------------------------------------------
    if verbose:
        print("\n[4/5] Extracting news embeddings (FinBERT fp16)...")
    t4 = time.time()
    news_path = Path(news_csv)
    if news_path.exists():
        news_embs = extract_news_embeddings(
            news_csv,
            prices_dir,
            targets_dir,
            device=device,
            batch_size=16,
        )
    else:
        news_embs = {}
    if verbose:
        print(f"  Done — {len(news_embs)} window embeddings in {time.time()-t4:.1f}s")

    # ------------------------------------------------------------------
    # E) Surprise features
    # ------------------------------------------------------------------
    if verbose:
        print("\n[5/5] Building surprise features...")
    t5 = time.time()
    surprise_feats = build_surprise_features(targets_dir, lookback_days=90)
    if verbose:
        print(f"  Done — {len(surprise_feats)} features in {time.time()-t5:.1f}s")

    # ------------------------------------------------------------------
    # Align everything by (ticker, date) from the price dataset
    # ------------------------------------------------------------------
    if verbose:
        print("\nAligning all modalities...")

    vol_df = pd.read_csv(Path(targets_dir) / "volatility_targets.csv")
    vol_df["date"] = pd.to_datetime(vol_df["date"])
    vol_lookup: dict[tuple[str, str], float] = {}
    for _, row in vol_df.iterrows():
        vol_lookup[(row["ticker"], str(row["date"].date()))] = float(
            row["realized_vol_20d_annualized"]
        )

    surp_target_df = pd.read_csv(Path(targets_dir) / "fundamental_surprise_targets.csv")
    surp_target_df["date"] = pd.to_datetime(surp_target_df["date"])
    surp_target_lookup: dict[tuple[str, str], int] = {}
    for _, row in surp_target_df.iterrows():
        sid = int(row["surprise_id"])
        if sid >= 0:
            surp_target_lookup[(row["ticker"], str(row["date"].date()))] = sid

    N = len(price_ds)
    PRICE_DIM = 256
    GAT_DIM = 256
    DOC_DIM = 768
    NEWS_DIM = 512
    SURP_DIM = 3

    all_price = np.zeros((N, PRICE_DIM), dtype=np.float32)
    all_gat = np.zeros((N, GAT_DIM), dtype=np.float32)
    all_doc = np.zeros((N, DOC_DIM), dtype=np.float32)
    all_news = np.zeros((N, NEWS_DIM), dtype=np.float32)
    all_surprise = np.zeros((N, SURP_DIM), dtype=np.float32)
    all_mask = np.zeros((N, 5), dtype=np.float32)  # [price, gat, doc, news, surprise]
    all_dir_label = np.full(N, -1, dtype=np.int64)
    all_vol_target = np.full(N, float("nan"), dtype=np.float32)
    all_surp_target = np.full(N, -1, dtype=np.int64)
    all_tickers: list[str] = []
    all_dates: list[str] = []

    for i in range(N):
        sample = price_ds._samples[i]
        ticker = sample["ticker"]
        date_str = sample["date"]
        key = (ticker, date_str)

        all_tickers.append(ticker)
        all_dates.append(date_str)

        # Direction label (5-day horizon)
        all_dir_label[i] = int(sample.get("direction_5d", -1))

        # Volatility target
        if key in vol_lookup:
            all_vol_target[i] = vol_lookup[key]

        # Surprise target
        if key in surp_target_lookup:
            all_surp_target[i] = surp_target_lookup[key]

        # Price embedding (always available)
        if key in price_embs:
            all_price[i] = price_embs[key]
            all_mask[i, 0] = 1.0

        # GAT embedding
        if key in gat_embs:
            all_gat[i] = gat_embs[key]
            all_mask[i, 1] = 1.0

        # Document embedding
        if key in doc_embs:
            all_doc[i] = doc_embs[key]
            all_mask[i, 2] = 1.0

        # News embedding
        if key in news_embs:
            all_news[i] = news_embs[key]
            all_mask[i, 3] = 1.0

        # Surprise features
        if key in surprise_feats:
            all_surprise[i] = surprise_feats[key]
            all_mask[i, 4] = 1.0

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    data = {
        "price_emb": torch.from_numpy(all_price),
        "gat_emb": torch.from_numpy(all_gat),
        "doc_emb": torch.from_numpy(all_doc),
        "news_emb": torch.from_numpy(all_news),
        "surprise_feat": torch.from_numpy(all_surprise),
        "modality_mask": torch.from_numpy(all_mask),
        "direction_label": torch.from_numpy(all_dir_label),
        "volatility_target": torch.from_numpy(all_vol_target),
        "surprise_target": torch.from_numpy(all_surp_target),
        "tickers": all_tickers,
        "dates": all_dates,
    }
    torch.save(data, save_path)

    summary = {
        "total_samples": N,
        "price_available": int(all_mask[:, 0].sum()),
        "gat_available": int(all_mask[:, 1].sum()),
        "doc_available": int(all_mask[:, 2].sum()),
        "news_available": int(all_mask[:, 3].sum()),
        "surprise_available": int(all_mask[:, 4].sum()),
        "dir_labels_valid": int((all_dir_label >= 0).sum()),
        "vol_targets_valid": int((~np.isnan(all_vol_target)).sum()),
        "surp_targets_valid": int((all_surp_target >= 0).sum()),
        "time_seconds": time.time() - t0,
        "save_path": str(save_path),
    }

    if verbose:
        print(f"\nExtraction complete — {N} samples")
        for k, v in summary.items():
            if k != "save_path":
                print(f"  {k}: {v}")
        print(f"  Saved to {save_path}")

    return summary
