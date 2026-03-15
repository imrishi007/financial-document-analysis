"""Phase 13 Pipeline: HAR-RV Features + Loss Optimization.

Closely mirrors run_phase12_pipeline.py but with:
  - 21 price features (18 + 3 HAR-RV autoregressive features)
  - Three fusion model variants (0.85/0.15, 0.50/0.50, 0.70/0.30)
  - Phase 13 checkpoints and embedding files

Sequence:
  A. GPU + seed setup
  B. Train price model (21 features, 30 stocks)
  C. Train GAT model (30 nodes, 21 features)
  D. Extract embeddings (price+GAT new; doc+macro reuse Phase 12)
  E. Train three Phase13FusionModel variants
  F. Benchmark comparison
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from src.utils.seed import set_global_seed
from src.utils.gpu import setup_gpu, log_gpu_usage, create_grad_scaler

# ======================================================================
# Constants
# ======================================================================
ALL_30 = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "AMD", "INTC", "ORCL",
    "QCOM", "TXN", "AVGO", "MRVL", "KLAC", "CRM", "ADBE", "NOW", "SNOW", "DDOG",
    "NFLX", "UBER", "PYPL", "SNAP", "DELL", "AMAT", "LRCX", "IBM", "CSCO", "HPE",
]

PRICES_DIR = "data/raw/prices"
TARGETS_DIR = "data/targets"
SAVE_DIR = "models"
EMBEDDINGS_DIR = "data/embeddings"

PHASE13_VARIANTS = {
    "vol_primary":  {"lambda_vol": 0.85, "lambda_dir": 0.15},
    "balanced":     {"lambda_vol": 0.50, "lambda_dir": 0.50},
    "dir_assisted": {"lambda_vol": 0.70, "lambda_dir": 0.30},
}


def check_nan(loss_val, step_name):
    if torch.is_tensor(loss_val):
        loss_val = loss_val.item()
    if np.isnan(loss_val) or np.isinf(loss_val):
        raise RuntimeError(f"NaN/Inf loss detected in {step_name}! Aborting.")


# ======================================================================
# B. Train Price Model (21 features)
# ======================================================================

def train_price_model_phase13():
    """Train PriceDirectionModel with 21 features on 30 stocks."""
    from src.data.preprocessing import SplitConfig, create_time_splits, fit_scaler
    from src.data.price_dataset import (
        ENGINEERED_FEATURES,
        PriceWindowDataset,
        load_price_csv_dir,
        prepare_price_features,
    )
    from src.models.price_model import PriceDirectionModel
    from src.train.common import (
        TrainingConfig,
        EarlyStopping,
        create_optimizer,
        save_checkpoint,
        make_dataloader,
        evaluate_epoch,
    )
    from src.evaluation.metrics import classification_metrics

    print("=" * 60)
    print("PHASE 13B: TRAINING PRICE MODEL (21 features, 30 stocks)")
    print("=" * 60)

    device = setup_gpu(verbose=True)
    dev = str(device)
    set_global_seed(42)

    assert len(ENGINEERED_FEATURES) == 21, f"Expected 21 features, got {len(ENGINEERED_FEATURES)}"
    print(f"Features: {len(ENGINEERED_FEATURES)} — last 3: {ENGINEERED_FEATURES[-3:]}")

    config = TrainingConfig(
        batch_size=1024, epochs=50, learning_rate=1e-3, patience=7, use_amp=True,
    )

    # Load & prepare
    prices = load_price_csv_dir(PRICES_DIR)
    price_feat = prepare_price_features(prices)
    print(f"Total feature rows: {len(price_feat)}")

    targets = pd.read_csv(Path(TARGETS_DIR) / "direction_labels_multi_horizon.csv")
    vol = pd.read_csv(Path(TARGETS_DIR) / "volatility_targets.csv")
    surprise = pd.read_csv(Path(TARGETS_DIR) / "fundamental_surprise_targets.csv")

    # Scale (train only)
    price_feat["date"] = pd.to_datetime(price_feat["date"])
    train_mask = price_feat["date"] <= "2022-12-31"
    scaler = fit_scaler(price_feat[train_mask], ENGINEERED_FEATURES)
    price_feat = scaler.transform(price_feat, ENGINEERED_FEATURES)

    dataset = PriceWindowDataset(
        price_feat, targets, vol_df=vol, surprise_df=surprise, window_size=60,
    )
    print(f"Dataset samples: {len(dataset)}")

    # Split
    split_cfg = SplitConfig()
    sample_dates = pd.Series(
        pd.to_datetime([dataset._samples[i]["date"] for i in range(len(dataset))])
    )
    masks = create_time_splits(sample_dates, split_cfg)
    idx_map = {s: [i for i, m in enumerate(mask) if m] for s, mask in masks.items()}

    train_ds = Subset(dataset, idx_map["train"])
    val_ds = Subset(dataset, idx_map["val"])
    test_ds = Subset(dataset, idx_map["test"])
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    train_loader = make_dataloader(train_ds, batch_size=config.batch_size, shuffle=True, config=config)
    val_loader = make_dataloader(val_ds, batch_size=config.batch_size * 2, config=config)
    test_loader = make_dataloader(test_ds, batch_size=config.batch_size * 2, config=config)

    model = PriceDirectionModel(num_features=len(ENGINEERED_FEATURES)).to(dev)
    optimizer = create_optimizer(model, config)
    criterion = nn.CrossEntropyLoss()
    stopper = EarlyStopping(patience=config.patience, mode="min")
    scaler_amp = create_grad_scaler()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"PriceDirectionModel: {n_params:,} params, {len(ENGINEERED_FEATURES)} features")

    save_path = Path(SAVE_DIR) / "phase13_price_best.pt"
    best_val_loss = float("inf")

    for epoch in range(1, config.epochs + 1):
        model.train()
        loss_sum, correct, total = 0.0, 0, 0

        for batch in train_loader:
            features = batch["features"].to(dev, non_blocking=True)
            labels = batch["direction_60d"].to(dev, non_blocking=True)

            optimizer.zero_grad()
            with torch.amp.autocast("cuda"):
                logits = model(features)
                loss = criterion(logits, labels)

            check_nan(loss, f"price epoch {epoch}")
            scaler_amp.scale(loss).backward()
            scaler_amp.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler_amp.step(optimizer)
            scaler_amp.update()

            loss_sum += loss.item() * features.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            total += features.size(0)

        train_loss = loss_sum / total
        train_acc = correct / total

        val_result = evaluate_epoch(model, val_loader, criterion, dev)
        val_cls = classification_metrics(
            val_result["y_true"], val_result["y_pred"], val_result["y_prob"]
        )

        if epoch % 5 == 0 or epoch == 1:
            alloc = torch.cuda.memory_allocated(0) / 1e9
            print(
                f"  Epoch {epoch:02d} | train={train_loss:.4f} acc={train_acc:.4f} | "
                f"val={val_result['loss']:.4f} acc={val_cls['accuracy']:.4f} | VRAM={alloc:.2f}GB"
            )

        if val_result["loss"] < best_val_loss:
            best_val_loss = val_result["loss"]
            save_checkpoint(model, optimizer, epoch, {"val_loss": val_result["loss"], **val_cls}, save_path)

        if stopper(val_result["loss"]):
            print(f"  Early stopping at epoch {epoch}")
            break

    # Test
    ckpt = torch.load(save_path, weights_only=False, map_location=dev)
    model.load_state_dict(ckpt["model_state_dict"])
    test_result = evaluate_epoch(model, test_loader, criterion, dev)
    test_cls = classification_metrics(
        test_result["y_true"], test_result["y_pred"], test_result["y_prob"]
    )
    print(
        f"\n  Test: loss={test_result['loss']:.4f}, acc={test_cls['accuracy']:.4f}, "
        f"auc={test_cls.get('auc', 0):.4f}"
    )

    del model, optimizer
    torch.cuda.empty_cache()
    print("PHASE 13 STEP B COMPLETE — Price model (21 features) trained")
    return test_cls


# ======================================================================
# C. Train GAT Model (30 nodes, 21 features)
# ======================================================================

def train_gat_model_phase13():
    """Train GraphEnhancedModel with 30 nodes and 21 features."""
    from src.data.preprocessing import SplitConfig, create_time_splits, fit_scaler
    from src.data.price_dataset import (
        ENGINEERED_FEATURES,
        load_price_csv_dir,
        prepare_price_features,
    )
    from src.data.graph_dataset import GraphSnapshotDataset
    from src.data.graph_utils import load_graph, make_bidirectional
    from src.models.gat_model import GraphEnhancedModel
    from src.train.common import EarlyStopping, save_checkpoint, create_optimizer

    print("\n" + "=" * 60)
    print("PHASE 13C: TRAINING GAT MODEL (30 nodes, 21 features)")
    print("=" * 60)

    device = setup_gpu(verbose=True)
    dev = str(device)
    set_global_seed(42)

    # Load graph
    graph = load_graph(
        "data/raw/graph/tech30_nodes.csv", "data/raw/graph/tech30_edges.csv"
    )
    ei, ew, et = make_bidirectional(
        graph["edge_index"], graph["edge_weight"], graph["edge_type"]
    )
    tickers = graph["tickers"]
    print(f"Graph: {graph['num_nodes']} nodes, {ei.shape[1]} edges (bidirectional)")

    # Load price data + features
    prices = load_price_csv_dir(PRICES_DIR)
    price_feat = prepare_price_features(prices)
    price_feat["date"] = pd.to_datetime(price_feat["date"])

    # Targets
    targets_df = pd.read_csv(Path(TARGETS_DIR) / "direction_labels.csv")
    targets_df["date"] = pd.to_datetime(targets_df["date"])

    # Scale (train only)
    train_mask = price_feat["date"] <= "2022-12-31"
    scaler = fit_scaler(price_feat[train_mask], ENGINEERED_FEATURES)
    price_feat = scaler.transform(price_feat, ENGINEERED_FEATURES)

    # Build graph snapshots
    graph_ds = GraphSnapshotDataset(price_feat, targets_df, tickers, window_size=60)
    print(f"Graph snapshots: {len(graph_ds)}")

    # Split by date
    dates_list = [graph_ds._snapshots[i]["date"] for i in range(len(graph_ds))]
    dates_pd = pd.to_datetime(dates_list)
    train_idx = [i for i, d in enumerate(dates_pd) if d <= pd.Timestamp("2022-12-31")]
    val_idx = [i for i, d in enumerate(dates_pd)
               if pd.Timestamp("2022-12-31") < d <= pd.Timestamp("2023-12-31")]
    test_idx = [i for i, d in enumerate(dates_pd) if d > pd.Timestamp("2023-12-31")]
    print(f"Snapshots — Train: {len(train_idx)} | Val: {len(val_idx)} | Test: {len(test_idx)}")

    # Model
    model = GraphEnhancedModel(num_features=len(ENGINEERED_FEATURES)).to(dev)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    stopper = EarlyStopping(patience=7, mode="min")
    scaler_amp = create_grad_scaler()

    ei_dev = ei.to(dev)
    ew_dev = ew.to(dev)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"GraphEnhancedModel: {n_params:,} params")

    save_path = Path(SAVE_DIR) / "phase13_gat_best.pt"
    best_val_loss = float("inf")

    for epoch in range(1, 51):
        model.train()
        epoch_loss, epoch_correct, epoch_total = 0.0, 0, 0

        for i in train_idx:
            sample = graph_ds[i]
            x = sample["features"].to(dev)
            labels = sample["labels"].to(dev)
            mask = sample["mask"]

            optimizer.zero_grad()
            with torch.amp.autocast("cuda"):
                logits = model(x, ei_dev, ew_dev)
                valid = labels >= 0
                if not valid.any():
                    continue
                loss = criterion(logits[valid], labels[valid])

            check_nan(loss, f"GAT epoch {epoch}")
            scaler_amp.scale(loss).backward()
            scaler_amp.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler_amp.step(optimizer)
            scaler_amp.update()

            n_valid = valid.sum().item()
            epoch_loss += loss.item() * n_valid
            epoch_correct += (logits[valid].argmax(1) == labels[valid]).sum().item()
            epoch_total += n_valid

        if epoch_total == 0:
            continue
        train_loss = epoch_loss / epoch_total

        # Validation
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for i in val_idx:
                sample = graph_ds[i]
                x = sample["features"].to(dev)
                labels = sample["labels"].to(dev)
                with torch.amp.autocast("cuda"):
                    logits = model(x, ei_dev, ew_dev)
                    valid = labels >= 0
                    if not valid.any():
                        continue
                    loss = criterion(logits[valid], labels[valid])
                n_v = valid.sum().item()
                val_loss += loss.item() * n_v
                val_correct += (logits[valid].argmax(1) == labels[valid]).sum().item()
                val_total += n_v

        if val_total > 0:
            vl = val_loss / val_total
            va = val_correct / val_total
        else:
            vl, va = float("inf"), 0.0

        if epoch % 5 == 0 or epoch == 1:
            alloc = torch.cuda.memory_allocated(0) / 1e9
            print(f"  Epoch {epoch:02d} | train={train_loss:.4f} | val={vl:.4f} acc={va:.4f} | VRAM={alloc:.2f}GB")

        if vl < best_val_loss:
            best_val_loss = vl
            save_checkpoint(model, optimizer, epoch, {"val_loss": vl, "val_acc": va}, save_path)

        if stopper(vl):
            print(f"  Early stopping at epoch {epoch}")
            break

    # Test
    ckpt = torch.load(save_path, weights_only=False, map_location=dev)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    test_correct, test_total = 0, 0
    with torch.no_grad():
        for i in test_idx:
            sample = graph_ds[i]
            x = sample["features"].to(dev)
            labels = sample["labels"].to(dev)
            logits = model(x, ei_dev, ew_dev)
            valid = labels >= 0
            if valid.any():
                test_correct += (logits[valid].argmax(1) == labels[valid]).sum().item()
                test_total += valid.sum().item()

    test_acc = test_correct / max(test_total, 1)
    print(f"\n  Test accuracy: {test_acc:.4f} ({test_total} samples)")

    del model, optimizer
    torch.cuda.empty_cache()
    print("PHASE 13 STEP C COMPLETE — GAT model (21 features) trained")
    return {"test_acc": test_acc}


# ======================================================================
# D. Extract Embeddings
# ======================================================================

def extract_phase13_embeddings():
    """Extract embeddings using Phase 13 trained models.
    
    Price + GAT: re-extracted with 21 features.
    Doc + Macro: reused from Phase 12 fusion embeddings.
    """
    import torch.nn.functional as F
    from src.data.preprocessing import fit_scaler
    from src.data.price_dataset import (
        ENGINEERED_FEATURES,
        PriceWindowDataset,
        load_price_csv_dir,
        prepare_price_features,
    )
    from src.data.graph_dataset import GraphSnapshotDataset
    from src.data.graph_utils import load_graph, make_bidirectional
    from src.models.price_model import PriceDirectionModel
    from src.models.gat_model import GraphEnhancedModel

    print("\n" + "=" * 60)
    print("PHASE 13D: EXTRACTING EMBEDDINGS (21 features)")
    print("=" * 60)

    device = setup_gpu(verbose=True)
    dev = str(device)
    set_global_seed(42)

    # ---- Load and prepare price data ----
    prices = load_price_csv_dir(PRICES_DIR)
    price_feat = prepare_price_features(prices)
    price_feat["date"] = pd.to_datetime(price_feat["date"])

    targets = pd.read_csv(Path(TARGETS_DIR) / "direction_labels_multi_horizon.csv")
    vol = pd.read_csv(Path(TARGETS_DIR) / "volatility_targets.csv")
    surprise = pd.read_csv(Path(TARGETS_DIR) / "fundamental_surprise_targets.csv")

    # Scale (train only)
    train_mask = price_feat["date"] <= "2022-12-31"
    scaler = fit_scaler(price_feat[train_mask], ENGINEERED_FEATURES)
    price_feat = scaler.transform(price_feat, ENGINEERED_FEATURES)

    price_ds = PriceWindowDataset(
        price_feat, targets, vol_df=vol, surprise_df=surprise, window_size=60,
    )
    print(f"Price dataset: {len(price_ds)} samples")

    # === A) Price embeddings (CNN-BiLSTM hidden state) ===
    print("\n[1/4] Extracting price embeddings (21 features)...")
    t1 = time.time()
    price_model = PriceDirectionModel(num_features=len(ENGINEERED_FEATURES))
    price_ckpt = torch.load(Path(SAVE_DIR) / "phase13_price_best.pt", map_location="cpu", weights_only=False)
    price_model.load_state_dict(price_ckpt["model_state_dict"])
    price_model.eval().to(dev)

    price_embs = {}
    loader = DataLoader(price_ds, batch_size=512, shuffle=False, num_workers=0)
    with torch.no_grad():
        for batch in loader:
            x = batch["features"].to(dev)
            tickers_b = batch["ticker"]
            dates_b = batch["date"]
            with torch.amp.autocast("cuda"):
                # Extract 256-d LSTM hidden state (before classification head)
                h = x.transpose(1, 2)
                h = price_model.conv(h)
                h = h.transpose(1, 2)
                h, _ = price_model.lstm(h)
                emb = h[:, -1, :].float().cpu().numpy()
            for i in range(len(tickers_b)):
                price_embs[(tickers_b[i], dates_b[i])] = emb[i]

    del price_model
    torch.cuda.empty_cache()
    print(f"  {len(price_embs)} price embeddings in {time.time()-t1:.1f}s")

    # === B) GAT embeddings ===
    print("\n[2/4] Extracting GAT embeddings (21 features)...")
    t2 = time.time()
    graph = load_graph(
        "data/raw/graph/tech30_nodes.csv", "data/raw/graph/tech30_edges.csv"
    )
    ei, ew, _ = make_bidirectional(
        graph["edge_index"], graph["edge_weight"], graph["edge_type"]
    )
    graph_tickers = graph["tickers"]

    gat_targets = pd.read_csv(Path(TARGETS_DIR) / "direction_labels.csv")
    gat_targets["date"] = pd.to_datetime(gat_targets["date"])

    gat_ds = GraphSnapshotDataset(price_feat, gat_targets, graph_tickers, window_size=60)
    print(f"  Graph snapshots: {len(gat_ds)}")

    gat_model = GraphEnhancedModel(num_features=len(ENGINEERED_FEATURES))
    gat_ckpt = torch.load(Path(SAVE_DIR) / "phase13_gat_best.pt", map_location="cpu", weights_only=False)
    gat_model.load_state_dict(gat_ckpt["model_state_dict"])
    gat_model.eval().to(dev)
    ei_dev = ei.to(dev)
    ew_dev = ew.to(dev)

    gat_embs = {}
    with torch.no_grad():
        for si in range(len(gat_ds)):
            sample = gat_ds[si]
            x = sample["features"].to(dev)
            m = sample["mask"]
            with torch.amp.autocast("cuda"):
                h = gat_model.encode_price(x)
                h = gat_model.node_proj(h)
                h1 = F.elu(gat_model.gat1(h, ei_dev, ew_dev))
                h = gat_model.gat_norm1(h + h1)
                h2 = F.elu(gat_model.gat2(h, ei_dev, ew_dev))
                h = gat_model.gat_norm2(h + h2)
            emb = h.float().cpu().numpy()
            date_str = gat_ds._snapshots[si]["date"]
            for j in range(len(graph_tickers)):
                if m[j]:
                    gat_embs[(graph_tickers[j], date_str)] = emb[j]

    del gat_model
    torch.cuda.empty_cache()
    print(f"  {len(gat_embs)} GAT embeddings in {time.time()-t2:.1f}s")

    # === C) Doc + Macro: reuse from Phase 12 ===
    print("\n[3/4] Loading doc+macro from Phase 12 embeddings (reuse)...")
    p12 = torch.load(f"{EMBEDDINGS_DIR}/phase12_fusion_embeddings.pt", weights_only=False)
    p12_index = {}
    for i, (t, d) in enumerate(zip(p12["tickers"], p12["dates"])):
        p12_index[(t, str(d)[:10])] = i
    print(f"  Phase 12 index: {len(p12_index)} entries")

    # === D) Surprise features ===
    print("\n[4/4] Building surprise features...")
    t5 = time.time()
    from src.features.extract_embeddings import build_surprise_features
    surprise_feats = build_surprise_features(TARGETS_DIR, lookback_days=90)
    print(f"  {len(surprise_feats)} surprise features in {time.time()-t5:.1f}s")

    # === Align everything ===
    print("\nAligning all modalities...")
    vol_df = pd.read_csv(Path(TARGETS_DIR) / "volatility_targets.csv")
    vol_df["date"] = pd.to_datetime(vol_df["date"])
    vol_lookup = {}
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
    all_mask = np.zeros((N, 4), dtype=np.float32)
    all_dir_label = np.full(N, -1, dtype=np.int64)
    all_vol_target = np.full(N, float("nan"), dtype=np.float32)
    all_tickers = []
    all_dates = []

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

        # Doc + Macro from Phase 12
        if key in p12_index:
            idx = p12_index[key]
            all_doc[i] = p12["doc_emb"][idx].numpy()
            all_macro[i] = p12["macro_emb"][idx].numpy()
            if p12["modality_mask"][idx, 2] > 0:
                all_mask[i, 2] = 1.0
            if p12["modality_mask"][idx, 3] > 0:
                all_mask[i, 3] = 1.0

        if key in surprise_feats:
            all_surprise[i] = surprise_feats[key]

    # Save
    save_path = Path(EMBEDDINGS_DIR) / "phase13_fusion_embeddings.pt"
    save_path.parent.mkdir(parents=True, exist_ok=True)

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

    print(f"\nEmbeddings saved: {save_path}")
    print(f"  Total:  {N} samples")
    print(f"  Price:  {int(all_mask[:, 0].sum())}")
    print(f"  GAT:    {int(all_mask[:, 1].sum())}")
    print(f"  Doc:    {int(all_mask[:, 2].sum())}")
    print(f"  Macro:  {int(all_mask[:, 3].sum())}")
    print(f"  Vol targets valid: {int((~np.isnan(all_vol_target)).sum())}")
    print(f"  Dir labels valid: {int((all_dir_label >= 0).sum())}")

    print("PHASE 13 STEP D COMPLETE — Embeddings extracted")
    return data


# ======================================================================
# E. Train Three Fusion Variants
# ======================================================================

def train_fusion_phase13():
    """Train three Phase13FusionModel variants with different loss weights."""
    from src.models.fusion_model import Phase13FusionModel
    from src.models.losses import CombinedVolatilityLoss
    from src.train.common import EarlyStopping, save_checkpoint
    from sklearn.metrics import roc_auc_score

    print("\n" + "=" * 60)
    print("PHASE 13E: TRAINING THREE FUSION VARIANTS")
    print("=" * 60)

    device = setup_gpu(verbose=True)
    dev = str(device)
    set_global_seed(42)

    # Load embeddings
    emb_path = Path(EMBEDDINGS_DIR) / "phase13_fusion_embeddings.pt"
    data = torch.load(emb_path, weights_only=False)
    print(f"Loaded embeddings: {data['price_emb'].shape[0]} samples")

    # Date-based splits
    dates = pd.to_datetime(pd.Series(data["dates"]))
    train_mask = (dates <= "2022-12-31").values
    val_mask = ((dates > "2022-12-31") & (dates <= "2023-12-31")).values
    test_mask = (dates > "2023-12-31").values

    # Build date indices for ListNet
    date_strs = data["dates"]
    unique_dates = sorted(set(date_strs))
    date_to_idx = {d: i for i, d in enumerate(unique_dates)}
    date_indices = torch.tensor([date_to_idx[d] for d in date_strs], dtype=torch.long)

    class FusionDataset(torch.utils.data.Dataset):
        def __init__(self, indices):
            self.indices = indices

        def __len__(self):
            return len(self.indices)

        def __getitem__(self, idx):
            i = self.indices[idx]
            return {
                "price_emb": data["price_emb"][i],
                "gat_emb": data["gat_emb"][i],
                "doc_emb": data["doc_emb"][i],
                "macro_emb": data["macro_emb"][i],
                "surprise_feat": data["surprise_feat"][i],
                "modality_mask": data["modality_mask"][i],
                "direction_label": data["direction_label"][i],
                "volatility_target": data["volatility_target"][i],
                "date_idx": date_indices[i],
            }

    train_idx = np.where(train_mask)[0].tolist()
    val_idx = np.where(val_mask)[0].tolist()
    test_idx = np.where(test_mask)[0].tolist()
    print(f"Train: {len(train_idx)} | Val: {len(val_idx)} | Test: {len(test_idx)}")

    train_ds = FusionDataset(train_idx)
    val_ds = FusionDataset(val_idx)
    test_ds = FusionDataset(test_idx)

    train_loader = DataLoader(train_ds, batch_size=2048, shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=4096, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=4096, shuffle=False, num_workers=0)

    all_variant_results = {}

    for variant_name, weights in PHASE13_VARIANTS.items():
        lv = weights["lambda_vol"]
        ld = weights["lambda_dir"]
        print(f"\n  {'='*56}")
        print(f"  Variant: {variant_name}  (lambda_vol={lv}, lambda_dir={ld})")
        print(f"  {'='*56}")

        set_global_seed(42)
        model = Phase13FusionModel(
            price_dim=256, gat_dim=256, doc_dim=768, macro_dim=32, surprise_dim=5,
            proj_dim=128, hidden_dim=256, dropout=0.3, mc_dropout=True,
            lambda_vol=lv, lambda_dir=ld,
        ).to(dev)

        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=60, eta_min=1e-6)
        criterion = CombinedVolatilityLoss(lambda_vol=lv, lambda_dir=ld)
        stopper = EarlyStopping(patience=7, mode="min")
        scaler_amp = create_grad_scaler()

        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Phase13FusionModel: {n_params:,} params")

        save_path = Path(SAVE_DIR) / f"phase13_fusion_{variant_name}.pt"
        best_val_loss = float("inf")

        for epoch in range(1, 61):
            model.train()
            epoch_loss, n_batches = 0.0, 0

            for batch in train_loader:
                price_emb = batch["price_emb"].to(dev)
                gat_emb = batch["gat_emb"].to(dev)
                doc_emb = batch["doc_emb"].to(dev)
                macro_emb = batch["macro_emb"].to(dev)
                surprise_feat = batch["surprise_feat"].to(dev)
                modality_mask = batch["modality_mask"].to(dev)
                dir_labels = batch["direction_label"].to(dev)
                vol_targets = batch["volatility_target"].to(dev)
                date_idx = batch["date_idx"].to(dev)

                optimizer.zero_grad()
                with torch.amp.autocast("cuda"):
                    out = model(price_emb, gat_emb, doc_emb, macro_emb, surprise_feat, modality_mask)
                    losses = criterion(out["direction_logits"], dir_labels,
                                       out["volatility_pred"], vol_targets, date_idx)

                total_loss = losses["total"]
                check_nan(total_loss, f"{variant_name} epoch {epoch}")

                scaler_amp.scale(total_loss).backward()
                scaler_amp.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler_amp.step(optimizer)
                scaler_amp.update()

                epoch_loss += total_loss.item()
                n_batches += 1

            scheduler.step()
            avg_loss = epoch_loss / max(n_batches, 1)

            # Validation
            model.eval()
            val_loss_sum, val_n = 0.0, 0
            val_vol_preds, val_vol_targets_ = [], []
            val_dir_preds, val_dir_labels_ = [], []

            with torch.no_grad():
                for batch in val_loader:
                    price_emb = batch["price_emb"].to(dev)
                    gat_emb = batch["gat_emb"].to(dev)
                    doc_emb = batch["doc_emb"].to(dev)
                    macro_emb = batch["macro_emb"].to(dev)
                    surprise_feat = batch["surprise_feat"].to(dev)
                    modality_mask = batch["modality_mask"].to(dev)
                    dir_labels = batch["direction_label"].to(dev)
                    vol_targets = batch["volatility_target"].to(dev)
                    date_idx = batch["date_idx"].to(dev)

                    with torch.amp.autocast("cuda"):
                        out = model(price_emb, gat_emb, doc_emb, macro_emb, surprise_feat, modality_mask)
                        losses = criterion(out["direction_logits"], dir_labels,
                                           out["volatility_pred"], vol_targets, date_idx)

                    bs = price_emb.size(0)
                    val_loss_sum += losses["total"].item() * bs
                    val_n += bs

                    valid_v = ~torch.isnan(vol_targets)
                    if valid_v.any():
                        val_vol_preds.append(out["volatility_pred"][valid_v].cpu())
                        val_vol_targets_.append(vol_targets[valid_v].cpu())

                    valid_d = dir_labels >= 0
                    if valid_d.any():
                        probs = torch.softmax(out["direction_logits"][valid_d], dim=1)[:, 1]
                        val_dir_preds.append(probs.cpu())
                        val_dir_labels_.append(dir_labels[valid_d].cpu())

            val_loss = val_loss_sum / max(val_n, 1)

            val_r2 = 0.0
            if val_vol_preds:
                vp = torch.cat(val_vol_preds).numpy()
                vt = torch.cat(val_vol_targets_).numpy()
                ss_res = np.sum((vt - vp) ** 2)
                ss_tot = np.sum((vt - vt.mean()) ** 2)
                val_r2 = 1 - ss_res / max(ss_tot, 1e-8)

            val_auc = 0.5
            if val_dir_preds:
                dp = torch.cat(val_dir_preds).numpy()
                dl = torch.cat(val_dir_labels_).numpy()
                try:
                    val_auc = roc_auc_score(dl, dp)
                except Exception:
                    pass

            if epoch % 5 == 0 or epoch == 1:
                alloc = torch.cuda.memory_allocated(0) / 1e9
                print(
                    f"    [{variant_name}] Ep {epoch:3d}/60 | Loss: {avg_loss:.4f} | "
                    f"Val R²: {val_r2:.4f} | Dir AUC: {val_auc:.4f} | VRAM: {alloc:.2f}GB"
                )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(model, optimizer, epoch,
                                {"val_loss": val_loss, "val_r2": val_r2, "val_auc": val_auc},
                                save_path)

            if stopper(val_loss):
                print(f"    Early stopping at epoch {epoch}")
                break

        # === Test evaluation ===
        print(f"  Evaluating {variant_name} on test set...")
        ckpt = torch.load(save_path, weights_only=False, map_location=dev)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        test_vol_preds, test_vol_targets_ = [], []
        test_dir_preds, test_dir_labels_ = [], []
        test_gate_weights = []

        with torch.no_grad():
            for batch in test_loader:
                price_emb = batch["price_emb"].to(dev)
                gat_emb = batch["gat_emb"].to(dev)
                doc_emb = batch["doc_emb"].to(dev)
                macro_emb = batch["macro_emb"].to(dev)
                surprise_feat = batch["surprise_feat"].to(dev)
                modality_mask = batch["modality_mask"].to(dev)
                dir_labels = batch["direction_label"].to(dev)
                vol_targets = batch["volatility_target"].to(dev)

                with torch.amp.autocast("cuda"):
                    out = model(price_emb, gat_emb, doc_emb, macro_emb, surprise_feat, modality_mask)

                valid_vol = ~torch.isnan(vol_targets)
                if valid_vol.any():
                    test_vol_preds.append(out["volatility_pred"][valid_vol].cpu())
                    test_vol_targets_.append(vol_targets[valid_vol].cpu())

                valid_dir = dir_labels >= 0
                if valid_dir.any():
                    probs = torch.softmax(out["direction_logits"][valid_dir], dim=1)[:, 1]
                    test_dir_preds.append(probs.cpu())
                    test_dir_labels_.append(dir_labels[valid_dir].cpu())

                test_gate_weights.append(out["gate_weights"].cpu().mean(0))

        vp = torch.cat(test_vol_preds).numpy()
        vt = torch.cat(test_vol_targets_).numpy()
        ss_res = np.sum((vt - vp) ** 2)
        ss_tot = np.sum((vt - vt.mean()) ** 2)
        test_r2 = 1 - ss_res / max(ss_tot, 1e-8)
        test_mae = np.mean(np.abs(vt - vp))
        test_rmse = np.sqrt(np.mean((vt - vp) ** 2))

        dp = torch.cat(test_dir_preds).numpy()
        dl = torch.cat(test_dir_labels_).numpy()
        try:
            from sklearn.metrics import roc_auc_score, accuracy_score
            test_auc = roc_auc_score(dl, dp)
            test_acc = accuracy_score(dl, (dp > 0.5).astype(int))
        except Exception:
            test_auc, test_acc = 0.5, 0.5

        avg_gates = torch.stack(test_gate_weights).mean(0).numpy()

        print(f"\n  {variant_name} TEST RESULTS:")
        print(f"    Vol R²:  {test_r2:.4f}")
        print(f"    QLIKE:   (val criterion)")
        print(f"    MAE:     {test_mae:.4f}")
        print(f"    RMSE:    {test_rmse:.4f}")
        print(f"    Dir AUC: {test_auc:.4f}")
        print(f"    Dir Acc: {test_acc:.4f}")
        print(f"    Gates: price={avg_gates[0]:.1%} gat={avg_gates[1]:.1%} "
              f"doc={avg_gates[2]:.1%} macro={avg_gates[3]:.1%}")

        all_variant_results[variant_name] = {
            "vol_r2": float(test_r2),
            "mae": float(test_mae),
            "rmse": float(test_rmse),
            "dir_auc": float(test_auc),
            "dir_acc": float(test_acc),
            "lambda_vol": lv,
            "lambda_dir": ld,
            "gates": avg_gates.tolist(),
        }

        del model, optimizer
        torch.cuda.empty_cache()

    # Comparison table
    print("\n" + "=" * 80)
    print("VARIANT COMPARISON")
    print(f"{'Variant':<25} | {'Vol R²':>8} | {'RMSE':>8} | {'Dir AUC':>8} | Notes")
    print("-" * 80)
    print(f"{'Phase 12 (ref)':<25} | {'0.7719':>8} | {'0.0958':>8} | {'0.5675':>8} | No HAR-RV")
    for name, res in all_variant_results.items():
        lv, ld = res["lambda_vol"], res["lambda_dir"]
        print(f"{name:<25} | {res['vol_r2']:>8.4f} | {res['rmse']:>8.4f} | "
              f"{res['dir_auc']:>8.4f} | HAR-RV + {lv}/{ld}")
    print(f"{'HAR-RV (ref)':<25} | {'0.9469':>8} | {'N/A':>8} | {'---':>8} | Benchmark to beat")
    print("=" * 80)

    return all_variant_results


# ======================================================================
# F. Benchmark Update
# ======================================================================

def update_benchmarks(variant_results):
    print("\n" + "=" * 60)
    print("PHASE 13F: BENCHMARK COMPARISON")
    print("=" * 60)

    p12_bmarks = json.loads(Path(f"{SAVE_DIR}/phase12_benchmark_results.json").read_text())
    print("Phase 12 benchmarks:")
    for k, v in p12_bmarks.items():
        if isinstance(v, dict) and "r2" in v:
            print(f"  {k}: R²={v['r2']:.4f}")

    har_rv_r2 = 0.9469
    best_vol = max(variant_results.items(), key=lambda x: x[1]["vol_r2"])
    best_dir = max(variant_results.items(), key=lambda x: x[1]["dir_auc"])

    print(f"\nBest vol R²: {best_vol[0]} = {best_vol[1]['vol_r2']:.4f}")
    print(f"Best dir AUC: {best_dir[0]} = {best_dir[1]['dir_auc']:.4f}")
    print(f"HAR-RV R²: {har_rv_r2}")

    results = {
        "phase12_benchmarks": p12_bmarks,
        "phase13_variants": variant_results,
        "best_vol_variant": best_vol[0],
        "best_dir_variant": best_dir[0],
    }

    if best_vol[1]["vol_r2"] > har_rv_r2:
        print(f"\n*** LANDMARK: Phase 13 BEATS HAR-RV! "
              f"{best_vol[0]}: R²={best_vol[1]['vol_r2']:.4f} > {har_rv_r2} ***")
        results["beats_har_rv"] = True
        results["margin"] = float(best_vol[1]["vol_r2"] - har_rv_r2)
    else:
        gap = har_rv_r2 - best_vol[1]["vol_r2"]
        improvement = best_vol[1]["vol_r2"] - 0.7719
        print(f"\nPhase 13 does NOT beat HAR-RV. Gap: {gap:.4f}")
        print(f"Improvement over Phase 12: {improvement:+.4f} (0.7719 → {best_vol[1]['vol_r2']:.4f})")
        results["beats_har_rv"] = False
        results["gap_to_har_rv"] = float(gap)
        results["improvement_over_phase12"] = float(improvement)

    out_path = f"{SAVE_DIR}/phase13_benchmark_results.json"
    Path(out_path).write_text(json.dumps(results, indent=2))
    print(f"\nSaved: {out_path}")
    print("PHASE 13 STEP 6 COMPLETE — Benchmarks updated")
    return results


# ======================================================================
# MAIN
# ======================================================================

if __name__ == "__main__":
    set_global_seed(42)

    print("=" * 70)
    print("PHASE 13 PIPELINE START — HAR-RV Features + Loss Optimization")
    print("=" * 70)

    # Step A: GPU verification
    device = setup_gpu(verbose=True)
    print("\nPHASE 13 STEP 2 COMPLETE — GPU verified")

    # Step B: Train price model (21 features)
    train_price_model_phase13()

    # Step C: Train GAT model (21 features)
    train_gat_model_phase13()

    # Step D: Extract embeddings
    extract_phase13_embeddings()
    print("\nPHASE 13 STEP 4 COMPLETE — Embeddings re-extracted")

    # Step E: Train three variants
    variant_results = train_fusion_phase13()
    print("\nPHASE 13 STEP 5 COMPLETE — Three variants trained")

    # Step F: Benchmarks
    update_benchmarks(variant_results)

    print("\n" + "=" * 70)
    print("PHASE 13 PIPELINE COMPLETE")
    print("=" * 70)
