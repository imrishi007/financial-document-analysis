"""Phase 12 Pipeline: Train component models, extract embeddings, train fusion.

Sequence:
  10A. Train price model (18 features, 30 stocks)
  10B. Train GAT model (30 nodes, 18 features)
  10C. Document model (reuse existing checkpoint)
  10D. Extract aligned embeddings for all 30 stocks
  10E. Train Phase12FusionModel (QLIKE-primary)
"""

from __future__ import annotations

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
    "AAPL",
    "MSFT",
    "GOOGL",
    "AMZN",
    "NVDA",
    "META",
    "TSLA",
    "AMD",
    "INTC",
    "ORCL",
    "QCOM",
    "TXN",
    "AVGO",
    "MRVL",
    "KLAC",
    "CRM",
    "ADBE",
    "NOW",
    "SNOW",
    "DDOG",
    "NFLX",
    "UBER",
    "PYPL",
    "SNAP",
    "DELL",
    "AMAT",
    "LRCX",
    "IBM",
    "CSCO",
    "HPE",
]

PRICES_DIR = "data/raw/prices"
TARGETS_DIR = "data/targets"
SAVE_DIR = "models"
EMBEDDINGS_DIR = "data/embeddings"


def check_nan(loss_val, step_name):
    """Abort if loss is NaN."""
    if torch.is_tensor(loss_val):
        loss_val = loss_val.item()
    if np.isnan(loss_val) or np.isinf(loss_val):
        raise RuntimeError(f"NaN/Inf loss detected in {step_name}! Aborting.")


# ======================================================================
# 10A. Train Price Model (18 features, 30 stocks)
# ======================================================================
def train_price_model_phase12():
    """Train PriceDirectionModel with 18 features on 30 stocks."""
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
    print("10A: TRAINING PRICE MODEL (18 features, 30 stocks)")
    print("=" * 60)

    device = setup_gpu(verbose=True)
    set_global_seed(42)

    config = TrainingConfig(
        batch_size=1024,
        epochs=30,
        learning_rate=1e-4,
        patience=7,
        use_amp=True,
    )

    # Load & prepare
    prices = load_price_csv_dir(PRICES_DIR)
    price_feat = prepare_price_features(prices)
    print(f"Features: {len(ENGINEERED_FEATURES)} -> {ENGINEERED_FEATURES}")
    print(f"Total feature rows: {len(price_feat)}")

    targets = pd.read_csv(Path(TARGETS_DIR) / "direction_labels_multi_horizon.csv")
    vol = pd.read_csv(Path(TARGETS_DIR) / "volatility_targets.csv")
    surprise = pd.read_csv(Path(TARGETS_DIR) / "fundamental_surprise_targets.csv")

    # Scale (train only)
    dates_s = pd.to_datetime(price_feat["date"])
    train_mask = dates_s <= "2022-12-31"
    scaler = fit_scaler(price_feat[train_mask], ENGINEERED_FEATURES)
    price_feat = scaler.transform(price_feat, ENGINEERED_FEATURES)

    dataset = PriceWindowDataset(
        price_feat,
        targets,
        vol_df=vol,
        surprise_df=surprise,
        window_size=60,
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

    train_loader = make_dataloader(
        train_ds, batch_size=config.batch_size, shuffle=True, config=config
    )
    val_loader = make_dataloader(
        val_ds, batch_size=config.batch_size * 2, config=config
    )
    test_loader = make_dataloader(
        test_ds, batch_size=config.batch_size * 2, config=config
    )

    # Model
    model = PriceDirectionModel(num_features=len(ENGINEERED_FEATURES)).to(str(device))
    optimizer = create_optimizer(model, config)
    criterion = nn.CrossEntropyLoss()
    stopper = EarlyStopping(patience=config.patience, mode="min")
    scaler_amp = create_grad_scaler()

    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"PriceDirectionModel: {n_params:,} params, {len(ENGINEERED_FEATURES)} features"
    )

    best_val_loss = float("inf")
    save_path = Path(SAVE_DIR) / "phase12_price_model_best.pt"

    for epoch in range(1, config.epochs + 1):
        model.train()
        loss_sum, correct, total = 0.0, 0, 0

        for batch in train_loader:
            features = batch["features"].to(str(device), non_blocking=True)
            labels = batch["direction_60d"].to(str(device), non_blocking=True)

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

        # Val
        val_result = evaluate_epoch(model, val_loader, criterion, str(device))
        val_cls = classification_metrics(
            val_result["y_true"], val_result["y_pred"], val_result["y_prob"]
        )

        if epoch % 5 == 0 or epoch == 1:
            alloc = torch.cuda.memory_allocated(0) / 1e9
            print(
                f"  Epoch {epoch:02d} | train_loss={train_loss:.4f} acc={train_acc:.4f} | "
                f"val_loss={val_result['loss']:.4f} val_acc={val_cls['accuracy']:.4f} | VRAM={alloc:.2f}GB"
            )

        if val_result["loss"] < best_val_loss:
            best_val_loss = val_result["loss"]
            save_checkpoint(
                model,
                optimizer,
                epoch,
                {"val_loss": val_result["loss"], **val_cls},
                save_path,
            )

        if stopper(val_result["loss"]):
            print(f"  Early stopping at epoch {epoch}")
            break

    # Test
    ckpt = torch.load(save_path, weights_only=False, map_location=str(device))
    model.load_state_dict(ckpt["model_state_dict"])
    test_result = evaluate_epoch(model, test_loader, criterion, str(device))
    test_cls = classification_metrics(
        test_result["y_true"], test_result["y_pred"], test_result["y_prob"]
    )
    print(
        f"\n  Test: loss={test_result['loss']:.4f}, acc={test_cls['accuracy']:.4f}, "
        f"f1={test_cls['f1']:.4f}, auc={test_cls.get('auc', 0):.4f}"
    )

    del model, optimizer
    torch.cuda.empty_cache()
    print("PHASE 12 STEP 10A COMPLETE — Price model trained")
    return test_cls


# ======================================================================
# 10B. Train GAT Model (30 nodes, 18 features)
# ======================================================================
def train_gat_model_phase12():
    """Train GraphEnhancedModel with 30 nodes and 18 features."""
    from src.data.preprocessing import SplitConfig, create_time_splits, fit_scaler
    from src.data.price_dataset import (
        ENGINEERED_FEATURES,
        load_price_csv_dir,
        prepare_price_features,
    )
    from src.data.graph_dataset import GraphSnapshotDataset, build_graph_snapshots
    from src.data.graph_utils import load_graph, make_bidirectional
    from src.models.gat_model import GraphEnhancedModel
    from src.train.common import (
        TrainingConfig,
        EarlyStopping,
        save_checkpoint,
        create_optimizer,
    )

    print("\n" + "=" * 60)
    print("10B: TRAINING GAT MODEL (30 nodes, 18 features)")
    print("=" * 60)

    device = setup_gpu(verbose=True)
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

    # Targets
    targets_df = pd.read_csv(Path(TARGETS_DIR) / "direction_labels.csv")
    targets_df["date"] = pd.to_datetime(targets_df["date"])
    price_feat["date"] = pd.to_datetime(price_feat["date"])

    # Scale
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
    val_idx = [
        i
        for i, d in enumerate(dates_pd)
        if pd.Timestamp("2022-12-31") < d <= pd.Timestamp("2023-12-31")
    ]
    test_idx = [i for i, d in enumerate(dates_pd) if d > pd.Timestamp("2023-12-31")]

    print(
        f"Snapshots — Train: {len(train_idx)} | Val: {len(val_idx)} | Test: {len(test_idx)}"
    )

    # Model
    model = GraphEnhancedModel(num_features=len(ENGINEERED_FEATURES)).to(str(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    stopper = EarlyStopping(patience=7, mode="min")
    scaler_amp = create_grad_scaler()

    ei_dev = ei.to(str(device))
    ew_dev = ew.to(str(device))

    n_params = sum(p.numel() for p in model.parameters())
    print(f"GraphEnhancedModel: {n_params:,} params")

    save_path = Path(SAVE_DIR) / "phase12_graph_model_best.pt"
    best_val_loss = float("inf")

    for epoch in range(1, 31):
        model.train()
        epoch_loss, epoch_correct, epoch_total = 0.0, 0, 0

        for i in train_idx:
            sample = graph_ds[i]
            x = sample["features"].to(str(device))  # [N, T, F]
            labels = sample["labels"].to(str(device))  # [N]
            mask = sample["mask"]  # [N]

            optimizer.zero_grad()
            with torch.amp.autocast("cuda"):
                logits = model(x, ei_dev, ew_dev)  # [N, 2]
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
                x = sample["features"].to(str(device))
                labels = sample["labels"].to(str(device))
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
            print(
                f"  Epoch {epoch:02d} | train_loss={train_loss:.4f} | "
                f"val_loss={vl:.4f} val_acc={va:.4f} | VRAM={alloc:.2f}GB"
            )

        if vl < best_val_loss:
            best_val_loss = vl
            save_checkpoint(
                model, optimizer, epoch, {"val_loss": vl, "val_acc": va}, save_path
            )

        if stopper(vl):
            print(f"  Early stopping at epoch {epoch}")
            break

    # Test
    ckpt = torch.load(save_path, weights_only=False, map_location=str(device))
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    test_correct, test_total = 0, 0
    with torch.no_grad():
        for i in test_idx:
            sample = graph_ds[i]
            x = sample["features"].to(str(device))
            labels = sample["labels"].to(str(device))
            logits = model(x, ei_dev, ew_dev)
            valid = labels >= 0
            if valid.any():
                test_correct += (logits[valid].argmax(1) == labels[valid]).sum().item()
                test_total += valid.sum().item()

    test_acc = test_correct / max(test_total, 1)
    print(f"\n  Test accuracy: {test_acc:.4f} ({test_total} samples)")

    del model, optimizer
    torch.cuda.empty_cache()
    print("PHASE 12 STEP 10B COMPLETE — GAT model trained")
    return {"test_acc": test_acc}


# ======================================================================
# 10C/D. Extract Embeddings for 30 stocks
# ======================================================================
def extract_phase12_embeddings():
    """Extract embeddings using Phase 12 trained models."""
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
    import torch.nn.functional as F

    print("\n" + "=" * 60)
    print("10C/D: EXTRACTING EMBEDDINGS FOR 30 STOCKS")
    print("=" * 60)

    device = setup_gpu(verbose=True)
    dev = str(device)
    set_global_seed(42)

    # Load data
    prices = load_price_csv_dir(PRICES_DIR)
    price_feat = prepare_price_features(prices)
    price_feat["date"] = pd.to_datetime(price_feat["date"])

    targets = pd.read_csv(Path(TARGETS_DIR) / "direction_labels_multi_horizon.csv")
    vol = pd.read_csv(Path(TARGETS_DIR) / "volatility_targets.csv")
    surprise = pd.read_csv(Path(TARGETS_DIR) / "fundamental_surprise_targets.csv")

    # Scale
    train_mask = price_feat["date"] <= "2022-12-31"
    scaler = fit_scaler(price_feat[train_mask], ENGINEERED_FEATURES)
    price_feat = scaler.transform(price_feat, ENGINEERED_FEATURES)

    # Price dataset
    price_ds = PriceWindowDataset(
        price_feat,
        targets,
        vol_df=vol,
        surprise_df=surprise,
        window_size=60,
    )
    print(f"Price dataset: {len(price_ds)} samples")

    # === A) Price embeddings ===
    print("\n[1/5] Extracting price embeddings...")
    t1 = time.time()
    price_model = PriceDirectionModel(num_features=len(ENGINEERED_FEATURES))
    price_ckpt_path = Path(SAVE_DIR) / "phase12_price_model_best.pt"
    if not price_ckpt_path.exists():
        # Fallback: use original if phase12 not trained
        price_ckpt_path = Path(SAVE_DIR) / "price_model_best.pt"
        print(f"  WARNING: Using fallback {price_ckpt_path}")
    ckpt = torch.load(price_ckpt_path, map_location="cpu", weights_only=False)
    price_model.load_state_dict(ckpt["model_state_dict"])
    price_model.eval().to(dev)

    price_embs = {}
    loader = DataLoader(price_ds, batch_size=512, shuffle=False, num_workers=0)
    with torch.no_grad():
        for batch in loader:
            x = batch["features"].to(dev)
            tickers_b = batch["ticker"]
            dates_b = batch["date"]
            with torch.amp.autocast("cuda"):
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
    print("\n[2/5] Extracting GAT embeddings...")
    t2 = time.time()
    graph = load_graph(
        "data/raw/graph/tech30_nodes.csv", "data/raw/graph/tech30_edges.csv"
    )
    ei, ew, _ = make_bidirectional(
        graph["edge_index"], graph["edge_weight"], graph["edge_type"]
    )
    graph_tickers = graph["tickers"]

    # GAT needs direction_labels for GraphSnapshotDataset
    gat_targets = pd.read_csv(Path(TARGETS_DIR) / "direction_labels.csv")
    gat_targets["date"] = pd.to_datetime(gat_targets["date"])

    gat_ds = GraphSnapshotDataset(
        price_feat, gat_targets, graph_tickers, window_size=60
    )
    print(f"  Graph snapshots: {len(gat_ds)}")

    gat_model = GraphEnhancedModel(num_features=len(ENGINEERED_FEATURES))
    gat_ckpt_path = Path(SAVE_DIR) / "phase12_graph_model_best.pt"
    if not gat_ckpt_path.exists():
        gat_ckpt_path = Path(SAVE_DIR) / "graph_model_best.pt"
        print(f"  WARNING: Using fallback {gat_ckpt_path}")
    ckpt = torch.load(gat_ckpt_path, map_location="cpu", weights_only=False)
    gat_model.load_state_dict(ckpt["model_state_dict"])
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

    # === C) Document embeddings ===
    print("\n[3/5] Extracting document embeddings (FinBERT)...")
    t3 = time.time()
    doc_embs_raw = {}
    doc_model_path = Path(SAVE_DIR) / "document_model_best.pt"
    if doc_model_path.exists():
        from transformers import AutoTokenizer
        from src.data.document_dataset import DocumentChunkDataset
        from src.models.document_model import DocumentDirectionModel

        backbone = "ProsusAI/finbert"
        tokenizer = AutoTokenizer.from_pretrained(backbone)
        doc_ds = DocumentChunkDataset("data/processed", tokenizer, max_chunks=64)
        doc_model = DocumentDirectionModel(backbone_name=backbone)
        ckpt = torch.load(doc_model_path, map_location="cpu", weights_only=False)
        doc_model.load_state_dict(ckpt["model_state_dict"])
        doc_model.eval()
        doc_model.encoder = doc_model.encoder.half().to(dev)
        doc_model.pool = doc_model.pool.to(dev)

        with torch.no_grad():
            for idx in range(len(doc_ds)):
                sample = doc_ds[idx]
                ticker = sample["ticker"]
                year = sample["year"]
                n_chunks = sample["num_chunks"]
                ids = sample["input_ids"][:n_chunks]
                mask = sample["attention_mask"][:n_chunks]

                cls_list = []
                for start in range(0, n_chunks, 64):
                    end = min(start + 64, n_chunks)
                    mini_ids = ids[start:end].to(dev)
                    mini_mask = mask[start:end].to(dev)
                    with torch.amp.autocast("cuda"):
                        enc_out = doc_model.encoder(
                            input_ids=mini_ids, attention_mask=mini_mask
                        )
                        cls_list.append(enc_out.last_hidden_state[:, 0, :].float())
                    del mini_ids, mini_mask, enc_out

                chunk_cls = torch.cat(cls_list, dim=0)
                embedding = chunk_cls.mean(dim=0).cpu().numpy()
                doc_embs_raw[(ticker, year)] = embedding

        del doc_model
        torch.cuda.empty_cache()
    print(f"  {len(doc_embs_raw)} filing embeddings in {time.time()-t3:.1f}s")

    # Map doc embeddings to (ticker, date) using previous year's filing
    doc_embs = {}
    for ticker, date_str in price_embs:
        year = int(date_str[:4])
        filing_year = year - 1
        key = (ticker, filing_year)
        if key in doc_embs_raw:
            doc_embs[(ticker, date_str)] = doc_embs_raw[key]

    # === D) Macro embeddings ===
    print("\n[4/5] Extracting macro embeddings...")
    t4 = time.time()
    macro_embs = {}
    macro_csv = Path("data/raw/macro/macro_prices.csv")
    if macro_csv.exists():
        from src.data.macro_features import (
            load_macro_data,
            compute_macro_features,
            MacroFeatureScaler,
            MACRO_FEATURE_NAMES,
        )

        raw_macro = load_macro_data(macro_csv)
        macro_feat = compute_macro_features(raw_macro, lag_days=1)
        mscaler = MacroFeatureScaler()
        mscaler.fit(macro_feat, train_end="2022-12-31")
        macro_feat_norm = mscaler.transform(macro_feat)
        macro_feat_norm.index = pd.to_datetime(macro_feat_norm.index)

        macro_model_path = Path(SAVE_DIR) / "macro_model_best.pt"
        if macro_model_path.exists():
            from src.models.macro_model import MacroStateModel

            macro_model = MacroStateModel(input_dim=12, hidden_dim=64, output_dim=32)
            ckpt = torch.load(macro_model_path, map_location="cpu", weights_only=False)
            macro_model.load_state_dict(ckpt["model_state_dict"])
            macro_model.eval().to(dev)

            with torch.no_grad():
                for date_idx in macro_feat_norm.index:
                    date_str = str(date_idx.date())
                    feat = macro_feat_norm.loc[date_idx, MACRO_FEATURE_NAMES].values
                    if np.any(np.isnan(feat)):
                        continue
                    x = torch.tensor(feat, dtype=torch.float32).unsqueeze(0).to(dev)
                    with torch.amp.autocast("cuda"):
                        emb = macro_model.encode(x)
                    macro_embs[date_str] = emb.float().cpu().numpy()[0]

            del macro_model
            torch.cuda.empty_cache()
        else:
            for date_idx in macro_feat_norm.index:
                date_str = str(date_idx.date())
                feat = macro_feat_norm.loc[date_idx, MACRO_FEATURE_NAMES].values
                if np.any(np.isnan(feat)):
                    continue
                padded = np.zeros(32, dtype=np.float32)
                padded[:12] = feat.astype(np.float32)
                macro_embs[date_str] = padded

    print(f"  {len(macro_embs)} macro embeddings in {time.time()-t4:.1f}s")

    # === E) Surprise features ===
    print("\n[5/5] Building surprise features...")
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
        if key in doc_embs:
            all_doc[i] = doc_embs[key]
            all_mask[i, 2] = 1.0
        if date_str in macro_embs:
            all_macro[i] = macro_embs[date_str]
            all_mask[i, 3] = 1.0
        if key in surprise_feats:
            all_surprise[i] = surprise_feats[key]

    # Save
    save_path = Path(EMBEDDINGS_DIR) / "phase12_fusion_embeddings.pt"
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

    print(f"\nEmbeddings saved to {save_path}")
    print(f"  Total samples: {N}")
    print(f"  Price available: {int(all_mask[:, 0].sum())}")
    print(f"  GAT available: {int(all_mask[:, 1].sum())}")
    print(f"  Doc available: {int(all_mask[:, 2].sum())}")
    print(f"  Macro available: {int(all_mask[:, 3].sum())}")
    print(f"  Vol targets valid: {int((~np.isnan(all_vol_target)).sum())}")
    print(f"  Dir labels valid: {int((all_dir_label >= 0).sum())}")

    print("PHASE 12 STEP 9/10C/D COMPLETE — Embeddings extracted")
    return data


# ======================================================================
# 10E. Train Phase12FusionModel
# ======================================================================
def train_fusion_phase12():
    """Train Phase12FusionModel with QLIKE-primary loss."""
    from src.models.fusion_model import Phase12FusionModel
    from src.models.losses import CombinedVolatilityLoss
    from src.train.common import EarlyStopping, save_checkpoint

    print("\n" + "=" * 60)
    print("10E: TRAINING PHASE 12 FUSION MODEL (QLIKE-primary)")
    print("=" * 60)

    device = setup_gpu(verbose=True)
    dev = str(device)
    set_global_seed(42)

    # Load embeddings
    emb_path = Path(EMBEDDINGS_DIR) / "phase12_fusion_embeddings.pt"
    data = torch.load(emb_path, weights_only=False)
    print(f"Loaded embeddings: {data['price_emb'].shape[0]} samples")

    # Date-based splits
    dates = pd.to_datetime(pd.Series(data["dates"]))
    train_mask = (dates <= "2022-12-31").values
    val_mask = ((dates > "2022-12-31") & (dates <= "2023-12-31")).values
    test_mask = (dates > "2023-12-31").values

    # Build date indices for ListNet (map each date to an integer)
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

    train_ds = FusionDataset(train_idx)
    val_ds = FusionDataset(val_idx)
    test_ds = FusionDataset(test_idx)

    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=2048, shuffle=True, num_workers=0, drop_last=True
    )
    val_loader = DataLoader(val_ds, batch_size=4096, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=4096, shuffle=False, num_workers=0)

    # Model
    model = Phase12FusionModel(
        price_dim=256,
        gat_dim=256,
        doc_dim=768,
        macro_dim=32,
        surprise_dim=5,
        proj_dim=128,
        hidden_dim=256,
        dropout=0.3,
        mc_dropout=True,
    ).to(dev)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=50, eta_min=1e-6
    )
    criterion = CombinedVolatilityLoss(lambda_vol=0.85, lambda_dir=0.15)
    stopper = EarlyStopping(patience=10, mode="min")
    scaler_amp = create_grad_scaler()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Phase12FusionModel: {n_params:,} params")
    print(f"Loss: 0.85×QLIKE + 0.15×ListNet")

    save_path = Path(SAVE_DIR) / "phase12_fusion_model_best.pt"
    best_val_loss = float("inf")

    for epoch in range(1, 51):
        model.train()
        epoch_loss, epoch_vol_loss, epoch_dir_loss = 0.0, 0.0, 0.0
        n_batches = 0

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
                out = model(
                    price_emb, gat_emb, doc_emb, macro_emb, surprise_feat, modality_mask
                )
                losses = criterion(
                    out["direction_logits"],
                    dir_labels,
                    out["volatility_pred"],
                    vol_targets,
                    date_idx,
                )

            total_loss = losses["total"]
            check_nan(total_loss, f"fusion epoch {epoch}")

            scaler_amp.scale(total_loss).backward()
            scaler_amp.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler_amp.step(optimizer)
            scaler_amp.update()

            epoch_loss += total_loss.item()
            epoch_vol_loss += losses["volatility"].item()
            epoch_dir_loss += losses["direction"].item()
            n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)

        # Validation
        model.eval()
        val_loss_sum, val_vol_sum, val_n = 0.0, 0.0, 0
        val_vol_preds, val_vol_targets = [], []

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
                    out = model(
                        price_emb,
                        gat_emb,
                        doc_emb,
                        macro_emb,
                        surprise_feat,
                        modality_mask,
                    )
                    losses = criterion(
                        out["direction_logits"],
                        dir_labels,
                        out["volatility_pred"],
                        vol_targets,
                        date_idx,
                    )

                bs = price_emb.size(0)
                val_loss_sum += losses["total"].item() * bs
                val_vol_sum += losses["volatility"].item() * bs
                val_n += bs

                # Collect for R²
                valid = ~torch.isnan(vol_targets)
                if valid.any():
                    val_vol_preds.append(out["volatility_pred"][valid].cpu())
                    val_vol_targets.append(vol_targets[valid].cpu())

        val_loss = val_loss_sum / max(val_n, 1)

        # Val R²
        val_r2 = 0.0
        if val_vol_preds:
            vp = torch.cat(val_vol_preds).numpy()
            vt = torch.cat(val_vol_targets).numpy()
            ss_res = np.sum((vt - vp) ** 2)
            ss_tot = np.sum((vt - vt.mean()) ** 2)
            val_r2 = 1 - ss_res / max(ss_tot, 1e-8)

        if epoch % 5 == 0 or epoch == 1:
            alloc = torch.cuda.memory_allocated(0) / 1e9
            print(
                f"  Epoch {epoch:02d} | loss={avg_loss:.4f} vol={epoch_vol_loss/max(n_batches,1):.4f} "
                f"dir={epoch_dir_loss/max(n_batches,1):.4f} | val_loss={val_loss:.4f} "
                f"val_R²={val_r2:.4f} | VRAM={alloc:.2f}GB"
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                model,
                optimizer,
                epoch,
                {"val_loss": val_loss, "val_r2": val_r2},
                save_path,
            )

        if stopper(val_loss):
            print(f"  Early stopping at epoch {epoch}")
            break

    # === Test evaluation ===
    ckpt = torch.load(save_path, weights_only=False, map_location=dev)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    test_vol_preds, test_vol_targets = [], []
    test_dir_preds, test_dir_labels = [], []

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
                out = model(
                    price_emb, gat_emb, doc_emb, macro_emb, surprise_feat, modality_mask
                )

            valid_vol = ~torch.isnan(vol_targets)
            if valid_vol.any():
                test_vol_preds.append(out["volatility_pred"][valid_vol].cpu())
                test_vol_targets.append(vol_targets[valid_vol].cpu())

            valid_dir = dir_labels >= 0
            if valid_dir.any():
                probs = torch.softmax(out["direction_logits"][valid_dir], dim=1)[:, 1]
                test_dir_preds.append(probs.cpu())
                test_dir_labels.append(dir_labels[valid_dir].cpu())

    # Vol metrics
    vp = torch.cat(test_vol_preds).numpy()
    vt = torch.cat(test_vol_targets).numpy()
    ss_res = np.sum((vt - vp) ** 2)
    ss_tot = np.sum((vt - vt.mean()) ** 2)
    test_r2 = 1 - ss_res / max(ss_tot, 1e-8)
    test_mae = np.mean(np.abs(vt - vp))
    test_rmse = np.sqrt(np.mean((vt - vp) ** 2))

    # Dir metrics
    dp = torch.cat(test_dir_preds).numpy()
    dl = torch.cat(test_dir_labels).numpy()
    from sklearn.metrics import roc_auc_score, accuracy_score

    test_auc = roc_auc_score(dl, dp)
    test_dir_acc = accuracy_score(dl, (dp > 0.5).astype(int))

    print(f"\n{'='*60}")
    print("PHASE 12 TEST RESULTS")
    print(f"{'='*60}")
    print(f"  Volatility R²:  {test_r2:.4f}")
    print(f"  Volatility MAE: {test_mae:.4f}")
    print(f"  Volatility RMSE: {test_rmse:.4f}")
    print(f"  Direction AUC:  {test_auc:.4f}")
    print(f"  Direction Acc:  {test_dir_acc:.4f}")

    results = {
        "vol_r2": test_r2,
        "vol_mae": test_mae,
        "vol_rmse": test_rmse,
        "dir_auc": test_auc,
        "dir_acc": test_dir_acc,
    }

    del model
    torch.cuda.empty_cache()
    print("PHASE 12 STEP 10E COMPLETE — Fusion model trained")
    return results


# ======================================================================
# Main
# ======================================================================
if __name__ == "__main__":
    t_start = time.time()

    # 10A: Price model
    price_results = train_price_model_phase12()

    # 10B: GAT model
    gat_results = train_gat_model_phase12()

    # 10C/D: Extract embeddings
    emb_data = extract_phase12_embeddings()

    # 10E: Fusion model
    fusion_results = train_fusion_phase12()

    total_time = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"PHASE 12 FULL PIPELINE COMPLETE in {total_time/60:.1f} minutes")
    print(f"{'='*60}")
