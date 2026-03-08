#!/usr/bin/env python
"""Model Improvements — 4 enhancements to boost prediction quality.

Implements and evaluates:
  1. Multi-task loss weight tuning   (lower λ_vol)
  2. Dynamic graph with rolling correlation edges
  3. Longer prediction horizons      (20-day, 60-day)
  4. Calibrated ensemble of per-modality classifiers

All results are saved to ``models/improvement_results.json``.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.fusion_dataset import FusionEmbeddingDataset
from src.evaluation.metrics import classification_metrics, majority_baseline_accuracy
from src.train.common import EarlyStopping, TrainingConfig, save_checkpoint
from src.train.train_fusion import run_fusion_training
from src.utils.seed import set_global_seed

EMB_PATH = Path("data/embeddings/fusion_embeddings.pt")
RESULTS_PATH = Path("models/improvement_results.json")

TICKERS = [
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
]


# ======================================================================
#  Improvement 1 — Multi-task loss weight tuning
# ======================================================================


def improvement_1_loss_weights(verbose: bool = True) -> dict:
    """Try different loss-weight configurations and compare direction AUC."""
    print("\n" + "=" * 70)
    print("  IMPROVEMENT 1: Multi-task loss weight tuning")
    print("=" * 70)

    configs = [
        {"name": "low_vol_0.1", "dir": 1.0, "vol": 0.1, "surp": 1.0},
        {"name": "no_vol", "dir": 1.0, "vol": 0.0, "surp": 1.0},
        {"name": "dir_only", "dir": 1.0, "vol": 0.0, "surp": 0.0},
        {"name": "high_dir", "dir": 2.0, "vol": 0.1, "surp": 0.5},
    ]

    training_config = TrainingConfig(
        epochs=60,
        patience=10,
        batch_size=512,
        learning_rate=5e-4,
        weight_decay=1e-3,
        seed=42,
    )

    results: dict[str, dict] = {}
    best_auc, best_name = 0.0, ""

    for cfg in configs:
        name = cfg["name"]
        save_dir = Path("models") / f"imp1_{name}"
        save_dir.mkdir(parents=True, exist_ok=True)

        if verbose:
            print(
                f"\n--- Config: {name}  "
                f"(λ_dir={cfg['dir']}, λ_vol={cfg['vol']}, λ_surp={cfg['surp']}) ---"
            )

        result = run_fusion_training(
            embeddings_path=str(EMB_PATH),
            config=training_config,
            save_dir=str(save_dir),
            lambda_dir=cfg["dir"],
            lambda_vol=cfg["vol"],
            lambda_surp=cfg["surp"],
            verbose=verbose,
        )

        test = result["test_result"]
        d = test.get("direction", {})
        v = test.get("volatility", {})

        entry = {
            "lambda_dir": cfg["dir"],
            "lambda_vol": cfg["vol"],
            "lambda_surp": cfg["surp"],
            "direction_auc": d.get("auc", 0.5),
            "direction_acc": d.get("accuracy", 0),
            "direction_f1": d.get("f1", 0),
            "volatility_r2": v.get("r2", None),
            "epochs_trained": len(result["history"]),
        }
        results[name] = entry

        if entry["direction_auc"] > best_auc:
            best_auc = entry["direction_auc"]
            best_name = name

    results["best_config"] = best_name
    results["best_direction_auc"] = best_auc

    if verbose:
        print(f"\n  Best: {best_name} → direction AUC = {best_auc:.4f}")

    return results


# ======================================================================
#  Improvement 2 — Dynamic graph with rolling correlation
# ======================================================================


def improvement_2_dynamic_graph(verbose: bool = True) -> dict:
    """Retrain GAT with dynamic correlation-based edges, rebuild fusion."""
    print("\n" + "=" * 70)
    print("  IMPROVEMENT 2: Dynamic graph (rolling correlation)")
    print("=" * 70)

    from src.data.dynamic_graph import compute_rolling_correlation_graphs
    from src.data.graph_dataset import GraphSnapshotDataset, build_graph_snapshots
    from src.data.graph_utils import load_graph, make_bidirectional
    from src.data.preprocessing import fit_scaler
    from src.data.price_dataset import ENGINEERED_FEATURES
    from src.models.gat_model import GraphEnhancedModel

    set_global_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Step 1: Compute per-date dynamic edges
    print("\n  Step 1: Computing dynamic edges from rolling correlations...")
    t0 = time.time()
    dynamic_edges = compute_rolling_correlation_graphs(
        prices_dir="data/raw/prices",
        tickers=TICKERS,
        window=60,
        corr_threshold=0.3,
        verbose=verbose,
    )
    print(f"  Done in {time.time()-t0:.1f}s")

    # Fallback to static edges for dates not in dynamic_edges
    static_graph = load_graph()
    static_ei, static_ew, _ = make_bidirectional(
        static_graph["edge_index"],
        static_graph["edge_weight"],
        static_graph["edge_type"],
    )
    static_ei_d = static_ei.to(device)
    static_ew_d = static_ew.to(device)

    def _get_edges(date_str: str):
        if date_str in dynamic_edges:
            return (
                dynamic_edges[date_str]["edge_index"].to(device),
                dynamic_edges[date_str]["edge_weight"].to(device),
            )
        return static_ei_d, static_ew_d

    # Step 2: Build GraphSnapshotDataset (same as Phase 5)
    print("\n  Step 2: Building graph snapshot dataset...")
    price_df, targets_df, tickers = build_graph_snapshots(
        "data/raw/prices",
        "data/targets",
        TICKERS,
        window_size=60,
    )
    train_mask = price_df["date"] <= "2022-12-31"
    scaler = fit_scaler(price_df.loc[train_mask], ENGINEERED_FEATURES)
    price_df = scaler.transform(price_df, ENGINEERED_FEATURES)

    dataset = GraphSnapshotDataset(
        price_df,
        targets_df,
        TICKERS,
        window_size=60,
    )

    train_idx, val_idx, test_idx = [], [], []
    for i, s in enumerate(dataset._snapshots):
        d = s["date"]
        if d <= "2022-12-31":
            train_idx.append(i)
        elif d <= "2023-12-31":
            val_idx.append(i)
        else:
            test_idx.append(i)

    if verbose:
        print(
            f"  Snapshots — train: {len(train_idx)}, "
            f"val: {len(val_idx)}, test: {len(test_idx)}"
        )

    # Step 3: Train GAT with dynamic edges
    print("\n  Step 3: Training GAT with dynamic edges...")
    model = GraphEnhancedModel(
        num_features=10,
        encoder_dim=256,
        gat_hidden=64,
        gat_heads=4,
        gat_dropout=0.1,
        num_classes=2,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    stopper = EarlyStopping(patience=7)
    best_val_loss = float("inf")
    save_path = Path("models") / "dynamic_gat_best.pt"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, 31):
        # --- Train ---
        model.train()
        loss_sum, correct, total = 0.0, 0, 0
        perm = np.random.permutation(train_idx)

        for si in perm:
            snap = dataset[si]
            x = snap["features"].to(device)  # [N, T, F]
            y = snap["labels"].to(device)  # [N]
            m = snap["mask"].to(device)  # [N]
            ei, ew = _get_edges(snap["date"])

            if m.sum() == 0:
                continue

            logits = model(x, ei, ew)  # [N, 2]
            valid_logits = logits[m]
            valid_labels = y[m]

            loss = criterion(valid_logits, valid_labels)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            loss_sum += loss.item() * valid_labels.size(0)
            correct += (valid_logits.argmax(1) == valid_labels).sum().item()
            total += valid_labels.size(0)

        # --- Validate ---
        model.eval()
        v_preds, v_labels, v_probs = [], [], []
        v_loss_sum, v_total = 0.0, 0
        with torch.no_grad():
            for si in val_idx:
                snap = dataset[si]
                x = snap["features"].to(device)
                y = snap["labels"].to(device)
                m = snap["mask"].to(device)
                ei, ew = _get_edges(snap["date"])
                if m.sum() == 0:
                    continue
                logits = model(x, ei, ew)
                vl, vla = logits[m], y[m]
                loss = criterion(vl, vla)
                v_loss_sum += loss.item() * vla.size(0)
                v_preds.extend(vl.argmax(1).cpu().tolist())
                v_labels.extend(vla.cpu().tolist())
                v_probs.extend(torch.softmax(vl, 1)[:, 1].cpu().tolist())
                v_total += vla.size(0)

        v_loss = v_loss_sum / max(v_total, 1)
        v_cls = classification_metrics(
            np.array(v_labels),
            np.array(v_preds),
            np.array(v_probs),
        )

        if verbose:
            print(
                f"  Epoch {epoch:02d} | "
                f"t_loss={loss_sum/max(total,1):.4f} "
                f"t_acc={correct/max(total,1):.4f} | "
                f"v_loss={v_loss:.4f} v_acc={v_cls['accuracy']:.4f} "
                f"v_auc={v_cls.get('auc', 0.5):.4f}"
            )

        if v_loss < best_val_loss:
            best_val_loss = v_loss
            save_checkpoint(
                model,
                optimizer,
                epoch,
                {"val_loss": v_loss, **v_cls},
                save_path,
            )

        if stopper(v_loss):
            if verbose:
                print(f"  Early stopping at epoch {epoch}")
            break

    # --- Test ---
    ckpt = torch.load(save_path, weights_only=False, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    t_preds, t_labels, t_probs = [], [], []
    with torch.no_grad():
        for si in test_idx:
            snap = dataset[si]
            x = snap["features"].to(device)
            y = snap["labels"].to(device)
            m = snap["mask"].to(device)
            ei, ew = _get_edges(snap["date"])
            if m.sum() == 0:
                continue
            logits = model(x, ei, ew)
            vl, vla = logits[m], y[m]
            t_preds.extend(vl.argmax(1).cpu().tolist())
            t_labels.extend(vla.cpu().tolist())
            t_probs.extend(torch.softmax(vl, 1)[:, 1].cpu().tolist())

    gat_metrics = classification_metrics(
        np.array(t_labels),
        np.array(t_preds),
        np.array(t_probs),
    )
    gat_baseline = majority_baseline_accuracy(np.array(t_labels))

    if verbose:
        print(
            f"\n  Dynamic GAT test:  "
            f"acc={gat_metrics['accuracy']:.4f}  "
            f"auc={gat_metrics['auc']:.4f}  "
            f"baseline={gat_baseline:.4f}"
        )

    # Step 4: Re-extract GAT embeddings with dynamic model
    print("\n  Step 4: Re-extracting GAT embeddings with dynamic model...")
    gat_embeddings: dict[tuple[str, str], np.ndarray] = {}
    with torch.no_grad():
        for si in range(len(dataset)):
            snap = dataset[si]
            x = snap["features"].to(device)
            m = snap["mask"]
            date_str = snap["date"]
            ei, ew = _get_edges(date_str)

            # Forward through encoder + GAT (no classification head)
            h = model.encode_price(x)
            h = model.node_proj(h)
            h1 = F.elu(model.gat1(h, ei, ew))
            h = model.gat_norm1(h + h1)
            h2 = F.elu(model.gat2(h, ei, ew))
            h = model.gat_norm2(h + h2)

            emb = h.cpu().numpy()
            for j in range(len(TICKERS)):
                if m[j]:
                    gat_embeddings[(TICKERS[j], date_str)] = emb[j]

    print(f"  Dynamic GAT embeddings: {len(gat_embeddings)} entries")

    # Step 5: Rebuild fusion embeddings with new GAT embeddings
    print("  Replacing GAT embeddings in fusion dataset...")
    emb_data = torch.load(str(EMB_PATH), weights_only=False, map_location="cpu")
    tickers_list = emb_data["tickers"]
    dates_list = emb_data["dates"]

    new_gat = torch.zeros_like(emb_data["gat_emb"])
    new_mask = emb_data["modality_mask"].clone()

    replaced = 0
    for i in range(len(tickers_list)):
        key = (tickers_list[i], dates_list[i])
        if key in gat_embeddings:
            new_gat[i] = torch.from_numpy(gat_embeddings[key])
            new_mask[i, 1] = 1.0
            replaced += 1

    emb_data["gat_emb"] = new_gat
    emb_data["modality_mask"] = new_mask

    dyn_emb_path = Path("data/embeddings/fusion_embeddings_dynamic_gat.pt")
    dyn_emb_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(emb_data, dyn_emb_path)
    print(f"  Replaced {replaced}/{len(tickers_list)} GAT embeddings → {dyn_emb_path}")

    # Step 6: Retrain fusion with new embeddings
    print("\n  Step 6: Retraining fusion with dynamic GAT embeddings...")
    fusion_result = run_fusion_training(
        embeddings_path=str(dyn_emb_path),
        config=TrainingConfig(
            epochs=60,
            patience=10,
            batch_size=512,
            learning_rate=5e-4,
            weight_decay=1e-3,
            seed=42,
        ),
        save_dir="models/imp2_dynamic_gat",
        lambda_dir=1.0,
        lambda_vol=0.1,
        lambda_surp=1.0,
        verbose=verbose,
    )

    ft = fusion_result["test_result"]
    fd = ft.get("direction", {})

    torch.cuda.empty_cache()

    return {
        "gat_test_metrics": {k: round(float(v), 4) for k, v in gat_metrics.items()},
        "gat_baseline": round(float(gat_baseline), 4),
        "fusion_direction_auc": fd.get("auc", 0.5),
        "fusion_direction_acc": fd.get("accuracy", 0),
        "fusion_direction_f1": fd.get("f1", 0),
        "fusion_volatility_r2": ft.get("volatility", {}).get("r2", None),
        "n_dynamic_dates": len(dynamic_edges),
        "gat_embeddings_replaced": replaced,
    }


# ======================================================================
#  Improvement 3 — Longer prediction horizons (20d, 60d)
# ======================================================================


def improvement_3_longer_horizons(verbose: bool = True) -> dict:
    """Generate 20-day and 60-day targets, retrain fusion for each."""
    print("\n" + "=" * 70)
    print("  IMPROVEMENT 3: Longer prediction horizons (20d, 60d)")
    print("=" * 70)

    import pandas as pd
    from src.data.price_dataset import load_price_csv_dir
    from src.data.target_builder import build_multi_horizon_direction_labels

    set_global_seed(42)

    # Step 1: Generate 20d and 60d direction labels
    print("\n  Step 1: Generating 20d and 60d direction labels...")
    prices = load_price_csv_dir("data/raw/prices")
    price_subset = prices[["ticker", "date", "close"]].copy()

    labels = build_multi_horizon_direction_labels(
        prices=price_subset,
        horizons=[20, 60],
        threshold=0.0,
    )
    labels["date"] = pd.to_datetime(labels["date"])

    # Build lookups
    dir_20d_lookup: dict[tuple[str, str], int] = {}
    dir_60d_lookup: dict[tuple[str, str], int] = {}

    for _, row in labels.iterrows():
        key = (row["ticker"], str(row["date"].date()))
        v20 = row.get("direction_20d_id")
        v60 = row.get("direction_60d_id")
        if pd.notna(v20):
            dir_20d_lookup[key] = int(v20)
        if pd.notna(v60):
            dir_60d_lookup[key] = int(v60)

    if verbose:
        print(f"  20d labels: {len(dir_20d_lookup)} entries")
        print(f"  60d labels: {len(dir_60d_lookup)} entries")
        # Class balance
        vals20 = list(dir_20d_lookup.values())
        vals60 = list(dir_60d_lookup.values())
        print(f"  20d UP fraction: {sum(vals20)/len(vals20):.3f}")
        print(f"  60d UP fraction: {sum(vals60)/len(vals60):.3f}")

    # Step 2: Load base fusion embeddings
    emb_data = torch.load(str(EMB_PATH), weights_only=False, map_location="cpu")
    tickers_list = emb_data["tickers"]
    dates_list = emb_data["dates"]

    horizon_results: dict[str, dict] = {}

    for horizon, lookup in [("20d", dir_20d_lookup), ("60d", dir_60d_lookup)]:
        print(f"\n  Step 2: Training fusion for {horizon} horizon...")

        # Replace direction_label
        new_labels = torch.full((len(tickers_list),), -1, dtype=torch.long)
        matched = 0
        for i in range(len(tickers_list)):
            key = (tickers_list[i], dates_list[i])
            if key in lookup:
                new_labels[i] = lookup[key]
                matched += 1

        if verbose:
            valid = (new_labels >= 0).sum().item()
            print(f"    Matched {matched} labels, {valid} valid for {horizon}")

        # Save modified embeddings
        emb_copy = {
            k: v.clone() if isinstance(v, torch.Tensor) else list(v)
            for k, v in emb_data.items()
        }
        emb_copy["direction_label"] = new_labels

        horizon_path = Path(f"data/embeddings/fusion_embeddings_{horizon}.pt")
        horizon_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(emb_copy, horizon_path)

        # Train fusion
        result = run_fusion_training(
            embeddings_path=str(horizon_path),
            config=TrainingConfig(
                epochs=60,
                patience=10,
                batch_size=512,
                learning_rate=5e-4,
                weight_decay=1e-3,
                seed=42,
            ),
            save_dir=f"models/imp3_{horizon}",
            lambda_dir=1.0,
            lambda_vol=0.1,
            lambda_surp=1.0,
            verbose=verbose,
        )

        test = result["test_result"]
        d = test.get("direction", {})

        horizon_results[horizon] = {
            "direction_auc": d.get("auc", 0.5),
            "direction_acc": d.get("accuracy", 0),
            "direction_f1": d.get("f1", 0),
            "n_valid_labels": int((new_labels >= 0).sum().item()),
            "epochs_trained": len(result["history"]),
            "gate_weights": test.get("mean_gate_weights", []),
        }

    return horizon_results


# ======================================================================
#  Improvement 4 — Calibrated ensemble of per-modality classifiers
# ======================================================================


class _ModalityMLP(nn.Module):
    """Simple MLP classifier for a single modality's embeddings."""

    def __init__(self, in_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def improvement_4_ensemble(verbose: bool = True) -> dict:
    """Train per-modality classifiers, calibrate, and build an ensemble."""
    print("\n" + "=" * 70)
    print("  IMPROVEMENT 4: Calibrated ensemble of per-modality classifiers")
    print("=" * 70)

    from sklearn.linear_model import LogisticRegression

    set_global_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load fusion embeddings
    dataset = FusionEmbeddingDataset(str(EMB_PATH))

    # Temporal split
    train_ids, val_ids, test_ids = [], [], []
    for i in range(len(dataset)):
        d = dataset.dates[i]
        lbl = dataset.direction_label[i].item()
        if lbl < 0:
            continue
        if d <= "2022-12-31":
            train_ids.append(i)
        elif d <= "2023-12-31":
            val_ids.append(i)
        else:
            test_ids.append(i)

    if verbose:
        print(
            f"  Samples — train: {len(train_ids)}, "
            f"val: {len(val_ids)}, test: {len(test_ids)}"
        )

    modalities = {
        "price": ("price_emb", 256),
        "gat": ("gat_emb", 256),
        "doc": ("doc_emb", 768),
        "news": ("news_emb", 512),
        "surprise": ("surprise_feat", 3),
    }

    modality_val_probs: dict[str, np.ndarray] = {}
    modality_test_probs: dict[str, np.ndarray] = {}
    modality_val_aucs: dict[str, float] = {}
    modality_test_aucs: dict[str, float] = {}

    # Get global labels
    train_labels = dataset.direction_label[train_ids].numpy()
    val_labels = dataset.direction_label[val_ids].numpy()
    test_labels = dataset.direction_label[test_ids].numpy()
    mask_tensor = dataset.modality_mask

    for mod_name, (attr_name, dim) in modalities.items():
        print(f"\n  --- Training {mod_name} classifier (dim={dim}) ---")

        # Get embeddings
        emb_tensor = getattr(dataset, attr_name)
        mod_idx = list(modalities.keys()).index(mod_name)

        # Filter to samples where this modality is available
        train_mask_mod = mask_tensor[train_ids, mod_idx] > 0.5
        val_mask_mod = mask_tensor[val_ids, mod_idx] > 0.5
        test_mask_mod = mask_tensor[test_ids, mod_idx] > 0.5

        train_subset = [
            train_ids[j] for j in range(len(train_ids)) if train_mask_mod[j]
        ]
        val_subset = [val_ids[j] for j in range(len(val_ids)) if val_mask_mod[j]]
        test_subset = [test_ids[j] for j in range(len(test_ids)) if test_mask_mod[j]]

        if not train_subset or not val_subset or not test_subset:
            print(f"    Skipping {mod_name}: insufficient samples")
            continue

        X_train = emb_tensor[train_subset].to(device)
        y_train = dataset.direction_label[train_subset].to(device)
        X_val = emb_tensor[val_subset].to(device)
        y_val = dataset.direction_label[val_subset].numpy()
        X_test = emb_tensor[test_subset].to(device)
        y_test = dataset.direction_label[test_subset].numpy()

        # Train MLP
        model = _ModalityMLP(dim, hidden_dim=max(64, dim // 4)).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
        ce = nn.CrossEntropyLoss()
        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(1, 51):
            model.train()
            # Mini-batch training
            perm = torch.randperm(len(train_subset), device=device)
            epoch_loss = 0.0
            for start in range(0, len(perm), 512):
                end = min(start + 512, len(perm))
                idx = perm[start:end]
                logits = model(X_train[idx])
                loss = ce(logits, y_train[idx])
                opt.zero_grad()
                loss.backward()
                opt.step()
                epoch_loss += loss.item() * len(idx)

            # Validate
            model.eval()
            with torch.no_grad():
                v_logits = model(X_val)
                v_loss = ce(v_logits, torch.tensor(y_val, device=device)).item()

            if v_loss < best_val_loss:
                best_val_loss = v_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= 8:
                    break

        # Load best and evaluate
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        model.eval()

        with torch.no_grad():
            val_logits = model(X_val)
            val_probs = torch.softmax(val_logits, 1)[:, 1].cpu().numpy()
            test_logits = model(X_test)
            test_probs_raw = torch.softmax(test_logits, 1)[:, 1].cpu().numpy()

        # Platt scaling calibration on validation set
        cal = LogisticRegression(C=1.0, solver="lbfgs", max_iter=200)
        cal.fit(val_probs.reshape(-1, 1), y_val)
        test_probs = cal.predict_proba(test_probs_raw.reshape(-1, 1))[:, 1]

        val_cls = classification_metrics(
            y_val,
            (val_probs > 0.5).astype(int),
            val_probs,
        )
        test_cls = classification_metrics(
            y_test,
            (test_probs > 0.5).astype(int),
            test_probs,
        )

        modality_val_aucs[mod_name] = val_cls.get("auc", 0.5)
        modality_test_aucs[mod_name] = test_cls.get("auc", 0.5)

        # Store test probs aligned to full test set (NaN where modality unavailable)
        full_test_probs = np.full(len(test_ids), np.nan)
        test_local_idx = 0
        for j in range(len(test_ids)):
            if test_mask_mod[j]:
                full_test_probs[j] = test_probs[test_local_idx]
                test_local_idx += 1
        modality_test_probs[mod_name] = full_test_probs

        full_val_probs = np.full(len(val_ids), np.nan)
        val_local_idx = 0
        for j in range(len(val_ids)):
            if val_mask_mod[j]:
                full_val_probs[j] = val_probs[val_local_idx]
                val_local_idx += 1
        modality_val_probs[mod_name] = full_val_probs

        if verbose:
            print(
                f"    Val  AUC: {val_cls.get('auc', 0.5):.4f}  "
                f"Acc: {val_cls['accuracy']:.4f}"
            )
            print(
                f"    Test AUC: {test_cls.get('auc', 0.5):.4f}  "
                f"Acc: {test_cls['accuracy']:.4f}"
            )

    del model, X_train, X_val, X_test
    torch.cuda.empty_cache()

    # ---- Combine via AUC-proportional weights ----
    print("\n  --- Building ensemble ---")

    # Weight each modality by (val_AUC - 0.5), floored at 0
    weights: dict[str, float] = {}
    for mod in modality_val_aucs:
        w = max(modality_val_aucs[mod] - 0.5, 0.0)
        weights[mod] = w

    total_w = sum(weights.values())
    if total_w > 0:
        weights = {k: v / total_w for k, v in weights.items()}
    else:
        n = len(weights)
        weights = {k: 1.0 / n for k in weights}

    if verbose:
        print("  Ensemble weights (AUC-proportional):")
        for mod, w in sorted(weights.items(), key=lambda x: -x[1]):
            print(f"    {mod:12s}  w={w:.4f}  val_auc={modality_val_aucs[mod]:.4f}")

    # Combine test predictions
    ensemble_probs = np.zeros(len(test_ids))
    weight_sums = np.zeros(len(test_ids))

    for mod, probs in modality_test_probs.items():
        valid = ~np.isnan(probs)
        ensemble_probs[valid] += weights.get(mod, 0.0) * probs[valid]
        weight_sums[valid] += weights.get(mod, 0.0)

    # Normalise
    nonzero = weight_sums > 0
    ensemble_probs[nonzero] /= weight_sums[nonzero]
    ensemble_probs[~nonzero] = 0.5

    ensemble_preds = (ensemble_probs > 0.5).astype(int)
    ensemble_cls = classification_metrics(test_labels, ensemble_preds, ensemble_probs)
    ensemble_baseline = majority_baseline_accuracy(test_labels)

    if verbose:
        print(
            f"\n  Ensemble test:  "
            f"acc={ensemble_cls['accuracy']:.4f}  "
            f"auc={ensemble_cls['auc']:.4f}  "
            f"f1={ensemble_cls['f1']:.4f}  "
            f"baseline={ensemble_baseline:.4f}"
        )

    return {
        "per_modality_val_auc": {k: round(v, 4) for k, v in modality_val_aucs.items()},
        "per_modality_test_auc": {
            k: round(v, 4) for k, v in modality_test_aucs.items()
        },
        "ensemble_weights": {k: round(v, 4) for k, v in weights.items()},
        "ensemble_test_auc": ensemble_cls.get("auc", 0.5),
        "ensemble_test_acc": ensemble_cls["accuracy"],
        "ensemble_test_f1": ensemble_cls["f1"],
        "ensemble_baseline": ensemble_baseline,
    }


# ======================================================================
#  Main
# ======================================================================


def main() -> None:
    set_global_seed(42)
    t_start = time.time()

    print("=" * 70)
    print("  MODEL IMPROVEMENTS — 4 enhancements")
    print("=" * 70)

    results: dict[str, dict] = {}

    # Original baseline
    results["baseline"] = {
        "direction_auc": 0.5161,
        "volatility_r2": 0.4115,
        "note": "Original fusion (λ_dir=1.0, λ_vol=0.5, λ_surp=1.0)",
    }

    # Run all improvements
    results["1_loss_weights"] = improvement_1_loss_weights()
    results["2_dynamic_graph"] = improvement_2_dynamic_graph()
    results["3_longer_horizons"] = improvement_3_longer_horizons()
    results["4_ensemble"] = improvement_4_ensemble()

    # Save
    results["total_time_seconds"] = round(time.time() - t_start, 1)

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Print summary
    print("\n" + "=" * 70)
    print("  SUMMARY — All improvements")
    print("=" * 70)

    bl_auc = results["baseline"]["direction_auc"]
    print(f"\n  Baseline direction AUC: {bl_auc:.4f}")

    # Improvement 1
    imp1 = results["1_loss_weights"]
    best_cfg = imp1.get("best_config", "?")
    best_auc = imp1.get("best_direction_auc", 0)
    print(f"\n  1. Loss weight tuning:")
    print(
        f"     Best config: {best_cfg} → AUC = {best_auc:.4f}  "
        f"(Δ = {best_auc - bl_auc:+.4f})"
    )

    # Improvement 2
    imp2 = results["2_dynamic_graph"]
    dyn_gat_auc = imp2.get("gat_test_metrics", {}).get("auc", 0)
    dyn_fus_auc = imp2.get("fusion_direction_auc", 0)
    print(f"\n  2. Dynamic graph:")
    print(f"     GAT-only AUC: {dyn_gat_auc:.4f}")
    print(
        f"     Fusion AUC:   {dyn_fus_auc:.4f}  " f"(Δ = {dyn_fus_auc - bl_auc:+.4f})"
    )

    # Improvement 3
    imp3 = results["3_longer_horizons"]
    for h in ["20d", "60d"]:
        if h in imp3:
            h_auc = imp3[h].get("direction_auc", 0)
            print(f"\n  3. {h} horizon: AUC = {h_auc:.4f}")

    # Improvement 4
    imp4 = results["4_ensemble"]
    ens_auc = imp4.get("ensemble_test_auc", 0)
    print(f"\n  4. Ensemble: AUC = {ens_auc:.4f}  " f"(Δ = {ens_auc - bl_auc:+.4f})")

    print(f"\n  Total time: {results['total_time_seconds']:.0f}s")
    print(f"  Results saved to {RESULTS_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    main()
