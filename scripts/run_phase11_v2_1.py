"""Phase 11B: V2.1 — Expanded macro encoder (32d → 128d) + retrained fusion.

Steps:
1. Train expanded macro model (hidden=128, output=128)
2. Re-extract all embeddings with the new macro model
3. Retrain fusion model with macro_dim=128
4. Save checkpoint as models/fusion_v2_1_best.pt

Saves results to: models/phase11_v2_1_results.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.train.common import TrainingConfig
from src.utils.gpu import setup_gpu, log_gpu_usage


def train_expanded_macro(verbose: bool = True) -> dict:
    """Train a macro model with hidden_dim=128, output_dim=128."""
    from src.data.macro_features import load_macro_data, compute_macro_features, MACRO_FEATURE_NAMES
    from src.models.macro_model import MacroStateModel
    from src.train.common import EarlyStopping, save_checkpoint

    import numpy as np
    import pandas as pd

    device = setup_gpu(verbose=verbose)
    device_str = str(device)

    # Load macro features
    macro_df = load_macro_data()
    features_df = compute_macro_features(macro_df)

    # Load direction labels for training
    labels_df = pd.read_csv("data/targets/direction_labels_multi_horizon.csv")
    labels_df["date"] = pd.to_datetime(labels_df["date"]).dt.strftime("%Y-%m-%d")

    # Build training data: for each date, get macro features + majority label
    date_labels = labels_df.groupby("date")["direction_60d_id"].agg(
        lambda x: int(x.mode().iloc[0])
    ).to_dict()

    X_list, y_list, dates_list = [], [], []
    for date_idx in features_df.index:
        date_str = str(date_idx.date())
        if date_str not in date_labels:
            continue
        feat = features_df.loc[date_idx, MACRO_FEATURE_NAMES].values
        if np.any(np.isnan(feat)):
            continue
        X_list.append(feat.astype(np.float32))
        y_list.append(date_labels[date_str])
        dates_list.append(date_str)

    X = torch.tensor(np.array(X_list))
    y = torch.tensor(np.array(y_list))

    # Split
    train_mask = torch.tensor([d <= "2022-12-31" for d in dates_list])
    val_mask = torch.tensor([(d > "2022-12-31") & (d <= "2023-12-31") for d in dates_list])
    test_mask = torch.tensor([d > "2023-12-31" for d in dates_list])

    X_train, y_train = X[train_mask].to(device_str), y[train_mask].to(device_str)
    X_val, y_val = X[val_mask].to(device_str), y[val_mask].to(device_str)
    X_test, y_test = X[test_mask].to(device_str), y[test_mask].to(device_str)

    if verbose:
        print(f"  Expanded macro: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")

    # V2.1 expanded model
    model = MacroStateModel(input_dim=12, hidden_dim=128, output_dim=128, dropout=0.1).to(device_str)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    criterion = torch.nn.CrossEntropyLoss()
    stopper = EarlyStopping(patience=10, mode="min")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-5)

    best_val_loss = float("inf")
    save_path = Path("models/macro_v2_1_best.pt")

    for epoch in range(1, 51):
        model.train()
        optimizer.zero_grad()
        logits = model(X_train)
        loss = criterion(logits, y_train)
        loss.backward()
        optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(X_val)
            val_loss = criterion(val_logits, y_val).item()
            val_acc = (val_logits.argmax(1) == y_val).float().mean().item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, epoch,
                           {"val_loss": val_loss, "val_acc": val_acc}, save_path)

        if epoch % 10 == 0 and verbose:
            print(f"  Epoch {epoch:3d} | loss={loss.item():.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        if stopper(val_loss):
            if verbose:
                print(f"  Early stopping at epoch {epoch}")
            break

    # Test
    ckpt = torch.load(save_path, weights_only=False, map_location=device_str)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    with torch.no_grad():
        test_logits = model(X_test)
        test_acc = (test_logits.argmax(1) == y_test).float().mean().item()
        from sklearn.metrics import roc_auc_score
        test_probs = torch.softmax(test_logits, dim=1)[:, 1].cpu().numpy()
        test_auc = roc_auc_score(y_test.cpu().numpy(), test_probs)

    if verbose:
        print(f"  Expanded macro test: acc={test_acc:.4f} auc={test_auc:.4f}")
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {n_params:,}")

    return {"test_acc": test_acc, "test_auc": test_auc, "epochs": epoch}


def extract_v2_1_embeddings(verbose: bool = True) -> str:
    """Re-extract all embeddings with expanded 128d macro encoder."""
    from src.features.extract_embeddings import extract_all_embeddings

    save_path = "data/embeddings/fusion_embeddings_v2_1.pt"
    if verbose:
        print(f"\n  Extracting V2.1 embeddings (macro 128d)...")

    result = extract_all_embeddings(
        save_path=save_path,
        verbose=verbose,
        macro_model_path="models/macro_v2_1_best.pt",
        macro_hidden_dim=128,
        macro_output_dim=128,
    )

    if verbose:
        print(f"  Saved V2.1 embeddings: {result.get('total_samples', 'N/A')} samples")

    return save_path


def retrain_v2_1_fusion(embeddings_path: str, verbose: bool = True) -> dict:
    """Retrain fusion model with macro_dim=128."""
    from src.data.fusion_dataset import FusionEmbeddingDataset
    from src.models.fusion_model import MultimodalFusionModel
    from src.models.losses import CombinedLoss
    from src.train.common import EarlyStopping, save_checkpoint, make_dataloader, TrainingConfig
    from src.utils.seed import set_global_seed
    from torch.utils.data import Subset
    import numpy as np

    device = setup_gpu(verbose=verbose)
    device_str = str(device)
    set_global_seed(42)

    config = TrainingConfig(
        epochs=60, patience=10, batch_size=4096,
        learning_rate=5e-4, weight_decay=1e-3, seed=42, use_amp=True,
    )

    dataset = FusionEmbeddingDataset(embeddings_path)
    macro_dim = dataset.macro_emb.shape[1]
    if verbose:
        print(f"  Macro dim from embeddings: {macro_dim}")

    if device_str == "cuda":
        dataset.to_device(device_str)
        log_gpu_usage("  After GPU preload: ")

    # Split
    train_idx, val_idx, test_idx = [], [], []
    for i in range(len(dataset)):
        d = dataset.dates[i]
        lab = dataset.direction_label[i]
        if lab < 0:
            continue
        if d <= "2022-12-31":
            train_idx.append(i)
        elif d <= "2023-12-31":
            val_idx.append(i)
        else:
            test_idx.append(i)

    if verbose:
        print(f"  Split: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

    dl_config = TrainingConfig(**{**config.__dict__, "pin_memory": device_str != "cuda"})
    train_loader = make_dataloader(Subset(dataset, train_idx), batch_size=config.batch_size, shuffle=True, config=dl_config)
    val_loader = make_dataloader(Subset(dataset, val_idx), batch_size=config.batch_size * 2, config=dl_config)
    test_loader = make_dataloader(Subset(dataset, test_idx), batch_size=config.batch_size * 2, config=dl_config)

    # V2.1 model with macro_dim from data
    model = MultimodalFusionModel(
        price_dim=256, gat_dim=256, doc_dim=768,
        macro_dim=macro_dim, surprise_dim=5,
        proj_dim=128, hidden_dim=256, dropout=0.3,
    ).to(device_str)

    n_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"  V2.1 Fusion: {n_params:,} parameters (macro_dim={macro_dim})")
        log_gpu_usage("  After model: ")

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=1e-6)
    stopper = EarlyStopping(patience=config.patience, mode="min")
    criterion = CombinedLoss(lambda_dir=0.7, lambda_vol=0.3)
    use_amp = config.use_amp and device_str == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    save_path = Path("models/fusion_v2_1_best.pt")

    best_val_loss = float("inf")
    t0 = time.time()

    for epoch in range(1, config.epochs + 1):
        # Train
        model.train()
        loss_sum, n_correct, n_total = 0.0, 0, 0
        for batch in train_loader:
            price = batch["price_emb"].to(device_str, non_blocking=True)
            gat = batch["gat_emb"].to(device_str, non_blocking=True)
            doc = batch["doc_emb"].to(device_str, non_blocking=True)
            macro = batch["macro_emb"].to(device_str, non_blocking=True)
            surp = batch["surprise_feat"].to(device_str, non_blocking=True)
            mask = batch["modality_mask"].to(device_str, non_blocking=True)
            dir_label = batch["direction_label"].to(device_str, non_blocking=True)
            vol_target = batch["volatility_target"].to(device_str, non_blocking=True)
            date_idx = batch["date_index"].to(device_str, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=use_amp):
                out = model(price, gat, doc, macro, surp, mask)
                losses = criterion(out["direction_logits"], dir_label, out["volatility_pred"], vol_target, date_idx)

            if scaler:
                scaler.scale(losses["total"]).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                losses["total"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            optimizer.zero_grad()

            loss_sum += losses["total"].item() * price.size(0)
            valid = dir_label >= 0
            if valid.any():
                preds = out["direction_logits"][valid].argmax(1)
                n_correct += (preds == dir_label[valid]).sum().item()
                n_total += valid.sum().item()

        scheduler.step()
        train_loss = loss_sum / max(n_total, 1)
        train_acc = n_correct / max(n_total, 1)

        # Validate
        model.eval()
        val_loss_sum, val_correct, val_total = 0.0, 0, 0
        val_probs_list, val_labels_list = [], []

        with torch.no_grad():
            for batch in val_loader:
                price = batch["price_emb"].to(device_str, non_blocking=True)
                gat = batch["gat_emb"].to(device_str, non_blocking=True)
                doc = batch["doc_emb"].to(device_str, non_blocking=True)
                macro = batch["macro_emb"].to(device_str, non_blocking=True)
                surp = batch["surprise_feat"].to(device_str, non_blocking=True)
                mask_v = batch["modality_mask"].to(device_str, non_blocking=True)
                dir_label = batch["direction_label"].to(device_str, non_blocking=True)
                vol_target = batch["volatility_target"].to(device_str, non_blocking=True)
                date_idx = batch["date_index"].to(device_str, non_blocking=True)

                with torch.amp.autocast("cuda", enabled=use_amp):
                    out = model(price, gat, doc, macro, surp, mask_v)
                    losses = criterion(out["direction_logits"], dir_label, out["volatility_pred"], vol_target, date_idx)

                val_loss_sum += losses["total"].item() * price.size(0)
                valid = dir_label >= 0
                if valid.any():
                    val_correct += (out["direction_logits"][valid].argmax(1) == dir_label[valid]).sum().item()
                    val_total += valid.sum().item()
                    probs = torch.softmax(out["direction_logits"][valid].float(), dim=1)[:, 1]
                    val_probs_list.extend(probs.cpu().tolist())
                    val_labels_list.extend(dir_label[valid].cpu().tolist())

        val_loss = val_loss_sum / max(val_total, 1)
        val_acc = val_correct / max(val_total, 1)
        from sklearn.metrics import roc_auc_score
        val_auc = roc_auc_score(val_labels_list, val_probs_list) if len(set(val_labels_list)) > 1 else 0.5

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, epoch, {"val_loss": val_loss, "val_auc": val_auc}, save_path)

        if verbose:
            print(f"  Epoch {epoch:02d} | loss={train_loss:.4f} acc={train_acc:.4f} | val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_auc={val_auc:.4f}")
            if epoch == 1:
                log_gpu_usage("  ")
            elif epoch % 5 == 0:
                peak = torch.cuda.max_memory_allocated(0) / 1e9 if device_str == "cuda" else 0
                print(f"    VRAM peak: {peak:.3f} GB | LR: {optimizer.param_groups[0]['lr']:.6f}")

        if stopper(val_loss):
            if verbose:
                print(f"  Early stopping at epoch {epoch}")
            break

    # Test
    ckpt = torch.load(save_path, weights_only=False, map_location=device_str)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    test_probs_list, test_labels_list, test_vol_preds, test_vol_true = [], [], [], []
    all_gates = []

    with torch.no_grad():
        for batch in test_loader:
            price = batch["price_emb"].to(device_str, non_blocking=True)
            gat = batch["gat_emb"].to(device_str, non_blocking=True)
            doc = batch["doc_emb"].to(device_str, non_blocking=True)
            macro = batch["macro_emb"].to(device_str, non_blocking=True)
            surp = batch["surprise_feat"].to(device_str, non_blocking=True)
            mask_t = batch["modality_mask"].to(device_str, non_blocking=True)
            dir_label = batch["direction_label"].to(device_str, non_blocking=True)
            vol_target = batch["volatility_target"].to(device_str, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=use_amp):
                out = model(price, gat, doc, macro, surp, mask_t)

            all_gates.append(out["gate_weights"].float().cpu())
            valid = dir_label >= 0
            if valid.any():
                probs = torch.softmax(out["direction_logits"][valid].float(), dim=1)[:, 1]
                test_probs_list.extend(probs.cpu().tolist())
                test_labels_list.extend(dir_label[valid].cpu().tolist())

            vol_valid = ~torch.isnan(vol_target)
            if vol_valid.any():
                test_vol_preds.extend(out["volatility_pred"][vol_valid].float().cpu().tolist())
                test_vol_true.extend(vol_target[vol_valid].cpu().tolist())

    test_auc = roc_auc_score(test_labels_list, test_probs_list) if len(set(test_labels_list)) > 1 else 0.5
    test_acc = np.mean(np.array(test_labels_list) == (np.array(test_probs_list) >= 0.5).astype(int))
    from sklearn.metrics import f1_score
    test_f1 = f1_score(test_labels_list, (np.array(test_probs_list) >= 0.5).astype(int))

    # Volatility R²
    vol_preds_arr = np.array(test_vol_preds)
    vol_true_arr = np.array(test_vol_true)
    ss_res = np.sum((vol_true_arr - vol_preds_arr) ** 2)
    ss_tot = np.sum((vol_true_arr - vol_true_arr.mean()) ** 2)
    vol_r2 = 1 - ss_res / (ss_tot + 1e-8)
    vol_rmse = np.sqrt(np.mean((vol_true_arr - vol_preds_arr) ** 2))

    # Gate weights
    gates_cat = torch.cat(all_gates, dim=0)
    gate_means = gates_cat.mean(0).tolist()
    gate_names = ["price", "gat", "doc", "macro"]

    elapsed = time.time() - t0

    if verbose:
        print(f"\n  V2.1 TEST RESULTS:")
        print(f"    Direction AUC: {test_auc:.4f}")
        print(f"    Direction Acc: {test_acc:.4f}")
        print(f"    Direction F1:  {test_f1:.4f}")
        print(f"    Vol RMSE:      {vol_rmse:.4f}")
        print(f"    Vol R2:        {vol_r2:.4f}")
        print(f"    Gate weights:")
        for n, w in zip(gate_names, gate_means):
            print(f"      {n:12s} {w:.4f}")
        print(f"    Training time: {elapsed:.1f}s")

    return {
        "direction_auc": round(test_auc, 4),
        "direction_acc": round(test_acc, 4),
        "direction_f1": round(test_f1, 4),
        "vol_rmse": round(vol_rmse, 4),
        "vol_r2": round(vol_r2, 4),
        "gate_weights": {n: round(w, 4) for n, w in zip(gate_names, gate_means)},
        "params": n_params,
        "macro_dim": macro_dim,
        "epochs_trained": epoch,
        "training_time_s": round(elapsed, 1),
    }


def main():
    print("=" * 60)
    print("PHASE 11B: V2.1 — EXPANDED MACRO ENCODER (32d -> 128d)")
    print("=" * 60)

    # Step 1: Train expanded macro model
    print("\n[1/3] Training expanded macro model (hidden=128, output=128)...")
    macro_results = train_expanded_macro(verbose=True)

    # Step 2: Re-extract embeddings with 128d macro
    # We need to modify the extract_macro_embeddings call in extract_all_embeddings
    # The simplest approach: patch the function call
    print("\n[2/3] Re-extracting embeddings with 128d macro...")
    emb_path = extract_v2_1_embeddings(verbose=True)

    # Step 3: Train V2.1 fusion
    print("\n[3/3] Training V2.1 fusion model (macro_dim=128)...")
    fusion_results = retrain_v2_1_fusion(emb_path, verbose=True)

    # Save combined results
    all_results = {
        "macro_expansion": macro_results,
        "fusion_v2_1": fusion_results,
    }
    with open("models/phase11_v2_1_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved results to models/phase11_v2_1_results.json")
    print("\n=== PHASE 11B COMPLETE ===")


if __name__ == "__main__":
    main()
