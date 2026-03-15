"""Phase 14 Pipeline: HAR-RV Skip Connection + Fixed Volatility Strategy.

Two goals:
  A. Close remaining gap to HAR-RV via skip connection architecture
  B. Fix vol trading strategy using VIX × beta as implied vol proxy

Sequence:
  Step 2: GPU verification
  Step 4: Extract HAR-RV raw features from price sequences
  Step 5: Train Phase14FusionModel with skip connection
  Step 6: Run benchmark comparison
  Step 7: VIX-adjusted vol strategy + full backtest
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
from torch.utils.data import DataLoader

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

# HAR-RV feature indices in the 21-feature tensor
HAR_RV_INDICES = [18, 19, 20]  # rv_lag1d, rv_lag5d, rv_lag22d


def check_nan(loss_val, step_name):
    if torch.is_tensor(loss_val):
        loss_val = loss_val.item()
    if np.isnan(loss_val) or np.isinf(loss_val):
        raise RuntimeError(f"NaN/Inf loss detected in {step_name}! Aborting.")


# ======================================================================
# Step 2: GPU Verification
# ======================================================================


def gpu_verification():
    print("=" * 60)
    print("PHASE 14 STEP 2: GPU VERIFICATION")
    print("=" * 60)

    device = setup_gpu(verbose=True)
    assert str(device) == "cuda", "CUDA required"

    props = torch.cuda.get_device_properties(0)
    print(f"\nGPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {props.total_memory / 1e9:.2f}GB")
    print("PHASE 14 STEP 2 COMPLETE — GPU verified")
    return device


# ======================================================================
# Step 4: Extract HAR-RV Raw Features
# ======================================================================


def extract_har_rv_raw():
    """Extract raw HAR-RV features (rv_lag1d, rv_lag5d, rv_lag22d) from
    the last timestep of each 60-day price window.

    These are saved alongside the Phase 13 embeddings for the skip connection.
    """
    from src.data.preprocessing import fit_scaler
    from src.data.price_dataset import (
        ENGINEERED_FEATURES,
        PriceWindowDataset,
        load_price_csv_dir,
        prepare_price_features,
    )

    print("\n" + "=" * 60)
    print("PHASE 14 STEP 4: EXTRACT HAR-RV RAW FEATURES")
    print("=" * 60)

    device = setup_gpu(verbose=False)
    set_global_seed(42)

    # Verify feature indices
    assert ENGINEERED_FEATURES[18] == "rv_lag1d"
    assert ENGINEERED_FEATURES[19] == "rv_lag5d"
    assert ENGINEERED_FEATURES[20] == "rv_lag22d"
    print(
        f"HAR-RV features at indices {HAR_RV_INDICES}: "
        f"{[ENGINEERED_FEATURES[i] for i in HAR_RV_INDICES]}"
    )

    # Load price data (same pipeline as Phase 13)
    prices = load_price_csv_dir(PRICES_DIR)
    price_feat = prepare_price_features(prices)
    price_feat["date"] = pd.to_datetime(price_feat["date"])

    targets = pd.read_csv(Path(TARGETS_DIR) / "direction_labels_multi_horizon.csv")
    vol = pd.read_csv(Path(TARGETS_DIR) / "volatility_targets.csv")
    surprise = pd.read_csv(Path(TARGETS_DIR) / "fundamental_surprise_targets.csv")

    # Scale using train only (same scaler as Phase 13)
    train_mask = price_feat["date"] <= "2022-12-31"
    scaler = fit_scaler(price_feat[train_mask], ENGINEERED_FEATURES)
    price_feat = scaler.transform(price_feat, ENGINEERED_FEATURES)

    price_ds = PriceWindowDataset(
        price_feat,
        targets,
        vol_df=vol,
        surprise_df=surprise,
        window_size=60,
    )
    print(f"Price dataset: {len(price_ds)} samples")

    # Load Phase 13 embeddings to get alignment
    emb = torch.load(
        f"{EMBEDDINGS_DIR}/phase13_fusion_embeddings.pt", weights_only=False
    )
    emb_keys = set()
    for i in range(len(emb["tickers"])):
        emb_keys.add((emb["tickers"][i], emb["dates"][i]))
    print(f"Phase 13 embedding keys: {len(emb_keys)}")

    # Extract HAR-RV raw features from last timestep of each window
    loader = DataLoader(price_ds, batch_size=512, shuffle=False, num_workers=0)
    har_rv_dict = {}
    t0 = time.time()

    for batch in loader:
        features = batch["features"]  # [B, 60, 21]
        tickers_b = batch["ticker"]
        dates_b = batch["date"]

        # Last timestep, HAR-RV indices
        har_rv_batch = features[:, -1, HAR_RV_INDICES]  # [B, 3]

        for i in range(len(tickers_b)):
            key = (tickers_b[i], dates_b[i])
            har_rv_dict[key] = har_rv_batch[i].numpy()

    print(f"Extracted {len(har_rv_dict)} HAR-RV raw features in {time.time()-t0:.1f}s")

    # Align with Phase 13 embeddings
    N = len(emb["tickers"])
    har_rv_aligned = np.zeros((N, 3), dtype=np.float32)
    matched = 0
    for i in range(N):
        key = (emb["tickers"][i], emb["dates"][i])
        if key in har_rv_dict:
            har_rv_aligned[i] = har_rv_dict[key]
            matched += 1

    print(f"Aligned: {matched}/{N} samples ({matched/N:.1%})")

    # Sanity checks
    har_rv_t = torch.from_numpy(har_rv_aligned)
    print(f"\nHAR-RV raw shape: {har_rv_t.shape}")
    for idx, name in zip(range(3), ["rv_lag1d", "rv_lag5d", "rv_lag22d"]):
        col = har_rv_aligned[:, idx]
        valid = col[col != 0]
        print(
            f"  {name}: mean={valid.mean():.4f}, std={valid.std():.4f}, "
            f"min={valid.min():.4f}, max={valid.max():.4f}"
        )

    # Correlation with vol target
    vol_targets = emb["volatility_target"].numpy()
    valid_mask = (~np.isnan(vol_targets)) & (har_rv_aligned[:, 0] != 0)
    if valid_mask.sum() > 100:
        from scipy.stats import spearmanr

        corr_lag1 = spearmanr(har_rv_aligned[valid_mask, 0], vol_targets[valid_mask])[0]
        corr_lag5 = spearmanr(har_rv_aligned[valid_mask, 1], vol_targets[valid_mask])[0]
        corr_lag22 = spearmanr(har_rv_aligned[valid_mask, 2], vol_targets[valid_mask])[
            0
        ]
        print(f"\nCorrelation with vol target:")
        print(f"  rv_lag1d:  {corr_lag1:.4f}")
        print(f"  rv_lag5d:  {corr_lag5:.4f}")
        print(f"  rv_lag22d: {corr_lag22:.4f}")

    # Save
    save_path = Path(EMBEDDINGS_DIR) / "phase14_har_rv_raw.pt"
    torch.save(har_rv_t, save_path)
    print(f"\nSaved: {save_path} shape={har_rv_t.shape}")
    print("PHASE 14 STEP 4 COMPLETE — HAR-RV raw features extracted")
    return har_rv_t


# ======================================================================
# Step 5: Train Phase14FusionModel
# ======================================================================


def train_phase14_fusion():
    """Train Phase14FusionModel with HAR-RV skip connection."""
    from src.models.fusion_model import Phase14FusionModel
    from src.models.losses import CombinedVolatilityLoss
    from src.train.common import EarlyStopping, save_checkpoint
    from sklearn.metrics import roc_auc_score

    print("\n" + "=" * 60)
    print("PHASE 14 STEP 5: TRAIN Phase14FusionModel (SKIP CONNECTION)")
    print("=" * 60)

    device = setup_gpu(verbose=True)
    dev = str(device)
    set_global_seed(42)

    # Load embeddings + HAR-RV raw features
    emb_path = Path(EMBEDDINGS_DIR) / "phase13_fusion_embeddings.pt"
    data = torch.load(emb_path, weights_only=False)
    har_rv = torch.load(
        Path(EMBEDDINGS_DIR) / "phase14_har_rv_raw.pt", weights_only=False
    )
    print(f"Loaded embeddings: {data['price_emb'].shape[0]} samples")
    print(f"HAR-RV raw: {har_rv.shape}")

    # Date-based splits
    dates = pd.to_datetime(pd.Series(data["dates"]))
    train_mask = (dates <= "2022-12-31").values
    val_mask = ((dates > "2022-12-31") & (dates <= "2023-12-31")).values
    test_mask = (dates > "2023-12-31").values

    # Date indices for ListNet
    date_strs = data["dates"]
    unique_dates = sorted(set(date_strs))
    date_to_idx = {d: i for i, d in enumerate(unique_dates)}
    date_indices = torch.tensor([date_to_idx[d] for d in date_strs], dtype=torch.long)

    class Phase14FusionDataset(torch.utils.data.Dataset):
        def __init__(self, indices):
            self.indices = indices

        def __len__(self):
            return len(self.indices)

        def __getitem__(self, idx):
            i = self.indices[idx]
            return {
                "price_emb": data["price_emb"][i],
                "har_rv_raw": har_rv[i],
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

    train_ds = Phase14FusionDataset(train_idx)
    val_ds = Phase14FusionDataset(val_idx)
    test_ds = Phase14FusionDataset(test_idx)

    train_loader = DataLoader(
        train_ds, batch_size=4096, shuffle=True, num_workers=0, drop_last=True
    )
    val_loader = DataLoader(val_ds, batch_size=4096, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=4096, shuffle=False, num_workers=0)

    # Model
    model = Phase14FusionModel(
        price_dim=256,
        har_rv_dim=3,
        har_proj_dim=32,
        gat_dim=256,
        doc_dim=768,
        macro_dim=32,
        surprise_dim=5,
        proj_dim=128,
        hidden_dim=256,
        dropout=0.3,
        trunk_dropout=0.3,
        mc_dropout=True,
        lambda_vol=0.85,
        lambda_dir=0.15,
    ).to(dev)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Phase14FusionModel: {n_params:,} params")

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=60, eta_min=1e-6
    )
    criterion = CombinedVolatilityLoss(lambda_vol=0.85, lambda_dir=0.15)
    stopper = EarlyStopping(patience=7, mode="min")
    scaler_amp = create_grad_scaler()

    save_path = Path(SAVE_DIR) / "phase14_fusion_best.pt"
    best_val_loss = float("inf")
    skip_contributions = []
    peak_vram = 0.0

    for epoch in range(1, 61):
        model.train()
        epoch_loss, n_batches = 0.0, 0

        for batch in train_loader:
            price_emb = batch["price_emb"].to(dev)
            har_rv_raw = batch["har_rv_raw"].to(dev)
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
                    price_emb,
                    har_rv_raw,
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

            total_loss = losses["total"]
            check_nan(total_loss, f"phase14 epoch {epoch}")

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
        val_gate_weights = []

        with torch.no_grad():
            for batch in val_loader:
                price_emb = batch["price_emb"].to(dev)
                har_rv_raw = batch["har_rv_raw"].to(dev)
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
                        har_rv_raw,
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
                val_n += bs

                valid_v = ~torch.isnan(vol_targets)
                if valid_v.any():
                    val_vol_preds.append(out["volatility_pred"][valid_v].float().cpu())
                    val_vol_targets_.append(vol_targets[valid_v].float().cpu())

                valid_d = dir_labels >= 0
                if valid_d.any():
                    probs = torch.softmax(
                        out["direction_logits"][valid_d].float(), dim=1
                    )[:, 1]
                    val_dir_preds.append(probs.cpu())
                    val_dir_labels_.append(dir_labels[valid_d].cpu())

                val_gate_weights.append(out["gate_weights"].float().cpu().mean(0))

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

        # Measure HAR skip contribution
        skip_contrib = 0.0
        with torch.no_grad():
            for batch in val_loader:
                pe = batch["price_emb"].to(dev)
                hr = batch["har_rv_raw"].to(dev)
                ge = batch["gat_emb"].to(dev)
                de = batch["doc_emb"].to(dev)
                me = batch["macro_emb"].to(dev)
                sf = batch["surprise_feat"].to(dev)
                mk = batch["modality_mask"].to(dev)

                with torch.amp.autocast("cuda"):
                    out_with = model(pe, hr, ge, de, me, sf, mk)
                    out_without = model(pe, torch.zeros_like(hr), ge, de, me, sf, mk)

                skip_contrib = (
                    (out_with["volatility_pred"] - out_without["volatility_pred"])
                    .abs()
                    .mean()
                    .item()
                )
                break  # single batch is enough for monitoring

        skip_contributions.append(skip_contrib)

        vram_gb = torch.cuda.memory_allocated(0) / 1e9
        peak_vram = max(peak_vram, vram_gb)

        avg_gates = torch.stack(val_gate_weights).mean(0).numpy()

        if epoch % 5 == 0 or epoch == 1:
            print(
                f"  Epoch {epoch:02d}/60 | QLIKE: {avg_loss:.4f} | Val R2: {val_r2:.4f} | "
                f"Dir AUC: {val_auc:.4f} | VRAM: {vram_gb:.2f}GB | "
                f"Gate: [price={avg_gates[0]:.0%} gat={avg_gates[1]:.0%} "
                f"doc={avg_gates[2]:.0%} macro={avg_gates[3]:.0%}] | "
                f"HAR skip: {skip_contrib:.4f}"
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                model,
                optimizer,
                epoch,
                {"val_loss": val_loss, "val_r2": val_r2, "val_auc": val_auc},
                save_path,
            )

        if stopper(val_loss):
            print(f"  Early stopping at epoch {epoch}")
            break

    # === Test evaluation ===
    print(f"\nEvaluating on test set...")
    ckpt = torch.load(save_path, weights_only=False, map_location=dev)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    test_vol_preds, test_vol_targets_ = [], []
    test_dir_preds, test_dir_labels_ = [], []
    test_gate_weights = []

    with torch.no_grad():
        for batch in test_loader:
            price_emb = batch["price_emb"].to(dev)
            har_rv_raw = batch["har_rv_raw"].to(dev)
            gat_emb = batch["gat_emb"].to(dev)
            doc_emb = batch["doc_emb"].to(dev)
            macro_emb = batch["macro_emb"].to(dev)
            surprise_feat = batch["surprise_feat"].to(dev)
            modality_mask = batch["modality_mask"].to(dev)
            dir_labels = batch["direction_label"].to(dev)
            vol_targets = batch["volatility_target"].to(dev)

            with torch.amp.autocast("cuda"):
                out = model(
                    price_emb,
                    har_rv_raw,
                    gat_emb,
                    doc_emb,
                    macro_emb,
                    surprise_feat,
                    modality_mask,
                )

            valid_vol = ~torch.isnan(vol_targets)
            if valid_vol.any():
                test_vol_preds.append(out["volatility_pred"][valid_vol].float().cpu())
                test_vol_targets_.append(vol_targets[valid_vol].float().cpu())

            valid_dir = dir_labels >= 0
            if valid_dir.any():
                probs = torch.softmax(
                    out["direction_logits"][valid_dir].float(), dim=1
                )[:, 1]
                test_dir_preds.append(probs.cpu())
                test_dir_labels_.append(dir_labels[valid_dir].cpu())

            test_gate_weights.append(out["gate_weights"].float().cpu().mean(0))

    vp = torch.cat(test_vol_preds).numpy()
    vt = torch.cat(test_vol_targets_).numpy()
    ss_res = np.sum((vt - vp) ** 2)
    ss_tot = np.sum((vt - vt.mean()) ** 2)
    test_r2 = 1 - ss_res / max(ss_tot, 1e-8)
    test_rmse = float(np.sqrt(np.mean((vt - vp) ** 2)))
    test_mae = float(np.mean(np.abs(vt - vp)))

    # QLIKE on test
    pred_var = np.clip(vp**2, 1e-6, None)
    true_var = np.clip(vt**2, 1e-6, None)
    qlike_vals = true_var / pred_var + np.log(pred_var)
    test_qlike = float(np.mean(qlike_vals))

    dp = torch.cat(test_dir_preds).numpy()
    dl = torch.cat(test_dir_labels_).numpy()
    try:
        test_auc = roc_auc_score(dl, dp)
        from sklearn.metrics import accuracy_score

        test_acc = accuracy_score(dl, (dp > 0.5).astype(int))
    except Exception:
        test_auc, test_acc = 0.5, 0.5

    avg_gates = torch.stack(test_gate_weights).mean(0).numpy()

    # Final skip contribution
    model.eval()
    final_skip_contrib = 0.0
    n_batches_skip = 0
    with torch.no_grad():
        for batch in test_loader:
            pe = batch["price_emb"].to(dev)
            hr = batch["har_rv_raw"].to(dev)
            ge = batch["gat_emb"].to(dev)
            de = batch["doc_emb"].to(dev)
            me = batch["macro_emb"].to(dev)
            sf = batch["surprise_feat"].to(dev)
            mk = batch["modality_mask"].to(dev)
            with torch.amp.autocast("cuda"):
                o1 = model(pe, hr, ge, de, me, sf, mk)
                o2 = model(pe, torch.zeros_like(hr), ge, de, me, sf, mk)
            final_skip_contrib += (
                (o1["volatility_pred"] - o2["volatility_pred"]).abs().mean().item()
            )
            n_batches_skip += 1
    final_skip_contrib /= max(n_batches_skip, 1)

    print(f"\nPHASE 14 TRAINING RESULTS")
    print(f"{'='*60}")
    print(f"Phase 13 vol_primary (reference): R2=0.867, RMSE=0.073, Dir AUC=0.585")
    print(
        f"Phase 14 (with skip connection):  R2={test_r2:.4f}, RMSE={test_rmse:.4f}, Dir AUC={test_auc:.4f}"
    )
    print(
        f"Improvement: {test_r2 - 0.8665:+.4f} R2 ({(test_r2 - 0.8665)/0.8665*100:+.1f}%)"
    )
    print(f"Beat HAR-RV (R2=0.947): {'YES' if test_r2 > 0.947 else 'NO'}")
    print(
        f"Gap remaining: {0.947 - test_r2:.4f}"
        if test_r2 < 0.947
        else "SURPASSED HAR-RV!"
    )
    print(f"QLIKE: {test_qlike:.4f}")
    print(f"MAE: {test_mae:.4f}")
    print(f"Dir Acc: {test_acc:.4f}")
    print(f"HAR skip avg contribution: {final_skip_contrib:.4f}")
    print(
        f"Gates: price={avg_gates[0]:.1%} gat={avg_gates[1]:.1%} doc={avg_gates[2]:.1%} macro={avg_gates[3]:.1%}"
    )
    print(f"Peak VRAM: {peak_vram:.2f}GB")

    results = {
        "vol_r2": float(test_r2),
        "rmse": float(test_rmse),
        "mae": float(test_mae),
        "qlike": float(test_qlike),
        "dir_auc": float(test_auc),
        "dir_acc": float(test_acc),
        "gates": avg_gates.tolist(),
        "skip_contribution": float(final_skip_contrib),
        "skip_contributions_history": skip_contributions,
        "peak_vram_gb": float(peak_vram),
        "n_params": n_params,
    }

    Path(f"{SAVE_DIR}/phase14_training_results.json").write_text(
        json.dumps(results, indent=2)
    )

    del model, optimizer
    torch.cuda.empty_cache()
    print("PHASE 14 STEP 5 COMPLETE — Phase14FusionModel trained")
    return results


# ======================================================================
# Step 6: Benchmark Comparison
# ======================================================================


def run_benchmarks(training_results):
    print("\n" + "=" * 60)
    print("PHASE 14 STEP 6: BENCHMARK COMPARISON")
    print("=" * 60)

    # Load Phase 13 benchmarks
    p13_bench = json.loads(
        Path(f"{SAVE_DIR}/phase13_benchmark_results.json").read_text()
    )
    p12_bench = p13_bench.get("phase12_benchmarks", {})

    ha_r2 = p12_bench.get("HA", {}).get("r2", 0.348)
    har_rv_r2 = p12_bench.get("HAR_RV", {}).get("r2", 0.947)
    har_rv_rmse = p12_bench.get("HAR_RV", {}).get("rmse", 0.0447)
    v2_r2 = p12_bench.get("V2_Baseline", {}).get("r2", 0.335)
    p12_r2 = p12_bench.get("Phase12_Multimodal", {}).get("r2", 0.772)
    p12_rmse = p12_bench.get("Phase12_Multimodal", {}).get("rmse", 0.096)
    p13_best = p13_bench.get("phase13_variants", {}).get("vol_primary", {})
    p13_r2 = p13_best.get("vol_r2", 0.867)
    p13_rmse = p13_best.get("rmse", 0.073)
    p13_auc = p13_best.get("dir_auc", 0.585)

    p14_r2 = training_results["vol_r2"]
    p14_rmse = training_results["rmse"]
    p14_qlike = training_results["qlike"]
    p14_auc = training_results["dir_auc"]

    print(f"\n{'='*80}")
    print(f"COMPLETE BENCHMARK TABLE")
    print(f"{'='*80}")
    print(
        f"{'Model':<25} | {'Vol R2':>8} | {'RMSE':>8} | {'QLIKE':>8} | {'Dir AUC':>8}"
    )
    print(f"{'-'*80}")
    print(
        f"{'Historical Average':<25} | {ha_r2:>8.4f} | {'---':>8} | {'---':>8} | {'---':>8}"
    )
    print(f"{'V2 Baseline':<25} | {v2_r2:>8.4f} | {'---':>8} | {'---':>8} | {'---':>8}")
    print(
        f"{'HAR-RV':<25} | {har_rv_r2:>8.4f} | {har_rv_rmse:>8.4f} | {'---':>8} | {'---':>8}"
    )
    print(
        f"{'Phase 12':<25} | {p12_r2:>8.4f} | {p12_rmse:>8.4f} | {'---':>8} | {'0.568':>8}"
    )
    print(
        f"{'Phase 13 vol_primary':<25} | {p13_r2:>8.4f} | {p13_rmse:>8.4f} | {'---':>8} | {p13_auc:>8.4f}"
    )
    print(
        f"{'Phase 14 (skip)':<25} | {p14_r2:>8.4f} | {p14_rmse:>8.4f} | {p14_qlike:>8.4f} | {p14_auc:>8.4f}"
    )
    print(f"{'='*80}")

    beats_har = p14_r2 > har_rv_r2
    if beats_har:
        print(
            f"\n*** LANDMARK: Phase 14 BEATS HAR-RV! R2={p14_r2:.4f} > {har_rv_r2:.4f} (margin={p14_r2-har_rv_r2:.4f}) ***"
        )
    else:
        gap = har_rv_r2 - p14_r2
        print(f"\nPhase 14 does NOT beat HAR-RV. Gap: {gap:.4f}")
        print(
            f"Improvement over Phase 13: {p14_r2 - p13_r2:+.4f} ({p13_r2:.4f} -> {p14_r2:.4f})"
        )

    bench_results = {
        "ha_r2": ha_r2,
        "v2_r2": v2_r2,
        "har_rv_r2": har_rv_r2,
        "p12_r2": p12_r2,
        "p13_r2": p13_r2,
        "p14_r2": p14_r2,
        "p14_rmse": p14_rmse,
        "p14_qlike": p14_qlike,
        "p14_auc": p14_auc,
        "beats_har_rv": beats_har,
        "gap_to_har_rv": float(har_rv_r2 - p14_r2) if not beats_har else 0.0,
        "improvement_over_p13": float(p14_r2 - p13_r2),
    }
    Path(f"{SAVE_DIR}/phase14_benchmark_results.json").write_text(
        json.dumps(bench_results, indent=2)
    )

    print("PHASE 14 STEP 6 COMPLETE — Benchmarks updated")
    return bench_results


# ======================================================================
# Step 7: VIX-Adjusted Vol Strategy + Full Backtest
# ======================================================================


def run_phase14_backtest(training_results):
    """Run all trading strategies including VIX-adjusted vol strategy."""
    from src.models.fusion_model import Phase14FusionModel
    from src.evaluation.vol_strategy_backtester import VolatilityStrategyBacktester

    print("\n" + "=" * 60)
    print("PHASE 14 STEP 7: VIX-ADJUSTED VOL STRATEGY + FULL BACKTEST")
    print("=" * 60)

    device = setup_gpu(verbose=False)
    dev = str(device)
    set_global_seed(42)

    # Load price data
    print("\n[1] Loading price data...")
    price_data = {}
    for ticker in ALL_30:
        csv_path = Path(f"data/raw/prices/{ticker}_ohlcv.csv")
        if csv_path.exists():
            price_data[ticker] = pd.read_csv(csv_path, parse_dates=["date"])
    print(f"  Loaded {len(price_data)} tickers")

    # Load embeddings + HAR-RV
    print("\n[2] Loading Phase 14 data...")
    emb = torch.load("data/embeddings/phase13_fusion_embeddings.pt", weights_only=False)
    har_rv = torch.load("data/embeddings/phase14_har_rv_raw.pt", weights_only=False)
    sample_tickers = np.array(emb["tickers"])
    sample_dates = np.array(emb["dates"])

    # Load best Phase 14 model and run MC Dropout
    print("\n[3] Running MC Dropout for uncertainty estimates...")
    model = Phase14FusionModel(
        price_dim=256,
        har_rv_dim=3,
        har_proj_dim=32,
        gat_dim=256,
        doc_dim=768,
        macro_dim=32,
        surprise_dim=5,
        proj_dim=128,
        hidden_dim=256,
        dropout=0.3,
        trunk_dropout=0.3,
        mc_dropout=True,
        lambda_vol=0.85,
        lambda_dir=0.15,
    ).to(dev)

    ckpt = torch.load(
        "models/phase14_fusion_best.pt", weights_only=False, map_location=dev
    )
    model.load_state_dict(ckpt["model_state_dict"])

    # Test set only
    dates_pd = pd.to_datetime(pd.Series(sample_dates))
    test_mask = (dates_pd > "2023-12-31").values
    test_idx = np.where(test_mask)[0]
    print(f"  Test samples: {len(test_idx)}")

    BATCH = 4096
    N_MC = 30
    all_vol_mean = []
    all_vol_std = []
    all_dir_mean = []

    model.train()  # keep dropout active for MC
    with torch.no_grad():
        for start in range(0, len(test_idx), BATCH):
            end = min(start + BATCH, len(test_idx))
            idx = test_idx[start:end]

            p_e = emb["price_emb"][idx].to(dev)
            h_r = har_rv[idx].to(dev)
            g_e = emb["gat_emb"][idx].to(dev)
            d_e = emb["doc_emb"][idx].to(dev)
            m_e = emb["macro_emb"][idx].to(dev)
            s = emb["surprise_feat"][idx].to(dev)
            mk = emb["modality_mask"][idx].to(dev)

            mc_out = model.predict_with_uncertainty(
                p_e, h_r, g_e, d_e, m_e, s, mk, n_samples=N_MC
            )
            all_vol_mean.append(mc_out["vol_mean"].cpu().numpy())
            all_vol_std.append(mc_out["vol_std"].cpu().numpy())
            dir_probs = torch.softmax(mc_out["dir_mean"], dim=1)[:, 1]
            all_dir_mean.append(dir_probs.cpu().numpy())

    vol_predictions = np.concatenate(all_vol_mean)
    vol_uncertainty = np.concatenate(all_vol_std)
    dir_predictions = np.concatenate(all_dir_mean)

    test_dates = sample_dates[test_idx]
    test_tickers = sample_tickers[test_idx]

    print(f"  Vol predictions shape: {vol_predictions.shape}")
    print(f"  Vol uncertainty mean: {vol_uncertainty.mean():.4f}")

    # Run standard backtest first
    print(
        "\n[4] Running standard backtest (Phase 13 strategies + Phase 14 predictions)..."
    )
    backtester = VolatilityStrategyBacktester(
        tickers=ALL_30,
        price_data=price_data,
        vol_predictions=vol_predictions,
        vol_uncertainty=vol_uncertainty,
        dir_predictions=dir_predictions,
        sample_dates=test_dates,
        sample_tickers=test_tickers,
        vol_threshold=0.15,
        rebalance_freq=5,
        holding_period=20,
        top_k=5,
        risk_free_rate=0.05,
    )

    standard_results = backtester.run_full_backtest(cost_levels=[0, 5, 10, 20])

    # VIX-adjusted strategy
    print("\n[5] Running VIX-adjusted volatility strategy...")

    # Load VIX data from macro features
    macro_path = Path("data/raw/macro")
    vix_df = None
    if (macro_path / "VIX_daily.csv").exists():
        vix_df = pd.read_csv(macro_path / "VIX_daily.csv", parse_dates=["date"])
    elif (macro_path / "vix_daily.csv").exists():
        vix_df = pd.read_csv(macro_path / "vix_daily.csv", parse_dates=["date"])
    else:
        # Try to find VIX in any macro CSV
        for f in macro_path.glob("*.csv"):
            df_tmp = pd.read_csv(
                f,
                parse_dates=(
                    ["date"] if "date" in pd.read_csv(f, nrows=0).columns else [0]
                ),
            )
            if "VIX" in str(f).upper() or "vix" in df_tmp.columns.str.lower().tolist():
                vix_df = df_tmp
                break

    if vix_df is not None:
        print(f"  VIX data loaded: {len(vix_df)} rows")
        vix_df["date"] = pd.to_datetime(vix_df["date"])
        vix_col = [c for c in vix_df.columns if c != "date"][0]
        vix_series = vix_df.set_index("date")[vix_col]
    else:
        # Fallback: use median VIX ~20
        print("  WARNING: VIX data not found, using fallback VIX=20")
        test_date_range = pd.bdate_range("2024-01-01", "2026-02-28")
        vix_series = pd.Series(20.0, index=test_date_range)

    # Compute beta for each stock vs SPY proxy (equal-weight market return)
    print("  Computing stock betas...")
    spy_rets = None
    spy_path = Path(f"{PRICES_DIR}/SPY_ohlcv.csv")
    if spy_path.exists():
        spy_df = pd.read_csv(spy_path, parse_dates=["date"])
        spy_df = spy_df.sort_values("date").set_index("date")
        spy_rets = spy_df["close"].pct_change()
    else:
        # Use equal-weight market return as proxy
        all_rets = []
        for ticker in ALL_30:
            if ticker in price_data:
                df = price_data[ticker].copy()
                df["date"] = pd.to_datetime(df["date"])
                df = df.sort_values("date").set_index("date")
                all_rets.append(df["close"].pct_change().rename(ticker))
        if all_rets:
            market_df = pd.concat(all_rets, axis=1)
            spy_rets = market_df.mean(axis=1)
            print("  Using equal-weight market return as SPY proxy")

    # Compute rolling beta per stock
    beta_dict = {}
    for ticker in ALL_30:
        if ticker not in price_data:
            continue
        df = price_data[ticker].copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").set_index("date")
        stock_rets = df["close"].pct_change()

        common = stock_rets.index.intersection(spy_rets.index)
        if len(common) < 60:
            beta_dict[ticker] = 1.0
            continue

        sr = stock_rets[common].values[-252:]
        mr = spy_rets[common].values[-252:]
        valid = ~(np.isnan(sr) | np.isnan(mr))
        sr, mr = sr[valid], mr[valid]
        if len(sr) < 60:
            beta_dict[ticker] = 1.0
        else:
            cov = np.cov(sr, mr)[0, 1]
            var_m = np.var(mr)
            beta = np.clip(cov / (var_m + 1e-8), 0.3, 3.0)
            beta_dict[ticker] = float(beta)

    print(f"  Betas computed for {len(beta_dict)} stocks")
    for t in sorted(beta_dict.keys())[:5]:
        print(f"    {t}: beta={beta_dict[t]:.2f}")

    # Run VIX-adjusted vol strategy at multiple cost levels
    vix_adj_results = {}
    for cost_bps in [0, 5, 10, 20]:
        cost = cost_bps / 10000.0
        test_dates_bt = backtester._get_test_dates()
        if len(test_dates_bt) == 0:
            continue

        hist_vol = backtester.compute_historical_avg_vol()
        portfolio_returns = []
        portfolio_dates = []
        prev_longs = set()
        prev_shorts = set()

        rebalance_dates = test_dates_bt[:: backtester.rebalance_freq]

        for reb_date in rebalance_dates:
            date_str = str(reb_date.date())

            signals = {}
            uncertainties = {}
            for ticker in ALL_30:
                key = (date_str, ticker)
                if key not in backtester._signals:
                    continue
                sig = backtester._signals[key]

                # Get VIX value
                try:
                    vix_val = vix_series.loc[:reb_date].iloc[-1] / 100.0
                except (KeyError, IndexError):
                    vix_val = 0.20

                beta = beta_dict.get(ticker, 1.0)
                implied_vol = vix_val * beta

                # Mispricing signal: predicted - implied
                mispricing = sig["pred_vol"] - implied_vol
                signals[ticker] = mispricing
                uncertainties[ticker] = sig["uncertainty"]

            if len(signals) < 2 * backtester.top_k:
                continue

            # Rank by mispricing: long most underpriced, short most overpriced
            sorted_tickers = sorted(signals.keys(), key=lambda t: signals[t])
            longs = sorted_tickers[
                -backtester.top_k :
            ]  # highest mispricing (underpriced)
            shorts = sorted_tickers[
                : backtester.top_k
            ]  # lowest mispricing (overpriced)

            # Uncertainty-weighted positions
            long_weights = {}
            for t in longs:
                long_weights[t] = 1.0 / (1.0 + uncertainties.get(t, 0.1))
            wsum = sum(long_weights.values())
            long_weights = {t: w / wsum for t, w in long_weights.items()}

            short_weights = {}
            for t in shorts:
                short_weights[t] = 1.0 / (1.0 + uncertainties.get(t, 0.1))
            wsum = sum(short_weights.values())
            short_weights = {t: w / wsum for t, w in short_weights.items()}

            # Turnover cost
            new_longs = set(longs)
            new_shorts = set(shorts)
            turnover = len(new_longs - prev_longs) + len(new_shorts - prev_shorts)
            turnover_cost = turnover * cost / max(len(longs) + len(shorts), 1)
            prev_longs = new_longs
            prev_shorts = new_shorts

            hold_end = min(
                len(test_dates_bt) - 1,
                np.searchsorted(test_dates_bt, reb_date) + backtester.rebalance_freq,
            )
            hold_start = np.searchsorted(test_dates_bt, reb_date)

            for day_idx in range(hold_start, hold_end):
                day = test_dates_bt[day_idx]
                daily_ret = 0.0

                for t, w in long_weights.items():
                    if (
                        t in backtester._return_df.columns
                        and day in backtester._return_df.index
                    ):
                        r = backtester._return_df.loc[day, t]
                        if not np.isnan(r):
                            daily_ret += w * r * 0.5

                for t, w in short_weights.items():
                    if (
                        t in backtester._return_df.columns
                        and day in backtester._return_df.index
                    ):
                        r = backtester._return_df.loc[day, t]
                        if not np.isnan(r):
                            daily_ret -= w * r * 0.5

                if day_idx == hold_start:
                    daily_ret -= turnover_cost

                portfolio_returns.append(daily_ret)
                portfolio_dates.append(day)

        if portfolio_returns:
            ret_df = pd.DataFrame({"date": portfolio_dates, "ret": portfolio_returns})
            ret_df = ret_df.groupby("date")["ret"].last()
            metrics = backtester.compute_metrics(ret_df, f"Vol VIX-adj ({cost_bps}bp)")
            vix_adj_results[cost_bps] = metrics
            print(
                f"  VIX-adj ({cost_bps}bp): Sharpe={metrics['sharpe']:.3f}, "
                f"Return={metrics['ann_return']:.2%}, MaxDD={metrics['max_drawdown']:.2%}"
            )

    # Combine all results
    all_results = standard_results.to_dict(orient="records")
    for cost_bps, metrics in vix_adj_results.items():
        all_results.append(metrics)

    print(f"\n{'='*100}")
    print("COMPLETE BACKTEST RESULTS")
    print(f"{'='*100}")
    print(
        f"{'Strategy':<40} | {'Sharpe':>8} | {'Return':>8} | {'MaxDD':>8} | {'WinRate':>8}"
    )
    print(f"{'-'*100}")
    for r in all_results:
        print(
            f"{r['strategy']:<40} | {r['sharpe']:>+8.3f} | {r['ann_return']:>+7.2%} | "
            f"{r['max_drawdown']:>+7.2%} | {r['win_rate']:>7.2%}"
        )
    print(f"{'='*100}")

    # Key findings
    vol_0 = next(
        (r for r in all_results if r["strategy"] == "Vol Strategy (0bp)"), None
    )
    vix_0 = vix_adj_results.get(0, {})
    vix_5 = vix_adj_results.get(5, {})
    dir_0 = next(
        (r for r in all_results if r["strategy"] == "Direction Strategy (0bp)"), None
    )
    dir_5 = next(
        (r for r in all_results if r["strategy"] == "Direction Strategy (5bp)"), None
    )
    dir_10 = next(
        (r for r in all_results if r["strategy"] == "Direction Strategy (10bp)"), None
    )

    print("\nKEY FINDINGS:")
    if vol_0 and vix_0:
        print(
            f"  1. VIX-adjusted vs Phase 13 raw vol: Sharpe {vix_0.get('sharpe', 0):.3f} vs {vol_0['sharpe']:.3f} "
            f"-> {'IMPROVED' if vix_0.get('sharpe', 0) > vol_0['sharpe'] else 'WORSE'}"
        )
    if vix_5:
        print(
            f"  2. VIX-adjusted at 5bp: Sharpe={vix_5.get('sharpe', 0):.3f} "
            f"-> {'POSITIVE' if vix_5.get('sharpe', 0) > 0 else 'NEGATIVE'}"
        )
    if dir_0 and dir_5:
        print(f"  3. Direction at 0bp: Sharpe={dir_0['sharpe']:.3f}")
        print(
            f"     Direction at 5bp: Sharpe={dir_5['sharpe']:.3f} -> {'VIABLE' if dir_5['sharpe'] > 0 else 'NOT VIABLE'}"
        )
    if dir_10:
        print(
            f"     Direction at 10bp: Sharpe={dir_10['sharpe']:.3f} -> {'VIABLE' if dir_10['sharpe'] > 0 else 'NOT VIABLE'}"
        )

    # Find break-even cost for VIX-adjusted
    if vix_adj_results:
        sorted_costs = sorted(vix_adj_results.keys())
        break_even = None
        for i in range(len(sorted_costs) - 1):
            c1 = sorted_costs[i]
            c2 = sorted_costs[i + 1]
            s1 = vix_adj_results[c1].get("sharpe", 0)
            s2 = vix_adj_results[c2].get("sharpe", 0)
            if s1 > 0 and s2 <= 0:
                break_even = c1 + (c2 - c1) * s1 / (s1 - s2 + 1e-8)
                break
        if break_even:
            print(f"  4. VIX-adj break-even cost: ~{break_even:.0f}bp")
        elif vix_adj_results.get(20, {}).get("sharpe", 0) > 0:
            print(f"  4. VIX-adj break-even cost: >20bp")
        else:
            print(f"  4. VIX-adj break-even cost: <0bp (not profitable at any cost)")

    # Save results
    Path("models/phase14_backtest_results.json").write_text(
        json.dumps(all_results, indent=2)
    )

    # Save daily returns
    vol_rets_0 = backtester.run_vol_strategy(0)
    dir_rets_0 = backtester.run_direction_strategy(0)
    bh_rets = backtester.run_buy_and_hold()

    returns_dict = {
        "vol_strategy_0bp": vol_rets_0,
        "dir_strategy_0bp": dir_rets_0,
        "buy_and_hold": bh_rets,
    }

    # Add VIX-adj returns if available
    if 0 in vix_adj_results:
        # Re-run to get return series (already computed but not saved)
        pass

    returns_df = pd.DataFrame(returns_dict)
    returns_df.to_csv("models/phase14_backtest_returns.csv")

    print(f"\nSaved: models/phase14_backtest_results.json")
    print(f"Saved: models/phase14_backtest_returns.csv")
    print("PHASE 14 STEP 7 COMPLETE — All strategies backtested")
    return all_results, vix_adj_results


# ======================================================================
# MAIN
# ======================================================================

if __name__ == "__main__":
    set_global_seed(42)

    print("=" * 70)
    print("PHASE 14 PIPELINE START — HAR-RV Skip Connection + Fixed Vol Strategy")
    print("=" * 70)

    # Step 2: GPU verification
    device = gpu_verification()

    # Step 4: Extract HAR-RV raw features
    extract_har_rv_raw()

    # Step 5: Train Phase14FusionModel
    training_results = train_phase14_fusion()

    # Step 6: Benchmark comparison
    bench_results = run_benchmarks(training_results)

    # Step 7: Full backtest with VIX-adjusted strategy
    backtest_results, vix_results = run_phase14_backtest(training_results)

    print("\n" + "=" * 70)
    print("PHASE 14 PIPELINE COMPLETE")
    print("=" * 70)
