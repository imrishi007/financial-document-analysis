"""Phase 9: Strategic model upgrades.

Phase 9A: Regime-Switching Ensemble (HMM + Gated Experts)
Phase 9B: Backtesting Engine with Transaction Costs
Phase 9C: Model Optimization (FP16 / INT8 Quantization / ONNX)
Phase 9D: Enhanced Cross-Stock Features & Retrain

All training uses CUDA with AMP (mixed precision) and maximal GPU utilization.
Results saved to models/phase9_results.json.
"""

from __future__ import annotations

import json
import time
import warnings
import gc
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.amp import autocast, GradScaler

# Suppress non-critical warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="hmmlearn")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.metrics import classification_metrics, regression_metrics, majority_baseline_accuracy
from src.train.common import EarlyStopping
from src.utils.seed import set_global_seed

# --------------- GPU Configuration ---------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
RESULTS_PATH = Path("models/phase9_results.json")

# Maximize GPU utilization
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def gpu_report(tag: str = ""):
    """Print current GPU memory usage."""
    if not torch.cuda.is_available():
        return
    alloc = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    pct = 100 * reserved / total
    print(f"  [GPU {tag}] Alloc={alloc:.2f}GB  Reserved={reserved:.2f}GB  Total={total:.1f}GB  Usage={pct:.0f}%")


def _save_results(results: dict):
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Results saved -> {RESULTS_PATH}")


# ======================================================================
# SHARED: CleanFusionModel (same arch used in Phase 8, imported here)
# ======================================================================

class CleanFusionModel(nn.Module):
    """Gated fusion: price(256) + gat(256) + doc(768) + surprise(1)."""
    NUM_MODALITIES = 4

    def __init__(self, price_dim=256, gat_dim=256, doc_dim=768, surprise_dim=1,
                 proj_dim=128, hidden_dim=256, dropout=0.3):
        super().__init__()
        self.proj_dim = proj_dim

        def _proj(in_dim):
            return nn.Sequential(
                nn.Linear(in_dim, proj_dim), nn.LayerNorm(proj_dim),
                nn.GELU(), nn.Dropout(dropout * 0.5))

        self.price_proj = _proj(price_dim)
        self.gat_proj = _proj(gat_dim)
        self.doc_proj = _proj(doc_dim)
        self.surprise_proj = _proj(surprise_dim)

        gate_in = proj_dim * self.NUM_MODALITIES
        self.gate = nn.Sequential(
            nn.Linear(gate_in, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, self.NUM_MODALITIES), nn.Sigmoid())

        self.trunk = nn.Sequential(
            nn.Linear(gate_in, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout * 0.5))

        self.direction_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(),
            nn.Dropout(dropout * 0.5), nn.Linear(hidden_dim // 2, 2))

        self.volatility_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4), nn.GELU(),
            nn.Linear(hidden_dim // 4, 1))

    def forward(self, price_emb, gat_emb, doc_emb, surprise_feat, modality_mask):
        p = self.price_proj(price_emb)
        g = self.gat_proj(gat_emb)
        d = self.doc_proj(doc_emb)
        s = self.surprise_proj(surprise_feat)
        stacked = torch.stack([p, g, d, s], dim=1)
        mask = modality_mask.unsqueeze(-1)
        stacked = stacked * mask
        flat = stacked.view(stacked.size(0), -1)
        gates = self.gate(flat) * modality_mask
        gates = gates / (gates.sum(dim=1, keepdim=True) + 1e-8)
        weighted = stacked * gates.unsqueeze(-1)
        fused = weighted.view(weighted.size(0), -1)
        h = self.trunk(fused)
        return {
            "direction_logits": self.direction_head(h),
            "volatility_pred": self.volatility_head(h).squeeze(-1),
            "gate_weights": gates,
        }


# ======================================================================
# SHARED HELPERS
# ======================================================================

def load_v2_data():
    """Load fixed embeddings to GPU (maximize VRAM usage)."""
    data = torch.load("data/embeddings/fusion_embeddings_v2.pt",
                       weights_only=False, map_location="cpu")
    print(f"  Loaded fusion_embeddings_v2.pt  ({data['price_emb'].shape[0]} samples)")
    return data


def load_returns_data():
    """Load actual 5-day returns for backtesting."""
    df = pd.read_csv("data/targets/direction_labels_multi_horizon.csv")
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    return df[["ticker", "date", "direction_5d_return"]].copy()


def split_by_date(dates, labels):
    """Temporal split: train<=2022, val=2023, test>=2024."""
    train, val, test = [], [], []
    for i, d in enumerate(dates):
        if labels[i] < 0:
            continue
        if d <= "2022-12-31":
            train.append(i)
        elif d <= "2023-12-31":
            val.append(i)
        else:
            test.append(i)
    return train, val, test


def make_loader(data, indices, keys, bs=2048, shuffle=False, pin=True):
    """Create a DataLoader from embedding dict."""
    idx_t = torch.tensor(indices, dtype=torch.long)
    tensors = [data[k][idx_t] for k in keys]
    ds = TensorDataset(*tensors)
    return DataLoader(ds, batch_size=bs, shuffle=shuffle,
                      pin_memory=pin, num_workers=0)


def train_fusion_amp(model, train_loader, val_loader, epochs=80, lr=5e-4,
                     patience=12, save_path=None, label_idx=-1):
    """Train a fusion model with AMP (mixed precision) for maximum GPU use."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    stopper = EarlyStopping(patience=patience, mode="max")
    ce = nn.CrossEntropyLoss()
    scaler = GradScaler("cuda")

    best_val_auc = 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        ep_loss, ep_n = 0.0, 0
        for batch in train_loader:
            tensors = [x.to(DEVICE, non_blocking=True) for x in batch]
            p, g, d, s, m, lbl = tensors[0], tensors[1], tensors[2], tensors[3], tensors[4], tensors[5]

            optimizer.zero_grad(set_to_none=True)
            with autocast("cuda"):
                out = model(p, g, d, s, m)
                loss = ce(out["direction_logits"], lbl)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            ep_loss += loss.item() * p.size(0)
            ep_n += p.size(0)

        scheduler.step()

        # Validation
        model.eval()
        vp, vl = [], []
        with torch.no_grad(), autocast("cuda"):
            for batch in val_loader:
                ts = [x.to(DEVICE, non_blocking=True) for x in batch]
                out = model(ts[0], ts[1], ts[2], ts[3], ts[4])
                probs = torch.softmax(out["direction_logits"], dim=1)[:, 1]
                vp.extend(probs.cpu().tolist())
                vl.extend(ts[5].cpu().tolist())

        val_met = classification_metrics(np.array(vl), (np.array(vp) > 0.5).astype(int), np.array(vp))
        val_auc = val_met.get("auc", 0.5)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            if save_path:
                Path(save_path).parent.mkdir(parents=True, exist_ok=True)
                torch.save({"model_state_dict": model.state_dict(), "epoch": epoch,
                            "val_auc": val_auc}, save_path)

        if epoch % 20 == 0 or epoch == 1:
            print(f"    Ep {epoch:02d} | loss={ep_loss/ep_n:.4f} | val_auc={val_auc:.4f}")
            gpu_report(f"ep{epoch}")

        if stopper(val_auc):
            print(f"    Early stop @ epoch {epoch}")
            break

    return best_val_auc


def evaluate_model(model, loader):
    """Evaluate model, return probs and labels."""
    model.eval()
    probs_all, labels_all, gates_all = [], [], []
    with torch.no_grad(), autocast("cuda"):
        for batch in loader:
            ts = [x.to(DEVICE, non_blocking=True) for x in batch]
            out = model(ts[0], ts[1], ts[2], ts[3], ts[4])
            probs = torch.softmax(out["direction_logits"], dim=1)[:, 1]
            probs_all.extend(probs.cpu().tolist())
            labels_all.extend(ts[5].cpu().tolist())
            if "gate_weights" in out:
                gates_all.append(out["gate_weights"].cpu())
    return np.array(probs_all), np.array(labels_all), gates_all


# ======================================================================
# PHASE 9A: REGIME-SWITCHING ENSEMBLE
# ======================================================================

def phase9a_regime_switching(data: dict) -> dict:
    """Train regime-specific expert models + gating network."""
    print("\n" + "=" * 70)
    print("  PHASE 9A: REGIME-SWITCHING ENSEMBLE (HMM + GATED EXPERTS)")
    print("=" * 70)
    t0 = time.time()
    set_global_seed(SEED)

    try:
        from hmmlearn.hmm import GaussianHMM
        USE_HMM = True
    except ImportError:
        from sklearn.cluster import KMeans
        USE_HMM = False
        print("  WARNING: hmmlearn not available, falling back to KMeans regime detection")

    dates = data["dates"]
    tickers = data["tickers"]
    dir_labels = data["direction_label"]

    # --- Step 1: Build market regime features from HISTORICAL returns ---
    # Use log_return (backward-looking daily return) NOT forward 5d return
    vol_df = pd.read_csv("data/targets/volatility_targets.csv")
    vol_df["date"] = pd.to_datetime(vol_df["date"]).dt.strftime("%Y-%m-%d")
    # Compute daily cross-stock average historical return per date
    daily_market = vol_df.groupby("date")["log_return"].agg(["mean", "std"]).reset_index()
    daily_market.columns = ["date", "market_ret", "market_disp"]
    daily_market = daily_market.sort_values("date").reset_index(drop=True)
    daily_market["market_disp"] = daily_market["market_disp"].fillna(0)
    # Rolling 5-day average for smoother regime detection
    daily_market["market_ret_5d"] = daily_market["market_ret"].rolling(5, min_periods=1).mean()
    daily_market["market_disp_5d"] = daily_market["market_disp"].rolling(5, min_periods=1).mean()

    # --- Step 2: Fit 3-state regime model on market features ---
    features = daily_market[["market_ret_5d", "market_disp_5d"]].values
    # Remove NaN rows
    valid_mask = ~np.isnan(features).any(axis=1)
    features_clean = features[valid_mask]

    if USE_HMM:
        hmm = GaussianHMM(n_components=3, covariance_type="full", n_iter=200,
                          random_state=SEED, verbose=False)
    else:
        hmm = KMeans(n_clusters=3, random_state=SEED, n_init=10)
    hmm.fit(features_clean)
    states = hmm.predict(features_clean)

    # Map states to interpretable names based on mean return
    state_means = [features_clean[states == s, 0].mean() for s in range(3)]
    state_order = np.argsort(state_means)  # low->high
    regime_names = {state_order[0]: "Bear", state_order[1]: "Sideways", state_order[2]: "Bull"}
    regime_map = {state_order[i]: i for i in range(3)}  # Bear=0, Side=1, Bull=2

    # Assign regime labels to dates
    valid_dates = daily_market["date"].values[valid_mask]
    date_to_regime = {}
    for dd, st in zip(valid_dates, states):
        date_to_regime[dd] = regime_map[st]

    regime_counts = {name: sum(1 for v in date_to_regime.values() if v == regime_map[k])
                     for k, name in regime_names.items()}
    print(f"  HMM Regimes: {regime_counts}")
    print(f"  State means (return): Bear={state_means[state_order[0]]:.4f}, "
          f"Sideways={state_means[state_order[1]]:.4f}, Bull={state_means[state_order[2]]:.4f}")

    # --- Step 3: Assign regime to each sample ---
    sample_regimes = []
    for d in dates:
        sample_regimes.append(date_to_regime.get(d, 1))  # default=Sideways
    sample_regimes = np.array(sample_regimes)

    # --- Step 4: Train 3 expert models (one per regime) ---
    keys = ["price_emb", "gat_emb", "doc_emb", "surprise_feat", "modality_mask", "direction_label"]
    train_idx, val_idx, test_idx = split_by_date(dates, dir_labels)

    expert_results = {}
    expert_models = {}

    for regime_id in range(3):
        rname = ["Bear", "Sideways", "Bull"][regime_id]
        # Filter indices by regime
        r_train = [i for i in train_idx if sample_regimes[i] == regime_id]
        r_val = [i for i in val_idx if sample_regimes[i] == regime_id]

        if len(r_train) < 200 or len(r_val) < 50:
            print(f"  Expert {rname}: insufficient data (train={len(r_train)}, val={len(r_val)}). Skipping.")
            continue

        print(f"\n  Training Expert: {rname} (train={len(r_train)}, val={len(r_val)})")

        model = CleanFusionModel(proj_dim=128, hidden_dim=256, dropout=0.3).to(DEVICE)
        train_ld = make_loader(data, r_train, keys, bs=2048, shuffle=True)
        val_ld = make_loader(data, r_val, keys, bs=2048)

        save_p = f"models/phase9_regime/{rname.lower()}_expert.pt"
        best_auc = train_fusion_amp(model, train_ld, val_ld, epochs=60, lr=5e-4,
                                     patience=10, save_path=save_p)
        expert_results[rname] = {"best_val_auc": best_auc, "n_train": len(r_train), "n_val": len(r_val)}
        expert_models[rname] = model
        print(f"    {rname} best val AUC: {best_auc:.4f}")

    # --- Step 5: Evaluate on test set using regime-based routing ---
    test_probs, test_labels = [], []
    for i in test_idx:
        regime_id = sample_regimes[i]
        rname = ["Bear", "Sideways", "Bull"][regime_id]
        if rname not in expert_models:
            rname = "Sideways"  # fallback
        if rname not in expert_models:
            # Use first available expert
            rname = list(expert_models.keys())[0]

        model = expert_models[rname]
        model.eval()
        with torch.no_grad(), autocast("cuda"):
            p = data["price_emb"][i:i+1].to(DEVICE)
            g = data["gat_emb"][i:i+1].to(DEVICE)
            d = data["doc_emb"][i:i+1].to(DEVICE)
            s = data["surprise_feat"][i:i+1].to(DEVICE)
            m = data["modality_mask"][i:i+1].to(DEVICE)
            out = model(p, g, d, s, m)
            prob = torch.softmax(out["direction_logits"], dim=1)[0, 1].item()

        test_probs.append(prob)
        test_labels.append(dir_labels[i].item())

    test_probs = np.array(test_probs)
    test_labels = np.array(test_labels)

    # Also evaluate in batch for per-regime breakdown
    regime_test_metrics = {}
    for regime_id in range(3):
        rname = ["Bear", "Sideways", "Bull"][regime_id]
        r_test = [j for j, i in enumerate(test_idx) if sample_regimes[i] == regime_id]
        if len(r_test) < 20:
            continue
        r_probs = test_probs[r_test]
        r_labels = test_labels[r_test]
        met = classification_metrics(r_labels, (r_probs > 0.5).astype(int), r_probs)
        regime_test_metrics[rname] = {"auc": met["auc"], "acc": met["accuracy"], "n": len(r_test)}
        print(f"  {rname} test: AUC={met['auc']:.4f}, Acc={met['accuracy']:.4f}, N={len(r_test)}")

    combined_met = classification_metrics(test_labels, (test_probs > 0.5).astype(int), test_probs)
    print(f"\n  Regime Ensemble Combined: AUC={combined_met['auc']:.4f}, Acc={combined_met['accuracy']:.4f}")
    gpu_report("9A-done")

    result = {
        "regime_counts": regime_counts,
        "expert_val_results": expert_results,
        "per_regime_test": regime_test_metrics,
        "combined_test_metrics": combined_met,
        "time_seconds": time.time() - t0,
    }
    return result


# ======================================================================
# PHASE 9B: BACKTESTING ENGINE WITH TRANSACTION COSTS
# ======================================================================

def phase9b_backtesting(data: dict) -> dict:
    """Full backtesting with slippage, commission, and risk-parity sizing."""
    print("\n" + "=" * 70)
    print("  PHASE 9B: BACKTESTING ENGINE WITH TRANSACTION COSTS")
    print("=" * 70)
    t0 = time.time()
    set_global_seed(SEED)

    dates = data["dates"]
    tickers = data["tickers"]
    dir_labels = data["direction_label"]

    # Load best model (clean fusion from Phase 8)
    model = CleanFusionModel(proj_dim=128, hidden_dim=256, dropout=0.3).to(DEVICE)
    ckpt_path = Path("models/v2_clean_fusion/clean_fusion_best.pt")
    if not ckpt_path.exists():
        print("  WARNING: Clean fusion checkpoint not found. Training fresh model...")
        keys = ["price_emb", "gat_emb", "doc_emb", "surprise_feat", "modality_mask", "direction_label"]
        train_idx, val_idx, _ = split_by_date(dates, dir_labels)
        train_ld = make_loader(data, train_idx, keys, bs=2048, shuffle=True)
        val_ld = make_loader(data, val_idx, keys, bs=2048)
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        train_fusion_amp(model, train_ld, val_ld, epochs=60, save_path=str(ckpt_path))

    ckpt = torch.load(str(ckpt_path), weights_only=False, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"  Loaded clean fusion model (epoch {ckpt.get('epoch', '?')})")

    # --- Generate predictions for test period ---
    _, _, test_idx = split_by_date(dates, dir_labels)
    keys = ["price_emb", "gat_emb", "doc_emb", "surprise_feat", "modality_mask", "direction_label"]
    test_loader = make_loader(data, test_idx, keys, bs=4096)
    test_probs, test_labels, _ = evaluate_model(model, test_loader)
    gpu_report("backtest-pred")

    # Load actual returns
    returns_df = load_returns_data()
    ret_lookup = {}
    for _, row in returns_df.iterrows():
        ret_lookup[(row["ticker"], row["date"])] = row["direction_5d_return"]

    # Build test prediction table
    test_records = []
    for j, i in enumerate(test_idx):
        d = dates[i]
        t = tickers[i]
        ret = ret_lookup.get((t, d), np.nan)
        test_records.append({
            "date": d, "ticker": t,
            "prob_up": test_probs[j],
            "true_label": int(test_labels[j]),
            "return_5d": ret,
        })

    pred_df = pd.DataFrame(test_records)
    pred_df = pred_df.dropna(subset=["return_5d"])
    print(f"  Predictions with returns: {len(pred_df)} samples across "
          f"{pred_df['date'].nunique()} dates, {pred_df['ticker'].nunique()} tickers")

    # --- Strategy 1: Equal-Weight Long-Short ---
    def run_strategy(pred_df, name, fee_pct=0.0005, top_k=3):
        """Simulate daily long-short: buy top-K, short bottom-K stocks by date."""
        daily_pnl = []
        daily_trades = []

        for date, group in pred_df.groupby("date"):
            if len(group) < 2 * top_k:
                continue
            sorted_g = group.sort_values("prob_up", ascending=False)

            # Long top-K (highest UP probability)
            longs = sorted_g.head(top_k)
            # Short bottom-K (lowest UP probability)
            shorts = sorted_g.tail(top_k)

            long_ret = longs["return_5d"].mean()
            short_ret = -shorts["return_5d"].mean()  # profit from shorts

            gross_ret = (long_ret + short_ret) / 2
            trade_cost = fee_pct * 2 * top_k * 2  # buy+sell for longs + shorts
            net_ret = gross_ret - trade_cost

            daily_pnl.append({"date": date, "gross_return": gross_ret,
                              "net_return": net_ret, "n_trades": 2 * top_k})
            daily_trades.append(2 * top_k)

        pnl_df = pd.DataFrame(daily_pnl).sort_values("date")
        if len(pnl_df) == 0:
            return {"strategy": name, "error": "no trades"}

        cum_gross = (1 + pnl_df["gross_return"]).cumprod()
        cum_net = (1 + pnl_df["net_return"]).cumprod()

        total_gross = cum_gross.iloc[-1] - 1
        total_net = cum_net.iloc[-1] - 1
        sharpe = pnl_df["net_return"].mean() / (pnl_df["net_return"].std() + 1e-8) * np.sqrt(252 / 5)
        max_dd = (cum_net / cum_net.cummax() - 1).min()
        win_rate = (pnl_df["net_return"] > 0).mean()

        return {
            "strategy": name,
            "total_return_gross": float(total_gross),
            "total_return_net": float(total_net),
            "sharpe_ratio": float(sharpe),
            "max_drawdown": float(max_dd),
            "win_rate": float(win_rate),
            "n_periods": len(pnl_df),
            "total_trades": int(sum(daily_trades)),
            "avg_daily_pnl": float(pnl_df["net_return"].mean()),
            "fee_pct": fee_pct,
        }

    # --- Strategy 2: Confidence-Weighted ---
    def run_confidence_strategy(pred_df, name, fee_pct=0.0005, conf_threshold=0.55):
        """Only trade when model confidence > threshold, size by confidence."""
        daily_pnl = []
        for date, group in pred_df.groupby("date"):
            confident = group[group["prob_up"].apply(lambda p: max(p, 1-p)) > conf_threshold]
            if len(confident) == 0:
                continue

            # Long stocks with prob > 0.5, short stocks with prob < 0.5
            long_mask = confident["prob_up"] > 0.5
            short_mask = confident["prob_up"] <= 0.5

            rets = []
            n_trades = 0
            if long_mask.any():
                long_wt = confident.loc[long_mask, "prob_up"]
                long_wt = long_wt / long_wt.sum()
                long_r = (confident.loc[long_mask, "return_5d"] * long_wt).sum()
                rets.append(long_r)
                n_trades += long_mask.sum()

            if short_mask.any():
                short_wt = (1 - confident.loc[short_mask, "prob_up"])
                short_wt = short_wt / short_wt.sum()
                short_r = -(confident.loc[short_mask, "return_5d"] * short_wt).sum()
                rets.append(short_r)
                n_trades += short_mask.sum()

            if rets:
                gross_r = np.mean(rets)
                net_r = gross_r - fee_pct * n_trades * 2
                daily_pnl.append({"date": date, "net_return": net_r, "gross_return": gross_r})

        pnl_df = pd.DataFrame(daily_pnl).sort_values("date")
        if len(pnl_df) == 0:
            return {"strategy": name, "error": "no trades"}

        cum_net = (1 + pnl_df["net_return"]).cumprod()
        total_net = cum_net.iloc[-1] - 1
        sharpe = pnl_df["net_return"].mean() / (pnl_df["net_return"].std() + 1e-8) * np.sqrt(252 / 5)
        max_dd = (cum_net / cum_net.cummax() - 1).min()
        win_rate = (pnl_df["net_return"] > 0).mean()

        return {
            "strategy": name,
            "total_return_net": float(total_net),
            "sharpe_ratio": float(sharpe),
            "max_drawdown": float(max_dd),
            "win_rate": float(win_rate),
            "n_periods": len(pnl_df),
        }

    # --- Strategy 3: Risk-Parity (vol-adjusted sizing) ---
    def run_risk_parity_strategy(pred_df, vol_lookup, name, fee_pct=0.0005, top_k=3):
        """Size positions inversely proportional to predicted volatility."""
        daily_pnl = []
        for date, group in pred_df.groupby("date"):
            if len(group) < 2 * top_k:
                continue
            sorted_g = group.sort_values("prob_up", ascending=False)
            longs = sorted_g.head(top_k).copy()
            shorts = sorted_g.tail(top_k).copy()

            # Get vol for sizing (inverse vol weighting)
            for df_part in [longs, shorts]:
                df_part["vol"] = df_part.apply(
                    lambda r: vol_lookup.get((r["ticker"], r["date"]), 0.25), axis=1)
                df_part["inv_vol"] = 1.0 / (df_part["vol"] + 0.01)
                df_part["weight"] = df_part["inv_vol"] / df_part["inv_vol"].sum()

            long_ret = (longs["return_5d"] * longs["weight"]).sum()
            short_ret = -(shorts["return_5d"] * shorts["weight"]).sum()
            gross_r = (long_ret + short_ret) / 2
            net_r = gross_r - fee_pct * 2 * top_k * 2

            daily_pnl.append({"date": date, "net_return": net_r, "gross_return": gross_r})

        pnl_df = pd.DataFrame(daily_pnl).sort_values("date")
        if len(pnl_df) == 0:
            return {"strategy": name, "error": "no trades"}

        cum_net = (1 + pnl_df["net_return"]).cumprod()
        total_net = cum_net.iloc[-1] - 1
        sharpe = pnl_df["net_return"].mean() / (pnl_df["net_return"].std() + 1e-8) * np.sqrt(252 / 5)
        max_dd = (cum_net / cum_net.cummax() - 1).min()
        win_rate = (pnl_df["net_return"] > 0).mean()

        return {
            "strategy": name,
            "total_return_net": float(total_net),
            "sharpe_ratio": float(sharpe),
            "max_drawdown": float(max_dd),
            "win_rate": float(win_rate),
            "n_periods": len(pnl_df),
        }

    # --- Strategy 4: Buy-and-Hold benchmark ---
    def buy_and_hold_benchmark(pred_df):
        daily = pred_df.groupby("date")["return_5d"].mean().sort_index()
        cum = (1 + daily).cumprod()
        total = cum.iloc[-1] - 1
        sharpe = daily.mean() / (daily.std() + 1e-8) * np.sqrt(252 / 5)
        max_dd = (cum / cum.cummax() - 1).min()
        return {
            "strategy": "Buy-and-Hold (Benchmark)",
            "total_return_net": float(total),
            "sharpe_ratio": float(sharpe),
            "max_drawdown": float(max_dd),
            "win_rate": float((daily > 0).mean()),
            "n_periods": len(daily),
        }

    # Load volatility for risk-parity
    vol_df = pd.read_csv("data/targets/volatility_targets.csv")
    vol_df["date"] = pd.to_datetime(vol_df["date"]).dt.strftime("%Y-%m-%d")
    vol_lookup = {(r["ticker"], r["date"]): r["realized_vol_20d_annualized"]
                  for _, r in vol_df.iterrows()}

    # Run all strategies
    results = {}

    # Benchmark
    bh = buy_and_hold_benchmark(pred_df)
    print(f"\n  Buy-and-Hold: Return={bh['total_return_net']:.2%}, Sharpe={bh['sharpe_ratio']:.2f}")
    results["buy_and_hold"] = bh

    # Equal-weight long-short variations
    for fee in [0.0, 0.0005, 0.001]:
        for k in [3, 5]:
            name = f"EqualWeight_top{k}_fee{fee:.4f}"
            r = run_strategy(pred_df, name, fee_pct=fee, top_k=k)
            if "error" not in r:
                print(f"  {name}: NetRet={r['total_return_net']:.2%}, Sharpe={r['sharpe_ratio']:.2f}, "
                      f"MaxDD={r['max_drawdown']:.2%}, WinRate={r['win_rate']:.1%}")
            results[name] = r

    # Confidence-weighted
    for conf in [0.52, 0.55, 0.60]:
        name = f"Confidence_{conf:.2f}"
        r = run_confidence_strategy(pred_df, name, conf_threshold=conf)
        if "error" not in r:
            print(f"  {name}: NetRet={r['total_return_net']:.2%}, Sharpe={r['sharpe_ratio']:.2f}")
        results[name] = r

    # Risk-parity
    for k in [3, 5]:
        name = f"RiskParity_top{k}"
        r = run_risk_parity_strategy(pred_df, vol_lookup, name, top_k=k)
        if "error" not in r:
            print(f"  {name}: NetRet={r['total_return_net']:.2%}, Sharpe={r['sharpe_ratio']:.2f}")
        results[name] = r

    results["time_seconds"] = time.time() - t0
    print(f"\n  Backtesting complete in {results['time_seconds']:.1f}s")
    return results


# ======================================================================
# PHASE 9C: MODEL OPTIMIZATION (FP16 / INT8 / ONNX)
# ======================================================================

def phase9c_optimization(data: dict) -> dict:
    """Benchmark FP16 vs INT8 vs ONNX inference."""
    print("\n" + "=" * 70)
    print("  PHASE 9C: MODEL OPTIMIZATION (FP16 / INT8 / ONNX)")
    print("=" * 70)
    t0 = time.time()
    set_global_seed(SEED)

    # Load model
    model = CleanFusionModel(proj_dim=128, hidden_dim=256, dropout=0.3).to(DEVICE)
    ckpt_path = "models/v2_clean_fusion/clean_fusion_best.pt"
    if Path(ckpt_path).exists():
        ckpt = torch.load(ckpt_path, weights_only=False, map_location=DEVICE)
        model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    dates = data["dates"]
    dir_labels = data["direction_label"]
    _, _, test_idx = split_by_date(dates, dir_labels)
    keys = ["price_emb", "gat_emb", "doc_emb", "surprise_feat", "modality_mask", "direction_label"]
    test_loader = make_loader(data, test_idx, keys, bs=4096)

    results = {}
    n_params = sum(p.numel() for p in model.parameters())
    model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e6
    print(f"  Base model: {n_params:,} params, {model_size_mb:.1f} MB (FP32)")

    # --- FP32 Baseline ---
    torch.cuda.synchronize()
    fp32_times = []
    for _ in range(3):
        start = time.perf_counter()
        with torch.no_grad():
            for batch in test_loader:
                ts = [x.to(DEVICE, non_blocking=True) for x in batch]
                _ = model(ts[0], ts[1], ts[2], ts[3], ts[4])
        torch.cuda.synchronize()
        fp32_times.append(time.perf_counter() - start)
    fp32_time = np.mean(fp32_times)
    print(f"  FP32 inference: {fp32_time*1000:.1f}ms (mean of 3 runs)")
    gpu_report("FP32")

    # --- FP16 Inference ---
    model_fp16 = model.half()
    torch.cuda.synchronize()
    fp16_times = []
    for _ in range(3):
        start = time.perf_counter()
        with torch.no_grad(), autocast("cuda"):
            for batch in test_loader:
                ts = [x.to(DEVICE, non_blocking=True) for x in batch]
                _ = model_fp16(ts[0].half(), ts[1].half(), ts[2].half(),
                               ts[3].half(), ts[4].half())
        torch.cuda.synchronize()
        fp16_times.append(time.perf_counter() - start)
    fp16_time = np.mean(fp16_times)
    fp16_size = model_size_mb / 2
    print(f"  FP16 inference: {fp16_time*1000:.1f}ms -> {fp16_time/fp32_time:.2f}x of FP32")
    gpu_report("FP16")

    # Put model back to FP32 for INT8
    model = model.float()

    # --- INT8 Dynamic Quantization (CPU only for PyTorch) ---
    model_cpu = model.cpu()
    model_int8 = torch.quantization.quantize_dynamic(
        model_cpu, {nn.Linear}, dtype=torch.qint8)
    int8_size = sum(
        p.numel() * (1 if p.dtype == torch.qint8 else p.element_size())
        for p in model_int8.parameters()
    ) / 1e6

    int8_times = []
    for _ in range(3):
        start = time.perf_counter()
        with torch.no_grad():
            for batch in test_loader:
                ts = [x.cpu() for x in batch]
                _ = model_int8(ts[0], ts[1], ts[2], ts[3], ts[4])
        int8_times.append(time.perf_counter() - start)
    int8_time = np.mean(int8_times)
    print(f"  INT8 inference (CPU): {int8_time*1000:.1f}ms -> {int8_time/fp32_time:.2f}x of FP32")

    # Verify INT8 preserves AUC
    model_int8.eval()
    int8_probs, int8_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            ts = [x.cpu() for x in batch]
            out = model_int8(ts[0], ts[1], ts[2], ts[3], ts[4])
            probs = torch.softmax(out["direction_logits"], dim=1)[:, 1]
            int8_probs.extend(probs.tolist())
            int8_labels.extend(ts[5].tolist())
    int8_met = classification_metrics(np.array(int8_labels),
                                       (np.array(int8_probs) > 0.5).astype(int),
                                       np.array(int8_probs))
    print(f"  INT8 AUC: {int8_met['auc']:.4f} (vs FP32 reference)")

    # --- ONNX Export ---
    model_export = model.cpu()
    onnx_path = "models/phase9_clean_fusion.onnx"
    try:
        dummy = (torch.randn(1, 256), torch.randn(1, 256), torch.randn(1, 768),
                 torch.randn(1, 1), torch.ones(1, 4))
        torch.onnx.export(model_export, dummy, onnx_path,
                          input_names=["price", "gat", "doc", "surprise", "mask"],
                          output_names=["direction_logits", "vol_pred", "gates"],
                          dynamic_axes={"price": {0: "batch"}, "gat": {0: "batch"},
                                       "doc": {0: "batch"}, "surprise": {0: "batch"},
                                       "mask": {0: "batch"}},
                          opset_version=17)
        onnx_size = os.path.getsize(onnx_path) / 1e6

        # Benchmark ONNX runtime
        import onnxruntime as ort
        sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

        onnx_times = []
        for _ in range(3):
            start = time.perf_counter()
            for batch in test_loader:
                ts = [x.cpu().numpy() for x in batch]
                _ = sess.run(None, {
                    "price": ts[0], "gat": ts[1], "doc": ts[2],
                    "surprise": ts[3], "mask": ts[4]})
            onnx_times.append(time.perf_counter() - start)
        onnx_time = np.mean(onnx_times)
        print(f"  ONNX inference (CPU): {onnx_time*1000:.1f}ms -> {onnx_time/fp32_time:.2f}x of FP32")
        print(f"  ONNX size: {onnx_size:.1f} MB")
        results["onnx"] = {"time_ms": onnx_time * 1000, "size_mb": onnx_size,
                           "speedup_vs_fp32": fp32_time / onnx_time}
    except Exception as e:
        print(f"  ONNX export failed: {e}")
        results["onnx"] = {"error": str(e)}

    # Move model back to GPU for subsequent phases
    model = model.to(DEVICE)

    results.update({
        "n_params": n_params,
        "fp32": {"time_ms": fp32_time * 1000, "size_mb": model_size_mb},
        "fp16": {"time_ms": fp16_time * 1000, "size_mb": fp16_size,
                 "speedup": fp32_time / fp16_time},
        "int8": {"time_ms": int8_time * 1000, "auc": int8_met["auc"],
                 "speedup_vs_fp32_gpu": fp32_time / int8_time},
        "time_seconds": time.time() - t0,
    })
    return results


# ======================================================================
# PHASE 9D: ENHANCED CROSS-STOCK FEATURES & RETRAIN
# ======================================================================

def phase9d_enhanced_features(data: dict) -> dict:
    """Add cross-stock and regime features, retrain fusion model."""
    print("\n" + "=" * 70)
    print("  PHASE 9D: ENHANCED CROSS-STOCK FEATURES & RETRAIN")
    print("=" * 70)
    t0 = time.time()
    set_global_seed(SEED)

    dates = data["dates"]
    tickers_list = data["tickers"]
    dir_labels = data["direction_label"]

    # --- Step 1: Build enhanced features from HISTORICAL data only ---
    # CRITICAL: Use only backward-looking data to avoid leakage.
    # Use log_return (daily close-to-close) NOT direction_5d_return (forward).
    vol_df = pd.read_csv("data/targets/volatility_targets.csv")
    vol_df["date"] = pd.to_datetime(vol_df["date"]).dt.strftime("%Y-%m-%d")

    # Build per-ticker rolling 5-day past return (backward-looking momentum)
    vol_df = vol_df.sort_values(["ticker", "date"])
    vol_df["past_5d_ret"] = vol_df.groupby("ticker")["log_return"].transform(
        lambda x: x.rolling(5, min_periods=1).sum())

    # Create lookup tables (all backward-looking)
    hist_ret_lookup = {}  # (ticker, date) -> past 5d log return
    vol_lookup = {}
    for _, r in vol_df.iterrows():
        hist_ret_lookup[(r["ticker"], r["date"])] = r["past_5d_ret"]
        vol_lookup[(r["ticker"], r["date"])] = r["realized_vol_20d_annualized"]

    # Define sector groups for cross-stock features
    SECTORS = {
        "FAANG+": ["AAPL", "AMZN", "GOOGL", "META", "MSFT"],
        "Chips": ["NVDA", "AMD", "INTC"],
        "Other": ["TSLA", "ORCL"],
    }
    ticker_to_sector = {}
    for sec, ticks in SECTORS.items():
        for t in ticks:
            ticker_to_sector[t] = sec

    # Compute per-date cross-stock features using HISTORICAL returns only
    unique_dates = sorted(set(dates))
    date_hist_ret = {}  # date -> {ticker -> past 5d return}
    date_vol = {}

    for d in unique_dates:
        date_hist_ret[d] = {}
        date_vol[d] = {}
        for tick in set(tickers_list):
            r = hist_ret_lookup.get((tick, d))
            v = vol_lookup.get((tick, d))
            if r is not None and not np.isnan(r):
                date_hist_ret[d][tick] = r
            if v is not None and not np.isnan(v):
                date_vol[d][tick] = v

    N = len(dates)
    # Enhanced features: [sector_past_mom, market_past_mom, cross_disp, vol_regime, sector_rank]
    enhanced_dim = 5
    enhanced_feats = torch.zeros(N, enhanced_dim)

    for i in range(N):
        d = dates[i]
        t = tickers_list[i]
        sector = ticker_to_sector.get(t, "Other")
        sector_tickers = SECTORS.get(sector, [t])

        # 1. Sector past momentum: avg PAST 5d return of sector peers (excl self)
        peers = [date_hist_ret[d].get(pt, 0) for pt in sector_tickers
                 if pt != t and pt in date_hist_ret.get(d, {})]
        sector_mom = np.mean(peers) if peers else 0.0

        # 2. Market past momentum: avg PAST 5d return of all stocks
        all_hist_rets = list(date_hist_ret.get(d, {}).values())
        market_mom = np.mean(all_hist_rets) if all_hist_rets else 0.0

        # 3. Cross-stock dispersion: std of PAST returns (volatility clustering)
        cross_disp = np.std(all_hist_rets) if len(all_hist_rets) > 1 else 0.0

        # 4. Volatility regime: current realized vol vs cross-sectional median
        all_vols = list(date_vol.get(d, {}).values())
        my_vol = date_vol.get(d, {}).get(t, 0.25)
        median_vol = np.median(all_vols) if all_vols else 0.25
        vol_regime = my_vol / (median_vol + 1e-6) - 1.0  # relative vol

        # 5. Sector rank by PAST return (relative performance within sector)
        sector_rets = [(pt, date_hist_ret[d].get(pt, 0)) for pt in sector_tickers
                       if pt in date_hist_ret.get(d, {})]
        if sector_rets:
            sector_rets.sort(key=lambda x: x[1])
            my_rank = next((j for j, (pt, _) in enumerate(sector_rets) if pt == t),
                           len(sector_rets) // 2)
            rank_pct = my_rank / max(len(sector_rets) - 1, 1)
        else:
            rank_pct = 0.5

        enhanced_feats[i] = torch.tensor([sector_mom, market_mom, cross_disp, vol_regime, rank_pct])

    # Normalize
    mean = enhanced_feats.mean(dim=0, keepdim=True)
    std = enhanced_feats.std(dim=0, keepdim=True) + 1e-8
    enhanced_feats = (enhanced_feats - mean) / std

    print(f"  Enhanced features shape: {enhanced_feats.shape}")
    print(f"  Feature names: [sector_momentum, market_momentum, cross_dispersion, vol_regime, sector_rank]")

    # --- Step 2: Build Enhanced Fusion Model ---
    class EnhancedFusionModel(nn.Module):
        """Gated fusion + cross-stock enhanced features."""
        NUM_MODALITIES = 4

        def __init__(self, price_dim=256, gat_dim=256, doc_dim=768, surprise_dim=1,
                     enhanced_dim=5, proj_dim=128, hidden_dim=256, dropout=0.3):
            super().__init__()
            self.proj_dim = proj_dim

            def _proj(in_dim):
                return nn.Sequential(
                    nn.Linear(in_dim, proj_dim), nn.LayerNorm(proj_dim),
                    nn.GELU(), nn.Dropout(dropout * 0.5))

            self.price_proj = _proj(price_dim)
            self.gat_proj = _proj(gat_dim)
            self.doc_proj = _proj(doc_dim)
            self.surprise_proj = _proj(surprise_dim)
            self.enhanced_proj = _proj(enhanced_dim)

            gate_in = proj_dim * (self.NUM_MODALITIES + 1)  # +1 for enhanced
            self.gate = nn.Sequential(
                nn.Linear(gate_in, hidden_dim), nn.GELU(),
                nn.Linear(hidden_dim, self.NUM_MODALITIES + 1), nn.Sigmoid())

            self.trunk = nn.Sequential(
                nn.Linear(gate_in, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout * 0.5))

            self.direction_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(),
                nn.Dropout(dropout * 0.5), nn.Linear(hidden_dim // 2, 2))

            self.volatility_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 4), nn.GELU(),
                nn.Linear(hidden_dim // 4, 1))

        def forward(self, price_emb, gat_emb, doc_emb, surprise_feat, modality_mask, enhanced_feat):
            p = self.price_proj(price_emb)
            g = self.gat_proj(gat_emb)
            d = self.doc_proj(doc_emb)
            s = self.surprise_proj(surprise_feat)
            e = self.enhanced_proj(enhanced_feat)

            stacked = torch.stack([p, g, d, s, e], dim=1)
            # enhanced features always available
            ext_mask = torch.cat([modality_mask, torch.ones(modality_mask.size(0), 1, device=modality_mask.device)], dim=1)
            stacked = stacked * ext_mask.unsqueeze(-1)

            flat = stacked.view(stacked.size(0), -1)
            gates = self.gate(flat) * ext_mask
            gates = gates / (gates.sum(dim=1, keepdim=True) + 1e-8)
            weighted = stacked * gates.unsqueeze(-1)
            fused = weighted.view(weighted.size(0), -1)

            h = self.trunk(fused)
            return {
                "direction_logits": self.direction_head(h),
                "volatility_pred": self.volatility_head(h).squeeze(-1),
                "gate_weights": gates,
            }

    # --- Step 3: Train enhanced model with AMP ---
    train_idx, val_idx, test_idx = split_by_date(dates, dir_labels)

    model = EnhancedFusionModel(enhanced_dim=enhanced_dim, proj_dim=128,
                                 hidden_dim=256, dropout=0.3).to(DEVICE)
    n_params = sum(pp.numel() for pp in model.parameters())
    print(f"  EnhancedFusionModel: {n_params:,} params")

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=80, eta_min=1e-6)
    stopper = EarlyStopping(patience=12, mode="max")
    ce = nn.CrossEntropyLoss()
    scaler = GradScaler("cuda")

    def make_enhanced_loader(indices, shuffle=False, bs=2048):
        idx_t = torch.tensor(indices, dtype=torch.long)
        ds = TensorDataset(
            data["price_emb"][idx_t], data["gat_emb"][idx_t],
            data["doc_emb"][idx_t], data["surprise_feat"][idx_t],
            data["modality_mask"][idx_t], enhanced_feats[idx_t],
            dir_labels[idx_t])
        return DataLoader(ds, batch_size=bs, shuffle=shuffle,
                          pin_memory=True, num_workers=0)

    train_ld = make_enhanced_loader(train_idx, shuffle=True, bs=2048)
    val_ld = make_enhanced_loader(val_idx, bs=4096)
    test_ld = make_enhanced_loader(test_idx, bs=4096)

    save_dir = Path("models/phase9_enhanced")
    save_dir.mkdir(parents=True, exist_ok=True)
    best_val_auc = 0.0

    for epoch in range(1, 81):
        model.train()
        ep_loss, ep_n = 0.0, 0

        for batch in train_ld:
            p_b, g_b, d_b, s_b, m_b, e_b, l_b = [x.to(DEVICE, non_blocking=True) for x in batch]
            optimizer.zero_grad(set_to_none=True)
            with autocast("cuda"):
                out = model(p_b, g_b, d_b, s_b, m_b, e_b)
                loss = ce(out["direction_logits"], l_b)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            ep_loss += loss.item() * p_b.size(0)
            ep_n += p_b.size(0)

        scheduler.step()

        model.eval()
        vp, vl = [], []
        with torch.no_grad(), autocast("cuda"):
            for batch in val_ld:
                ts = [x.to(DEVICE, non_blocking=True) for x in batch]
                out = model(ts[0], ts[1], ts[2], ts[3], ts[4], ts[5])
                probs = torch.softmax(out["direction_logits"], dim=1)[:, 1]
                vp.extend(probs.cpu().tolist())
                vl.extend(ts[6].cpu().tolist())

        val_met = classification_metrics(np.array(vl), (np.array(vp) > 0.5).astype(int), np.array(vp))
        val_auc = val_met.get("auc", 0.5)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch,
                         "val_auc": val_auc}, save_dir / "enhanced_best.pt")

        if epoch % 20 == 0 or epoch == 1:
            print(f"    Ep {epoch:02d} | loss={ep_loss/ep_n:.4f} | val_auc={val_auc:.4f}")
            gpu_report(f"9D-ep{epoch}")

        if stopper(val_auc):
            print(f"    Early stop @ epoch {epoch}")
            break

    # --- Step 4: Test evaluation ---
    ckpt = torch.load(save_dir / "enhanced_best.pt", weights_only=False, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    test_probs, test_labels, test_gates = [], [], []
    with torch.no_grad(), autocast("cuda"):
        for batch in test_ld:
            ts = [x.to(DEVICE, non_blocking=True) for x in batch]
            out = model(ts[0], ts[1], ts[2], ts[3], ts[4], ts[5])
            probs = torch.softmax(out["direction_logits"], dim=1)[:, 1]
            test_probs.extend(probs.cpu().tolist())
            test_labels.extend(ts[6].cpu().tolist())
            test_gates.append(out["gate_weights"].cpu())

    test_met = classification_metrics(np.array(test_labels),
                                       (np.array(test_probs) > 0.5).astype(int),
                                       np.array(test_probs))
    gate_w = torch.cat(test_gates, dim=0).mean(0).tolist()
    gate_names = ["price", "gat", "doc", "surprise", "enhanced"]

    print(f"\n  Enhanced Model TEST AUC: {test_met['auc']:.4f}, Acc: {test_met['accuracy']:.4f}")
    print(f"  Gate weights: " + ", ".join(f"{n}={w:.3f}" for n, w in zip(gate_names, gate_w)))
    gpu_report("9D-done")

    # Save enhanced embeddings
    enhanced_path = "data/embeddings/fusion_embeddings_v3_enhanced.pt"
    v3_data = {
        "price_emb": data["price_emb"],
        "gat_emb": data["gat_emb"],
        "doc_emb": data["doc_emb"],
        "surprise_feat": data["surprise_feat"],
        "enhanced_feat": enhanced_feats,
        "modality_mask": data["modality_mask"],
        "direction_label": data["direction_label"],
        "volatility_target": data["volatility_target"],
        "tickers": data["tickers"],
        "dates": data["dates"],
    }
    torch.save(v3_data, enhanced_path)
    print(f"  Saved v3 embeddings -> {enhanced_path}")

    return {
        "test_metrics": test_met,
        "gate_weights": {n: w for n, w in zip(gate_names, gate_w)},
        "best_val_auc": best_val_auc,
        "n_params": n_params,
        "enhanced_features": ["sector_momentum", "market_momentum", "cross_dispersion",
                              "vol_regime", "sector_rank"],
        "time_seconds": time.time() - t0,
    }


# ======================================================================
# MASTER PIPELINE
# ======================================================================

def main():
    print("=" * 70)
    print("  PHASE 9: STRATEGIC MODEL UPGRADES")
    print("=" * 70)

    if not torch.cuda.is_available():
        print("  ERROR: CUDA not available! This script requires GPU.")
        print("  Install: pip install torch --index-url https://download.pytorch.org/whl/cu126")
        sys.exit(1)

    print(f"  Device: {DEVICE}")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    props = torch.cuda.get_device_properties(0)
    vram = getattr(props, "total_memory", getattr(props, "total_mem", 0))
    print(f"  VRAM: {vram / 1e9:.1f} GB")
    print(f"  CUDA: {torch.version.cuda}")
    print(f"  cuDNN benchmark: {torch.backends.cudnn.benchmark}")
    print(f"  TF32: {torch.backends.cuda.matmul.allow_tf32}")
    gpu_report("start")

    total_t0 = time.time()
    results = {}

    # Load data once
    data = load_v2_data()

    # --- Phase 9A ---
    try:
        r9a = phase9a_regime_switching(data)
        results["phase9a_regime_switching"] = r9a
        _save_results(results)
    except Exception as e:
        print(f"  Phase 9A failed: {e}")
        import traceback; traceback.print_exc()
        results["phase9a_regime_switching"] = {"error": str(e)}

    torch.cuda.empty_cache()
    gc.collect()

    # --- Phase 9B ---
    try:
        r9b = phase9b_backtesting(data)
        results["phase9b_backtesting"] = r9b
        _save_results(results)
    except Exception as e:
        print(f"  Phase 9B failed: {e}")
        import traceback; traceback.print_exc()
        results["phase9b_backtesting"] = {"error": str(e)}

    torch.cuda.empty_cache()
    gc.collect()

    # --- Phase 9C ---
    try:
        r9c = phase9c_optimization(data)
        results["phase9c_optimization"] = r9c
        _save_results(results)
    except Exception as e:
        print(f"  Phase 9C failed: {e}")
        import traceback; traceback.print_exc()
        results["phase9c_optimization"] = {"error": str(e)}

    torch.cuda.empty_cache()
    gc.collect()

    # --- Phase 9D ---
    try:
        r9d = phase9d_enhanced_features(data)
        results["phase9d_enhanced_features"] = r9d
        _save_results(results)
    except Exception as e:
        print(f"  Phase 9D failed: {e}")
        import traceback; traceback.print_exc()
        results["phase9d_enhanced_features"] = {"error": str(e)}

    # --- Final Summary ---
    print("\n" + "=" * 70)
    print("  PHASE 9 FINAL SUMMARY")
    print("=" * 70)

    summary_lines = []
    if "phase9a_regime_switching" in results and "combined_test_metrics" in results.get("phase9a_regime_switching", {}):
        auc = results["phase9a_regime_switching"]["combined_test_metrics"]["auc"]
        summary_lines.append(f"  9A Regime Ensemble:    AUC={auc:.4f}")

    if "phase9b_backtesting" in results:
        bt = results["phase9b_backtesting"]
        if "buy_and_hold" in bt:
            bh = bt["buy_and_hold"]
            summary_lines.append(f"  9B Buy-Hold Benchmark: Return={bh['total_return_net']:.2%}, Sharpe={bh['sharpe_ratio']:.2f}")
        # Find best strategy
        best_sharpe = -999
        best_name = ""
        for k, v in bt.items():
            if isinstance(v, dict) and "sharpe_ratio" in v and v["sharpe_ratio"] > best_sharpe:
                best_sharpe = v["sharpe_ratio"]
                best_name = v.get("strategy", k)
        if best_name:
            summary_lines.append(f"  9B Best Strategy:      {best_name} (Sharpe={best_sharpe:.2f})")

    if "phase9c_optimization" in results and "fp16" in results.get("phase9c_optimization", {}):
        opt = results["phase9c_optimization"]
        summary_lines.append(f"  9C FP32: {opt['fp32']['time_ms']:.1f}ms | "
                             f"FP16: {opt['fp16']['time_ms']:.1f}ms ({opt['fp16']['speedup']:.1f}x)")

    if "phase9d_enhanced_features" in results and "test_metrics" in results.get("phase9d_enhanced_features", {}):
        auc = results["phase9d_enhanced_features"]["test_metrics"]["auc"]
        summary_lines.append(f"  9D Enhanced Features:  AUC={auc:.4f}")

    for line in summary_lines:
        print(line)

    results["total_time_seconds"] = time.time() - total_t0
    _save_results(results)

    print(f"\n  Total time: {time.time()-total_t0:.1f}s")
    print(f"  Results: {RESULTS_PATH}")
    gpu_report("final")


if __name__ == "__main__":
    main()
