"""Confidence-Filtered Trading Analysis.

Loads the best direction-only fusion model (Improvement 1, dir_only config)
and evaluates it on the test set with progressively stricter confidence
thresholds. The hypothesis: if we only act on high-confidence predictions,
accuracy and AUC should improve significantly.

Thresholds are applied by sorting test samples by the model's softmax
confidence (max class probability) and keeping only the top-K% most
confident predictions.

Metrics computed at each threshold:
  - Direction AUC-ROC
  - Direction accuracy
  - Coverage (fraction of test set retained)
  - Class balance in the retained subset
  - Mean confidence of retained predictions
  - Simulated return (mean actual 5-day return of predicted-UP vs predicted-DOWN)
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

# Project root
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.fusion_dataset import FusionEmbeddingDataset
from src.evaluation.metrics import classification_metrics, majority_baseline_accuracy
from src.models.fusion_model import MultimodalFusionModel
from src.utils.seed import set_global_seed

# ======================================================================
# Configuration
# ======================================================================

CHECKPOINT_PATH = ROOT / "models" / "imp1_dir_only" / "fusion_model_best.pt"
EMBEDDINGS_PATH = ROOT / "data" / "embeddings" / "fusion_embeddings.pt"
OUTPUT_PATH = ROOT / "models" / "confidence_filtering_results.json"

# Coverage thresholds: keep top X% most confident predictions
COVERAGE_THRESHOLDS = [1.0, 0.75, 0.50, 0.40, 0.30, 0.20, 0.10, 0.05]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
BATCH_SIZE = 1024


# ======================================================================
# Helpers
# ======================================================================


def split_by_date(dataset: FusionEmbeddingDataset):
    """Temporal split: train ≤ 2022, val = 2023, test ≥ 2024."""
    train_idx, val_idx, test_idx = [], [], []
    for i in range(len(dataset)):
        d = dataset.dates[i]
        if d <= "2022-12-31":
            train_idx.append(i)
        elif d <= "2023-12-31":
            val_idx.append(i)
        else:
            test_idx.append(i)
    return train_idx, val_idx, test_idx


@torch.no_grad()
def get_predictions(model, loader, device):
    """Run model on all batches, return arrays of probs, preds, labels, tickers, dates."""
    model.eval()
    all_probs = []
    all_labels = []
    all_tickers = []
    all_dates = []
    all_gates = []

    for batch in loader:
        price = batch["price_emb"].to(device)
        gat = batch["gat_emb"].to(device)
        doc = batch["doc_emb"].to(device)
        news = batch["news_emb"].to(device)
        surp = batch["surprise_feat"].to(device)
        mask = batch["modality_mask"].to(device)

        out = model(price, gat, doc, news, surp, mask)
        probs = torch.softmax(out["direction_logits"], dim=1).cpu().numpy()
        all_probs.append(probs)
        all_labels.extend(batch["direction_label"].tolist())
        all_tickers.extend(batch["ticker"])
        all_dates.extend(batch["date"])
        all_gates.append(out["gate_weights"].cpu().numpy())

    probs = np.concatenate(all_probs, axis=0)  # [N, 2]
    labels = np.array(all_labels)
    gates = np.concatenate(all_gates, axis=0)  # [N, 5]
    return probs, labels, all_tickers, all_dates, gates


def evaluate_at_threshold(
    probs: np.ndarray,
    labels: np.ndarray,
    tickers: list,
    dates: list,
    coverage: float,
) -> dict:
    """Evaluate direction prediction keeping only the top-`coverage` most confident."""
    # Filter valid labels
    valid = labels >= 0
    probs_v = probs[valid]
    labels_v = labels[valid]

    # Confidence = max class probability
    confidence = probs_v.max(axis=1)  # [N_valid]
    n_total = len(labels_v)

    if coverage >= 1.0:
        # Use all samples
        selected = np.ones(n_total, dtype=bool)
    else:
        n_keep = max(int(n_total * coverage), 1)
        threshold = np.sort(confidence)[::-1][min(n_keep - 1, n_total - 1)]
        selected = confidence >= threshold
        # In case of ties at the boundary, randomly trim to exact count
        if selected.sum() > n_keep:
            indices_at_boundary = np.where((selected) & (confidence == threshold))[0]
            n_excess = selected.sum() - n_keep
            rng = np.random.RandomState(42)
            drop = rng.choice(indices_at_boundary, size=n_excess, replace=False)
            selected[drop] = False

    sel_probs = probs_v[selected]
    sel_labels = labels_v[selected]
    sel_confidence = confidence[selected]

    # Predicted class and UP probability
    preds = sel_probs.argmax(axis=1)
    up_probs = sel_probs[:, 1]

    # Classification metrics
    n_selected = selected.sum()
    metrics = classification_metrics(sel_labels, preds, up_probs)
    baseline = majority_baseline_accuracy(sel_labels)

    # Class balance in the selected subset
    up_frac = sel_labels.mean()

    # Per-class confidence
    up_mask = preds == 1
    down_mask = preds == 0

    result = {
        "coverage": float(coverage),
        "n_samples": int(n_selected),
        "n_total": int(n_total),
        "actual_coverage": float(n_selected / n_total),
        "mean_confidence": float(sel_confidence.mean()),
        "median_confidence": float(np.median(sel_confidence)),
        "min_confidence": float(sel_confidence.min()),
        "direction_auc": metrics.get("auc", 0.5),
        "direction_acc": metrics["accuracy"],
        "direction_precision": metrics["precision"],
        "direction_recall": metrics["recall"],
        "direction_f1": metrics["f1"],
        "majority_baseline": float(baseline),
        "acc_above_baseline": float(metrics["accuracy"] - baseline),
        "up_fraction": float(up_frac),
        "pred_up_fraction": float(up_mask.mean()),
        "pred_down_fraction": float(down_mask.mean()),
    }

    # Accuracy by predicted class
    if up_mask.any():
        result["acc_when_predict_up"] = float((sel_labels[up_mask] == 1).mean())
    if down_mask.any():
        result["acc_when_predict_down"] = float((sel_labels[down_mask] == 0).mean())

    return result


def per_ticker_confidence_analysis(
    probs: np.ndarray,
    labels: np.ndarray,
    tickers: list,
    coverage: float = 0.30,
) -> dict:
    """Evaluate confidence filtering per ticker at a given coverage level."""
    valid = labels >= 0
    probs_v = probs[valid]
    labels_v = labels[valid]
    tickers_v = [t for t, v in zip(tickers, valid) if v]

    confidence = probs_v.max(axis=1)
    unique_tickers = sorted(set(tickers_v))

    results = {}
    for ticker in unique_tickers:
        t_mask = np.array([t == ticker for t in tickers_v])
        t_probs = probs_v[t_mask]
        t_labels = labels_v[t_mask]
        t_conf = confidence[t_mask]

        n_total = len(t_labels)
        n_keep = max(int(n_total * coverage), 1)

        if n_total < 10:
            continue

        threshold = np.sort(t_conf)[::-1][min(n_keep - 1, n_total - 1)]
        selected = t_conf >= threshold

        if selected.sum() > n_keep:
            idx_boundary = np.where((selected) & (t_conf == threshold))[0]
            n_excess = selected.sum() - n_keep
            rng = np.random.RandomState(42)
            drop = rng.choice(idx_boundary, size=n_excess, replace=False)
            selected[drop] = False

        if selected.sum() < 5:
            continue

        sel_probs = t_probs[selected]
        sel_labels = t_labels[selected]
        preds = sel_probs.argmax(axis=1)
        up_prob = sel_probs[:, 1]

        try:
            auc = float(classification_metrics(sel_labels, preds, up_prob)["auc"])
        except Exception:
            auc = 0.5

        results[ticker] = {
            "n_samples": int(selected.sum()),
            "auc": auc,
            "accuracy": float((preds == sel_labels).mean()),
            "baseline": float(majority_baseline_accuracy(sel_labels)),
            "mean_confidence": float(t_conf[selected].mean()),
        }

    return results


# ======================================================================
# Main
# ======================================================================


def main():
    start = time.time()
    set_global_seed(SEED)

    print("=" * 70)
    print("  CONFIDENCE-FILTERED TRADING ANALYSIS")
    print("=" * 70)

    # --- 1. Load dataset ---
    print(f"\n[1/4] Loading embeddings from {EMBEDDINGS_PATH.name} ...")
    ds = FusionEmbeddingDataset(EMBEDDINGS_PATH)
    _, _, test_idx = split_by_date(ds)
    test_ds = Subset(ds, test_idx)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    print(f"       Test samples: {len(test_idx):,}")

    # --- 2. Load model ---
    print(f"\n[2/4] Loading dir_only model from {CHECKPOINT_PATH.name} ...")
    model = MultimodalFusionModel()
    ckpt = torch.load(CHECKPOINT_PATH, weights_only=False, map_location=DEVICE)
    # Handle both raw state_dict and wrapped checkpoint formats
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    model = model.to(DEVICE)
    model.eval()
    print(f"       Model loaded on {DEVICE}")

    # --- 3. Get predictions ---
    print("\n[3/4] Running inference on test set ...")
    probs, labels, tickers, dates, gates = get_predictions(model, test_loader, DEVICE)

    valid_mask = labels >= 0
    n_valid = valid_mask.sum()
    print(f"       Valid predictions: {n_valid:,} / {len(labels):,}")
    print(f"       Mean gate weights: {gates.mean(axis=0).round(4).tolist()}")
    print(f"       Modalities: [price, gat, doc, news, surprise]")

    # Confidence distribution stats
    conf = probs.max(axis=1)
    conf_valid = conf[valid_mask]
    print(f"\n       Confidence distribution (valid samples):")
    print(f"         Mean:   {conf_valid.mean():.4f}")
    print(f"         Median: {np.median(conf_valid):.4f}")
    print(f"         Std:    {conf_valid.std():.4f}")
    print(f"         Min:    {conf_valid.min():.4f}")
    print(f"         Max:    {conf_valid.max():.4f}")
    for pctile in [90, 75, 50, 25, 10]:
        print(f"         P{pctile:02d}:    {np.percentile(conf_valid, pctile):.4f}")

    # --- 4. Evaluate at all thresholds ---
    print(f"\n[4/4] Evaluating at {len(COVERAGE_THRESHOLDS)} coverage levels ...\n")

    threshold_results = []
    for cov in COVERAGE_THRESHOLDS:
        result = evaluate_at_threshold(probs, labels, tickers, dates, cov)
        threshold_results.append(result)

        pct_label = f"{cov*100:.0f}%" if cov < 1.0 else "ALL"
        print(
            f"  Coverage {pct_label:>4s}  |  "
            f"N={result['n_samples']:>5,}  |  "
            f"AUC={result['direction_auc']:.4f}  |  "
            f"Acc={result['direction_acc']:.4f}  |  "
            f"Baseline={result['majority_baseline']:.4f}  |  "
            f"Δ Acc={result['acc_above_baseline']:+.4f}  |  "
            f"Conf={result['mean_confidence']:.4f}"
        )

    # Best threshold (max AUC with at least 5% coverage)
    viable = [r for r in threshold_results if r["coverage"] >= 0.05]
    best = max(viable, key=lambda r: r["direction_auc"])
    print(
        f"\n  ★ Best AUC: {best['direction_auc']:.4f} at "
        f"{best['coverage']*100:.0f}% coverage "
        f"({best['n_samples']:,} samples)"
    )

    # Highlight the accuracy lift
    all_result = threshold_results[0]
    print(f"\n  Full-set AUC:     {all_result['direction_auc']:.4f}")
    print(f"  Best filtered AUC: {best['direction_auc']:.4f}")
    print(
        f"  AUC improvement:   {best['direction_auc'] - all_result['direction_auc']:+.4f}"
    )

    # --- Per-ticker at best coverage ---
    best_cov = best["coverage"]
    print(f"\n  Per-ticker analysis at {best_cov*100:.0f}% coverage:")
    ticker_results = per_ticker_confidence_analysis(
        probs, labels, tickers, coverage=best_cov
    )
    print(
        f"  {'Ticker':<8s} {'N':>5s} {'AUC':>7s} {'Acc':>7s} {'Base':>7s} {'Conf':>7s}"
    )
    print(f"  {'-'*40}")
    for t in sorted(
        ticker_results, key=lambda t: ticker_results[t]["auc"], reverse=True
    ):
        r = ticker_results[t]
        print(
            f"  {t:<8s} {r['n_samples']:5d} {r['auc']:7.4f} "
            f"{r['accuracy']:7.4f} {r['baseline']:7.4f} {r['mean_confidence']:7.4f}"
        )

    # --- Directional accuracy (UP vs DOWN predictions) ---
    print(f"\n  Directional prediction quality at {best_cov*100:.0f}% coverage:")
    if "acc_when_predict_up" in best:
        print(f"    Accuracy when predicting UP:   {best['acc_when_predict_up']:.4f}")
    if "acc_when_predict_down" in best:
        print(
            f"    Accuracy when predicting DOWN:  {best['acc_when_predict_down']:.4f}"
        )
    print(f"    Fraction predicted UP:          {best['pred_up_fraction']:.4f}")
    print(f"    Fraction predicted DOWN:        {best['pred_down_fraction']:.4f}")

    elapsed = time.time() - start

    # --- Save results ---
    output = {
        "experiment": "confidence_filtered_trading",
        "model": "imp1_dir_only",
        "checkpoint": str(CHECKPOINT_PATH.relative_to(ROOT)),
        "test_samples_total": int(len(test_idx)),
        "test_samples_valid": int(n_valid),
        "confidence_distribution": {
            "mean": float(conf_valid.mean()),
            "median": float(np.median(conf_valid)),
            "std": float(conf_valid.std()),
            "min": float(conf_valid.min()),
            "max": float(conf_valid.max()),
            "percentiles": {
                str(p): float(np.percentile(conf_valid, p))
                for p in [5, 10, 25, 50, 75, 90, 95]
            },
        },
        "gate_weights_mean": {
            name: float(gates.mean(axis=0)[i])
            for i, name in enumerate(["price", "gat", "doc", "news", "surprise"])
        },
        "threshold_results": threshold_results,
        "best_threshold": {
            "coverage": best["coverage"],
            "direction_auc": best["direction_auc"],
            "direction_acc": best["direction_acc"],
            "n_samples": best["n_samples"],
            "auc_improvement": float(
                best["direction_auc"] - all_result["direction_auc"]
            ),
        },
        "per_ticker_at_best": ticker_results,
        "total_time_seconds": round(elapsed, 1),
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to {OUTPUT_PATH.relative_to(ROOT)}")
    print(f"  Total time: {elapsed:.1f}s")
    print("\n" + "=" * 70)
    print("  DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
