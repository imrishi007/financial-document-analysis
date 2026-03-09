"""Phase 11A: Confidence-filtered trading analysis on V2 fusion model.

Reuses V1 confidence filtering logic but on the V2 model + 60-day labels.

Key differences from V1:
- Uses V2 fusion model checkpoint (60-day horizon)
- Uses V2 embeddings (4 modalities, no news, no leaky surprise)
- Confidence = max(P(UP_60d), P(DOWN_60d))
- Per-ticker analysis with filtered AUC
- Compare against V1 confidence filtering results

Saves results to: models/phase11_confidence_results.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.fusion_dataset import FusionEmbeddingDataset
from src.models.fusion_model import MultimodalFusionModel
from src.utils.gpu import setup_gpu


THRESHOLDS = [0.50, 0.52, 0.54, 0.55, 0.56, 0.57, 0.58, 0.60, 0.62, 0.65, 0.70]


def run_confidence_filtering(
    model_path: str = "models/fusion_model_best.pt",
    embeddings_path: str = "data/embeddings/fusion_embeddings.pt",
    macro_dim: int = 32,
    save_path: str = "models/phase11_confidence_results.json",
    label: str = "V2",
) -> dict:
    """Run confidence filtering sweep and per-ticker analysis."""
    device = setup_gpu(verbose=False)
    device_str = str(device)

    # Load dataset
    dataset = FusionEmbeddingDataset(embeddings_path)

    # Split: test >= 2024
    test_idx = [
        i for i in range(len(dataset))
        if dataset.dates[i] > "2023-12-31" and dataset.direction_label[i] >= 0
    ]
    print(f"[{label}] Test samples: {len(test_idx)}")

    # Load model
    model = MultimodalFusionModel(
        price_dim=256, gat_dim=256, doc_dim=768,
        macro_dim=macro_dim, surprise_dim=5,
        proj_dim=128, hidden_dim=256, dropout=0.3,
    ).to(device_str)
    ckpt = torch.load(model_path, weights_only=False, map_location=device_str)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Run inference on test set
    all_probs = []
    all_labels = []
    all_tickers = []

    with torch.no_grad():
        for idx in test_idx:
            sample = dataset[idx]
            for k in ["price_emb", "gat_emb", "doc_emb", "macro_emb",
                       "surprise_feat", "modality_mask"]:
                sample[k] = sample[k].unsqueeze(0).to(device_str)
            out = model(
                sample["price_emb"], sample["gat_emb"], sample["doc_emb"],
                sample["macro_emb"], sample["surprise_feat"], sample["modality_mask"],
            )
            probs = torch.softmax(out["direction_logits"], dim=1)
            up_prob = probs[0, 1].item()
            all_probs.append(up_prob)
            all_labels.append(dataset.direction_label[idx].item())
            all_tickers.append(dataset.tickers[idx])

    probs_arr = np.array(all_probs)
    labels_arr = np.array(all_labels)
    tickers_arr = np.array(all_tickers)
    confidences = np.maximum(probs_arr, 1 - probs_arr)

    # Full unfiltered metrics
    full_auc = roc_auc_score(labels_arr, probs_arr) if len(np.unique(labels_arr)) > 1 else 0.5
    full_acc = accuracy_score(labels_arr, (probs_arr >= 0.5).astype(int))
    majority = max(np.mean(labels_arr), 1 - np.mean(labels_arr))

    print(f"[{label}] Full test: AUC={full_auc:.4f}, Acc={full_acc:.4f}, Baseline={majority:.4f}")

    # Threshold sweep
    sweep_results = []
    for thresh in THRESHOLDS:
        mask = confidences >= thresh
        n_retained = mask.sum()
        coverage = n_retained / len(labels_arr)

        if n_retained < 10 or len(np.unique(labels_arr[mask])) < 2:
            sweep_results.append({
                "threshold": thresh, "N": int(n_retained),
                "coverage_pct": float(coverage * 100),
                "auc": None, "accuracy": None, "f1": None,
            })
            continue

        filtered_auc = roc_auc_score(labels_arr[mask], probs_arr[mask])
        filtered_acc = accuracy_score(labels_arr[mask], (probs_arr[mask] >= 0.5).astype(int))
        filtered_f1 = f1_score(labels_arr[mask], (probs_arr[mask] >= 0.5).astype(int))
        filtered_baseline = max(np.mean(labels_arr[mask]), 1 - np.mean(labels_arr[mask]))

        sweep_results.append({
            "threshold": thresh,
            "N": int(n_retained),
            "coverage_pct": round(float(coverage * 100), 2),
            "auc": round(float(filtered_auc), 4),
            "accuracy": round(float(filtered_acc), 4),
            "f1": round(float(filtered_f1), 4),
            "baseline_acc": round(float(filtered_baseline), 4),
        })

    # Per-ticker analysis (unfiltered vs filtered at ~10% coverage)
    # Find threshold closest to 10% coverage
    for s in sorted(sweep_results, key=lambda x: abs((x["coverage_pct"] or 100) - 10)):
        if s["auc"] is not None and (s["coverage_pct"] or 100) <= 15:
            best_filter_thresh = s["threshold"]
            break
    else:
        best_filter_thresh = 0.60

    ticker_results = {}
    for ticker in sorted(set(tickers_arr)):
        tmask = tickers_arr == ticker
        if np.sum(tmask) < 10 or len(np.unique(labels_arr[tmask])) < 2:
            continue
        t_auc = roc_auc_score(labels_arr[tmask], probs_arr[tmask])
        # Filtered
        cmask = tmask & (confidences >= best_filter_thresh)
        if np.sum(cmask) >= 5 and len(np.unique(labels_arr[cmask])) >= 2:
            t_auc_filt = roc_auc_score(labels_arr[cmask], probs_arr[cmask])
        else:
            t_auc_filt = None
        ticker_results[ticker] = {
            "unfiltered_auc": round(float(t_auc), 4),
            "filtered_auc": round(float(t_auc_filt), 4) if t_auc_filt else None,
            "n_total": int(np.sum(tmask)),
            "n_filtered": int(np.sum(cmask)),
        }

    # Find best AUC from sweep
    valid_sweeps = [s for s in sweep_results if s["auc"] is not None]
    best_entry = max(valid_sweeps, key=lambda x: x["auc"]) if valid_sweeps else None

    results = {
        "model": label,
        "macro_dim": macro_dim,
        "test_samples": len(test_idx),
        "full_auc": round(full_auc, 4),
        "full_accuracy": round(full_acc, 4),
        "majority_baseline": round(majority, 4),
        "sweep": sweep_results,
        "per_ticker": ticker_results,
        "best_filtered_auc": best_entry["auc"] if best_entry else None,
        "best_filtered_coverage": best_entry["coverage_pct"] if best_entry else None,
        "best_filter_threshold": best_entry["threshold"] if best_entry else None,
    }

    # Print sweep table
    print(f"\n[{label}] Confidence Filtering Sweep:")
    print(f"{'Thresh':<8} {'N':>6} {'Coverage':>9} {'AUC':>8} {'Acc':>8} {'F1':>8}")
    print("-" * 55)
    for s in sweep_results:
        auc_str = f"{s['auc']:.4f}" if s["auc"] is not None else "  n/a"
        acc_str = f"{s['accuracy']:.4f}" if s.get("accuracy") is not None else "  n/a"
        f1_str = f"{s['f1']:.4f}" if s.get("f1") is not None else "  n/a"
        print(f"{s['threshold']:<8.2f} {s['N']:>6} {s['coverage_pct']:>8.1f}% {auc_str:>8} {acc_str:>8} {f1_str:>8}")

    if best_entry:
        print(f"\nBest: AUC={best_entry['auc']:.4f} at {best_entry['coverage_pct']:.1f}% coverage (threshold={best_entry['threshold']})")

    # Save
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {save_path}")

    return results


if __name__ == "__main__":
    print("=" * 60)
    print("PHASE 11A: CONFIDENCE FILTERING ON V2 MODEL")
    print("=" * 60)
    run_confidence_filtering()
    print("\n=== PHASE 11A COMPLETE ===")
