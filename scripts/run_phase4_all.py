"""Run all Phase 4 training: Price, Document, News, Surprise.

Results are printed to console and saved to models/ directory.
"""

import json
import sys
import time
import traceback
from pathlib import Path

import torch

from src.utils.seed import set_global_seed

print("=" * 70)
print("PHASE 4: Individual Model Training")
print(
    f"Device: {'cuda — ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}"
)
print("=" * 70)

results_summary = {}

# ======================================================================
# 4A: Price Direction Model (CNN-LSTM)
# ======================================================================
print("\n" + "=" * 70)
print("4A: Price Direction Model (CNN-LSTM)")
print("=" * 70)

try:
    from src.train.train_price import run_price_training
    from src.train.common import TrainingConfig

    t0 = time.time()
    price_results = run_price_training(
        config=TrainingConfig(epochs=20, patience=5, batch_size=64, seed=42),
        verbose=True,
    )
    elapsed = time.time() - t0

    results_summary["price"] = {
        "status": "OK",
        "test_metrics": price_results["test_metrics"],
        "baseline_accuracy": price_results["baseline_accuracy"],
        "epochs_trained": len(price_results["history"]),
        "time_seconds": round(elapsed, 1),
    }
    print(f"\n  4A completed in {elapsed:.1f}s")

except Exception as e:
    traceback.print_exc()
    results_summary["price"] = {"status": "FAILED", "error": str(e)}

# Free GPU memory
torch.cuda.empty_cache() if torch.cuda.is_available() else None

# ======================================================================
# 4D: Surprise Prediction (BEAT/MISS) - price-based, fast
# ======================================================================
print("\n" + "=" * 70)
print("4D: Fundamental Surprise Prediction (BEAT/MISS)")
print("=" * 70)

try:
    from src.train.train_surprise import run_surprise_training

    t0 = time.time()
    surprise_results = run_surprise_training(
        config=TrainingConfig(epochs=30, patience=8, batch_size=16, seed=42),
        verbose=True,
    )
    elapsed = time.time() - t0

    results_summary["surprise"] = {
        "status": "OK",
        "test_metrics": surprise_results["test_metrics"],
        "baseline_accuracy": surprise_results["baseline_accuracy"],
        "epochs_trained": len(surprise_results["history"]),
        "time_seconds": round(elapsed, 1),
    }
    print(f"\n  4D completed in {elapsed:.1f}s")

except Exception as e:
    traceback.print_exc()
    results_summary["surprise"] = {"status": "FAILED", "error": str(e)}

torch.cuda.empty_cache() if torch.cuda.is_available() else None

# ======================================================================
# 4B: Document Direction Model (FinBERT + Attention Pooling)
# ======================================================================
print("\n" + "=" * 70)
print("4B: Document Direction Model (FinBERT + AttentionPooling)")
print("=" * 70)

try:
    from src.train.train_document import run_document_training

    t0 = time.time()
    # Small batch size for 4GB VRAM — FinBERT chunks are memory-heavy
    doc_results = run_document_training(
        config=TrainingConfig(epochs=30, patience=8, batch_size=2, seed=42),
        verbose=True,
    )
    elapsed = time.time() - t0

    results_summary["document"] = {
        "status": "OK",
        "test_metrics": doc_results["test_metrics"],
        "baseline_accuracy": doc_results["baseline_accuracy"],
        "epochs_trained": len(doc_results["history"]),
        "n_train": doc_results["n_train"],
        "n_val": doc_results["n_val"],
        "n_test": doc_results["n_test"],
        "time_seconds": round(elapsed, 1),
    }
    print(f"\n  4B completed in {elapsed:.1f}s")

except Exception as e:
    traceback.print_exc()
    results_summary["document"] = {"status": "FAILED", "error": str(e)}

torch.cuda.empty_cache() if torch.cuda.is_available() else None

# ======================================================================
# 4C: News Temporal Model (FinBERT + BiGRU)
# ======================================================================
print("\n" + "=" * 70)
print("4C: News Temporal Model -- REMOVED IN V2")
print("=" * 70)
print("  Skipped: train_news.py was removed. News had <2% coverage and negligible")
print("  gate weight. Use run_v2_pipeline.py for the V2 pipeline.")
results_summary["news"] = {"status": "SKIPPED", "reason": "removed in V2"}

torch.cuda.empty_cache() if torch.cuda.is_available() else None

# ======================================================================
# Final Summary
# ======================================================================
print("\n" + "=" * 70)
print("PHASE 4 SUMMARY")
print("=" * 70)

for model_name, info in results_summary.items():
    status = info.get("status", "UNKNOWN")
    if status == "OK":
        tm = info.get("test_metrics", {})
        baseline = info.get("baseline_accuracy", 0)
        acc = tm.get("accuracy", 0)
        f1 = tm.get("f1", 0)
        auc = tm.get("auc", 0)
        secs = info.get("time_seconds", 0)
        print(
            f"\n  {model_name.upper():12s} | acc={acc:.4f} f1={f1:.4f} auc={auc:.4f} | baseline={baseline:.4f} | {secs:.0f}s"
        )
    else:
        print(
            f"\n  {model_name.upper():12s} | FAILED: {info.get('error', 'unknown')[:80]}"
        )

# Save summary to JSON
summary_path = Path("models") / "phase4_results.json"
summary_path.parent.mkdir(parents=True, exist_ok=True)


# Convert numpy types for JSON serialization
def make_serializable(obj):
    if hasattr(obj, "item"):
        return obj.item()
    if hasattr(obj, "tolist"):
        return obj.tolist()
    return obj


clean_summary = {}
for k, v in results_summary.items():
    clean_summary[k] = {}
    for kk, vv in v.items():
        if isinstance(vv, dict):
            clean_summary[k][kk] = {
                kkk: make_serializable(vvv) for kkk, vvv in vv.items()
            }
        else:
            clean_summary[k][kk] = make_serializable(vv)

with open(summary_path, "w") as f:
    json.dump(clean_summary, f, indent=2)

print(f"\nResults saved to {summary_path}")
print("=" * 70)
