"""Run Phase 4B: Document (10-K) model training."""

from src.train.train_document import run_document_training
from src.train.common import TrainingConfig

config = TrainingConfig(epochs=30, patience=8, batch_size=4, seed=42)
results = run_document_training(config=config, verbose=True)

if results.get("test_metrics"):
    print()
    print("--- Summary ---")
    tm = results["test_metrics"]
    for k, v in tm.items():
        print(f"  {k}: {v:.4f}")
    print(f"  Majority baseline: {results['baseline_accuracy']:.4f}")
else:
    print("--- Summary ---")
    print(
        f"  Train: {results.get('n_train', 0)}, Val: {results.get('n_val', 0)}, Test: {results.get('n_test', 0)}"
    )
    print("  No test metrics (insufficient test data)")
