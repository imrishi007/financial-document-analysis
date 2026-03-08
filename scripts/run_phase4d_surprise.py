"""Run Phase 4D: Fundamental surprise prediction training."""

from src.train.train_surprise import run_surprise_training
from src.train.common import TrainingConfig

config = TrainingConfig(epochs=30, patience=8, batch_size=16, seed=42)
results = run_surprise_training(config=config, verbose=True)

if results.get("test_metrics"):
    print()
    print("--- Summary ---")
    tm = results["test_metrics"]
    for k, v in tm.items():
        print(f"  {k}: {v:.4f}")
    print(f"  Majority baseline: {results['baseline_accuracy']:.4f}")
else:
    print("No test results available.")
