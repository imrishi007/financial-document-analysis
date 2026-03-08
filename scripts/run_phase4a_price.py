"""Run Phase 4A: Price model training."""

from src.train.train_price import run_price_training
from src.train.common import TrainingConfig

config = TrainingConfig(epochs=20, patience=5, batch_size=32, seed=42)
results = run_price_training(config=config, verbose=True)

print()
print("--- Summary ---")
tm = results["test_metrics"]
print(f"Test Accuracy: {tm['accuracy']:.4f}")
print(f"Test F1:       {tm['f1']:.4f}")
print(f"Test AUC:      {tm.get('auc', 0):.4f}")
print(f"Baseline Acc:  {results['baseline_accuracy']:.4f}")
