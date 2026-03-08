"""Phase 6 runner — Multimodal Fusion.

Steps:
  1. Extract embeddings from all pretrained models (one-time, saved to disk).
  2. Train the gated-attention fusion model with multi-task loss.
  3. Save results to ``models/phase6_results.json``.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from src.features.extract_embeddings import extract_all_embeddings
from src.train.common import TrainingConfig
from src.train.train_fusion import run_fusion_training
from src.utils.seed import set_global_seed


def main() -> None:
    set_global_seed(42)

    EMB_PATH = Path("data/embeddings/fusion_embeddings.pt")
    RESULTS_PATH = Path("models/phase6_results.json")

    print("=" * 70)
    print("  PHASE 6: Multimodal Fusion")
    print("=" * 70)
    t0 = time.time()

    # ------------------------------------------------------------------
    # Step 1: Extract embeddings (skip if already done)
    # ------------------------------------------------------------------
    if not EMB_PATH.exists():
        print("\n--- Step 1: Extracting embeddings from all pretrained models ---")
        extract_summary = extract_all_embeddings(
            save_path=EMB_PATH,
            device="cuda",
            verbose=True,
        )
    else:
        print(f"\n--- Step 1: Embeddings already exist at {EMB_PATH} (skipping) ---")
        extract_summary = {"note": "pre-extracted"}

    # ------------------------------------------------------------------
    # Step 2: Train fusion model
    # ------------------------------------------------------------------
    print("\n--- Step 2: Training fusion model ---")
    config = TrainingConfig(
        epochs=60,
        patience=10,
        batch_size=512,
        learning_rate=5e-4,
        weight_decay=1e-3,
        seed=42,
    )

    result = run_fusion_training(
        embeddings_path=EMB_PATH,
        config=config,
        save_dir="models",
        lambda_dir=1.0,
        lambda_vol=0.5,
        lambda_surp=1.0,
        verbose=True,
    )

    # ------------------------------------------------------------------
    # Step 3: Save results
    # ------------------------------------------------------------------
    test = result["test_result"]

    output = {
        "fusion": {
            "status": "OK",
            "direction_metrics": test.get("direction", {}),
            "direction_baseline": test.get("direction_baseline", 0),
            "volatility_metrics": test.get("volatility", {}),
            "surprise_metrics": test.get("surprise", {}),
            "surprise_baseline": test.get("surprise_baseline", 0),
            "mean_gate_weights": test.get("mean_gate_weights", []),
            "epochs_trained": len(result["history"]),
            "n_train": result["n_train"],
            "n_val": result["n_val"],
            "n_test": result["n_test"],
            "time_seconds": round(result["time_seconds"], 1),
        },
        "extraction_summary": extract_summary,
    }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(output, f, indent=2, default=str)

    total_time = time.time() - t0
    print(f"\n{'='*70}")
    print(f"  PHASE 6 COMPLETE — total time: {total_time:.1f}s")
    print(f"  Results saved to {RESULTS_PATH}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
