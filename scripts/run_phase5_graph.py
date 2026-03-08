"""Run Phase 5 — Graph Attention Network training + No-GAT ablation."""

from __future__ import annotations

import json
import time
import sys
from pathlib import Path

# Ensure project root on path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main() -> None:
    from src.train.train_graph import run_graph_training

    print("=" * 70)
    print("  PHASE 5: Graph-Enhanced Direction Model (GAT)")
    print("=" * 70)

    t0 = time.time()

    result = run_graph_training(
        prices_dir="data/raw/prices",
        targets_dir="data/targets",
        graph_nodes_csv="data/raw/graph/tech10_nodes.csv",
        graph_edges_csv="data/raw/graph/tech10_edges.csv",
        save_dir="models",
        verbose=True,
    )

    elapsed = time.time() - t0

    # Save summary
    summary = {
        "graph_gat": {
            "status": "OK",
            "test_metrics": {k: float(v) for k, v in result["test_metrics"].items()},
            "baseline_accuracy": float(result["baseline_accuracy"]),
            "epochs_trained": len(result["history"]),
            "n_train": result["n_train"],
            "n_val": result["n_val"],
            "n_test": result["n_test"],
            "time_seconds": round(elapsed, 1),
        },
        "no_gat_ablation": {
            "status": "OK",
            "test_metrics": {
                k: float(v) for k, v in result["no_gat_test_metrics"].items()
            },
        },
    }

    out_path = Path("models") / "phase5_results.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'=' * 70}")
    print("  PHASE 5 SUMMARY")
    print(f"{'=' * 70}")
    gat = summary["graph_gat"]["test_metrics"]
    nogat = summary["no_gat_ablation"]["test_metrics"]
    bl = summary["graph_gat"]["baseline_accuracy"]
    print(
        f"  GAT     | acc={gat['accuracy']:.4f} f1={gat['f1']:.4f} auc={gat['auc']:.4f} | baseline={bl:.4f}"
    )
    print(
        f"  No-GAT  | acc={nogat['accuracy']:.4f} f1={nogat['f1']:.4f} auc={nogat['auc']:.4f}"
    )
    delta = gat["accuracy"] - nogat["accuracy"]
    print(f"  GAT vs No-GAT delta: {delta:+.4f}")
    print(f"  Total time: {elapsed:.1f}s")
    print(f"  Results saved to {out_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
