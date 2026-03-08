"""Phase 7 — Ablation Studies & Consolidated Final Report.

Runs the following analyses using pre-extracted embeddings and saved results:
  1. Single-modality vs multimodal fusion (direction AUC)
  2. With-GAT vs without-GAT ablation
  3. Multi-task vs single-task comparison
  4. Per-ticker direction AUC breakdown
  5. Modality coverage impact analysis
  6. Consolidated cross-phase results table

No new training is required — all metrics are read from phase4/5/6 JSON files
and the pre-extracted fusion embeddings.
"""

from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.fusion_dataset import FusionEmbeddingDataset
from src.models.fusion_model import MultimodalFusionModel


def load_results():
    p4 = json.loads((ROOT / "models/phase4_results.json").read_text())
    p5 = json.loads((ROOT / "models/phase5_results.json").read_text())
    p6 = json.loads((ROOT / "models/phase6_results.json").read_text())
    return p4, p5, p6


def single_modality_ablation(model, dataset, device):
    """Zero out all modalities except one at a time and evaluate direction AUC."""
    model.eval()
    modality_names = ["price", "gat", "doc", "news", "surprise"]
    results = {}

    # Test split: dates >= 2024
    test_indices = [i for i in range(len(dataset)) if dataset.dates[i] >= "2024"]

    for mod_idx, mod_name in enumerate(modality_names):
        all_probs, all_labels = [], []

        # Process in batches
        batch_size = 512
        for start in range(0, len(test_indices), batch_size):
            batch_idx = test_indices[start : start + batch_size]
            batch = [dataset[i] for i in batch_idx]

            price = torch.stack([b["price_emb"] for b in batch]).to(device)
            gat = torch.stack([b["gat_emb"] for b in batch]).to(device)
            doc = torch.stack([b["doc_emb"] for b in batch]).to(device)
            news = torch.stack([b["news_emb"] for b in batch]).to(device)
            surp = torch.stack([b["surprise_feat"] for b in batch]).to(device)
            mask = torch.stack([b["modality_mask"] for b in batch]).to(device)
            labels = torch.stack([b["direction_label"] for b in batch])

            # Create single-modality mask: only mod_idx is 1
            ablation_mask = torch.zeros_like(mask)
            ablation_mask[:, mod_idx] = mask[:, mod_idx]

            with torch.no_grad():
                out = model(price, gat, doc, news, surp, ablation_mask)
                probs = torch.softmax(out["direction_logits"], dim=1)[:, 1].cpu()

            all_probs.append(probs)
            all_labels.append(labels)

        all_probs = torch.cat(all_probs).numpy()
        all_labels = torch.cat(all_labels).numpy()

        try:
            auc = roc_auc_score(all_labels, all_probs)
        except ValueError:
            auc = 0.5

        results[mod_name] = {
            "auc": float(auc),
            "n_test": len(test_indices),
        }
        print(f"    {mod_name:10s}  AUC={auc:.4f}")

    return results


def per_ticker_analysis(model, dataset, device):
    """Compute direction AUC per ticker on test split."""
    model.eval()
    test_indices = [i for i in range(len(dataset)) if dataset.dates[i] >= "2024"]

    # Group by ticker
    ticker_data = defaultdict(lambda: {"probs": [], "labels": []})

    batch_size = 512
    for start in range(0, len(test_indices), batch_size):
        batch_idx = test_indices[start : start + batch_size]
        batch = [dataset[i] for i in batch_idx]

        price = torch.stack([b["price_emb"] for b in batch]).to(device)
        gat = torch.stack([b["gat_emb"] for b in batch]).to(device)
        doc = torch.stack([b["doc_emb"] for b in batch]).to(device)
        news = torch.stack([b["news_emb"] for b in batch]).to(device)
        surp = torch.stack([b["surprise_feat"] for b in batch]).to(device)
        mask = torch.stack([b["modality_mask"] for b in batch]).to(device)
        labels = [b["direction_label"].item() for b in batch]

        with torch.no_grad():
            out = model(price, gat, doc, news, surp, mask)
            probs = torch.softmax(out["direction_logits"], dim=1)[:, 1].cpu().numpy()

        for j, idx in enumerate(batch_idx):
            ticker = dataset.tickers[idx]
            ticker_data[ticker]["probs"].append(probs[j])
            ticker_data[ticker]["labels"].append(labels[j])

    results = {}
    for ticker in sorted(ticker_data.keys()):
        td = ticker_data[ticker]
        p = np.array(td["probs"])
        l = np.array(td["labels"])
        try:
            auc = roc_auc_score(l, p)
        except ValueError:
            auc = 0.5

        up_pct = l.mean() * 100
        results[ticker] = {
            "auc": float(auc),
            "n_samples": len(l),
            "up_pct": float(up_pct),
        }
        print(f"    {ticker:6s}  AUC={auc:.4f}  n={len(l):5d}  UP%={up_pct:.1f}")

    return results


def main():
    t0 = time.time()
    print("=" * 60)
    print("  PHASE 7 — ABLATION STUDIES & FINAL REPORT")
    print("=" * 60)

    p4, p5, p6 = load_results()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # Load model and dataset
    print("Loading fusion model and embeddings...")
    model = MultimodalFusionModel().to(device)
    ckpt = torch.load(
        ROOT / "models/fusion_model_best.pt", map_location=device, weights_only=False
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    emb_path = ROOT / "data/embeddings/fusion_embeddings.pt"
    dataset = FusionEmbeddingDataset(emb_path)
    print(f"  Dataset: {len(dataset)} samples\n")

    # ================================================================
    # Ablation 1: Single-modality vs Full Fusion
    # ================================================================
    print("[1/5] Single-Modality Ablation (Direction AUC, test split):")
    single_mod_results = single_modality_ablation(model, dataset, device)

    # ================================================================
    # Ablation 2: GAT vs No-GAT (from Phase 5)
    # ================================================================
    print("\n[2/5] GAT vs No-GAT Ablation (from Phase 5):")
    gat_auc = p5["graph_gat"]["test_metrics"]["auc"]
    nogat_auc = p5["no_gat_ablation"]["test_metrics"]["auc"]
    delta = gat_auc - nogat_auc
    print(f"    With GAT:    AUC={gat_auc:.4f}")
    print(f"    Without GAT: AUC={nogat_auc:.4f}")
    print(f"    Delta:       {delta:+.4f}")

    gat_ablation = {
        "with_gat_auc": gat_auc,
        "without_gat_auc": nogat_auc,
        "delta": delta,
    }

    # ================================================================
    # Ablation 3: Multi-task vs Single-task
    # ================================================================
    print("\n[3/5] Multi-Task vs Single-Task Comparison:")
    # Single-task = Phase 4 price model (direction only)
    # Multi-task = Phase 6 fusion (direction + vol + surprise)
    price_single_auc = p4["price"]["test_metrics"]["auc"]
    fusion_multi_auc = p6["fusion"]["direction_metrics"]["auc"]
    vol_r2 = p6["fusion"]["volatility_metrics"]["r2"]

    print(f"    Single-task (price, direction only): AUC={price_single_auc:.4f}")
    print(f"    Multi-task (fusion, direction head):  AUC={fusion_multi_auc:.4f}")
    print(f"    Multi-task volatility R²:             {vol_r2:.4f}")
    print(f"    Multi-task adds volatility + surprise predictions")

    multitask_comparison = {
        "single_task_direction_auc": price_single_auc,
        "multitask_direction_auc": fusion_multi_auc,
        "multitask_volatility_r2": vol_r2,
        "multitask_surprise_auc": p6["fusion"]["surprise_metrics"]["auc"],
    }

    # ================================================================
    # Ablation 4: Per-Ticker Analysis
    # ================================================================
    print("\n[4/5] Per-Ticker Direction AUC (fusion model, test split):")
    ticker_results = per_ticker_analysis(model, dataset, device)

    # ================================================================
    # Ablation 5: Consolidated Cross-Phase Comparison
    # ================================================================
    print("\n[5/5] Consolidated Cross-Phase Direction AUC Comparison:")
    cross_phase = {
        "Price (CNN-BiLSTM)": p4["price"]["test_metrics"]["auc"],
        "Document (FinBERT)": p4["document"]["test_metrics"]["auc"],
        "News (FinBERT+BiGRU)": p4["news"]["test_metrics"]["auc"],
        "Surprise (MLP)": p4["surprise"]["test_metrics"]["auc"],
        "GAT (Graph Attention)": p5["graph_gat"]["test_metrics"]["auc"],
        "No-GAT (Ablation)": p5["no_gat_ablation"]["test_metrics"]["auc"],
        "Fusion (Gated Multi-Task)": p6["fusion"]["direction_metrics"]["auc"],
    }

    print(f"\n    {'Model':<28s}  {'AUC':>7s}  {'Δ vs 0.50':>10s}")
    print("    " + "-" * 50)
    for name, auc in sorted(cross_phase.items(), key=lambda x: -x[1]):
        d = auc - 0.5
        print(f"    {name:<28s}  {auc:.4f}    {d:+.4f}")

    # Gate weights
    gate_names = ["price", "gat", "doc", "news", "surprise"]
    gate_vals = p6["fusion"]["mean_gate_weights"]
    print(f"\n    Fusion Gate Weights:")
    for gn, gv in zip(gate_names, gate_vals):
        print(f"      {gn:10s}: {gv:.4f}")

    # ================================================================
    # Save all ablation results
    # ================================================================
    results = {
        "single_modality_ablation": single_mod_results,
        "gat_ablation": gat_ablation,
        "multitask_comparison": multitask_comparison,
        "per_ticker_analysis": ticker_results,
        "cross_phase_direction_auc": cross_phase,
        "fusion_gate_weights": dict(zip(gate_names, [float(g) for g in gate_vals])),
        "fusion_volatility": p6["fusion"]["volatility_metrics"],
        "fusion_surprise": p6["fusion"]["surprise_metrics"],
        "time_seconds": time.time() - t0,
    }

    out_path = ROOT / "models/phase7_ablation_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {out_path}")

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"  PHASE 7 COMPLETE — {elapsed:.1f}s")
    print(f"{'='*60}")

    return results


if __name__ == "__main__":
    main()
