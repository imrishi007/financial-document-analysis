"""Phase 11C: Weekly rebalancing backtest on V2 model predictions.

Uses actual model predictions from the V2 test set (not random).

Saves results to: models/phase11_backtest_results.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.fusion_dataset import FusionEmbeddingDataset
from src.evaluation.backtester_v2 import run_full_backtest
from src.models.fusion_model import MultimodalFusionModel
from src.utils.gpu import setup_gpu


def get_model_predictions(
    model_path: str = "models/fusion_model_best.pt",
    embeddings_path: str = "data/embeddings/fusion_embeddings.pt",
    macro_dim: int = 32,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Get model predictions and prices for test set."""
    device = setup_gpu(verbose=False)
    device_str = str(device)

    dataset = FusionEmbeddingDataset(embeddings_path)

    # Test set: >= 2024
    test_idx = [
        i for i in range(len(dataset))
        if dataset.dates[i] > "2023-12-31" and dataset.direction_label[i] >= 0
    ]

    model = MultimodalFusionModel(
        price_dim=256, gat_dim=256, doc_dim=768,
        macro_dim=macro_dim, surprise_dim=5,
        proj_dim=128, hidden_dim=256, dropout=0.3,
    ).to(device_str)
    ckpt = torch.load(model_path, weights_only=False, map_location=device_str)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    pred_rows = []
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
            prob = torch.softmax(out["direction_logits"], dim=1)[0, 1].item()
            pred_rows.append({
                "date": dataset.dates[idx],
                "ticker": dataset.tickers[idx],
                "pred_prob": prob,
            })

    pred_df = pd.DataFrame(pred_rows)
    pred_df["date"] = pd.to_datetime(pred_df["date"])

    # Load actual prices
    prices_dir = Path("data/raw/prices")
    price_rows = []
    for csv_path in sorted(prices_dir.glob("*_ohlcv.csv")):
        if csv_path.name == "tech10_ohlcv.csv":
            continue
        df = pd.read_csv(csv_path)
        df["date"] = pd.to_datetime(df["date"])
        # Filter to test period
        df = df[df["date"] >= "2024-01-01"]
        price_rows.append(df[["date", "ticker", "close"]])

    price_df = pd.concat(price_rows, ignore_index=True)

    return pred_df, price_df


def main():
    print("=" * 60)
    print("PHASE 11C: WEEKLY REBALANCING BACKTEST ON V2 MODEL")
    print("=" * 60)

    print("\n[1/2] Getting V2 model predictions on test set...")
    pred_df, price_df = get_model_predictions()
    print(f"  Predictions: {len(pred_df)} rows, {pred_df['ticker'].nunique()} tickers")
    print(f"  Date range: {pred_df['date'].min().date()} to {pred_df['date'].max().date()}")

    print("\n[2/2] Running weekly long-short backtest...")
    results = run_full_backtest(
        pred_df, price_df,
        cost_bps_list=[0, 5, 10, 20],
        n_long=2, n_short=2,
        rebalance_every=5,
        verbose=True,
    )

    # Save
    with open("models/phase11_backtest_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to models/phase11_backtest_results.json")
    print("\n=== PHASE 11C COMPLETE ===")


if __name__ == "__main__":
    main()
