"""Phase 13 Step 7: Run volatility trading strategy backtest."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch

from src.utils.seed import set_global_seed
from src.evaluation.vol_strategy_backtester import VolatilityStrategyBacktester


def main():
    set_global_seed(42)
    print("=" * 60)
    print("PHASE 13 STEP 7: VOLATILITY TRADING STRATEGY BACKTEST")
    print("=" * 60)

    ALL_30 = [
        "AAPL",
        "MSFT",
        "GOOGL",
        "AMZN",
        "NVDA",
        "META",
        "TSLA",
        "AMD",
        "INTC",
        "ORCL",
        "QCOM",
        "TXN",
        "AVGO",
        "MRVL",
        "KLAC",
        "CRM",
        "ADBE",
        "NOW",
        "SNOW",
        "DDOG",
        "NFLX",
        "UBER",
        "PYPL",
        "SNAP",
        "DELL",
        "AMAT",
        "LRCX",
        "IBM",
        "CSCO",
        "HPE",
    ]

    # Load price data
    print("\n[1] Loading price data...")
    price_data = {}
    for ticker in ALL_30:
        csv_path = Path(f"data/raw/prices/{ticker}_ohlcv.csv")
        if csv_path.exists():
            price_data[ticker] = pd.read_csv(csv_path, parse_dates=["date"])
    print(f"  Loaded {len(price_data)} tickers")

    # Load Phase 13 embeddings for model predictions
    print("\n[2] Loading Phase 13 embeddings...")
    emb = torch.load("data/embeddings/phase13_fusion_embeddings.pt", weights_only=False)
    sample_tickers = np.array(emb["tickers"])
    sample_dates = np.array(emb["dates"])

    # Load best vol model and run MC Dropout for uncertainty
    print("\n[3] Running MC Dropout for uncertainty estimates...")
    from src.models.fusion_model import Phase13FusionModel
    from src.utils.gpu import setup_gpu

    device = setup_gpu(verbose=False)
    dev = str(device)

    model = Phase13FusionModel(
        price_dim=256,
        gat_dim=256,
        doc_dim=768,
        macro_dim=32,
        surprise_dim=5,
        proj_dim=128,
        hidden_dim=256,
        dropout=0.3,
        mc_dropout=True,
        lambda_vol=0.85,
        lambda_dir=0.15,
    ).to(dev)

    ckpt = torch.load(
        "models/phase13_fusion_vol_primary.pt", weights_only=False, map_location=dev
    )
    model.load_state_dict(ckpt["model_state_dict"])

    # Test set only (dates > 2023-12-31)
    dates_pd = pd.to_datetime(pd.Series(sample_dates))
    test_mask = (dates_pd > "2023-12-31").values
    test_idx = np.where(test_mask)[0]
    print(f"  Test samples: {len(test_idx)}")

    # Run MC Dropout in batches
    BATCH = 4096
    N_MC = 30
    all_vol_mean = []
    all_vol_std = []
    all_dir_mean = []

    model.train()  # keep dropout active for MC
    with torch.no_grad():
        for start in range(0, len(test_idx), BATCH):
            end = min(start + BATCH, len(test_idx))
            idx = test_idx[start:end]

            p_e = emb["price_emb"][idx].to(dev)
            g_e = emb["gat_emb"][idx].to(dev)
            d_e = emb["doc_emb"][idx].to(dev)
            m_e = emb["macro_emb"][idx].to(dev)
            s = emb["surprise_feat"][idx].to(dev)
            mk = emb["modality_mask"][idx].to(dev)

            mc_out = model.predict_with_uncertainty(
                p_e, g_e, d_e, m_e, s, mk, n_samples=N_MC
            )
            all_vol_mean.append(mc_out["vol_mean"].cpu().numpy())
            all_vol_std.append(mc_out["vol_std"].cpu().numpy())
            dir_probs = torch.softmax(mc_out["dir_mean"], dim=1)[:, 1]
            all_dir_mean.append(dir_probs.cpu().numpy())

    vol_predictions = np.concatenate(all_vol_mean)
    vol_uncertainty = np.concatenate(all_vol_std)
    dir_predictions = np.concatenate(all_dir_mean)

    test_dates = sample_dates[test_idx]
    test_tickers = sample_tickers[test_idx]

    print(f"  Vol predictions shape: {vol_predictions.shape}")
    print(f"  Vol uncertainty mean: {vol_uncertainty.mean():.4f}")
    print(f"  Dir predictions shape: {dir_predictions.shape}")

    # Run backtest
    print("\n[4] Running full backtest...")
    backtester = VolatilityStrategyBacktester(
        tickers=ALL_30,
        price_data=price_data,
        vol_predictions=vol_predictions,
        vol_uncertainty=vol_uncertainty,
        dir_predictions=dir_predictions,
        sample_dates=test_dates,
        sample_tickers=test_tickers,
        vol_threshold=0.15,
        rebalance_freq=5,
        holding_period=20,
        top_k=5,
        risk_free_rate=0.05,
    )

    results_df = backtester.run_full_backtest(cost_levels=[0, 5, 10, 20])

    print("\n" + "=" * 100)
    print("BACKTEST RESULTS")
    print("=" * 100)
    print(
        results_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )
    print("=" * 100)

    # Save results
    results_dict = results_df.to_dict(orient="records")
    Path("models/phase13_backtest_results.json").write_text(
        json.dumps(results_dict, indent=2)
    )

    # Save daily returns for the main strategies
    vol_rets_0 = backtester.run_vol_strategy(0)
    dir_rets_0 = backtester.run_direction_strategy(0)
    bh_rets = backtester.run_buy_and_hold()

    returns_df = pd.DataFrame(
        {
            "vol_strategy_0bp": vol_rets_0,
            "dir_strategy_0bp": dir_rets_0,
            "buy_and_hold": bh_rets,
        }
    )
    returns_df.to_csv("models/phase13_backtest_returns.csv")

    # Highlight key metrics
    vol_0 = next(
        (r for r in results_dict if r["strategy"] == "Vol Strategy (0bp)"), None
    )
    dir_0 = next(
        (r for r in results_dict if r["strategy"] == "Direction Strategy (0bp)"), None
    )
    bh = next((r for r in results_dict if r["strategy"] == "Buy and Hold"), None)

    print("\n  KEY FINDINGS:")
    if vol_0:
        print(
            f"    Vol Strategy (0bp): Sharpe={vol_0['sharpe']:.3f}, Return={vol_0['ann_return']:.2%}, MaxDD={vol_0['max_drawdown']:.2%}"
        )
    if dir_0:
        print(
            f"    Dir Strategy (0bp): Sharpe={dir_0['sharpe']:.3f}, Return={dir_0['ann_return']:.2%}, MaxDD={dir_0['max_drawdown']:.2%}"
        )
    if bh:
        print(
            f"    Buy and Hold:       Sharpe={bh['sharpe']:.3f}, Return={bh['ann_return']:.2%}, MaxDD={bh['max_drawdown']:.2%}"
        )

    print("\n  Saved: models/phase13_backtest_results.json")
    print("  Saved: models/phase13_backtest_returns.csv")
    print("\nPHASE 13 STEP 7 COMPLETE — Volatility strategy backtested")


if __name__ == "__main__":
    main()
