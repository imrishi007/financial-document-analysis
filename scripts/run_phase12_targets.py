"""Phase 12 Step 4: Target generation for all 30 stocks."""

from __future__ import annotations
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
from src.data.target_builder import (
    build_binary_direction_labels,
    build_multi_horizon_direction_labels,
    build_realized_volatility_targets,
    build_fundamental_surprise_targets,
)

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


def main():
    print("=" * 60)
    print("PHASE 12 STEP 4: TARGET GENERATION FOR ALL 30 STOCKS")
    print("=" * 60)

    # Load all price data
    prices_dir = Path("data/raw/prices")
    all_prices = []
    for ticker in ALL_30:
        csv_path = prices_dir / f"{ticker}_ohlcv.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            df["date"] = pd.to_datetime(df["date"]).dt.date
            all_prices.append(df)
    prices = pd.concat(all_prices, ignore_index=True)
    price_subset = prices[["ticker", "date", "close"]].copy()
    print(f"Total price rows: {len(prices)} across {len(ALL_30)} tickers")

    targets_dir = Path("data/targets")
    targets_dir.mkdir(parents=True, exist_ok=True)

    # 1. Legacy direction labels (60-day)
    print("\n--- Building direction labels (60d) ---")
    labels = build_binary_direction_labels(price_subset, horizon_days=60, threshold=0.0)
    labels.to_csv(targets_dir / "direction_labels.csv", index=False)
    vc = labels["target_label"].value_counts().to_dict()
    print(f"  {len(labels)} rows: {vc}")

    # 2. Multi-horizon direction labels (5d + 60d)
    print("\n--- Building multi-horizon direction labels (5d + 60d) ---")
    multi_labels = build_multi_horizon_direction_labels(
        price_subset, horizons=[5, 60], threshold=0.0
    )
    multi_labels.to_csv(targets_dir / "direction_labels_multi_horizon.csv", index=False)
    print(f"  {len(multi_labels)} rows")
    for h in [5, 60]:
        col = f"direction_{h}d_label"
        if col in multi_labels.columns:
            print(f"  {h}d: {multi_labels[col].value_counts().to_dict()}")

    # 3. Realized volatility
    print("\n--- Building volatility targets (20d) ---")
    vol_targets = build_realized_volatility_targets(price_subset, lookback_days=20)
    vol_targets.to_csv(targets_dir / "volatility_targets.csv", index=False)
    vol_col = "realized_vol_20d_annualized"
    print(f"  {len(vol_targets)} rows")
    print(f"  Mean vol: {vol_targets[vol_col].mean():.4f}")
    print(f"  Std vol:  {vol_targets[vol_col].std():.4f}")
    q25 = vol_targets[vol_col].quantile(0.25)
    q50 = vol_targets[vol_col].quantile(0.5)
    q75 = vol_targets[vol_col].quantile(0.75)
    print(f"  Percentiles: 25%={q25:.4f}, 50%={q50:.4f}, 75%={q75:.4f}")

    # 4. Fundamental surprise targets
    print("\n--- Building fundamental surprise targets ---")
    earnings_csv = Path("data/raw/earnings/earnings_surprise.csv")
    if earnings_csv.exists():
        earnings = pd.read_csv(earnings_csv)
        surprise_targets = build_fundamental_surprise_targets(
            earnings=earnings,
            prices=prices[["ticker", "date"]],
        )
        surprise_targets.to_csv(
            targets_dir / "fundamental_surprise_targets.csv", index=False
        )
        sc = surprise_targets["surprise_label"].value_counts().to_dict()
        print(f"  {len(surprise_targets)} rows: {sc}")
    else:
        print("  WARNING: No earnings CSV found")

    # Split statistics
    print("\n--- Split Statistics ---")
    multi_labels["date"] = pd.to_datetime(multi_labels["date"])
    train = multi_labels[multi_labels["date"] <= "2022-12-31"]
    val = multi_labels[
        (multi_labels["date"] > "2022-12-31") & (multi_labels["date"] <= "2023-12-31")
    ]
    test = multi_labels[multi_labels["date"] > "2023-12-31"]
    print(f"Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")

    # Volatility by split
    vol_targets["date"] = pd.to_datetime(vol_targets["date"])
    for name, mask in [
        ("Train", vol_targets["date"] <= "2022-12-31"),
        (
            "Val",
            (vol_targets["date"] > "2022-12-31")
            & (vol_targets["date"] <= "2023-12-31"),
        ),
        ("Test", vol_targets["date"] > "2023-12-31"),
    ]:
        subset = vol_targets[mask][vol_col]
        print(
            f"  Vol {name}: mean={subset.mean():.4f}, std={subset.std():.4f}, n={len(subset)}"
        )

    # Direction balance
    for name, df_split in [("Train", train), ("Val", val), ("Test", test)]:
        col = "direction_60d_label"
        if col in df_split.columns:
            up_pct = (df_split[col] == "UP").mean() * 100
            print(f"  Dir {name}: {up_pct:.1f}% UP")

    print("\nPHASE 12 STEP 4 COMPLETE — Targets generated for all 30 stocks")


if __name__ == "__main__":
    main()
