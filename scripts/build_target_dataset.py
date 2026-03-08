from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.price_loader import PriceRequest, download_price_history
from src.data.target_builder import (
    build_binary_direction_labels,
    build_multi_horizon_direction_labels,
    build_realized_volatility_targets,
    build_fundamental_surprise_targets,
)


def main() -> None:
    config_path = Path("configs/experiment.yaml")
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    target_cfg = config["target"]
    data_cfg = config["data"]
    labels_cfg = config.get("labels", {})

    # ---- Download prices ----
    all_prices = []
    for ticker in data_cfg["tickers"]:
        request = PriceRequest(
            ticker=ticker,
            start_date=data_cfg["start_date"],
            end_date=data_cfg["end_date"],
        )
        all_prices.append(download_price_history(request))

    prices = pd.concat(all_prices, axis=0, ignore_index=True)
    price_subset = prices[["ticker", "date", "close"]].copy()

    targets_dir = Path("data/targets")
    targets_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1. Legacy single-horizon direction labels (backward compat) ----
    labels = build_binary_direction_labels(
        prices=price_subset,
        horizon_days=int(target_cfg["horizon_days"]),
        threshold=float(target_cfg["threshold"]),
    )
    labels.to_csv(targets_dir / "direction_labels.csv", index=False)
    print(f"[direction_labels.csv] {len(labels)} rows")
    print(f"  {labels['target_label'].value_counts().to_dict()}")

    # ---- 2. Multi-horizon direction labels ----
    horizons = [int(h) for h in target_cfg.get("horizons", [1, 3, 5])]
    multi_labels = build_multi_horizon_direction_labels(
        prices=price_subset,
        horizons=horizons,
        threshold=float(target_cfg["threshold"]),
    )
    multi_labels.to_csv(targets_dir / "direction_labels_multi_horizon.csv", index=False)
    print(
        f"\n[direction_labels_multi_horizon.csv] {len(multi_labels)} rows, horizons={horizons}"
    )
    for h in horizons:
        col = f"direction_{h}d_label"
        if col in multi_labels.columns:
            print(f"  {h}d: {multi_labels[col].value_counts().to_dict()}")

    # ---- 3. Realized volatility targets ----
    vol_cfg = labels_cfg.get("realized_volatility", {})
    lookback = int(vol_cfg.get("lookback_days", 20))
    vol_targets = build_realized_volatility_targets(
        prices=price_subset,
        lookback_days=lookback,
    )
    vol_targets.to_csv(targets_dir / "volatility_targets.csv", index=False)
    print(f"\n[volatility_targets.csv] {len(vol_targets)} rows, lookback={lookback}d")
    vol_col = f"realized_vol_{lookback}d"
    print(f"  Mean vol: {vol_targets[vol_col].mean():.6f}")
    print(f"  Std vol:  {vol_targets[vol_col].std():.6f}")

    # ---- 4. Fundamental surprise targets ----
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
        print(f"\n[fundamental_surprise_targets.csv] {len(surprise_targets)} rows")
        print(f"  {surprise_targets['surprise_label'].value_counts().to_dict()}")
    else:
        print(f"\n[SKIP] Earnings CSV not found at {earnings_csv}")

    print("\nAll targets saved to data/targets/")


if __name__ == "__main__":
    main()
