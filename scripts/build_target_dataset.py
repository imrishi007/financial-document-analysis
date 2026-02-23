from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.price_loader import PriceRequest, download_price_history
from src.data.target_builder import build_binary_direction_labels


def main() -> None:
    config_path = Path("configs/experiment.yaml")
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    target_cfg = config["target"]
    data_cfg = config["data"]

    all_prices = []
    for ticker in data_cfg["tickers"]:
        request = PriceRequest(
            ticker=ticker,
            start_date=data_cfg["start_date"],
            end_date=data_cfg["end_date"],
        )
        all_prices.append(download_price_history(request))

    prices = pd.concat(all_prices, axis=0, ignore_index=True)
    labels = build_binary_direction_labels(
        prices=prices[["ticker", "date", "close"]],
        horizon_days=int(target_cfg["horizon_days"]),
        threshold=float(target_cfg["threshold"]),
    )

    targets_dir = Path("data/targets")
    targets_dir.mkdir(parents=True, exist_ok=True)
    output_file = targets_dir / "direction_labels.csv"
    labels.to_csv(output_file, index=False)

    print(f"Saved target dataset: {output_file}")
    print(f"Rows: {len(labels)}")
    print(labels["target_label"].value_counts().to_string())


if __name__ == "__main__":
    main()
