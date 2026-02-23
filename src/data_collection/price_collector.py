from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
import yfinance as yf
import yaml

from src.data_collection.io_utils import append_rows_to_csv, ensure_directory, utc_now_iso


def load_config(config_path: str | Path) -> dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def normalize_ohlcv_frame(frame: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(
            columns=["ticker", "date", "open", "high", "low", "close", "adj_close", "volume"]
        )

    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = [str(price).lower().replace(" ", "_") for price, _ in frame.columns]
    else:
        frame.columns = [str(column).lower().replace(" ", "_") for column in frame.columns]

    frame = frame.reset_index()
    frame.columns = [str(column).lower().replace(" ", "_") for column in frame.columns]

    if "adj_close" not in frame.columns:
        frame["adj_close"] = frame["close"]

    frame["ticker"] = ticker
    frame["date"] = pd.to_datetime(frame["date"]).dt.date

    return frame[["ticker", "date", "open", "high", "low", "close", "adj_close", "volume"]]


def collect_price_data(
    tickers: list[str],
    start_date: str,
    end_date: str,
    interval: str,
    output_dir: str | Path,
    log_csv: str | Path,
) -> dict[str, int]:
    out_dir = ensure_directory(output_dir)
    logs: list[dict[str, Any]] = []
    frames: list[pd.DataFrame] = []

    for ticker in tickers:
        ticker = ticker.upper()
        try:
            raw_frame = yf.download(
                tickers=ticker,
                start=start_date,
                end=end_date,
                interval=interval,
                auto_adjust=False,
                progress=False,
            )
            normalized = normalize_ohlcv_frame(raw_frame, ticker)
            if normalized.empty:
                raise ValueError("No rows returned.")

            output_path = out_dir / f"{ticker}_ohlcv.csv"
            normalized.to_csv(output_path, index=False)
            frames.append(normalized)

            logs.append(
                {
                    "collected_at": utc_now_iso(),
                    "ticker": ticker,
                    "status": "downloaded",
                    "rows": len(normalized),
                    "local_path": str(output_path),
                    "message": "",
                }
            )
        except Exception as exc:
            logs.append(
                {
                    "collected_at": utc_now_iso(),
                    "ticker": ticker,
                    "status": "error",
                    "rows": 0,
                    "local_path": "",
                    "message": str(exc),
                }
            )

    if frames:
        combined = pd.concat(frames, axis=0, ignore_index=True)
        combined_output = out_dir / "tech10_ohlcv.csv"
        combined.to_csv(combined_output, index=False)

    append_rows_to_csv(log_csv, logs)
    counts = pd.DataFrame(logs)["status"].value_counts().to_dict() if logs else {}
    return counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect OHLCV data for configured tickers.")
    parser.add_argument("--config", type=str, default="configs/data_collection.yaml")
    parser.add_argument("--tickers", nargs="*", default=None, help="Optional ticker subset.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    tickers = [item["ticker"].upper() for item in config["companies"]]
    if args.tickers:
        tickers = [ticker.upper() for ticker in args.tickers]

    date_cfg = config["date_range"]
    price_cfg = config["sources"]["price"]
    paths = config["paths"]

    counts = collect_price_data(
        tickers=tickers,
        start_date=date_cfg["start_date"],
        end_date=date_cfg["end_date"],
        interval=price_cfg.get("interval", "1d"),
        output_dir=paths["price_dir"],
        log_csv=paths["price_log_csv"],
    )

    print(f"Price files saved in {paths['price_dir']}")
    print(counts)


if __name__ == "__main__":
    main()
