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


def normalize_earnings_frame(frame: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame(
            columns=[
                "ticker",
                "announcement_date",
                "eps_estimate",
                "reported_eps",
                "surprise_pct",
                "surprise_label",
            ]
        )

    normalized = frame.copy()
    normalized = normalized.reset_index().rename(columns={"Earnings Date": "announcement_date"})
    normalized.columns = [str(column).strip().lower().replace(" ", "_") for column in normalized.columns]

    if "eps_estimate" not in normalized.columns:
        normalized["eps_estimate"] = pd.NA
    if "reported_eps" not in normalized.columns:
        normalized["reported_eps"] = pd.NA
    if "surprise(%)" in normalized.columns:
        normalized = normalized.rename(columns={"surprise(%)": "surprise_pct"})
    if "surprise_pct" not in normalized.columns:
        normalized["surprise_pct"] = pd.NA

    normalized["eps_estimate"] = pd.to_numeric(normalized["eps_estimate"], errors="coerce")
    normalized["reported_eps"] = pd.to_numeric(normalized["reported_eps"], errors="coerce")
    normalized["surprise_pct"] = pd.to_numeric(normalized["surprise_pct"], errors="coerce")
    normalized["announcement_date"] = pd.to_datetime(normalized["announcement_date"]).dt.date
    normalized["ticker"] = ticker

    def label_row(row: pd.Series) -> str:
        estimate = row["eps_estimate"]
        reported = row["reported_eps"]
        if pd.isna(estimate) or pd.isna(reported):
            return "UNKNOWN"
        return "BEAT" if reported > estimate else "MISS"

    normalized["surprise_label"] = normalized.apply(label_row, axis=1)
    return normalized[
        [
            "ticker",
            "announcement_date",
            "eps_estimate",
            "reported_eps",
            "surprise_pct",
            "surprise_label",
        ]
    ]


def collect_earnings_surprise_data(
    tickers: list[str],
    max_quarters: int,
    output_csv: str | Path,
    log_csv: str | Path,
) -> dict[str, int]:
    output_path = Path(output_csv)
    ensure_directory(output_path.parent)
    logs: list[dict[str, Any]] = []
    all_frames: list[pd.DataFrame] = []

    for ticker in tickers:
        ticker = ticker.upper()
        try:
            ticker_obj = yf.Ticker(ticker)
            frame = ticker_obj.get_earnings_dates(limit=max_quarters)
            normalized = normalize_earnings_frame(frame, ticker)
            if normalized.empty:
                raise ValueError("No earnings rows returned.")

            all_frames.append(normalized)
            logs.append(
                {
                    "collected_at": utc_now_iso(),
                    "ticker": ticker,
                    "status": "downloaded",
                    "rows": len(normalized),
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
                    "message": str(exc),
                }
            )

    if all_frames:
        combined = pd.concat(all_frames, axis=0, ignore_index=True)
        combined = combined.sort_values(["ticker", "announcement_date"]).reset_index(drop=True)
        combined.to_csv(output_path, index=False)

    append_rows_to_csv(log_csv, logs)
    counts = pd.DataFrame(logs)["status"].value_counts().to_dict() if logs else {}
    return counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect earnings surprise data for configured tickers.")
    parser.add_argument("--config", type=str, default="configs/data_collection.yaml")
    parser.add_argument("--tickers", nargs="*", default=None, help="Optional ticker subset.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    tickers = [item["ticker"].upper() for item in config["companies"]]
    if args.tickers:
        tickers = [ticker.upper() for ticker in args.tickers]

    earnings_cfg = config["sources"]["earnings"]
    paths = config["paths"]

    counts = collect_earnings_surprise_data(
        tickers=tickers,
        max_quarters=int(earnings_cfg.get("max_quarters", 40)),
        output_csv=paths["earnings_csv"],
        log_csv=paths["earnings_log_csv"],
    )

    print(f"Earnings surprise file saved to {paths['earnings_csv']}")
    print(counts)


if __name__ == "__main__":
    main()
