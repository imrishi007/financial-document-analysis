from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_collection.earnings_collector import collect_earnings_surprise_data
from src.data_collection.graph_builder import build_graph_files
from src.data_collection.price_collector import collect_price_data
from src.data_collection.sec_10k_collector import SEC10KCollector
from src.data_collection.news_collector import collect_news_data


def load_config(config_path: str | Path) -> dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase 1 raw data collection.")
    parser.add_argument("--config", type=str, default="configs/data_collection.yaml")
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        choices=["sec", "price", "news", "earnings", "graph"],
        help="Run only a single data source.",
    )
    parser.add_argument("--tickers", nargs="*", default=None, help="Optional ticker subset.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    tickers = [item["ticker"].upper() for item in config["companies"]]
    company_name_by_ticker = {item["ticker"].upper(): item["name"] for item in config["companies"]}
    if args.tickers:
        tickers = [ticker.upper() for ticker in args.tickers]

    date_cfg = config["date_range"]
    source_cfg = config["sources"]
    paths = config["paths"]

    summary_rows: list[dict[str, Any]] = []

    if args.only in {None, "sec"} and source_cfg["sec"].get("enabled", True):
        sec_collector = SEC10KCollector(
            sec_raw_dir=paths["sec_raw_dir"],
            log_csv=paths["sec_log_csv"],
            coverage_csv=paths["sec_coverage_csv"],
            user_agent=source_cfg["sec"]["user_agent"],
            request_timeout_seconds=int(source_cfg["sec"].get("request_timeout_seconds", 30)),
        )
        coverage = sec_collector.build_coverage_report(
            tickers=tickers,
            start_year=int(date_cfg["sec_start_year"]),
            end_year=int(date_cfg["sec_end_year"]),
            include_amended=bool(source_cfg["sec"].get("include_amended", False)),
        )
        sec_logs = sec_collector.collect_missing_filings(
            tickers=tickers,
            start_year=int(date_cfg["sec_start_year"]),
            end_year=int(date_cfg["sec_end_year"]),
            include_amended=bool(source_cfg["sec"].get("include_amended", False)),
            overwrite=False,
        )
        sec_status = pd.DataFrame(sec_logs)["status"].value_counts().to_dict() if sec_logs else {}
        coverage_status = coverage["status"].value_counts().to_dict() if not coverage.empty else {}
        summary_rows.append(
            {
                "source": "sec",
                "status": "completed",
                "details": f"coverage={coverage_status}; collection={sec_status}",
            }
        )

    if args.only in {None, "price"} and source_cfg["price"].get("enabled", True):
        counts = collect_price_data(
            tickers=tickers,
            start_date=date_cfg["start_date"],
            end_date=date_cfg["end_date"],
            interval=source_cfg["price"].get("interval", "1d"),
            output_dir=paths["price_dir"],
            log_csv=paths["price_log_csv"],
        )
        summary_rows.append({"source": "price", "status": "completed", "details": str(counts)})

    if args.only in {None, "news"} and source_cfg["news"].get("enabled", True):
        counts = collect_news_data(
            tickers=tickers,
            company_name_by_ticker=company_name_by_ticker,
            news_csv=paths["news_csv"],
            log_csv=paths["news_log_csv"],
            provider=str(source_cfg["news"].get("provider", "rss_then_newsapi")),
            per_ticker_limit=int(source_cfg["news"].get("per_ticker_limit", 200)),
            fetch_full_text=bool(source_cfg["news"].get("fetch_full_text", False)),
            newsapi_key_env=str(source_cfg["news"].get("newsapi_key_env", "NEWSAPI_KEY")),
            request_timeout_seconds=int(source_cfg["news"].get("request_timeout_seconds", 20)),
        )
        summary_rows.append({"source": "news", "status": "completed", "details": str(counts)})

    if args.only in {None, "earnings"} and source_cfg["earnings"].get("enabled", True):
        counts = collect_earnings_surprise_data(
            tickers=tickers,
            max_quarters=int(source_cfg["earnings"].get("max_quarters", 40)),
            output_csv=paths["earnings_csv"],
            log_csv=paths["earnings_log_csv"],
        )
        summary_rows.append({"source": "earnings", "status": "completed", "details": str(counts)})

    if args.only in {None, "graph"} and source_cfg["graph"].get("enabled", True):
        build_graph_files(
            companies=config["companies"],
            nodes_csv=paths["graph_nodes_csv"],
            edges_csv=paths["graph_edges_csv"],
        )
        summary_rows.append({"source": "graph", "status": "completed", "details": "nodes and edges exported"})

    print("Phase 1 data collection summary")
    print(pd.DataFrame(summary_rows).to_string(index=False))


if __name__ == "__main__":
    main()
