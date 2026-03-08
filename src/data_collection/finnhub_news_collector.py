"""Finnhub-based news collector for historical financial news.

Requires a free API key from https://finnhub.io
Set environment variable FINNHUB_API_KEY before running.
"""

from __future__ import annotations

import argparse
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import yaml

from src.data_collection.io_utils import (
    append_rows_to_csv,
    ensure_directory,
    utc_now_iso,
)


def load_config(config_path: str | Path) -> dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def fetch_finnhub_news(
    ticker: str,
    from_date: str,
    to_date: str,
    api_key: str,
    timeout_seconds: int = 20,
) -> list[dict[str, Any]]:
    """Fetch company news from Finnhub for a date range.

    Finnhub free tier limits: 60 API calls/minute, 1 year max per call.
    """
    url = "https://finnhub.io/api/v1/company-news"
    params = {
        "symbol": ticker,
        "from": from_date,
        "to": to_date,
        "token": api_key,
    }
    response = requests.get(url, params=params, timeout=timeout_seconds)
    response.raise_for_status()
    articles = response.json()

    if not isinstance(articles, list):
        return []

    rows: list[dict[str, Any]] = []
    for article in articles:
        published_ts = article.get("datetime", 0)
        published_at = ""
        if published_ts:
            try:
                published_at = (
                    datetime.utcfromtimestamp(published_ts).isoformat() + "+00:00"
                )
            except (OSError, ValueError):
                pass

        rows.append(
            {
                "ticker": ticker,
                "source_provider": "finnhub",
                "source_name": str(article.get("source", "")).strip(),
                "published_at": published_at,
                "title": str(article.get("headline", "")).strip(),
                "summary": str(article.get("summary", "")).strip(),
                "article_url": str(article.get("url", "")).strip(),
                "raw_content": "",
                "finnhub_id": article.get("id", ""),
                "finnhub_category": str(article.get("category", "")).strip(),
                "finnhub_image": str(article.get("image", "")).strip(),
                "finnhub_related": str(article.get("related", "")).strip(),
            }
        )
    return rows


def collect_finnhub_news(
    tickers: list[str],
    company_name_by_ticker: dict[str, str],
    news_csv: str | Path,
    log_csv: str | Path,
    api_key: str,
    start_date: str,
    end_date: str,
    request_timeout_seconds: int = 20,
    rate_limit_delay: float = 1.1,
    chunk_months: int = 3,
) -> dict[str, int]:
    """Collect Finnhub news across all tickers, chunked by time windows.

    Parameters
    ----------
    chunk_months : int
        Number of months per API call to stay within Finnhub limits.
    rate_limit_delay : float
        Seconds to sleep between API calls (free tier: 60/min = 1s/call).
    """
    news_csv_path = Path(news_csv)
    ensure_directory(news_csv_path.parent)

    # Load existing URLs to avoid duplicates
    existing_urls: set[str] = set()
    if news_csv_path.exists() and news_csv_path.stat().st_size > 0:
        existing = pd.read_csv(news_csv_path)
        if "article_url" in existing.columns:
            existing_urls = set(existing["article_url"].dropna().astype(str).tolist())

    logs: list[dict[str, Any]] = []
    total_new = 0

    # Build date chunks
    dt_start = datetime.strptime(start_date, "%Y-%m-%d")
    dt_end = datetime.strptime(end_date, "%Y-%m-%d")

    for ticker in tickers:
        ticker = ticker.upper()
        company_name = company_name_by_ticker.get(ticker, ticker)
        ticker_rows: list[dict[str, Any]] = []

        # Walk through date chunks
        chunk_start = dt_start
        while chunk_start < dt_end:
            chunk_end = chunk_start + timedelta(days=chunk_months * 30)
            if chunk_end > dt_end:
                chunk_end = dt_end

            from_str = chunk_start.strftime("%Y-%m-%d")
            to_str = chunk_end.strftime("%Y-%m-%d")

            try:
                raw_rows = fetch_finnhub_news(
                    ticker=ticker,
                    from_date=from_str,
                    to_date=to_str,
                    api_key="d6h6p8pr01qnjncniee0d6h6p8pr01qnjncnieeg",
                    timeout_seconds=request_timeout_seconds,
                )

                for row in raw_rows:
                    url = row["article_url"]
                    if not url or url in existing_urls:
                        continue
                    existing_urls.add(url)
                    row["collected_at"] = utc_now_iso()
                    row["company_name"] = company_name
                    row["collection_source"] = "phase2_finnhub_collection"
                    ticker_rows.append(row)

                # Rate limiting
                time.sleep(rate_limit_delay)

            except Exception as exc:
                logs.append(
                    {
                        "collected_at": utc_now_iso(),
                        "ticker": ticker,
                        "status": "finnhub_chunk_error",
                        "date_range": f"{from_str} to {to_str}",
                        "rows_added": 0,
                        "message": str(exc),
                    }
                )
                time.sleep(rate_limit_delay)

            chunk_start = chunk_end + timedelta(days=1)

        # Save ticker batch
        if ticker_rows:
            frame = pd.DataFrame(ticker_rows)
            # Keep only the standard columns (drop finnhub-specific extras for CSV compat)
            standard_cols = [
                "ticker",
                "source_provider",
                "source_name",
                "published_at",
                "title",
                "summary",
                "article_url",
                "raw_content",
                "collected_at",
                "company_name",
                "collection_source",
            ]
            frame = frame[[c for c in standard_cols if c in frame.columns]]

            if news_csv_path.exists() and news_csv_path.stat().st_size > 0:
                frame.to_csv(news_csv_path, mode="a", index=False, header=False)
            else:
                frame.to_csv(news_csv_path, mode="w", index=False, header=True)

        total_new += len(ticker_rows)
        logs.append(
            {
                "collected_at": utc_now_iso(),
                "ticker": ticker,
                "status": "collected",
                "date_range": f"{start_date} to {end_date}",
                "rows_added": len(ticker_rows),
                "message": "",
            }
        )
        print(f"  {ticker}: {len(ticker_rows)} new articles")

    append_rows_to_csv(log_csv, logs)
    return {"total_new_articles": total_new, "tickers_processed": len(tickers)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect financial news from Finnhub.")
    parser.add_argument("--config", type=str, default="configs/data_collection.yaml")
    parser.add_argument(
        "--tickers", nargs="*", default=None, help="Optional ticker subset."
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Finnhub API key (or set FINNHUB_API_KEY env var).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    tickers = [item["ticker"].upper() for item in config["companies"]]
    company_name_by_ticker = {
        item["ticker"].upper(): item["name"] for item in config["companies"]
    }
    if args.tickers:
        tickers = [ticker.upper() for ticker in args.tickers]

    api_key = args.api_key or os.getenv("FINNHUB_API_KEY", "").strip()
    if not api_key:
        print(
            "ERROR: No Finnhub API key. Set FINNHUB_API_KEY env var or use --api-key."
        )
        return

    date_range = config.get("date_range", {})
    paths = config["paths"]

    print(f"Collecting Finnhub news for {len(tickers)} tickers...")
    counts = collect_finnhub_news(
        tickers=tickers,
        company_name_by_ticker=company_name_by_ticker,
        news_csv=paths["news_csv"],
        log_csv=paths["news_log_csv"],
        api_key=api_key,
        start_date=date_range.get("start_date", "2016-01-01"),
        end_date=date_range.get("end_date", "2026-02-28"),
        request_timeout_seconds=int(
            config["sources"]["news"].get("request_timeout_seconds", 20)
        ),
    )

    print(f"\nDone! {counts}")


if __name__ == "__main__":
    main()
