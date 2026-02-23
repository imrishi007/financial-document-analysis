from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import feedparser
import pandas as pd
import requests
import yaml
from bs4 import BeautifulSoup

from src.data_collection.io_utils import append_rows_to_csv, ensure_directory, utc_now_iso


def load_config(config_path: str | Path) -> dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def normalize_text(text: str) -> str:
    return " ".join((text or "").split())


def extract_full_text_from_url(url: str, timeout_seconds: int) -> str:
    response = requests.get(url, timeout=timeout_seconds, headers={"User-Agent": "Mozilla/5.0"})
    response.raise_for_status()
    soup = BeautifulSoup(response.content, "lxml")
    for tag in soup(["script", "style", "noscript", "header", "footer", "nav"]):
        tag.decompose()
    return normalize_text(soup.get_text(separator=" "))


def fetch_yahoo_rss_news(ticker: str, max_items: int) -> list[dict[str, Any]]:
    feed_urls = [
        f"https://finance.yahoo.com/rss/headline?s={ticker}",
        f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US",
    ]
    entries: list[dict[str, Any]] = []
    seen_urls: set[str] = set()

    for feed_url in feed_urls:
        feed = feedparser.parse(feed_url)
        for entry in feed.entries:
            url = str(entry.get("link", "")).strip()
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)

            entries.append(
                {
                    "ticker": ticker,
                    "source_provider": "yahoo_rss",
                    "source_name": "Yahoo Finance RSS",
                    "published_at": entry.get("published", ""),
                    "title": normalize_text(entry.get("title", "")),
                    "summary": normalize_text(entry.get("summary", "")),
                    "article_url": url,
                    "raw_content": "",
                }
            )
            if len(entries) >= max_items:
                return entries

    return entries


def fetch_newsapi_news(
    ticker: str,
    company_name: str,
    api_key: str,
    max_items: int,
    timeout_seconds: int,
) -> list[dict[str, Any]]:
    query = f'("{ticker}" OR "{company_name}") AND (earnings OR revenue OR guidance OR risk OR outlook)'
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": query,
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": min(max_items, 100),
        "apiKey": api_key,
    }
    response = requests.get(url, params=params, timeout=timeout_seconds)
    response.raise_for_status()
    payload = response.json()
    articles = payload.get("articles", [])

    rows: list[dict[str, Any]] = []
    for article in articles[:max_items]:
        rows.append(
            {
                "ticker": ticker,
                "source_provider": "newsapi",
                "source_name": normalize_text((article.get("source") or {}).get("name", "")),
                "published_at": article.get("publishedAt", ""),
                "title": normalize_text(article.get("title", "")),
                "summary": normalize_text(article.get("description", "")),
                "article_url": normalize_text(article.get("url", "")),
                "raw_content": normalize_text(article.get("content", "")),
            }
        )
    return rows


def collect_news_data(
    tickers: list[str],
    company_name_by_ticker: dict[str, str],
    news_csv: str | Path,
    log_csv: str | Path,
    provider: str,
    per_ticker_limit: int,
    fetch_full_text: bool,
    newsapi_key_env: str,
    request_timeout_seconds: int,
) -> dict[str, int]:
    news_csv_path = Path(news_csv)
    ensure_directory(news_csv_path.parent)

    existing_urls: set[str] = set()
    if news_csv_path.exists() and news_csv_path.stat().st_size > 0:
        existing = pd.read_csv(news_csv_path)
        if "article_url" in existing.columns:
            existing_urls = set(existing["article_url"].dropna().astype(str).tolist())

    newsapi_key = os.getenv(newsapi_key_env, "").strip()
    all_rows: list[dict[str, Any]] = []
    logs: list[dict[str, Any]] = []

    for ticker in tickers:
        ticker = ticker.upper()
        company_name = company_name_by_ticker.get(ticker, ticker)
        collected_rows: list[dict[str, Any]] = []

        if provider in {"rss", "rss_then_newsapi", "all"}:
            rss_rows = fetch_yahoo_rss_news(ticker=ticker, max_items=per_ticker_limit)
            collected_rows.extend(rss_rows)

        if provider in {"newsapi", "rss_then_newsapi", "all"} and len(collected_rows) < per_ticker_limit:
            if newsapi_key:
                remaining = per_ticker_limit - len(collected_rows)
                try:
                    api_rows = fetch_newsapi_news(
                        ticker=ticker,
                        company_name=company_name,
                        api_key=newsapi_key,
                        max_items=remaining,
                        timeout_seconds=request_timeout_seconds,
                    )
                    collected_rows.extend(api_rows)
                except Exception as exc:
                    logs.append(
                        {
                            "collected_at": utc_now_iso(),
                            "ticker": ticker,
                            "status": "newsapi_error",
                            "rows_added": 0,
                            "message": str(exc),
                        }
                    )

        deduped_rows: list[dict[str, Any]] = []
        for row in collected_rows:
            article_url = row["article_url"]
            if not article_url or article_url in existing_urls:
                continue

            existing_urls.add(article_url)
            row["collected_at"] = utc_now_iso()
            row["company_name"] = company_name
            row["collection_source"] = "phase1_news_collection"

            if fetch_full_text and article_url:
                try:
                    row["raw_content"] = extract_full_text_from_url(
                        article_url,
                        timeout_seconds=request_timeout_seconds,
                    )
                except Exception:
                    pass

            deduped_rows.append(row)

        all_rows.extend(deduped_rows)
        logs.append(
            {
                "collected_at": utc_now_iso(),
                "ticker": ticker,
                "status": "collected",
                "rows_added": len(deduped_rows),
                "message": "",
            }
        )

    if all_rows:
        frame = pd.DataFrame(all_rows)
        if news_csv_path.exists() and news_csv_path.stat().st_size > 0:
            frame.to_csv(news_csv_path, mode="a", index=False, header=False)
        else:
            frame.to_csv(news_csv_path, mode="w", index=False, header=True)

    append_rows_to_csv(log_csv, logs)
    counts = pd.DataFrame(logs)["status"].value_counts().to_dict() if logs else {}
    return counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect financial news for configured tickers.")
    parser.add_argument("--config", type=str, default="configs/data_collection.yaml")
    parser.add_argument("--tickers", nargs="*", default=None, help="Optional ticker subset.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    tickers = [item["ticker"].upper() for item in config["companies"]]
    company_name_by_ticker = {item["ticker"].upper(): item["name"] for item in config["companies"]}
    if args.tickers:
        tickers = [ticker.upper() for ticker in args.tickers]

    news_cfg = config["sources"]["news"]
    paths = config["paths"]

    counts = collect_news_data(
        tickers=tickers,
        company_name_by_ticker=company_name_by_ticker,
        news_csv=paths["news_csv"],
        log_csv=paths["news_log_csv"],
        provider=str(news_cfg.get("provider", "rss_then_newsapi")),
        per_ticker_limit=int(news_cfg.get("per_ticker_limit", 200)),
        fetch_full_text=bool(news_cfg.get("fetch_full_text", False)),
        newsapi_key_env=str(news_cfg.get("newsapi_key_env", "NEWSAPI_KEY")),
        request_timeout_seconds=int(news_cfg.get("request_timeout_seconds", 20)),
    )

    print(f"News file saved to {paths['news_csv']}")
    print(counts)


if __name__ == "__main__":
    main()
