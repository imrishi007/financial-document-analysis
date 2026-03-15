"""Phase 12 — Data collection for 20 new stocks.

Collects:
- OHLCV price data via yfinance
- 10-K filings via SEC EDGAR (reuses existing collector)
- Earnings surprise data (reuses existing collector)
- Macro data update (if needed)

New tickers:
  Semiconductors: QCOM, TXN, AVGO, MRVL, KLAC
  Cloud/Software: CRM, ADBE, NOW, SNOW, DDOG
  Internet/Consumer: NFLX, UBER, PYPL, SNAP
  Hardware/Infra: DELL, AMAT, LRCX
  Diversified Tech: IBM, CSCO, HPE
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np

ORIGINAL_10 = [
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
]
NEW_20 = [
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
ALL_30 = ORIGINAL_10 + NEW_20

# IPO-limited tickers
LATE_IPO = {
    "SNOW": "2020-09-16",
    "DDOG": "2019-09-19",
    "UBER": "2019-05-10",
    "DELL": "2018-12-28",  # re-IPO
}


def collect_price_data():
    """3A: Download OHLCV price data for 20 new tickers."""
    import yfinance as yf

    prices_dir = Path("data/raw/prices")
    prices_dir.mkdir(parents=True, exist_ok=True)

    start_date = "2015-01-01"
    end_date = "2026-02-28"

    results = {}
    for ticker in NEW_20:
        csv_path = prices_dir / f"{ticker}_ohlcv.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            print(f"  [SKIP] {ticker}: already exists ({len(df)} rows)")
            results[ticker] = len(df)
            continue

        print(f"  Downloading {ticker}...", end=" ")
        try:
            raw = yf.download(
                ticker,
                start=start_date,
                end=end_date,
                progress=False,
                auto_adjust=False,
            )
            if len(raw) == 0:
                print(f"WARNING: No data for {ticker}")
                results[ticker] = 0
                continue

            # Handle MultiIndex columns from newer yfinance
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)

            df = pd.DataFrame(
                {
                    "ticker": ticker,
                    "date": raw.index,
                    "open": raw["Open"].values,
                    "high": raw["High"].values,
                    "low": raw["Low"].values,
                    "close": (
                        raw["Close"].values
                        if "Adj Close" not in raw.columns
                        else raw["Adj Close"].values
                    ),
                    "volume": raw["Volume"].values,
                }
            )
            df["date"] = pd.to_datetime(df["date"]).dt.date
            df = df.sort_values("date").reset_index(drop=True)
            df.to_csv(csv_path, index=False)

            if ticker in LATE_IPO:
                ipo = LATE_IPO[ticker]
                print(f"{len(df)} rows (IPO-limited, starts ~{ipo})")
            else:
                print(f"{len(df)} rows ({df['date'].iloc[0]} to {df['date'].iloc[-1]})")
            results[ticker] = len(df)

        except Exception as e:
            print(f"ERROR: {e}")
            results[ticker] = 0

    return results


def collect_10k_filings():
    """3B: Collect 10-K filings for new tickers via SEC EDGAR."""
    from src.data_collection.sec_10k_collector import SEC10KCollector

    collector = SEC10KCollector(
        sec_raw_dir="data/raw/sec_10k",
        log_csv="data/interim/sec_10k_collection_log.csv",
        coverage_csv="data/interim/sec_10k_coverage.csv",
        user_agent="HeterogeneousGraphFusionResearch/1.0 (contact: research@example.com)",
        request_timeout_seconds=30,
    )

    filing_counts = {}
    for ticker in NEW_20:
        print(f"  Collecting 10-K for {ticker}...", end=" ")
        try:
            logs = collector.collect_missing_filings(
                tickers=[ticker],
                start_year=2015,
                end_year=2025,
                include_amended=False,
                overwrite=False,
            )
            count = (
                sum(1 for log in logs if log.get("status") == "downloaded")
                if logs
                else 0
            )
            # Also count existing files
            existing = list(Path("data/raw").glob(f"{ticker}_*_10K.txt"))
            total = len(existing)
            filing_counts[ticker] = total
            print(f"{total} filings ({count} new)")
        except Exception as e:
            print(f"ERROR: {e}")
            filing_counts[ticker] = 0

    return filing_counts


def _extract_section(text, section_start_pattern, section_end_pattern):
    """Extract a section from 10-K text using regex patterns."""
    import re
    match = re.search(section_start_pattern, text, re.IGNORECASE | re.MULTILINE)
    if not match:
        return ""
    start = match.start()
    remaining = text[start + len(match.group()):]
    end_match = re.search(section_end_pattern, remaining, re.IGNORECASE | re.MULTILINE)
    if end_match:
        return remaining[:end_match.start()].strip()
    return remaining[:50000].strip()


def _extract_10k_sections(text):
    """Extract standard 10-K sections from raw filing text."""
    sections = {}
    section_patterns = [
        ("item_1", r"(?:^|\n)\s*item\s+1[\.\s:]+(?:business)", r"(?:^|\n)\s*item\s+1[aA]"),
        ("item_1a", r"(?:^|\n)\s*item\s+1[aA][\.\s:]+(?:risk\s+factors)", r"(?:^|\n)\s*item\s+(?:1[bB]|2)"),
        ("item_7", r"(?:^|\n)\s*item\s+7[\.\s:]+(?:management)", r"(?:^|\n)\s*item\s+7[aA]"),
        ("item_7a", r"(?:^|\n)\s*item\s+7[aA]", r"(?:^|\n)\s*item\s+8"),
        ("item_8", r"(?:^|\n)\s*item\s+8[\.\s:]+(?:financial)", r"(?:^|\n)\s*item\s+9"),
    ]
    for name, start_pat, end_pat in section_patterns:
        try:
            content = _extract_section(text, start_pat, end_pat)
            if content and len(content) > 100:
                sections[name] = content[:200000]
        except Exception:
            pass
    if not sections:
        sections["item_7"] = text[:200000]
    return sections


def process_10k_filings():
    """Process raw 10-K filings into JSON for document model."""
    import json
    from datetime import datetime

    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)

    search_dirs = [Path("data/raw"), Path("data/raw/sec_10k")]
    count = 0

    for ticker in NEW_20:
        for raw_dir in search_dirs:
            if not raw_dir.exists():
                continue
            raw_files = sorted(raw_dir.glob(f"{ticker}_*_10K.txt"))
            for raw_file in raw_files:
                parts = raw_file.stem.split("_")
                if len(parts) >= 2:
                    year = parts[1]
                else:
                    continue

                processed_path = processed_dir / f"{ticker}_{year}_processed.json"
                if processed_path.exists():
                    continue

                try:
                    text = raw_file.read_text(encoding="utf-8", errors="replace")
                    sections = _extract_10k_sections(text)
                    word_count = sum(len(s.split()) for s in sections.values())
                    stats = {k: len(v.split()) for k, v in sections.items()}

                    processed = {
                        "ticker": ticker,
                        "year": year,
                        "filename": raw_file.name,
                        "sections": sections,
                        "stats": stats,
                        "total_words": word_count,
                        "sections_found": len(sections),
                        "processed_at": datetime.now().isoformat(),
                    }
                    processed_path.write_text(json.dumps(processed), encoding="utf-8")
                    count += 1
                except Exception as e:
                    print(f"  WARNING: Failed to process {raw_file.name}: {e}")

    print(f"  Processed {count} new 10-K filings into data/processed/")
    return count


def collect_earnings_data():
    """3C: Collect earnings data for new tickers."""
    from src.data_collection.earnings_collector import collect_earnings_surprise_data

    print("  Collecting earnings data for new tickers...")
    try:
        counts = collect_earnings_surprise_data(
            tickers=NEW_20,
            max_quarters=40,
            output_csv="data/raw/earnings/earnings_surprise.csv",
            log_csv="data/interim/earnings_collection_log.csv",
        )
        print(f"  Earnings data updated: {counts}")
        return counts
    except Exception as e:
        print(f"  WARNING: Earnings collection failed: {e}")
        return {}


def check_macro_data():
    """3D: Check if macro data needs updating."""
    macro_path = Path("data/raw/macro/macro_prices.csv")
    if macro_path.exists():
        df = pd.read_csv(macro_path, parse_dates=["date"], index_col="date")
        print(
            f"  Macro data exists: {df.index[0].date()} to {df.index[-1].date()}, {len(df)} rows"
        )
        if df.index[-1] >= pd.Timestamp("2026-01-01"):
            print("  Macro data is up to date — no collection needed")
            return True
        else:
            print("  Macro data needs updating...")
    else:
        print("  No macro data found — downloading...")

    from src.data.macro_features import download_macro_data

    download_macro_data(start_date="2015-06-01", end_date="2026-03-01")
    return True


def print_summary(price_counts, filing_counts):
    """3E: Print data collection summary."""
    print("\n" + "=" * 60)
    print("DATA COLLECTION SUMMARY")
    print("=" * 60)

    # Price data
    print("\nPrice data:")
    print("  Original 10 tickers:")
    prices_dir = Path("data/raw/prices")
    for ticker in ORIGINAL_10:
        csv = prices_dir / f"{ticker}_ohlcv.csv"
        if csv.exists():
            df = pd.read_csv(csv)
            print(
                f"    {ticker}: {len(df)} rows ({df['date'].iloc[0]} to {df['date'].iloc[-1]})"
            )

    print("  New 20 tickers:")
    limited = []
    for ticker in NEW_20:
        csv = prices_dir / f"{ticker}_ohlcv.csv"
        if csv.exists():
            df = pd.read_csv(csv)
            rows = len(df)
            start = df["date"].iloc[0]
            end = df["date"].iloc[-1]
            note = " (IPO-limited)" if ticker in LATE_IPO else ""
            print(f"    {ticker}: {rows} rows ({start} to {end}){note}")
            if ticker in LATE_IPO:
                limited.append(ticker)
        else:
            print(f"    {ticker}: NO DATA")

    if limited:
        print(f"  Tickers with limited history: {limited}")

    # 10-K filings
    print("\n10-K filings:")
    orig_count = sum(
        1
        for f in Path("data/raw").glob("*_*_10K.txt")
        if f.stem.split("_")[0] in ORIGINAL_10
    )
    new_count = sum(
        1
        for f in Path("data/raw").glob("*_*_10K.txt")
        if f.stem.split("_")[0] in NEW_20
    )
    print(f"  Original 10: {orig_count} filings")
    print(f"  New 20: {new_count} filings")
    print(f"  Total: {orig_count + new_count} filings")

    # Processed documents
    proc_count = len(list(Path("data/processed").glob("*_processed.json")))
    print(f"  Processed documents: {proc_count}")

    # Earnings data
    earnings_path = Path("data/raw/earnings/earnings_surprise.csv")
    if earnings_path.exists():
        earn_df = pd.read_csv(earnings_path)
        orig_earn = earn_df[earn_df["ticker"].isin(ORIGINAL_10)]
        new_earn = earn_df[earn_df["ticker"].isin(NEW_20)]
        print(f"\nEarnings data:")
        print(f"  Original 10: {len(orig_earn)} events")
        print(f"  New 20: {len(new_earn)} events")
        print(f"  Total: {len(earn_df)} events")

    # Macro data
    macro_path = Path("data/raw/macro/macro_prices.csv")
    if macro_path.exists():
        macro_df = pd.read_csv(macro_path, parse_dates=["date"], index_col="date")
        print(
            f"\nMacro data: {macro_df.index[0].date()} to {macro_df.index[-1].date()}, {len(macro_df.columns)} features"
        )


def main():
    print("=" * 60)
    print("PHASE 12 STEP 3: DATA COLLECTION FOR 20 NEW STOCKS")
    print("=" * 60)

    # 3A: Price data
    print("\n--- 3A: Price/OHLCV Data ---")
    price_counts = collect_price_data()

    # 3B: 10-K filings
    print("\n--- 3B: SEC 10-K Filings ---")
    filing_counts = collect_10k_filings()

    # Process 10-K filings
    print("\n--- 3B (cont): Processing 10-K filings ---")
    process_10k_filings()

    # 3C: Earnings
    print("\n--- 3C: Earnings Data ---")
    collect_earnings_data()

    # 3D: Macro data
    print("\n--- 3D: Macro Data ---")
    check_macro_data()

    # 3E: Summary
    print_summary(price_counts, filing_counts)

    print("\nPHASE 12 STEP 3 COMPLETE — Data collected for all 30 stocks")


if __name__ == "__main__":
    main()
