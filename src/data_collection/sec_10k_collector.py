from __future__ import annotations

import argparse
import re
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import yaml
from bs4 import BeautifulSoup

from src.data_collection.io_utils import append_rows_to_csv, ensure_directory, utc_now_iso


def load_config(config_path: str | Path) -> dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def normalize_text(raw_text: str) -> str:
    text = re.sub(r"\r\n?", "\n", raw_text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def html_to_text(content: bytes, parser: str = "lxml") -> str:
    soup = BeautifulSoup(content, parser)
    for tag in soup(["script", "style", "noscript", "header", "footer", "nav"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    return normalize_text(text)


class SEC10KCollector:
    def __init__(
        self,
        sec_raw_dir: str | Path,
        log_csv: str | Path,
        coverage_csv: str | Path,
        user_agent: str,
        request_timeout_seconds: int = 30,
    ) -> None:
        self.sec_raw_dir = ensure_directory(sec_raw_dir)
        self.log_csv = Path(log_csv)
        self.coverage_csv = Path(coverage_csv)
        self.request_timeout_seconds = request_timeout_seconds

        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": user_agent,
                "Accept-Encoding": "gzip, deflate",
            }
        )
        self._cik_lookup: dict[str, dict[str, str]] | None = None

    def _load_cik_lookup(self) -> dict[str, dict[str, str]]:
        if self._cik_lookup is not None:
            return self._cik_lookup

        url = "https://www.sec.gov/files/company_tickers.json"
        response = self.session.get(url, timeout=self.request_timeout_seconds)
        response.raise_for_status()
        payload = response.json()

        lookup: dict[str, dict[str, str]] = {}
        for company in payload.values():
            ticker = str(company["ticker"]).upper().strip()
            lookup[ticker] = {
                "cik_10": str(company["cik_str"]).zfill(10),
                "company_name": str(company["title"]).strip(),
            }
        self._cik_lookup = lookup
        return lookup

    def get_company_identity(self, ticker: str) -> dict[str, str] | None:
        lookup = self._load_cik_lookup()
        return lookup.get(ticker.upper())

    def _get_submissions_payload(self, cik_10: str) -> dict[str, Any]:
        url = f"https://data.sec.gov/submissions/CIK{cik_10}.json"
        response = self.session.get(url, timeout=self.request_timeout_seconds)
        response.raise_for_status()
        return response.json()

    def _extract_recent_10k_filings(
        self,
        submissions_payload: dict[str, Any],
        include_amended: bool,
    ) -> list[dict[str, str]]:
        recent = submissions_payload.get("filings", {}).get("recent", {})
        forms = recent.get("form", [])
        filing_dates = recent.get("filingDate", [])
        report_dates = recent.get("reportDate", [])
        accession_numbers = recent.get("accessionNumber", [])
        primary_documents = recent.get("primaryDocument", [])

        allowed_forms = {"10-K", "10-K/A"} if include_amended else {"10-K"}
        rows: list[dict[str, str]] = []

        for idx, form in enumerate(forms):
            if form not in allowed_forms:
                continue

            filing_date = filing_dates[idx] if idx < len(filing_dates) else ""
            report_date = report_dates[idx] if idx < len(report_dates) else ""
            accession_number = accession_numbers[idx] if idx < len(accession_numbers) else ""
            primary_document = primary_documents[idx] if idx < len(primary_documents) else ""
            year = filing_date[:4] if filing_date else ""

            if not year.isdigit():
                continue

            rows.append(
                {
                    "form": form,
                    "filing_date": filing_date,
                    "report_date": report_date,
                    "year": year,
                    "accession_number": accession_number,
                    "primary_document": primary_document,
                }
            )

        rows.sort(key=lambda item: item["filing_date"], reverse=True)
        return rows

    def _build_filing_url(self, cik_10: str, accession_number: str, primary_document: str) -> str:
        cik_no_leading_zeros = str(int(cik_10))
        accession_compact = accession_number.replace("-", "")
        return (
            f"https://www.sec.gov/Archives/edgar/data/"
            f"{cik_no_leading_zeros}/{accession_compact}/{primary_document}"
        )

    def _download_filing_text(self, filing_url: str) -> str:
        response = self.session.get(filing_url, timeout=self.request_timeout_seconds)
        response.raise_for_status()

        if filing_url.lower().endswith(".txt"):
            return normalize_text(response.text)

        content_type = response.headers.get("Content-Type", "").lower()
        if "html" in content_type or "xml" in content_type:
            parser = "xml" if "xml" in content_type else "lxml"
            return html_to_text(response.content, parser=parser)

        return normalize_text(response.text)

    def _existing_years(self, ticker: str) -> set[int]:
        years: set[int] = set()
        scan_dirs = [self.sec_raw_dir]
        legacy_dir = self.sec_raw_dir.parent
        if legacy_dir != self.sec_raw_dir:
            scan_dirs.append(legacy_dir)

        for directory in scan_dirs:
            for path in directory.glob(f"{ticker.upper()}_*_10K.txt"):
                match = re.match(rf"{ticker.upper()}_(\d{{4}})_10K\.txt$", path.name)
                if match:
                    years.add(int(match.group(1)))
        return years

    def build_coverage_report(
        self,
        tickers: list[str],
        start_year: int,
        end_year: int,
        include_amended: bool,
    ) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []

        for ticker in tickers:
            identity = self.get_company_identity(ticker)
            if identity is None:
                for year in range(start_year, end_year + 1):
                    rows.append(
                        {
                            "ticker": ticker,
                            "year": year,
                            "status": "unknown_ticker",
                            "company_name": "",
                            "cik_10": "",
                        }
                    )
                continue

            cik_10 = identity["cik_10"]
            company_name = identity["company_name"]
            local_years = self._existing_years(ticker)

            remote_years: set[int] = set()
            try:
                payload = self._get_submissions_payload(cik_10)
                filings = self._extract_recent_10k_filings(payload, include_amended=include_amended)
                remote_years = {int(item["year"]) for item in filings}
            except Exception:
                remote_years = set()

            for year in range(start_year, end_year + 1):
                if year in local_years:
                    status = "present_local"
                elif year in remote_years:
                    status = "missing_downloadable"
                else:
                    status = "missing_unavailable"

                rows.append(
                    {
                        "ticker": ticker,
                        "year": year,
                        "status": status,
                        "company_name": company_name,
                        "cik_10": cik_10,
                    }
                )

            time.sleep(0.2)

        frame = pd.DataFrame(rows).sort_values(["ticker", "year"]).reset_index(drop=True)
        ensure_directory(self.coverage_csv.parent)
        frame.to_csv(self.coverage_csv, index=False)
        return frame

    def collect_missing_filings(
        self,
        tickers: list[str],
        start_year: int,
        end_year: int,
        include_amended: bool,
        overwrite: bool = False,
    ) -> list[dict[str, Any]]:
        target_years = set(range(start_year, end_year + 1))
        logs: list[dict[str, Any]] = []

        for ticker in tickers:
            ticker = ticker.upper()
            identity = self.get_company_identity(ticker)

            if identity is None:
                logs.append(
                    {
                        "collected_at": utc_now_iso(),
                        "ticker": ticker,
                        "year": "",
                        "status": "unknown_ticker",
                        "message": "Ticker not found in SEC company list.",
                    }
                )
                continue

            cik_10 = identity["cik_10"]
            company_name = identity["company_name"]
            local_years = self._existing_years(ticker)
            missing_years = sorted(target_years - local_years)

            if not missing_years:
                logs.append(
                    {
                        "collected_at": utc_now_iso(),
                        "ticker": ticker,
                        "year": "",
                        "status": "no_missing_years",
                        "message": "All target years already available locally.",
                    }
                )
                continue

            try:
                payload = self._get_submissions_payload(cik_10)
                filings = self._extract_recent_10k_filings(payload, include_amended=include_amended)
            except Exception as exc:
                logs.append(
                    {
                        "collected_at": utc_now_iso(),
                        "ticker": ticker,
                        "year": "",
                        "status": "submissions_error",
                        "message": str(exc),
                    }
                )
                continue

            first_filing_per_year: dict[int, dict[str, str]] = {}
            for filing in filings:
                year = int(filing["year"])
                if year not in first_filing_per_year:
                    first_filing_per_year[year] = filing

            for year in missing_years:
                if year not in first_filing_per_year:
                    logs.append(
                        {
                            "collected_at": utc_now_iso(),
                            "ticker": ticker,
                            "year": year,
                            "status": "missing_unavailable",
                            "message": "No eligible 10-K filing found in recent SEC submissions.",
                        }
                    )
                    continue

                filing = first_filing_per_year[year]
                accession_number = filing["accession_number"]
                primary_document = filing["primary_document"]
                filing_date = filing["filing_date"]
                report_date = filing["report_date"]
                form = filing["form"]

                filing_url = self._build_filing_url(cik_10, accession_number, primary_document)
                output_path = self.sec_raw_dir / f"{ticker}_{year}_10K.txt"

                if output_path.exists() and not overwrite:
                    logs.append(
                        {
                            "collected_at": utc_now_iso(),
                            "ticker": ticker,
                            "year": year,
                            "status": "skipped_exists",
                            "message": f"{output_path} already exists.",
                        }
                    )
                    continue

                try:
                    text = self._download_filing_text(filing_url)
                    if len(text) < 10_000:
                        raise ValueError(f"Downloaded text too short: {len(text)} characters.")

                    with open(output_path, "w", encoding="utf-8") as handle:
                        handle.write(text)

                    logs.append(
                        {
                            "collected_at": utc_now_iso(),
                            "ticker": ticker,
                            "year": year,
                            "status": "downloaded",
                            "message": f"Saved {output_path.name}",
                            "company_name": company_name,
                            "cik_10": cik_10,
                            "form": form,
                            "filing_date": filing_date,
                            "report_date": report_date,
                            "accession_number": accession_number,
                            "primary_document": primary_document,
                            "source_url": filing_url,
                            "local_path": str(output_path),
                        }
                    )
                except Exception as exc:
                    logs.append(
                        {
                            "collected_at": utc_now_iso(),
                            "ticker": ticker,
                            "year": year,
                            "status": "download_error",
                            "message": str(exc),
                            "source_url": filing_url,
                        }
                    )

                time.sleep(0.2)

        append_rows_to_csv(self.log_csv, logs)
        return logs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SEC 10-K coverage audit and missing-file collection.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/data_collection.yaml",
        help="Path to collection config yaml.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["coverage", "collect-missing"],
        default="coverage",
        help="coverage: build a missing-year report; collect-missing: download only missing target years.",
    )
    parser.add_argument(
        "--tickers",
        nargs="*",
        default=None,
        help="Optional ticker subset. If omitted, tickers from config are used.",
    )
    parser.add_argument("--start-year", type=int, default=None, help="Override start year.")
    parser.add_argument("--end-year", type=int, default=None, help="Override end year.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing local files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    tickers = [item["ticker"].upper() for item in config["companies"]]
    if args.tickers:
        tickers = [ticker.upper() for ticker in args.tickers]

    sec_cfg = config["sources"]["sec"]
    start_year = int(args.start_year or config["date_range"]["sec_start_year"])
    end_year = int(args.end_year or config["date_range"]["sec_end_year"])

    collector = SEC10KCollector(
        sec_raw_dir=config["paths"]["sec_raw_dir"],
        log_csv=config["paths"]["sec_log_csv"],
        coverage_csv=config["paths"]["sec_coverage_csv"],
        user_agent=sec_cfg["user_agent"],
        request_timeout_seconds=int(sec_cfg.get("request_timeout_seconds", 30)),
    )

    if args.mode == "coverage":
        coverage = collector.build_coverage_report(
            tickers=tickers,
            start_year=start_year,
            end_year=end_year,
            include_amended=bool(sec_cfg.get("include_amended", False)),
        )
        summary = coverage["status"].value_counts().to_dict()
        print(f"Coverage report saved to {collector.coverage_csv}")
        print(summary)
        return

    logs = collector.collect_missing_filings(
        tickers=tickers,
        start_year=start_year,
        end_year=end_year,
        include_amended=bool(sec_cfg.get("include_amended", False)),
        overwrite=args.overwrite,
    )
    status_counts = pd.DataFrame(logs)["status"].value_counts().to_dict() if logs else {}
    print(f"Collection log saved to {collector.log_csv}")
    print(status_counts)


if __name__ == "__main__":
    main()
