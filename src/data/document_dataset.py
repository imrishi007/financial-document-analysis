from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def load_processed_filings(processed_dir: str | Path) -> pd.DataFrame:
    processed_path = Path(processed_dir)
    rows = []

    for file_path in processed_path.glob("*_processed.json"):
        with open(file_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)

        sections = payload.get("sections", {})
        document_text = "\n\n".join(
            filter(
                None,
                [
                    sections.get("item_1", ""),
                    sections.get("item_1a", ""),
                    sections.get("item_7", ""),
                    sections.get("item_7a", ""),
                    sections.get("item_8", ""),
                ],
            )
        )

        rows.append(
            {
                "ticker": payload.get("ticker"),
                "year": payload.get("year"),
                "filename": payload.get("filename"),
                "document_text": document_text,
            }
        )

    if not rows:
        return pd.DataFrame(columns=["ticker", "year", "filename", "document_text"])

    return pd.DataFrame(rows)
