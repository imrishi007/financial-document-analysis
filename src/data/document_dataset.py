"""10-K document loading and chunked FinBERT dataset.

Provides:
- ``load_processed_filings``: load all processed JSONs into a DataFrame.
- ``DocumentChunkDataset``: a PyTorch Dataset that splits long 10-K texts
  into overlapping 512-token chunks for FinBERT encoding.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.data.preprocessing import chunk_text


# Sections to concatenate from each 10-K filing
_SECTION_KEYS = ["item_1", "item_1a", "item_7", "item_7a", "item_8"]


def load_processed_filings(processed_dir: str | Path) -> pd.DataFrame:
    """Load all ``*_processed.json`` files into a DataFrame.

    Returns columns: ticker, year, filename, document_text.
    """
    processed_path = Path(processed_dir)
    rows = []

    for file_path in sorted(processed_path.glob("*_processed.json")):
        with open(file_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)

        sections = payload.get("sections", {})
        document_text = "\n\n".join(
            filter(
                None,
                [sections.get(k, "") for k in _SECTION_KEYS],
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


class DocumentChunkDataset(Dataset):
    """Dataset of chunked 10-K filings tokenized for FinBERT.

    Each sample corresponds to a single (ticker, year) filing.
    The full document text is split into overlapping 512-token chunks.
    A sample returns *all* chunks for that filing so the model can
    aggregate them (e.g., via attention pooling over chunks).

    Parameters
    ----------
    processed_dir : Path to the ``data/processed/`` directory.
    tokenizer : A HuggingFace tokenizer (e.g., ``AutoTokenizer.from_pretrained("ProsusAI/finbert")``).
    max_chunks : Maximum number of chunks to keep per document.
        Longer documents are truncated to this many chunks.
    max_length : Max tokens per chunk (before [CLS]/[SEP]).
    stride : Overlap between consecutive chunks.
    """

    def __init__(
        self,
        processed_dir: str | Path,
        tokenizer,
        max_chunks: int = 64,
        max_length: int = 510,
        stride: int = 128,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_chunks = max_chunks
        self.max_length = max_length
        self.stride = stride
        self.seq_len = max_length + 2  # [CLS] + tokens + [SEP]

        filings_df = load_processed_filings(processed_dir)
        if filings_df.empty:
            raise FileNotFoundError(f"No processed filings in {processed_dir}")

        # Pre-chunk and tokenize every filing
        self._samples: list[dict] = []
        for _, row in filings_df.iterrows():
            chunks = chunk_text(
                row["document_text"],
                tokenizer,
                max_length=max_length,
                stride=stride,
            )
            # Truncate to max_chunks
            chunks = chunks[: self.max_chunks]

            ids = [c["input_ids"] for c in chunks]
            masks = [c["attention_mask"] for c in chunks]
            num_chunks = len(chunks)

            # Pad to max_chunks
            pad_chunk_ids = [tokenizer.pad_token_id] * self.seq_len
            pad_chunk_mask = [0] * self.seq_len
            while len(ids) < self.max_chunks:
                ids.append(pad_chunk_ids)
                masks.append(pad_chunk_mask)

            self._samples.append(
                {
                    "input_ids": np.array(ids, dtype=np.int64),  # [max_chunks, seq_len]
                    "attention_mask": np.array(
                        masks, dtype=np.int64
                    ),  # [max_chunks, seq_len]
                    "num_chunks": num_chunks,
                    "ticker": str(row["ticker"]),
                    "year": int(row["year"]),
                }
            )

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> dict:
        s = self._samples[idx]
        return {
            "input_ids": torch.from_numpy(s["input_ids"]),  # [C, S]
            "attention_mask": torch.from_numpy(s["attention_mask"]),  # [C, S]
            "num_chunks": torch.tensor(s["num_chunks"], dtype=torch.long),
            "ticker": s["ticker"],
            "year": s["year"],
        }

    def get_filing_lookup(self) -> dict[tuple[str, int], int]:
        """Return a dict mapping (ticker, year) -> dataset index."""
        return {(s["ticker"], s["year"]): i for i, s in enumerate(self._samples)}
