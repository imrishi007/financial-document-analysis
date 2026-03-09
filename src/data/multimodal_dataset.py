"""Aligned multimodal dataset joining price and document modalities.

Each sample is indexed by (ticker, date) and returns:
- Price window features (always available).
- Most recent 10-K filing chunks (may be missing for early dates).
- Multi-task labels: direction (60-day primary), volatility.

Missing modalities are zero-padded and flagged via ``*_available`` masks
so the fusion model can gate them appropriately.

V2: News modality removed entirely.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.data.document_dataset import DocumentChunkDataset
from src.data.price_dataset import PriceWindowDataset


class MultimodalAlignedDataset(Dataset):
    """Combines price and document into one aligned sample per (ticker, date).

    This dataset wraps pre-built modality-specific datasets and aligns them
    by (ticker, date) so a single ``__getitem__`` call returns all inputs
    needed by the full fusion pipeline.

    Parameters
    ----------
    price_dataset : ``PriceWindowDataset`` -- provides feature windows and labels.
    doc_dataset : ``DocumentChunkDataset`` or None -- provides 10-K chunks.
    doc_year_for_date_fn : Optional callable ``(ticker: str, date: str) -> int | None``
        that maps a prediction date to the relevant 10-K filing year.
        Default: ``year_of_date - 1`` (most recent annual filing).
    """

    def __init__(
        self,
        price_dataset: "PriceWindowDataset",
        doc_dataset: Optional["DocumentChunkDataset"] = None,
        doc_year_for_date_fn=None,
    ) -> None:
        self.price_ds = price_dataset
        self.doc_ds = doc_dataset

        # Build a lookup for document dataset: (ticker, year) -> index
        self.doc_lookup: dict[tuple[str, int], int] = {}
        if doc_dataset is not None:
            self.doc_lookup = doc_dataset.get_filing_lookup()

        # Default: prediction date's year - 1 gives the most recent 10-K
        self.doc_year_fn = doc_year_for_date_fn or self._default_doc_year

    @staticmethod
    def _default_doc_year(ticker: str, date_str: str) -> int:
        """Map a prediction date to the 10-K filing year (prior year)."""
        year = int(date_str[:4])
        return year - 1

    def __len__(self) -> int:
        return len(self.price_ds)

    def __getitem__(self, idx: int) -> dict:
        sample = self.price_ds[idx]
        ticker = sample["ticker"]
        date_str = sample["date"]

        # --- Document modality ---
        doc_available = False
        if self.doc_ds is not None:
            doc_year = self.doc_year_fn(ticker, date_str)
            doc_idx = self.doc_lookup.get((ticker, doc_year))
            if doc_idx is not None:
                doc_sample = self.doc_ds[doc_idx]
                sample["doc_input_ids"] = doc_sample["input_ids"]
                sample["doc_attention_mask"] = doc_sample["attention_mask"]
                sample["doc_num_chunks"] = doc_sample["num_chunks"]
                doc_available = True

        if not doc_available:
            # Zero-pad document tensors
            c = self.doc_ds.max_chunks if self.doc_ds else 64
            s = self.doc_ds.seq_len if self.doc_ds else 512
            sample["doc_input_ids"] = torch.zeros(c, s, dtype=torch.long)
            sample["doc_attention_mask"] = torch.zeros(c, s, dtype=torch.long)
            sample["doc_num_chunks"] = torch.tensor(0, dtype=torch.long)

        sample["doc_available"] = torch.tensor(int(doc_available), dtype=torch.long)

        return sample
