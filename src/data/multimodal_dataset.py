"""Aligned multimodal dataset joining price, document, and news modalities.

Each sample is indexed by (ticker, date) and returns:
- Price window features (always available).
- Most recent 10-K filing chunks (may be missing for early dates).
- Recent news articles (may be missing for dates before news coverage).
- Multi-task labels: direction, volatility, fundamental surprise.

Missing modalities are zero-padded and flagged via ``*_available`` masks
so the fusion model can gate them appropriately.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.data.document_dataset import DocumentChunkDataset
from src.data.news_dataset import NewsWindowDataset
from src.data.price_dataset import PriceWindowDataset


class MultimodalAlignedDataset(Dataset):
    """Combines price, document, and news into one aligned sample per (ticker, date).

    This dataset wraps pre-built modality-specific datasets and aligns them
    by (ticker, date) so a single ``__getitem__`` call returns all inputs
    needed by the full fusion pipeline.

    Parameters
    ----------
    price_dataset : ``PriceWindowDataset`` — provides feature windows and labels.
    doc_dataset : ``DocumentChunkDataset`` or None — provides 10-K chunks.
    news_dataset : ``NewsWindowDataset`` or None — provides temporally windowed news.
    doc_year_for_date_fn : Optional callable ``(ticker: str, date: str) -> int | None``
        that maps a prediction date to the relevant 10-K filing year.
        Default: ``year_of_date - 1`` (most recent annual filing).
    """

    def __init__(
        self,
        price_dataset: "PriceWindowDataset",
        doc_dataset: Optional["DocumentChunkDataset"] = None,
        news_dataset: Optional["NewsWindowDataset"] = None,
        doc_year_for_date_fn=None,
    ) -> None:
        self.price_ds = price_dataset
        self.doc_ds = doc_dataset
        self.news_ds = news_dataset

        # Build a lookup for document dataset: (ticker, year) -> index
        self.doc_lookup: dict[tuple[str, int], int] = {}
        if doc_dataset is not None:
            self.doc_lookup = doc_dataset.get_filing_lookup()

        # Build a lookup for news dataset: (ticker, date_str) -> index
        self.news_lookup: dict[tuple[str, str], int] = {}
        if news_dataset is not None:
            for i in range(len(news_dataset)):
                s = news_dataset._samples[i]
                self.news_lookup[(s["ticker"], s["date"])] = i

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

        # --- News modality ---
        news_available = False
        if self.news_ds is not None:
            news_idx = self.news_lookup.get((ticker, date_str))
            if news_idx is not None:
                news_sample = self.news_ds[news_idx]
                if news_sample["num_articles"].item() > 0:
                    sample["news_input_ids"] = news_sample["input_ids"]
                    sample["news_attention_mask"] = news_sample["attention_mask"]
                    sample["news_num_articles"] = news_sample["num_articles"]
                    news_available = True

        if not news_available:
            a = self.news_ds.max_articles if self.news_ds else 16
            s = self.news_ds.max_length if self.news_ds else 128
            sample["news_input_ids"] = torch.zeros(a, s, dtype=torch.long)
            sample["news_attention_mask"] = torch.zeros(a, s, dtype=torch.long)
            sample["news_num_articles"] = torch.tensor(0, dtype=torch.long)

        sample["news_available"] = torch.tensor(int(news_available), dtype=torch.long)

        return sample
