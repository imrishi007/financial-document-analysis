"""Common preprocessing utilities: normalization, time-based splits, text chunking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Time-based splits
# ---------------------------------------------------------------------------


@dataclass
class SplitConfig:
    """Configuration for chronological train / val / test split."""

    train_end: str = "2022-12-31"
    val_end: str = "2023-12-31"
    # Everything after val_end is test


def create_time_splits(
    dates: pd.Series,
    cfg: Optional[SplitConfig] = None,
) -> dict[str, np.ndarray]:
    """Return boolean index arrays for train / val / test based on date cutoffs.

    Parameters
    ----------
    dates : pd.Series of datetime-like values (will be coerced).
    cfg : SplitConfig with cutoff dates.

    Returns
    -------
    dict with keys 'train', 'val', 'test', each a boolean np.ndarray.
    """
    if cfg is None:
        cfg = SplitConfig()

    dt = pd.to_datetime(dates)
    train_end = pd.Timestamp(cfg.train_end)
    val_end = pd.Timestamp(cfg.val_end)

    train_mask = (dt <= train_end).values
    val_mask = ((dt > train_end) & (dt <= val_end)).values
    test_mask = (dt > val_end).values

    return {"train": train_mask, "val": val_mask, "test": test_mask}


# ---------------------------------------------------------------------------
# Feature normalization (z-score)
# ---------------------------------------------------------------------------


@dataclass
class FeatureScaler:
    """Stores mean / std computed on training data for z-score normalization."""

    mean: pd.Series
    std: pd.Series

    def transform(self, df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
        out = df.copy()
        for c in cols:
            out[c] = (out[c] - self.mean[c]) / (self.std[c] + 1e-8)
        return out


def fit_scaler(df: pd.DataFrame, cols: list[str]) -> FeatureScaler:
    """Fit a z-score scaler on the given dataframe (should be training set only)."""
    return FeatureScaler(mean=df[cols].mean(), std=df[cols].std())


# ---------------------------------------------------------------------------
# Text chunking for long documents (FinBERT 512-token limit)
# ---------------------------------------------------------------------------


def chunk_text(
    text: str,
    tokenizer,
    max_length: int = 510,
    stride: int = 128,
) -> list[dict]:
    """Split a long text into overlapping token chunks suitable for BERT.

    Each chunk has at most `max_length` tokens (excluding [CLS]/[SEP]).
    Consecutive chunks overlap by `stride` tokens for context continuity.

    Parameters
    ----------
    text : The long text to chunk.
    tokenizer : A HuggingFace tokenizer.
    max_length : Max tokens per chunk (before adding special tokens).
    stride : Overlap between consecutive chunks.

    Returns
    -------
    List of dicts, each with 'input_ids' and 'attention_mask' as lists of ints.
    """
    # Tokenize without truncation to get full token ids
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_attention_mask=False,
        return_token_type_ids=False,
    )
    all_ids = encoded["input_ids"]

    if len(all_ids) == 0:
        # Empty text — return a single padding chunk
        dummy = tokenizer(
            "",
            max_length=max_length + 2,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return [
            {
                "input_ids": dummy["input_ids"][0].tolist(),
                "attention_mask": dummy["attention_mask"][0].tolist(),
            }
        ]

    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id

    chunks = []
    start = 0
    while start < len(all_ids):
        end = start + max_length
        chunk_ids = all_ids[start:end]

        # Add [CLS] and [SEP]
        input_ids = [cls_id] + chunk_ids + [sep_id]
        attention_mask = [1] * len(input_ids)

        # Pad to max_length + 2 (for [CLS] and [SEP])
        pad_len = (max_length + 2) - len(input_ids)
        if pad_len > 0:
            input_ids += [tokenizer.pad_token_id] * pad_len
            attention_mask += [0] * pad_len

        chunks.append(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }
        )

        if end >= len(all_ids):
            break
        start += max_length - stride

    return chunks


def tokenize_short_text(
    text: str,
    tokenizer,
    max_length: int = 128,
) -> dict:
    """Tokenize a short text (news title/summary) with truncation and padding.

    Returns dict with 'input_ids' and 'attention_mask' as lists.
    """
    if not text or (isinstance(text, float) and np.isnan(text)):
        text = ""

    encoded = tokenizer(
        text,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    return {
        "input_ids": encoded["input_ids"][0].tolist(),
        "attention_mask": encoded["attention_mask"][0].tolist(),
    }
