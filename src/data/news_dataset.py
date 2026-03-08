"""News article dataset with temporal windowing for FinBERT encoding.

For each (ticker, date) pair, collects articles published within a lookback
window and tokenizes the title + summary for each article.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.data.preprocessing import tokenize_short_text


def load_news_articles(news_csv: str | Path) -> pd.DataFrame:
    """Load news articles CSV and parse dates to UTC datetime."""
    df = pd.read_csv(news_csv)
    df["published_at"] = pd.to_datetime(df["published_at"], format="mixed", utc=True)
    # Normalize to date (tz-naive) for easy merging
    df["date"] = df["published_at"].dt.tz_localize(None).dt.normalize()
    return df


def _build_text(row: pd.Series) -> str:
    """Combine title and summary into a single string for tokenization."""
    title = str(row.get("title", "")) if pd.notna(row.get("title")) else ""
    summary = str(row.get("summary", "")) if pd.notna(row.get("summary")) else ""
    if summary:
        return f"{title}. {summary}"
    return title


class NewsWindowDataset(Dataset):
    """Dataset of news articles grouped by (ticker, date-window).

    For each anchor date, gathers articles published within
    ``[date - window_days, date]`` for a given ticker, tokenizes them, and
    pads/truncates to a fixed ``max_articles`` count.

    The existing ``NewsTemporalDirectionModel`` expects input of shape
    ``(batch, num_articles, seq_len)``, which this dataset provides.

    Parameters
    ----------
    news_df : DataFrame from ``load_news_articles``.
    anchor_dates : DataFrame with columns [ticker, date] — the prediction
        dates we want news features for.
    tokenizer : HuggingFace tokenizer.
    max_articles : Maximum articles per window (others are truncated, most recent first).
    max_length : Maximum tokens per article.
    window_days : Number of calendar days to look back for articles.
    """

    def __init__(
        self,
        news_df: pd.DataFrame,
        anchor_dates: pd.DataFrame,
        tokenizer,
        max_articles: int = 16,
        max_length: int = 128,
        window_days: int = 7,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_articles = max_articles
        self.max_length = max_length
        self.window_days = window_days

        news = news_df.copy()
        news["date"] = pd.to_datetime(news["date"])

        anchors = anchor_dates.copy()
        anchors["date"] = pd.to_datetime(anchors["date"])

        # Pre-compute article text
        news["text"] = news.apply(_build_text, axis=1)

        # Build samples: for each (ticker, date), gather windowed articles
        self._samples: list[dict] = []
        self._pad_ids = [tokenizer.pad_token_id] * max_length
        self._pad_mask = [0] * max_length

        for _, anchor in anchors.iterrows():
            ticker = anchor["ticker"]
            dt = anchor["date"]
            window_start = dt - pd.Timedelta(days=window_days)

            # Filter articles for this ticker in the window
            mask = (
                (news["ticker"] == ticker)
                & (news["date"] >= window_start)
                & (news["date"] <= dt)
            )
            window_articles = news.loc[mask].sort_values(
                "published_at", ascending=False
            )

            # Tokenize up to max_articles
            article_ids = []
            article_masks = []
            n_articles = min(len(window_articles), max_articles)

            for _, art in window_articles.head(max_articles).iterrows():
                tok = tokenize_short_text(art["text"], tokenizer, max_length)
                article_ids.append(tok["input_ids"])
                article_masks.append(tok["attention_mask"])

            # Pad to max_articles
            while len(article_ids) < max_articles:
                article_ids.append(self._pad_ids)
                article_masks.append(self._pad_mask)

            self._samples.append(
                {
                    "input_ids": np.array(article_ids, dtype=np.int64),
                    "attention_mask": np.array(article_masks, dtype=np.int64),
                    "num_articles": n_articles,
                    "ticker": str(ticker),
                    "date": str(dt.date()),
                }
            )

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> dict:
        s = self._samples[idx]
        return {
            "input_ids": torch.from_numpy(s["input_ids"]),  # [A, S]
            "attention_mask": torch.from_numpy(s["attention_mask"]),  # [A, S]
            "num_articles": torch.tensor(s["num_articles"], dtype=torch.long),
            "ticker": s["ticker"],
            "date": s["date"],
        }
