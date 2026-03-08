"""Graph-snapshot dataset: all companies for a single date.

The GAT model processes all N companies simultaneously on each date so that
inter-company attention can propagate signals. This dataset groups the
per-company sliding windows by date, producing one "snapshot" per trading day
that contains the price features and direction labels for every company
present on that date.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.data.price_dataset import (
    ENGINEERED_FEATURES,
    load_price_csv_dir,
    prepare_price_features,
)
from src.data.preprocessing import fit_scaler


class GraphSnapshotDataset(Dataset):
    """Each sample is a full-graph snapshot: N companies × T time steps × F features.

    For dates where a company is missing (e.g. not enough history yet), that
    company's features are zero-padded and a boolean ``mask`` tensor indicates
    which companies are valid.

    Parameters
    ----------
    price_df : DataFrame with [ticker, date] + feature_cols (already normalized).
    targets_df : DataFrame with [ticker, date, direction_1d_id].
    tickers : ordered list of tickers (defines node ordering).
    window_size : number of consecutive days per company window.
    feature_cols : which columns to use as features.
    """

    def __init__(
        self,
        price_df: pd.DataFrame,
        targets_df: pd.DataFrame,
        tickers: list[str],
        window_size: int = 60,
        feature_cols: list[str] | None = None,
    ) -> None:
        self.tickers = tickers
        self.ticker_to_idx = {t: i for i, t in enumerate(tickers)}
        self.num_nodes = len(tickers)
        self.window_size = window_size
        self.feature_cols = feature_cols or ENGINEERED_FEATURES
        self.num_features = len(self.feature_cols)

        price_df = price_df.copy()
        targets_df = targets_df.copy()
        price_df["date"] = pd.to_datetime(price_df["date"])
        targets_df["date"] = pd.to_datetime(targets_df["date"])

        # Merge direction labels
        merged = price_df.merge(
            targets_df[["ticker", "date", "target_id"]],
            on=["ticker", "date"],
            how="inner",
        )
        merged = merged.sort_values(["ticker", "date"]).reset_index(drop=True)

        # Build per-ticker window indices
        # For each ticker, store (date -> feature_window) mapping
        per_ticker: dict[str, dict[str, tuple[np.ndarray, int]]] = {}
        for ticker in tickers:
            tk_data = merged[merged["ticker"] == ticker].reset_index(drop=True)
            if len(tk_data) < window_size:
                per_ticker[ticker] = {}
                continue
            tk_windows: dict[str, tuple[np.ndarray, int]] = {}
            for i in range(window_size, len(tk_data)):
                window = tk_data.iloc[i - window_size : i][self.feature_cols].values
                label = int(tk_data.iloc[i]["target_id"])
                date_str = str(tk_data.iloc[i]["date"].date())
                tk_windows[date_str] = (window.astype(np.float32), label)
            per_ticker[ticker] = tk_windows

        # Find all dates where at least 1 company is present
        all_dates: set[str] = set()
        for tk_windows in per_ticker.values():
            all_dates.update(tk_windows.keys())

        # Build snapshots — one per date
        self._snapshots: list[dict] = []
        for date_str in sorted(all_dates):
            features = np.zeros(
                (self.num_nodes, window_size, self.num_features), dtype=np.float32
            )
            labels = np.full(self.num_nodes, -1, dtype=np.int64)  # -1 = missing
            mask = np.zeros(self.num_nodes, dtype=np.bool_)

            for ticker, tk_windows in per_ticker.items():
                if date_str not in tk_windows:
                    continue
                idx = self.ticker_to_idx[ticker]
                feat_window, label = tk_windows[date_str]
                features[idx] = feat_window
                labels[idx] = label
                mask[idx] = True

            # Only keep dates with sufficient coverage (≥ 5 companies)
            if mask.sum() >= 5:
                self._snapshots.append(
                    {
                        "features": features,  # [N, T, F]
                        "labels": labels,  # [N]
                        "mask": mask,  # [N]
                        "date": date_str,
                    }
                )

    def __len__(self) -> int:
        return len(self._snapshots)

    def __getitem__(self, idx: int) -> dict:
        s = self._snapshots[idx]
        return {
            "features": torch.from_numpy(s["features"]),  # [N, T, F]
            "labels": torch.from_numpy(s["labels"]),  # [N]
            "mask": torch.from_numpy(s["mask"]),  # [N]
            "date": s["date"],
        }


def build_graph_snapshots(
    prices_dir: str | Path = "data/raw/prices",
    targets_dir: str | Path = "data/targets",
    tickers: list[str] | None = None,
    window_size: int = 60,
    feature_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Load and prepare data for GraphSnapshotDataset.

    Returns
    -------
    (price_df, targets_df, tickers)
    """
    if tickers is None:
        tickers = [
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

    prices = load_price_csv_dir(prices_dir)
    price_df = prepare_price_features(prices, feature_cols)

    targets_path = Path(targets_dir) / "direction_labels.csv"
    targets_df = pd.read_csv(targets_path, parse_dates=["date"])

    return price_df, targets_df, tickers
