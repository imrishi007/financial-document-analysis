"""Sliding-window price dataset with integrated feature engineering."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.features.price_features import build_price_features
from src.data.preprocessing import FeatureScaler, fit_scaler


# Features produced by build_price_features (excluding ticker/date/raw OHLCV)
ENGINEERED_FEATURES = [
    "close",
    "volume",
    "ret_1",
    "ret_5",
    "ret_10",
    "volatility_10",
    "volume_change_1",
    "ma_5",
    "ma_20",
    "ma_ratio_5_20",
]


def load_price_csv_dir(prices_dir: str | Path) -> pd.DataFrame:
    """Load and concatenate all per-ticker price CSVs from a directory."""
    frames = []
    for f in sorted(Path(prices_dir).glob("*.csv")):
        df = pd.read_csv(f, parse_dates=["date"])
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No CSV files found in {prices_dir}")
    prices = pd.concat(frames, ignore_index=True)
    prices["date"] = pd.to_datetime(prices["date"])
    return prices.sort_values(["ticker", "date"]).reset_index(drop=True)


def prepare_price_features(
    prices: pd.DataFrame,
    feature_cols: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Apply feature engineering and return a clean DataFrame ready for windowing.

    Returns DataFrame with columns: ticker, date, + feature_cols (all numeric).
    """
    feat = build_price_features(prices)
    if feature_cols is None:
        feature_cols = ENGINEERED_FEATURES
    # Drop rows with NaN in any feature (from rolling calculations)
    subset = feat[["ticker", "date"] + feature_cols].dropna().copy()
    subset = subset.sort_values(["ticker", "date"]).reset_index(drop=True)
    return subset


class PriceWindowDataset(Dataset):
    """Sliding-window dataset over engineered price features.

    For each sample, returns a window of `window_size` consecutive trading days
    of feature vectors, plus the aligned prediction targets.

    PRIMARY target: 60-day direction (direction_60d)
    Secondary targets: 5-day direction, volatility

    Parameters
    ----------
    price_df : DataFrame with columns [ticker, date] + feature_cols (already
        feature-engineered and optionally normalized).
    targets_df : DataFrame with columns [ticker, date, direction_5d_id,
        direction_60d_id] (from multi-horizon direction labels).
    vol_df : Optional DataFrame with [ticker, date, realized_vol_20d_annualized].
    surprise_df : Optional DataFrame with [ticker, date, surprise_id].
    window_size : Number of consecutive days per sample.
    feature_cols : Columns to include as features (must exist in price_df).
    """

    def __init__(
        self,
        price_df: pd.DataFrame,
        targets_df: pd.DataFrame,
        vol_df: Optional[pd.DataFrame] = None,
        surprise_df: Optional[pd.DataFrame] = None,
        window_size: int = 60,
        feature_cols: Optional[list[str]] = None,
    ) -> None:
        self.window_size = window_size
        self.feature_cols = feature_cols or ENGINEERED_FEATURES

        # Ensure date types match for merging
        price_df = price_df.copy()
        targets_df = targets_df.copy()
        price_df["date"] = pd.to_datetime(price_df["date"])
        targets_df["date"] = pd.to_datetime(targets_df["date"])

        # Determine which direction columns are available
        dir_cols = ["ticker", "date"]
        has_5d = "direction_5d_id" in targets_df.columns
        has_60d = "direction_60d_id" in targets_df.columns
        has_1d = "direction_1d_id" in targets_df.columns

        if has_5d:
            dir_cols.append("direction_5d_id")
        if has_60d:
            dir_cols.append("direction_60d_id")
        if has_1d:
            dir_cols.append("direction_1d_id")

        # Merge price features with direction targets
        merged = price_df.merge(
            targets_df[dir_cols],
            on=["ticker", "date"],
            how="inner",
        )

        # Optionally merge volatility
        if vol_df is not None:
            vol_df = vol_df.copy()
            vol_df["date"] = pd.to_datetime(vol_df["date"])
            merged = merged.merge(
                vol_df[["ticker", "date", "realized_vol_20d_annualized"]],
                on=["ticker", "date"],
                how="left",
            )
        else:
            merged["realized_vol_20d_annualized"] = np.nan

        # Optionally merge fundamental surprise
        if surprise_df is not None:
            surprise_df = surprise_df.copy()
            surprise_df["date"] = pd.to_datetime(surprise_df["date"])
            merged = merged.merge(
                surprise_df[["ticker", "date", "surprise_id"]],
                on=["ticker", "date"],
                how="left",
            )
        else:
            merged["surprise_id"] = -1

        merged = merged.sort_values(["ticker", "date"]).reset_index(drop=True)

        # Build valid window indices: for each ticker, we need at least
        # window_size consecutive rows. Store (start_idx, end_idx) pairs.
        self._samples: list[dict] = []
        for ticker, group in merged.groupby("ticker"):
            group = group.reset_index(drop=True)
            n = len(group)
            if n < window_size:
                continue
            for i in range(window_size, n):
                window = group.iloc[i - window_size : i]
                target_row = group.iloc[i]

                sample = {
                    "features": window[self.feature_cols].values.astype(np.float32),
                    "direction_5d": int(target_row.get("direction_5d_id", -1)),
                    "direction_60d": int(target_row.get("direction_60d_id", -1)),
                    "direction_1d": int(target_row.get("direction_1d_id", -1)),
                    "volatility": (
                        float(target_row["realized_vol_20d_annualized"])
                        if not np.isnan(target_row["realized_vol_20d_annualized"])
                        else 0.0
                    ),
                    "surprise_id": int(target_row.get("surprise_id", -1)),
                    "ticker": str(ticker),
                    "date": str(target_row["date"].date()),
                }
                self._samples.append(sample)

        if not self._samples:
            raise ValueError("No valid windows could be created. Check data.")

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> dict:
        s = self._samples[idx]
        return {
            "features": torch.from_numpy(s["features"]),  # [window, F]
            "direction_60d": torch.tensor(s["direction_60d"], dtype=torch.long),
            "direction_5d": torch.tensor(s["direction_5d"], dtype=torch.long),
            "direction_1d": torch.tensor(s["direction_1d"], dtype=torch.long),
            "volatility": torch.tensor(s["volatility"], dtype=torch.float32),
            "surprise_id": torch.tensor(s["surprise_id"], dtype=torch.long),
            "ticker": s["ticker"],
            "date": s["date"],
        }
