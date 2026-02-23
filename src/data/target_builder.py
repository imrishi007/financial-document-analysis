from __future__ import annotations

import pandas as pd


def build_binary_direction_labels(
    prices: pd.DataFrame,
    horizon_days: int,
    threshold: float = 0.0,
) -> pd.DataFrame:
    required_columns = {"ticker", "date", "close"}
    missing = required_columns - set(prices.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    frame = prices.copy()
    frame = frame.sort_values(["ticker", "date"]).reset_index(drop=True)
    frame["close_t"] = frame["close"]
    frame["close_t_plus_h"] = frame.groupby("ticker")["close"].shift(-horizon_days)
    frame["future_return"] = frame["close_t_plus_h"] / frame["close_t"] - 1.0
    frame = frame.dropna(subset=["close_t_plus_h"]).copy()

    frame["target_id"] = (frame["future_return"] > threshold).astype(int)
    frame["target_label"] = frame["target_id"].map({1: "UP", 0: "DOWN"})

    return frame[
        ["ticker", "date", "close_t", "close_t_plus_h", "future_return", "target_id", "target_label"]
    ].reset_index(drop=True)
