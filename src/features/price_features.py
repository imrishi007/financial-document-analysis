from __future__ import annotations

import pandas as pd


def build_price_features(prices: pd.DataFrame) -> pd.DataFrame:
    required_columns = {"ticker", "date", "close", "volume"}
    missing = required_columns - set(prices.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    frame = prices.copy().sort_values(["ticker", "date"]).reset_index(drop=True)
    by_ticker = frame.groupby("ticker")

    frame["ret_1"] = by_ticker["close"].pct_change(1)
    frame["ret_5"] = by_ticker["close"].pct_change(5)
    frame["ret_10"] = by_ticker["close"].pct_change(10)
    frame["volatility_10"] = by_ticker["close"].pct_change().rolling(10).std().reset_index(level=0, drop=True)
    frame["volume_change_1"] = by_ticker["volume"].pct_change(1)
    frame["ma_5"] = by_ticker["close"].rolling(5).mean().reset_index(level=0, drop=True)
    frame["ma_20"] = by_ticker["close"].rolling(20).mean().reset_index(level=0, drop=True)
    frame["ma_ratio_5_20"] = frame["ma_5"] / frame["ma_20"]

    return frame
