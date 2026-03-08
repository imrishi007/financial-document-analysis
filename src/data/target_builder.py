from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Direction targets
# ---------------------------------------------------------------------------


def build_binary_direction_labels(
    prices: pd.DataFrame,
    horizon_days: int,
    threshold: float = 0.0,
) -> pd.DataFrame:
    """Build binary UP/DOWN labels for a single horizon."""
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
        [
            "ticker",
            "date",
            "close_t",
            "close_t_plus_h",
            "future_return",
            "target_id",
            "target_label",
        ]
    ].reset_index(drop=True)


def build_multi_horizon_direction_labels(
    prices: pd.DataFrame,
    horizons: list[int],
    threshold: float = 0.0,
) -> pd.DataFrame:
    """Build direction labels for multiple horizons (1, 3, 5 days etc).

    Returns a DataFrame with columns:
        ticker, date, close, direction_Xd_return, direction_Xd_id, direction_Xd_label
    for each horizon X.
    """
    required_columns = {"ticker", "date", "close"}
    missing = required_columns - set(prices.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    frame = prices[["ticker", "date", "close"]].copy()
    frame = frame.sort_values(["ticker", "date"]).reset_index(drop=True)

    for h in horizons:
        col_return = f"direction_{h}d_return"
        col_id = f"direction_{h}d_id"
        col_label = f"direction_{h}d_label"

        close_future = frame.groupby("ticker")["close"].shift(-h)
        frame[col_return] = close_future / frame["close"] - 1.0
        frame[col_id] = (frame[col_return] > threshold).astype("Int64")
        frame[col_label] = frame[col_id].map({1: "UP", 0: "DOWN"})

    # Drop rows where the longest horizon is NaN
    longest = max(horizons)
    frame = frame.dropna(subset=[f"direction_{longest}d_return"]).copy()
    frame = frame.reset_index(drop=True)
    return frame


# ---------------------------------------------------------------------------
# Volatility targets
# ---------------------------------------------------------------------------


def build_realized_volatility_targets(
    prices: pd.DataFrame,
    lookback_days: int = 20,
) -> pd.DataFrame:
    """Compute realized volatility as the rolling std of daily log returns.

    Returns a DataFrame with columns:
        ticker, date, close, log_return, realized_vol_{lookback_days}d
    """
    required_columns = {"ticker", "date", "close"}
    missing = required_columns - set(prices.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    frame = prices[["ticker", "date", "close"]].copy()
    frame = frame.sort_values(["ticker", "date"]).reset_index(drop=True)

    frame["log_return"] = frame.groupby("ticker")["close"].transform(
        lambda s: np.log(s / s.shift(1))
    )

    vol_col = f"realized_vol_{lookback_days}d"
    frame[vol_col] = frame.groupby("ticker")["log_return"].transform(
        lambda s: s.rolling(lookback_days, min_periods=lookback_days).std()
    )

    # Annualize: multiply by sqrt(252)
    frame[f"{vol_col}_annualized"] = frame[vol_col] * np.sqrt(252)

    frame = frame.dropna(subset=[vol_col]).copy()
    frame = frame.reset_index(drop=True)
    return frame


# ---------------------------------------------------------------------------
# Fundamental surprise targets (aligned to trading dates)
# ---------------------------------------------------------------------------


def build_fundamental_surprise_targets(
    earnings: pd.DataFrame,
    prices: pd.DataFrame,
) -> pd.DataFrame:
    """Align earnings surprise labels to the nearest trading dates.

    For each earnings announcement, finds the closest trading date and
    marks it with the BEAT/MISS label + surprise percentage. Trading dates
    without an earning event get label NONE.

    Parameters
    ----------
    earnings : DataFrame with columns ticker, announcement_date, surprise_pct, surprise_label
    prices : DataFrame with columns ticker, date (trading dates)

    Returns
    -------
    DataFrame with columns:
        ticker, date, has_earnings_event, surprise_pct, surprise_label
    """
    required_earnings = {
        "ticker",
        "announcement_date",
        "surprise_pct",
        "surprise_label",
    }
    missing_e = required_earnings - set(earnings.columns)
    if missing_e:
        raise ValueError(f"Missing earnings columns: {sorted(missing_e)}")

    required_prices = {"ticker", "date"}
    missing_p = required_prices - set(prices.columns)
    if missing_p:
        raise ValueError(f"Missing price columns: {sorted(missing_p)}")

    # Build the base frame from trading dates
    base = prices[["ticker", "date"]].copy()
    base["date"] = pd.to_datetime(base["date"])
    base = base.sort_values(["ticker", "date"]).reset_index(drop=True)

    # Clean earnings
    earn = earnings[
        ["ticker", "announcement_date", "surprise_pct", "surprise_label"]
    ].copy()
    earn["announcement_date"] = pd.to_datetime(earn["announcement_date"])
    # Drop UNKNOWN labels and future dates
    earn = earn[earn["surprise_label"].isin(["BEAT", "MISS"])].copy()

    # For each earnings announcement, find nearest trading date per ticker
    merged_rows = []
    for ticker in earn["ticker"].unique():
        tick_earn = earn[earn["ticker"] == ticker].copy()
        tick_prices = base[base["ticker"] == ticker]["date"].sort_values().values

        if len(tick_prices) == 0:
            continue

        for _, row in tick_earn.iterrows():
            ann_date = row["announcement_date"]
            # Find nearest trading date (on or after announcement)
            idx = np.searchsorted(tick_prices, ann_date, side="left")
            if idx >= len(tick_prices):
                idx = len(tick_prices) - 1
            nearest = pd.Timestamp(tick_prices[idx])

            # If announcement was before market open, use same day;
            # otherwise use next trading day
            if idx < len(tick_prices) - 1 and nearest < ann_date:
                nearest = pd.Timestamp(tick_prices[idx + 1])

            merged_rows.append(
                {
                    "ticker": ticker,
                    "date": nearest,
                    "surprise_pct": row["surprise_pct"],
                    "surprise_label": row["surprise_label"],
                }
            )

    if merged_rows:
        events = pd.DataFrame(merged_rows)
        # Deduplicate: if multiple events on same date, keep the latest
        events = events.drop_duplicates(subset=["ticker", "date"], keep="last")

        result = base.merge(events, on=["ticker", "date"], how="left")
    else:
        result = base.copy()
        result["surprise_pct"] = np.nan
        result["surprise_label"] = None

    result["has_earnings_event"] = result["surprise_label"].notna().astype(int)
    result["surprise_label"] = result["surprise_label"].fillna("NONE")
    result["surprise_pct"] = result["surprise_pct"].fillna(0.0)

    # Encode: BEAT=1, MISS=0, NONE=-1
    label_map = {"BEAT": 1, "MISS": 0, "NONE": -1}
    result["surprise_id"] = result["surprise_label"].map(label_map)

    return result.reset_index(drop=True)
