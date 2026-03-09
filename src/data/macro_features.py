"""Macro features modality -- 12-dimensional backward-looking macro state vector.

All features computed from data available BEFORE the prediction date.
No forward-looking information. Lagged by 1 day to prevent leakage.

Features:
1.  VIX level (normalized, 20-day z-score)
2.  VIX 5-day change
3.  10Y Treasury yield proxy (TLT price inverted, normalized)
4.  10Y-2Y yield curve spread proxy (IEF/TLT ratio momentum)
5.  DXY (USD index) 5-day momentum proxy (UUP ETF)
6.  SPY 5-day momentum (market momentum)
7.  SPY 20-day realized volatility
8.  QQQ vs SPY relative strength (tech vs broad market)
9.  Fed funds rate proxy (SHY price level, normalized)
10. Credit spread proxy (HYG/LQD ratio 5-day momentum)
11. Gold 5-day momentum (GLD, risk-off indicator)
12. Market breadth proxy (SPY RSI-14)

Data sources: Yahoo Finance (yfinance) for ^VIX, TLT, IEF, UUP, SPY, QQQ,
SHY, GLD, HYG, LQD -- all freely available.

Compute daily, lag by 1 day to prevent leakage.
Z-score normalize using training set statistics only.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


MACRO_TICKERS = ["^VIX", "TLT", "IEF", "SPY", "QQQ", "SHY", "GLD", "HYG", "LQD"]
MACRO_FEATURE_NAMES = [
    "vix_zscore_20d",
    "vix_5d_change",
    "treasury_10y_proxy",
    "yield_curve_spread_proxy",
    "usd_momentum_5d",
    "spy_momentum_5d",
    "spy_realized_vol_20d",
    "qqq_spy_relative_strength",
    "fed_funds_proxy",
    "credit_spread_momentum",
    "gold_momentum_5d",
    "spy_rsi_14",
]


def _compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Compute RSI (Relative Strength Index)."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
    rs = gain / (loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def download_macro_data(
    start_date: str = "2015-06-01",
    end_date: str = "2026-03-01",
    save_dir: str | Path = "data/raw/macro",
) -> pd.DataFrame:
    """Download macro data from Yahoo Finance and save to CSV.

    Downloads data starting 6 months before the project start date
    to have enough history for rolling calculations.
    """
    import yfinance as yf

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    all_data = {}
    for ticker in MACRO_TICKERS:
        print(f"  Downloading {ticker}...")
        df = yf.download(
            ticker, start=start_date, end=end_date, progress=False, auto_adjust=False
        )
        if len(df) > 0:
            # Handle MultiIndex columns from newer yfinance
            if isinstance(df.columns, pd.MultiIndex):
                level0 = df.columns.get_level_values(0)
                col = "Adj Close" if "Adj Close" in level0 else "Close"
                series = df[col]
                if isinstance(series, pd.DataFrame):
                    series = series.iloc[:, 0]
            else:
                col = "Adj Close" if "Adj Close" in df.columns else "Close"
                series = df[col]
            series.name = ticker
            all_data[ticker] = series

    macro_df = pd.DataFrame(all_data)
    macro_df.index.name = "date"
    macro_df = macro_df.sort_index()

    # Forward-fill missing data (holidays may differ across instruments)
    macro_df = macro_df.ffill()

    csv_path = save_dir / "macro_prices.csv"
    macro_df.to_csv(csv_path)
    print(f"  Saved macro data: {len(macro_df)} rows to {csv_path}")
    return macro_df


def load_macro_data(
    macro_csv: str | Path = "data/raw/macro/macro_prices.csv",
) -> pd.DataFrame:
    """Load pre-downloaded macro price data."""
    df = pd.read_csv(macro_csv, parse_dates=["date"], index_col="date")
    return df.sort_index()


def compute_macro_features(
    macro_df: pd.DataFrame,
    lag_days: int = 1,
) -> pd.DataFrame:
    """Compute the 12-dimensional macro feature vector.

    All features are backward-looking. A 1-day lag is applied to prevent
    any look-ahead bias (features on date t use data up to t-1).

    Parameters
    ----------
    macro_df : DataFrame with columns for macro tickers, indexed by date.
    lag_days : Number of days to lag features (default 1).

    Returns
    -------
    DataFrame indexed by date with 12 macro feature columns.
    """
    features = pd.DataFrame(index=macro_df.index)

    vix = macro_df.get("^VIX", pd.Series(dtype=float))
    spy = macro_df.get("SPY", pd.Series(dtype=float))
    qqq = macro_df.get("QQQ", pd.Series(dtype=float))
    tlt = macro_df.get("TLT", pd.Series(dtype=float))
    ief = macro_df.get("IEF", pd.Series(dtype=float))
    shy = macro_df.get("SHY", pd.Series(dtype=float))
    gld = macro_df.get("GLD", pd.Series(dtype=float))
    hyg = macro_df.get("HYG", pd.Series(dtype=float))
    lqd = macro_df.get("LQD", pd.Series(dtype=float))

    # 1. VIX level (20-day z-score)
    vix_mean = vix.rolling(20).mean()
    vix_std = vix.rolling(20).std()
    features["vix_zscore_20d"] = (vix - vix_mean) / (vix_std + 1e-8)

    # 2. VIX 5-day change
    features["vix_5d_change"] = vix.pct_change(5)

    # 3. 10Y Treasury yield proxy (TLT inversely related to yields)
    tlt_mean = tlt.rolling(20).mean()
    tlt_std = tlt.rolling(20).std()
    features["treasury_10y_proxy"] = -(tlt - tlt_mean) / (tlt_std + 1e-8)

    # 4. Yield curve spread proxy (IEF/TLT ratio momentum)
    ief_tlt_ratio = ief / (tlt + 1e-8)
    features["yield_curve_spread_proxy"] = ief_tlt_ratio.pct_change(5)

    # 5. USD momentum proxy (use SPY/QQQ as indirect -- or UUP if available)
    # Since UUP may not be in our data, use inverse of Gold as USD proxy
    features["usd_momentum_5d"] = -gld.pct_change(5)

    # 6. SPY 5-day momentum
    features["spy_momentum_5d"] = spy.pct_change(5)

    # 7. SPY 20-day realized volatility
    spy_log_ret = np.log(spy / spy.shift(1))
    features["spy_realized_vol_20d"] = spy_log_ret.rolling(20).std() * np.sqrt(252)

    # 8. QQQ vs SPY relative strength
    qqq_ret_5d = qqq.pct_change(5)
    spy_ret_5d = spy.pct_change(5)
    features["qqq_spy_relative_strength"] = qqq_ret_5d - spy_ret_5d

    # 9. Fed funds rate proxy (SHY level, normalized)
    shy_mean = shy.rolling(60).mean()
    shy_std = shy.rolling(60).std()
    features["fed_funds_proxy"] = -(shy - shy_mean) / (shy_std + 1e-8)

    # 10. Credit spread proxy (HYG/LQD ratio momentum)
    hyg_lqd_ratio = hyg / (lqd + 1e-8)
    features["credit_spread_momentum"] = hyg_lqd_ratio.pct_change(5)

    # 11. Gold 5-day momentum (risk-off indicator)
    features["gold_momentum_5d"] = gld.pct_change(5)

    # 12. Market breadth proxy (SPY RSI-14)
    features["spy_rsi_14"] = (_compute_rsi(spy, 14) - 50) / 50  # Normalize to [-1, 1]

    # Apply lag to prevent leakage
    if lag_days > 0:
        features = features.shift(lag_days)

    # Drop NaN rows from rolling calculations
    features = features.dropna()

    return features


def build_macro_feature_vectors(
    macro_features_df: pd.DataFrame,
    dates: list[str],
) -> dict[str, np.ndarray]:
    """Build macro feature vectors aligned to specific trading dates.

    Parameters
    ----------
    macro_features_df : DataFrame with 12 macro features, indexed by date.
    dates : List of date strings to align to.

    Returns
    -------
    dict mapping date_str -> np.ndarray of shape (12,)
    """
    macro_features_df = macro_features_df.copy()
    macro_features_df.index = pd.to_datetime(macro_features_df.index)

    result = {}
    for date_str in dates:
        dt = pd.Timestamp(date_str)
        # Find the most recent macro data point on or before this date
        valid = macro_features_df.index <= dt
        if valid.any():
            idx = macro_features_df.index[valid][-1]
            vec = macro_features_df.loc[idx, MACRO_FEATURE_NAMES].values.astype(
                np.float32
            )
            if not np.any(np.isnan(vec)):
                result[date_str] = vec

    return result


class MacroFeatureScaler:
    """Z-score normalization for macro features using training set statistics."""

    def __init__(self):
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None

    def fit(
        self, features_df: pd.DataFrame, train_end: str = "2022-12-31"
    ) -> "MacroFeatureScaler":
        """Fit scaler on training data only."""
        train_mask = features_df.index <= pd.Timestamp(train_end)
        train_data = features_df.loc[train_mask, MACRO_FEATURE_NAMES]
        self.mean = train_data.mean().values.astype(np.float32)
        self.std = train_data.std().values.astype(np.float32)
        self.std[self.std < 1e-8] = 1.0  # Prevent division by zero
        return self

    def transform(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Apply z-score normalization."""
        if self.mean is None:
            raise ValueError("Scaler not fitted. Call fit() first.")
        result = features_df.copy()
        for i, col in enumerate(MACRO_FEATURE_NAMES):
            if col in result.columns:
                result[col] = (result[col] - self.mean[i]) / self.std[i]
        return result
