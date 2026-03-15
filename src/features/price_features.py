from __future__ import annotations

import numpy as np
import pandas as pd


def build_price_features(prices: pd.DataFrame) -> pd.DataFrame:
    required_columns = {"ticker", "date", "close", "volume"}
    missing = required_columns - set(prices.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    frame = prices.copy().sort_values(["ticker", "date"]).reset_index(drop=True)
    by_ticker = frame.groupby("ticker")

    # Original features
    frame["ret_1"] = by_ticker["close"].pct_change(1)
    frame["ret_5"] = by_ticker["close"].pct_change(5)
    frame["ret_10"] = by_ticker["close"].pct_change(10)
    frame["volatility_10"] = (
        by_ticker["close"]
        .pct_change()
        .rolling(10)
        .std()
        .reset_index(level=0, drop=True)
    )
    frame["volume_change_1"] = by_ticker["volume"].pct_change(1)
    frame["ma_5"] = by_ticker["close"].rolling(5).mean().reset_index(level=0, drop=True)
    frame["ma_20"] = (
        by_ticker["close"].rolling(20).mean().reset_index(level=0, drop=True)
    )
    frame["ma_ratio_5_20"] = frame["ma_5"] / frame["ma_20"]

    # --- Phase 12: 8 new volatility features ---
    has_hloc = {"high", "low", "open"}.issubset(set(prices.columns))

    # 1. Parkinson volatility (uses high/low range, 10-day rolling)
    if has_hloc:
        log_hl = np.log(frame["high"] / frame["low"])
        parkinson_daily = log_hl**2 / (4.0 * np.log(2.0))
        frame["parkinson_vol"] = (
            parkinson_daily.groupby(frame["ticker"])
            .rolling(10)
            .mean()
            .reset_index(level=0, drop=True)
            .pipe(np.sqrt)
        )
    else:
        frame["parkinson_vol"] = np.nan

    # 2. Garman-Klass volatility (uses OHLC, 10-day rolling)
    if has_hloc:
        log_hl2 = np.log(frame["high"] / frame["low"]) ** 2
        log_co2 = np.log(frame["close"] / frame["open"]) ** 2
        gk_daily = 0.5 * log_hl2 - (2.0 * np.log(2.0) - 1.0) * log_co2
        frame["garman_klass_vol"] = (
            gk_daily.groupby(frame["ticker"])
            .rolling(10)
            .mean()
            .reset_index(level=0, drop=True)
            .pipe(np.sqrt)
        )
    else:
        frame["garman_klass_vol"] = np.nan

    # 3. 5-day realized volatility (short-term vol for regime detection)
    frame["rv_5d"] = by_ticker["close"].pct_change().rolling(5).std().reset_index(
        level=0, drop=True
    ) * np.sqrt(252)

    # 4. Volatility ratio (short/long regime indicator)
    vol_20 = (
        by_ticker["close"]
        .pct_change()
        .rolling(20)
        .std()
        .reset_index(level=0, drop=True)
    )
    frame["vol_ratio"] = frame["volatility_10"] / vol_20.replace(0, np.nan)

    # 5. Squared return (instantaneous variance proxy)
    frame["squared_return"] = frame["ret_1"] ** 2

    # 6. Absolute return (robust volatility proxy)
    frame["abs_return"] = frame["ret_1"].abs()

    # 7. Jump indicator (|ret| > 3σ of rolling 20d vol)
    rolling_20_std = (
        by_ticker["close"]
        .pct_change()
        .rolling(20)
        .std()
        .reset_index(level=0, drop=True)
    )
    frame["jump_indicator"] = (frame["ret_1"].abs() > 3.0 * rolling_20_std).astype(
        np.float32
    )

    # 8. EWMA variance (exponentially weighted, span=10)
    frame["ewma_var"] = (
        by_ticker["close"]
        .pct_change()
        .pow(2)
        .ewm(span=10, adjust=False)
        .mean()
        .reset_index(level=0, drop=True)
    )

    # --- Phase 13: 3 HAR-RV autoregressive features ---
    # HAR-RV model (Corsi 2009): RV_t = β0 + β_d×RV_{t-1} + β_w×RV^w_{t-5} + β_m×RV^m_{t-22}
    # Base realized volatility using log returns (standard for HAR-RV)
    log_ret = np.log(frame["close"] / frame["close"].groupby(frame["ticker"]).shift(1))
    # 20-day rolling realized vol, annualized
    rv_base = (
        log_ret.groupby(frame["ticker"])
        .rolling(20)
        .std()
        .reset_index(level=0, drop=True)
        * np.sqrt(252)
    )

    # HAR-RV Feature 1: daily lag (t-1) — strongest predictor
    rv_lag1d = rv_base.groupby(frame["ticker"]).shift(1).reset_index(level=0, drop=True)
    # HAR-RV Feature 2: weekly average (t-1 to t-5) — medium-term regime
    rv_lag5d = (
        rv_base.groupby(frame["ticker"])
        .rolling(5)
        .mean()
        .reset_index(level=0, drop=True)
        .groupby(frame["ticker"])
        .shift(1)
        .reset_index(level=0, drop=True)
    )
    # HAR-RV Feature 3: monthly average (t-1 to t-22) — long-term regime
    rv_lag22d = (
        rv_base.groupby(frame["ticker"])
        .rolling(22)
        .mean()
        .reset_index(level=0, drop=True)
        .groupby(frame["ticker"])
        .shift(1)
        .reset_index(level=0, drop=True)
    )

    # Fill NaN from rolling windows using per-ticker medians
    for col_name, col_data in [("rv_lag1d", rv_lag1d), ("rv_lag5d", rv_lag5d), ("rv_lag22d", rv_lag22d)]:
        frame[col_name] = col_data.values
        ticker_medians = frame.groupby("ticker")[col_name].transform("median")
        frame[col_name] = frame[col_name].fillna(ticker_medians)

    return frame
