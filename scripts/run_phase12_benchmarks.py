"""Phase 12 Step 11: Volatility benchmarks (HA, GARCH, GJR-GARCH, HAR-RV)."""

from __future__ import annotations
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

ALL_30 = [
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
    "QCOM",
    "TXN",
    "AVGO",
    "MRVL",
    "KLAC",
    "CRM",
    "ADBE",
    "NOW",
    "SNOW",
    "DDOG",
    "NFLX",
    "UBER",
    "PYPL",
    "SNAP",
    "DELL",
    "AMAT",
    "LRCX",
    "IBM",
    "CSCO",
    "HPE",
]


def load_returns():
    """Load daily log returns for all 30 stocks."""
    frames = []
    for ticker in ALL_30:
        fp = Path(f"data/raw/prices/{ticker}_ohlcv.csv")
        if not fp.exists():
            continue
        df = pd.read_csv(fp, parse_dates=["date"])
        df = df.sort_values("date")
        df["log_ret"] = np.log(df["close"] / df["close"].shift(1))
        df["ticker"] = ticker
        frames.append(df[["ticker", "date", "log_ret", "close"]].dropna())
    return pd.concat(frames, ignore_index=True)


def realized_vol(returns, window=20):
    """Annualized realized volatility."""
    return returns.rolling(window).std() * np.sqrt(252)


def qlike(true_var, pred_var, eps=1e-8):
    """QLIKE loss."""
    valid = (true_var > eps) & (pred_var > eps)
    t = true_var[valid]
    p = pred_var[valid]
    return np.mean(t / p + np.log(p))


def r_squared(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return 1 - ss_res / max(ss_tot, 1e-8)


def benchmark_historical_average(returns_by_ticker, test_mask):
    """Historical Average: predict vol as rolling 60-day average of past vol."""
    preds, actuals = [], []
    for ticker, df in returns_by_ticker.items():
        rv = realized_vol(df["log_ret"], 20)
        ha_pred = rv.rolling(60).mean()
        test_df = df[test_mask[ticker]]
        test_rv = rv[test_mask[ticker]]
        test_ha = ha_pred[test_mask[ticker]]
        valid = test_rv.notna() & test_ha.notna()
        preds.extend(test_ha[valid].values)
        actuals.extend(test_rv[valid].values)
    return np.array(preds), np.array(actuals)


def benchmark_garch(returns_by_ticker, test_mask):
    """GARCH(1,1) benchmark."""
    try:
        from arch import arch_model
    except ImportError:
        return None, None

    preds, actuals = [], []
    for ticker, df in returns_by_ticker.items():
        try:
            train_df = df[~test_mask[ticker]]
            test_df = df[test_mask[ticker]]
            if len(train_df) < 252 or len(test_df) < 20:
                continue

            rets_pct = train_df["log_ret"].values * 100
            am = arch_model(
                rets_pct, vol="Garch", p=1, q=1, mean="Constant", dist="normal"
            )
            res = am.fit(disp="off", show_warning=False)

            # Forecast for each test day
            rv = realized_vol(df["log_ret"], 20)
            test_rv = rv[test_mask[ticker]].values

            # Use rolling 1-step forecast
            all_rets = df["log_ret"].values * 100
            train_end = (~test_mask[ticker]).sum()
            forecasts = []
            for i in range(len(test_df)):
                end = train_end + i
                if end < 252:
                    continue
                try:
                    am2 = arch_model(
                        all_rets[:end],
                        vol="Garch",
                        p=1,
                        q=1,
                        mean="Constant",
                        dist="normal",
                    )
                    res2 = am2.fit(disp="off", show_warning=False, last_obs=end)
                    fc = res2.forecast(horizon=1)
                    vol_forecast = (
                        np.sqrt(fc.variance.values[-1, 0]) / 100 * np.sqrt(252)
                    )
                    forecasts.append(vol_forecast)
                except Exception:
                    forecasts.append(np.nan)

            if len(forecasts) > 0:
                forecasts = np.array(forecasts)
                test_rv_aligned = test_rv[: len(forecasts)]
                valid = ~np.isnan(forecasts) & ~np.isnan(test_rv_aligned)
                preds.extend(forecasts[valid])
                actuals.extend(test_rv_aligned[valid])
        except Exception:
            continue

    if not preds:
        return None, None
    return np.array(preds), np.array(actuals)


def benchmark_har_rv(returns_by_ticker, test_mask):
    """HAR-RV (Heterogeneous Autoregressive) model for realized volatility."""
    preds, actuals = [], []
    for ticker, df in returns_by_ticker.items():
        rv = realized_vol(df["log_ret"], 20).values
        # HAR components: RV_d (1-day), RV_w (5-day avg), RV_m (22-day avg)
        rv_d = pd.Series(rv).shift(1).values
        rv_w = pd.Series(rv).rolling(5).mean().shift(1).values
        rv_m = pd.Series(rv).rolling(22).mean().shift(1).values

        test_idx = (
            test_mask[ticker].values
            if hasattr(test_mask[ticker], "values")
            else test_mask[ticker]
        )
        train_idx = ~test_idx

        # Build regression
        valid_train = (
            train_idx
            & ~np.isnan(rv)
            & ~np.isnan(rv_d)
            & ~np.isnan(rv_w)
            & ~np.isnan(rv_m)
        )
        if valid_train.sum() < 100:
            continue

        X_train = np.column_stack(
            [rv_d[valid_train], rv_w[valid_train], rv_m[valid_train]]
        )
        y_train = rv[valid_train]

        # OLS fit
        X_aug = np.column_stack([np.ones(len(X_train)), X_train])
        try:
            beta = np.linalg.lstsq(X_aug, y_train, rcond=None)[0]
        except Exception:
            continue

        # Predict on test
        valid_test = (
            test_idx
            & ~np.isnan(rv)
            & ~np.isnan(rv_d)
            & ~np.isnan(rv_w)
            & ~np.isnan(rv_m)
        )
        if valid_test.sum() == 0:
            continue

        X_test = np.column_stack([rv_d[valid_test], rv_w[valid_test], rv_m[valid_test]])
        y_test = rv[valid_test]

        X_test_aug = np.column_stack([np.ones(len(X_test)), X_test])
        y_pred = X_test_aug @ beta
        y_pred = np.clip(y_pred, 0.01, 5.0)

        preds.extend(y_pred)
        actuals.extend(y_test)

    return np.array(preds), np.array(actuals)


def main():
    print("=" * 60)
    print("PHASE 12 STEP 11: VOLATILITY BENCHMARK COMPARISON")
    print("=" * 60)

    rets = load_returns()
    print(f"Total returns: {len(rets)} rows, {rets['ticker'].nunique()} tickers")

    # Build per-ticker data and test mask
    returns_by_ticker = {}
    test_mask = {}
    for ticker in ALL_30:
        tdf = rets[rets["ticker"] == ticker].copy().reset_index(drop=True)
        if len(tdf) < 100:
            continue
        returns_by_ticker[ticker] = tdf
        tdf_dates = pd.to_datetime(tdf["date"])
        test_mask[ticker] = tdf_dates > "2023-12-31"

    results = {}

    # 1. Historical Average
    print("\n--- Historical Average ---")
    ha_pred, ha_actual = benchmark_historical_average(returns_by_ticker, test_mask)
    if len(ha_pred) > 0:
        ha_r2 = r_squared(ha_actual, ha_pred)
        ha_mae = np.mean(np.abs(ha_actual - ha_pred))
        ha_rmse = np.sqrt(np.mean((ha_actual - ha_pred) ** 2))
        print(
            f"  R²={ha_r2:.4f}, MAE={ha_mae:.4f}, RMSE={ha_rmse:.4f}, N={len(ha_pred)}"
        )
        results["HA"] = {"r2": ha_r2, "mae": ha_mae, "rmse": ha_rmse, "n": len(ha_pred)}

    # 2. HAR-RV
    print("\n--- HAR-RV ---")
    har_pred, har_actual = benchmark_har_rv(returns_by_ticker, test_mask)
    if len(har_pred) > 0:
        har_r2 = r_squared(har_actual, har_pred)
        har_mae = np.mean(np.abs(har_actual - har_pred))
        har_rmse = np.sqrt(np.mean((har_actual - har_pred) ** 2))
        print(
            f"  R²={har_r2:.4f}, MAE={har_mae:.4f}, RMSE={har_rmse:.4f}, N={len(har_pred)}"
        )
        results["HAR_RV"] = {
            "r2": har_r2,
            "mae": har_mae,
            "rmse": har_rmse,
            "n": len(har_pred),
        }

    # 3. GARCH(1,1) — sample of tickers to avoid long runtime
    print("\n--- GARCH(1,1) [sample of 5 tickers] ---")
    sample_tickers = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL"]
    sample_returns = {
        t: returns_by_ticker[t] for t in sample_tickers if t in returns_by_ticker
    }
    sample_mask = {t: test_mask[t] for t in sample_tickers if t in test_mask}
    garch_pred, garch_actual = benchmark_garch(sample_returns, sample_mask)
    if garch_pred is not None and len(garch_pred) > 0:
        garch_r2 = r_squared(garch_actual, garch_pred)
        garch_mae = np.mean(np.abs(garch_actual - garch_pred))
        garch_rmse = np.sqrt(np.mean((garch_actual - garch_pred) ** 2))
        print(
            f"  R²={garch_r2:.4f}, MAE={garch_mae:.4f}, RMSE={garch_rmse:.4f}, N={len(garch_pred)}"
        )
        results["GARCH"] = {
            "r2": garch_r2,
            "mae": garch_mae,
            "rmse": garch_rmse,
            "n": len(garch_pred),
        }
    else:
        print("  GARCH not available (install 'arch' package for this benchmark)")

    # 4. Phase 12 model results (from training)
    print("\n--- Phase 12 Multimodal (from training) ---")
    results["Phase12_Multimodal"] = {
        "r2": 0.7719,
        "mae": 0.0615,
        "rmse": 0.0958,
        "dir_auc": 0.5675,
        "n": 17204,
    }
    print(f"  R²=0.7719, MAE=0.0615, RMSE=0.0958")

    # 5. V2 baseline
    results["V2_Baseline"] = {"r2": 0.335, "note": "from Phase 11 results"}

    # Summary table
    print("\n" + "=" * 60)
    print("BENCHMARK COMPARISON SUMMARY")
    print("=" * 60)
    print(f"{'Model':<25} {'R²':>8} {'MAE':>8} {'RMSE':>8}")
    print("-" * 55)
    for name, m in results.items():
        r2 = m.get("r2", "N/A")
        mae = m.get("mae", "N/A")
        rmse = m.get("rmse", "N/A")
        r2_str = f"{r2:.4f}" if isinstance(r2, float) else r2
        mae_str = f"{mae:.4f}" if isinstance(mae, float) else mae
        rmse_str = f"{rmse:.4f}" if isinstance(rmse, float) else rmse
        print(f"  {name:<23} {r2_str:>8} {mae_str:>8} {rmse_str:>8}")

    # Save results
    save_path = Path("models/phase12_benchmark_results.json")
    # Convert numpy types for JSON serialization
    clean_results = {}
    for k, v in results.items():
        clean_results[k] = {
            kk: float(vv) if isinstance(vv, (np.floating, float)) else vv
            for kk, vv in v.items()
        }
    with open(save_path, "w") as f:
        json.dump(clean_results, f, indent=2, default=str)
    print(f"\nResults saved to {save_path}")

    print("\nPHASE 12 STEP 11 COMPLETE — Benchmarks computed")


if __name__ == "__main__":
    main()
