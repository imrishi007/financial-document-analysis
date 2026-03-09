"""V2 Backtester: Weekly rebalancing with 60-day predictions.

Key design decisions:
1. WEEKLY rebalancing (every 5 trading days) instead of daily
2. LONG-SHORT portfolio: long top-N, short bottom-N
3. Equal weight and confidence-weighted position sizing
4. Transaction costs: 0bp, 5bp, 10bp, 20bp per trade
5. Benchmark: equal-weight long-only all stocks (monthly rebalance)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


def weekly_long_short_backtest(
    predictions: pd.DataFrame,
    prices: pd.DataFrame,
    n_long: int = 2,
    n_short: int = 2,
    rebalance_every: int = 5,
    cost_bps: float = 0.0,
    confidence_weighted: bool = False,
    risk_free_annual: float = 0.05,
) -> dict:
    """Run weekly long-short backtest on model predictions.

    Parameters
    ----------
    predictions : DataFrame with columns [date, ticker, pred_prob]
    prices : DataFrame with columns [date, ticker, close]
    n_long, n_short : Number of stocks in each leg
    rebalance_every : Rebalance interval in trading days (5 = weekly)
    cost_bps : One-way transaction cost in basis points
    confidence_weighted : If True, weight by |pred_prob - 0.5|
    risk_free_annual : Annual risk-free rate for Sharpe ratio

    Returns
    -------
    dict with strategy metrics
    """
    preds = predictions.copy()
    prices_df = prices.copy()
    preds["date"] = pd.to_datetime(preds["date"])
    prices_df["date"] = pd.to_datetime(prices_df["date"])

    # Pivot prices to wide format
    price_wide = prices_df.pivot(index="date", columns="ticker", values="close").sort_index()
    tickers = sorted(price_wide.columns)
    trading_dates = sorted(price_wide.index)

    # Build prediction lookup
    pred_lookup = {}
    for _, row in preds.iterrows():
        pred_lookup[(row["date"], row["ticker"])] = row["pred_prob"]

    # Run backtest
    daily_returns = []
    positions_history = []
    turnover_list = []
    prev_long, prev_short = set(), set()
    portfolio_value = 1.0
    cost_rate = cost_bps / 10000.0

    rebalance_dates = trading_dates[::rebalance_every]

    for i, rebal_date in enumerate(rebalance_dates[:-1]):
        # Get predictions for this rebalance date
        date_preds = {}
        for t in tickers:
            key = (rebal_date, t)
            if key in pred_lookup:
                date_preds[t] = pred_lookup[key]
            else:
                # Try nearest date
                for offset in range(3):
                    for d in [rebal_date + pd.Timedelta(days=offset),
                              rebal_date - pd.Timedelta(days=offset)]:
                        if (d, t) in pred_lookup:
                            date_preds[t] = pred_lookup[(d, t)]
                            break
                    if t in date_preds:
                        break

        if len(date_preds) < n_long + n_short:
            continue

        # Rank by predicted UP probability
        sorted_tickers = sorted(date_preds.keys(), key=lambda t: date_preds[t], reverse=True)
        long_tickers = set(sorted_tickers[:n_long])
        short_tickers = set(sorted_tickers[-n_short:])

        # Compute turnover
        if prev_long or prev_short:
            long_changes = len(long_tickers ^ prev_long)
            short_changes = len(short_tickers ^ prev_short)
            turnover = (long_changes + short_changes) / (2 * (n_long + n_short))
        else:
            turnover = 1.0  # Initial portfolio setup
        turnover_list.append(turnover)

        # Position weights
        if confidence_weighted:
            long_weights = {}
            short_weights = {}
            for t in long_tickers:
                long_weights[t] = abs(date_preds[t] - 0.5)
            for t in short_tickers:
                short_weights[t] = abs(date_preds[t] - 0.5)
            # Normalize
            lw_sum = sum(long_weights.values()) or 1
            sw_sum = sum(short_weights.values()) or 1
            long_weights = {t: 0.5 * w / lw_sum for t, w in long_weights.items()}
            short_weights = {t: 0.5 * w / sw_sum for t, w in short_weights.items()}
        else:
            long_weights = {t: 0.5 / n_long for t in long_tickers}
            short_weights = {t: 0.5 / n_short for t in short_tickers}

        # Compute returns until next rebalance
        next_rebal_idx = min(i + 1, len(rebalance_dates) - 1)
        next_rebal_date = rebalance_dates[next_rebal_idx]

        # Get prices between rebalance dates
        mask = (price_wide.index >= rebal_date) & (price_wide.index < next_rebal_date)
        period_prices = price_wide.loc[mask]

        if len(period_prices) < 2:
            continue

        # Daily returns for the period
        period_returns = period_prices.pct_change().iloc[1:]

        for _, day_ret in period_returns.iterrows():
            long_ret = sum(long_weights.get(t, 0) * day_ret.get(t, 0) for t in long_tickers)
            short_ret = -sum(short_weights.get(t, 0) * day_ret.get(t, 0) for t in short_tickers)
            daily_ret = long_ret + short_ret
            daily_returns.append(daily_ret)

        # Transaction cost on turnover
        if turnover > 0:
            cost = cost_rate * turnover * 2  # Round-trip cost on turnover fraction
            portfolio_value *= (1 - cost)

        prev_long, prev_short = long_tickers, short_tickers
        positions_history.append({
            "date": str(rebal_date.date()),
            "long": sorted(long_tickers),
            "short": sorted(short_tickers),
            "turnover": round(turnover, 4),
        })

    # Apply daily returns to portfolio
    daily_returns = np.array(daily_returns)
    # Apply transaction costs proportionally
    n_rebalances = len(turnover_list)
    avg_turnover = np.mean(turnover_list) if turnover_list else 0
    daily_cost = cost_rate * avg_turnover * 2 / max(rebalance_every, 1)
    net_daily_returns = daily_returns - daily_cost

    # Metrics
    cumulative = np.cumprod(1 + net_daily_returns)
    total_return = cumulative[-1] - 1 if len(cumulative) > 0 else 0
    n_days = len(net_daily_returns)
    annual_factor = 252 / max(n_days, 1)
    annual_return = (1 + total_return) ** annual_factor - 1

    daily_rf = (1 + risk_free_annual) ** (1 / 252) - 1
    excess_returns = net_daily_returns - daily_rf
    sharpe = np.mean(excess_returns) / (np.std(excess_returns) + 1e-10) * np.sqrt(252)

    # Max drawdown
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min() if len(drawdown) > 0 else 0

    # Win rate (weekly)
    weekly_returns = []
    for i in range(0, len(net_daily_returns), 5):
        chunk = net_daily_returns[i:i + 5]
        if len(chunk) > 0:
            weekly_returns.append(np.prod(1 + chunk) - 1)
    win_rate = np.mean(np.array(weekly_returns) > 0) if weekly_returns else 0

    annual_turnover = avg_turnover * 252 / max(rebalance_every, 1)

    return {
        "total_return": round(float(total_return), 4),
        "annual_return": round(float(annual_return), 4),
        "sharpe": round(float(sharpe), 4),
        "max_drawdown": round(float(max_drawdown), 4),
        "win_rate": round(float(win_rate), 4),
        "n_trading_days": n_days,
        "n_rebalances": n_rebalances,
        "avg_turnover": round(float(avg_turnover), 4),
        "annual_turnover": round(float(annual_turnover), 4),
        "cost_bps": cost_bps,
        "confidence_weighted": confidence_weighted,
    }


def run_full_backtest(
    predictions: pd.DataFrame,
    prices: pd.DataFrame,
    cost_bps_list: list[float] = [0, 5, 10, 20],
    n_long: int = 2,
    n_short: int = 2,
    rebalance_every: int = 5,
    verbose: bool = True,
) -> dict:
    """Run backtest for all cost levels and both position sizing methods."""
    results = {"equal_weight": {}, "confidence_weighted": {}}

    for cw in [False, True]:
        key = "confidence_weighted" if cw else "equal_weight"
        for cost in cost_bps_list:
            r = weekly_long_short_backtest(
                predictions, prices,
                n_long=n_long, n_short=n_short,
                rebalance_every=rebalance_every,
                cost_bps=cost,
                confidence_weighted=cw,
            )
            results[key][str(cost)] = r

            if verbose:
                print(f"  {key}, {cost}bp: return={r['total_return']*100:.1f}%, "
                      f"sharpe={r['sharpe']:.3f}, winrate={r['win_rate']*100:.1f}%")

    # Find break-even cost
    for key in ["equal_weight", "confidence_weighted"]:
        ew_results = results[key]
        costs = sorted([float(c) for c in ew_results.keys()])
        break_even = 0
        for c in costs:
            if ew_results[str(int(c))]["sharpe"] > 0:
                break_even = c
        results[f"{key}_break_even_bps"] = break_even

    return results
