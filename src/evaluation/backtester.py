"""Vectorized backtester with realistic assumptions.

Key design decisions:
- Weekly rebalancing only (not daily) -- reduces turnover ~5x
- Transaction costs: test at 0bp, 5bp, 10bp, 20bp
- Long-short portfolio: top 2 predicted stocks LONG, bottom 2 SHORT
  (this hedges market beta -- makes signal purely about relative performance)
- Position sizing: equal weight within long and short legs
- Benchmark: equal-weight buy-and-hold all 10 stocks
"""

from __future__ import annotations

from collections import defaultdict
from typing import Optional

import numpy as np
import pandas as pd


def run_backtest(
    predictions_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    n_long: int = 2,
    n_short: int = 2,
    rebalance_freq: str = "W",
    cost_bps_list: list[float] | None = None,
    verbose: bool = True,
) -> dict:
    """Run a long-short backtest using model predictions.

    Parameters
    ----------
    predictions_df : DataFrame with columns [date, ticker, pred_prob]
        pred_prob is P(UP) from the model. Higher = more likely to go up.
    prices_df : DataFrame with columns [date, ticker, close]
    n_long : Number of stocks to go long (top predicted)
    n_short : Number of stocks to go short (bottom predicted)
    rebalance_freq : Rebalancing frequency ('W' for weekly, 'D' for daily)
    cost_bps_list : List of transaction cost levels to test (in basis points)
    verbose : Print progress

    Returns
    -------
    dict with backtest results
    """
    if cost_bps_list is None:
        cost_bps_list = [0, 5, 10, 20]

    # Prepare data
    predictions_df = predictions_df.copy()
    predictions_df["date"] = pd.to_datetime(predictions_df["date"])
    prices_df = prices_df.copy()
    prices_df["date"] = pd.to_datetime(prices_df["date"])

    # Get weekly rebalance dates (every Friday or last trading day of week)
    all_dates = sorted(predictions_df["date"].unique())
    date_series = pd.Series(all_dates)

    if rebalance_freq == "W":
        # Group by week and take the last date in each week
        week_groups = date_series.groupby(date_series.dt.isocalendar().week)
        rebalance_dates = [
            group.iloc[-1]
            for _, group in date_series.groupby(
                [date_series.dt.year, date_series.dt.isocalendar().week]
            )
        ]
    else:
        rebalance_dates = all_dates

    # Build price lookup
    price_lookup = {}
    for _, row in prices_df.iterrows():
        price_lookup[(row["date"], row["ticker"])] = row["close"]

    # Compute weekly returns for each stock
    all_tickers = sorted(predictions_df["ticker"].unique())

    # Build returns between consecutive dates
    returns_data = []
    for i in range(1, len(all_dates)):
        prev_date = all_dates[i - 1]
        curr_date = all_dates[i]
        for ticker in all_tickers:
            prev_price = price_lookup.get((prev_date, ticker))
            curr_price = price_lookup.get((curr_date, ticker))
            if prev_price is not None and curr_price is not None and prev_price > 0:
                ret = (curr_price - prev_price) / prev_price
                returns_data.append(
                    {
                        "date": curr_date,
                        "ticker": ticker,
                        "daily_return": ret,
                    }
                )

    returns_df = pd.DataFrame(returns_data)
    if len(returns_df) == 0:
        return {"error": "No valid return data"}

    # Run the strategy
    results_by_cost = {}

    for cost_bps in cost_bps_list:
        cost_frac = cost_bps / 10000.0

        portfolio_returns = []
        benchmark_returns = []
        positions_history = []
        turnover_values = []
        prev_longs = set()
        prev_shorts = set()

        for i, rebal_date in enumerate(rebalance_dates):
            # Get predictions for this date
            preds = predictions_df[predictions_df["date"] == rebal_date]
            if len(preds) < n_long + n_short:
                continue

            # Rank stocks by predicted probability
            ranked = preds.sort_values("pred_prob", ascending=False)
            longs = set(ranked.head(n_long)["ticker"].tolist())
            shorts = set(ranked.tail(n_short)["ticker"].tolist())

            # Compute turnover
            if i > 0:
                changed_longs = len(longs.symmetric_difference(prev_longs))
                changed_shorts = len(shorts.symmetric_difference(prev_shorts))
                turnover = (changed_longs + changed_shorts) / (2 * (n_long + n_short))
                turnover_values.append(turnover)

            # Get returns until next rebalance
            if i + 1 < len(rebalance_dates):
                next_date = rebalance_dates[i + 1]
            else:
                next_date = all_dates[-1]

            # Period returns
            period_mask = (returns_df["date"] > rebal_date) & (
                returns_df["date"] <= next_date
            )
            period_rets = returns_df[period_mask]

            if len(period_rets) == 0:
                prev_longs = longs
                prev_shorts = shorts
                continue

            # Compute portfolio return for this period
            long_ret = 0.0
            short_ret = 0.0
            bench_ret = 0.0

            for ticker in all_tickers:
                ticker_rets = period_rets[period_rets["ticker"] == ticker][
                    "daily_return"
                ]
                if len(ticker_rets) == 0:
                    continue
                cumulative = (1 + ticker_rets).prod() - 1

                # Benchmark: equal weight all stocks
                bench_ret += cumulative / len(all_tickers)

                # Long-short
                if ticker in longs:
                    long_ret += cumulative / n_long
                elif ticker in shorts:
                    short_ret += cumulative / n_short

            # Portfolio return = long - short - costs
            transaction_cost = (
                cost_frac * (turnover_values[-1] if turnover_values else 0) * 2
            )
            port_ret = (long_ret - short_ret) / 2 - transaction_cost

            portfolio_returns.append(
                {
                    "date": next_date,
                    "portfolio_return": port_ret,
                    "long_return": long_ret,
                    "short_return": short_ret,
                    "benchmark_return": bench_ret,
                    "longs": list(longs),
                    "shorts": list(shorts),
                }
            )

            prev_longs = longs
            prev_shorts = shorts

        if not portfolio_returns:
            results_by_cost[cost_bps] = {"error": "No valid periods"}
            continue

        result_df = pd.DataFrame(portfolio_returns)

        # Compute metrics
        port_rets = result_df["portfolio_return"].values
        bench_rets = result_df["benchmark_return"].values

        cumulative_port = (1 + port_rets).cumprod()
        cumulative_bench = (1 + bench_rets).cumprod()

        # Annualized metrics (assuming weekly rebalancing)
        periods_per_year = 52 if rebalance_freq == "W" else 252
        n_periods = len(port_rets)

        total_return = cumulative_port[-1] - 1
        bench_total_return = cumulative_bench[-1] - 1

        ann_return = (1 + total_return) ** (periods_per_year / max(n_periods, 1)) - 1
        ann_vol = np.std(port_rets) * np.sqrt(periods_per_year)
        sharpe = ann_return / (ann_vol + 1e-8)

        # Maximum drawdown
        peak = np.maximum.accumulate(cumulative_port)
        drawdown = (cumulative_port - peak) / peak
        max_drawdown = drawdown.min()

        # Win rate
        win_rate = np.mean(port_rets > 0) if len(port_rets) > 0 else 0

        # Average turnover
        avg_turnover = np.mean(turnover_values) if turnover_values else 0

        results_by_cost[cost_bps] = {
            "total_return": float(total_return),
            "benchmark_return": float(bench_total_return),
            "annualized_return": float(ann_return),
            "annualized_volatility": float(ann_vol),
            "sharpe_ratio": float(sharpe),
            "max_drawdown": float(max_drawdown),
            "win_rate": float(win_rate),
            "avg_turnover": float(avg_turnover),
            "n_periods": int(n_periods),
            "cumulative_portfolio": cumulative_port.tolist(),
            "cumulative_benchmark": cumulative_bench.tolist(),
            "period_returns": port_rets.tolist(),
            "dates": result_df["date"].dt.strftime("%Y-%m-%d").tolist(),
        }

    # Find break-even cost
    break_even_bps = None
    for cost in sorted(results_by_cost.keys()):
        r = results_by_cost[cost]
        if isinstance(r, dict) and "total_return" in r:
            if r["total_return"] <= 0:
                break_even_bps = cost
                break

    if verbose:
        print("=" * 70)
        print("BACKTEST RESULTS: Long-Short Weekly Strategy")
        print("=" * 70)
        print(f"  Long: top {n_long} stocks | Short: bottom {n_short} stocks")
        print(f"  Rebalance: {rebalance_freq}")
        print()

        for cost in sorted(results_by_cost.keys()):
            r = results_by_cost[cost]
            if isinstance(r, dict) and "total_return" in r:
                print(f"  Cost = {cost}bp:")
                print(f"    Total Return:  {r['total_return']:+.2%}")
                print(f"    Sharpe Ratio:  {r['sharpe_ratio']:.3f}")
                print(f"    Max Drawdown:  {r['max_drawdown']:.2%}")
                print(f"    Win Rate:      {r['win_rate']:.1%}")
                print(f"    Avg Turnover:  {r['avg_turnover']:.1%}")
                print()

        if break_even_bps is not None:
            print(f"  Break-even cost: ~{break_even_bps}bp")
        else:
            print(f"  Strategy profitable at all tested cost levels!")

    return {
        "results_by_cost": results_by_cost,
        "break_even_bps": break_even_bps,
        "n_long": n_long,
        "n_short": n_short,
        "rebalance_freq": rebalance_freq,
    }
