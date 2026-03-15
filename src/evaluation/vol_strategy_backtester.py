"""Volatility-Based Trading Strategy Backtester — Phase 13.

Implements a market-neutral volatility mispricing strategy:
  - Long top-k highest predicted/historical vol ratio stocks
  - Short bottom-k lowest ratio stocks
  - Position sizing via MC Dropout uncertainty
  - Weekly rebalancing, 20-day holding horizon

Benchmarks against: Buy-and-Hold, Direction Strategy, Random Strategy.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


class VolatilityStrategyBacktester:

    def __init__(
        self,
        tickers: List[str],
        price_data: Dict[str, pd.DataFrame],
        vol_predictions: np.ndarray,
        vol_uncertainty: np.ndarray,
        dir_predictions: np.ndarray,
        sample_dates: np.ndarray,
        sample_tickers: np.ndarray,
        vol_threshold: float = 0.15,
        rebalance_freq: int = 5,
        holding_period: int = 20,
        top_k: int = 5,
        risk_free_rate: float = 0.05,
    ):
        self.tickers = tickers
        self.price_data = price_data
        self.vol_preds = vol_predictions
        self.vol_unc = vol_uncertainty
        self.dir_preds = dir_predictions
        self.dates = np.array([str(d)[:10] for d in sample_dates])
        self.sample_tickers = np.array(sample_tickers)
        self.vol_threshold = vol_threshold
        self.rebalance_freq = rebalance_freq
        self.holding_period = holding_period
        self.top_k = top_k
        self.rfr = risk_free_rate

        # Build signal lookup: (date, ticker) -> {pred_vol, uncertainty, dir_score}
        self._signals = {}
        for i in range(len(self.dates)):
            key = (self.dates[i], self.sample_tickers[i])
            self._signals[key] = {
                "pred_vol": float(self.vol_preds[i]),
                "uncertainty": float(self.vol_unc[i]),
                "dir_score": float(self.dir_preds[i]),
            }

        # Build price return lookup per ticker
        self._returns = {}
        self._all_dates = None
        self._build_returns()

    def _build_returns(self):
        """Build daily return series per ticker and a shared date index."""
        all_dfs = []
        for ticker in self.tickers:
            if ticker not in self.price_data:
                continue
            df = self.price_data[ticker].copy()
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date").set_index("date")
            rets = df["close"].pct_change()
            self._returns[ticker] = rets
            all_dfs.append(rets.rename(ticker))

        if all_dfs:
            combined = pd.concat(all_dfs, axis=1).sort_index()
            self._all_dates = combined.index
            self._return_df = combined
        else:
            self._all_dates = pd.DatetimeIndex([])
            self._return_df = pd.DataFrame()

    def compute_historical_avg_vol(self) -> Dict[str, float]:
        """Historical avg vol per ticker using train period (≤2022-12-31)."""
        avg_vols = {}
        for ticker in self.tickers:
            if ticker not in self._returns:
                continue
            train_rets = self._returns[ticker][
                self._returns[ticker].index <= "2022-12-31"
            ]
            rv = train_rets.rolling(20).std() * np.sqrt(252)
            avg_vols[ticker] = float(rv.mean()) if not rv.isna().all() else 0.2
        return avg_vols

    def _get_test_dates(self) -> pd.DatetimeIndex:
        """Return test-period trading dates (>2023-12-31)."""
        return self._all_dates[self._all_dates > "2023-12-31"]

    def run_vol_strategy(self, cost_bps: float = 5.0) -> pd.Series:
        """Vol mispricing strategy: long high-ratio, short low-ratio."""
        test_dates = self._get_test_dates()
        if len(test_dates) == 0:
            return pd.Series(dtype=float)

        hist_vol = self.compute_historical_avg_vol()
        cost = cost_bps / 10000.0
        portfolio_returns = []
        portfolio_dates = []

        prev_longs = set()
        prev_shorts = set()

        rebalance_dates = test_dates[:: self.rebalance_freq]

        for reb_date in rebalance_dates:
            date_str = str(reb_date.date())

            # Compute vol ratio for each ticker
            ratios = {}
            uncertainties = {}
            for ticker in self.tickers:
                key = (date_str, ticker)
                if key not in self._signals:
                    continue
                sig = self._signals[key]
                hv = hist_vol.get(ticker, 0.2)
                if hv > 0:
                    ratios[ticker] = sig["pred_vol"] / hv
                    uncertainties[ticker] = sig["uncertainty"]

            if len(ratios) < 2 * self.top_k:
                continue

            # Rank by vol ratio
            sorted_tickers = sorted(ratios.keys(), key=lambda t: ratios[t])
            longs = sorted_tickers[-self.top_k :]  # highest ratio
            shorts = sorted_tickers[: self.top_k]  # lowest ratio

            # Position sizing via uncertainty
            long_weights = {}
            for t in longs:
                long_weights[t] = 1.0 / (1.0 + uncertainties.get(t, 0.1))
            wsum = sum(long_weights.values())
            long_weights = {t: w / wsum for t, w in long_weights.items()}

            short_weights = {}
            for t in shorts:
                short_weights[t] = 1.0 / (1.0 + uncertainties.get(t, 0.1))
            wsum = sum(short_weights.values())
            short_weights = {t: w / wsum for t, w in short_weights.items()}

            # Turnover cost
            new_longs = set(longs)
            new_shorts = set(shorts)
            turnover = len(new_longs - prev_longs) + len(new_shorts - prev_shorts)
            turnover_cost = turnover * cost / max(len(longs) + len(shorts), 1)
            prev_longs = new_longs
            prev_shorts = new_shorts

            # Hold for rebalance_freq days
            hold_end = min(
                len(test_dates) - 1,
                np.searchsorted(test_dates, reb_date) + self.rebalance_freq,
            )
            hold_start = np.searchsorted(test_dates, reb_date)

            for day_idx in range(hold_start, hold_end):
                day = test_dates[day_idx]
                daily_ret = 0.0

                for t, w in long_weights.items():
                    if t in self._return_df.columns and day in self._return_df.index:
                        r = self._return_df.loc[day, t]
                        if not np.isnan(r):
                            daily_ret += w * r * 0.5  # 50% long

                for t, w in short_weights.items():
                    if t in self._return_df.columns and day in self._return_df.index:
                        r = self._return_df.loc[day, t]
                        if not np.isnan(r):
                            daily_ret -= w * r * 0.5  # 50% short

                # Apply turnover cost on rebalance day only
                if day_idx == hold_start:
                    daily_ret -= turnover_cost

                portfolio_returns.append(daily_ret)
                portfolio_dates.append(day)

        if not portfolio_returns:
            return pd.Series(dtype=float)

        # Deduplicate (take last value per date)
        ret_df = pd.DataFrame({"date": portfolio_dates, "ret": portfolio_returns})
        ret_df = ret_df.groupby("date")["ret"].last()
        return ret_df

    def run_direction_strategy(self, cost_bps: float = 5.0) -> pd.Series:
        """Long-short based on direction score (Phase 11 style)."""
        test_dates = self._get_test_dates()
        if len(test_dates) == 0:
            return pd.Series(dtype=float)

        cost = cost_bps / 10000.0
        portfolio_returns = []
        portfolio_dates = []
        prev_longs = set()
        prev_shorts = set()

        rebalance_dates = test_dates[:: self.rebalance_freq]

        for reb_date in rebalance_dates:
            date_str = str(reb_date.date())

            scores = {}
            for ticker in self.tickers:
                key = (date_str, ticker)
                if key in self._signals:
                    scores[ticker] = self._signals[key]["dir_score"]

            if len(scores) < 2 * self.top_k:
                continue

            sorted_tickers = sorted(scores.keys(), key=lambda t: scores[t])
            longs = sorted_tickers[-self.top_k :]
            shorts = sorted_tickers[: self.top_k]

            ew = 1.0 / self.top_k
            new_longs = set(longs)
            new_shorts = set(shorts)
            turnover = len(new_longs - prev_longs) + len(new_shorts - prev_shorts)
            turnover_cost = turnover * cost / max(len(longs) + len(shorts), 1)
            prev_longs = new_longs
            prev_shorts = new_shorts

            hold_end = min(
                len(test_dates) - 1,
                np.searchsorted(test_dates, reb_date) + self.rebalance_freq,
            )
            hold_start = np.searchsorted(test_dates, reb_date)

            for day_idx in range(hold_start, hold_end):
                day = test_dates[day_idx]
                daily_ret = 0.0

                for t in longs:
                    if t in self._return_df.columns and day in self._return_df.index:
                        r = self._return_df.loc[day, t]
                        if not np.isnan(r):
                            daily_ret += ew * r * 0.5

                for t in shorts:
                    if t in self._return_df.columns and day in self._return_df.index:
                        r = self._return_df.loc[day, t]
                        if not np.isnan(r):
                            daily_ret -= ew * r * 0.5

                if day_idx == hold_start:
                    daily_ret -= turnover_cost

                portfolio_returns.append(daily_ret)
                portfolio_dates.append(day)

        if not portfolio_returns:
            return pd.Series(dtype=float)

        ret_df = pd.DataFrame({"date": portfolio_dates, "ret": portfolio_returns})
        ret_df = ret_df.groupby("date")["ret"].last()
        return ret_df

    def run_buy_and_hold(self) -> pd.Series:
        """Equal-weight buy-and-hold all 30 stocks."""
        test_dates = self._get_test_dates()
        if len(test_dates) == 0:
            return pd.Series(dtype=float)

        available = [t for t in self.tickers if t in self._return_df.columns]
        if not available:
            return pd.Series(dtype=float)

        ew = 1.0 / len(available)
        daily_rets = []
        for day in test_dates:
            if day not in self._return_df.index:
                continue
            day_ret = 0.0
            for t in available:
                r = self._return_df.loc[day, t]
                if not np.isnan(r):
                    day_ret += ew * r
            daily_rets.append((day, day_ret))

        if not daily_rets:
            return pd.Series(dtype=float)
        df = pd.DataFrame(daily_rets, columns=["date", "ret"]).set_index("date")
        return df["ret"]

    def run_random_strategy(
        self, cost_bps: float = 5.0, n_sims: int = 100
    ) -> pd.Series:
        """Random long-short selection (averaged over n_sims runs)."""
        test_dates = self._get_test_dates()
        if len(test_dates) == 0:
            return pd.Series(dtype=float)

        cost = cost_bps / 10000.0
        all_rets = []

        rng = np.random.RandomState(42)
        rebalance_dates = test_dates[:: self.rebalance_freq]

        for _ in range(n_sims):
            portfolio_returns = []
            portfolio_dates = []

            for reb_date in rebalance_dates:
                date_str = str(reb_date.date())
                available = [t for t in self.tickers if (date_str, t) in self._signals]
                if len(available) < 2 * self.top_k:
                    continue

                chosen = rng.choice(available, 2 * self.top_k, replace=False)
                longs = chosen[: self.top_k]
                shorts = chosen[self.top_k :]
                ew = 1.0 / self.top_k

                hold_end = min(
                    len(test_dates) - 1,
                    np.searchsorted(test_dates, reb_date) + self.rebalance_freq,
                )
                hold_start = np.searchsorted(test_dates, reb_date)

                for day_idx in range(hold_start, hold_end):
                    day = test_dates[day_idx]
                    daily_ret = 0.0

                    for t in longs:
                        if (
                            t in self._return_df.columns
                            and day in self._return_df.index
                        ):
                            r = self._return_df.loc[day, t]
                            if not np.isnan(r):
                                daily_ret += ew * r * 0.5

                    for t in shorts:
                        if (
                            t in self._return_df.columns
                            and day in self._return_df.index
                        ):
                            r = self._return_df.loc[day, t]
                            if not np.isnan(r):
                                daily_ret -= ew * r * 0.5

                    if day_idx == hold_start:
                        daily_ret -= cost * 0.5

                    portfolio_returns.append(daily_ret)
                    portfolio_dates.append(day)

            if portfolio_returns:
                ret_s = (
                    pd.DataFrame({"date": portfolio_dates, "ret": portfolio_returns})
                    .groupby("date")["ret"]
                    .last()
                )
                all_rets.append(ret_s)

        if not all_rets:
            return pd.Series(dtype=float)

        combined = pd.concat(all_rets, axis=1)
        return combined.mean(axis=1)

    def compute_metrics(self, returns: pd.Series, strategy_name: str) -> Dict:
        """Compute full performance metrics for a return series."""
        if len(returns) == 0 or returns.isna().all():
            return {
                "strategy": strategy_name,
                "total_return": 0.0,
                "ann_return": 0.0,
                "ann_vol": 0.0,
                "sharpe": 0.0,
                "max_drawdown": 0.0,
                "win_rate": 0.0,
                "calmar": 0.0,
            }

        returns = returns.fillna(0.0)
        total_return = float((1 + returns).prod() - 1)
        n_days = len(returns)
        ann_return = float((1 + total_return) ** (252 / max(n_days, 1)) - 1)
        ann_vol = float(returns.std() * np.sqrt(252))
        sharpe = float((ann_return - self.rfr) / (ann_vol + 1e-8))

        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.cummax()
        drawdown = (cumulative - rolling_max) / rolling_max
        max_dd = float(drawdown.min())

        win_rate = float((returns > 0).mean())
        calmar = float(ann_return / (abs(max_dd) + 1e-8))

        return {
            "strategy": strategy_name,
            "total_return": total_return,
            "ann_return": ann_return,
            "ann_vol": ann_vol,
            "sharpe": sharpe,
            "max_drawdown": max_dd,
            "win_rate": win_rate,
            "calmar": calmar,
        }

    def run_full_backtest(
        self, cost_levels: List[float] = [0, 5, 10, 20]
    ) -> pd.DataFrame:
        """Run all strategies at all cost levels."""
        results = []

        for cost in cost_levels:
            vol_rets = self.run_vol_strategy(cost)
            results.append(self.compute_metrics(vol_rets, f"Vol Strategy ({cost}bp)"))

            dir_rets = self.run_direction_strategy(cost)
            results.append(
                self.compute_metrics(dir_rets, f"Direction Strategy ({cost}bp)")
            )

            rand_rets = self.run_random_strategy(cost)
            results.append(
                self.compute_metrics(rand_rets, f"Random Strategy ({cost}bp)")
            )

        bh_rets = self.run_buy_and_hold()
        results.append(self.compute_metrics(bh_rets, "Buy and Hold"))

        return pd.DataFrame(results)
