"""Phase 14 Step 7 re-run: VIX-adjusted strategy with real VIX data."""

import json, sys, time
import numpy as np
import pandas as pd
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.seed import set_global_seed
from src.utils.gpu import setup_gpu
from src.models.fusion_model import Phase14FusionModel
from src.evaluation.vol_strategy_backtester import VolatilityStrategyBacktester

set_global_seed(42)
device = setup_gpu(verbose=False)
dev = str(device)

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

# Load price data
price_data = {}
for ticker in ALL_30:
    csv_path = Path(f"data/raw/prices/{ticker}_ohlcv.csv")
    if csv_path.exists():
        price_data[ticker] = pd.read_csv(csv_path, parse_dates=["date"])
print(f"Loaded {len(price_data)} tickers")

# Load embeddings + HAR-RV
emb = torch.load("data/embeddings/phase13_fusion_embeddings.pt", weights_only=False)
har_rv = torch.load("data/embeddings/phase14_har_rv_raw.pt", weights_only=False)
sample_tickers = np.array(emb["tickers"])
sample_dates = np.array(emb["dates"])

# Load Phase 14 model
model = Phase14FusionModel(
    price_dim=256,
    har_rv_dim=3,
    har_proj_dim=32,
    gat_dim=256,
    doc_dim=768,
    macro_dim=32,
    surprise_dim=5,
    proj_dim=128,
    hidden_dim=256,
    dropout=0.3,
    trunk_dropout=0.3,
    mc_dropout=True,
    lambda_vol=0.85,
    lambda_dir=0.15,
).to(dev)
ckpt = torch.load("models/phase14_fusion_best.pt", weights_only=False, map_location=dev)
model.load_state_dict(ckpt["model_state_dict"])

# Test set
dates_pd = pd.to_datetime(pd.Series(sample_dates))
test_mask = (dates_pd > "2023-12-31").values
test_idx = np.where(test_mask)[0]
print(f"Test samples: {len(test_idx)}")

# MC Dropout predictions
BATCH = 4096
N_MC = 30
all_vol_mean, all_vol_std, all_dir_mean = [], [], []
model.train()
with torch.no_grad():
    for start in range(0, len(test_idx), BATCH):
        end = min(start + BATCH, len(test_idx))
        idx = test_idx[start:end]
        p_e = emb["price_emb"][idx].to(dev)
        h_r = har_rv[idx].to(dev)
        g_e = emb["gat_emb"][idx].to(dev)
        d_e = emb["doc_emb"][idx].to(dev)
        m_e = emb["macro_emb"][idx].to(dev)
        s = emb["surprise_feat"][idx].to(dev)
        mk = emb["modality_mask"][idx].to(dev)
        mc_out = model.predict_with_uncertainty(
            p_e, h_r, g_e, d_e, m_e, s, mk, n_samples=N_MC
        )
        all_vol_mean.append(mc_out["vol_mean"].cpu().numpy())
        all_vol_std.append(mc_out["vol_std"].cpu().numpy())
        dir_probs = torch.softmax(mc_out["dir_mean"], dim=1)[:, 1]
        all_dir_mean.append(dir_probs.cpu().numpy())

vol_predictions = np.concatenate(all_vol_mean)
vol_uncertainty = np.concatenate(all_vol_std)
dir_predictions = np.concatenate(all_dir_mean)
test_dates_arr = sample_dates[test_idx]
test_tickers_arr = sample_tickers[test_idx]

# Create backtester
backtester = VolatilityStrategyBacktester(
    tickers=ALL_30,
    price_data=price_data,
    vol_predictions=vol_predictions,
    vol_uncertainty=vol_uncertainty,
    dir_predictions=dir_predictions,
    sample_dates=test_dates_arr,
    sample_tickers=test_tickers_arr,
    vol_threshold=0.15,
    rebalance_freq=5,
    holding_period=20,
    top_k=5,
    risk_free_rate=0.05,
)

# Load VIX + SPY from macro_prices.csv
macro_df = pd.read_csv("data/raw/macro/macro_prices.csv", parse_dates=["date"])
macro_df = macro_df.sort_values("date").set_index("date")
vix_series = macro_df["^VIX"]
spy_prices = macro_df["SPY"]
spy_rets = spy_prices.pct_change()
print(
    f"VIX data: {len(vix_series)} rows, range: {vix_series.min():.1f}-{vix_series.max():.1f}"
)
print(f"SPY data: {len(spy_prices)} rows")

# Compute beta for each stock vs SPY
beta_dict = {}
for ticker in ALL_30:
    if ticker not in price_data:
        continue
    df = price_data[ticker].copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").set_index("date")
    stock_rets = df["close"].pct_change()
    common = stock_rets.index.intersection(spy_rets.index)
    if len(common) < 60:
        beta_dict[ticker] = 1.0
        continue
    sr = stock_rets[common].values[-252:]
    mr = spy_rets[common].values[-252:]
    valid = ~(np.isnan(sr) | np.isnan(mr))
    sr, mr = sr[valid], mr[valid]
    if len(sr) < 60:
        beta_dict[ticker] = 1.0
    else:
        cov_val = np.cov(sr, mr)[0, 1]
        var_m = np.var(mr)
        beta = np.clip(cov_val / (var_m + 1e-8), 0.3, 3.0)
        beta_dict[ticker] = float(beta)

print("\nBetas:")
for t in sorted(beta_dict.keys()):
    print(f"  {t}: {beta_dict[t]:.2f}")

# Standard strategies
all_results = []
std_results = backtester.run_full_backtest(cost_levels=[0, 5, 10, 20])
all_results.extend(std_results.to_dict(orient="records"))

# VIX-adjusted strategy
test_dates_bt = backtester._get_test_dates()
vix_adj_metrics = {}

for cost_bps in [0, 5, 10, 20]:
    cost = cost_bps / 10000.0
    portfolio_returns = []
    portfolio_dates = []
    prev_longs = set()
    prev_shorts = set()
    rebalance_dates = test_dates_bt[:: backtester.rebalance_freq]

    for reb_date in rebalance_dates:
        date_str = str(reb_date.date())
        signals = {}
        uncertainties = {}
        for ticker in ALL_30:
            key = (date_str, ticker)
            if key not in backtester._signals:
                continue
            sig = backtester._signals[key]
            try:
                vix_val = vix_series.loc[:reb_date].iloc[-1] / 100.0
            except (KeyError, IndexError):
                vix_val = 0.20
            beta = beta_dict.get(ticker, 1.0)
            implied_vol = vix_val * beta
            mispricing = sig["pred_vol"] - implied_vol
            signals[ticker] = mispricing
            uncertainties[ticker] = sig["uncertainty"]

        if len(signals) < 2 * backtester.top_k:
            continue

        sorted_tickers = sorted(signals.keys(), key=lambda t: signals[t])
        longs = sorted_tickers[-backtester.top_k :]
        shorts = sorted_tickers[: backtester.top_k]

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

        new_longs = set(longs)
        new_shorts = set(shorts)
        turnover = len(new_longs - prev_longs) + len(new_shorts - prev_shorts)
        turnover_cost = turnover * cost / max(len(longs) + len(shorts), 1)
        prev_longs = new_longs
        prev_shorts = new_shorts

        hold_end = min(
            len(test_dates_bt) - 1,
            np.searchsorted(test_dates_bt, reb_date) + backtester.rebalance_freq,
        )
        hold_start = np.searchsorted(test_dates_bt, reb_date)

        for day_idx in range(hold_start, hold_end):
            day = test_dates_bt[day_idx]
            daily_ret = 0.0
            for t, w in long_weights.items():
                if (
                    t in backtester._return_df.columns
                    and day in backtester._return_df.index
                ):
                    r = backtester._return_df.loc[day, t]
                    if not np.isnan(r):
                        daily_ret += w * r * 0.5
            for t, w in short_weights.items():
                if (
                    t in backtester._return_df.columns
                    and day in backtester._return_df.index
                ):
                    r = backtester._return_df.loc[day, t]
                    if not np.isnan(r):
                        daily_ret -= w * r * 0.5
            if day_idx == hold_start:
                daily_ret -= turnover_cost
            portfolio_returns.append(daily_ret)
            portfolio_dates.append(day)

    if portfolio_returns:
        ret_df = pd.DataFrame({"date": portfolio_dates, "ret": portfolio_returns})
        ret_df = ret_df.groupby("date")["ret"].last()
        metrics = backtester.compute_metrics(ret_df, f"Vol VIX-adj ({cost_bps}bp)")
        all_results.append(metrics)
        vix_adj_metrics[cost_bps] = metrics
        sh = metrics["sharpe"]
        ar = metrics["ann_return"]
        dd = metrics["max_drawdown"]
        print(
            f"  VIX-adj ({cost_bps}bp): Sharpe={sh:.3f}, Return={ar:.2%}, MaxDD={dd:.2%}"
        )

# Print complete table
print()
print("=" * 100)
print("COMPLETE BACKTEST RESULTS (with real VIX data)")
print("=" * 100)
print(
    f"{'Strategy':<40} | {'Sharpe':>8} | {'Return':>8} | {'MaxDD':>8} | {'WinRate':>8}"
)
print("-" * 100)
for r in all_results:
    name = r["strategy"]
    sh = r["sharpe"]
    ar = r["ann_return"]
    dd = r["max_drawdown"]
    wr = r["win_rate"]
    print(f"{name:<40} | {sh:>+8.3f} | {ar:>+7.2%} | {dd:>+7.2%} | {wr:>7.2%}")
print("=" * 100)

# Key findings
vol_0 = next((r for r in all_results if r["strategy"] == "Vol Strategy (0bp)"), None)
vix_0 = vix_adj_metrics.get(0)
vix_5 = vix_adj_metrics.get(5)
dir_0 = next(
    (r for r in all_results if r["strategy"] == "Direction Strategy (0bp)"), None
)
dir_5 = next(
    (r for r in all_results if r["strategy"] == "Direction Strategy (5bp)"), None
)
dir_10 = next(
    (r for r in all_results if r["strategy"] == "Direction Strategy (10bp)"), None
)
dir_20 = next(
    (r for r in all_results if r["strategy"] == "Direction Strategy (20bp)"), None
)

print("\nKEY FINDINGS:")
if vol_0 and vix_0:
    improved = vix_0["sharpe"] > vol_0["sharpe"]
    print(
        f"  1. VIX-adjusted vs raw vol: Sharpe {vix_0['sharpe']:.3f} vs {vol_0['sharpe']:.3f} -> {'IMPROVED' if improved else 'NOT IMPROVED'}"
    )
if vix_5:
    print(
        f"  2. VIX-adjusted at 5bp: Sharpe={vix_5['sharpe']:.3f} -> {'POSITIVE' if vix_5['sharpe'] > 0 else 'NEGATIVE'}"
    )
if dir_5:
    print(
        f"  3. Direction at 5bp: Sharpe={dir_5['sharpe']:.3f} -> {'VIABLE' if dir_5['sharpe'] > 0 else 'NOT VIABLE'}"
    )
if dir_10:
    print(
        f"  4. Direction at 10bp: Sharpe={dir_10['sharpe']:.3f} -> {'VIABLE' if dir_10['sharpe'] > 0 else 'NOT VIABLE'}"
    )
if dir_20:
    print(
        f"  5. Direction at 20bp: Sharpe={dir_20['sharpe']:.3f} -> {'VIABLE' if dir_20['sharpe'] > 0 else 'NOT VIABLE'}"
    )

# Save
Path("models/phase14_backtest_results.json").write_text(
    json.dumps(all_results, indent=2)
)

# Save daily returns for main strategies
vol_rets_0 = backtester.run_vol_strategy(0)
dir_rets_0 = backtester.run_direction_strategy(0)
bh_rets = backtester.run_buy_and_hold()
returns_df = pd.DataFrame(
    {
        "vol_strategy_0bp": vol_rets_0,
        "dir_strategy_0bp": dir_rets_0,
        "buy_and_hold": bh_rets,
    }
)
returns_df.to_csv("models/phase14_backtest_returns.csv")

print("\nSaved: models/phase14_backtest_results.json")
print("Saved: models/phase14_backtest_returns.csv")
print("PHASE 14 STEP 7 COMPLETE — All strategies backtested (real VIX data)")
