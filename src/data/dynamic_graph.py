"""Dynamic graph construction from rolling return correlations.

Instead of a fixed knowledge-based graph, this module builds time-varying
graphs where edges reflect the actual statistical relationships between
companies' stock returns over a rolling window.

Each trading date gets its own edge structure: companies whose returns are
highly correlated (above a threshold) within the past ``window`` days are
connected. Self-loops are always included.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch

from src.data.price_dataset import load_price_csv_dir


def compute_rolling_correlation_graphs(
    prices_dir: str | Path = "data/raw/prices",
    tickers: Optional[list[str]] = None,
    window: int = 60,
    corr_threshold: float = 0.3,
    verbose: bool = False,
) -> dict[str, dict[str, torch.Tensor]]:
    """Compute per-date dynamic graphs from rolling return correlations.

    For each trading date, computes the N×N Pearson correlation matrix of
    daily log-returns over the past ``window`` trading days.  Edges are
    created between company pairs whose |correlation| ≥ ``corr_threshold``.
    Self-loops are always included.

    Parameters
    ----------
    prices_dir : Directory containing per-ticker price CSVs.
    tickers : Ordered list of tickers (defines node ordering).
    window : Rolling window size in trading days.
    corr_threshold : Minimum |correlation| to create an edge.
    verbose : Print progress information.

    Returns
    -------
    dict mapping date_str → {"edge_index": [2, E], "edge_weight": [E]}
    """
    if tickers is None:
        tickers = [
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
        ]

    N = len(tickers)

    # Load price data
    prices = load_price_csv_dir(prices_dir)
    prices["date"] = pd.to_datetime(prices["date"])

    # Compute daily log-returns per ticker
    returns_frames = {}
    for ticker in tickers:
        tk = (
            prices[prices["ticker"] == ticker]
            .sort_values("date")
            .set_index("date")["close"]
        )
        returns_frames[ticker] = np.log(tk / tk.shift(1))

    returns_df = pd.DataFrame(returns_frames).dropna()

    if verbose:
        print(
            f"  Returns matrix: {returns_df.shape[0]} dates × "
            f"{returns_df.shape[1]} tickers"
        )

    date_graphs: dict[str, dict[str, torch.Tensor]] = {}
    dates = returns_df.index.tolist()

    for i in range(window, len(dates)):
        date_str = str(dates[i].date())

        # Correlation matrix from returns in [i-window, i)
        window_returns = returns_df.iloc[i - window : i]
        corr = window_returns.corr().values  # [N, N]

        # Build edges from significant correlations (both directions)
        src_list, tgt_list, weight_list = [], [], []
        for r in range(N):
            for c in range(r + 1, N):  # upper triangle only
                corr_val = abs(corr[r, c])
                if corr_val >= corr_threshold and not np.isnan(corr_val):
                    src_list.extend([r, c])
                    tgt_list.extend([c, r])
                    weight_list.extend([corr_val, corr_val])

        # Always add self-loops
        for n in range(N):
            src_list.append(n)
            tgt_list.append(n)
            weight_list.append(1.0)

        edge_index = torch.tensor([src_list, tgt_list], dtype=torch.long)
        edge_weight = torch.tensor(weight_list, dtype=torch.float32)

        date_graphs[date_str] = {
            "edge_index": edge_index,
            "edge_weight": edge_weight,
        }

    if verbose:
        n_edges = [g["edge_index"].size(1) - N for g in date_graphs.values()]
        print(f"  Dynamic graphs: {len(date_graphs)} dates")
        print(
            f"  Correlation edges per date: "
            f"mean={np.mean(n_edges):.1f}, "
            f"min={np.min(n_edges)}, "
            f"max={np.max(n_edges)}"
        )

    return date_graphs
