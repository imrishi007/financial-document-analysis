from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import yfinance as yf


@dataclass
class PriceRequest:
    ticker: str
    start_date: str
    end_date: str
    interval: str = "1d"


def download_price_history(request: PriceRequest) -> pd.DataFrame:
    frame = yf.download(
        tickers=request.ticker,
        start=request.start_date,
        end=request.end_date,
        interval=request.interval,
        auto_adjust=False,
        progress=False,
    )

    if frame.empty:
        raise ValueError(f"No price data returned for {request.ticker}.")

    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = [str(price).lower().replace(" ", "_") for price, _ in frame.columns]
    else:
        frame.columns = [str(column).lower().replace(" ", "_") for column in frame.columns]

    frame = frame.reset_index()
    frame.columns = [str(column).lower().replace(" ", "_") for column in frame.columns]

    if "adj_close" not in frame.columns:
        frame["adj_close"] = frame["close"]

    frame["ticker"] = request.ticker
    frame["date"] = pd.to_datetime(frame["date"]).dt.date
    return frame[["ticker", "date", "open", "high", "low", "close", "adj_close", "volume"]]
