# Next Steps

1. Validate Phase 1 data completeness:
   - 10-K coverage by ticker-year,
   - OHLCV continuity by ticker-date,
   - earnings surprise coverage,
   - news volume by ticker and month.
2. Build aligned multimodal index at `(ticker, date)` granularity.
3. Define label generators for:
   - directional trend (`UP`/`DOWN`),
   - realized volatility (windowed standard deviation),
   - fundamental surprise (`BEAT`/`MISS`).
4. Start Phase 2 preprocessing and tokenization.
5. Implement encoder-specific dataset classes for text, price, and graph inputs.
6. Implement baseline multitask model training with time-based validation splits.
