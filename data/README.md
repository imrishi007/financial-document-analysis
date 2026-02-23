# Data Layout

## Raw Data (`data/raw`)
- `sec_10k/`: SEC 10-K filing text files.
- `prices/`: daily OHLCV files (`{ticker}_ohlcv.csv`, `tech10_ohlcv.csv`).
- `news/`: article-level news dataset (`news_articles.csv`).
- `earnings/`: earnings surprise dataset (`earnings_surprise.csv`).
- `graph/`: graph node and edge files (`tech10_nodes.csv`, `tech10_edges.csv`).

## Interim Data (`data/interim`)
- collection logs per source:
  - `sec_10k_collection_log.csv`
  - `price_collection_log.csv`
  - `news_collection_log.csv`
  - `earnings_collection_log.csv`
- SEC coverage report:
  - `sec_10k_coverage.csv`

## Processed and Targets
- `data/processed/`: cleaned datasets for modeling.
- `data/targets/`: derived training targets for directional trend, realized volatility, and fundamental surprise.

## Phase 1 Scope
Phase 1 only collects and validates raw source data.
Tokenization and feature engineering start in the next phase.
