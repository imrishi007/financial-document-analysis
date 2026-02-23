# Phase 1: Raw Data Collection

## Goal
Collect all raw modalities required by the multimodal graph model:

1. SEC 10-K filings,
2. OHLCV market data,
3. financial news,
4. earnings surprise labels,
5. graph structure files.

## Configuration
Edit `configs/data_collection.yaml`:
- company universe,
- date range,
- source settings,
- output paths.

## Source Runners
SEC 10-K:
```bash
python -m src.data_collection.sec_10k_collector --mode coverage
python -m src.data_collection.sec_10k_collector --mode collect-missing
```

OHLCV:
```bash
python -m src.data_collection.price_collector
```

News:
```bash
python -m src.data_collection.news_collector
```

Earnings surprise:
```bash
python -m src.data_collection.earnings_collector
```

Graph files:
```bash
python -m src.data_collection.graph_builder
```

Run all sources:
```bash
python scripts/run_phase1_data_collection.py
```

## News API Key
If NewsAPI is used, set:
```bash
set NEWSAPI_KEY=your_key_here
```

If no key is set, the collector still runs with RSS source data.
