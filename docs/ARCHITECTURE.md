# Model Architecture

## Stage 1: Multimodal Encoders
1. Text encoder
   - FinBERT for 10-K and news encoding.
   - Produces sentiment/risk-aware embeddings.
2. Price encoder
   - Custom Bi-LSTM stack over OHLCV sequences.
   - Produces momentum/trend embeddings.

## Stage 2: Graph Attention Layer
1. Company-level embeddings are graph nodes.
2. Static edges define supplier, competitor, and sector relationships.
3. Graph attention learns context-dependent influence weights between companies.

## Stage 3: Multi-Task Heads
1. Direction head
   - Binary classification (`UP`/`DOWN`).
2. Volatility head
   - Regression for realized volatility.
3. Fundamental surprise head
   - Binary classification (`BEAT`/`MISS`).

## Optimization
Total loss is a weighted sum:

`L = lambda_dir * L_dir + lambda_vol * L_vol + lambda_fund * L_fund`

Weighting strategy and task balancing will be tuned after baseline training.
