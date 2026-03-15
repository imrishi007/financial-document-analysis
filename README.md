# Multimodal Financial Forecasting

> **Volatility forecasting and direction prediction for 30 S&P 500 tech stocks using price, SEC filings, sector graph, macro, and earnings surprise signals.**
>
> Final model: **Vol R² = 0.921** (HAR-RV benchmark: 0.947) | **Direction AUC = 0.591** | **Direction Sharpe = 0.620 at 5bp transaction costs**

---

## What This Project Does

This project builds a multimodal deep learning system that jointly forecasts **realized volatility** and **price direction** for 30 technology stocks over a 5-day horizon. The model fuses five heterogeneous information sources — price sequences, SEC 10-K filings, sector dependency graphs, macroeconomic indicators, and earnings surprises — through a gated fusion architecture with learnable modality weighting.

The project evolved significantly across 14 development phases. It began as a direction classifier for 10 stocks and pivoted in Phase 12 to volatility-primary forecasting after recognizing that volatility is more forecastable and more directly applicable to options/variance strategies.

---

## Evolution

| Phase | Focus | Vol R² | Dir AUC |
|-------|-------|--------|---------|
| 4–6 | Initial fusion (10 stocks, direction-only) | 0.440 | 0.545 |
| 11 | ListNet ranking loss, V2 architecture | 0.335 | 0.565 |
| 12 | Pivoted to vol-primary; QLIKE loss, 30 stocks | 0.772 | 0.568 |
| 13 | HAR-RV autoregressive features; 30 stocks | 0.867 | 0.585 |
| **14** | **HAR-RV skip connection bypassing CNN-BiLSTM** | **0.921** | **0.591** |
| HAR-RV | Econometric benchmark | 0.947 | — |

The jump from Phase 12 → 13 → 14 demonstrates systematic gap-closing against the HAR-RV econometric baseline. Phase 14 closes **95.7%** of the gap between the V2 baseline and HAR-RV.

---

## Architecture

```
Price OHLCV [B, 60, 21]
  ├─ CNN-BiLSTM ──────────────────────────────────► price_emb   [B, 256]
  └─ HAR-RV skip (rv_lag1d/5d/22d, last timestep) ► har_proj    [B, 32]  ← KEY P14 ADDITION

SEC 10-K text  ──► FinBERT ──► attention pool ──────► doc_emb    [B, 768]
Sector graph   ──► 2-layer GAT ─────────────────────► gat_emb    [B, 256]
Macro series   ──► MLP encoder ─────────────────────► macro_emb  [B, 32]
Earnings surp. ──────────────────────────────────────► surprise   [B, 5]

                   ┌─────────────────────────────────────┐
                   │  Gated Fusion Trunk                  │
                   │  gates = sigmoid(concat(4 modals))   │
                   │  trunk_in = [gated × 4 + har_proj]   │
                   │  = [B, 512 + 32] = [B, 544]          │
                   └─────────────────┬───────────────────┘
                                     │
                   ┌─────────────────┴───────────────────┐
                   │                                       │
              Vol head                              Dir head
          Softplus → σ̂ [B]                      logit → p [B]
       (QLIKE + ListNet loss)               (BCE + ListNet loss)
```

**Why the skip connection matters**: HAR-RV succeeds because lagged realized volatility directly predicts future volatility through a linear path. The CNN-BiLSTM compresses the 60-day sequence into 256 dimensions, losing some of this autoregressive signal. The skip connection provides a guaranteed direct path — the model can leverage the HAR-RV relationship without learning it through layers of convolutions.

---

## Results

### Volatility Forecasting

| Model | Vol R² | RMSE | Notes |
|-------|--------|------|-------|
| Historical Average | 0.348 | — | Naive baseline |
| GARCH(1,1) | ~0.60 | — | Univariate time-series |
| V2 Fusion Baseline | 0.335 | — | Phase 11 |
| Phase 12 | 0.772 | 0.096 | First QLIKE training |
| Phase 13 | 0.867 | 0.073 | +HAR-RV features |
| **Phase 14 (this model)** | **0.921** | **0.056** | +skip connection |
| HAR-RV | 0.947 | 0.045 | Econometric benchmark |

Phase 14 does not beat HAR-RV (gap: 0.026). The remaining gap reflects HAR-RV's irreducible advantage as a zero-compression linear model on strongly autoregressive data. The multimodal model's advantage lies in incorporating document, graph, and macro signals that HAR-RV cannot use.

### Trading Strategy

| Strategy | Sharpe | Ann. Return | Max DD |
|----------|--------|-------------|--------|
| Direction (0bp) | +0.697 | +12.9% | −11.4% |
| **Direction (5bp)** | **+0.620** | **+12.0%** | −11.6% |
| Direction (10bp) | +0.543 | +11.2% | −11.9% |
| Direction (20bp) | +0.392 | +9.5% | −12.4% |
| Buy and Hold | +0.652 | +14.2% | −18.3% |
| Vol strategy (0bp) | −0.763 | −6.4% | −26.3% |

The direction strategy is viable at realistic transaction costs (Sharpe > 0 at 20bp). The raw volatility long/short strategy does not generate positive returns — a finding that motivates options-based applications where the predicted vol level itself is the product, not a trading signal.

---

## Dataset

- **Universe**: 30 S&P 500 technology stocks (AAPL, MSFT, NVDA, GOOGL, AMD, META, AMZN, TSLA, ORCL, INTC + 20 others)
- **Time range**: January 2015 – February 2026
- **Training samples**: 73,793 (60-day rolling windows, 5-day forward targets)
- **Train / Val / Test split**: 2015–2021 / 2022–2023 / 2024–2026 (temporal, no leakage)
- **Modalities**:
  - Price OHLCV: daily open/high/low/close/volume → 21 engineered features including HAR-RV lags
  - SEC 10-K filings: 200+ company-year documents, encoded with `ProsusAI/finbert`
  - Sector graph: 30-node tech dependency graph with supply-chain and competitive edges
  - Macro: VIX, SPY, QQQ, TLT, DXY, interest rates (from `data/raw/macro/macro_prices.csv`)
  - Earnings surprise: EPS beat/miss magnitude (from `data/raw/earnings/earnings_surprise.csv`)

---

## Repository Structure

```
financial-document-analysis/
├── configs/
│   ├── experiment.yaml           # Hyperparameters, model config
│   └── data_collection.yaml      # Ticker list, date ranges
│
├── data/
│   ├── embeddings/
│   │   ├── phase13_fusion_embeddings.pt   # Pre-extracted embeddings [73793 samples]
│   │   └── phase14_har_rv_raw.pt          # HAR-RV skip features [73793, 3]
│   ├── raw/
│   │   ├── prices/               # 30 × OHLCV CSV files
│   │   ├── sec_10k/              # SEC 10-K filings (20 tickers, clean)
│   │   ├── macro/macro_prices.csv  # VIX, SPY, macro indices
│   │   ├── earnings/             # Earnings surprise data
│   │   └── graph/                # tech30 node/edge lists
│   └── targets/
│       ├── volatility_targets.csv
│       └── direction_labels.csv
│
├── src/
│   ├── models/
│   │   ├── fusion_model.py       # Phase14FusionModel (+ Phase12/13 for reference)
│   │   ├── price_model.py        # CNN-BiLSTM price encoder
│   │   ├── gat_model.py          # 2-layer GAT sector graph encoder
│   │   ├── document_model.py     # FinBERT document encoder
│   │   ├── macro_model.py        # Macro MLP encoder
│   │   └── losses.py             # QLIKELoss, ListNetRankingLoss, CombinedVolatilityLoss
│   ├── data/
│   │   ├── price_dataset.py      # 21-feature pipeline; HAR-RV at indices [18,19,20]
│   │   ├── fusion_dataset.py     # FusionEmbeddingDataset (loads from .pt files)
│   │   ├── preprocessing.py      # Z-score normalization, rolling windows
│   │   ├── target_builder.py     # Realized vol + direction label generation
│   │   └── macro_features.py     # Macro feature alignment
│   ├── evaluation/
│   │   ├── vol_strategy_backtester.py  # Direction + vol strategy backtests
│   │   ├── walk_forward.py             # Walk-forward R² validation
│   │   ├── metrics.py                  # QLIKE, R², RMSE, AUC
│   │   └── calibration.py
│   ├── features/
│   │   └── extract_embeddings.py  # Runs all encoders → saves .pt embedding files
│   └── train/
│       ├── train_price.py         # CNN-BiLSTM training
│       ├── train_graph.py         # GAT training
│       ├── train_document.py      # FinBERT fine-tuning
│       └── train_fusion.py        # Fusion model training loop
│
├── scripts/
│   ├── run_phase14_pipeline.py    # Full Phase 14 training pipeline (main entry point)
│   ├── run_phase14_backtest.py    # Backtest with VIX-adjusted signals
│   ├── create_phase14_notebook.py # Generates results notebook
│   ├── run_phase13_pipeline.py    # Phase 13 reference pipeline
│   └── run_phase12_pipeline.py    # Phase 12 reference pipeline
│
├── notebooks/
│   ├── 01_data_validation.ipynb       # Data quality checks
│   ├── 04_phase5_graph_results.ipynb  # GAT sector graph analysis
│   ├── 11_phase11_results.ipynb       # V2 architecture results
│   ├── 12_phase12_results.ipynb       # Vol-primary pivot
│   ├── 13_phase13_results.ipynb       # HAR-RV features
│   └── 14_phase14_results.ipynb       # Final results ← start here
│
├── models/
│   ├── document_model_best.pt         # FinBERT encoder (420 MB)
│   ├── phase14_fusion_best.pt         # Final Phase 14 model (6.5 MB)
│   ├── phase13_fusion_vol_primary.pt  # Phase 13 reference
│   ├── phase13_price_best.pt          # CNN-BiLSTM checkpoint
│   ├── phase13_gat_best.pt            # GAT checkpoint
│   ├── macro_v2_1_best.pt             # Macro encoder
│   ├── phase14_training_results.json
│   ├── phase14_benchmark_results.json
│   └── phase14_backtest_results.json
│
└── docs/
    ├── MODEL_DOCUMENTATION.tex        # Full technical write-up (gitignored)
    ├── ARCHITECTURE.md
    └── PROJECT_SCOPE.md
```

---

## Reproducing Results

### Prerequisites
```bash
git clone https://github.com/imrishi007/financial-document-analysis.git
cd financial-document-analysis
pip install -r requirements.txt
```

Requires CUDA GPU. Tested on NVIDIA RTX 2050 (4GB VRAM). Training uses AMP fp16.

### Option A — Load pre-trained results (fastest)

```python
import json
results = json.load(open("models/phase14_training_results.json"))
print(f"Vol R² = {results['vol_r2']:.4f}")
print(f"Dir AUC = {results['dir_auc']:.4f}")
# Open notebooks/14_phase14_results.ipynb for full analysis
```

### Option B — Re-run training from embeddings

```bash
# Embeddings already extracted; re-trains Phase 14 fusion model (~5 min on GPU)
python scripts/run_phase14_pipeline.py
```

### Option C — Full pipeline from raw data

```bash
# Step 1: Extract embeddings from all modalities (requires document_model_best.pt)
python -c "from src.features.extract_embeddings import run_phase13_extraction; run_phase13_extraction()"

# Step 2: Train Phase 14 fusion model
python scripts/run_phase14_pipeline.py

# Step 3: Run backtest
python scripts/run_phase14_backtest.py
```

---

## Key Technical Findings

1. **Volatility is forecastable, direction is not (much)**: Vol R²=0.921 vs direction AUC=0.591 — the model can almost perfectly rank stocks by upcoming volatility but cannot reliably predict direction.

2. **The CNN-BiLSTM bottleneck is real**: The HAR-RV skip connection improved R² from 0.867 → 0.921 by providing a direct linear path from lagged RV to the prediction, bypassing the 256-dim compression. Skip contribution metric = 0.115.

3. **QLIKE loss outperforms MSE for volatility**: The switch to QLIKE in Phase 12 drove the biggest single improvement (+0.437 R²). QLIKE penalizes forecast errors asymmetrically — underestimating volatility is worse than overestimating.

4. **Direction strategy survives realistic costs**: Sharpe drops from 0.697 → 0.620 → 0.543 at 0 / 5 / 10bp. Still positive at 20bp (Sharpe = 0.392), suggesting the directional signal is genuine.

5. **Multimodal > price-only for direction, not volatility**: Gate weights show price dominates (58.6%) after adding the skip connection. Documents and macro contribute 19.5% and 6.8% respectively — relevant for direction but less so for vol.

---

## Tech Stack

| Component | Library |
|-----------|---------|
| Deep learning | PyTorch 2.2+, AMP (fp16) |
| NLP / Document encoding | HuggingFace Transformers, `ProsusAI/finbert` |
| Graph neural network | PyTorch Geometric (GAT) |
| Financial data | yfinance, SEC EDGAR |
| Backtesting | Custom (src/evaluation/) |
| Analysis | NumPy, Pandas, SciPy, scikit-learn |

---

## Limitations

- **No options data**: The volatility trading strategy uses VIX × β as an implied vol proxy, which is too crude. Real mispricing detection requires individual stock option chains.
- **Survivorship bias**: Dataset includes only companies that remained in S&P 500 throughout the study period.
- **HAR-RV gap**: The 0.026 R² gap to HAR-RV reflects an irreducible advantage of zero-compression linear models on autoregressive data. The multimodal value is in cross-asset and sentiment signals that HAR-RV cannot incorporate.
- **Document encoding is static**: 10-K embeddings are computed once per year. Intra-year document updates are not captured.

---

## References

- Corsi (2009): HAR-RV model — *Journal of Financial Econometrics*
- Araci (2019): FinBERT — [`arxiv:1908.10063`](https://arxiv.org/abs/1908.10063)
- Veličković et al. (2018): Graph Attention Networks — [`arxiv:1710.10903`](https://arxiv.org/abs/1710.10903)
- Cao et al. (2015): ListNet ranking loss — *ICML 2015*
