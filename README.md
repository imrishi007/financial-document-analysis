# Financial Direction Forecasting

A multimodal deep learning system for predicting stock price direction using corporate filings, financial news, and market data.

## Overview

This project addresses the challenge of short-term stock direction prediction by fusing information from heterogeneous data sources. Given information available at time $t$, the system predicts whether a stock will move UP or DOWN after a configurable horizon $H$ (default: 5 trading days).

The approach combines three complementary signals:
- **Regulatory disclosures**: SEC 10-K annual reports containing business outlook, risk factors, and financial statements
- **Media coverage**: Time-stamped financial news articles reflecting market sentiment and events
- **Market microstructure**: Historical price and volume patterns capturing technical dynamics

A four-model ensemble architecture learns specialized representations for each modality before fusing them for final predictions.

## Research Question

**Can multimodal deep learning improve directional forecasting beyond single-modality baselines?**

For ticker $i$ at time $t$, we define:

$$r(t, H) = \frac{\text{Close}(t + H)}{\text{Close}(t)} - 1$$

Binary target: 
- $\text{UP}$ if $r(t, H) > 0$
- $\text{DOWN}$ otherwise

## Architecture

The system employs a staged ensemble approach:

1. **Document Model**: FinBERT encoder with attention pooling processes 10-K filings to extract long-term strategic signals.

2. **News Model**: FinBERT encoder with bidirectional GRU aggregates temporal sequences of news articles to capture evolving sentiment.

3. **Price Model**: 1D-CNN feature extractor with bidirectional LSTM models local patterns and momentum in OHLCV sequences.

4. **Fusion Model**: Gated multi-layer perceptron combines embeddings from all three base models with learned importance weighting.

Each base model is trained independently before fusion to enable modular evaluation and interpretability.

## Dataset

**Scope**: 10 technology sector companies (AAPL, MSFT, GOOGL, AMZN, NVDA, META, TSLA, AMD, INTC, ORCL)

**Time Range**: January 2016 - February 2026

**Data Sources**:
- SEC EDGAR: 97 annual 10-K filings
- Yahoo Finance: 25,432 daily OHLCV records
- News APIs: Timestamped articles from financial media outlets

**Target Labels**: 25,432 binary direction labels (51.7% UP, 48.3% DOWN)

The dataset exhibits near-perfect class balance and is split temporally to prevent look-ahead bias:
- Training: 2016-2022
- Validation: 2023-2024  
- Testing: 2025-2026

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

Clone the repository and install dependencies:

```bash
git clone https://github.com/imrishi007/financial-document-analysis.git
cd financial-document-analysis
python -m venv .venv
.venv\Scripts\activate  # On Windows
pip install -r requirements.txt
```

## Usage

### Data Collection

Collect all data sources automatically:

```bash
python scripts/run_phase1_data_collection.py
```

Collect specific modalities:

```bash
python scripts/run_phase1_data_collection.py --only sec
python scripts/run_phase1_data_collection.py --only price
python scripts/run_phase1_data_collection.py --only news
```

### Target Label Generation

Generate binary direction labels from price data:

```bash
python scripts/build_target_dataset.py
```

Output: `data/targets/direction_labels.csv`

### Model Training

Train individual base models:

```bash
python -m src.train.train_document_model
python -m src.train.train_news_model
python -m src.train.train_price_model
```

Train fusion model:

```bash
python -m src.train.train_fusion_model
```

### Evaluation

Evaluate model performance on test set:

```bash
python -m src.evaluation.evaluate_model --model fusion --split test
```

## Expected Outcomes

1. **Performance Metrics**: Accuracy, precision, recall, F1-score for binary classification with comparison against single-modality baselines.

2. **Ablation Analysis**: Quantification of each modality's contribution to final predictions through systematic removal experiments.

3. **Temporal Robustness**: Evaluation across different market regimes (bull vs. bear markets) and volatility levels.

4. **Interpretability**: Attention weight visualization showing which document sections and news events drive predictions.

5. **Financial Validation**: Backtested returns of model-guided trading strategy versus buy-and-hold benchmark.

## Project Structure

```
financial-document-analysis/
├── configs/                  # Configuration files for data collection and experiments
├── data/                     # All datasets (raw, processed, targets)
│   ├── raw/                 # Original downloaded data
│   ├── processed/           # Parsed and normalized data
│   ├── targets/             # Binary direction labels
│   └── interim/             # Collection logs and metadata
├── docs/                     # Technical documentation
├── src/                      # Source code
│   ├── data_collection/     # Automated data fetching modules
│   ├── data/                # Dataset builders and loaders
│   ├── models/              # Neural network architectures
│   ├── train/               # Training loops and utilities
│   └── evaluation/          # Metrics and analysis
├── scripts/                  # Standalone execution scripts
├── models/                   # Saved model checkpoints
├── logs/                     # Training logs and metrics
└── requirements.txt          # Python dependencies
```

## Technical Stack

- **Deep Learning**: PyTorch 2.2+
- **NLP**: Transformers (Hugging Face), FinBERT
- **Data Processing**: Pandas, NumPy
- **Financial Data**: yfinance, SEC EDGAR API
- **Visualization**: Matplotlib, Seaborn

## Limitations

- **Sector Specificity**: Current dataset limited to technology sector; cross-sector generalization untested.
- **Survivorship Bias**: Only includes companies that remained publicly traded throughout the study period.
- **Short Horizon**: Optimized for 5-day predictions; longer horizons may require architectural modifications.
- **No Event Detection**: Does not explicitly model discrete corporate events (M&A, product launches).

## References

- FinBERT: [Araci (2019)](https://arxiv.org/abs/1908.10063) - Financial sentiment analysis with BERT
- SEC EDGAR: [U.S. Securities and Exchange Commission](https://www.sec.gov/edgar)

## License

This project is released under the MIT License for academic and research purposes. See LICENSE file for details.