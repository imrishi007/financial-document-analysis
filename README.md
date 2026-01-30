# Financial Document Analysis System

A deep learning project for extracting insights and predicting risk levels from unstructured financial documents such as annual reports, earnings call transcripts, and financial news articles.

## Project Overview

**Problem Statement**: Manual analysis of financial documents is time-consuming and prone to human bias. This project aims to build a deep learning-based system that can automatically classify financial documents by sentiment, risk level, and market outlook.

**Objective**: Build an end-to-end deep learning pipeline that:
- Collects and curates a custom dataset of financial documents
- Preprocesses and extracts meaningful features from text
- Trains a deep learning model for multi-class classification
- Evaluates and analyzes model performance

## Project Structure

```
financial-document-analysis/
├── data/
│   ├── raw/                 # Original downloaded documents
│   ├── processed/           # Cleaned and preprocessed data
│   └── labeled/             # Manually labeled dataset
├── src/
│   ├── data_collection/     # Scripts for scraping and downloading
│   ├── preprocessing/       # Text cleaning and feature extraction
│   ├── models/              # Model architecture definitions
│   ├── training/            # Training scripts and utilities
│   └── evaluation/          # Evaluation and analysis scripts
├── notebooks/               # Jupyter notebooks for exploration
├── models/                  # Saved model checkpoints
├── reports/                 # Generated reports and figures
├── logs/                    # Training logs and metrics
├── scripts/                 # Utility scripts
├── requirements.txt         # Python dependencies
└── README.md
```

## Dataset

This project uses a custom-collected dataset consisting of:
- Annual reports (10-K filings) from SEC EDGAR
- Earnings call transcripts from public sources
- Financial news articles

**Target Labels**:
- Sentiment: Positive, Negative, Neutral
- Risk Level: Low, Medium, High
- Market Outlook: Bullish, Bearish, Neutral

Detailed dataset documentation is available in `data/README.md`.

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Git

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/financial-document-analysis.git
cd financial-document-analysis
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Instructions for running each phase of the project will be added as development progresses.

## Project Timeline

| Phase | Description | Duration |
|-------|-------------|----------|
| Phase 1 | Problem Identification and Proposal | Jan 12 - Jan 24, 2026 |
| Phase 2 | Dataset Design and Collection | Jan 25 - Feb 15, 2026 |
| Phase 3 | Data Preprocessing and Analysis | Feb 16 - Feb 29, 2026 |
| Phase 4 | Model Design and Implementation | Mar 01 - Mar 20, 2026 |
| Phase 5 | Model Evaluation and Optimization | Mar 21 - Apr 05, 2026 |
| Phase 6 | Documentation and Final Presentation | Apr 06 - Apr 25, 2026 |

## Technologies Used

- Python 3.8+
- PyTorch / TensorFlow
- Transformers (Hugging Face)
- NumPy, Pandas
- Matplotlib, Seaborn
- BeautifulSoup, Requests
- Jupyter Notebook

## Author

Rishi - 3rd Year, Computer Science

## License

This project is for educational purposes as part of the Deep Learning course.

## Acknowledgments

- SEC EDGAR for public financial filings
- Course instructors for project guidance
