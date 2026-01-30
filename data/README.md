# Dataset Documentation

## Overview

This directory contains all data related to the Financial Document Analysis project. The dataset is custom-collected as per project requirements.

## Directory Structure

```
data/
├── raw/                 # Original documents as downloaded
├── processed/           # Cleaned and preprocessed text files
├── labeled/             # CSV files with labels
└── README.md            # This file
```

## Data Sources

### Primary Sources (for custom dataset creation)

1. **SEC EDGAR Database**
   - URL: https://www.sec.gov/cgi-bin/browse-edgar
   - Contains: 10-K annual reports, 10-Q quarterly reports, 8-K current reports
   - Format: HTML, TXT
   - How to access: Free, no registration required
   - Best for: Corporate financial statements and risk disclosures

2. **Financial News APIs**
   - NewsAPI (https://newsapi.org/) - Free tier available, 100 requests/day
   - Alpha Vantage (https://www.alphavantage.co/) - Free API key required
   - Finnhub (https://finnhub.io/) - Free tier with rate limits

3. **Company Investor Relations Pages**
   - Direct access to earnings call transcripts
   - Quarterly and annual reports in PDF format
   - Presentations and shareholder letters

4. **Public Financial News Websites**
   - Reuters (reuters.com)
   - Bloomberg (bloomberg.com)
   - MarketWatch (marketwatch.com)
   - Yahoo Finance (finance.yahoo.com)

### Data Collection Guidelines

- Collect documents from at least 50 different companies
- Include multiple sectors (Technology, Finance, Healthcare, Energy, Consumer)
- Time range: 2020-2025 for relevance
- Target minimum: 500 documents for training

## Labeling Scheme

### Labels to Assign

| Label Category | Classes | Description |
|----------------|---------|-------------|
| Sentiment | Positive, Negative, Neutral | Overall tone of the document |
| Risk Level | Low, Medium, High | Assessed financial risk from content |
| Market Outlook | Bullish, Bearish, Neutral | Predicted market direction |

### Labeling Guidelines

1. Read the document summary or key sections
2. Identify key phrases indicating sentiment (growth, decline, risk, opportunity)
3. Assess risk based on language about debt, market conditions, regulatory concerns
4. Determine outlook based on forward-looking statements

## Data Collection Log

Maintain a log of all collected data in `collection_log.csv` with the following columns:
- document_id
- source
- company_name
- sector
- document_type
- collection_date
- file_path
- notes

## File Naming Convention

```
{company_ticker}_{document_type}_{year}_{sequence}.{extension}

Examples:
AAPL_10K_2024_001.txt
MSFT_earnings_2024Q3_001.txt
GOOGL_news_20240115_001.txt
```

## Data Statistics

To be updated as data collection progresses.

| Metric | Target | Current |
|--------|--------|---------|
| Total Documents | 500+ | 0 |
| Companies Covered | 50+ | 0 |
| Sectors Covered | 5+ | 0 |
| Labeled Documents | 500+ | 0 |
