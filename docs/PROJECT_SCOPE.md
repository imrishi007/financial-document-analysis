# Project Scope

## Problem Statement
Model individual stock behavior while accounting for inter-company dependency across a 10-company technology graph.

## Inputs
1. 10-K filings and financial news text.
2. Daily OHLCV price sequences.
3. Static company relationship graph edges.

## Multi-Task Outputs
1. Directional trend (`UP`/`DOWN`) for 1-day to 5-day horizons.
2. Realized volatility (regression).
3. Fundamental surprise (`BEAT`/`MISS`) based on earnings outcomes.

## Modeling Constraints
1. Deep learning only for predictive models.
2. Pre-trained NLP is allowed only with explicit methodological justification.
3. Price and graph modules must be custom deep learning architectures.

## Current Phase Boundary
Phase 1 includes only raw data collection and quality checks for all modalities.
