# Pre-Trained NLP Rationale

## Scope
Pre-trained NLP is used only in text branches:
- document model (10-K text)
- news model (article text)

The price encoder, graph attention layer, and multitask output heads are custom deep learning modules.

## What Is Reused
- tokenizer and encoder weights from `ProsusAI/finbert`
- contextual embeddings produced by the transformer backbone

## What Is Built In This Project
- attention pooling and classification head for document model
- temporal sequence model (BiGRU) over article embeddings for news model
- all loss definitions, training loops, evaluation logic, and fusion network

## Why Pre-Trained NLP Is Used
1. Financial language has domain-specific vocabulary and syntax.
2. Labelled task data is limited relative to transformer parameter count.
3. Transfer learning reduces training cost and improves convergence.
4. Fine-tuning enables adaptation to project-specific objectives (direction, volatility context, and fundamental signals).

## Why This Is Still a Deep Learning Project
- all prediction modules are neural networks
- two text modules are fine-tuned end-to-end
- non-text modules are custom neural architectures
- cross-company dependence is modeled with graph attention
- final outputs are produced by multitask neural heads

## Validation Requirements
- compare frozen vs fine-tuned text encoders
- compare FinBERT vs generic BERT backbone
- compare multimodal model vs ablations without graph attention
