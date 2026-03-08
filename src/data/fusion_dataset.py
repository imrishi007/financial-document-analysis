"""Fusion dataset: loads pre-extracted modality embeddings from disk.

The embedding file is produced by ``src.features.extract_embeddings``.
Each sample is a set of modality embeddings + availability mask + targets,
all aligned by (ticker, date).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset


class FusionEmbeddingDataset(Dataset):
    """Lightweight dataset over pre-extracted embeddings.

    Parameters
    ----------
    embeddings_path : path to ``fusion_embeddings.pt``
    """

    # Modality names in mask order
    MODALITIES = ["price", "gat", "doc", "news", "surprise"]

    def __init__(self, embeddings_path: str | Path) -> None:
        data = torch.load(embeddings_path, weights_only=False, map_location="cpu")

        self.price_emb: torch.Tensor = data["price_emb"]  # [N, 256]
        self.gat_emb: torch.Tensor = data["gat_emb"]  # [N, 256]
        self.doc_emb: torch.Tensor = data["doc_emb"]  # [N, 768]
        self.news_emb: torch.Tensor = data["news_emb"]  # [N, 512]
        self.surprise_feat: torch.Tensor = data["surprise_feat"]  # [N, 3]
        self.modality_mask: torch.Tensor = data["modality_mask"]  # [N, 5]
        self.direction_label: torch.Tensor = data["direction_label"]  # [N]
        self.volatility_target: torch.Tensor = data["volatility_target"]  # [N]
        self.surprise_target: torch.Tensor = data["surprise_target"]  # [N]
        self.tickers: list[str] = data["tickers"]
        self.dates: list[str] = data["dates"]

    def __len__(self) -> int:
        return self.price_emb.size(0)

    def __getitem__(self, idx: int) -> dict:
        return {
            "price_emb": self.price_emb[idx],
            "gat_emb": self.gat_emb[idx],
            "doc_emb": self.doc_emb[idx],
            "news_emb": self.news_emb[idx],
            "surprise_feat": self.surprise_feat[idx],
            "modality_mask": self.modality_mask[idx],
            "direction_label": self.direction_label[idx],
            "volatility_target": self.volatility_target[idx],
            "surprise_target": self.surprise_target[idx],
            "ticker": self.tickers[idx],
            "date": self.dates[idx],
        }
