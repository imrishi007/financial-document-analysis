"""Fusion dataset: loads pre-extracted modality embeddings from disk.

The embedding file is produced by ``src.features.extract_embeddings``.
Each sample is a set of modality embeddings + availability mask + targets,
all aligned by (ticker, date).

V2 changes:
- 4 modalities: price, gat, doc, macro (no news)
- Surprise features: 5-d fixed vector (no leakage)
- Primary target: 60-day direction
- No surprise classification target
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset


class FusionEmbeddingDataset(Dataset):
    """Lightweight dataset over pre-extracted embeddings.

    Parameters
    ----------
    embeddings_path : path to ``fusion_embeddings.pt``
    """

    # Modality names in mask order (4 modalities, no news)
    MODALITIES = ["price", "gat", "doc", "macro"]

    def __init__(self, embeddings_path: str | Path) -> None:
        data = torch.load(embeddings_path, weights_only=False, map_location="cpu")

        self.price_emb: torch.Tensor = data["price_emb"]  # [N, 256]
        self.gat_emb: torch.Tensor = data["gat_emb"]  # [N, 256]
        self.doc_emb: torch.Tensor = data["doc_emb"]  # [N, 768]
        self.macro_emb: torch.Tensor = data["macro_emb"]  # [N, 32]
        self.surprise_feat: torch.Tensor = data["surprise_feat"]  # [N, 5]
        self.modality_mask: torch.Tensor = data["modality_mask"]  # [N, 4]
        self.direction_label: torch.Tensor = data["direction_label"]  # [N] (60-day)
        self.volatility_target: torch.Tensor = data["volatility_target"]  # [N]
        self.tickers: list[str] = data["tickers"]
        self.dates: list[str] = data["dates"]

        # Date index for ListNet grouping
        unique_dates = sorted(set(self.dates))
        self.date_to_idx = {d: i for i, d in enumerate(unique_dates)}
        self.date_indices = torch.tensor(
            [self.date_to_idx[d] for d in self.dates], dtype=torch.long
        )

    def to_device(self, device: str | torch.device) -> "FusionEmbeddingDataset":
        """Move all tensor data to the specified device (e.g., 'cuda').

        This pre-loads everything to GPU so the DataLoader doesn't need
        per-batch device transfers, reducing overhead significantly.
        """
        self.price_emb = self.price_emb.to(device)
        self.gat_emb = self.gat_emb.to(device)
        self.doc_emb = self.doc_emb.to(device)
        self.macro_emb = self.macro_emb.to(device)
        self.surprise_feat = self.surprise_feat.to(device)
        self.modality_mask = self.modality_mask.to(device)
        self.direction_label = self.direction_label.to(device)
        self.volatility_target = self.volatility_target.to(device)
        self.date_indices = self.date_indices.to(device)
        return self

    def __len__(self) -> int:
        return self.price_emb.size(0)

    def __getitem__(self, idx: int) -> dict:
        return {
            "price_emb": self.price_emb[idx],
            "gat_emb": self.gat_emb[idx],
            "doc_emb": self.doc_emb[idx],
            "macro_emb": self.macro_emb[idx],
            "surprise_feat": self.surprise_feat[idx],
            "modality_mask": self.modality_mask[idx],
            "direction_label": self.direction_label[idx],
            "volatility_target": self.volatility_target[idx],
            "date_index": self.date_indices[idx],
            "ticker": self.tickers[idx],
            "date": self.dates[idx],
        }
