"""Macro State Model: 3-layer MLP encoder for macro features.

Architecture: Linear(12, 64) -> LayerNorm -> GELU -> Linear(64, 64)
-> LayerNorm -> GELU -> Linear(64, 32)

Output: 32-dimensional embedding representing the macro state.
"""

from __future__ import annotations

import torch
from torch import nn


class MacroStateModel(nn.Module):
    """Simple MLP encoder for macro features.

    Parameters
    ----------
    input_dim : Dimension of macro feature vector (12).
    hidden_dim : Hidden layer dimension (64).
    output_dim : Output embedding dimension (32).
    num_classes : Number of direction classes for the classification head.
    dropout : Dropout probability.
    """

    def __init__(
        self,
        input_dim: int = 12,
        hidden_dim: int = 64,
        output_dim: int = 32,
        num_classes: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

        self.head = nn.Sequential(
            nn.Linear(output_dim, num_classes),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Extract macro state embedding.

        Parameters
        ----------
        x : [B, 12] macro features

        Returns
        -------
        [B, 32] macro state embedding
        """
        return self.encoder(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full forward pass with classification head.

        Parameters
        ----------
        x : [B, 12] macro features

        Returns
        -------
        [B, 2] direction logits
        """
        emb = self.encode(x)
        return self.head(emb)
