"""Multimodal Gated Attention Fusion Model.

Fuses pre-extracted embeddings from five modalities (price, graph-GAT,
document, news, fundamental-surprise) via learned gating and produces
multi-task outputs for direction, volatility, and surprise prediction.

Architecture
------------
1. Per-modality linear projectors → common ``proj_dim``
2. Sigmoid-gated importance weighting (handles missing modalities via mask)
3. Shared trunk (2-layer MLP on concatenated gated projections)
4. Three task-specific heads:
   - **Direction** (binary CE): UP / DOWN
   - **Volatility** (MSE): realised 20-d annualised volatility
   - **Surprise** (binary CE): BEAT / MISS (sparse)
"""

from __future__ import annotations

import torch
from torch import nn


class MultimodalFusionModel(nn.Module):
    """Gated attention fusion of pre-extracted modality embeddings.

    Parameters
    ----------
    price_dim : Price encoder embedding dimension (256).
    gat_dim : GAT encoder embedding dimension (256).
    doc_dim : Document encoder embedding dimension (768).
    news_dim : News encoder embedding dimension (512).
    surprise_dim : Surprise feature dimension (3).
    proj_dim : Common projection size per modality.
    hidden_dim : Hidden size for the shared trunk.
    dropout : Dropout probability.
    """

    NUM_MODALITIES = 5

    def __init__(
        self,
        price_dim: int = 256,
        gat_dim: int = 256,
        doc_dim: int = 768,
        news_dim: int = 512,
        surprise_dim: int = 3,
        proj_dim: int = 128,
        hidden_dim: int = 256,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.proj_dim = proj_dim

        # ----- Per-modality projectors -----
        def _proj(in_dim: int) -> nn.Module:
            return nn.Sequential(
                nn.Linear(in_dim, proj_dim),
                nn.LayerNorm(proj_dim),
                nn.GELU(),
                nn.Dropout(dropout * 0.5),
            )

        self.price_proj = _proj(price_dim)
        self.gat_proj = _proj(gat_dim)
        self.doc_proj = _proj(doc_dim)
        self.news_proj = _proj(news_dim)
        self.surprise_proj = _proj(surprise_dim)

        # ----- Gating network -----
        # Learns per-modality importance from all projected features
        gate_in = proj_dim * self.NUM_MODALITIES
        self.gate = nn.Sequential(
            nn.Linear(gate_in, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.NUM_MODALITIES),
            nn.Sigmoid(),
        )

        # ----- Shared trunk -----
        trunk_in = proj_dim * self.NUM_MODALITIES
        self.trunk = nn.Sequential(
            nn.Linear(trunk_in, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
        )

        # ----- Task heads -----
        self.direction_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim // 2, 2),
        )

        self.volatility_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, 1),
        )

        self.surprise_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, 2),
        )

    def forward(
        self,
        price_emb: torch.Tensor,
        gat_emb: torch.Tensor,
        doc_emb: torch.Tensor,
        news_emb: torch.Tensor,
        surprise_feat: torch.Tensor,
        modality_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Forward pass.

        Parameters
        ----------
        price_emb : [B, 256]
        gat_emb : [B, 256]
        doc_emb : [B, 768]
        news_emb : [B, 512]
        surprise_feat : [B, 3]
        modality_mask : [B, 5] float — 1.0 if modality present, else 0.0

        Returns
        -------
        dict with keys ``direction_logits`` [B, 2], ``volatility_pred`` [B],
        ``surprise_logits`` [B, 2], ``gate_weights`` [B, 5].
        """
        # 1. Project each modality
        p = self.price_proj(price_emb)  # [B, proj]
        g = self.gat_proj(gat_emb)  # [B, proj]
        d = self.doc_proj(doc_emb)  # [B, proj]
        n = self.news_proj(news_emb)  # [B, proj]
        s = self.surprise_proj(surprise_feat)  # [B, proj]

        # Stack → [B, 5, proj]
        stacked = torch.stack([p, g, d, n, s], dim=1)

        # 2. Zero out unavailable modalities
        mask = modality_mask.unsqueeze(-1)  # [B, 5, 1]
        stacked = stacked * mask

        # 3. Compute gated importance
        flat = stacked.view(stacked.size(0), -1)  # [B, 5*proj]
        gates = self.gate(flat)  # [B, 5]
        gates = gates * modality_mask  # mask unavailable
        # Normalise so available gates sum to 1
        gates = gates / (gates.sum(dim=1, keepdim=True) + 1e-8)

        # 4. Apply gates to projections
        weighted = stacked * gates.unsqueeze(-1)  # [B, 5, proj]
        fused = weighted.view(weighted.size(0), -1)  # [B, 5*proj]

        # 5. Shared trunk
        h = self.trunk(fused)  # [B, hidden]

        # 6. Task heads
        dir_logits = self.direction_head(h)  # [B, 2]
        vol_pred = self.volatility_head(h).squeeze(-1)  # [B]
        surp_logits = self.surprise_head(h)  # [B, 2]

        return {
            "direction_logits": dir_logits,
            "volatility_pred": vol_pred,
            "surprise_logits": surp_logits,
            "gate_weights": gates,
        }


# Backward-compatible alias
FusionDirectionModel = MultimodalFusionModel
