"""Multimodal Gated Attention Fusion Model (V2).

Fuses pre-extracted embeddings from four modalities (price, graph-GAT,
document, macro) via learned gating. Surprise features are used as
additional input to the gating network, not as a separate modality.

Architecture
------------
1. Per-modality linear projectors -> common ``proj_dim``
2. Sigmoid-gated importance weighting (handles missing modalities via mask)
   Surprise features (5-d) feed into the gating network as conditioning signal
3. Shared trunk (2-layer MLP on concatenated gated projections)
4. Two task-specific heads:
   - **Direction** (ListNet ranking loss): UP / DOWN (60-day primary)
   - **Volatility** (MSE): realized 20-d annualized volatility
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
    macro_dim : Macro feature embedding dimension (32).
    surprise_dim : Surprise feature dimension (5) -- used in gating, not as modality.
    proj_dim : Common projection size per modality.
    hidden_dim : Hidden size for the shared trunk.
    dropout : Dropout probability.
    """

    NUM_MODALITIES = 4  # price, gat, doc, macro

    def __init__(
        self,
        price_dim: int = 256,
        gat_dim: int = 256,
        doc_dim: int = 768,
        macro_dim: int = 32,
        surprise_dim: int = 5,
        proj_dim: int = 128,
        hidden_dim: int = 256,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.proj_dim = proj_dim
        self.surprise_dim = surprise_dim

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
        self.macro_proj = _proj(macro_dim)

        # ----- Gating network -----
        # Takes concatenated modality projections + surprise features as input
        gate_in = proj_dim * self.NUM_MODALITIES + surprise_dim
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
        # Direction head: outputs 2 logits (DOWN, UP) for 60-day direction
        self.direction_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim // 2, 2),
        )

        # Volatility head: outputs scalar (realized vol prediction)
        self.volatility_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, 1),
        )

    def forward(
        self,
        price_emb: torch.Tensor,
        gat_emb: torch.Tensor,
        doc_emb: torch.Tensor,
        macro_emb: torch.Tensor,
        surprise_feat: torch.Tensor,
        modality_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Forward pass.

        Parameters
        ----------
        price_emb : [B, 256]
        gat_emb : [B, 256]
        doc_emb : [B, 768]
        macro_emb : [B, 32]
        surprise_feat : [B, 5] -- conditioning signal for gating
        modality_mask : [B, 4] float -- 1.0 if modality present, else 0.0

        Returns
        -------
        dict with keys ``direction_logits`` [B, 2], ``volatility_pred`` [B],
        ``gate_weights`` [B, 4].
        """
        # 1. Project each modality
        p = self.price_proj(price_emb)  # [B, proj]
        g = self.gat_proj(gat_emb)  # [B, proj]
        d = self.doc_proj(doc_emb)  # [B, proj]
        m = self.macro_proj(macro_emb)  # [B, proj]

        # Stack -> [B, 4, proj]
        stacked = torch.stack([p, g, d, m], dim=1)

        # 2. Zero out unavailable modalities
        mask = modality_mask.unsqueeze(-1)  # [B, 4, 1]
        stacked = stacked * mask

        # 3. Compute gated importance (conditioned on surprise features)
        flat = stacked.view(stacked.size(0), -1)  # [B, 4*proj]
        gate_input = torch.cat([flat, surprise_feat], dim=1)  # [B, 4*proj + 5]
        gates = self.gate(gate_input)  # [B, 4]
        gates = gates * modality_mask  # mask unavailable
        # Normalize so available gates sum to 1
        gates = gates / (gates.sum(dim=1, keepdim=True) + 1e-8)

        # 4. Apply gates to projections
        weighted = stacked * gates.unsqueeze(-1)  # [B, 4, proj]
        fused = weighted.view(weighted.size(0), -1)  # [B, 4*proj]

        # 5. Shared trunk
        h = self.trunk(fused)  # [B, hidden]

        # 6. Task heads
        dir_logits = self.direction_head(h)  # [B, 2]
        vol_pred = self.volatility_head(h).squeeze(-1)  # [B]

        return {
            "direction_logits": dir_logits,
            "volatility_pred": vol_pred,
            "gate_weights": gates,
        }


# Backward-compatible alias
FusionDirectionModel = MultimodalFusionModel
