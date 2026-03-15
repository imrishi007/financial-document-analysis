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


# ────────────────────────────────────────────────────────────────────
# Phase 12: Volatility-Primary Fusion Model
# ────────────────────────────────────────────────────────────────────


class Phase12FusionModel(nn.Module):
    """Volatility-primary multimodal fusion model.

    Key differences from V2 MultimodalFusionModel:
    - Primary head: volatility with Softplus output (guarantees σ > 0)
    - Auxiliary head: simplified direction (2 logits)
    - MC Dropout: dropout stays active at inference for uncertainty
    - Deeper volatility head for better expressiveness
    """

    NUM_MODALITIES = 4

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
        mc_dropout: bool = True,
    ) -> None:
        super().__init__()
        self.proj_dim = proj_dim
        self.surprise_dim = surprise_dim
        self.mc_dropout = mc_dropout

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

        # Gating (same architecture, surprise-conditioned)
        gate_in = proj_dim * self.NUM_MODALITIES + surprise_dim
        self.gate = nn.Sequential(
            nn.Linear(gate_in, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.NUM_MODALITIES),
            nn.Sigmoid(),
        )

        # Shared trunk
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

        # PRIMARY: Volatility head — deeper, with Softplus output
        self.volatility_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Softplus(),  # guarantees positive output
        )

        # AUXILIARY: Direction head — lightweight
        self.direction_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, 2),
        )

        # MC Dropout layer (applied at inference too)
        self._mc_drop = nn.Dropout(dropout)

    def forward(
        self,
        price_emb: torch.Tensor,
        gat_emb: torch.Tensor,
        doc_emb: torch.Tensor,
        macro_emb: torch.Tensor,
        surprise_feat: torch.Tensor,
        modality_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        p = self.price_proj(price_emb)
        g = self.gat_proj(gat_emb)
        d = self.doc_proj(doc_emb)
        m = self.macro_proj(macro_emb)

        stacked = torch.stack([p, g, d, m], dim=1)
        mask = modality_mask.unsqueeze(-1)
        stacked = stacked * mask

        flat = stacked.view(stacked.size(0), -1)
        gate_input = torch.cat([flat, surprise_feat], dim=1)
        gates = self.gate(gate_input)
        gates = gates * modality_mask
        gates = gates / (gates.sum(dim=1, keepdim=True) + 1e-8)

        weighted = stacked * gates.unsqueeze(-1)
        fused = weighted.view(weighted.size(0), -1)

        h = self.trunk(fused)

        # MC Dropout — stays active even in eval mode
        if self.mc_dropout:
            h = self._mc_drop(h)

        vol_pred = self.volatility_head(h).squeeze(-1)  # [B], always > 0
        dir_logits = self.direction_head(h)  # [B, 2]

        return {
            "volatility_pred": vol_pred,
            "direction_logits": dir_logits,
            "gate_weights": gates,
        }

    def predict_with_uncertainty(
        self,
        price_emb: torch.Tensor,
        gat_emb: torch.Tensor,
        doc_emb: torch.Tensor,
        macro_emb: torch.Tensor,
        surprise_feat: torch.Tensor,
        modality_mask: torch.Tensor,
        n_samples: int = 30,
    ) -> dict[str, torch.Tensor]:
        """MC Dropout uncertainty estimation.

        Returns mean, std, and individual samples for volatility predictions.
        """
        self.train()  # keep dropout active
        vol_samples = []
        dir_samples = []
        with torch.no_grad():
            for _ in range(n_samples):
                out = self.forward(
                    price_emb,
                    gat_emb,
                    doc_emb,
                    macro_emb,
                    surprise_feat,
                    modality_mask,
                )
                vol_samples.append(out["volatility_pred"])
                dir_samples.append(out["direction_logits"])

        vol_stack = torch.stack(vol_samples, dim=0)  # [S, B]
        dir_stack = torch.stack(dir_samples, dim=0)  # [S, B, 2]

        return {
            "vol_mean": vol_stack.mean(dim=0),
            "vol_std": vol_stack.std(dim=0),
            "vol_samples": vol_stack,
            "dir_mean": dir_stack.mean(dim=0),
            "dir_std": dir_stack.std(dim=0),
        }


# ────────────────────────────────────────────────────────────────────
# Phase 13: Configurable loss-weight fusion model (HAR-RV features)
# ────────────────────────────────────────────────────────────────────


class Phase13FusionModel(Phase12FusionModel):
    """Phase 13 fusion model — identical architecture to Phase12FusionModel.

    The only addition is the ``loss_weights`` config attribute so that
    callers can instantiate three variants (0.85/0.15, 0.50/0.50, 0.70/0.30)
    without changing the architecture.  All training-time logic remains in
    the training script; this class merely exposes the weight config.
    """

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
        mc_dropout: bool = True,
        lambda_vol: float = 0.85,
        lambda_dir: float = 0.15,
    ) -> None:
        super().__init__(
            price_dim=price_dim,
            gat_dim=gat_dim,
            doc_dim=doc_dim,
            macro_dim=macro_dim,
            surprise_dim=surprise_dim,
            proj_dim=proj_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            mc_dropout=mc_dropout,
        )
        self.lambda_vol = lambda_vol
        self.lambda_dir = lambda_dir


# ────────────────────────────────────────────────────────────────────
# Phase 14: HAR-RV Skip Connection Fusion Model
# ────────────────────────────────────────────────────────────────────


class Phase14FusionModel(nn.Module):
    """Phase 14 fusion model: adds HAR-RV skip connection to Phase 13 architecture.

    The three HAR-RV features (rv_lag1d, rv_lag5d, rv_lag22d) are extracted
    from the last timestep of the raw price sequence and fed DIRECTLY into
    the fusion trunk via a small projection, bypassing the CNN-BiLSTM.

    This gives the model an uncompressed linear path from lagged RV to the
    volatility prediction — the same path that makes HAR-RV R²=0.947.

    Fusion trunk input: [price_emb(128 gated) + gat(128 gated) +
                         doc(128 gated) + macro(128 gated) + har_proj(32)]
    = [B, 544]
    """

    NUM_MODALITIES = 4  # price, gat, doc, macro (HAR skip bypasses gating)

    def __init__(
        self,
        price_dim: int = 256,
        har_rv_dim: int = 3,
        har_proj_dim: int = 32,
        gat_dim: int = 256,
        doc_dim: int = 768,
        macro_dim: int = 32,
        surprise_dim: int = 5,
        proj_dim: int = 128,
        hidden_dim: int = 256,
        dropout: float = 0.3,
        trunk_dropout: float = 0.3,
        mc_dropout: bool = True,
        lambda_vol: float = 0.85,
        lambda_dir: float = 0.15,
    ) -> None:
        super().__init__()
        self.proj_dim = proj_dim
        self.surprise_dim = surprise_dim
        self.mc_dropout = mc_dropout
        self.lambda_vol = lambda_vol
        self.lambda_dir = lambda_dir

        # HAR-RV skip connection projector (deliberately small, near-linear path)
        self.har_rv_skip = nn.Sequential(
            nn.Linear(har_rv_dim, har_proj_dim),
            nn.LayerNorm(har_proj_dim),
            nn.GELU(),
        )

        # Per-modality projectors (identical to Phase 13)
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

        # Gating network (HAR-RV skip bypasses gating — always contributes)
        gate_in = proj_dim * self.NUM_MODALITIES + surprise_dim
        self.gate = nn.Sequential(
            nn.Linear(gate_in, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.NUM_MODALITIES),
            nn.Sigmoid(),
        )

        # Shared trunk: 4 gated projections + HAR-RV skip
        trunk_in = proj_dim * self.NUM_MODALITIES + har_proj_dim
        self.trunk = nn.Sequential(
            nn.Linear(trunk_in, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(trunk_dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
        )

        # PRIMARY: Volatility head — deeper, with Softplus output
        self.volatility_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Softplus(),
        )

        # AUXILIARY: Direction head — lightweight
        self.direction_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, 2),
        )

        # MC Dropout layer
        self._mc_drop = nn.Dropout(dropout)

    def forward(
        self,
        price_emb: torch.Tensor,
        har_rv_raw: torch.Tensor,
        gat_emb: torch.Tensor,
        doc_emb: torch.Tensor,
        macro_emb: torch.Tensor,
        surprise_feat: torch.Tensor,
        modality_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Forward pass.

        Parameters
        ----------
        price_emb : [B, 256] from CNN-BiLSTM
        har_rv_raw : [B, 3] raw HAR-RV features (last timestep)
        gat_emb : [B, 256]
        doc_emb : [B, 768]
        macro_emb : [B, 32]
        surprise_feat : [B, 5]
        modality_mask : [B, 4]
        """
        p = self.price_proj(price_emb)
        g = self.gat_proj(gat_emb)
        d = self.doc_proj(doc_emb)
        m = self.macro_proj(macro_emb)

        # HAR-RV skip (direct, ungated)
        har = self.har_rv_skip(har_rv_raw)  # [B, 32]

        # Gating
        stacked = torch.stack([p, g, d, m], dim=1)  # [B, 4, proj]
        mask = modality_mask.unsqueeze(-1)
        stacked = stacked * mask

        flat = stacked.view(stacked.size(0), -1)
        gate_input = torch.cat([flat, surprise_feat], dim=1)
        gates = self.gate(gate_input)
        gates = gates * modality_mask
        gates = gates / (gates.sum(dim=1, keepdim=True) + 1e-8)

        weighted = stacked * gates.unsqueeze(-1)
        fused = weighted.view(weighted.size(0), -1)  # [B, 4*proj]

        # Concatenate gated modalities + HAR-RV skip
        trunk_input = torch.cat([fused, har], dim=-1)  # [B, 4*proj + 32]

        h = self.trunk(trunk_input)

        # MC Dropout
        if self.mc_dropout:
            h = self._mc_drop(h)

        vol_pred = self.volatility_head(h).squeeze(-1)
        dir_logits = self.direction_head(h)

        return {
            "volatility_pred": vol_pred,
            "direction_logits": dir_logits,
            "gate_weights": gates,
        }

    def predict_with_uncertainty(
        self,
        price_emb: torch.Tensor,
        har_rv_raw: torch.Tensor,
        gat_emb: torch.Tensor,
        doc_emb: torch.Tensor,
        macro_emb: torch.Tensor,
        surprise_feat: torch.Tensor,
        modality_mask: torch.Tensor,
        n_samples: int = 30,
    ) -> dict[str, torch.Tensor]:
        """MC Dropout uncertainty estimation."""
        self.train()
        vol_samples = []
        dir_samples = []
        with torch.no_grad():
            for _ in range(n_samples):
                out = self.forward(
                    price_emb,
                    har_rv_raw,
                    gat_emb,
                    doc_emb,
                    macro_emb,
                    surprise_feat,
                    modality_mask,
                )
                vol_samples.append(out["volatility_pred"])
                dir_samples.append(out["direction_logits"])

        vol_stack = torch.stack(vol_samples, dim=0)
        dir_stack = torch.stack(dir_samples, dim=0)

        return {
            "vol_mean": vol_stack.mean(dim=0),
            "vol_std": vol_stack.std(dim=0),
            "vol_samples": vol_stack,
            "dir_mean": dir_stack.mean(dim=0),
            "dir_std": dir_stack.std(dim=0),
        }
