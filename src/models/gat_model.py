"""Custom Graph Attention Network for inter-company signal propagation.

Pure PyTorch implementation (no PyTorch Geometric dependency) to satisfy the
project constraint that graph modules must be custom deep learning architectures.

Architecture
------------
1. ``GATLayer``  — single-head graph attention with LeakyReLU + softmax
2. ``MultiHeadGAT`` — multi-head wrapper with concat or average reduction
3. ``GraphEnhancedModel`` — wraps a per-company encoder (e.g. PriceDirectionModel
   without head) → GAT → direction classifier. Takes a snapshot of all 10 companies
   at once and produces per-company logits.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


# ---------------------------------------------------------------------------
# Graph Attention Layer (single head)
# ---------------------------------------------------------------------------


class GATLayer(nn.Module):
    """Single-head graph attention layer.

    Implements the attention mechanism from Veličković et al. (2018):
        e_ij = LeakyReLU(a^T [W h_i || W h_j])
        α_ij = softmax_j(e_ij)
        h'_i  = σ(Σ_j α_ij W h_j)

    Parameters
    ----------
    in_features : input node feature dimension
    out_features : output node feature dimension
    negative_slope : LeakyReLU slope for attention coefficients
    dropout : dropout probability on attention weights
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        negative_slope: float = 0.2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.W = nn.Linear(in_features, out_features, bias=False)
        # Attention vector: a ∈ R^{2 * out_features}
        self.a = nn.Parameter(torch.empty(2 * out_features))
        nn.init.xavier_uniform_(self.a.unsqueeze(0))
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x : [N, in_features] — node features
        edge_index : [2, E] — directed edges (src → tgt)
        edge_weight : [E] optional prior weights to scale attention

        Returns
        -------
        [N, out_features] — updated node features
        """
        N = x.size(0)
        Wh = self.W(x)  # [N, out_features]

        src, tgt = edge_index  # each [E]

        # Compute attention coefficients
        Wh_src = Wh[src]  # [E, out_features]
        Wh_tgt = Wh[tgt]  # [E, out_features]
        cat_features = torch.cat([Wh_src, Wh_tgt], dim=1)  # [E, 2*out_features]
        e = self.leaky_relu(cat_features @ self.a)  # [E]

        # Scale by prior edge weight if provided
        if edge_weight is not None:
            e = e * edge_weight

        # Softmax over neighbors: for each target node, softmax over incoming edges
        # Use scatter with -inf masking for stable softmax
        e_max = torch.full((N,), float("-inf"), device=x.device)
        e_max.scatter_reduce_(0, tgt, e, reduce="amax", include_self=False)
        e_exp = torch.exp(e - e_max[tgt])

        # Sum of exp per target node
        e_sum = torch.zeros(N, device=x.device)
        e_sum.scatter_add_(0, tgt, e_exp)
        alpha = e_exp / (e_sum[tgt] + 1e-10)  # [E]

        alpha = self.dropout(alpha)

        # Weighted aggregation
        messages = alpha.unsqueeze(1) * Wh_src  # [E, out_features]
        out = torch.zeros(N, Wh.size(1), device=x.device)
        out.scatter_add_(0, tgt.unsqueeze(1).expand_as(messages), messages)

        return out


# ---------------------------------------------------------------------------
# Multi-Head GAT
# ---------------------------------------------------------------------------


class MultiHeadGAT(nn.Module):
    """Multi-head graph attention with concat or mean reduction.

    Parameters
    ----------
    in_features : input dimension
    out_features : per-head output dimension
    num_heads : number of attention heads
    concat : if True, output dim = num_heads * out_features; else = out_features
    dropout : attention dropout
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_heads: int = 4,
        concat: bool = True,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.heads = nn.ModuleList(
            [
                GATLayer(in_features, out_features, dropout=dropout)
                for _ in range(num_heads)
            ]
        )
        self.concat = concat
        self.out_dim = out_features * num_heads if concat else out_features

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        head_outs = [head(x, edge_index, edge_weight) for head in self.heads]
        if self.concat:
            return torch.cat(head_outs, dim=1)  # [N, H*out]
        return torch.stack(head_outs, dim=0).mean(dim=0)  # [N, out]


# ---------------------------------------------------------------------------
# Graph-Enhanced Direction Model
# ---------------------------------------------------------------------------


class GraphEnhancedModel(nn.Module):
    """Price encoder → GAT → direction classifier.

    Takes a per-company price window batch (all 10 companies for one date)
    and produces per-company direction logits. The GAT propagates information
    across the inter-company graph edges.

    Architecture
    ------------
    1. Shared CNN-LSTM price encoder (without classification head)
       → [N, encoder_dim] per-company embeddings
    2. Two-layer GAT with residual connection
       → [N, gat_dim] graph-enhanced embeddings
    3. Direction classification head
       → [N, 2] logits (UP/DOWN)

    Parameters
    ----------
    num_features : number of price features per time step
    encoder_dim : price encoder output dimension (BiLSTM hidden * 2)
    gat_hidden : per-head hidden dimension
    gat_heads : number of attention heads
    gat_dropout : attention dropout
    num_classes : output classes
    """

    def __init__(
        self,
        num_features: int = 10,
        encoder_dim: int = 256,
        gat_hidden: int = 64,
        gat_heads: int = 4,
        gat_dropout: float = 0.1,
        num_classes: int = 2,
    ) -> None:
        super().__init__()

        # --- Price encoder (CNN + BiLSTM, shared across companies) ---
        self.conv = nn.Sequential(
            nn.Conv1d(num_features, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(64),
        )
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            dropout=0.2,
            bidirectional=True,
        )
        # encoder_dim = 128 * 2 = 256

        # --- Projection to GAT input dim ---
        gat_in_dim = gat_hidden * gat_heads
        self.node_proj = nn.Linear(encoder_dim, gat_in_dim)

        # --- Two-layer GAT ---
        self.gat1 = MultiHeadGAT(
            gat_in_dim,
            gat_hidden,
            num_heads=gat_heads,
            concat=True,
            dropout=gat_dropout,
        )
        self.gat_norm1 = nn.LayerNorm(gat_in_dim)

        self.gat2 = MultiHeadGAT(
            gat_in_dim,
            gat_hidden,
            num_heads=gat_heads,
            concat=True,
            dropout=gat_dropout,
        )
        self.gat_norm2 = nn.LayerNorm(gat_in_dim)

        # --- Direction head ---
        self.head = nn.Sequential(
            nn.Linear(gat_in_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def encode_price(self, x: torch.Tensor) -> torch.Tensor:
        """Run CNN-LSTM encoder on price windows.

        Parameters
        ----------
        x : [B, T, F] price features

        Returns
        -------
        [B, encoder_dim] company embeddings
        """
        h = x.transpose(1, 2)  # [B, F, T]
        h = self.conv(h)  # [B, 64, T]
        h = h.transpose(1, 2)  # [B, T, 64]
        h, _ = self.lstm(h)  # [B, T, 256]
        return h[:, -1, :]  # [B, 256] — last time step

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x : [N, T, F] — price windows for N companies
        edge_index : [2, E] — graph structure
        edge_weight : [E] — optional prior edge weights

        Returns
        -------
        [N, num_classes] — per-company direction logits
        """
        # 1. Encode each company's price window
        h = self.encode_price(x)  # [N, 256]

        # 2. Project to GAT dimension
        h = self.node_proj(h)  # [N, gat_in_dim]

        # 3. GAT layer 1 + residual + LayerNorm
        h1 = F.elu(self.gat1(h, edge_index, edge_weight))
        h = self.gat_norm1(h + h1)  # residual

        # 4. GAT layer 2 + residual + LayerNorm
        h2 = F.elu(self.gat2(h, edge_index, edge_weight))
        h = self.gat_norm2(h + h2)  # residual

        # 5. Per-company classification
        return self.head(h)  # [N, 2]
