"""Graph utilities: load static company graph and convert to PyTorch tensors.

The graph is stored as two CSVs (nodes + edges) built by
``src/data_collection/graph_builder.py`` during Phase 1.

Key outputs
-----------
- ``ticker_to_idx``: mapping from ticker string to integer node index
- ``edge_index``:   [2, E] LongTensor of directed edges (COO format)
- ``edge_weight``:  [E] FloatTensor of prior edge weights
- ``edge_type``:    [E] LongTensor of relation-type indices
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import torch


# Default relation types from `graph_builder.build_default_edges`
RELATION_TYPES: list[str] = [
    "supplier_to",
    "platform_competitor",
    "ad_competitor",
    "cloud_competitor",
    "consumer_attention_overlap",
    "enterprise_competitor",
    "index_correlation",
]


def load_graph(
    nodes_csv: str | Path = "data/raw/graph/tech10_nodes.csv",
    edges_csv: str | Path = "data/raw/graph/tech10_edges.csv",
) -> dict[str, Any]:
    """Load the static company graph and return tensors + metadata.

    Returns
    -------
    dict with keys:
        tickers        : list[str] — ordered ticker list
        ticker_to_idx  : dict[str, int]
        num_nodes      : int
        edge_index     : Tensor [2, E]
        edge_weight    : Tensor [E]
        edge_type      : Tensor [E]  (index into RELATION_TYPES)
        relation_names : list[str]
    """
    nodes_df = pd.read_csv(nodes_csv)
    edges_df = pd.read_csv(edges_csv)

    tickers = nodes_df["ticker"].tolist()
    ticker_to_idx = {t: i for i, t in enumerate(tickers)}
    num_nodes = len(tickers)

    # Build relation-type mapping
    relation_to_idx = {r: i for i, r in enumerate(RELATION_TYPES)}

    src_indices: list[int] = []
    tgt_indices: list[int] = []
    weights: list[float] = []
    types: list[int] = []

    for _, row in edges_df.iterrows():
        s = row["source"]
        t = row["target"]
        if s not in ticker_to_idx or t not in ticker_to_idx:
            continue
        src_indices.append(ticker_to_idx[s])
        tgt_indices.append(ticker_to_idx[t])
        weights.append(float(row["weight"]))
        rel = row["relation"]
        types.append(relation_to_idx.get(rel, len(RELATION_TYPES)))

    # Add self-loops for every node (important for GAT message passing)
    for i in range(num_nodes):
        src_indices.append(i)
        tgt_indices.append(i)
        weights.append(1.0)
        types.append(len(RELATION_TYPES))  # self-loop type

    edge_index = torch.tensor([src_indices, tgt_indices], dtype=torch.long)
    edge_weight = torch.tensor(weights, dtype=torch.float32)
    edge_type = torch.tensor(types, dtype=torch.long)

    return {
        "tickers": tickers,
        "ticker_to_idx": ticker_to_idx,
        "num_nodes": num_nodes,
        "edge_index": edge_index,  # [2, E+N]
        "edge_weight": edge_weight,  # [E+N]
        "edge_type": edge_type,  # [E+N]
        "relation_names": RELATION_TYPES + ["self_loop"],
    }


def make_bidirectional(
    edge_index: torch.Tensor, edge_weight: torch.Tensor, edge_type: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Make directed edges bidirectional by adding reverse edges.

    Self-loops are not duplicated.
    """
    src, tgt = edge_index[0], edge_index[1]
    mask = src != tgt  # non-self-loop edges

    rev_src = tgt[mask]
    rev_tgt = src[mask]
    rev_w = edge_weight[mask]
    rev_t = edge_type[mask]

    new_src = torch.cat([src, rev_src])
    new_tgt = torch.cat([tgt, rev_tgt])
    new_w = torch.cat([edge_weight, rev_w])
    new_t = torch.cat([edge_type, rev_t])

    return (
        torch.stack([new_src, new_tgt]),
        new_w,
        new_t,
    )
