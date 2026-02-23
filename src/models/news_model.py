from __future__ import annotations

import torch
from torch import nn
from transformers import AutoModel


class NewsTemporalDirectionModel(nn.Module):
    def __init__(
        self,
        backbone_name: str = "ProsusAI/finbert",
        hidden_size: int = 768,
        temporal_hidden: int = 256,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(backbone_name)
        self.temporal = nn.GRU(
            input_size=hidden_size,
            hidden_size=temporal_hidden,
            num_layers=2,
            dropout=dropout,
            batch_first=True,
            bidirectional=True,
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(temporal_hidden * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
        )

    def encode_articles(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        batch_size, num_articles, seq_len = input_ids.shape
        ids = input_ids.view(batch_size * num_articles, seq_len)
        mask = attention_mask.view(batch_size * num_articles, seq_len)
        output = self.encoder(input_ids=ids, attention_mask=mask)
        cls_embeddings = output.last_hidden_state[:, 0, :]
        return cls_embeddings.view(batch_size, num_articles, -1)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        article_embeddings = self.encode_articles(input_ids, attention_mask)
        temporal_output, _ = self.temporal(article_embeddings)
        last_step = temporal_output[:, -1, :]
        return self.head(last_step)
