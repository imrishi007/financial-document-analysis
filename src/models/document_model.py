from __future__ import annotations

import torch
from torch import nn
from transformers import AutoModel


class AttentionPooling(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.score = nn.Linear(hidden_size, 1)

    def forward(self, sequence_output: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        scores = self.score(sequence_output).squeeze(-1)
        scores = scores.masked_fill(attention_mask == 0, float("-inf"))
        weights = torch.softmax(scores, dim=-1).unsqueeze(-1)
        return torch.sum(sequence_output * weights, dim=1)


class DocumentDirectionModel(nn.Module):
    def __init__(self, backbone_name: str = "ProsusAI/finbert", dropout: float = 0.2) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(backbone_name)
        hidden_size = self.encoder.config.hidden_size
        self.pool = AttentionPooling(hidden_size)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 2),
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        output = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.pool(output.last_hidden_state, attention_mask)
        return self.head(pooled)
