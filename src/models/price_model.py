from __future__ import annotations

import torch
from torch import nn


class PriceDirectionModel(nn.Module):
    def __init__(self, num_features: int, conv_channels: int = 64, lstm_hidden: int = 128) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(num_features, conv_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(conv_channels),
            nn.Conv1d(conv_channels, conv_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(conv_channels),
        )
        self.lstm = nn.LSTM(
            input_size=conv_channels,
            hidden_size=lstm_hidden,
            num_layers=2,
            batch_first=True,
            dropout=0.2,
            bidirectional=True,
        )
        self.head = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.transpose(1, 2)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        return self.head(x)
