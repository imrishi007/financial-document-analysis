from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class TrainingConfig:
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    epochs: int = 20


def create_optimizer(model: torch.nn.Module, config: TrainingConfig) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
