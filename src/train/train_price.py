from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.models.price_model import PriceDirectionModel
from src.train.common import TrainingConfig, create_optimizer


class PriceSequenceDataset(Dataset):
    def __init__(self, features: torch.Tensor, labels: torch.Tensor) -> None:
        self.features = features
        self.labels = labels

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx]


@dataclass
class PriceTrainResult:
    train_loss: float
    train_accuracy: float


def train_one_epoch(
    model: PriceDirectionModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
) -> PriceTrainResult:
    criterion = nn.CrossEntropyLoss()
    model.train()

    loss_sum = 0.0
    correct = 0
    total = 0

    for features, labels in loader:
        features = features.to(device)
        labels = labels.to(device)

        logits = model(features)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_sum += float(loss.item()) * features.size(0)
        predictions = logits.argmax(dim=1)
        correct += int((predictions == labels).sum().item())
        total += int(features.size(0))

    return PriceTrainResult(
        train_loss=loss_sum / max(total, 1),
        train_accuracy=correct / max(total, 1),
    )


def build_price_trainer(num_features: int, config: TrainingConfig) -> tuple[PriceDirectionModel, torch.optim.Optimizer]:
    model = PriceDirectionModel(num_features=num_features)
    optimizer = create_optimizer(model, config)
    return model, optimizer
