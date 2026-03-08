"""Common training utilities: config, early stopping, checkpointing."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class TrainingConfig:
    """Hyper-parameter bundle passed to all training routines."""

    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    epochs: int = 20
    batch_size: int = 32
    patience: int = 5  # early-stopping patience
    seed: int = 42
    device: str = "auto"

    def resolve_device(self) -> str:
        if self.device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.device


def create_optimizer(
    model: nn.Module,
    config: TrainingConfig,
    *,
    finetune_lr: Optional[float] = None,
    finetune_params: Optional[list] = None,
) -> torch.optim.Optimizer:
    """Create AdamW optimizer, optionally with separate LR for fine-tune params."""
    if finetune_lr is not None and finetune_params is not None:
        finetune_ids = {id(p) for p in finetune_params}
        head_params = [p for p in model.parameters() if id(p) not in finetune_ids]
        param_groups = [
            {"params": head_params, "lr": config.learning_rate},
            {"params": finetune_params, "lr": finetune_lr},
        ]
        return torch.optim.AdamW(param_groups, weight_decay=config.weight_decay)
    return torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )


# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------


class EarlyStopping:
    """Stop training when a monitored metric stops improving."""

    def __init__(self, patience: int = 5, mode: str = "min", min_delta: float = 1e-4):
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.counter: int = 0
        self.best_score: Optional[float] = None
        self.should_stop: bool = False

    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False

        improved = (
            score < self.best_score - self.min_delta
            if self.mode == "min"
            else score > self.best_score + self.min_delta
        )

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict,
    path: str | Path,
) -> None:
    """Save model + optimizer state and metadata."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
        },
        path,
    )


def load_checkpoint(
    model: nn.Module,
    path: str | Path,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> dict:
    """Restore model state and optionally optimizer state."""
    ckpt = torch.load(path, weights_only=False, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return ckpt


# ---------------------------------------------------------------------------
# Generic train / evaluate helpers
# ---------------------------------------------------------------------------


@torch.no_grad()
def evaluate_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str,
    *,
    input_key: str = "features",
    label_key: str = "direction_1d",
) -> dict[str, Any]:
    """Run one evaluation epoch and return loss, predictions, and true labels."""
    model.eval()
    loss_sum = 0.0
    all_preds: list[int] = []
    all_labels: list[int] = []
    all_probs: list[float] = []
    total = 0

    for batch in loader:
        features = batch[input_key].to(device)
        labels = batch[label_key].to(device)
        logits = model(features)
        loss = criterion(logits, labels)

        loss_sum += loss.item() * features.size(0)
        probs = torch.softmax(logits, dim=1)[:, 1]
        preds = logits.argmax(dim=1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
        all_probs.extend(probs.cpu().tolist())
        total += features.size(0)

    return {
        "loss": loss_sum / max(total, 1),
        "y_true": np.array(all_labels),
        "y_pred": np.array(all_preds),
        "y_prob": np.array(all_probs),
    }
