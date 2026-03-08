"""Full training pipeline for PriceDirectionModel (CNN-LSTM).

Usage from a script or notebook::

    from src.train.train_price import run_price_training
    results = run_price_training(prices_dir="data/raw/prices", targets_dir="data/targets")
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from src.data.preprocessing import SplitConfig, create_time_splits, fit_scaler
from src.data.price_dataset import (
    ENGINEERED_FEATURES,
    PriceWindowDataset,
    load_price_csv_dir,
    prepare_price_features,
)
from src.evaluation.metrics import classification_metrics, majority_baseline_accuracy
from src.models.price_model import PriceDirectionModel
from src.train.common import (
    EarlyStopping,
    TrainingConfig,
    create_optimizer,
    evaluate_epoch,
    save_checkpoint,
)
from src.utils.seed import set_global_seed


# ---------------------------------------------------------------------------
# Train one epoch
# ---------------------------------------------------------------------------


def train_one_epoch(
    model: PriceDirectionModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str,
) -> dict[str, float]:
    """Train for one epoch and return average loss + accuracy."""
    model.train()
    loss_sum = 0.0
    correct = 0
    total = 0

    for batch in loader:
        features = batch["features"].to(device)
        labels = batch["direction_1d"].to(device)

        logits = model(features)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_sum += loss.item() * features.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += features.size(0)

    return {
        "train_loss": loss_sum / max(total, 1),
        "train_acc": correct / max(total, 1),
    }


# ---------------------------------------------------------------------------
# Split helper
# ---------------------------------------------------------------------------


def _split_dataset_by_date(
    dataset: PriceWindowDataset,
    split_cfg: Optional[SplitConfig] = None,
) -> dict[str, list[int]]:
    """Return index lists for train / val / test using sample dates."""
    if split_cfg is None:
        split_cfg = SplitConfig()

    dates = pd.Series(
        pd.to_datetime([dataset._samples[i]["date"] for i in range(len(dataset))])
    )
    masks = create_time_splits(dates, split_cfg)

    return {
        split: [i for i, m in enumerate(mask) if m] for split, mask in masks.items()
    }


# ---------------------------------------------------------------------------
# Full training pipeline
# ---------------------------------------------------------------------------


def run_price_training(
    prices_dir: str | Path = "data/raw/prices",
    targets_dir: str | Path = "data/targets",
    config: Optional[TrainingConfig] = None,
    split_cfg: Optional[SplitConfig] = None,
    save_dir: str | Path = "models",
    verbose: bool = True,
) -> dict[str, Any]:
    """End-to-end price model training with evaluation.

    Returns a dict with training history, final metrics, and the trained model.
    """
    if config is None:
        config = TrainingConfig()
    if split_cfg is None:
        split_cfg = SplitConfig()

    device = config.resolve_device()
    set_global_seed(config.seed)
    if verbose:
        print(f"Device: {device} | Seed: {config.seed}")

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    prices = load_price_csv_dir(prices_dir)
    price_feat = prepare_price_features(prices)

    targets = pd.read_csv(Path(targets_dir) / "direction_labels_multi_horizon.csv")
    vol = pd.read_csv(Path(targets_dir) / "volatility_targets.csv")
    surprise = pd.read_csv(Path(targets_dir) / "fundamental_surprise_targets.csv")

    # ------------------------------------------------------------------
    # 2. Fit scaler on TRAINING data only (prevents leakage)
    # ------------------------------------------------------------------
    dates_series = pd.to_datetime(price_feat["date"])
    train_end = pd.Timestamp(split_cfg.train_end)
    train_mask = dates_series <= train_end

    scaler = fit_scaler(price_feat[train_mask], ENGINEERED_FEATURES)
    price_feat = scaler.transform(price_feat, ENGINEERED_FEATURES)

    if verbose:
        print(f"Features scaled (z-score fit on {train_mask.sum()} train rows)")

    # ------------------------------------------------------------------
    # 3. Build dataset and split
    # ------------------------------------------------------------------
    dataset = PriceWindowDataset(
        price_feat,
        targets,
        vol_df=vol,
        surprise_df=surprise,
        window_size=60,
    )
    idx_map = _split_dataset_by_date(dataset, split_cfg)
    train_ds = Subset(dataset, idx_map["train"])
    val_ds = Subset(dataset, idx_map["val"])
    test_ds = Subset(dataset, idx_map["test"])

    if verbose:
        print(
            f"Samples — train: {len(train_ds)}, val: {len(val_ds)}, test: {len(test_ds)}"
        )

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size)
    test_loader = DataLoader(test_ds, batch_size=config.batch_size)

    # ------------------------------------------------------------------
    # 4. Build model
    # ------------------------------------------------------------------
    model = PriceDirectionModel(num_features=len(ENGINEERED_FEATURES)).to(device)
    optimizer = create_optimizer(model, config)
    criterion = nn.CrossEntropyLoss()
    stopper = EarlyStopping(patience=config.patience, mode="min")

    if verbose:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"PriceDirectionModel — {n_params:,} parameters")

    # ------------------------------------------------------------------
    # 5. Training loop
    # ------------------------------------------------------------------
    history: list[dict] = []
    best_val_loss = float("inf")

    for epoch in range(1, config.epochs + 1):
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_result = evaluate_epoch(model, val_loader, criterion, device)
        val_cls = classification_metrics(
            val_result["y_true"], val_result["y_pred"], val_result["y_prob"]
        )

        row = {
            "epoch": epoch,
            **train_metrics,
            "val_loss": val_result["loss"],
            **{f"val_{k}": v for k, v in val_cls.items()},
        }
        history.append(row)

        if val_result["loss"] < best_val_loss:
            best_val_loss = val_result["loss"]
            save_checkpoint(
                model,
                optimizer,
                epoch,
                {"val_loss": val_result["loss"], **val_cls},
                Path(save_dir) / "price_model_best.pt",
            )

        if verbose:
            print(
                f"  Epoch {epoch:02d} | "
                f"train_loss={train_metrics['train_loss']:.4f} "
                f"train_acc={train_metrics['train_acc']:.4f} | "
                f"val_loss={val_result['loss']:.4f} "
                f"val_acc={val_cls['accuracy']:.4f} "
                f"val_f1={val_cls['f1']:.4f}"
            )

        if stopper(val_result["loss"]):
            if verbose:
                print(f"  Early stopping at epoch {epoch}")
            break

    # ------------------------------------------------------------------
    # 6. Evaluate on test set (load best model)
    # ------------------------------------------------------------------
    best_path = Path(save_dir) / "price_model_best.pt"
    if best_path.exists():
        ckpt = torch.load(best_path, weights_only=False, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])

    test_result = evaluate_epoch(model, test_loader, criterion, device)
    test_cls = classification_metrics(
        test_result["y_true"], test_result["y_pred"], test_result["y_prob"]
    )
    baseline_acc = majority_baseline_accuracy(test_result["y_true"])

    if verbose:
        print(f"\n  Test results (best model):")
        print(f"    Loss: {test_result['loss']:.4f}")
        for k, v in test_cls.items():
            print(f"    {k}: {v:.4f}")
        print(f"    Majority baseline accuracy: {baseline_acc:.4f}")

    return {
        "model": model,
        "history": history,
        "test_metrics": test_cls,
        "test_loss": test_result["loss"],
        "baseline_accuracy": baseline_acc,
        "test_predictions": {
            "y_true": test_result["y_true"],
            "y_pred": test_result["y_pred"],
            "y_prob": test_result["y_prob"],
        },
        "config": config,
        "device": device,
    }
