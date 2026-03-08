"""Training pipeline for fundamental surprise prediction (BEAT/MISS).

Uses price features before earnings announcements to predict whether the
company will beat or miss analyst expectations. This is a sparse-event
binary classification task.

Usage::

    from src.train.train_surprise import run_surprise_training
    results = run_surprise_training()
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset

from src.data.preprocessing import SplitConfig, create_time_splits, fit_scaler
from src.data.price_dataset import (
    ENGINEERED_FEATURES,
    load_price_csv_dir,
    prepare_price_features,
)
from src.evaluation.metrics import classification_metrics, majority_baseline_accuracy
from src.models.price_model import PriceDirectionModel
from src.train.common import (
    EarlyStopping,
    TrainingConfig,
    create_optimizer,
    save_checkpoint,
)
from src.utils.seed import set_global_seed


# ---------------------------------------------------------------------------
# Surprise dataset (price windows around earnings events)
# ---------------------------------------------------------------------------


class SurpriseDataset(Dataset):
    """Sliding price-window dataset filtered to earnings event dates only.

    For each earnings announcement, takes the preceding ``window_size``
    trading days of price features and predicts BEAT (1) vs MISS (0).

    Parameters
    ----------
    price_df : Feature-engineered and scaled price DataFrame.
    surprise_df : Fundamental surprise targets (from ``target_builder``).
    window_size : Number of lookback trading days.
    feature_cols : Price feature columns.
    """

    def __init__(
        self,
        price_df: pd.DataFrame,
        surprise_df: pd.DataFrame,
        window_size: int = 60,
        feature_cols: Optional[list[str]] = None,
    ) -> None:
        self.window_size = window_size
        self.feature_cols = feature_cols or ENGINEERED_FEATURES

        price_df = price_df.copy()
        surprise_df = surprise_df.copy()
        price_df["date"] = pd.to_datetime(price_df["date"])
        surprise_df["date"] = pd.to_datetime(surprise_df["date"])

        # Filter to actual earnings events (BEAT=1 or MISS=0, exclude NONE=-1)
        events = surprise_df[surprise_df["surprise_id"].isin([0, 1])].copy()

        self._samples: list[dict] = []

        for ticker, group in price_df.groupby("ticker"):
            group = group.sort_values("date").reset_index(drop=True)
            ticker_events = events[events["ticker"] == ticker]

            for _, ev_row in ticker_events.iterrows():
                ev_date = ev_row["date"]
                # Find the position of this date in the price data
                date_matches = group[group["date"] <= ev_date]
                if len(date_matches) < window_size:
                    continue

                window = date_matches.tail(window_size)
                self._samples.append(
                    {
                        "features": window[self.feature_cols].values.astype(np.float32),
                        "label": int(ev_row["surprise_id"]),
                        "ticker": str(ticker),
                        "date": str(ev_date.date()),
                    }
                )

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> dict:
        s = self._samples[idx]
        return {
            "features": torch.from_numpy(s["features"]),
            "label": torch.tensor(s["label"], dtype=torch.long),
            "ticker": s["ticker"],
            "date": s["date"],
        }


# ---------------------------------------------------------------------------
# Train / evaluate
# ---------------------------------------------------------------------------


def train_one_epoch_surprise(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str,
) -> dict[str, float]:
    model.train()
    loss_sum, correct, total = 0.0, 0, 0

    for batch in loader:
        features = batch["features"].to(device)
        labels = batch["label"].to(device)
        logits = model(features)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_sum += loss.item() * features.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += features.size(0)

    return {
        "train_loss": loss_sum / max(total, 1),
        "train_acc": correct / max(total, 1),
    }


@torch.no_grad()
def evaluate_surprise(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str,
) -> dict[str, Any]:
    model.eval()
    loss_sum, total = 0.0, 0
    all_preds, all_labels, all_probs = [], [], []

    for batch in loader:
        features = batch["features"].to(device)
        labels = batch["label"].to(device)
        logits = model(features)
        loss = criterion(logits, labels)

        probs = torch.softmax(logits, dim=1)[:, 1]
        loss_sum += loss.item() * features.size(0)
        all_preds.extend(logits.argmax(1).cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
        all_probs.extend(probs.cpu().tolist())
        total += features.size(0)

    return {
        "loss": loss_sum / max(total, 1),
        "y_true": np.array(all_labels),
        "y_pred": np.array(all_preds),
        "y_prob": np.array(all_probs),
    }


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------


def run_surprise_training(
    prices_dir: str | Path = "data/raw/prices",
    targets_dir: str | Path = "data/targets",
    config: Optional[TrainingConfig] = None,
    split_cfg: Optional[SplitConfig] = None,
    save_dir: str | Path = "models",
    verbose: bool = True,
) -> dict[str, Any]:
    """Train a price-based surprise predictor (BEAT vs MISS).

    Uses the same PriceDirectionModel architecture but with surprise labels.
    """
    if config is None:
        config = TrainingConfig(epochs=30, patience=8, batch_size=16)
    if split_cfg is None:
        split_cfg = SplitConfig()

    device = config.resolve_device()
    set_global_seed(config.seed)
    if verbose:
        print(f"Device: {device}")

    # ------------------------------------------------------------------
    # 1. Load and scale features
    # ------------------------------------------------------------------
    prices = load_price_csv_dir(prices_dir)
    price_feat = prepare_price_features(prices)

    surprise_df = pd.read_csv(Path(targets_dir) / "fundamental_surprise_targets.csv")

    dates_series = pd.to_datetime(price_feat["date"])
    train_mask = dates_series <= pd.Timestamp(split_cfg.train_end)
    scaler = fit_scaler(price_feat[train_mask], ENGINEERED_FEATURES)
    price_feat = scaler.transform(price_feat, ENGINEERED_FEATURES)

    # ------------------------------------------------------------------
    # 2. Build dataset
    # ------------------------------------------------------------------
    full_ds = SurpriseDataset(price_feat, surprise_df, window_size=60)

    if verbose:
        print(f"Earnings event samples: {len(full_ds)}")
        labels = [s["label"] for s in full_ds._samples]
        beat_pct = sum(labels) / len(labels) if labels else 0
        print(
            f"  BEAT: {sum(labels)}, MISS: {len(labels) - sum(labels)} ({beat_pct:.1%} BEAT)"
        )

    # ------------------------------------------------------------------
    # 3. Time-based split
    # ------------------------------------------------------------------
    dates = pd.Series(pd.to_datetime([s["date"] for s in full_ds._samples]))
    masks = create_time_splits(dates, split_cfg)

    train_idx = [i for i, m in enumerate(masks["train"]) if m]
    val_idx = [i for i, m in enumerate(masks["val"]) if m]
    test_idx = [i for i, m in enumerate(masks["test"]) if m]

    train_ds = Subset(full_ds, train_idx)
    val_ds = Subset(full_ds, val_idx) if val_idx else None
    test_ds = Subset(full_ds, test_idx) if test_idx else None

    if verbose:
        print(
            f"Split — train: {len(train_idx)}, val: {len(val_idx)}, test: {len(test_idx)}"
        )

    if len(train_idx) < 5:
        if verbose:
            print("Too few training samples for surprise prediction.")
        return {
            "model": None,
            "history": [],
            "test_metrics": {},
            "baseline_accuracy": 0.0,
        }

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size) if val_ds else None
    test_loader = DataLoader(test_ds, batch_size=config.batch_size) if test_ds else None

    # ------------------------------------------------------------------
    # 4. Build model (reuse PriceDirectionModel architecture)
    # ------------------------------------------------------------------
    model = PriceDirectionModel(num_features=len(ENGINEERED_FEATURES)).to(device)
    optimizer = create_optimizer(model, config)
    criterion = nn.CrossEntropyLoss()
    stopper = EarlyStopping(patience=config.patience, mode="min")

    # ------------------------------------------------------------------
    # 5. Training loop
    # ------------------------------------------------------------------
    history: list[dict] = []
    best_val_loss = float("inf")

    for epoch in range(1, config.epochs + 1):
        train_metrics = train_one_epoch_surprise(
            model, train_loader, optimizer, criterion, device
        )
        row = {"epoch": epoch, **train_metrics}

        if val_loader is not None:
            val_result = evaluate_surprise(model, val_loader, criterion, device)
            val_cls = classification_metrics(
                val_result["y_true"],
                val_result["y_pred"],
                val_result["y_prob"],
            )
            row.update(
                {
                    "val_loss": val_result["loss"],
                    **{f"val_{k}": v for k, v in val_cls.items()},
                }
            )

            if val_result["loss"] < best_val_loss:
                best_val_loss = val_result["loss"]
                save_checkpoint(
                    model,
                    optimizer,
                    epoch,
                    {"val_loss": val_result["loss"], **val_cls},
                    Path(save_dir) / "surprise_model_best.pt",
                )

            if stopper(val_result["loss"]):
                if verbose:
                    print(f"  Early stopping at epoch {epoch}")
                break

        history.append(row)
        if verbose:
            val_info = (
                f"val_loss={row.get('val_loss', 0):.4f} val_acc={row.get('val_accuracy', 0):.4f}"
                if val_loader
                else "no val"
            )
            print(
                f"  Epoch {epoch:02d} | train_loss={train_metrics['train_loss']:.4f} "
                f"train_acc={train_metrics['train_acc']:.4f} | {val_info}"
            )

    # ------------------------------------------------------------------
    # 6. Test evaluation
    # ------------------------------------------------------------------
    test_metrics: dict = {}
    baseline_acc = 0.0

    if test_loader is not None:
        best_path = Path(save_dir) / "surprise_model_best.pt"
        if best_path.exists():
            ckpt = torch.load(best_path, weights_only=False, map_location=device)
            model.load_state_dict(ckpt["model_state_dict"])

        test_result = evaluate_surprise(model, test_loader, criterion, device)
        test_metrics = classification_metrics(
            test_result["y_true"],
            test_result["y_pred"],
            test_result["y_prob"],
        )
        baseline_acc = majority_baseline_accuracy(test_result["y_true"])

        if verbose:
            print(f"\n  Test results:")
            for k, v in test_metrics.items():
                print(f"    {k}: {v:.4f}")
            print(f"    Majority baseline: {baseline_acc:.4f}")

    return {
        "model": model,
        "history": history,
        "test_metrics": test_metrics,
        "baseline_accuracy": baseline_acc,
        "config": config,
        "device": device,
        "n_train": len(train_idx),
        "n_val": len(val_idx),
        "n_test": len(test_idx),
    }
