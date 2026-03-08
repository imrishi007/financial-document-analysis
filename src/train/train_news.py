"""Training pipeline for NewsTemporalDirectionModel (FinBERT + BiGRU).

**Important limitation**: news data covers Nov 2024–Feb 2026 only, which falls
entirely in the test split.  Therefore this module provides:

1. ``extract_news_embeddings`` — extract FinBERT embeddings from available news.
2. ``evaluate_news_signal`` — evaluate how well news embeddings predict direction
   on available dates (test-split only, no proper train/val).
3. ``run_news_training`` — full pipeline that trains a small classifier on the
   available news embeddings using a within-test temporal split (demonstration).

For production use, the news encoder is best leveraged inside the multimodal
fusion (Phase 6) where date-aligned price data provides the training signal.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from transformers import AutoTokenizer

from src.data.news_dataset import NewsWindowDataset, load_news_articles
from src.evaluation.metrics import classification_metrics, majority_baseline_accuracy
from src.models.news_model import NewsTemporalDirectionModel
from src.train.common import (
    EarlyStopping,
    TrainingConfig,
    create_optimizer,
    save_checkpoint,
)
from src.utils.seed import set_global_seed


# ---------------------------------------------------------------------------
# News + direction aligned dataset
# ---------------------------------------------------------------------------


class NewsDirectionDataset(Dataset):
    """Wraps a ``NewsWindowDataset`` with direction labels.

    Each sample is a (ticker, date) news window paired with the 1-day
    direction label for that date.
    """

    def __init__(
        self,
        news_dataset: NewsWindowDataset,
        direction_df: pd.DataFrame,
    ) -> None:
        self.news_ds = news_dataset
        dir_df = direction_df.copy()
        dir_df["date"] = pd.to_datetime(dir_df["date"]).dt.strftime("%Y-%m-%d")

        # Build lookup: (ticker, date_str) -> direction_1d_id
        dir_lookup: dict[tuple[str, str], int] = {}
        for _, row in dir_df.iterrows():
            dir_lookup[(row["ticker"], row["date"])] = int(row["direction_1d_id"])

        self._samples: list[dict] = []
        for i in range(len(news_dataset)):
            s = news_dataset._samples[i]
            if s["num_articles"] == 0:
                continue  # skip dates with no news
            key = (s["ticker"], s["date"])
            label = dir_lookup.get(key)
            if label is None:
                continue
            self._samples.append({"news_idx": i, "label": label})

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> dict:
        s = self._samples[idx]
        news_sample = self.news_ds[s["news_idx"]]
        return {
            "input_ids": news_sample["input_ids"],  # [A, S]
            "attention_mask": news_sample["attention_mask"],  # [A, S]
            "num_articles": news_sample["num_articles"],
            "label": torch.tensor(s["label"], dtype=torch.long),
            "ticker": news_sample["ticker"],
            "date": news_sample["date"],
        }


# ---------------------------------------------------------------------------
# Train / evaluate
# ---------------------------------------------------------------------------


def train_one_epoch_news(
    model: NewsTemporalDirectionModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str,
    *,
    freeze_encoder: bool = True,
) -> dict[str, float]:
    """Train for one epoch. Encoder can be frozen or fine-tuned."""
    model.train()
    if freeze_encoder:
        model.encoder.eval()

    loss_sum, correct, total = 0.0, 0, 0

    for batch in loader:
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        if freeze_encoder:
            with torch.no_grad():
                article_emb = model.encode_articles(ids, mask)
            article_emb = article_emb.detach()
            temporal_out, _ = model.temporal(article_emb)
            logits = model.head(temporal_out[:, -1, :])
        else:
            logits = model(ids, mask)

        loss = criterion(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_sum += loss.item() * labels.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)

    return {
        "train_loss": loss_sum / max(total, 1),
        "train_acc": correct / max(total, 1),
    }


@torch.no_grad()
def evaluate_news(
    model: NewsTemporalDirectionModel,
    loader: DataLoader,
    criterion: nn.Module,
    device: str,
) -> dict[str, Any]:
    """Evaluate news model."""
    model.eval()
    loss_sum, total = 0.0, 0
    all_preds, all_labels, all_probs = [], [], []

    for batch in loader:
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        logits = model(ids, mask)
        loss = criterion(logits, labels)

        probs = torch.softmax(logits, dim=1)[:, 1]
        loss_sum += loss.item() * labels.size(0)
        all_preds.extend(logits.argmax(1).cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
        all_probs.extend(probs.cpu().tolist())
        total += labels.size(0)

    return {
        "loss": loss_sum / max(total, 1),
        "y_true": np.array(all_labels),
        "y_pred": np.array(all_preds),
        "y_prob": np.array(all_probs),
    }


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------


def run_news_training(
    news_csv: str | Path = "data/raw/news/news_articles.csv",
    prices_dir: str | Path = "data/raw/prices",
    targets_dir: str | Path = "data/targets",
    backbone: str = "ProsusAI/finbert",
    config: Optional[TrainingConfig] = None,
    save_dir: str | Path = "models",
    verbose: bool = True,
) -> dict[str, Any]:
    """Train the news model on available data.

    Because news covers only Nov 2024–Feb 2026 (test split), we use a
    temporal within-test split: first 60% for training, next 20% for
    validation, last 20% for testing. This is for demonstration purposes.
    """
    if config is None:
        config = TrainingConfig(epochs=15, patience=5, batch_size=8)

    device = config.resolve_device()
    set_global_seed(config.seed)
    if verbose:
        print(f"Device: {device} | Backbone: {backbone}")

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    news_df = load_news_articles(news_csv)
    direction_df = pd.read_csv(Path(targets_dir) / "direction_labels_multi_horizon.csv")

    # Build anchor dates from direction labels (test split only, where news exists)
    dir_dates = direction_df[["ticker", "date"]].copy()
    dir_dates["date"] = pd.to_datetime(dir_dates["date"])
    news_start = news_df["date"].min()
    anchors = dir_dates[dir_dates["date"] >= news_start].copy()

    if verbose:
        print(f"News articles: {len(news_df)}, anchor dates: {len(anchors)}")

    # ------------------------------------------------------------------
    # 2. Build datasets
    # ------------------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(backbone)
    if verbose:
        print("Tokenizing news articles...")

    news_ds = NewsWindowDataset(
        news_df,
        anchors,
        tokenizer,
        max_articles=16,
        max_length=128,
        window_days=7,
    )
    full_ds = NewsDirectionDataset(news_ds, direction_df)

    if verbose:
        print(f"News-direction samples (with articles): {len(full_ds)}")

    if len(full_ds) < 10:
        if verbose:
            print(
                "Too few samples with news articles for training. Returning empty results."
            )
        return {
            "model": None,
            "history": [],
            "test_metrics": {},
            "baseline_accuracy": 0.0,
            "n_samples": len(full_ds),
        }

    # ------------------------------------------------------------------
    # 3. Within-test temporal split (demonstration)
    # ------------------------------------------------------------------
    # Sort by date
    dates = [full_ds._samples[i]["label"] for i in range(len(full_ds))]
    sorted_news_dates = []
    for i in range(len(full_ds)):
        ns = full_ds.news_ds._samples[full_ds._samples[i]["news_idx"]]
        sorted_news_dates.append(ns["date"])
    order = np.argsort(sorted_news_dates)

    n = len(order)
    n_train = int(n * 0.6)
    n_val = int(n * 0.2)

    train_idx = order[:n_train].tolist()
    val_idx = order[n_train : n_train + n_val].tolist()
    test_idx = order[n_train + n_val :].tolist()

    train_ds = Subset(full_ds, train_idx)
    val_ds = Subset(full_ds, val_idx)
    test_ds = Subset(full_ds, test_idx)

    if verbose:
        print(
            f"Split — train: {len(train_idx)}, val: {len(val_idx)}, test: {len(test_idx)}"
        )

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size)
    test_loader = DataLoader(test_ds, batch_size=config.batch_size)

    # ------------------------------------------------------------------
    # 4. Build model (freeze encoder)
    # ------------------------------------------------------------------
    model = NewsTemporalDirectionModel(backbone_name=backbone).to(device)
    for p in model.encoder.parameters():
        p.requires_grad = False

    optimizer = create_optimizer(model, config)
    criterion = nn.CrossEntropyLoss()
    stopper = EarlyStopping(patience=config.patience, mode="min")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if verbose:
        print(f"Trainable parameters: {trainable:,}")

    # ------------------------------------------------------------------
    # 5. Training loop
    # ------------------------------------------------------------------
    history: list[dict] = []
    best_val_loss = float("inf")

    for epoch in range(1, config.epochs + 1):
        train_metrics = train_one_epoch_news(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            freeze_encoder=True,
        )
        val_result = evaluate_news(model, val_loader, criterion, device)
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
                Path(save_dir) / "news_model_best.pt",
            )

        if verbose:
            print(
                f"  Epoch {epoch:02d} | train_loss={train_metrics['train_loss']:.4f} "
                f"train_acc={train_metrics['train_acc']:.4f} | "
                f"val_loss={val_result['loss']:.4f} val_acc={val_cls['accuracy']:.4f}"
            )

        if stopper(val_result["loss"]):
            if verbose:
                print(f"  Early stopping at epoch {epoch}")
            break

    # ------------------------------------------------------------------
    # 6. Evaluate on test
    # ------------------------------------------------------------------
    best_path = Path(save_dir) / "news_model_best.pt"
    if best_path.exists():
        ckpt = torch.load(best_path, weights_only=False, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])

    test_result = evaluate_news(model, test_loader, criterion, device)
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
