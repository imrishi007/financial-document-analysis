"""Training pipeline for DocumentDirectionModel (FinBERT + AttentionPooling).

The document model predicts stock direction from 10-K filings. Since each
filing is a single sample (95 total), the model is trained with FinBERT
frozen and only the attention-pooling + classifier head learns.

Usage::

    from src.train.train_document import run_document_training
    results = run_document_training(
        processed_dir="data/processed",
        targets_dir="data/targets",
    )
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from src.data.document_dataset import DocumentChunkDataset
from src.evaluation.metrics import classification_metrics, majority_baseline_accuracy
from src.models.document_model import DocumentDirectionModel
from src.train.common import (
    EarlyStopping,
    TrainingConfig,
    create_optimizer,
    save_checkpoint,
)
from src.utils.seed import set_global_seed


# ---------------------------------------------------------------------------
# Document-level classification dataset
# ---------------------------------------------------------------------------


class DocumentDirectionDataset(Dataset):
    """Pairs each 10-K filing with a direction label for the following year.

    For filing (ticker, year), the label is the *majority* daily direction
    over the first 120 trading days of year+1 (≈ 1 semester). This gives a
    reasonable signal for whether the 10-K content is associated with a
    positive or negative outlook.

    Parameters
    ----------
    doc_dataset : Pre-built ``DocumentChunkDataset``.
    direction_df : Multi-horizon direction labels (with ``direction_1d_id``).
    lookforward_days : Number of trading days after Jan 1 of year+1 to
        aggregate the direction label over.
    """

    def __init__(
        self,
        doc_dataset: DocumentChunkDataset,
        direction_df: pd.DataFrame,
        lookforward_days: int = 120,
    ) -> None:
        self.doc_ds = doc_dataset
        dir_df = direction_df.copy()
        dir_df["date"] = pd.to_datetime(dir_df["date"])

        self._samples: list[dict] = []

        for idx in range(len(doc_dataset)):
            s = doc_dataset._samples[idx]
            ticker, year = s["ticker"], s["year"]

            # Label = majority daily direction in the first N days of year+1
            start = pd.Timestamp(f"{year + 1}-01-01")
            end = start + pd.Timedelta(days=int(lookforward_days * 1.5))  # buffer
            mask = (
                (dir_df["ticker"] == ticker)
                & (dir_df["date"] >= start)
                & (dir_df["date"] <= end)
            )
            subset = dir_df.loc[mask].head(lookforward_days)

            if len(subset) < 20:
                continue  # skip filings without enough future data

            label = int(subset["direction_1d_id"].mode().iloc[0])

            self._samples.append(
                {
                    "doc_idx": idx,
                    "label": label,
                    "ticker": ticker,
                    "year": year,
                }
            )

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> dict:
        s = self._samples[idx]
        doc_sample = self.doc_ds[s["doc_idx"]]
        return {
            "input_ids": doc_sample["input_ids"],  # [C, S]
            "attention_mask": doc_sample["attention_mask"],  # [C, S]
            "num_chunks": doc_sample["num_chunks"],
            "label": torch.tensor(s["label"], dtype=torch.long),
            "ticker": s["ticker"],
            "year": s["year"],
        }


# ---------------------------------------------------------------------------
# Model wrapper for chunk-level encoding
# ---------------------------------------------------------------------------


def _encode_chunked_batch(
    model: DocumentDirectionModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    num_chunks: torch.Tensor,
    device: str,
    chunk_batch_size: int = 8,
) -> torch.Tensor:
    """Encode a batch of chunked documents through FinBERT and aggregate.

    Each sample has [C, S] chunks. We process chunks in mini-batches of
    ``chunk_batch_size`` to keep VRAM usage manageable on 4GB GPUs, then
    mean-pool per-chunk CLS embeddings per document.
    """
    batch_size, max_chunks, seq_len = input_ids.shape
    all_embeddings = []

    for i in range(batch_size):
        n = num_chunks[i].item()
        if n == 0:
            n = 1  # at least 1 chunk

        ids = input_ids[i, :n]  # [n, S] — stay on CPU
        mask = attention_mask[i, :n]  # [n, S]

        # Process in mini-batches to limit VRAM
        chunk_cls_list = []
        for start in range(0, n, chunk_batch_size):
            end = min(start + chunk_batch_size, n)
            mini_ids = ids[start:end].to(device)
            mini_mask = mask[start:end].to(device)

            with torch.no_grad():
                enc_out = model.encoder(input_ids=mini_ids, attention_mask=mini_mask)
                chunk_cls_list.append(enc_out.last_hidden_state[:, 0, :].cpu())

            del mini_ids, mini_mask, enc_out

        chunk_cls = torch.cat(chunk_cls_list, dim=0)  # [n, H]
        embedding = chunk_cls.mean(dim=0)  # [H]
        all_embeddings.append(embedding)

    return torch.stack(all_embeddings)  # [B, H]


# ---------------------------------------------------------------------------
# Train / evaluate helpers
# ---------------------------------------------------------------------------


def train_one_epoch_doc(
    model: DocumentDirectionModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str,
) -> dict[str, float]:
    """Train classifier head for one epoch (encoder is frozen)."""
    model.train()
    loss_sum, correct, total = 0.0, 0, 0

    for batch in loader:
        labels = batch["label"].to(device)

        # Get frozen embeddings (returned on CPU to save VRAM)
        embeddings = _encode_chunked_batch(
            model,
            batch["input_ids"],
            batch["attention_mask"],
            batch["num_chunks"],
            device,
        )
        embeddings = embeddings.detach().to(device)  # move to GPU for head

        # Forward through trainable head
        logits = model.head(embeddings)
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
def evaluate_doc(
    model: DocumentDirectionModel,
    loader: DataLoader,
    criterion: nn.Module,
    device: str,
) -> dict[str, Any]:
    """Evaluate document model and return predictions."""
    model.eval()
    loss_sum, total = 0.0, 0
    all_preds, all_labels, all_probs = [], [], []

    for batch in loader:
        labels = batch["label"].to(device)
        embeddings = _encode_chunked_batch(
            model,
            batch["input_ids"],
            batch["attention_mask"],
            batch["num_chunks"],
            device,
        ).to(
            device
        )  # move from CPU to GPU for head
        logits = model.head(embeddings)
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


def run_document_training(
    processed_dir: str | Path = "data/processed",
    targets_dir: str | Path = "data/targets",
    backbone: str = "ProsusAI/finbert",
    config: Optional[TrainingConfig] = None,
    save_dir: str | Path = "models",
    verbose: bool = True,
) -> dict[str, Any]:
    """End-to-end document model training.

    FinBERT is frozen; only the classifier head is trained.
    """
    if config is None:
        config = TrainingConfig(epochs=30, patience=8, batch_size=4)

    device = config.resolve_device()
    set_global_seed(config.seed)
    if verbose:
        print(f"Device: {device} | Backbone: {backbone}")

    # ------------------------------------------------------------------
    # 1. Build document dataset
    # ------------------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(backbone)
    if verbose:
        print("Tokenizing 10-K filings (this may take a minute)...")

    doc_ds = DocumentChunkDataset(processed_dir, tokenizer, max_chunks=64)

    direction_df = pd.read_csv(Path(targets_dir) / "direction_labels_multi_horizon.csv")
    full_ds = DocumentDirectionDataset(doc_ds, direction_df, lookforward_days=120)

    if verbose:
        print(f"Document samples: {len(full_ds)}")

    # ------------------------------------------------------------------
    # 2. Time-based split (filing year)
    # ------------------------------------------------------------------
    train_idx, val_idx, test_idx = [], [], []
    for i, s in enumerate(full_ds._samples):
        if s["year"] <= 2021:
            train_idx.append(i)
        elif s["year"] <= 2022:
            val_idx.append(i)
        else:
            test_idx.append(i)

    from torch.utils.data import Subset

    train_ds = Subset(full_ds, train_idx)
    val_ds = Subset(full_ds, val_idx) if val_idx else None
    test_ds = Subset(full_ds, test_idx) if test_idx else None

    if verbose:
        print(
            f"Split — train: {len(train_idx)}, val: {len(val_idx)}, test: {len(test_idx)}"
        )

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size) if val_ds else None
    test_loader = DataLoader(test_ds, batch_size=config.batch_size) if test_ds else None

    # ------------------------------------------------------------------
    # 3. Build model (freeze encoder)
    # ------------------------------------------------------------------
    model = DocumentDirectionModel(backbone_name=backbone).to(device)
    for p in model.encoder.parameters():
        p.requires_grad = False

    optimizer = create_optimizer(model, config)
    criterion = nn.CrossEntropyLoss()
    stopper = EarlyStopping(patience=config.patience, mode="min")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"Parameters — total: {total_params:,}, trainable: {trainable:,}")

    # ------------------------------------------------------------------
    # 4. Training loop
    # ------------------------------------------------------------------
    history: list[dict] = []
    best_val_loss = float("inf")

    for epoch in range(1, config.epochs + 1):
        train_metrics = train_one_epoch_doc(
            model, train_loader, optimizer, criterion, device
        )

        row = {"epoch": epoch, **train_metrics}

        if val_loader is not None:
            val_result = evaluate_doc(model, val_loader, criterion, device)
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
                    Path(save_dir) / "document_model_best.pt",
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
    # 5. Evaluate on test
    # ------------------------------------------------------------------
    test_metrics: dict = {}
    baseline_acc = 0.0

    if test_loader is not None:
        best_path = Path(save_dir) / "document_model_best.pt"
        if best_path.exists():
            ckpt = torch.load(best_path, weights_only=False, map_location=device)
            model.load_state_dict(ckpt["model_state_dict"])

        test_result = evaluate_doc(model, test_loader, criterion, device)
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
