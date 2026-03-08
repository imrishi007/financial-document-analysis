"""Phase 6 — Train Multimodal Fusion Model.

Pipeline
--------
1. (Optional) extract embeddings from all pretrained models
2. Load pre-extracted embeddings into ``FusionEmbeddingDataset``
3. Temporal split: train ≤ 2022, val = 2023, test ≥ 2024
4. Train ``MultimodalFusionModel`` with multi-task loss:
       L = λ_dir × CE(direction) + λ_vol × MSE(volatility) + λ_surp × CE(surprise)
5. Evaluate all tasks on test set
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from src.data.fusion_dataset import FusionEmbeddingDataset
from src.evaluation.metrics import (
    classification_metrics,
    majority_baseline_accuracy,
    regression_metrics,
)
from src.models.fusion_model import MultimodalFusionModel
from src.train.common import (
    EarlyStopping,
    TrainingConfig,
    save_checkpoint,
)
from src.utils.seed import set_global_seed


# ======================================================================
# Data helpers
# ======================================================================


def _split_by_date(
    dataset: FusionEmbeddingDataset,
) -> tuple[list[int], list[int], list[int]]:
    """Temporal split: train ≤ 2022, val = 2023, test ≥ 2024."""
    train, val, test = [], [], []
    for i in range(len(dataset)):
        date_str = dataset.dates[i]
        if date_str <= "2022-12-31":
            train.append(i)
        elif date_str <= "2023-12-31":
            val.append(i)
        else:
            test.append(i)
    return train, val, test


# ======================================================================
# Train / evaluate one epoch
# ======================================================================


def train_one_epoch(
    model: MultimodalFusionModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    *,
    lambda_dir: float = 1.0,
    lambda_vol: float = 0.5,
    lambda_surp: float = 1.0,
) -> dict[str, float]:
    """Train for one epoch with multi-task masked loss."""
    model.train()
    ce_loss = nn.CrossEntropyLoss(reduction="none")
    mse_loss = nn.MSELoss(reduction="none")

    loss_sum = 0.0
    dir_correct, dir_total = 0, 0
    surp_correct, surp_total = 0, 0
    vol_se_sum, vol_total = 0.0, 0

    for batch in loader:
        price = batch["price_emb"].to(device)
        gat = batch["gat_emb"].to(device)
        doc = batch["doc_emb"].to(device)
        news = batch["news_emb"].to(device)
        surp = batch["surprise_feat"].to(device)
        mask = batch["modality_mask"].to(device)

        dir_label = batch["direction_label"].to(device)
        vol_target = batch["volatility_target"].to(device)
        surp_target = batch["surprise_target"].to(device)

        out = model(price, gat, doc, news, surp, mask)

        # --- Direction loss (always valid when label >= 0) ---
        dir_valid = dir_label >= 0
        if dir_valid.any():
            dl = ce_loss(out["direction_logits"][dir_valid], dir_label[dir_valid])
            dir_loss = dl.mean()
            preds = out["direction_logits"][dir_valid].argmax(1)
            dir_correct += (preds == dir_label[dir_valid]).sum().item()
            dir_total += dir_valid.sum().item()
        else:
            dir_loss = torch.tensor(0.0, device=device)

        # --- Volatility loss (valid when not NaN) ---
        vol_valid = ~torch.isnan(vol_target)
        if vol_valid.any():
            vl = mse_loss(out["volatility_pred"][vol_valid], vol_target[vol_valid])
            vol_loss = vl.mean()
            vol_se_sum += vl.sum().item()
            vol_total += vol_valid.sum().item()
        else:
            vol_loss = torch.tensor(0.0, device=device)

        # --- Surprise loss (valid when target >= 0, very sparse) ---
        surp_valid = surp_target >= 0
        if surp_valid.any():
            sl = ce_loss(out["surprise_logits"][surp_valid], surp_target[surp_valid])
            surp_loss = sl.mean()
            s_preds = out["surprise_logits"][surp_valid].argmax(1)
            surp_correct += (s_preds == surp_target[surp_valid]).sum().item()
            surp_total += surp_valid.sum().item()
        else:
            surp_loss = torch.tensor(0.0, device=device)

        # --- Combined loss ---
        total_loss = (
            lambda_dir * dir_loss + lambda_vol * vol_loss + lambda_surp * surp_loss
        )

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        loss_sum += total_loss.item() * price.size(0)

    n = max(dir_total, 1)
    return {
        "train_loss": loss_sum / max(len(loader.dataset), 1),
        "train_dir_acc": dir_correct / n,
        "train_vol_mse": vol_se_sum / max(vol_total, 1),
        "train_surp_acc": surp_correct / max(surp_total, 1),
    }


@torch.no_grad()
def evaluate(
    model: MultimodalFusionModel,
    loader: DataLoader,
    device: str,
    *,
    lambda_dir: float = 1.0,
    lambda_vol: float = 0.5,
    lambda_surp: float = 1.0,
) -> dict[str, Any]:
    """Evaluate and return detailed predictions."""
    model.eval()
    ce_loss = nn.CrossEntropyLoss(reduction="none")
    mse_loss = nn.MSELoss(reduction="none")

    loss_sum = 0.0
    dir_preds, dir_probs, dir_labels = [], [], []
    vol_preds, vol_labels = [], []
    surp_preds, surp_probs, surp_labels = [], [], []
    all_gates = []

    for batch in loader:
        price = batch["price_emb"].to(device)
        gat = batch["gat_emb"].to(device)
        doc = batch["doc_emb"].to(device)
        news = batch["news_emb"].to(device)
        surp = batch["surprise_feat"].to(device)
        mask = batch["modality_mask"].to(device)

        dir_label = batch["direction_label"].to(device)
        vol_target = batch["volatility_target"].to(device)
        surp_target = batch["surprise_target"].to(device)

        out = model(price, gat, doc, news, surp, mask)
        all_gates.append(out["gate_weights"].cpu())

        # Direction
        dir_valid = dir_label >= 0
        if dir_valid.any():
            dl = ce_loss(out["direction_logits"][dir_valid], dir_label[dir_valid])
            dir_loss = dl.mean()
            probs = torch.softmax(out["direction_logits"][dir_valid], dim=1)[:, 1]
            dir_preds.extend(
                out["direction_logits"][dir_valid].argmax(1).cpu().tolist()
            )
            dir_probs.extend(probs.cpu().tolist())
            dir_labels.extend(dir_label[dir_valid].cpu().tolist())
        else:
            dir_loss = torch.tensor(0.0, device=device)

        # Volatility
        vol_valid = ~torch.isnan(vol_target)
        if vol_valid.any():
            vl = mse_loss(out["volatility_pred"][vol_valid], vol_target[vol_valid])
            vol_loss = vl.mean()
            vol_preds.extend(out["volatility_pred"][vol_valid].cpu().tolist())
            vol_labels.extend(vol_target[vol_valid].cpu().tolist())
        else:
            vol_loss = torch.tensor(0.0, device=device)

        # Surprise
        surp_valid = surp_target >= 0
        if surp_valid.any():
            sl = ce_loss(out["surprise_logits"][surp_valid], surp_target[surp_valid])
            surp_loss = sl.mean()
            sprobs = torch.softmax(out["surprise_logits"][surp_valid], dim=1)[:, 1]
            surp_preds.extend(
                out["surprise_logits"][surp_valid].argmax(1).cpu().tolist()
            )
            surp_probs.extend(sprobs.cpu().tolist())
            surp_labels.extend(surp_target[surp_valid].cpu().tolist())
        else:
            surp_loss = torch.tensor(0.0, device=device)

        total_loss = (
            lambda_dir * dir_loss + lambda_vol * vol_loss + lambda_surp * surp_loss
        )
        loss_sum += total_loss.item() * price.size(0)

    results: dict[str, Any] = {
        "loss": loss_sum / max(len(loader.dataset), 1),
    }

    # Direction metrics
    if dir_labels:
        results["direction"] = classification_metrics(
            np.array(dir_labels),
            np.array(dir_preds),
            np.array(dir_probs),
        )
        results["direction_baseline"] = majority_baseline_accuracy(np.array(dir_labels))

    # Volatility metrics
    if vol_labels:
        results["volatility"] = regression_metrics(
            np.array(vol_labels),
            np.array(vol_preds),
        )

    # Surprise metrics
    if surp_labels and len(set(surp_labels)) > 1:
        results["surprise"] = classification_metrics(
            np.array(surp_labels),
            np.array(surp_preds),
            np.array(surp_probs),
        )
        results["surprise_baseline"] = majority_baseline_accuracy(np.array(surp_labels))

    # Gate analysis
    if all_gates:
        gates_cat = torch.cat(all_gates, dim=0)  # [N, 5]
        results["mean_gate_weights"] = gates_cat.mean(0).tolist()

    return results


# ======================================================================
# Full pipeline
# ======================================================================


def run_fusion_training(
    embeddings_path: str | Path = "data/embeddings/fusion_embeddings.pt",
    config: Optional[TrainingConfig] = None,
    save_dir: str | Path = "models",
    lambda_dir: float = 1.0,
    lambda_vol: float = 0.5,
    lambda_surp: float = 1.0,
    verbose: bool = True,
) -> dict[str, Any]:
    """End-to-end fusion model training on pre-extracted embeddings.

    Because embeddings are pre-computed dense tensors, this trains very fast
    (seconds per epoch) with large batch sizes.
    """
    if config is None:
        config = TrainingConfig(
            epochs=60,
            patience=10,
            batch_size=512,
            learning_rate=5e-4,
            weight_decay=1e-3,
            seed=42,
        )

    device = config.resolve_device()
    set_global_seed(config.seed)
    t0 = time.time()

    # ------------------------------------------------------------------
    # 1. Load dataset
    # ------------------------------------------------------------------
    dataset = FusionEmbeddingDataset(embeddings_path)
    train_idx, val_idx, test_idx = _split_by_date(dataset)

    # Filter to samples with valid direction labels
    train_idx = [i for i in train_idx if dataset.direction_label[i] >= 0]
    val_idx = [i for i in val_idx if dataset.direction_label[i] >= 0]
    test_idx = [i for i in test_idx if dataset.direction_label[i] >= 0]

    if verbose:
        print(f"Device: {device}")
        print(f"Dataset: {len(dataset)} total")
        print(
            f"Split — train: {len(train_idx)}, val: {len(val_idx)}, test: {len(test_idx)}"
        )

    train_loader = DataLoader(
        Subset(dataset, train_idx),
        batch_size=config.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        Subset(dataset, val_idx),
        batch_size=config.batch_size * 2,
        pin_memory=True,
    )
    test_loader = DataLoader(
        Subset(dataset, test_idx),
        batch_size=config.batch_size * 2,
        pin_memory=True,
    )

    # ------------------------------------------------------------------
    # 2. Build model
    # ------------------------------------------------------------------
    model = MultimodalFusionModel(
        price_dim=256,
        gat_dim=256,
        doc_dim=768,
        news_dim=512,
        surprise_dim=3,
        proj_dim=128,
        hidden_dim=256,
        dropout=0.3,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"MultimodalFusionModel — {total_params:,} parameters")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.epochs,
        eta_min=1e-6,
    )
    stopper = EarlyStopping(patience=config.patience, mode="min")

    # ------------------------------------------------------------------
    # 3. Training loop
    # ------------------------------------------------------------------
    history: list[dict] = []
    best_val_loss = float("inf")

    for epoch in range(1, config.epochs + 1):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            lambda_dir=lambda_dir,
            lambda_vol=lambda_vol,
            lambda_surp=lambda_surp,
        )
        val_result = evaluate(
            model,
            val_loader,
            device,
            lambda_dir=lambda_dir,
            lambda_vol=lambda_vol,
            lambda_surp=lambda_surp,
        )
        scheduler.step()

        row = {"epoch": epoch, **train_metrics, "val_loss": val_result["loss"]}
        if "direction" in val_result:
            row["val_dir_acc"] = val_result["direction"]["accuracy"]
            row["val_dir_auc"] = val_result["direction"].get("auc", 0.5)
        history.append(row)

        if val_result["loss"] < best_val_loss:
            best_val_loss = val_result["loss"]
            save_checkpoint(
                model,
                optimizer,
                epoch,
                {"val_loss": val_result["loss"], **val_result.get("direction", {})},
                Path(save_dir) / "fusion_model_best.pt",
            )

        if verbose:
            val_acc = row.get("val_dir_acc", 0)
            val_auc = row.get("val_dir_auc", 0)
            print(
                f"  Epoch {epoch:02d} | "
                f"train_loss={train_metrics['train_loss']:.4f} "
                f"dir_acc={train_metrics['train_dir_acc']:.4f} | "
                f"val_loss={val_result['loss']:.4f} "
                f"val_dir_acc={val_acc:.4f} val_auc={val_auc:.4f}"
            )

        if stopper(val_result["loss"]):
            if verbose:
                print(f"  Early stopping at epoch {epoch}")
            break

    # ------------------------------------------------------------------
    # 4. Test evaluation
    # ------------------------------------------------------------------
    best_path = Path(save_dir) / "fusion_model_best.pt"
    if best_path.exists():
        ckpt = torch.load(best_path, weights_only=False, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])

    test_result = evaluate(
        model,
        test_loader,
        device,
        lambda_dir=lambda_dir,
        lambda_vol=lambda_vol,
        lambda_surp=lambda_surp,
    )

    if verbose:
        print(f"\n{'='*60}")
        print("  TEST RESULTS")
        print(f"{'='*60}")
        if "direction" in test_result:
            d = test_result["direction"]
            bl = test_result.get("direction_baseline", 0)
            print(
                f"  Direction:  acc={d['accuracy']:.4f}  f1={d['f1']:.4f}  "
                f"auc={d['auc']:.4f}  baseline={bl:.4f}"
            )
        if "volatility" in test_result:
            v = test_result["volatility"]
            print(f"  Volatility: rmse={v['rmse']:.4f}  r2={v['r2']:.4f}")
        if "surprise" in test_result:
            s = test_result["surprise"]
            bl = test_result.get("surprise_baseline", 0)
            print(
                f"  Surprise:   acc={s['accuracy']:.4f}  f1={s['f1']:.4f}  "
                f"auc={s['auc']:.4f}  baseline={bl:.4f}"
            )
        if "mean_gate_weights" in test_result:
            names = ["price", "gat", "doc", "news", "surprise"]
            gw = test_result["mean_gate_weights"]
            print(f"\n  Gate weights (test):")
            for n, w in zip(names, gw):
                print(f"    {n:12s} {w:.4f}")
        print(f"\n  Total time: {time.time()-t0:.1f}s")

    return {
        "model": model,
        "history": history,
        "test_result": test_result,
        "config": config,
        "time_seconds": time.time() - t0,
        "n_train": len(train_idx),
        "n_val": len(val_idx),
        "n_test": len(test_idx),
    }
