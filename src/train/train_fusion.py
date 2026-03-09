"""Phase 6 -- Train Multimodal Fusion Model (V2).

Pipeline
--------
1. Load pre-extracted embeddings into ``FusionEmbeddingDataset``
2. Temporal split: train <= 2022, val = 2023, test >= 2024
3. Train ``MultimodalFusionModel`` with combined loss:
       L = 0.7 * ListNet(direction) + 0.3 * MSE(volatility)
4. Evaluate on test set with 60-day direction as primary metric

V2 changes:
- 4 modalities (no news): price, gat, doc, macro
- Surprise features as gating input (5-d), NOT a modality
- ListNet ranking loss as primary objective
- 60-day direction as primary target
- AMP + GPU optimization
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
from src.models.losses import CombinedLoss
from src.train.common import (
    EarlyStopping,
    TrainingConfig,
    save_checkpoint,
    make_dataloader,
)
from src.utils.seed import set_global_seed
from src.utils.gpu import setup_gpu, log_gpu_usage, create_grad_scaler


# ======================================================================
# Data helpers
# ======================================================================


def _split_by_date(
    dataset: FusionEmbeddingDataset,
) -> tuple[list[int], list[int], list[int]]:
    """Temporal split: train <= 2022, val = 2023, test >= 2024."""
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
    criterion: CombinedLoss,
    device: str,
    scaler: torch.amp.GradScaler | None = None,
    use_amp: bool = True,
) -> dict[str, float]:
    """Train for one epoch with ListNet + MSE combined loss."""
    model.train()

    loss_sum = 0.0
    dir_correct, dir_total = 0, 0
    vol_se_sum, vol_total = 0.0, 0

    for batch in loader:
        price = batch["price_emb"].to(device, non_blocking=True)
        gat = batch["gat_emb"].to(device, non_blocking=True)
        doc = batch["doc_emb"].to(device, non_blocking=True)
        macro = batch["macro_emb"].to(device, non_blocking=True)
        surp = batch["surprise_feat"].to(device, non_blocking=True)
        mask = batch["modality_mask"].to(device, non_blocking=True)

        dir_label = batch["direction_label"].to(device, non_blocking=True)
        vol_target = batch["volatility_target"].to(device, non_blocking=True)
        date_idx = batch["date_index"].to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=use_amp):
            out = model(price, gat, doc, macro, surp, mask)

            losses = criterion(
                out["direction_logits"],
                dir_label,
                out["volatility_pred"],
                vol_target,
                date_idx,
            )
            total_loss = losses["total"]

        if scaler is not None:
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        optimizer.zero_grad()

        loss_sum += total_loss.item() * price.size(0)

        # Track direction accuracy
        dir_valid = dir_label >= 0
        if dir_valid.any():
            preds = out["direction_logits"][dir_valid].argmax(1)
            dir_correct += (preds == dir_label[dir_valid]).sum().item()
            dir_total += dir_valid.sum().item()

        # Track volatility MSE
        vol_valid = ~torch.isnan(vol_target)
        if vol_valid.any():
            with torch.no_grad():
                vol_se = (
                    out["volatility_pred"][vol_valid] - vol_target[vol_valid]
                ).pow(2)
                vol_se_sum += vol_se.sum().item()
                vol_total += vol_valid.sum().item()

    return {
        "train_loss": loss_sum / max(len(loader.dataset), 1),
        "train_dir_acc": dir_correct / max(dir_total, 1),
        "train_vol_mse": vol_se_sum / max(vol_total, 1),
    }


@torch.no_grad()
def evaluate(
    model: MultimodalFusionModel,
    loader: DataLoader,
    criterion: CombinedLoss,
    device: str,
    use_amp: bool = True,
) -> dict[str, Any]:
    """Evaluate and return detailed predictions."""
    model.eval()

    loss_sum = 0.0
    dir_preds, dir_probs, dir_labels = [], [], []
    dir_logits_list = []
    vol_preds, vol_labels = [], []
    all_gates = []

    for batch in loader:
        price = batch["price_emb"].to(device, non_blocking=True)
        gat = batch["gat_emb"].to(device, non_blocking=True)
        doc = batch["doc_emb"].to(device, non_blocking=True)
        macro = batch["macro_emb"].to(device, non_blocking=True)
        surp = batch["surprise_feat"].to(device, non_blocking=True)
        mask = batch["modality_mask"].to(device, non_blocking=True)

        dir_label = batch["direction_label"].to(device, non_blocking=True)
        vol_target = batch["volatility_target"].to(device, non_blocking=True)
        date_idx = batch["date_index"].to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=use_amp):
            out = model(price, gat, doc, macro, surp, mask)
            losses = criterion(
                out["direction_logits"],
                dir_label,
                out["volatility_pred"],
                vol_target,
                date_idx,
            )

        all_gates.append(out["gate_weights"].float().cpu())
        loss_sum += losses["total"].item() * price.size(0)

        # Direction
        dir_valid = dir_label >= 0
        if dir_valid.any():
            probs = torch.softmax(out["direction_logits"][dir_valid].float(), dim=1)[
                :, 1
            ]
            dir_preds.extend(
                out["direction_logits"][dir_valid].float().argmax(1).cpu().tolist()
            )
            dir_probs.extend(probs.cpu().tolist())
            dir_labels.extend(dir_label[dir_valid].cpu().tolist())
            dir_logits_list.append(out["direction_logits"][dir_valid].float().cpu())

        # Volatility
        vol_valid = ~torch.isnan(vol_target)
        if vol_valid.any():
            vol_preds.extend(out["volatility_pred"][vol_valid].float().cpu().tolist())
            vol_labels.extend(vol_target[vol_valid].cpu().tolist())

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
        results["direction_logits"] = torch.cat(dir_logits_list, dim=0).numpy()
        results["direction_labels"] = np.array(dir_labels)
        results["direction_probs"] = np.array(dir_probs)

    # Volatility metrics
    if vol_labels:
        results["volatility"] = regression_metrics(
            np.array(vol_labels),
            np.array(vol_preds),
        )

    # Gate analysis
    if all_gates:
        gates_cat = torch.cat(all_gates, dim=0)
        results["mean_gate_weights"] = gates_cat.mean(0).tolist()

    return results


# ======================================================================
# Full pipeline
# ======================================================================


def run_fusion_training(
    embeddings_path: str | Path = "data/embeddings/fusion_embeddings.pt",
    config: Optional[TrainingConfig] = None,
    save_dir: str | Path = "models",
    lambda_dir: float = 0.7,
    lambda_vol: float = 0.3,
    verbose: bool = True,
) -> dict[str, Any]:
    """End-to-end fusion model training on pre-extracted embeddings.

    Uses ListNet ranking loss for direction + MSE for volatility.
    60-day direction is the PRIMARY target.
    """
    if config is None:
        config = TrainingConfig(
            epochs=60,
            patience=10,
            batch_size=4096,
            learning_rate=5e-4,
            weight_decay=1e-3,
            seed=42,
            use_amp=True,
        )

    gpu_device = setup_gpu(verbose=verbose)
    device = str(gpu_device)
    set_global_seed(config.seed)
    use_amp = config.use_amp and device == "cuda"
    amp_scaler = create_grad_scaler() if use_amp else None
    t0 = time.time()

    # ------------------------------------------------------------------
    # 1. Load dataset and pre-load to GPU for minimal overhead
    # ------------------------------------------------------------------
    dataset = FusionEmbeddingDataset(embeddings_path)
    if device == "cuda":
        dataset.to_device(device)
        if verbose:
            log_gpu_usage("  After GPU preload: ")
    train_idx, val_idx, test_idx = _split_by_date(dataset)

    # Filter to samples with valid direction labels
    train_idx = [i for i in train_idx if dataset.direction_label[i] >= 0]
    val_idx = [i for i in val_idx if dataset.direction_label[i] >= 0]
    test_idx = [i for i in test_idx if dataset.direction_label[i] >= 0]

    if verbose:
        print(f"Device: {device} | AMP: {use_amp}")
        print(f"Dataset: {len(dataset)} total")
        print(
            f"Split -- train: {len(train_idx)}, val: {len(val_idx)}, test: {len(test_idx)}"
        )

    # When data is pre-loaded to GPU, disable pin_memory to avoid overhead
    dl_config = TrainingConfig(
        **{**config.__dict__, "pin_memory": device != "cuda"}
    )

    train_loader = make_dataloader(
        Subset(dataset, train_idx),
        batch_size=config.batch_size,
        shuffle=True,
        config=dl_config,
    )
    val_loader = make_dataloader(
        Subset(dataset, val_idx),
        batch_size=config.batch_size * 2,
        config=dl_config,
    )
    test_loader = make_dataloader(
        Subset(dataset, test_idx),
        batch_size=config.batch_size * 2,
        config=dl_config,
    )

    # ------------------------------------------------------------------
    # 2. Build model
    # ------------------------------------------------------------------
    model = MultimodalFusionModel(
        price_dim=256,
        gat_dim=256,
        doc_dim=768,
        macro_dim=32,
        surprise_dim=5,
        proj_dim=128,
        hidden_dim=256,
        dropout=0.3,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"MultimodalFusionModel -- {total_params:,} parameters")
        log_gpu_usage("  After model load: ")

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

    # Combined loss: ListNet + MSE
    criterion = CombinedLoss(lambda_dir=lambda_dir, lambda_vol=lambda_vol)

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
            criterion,
            device,
            scaler=amp_scaler,
            use_amp=use_amp,
        )
        val_result = evaluate(model, val_loader, criterion, device, use_amp=use_amp)
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
            if epoch == 1:
                log_gpu_usage("  ")
            elif epoch % 5 == 0:
                peak = torch.cuda.max_memory_allocated(0) / 1e9 if device == "cuda" else 0
                lr_now = optimizer.param_groups[0]["lr"]
                print(f"    VRAM peak: {peak:.3f} GB | LR: {lr_now:.6f}")

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

    test_result = evaluate(model, test_loader, criterion, device, use_amp=use_amp)

    if verbose:
        print(f"\n{'='*60}")
        print("  TEST RESULTS (60-day Direction, ListNet Loss)")
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
        if "mean_gate_weights" in test_result:
            names = ["price", "gat", "doc", "macro"]
            gw = test_result["mean_gate_weights"]
            print(f"\n  Gate weights (test):")
            for n, w in zip(names, gw):
                print(f"    {n:12s} {w:.4f}")
        print(f"\n  Total time: {time.time()-t0:.1f}s")
        log_gpu_usage("  Final: ")

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
