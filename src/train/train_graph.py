"""Phase 5 -- Train Graph-Enhanced Direction Model (GAT).

Pipeline
--------
1. Load price features + direction labels
2. Build ``GraphSnapshotDataset`` (all 10 companies per date)
3. Load static graph -> edge_index, edge_weight
4. Split by date: train <= 2022, val = 2023, test >= 2024
5. Train ``GraphEnhancedModel`` (CNN-LSTM -> GAT -> classifier)
6. Evaluate with / without GAT (ablation baseline)

V2: AMP + GPU optimization, 60-day direction as primary target.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from src.data.graph_dataset import GraphSnapshotDataset, build_graph_snapshots
from src.data.graph_utils import load_graph, make_bidirectional
from src.data.preprocessing import fit_scaler
from src.evaluation.metrics import classification_metrics, majority_baseline_accuracy
from src.models.gat_model import GraphEnhancedModel
from src.train.common import (
    EarlyStopping,
    TrainingConfig,
    save_checkpoint,
)
from src.utils.gpu import setup_gpu, log_gpu_usage, create_grad_scaler


# ---------------------------------------------------------------------------
# Dataset construction with normalization & split
# ---------------------------------------------------------------------------


def _build_datasets(
    prices_dir: str | Path,
    targets_dir: str | Path,
    tickers: list[str],
    window_size: int = 60,
) -> tuple[GraphSnapshotDataset, list[int], list[int], list[int]]:
    """Build the full dataset and return temporal split indices."""
    price_df, targets_df, tickers = build_graph_snapshots(
        prices_dir, targets_dir, tickers, window_size
    )

    # Fit z-score scaler on train dates only (≤ 2022-12-31)
    from src.data.price_dataset import ENGINEERED_FEATURES

    train_mask = price_df["date"] <= "2022-12-31"
    scaler = fit_scaler(price_df.loc[train_mask], ENGINEERED_FEATURES)
    price_df = scaler.transform(price_df, ENGINEERED_FEATURES)

    dataset = GraphSnapshotDataset(
        price_df, targets_df, tickers, window_size=window_size
    )

    # Split by date
    train_idx, val_idx, test_idx = [], [], []
    for i, snap in enumerate(dataset._snapshots):
        d = snap["date"]
        if d <= "2022-12-31":
            train_idx.append(i)
        elif d <= "2023-12-31":
            val_idx.append(i)
        else:
            test_idx.append(i)

    return dataset, train_idx, val_idx, test_idx


# ---------------------------------------------------------------------------
# Train / Evaluate one epoch
# ---------------------------------------------------------------------------


def train_one_epoch(
    model: GraphEnhancedModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    device: str,
    *,
    use_amp: bool = False,
    scaler: torch.amp.GradScaler | None = None,
) -> dict[str, float]:
    """Train for one epoch over graph snapshots with optional AMP."""
    model.train()
    loss_sum, correct, total = 0.0, 0, 0

    for batch in loader:
        features = batch["features"].to(device, non_blocking=True)  # [B, N, T, F]
        labels = batch["labels"].to(device, non_blocking=True)  # [B, N]
        mask = batch["mask"].to(device, non_blocking=True)  # [B, N]

        B, N, T, F = features.shape

        # Process each snapshot in the batch
        batch_loss = torch.tensor(0.0, device=device)
        batch_correct = 0
        batch_total = 0

        with torch.amp.autocast("cuda", enabled=use_amp):
            for b in range(B):
                x = features[b]  # [N, T, F]
                y = labels[b]  # [N]
                m = mask[b]  # [N]

                if m.sum() == 0:
                    continue

                logits = model(x, edge_index, edge_weight)  # [N, 2]

                # Only compute loss for valid (non-masked) companies
                valid_logits = logits[m]
                valid_labels = y[m]

                loss = criterion(valid_logits, valid_labels)
                batch_loss = batch_loss + loss

                preds = valid_logits.argmax(dim=1)
                batch_correct += (preds == valid_labels).sum().item()
                batch_total += valid_labels.size(0)

        if batch_total == 0:
            continue

        avg_loss = batch_loss / B
        optimizer.zero_grad()

        if scaler is not None:
            scaler.scale(avg_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            avg_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        loss_sum += batch_loss.item()
        correct += batch_correct
        total += batch_total

    return {
        "train_loss": loss_sum / max(total, 1),
        "train_acc": correct / max(total, 1),
    }


@torch.no_grad()
def evaluate(
    model: GraphEnhancedModel,
    loader: DataLoader,
    criterion: nn.Module,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    device: str,
    *,
    use_amp: bool = False,
) -> dict[str, Any]:
    """Evaluate model and return predictions with optional AMP."""
    model.eval()
    loss_sum, total = 0.0, 0
    all_preds, all_labels, all_probs = [], [], []

    for batch in loader:
        features = batch["features"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)

        B, N, T, F = features.shape

        with torch.amp.autocast("cuda", enabled=use_amp):
            for b in range(B):
                x = features[b]
                y = labels[b]
                m = mask[b]

                if m.sum() == 0:
                    continue

                logits = model(x, edge_index, edge_weight)
                valid_logits = logits[m]
                valid_labels = y[m]

                loss = criterion(valid_logits, valid_labels)
                probs = torch.softmax(valid_logits, dim=1)[:, 1]

                loss_sum += loss.item() * valid_labels.size(0)
                all_preds.extend(valid_logits.argmax(1).cpu().tolist())
                all_labels.extend(valid_labels.cpu().tolist())
                all_probs.extend(probs.cpu().tolist())
                total += valid_labels.size(0)

    return {
        "loss": loss_sum / max(total, 1),
        "y_true": np.array(all_labels),
        "y_pred": np.array(all_preds),
        "y_prob": np.array(all_probs),
    }


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------


def run_graph_training(
    prices_dir: str | Path = "data/raw/prices",
    targets_dir: str | Path = "data/targets",
    graph_nodes_csv: str | Path = "data/raw/graph/tech10_nodes.csv",
    graph_edges_csv: str | Path = "data/raw/graph/tech10_edges.csv",
    save_dir: str | Path = "models",
    verbose: bool = True,
) -> dict[str, Any]:
    """Full Phase 5 GAT training pipeline with AMP."""

    config = TrainingConfig(
        learning_rate=5e-4,
        weight_decay=1e-4,
        epochs=30,
        batch_size=8,
        patience=7,
        seed=42,
        use_amp=True,
    )
    gpu_device = setup_gpu(verbose=verbose)
    device = str(gpu_device)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    use_amp = config.use_amp and device == "cuda"
    scaler = create_grad_scaler() if use_amp else None

    if verbose:
        print(f"Device: {device} | AMP: {use_amp}")

    # --- 1. Load graph ---
    graph = load_graph(graph_nodes_csv, graph_edges_csv)
    tickers = graph["tickers"]

    # Make edges bidirectional for message passing
    edge_index, edge_weight, edge_type = make_bidirectional(
        graph["edge_index"], graph["edge_weight"], graph["edge_type"]
    )
    edge_index = edge_index.to(device)
    edge_weight = edge_weight.to(device)

    if verbose:
        print(
            f"Graph: {graph['num_nodes']} nodes, {edge_index.size(1)} edges (bidirectional + self-loops)"
        )

    # --- 2. Build dataset ---
    dataset, train_idx, val_idx, test_idx = _build_datasets(
        prices_dir, targets_dir, tickers
    )

    if verbose:
        print(
            f"Snapshots — total: {len(dataset)}, train: {len(train_idx)}, "
            f"val: {len(val_idx)}, test: {len(test_idx)}"
        )

    train_loader = DataLoader(
        Subset(dataset, train_idx), batch_size=config.batch_size, shuffle=True
    )
    val_loader = DataLoader(
        Subset(dataset, val_idx), batch_size=config.batch_size, shuffle=False
    )
    test_loader = DataLoader(
        Subset(dataset, test_idx), batch_size=config.batch_size, shuffle=False
    )

    # --- 3. Build model ---
    model = GraphEnhancedModel(
        num_features=10,
        encoder_dim=256,
        gat_hidden=64,
        gat_heads=4,
        gat_dropout=0.1,
        num_classes=2,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"GraphEnhancedModel — {num_params:,} parameters")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    criterion = nn.CrossEntropyLoss()
    stopper = EarlyStopping(patience=config.patience)

    # --- 4. Training loop ---
    history: list[dict] = []
    best_val_loss = float("inf")

    for epoch in range(1, config.epochs + 1):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            edge_index,
            edge_weight,
            device,
            use_amp=use_amp,
            scaler=scaler,
        )

        val_result = evaluate(
            model,
            val_loader,
            criterion,
            edge_index,
            edge_weight,
            device,
            use_amp=use_amp,
        )
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
                Path(save_dir) / "graph_model_best.pt",
            )

        if verbose:
            print(
                f"  Epoch {epoch:02d} | "
                f"train_loss={train_metrics['train_loss']:.4f} "
                f"train_acc={train_metrics['train_acc']:.4f} | "
                f"val_loss={val_result['loss']:.4f} "
                f"val_acc={val_cls['accuracy']:.4f}"
            )
            if epoch == 1:
                log_gpu_usage("  [Epoch 1] ")

        if stopper(val_result["loss"]):
            if verbose:
                print(f"  Early stopping at epoch {epoch}")
            break

    # --- 5. Test ---
    test_metrics: dict = {}
    baseline_acc = 0.0

    best_path = Path(save_dir) / "graph_model_best.pt"
    if best_path.exists():
        ckpt = torch.load(best_path, weights_only=False, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])

    test_result = evaluate(
        model,
        test_loader,
        criterion,
        edge_index,
        edge_weight,
        device,
        use_amp=use_amp,
    )
    test_metrics = classification_metrics(
        test_result["y_true"], test_result["y_pred"], test_result["y_prob"]
    )
    baseline_acc = majority_baseline_accuracy(test_result["y_true"])

    if verbose:
        print(f"\n  Test results (best model):")
        for k, v in test_metrics.items():
            print(f"    {k}: {v:.4f}")
        print(f"    Majority baseline: {baseline_acc:.4f}")

    # --- 6. Ablation: No-GAT baseline ---
    # Create the same model but bypass GAT (just encoder + head)
    no_gat_metrics = _run_no_gat_ablation(
        dataset,
        train_idx,
        val_idx,
        test_idx,
        config,
        device,
        verbose,
        use_amp=use_amp,
    )

    return {
        "model": model,
        "history": history,
        "test_metrics": test_metrics,
        "baseline_accuracy": baseline_acc,
        "no_gat_test_metrics": no_gat_metrics,
        "config": config,
        "device": device,
        "n_train": len(train_idx),
        "n_val": len(val_idx),
        "n_test": len(test_idx),
    }


# ---------------------------------------------------------------------------
# No-GAT ablation
# ---------------------------------------------------------------------------


class _NoGATModel(nn.Module):
    """Same encoder + head as GraphEnhancedModel but without GAT layers.

    This serves as the ablation baseline to measure GAT's contribution.
    """

    def __init__(self, num_features: int = 10, encoder_dim: int = 256) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(num_features, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(64),
        )
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            dropout=0.2,
            bidirectional=True,
        )
        self.head = nn.Sequential(
            nn.Linear(encoder_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [N, T, F] → [N, 2]"""
        h = x.transpose(1, 2)
        h = self.conv(h)
        h = h.transpose(1, 2)
        h, _ = self.lstm(h)
        h = h[:, -1, :]
        return self.head(h)


def _run_no_gat_ablation(
    dataset: GraphSnapshotDataset,
    train_idx: list[int],
    val_idx: list[int],
    test_idx: list[int],
    config: TrainingConfig,
    device: str,
    verbose: bool,
    *,
    use_amp: bool = False,
) -> dict:
    """Train the same encoder architecture without GAT for ablation."""
    if verbose:
        print("\n" + "=" * 60)
        print("ABLATION: No-GAT baseline (encoder + head only)")
        print("=" * 60)

    model = _NoGATModel().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    criterion = nn.CrossEntropyLoss()
    stopper = EarlyStopping(patience=config.patience)
    scaler = create_grad_scaler() if use_amp else None

    train_loader = DataLoader(
        Subset(dataset, train_idx), batch_size=config.batch_size, shuffle=True
    )
    val_loader = DataLoader(
        Subset(dataset, val_idx), batch_size=config.batch_size, shuffle=False
    )
    test_loader = DataLoader(
        Subset(dataset, test_idx), batch_size=config.batch_size, shuffle=False
    )

    best_val_loss = float("inf")

    for epoch in range(1, config.epochs + 1):
        # Train
        model.train()
        loss_sum, correct, total = 0.0, 0, 0
        for batch in train_loader:
            features = batch["features"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            B, N, T, F = features.shape

            for b_i in range(B):
                m = mask[b_i]
                if m.sum() == 0:
                    continue
                x = features[b_i][m]  # [valid_N, T, F]
                y = labels[b_i][m]  # [valid_N]

                with torch.amp.autocast("cuda", enabled=use_amp):
                    logits = model(x)
                    loss = criterion(logits, y)

                optimizer.zero_grad()
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

                loss_sum += loss.item() * y.size(0)
                correct += (logits.argmax(1) == y).sum().item()
                total += y.size(0)

        t_loss = loss_sum / max(total, 1)
        t_acc = correct / max(total, 1)

        # Validate
        model.eval()
        v_preds, v_labels_list, v_probs = [], [], []
        v_loss_sum, v_total = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                features = batch["features"].to(device, non_blocking=True)
                labels_b = batch["labels"].to(device, non_blocking=True)
                mask_b = batch["mask"].to(device, non_blocking=True)
                B, N, T, F = features.shape
                for b_i in range(B):
                    m = mask_b[b_i]
                    if m.sum() == 0:
                        continue
                    x = features[b_i][m]
                    y = labels_b[b_i][m]
                    with torch.amp.autocast("cuda", enabled=use_amp):
                        logits = model(x)
                        loss = criterion(logits, y)
                    v_loss_sum += loss.item() * y.size(0)
                    v_preds.extend(logits.argmax(1).cpu().tolist())
                    v_labels_list.extend(y.cpu().tolist())
                    v_probs.extend(torch.softmax(logits, 1)[:, 1].cpu().tolist())
                    v_total += y.size(0)

        v_loss = v_loss_sum / max(v_total, 1)
        v_cls = classification_metrics(
            np.array(v_labels_list), np.array(v_preds), np.array(v_probs)
        )

        if v_loss < best_val_loss:
            best_val_loss = v_loss
            save_checkpoint(
                model,
                optimizer,
                epoch,
                {"val_loss": v_loss, **v_cls},
                Path("models") / "graph_nogat_best.pt",
            )

        if verbose:
            print(
                f"  Epoch {epoch:02d} | train_loss={t_loss:.4f} train_acc={t_acc:.4f} | "
                f"val_loss={v_loss:.4f} val_acc={v_cls['accuracy']:.4f}"
            )

        if stopper(v_loss):
            if verbose:
                print(f"  Early stopping at epoch {epoch}")
            break

    # Test
    best_path = Path("models") / "graph_nogat_best.pt"
    if best_path.exists():
        ckpt = torch.load(best_path, weights_only=False, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])

    model.eval()
    t_preds, t_labels_list, t_probs = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            features = batch["features"].to(device, non_blocking=True)
            labels_b = batch["labels"].to(device, non_blocking=True)
            mask_b = batch["mask"].to(device, non_blocking=True)
            B, N, T, F = features.shape
            for b_i in range(B):
                m = mask_b[b_i]
                if m.sum() == 0:
                    continue
                x = features[b_i][m]
                y = labels_b[b_i][m]
                with torch.amp.autocast("cuda", enabled=use_amp):
                    logits = model(x)
                t_preds.extend(logits.argmax(1).cpu().tolist())
                t_labels_list.extend(y.cpu().tolist())
                t_probs.extend(torch.softmax(logits, 1)[:, 1].cpu().tolist())

    no_gat_metrics = classification_metrics(
        np.array(t_labels_list), np.array(t_preds), np.array(t_probs)
    )
    no_gat_baseline = majority_baseline_accuracy(np.array(t_labels_list))

    if verbose:
        print(f"\n  No-GAT test results:")
        for k, v in no_gat_metrics.items():
            print(f"    {k}: {v:.4f}")
        print(f"    Majority baseline: {no_gat_baseline:.4f}")

    return no_gat_metrics
