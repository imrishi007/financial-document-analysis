"""Master improvement script - all 4 phases of model improvements.

Phase 1: Emergency Fixes (surprise leakage, remove news, ensemble)
Phase 2: Scientific Rigor (GradNorm, walk-forward validation)
Phase 3: Performance Scaling (long-horizon pretrain, per-ticker, vol-adjusted labels)
Phase 4: Advanced Upgrades (calibration, cross-attention, ranking loss)

Results are saved to models/v2_improvement_results.json after each phase.
"""

from __future__ import annotations

import json
import time
import copy
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, TensorDataset

from src.evaluation.metrics import (
    classification_metrics,
    majority_baseline_accuracy,
    regression_metrics,
)
from src.train.common import EarlyStopping, TrainingConfig, save_checkpoint
from src.utils.seed import set_global_seed

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS_PATH = Path("models/v2_improvement_results.json")
SEED = 42


def _save_results(results: dict):
    """Append results to disk."""
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results saved to {RESULTS_PATH}")


# ======================================================================
# PHASE 1: EMERGENCY FIXES
# ======================================================================


class CleanFusionModel(nn.Module):
    """Fusion model with 4 modalities (no news), fixed surprise features.

    Modalities: price(256), gat(256), doc(768), surprise(1 = recency only)
    """

    NUM_MODALITIES = 4

    def __init__(
        self,
        price_dim: int = 256,
        gat_dim: int = 256,
        doc_dim: int = 768,
        surprise_dim: int = 1,
        proj_dim: int = 128,
        hidden_dim: int = 256,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.proj_dim = proj_dim

        def _proj(in_dim: int) -> nn.Module:
            return nn.Sequential(
                nn.Linear(in_dim, proj_dim),
                nn.LayerNorm(proj_dim),
                nn.GELU(),
                nn.Dropout(dropout * 0.5),
            )

        self.price_proj = _proj(price_dim)
        self.gat_proj = _proj(gat_dim)
        self.doc_proj = _proj(doc_dim)
        self.surprise_proj = _proj(surprise_dim)

        gate_in = proj_dim * self.NUM_MODALITIES
        self.gate = nn.Sequential(
            nn.Linear(gate_in, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.NUM_MODALITIES),
            nn.Sigmoid(),
        )

        trunk_in = proj_dim * self.NUM_MODALITIES
        self.trunk = nn.Sequential(
            nn.Linear(trunk_in, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
        )

        self.direction_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim // 2, 2),
        )

        self.volatility_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, 1),
        )

    def forward(self, price_emb, gat_emb, doc_emb, surprise_feat, modality_mask):
        p = self.price_proj(price_emb)
        g = self.gat_proj(gat_emb)
        d = self.doc_proj(doc_emb)
        s = self.surprise_proj(surprise_feat)

        stacked = torch.stack([p, g, d, s], dim=1)  # [B, 4, proj]
        mask = modality_mask.unsqueeze(-1)  # [B, 4, 1]
        stacked = stacked * mask

        flat = stacked.view(stacked.size(0), -1)
        gates = self.gate(flat)
        gates = gates * modality_mask
        gates = gates / (gates.sum(dim=1, keepdim=True) + 1e-8)

        weighted = stacked * gates.unsqueeze(-1)
        fused = weighted.view(weighted.size(0), -1)

        h = self.trunk(fused)
        dir_logits = self.direction_head(h)
        vol_pred = self.volatility_head(h).squeeze(-1)

        return {
            "direction_logits": dir_logits,
            "volatility_pred": vol_pred,
            "gate_weights": gates,
        }


def fix_embeddings():
    """Fix surprise data leakage and remove news modality."""
    print("\n" + "=" * 60)
    print("PHASE 1a: Fixing surprise data leakage & removing news")
    print("=" * 60)

    data = torch.load(
        "data/embeddings/fusion_embeddings.pt", weights_only=False, map_location="cpu"
    )

    # Diagnostic: show what surprise features look like before fix
    surprise = data["surprise_feat"]
    mask = data["modality_mask"]
    surp_available = mask[:, 4] > 0
    print(f"\nBefore fix:")
    print(f"  Total samples: {len(surprise)}")
    print(f"  Surprise available: {surp_available.sum().item()}")
    print(f"  is_beat mean (where available): {surprise[surp_available, 0].mean():.4f}")
    print(f"  is_miss mean (where available): {surprise[surp_available, 1].mean():.4f}")
    print(f"  recency mean (where available): {surprise[surp_available, 2].mean():.4f}")
    print(f"  News available: {(mask[:, 3] > 0).sum().item()} / {len(mask)}")

    # FIX 1: Remove is_beat and is_miss (leaking target info)
    # Keep only recency feature (index 2)
    fixed_surprise = surprise[:, 2:3].clone()  # [N, 1] - just recency

    # FIX 2: Remove news modality
    fixed_mask = torch.stack(
        [
            mask[:, 0],  # price
            mask[:, 1],  # gat
            mask[:, 2],  # doc
            mask[:, 4],  # surprise (was index 4, now index 3)
        ],
        dim=1,
    )  # [N, 4]

    fixed_data = {
        "price_emb": data["price_emb"],
        "gat_emb": data["gat_emb"],
        "doc_emb": data["doc_emb"],
        "surprise_feat": fixed_surprise,
        "modality_mask": fixed_mask,
        "direction_label": data["direction_label"],
        "volatility_target": data["volatility_target"],
        "surprise_target": data["surprise_target"],
        "tickers": data["tickers"],
        "dates": data["dates"],
    }

    save_path = Path("data/embeddings/fusion_embeddings_v2.pt")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(fixed_data, save_path)

    print(f"\nAfter fix:")
    print(f"  Surprise features: [recency_only] shape={fixed_surprise.shape}")
    print(f"  Modalities: [price, gat, doc, surprise] shape={fixed_mask.shape}")
    print(f"  News: REMOVED")
    print(f"  Saved to {save_path}")

    return fixed_data


def _split_by_date(dates, direction_labels):
    """Temporal split: train <= 2022, val = 2023, test >= 2024."""
    train, val, test = [], [], []
    for i, d in enumerate(dates):
        if direction_labels[i] < 0:
            continue
        if d <= "2022-12-31":
            train.append(i)
        elif d <= "2023-12-31":
            val.append(i)
        else:
            test.append(i)
    return train, val, test


def train_clean_fusion(data: dict, save_dir: str = "models/v2_clean_fusion") -> dict:
    """Train the cleaned fusion model (no news, fixed surprise)."""
    print("\n" + "=" * 60)
    print("PHASE 1b: Training clean fusion model")
    print("=" * 60)

    set_global_seed(SEED)
    t0 = time.time()

    dates = data["dates"]
    dir_labels = data["direction_label"]
    train_idx, val_idx, test_idx = _split_by_date(dates, dir_labels)

    print(f"Device: {DEVICE}")
    print(f"Split: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

    # Build tensors
    price = data["price_emb"]
    gat = data["gat_emb"]
    doc = data["doc_emb"]
    surp = data["surprise_feat"]
    mask = data["modality_mask"]
    labels = data["direction_label"]
    vol_targets = data["volatility_target"]

    # Create model - direction only (no surprise task, no vol dilution)
    model = CleanFusionModel(
        price_dim=256,
        gat_dim=256,
        doc_dim=768,
        surprise_dim=1,
        proj_dim=128,
        hidden_dim=256,
        dropout=0.3,
    ).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"CleanFusionModel: {total_params:,} parameters")

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=60, eta_min=1e-6
    )
    stopper = EarlyStopping(patience=12, mode="max")
    ce_loss = nn.CrossEntropyLoss()

    # DataLoader helper
    def make_loader(indices, shuffle=False, bs=512):
        idx_t = torch.tensor(indices, dtype=torch.long)
        ds = TensorDataset(
            price[idx_t],
            gat[idx_t],
            doc[idx_t],
            surp[idx_t],
            mask[idx_t],
            labels[idx_t],
        )
        return DataLoader(
            ds, batch_size=bs, shuffle=shuffle, pin_memory=True, num_workers=0
        )

    train_loader = make_loader(train_idx, shuffle=True, bs=1024)
    val_loader = make_loader(val_idx, bs=2048)
    test_loader = make_loader(test_idx, bs=2048)

    best_val_auc = 0.0
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, 81):
        model.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_total = 0

        for batch in train_loader:
            p_b, g_b, d_b, s_b, m_b, l_b = [x.to(DEVICE) for x in batch]
            out = model(p_b, g_b, d_b, s_b, m_b)
            loss = ce_loss(out["direction_logits"], l_b)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss_sum += loss.item() * p_b.size(0)
            preds = out["direction_logits"].argmax(1)
            train_correct += (preds == l_b).sum().item()
            train_total += p_b.size(0)

        scheduler.step()

        # Validation
        model.eval()
        val_probs, val_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                p_b, g_b, d_b, s_b, m_b, l_b = [x.to(DEVICE) for x in batch]
                out = model(p_b, g_b, d_b, s_b, m_b)
                probs = torch.softmax(out["direction_logits"], dim=1)[:, 1]
                val_probs.extend(probs.cpu().tolist())
                val_labels.extend(l_b.cpu().tolist())

        val_metrics = classification_metrics(
            np.array(val_labels),
            (np.array(val_probs) > 0.5).astype(int),
            np.array(val_probs),
        )
        val_auc = val_metrics.get("auc", 0.5)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_auc": val_auc,
                },
                save_path / "clean_fusion_best.pt",
            )

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"  Epoch {epoch:02d} | loss={train_loss_sum/train_total:.4f} "
                f"acc={train_correct/train_total:.4f} | val_auc={val_auc:.4f}"
            )

        if stopper(val_auc):
            print(f"  Early stopping at epoch {epoch}")
            break

    # Test evaluation
    ckpt = torch.load(
        save_path / "clean_fusion_best.pt", weights_only=False, map_location=DEVICE
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    test_probs, test_labels, test_gates = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            p_b, g_b, d_b, s_b, m_b, l_b = [x.to(DEVICE) for x in batch]
            out = model(p_b, g_b, d_b, s_b, m_b)
            probs = torch.softmax(out["direction_logits"], dim=1)[:, 1]
            test_probs.extend(probs.cpu().tolist())
            test_labels.extend(l_b.cpu().tolist())
            test_gates.append(out["gate_weights"].cpu())

    test_metrics = classification_metrics(
        np.array(test_labels),
        (np.array(test_probs) > 0.5).astype(int),
        np.array(test_probs),
    )
    gate_weights = torch.cat(test_gates, dim=0).mean(0).tolist()

    print(f"\n{'='*60}")
    print("  CLEAN FUSION TEST RESULTS")
    print(f"{'='*60}")
    print(f"  AUC:  {test_metrics['auc']:.4f}")
    print(f"  Acc:  {test_metrics['accuracy']:.4f}")
    print(f"  F1:   {test_metrics['f1']:.4f}")
    print(
        f"  Gate weights: price={gate_weights[0]:.3f} gat={gate_weights[1]:.3f} "
        f"doc={gate_weights[2]:.3f} surp={gate_weights[3]:.3f}"
    )
    print(f"  Time: {time.time()-t0:.1f}s")

    return {
        "test_metrics": test_metrics,
        "gate_weights": {
            "price": gate_weights[0],
            "gat": gate_weights[1],
            "doc": gate_weights[2],
            "surprise": gate_weights[3],
        },
        "best_val_auc": best_val_auc,
        "epochs_trained": ckpt["epoch"],
        "time_seconds": time.time() - t0,
        "test_probs": test_probs,
        "test_labels": test_labels,
    }


def build_ensemble(data: dict) -> dict:
    """Build a calibrated ensemble from individual modality predictions."""
    print("\n" + "=" * 60)
    print("PHASE 1c: Building calibrated ensemble")
    print("=" * 60)

    dates = data["dates"]
    dir_labels = data["direction_label"]
    train_idx, val_idx, test_idx = _split_by_date(dates, dir_labels)

    # Use embeddings directly for simple logistic models per modality
    price = data["price_emb"]
    gat = data["gat_emb"]
    doc = data["doc_emb"]

    # Train simple linear classifiers per modality
    modality_results = {}
    modality_val_probs = {}
    modality_test_probs = {}

    for name, emb, dim in [("price", price, 256), ("gat", gat, 256), ("doc", doc, 768)]:
        print(f"\n  Training {name} linear classifier...")
        set_global_seed(SEED)

        classifier = nn.Sequential(
            nn.Linear(dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2),
        ).to(DEVICE)

        opt = torch.optim.AdamW(classifier.parameters(), lr=1e-3, weight_decay=1e-3)
        ce = nn.CrossEntropyLoss()

        idx_t = torch.tensor(train_idx, dtype=torch.long)
        train_ds = TensorDataset(emb[idx_t], dir_labels[idx_t])
        train_loader = DataLoader(
            train_ds, batch_size=2048, shuffle=True, pin_memory=True
        )

        for ep in range(50):
            classifier.train()
            for x_b, y_b in train_loader:
                x_b, y_b = x_b.to(DEVICE), y_b.to(DEVICE)
                loss = ce(classifier(x_b), y_b)
                opt.zero_grad()
                loss.backward()
                opt.step()

        # Val/Test predictions
        classifier.eval()
        with torch.no_grad():
            val_t = torch.tensor(val_idx, dtype=torch.long)
            val_logits = classifier(emb[val_t].to(DEVICE))
            val_p = torch.softmax(val_logits, dim=1)[:, 1].cpu().numpy()
            modality_val_probs[name] = val_p

            test_t = torch.tensor(test_idx, dtype=torch.long)
            test_logits = classifier(emb[test_t].to(DEVICE))
            test_p = torch.softmax(test_logits, dim=1)[:, 1].cpu().numpy()
            modality_test_probs[name] = test_p

        val_labels_np = dir_labels[val_t].numpy()
        val_met = classification_metrics(
            val_labels_np, (val_p > 0.5).astype(int), val_p
        )
        print(f"    {name} val AUC: {val_met['auc']:.4f}")
        modality_results[name] = {"val_auc": val_met["auc"]}

    # Optimize ensemble weights on validation set
    val_labels_np = dir_labels[torch.tensor(val_idx, dtype=torch.long)].numpy()
    test_labels_np = dir_labels[torch.tensor(test_idx, dtype=torch.long)].numpy()

    best_auc = 0
    best_weights = None
    modalities = ["price", "gat", "doc"]

    for w1 in np.arange(0, 1.05, 0.05):
        for w2 in np.arange(0, 1.05 - w1, 0.05):
            w3 = 1.0 - w1 - w2
            if w3 < -0.01:
                continue
            combo = (
                w1 * modality_val_probs["price"]
                + w2 * modality_val_probs["gat"]
                + w3 * modality_val_probs["doc"]
            )
            try:
                auc = float(roc_auc_score(val_labels_np, combo))
            except:
                auc = 0.5
            if auc > best_auc:
                best_auc = auc
                best_weights = {"price": w1, "gat": w2, "doc": w3}

    print(f"\n  Best ensemble weights (val): {best_weights}")
    print(f"  Val AUC: {best_auc:.4f}")

    # Test with best weights
    ensemble_test = (
        best_weights["price"] * modality_test_probs["price"]
        + best_weights["gat"] * modality_test_probs["gat"]
        + best_weights["doc"] * modality_test_probs["doc"]
    )

    test_metrics = classification_metrics(
        test_labels_np,
        (ensemble_test > 0.5).astype(int),
        ensemble_test,
    )

    print(f"\n  Ensemble TEST AUC:  {test_metrics['auc']:.4f}")
    print(f"  Ensemble TEST Acc:  {test_metrics['accuracy']:.4f}")

    return {
        "ensemble_weights": best_weights,
        "val_auc": best_auc,
        "test_metrics": test_metrics,
        "per_modality": modality_results,
    }


# ======================================================================
# PHASE 2: SCIENTIFIC RIGOR
# ======================================================================


class GradNormFusionModel(CleanFusionModel):
    """Fusion model with GradNorm automatic loss weighting.

    Adds learnable log-variance parameters for direction and volatility tasks
    that are automatically tuned during training.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Learnable task weights (log-variance parameterization)
        self.log_var_dir = nn.Parameter(torch.zeros(1))
        self.log_var_vol = nn.Parameter(torch.zeros(1))

    def compute_loss(
        self, out: dict, labels: torch.Tensor, vol_targets: torch.Tensor
    ) -> dict:
        """Compute uncertainty-weighted multi-task loss."""
        ce = nn.CrossEntropyLoss()
        mse = nn.MSELoss()

        dir_loss = ce(out["direction_logits"], labels)

        vol_valid = ~torch.isnan(vol_targets)
        if vol_valid.any():
            vol_loss = mse(out["volatility_pred"][vol_valid], vol_targets[vol_valid])
        else:
            vol_loss = torch.tensor(0.0, device=labels.device)

        # Uncertainty weighting: L = (1/2*var) * L_task + log(sigma)
        precision_dir = torch.exp(-self.log_var_dir)
        precision_vol = torch.exp(-self.log_var_vol)

        total_loss = (
            precision_dir * dir_loss
            + 0.5 * self.log_var_dir
            + precision_vol * vol_loss
            + 0.5 * self.log_var_vol
        )

        return {
            "total_loss": total_loss,
            "dir_loss": dir_loss.item(),
            "vol_loss": vol_loss.item(),
            "dir_weight": precision_dir.item(),
            "vol_weight": precision_vol.item(),
        }


def train_gradnorm_fusion(data: dict) -> dict:
    """Train fusion model with GradNorm uncertainty-weighted loss."""
    print("\n" + "=" * 60)
    print("PHASE 2a: Training with GradNorm uncertainty-weighted loss")
    print("=" * 60)

    set_global_seed(SEED)
    t0 = time.time()

    dates = data["dates"]
    dir_labels = data["direction_label"]
    vol_targets = data["volatility_target"]
    train_idx, val_idx, test_idx = _split_by_date(dates, dir_labels)

    price = data["price_emb"]
    gat = data["gat_emb"]
    doc = data["doc_emb"]
    surp = data["surprise_feat"]
    mask = data["modality_mask"]

    model = GradNormFusionModel(
        price_dim=256,
        gat_dim=256,
        doc_dim=768,
        surprise_dim=1,
        proj_dim=128,
        hidden_dim=256,
        dropout=0.3,
    ).to(DEVICE)

    print(f"GradNormFusionModel: {sum(p.numel() for p in model.parameters()):,} params")

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=80, eta_min=1e-6
    )
    stopper = EarlyStopping(patience=12, mode="max")

    def make_loader(indices, shuffle=False, bs=1024):
        idx_t = torch.tensor(indices, dtype=torch.long)
        ds = TensorDataset(
            price[idx_t],
            gat[idx_t],
            doc[idx_t],
            surp[idx_t],
            mask[idx_t],
            dir_labels[idx_t],
            vol_targets[idx_t],
        )
        return DataLoader(
            ds, batch_size=bs, shuffle=shuffle, pin_memory=True, num_workers=0
        )

    train_loader = make_loader(train_idx, shuffle=True)
    val_loader = make_loader(val_idx, bs=2048)
    test_loader = make_loader(test_idx, bs=2048)

    save_dir = Path("models/v2_gradnorm")
    save_dir.mkdir(parents=True, exist_ok=True)
    best_val_auc = 0.0

    for epoch in range(1, 81):
        model.train()
        ep_loss = 0.0
        ep_n = 0

        for batch in train_loader:
            p_b, g_b, d_b, s_b, m_b, l_b, v_b = [x.to(DEVICE) for x in batch]
            out = model(p_b, g_b, d_b, s_b, m_b)
            loss_dict = model.compute_loss(out, l_b, v_b)

            optimizer.zero_grad()
            loss_dict["total_loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            ep_loss += loss_dict["total_loss"].item() * p_b.size(0)
            ep_n += p_b.size(0)

        scheduler.step()

        # Validation
        model.eval()
        val_probs, val_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                p_b, g_b, d_b, s_b, m_b, l_b, v_b = [x.to(DEVICE) for x in batch]
                out = model(p_b, g_b, d_b, s_b, m_b)
                probs = torch.softmax(out["direction_logits"], dim=1)[:, 1]
                val_probs.extend(probs.cpu().tolist())
                val_labels.extend(l_b.cpu().tolist())

        val_met = classification_metrics(
            np.array(val_labels),
            (np.array(val_probs) > 0.5).astype(int),
            np.array(val_probs),
        )
        val_auc = val_met.get("auc", 0.5)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(
                {"model_state_dict": model.state_dict(), "epoch": epoch},
                save_dir / "gradnorm_best.pt",
            )

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"  Epoch {epoch:02d} | loss={ep_loss/ep_n:.4f} | val_auc={val_auc:.4f} | "
                f"w_dir={torch.exp(-model.log_var_dir).item():.3f} w_vol={torch.exp(-model.log_var_vol).item():.3f}"
            )

        if stopper(val_auc):
            print(f"  Early stopping at epoch {epoch}")
            break

    # Test
    ckpt = torch.load(
        save_dir / "gradnorm_best.pt", weights_only=False, map_location=DEVICE
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    test_probs, test_labels = [], []
    vol_preds, vol_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            p_b, g_b, d_b, s_b, m_b, l_b, v_b = [x.to(DEVICE) for x in batch]
            out = model(p_b, g_b, d_b, s_b, m_b)
            probs = torch.softmax(out["direction_logits"], dim=1)[:, 1]
            test_probs.extend(probs.cpu().tolist())
            test_labels.extend(l_b.cpu().tolist())
            v_valid = ~torch.isnan(v_b)
            if v_valid.any():
                vol_preds.extend(out["volatility_pred"][v_valid].cpu().tolist())
                vol_labels.extend(v_b[v_valid].cpu().tolist())

    test_metrics = classification_metrics(
        np.array(test_labels),
        (np.array(test_probs) > 0.5).astype(int),
        np.array(test_probs),
    )
    vol_met = (
        regression_metrics(np.array(vol_labels), np.array(vol_preds))
        if vol_labels
        else {}
    )

    print(f"\n  GradNorm TEST AUC:  {test_metrics['auc']:.4f}")
    print(f"  GradNorm TEST Acc:  {test_metrics['accuracy']:.4f}")
    if vol_met:
        print(f"  Volatility R2:     {vol_met['r2']:.4f}")

    return {
        "test_metrics": test_metrics,
        "vol_metrics": vol_met,
        "learned_weights": {
            "direction": torch.exp(-model.log_var_dir).item(),
            "volatility": torch.exp(-model.log_var_vol).item(),
        },
        "best_val_auc": best_val_auc,
        "time_seconds": time.time() - t0,
    }


def walk_forward_validation(data: dict) -> dict:
    """Walk-forward expanding window validation."""
    print("\n" + "=" * 60)
    print("PHASE 2b: Walk-forward validation")
    print("=" * 60)

    dates = data["dates"]
    dir_labels = data["direction_label"]
    price = data["price_emb"]
    gat = data["gat_emb"]
    doc = data["doc_emb"]
    surp = data["surprise_feat"]
    mask = data["modality_mask"]

    # Walk-forward: train on expanding window, test on next year
    # Windows: train<=2019/test=2020, train<=2020/test=2021, etc.
    fold_results = []

    for test_year in range(2020, 2026):
        set_global_seed(SEED)
        train_end = f"{test_year-1}-12-31"
        test_start = f"{test_year}-01-01"
        test_end = f"{test_year}-12-31"

        train_idx = [
            i for i, d in enumerate(dates) if d <= train_end and dir_labels[i] >= 0
        ]
        test_idx = [
            i
            for i, d in enumerate(dates)
            if test_start <= d <= test_end and dir_labels[i] >= 0
        ]

        if len(train_idx) < 100 or len(test_idx) < 50:
            continue

        print(
            f"\n  Fold: train <= {test_year-1}, test = {test_year} "
            f"(train={len(train_idx)}, test={len(test_idx)})"
        )

        model = CleanFusionModel(
            price_dim=256,
            gat_dim=256,
            doc_dim=768,
            surprise_dim=1,
            proj_dim=128,
            hidden_dim=256,
            dropout=0.3,
        ).to(DEVICE)

        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-3)
        ce = nn.CrossEntropyLoss()

        idx_t = torch.tensor(train_idx, dtype=torch.long)
        train_ds = TensorDataset(
            price[idx_t],
            gat[idx_t],
            doc[idx_t],
            surp[idx_t],
            mask[idx_t],
            dir_labels[idx_t],
        )
        train_loader = DataLoader(
            train_ds, batch_size=1024, shuffle=True, pin_memory=True
        )

        # Quick training (30 epochs, no early stopping for speed)
        for ep in range(30):
            model.train()
            for batch in train_loader:
                p_b, g_b, d_b, s_b, m_b, l_b = [x.to(DEVICE) for x in batch]
                out = model(p_b, g_b, d_b, s_b, m_b)
                loss = ce(out["direction_logits"], l_b)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Test
        model.eval()
        test_t = torch.tensor(test_idx, dtype=torch.long)
        test_ds = TensorDataset(
            price[test_t],
            gat[test_t],
            doc[test_t],
            surp[test_t],
            mask[test_t],
            dir_labels[test_t],
        )
        test_loader = DataLoader(test_ds, batch_size=2048)

        test_probs, test_labels = [], []
        with torch.no_grad():
            for batch in test_loader:
                p_b, g_b, d_b, s_b, m_b, l_b = [x.to(DEVICE) for x in batch]
                out = model(p_b, g_b, d_b, s_b, m_b)
                probs = torch.softmax(out["direction_logits"], dim=1)[:, 1]
                test_probs.extend(probs.cpu().tolist())
                test_labels.extend(l_b.cpu().tolist())

        met = classification_metrics(
            np.array(test_labels),
            (np.array(test_probs) > 0.5).astype(int),
            np.array(test_probs),
        )
        bl = majority_baseline_accuracy(np.array(test_labels))

        fold_results.append(
            {
                "test_year": test_year,
                "n_train": len(train_idx),
                "n_test": len(test_idx),
                "auc": met["auc"],
                "accuracy": met["accuracy"],
                "baseline": bl,
            }
        )
        print(f"    AUC={met['auc']:.4f}, Acc={met['accuracy']:.4f}, Baseline={bl:.4f}")

    mean_auc = np.mean([f["auc"] for f in fold_results])
    std_auc = np.std([f["auc"] for f in fold_results])
    print(f"\n  Walk-Forward Mean AUC: {mean_auc:.4f} ± {std_auc:.4f}")

    return {
        "folds": fold_results,
        "mean_auc": mean_auc,
        "std_auc": std_auc,
    }


# ======================================================================
# PHASE 3: PERFORMANCE SCALING
# ======================================================================


def volatility_adjusted_labels(data: dict) -> dict:
    """Create volatility-adjusted direction labels and retrain."""
    print("\n" + "=" * 60)
    print("PHASE 3a: Volatility-adjusted labels")
    print("=" * 60)

    set_global_seed(SEED)
    t0 = time.time()

    # Load raw targets to get returns and volatility
    targets_df = pd.read_csv("data/targets/direction_labels_multi_horizon.csv")
    vol_df = pd.read_csv("data/targets/volatility_targets.csv")

    targets_df["date"] = pd.to_datetime(targets_df["date"])
    vol_df["date"] = pd.to_datetime(vol_df["date"])

    # Merge to get volatility alongside returns
    merged = targets_df.merge(
        vol_df[["ticker", "date", "realized_vol_20d_annualized"]],
        on=["ticker", "date"],
        how="left",
    )

    # Build vol-adjusted labels: z-score = return / volatility
    # Use 5-day return if available
    if "return_5d" in merged.columns:
        returns_col = "return_5d"
    else:
        # Compute from direction: just use original labels but weight by inverse vol
        returns_col = None

    dates = data["dates"]
    dir_labels = data["direction_label"]

    if returns_col:
        # Create lookup: (ticker, date) -> vol-adjusted return
        merged["date_str"] = merged["date"].dt.strftime("%Y-%m-%d")
        vol_adj_lookup = {}
        for _, row in merged.iterrows():
            vol = row["realized_vol_20d_annualized"]
            ret = row[returns_col]
            if pd.notna(vol) and vol > 0.01 and pd.notna(ret):
                vol_adj_lookup[(row["ticker"], row["date_str"])] = ret / vol

        # Create new labels: UP if vol-adjusted return > 0, else DOWN
        tickers = data["tickers"]
        new_labels = data["direction_label"].clone()
        changed = 0
        for i in range(len(dates)):
            key = (tickers[i], dates[i])
            if key in vol_adj_lookup:
                adj_ret = vol_adj_lookup[key]
                new_label = 1 if adj_ret > 0 else 0
                if new_labels[i] != new_label and new_labels[i] >= 0:
                    changed += 1
                if new_labels[i] >= 0:
                    new_labels[i] = new_label

        print(
            f"  Labels changed: {changed} / {(new_labels >= 0).sum().item()} "
            f"({100*changed/(new_labels >= 0).sum().item():.1f}%)"
        )
    else:
        print("  No return column found, using original labels")
        new_labels = dir_labels

    # Train with vol-adjusted labels
    price = data["price_emb"]
    gat = data["gat_emb"]
    doc = data["doc_emb"]
    surp = data["surprise_feat"]
    mask = data["modality_mask"]

    train_idx, val_idx, test_idx = _split_by_date(dates, new_labels)

    model = CleanFusionModel(
        price_dim=256,
        gat_dim=256,
        doc_dim=768,
        surprise_dim=1,
        proj_dim=128,
        hidden_dim=256,
        dropout=0.3,
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=60, eta_min=1e-6
    )
    stopper = EarlyStopping(patience=12, mode="max")
    ce = nn.CrossEntropyLoss()

    def make_loader(indices, labels, shuffle=False, bs=1024):
        idx_t = torch.tensor(indices, dtype=torch.long)
        ds = TensorDataset(
            price[idx_t],
            gat[idx_t],
            doc[idx_t],
            surp[idx_t],
            mask[idx_t],
            labels[idx_t],
        )
        return DataLoader(ds, batch_size=bs, shuffle=shuffle, pin_memory=True)

    train_loader = make_loader(train_idx, new_labels, shuffle=True)
    val_loader = make_loader(val_idx, new_labels, bs=2048)
    # Test with ORIGINAL labels for fair comparison
    orig_test_idx = [i for i in test_idx if dir_labels[i] >= 0]
    test_loader = make_loader(orig_test_idx, dir_labels, bs=2048)

    save_dir = Path("models/v2_vol_adj")
    save_dir.mkdir(parents=True, exist_ok=True)
    best_val_auc = 0.0

    for epoch in range(1, 61):
        model.train()
        for batch in train_loader:
            p_b, g_b, d_b, s_b, m_b, l_b = [x.to(DEVICE) for x in batch]
            out = model(p_b, g_b, d_b, s_b, m_b)
            loss = ce(out["direction_logits"], l_b)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        model.eval()
        val_probs, val_labels_list = [], []
        with torch.no_grad():
            for batch in val_loader:
                p_b, g_b, d_b, s_b, m_b, l_b = [x.to(DEVICE) for x in batch]
                out = model(p_b, g_b, d_b, s_b, m_b)
                probs = torch.softmax(out["direction_logits"], dim=1)[:, 1]
                val_probs.extend(probs.cpu().tolist())
                val_labels_list.extend(l_b.cpu().tolist())

        val_met = classification_metrics(
            np.array(val_labels_list),
            (np.array(val_probs) > 0.5).astype(int),
            np.array(val_probs),
        )
        if val_met.get("auc", 0.5) > best_val_auc:
            best_val_auc = val_met["auc"]
            torch.save(
                {"model_state_dict": model.state_dict(), "epoch": epoch},
                save_dir / "vol_adj_best.pt",
            )

        if stopper(val_met.get("auc", 0.5)):
            print(f"  Early stopping at epoch {epoch}")
            break

    # Test with original labels
    ckpt = torch.load(
        save_dir / "vol_adj_best.pt", weights_only=False, map_location=DEVICE
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    test_probs, test_labels_list = [], []
    with torch.no_grad():
        for batch in test_loader:
            p_b, g_b, d_b, s_b, m_b, l_b = [x.to(DEVICE) for x in batch]
            out = model(p_b, g_b, d_b, s_b, m_b)
            probs = torch.softmax(out["direction_logits"], dim=1)[:, 1]
            test_probs.extend(probs.cpu().tolist())
            test_labels_list.extend(l_b.cpu().tolist())

    test_metrics = classification_metrics(
        np.array(test_labels_list),
        (np.array(test_probs) > 0.5).astype(int),
        np.array(test_probs),
    )
    print(f"  Vol-adjusted TEST AUC: {test_metrics['auc']:.4f}")
    print(f"  Time: {time.time()-t0:.1f}s")

    return {
        "test_metrics": test_metrics,
        "best_val_auc": best_val_auc,
        "time_seconds": time.time() - t0,
    }


def per_ticker_specialization(data: dict) -> dict:
    """Train ticker-specific direction models."""
    print("\n" + "=" * 60)
    print("PHASE 3b: Per-ticker specialization")
    print("=" * 60)

    dates = data["dates"]
    tickers = data["tickers"]
    dir_labels = data["direction_label"]
    price = data["price_emb"]
    gat = data["gat_emb"]
    doc = data["doc_emb"]
    surp = data["surprise_feat"]
    mask = data["modality_mask"]

    unique_tickers = sorted(set(tickers))
    ticker_results = {}
    all_test_probs = {}
    all_test_labels = {}

    for ticker in unique_tickers:
        set_global_seed(SEED)
        # Get indices for this ticker
        ticker_idx = [
            i for i, t in enumerate(tickers) if t == ticker and dir_labels[i] >= 0
        ]
        ticker_dates = [dates[i] for i in ticker_idx]

        train_idx = [
            ticker_idx[j] for j, d in enumerate(ticker_dates) if d <= "2022-12-31"
        ]
        val_idx = [
            ticker_idx[j]
            for j, d in enumerate(ticker_dates)
            if "2023-01-01" <= d <= "2023-12-31"
        ]
        test_idx = [
            ticker_idx[j] for j, d in enumerate(ticker_dates) if d > "2023-12-31"
        ]

        if len(train_idx) < 50 or len(test_idx) < 20:
            continue

        # Simple MLP per ticker (lightweight)
        emb_dim = 256 + 256 + 768 + 1  # price + gat + doc + surprise
        model = nn.Sequential(
            nn.Linear(emb_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2),
        ).to(DEVICE)

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
        ce = nn.CrossEntropyLoss()

        # Concatenate all features
        def get_features(indices):
            idx_t = torch.tensor(indices, dtype=torch.long)
            return torch.cat([price[idx_t], gat[idx_t], doc[idx_t], surp[idx_t]], dim=1)

        train_x = get_features(train_idx)
        train_y = dir_labels[torch.tensor(train_idx)]
        train_ds = TensorDataset(train_x, train_y)
        train_loader = DataLoader(
            train_ds, batch_size=512, shuffle=True, pin_memory=True
        )

        for ep in range(60):
            model.train()
            for x_b, y_b in train_loader:
                x_b, y_b = x_b.to(DEVICE), y_b.to(DEVICE)
                loss = ce(model(x_b), y_b)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        model.eval()
        test_x = get_features(test_idx).to(DEVICE)
        test_y = dir_labels[torch.tensor(test_idx)].numpy()
        with torch.no_grad():
            logits = model(test_x)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()

        met = classification_metrics(test_y, (probs > 0.5).astype(int), probs)
        ticker_results[ticker] = met
        all_test_probs[ticker] = probs.tolist()
        all_test_labels[ticker] = test_y.tolist()
        print(
            f"  {ticker}: AUC={met['auc']:.4f}, Acc={met['accuracy']:.4f} (n={len(test_idx)})"
        )

    # Combined metrics
    all_probs_combined = []
    all_labels_combined = []
    for t in unique_tickers:
        if t in all_test_probs:
            all_probs_combined.extend(all_test_probs[t])
            all_labels_combined.extend(all_test_labels[t])

    combined_met = classification_metrics(
        np.array(all_labels_combined),
        (np.array(all_probs_combined) > 0.5).astype(int),
        np.array(all_probs_combined),
    )
    print(f"\n  Combined (all tickers) AUC: {combined_met['auc']:.4f}")

    return {
        "per_ticker": {k: v for k, v in ticker_results.items()},
        "combined_metrics": combined_met,
    }


# ======================================================================
# PHASE 4: ADVANCED UPGRADES
# ======================================================================


class CrossAttentionFusionModel(nn.Module):
    """Fusion model with cross-attention between modalities."""

    NUM_MODALITIES = 4

    def __init__(
        self,
        price_dim: int = 256,
        gat_dim: int = 256,
        doc_dim: int = 768,
        surprise_dim: int = 1,
        proj_dim: int = 128,
        n_heads: int = 4,
        hidden_dim: int = 256,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.proj_dim = proj_dim

        # Projectors
        self.price_proj = nn.Linear(price_dim, proj_dim)
        self.gat_proj = nn.Linear(gat_dim, proj_dim)
        self.doc_proj = nn.Linear(doc_dim, proj_dim)
        self.surp_proj = nn.Linear(surprise_dim, proj_dim)

        # Cross-attention: each modality attends to all others
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=proj_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(proj_dim)

        # Self-attention refinement
        self.self_attn = nn.MultiheadAttention(
            embed_dim=proj_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.self_norm = nn.LayerNorm(proj_dim)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(proj_dim * self.NUM_MODALITIES, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
        )

        self.direction_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim // 2, 2),
        )

    def forward(self, price_emb, gat_emb, doc_emb, surprise_feat, modality_mask):
        p = self.price_proj(price_emb)
        g = self.gat_proj(gat_emb)
        d = self.doc_proj(doc_emb)
        s = self.surp_proj(surprise_feat)

        # Stack as sequence: [B, 4, proj_dim]
        tokens = torch.stack([p, g, d, s], dim=1)

        # Mask unavailable modalities
        mask_expanded = modality_mask.unsqueeze(-1)
        tokens = tokens * mask_expanded

        # Cross-attention
        attn_out, _ = self.cross_attn(tokens, tokens, tokens)
        tokens = self.attn_norm(tokens + attn_out)

        # Self-attention
        self_out, _ = self.self_attn(tokens, tokens, tokens)
        tokens = self.self_norm(tokens + self_out)

        # Flatten and predict
        flat = tokens.reshape(tokens.size(0), -1)
        h = self.ffn(flat)
        return {"direction_logits": self.direction_head(h)}


def train_cross_attention(data: dict) -> dict:
    """Train cross-attention fusion model."""
    print("\n" + "=" * 60)
    print("PHASE 4a: Cross-attention fusion model")
    print("=" * 60)

    set_global_seed(SEED)
    t0 = time.time()

    dates = data["dates"]
    dir_labels = data["direction_label"]
    train_idx, val_idx, test_idx = _split_by_date(dates, dir_labels)

    price = data["price_emb"]
    gat = data["gat_emb"]
    doc = data["doc_emb"]
    surp = data["surprise_feat"]
    mask = data["modality_mask"]

    model = CrossAttentionFusionModel(
        price_dim=256,
        gat_dim=256,
        doc_dim=768,
        surprise_dim=1,
        proj_dim=128,
        n_heads=4,
        hidden_dim=256,
        dropout=0.3,
    ).to(DEVICE)

    print(
        f"CrossAttentionFusionModel: {sum(p.numel() for p in model.parameters()):,} params"
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=80, eta_min=1e-6
    )
    stopper = EarlyStopping(patience=15, mode="max")
    ce = nn.CrossEntropyLoss()

    def make_loader(indices, shuffle=False, bs=1024):
        idx_t = torch.tensor(indices, dtype=torch.long)
        ds = TensorDataset(
            price[idx_t],
            gat[idx_t],
            doc[idx_t],
            surp[idx_t],
            mask[idx_t],
            dir_labels[idx_t],
        )
        return DataLoader(ds, batch_size=bs, shuffle=shuffle, pin_memory=True)

    train_loader = make_loader(train_idx, shuffle=True)
    val_loader = make_loader(val_idx, bs=2048)
    test_loader = make_loader(test_idx, bs=2048)

    save_dir = Path("models/v2_cross_attn")
    save_dir.mkdir(parents=True, exist_ok=True)
    best_val_auc = 0.0

    for epoch in range(1, 81):
        model.train()
        ep_loss = 0.0
        ep_n = 0
        for batch in train_loader:
            p_b, g_b, d_b, s_b, m_b, l_b = [x.to(DEVICE) for x in batch]
            out = model(p_b, g_b, d_b, s_b, m_b)
            loss = ce(out["direction_logits"], l_b)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            ep_loss += loss.item() * p_b.size(0)
            ep_n += p_b.size(0)

        scheduler.step()

        model.eval()
        val_probs, val_labels_list = [], []
        with torch.no_grad():
            for batch in val_loader:
                p_b, g_b, d_b, s_b, m_b, l_b = [x.to(DEVICE) for x in batch]
                out = model(p_b, g_b, d_b, s_b, m_b)
                probs = torch.softmax(out["direction_logits"], dim=1)[:, 1]
                val_probs.extend(probs.cpu().tolist())
                val_labels_list.extend(l_b.cpu().tolist())

        val_met = classification_metrics(
            np.array(val_labels_list),
            (np.array(val_probs) > 0.5).astype(int),
            np.array(val_probs),
        )
        val_auc = val_met.get("auc", 0.5)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(
                {"model_state_dict": model.state_dict(), "epoch": epoch},
                save_dir / "cross_attn_best.pt",
            )

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"  Epoch {epoch:02d} | loss={ep_loss/ep_n:.4f} | val_auc={val_auc:.4f}"
            )

        if stopper(val_auc):
            print(f"  Early stopping at epoch {epoch}")
            break

    # Test
    ckpt = torch.load(
        save_dir / "cross_attn_best.pt", weights_only=False, map_location=DEVICE
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    test_probs, test_labels_list = [], []
    with torch.no_grad():
        for batch in test_loader:
            p_b, g_b, d_b, s_b, m_b, l_b = [x.to(DEVICE) for x in batch]
            out = model(p_b, g_b, d_b, s_b, m_b)
            probs = torch.softmax(out["direction_logits"], dim=1)[:, 1]
            test_probs.extend(probs.cpu().tolist())
            test_labels_list.extend(l_b.cpu().tolist())

    test_metrics = classification_metrics(
        np.array(test_labels_list),
        (np.array(test_probs) > 0.5).astype(int),
        np.array(test_probs),
    )

    print(f"\n  Cross-Attention TEST AUC: {test_metrics['auc']:.4f}")
    print(f"  Time: {time.time()-t0:.1f}s")

    return {
        "test_metrics": test_metrics,
        "best_val_auc": best_val_auc,
        "time_seconds": time.time() - t0,
    }


def calibration_layer(data: dict, model_dir: str = "models/v2_clean_fusion") -> dict:
    """Apply temperature scaling calibration to the best model."""
    print("\n" + "=" * 60)
    print("PHASE 4b: Temperature scaling calibration")
    print("=" * 60)

    dates = data["dates"]
    dir_labels = data["direction_label"]
    train_idx, val_idx, test_idx = _split_by_date(dates, dir_labels)

    price = data["price_emb"]
    gat = data["gat_emb"]
    doc = data["doc_emb"]
    surp = data["surprise_feat"]
    mask = data["modality_mask"]

    # Load best clean fusion model
    model = CleanFusionModel(
        price_dim=256,
        gat_dim=256,
        doc_dim=768,
        surprise_dim=1,
        proj_dim=128,
        hidden_dim=256,
        dropout=0.3,
    ).to(DEVICE)
    ckpt = torch.load(
        Path(model_dir) / "clean_fusion_best.pt",
        weights_only=False,
        map_location=DEVICE,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Get validation logits for temperature fitting
    val_t = torch.tensor(val_idx, dtype=torch.long)
    val_ds = TensorDataset(
        price[val_t],
        gat[val_t],
        doc[val_t],
        surp[val_t],
        mask[val_t],
        dir_labels[val_t],
    )
    val_loader = DataLoader(val_ds, batch_size=2048)

    val_logits_list, val_labels_list = [], []
    with torch.no_grad():
        for batch in val_loader:
            p_b, g_b, d_b, s_b, m_b, l_b = [x.to(DEVICE) for x in batch]
            out = model(p_b, g_b, d_b, s_b, m_b)
            val_logits_list.append(out["direction_logits"].cpu())
            val_labels_list.append(l_b.cpu())

    val_logits = torch.cat(val_logits_list)
    val_labels = torch.cat(val_labels_list)

    # Optimize temperature on validation set
    temperature = nn.Parameter(torch.ones(1) * 1.5)
    opt = torch.optim.LBFGS([temperature], lr=0.01, max_iter=100)
    nll = nn.CrossEntropyLoss()

    def closure():
        opt.zero_grad()
        loss = nll(val_logits / temperature, val_labels)
        loss.backward()
        return loss

    opt.step(closure)
    T = temperature.item()
    print(f"  Learned temperature: {T:.4f}")

    # Apply to test set
    test_t = torch.tensor(test_idx, dtype=torch.long)
    test_ds = TensorDataset(
        price[test_t],
        gat[test_t],
        doc[test_t],
        surp[test_t],
        mask[test_t],
        dir_labels[test_t],
    )
    test_loader = DataLoader(test_ds, batch_size=2048)

    test_probs, test_labels_out = [], []
    with torch.no_grad():
        for batch in test_loader:
            p_b, g_b, d_b, s_b, m_b, l_b = [x.to(DEVICE) for x in batch]
            out = model(p_b, g_b, d_b, s_b, m_b)
            calibrated = torch.softmax(out["direction_logits"] / T, dim=1)[:, 1]
            test_probs.extend(calibrated.cpu().tolist())
            test_labels_out.extend(l_b.cpu().tolist())

    test_metrics = classification_metrics(
        np.array(test_labels_out),
        (np.array(test_probs) > 0.5).astype(int),
        np.array(test_probs),
    )
    print(f"  Calibrated TEST AUC: {test_metrics['auc']:.4f}")
    print(f"  Calibrated TEST Acc: {test_metrics['accuracy']:.4f}")

    # Confidence filtering with calibrated model
    probs_arr = np.array(test_probs)
    labels_arr = np.array(test_labels_out)
    confidence = np.maximum(probs_arr, 1 - probs_arr)

    thresholds = [1.0, 0.75, 0.5, 0.25, 0.1]
    cf_results = []
    for coverage in thresholds:
        n_keep = int(len(confidence) * coverage)
        if n_keep < 10:
            continue
        top_idx = np.argsort(confidence)[::-1][:n_keep]
        subset_probs = probs_arr[top_idx]
        subset_labels = labels_arr[top_idx]
        met = classification_metrics(
            subset_labels, (subset_probs > 0.5).astype(int), subset_probs
        )
        cf_results.append(
            {
                "coverage": coverage,
                "n": n_keep,
                "auc": met["auc"],
                "acc": met["accuracy"],
            }
        )
        print(
            f"    Coverage {coverage:.0%}: AUC={met['auc']:.4f}, Acc={met['accuracy']:.4f} (n={n_keep})"
        )

    return {
        "temperature": T,
        "test_metrics": test_metrics,
        "confidence_filtering": cf_results,
    }


def ranking_loss_model(data: dict) -> dict:
    """Train with ListNet-style ranking loss for relative stock ordering."""
    print("\n" + "=" * 60)
    print("PHASE 4c: Ranking loss (ListNet-style)")
    print("=" * 60)

    set_global_seed(SEED)
    t0 = time.time()

    dates = data["dates"]
    tickers_list = data["tickers"]
    dir_labels = data["direction_label"]
    price = data["price_emb"]
    gat = data["gat_emb"]
    doc = data["doc_emb"]
    surp = data["surprise_feat"]
    mask = data["modality_mask"]

    # Build per-date groups for ranking
    date_groups = {}
    for i in range(len(dates)):
        if dir_labels[i] < 0:
            continue
        d = dates[i]
        if d not in date_groups:
            date_groups[d] = []
        date_groups[d].append(i)

    # Filter to dates with at least 5 stocks
    valid_dates = {d: idxs for d, idxs in date_groups.items() if len(idxs) >= 5}

    train_dates = [d for d in valid_dates if d <= "2022-12-31"]
    val_dates = [d for d in valid_dates if "2023-01-01" <= d <= "2023-12-31"]
    test_dates = [d for d in valid_dates if d > "2023-12-31"]

    print(
        f"  Dates: train={len(train_dates)}, val={len(val_dates)}, test={len(test_dates)}"
    )

    # Simple ranking model: score each stock independently
    emb_dim = 256 + 256 + 768 + 1
    scorer = nn.Sequential(
        nn.Linear(emb_dim, 256),
        nn.LayerNorm(256),
        nn.GELU(),
        nn.Dropout(0.3),
        nn.Linear(256, 128),
        nn.GELU(),
        nn.Dropout(0.2),
        nn.Linear(128, 1),
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(scorer.parameters(), lr=1e-3, weight_decay=1e-3)

    # ListNet loss: KL divergence between predicted and true ranking distributions
    def listnet_loss(scores, labels):
        """ListNet: KL(softmax(labels) || softmax(scores))"""
        true_dist = F.softmax(labels.float(), dim=0)
        pred_dist = F.log_softmax(scores, dim=0)
        return F.kl_div(pred_dist, true_dist, reduction="batchmean")

    best_val_auc = 0.0
    save_dir = Path("models/v2_ranking")
    save_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, 61):
        scorer.train()
        np.random.shuffle(train_dates)
        ep_loss = 0.0

        for d in train_dates:
            idxs = valid_dates[d]
            idx_t = torch.tensor(idxs, dtype=torch.long)
            x = torch.cat(
                [price[idx_t], gat[idx_t], doc[idx_t], surp[idx_t]], dim=1
            ).to(DEVICE)
            y = dir_labels[idx_t].to(DEVICE)

            scores = scorer(x).squeeze(-1)
            loss = listnet_loss(scores, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ep_loss += loss.item()

        # Val evaluation - use scores as probabilities for AUC
        scorer.eval()
        val_probs, val_labels_list = [], []
        with torch.no_grad():
            for d in val_dates:
                idxs = valid_dates[d]
                idx_t = torch.tensor(idxs, dtype=torch.long)
                x = torch.cat(
                    [price[idx_t], gat[idx_t], doc[idx_t], surp[idx_t]], dim=1
                ).to(DEVICE)
                scores = torch.sigmoid(scorer(x).squeeze(-1)).cpu().tolist()
                labels = dir_labels[idx_t].tolist()
                val_probs.extend(scores)
                val_labels_list.extend(labels)

        val_met = classification_metrics(
            np.array(val_labels_list),
            (np.array(val_probs) > 0.5).astype(int),
            np.array(val_probs),
        )
        if val_met.get("auc", 0.5) > best_val_auc:
            best_val_auc = val_met["auc"]
            torch.save(
                {"model_state_dict": scorer.state_dict()}, save_dir / "ranking_best.pt"
            )

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"  Epoch {epoch:02d} | loss={ep_loss/len(train_dates):.4f} | val_auc={val_met.get('auc', 0.5):.4f}"
            )

    # Test
    ckpt = torch.load(
        save_dir / "ranking_best.pt", weights_only=False, map_location=DEVICE
    )
    scorer.load_state_dict(ckpt["model_state_dict"])
    scorer.eval()

    test_probs, test_labels_list = [], []
    with torch.no_grad():
        for d in test_dates:
            idxs = valid_dates[d]
            idx_t = torch.tensor(idxs, dtype=torch.long)
            x = torch.cat(
                [price[idx_t], gat[idx_t], doc[idx_t], surp[idx_t]], dim=1
            ).to(DEVICE)
            scores = torch.sigmoid(scorer(x).squeeze(-1)).cpu().tolist()
            labels = dir_labels[idx_t].tolist()
            test_probs.extend(scores)
            test_labels_list.extend(labels)

    test_metrics = classification_metrics(
        np.array(test_labels_list),
        (np.array(test_probs) > 0.5).astype(int),
        np.array(test_probs),
    )
    print(f"\n  Ranking TEST AUC: {test_metrics['auc']:.4f}")
    return {
        "test_metrics": test_metrics,
        "best_val_auc": best_val_auc,
        "time_seconds": time.time() - t0,
    }


# ======================================================================
# MASTER PIPELINE
# ======================================================================

from sklearn.metrics import roc_auc_score


def main():
    """Run all improvement phases sequentially."""
    print("=" * 70)
    print("  FINANCIAL DIRECTION FORECASTING - COMPREHENSIVE IMPROVEMENTS")
    print("=" * 70)
    print(f"  Device: {DEVICE}")
    print(f"  CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name()}")
        props = torch.cuda.get_device_properties(0)
        vram = getattr(props, "total_memory", getattr(props, "total_mem", 0))
        print(f"  VRAM: {vram / 1e9:.1f} GB")
    print()

    results = {}
    total_t0 = time.time()

    # ---- PHASE 1: Emergency Fixes ----
    print("\n" + "#" * 70)
    print("# PHASE 1: EMERGENCY FIXES")
    print("#" * 70)

    fixed_data = fix_embeddings()
    results["phase1a_surprise_fix"] = (
        "Applied - removed is_beat/is_miss leakage, kept recency only"
    )

    clean_fusion_result = train_clean_fusion(fixed_data)
    results["phase1b_clean_fusion"] = {
        k: v
        for k, v in clean_fusion_result.items()
        if k not in ("test_probs", "test_labels")
    }

    ensemble_result = build_ensemble(fixed_data)
    results["phase1c_ensemble"] = ensemble_result
    _save_results(results)

    # ---- PHASE 2: Scientific Rigor ----
    print("\n" + "#" * 70)
    print("# PHASE 2: SCIENTIFIC RIGOR")
    print("#" * 70)

    gradnorm_result = train_gradnorm_fusion(fixed_data)
    results["phase2a_gradnorm"] = gradnorm_result

    walkforward_result = walk_forward_validation(fixed_data)
    results["phase2b_walk_forward"] = walkforward_result
    _save_results(results)

    # ---- PHASE 3: Performance Scaling ----
    print("\n" + "#" * 70)
    print("# PHASE 3: PERFORMANCE SCALING")
    print("#" * 70)

    vol_adj_result = volatility_adjusted_labels(fixed_data)
    results["phase3a_vol_adjusted"] = vol_adj_result

    ticker_result = per_ticker_specialization(fixed_data)
    results["phase3b_per_ticker"] = ticker_result
    _save_results(results)

    # ---- PHASE 4: Advanced Upgrades ----
    print("\n" + "#" * 70)
    print("# PHASE 4: ADVANCED UPGRADES")
    print("#" * 70)

    cross_attn_result = train_cross_attention(fixed_data)
    results["phase4a_cross_attention"] = cross_attn_result

    calibration_result = calibration_layer(fixed_data)
    results["phase4b_calibration"] = calibration_result

    ranking_result = ranking_loss_model(fixed_data)
    results["phase4c_ranking_loss"] = ranking_result
    _save_results(results)

    # ---- FINAL SUMMARY ----
    print("\n" + "=" * 70)
    print("  FINAL RESULTS SUMMARY")
    print("=" * 70)

    summary = {
        "Old Baseline (Phase 6)": 0.5161,
        "Old Best (conf filtered)": 0.6024,
    }

    if "test_metrics" in clean_fusion_result:
        summary["Clean Fusion (no news, fixed surp)"] = clean_fusion_result[
            "test_metrics"
        ]["auc"]
    if "test_metrics" in ensemble_result:
        summary["Calibrated Ensemble"] = ensemble_result["test_metrics"]["auc"]
    if "test_metrics" in gradnorm_result:
        summary["GradNorm Fusion"] = gradnorm_result["test_metrics"]["auc"]
    if "mean_auc" in walkforward_result:
        summary["Walk-Forward Mean"] = walkforward_result["mean_auc"]
    if "test_metrics" in vol_adj_result:
        summary["Vol-Adjusted Labels"] = vol_adj_result["test_metrics"]["auc"]
    if "combined_metrics" in ticker_result:
        summary["Per-Ticker Ensemble"] = ticker_result["combined_metrics"]["auc"]
    if "test_metrics" in cross_attn_result:
        summary["Cross-Attention Fusion"] = cross_attn_result["test_metrics"]["auc"]
    if "test_metrics" in calibration_result:
        summary["Calibrated (temp scaling)"] = calibration_result["test_metrics"]["auc"]
    if "test_metrics" in ranking_result:
        summary["Ranking Loss (ListNet)"] = ranking_result["test_metrics"]["auc"]

    for name, auc in sorted(summary.items(), key=lambda x: x[1], reverse=True):
        delta = auc - 0.5161
        print(f"  {name:40s} AUC={auc:.4f} (Δ={delta:+.4f})")

    results["summary"] = summary
    results["total_time_seconds"] = time.time() - total_t0
    _save_results(results)

    print(f"\n  Total time: {time.time()-total_t0:.1f}s")
    print("  All results saved to models/v2_improvement_results.json")


if __name__ == "__main__":
    main()
