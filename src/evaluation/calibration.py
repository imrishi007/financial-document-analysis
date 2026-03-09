"""Calibration utilities: temperature scaling, reliability diagrams, ECE.

Standard output for every trained model.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader


class TemperatureScaler(nn.Module):
    """Temperature scaling for model calibration.

    Optimizes a single scalar temperature T on the validation set
    to minimize NLL. Applied as: calibrated_probs = softmax(logits / T).
    """

    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature


def optimize_temperature(
    logits: np.ndarray,
    labels: np.ndarray,
    max_iter: int = 100,
    lr: float = 0.01,
) -> float:
    """Optimize temperature scaling on validation data.

    Parameters
    ----------
    logits : [N, 2] raw model logits
    labels : [N] integer labels
    max_iter : Number of optimization steps
    lr : Learning rate for temperature optimization

    Returns
    -------
    Optimal temperature value (float)
    """
    logits_t = torch.from_numpy(logits).float()
    labels_t = torch.from_numpy(labels).long()

    scaler = TemperatureScaler()
    optimizer = torch.optim.LBFGS([scaler.temperature], lr=lr, max_iter=max_iter)
    criterion = nn.CrossEntropyLoss()

    def closure():
        optimizer.zero_grad()
        scaled = scaler(logits_t)
        loss = criterion(scaled, labels_t)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(scaler.temperature.item())


def compute_ece(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Compute Expected Calibration Error (ECE).

    Parameters
    ----------
    probs : [N] predicted probabilities for positive class
    labels : [N] binary labels
    n_bins : Number of bins for calibration

    Returns
    -------
    ECE value (lower is better, 0 = perfectly calibrated)
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total = len(probs)

    for i in range(n_bins):
        lower = bin_boundaries[i]
        upper = bin_boundaries[i + 1]
        mask = (probs >= lower) & (probs < upper)
        n_in_bin = mask.sum()

        if n_in_bin == 0:
            continue

        avg_confidence = probs[mask].mean()
        avg_accuracy = labels[mask].mean()
        ece += (n_in_bin / total) * abs(avg_accuracy - avg_confidence)

    return float(ece)


def reliability_diagram_data(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
) -> dict:
    """Compute data for a reliability diagram.

    Returns
    -------
    dict with 'bin_centers', 'bin_accuracies', 'bin_confidences', 'bin_counts'
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_centers = []
    bin_accuracies = []
    bin_confidences = []
    bin_counts = []

    for i in range(n_bins):
        lower = bin_boundaries[i]
        upper = bin_boundaries[i + 1]
        mask = (probs >= lower) & (probs < upper)
        n_in_bin = mask.sum()

        center = (lower + upper) / 2
        bin_centers.append(center)
        bin_counts.append(int(n_in_bin))

        if n_in_bin > 0:
            bin_accuracies.append(float(labels[mask].mean()))
            bin_confidences.append(float(probs[mask].mean()))
        else:
            bin_accuracies.append(0.0)
            bin_confidences.append(center)

    return {
        "bin_centers": bin_centers,
        "bin_accuracies": bin_accuracies,
        "bin_confidences": bin_confidences,
        "bin_counts": bin_counts,
    }


def evaluate_filtered(
    probs: np.ndarray,
    labels: np.ndarray,
    threshold: float,
    temperature: float = 1.0,
) -> dict:
    """Evaluate model at a confidence threshold.

    Only considers predictions where max(P(UP), P(DOWN)) >= threshold.

    Parameters
    ----------
    probs : [N] predicted probabilities for positive class
    labels : [N] binary labels
    threshold : Minimum confidence to include
    temperature : Temperature for calibration

    Returns
    -------
    dict with 'auc', 'coverage', 'accuracy'
    """
    from sklearn.metrics import roc_auc_score

    # Apply temperature if not 1.0
    if abs(temperature - 1.0) > 1e-6:
        # Convert prob to logit, scale, convert back
        logit = np.log(probs / (1 - probs + 1e-8) + 1e-8)
        scaled_logit = logit / temperature
        probs = 1 / (1 + np.exp(-scaled_logit))

    confidence = np.maximum(probs, 1 - probs)
    mask = confidence >= threshold
    coverage = mask.mean()

    if mask.sum() < 10 or len(np.unique(labels[mask])) < 2:
        return {"auc": 0.5, "coverage": float(coverage), "accuracy": 0.5}

    filtered_probs = probs[mask]
    filtered_labels = labels[mask]

    try:
        auc = float(roc_auc_score(filtered_labels, filtered_probs))
    except ValueError:
        auc = 0.5

    accuracy = float((filtered_labels == (filtered_probs >= 0.5).astype(int)).mean())

    return {
        "auc": auc,
        "coverage": float(coverage),
        "accuracy": accuracy,
    }


def calibrate_and_report(
    logits: np.ndarray,
    labels: np.ndarray,
    val_logits: np.ndarray,
    val_labels: np.ndarray,
) -> dict:
    """Full calibration pipeline. Run after every model training.

    Parameters
    ----------
    logits : [N, 2] test set logits
    labels : [N] test set labels
    val_logits : [N_val, 2] validation set logits (for temperature fitting)
    val_labels : [N_val] validation set labels

    Returns
    -------
    dict with calibration results
    """
    # 1. Optimize temperature on validation set
    T = optimize_temperature(val_logits, val_labels)
    print(f"\nOptimal temperature: {T:.4f}")

    # 2. Get calibrated probabilities
    probs_raw = torch.softmax(torch.from_numpy(logits).float(), dim=1)[:, 1].numpy()
    probs_cal = torch.softmax(torch.from_numpy(logits).float() / T, dim=1)[:, 1].numpy()

    # 3. ECE before and after calibration
    ece_before = compute_ece(probs_raw, labels)
    ece_after = compute_ece(probs_cal, labels)
    print(f"ECE before calibration: {ece_before:.4f}")
    print(f"ECE after calibration:  {ece_after:.4f}")

    # 4. Confidence-filtered evaluation
    thresholds = [0.50, 0.52, 0.55, 0.57, 0.60]
    filtered_results = {}
    print("\nConfidence-filtered evaluation (calibrated):")
    for thresh in thresholds:
        result = evaluate_filtered(probs_cal, labels, thresh, temperature=1.0)
        filtered_results[thresh] = result
        print(
            f"  Threshold {thresh:.2f}: "
            f"AUC={result['auc']:.4f}, "
            f"Coverage={result['coverage']:.1%}, "
            f"Acc={result['accuracy']:.4f}"
        )

    # 5. Reliability diagram data
    rel_data = reliability_diagram_data(probs_cal, labels)

    return {
        "temperature": T,
        "ece_before": ece_before,
        "ece_after": ece_after,
        "filtered_results": filtered_results,
        "reliability_diagram": rel_data,
        "probs_calibrated": probs_cal,
        "probs_raw": probs_raw,
    }
