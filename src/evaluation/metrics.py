"""Evaluation metrics for classification and regression tasks."""

from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_fscore_support,
    r2_score,
    roc_auc_score,
)


def classification_metrics(y_true, y_pred, y_prob=None) -> Dict[str, float]:
    """Compute standard binary classification metrics."""
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="binary",
        zero_division=0,
    )

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }

    if y_prob is not None:
        try:
            metrics["auc"] = float(roc_auc_score(y_true, y_prob))
        except ValueError:
            metrics["auc"] = 0.5

    return metrics


def regression_metrics(y_true, y_pred) -> Dict[str, float]:
    """Compute standard regression metrics."""
    return {
        "mse": float(mean_squared_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def majority_baseline_accuracy(y_true) -> float:
    """Accuracy of always predicting the majority class."""
    unique, counts = np.unique(y_true, return_counts=True)
    return float(counts.max() / len(y_true)) if len(y_true) > 0 else 0.0
