"""Evaluation utilities."""

from src.evaluation.metrics import (
    classification_metrics,
    regression_metrics,
    majority_baseline_accuracy,
)
from src.evaluation.backtester import run_backtest
from src.evaluation.calibration import calibrate_and_report, compute_ece
from src.evaluation.walk_forward import (
    walk_forward_splits,
    run_walk_forward_validation,
    check_overfitting,
)

__all__ = [
    "classification_metrics",
    "regression_metrics",
    "majority_baseline_accuracy",
    "run_backtest",
    "calibrate_and_report",
    "compute_ece",
    "walk_forward_splits",
    "run_walk_forward_validation",
    "check_overfitting",
]
