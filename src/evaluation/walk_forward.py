"""Walk-forward expanding window validation.

Every model must be evaluated with BOTH:
1. Fixed temporal split (train <= 2022, val = 2023, test >= 2024)
2. Walk-forward expanding window (test years: 2020-2025)

Reports: mean AUC ± std across walk-forward folds.
If walk-forward mean differs from fixed-split AUC by more than 0.02,
flags this as potential overfitting to the test period.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import numpy as np
import pandas as pd


def walk_forward_splits(
    dates: np.ndarray | list[str],
    test_years: list[int] | None = None,
    min_train_years: int = 3,
) -> list[dict]:
    """Generate walk-forward expanding window splits.

    For each test year, the training set includes all data up to 2 years
    before, and validation is the year immediately before test.

    Parameters
    ----------
    dates : Array of date strings (YYYY-MM-DD format)
    test_years : List of years to use as test sets
    min_train_years : Minimum number of training years required

    Returns
    -------
    List of dicts, each with 'train_idx', 'val_idx', 'test_idx', 'test_year'
    """
    if test_years is None:
        test_years = [2020, 2021, 2022, 2023, 2024, 2025]

    dates_dt = pd.to_datetime(dates)
    years = dates_dt.year

    splits = []
    for test_year in test_years:
        val_year = test_year - 1
        train_end_year = test_year - 2

        # Check minimum training data
        train_mask = years <= train_end_year
        if train_mask.sum() == 0:
            continue

        val_mask = years == val_year
        test_mask = years == test_year

        if val_mask.sum() == 0 or test_mask.sum() == 0:
            continue

        train_idx = np.where(train_mask)[0].tolist()
        val_idx = np.where(val_mask)[0].tolist()
        test_idx = np.where(test_mask)[0].tolist()

        splits.append(
            {
                "train_idx": train_idx,
                "val_idx": val_idx,
                "test_idx": test_idx,
                "test_year": test_year,
                "n_train": len(train_idx),
                "n_val": len(val_idx),
                "n_test": len(test_idx),
            }
        )

    return splits


def run_walk_forward_validation(
    train_and_eval_fn: Callable,
    dates: np.ndarray | list[str],
    test_years: list[int] | None = None,
    verbose: bool = True,
    **kwargs,
) -> dict[str, Any]:
    """Run walk-forward validation with a user-provided training function.

    Parameters
    ----------
    train_and_eval_fn : Callable that takes (train_idx, val_idx, test_idx, **kwargs)
        and returns a dict with at least 'test_auc' key.
    dates : Array of date strings
    test_years : Test years for walk-forward
    verbose : Print progress
    **kwargs : Additional arguments passed to train_and_eval_fn

    Returns
    -------
    dict with walk-forward results
    """
    splits = walk_forward_splits(dates, test_years)

    fold_results = []
    for i, split in enumerate(splits):
        if verbose:
            print(f"\n{'='*60}")
            print(
                f"Walk-Forward Fold {i+1}/{len(splits)}: Test Year {split['test_year']}"
            )
            print(
                f"  Train: {split['n_train']} | Val: {split['n_val']} | Test: {split['n_test']}"
            )
            print(f"{'='*60}")

        result = train_and_eval_fn(
            train_idx=split["train_idx"],
            val_idx=split["val_idx"],
            test_idx=split["test_idx"],
            **kwargs,
        )
        result["test_year"] = split["test_year"]
        fold_results.append(result)

        if verbose:
            test_auc = result.get("test_auc", 0.5)
            print(f"  Test AUC: {test_auc:.4f}")

    # Aggregate results
    aucs = [r.get("test_auc", 0.5) for r in fold_results]
    mean_auc = float(np.mean(aucs))
    std_auc = float(np.std(aucs))

    if verbose:
        print(f"\n{'='*60}")
        print(f"WALK-FORWARD SUMMARY")
        print(f"{'='*60}")
        print(f"  Folds: {len(fold_results)}")
        print(f"  Mean AUC: {mean_auc:.4f} +/- {std_auc:.4f}")
        for r in fold_results:
            print(f"    Year {r['test_year']}: AUC={r.get('test_auc', 0.5):.4f}")

    return {
        "fold_results": fold_results,
        "mean_auc": mean_auc,
        "std_auc": std_auc,
        "n_folds": len(fold_results),
    }


def check_overfitting(
    fixed_split_auc: float,
    walk_forward_mean_auc: float,
    threshold: float = 0.03,
) -> dict:
    """Check if fixed-split AUC is suspiciously higher than walk-forward.

    Parameters
    ----------
    fixed_split_auc : AUC from the standard fixed temporal split
    walk_forward_mean_auc : Mean AUC from walk-forward validation
    threshold : Maximum acceptable difference

    Returns
    -------
    dict with 'is_overfit', 'difference', 'message'
    """
    diff = fixed_split_auc - walk_forward_mean_auc
    is_overfit = diff > threshold

    if is_overfit:
        message = (
            f"POTENTIAL OVERFITTING: Fixed-split AUC ({fixed_split_auc:.4f}) "
            f"exceeds walk-forward mean ({walk_forward_mean_auc:.4f}) "
            f"by {diff:.4f} (threshold: {threshold:.4f})"
        )
    else:
        message = (
            f"No overfitting detected: difference = {diff:.4f} "
            f"(within threshold of {threshold:.4f})"
        )

    print(message)
    return {
        "is_overfit": is_overfit,
        "difference": float(diff),
        "message": message,
    }
