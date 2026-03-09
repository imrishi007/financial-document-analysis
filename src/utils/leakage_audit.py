"""Feature leakage detection and prevention utilities.

Provides audit functions that should be run after constructing ANY new feature
to detect potential data leakage (features that are too correlated with the target).
"""

from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr


def audit_features_for_leakage(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    threshold: float = 0.15,
    raise_on_leak: bool = True,
) -> dict[str, float]:
    """Compute Spearman correlation between each feature and the target.

    Flag any feature with |correlation| > threshold as potential leakage.

    Run this after EVERY new feature addition. Print results always.
    Raise an AssertionError if any feature exceeds threshold (when raise_on_leak=True).

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
    y : np.ndarray of shape (n_samples,)
    feature_names : list of feature names
    threshold : Maximum absolute correlation before flagging
    raise_on_leak : Whether to raise AssertionError on detected leakage

    Returns
    -------
    dict mapping feature_name -> correlation value
    """
    print("=" * 60)
    print("LEAKAGE AUDIT RESULTS")
    print("=" * 60)

    correlations = {}
    leakage_detected = False

    for i, name in enumerate(feature_names):
        # Skip features with zero variance
        if np.std(X[:, i]) < 1e-10:
            print(f"{name:40s}: SKIPPED (zero variance)")
            correlations[name] = 0.0
            continue

        # Remove NaN values for this feature
        valid = ~(np.isnan(X[:, i]) | np.isnan(y))
        if valid.sum() < 10:
            print(f"{name:40s}: SKIPPED (too few valid samples)")
            correlations[name] = 0.0
            continue

        corr, pval = spearmanr(X[valid, i], y[valid])
        correlations[name] = float(corr)

        flag = " <<< POTENTIAL LEAKAGE" if abs(corr) > threshold else ""
        print(f"{name:40s}: corr={corr:+.4f}, p={pval:.4f}{flag}")

        if abs(corr) > threshold:
            leakage_detected = True

    if leakage_detected and raise_on_leak:
        raise AssertionError(
            "LEAKAGE DETECTED. Fix features before training. "
            "See audit results above for flagged features."
        )

    if not leakage_detected:
        print(
            "\nAll features passed leakage audit (|corr| < {:.2f}).".format(threshold)
        )

    return correlations


def audit_cross_stock_features(
    feature_df,
    target_col: str = "direction_60d",
    feature_cols: list[str] | None = None,
    threshold: float = 0.15,
) -> dict[str, float]:
    """Audit cross-stock features for leakage against direction targets.

    Parameters
    ----------
    feature_df : DataFrame with features and target columns
    target_col : Name of the target column
    feature_cols : List of feature column names to audit
    threshold : Maximum absolute correlation

    Returns
    -------
    dict mapping feature_name -> correlation value
    """
    import pandas as pd

    if feature_cols is None:
        feature_cols = [
            c for c in feature_df.columns if c not in ["ticker", "date", target_col]
        ]

    valid = feature_df.dropna(subset=[target_col] + feature_cols)
    X = valid[feature_cols].values
    y = valid[target_col].values

    return audit_features_for_leakage(X, y, feature_cols, threshold, raise_on_leak=True)
