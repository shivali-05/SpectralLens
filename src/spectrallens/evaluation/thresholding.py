"""Threshold estimation from out-of-sample calibration scores."""

from __future__ import annotations

import numpy as np


def fit_threshold_from_calibration(
    calibration_scores: np.ndarray,
    quantile: float = 0.99,
) -> float:
    """Derive an anomaly decision threshold from calibration scores.

    As documented in DECISIONS.md (D-002), Isolation Forest and unsupervised
    detectors score their own training data artificially well. Deriving thresholds
    from an unseen normal calibration split prevents optimistic threshold bias.

    Args:
        calibration_scores: Anomaly scores on unseen normal calibration data.
        quantile: Quantile cutoff (default: 0.99 for ~1% nominal false positive rate).

    Returns:
        Float threshold value.
    """
    scores = np.asarray(calibration_scores, dtype=np.float64)
    if scores.ndim != 1:
        raise ValueError(f"calibration_scores must be 1D, got ndim={scores.ndim}")
    if len(scores) == 0:
        raise ValueError("calibration_scores cannot be empty")
    if not (0.0 <= quantile <= 1.0):
        raise ValueError(f"quantile must be in [0, 1], got {quantile}")

    # Exclude any NaNs (e.g. from uncovered margin points)
    valid_scores = scores[~np.isnan(scores)]
    if len(valid_scores) == 0:
        raise ValueError("All calibration_scores are NaN")

    return float(np.quantile(valid_scores, quantile))
