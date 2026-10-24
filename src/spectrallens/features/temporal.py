"""Temporal feature extraction from time-series windows.

Feature set v1 (11 features) — frozen at M2 for the main RQ1 experiment.

For every window of length W the following descriptors are computed:
  1.  mean           — central tendency
  2.  median         — robust central tendency
  3.  std            — spread / standard deviation
  4.  min            — minimum value
  5.  max            — maximum value
  6.  range          — max − min (signal excursion)
  7.  iqr            — interquartile range Q75 − Q25 (robust spread)
  8.  slope          — linear trend slope via least-squares
  9.  mean_abs_diff  — mean |x[t] − x[t−1]| (roughness / rate of change)
  10. std_diff       — std of first differences
  11. acf1           — lag-1 autocorrelation

Design note (D-008): Isolation Forest is nearly scale-invariant, but a train-only
StandardScaler is still applied (leakage rule, and matters for local spectral detector).
"""

from __future__ import annotations

import numpy as np

FEATURE_NAMES = [
    "mean",
    "median",
    "std",
    "min",
    "max",
    "range",
    "iqr",
    "slope",
    "mean_abs_diff",
    "std_diff",
    "acf1",
]
N_TEMPORAL_FEATURES = len(FEATURE_NAMES)


def extract_temporal_features(windows: np.ndarray) -> np.ndarray:
    """Compute temporal feature set v1 for each window.

    Args:
        windows: 2D array of shape (n_windows, window_size). Each row is one window.

    Returns:
        2D feature matrix of shape (n_windows, 11).
    """
    windows = np.asarray(windows, dtype=np.float64)
    if windows.ndim != 2:
        raise ValueError(
            f"windows must be 2D (n_windows, window_size), got shape {windows.shape}"
        )
    if windows.shape[1] < 2:
        raise ValueError(
            f"window_size must be >= 2 for temporal features, got {windows.shape[1]}"
        )

    n, W = windows.shape

    # ── location ──────────────────────────────────────────────────────────────
    mean = np.mean(windows, axis=1)
    median = np.median(windows, axis=1)

    # ── scale/dispersion ──────────────────────────────────────────────────────
    std = np.std(windows, axis=1, ddof=0)  # population std (consistent across window sizes)
    vmin = np.min(windows, axis=1)
    vmax = np.max(windows, axis=1)
    vrange = vmax - vmin

    q25 = np.quantile(windows, 0.25, axis=1)
    q75 = np.quantile(windows, 0.75, axis=1)
    iqr = q75 - q25

    # ── trend ──────────────────────────────────────────────────────────────────
    # Least-squares slope: cov(t, x) / var(t) for t = 0..W-1
    t = np.arange(W, dtype=np.float64)
    t_centered = t - t.mean()
    t_var = np.dot(t_centered, t_centered)  # scalar
    x_centered = windows - mean[:, np.newaxis]
    slope = (x_centered @ t_centered) / t_var  # shape (n,)

    # ── dynamics ──────────────────────────────────────────────────────────────
    diffs = np.diff(windows, axis=1)  # shape (n, W-1)
    mean_abs_diff = np.mean(np.abs(diffs), axis=1)
    std_diff = np.std(diffs, axis=1, ddof=0)

    # ── lag-1 autocorrelation ─────────────────────────────────────────────────
    # ACF(1) = cov(x[1:], x[:-1]) / var(x)
    # We guard against zero variance (flat windows) by returning 0.
    x0 = x_centered[:, :-1]   # x[t]   shape (n, W-1)
    x1 = x_centered[:, 1:]    # x[t+1] shape (n, W-1)
    cov = np.sum(x0 * x1, axis=1) / W        # biased covariance estimate
    var = std ** 2                             # biased variance
    acf1 = np.where(var > 1e-12, np.divide(cov, var, where=var > 1e-12, out=np.zeros(n)), 0.0)

    return np.column_stack(
        [mean, median, std, vmin, vmax, vrange, iqr, slope, mean_abs_diff, std_diff, acf1]
    )
