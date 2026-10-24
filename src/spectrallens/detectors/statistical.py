"""Statistical baseline detectors for control experiments and sanity checks."""

from __future__ import annotations

import numpy as np


class RandomDetector:
    """Assigns independent uniform random scores in [0, 1).

    Used as a primary negative control: Average Precision (PR-AUC) must be close
    to the dataset prevalence floor on any dataset when evaluated with RandomDetector.
    """

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def fit(self, X: np.ndarray) -> RandomDetector:
        """Fit does nothing for a random detector."""
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        """Produce random scores of length len(X)."""
        return self.rng.uniform(0.0, 1.0, size=len(X))


class RobustZDetector:
    """Robust Z-score anomaly detector based on Median and Median Absolute Deviation (MAD).

    Unlike standard mean/std z-scores, median and MAD are robust against outliers.
    For normal distributions, 1.4826 * MAD approximates the standard deviation.

    When scoring 2D windows (n_windows, window_size), the window score is the
    maximum absolute robust z-score observed within the window. When scoring 1D points,
    the point score is the absolute robust z-score of each point.
    """

    def __init__(self, eps: float = 1e-8) -> None:
        self.eps = eps
        self.median_: float | None = None
        self.mad_: float | None = None
        self.scale_: float | None = None

    def fit(self, X: np.ndarray) -> RobustZDetector:
        """Fit median and MAD on normal training data (1D points or 2D windows)."""
        arr = np.asarray(X, dtype=np.float64)
        flat = arr.ravel()
        if len(flat) == 0:
            raise ValueError("Cannot fit RobustZDetector on empty data")

        self.median_ = float(np.median(flat))
        deviations = np.abs(flat - self.median_)
        self.mad_ = float(np.median(deviations))
        # Normal-consistent scale estimator
        self.scale_ = max(1.4826 * self.mad_, self.eps)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        """Compute anomaly scores for points (1D) or windows (2D)."""
        if self.median_ is None or self.scale_ is None:
            raise RuntimeError("RobustZDetector must be fit before scoring")

        arr = np.asarray(X, dtype=np.float64)
        if arr.ndim == 1:
            return np.abs(arr - self.median_) / self.scale_
        elif arr.ndim == 2:
            # Window score is the maximum deviation within the window
            point_z = np.abs(arr - self.median_) / self.scale_
            return np.max(point_z, axis=1)
        else:
            raise ValueError(f"Expected 1D or 2D array, got shape {arr.shape}")
