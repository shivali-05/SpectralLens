"""Point-score mapping and score aggregation across overlapping windows."""

from __future__ import annotations

import numpy as np

from spectrallens.data.windowing import Windows


def windows_to_points(
    scores: np.ndarray,
    indices: np.ndarray,
    length: int,
    default: float = np.nan,
) -> np.ndarray:
    """Map window-level anomaly scores to point-level scores using mean aggregation.

    For every sample point t, its score is the mean of the scores of all windows
    that cover t (i.e. where w_start <= t < w_end). Points with zero covering windows
    are set to ``default``.

    Args:
        scores: 1D array of window scores, shape (n_windows,).
        indices: 2D array of window bounds, shape (n_windows, 2), each row is [start, end).
        length: Total number of points in the target array.
        default: Fill value for uncovered points (default: np.nan).

    Returns:
        1D array of shape (length,) with aggregated point anomaly scores.
    """
    scores = np.asarray(scores, dtype=np.float64)
    indices = np.asarray(indices, dtype=np.int64)

    if scores.ndim != 1:
        raise ValueError(f"scores must be 1D, got ndim={scores.ndim}")
    if indices.ndim != 2 or indices.shape[1] != 2:
        raise ValueError(f"indices must have shape (n_windows, 2), got {indices.shape}")
    if len(scores) != len(indices):
        raise ValueError(
            f"Length mismatch: {len(scores)} scores vs {len(indices)} window indices"
        )
    if length <= 0:
        raise ValueError(f"length must be positive, got {length}")

    if len(scores) == 0:
        return np.full(length, default, dtype=np.float64)

    if np.any(indices[:, 0] < 0) or np.any(indices[:, 1] > length):
        raise ValueError(f"Window indices exceed bounds [0, {length})")
    if np.any(indices[:, 0] >= indices[:, 1]):
        raise ValueError("Invalid window indices: found window where start >= end")

    sum_scores = np.zeros(length, dtype=np.float64)
    counts = np.zeros(length, dtype=np.int64)

    for score, (start, end) in zip(scores, indices, strict=True):
        sum_scores[start:end] += score
        counts[start:end] += 1

    point_scores = np.full(length, default, dtype=np.float64)
    covered = counts > 0
    point_scores[covered] = sum_scores[covered] / counts[covered]

    return point_scores


def point_scores_for_split(
    scores: np.ndarray,
    windows: Windows,
    split_range: tuple[int, int],
    default: float = np.nan,
) -> np.ndarray:
    """Map window scores to points strictly within a split region [start, end).

    Shifts absolute window coordinates relative to split start so the resulting
    array has length (end - start), matching the split's local indices.

    Args:
        scores: 1D array of window scores, shape (n_windows,).
        windows: Windows dataclass instance extracted for this split.
        split_range: Half-open tuple (start, end) of the split.
        default: Fill value for uncovered points.

    Returns:
        1D array of shape (end - start,) with point-level anomaly scores.
    """
    start, end = split_range
    split_len = end - start
    if split_len <= 0:
        raise ValueError(f"Invalid split range {split_range}")

    if windows.n_windows == 0:
        return np.full(split_len, default, dtype=np.float64)

    # Shift indices to be relative to the split start
    rel_indices = windows.indices - start
    return windows_to_points(scores, rel_indices, length=split_len, default=default)
