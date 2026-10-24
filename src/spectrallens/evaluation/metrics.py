"""Evaluation metrics for time-series anomaly detection."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def compute_prevalence(labels: np.ndarray) -> float:
    """Compute the positive class prevalence (fraction of anomalous points)."""
    labels = np.asarray(labels, dtype=np.int64)
    if len(labels) == 0:
        return 0.0
    return float(np.mean(labels == 1))


def point_average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    """Compute Point-Level Average Precision (area under the precision-recall curve).

    Returns np.nan if there are no positive anomalies in labels (e.g. null series).
    """
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    if len(labels) != len(scores):
        raise ValueError(f"Length mismatch: {len(labels)} labels vs {len(scores)} scores")
    if np.all(labels == 0):
        return float(np.nan)
    if np.all(labels == 1):
        return 1.0
    return float(average_precision_score(labels, scores))


def point_roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Compute Point-Level ROC-AUC.

    Returns np.nan if there is only one class present in labels.
    """
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    if len(labels) != len(scores):
        raise ValueError(f"Length mismatch: {len(labels)} labels vs {len(scores)} scores")
    if np.all(labels == 0) or np.all(labels == 1):
        return float(np.nan)
    return float(roc_auc_score(labels, scores))


def binary_metrics_at_threshold(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    """Compute precision, recall, and F1 score at a specified score threshold."""
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    if len(labels) != len(scores):
        raise ValueError(f"Length mismatch: {len(labels)} labels vs {len(scores)} scores")

    preds = (scores >= threshold).astype(np.int64)
    tp = int(np.sum((preds == 1) & (labels == 1)))
    fp = int(np.sum((preds == 1) & (labels == 0)))
    fn = int(np.sum((preds == 0) & (labels == 1)))

    prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float(2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

    return {"precision": prec, "recall": rec, "f1": f1}


def event_hit(
    labels: np.ndarray,
    scores: np.ndarray,
    tolerance: int = 100,
    anomaly_range: tuple[int, int] | None = None,
) -> bool:
    """Determine whether the highest-scoring point falls within tolerance of the anomaly.

    As defined in UCR and DECISIONS.md (D-004), an event hit occurs if the location
    of the global score peak argmax(scores) satisfies:
        anomaly_start - tolerance <= argmax < anomaly_end + tolerance

    Args:
        labels: 1D binary ground truth array.
        scores: 1D point anomaly scores.
        tolerance: Allowed sample margin around the true anomaly segment.
        anomaly_range: Optional known (start, end) tuple. If None, derived from labels.

    Returns:
        True if the peak score falls within the anomaly segment (+- tolerance).
    """
    scores = np.asarray(scores, dtype=np.float64)
    if anomaly_range is None:
        pos_indices = np.flatnonzero(labels == 1)
        if len(pos_indices) == 0:
            return False
        start, end = int(pos_indices[0]), int(pos_indices[-1]) + 1
    else:
        start, end = anomaly_range

    peak_idx = int(np.argmax(scores))
    return bool((start - tolerance) <= peak_idx < (end + tolerance))


def detection_delay(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    anomaly_range: tuple[int, int] | None = None,
) -> int | None:
    """Calculate the number of samples from anomaly start to the first threshold crossing.

    Returns None if no detection occurs inside the anomaly segment [start, end).
    """
    scores = np.asarray(scores, dtype=np.float64)
    if anomaly_range is None:
        pos_indices = np.flatnonzero(labels == 1)
        if len(pos_indices) == 0:
            return None
        start, end = int(pos_indices[0]), int(pos_indices[-1]) + 1
    else:
        start, end = anomaly_range

    seg_scores = scores[start:end]
    detected_offsets = np.flatnonzero(seg_scores >= threshold)
    if len(detected_offsets) == 0:
        return None
    return int(detected_offsets[0])


def evaluate_predictions(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float | None = None,
    event_tolerance: int = 100,
    anomaly_range: tuple[int, int] | None = None,
) -> dict[str, Any]:
    """Calculate the full evaluation suite for predicted point anomaly scores.

    Includes primary metric (Average Precision), prevalence floor, ROC-AUC, event hit,
    and optional thresholded metrics (F1, precision, recall, detection delay).
    """
    out: dict[str, Any] = {
        "prevalence": compute_prevalence(labels),
        "point_average_precision": point_average_precision(labels, scores),
        "roc_auc": point_roc_auc(labels, scores),
        "event_hit": event_hit(
            labels, scores, tolerance=event_tolerance, anomaly_range=anomaly_range
        ),
    }

    if threshold is not None:
        bin_metrics = binary_metrics_at_threshold(labels, scores, threshold=threshold)
        out.update(bin_metrics)
        out["threshold"] = float(threshold)
        out["detection_delay"] = detection_delay(
            labels, scores, threshold=threshold, anomaly_range=anomaly_range
        )

    return out
