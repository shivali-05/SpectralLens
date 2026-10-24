import numpy as np

from spectrallens.evaluation.metrics import (
    binary_metrics_at_threshold,
    compute_prevalence,
    detection_delay,
    evaluate_predictions,
    event_hit,
    point_average_precision,
    point_roc_auc,
)


def test_compute_prevalence():
    labels = np.array([0, 0, 1, 0, 0, 1, 0, 0, 0, 0])
    assert compute_prevalence(labels) == 0.2
    assert compute_prevalence(np.zeros(50)) == 0.0


def test_point_average_precision_perfect_and_null():
    labels = np.zeros(100, dtype=int)
    labels[20:30] = 1

    # Perfect scores
    scores_perfect = labels.astype(float)
    assert point_average_precision(labels, scores_perfect) == 1.0

    # Inverted scores
    scores_inverted = 1.0 - scores_perfect
    ap_inverted = point_average_precision(labels, scores_inverted)
    assert ap_inverted < 0.15

    # Null series (no positive label) returns NaN
    null_labels = np.zeros(100, dtype=int)
    assert np.isnan(point_average_precision(null_labels, scores_perfect))


def test_point_roc_auc():
    labels = np.zeros(100, dtype=int)
    labels[10:20] = 1
    scores = labels.astype(float)
    assert point_roc_auc(labels, scores) == 1.0
    assert point_roc_auc(labels, 1.0 - scores) == 0.0
    assert np.isnan(point_roc_auc(np.zeros(50), np.ones(50)))


def test_binary_metrics_at_threshold():
    # 4 samples: 2 positive, 2 negative
    labels = np.array([0, 1, 0, 1])
    scores = np.array([0.1, 0.8, 0.9, 0.7])  # at th=0.75: preds = [0, 1, 1, 0] -> TP=1, FP=1, FN=1
    metrics = binary_metrics_at_threshold(labels, scores, threshold=0.75)
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5
    assert metrics["f1"] == 0.5


def test_event_hit():
    labels = np.zeros(200, dtype=int)
    labels[100:120] = 1  # anomaly at [100, 120)

    # Peak exactly inside anomaly
    scores_inside = np.zeros(200)
    scores_inside[105] = 10.0
    assert event_hit(labels, scores_inside, tolerance=50)

    # Peak within tolerance (e.g. at sample 70, distance 30 <= 50)
    scores_margin = np.zeros(200)
    scores_margin[70] = 10.0
    assert event_hit(labels, scores_margin, tolerance=50)

    # Peak far outside tolerance (e.g. at sample 10)
    scores_outside = np.zeros(200)
    scores_outside[10] = 10.0
    assert not event_hit(labels, scores_outside, tolerance=50)


def test_detection_delay():
    labels = np.zeros(100, dtype=int)
    labels[40:60] = 1
    scores = np.zeros(100)
    scores[47] = 2.0  # first threshold crossing at sample 47 -> delay = 7

    assert detection_delay(labels, scores, threshold=1.0) == 7
    # If not detected inside anomaly segment
    assert detection_delay(labels, scores, threshold=5.0) is None


def test_evaluate_predictions_suite():
    labels = np.zeros(100, dtype=int)
    labels[50:60] = 1
    scores = np.zeros(100)
    scores[50:60] = 2.0

    res = evaluate_predictions(labels, scores, threshold=1.0, event_tolerance=20)
    assert res["prevalence"] == 0.1
    assert res["point_average_precision"] == 1.0
    assert res["roc_auc"] == 1.0
    assert res["precision"] == 1.0
    assert res["recall"] == 1.0
    assert res["f1"] == 1.0
    assert res["detection_delay"] == 0
    assert res["event_hit"] is True
