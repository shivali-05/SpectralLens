"""Tests for IsolationForestDetector and the end-to-end temporal pipeline.

Two categories:
  1. Unit tests for the detector class itself (leakage guard, score ordering).
  2. Integration tests running the full vertical slice on synthetic series and checking
     that the positive control and negative control behave as expected.
"""

import numpy as np
import pytest

from spectrallens.config import load_config
from spectrallens.data.protocol import Protocol
from spectrallens.data.synthetic import generate_series
from spectrallens.data.windowing import extract_split_windows
from spectrallens.detectors.isolation_forest import IsolationForestDetector
from spectrallens.evaluation.metrics import event_hit, point_average_precision
from spectrallens.evaluation.scoring import windows_to_points
from spectrallens.evaluation.thresholding import fit_threshold_from_calibration
from spectrallens.features.temporal import extract_temporal_features

CFG = load_config("base", "synthetic")
PROTO = Protocol.from_config(CFG)
W = CFG["windowing"]["size"]     # 128
STEP = CFG["windowing"]["step"]  # 8


# ── helper: run the full temporal-IF pipeline on one series ──────────────────

def _run_temporal_pipeline(series):
    """Run the full temporal-IF pipeline and return (labels, point_scores)."""
    split_windows = extract_split_windows(series.x, PROTO, window_size=W, step=STEP)

    fit_feats = extract_temporal_features(split_windows["fit"].data)
    cal_feats = extract_temporal_features(split_windows["calibration"].data)
    test_feats = extract_temporal_features(split_windows["test"].data)

    detector = IsolationForestDetector(CFG)
    detector.fit(fit_feats)  # scaler AND forest fit ONLY on fit features

    cal_win_scores = detector.score(cal_feats)
    test_win_scores = detector.score(test_feats)

    threshold = fit_threshold_from_calibration(cal_win_scores, quantile=0.99)

    # Map window scores to point scores for the test split only
    test_pt_scores = windows_to_points(
        test_win_scores,
        split_windows["test"].indices - PROTO.test[0],  # shift to local indices
        length=PROTO.test[1] - PROTO.test[0],
    )

    # Ground-truth labels for the test split only
    test_labels = series.labels[PROTO.test[0]:PROTO.test[1]]
    return test_labels, test_pt_scores, threshold


# ── unit tests for IsolationForestDetector ───────────────────────────────────

def test_detector_score_before_fit_raises():
    det = IsolationForestDetector(CFG)
    with pytest.raises(RuntimeError, match="must be fit"):
        det.score(np.ones((5, 11)))


def test_detector_fit_rejects_empty_features():
    det = IsolationForestDetector(CFG)
    with pytest.raises(ValueError, match="empty"):
        det.fit(np.empty((0, 11)))


def test_detector_fit_rejects_1d_input():
    det = IsolationForestDetector(CFG)
    with pytest.raises(ValueError, match="2D"):
        det.fit(np.ones(11))


def test_scaler_fit_only_on_training_data():
    """Critical leakage guard: scaler must never be re-fit on calibration/test data."""
    rng = np.random.default_rng(99)
    fit_feats = rng.standard_normal((300, 11))
    cal_feats = rng.standard_normal((100, 11)) * 10 + 50  # wildly different scale

    det = IsolationForestDetector(CFG)
    det.fit(fit_feats)

    # After fitting on fit_feats, the scaler mean must reflect fit_feats, not cal_feats
    np.testing.assert_allclose(det._scaler.mean_, fit_feats.mean(axis=0), atol=0.5)

    # Scoring cal_feats must work without re-fitting the scaler
    cal_scores = det.score(cal_feats)
    assert cal_scores.shape == (100,)


def test_anomalous_windows_score_higher_than_normal():
    """Normal windows should have lower anomaly scores than injected anomalous windows.

    We generate 300 training windows (> max_samples=256) from a clean sine wave so
    the Isolation Forest is well-calibrated, then score an extreme outlier window where
    a single sample is set to 1000x the signal amplitude.
    """
    rng = np.random.default_rng(42)
    # 300 windows of length 128 from a clean sine + tiny noise
    t = np.linspace(0, 60 * np.pi, 300 * 128)
    clean = np.sin(t) + 0.05 * rng.standard_normal(len(t))
    normal_windows = clean.reshape(300, 128)

    det = IsolationForestDetector(CFG)
    det.fit(extract_temporal_features(normal_windows))

    normal_feats = extract_temporal_features(normal_windows)
    normal_scores = det.score(normal_feats)

    # Outlier: a window where one sample is 1000x the signal range
    anomaly_window = normal_windows[0:1].copy()
    anomaly_window[0, 64] = 1000.0
    anomaly_feats = extract_temporal_features(anomaly_window)
    anomaly_score = det.score(anomaly_feats)[0]

    assert anomaly_score > np.percentile(normal_scores, 95), (
        f"Extreme outlier score {anomaly_score:.4f} should exceed 95th percentile of "
        f"normal scores {np.percentile(normal_scores, 95):.4f}"
    )


# ── integration tests: positive & negative controls ──────────────────────────

def test_null_series_ap_is_nan():
    """Null series has no anomaly — AP must be NaN (no positive labels)."""
    series = generate_series(CFG, "single_sinusoid", None, 0)
    test_labels, test_pt_scores, _ = _run_temporal_pipeline(series)
    assert np.all(test_labels == 0)
    assert np.isnan(point_average_precision(test_labels, test_pt_scores))


def test_easy_point_spike_is_detected_event_hit():
    """Positive control: an 8-sigma spike with temporal features must be an event hit."""
    series = generate_series(
        CFG, "single_sinusoid", "point_spike", 0, difficulty="easy"
    )
    test_labels, test_pt_scores, _ = _run_temporal_pipeline(series)
    assert event_hit(test_labels, test_pt_scores, tolerance=100)


def test_temporal_pipeline_runs_all_four_anomaly_types():
    """Pipeline must complete end-to-end for every anomaly type (not a metric check)."""
    anomaly_types = ["point_spike", "amplitude_change", "frequency_shift",
                     "local_frequency_insertion"]
    for anom in anomaly_types:
        series = generate_series(CFG, "single_sinusoid", anom, 1)
        labels, scores, threshold = _run_temporal_pipeline(series)
        assert len(labels) == PROTO.test[1] - PROTO.test[0]
        assert len(scores) == len(labels)
        assert np.isfinite(threshold)


def test_amplitude_change_ap_above_prevalence():
    """Amplitude change is visible to temporal std/range — AP should beat prevalence."""
    series = generate_series(CFG, "single_sinusoid", "amplitude_change", 2, difficulty="easy")
    test_labels, test_pt_scores, _ = _run_temporal_pipeline(series)
    prevalence = float(test_labels.mean())
    ap = point_average_precision(test_labels, test_pt_scores)
    assert ap > prevalence * 3, (
        f"Expected AP >> prevalence ({prevalence:.4f}), got AP={ap:.4f}"
    )
