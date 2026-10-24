import numpy as np

from spectrallens.detectors.statistical import RandomDetector, RobustZDetector
from spectrallens.evaluation.metrics import event_hit, point_average_precision


def test_random_detector_average_precision_equals_prevalence():
    """Hypothesis Control check: Random scores must give AP approx equal to prevalence."""
    n_samples = 10_000
    prevalence = 0.05
    n_positives = int(n_samples * prevalence)

    labels = np.zeros(n_samples, dtype=int)
    labels[:n_positives] = 1

    detector = RandomDetector(seed=2026)
    scores = detector.score(labels)

    ap = point_average_precision(labels, scores)
    # Average precision of random ranking should be close to prevalence (0.05)
    assert abs(ap - prevalence) < 0.02


def test_robust_z_detector_on_point_spike():
    """Positive Control check: Robust-Z easily detects an 8-sigma point spike."""
    rng = np.random.default_rng(123)
    clean_signal = np.sin(np.linspace(0, 20 * np.pi, 2000)) + 0.1 * rng.standard_normal(2000)

    # Fit on the first 1000 points (normal training)
    detector = RobustZDetector()
    detector.fit(clean_signal[:1000])

    # Inject an 8-sigma spike in the test split at index 1500
    test_signal = clean_signal.copy()
    spike_idx = 1500
    test_signal[spike_idx] += 8.0 * np.std(clean_signal[:1000])

    scores = detector.score(test_signal[1000:])
    rel_spike_idx = spike_idx - 1000

    # The highest score must be at the spike location
    assert np.argmax(scores) == rel_spike_idx

    labels = np.zeros(1000, dtype=int)
    labels[rel_spike_idx] = 1
    assert event_hit(labels, scores, tolerance=5)
    assert point_average_precision(labels, scores) == 1.0


def test_robust_z_detector_window_scoring():
    # 2D windows shape (n_windows, window_size)
    windows = np.array(
        [
            [0.1, 0.2, -0.1, 0.0],  # normal window
            [0.0, 50.0, 0.1, -0.2],  # anomalous window with huge spike at index 1
            [0.0, -0.1, 0.1, 0.0],  # normal window
        ]
    )
    detector = RobustZDetector()
    detector.fit(windows[0:1])  # fit on normal window

    win_scores = detector.score(windows)
    assert len(win_scores) == 3
    # Window 1 must have massively higher score than window 0 and 2
    assert win_scores[1] > 10.0 * win_scores[0]
    assert np.argmax(win_scores) == 1
