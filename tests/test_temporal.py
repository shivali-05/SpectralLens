"""Unit tests for temporal feature extraction.

Every feature is checked against an analytically derivable value on simple toy windows
so bugs in vectorised NumPy code are caught immediately.
"""

import numpy as np
import pytest

from spectrallens.features.temporal import (
    FEATURE_NAMES,
    N_TEMPORAL_FEATURES,
    extract_temporal_features,
)


def _single_window(values: list[float]) -> np.ndarray:
    """Wrap a list as a (1, W) window array."""
    return np.array([values], dtype=np.float64)


# ── feature count ────────────────────────────────────────────────────────────

def test_feature_names_and_count_consistent():
    assert len(FEATURE_NAMES) == N_TEMPORAL_FEATURES == 11


def test_output_shape():
    windows = np.random.default_rng(0).standard_normal((50, 64))
    feats = extract_temporal_features(windows)
    assert feats.shape == (50, N_TEMPORAL_FEATURES)


# ── per-feature analytical checks ────────────────────────────────────────────

def test_mean_feature():
    w = _single_window([1.0, 2.0, 3.0, 4.0])
    f = extract_temporal_features(w)
    np.testing.assert_allclose(f[0, FEATURE_NAMES.index("mean")], 2.5)


def test_median_feature():
    w = _single_window([1.0, 2.0, 3.0, 4.0])
    f = extract_temporal_features(w)
    np.testing.assert_allclose(f[0, FEATURE_NAMES.index("median")], 2.5)

    w_odd = _single_window([1.0, 5.0, 3.0])
    f_odd = extract_temporal_features(w_odd)
    np.testing.assert_allclose(f_odd[0, FEATURE_NAMES.index("median")], 3.0)


def test_std_feature():
    w = _single_window([2.0, 2.0, 2.0, 2.0])  # flat → std = 0
    f = extract_temporal_features(w)
    np.testing.assert_allclose(f[0, FEATURE_NAMES.index("std")], 0.0)

    w2 = _single_window([0.0, 2.0])  # population std = 1.0
    f2 = extract_temporal_features(w2)
    np.testing.assert_allclose(f2[0, FEATURE_NAMES.index("std")], 1.0)


def test_min_max_range_features():
    w = _single_window([-3.0, 0.0, 5.0, 1.0])
    f = extract_temporal_features(w)
    assert f[0, FEATURE_NAMES.index("min")] == -3.0
    assert f[0, FEATURE_NAMES.index("max")] == 5.0
    np.testing.assert_allclose(f[0, FEATURE_NAMES.index("range")], 8.0)


def test_iqr_feature():
    # linearly spaced 0..1 — Q25=0.25, Q75=0.75 → IQR=0.5
    w = _single_window(list(np.linspace(0, 1, 101)))
    f = extract_temporal_features(w)
    np.testing.assert_allclose(f[0, FEATURE_NAMES.index("iqr")], 0.5, atol=0.01)


def test_slope_feature_increasing():
    # Perfectly linear increasing: y = t  → slope = 1.0
    w = _single_window(list(range(10)))
    f = extract_temporal_features(w)
    np.testing.assert_allclose(f[0, FEATURE_NAMES.index("slope")], 1.0, atol=1e-10)


def test_slope_feature_constant():
    # Flat signal → slope = 0
    w = _single_window([3.0] * 8)
    f = extract_temporal_features(w)
    np.testing.assert_allclose(f[0, FEATURE_NAMES.index("slope")], 0.0, atol=1e-12)


def test_mean_abs_diff_feature():
    # [0, 1, 0, 1] → diffs = [1, -1, 1] → mean |diff| = 1.0
    w = _single_window([0.0, 1.0, 0.0, 1.0])
    f = extract_temporal_features(w)
    np.testing.assert_allclose(f[0, FEATURE_NAMES.index("mean_abs_diff")], 1.0)


def test_std_diff_feature():
    # [0, 2, 0, 2] → diffs = [2, -2, 2] → std_diff (population) ≈ 2.0 - 2/3 ≈ ...
    # diffs = [2, -2, 2], mean = 2/3, deviations = [4/3, -8/3, 4/3]
    # var = (16/9 + 64/9 + 16/9) / 3 = 96/27 = 32/9
    # std = sqrt(32/9) ≈ 1.886
    w = _single_window([0.0, 2.0, 0.0, 2.0])
    f = extract_temporal_features(w)
    diffs = np.diff([0.0, 2.0, 0.0, 2.0])
    expected_std_diff = np.std(diffs, ddof=0)
    np.testing.assert_allclose(
        f[0, FEATURE_NAMES.index("std_diff")], expected_std_diff, atol=1e-12
    )


def test_acf1_feature_perfect_sine():
    # A clean sinusoid has high positive acf1 near 1 (neighboring samples correlated)
    t = np.linspace(0, 2 * np.pi, 128, endpoint=False)
    sine = np.sin(t)
    w = sine[np.newaxis, :]
    f = extract_temporal_features(w)
    # ACF1 of a dense sinusoid with 128 samples per cycle should be close to 1.0
    assert f[0, FEATURE_NAMES.index("acf1")] > 0.95


def test_acf1_flat_window_returns_zero():
    w = _single_window([5.0] * 16)
    f = extract_temporal_features(w)
    np.testing.assert_allclose(f[0, FEATURE_NAMES.index("acf1")], 0.0, atol=1e-12)


def test_acf1_iid_noise_near_zero():
    rng = np.random.default_rng(7)
    noise = rng.standard_normal((1, 10_000))
    f = extract_temporal_features(noise)
    # Large-sample iid noise: ACF1 should be near 0
    assert abs(f[0, FEATURE_NAMES.index("acf1")]) < 0.05


# ── validation errors ─────────────────────────────────────────────────────────

def test_invalid_input_raises():
    with pytest.raises(ValueError, match="2D"):
        extract_temporal_features(np.ones(10))
    with pytest.raises(ValueError, match="window_size must be >= 2"):
        extract_temporal_features(np.ones((5, 1)))
