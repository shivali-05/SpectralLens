"""Unit tests for spectral and hybrid feature extraction.

Verifies:
  1. Dimensionality, naming, and dispatch consistency across representations.
  2. Fourier transform accuracy and bin recovery on known sinusoids.
  3. Spectral leakage suppression of Hann vs rectangular windowing.
  4. Spectral entropy separation (pure sinusoid vs white noise).
  5. Centroid, bandwidth, and band energy partitioning.
  6. Peak counting accuracy on multi-component signals.
  7. Numerical stability on zero-variance and edge-case windows (no NaN / Inf / warnings).
"""

from __future__ import annotations

import numpy as np
import pytest

from spectrallens.config import load_config
from spectrallens.features import (
    FEATURE_SET_VERSION,
    HYBRID_FEATURE_NAMES,
    N_HYBRID_FEATURES,
    N_SPECTRAL_FEATURES,
    N_TEMPORAL_FEATURES,
    SPECTRAL_FEATURE_NAMES,
    TEMPORAL_FEATURE_NAMES,
    extract_features,
    extract_hybrid_features,
    extract_spectral_features,
)


def _sine_window(freq: float, fs: float = 100.0, W: int = 128, amp: float = 1.0) -> np.ndarray:
    """Generate a single window containing a pure sine wave."""
    t = np.arange(W) / fs
    x = amp * np.sin(2 * np.pi * freq * t)
    return x[np.newaxis, :]


# ── Feature count and dimensionality ─────────────────────────────────────────

def test_feature_set_v1_frozen_and_immutable():
    """Verify Feature Set v1 tag (D-010) and exact immutable feature composition."""
    assert FEATURE_SET_VERSION == "v1"

    expected_temporal = [
        "mean", "median", "std", "min", "max", "range",
        "iqr", "slope", "mean_abs_diff", "std_diff", "acf1",
    ]
    expected_spectral = [
        "dominant_freq", "dominant_magnitude", "spectral_energy",
        "spectral_entropy", "spectral_centroid", "spectral_bandwidth",
        "band_ratio_low", "band_ratio_mid", "band_ratio_high", "peak_count",
    ]

    assert TEMPORAL_FEATURE_NAMES == expected_temporal
    assert SPECTRAL_FEATURE_NAMES == expected_spectral
    assert HYBRID_FEATURE_NAMES == expected_temporal + expected_spectral
    assert len(TEMPORAL_FEATURE_NAMES) == N_TEMPORAL_FEATURES == 11
    assert len(SPECTRAL_FEATURE_NAMES) == N_SPECTRAL_FEATURES == 10
    assert len(HYBRID_FEATURE_NAMES) == N_HYBRID_FEATURES == 21


def test_spectral_output_shape():
    windows = np.random.default_rng(42).standard_normal((25, 64))
    feats = extract_spectral_features(windows)
    assert feats.shape == (25, 10)


def test_hybrid_output_shape_and_composition():
    windows = np.random.default_rng(42).standard_normal((15, 128))
    hybrid = extract_hybrid_features(windows)
    assert hybrid.shape == (15, 21)

    temporal = extract_features(windows, representation="temporal")
    spectral = extract_features(windows, representation="spectral")
    np.testing.assert_allclose(hybrid[:, :11], temporal)
    np.testing.assert_allclose(hybrid[:, 11:], spectral)


def test_extract_features_dispatcher():
    windows = np.ones((5, 32)) * 3.0
    t = extract_features(windows, representation="temporal")
    s = extract_features(windows, representation="spectral")
    h = extract_features(windows, representation="hybrid")

    assert t.shape == (5, 11)
    assert s.shape == (5, 10)
    assert h.shape == (5, 21)

    with pytest.raises(ValueError, match="Unknown representation"):
        extract_features(windows, representation="wavelet")


def test_invalid_input_shapes():
    with pytest.raises(ValueError, match="must be 2D"):
        extract_spectral_features(np.array([1.0, 2.0, 3.0]))

    with pytest.raises(ValueError, match="window_size must be >= 4"):
        extract_spectral_features(np.zeros((2, 3)))


# ── Fourier Transform & Frequency Recovery ───────────────────────────────────

def test_dominant_frequency_recovery():
    """Dominant frequency should match the injected tone within bin resolution."""
    fs = 100.0
    W = 128
    df = fs / W  # 0.78125 Hz

    # 5 Hz tone
    w5 = _sine_window(freq=5.0, fs=fs, W=W)
    f5 = extract_spectral_features(w5, fs=fs)
    dom_f5 = f5[0, SPECTRAL_FEATURE_NAMES.index("dominant_freq")]
    assert abs(dom_f5 - 5.0) <= df

    # 8 Hz tone (frequency shift)
    w8 = _sine_window(freq=8.0, fs=fs, W=W)
    f8 = extract_spectral_features(w8, fs=fs)
    dom_f8 = f8[0, SPECTRAL_FEATURE_NAMES.index("dominant_freq")]
    assert abs(dom_f8 - 8.0) <= df

    # 20 Hz tone (local insertion)
    w20 = _sine_window(freq=20.0, fs=fs, W=W)
    f20 = extract_spectral_features(w20, fs=fs)
    dom_f20 = f20[0, SPECTRAL_FEATURE_NAMES.index("dominant_freq")]
    assert abs(dom_f20 - 20.0) <= df


def test_amplitude_scaling_and_parseval():
    """Doubling amplitude quadruples spectral energy and doubles dominant magnitude."""
    w1 = _sine_window(freq=5.0, amp=1.0)
    w2 = _sine_window(freq=5.0, amp=2.0)

    f1 = extract_spectral_features(w1)
    f2 = extract_spectral_features(w2)

    mag1 = f1[0, SPECTRAL_FEATURE_NAMES.index("dominant_magnitude")]
    mag2 = f2[0, SPECTRAL_FEATURE_NAMES.index("dominant_magnitude")]
    np.testing.assert_allclose(mag2 / mag1, 2.0, rtol=1e-5)

    e1 = f1[0, SPECTRAL_FEATURE_NAMES.index("spectral_energy")]
    e2 = f2[0, SPECTRAL_FEATURE_NAMES.index("spectral_energy")]
    np.testing.assert_allclose(e2 / e1, 4.0, rtol=1e-5)


# ── Leakage Control: Hann vs Boxcar ──────────────────────────────────────────

def test_hann_window_reduces_spectral_leakage():
    """Hann window should suppress distant sidelobe leakage by over 100x vs boxcar."""
    fs = 100.0
    W = 128
    # 5.4 Hz is intentionally off-grid (between bin 6 and 7)
    w = _sine_window(freq=5.4, fs=fs, W=W)

    # With rectangular (boxcar) window
    f_box = extract_spectral_features(w, fs=fs, window_fn="boxcar")
    # With default Hann window
    f_hann = extract_spectral_features(w, fs=fs, window_fn="hann")

    # High band (15 to 50 Hz) should have far less leakage power with Hann window
    high_box = f_box[0, SPECTRAL_FEATURE_NAMES.index("band_ratio_high")]
    high_hann = f_hann[0, SPECTRAL_FEATURE_NAMES.index("band_ratio_high")]

    assert high_box > 0.001
    assert high_hann < 1e-6
    assert high_box / max(high_hann, 1e-12) > 100.0


# ── Entropy: Sinusoid vs White Noise ─────────────────────────────────────────

def test_spectral_entropy_separation():
    """Pure sinusoid has low entropy; white noise has high entropy."""
    w_sine = _sine_window(freq=5.0)
    rng = np.random.default_rng(2026)
    w_noise = rng.standard_normal((1, 128))

    f_sine = extract_spectral_features(w_sine)
    f_noise = extract_spectral_features(w_noise)

    ent_sine = f_sine[0, SPECTRAL_FEATURE_NAMES.index("spectral_entropy")]
    ent_noise = f_noise[0, SPECTRAL_FEATURE_NAMES.index("spectral_entropy")]

    assert 0.0 <= ent_sine < 0.35
    assert 0.85 < ent_noise <= 1.0


# ── Centroid and Bandwidth ───────────────────────────────────────────────────

def test_centroid_and_bandwidth():
    """Centroid of 5 Hz sine should be ~5 Hz, bandwidth tight; noise has wide spread."""
    w5 = _sine_window(freq=5.0)
    rng = np.random.default_rng(2026)
    w_noise = rng.standard_normal((1, 128))

    f5 = extract_spectral_features(w5)
    fn = extract_spectral_features(w_noise)

    c5 = f5[0, SPECTRAL_FEATURE_NAMES.index("spectral_centroid")]
    b5 = f5[0, SPECTRAL_FEATURE_NAMES.index("spectral_bandwidth")]

    cn = fn[0, SPECTRAL_FEATURE_NAMES.index("spectral_centroid")]
    bn = fn[0, SPECTRAL_FEATURE_NAMES.index("spectral_bandwidth")]

    # 5 Hz sine centroid is close to 5 Hz with small spread
    assert abs(c5 - 5.0) < 0.15
    assert b5 < 1.0

    # White noise centroid is around mid-spectrum (25 Hz) with large spread (> 10 Hz)
    assert 20.0 < cn < 30.0
    assert bn > 10.0


# ── Band Energy Ratios ───────────────────────────────────────────────────────

def test_band_energy_ratios_partition_and_response():
    """Band ratios must sum to 1.0 and concentrate energy in the expected band."""
    fs = 100.0
    w_low = _sine_window(freq=5.0, fs=fs)    # 5 Hz is in low band [0, 6)
    w_mid = _sine_window(freq=8.0, fs=fs)    # 8 Hz is in mid band [6, 15)
    w_high = _sine_window(freq=20.0, fs=fs)  # 20 Hz is in high band [15, 50]

    for w, target_col in [
        (w_low, "band_ratio_low"),
        (w_mid, "band_ratio_mid"),
        (w_high, "band_ratio_high"),
    ]:
        f = extract_spectral_features(w, fs=fs)
        r_low = f[0, SPECTRAL_FEATURE_NAMES.index("band_ratio_low")]
        r_mid = f[0, SPECTRAL_FEATURE_NAMES.index("band_ratio_mid")]
        r_high = f[0, SPECTRAL_FEATURE_NAMES.index("band_ratio_high")]

        # Partition: sum to 1.0
        np.testing.assert_allclose(r_low + r_mid + r_high, 1.0, atol=1e-6)

        # Concentrates in target band
        target_val = f[0, SPECTRAL_FEATURE_NAMES.index(target_col)]
        assert target_val > 0.95


# ── Peak Counting ────────────────────────────────────────────────────────────

def test_peak_counts():
    """Single sinusoid -> 1 peak; 3 sinusoids -> 3 peaks."""
    fs = 100.0
    W = 128
    t = np.arange(W) / fs

    # 1 peak: 5 Hz
    x1 = np.sin(2 * np.pi * 5 * t)[np.newaxis, :]
    f1 = extract_spectral_features(x1, fs=fs)
    assert f1[0, SPECTRAL_FEATURE_NAMES.index("peak_count")] == 1.0

    # 3 peaks: 2 Hz, 5 Hz, 11 Hz
    x3 = (
        np.sin(2 * np.pi * 2 * t)
        + np.sin(2 * np.pi * 5 * t)
        + np.sin(2 * np.pi * 11 * t)
    )[np.newaxis, :]
    f3 = extract_spectral_features(x3, fs=fs)
    assert f3[0, SPECTRAL_FEATURE_NAMES.index("peak_count")] == 3.0


# ── Numerical Stability & Edge Cases ─────────────────────────────────────────

def test_zero_variance_windows():
    """Flat / zero windows should produce finite, valid features with zero warnings."""
    flat_zero = np.zeros((3, 64))
    flat_const = np.ones((3, 64)) * 42.0

    for w in [flat_zero, flat_const]:
        f = extract_spectral_features(w)
        assert np.all(np.isfinite(f))
        assert np.all(f[:, SPECTRAL_FEATURE_NAMES.index("spectral_energy")] == 0.0)
        assert np.all(f[:, SPECTRAL_FEATURE_NAMES.index("dominant_freq")] == 0.0)
        assert np.all(f[:, SPECTRAL_FEATURE_NAMES.index("spectral_entropy")] == 0.0)
        assert np.all(f[:, SPECTRAL_FEATURE_NAMES.index("peak_count")] == 0.0)


def test_config_integration():
    """Passing a full config dict should match default parameters."""
    cfg = load_config("base", "synthetic")
    w = _sine_window(freq=5.0)

    f_cfg = extract_spectral_features(w, cfg=cfg)
    f_def = extract_spectral_features(w)
    np.testing.assert_allclose(f_cfg, f_def)
