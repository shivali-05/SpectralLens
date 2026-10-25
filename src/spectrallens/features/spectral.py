"""Spectral feature extraction from time-series windows via FFT.

Feature set v1 (10 features) — evaluated in the RQ1 representation study.

For every window of length W sampled at rate fs (default 100 Hz), the signal is
optionally mean-centered, windowed (default Hann) to mitigate spectral leakage,
and transformed via real FFT (rfft). The magnitude and power spectra are then
condensed into 10 physical and statistical spectral descriptors:
  1.  dominant_freq      — frequency (Hz) of highest magnitude bin (excluding DC)
  2.  dominant_magnitude — magnitude of the peak frequency bin
  3.  spectral_energy    — total spectral energy (sum of power spectrum)
  4.  spectral_entropy   — normalized Shannon entropy of power distribution in [0, 1]
  5.  spectral_centroid  — spectral center of mass (frequency-weighted power average, Hz)
  6.  spectral_bandwidth — spectral spread (std of frequency around centroid, Hz)
  7.  band_ratio_low     — fraction of spectral energy in [0, 0.12 * f_Nyquist)
  8.  band_ratio_mid     — fraction of spectral energy in [0.12, 0.30 * f_Nyquist)
  9.  band_ratio_high    — fraction of spectral energy in [0.30, 1.0 * f_Nyquist]
  10. peak_count         — count of prominent peaks in magnitude spectrum
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.signal import find_peaks
from scipy.signal.windows import get_window

FEATURE_NAMES = [
    "dominant_freq",
    "dominant_magnitude",
    "spectral_energy",
    "spectral_entropy",
    "spectral_centroid",
    "spectral_bandwidth",
    "band_ratio_low",
    "band_ratio_mid",
    "band_ratio_high",
    "peak_count",
]
N_SPECTRAL_FEATURES = len(FEATURE_NAMES)


def extract_spectral_features(
    windows: np.ndarray,
    cfg: dict[str, Any] | None = None,
    *,
    fs: float = 100.0,
    window_fn: str | None = "hann",
    remove_mean: bool = True,
    band_edges_nyquist_frac: tuple[float, ...] | list[float] = (0.0, 0.12, 0.30, 1.0),
    peak_rule: dict[str, Any] | None = None,
    entropy: str = "normalized",
) -> np.ndarray:
    """Compute spectral feature set v1 for each window.

    Args:
        windows: 2D array of shape (n_windows, window_size). Each row is one window.
        cfg: Optional full or partial config dict (e.g. loaded from base.yaml).
            If supplied, sampling and spectral feature parameters are read from cfg.
        fs: Sampling frequency in Hz.
        window_fn: Window function name (e.g. 'hann', 'hamming', 'boxcar', None).
        remove_mean: If True, subtract the window mean before applying window_fn.
        band_edges_nyquist_frac: Fractions of Nyquist frequency defining band edges.
            Default [0.0, 0.12, 0.30, 1.0] corresponds to 0-6, 6-15, 15-50 Hz at fs=100.
        peak_rule: Dict specifying peak finding criteria, e.g. {"prominence_frac_of_max": 0.10}.
        entropy: 'normalized' for Shannon entropy divided by ln(K-1), or 'raw'.

    Returns:
        2D feature matrix of shape (n_windows, 10).
    """
    windows = np.asarray(windows, dtype=np.float64)
    if windows.ndim != 2:
        raise ValueError(
            f"windows must be 2D (n_windows, window_size), got shape {windows.shape}"
        )
    if windows.shape[1] < 4:
        raise ValueError(
            f"window_size must be >= 4 for spectral features, got {windows.shape[1]}"
        )

    # Resolve parameters from cfg if provided
    if cfg is not None:
        fs = float(cfg.get("sampling", {}).get("fs", fs))
        spec_cfg = cfg.get("features", {}).get("spectral", {})
        window_fn = spec_cfg.get("window_fn", window_fn)
        remove_mean = bool(spec_cfg.get("remove_mean", remove_mean))
        band_edges_nyquist_frac = spec_cfg.get(
            "band_edges_nyquist_frac", band_edges_nyquist_frac
        )
        peak_rule = spec_cfg.get("peak_rule", peak_rule)
        entropy = spec_cfg.get("entropy", entropy)

    if peak_rule is None:
        peak_rule = {"prominence_frac_of_max": 0.10}

    n, W = windows.shape

    # 1. Mean removal (DC suppression)
    if remove_mean:
        x = windows - np.mean(windows, axis=1, keepdims=True)
    else:
        x = windows.copy()

    # 2. Windowing to reduce spectral leakage
    if window_fn in (None, "none", "boxcar", "rectangular"):
        xw = x
    else:
        win = get_window(window_fn, W, fftbins=True)
        xw = x * win[np.newaxis, :]

    # 3. Real FFT and frequency bins
    X = np.fft.rfft(xw, axis=1)  # shape (n, K) where K = W // 2 + 1
    freqs = np.fft.rfftfreq(W, d=1.0 / fs)  # shape (K,)

    mag = np.abs(X)
    power = mag ** 2

    # AC components (excluding DC bin 0)
    mag_ac = mag[:, 1:]
    freqs_ac = freqs[1:]
    power_ac = power[:, 1:]
    ac_power = np.sum(power_ac, axis=1)  # shape (n,)
    total_power = np.sum(power, axis=1)  # shape (n,)

    # ── Feature 1 & 2: Dominant frequency and dominant magnitude ─────────────
    argmax_ac = np.argmax(mag_ac, axis=1)
    dom_freq = freqs_ac[argmax_ac]
    dom_mag = mag_ac[np.arange(n), argmax_ac]

    # Flat / zero-power windows guard
    flat_mask = ac_power <= 1e-12
    dom_freq[flat_mask] = 0.0
    dom_mag[flat_mask] = 0.0

    # ── Feature 3: Spectral energy ───────────────────────────────────────────
    spectral_energy = total_power

    # ── Feature 4: Spectral entropy ──────────────────────────────────────────
    # Normalized Shannon entropy of the AC power distribution: H = -sum(p * log(p)) / log(K-1)
    p = np.where(
        ac_power[:, np.newaxis] > 1e-12,
        power_ac / np.maximum(ac_power[:, np.newaxis], 1e-12),
        0.0,
    )
    log_p = np.where(p > 1e-12, np.log(np.maximum(p, 1e-12)), 0.0)
    ent_raw = -np.sum(p * log_p, axis=1)

    n_ac_bins = power_ac.shape[1]
    if entropy == "normalized" and n_ac_bins > 1:
        spectral_entropy = np.where(
            ac_power > 1e-12,
            ent_raw / np.log(n_ac_bins),
            0.0,
        )
        spectral_entropy = np.clip(spectral_entropy, 0.0, 1.0)
    else:
        spectral_entropy = np.where(ac_power > 1e-12, ent_raw, 0.0)

    # ── Feature 5: Spectral centroid ─────────────────────────────────────────
    spectral_centroid = np.where(
        ac_power > 1e-12,
        np.sum(freqs_ac[np.newaxis, :] * power_ac, axis=1)
        / np.maximum(ac_power, 1e-12),
        0.0,
    )

    # ── Feature 6: Spectral bandwidth (spread around centroid) ───────────────
    freq_diff = freqs_ac[np.newaxis, :] - spectral_centroid[:, np.newaxis]
    var_f = np.where(
        ac_power > 1e-12,
        np.sum((freq_diff ** 2) * power_ac, axis=1) / np.maximum(ac_power, 1e-12),
        0.0,
    )
    spectral_bandwidth = np.sqrt(np.maximum(var_f, 0.0))

    # ── Feature 7, 8, 9: Band energy ratios ──────────────────────────────────
    nyquist = fs / 2.0
    band_edges = list(band_edges_nyquist_frac)
    n_bands = len(band_edges) - 1

    band_ratios = []
    for b in range(n_bands):
        low_f = band_edges[b] * nyquist
        high_f = band_edges[b + 1] * nyquist
        if b < n_bands - 1:
            mask = (freqs >= low_f) & (freqs < high_f)
        else:
            mask = (freqs >= low_f) & (freqs <= high_f)

        bp = np.sum(power[:, mask], axis=1)
        ratio = np.where(
            total_power > 1e-12,
            bp / np.maximum(total_power, 1e-12),
            0.0,
        )
        band_ratios.append(ratio)

    # For default 3 bands, unpack low, mid, high
    if n_bands >= 3:
        band_ratio_low = band_ratios[0]
        band_ratio_mid = band_ratios[1]
        band_ratio_high = band_ratios[2]
    else:
        band_ratio_low = band_ratios[0] if n_bands > 0 else np.zeros(n)
        band_ratio_mid = band_ratios[1] if n_bands > 1 else np.zeros(n)
        band_ratio_high = np.zeros(n)

    # ── Feature 10: Peak count ───────────────────────────────────────────────
    prom_frac = float(peak_rule.get("prominence_frac_of_max", 0.10))
    peak_counts = np.zeros(n, dtype=np.float64)

    for i in range(n):
        if not flat_mask[i]:
            m = mag[i].copy()
            m[0] = 0.0  # zero DC bin so it cannot trigger a peak
            max_m = np.max(m)
            # pad with 0 at the end to allow peak detection at Nyquist
            m_pad = np.pad(m, (0, 1), mode="constant", constant_values=0.0)
            peaks, _ = find_peaks(m_pad, prominence=prom_frac * max_m)
            peak_counts[i] = float(len(peaks))

    return np.column_stack(
        [
            dom_freq,
            dom_mag,
            spectral_energy,
            spectral_entropy,
            spectral_centroid,
            spectral_bandwidth,
            band_ratio_low,
            band_ratio_mid,
            band_ratio_high,
            peak_counts,
        ]
    )


def extract_hybrid_features(
    windows: np.ndarray,
    cfg: dict[str, Any] | None = None,
    **spectral_kwargs: Any,
) -> np.ndarray:
    """Compute hybrid (temporal + spectral) feature set v1 for each window.

    Concatenates the 11 temporal features and 10 spectral features into a
    single 21-dimensional representation per window.

    Args:
        windows: 2D array of shape (n_windows, window_size).
        cfg: Optional config dict passed to both extractors.
        **spectral_kwargs: Keyword arguments forwarded to extract_spectral_features.

    Returns:
        2D feature matrix of shape (n_windows, 21).
    """
    from spectrallens.features.temporal import extract_temporal_features

    temporal = extract_temporal_features(windows)
    spectral = extract_spectral_features(windows, cfg=cfg, **spectral_kwargs)
    return np.column_stack([temporal, spectral])


def extract_features(
    windows: np.ndarray,
    representation: str = "temporal",
    cfg: dict[str, Any] | None = None,
    **kwargs: Any,
) -> np.ndarray:
    """Unified entry point for window feature extraction across representations.

    Args:
        windows: 2D array of shape (n_windows, window_size).
        representation: One of 'temporal', 'spectral', or 'hybrid'.
        cfg: Optional config dict.
        **kwargs: Extra parameters passed to the feature extractor.

    Returns:
        2D feature matrix (n_windows, n_features).
    """
    from spectrallens.features.temporal import extract_temporal_features

    rep = representation.lower().strip()
    if rep == "temporal":
        return extract_temporal_features(windows)
    elif rep == "spectral":
        return extract_spectral_features(windows, cfg=cfg, **kwargs)
    elif rep == "hybrid":
        return extract_hybrid_features(windows, cfg=cfg, **kwargs)
    else:
        raise ValueError(
            f"Unknown representation '{representation}', "
            "expected 'temporal', 'spectral', or 'hybrid'"
        )
