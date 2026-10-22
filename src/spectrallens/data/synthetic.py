"""Controlled synthetic signals with a single injected anomaly and known ground truth.

Design rules (see research/DECISIONS.md):

* Random streams are keyed by identity, not by parameters. For a given (signal_type, index) the
  phases and the unit-variance noise draw are identical across anomaly types, difficulty levels,
  durations and noise levels. Comparisons across those factors are therefore paired.
* Anomalies live in the test split only, at least ``anomaly_margin`` samples from its edges.
* Injections avoid trivially detectable artifacts: frequency shifts are phase-continuous and
  blend back to the baseline at their end, amplitude changes use raised-cosine edges, insertions
  are Hann-tapered. Outside its labeled range every series equals its null series exactly.
* Noise is added last, so ``x(noise_std=a) - x(noise_std=b) == (a - b) * z`` exactly.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from spectrallens.config import config_hash
from spectrallens.data.protocol import Protocol
from spectrallens.seeding import make_rng

GENERATOR_VERSION = "0.1.0"
SCHEMA_VERSION = 1
ANOMALY_TYPES = (
    "point_spike",
    "amplitude_change",
    "frequency_shift",
    "local_frequency_insertion",
)


@dataclass
class SyntheticSeries:
    x: np.ndarray            # float64, shape (N,)
    labels: np.ndarray       # int8, shape (N,), 1 inside the anomaly
    metadata: dict[str, Any]

    @property
    def series_id(self) -> str:
        return self.metadata["series_id"]


# ----------------------------------------------------------------------------- helpers


def data_hash(x: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(x, dtype="<f8").tobytes()).hexdigest()


def make_series_id(
    signal_type: str,
    anomaly_type: str | None,
    difficulty: str,
    duration: int,
    noise_std: float,
    index: int,
) -> str:
    return (
        f"{signal_type}__{anomaly_type or 'null'}__{difficulty}"
        f"__d{duration}__n{noise_std:g}__{index:05d}"
    )


def resolve_anomaly_params(
    cfg: dict, anomaly_type: str, difficulty: str = "default", duration: int | None = None
) -> dict[str, Any]:
    """Merge the base anomaly spec, the difficulty level and an optional duration override."""
    specs = cfg["anomalies"]
    if anomaly_type not in specs:
        raise ValueError(f"Unknown anomaly type '{anomaly_type}'. Known: {sorted(specs)}")
    spec = specs[anomaly_type]
    levels = spec.get("levels", {})
    if difficulty not in levels:
        raise ValueError(f"Unknown difficulty '{difficulty}' for {anomaly_type}: {sorted(levels)}")
    params = {k: v for k, v in spec.items() if k != "levels"}
    params.update(levels[difficulty])
    if duration is not None:
        if anomaly_type == "point_spike" and int(duration) != 1:
            raise ValueError("point_spike has a fixed duration of 1 sample")
        params["duration"] = int(duration)
    if "duration" not in params:
        raise ValueError(f"No duration defined for {anomaly_type}")
    params["duration"] = int(params["duration"])
    return params


def _periodic(
    components: list[dict],
    phases: np.ndarray,
    n: int,
    fs: float,
    freq_mult: np.ndarray | None = None,
    envelope: np.ndarray | None = None,
) -> np.ndarray:
    """Sum of sinusoids. A frequency multiplier is integrated so phase stays continuous."""
    m = np.ones(n) if freq_mult is None else freq_mult
    cum = np.concatenate(([0.0], np.cumsum(m)[:-1]))   # cum[k] = sum of m[:k]; arange(n) if m=1
    out = np.zeros(n)
    for comp, phi in zip(components, phases, strict=True):
        out += comp["amp"] * np.sin(phi + 2.0 * np.pi * comp["f"] * cum / fs)
    return out if envelope is None else out * envelope


def _raised_cosine_envelope(n: int, start: int, end: int, factor: float, ramp: int) -> np.ndarray:
    """Envelope equal to 1 outside [start, end), ``factor`` in the middle, smooth edges inside."""
    ramp = int(min(ramp, (end - start) // 2))
    w = np.zeros(n)
    w[start:end] = 1.0
    if ramp > 0:
        k = np.arange(1, ramp + 1)
        up = 0.5 * (1.0 - np.cos(np.pi * k / (ramp + 1)))
        w[start : start + ramp] = up
        w[end - ramp : end] = up[::-1]
    return 1.0 + (factor - 1.0) * w


def _blend_back_weights(n: int, start: int, end: int, ramp: int) -> np.ndarray:
    """Weight 1 inside [start, end), 0 outside, easing 1 -> 0 over the last ``ramp`` samples."""
    ramp = int(min(ramp, (end - start) // 2))
    w = np.zeros(n)
    w[start:end] = 1.0
    if ramp > 0:
        k = np.arange(1, ramp + 1)
        w[end - ramp : end] = 0.5 * (1.0 + np.cos(np.pi * k / (ramp + 1)))
    return w


# --------------------------------------------------------------------------- generator


def generate_series(
    cfg: dict,
    signal_type: str,
    anomaly_type: str | None,
    index: int,
    *,
    difficulty: str = "default",
    noise_std: float | None = None,
    duration: int | None = None,
) -> SyntheticSeries:
    """Generate one labeled series. ``anomaly_type=None`` gives a null (no-anomaly) series."""
    if signal_type not in cfg["signals"]:
        raise ValueError(f"Unknown signal type '{signal_type}'. Known: {sorted(cfg['signals'])}")
    seed = int(cfg["seed"])
    fs = float(cfg["sampling"]["fs"])
    protocol = Protocol.from_config(cfg)
    n = protocol.series_length
    noise_std = float(cfg["noise_std"] if noise_std is None else noise_std)

    sig = cfg["signals"][signal_type]
    comps = sig["components"]
    amp_ref = max(c["amp"] for c in comps)

    # Identity-keyed streams: independent of anomaly type / difficulty / noise level.
    if sig.get("phase", "random") == "random":
        phases = make_rng(seed, "phase", signal_type, index).uniform(0, 2 * np.pi, len(comps))
    else:
        phases = np.zeros(len(comps))
    z = make_rng(seed, "noise", signal_type, index).standard_normal(n)

    baseline = _periodic(comps, phases, n, fs)
    sigma_ref = float(np.std(baseline))
    trend = np.zeros(n)
    if sig.get("trend"):
        drift = float(sig["trend"]["total_drift_in_amp"]) * amp_ref
        trend = drift * np.arange(n) / (n - 1)

    labels = np.zeros(n, dtype=np.int8)
    anomaly_meta: dict[str, Any] | None = None
    freq_mult = None
    envelope = None
    blend = None
    additive = np.zeros(n)

    if anomaly_type is None:
        difficulty, dur = "none", 0
    else:
        params = resolve_anomaly_params(cfg, anomaly_type, difficulty, duration)
        dur = params["duration"]
        rng_a = make_rng(seed, "anomaly", signal_type, anomaly_type, index)
        lo, hi = protocol.start_bounds(dur)
        start = int(rng_a.integers(lo, hi + 1))
        end = start + dur
        labels[start:end] = 1

        if anomaly_type == "point_spike":
            sign = 1.0 if rng_a.random() < 0.5 else -1.0
            additive[start] = sign * float(params["magnitude_sigma"]) * sigma_ref
        elif anomaly_type == "amplitude_change":
            envelope = _raised_cosine_envelope(
                n, start, end, float(params["factor"]), int(params.get("ramp", 0))
            )
        elif anomaly_type == "frequency_shift":
            freq_mult = np.ones(n)
            freq_mult[start:end] = float(params["ratio"])
            # Extra phase gained during the shift would otherwise persist after the anomaly.
            blend = _blend_back_weights(n, start, end, int(params.get("ramp", 0)))
        elif anomaly_type == "local_frequency_insertion":
            phi_ins = float(rng_a.uniform(0, 2 * np.pi))
            tt = np.arange(dur)
            taper = np.hanning(dur + 2)[1:-1]
            additive[start:end] = (
                float(params["rel_amp"]) * amp_ref
                * np.sin(phi_ins + 2 * np.pi * float(params["f"]) * tt / fs)
                * taper
            )
        else:  # pragma: no cover - guarded by resolve_anomaly_params
            raise ValueError(anomaly_type)

        anomaly_meta = {
            "type": anomaly_type,
            "start": start,
            "end": end,
            "duration": dur,
            "difficulty": difficulty,
            "params": params,
            "region": "test",
        }

    periodic = _periodic(comps, phases, n, fs, freq_mult, envelope)
    if blend is not None:
        periodic = (1.0 - blend) * baseline + blend * periodic
    clean = periodic + trend + additive
    x = clean + noise_std * z

    series_id = make_series_id(signal_type, anomaly_type, difficulty, dur, noise_std, index)
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "generator_version": GENERATOR_VERSION,
        "series_id": series_id,
        "index": int(index),
        "global_seed": seed,
        "signal_type": signal_type,
        "signal_components": [dict(c) for c in comps],
        "sampling_rate": fs,
        "noise_std": noise_std,
        "series_length": n,
        "split": {k: list(v) for k, v in protocol.splits().items()},
        "anomaly": anomaly_meta,
        "config_hash": config_hash({k: cfg[k] for k in ("seed", "sampling", "protocol", "signals",
                                                        "noise_std", "anomalies")}),
        "data_sha256": data_hash(x),
    }
    return SyntheticSeries(x=x, labels=labels, metadata=metadata)


# ------------------------------------------------------------------------------- I/O


def save_series(series: SyntheticSeries, out_dir: str | Path) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    npz = out / f"{series.series_id}.npz"
    np.savez_compressed(npz, x=series.x, labels=series.labels)
    with open(out / f"{series.series_id}.json", "w", encoding="utf-8") as fh:
        json.dump(series.metadata, fh, indent=2, sort_keys=True)
    return npz


def load_series(out_dir: str | Path, series_id: str, verify: bool = True) -> SyntheticSeries:
    out = Path(out_dir)
    with open(out / f"{series_id}.json", encoding="utf-8") as fh:
        metadata = json.load(fh)
    with np.load(out / f"{series_id}.npz") as arrays:
        x, labels = arrays["x"], arrays["labels"]
    if verify and data_hash(x) != metadata["data_sha256"]:
        raise ValueError(f"Data hash mismatch for {series_id}: file changed or corrupted")
    return SyntheticSeries(x=x, labels=labels, metadata=metadata)


def generate_dataset(
    cfg: dict,
    out_dir: str | Path,
    *,
    signals: list[str],
    anomalies: list[str | None],
    n_per_cell: int,
    difficulty: str = "default",
    noise_std: float | None = None,
    duration: int | None = None,
    start_index: int = 0,
) -> list[str]:
    """Generate and save a grid of series; write a manifest. ``None`` in anomalies = null series."""
    ids: list[str] = []
    for sig in signals:
        for anom in anomalies:
            for i in range(start_index, start_index + n_per_cell):
                dur = None if anom in (None, "point_spike") else duration
                s = generate_series(cfg, sig, anom, i, difficulty=difficulty,
                                    noise_std=noise_std, duration=dur)
                save_series(s, out_dir)
                ids.append(s.series_id)
    manifest = {
        "generator_version": GENERATOR_VERSION,
        "schema_version": SCHEMA_VERSION,
        "n_series": len(ids),
        "series_ids": ids,
    }
    with open(Path(out_dir) / "manifest.json", "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    return ids
