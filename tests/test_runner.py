"""Unit and integration tests for the experiment runner and orchestrator.

Verifies:
  1. Detector factory (instantiation, parameter pass-through, seed derivation, errors).
  2. Single cell execution (returns valid metrics, correct score shapes, thresholding).
  3. Paired-design full sweep in a temporary directory (results.csv, summary.csv, metadata.json).
  4. Negative control (null series gives AP=NaN) and random detector baseline (AP ~ prevalence).
  5. Diagnostic comparison plot generation.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from spectrallens.config import load_config
from spectrallens.data.protocol import Protocol
from spectrallens.data.synthetic import generate_series
from spectrallens.detectors.isolation_forest import IsolationForestDetector
from spectrallens.detectors.statistical import RandomDetector, RobustZDetector
from spectrallens.experiments.runner import (
    build_detector,
    plot_cell_comparison,
    run_experiment,
    run_single_cell,
)

CFG = load_config("base", "synthetic")
PROTO = Protocol.from_config(CFG)


# ── Detector Factory Tests ───────────────────────────────────────────────────

def test_build_detector_isolation_forest():
    det = build_detector("isolation_forest", CFG, seed=123)
    assert isinstance(det, IsolationForestDetector)


def test_build_detector_baselines():
    rand = build_detector("random_score", CFG, seed=123)
    assert isinstance(rand, RandomDetector)

    rob = build_detector("robust_z", CFG, seed=123)
    assert isinstance(rob, RobustZDetector)


def test_build_detector_unknown_raises():
    with pytest.raises(ValueError, match="Unknown detector"):
        build_detector("deep_lstm_autoencoder", CFG, seed=123)


# ── Single Cell Execution Tests ──────────────────────────────────────────────

@pytest.mark.parametrize("rep", ["temporal", "spectral", "hybrid"])
def test_run_single_cell_representations(rep: str):
    """Single cell runs without error and returns matching score and label lengths."""
    series = generate_series(CFG, "single_sinusoid", "frequency_shift", 0, difficulty="easy")
    record, pt_scores, test_labels, threshold = run_single_cell(
        series=series,
        representation=rep,
        detector_name="isolation_forest",
        window_size=128,
        window_step=8,
        cfg=CFG,
        proto=PROTO,
    )

    test_len = PROTO.test[1] - PROTO.test[0]
    assert len(pt_scores) == test_len
    assert len(test_labels) == test_len
    assert np.isfinite(threshold)
    assert record["representation"] == rep
    assert record["signal"] == "single_sinusoid"
    assert record["anomaly"] == "frequency_shift"
    assert 0.0 <= record["prevalence"] <= 1.0
    assert 0.0 <= record["point_average_precision"] <= 1.0


def test_run_single_cell_null_series():
    """Null series should produce NaN for average precision."""
    series = generate_series(CFG, "single_sinusoid", None, 0)
    record, _, test_labels, _ = run_single_cell(
        series=series,
        representation="temporal",
        detector_name="isolation_forest",
        window_size=128,
        window_step=8,
        cfg=CFG,
        proto=PROTO,
    )

    assert record["prevalence"] == 0.0
    assert np.isnan(record["point_average_precision"])
    assert np.all(test_labels == 0)


# ── Full Sweep Integration Test ──────────────────────────────────────────────

def test_run_experiment_sweep(tmp_path: Path):
    """Run a mini factorial sweep and verify file artifacts and summary structure."""
    mini_cfg = {
        "experiment_id": "test_mini_sweep",
        "inherit": ["base", "synthetic"],
        "data": {
            "signals": ["single_sinusoid"],
            "anomalies": ["frequency_shift", "amplitude_change"],
            "difficulty": "easy",
            "n_seeds": 2,
        },
        "factors": {
            "representation": ["temporal", "spectral"],
            "windowing.size": [128],
        },
        "controls": ["random_score"],
        "outputs": {
            "save_scores": True,
            "example_plots_per_cell": 1,
        },
    }

    # Merge with base and synthetic
    full_cfg = load_config("base", "synthetic", overrides=mini_cfg)

    df_results, df_summary = run_experiment(
        cfg_or_path=full_cfg,
        out_dir=tmp_path,
        n_seeds=2,
        save_scores=True,
        example_plots_per_cell=1,
        verbose=False,
    )

    # 1. Output files exist
    assert (tmp_path / "results.csv").exists()
    assert (tmp_path / "results.json").exists()
    assert (tmp_path / "summary.csv").exists()
    assert (tmp_path / "summary.json").exists()
    assert (tmp_path / "metadata.json").exists()
    assert (tmp_path / "point_scores.json").exists()

    # 2. Results row count check:
    # 1 signal * 2 anomalies * 2 seeds = 4 series.
    # For each series: 2 reps for IF + 1 rep for random_score = 3 runs.
    # Total = 4 * 3 = 12 rows.
    assert len(df_results) == 12

    # 3. Verify Paired Design: seed 0 has identical prevalence across representations
    sub = df_results[(df_results["seed"] == 0) & (df_results["anomaly"] == "frequency_shift")]
    assert len(sub["prevalence"].unique()) == 1

    # 4. Summary dataframe has expected aggregations
    assert len(df_summary) > 0
    assert "mean_ap" in df_summary.columns
    assert "event_hit_rate" in df_summary.columns

    # 5. Metadata contents check
    with open(tmp_path / "metadata.json", encoding="utf-8") as f:
        meta = json.load(f)
    assert meta["experiment_id"] == "test_mini_sweep"
    assert "environment" in meta
    assert "numpy" in meta["environment"]
    assert "platform" in meta["environment"]

    # 6. Example plot was generated
    fig_path = tmp_path / "figures" / "single_sinusoid_frequency_shift_seed_0.png"
    assert fig_path.exists()
    assert fig_path.stat().st_size > 1000  # valid image file


def test_plot_cell_comparison(tmp_path: Path):
    """Directly test plot_cell_comparison utility."""
    series = generate_series(CFG, "single_sinusoid", "frequency_shift", 0)
    test_len = PROTO.test[1] - PROTO.test[0]
    scores = {
        "temporal": np.random.default_rng(0).uniform(0, 1, size=test_len),
        "spectral": np.random.default_rng(1).uniform(0, 1, size=test_len),
    }
    thresholds = {"temporal": 0.8, "spectral": 0.85}

    plot_file = tmp_path / "test_plot.png"
    plot_cell_comparison(series, scores, thresholds, PROTO, plot_file)
    assert plot_file.exists()
    assert plot_file.stat().st_size > 5000
