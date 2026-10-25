"""Experiment runner and orchestrator for SpectralLens representation studies.

Executes controlled empirical sweeps across:
  - Signal types (single_sinusoid, multi_sinusoid)
  - Anomaly types (point_spike, amplitude_change, frequency_shift, local_frequency_insertion)
  - Random seeds (paired design: identical series evaluated across representations)
  - Representations (temporal, spectral, hybrid)
  - Detectors (Isolation Forest, plus Random and Robust-Z baselines)
  - Windowing configurations

Outputs:
  - Tabular results (results.csv, results.json)
  - Aggregated metrics summary (summary.csv, summary.json)
  - Provenance metadata (metadata.json) with exact library versions and config hashes
  - Optional visual comparative diagnostic plots
"""

from __future__ import annotations

import copy
import datetime
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn

from spectrallens.config import config_hash, load_experiment
from spectrallens.data.protocol import Protocol
from spectrallens.data.synthetic import SyntheticSeries, generate_series
from spectrallens.data.windowing import extract_split_windows
from spectrallens.detectors.isolation_forest import IsolationForestDetector
from spectrallens.detectors.statistical import RandomDetector, RobustZDetector
from spectrallens.evaluation.metrics import evaluate_predictions
from spectrallens.evaluation.scoring import windows_to_points
from spectrallens.evaluation.thresholding import fit_threshold_from_calibration
from spectrallens.features import extract_features
from spectrallens.seeding import derive_seed


def build_detector(name: str, cfg: dict[str, Any], seed: int) -> Any:
    """Instantiate a detector by name with proper deterministic seeding.

    Args:
        name: Name of detector ('isolation_forest', 'random_score', or 'robust_z').
        cfg: Configuration dictionary.
        seed: Deterministic integer seed for this specific experiment run.

    Returns:
        Configured detector instance with .fit() and .score() methods.
    """
    det_name = name.lower().strip()
    if det_name == "isolation_forest":
        detector_cfg = copy.deepcopy(cfg)
        detector_cfg["seed"] = seed
        return IsolationForestDetector(detector_cfg)
    elif det_name == "random_score":
        random_seed = derive_seed(seed, "random_detector")
        return RandomDetector(seed=random_seed)
    elif det_name == "robust_z":
        return RobustZDetector()
    else:
        raise ValueError(
            f"Unknown detector '{name}'. Expected 'isolation_forest', 'random_score', or 'robust_z'"
        )


def run_single_cell(
    series: SyntheticSeries,
    representation: str,
    detector_name: str,
    window_size: int,
    window_step: int,
    cfg: dict[str, Any],
    proto: Protocol | None = None,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, float]:
    """Execute a single vertical slice evaluation for one series and representation.

    Args:
        series: SyntheticSeries dataclass with signal array, labels, and metadata.
        representation: Feature set ('temporal', 'spectral', or 'hybrid').
        detector_name: Detector identifier ('isolation_forest', 'random_score', 'robust_z').
        window_size: Length of window W in samples.
        window_step: Step size between consecutive windows.
        cfg: Configuration dictionary.
        proto: Protocol instance. If None, derived from cfg.

    Returns:
        Tuple of (metrics_dict, test_point_scores, test_labels, threshold).
    """
    if proto is None:
        proto = Protocol.from_config(cfg)

    # 1. Extract windows strictly within each split (D-001)
    split_windows = extract_split_windows(
        series.x, proto, window_size=window_size, step=window_step
    )

    # 2. Extract feature representations
    fit_feats = extract_features(split_windows["fit"].data, representation, cfg=cfg)
    cal_feats = extract_features(split_windows["calibration"].data, representation, cfg=cfg)
    test_feats = extract_features(split_windows["test"].data, representation, cfg=cfg)

    # 3. Fit detector strictly on fit split (D-008 leakage rule)
    cell_seed = derive_seed(
        cfg.get("seed", 42),
        f"{series.signal_type}_{series.anomaly_type}_{series.index}_{representation}_{detector_name}",
    )
    detector = build_detector(detector_name, cfg, seed=cell_seed)
    detector.fit(fit_feats)

    # 4. Calibration split threshold (D-002)
    cal_win_scores = detector.score(cal_feats)
    quantile = float(cfg.get("threshold", {}).get("quantile", 0.99))
    threshold = fit_threshold_from_calibration(cal_win_scores, quantile=quantile)

    # 5. Test split scoring and point-level mapping (D-003)
    test_win_scores = detector.score(test_feats)
    test_len = proto.test[1] - proto.test[0]
    test_pt_scores = windows_to_points(
        test_win_scores,
        split_windows["test"].indices - proto.test[0],
        length=test_len,
    )

    # 6. Evaluation against ground-truth labels
    test_labels = series.labels[proto.test[0]:proto.test[1]]
    event_tol = int(cfg.get("evaluation", {}).get("event_tolerance", 100))
    metrics = evaluate_predictions(
        test_labels,
        test_pt_scores,
        threshold=threshold,
        event_tolerance=event_tol,
    )

    record: dict[str, Any] = {
        "signal": series.signal_type,
        "anomaly": series.anomaly_type if series.anomaly_type is not None else "null",
        "difficulty": series.difficulty,
        "seed": series.index,
        "representation": representation,
        "detector": detector_name,
        "window_size": window_size,
        "window_step": window_step,
        "prevalence": metrics["prevalence"],
        "point_average_precision": metrics["point_average_precision"],
        "roc_auc": metrics["roc_auc"],
        "event_hit": metrics["event_hit"],
        "f1": metrics.get("f1", float(np.nan)),
        "precision": metrics.get("precision", float(np.nan)),
        "recall": metrics.get("recall", float(np.nan)),
        "threshold": float(threshold),
        "detection_delay": metrics.get("detection_delay"),
    }

    return record, test_pt_scores, test_labels, threshold


def plot_cell_comparison(
    series: SyntheticSeries,
    scores_by_rep: dict[str, np.ndarray],
    threshold_by_rep: dict[str, float],
    proto: Protocol,
    out_path: Path,
) -> None:
    """Generate diagnostic comparative plot showing raw signal and score curves."""
    test_start, test_end = proto.test
    t = np.arange(test_start, test_end)
    sig_test = series.x[test_start:test_end]
    labels_test = series.labels[test_start:test_end]

    n_reps = len(scores_by_rep)
    fig, axes = plt.subplots(
        nrows=n_reps + 1,
        ncols=1,
        figsize=(12, 2.4 * (n_reps + 1)),
        sharex=True,
    )

    # Plot 1: Raw test signal with ground truth anomaly segment
    ax0 = axes[0]
    ax0.plot(t, sig_test, color="#1f77b4", lw=1.0, label="Test signal x[t]")
    if np.any(labels_test == 1):
        ax0.fill_between(
            t,
            sig_test.min(),
            sig_test.max(),
            where=(labels_test == 1),
            color="#d62728",
            alpha=0.25,
            label="True anomaly segment",
        )
    ax0.set_title(
        f"Signal: {series.signal_type} | Anomaly: {series.anomaly_type or 'null'} "
        f"({series.difficulty}) | Seed: {series.index}",
        fontsize=11,
        fontweight="bold",
    )
    ax0.legend(loc="upper right", fontsize=9)
    ax0.grid(True, alpha=0.3)

    # Subsequent plots: representation anomaly scores
    colors = {"temporal": "#ff7f0e", "spectral": "#2ca02c", "hybrid": "#9467bd"}
    for idx, (rep, scores) in enumerate(scores_by_rep.items(), start=1):
        ax = axes[idx]
        c = colors.get(rep, "#333333")
        ax.plot(t, scores, color=c, lw=1.2, label=f"{rep.capitalize()} score")
        th = threshold_by_rep.get(rep)
        if th is not None and np.isfinite(th):
            ax.axhline(th, color="#d62728", ls="--", lw=1.0, label=f"Threshold (q=0.99): {th:.3f}")

        if np.any(labels_test == 1):
            ax.fill_between(
                t,
                np.nanmin(scores) if np.any(np.isfinite(scores)) else 0.0,
                np.nanmax(scores) if np.any(np.isfinite(scores)) else 1.0,
                where=(labels_test == 1),
                color="#d62728",
                alpha=0.15,
            )
        ax.set_ylabel("Score", fontsize=9)
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (sample index)", fontsize=10)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run_experiment(
    cfg_or_path: dict[str, Any] | str | Path,
    out_dir: str | Path | None = None,
    n_seeds: int | None = None,
    save_scores: bool = True,
    example_plots_per_cell: int | None = None,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Execute the factorial experiment sweep defined by configuration.

    Args:
        cfg_or_path: Full config dict or path to an experiment YAML file.
        out_dir: Output directory path. If None, derived from results/<experiment_id>.
        n_seeds: Optional override for number of seeds per cell.
        save_scores: Whether to save per-run raw score arrays.
        example_plots_per_cell: Number of example comparison plots to generate per cell.
        verbose: If True, prints execution progress and summary tables.

    Returns:
        Tuple of (results_dataframe, summary_dataframe).
    """
    t_start = time.perf_counter()

    # 1. Resolve configuration
    if isinstance(cfg_or_path, (str, Path)):
        cfg = load_experiment(cfg_or_path)
    else:
        cfg = copy.deepcopy(cfg_or_path)

    exp_id = str(cfg.get("experiment_id", "experiment"))
    if out_dir is None:
        out_path = Path("results") / exp_id
    else:
        out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    proto = Protocol.from_config(cfg)

    # 2. Extract factorial factors
    data_cfg = cfg.get("data", {})
    signals = list(data_cfg.get("signals", ["single_sinusoid", "multi_sinusoid"]))
    raw_anomalies = list(
        data_cfg.get(
            "anomalies",
            ["point_spike", "amplitude_change", "frequency_shift", "local_frequency_insertion"],
        )
    )
    # Convert 'null' string to None for synthetic generator
    anomalies = [None if a == "null" else a for a in raw_anomalies]
    difficulty = str(data_cfg.get("difficulty", "default"))

    if n_seeds is None:
        n_seeds = int(data_cfg.get("n_seeds", 30))
    seed_list = list(data_cfg.get("seed_list", list(range(n_seeds))))

    factors_cfg = cfg.get("factors", {})
    representations = list(factors_cfg.get("representation", ["temporal", "spectral", "hybrid"]))
    window_sizes = list(
        factors_cfg.get("windowing.size", [int(cfg.get("windowing", {}).get("size", 128))])
    )
    window_step = int(cfg.get("windowing", {}).get("step", 8))

    # Detectors & Controls
    primary_detector = str(cfg.get("detector", {}).get("name", "isolation_forest"))
    control_detectors = list(cfg.get("controls", []))
    detectors = [primary_detector, *control_detectors]

    # Outputs config
    outputs_cfg = cfg.get("outputs", {})
    if example_plots_per_cell is None:
        example_plots_per_cell = int(outputs_cfg.get("example_plots_per_cell", 0))

    if verbose:
        print(f"══ Running Experiment: {exp_id} ══")
        print(f"  Signals ({len(signals)}): {signals}")
        print(f"  Anomalies ({len(anomalies)}): {[a or 'null' for a in anomalies]}")
        print(f"  Seeds ({len(seed_list)}): {seed_list[:5]}{'...' if len(seed_list) > 5 else ''}")
        print(f"  Representations ({len(representations)}): {representations}")
        print(f"  Detectors ({len(detectors)}): {detectors}")
        print(f"  Output directory: {out_path}")

    # 3. Factorial execution loop (Paired Design)
    rows: list[dict[str, Any]] = []
    saved_scores: dict[str, Any] = {}
    total_series = len(signals) * len(anomalies) * len(seed_list)
    series_count = 0

    for signal_type in signals:
        for anom in anomalies:
            anom_name = anom or "null"
            plots_created = 0

            for seed in seed_list:
                series_count += 1
                # Generate synthetic series ONCE for this (signal, anomaly, seed) cell
                series = generate_series(
                    cfg,
                    signal_type=signal_type,
                    anomaly_type=anom,
                    index=seed,
                    difficulty=difficulty,
                )

                scores_by_rep: dict[str, np.ndarray] = {}
                threshold_by_rep: dict[str, float] = {}

                # Evaluate across window sizes, representations, and detectors
                for w_size in window_sizes:
                    for det_name in detectors:
                        # Baseline controls (like random_score) do not depend on representation;
                        # evaluate on first representation to avoid redundant duplicate runs
                        reps_to_run = (
                            [representations[0]]
                            if det_name == "random_score"
                            else representations
                        )

                        for rep in reps_to_run:
                            record, pt_scores, _, th = run_single_cell(
                                series=series,
                                representation=rep,
                                detector_name=det_name,
                                window_size=w_size,
                                window_step=window_step,
                                cfg=cfg,
                                proto=proto,
                            )
                            record["experiment_id"] = exp_id
                            rows.append(record)

                            if det_name == primary_detector:
                                scores_by_rep[rep] = pt_scores
                                threshold_by_rep[rep] = th

                            if save_scores:
                                key = f"{signal_type}_{anom_name}_s{seed}_{w_size}_{rep}_{det_name}"
                                saved_scores[key] = pt_scores.tolist()

                # Generate comparison plot if within plot limit for this cell
                if example_plots_per_cell > 0 and plots_created < example_plots_per_cell:
                    plot_filename = (
                        out_path / "figures" / f"{signal_type}_{anom_name}_seed_{seed}.png"
                    )
                    plot_cell_comparison(
                        series, scores_by_rep, threshold_by_rep, proto, plot_filename
                    )
                    plots_created += 1

                if verbose and series_count % 20 == 0:
                    pct = (series_count / total_series) * 100
                    print(
                        f"  Progress: {series_count}/{total_series} "
                        f"series evaluated ({pct:.1f}%)"
                    )

    # 4. Tabulate results
    df_results = pd.DataFrame(rows)

    # 5. Compute summary grouped by (signal, anomaly, representation, detector)
    valid_ap = df_results[df_results["point_average_precision"].notna()]
    if len(valid_ap) > 0:
        summary_df = (
            valid_ap.groupby(["signal", "anomaly", "representation", "detector"], as_index=False)
            .agg(
                mean_ap=("point_average_precision", "mean"),
                std_ap=("point_average_precision", "std"),
                event_hit_rate=("event_hit", "mean"),
                mean_roc_auc=("roc_auc", "mean"),
                mean_f1=("f1", "mean"),
                prevalence=("prevalence", "mean"),
                n_seeds=("point_average_precision", "count"),
            )
        )
    else:
        summary_df = pd.DataFrame(
            columns=[
                "signal",
                "anomaly",
                "representation",
                "detector",
                "mean_ap",
                "std_ap",
                "event_hit_rate",
                "mean_roc_auc",
                "mean_f1",
                "prevalence",
                "n_seeds",
            ]
        )

    # 6. Save results, summary, metadata, and scores
    results_csv_path = out_path / "results.csv"
    results_json_path = out_path / "results.json"
    summary_csv_path = out_path / "summary.csv"
    summary_json_path = out_path / "summary.json"
    metadata_json_path = out_path / "metadata.json"

    df_results.to_csv(results_csv_path, index=False)
    with open(results_json_path, "w", encoding="utf-8") as f:
        json.dump(df_results.to_dict(orient="records"), f, indent=2)

    summary_df.to_csv(summary_csv_path, index=False)
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_df.to_dict(orient="records"), f, indent=2)

    if save_scores:
        scores_path = out_path / "point_scores.json"
        with open(scores_path, "w", encoding="utf-8") as f:
            json.dump(saved_scores, f)

    t_end = time.perf_counter()
    duration = t_end - t_start

    # Build D-007 metadata record
    metadata = {
        "experiment_id": exp_id,
        "config_hash": config_hash(cfg),
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "runtime_seconds": round(duration, 3),
        "total_runs": len(rows),
        "total_series": total_series,
        "environment": {
            "platform": platform.platform(),
            "python": sys.version,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "sklearn": sklearn.__version__,
            "pandas": pd.__version__,
        },
        "config": cfg,
    }

    with open(metadata_json_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    if verbose:
        print(f"\n══ Completed {len(rows)} evaluations in {duration:.2f} s ══")
        print(f"Results written to: {out_path.resolve()}")
        if len(summary_df) > 0:
            print("\nSummary Results (Primary Metric: Point Average Precision):")
            # Format and display concise summary
            view_cols = [
                "signal",
                "anomaly",
                "representation",
                "detector",
                "mean_ap",
                "std_ap",
                "event_hit_rate",
            ]
            print(summary_df[view_cols].to_string(index=False))

    return df_results, summary_df
