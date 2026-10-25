"""CLI entry point to execute SpectralLens experiments.

Example usage:
    # Run preview with 5 seeds
    python scripts/run_experiment.py --config configs/experiments/rq1_representation_v1.yaml \
        --n-seeds 5

    # Run full experiment as configured
    python scripts/run_experiment.py --config configs/experiments/rq1_representation_v1.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

from spectrallens.experiments.runner import run_experiment


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument(
        "--config",
        "-c",
        required=True,
        help="Path to experiment YAML config (e.g. configs/experiments/rq1_representation_v1.yaml)",
    )
    p.add_argument(
        "--out",
        "-o",
        default=None,
        help="Custom output directory (default: results/<experiment_id>)",
    )
    p.add_argument(
        "--n-seeds",
        "-n",
        type=int,
        default=None,
        help="Override number of seeds to run per cell (useful for fast smoke tests)",
    )
    p.add_argument(
        "--plots",
        type=int,
        default=None,
        help="Number of example diagnostic plots to generate per cell",
    )
    p.add_argument(
        "--no-scores",
        action="store_true",
        help="Disable saving raw point score arrays (saves disk space)",
    )

    args = p.parse_args()

    out_dir = Path(args.out) if args.out else None
    run_experiment(
        cfg_or_path=args.config,
        out_dir=out_dir,
        n_seeds=args.n_seeds,
        save_scores=not args.no_scores,
        example_plots_per_cell=args.plots,
        verbose=True,
    )


if __name__ == "__main__":
    main()
