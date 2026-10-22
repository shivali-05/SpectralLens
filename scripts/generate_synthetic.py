"""Generate a labeled synthetic dataset from the configs.

Example:
    python scripts/generate_synthetic.py --n 5 --out data/synthetic
"""

from __future__ import annotations

import argparse
from pathlib import Path

from spectrallens.config import load_config
from spectrallens.data.synthetic import ANOMALY_TYPES, generate_dataset


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--configs", nargs="+", default=["base", "synthetic"])
    p.add_argument("--signals", nargs="+", default=["single_sinusoid", "multi_sinusoid"])
    p.add_argument("--anomalies", nargs="+", default=["null", *ANOMALY_TYPES],
                   help="use 'null' for series without an anomaly")
    p.add_argument("--n", type=int, default=5, help="series per (signal, anomaly) cell")
    p.add_argument("--difficulty", default="default", choices=["easy", "default", "hard"])
    p.add_argument("--noise-std", type=float, default=None)
    p.add_argument("--duration", type=int, default=None)
    p.add_argument("--start-index", type=int, default=0)
    p.add_argument("--out", default="data/synthetic")
    args = p.parse_args()

    cfg = load_config(*args.configs)
    anomalies = [None if a == "null" else a for a in args.anomalies]
    ids = generate_dataset(
        cfg, Path(args.out), signals=args.signals, anomalies=anomalies, n_per_cell=args.n,
        difficulty=args.difficulty, noise_std=args.noise_std, duration=args.duration,
        start_index=args.start_index,
    )
    print(f"Wrote {len(ids)} series to {args.out}")


if __name__ == "__main__":
    main()
