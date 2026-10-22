"""Quick visual sanity check: one example of each anomaly type around its labeled range."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from spectrallens.config import load_config  # noqa: E402
from spectrallens.data.synthetic import ANOMALY_TYPES, generate_series  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--signal", default="single_sinusoid")
    p.add_argument("--index", type=int, default=0)
    p.add_argument("--pad", type=int, default=250)
    p.add_argument("--out", default="results/figures/synthetic_examples.png")
    args = p.parse_args()

    cfg = load_config("base", "synthetic")
    fs = cfg["sampling"]["fs"]
    kinds = [None, *ANOMALY_TYPES]
    fig, axes = plt.subplots(len(kinds), 1, figsize=(11, 2.2 * len(kinds)))
    ref = generate_series(cfg, args.signal, "frequency_shift", args.index).metadata["anomaly"]
    for ax, kind in zip(axes, kinds, strict=True):
        s = generate_series(cfg, args.signal, kind, args.index)
        a = s.metadata["anomaly"] or ref
        lo, hi = a["start"] - args.pad, a["end"] + args.pad
        t = range(lo, hi)
        ax.plot([i / fs for i in t], s.x[lo:hi], lw=0.9)
        ax.axvspan(a["start"] / fs, a["end"] / fs, alpha=0.2, color="red" if kind else "gray")
        ax.set_ylabel(kind or "null", fontsize=9)
        ax.margins(x=0)
    axes[-1].set_xlabel("time (s)")
    axes[0].set_title(
        f"{args.signal}, index {args.index}, default difficulty "
        "(shaded = labeled range; null shows the frequency_shift range for reference)",
        fontsize=9,
    )
    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
