# SpectralLens

**When does frequency-domain information improve time-series anomaly detection, and which anomaly
types benefit most from it?**

A controlled empirical study comparing temporal, spectral and hybrid feature representations with
a constant detector (Isolation Forest) on labeled synthetic signals, then on a small UCR subset.

Status: **M1 (foundation) in progress.** See `research/PROJECT_LOG.md`.

## Quick start

```bash
pip install -e ".[dev]"
pytest -q
python scripts/generate_synthetic.py --n 5 --out data/synthetic
python scripts/plot_examples.py        # writes results/figures/synthetic_examples.png
```

## Layout

| Path | Purpose |
|------|---------|
| `configs/` | `base.yaml` (protocol, windows, features, detector), `synthetic.yaml` (signals, anomalies), `experiments/` (specs that inherit the others) |
| `src/spectrallens/` | `seeding`, `config`, `data/protocol`, `data/synthetic` so far; windowing, features, detectors and evaluation arrive in M2-M5 |
| `research/` | Research questions, pre-registered hypotheses, decision log, literature, project log |
| `tests/` | Unit tests (determinism, protocol, injection properties) |

## Experimental protocol (short version)

- Each series: fit region (normal), calibration region (normal, thresholds only), test region (one
  anomaly). No labels or test data touch the detector, scaler or threshold.
- Random streams are keyed by identity, so comparisons across anomaly type, difficulty, duration and
  noise level are paired.
- Hypotheses are written before results (`research/HYPOTHESES.md`); design choices are in
  `research/DECISIONS.md`.

## Reproducibility

Same config and seed give byte-identical series on the same platform and numpy version; every
series stores a SHA-256 of its data and a config hash. Library versions are recorded with results.
