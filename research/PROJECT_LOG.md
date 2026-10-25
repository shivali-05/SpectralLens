# Project log

## 2026-10-12: M1 started
- Repo skeleton, packaging, CI config, seeding, config loader with inheritance and hashing.
- Protocol (fit / calibration / test) with validation.
- Synthetic generator and injector for four anomaly types plus null series; save/load with a data
  hash; dataset generation script; example plot script.
- 30 tests pass. Mutation check: the phase-continuity test separates a proper frequency shift
  (edge step ratio 1.61 vs limit 1.8) from a naive splice (4.64).
- Decisions D-001..D-008 and hypotheses H1..H5 written before any detector exists.

## 2026-10-12 (later): first real-environment run and a generator fix
- Real pytest on Python 3.13 / macOS: 30 passed. Ruff reported 4 style issues (2x B905 `zip`
  without `strict=`, 1x E501 line too long, 1x I001 import order); fixed.
- Found and fixed a flaw in frequency-shift injection (permanent phase offset after the anomaly
  for multi-component signals). See D-009. Added a regression test over every signal and anomaly
  type. Replaced a weak amplitude edge test (it passed with the ramp removed for 3 of 5 series)
  with a stronger one; mutation-checked both. Suite is now 32 tests.
- Lesson: a measurement without a control can look like evidence when it is not.

### Next
- Confirm CI is green on GitHub (ruff + pytest on 3.10 and 3.12).
- M2: windowing, point mapping, metrics, runner, following docs/WORKING_AGREEMENT.md.

## 2026-10-24: Features & Representation Pipeline (Step 2.3 & Step 2.4)
- Implemented Temporal Feature Set v1 (11 features): mean, median, std, min, max, range, iqr, slope, mean_abs_diff, std_diff, acf1.
- Implemented Leakage-Free Isolation Forest pipeline with train-only StandardScaler.
- Implemented Spectral Feature Set v1 (10 features): dominant_freq, dominant_magnitude, spectral_energy, spectral_entropy, spectral_centroid, spectral_bandwidth, band_ratio_low, band_ratio_mid, band_ratio_high, peak_count.
- Leakage control: Hann window reduces sidelobe leakage by >100x vs boxcar on off-grid frequencies.
- Implemented Hybrid representation (21 features = 11 temporal + 10 spectral) and unified `extract_features` dispatcher.
- Full test suite: 97 unit and integration tests passing with zero warnings and zero ruff lint errors.

## 2026-10-25: Experiment Runner & Orchestration (Step 2.5)
- Implemented full factorial experiment runner (`src/spectrallens/experiments/runner.py`) with paired design (identical generated series evaluated across representations).
- Added detector factory supporting Isolation Forest, RandomDetector (control), and RobustZDetector (control).
- Comprehensive tabulation: per-cell metrics (results.csv, results.json), grouped summary aggregations (summary.csv, summary.json), and D-007 provenance metadata (metadata.json).
- Implemented visual comparative plotting utility generating multi-panel diagnostic figures.
- Created CLI entry point (`scripts/run_experiment.py`) with argument parsing for seed sweeps and diagnostic plots.
- Test suite expanded to 106 tests, all passing in <8s with zero warnings and zero ruff lint errors.

## 2026-10-25: Milestone 3 Started — Feature Set v1 Frozen & Active (Step 3.1)
- Resolved DECISIONS.md D-010: Feature Set v1 formally frozen with tag `features_v1`.
- Declared `FEATURE_SET_VERSION = "v1"` in `src/spectrallens/features/__init__.py`.
- Activated `configs/experiments/rq1_representation_v1.yaml` (`status: active`).
- Added strict immutability unit test `test_feature_set_v1_frozen_and_immutable` in `tests/test_spectral.py`.
- 106 tests passing with zero warnings and zero ruff lint errors.



