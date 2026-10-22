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
