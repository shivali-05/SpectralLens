# Decision log

Newest last. A decision is changed by adding a new entry that supersedes the old one.

## D-001 Split layout (2026-10-12)
Each series has three contiguous regions: fit [0, 4000), calibration [4000, 6000), test
[6000, 12000). Fit and calibration contain only normal data. The single anomaly lies in the test
region at least 600 samples from its edges. Windows must lie entirely inside one split.
*Why:* no anomaly or test information can reach the detector, scaler or threshold.

## D-002 Thresholds come from the calibration split (2026-10-12)
`threshold = quantile(calibration_scores, q)`, q = 0.99. *Why:* Isolation Forest scores its own
training windows as more normal than unseen normal windows, so training-score quantiles
under-estimate the threshold. Thresholded metrics (precision, recall, F1) are secondary because the
calibration set is small for large windows.

## D-003 Primary metric and score mapping (2026-10-12)
Primary: point-level average precision after mapping window scores to points (mean of covering
windows). Point-level prevalence is `duration / test_length`, independent of window size, so AP
stays comparable across window sweeps. Always report prevalence next to AP.

## D-004 Point spikes use event metrics (2026-10-12)
A one-sample positive makes point-wise AP degenerate. Spikes are scored with event hit (top-scoring
location within +-100 samples of the anomaly, as in the UCR archive) and the rank of the spike.

## D-005 Injection rules (2026-10-12)
- Frequency shift: all components scaled, phase-continuous (instantaneous frequency integrated).
  (See D-009: the original version left a permanent phase offset after the segment; that was
  found to be a flaw and fixed.)
- Amplitude change: raised-cosine edges of 10 samples, inside the labeled range.
- Local insertion: additive, Hann-tapered, at 20 Hz (not a component of any base signal).
- Point spike: single sample, magnitude in units of the clean periodic signal's std, random sign.
*Why:* naive splices create edge artifacts (jumps) that temporal detectors catch for free.

## D-006 Seeding and paired design (2026-10-12)
Random streams are keyed by identity (SHA-256 -> `SeedSequence`), never by Python `hash()`.
Phases and the unit-variance noise draw depend only on (global seed, signal type, index). So the
same series differs across anomaly type, difficulty, duration and noise level *only* by the
intended factor, which enables paired statistics. Noise is added last and scales exactly.

## D-007 Reproducibility scope (2026-10-12)
Same config + seed -> byte-identical arrays on the same platform and numpy version (tested by
hash). Across platforms, floating-point results may differ in the last bits (sin/cos
implementations), so published numbers record library versions in `metadata.json`.

## D-008 Isolation Forest is nearly scale-invariant (2026-10-12)
Splits are drawn uniformly between a feature's min and max, so standardization changes little for
IF. Train-only scaling is still applied (leakage rule, and it matters for the distance-based local
spectral detector). Do not expect scaling to explain RQ1 differences.

## D-009 Frequency shifts blend back to the baseline (2026-10-12)
*Problem found.* Integrating the instantaneous frequency keeps phase continuous, but the extra phase
gained during the shift persists afterwards. For `single_sinusoid` at the default settings the
extra phase is exactly 6 whole cycles (5 Hz x 0.6 x 2 s), so nothing remained; for
`multi_sinusoid` (2, 5, 11 Hz) it is 2.4, 6.0 and 13.2 cycles, so the components' *relative phases*
changed and the waveform shape after the anomaly differed from training. D-005 wrongly called this
harmless for window-level features.
*Evidence.* Noise-free, window-max after the anomaly drifted by |z| of about 82 (std of pre-anomaly
window-max) versus 0.74 for the null control. At sigma = 0.1 the drift was indistinguishable from
the null control (0.37 vs 0.35), so the effect is hidden by noise at the default level but may
matter at lower noise (sigma = 0.05 in the noise sweep). An earlier measurement without a null
control wrongly suggested a small real effect at sigma = 0.1.
*Fix.* The last `ramp` (10) samples of the segment cross-fade from the shifted signal back to the
baseline, so everything outside the labeled range equals the null series exactly. Guarded by
`test_series_equals_the_null_series_outside_the_labeled_range`.
*Cost.* The cross-fade region is inside the labeled range, so it counts as anomalous; the pure
shifted part is `duration - ramp` samples.

## D-010 Feature Set v1 frozen for RQ1 representation study (2026-10-25)
Feature Set v1 is frozen with version tag `features_v1`.
- **11 Temporal features**: `mean`, `median`, `std`, `min`, `max`, `range`, `iqr`, `slope`,
  `mean_abs_diff`, `std_diff`, `acf1`.
- **10 Spectral features**: `dominant_freq`, `dominant_magnitude`, `spectral_energy`,
  `spectral_entropy`, `spectral_centroid`, `spectral_bandwidth`, `band_ratio_low`,
  `band_ratio_mid`, `band_ratio_high`, `peak_count`.
- **Hyperparameters**: $f_s = 100$ Hz, Hann window (`window_fn: hann`), DC mean removed
  (`remove_mean: true`), Nyquist fraction band edges $[0.0, 0.12, 0.30, 1.0]$ ($0-6$ Hz low,
  $6-15$ Hz mid, $15-50$ Hz high), peak prominence ratio $0.10$, normalized entropy.
- **21 Hybrid features**: exact concatenation of temporal features followed by spectral features.
*Why:* Pre-registering and freezing the representation definitions prior to the 30-seed RQ1
evaluation prevents post-hoc feature engineering bias and p-hacking. Any subsequent feature
modifications belong to v2 or the parking lot.
