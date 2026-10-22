# Hypotheses (pre-registered)

Written on 2026-10-12, **before any detector has been run**. Do not edit a prediction after seeing
results. Add a dated "Outcome" line beneath it instead. Predictions are expectations from signal
processing, not findings.

Reference numbers (fs = 100 Hz, base frequency 5 Hz, period = 20 samples): FFT bin width is
`df = fs / W`, so W=32 -> 3.125 Hz, W=64 -> 1.56 Hz, W=128 -> 0.78 Hz, W=256 -> 0.39 Hz.

## RQ1
- **H1 (frequency shift).** Spectral and hybrid beat temporal when the shift spans about 2 bins or
  more. At W=32 a 5 -> 8 Hz shift is about 1 bin, so spectral features should *not* help there.
  At W=128 and ratio 1.2 (1 Hz, about 1.3 bins) the advantage should be small or absent.
  *Status: untested.*
- **H2 (point spike).** Temporal is at least as good as spectral. A single sample spreads small
  energy across all bins, diluted further as W grows. *Status: untested.*
- **H3 (amplitude change).** Temporal and spectral both detect it and roughly tie, since std/range
  and spectral energy/dominant magnitude carry the same information. *Status: untested.*
- **H4 (hybrid).** Hybrid is roughly the better of the two on each anomaly type, possibly slightly
  below it from feature dilution in Isolation Forest. *Status: untested.*

## RQ3
- **H5 (locality).** For local frequency insertion, detection with global-window features degrades
  as d/W shrinks (watch d/W below about 0.25). Local STFT or sliding-window spectral comparison
  recovers it. *Status: untested.*

## Controls that must hold before trusting any result
- Random-score baseline: AP close to prevalence.
- Null series (no anomaly): scores on the test split are indistinguishable from calibration scores.
- Easy case (e.g. spike at 8 sigma, temporal features): clearly above chance, otherwise the pipeline
  is broken.
