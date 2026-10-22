# Research questions

**Central question.** When does frequency-domain information improve time-series anomaly
detection, and which anomaly types benefit most from it?

This is a controlled study of *representations*, not a search for the newest model. The detector
is held constant (Isolation Forest) so that the representation is the experimental variable.

| ID | Question | Comparison | Main factors |
|----|----------|------------|--------------|
| RQ1 | Does adding spectral information beat temporal features alone? | Temporal vs spectral vs hybrid features -> Isolation Forest | representation |
| RQ2 | Which anomaly types benefit most? | RQ1 results split by anomaly type | point spike, amplitude change, frequency shift, local frequency insertion |
| RQ3 | When is a global FFT insufficient and local time-frequency features help? | Global FFT vs sliding-window FFT vs STFT | anomaly duration relative to window (d/W) |
| RQ4 | How do window size, noise, and anomaly duration affect results? | Sweeps over RQ1/RQ3 | window size, noise sigma, duration (trend/detrending as stress test) |

## Control rule for RQ1
Hold constant: signals, anomaly type and magnitude, split, window size and step, Isolation Forest
settings, thresholding, metrics, seeds. Change only: temporal vs spectral vs hybrid features.

## Explicitly out of scope for v1
Deep models (Transformers, diffusion, foundation models, CATCH), multivariate data, TSB-UAD/TSB-AD,
a large benchmark suite. Matrix Profile/DAMP is optional, only after the core study is complete.
