"""Train / calibration / test layout shared by every experiment."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

Range = tuple[int, int]


@dataclass(frozen=True)
class Protocol:
    series_length: int
    fit: Range
    calibration: Range
    test: Range
    anomaly_margin: int
    n_anomalies: int = 1

    def __post_init__(self) -> None:
        f0, f1 = self.fit
        c0, c1 = self.calibration
        t0, t1 = self.test
        if not (0 <= f0 < f1 <= c0 < c1 <= t0 < t1 <= self.series_length):
            raise ValueError(
                "Splits must satisfy 0 <= fit < calibration < test <= series_length "
                f"(got fit={self.fit}, calibration={self.calibration}, test={self.test}, "
                f"series_length={self.series_length})"
            )
        if self.n_anomalies != 1:
            raise NotImplementedError("v1 supports exactly one anomaly per series")
        if t1 - t0 - 2 * self.anomaly_margin < 1:
            raise ValueError("anomaly_margin leaves no room inside the test split")

    @classmethod
    def from_config(cls, cfg: dict) -> Protocol:
        p = cfg["protocol"]
        return cls(
            series_length=int(p["series_length"]),
            fit=tuple(p["fit"]),
            calibration=tuple(p["calibration"]),
            test=tuple(p["test"]),
            anomaly_margin=int(p["anomaly_margin"]),
            n_anomalies=int(p.get("n_anomalies", 1)),
        )

    def start_bounds(self, duration: int) -> Range:
        """Inclusive (lowest, highest) legal anomaly start index for a given duration."""
        lo = self.test[0] + self.anomaly_margin
        hi = self.test[1] - self.anomaly_margin - duration
        if hi < lo:
            raise ValueError(f"duration {duration} does not fit inside the test split margins")
        return lo, hi

    def splits(self) -> dict[str, Range]:
        return {"fit": self.fit, "calibration": self.calibration, "test": self.test}

    def masks(self) -> dict[str, np.ndarray]:
        out = {}
        for name, (a, b) in self.splits().items():
            m = np.zeros(self.series_length, dtype=bool)
            m[a:b] = True
            out[name] = m
        return out
