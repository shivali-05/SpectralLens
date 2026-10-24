"""Isolation Forest anomaly detector with train-only scaling.

Design rules (DECISIONS.md D-008):
  - StandardScaler is fit ONLY on fit-split features; calibration and test features
    are only transformed, never used to re-fit the scaler.
  - IsolationForest is fit ONLY on fit-split features.
  - sklearn outputs decision_function scores where more negative = more anomalous.
    We negate them so that HIGHER score = more anomalous (matches evaluation convention).
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import IsolationForest as _SklearnIF
from sklearn.preprocessing import StandardScaler

from spectrallens.seeding import derive_seed


class IsolationForestDetector:
    """Leakage-free Isolation Forest with train-only StandardScaler.

    Usage::

        detector = IsolationForestDetector(cfg)
        detector.fit(fit_features)           # fit scaler + forest on train windows
        cal_scores = detector.score(cal_features)   # calibration scores for threshold
        test_scores = detector.score(test_features) # final evaluation

    Parameters
    ----------
    cfg : dict
        Full config dict (merged base + synthetic). Reads:
            cfg["detector"]["params"]   — IsolationForest kwargs
            cfg["seed"]                 — global reproducibility seed
    """

    def __init__(self, cfg: dict) -> None:
        det_params = cfg.get("detector", {}).get("params", {})
        global_seed = cfg.get("seed", 42)
        random_state = derive_seed(global_seed, "isolation_forest")

        self._scaler = StandardScaler()
        self._forest = _SklearnIF(
            n_estimators=int(det_params.get("n_estimators", 200)),
            max_samples=det_params.get("max_samples", 256),
            max_features=float(det_params.get("max_features", 1.0)),
            bootstrap=bool(det_params.get("bootstrap", False)),
            contamination=det_params.get("contamination", "auto"),
            random_state=random_state,
        )
        self._fitted = False

    def fit(self, X: np.ndarray) -> IsolationForestDetector:
        """Fit the scaler and Isolation Forest on normal training features.

        Args:
            X: 2D feature matrix shape (n_windows, n_features) from the fit split ONLY.

        Returns:
            self
        """
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError(f"X must be 2D (n_windows, n_features), got shape {X.shape}")
        if len(X) == 0:
            raise ValueError("Cannot fit on empty feature matrix")

        X_scaled = self._scaler.fit_transform(X)
        self._forest.fit(X_scaled)
        self._fitted = True
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        """Compute anomaly scores for feature windows.

        Uses the scaler fit from training — never re-fits. Score convention:
        HIGHER score = MORE anomalous (negated sklearn decision_function output).

        Args:
            X: 2D feature matrix shape (n_windows, n_features).

        Returns:
            1D array of anomaly scores, shape (n_windows,).
        """
        if not self._fitted:
            raise RuntimeError("IsolationForestDetector must be fit before scoring")
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError(f"X must be 2D (n_windows, n_features), got shape {X.shape}")
        X_scaled = self._scaler.transform(X)
        # sklearn: decision_function → more negative = more anomalous
        # We negate so higher = more anomalous (matches PR-AUC convention)
        return -self._forest.decision_function(X_scaled)
