import numpy as np
import pytest

from spectrallens.evaluation.thresholding import fit_threshold_from_calibration


def test_fit_threshold_from_calibration():
    scores = np.linspace(0.0, 1.0, 101)  # 0.00, 0.01, ..., 1.00
    th = fit_threshold_from_calibration(scores, quantile=0.99)
    assert abs(th - 0.99) < 1e-4


def test_fit_threshold_handles_nans():
    scores = np.array([np.nan, 0.1, 0.2, 0.5, 0.9, np.nan])
    th = fit_threshold_from_calibration(scores, quantile=0.50)
    assert abs(th - 0.35) < 1e-4


def test_fit_threshold_validation_errors():
    with pytest.raises(ValueError, match="calibration_scores must be 1D"):
        fit_threshold_from_calibration(np.zeros((2, 2)))
    with pytest.raises(ValueError, match="cannot be empty"):
        fit_threshold_from_calibration(np.array([]))
    with pytest.raises(ValueError, match="quantile must be in"):
        fit_threshold_from_calibration(np.array([1.0, 2.0]), quantile=1.5)
    with pytest.raises(ValueError, match="All calibration_scores are NaN"):
        fit_threshold_from_calibration(np.array([np.nan, np.nan]))
