import numpy as np
import pytest

from spectrallens.data.windowing import slice_windows
from spectrallens.evaluation.scoring import point_scores_for_split, windows_to_points


def test_windows_to_points_toy_example():
    # Length 6 series
    # Window 0: [0, 4) with score 2.0
    # Window 1: [2, 6) with score 4.0
    # Points 0, 1 covered only by win 0 -> score 2.0
    # Points 2, 3 covered by win 0 and win 1 -> score (2.0 + 4.0)/2 = 3.0
    # Points 4, 5 covered only by win 1 -> score 4.0
    scores = np.array([2.0, 4.0])
    indices = np.array([[0, 4], [2, 6]])
    points = windows_to_points(scores, indices, length=6)
    np.testing.assert_allclose(points, [2.0, 2.0, 3.0, 3.0, 4.0, 4.0])


def test_uncovered_points_filled_with_default():
    # Only window [2, 5) is scored
    scores = np.array([10.0])
    indices = np.array([[2, 5]])
    points = windows_to_points(scores, indices, length=8, default=np.nan)
    assert np.isnan(points[0]) and np.isnan(points[1])
    np.testing.assert_allclose(points[2:5], [10.0, 10.0, 10.0])
    assert np.isnan(points[5]) and np.isnan(points[6]) and np.isnan(points[7])


def test_point_scores_for_split_relative_indexing():
    # A series with test split [6000, 6010)
    x = np.arange(10000, dtype=np.float64)
    # Windows inside [6000, 6010): window_size=4, step=2 ->
    # [6000, 6004), [6002, 6006), [6004, 6008), [6006, 6010)
    windows = slice_windows(x, window_size=4, step=2, start=6000, end=6010, split_name="test")
    assert windows.n_windows == 4

    scores = np.array([1.0, 3.0, 5.0, 7.0])
    pts = point_scores_for_split(scores, windows, split_range=(6000, 6010))

    assert len(pts) == 10
    # Point 0 (original 6000): win 0 -> 1.0
    assert pts[0] == 1.0
    # Point 1 (original 6001): win 0 -> 1.0
    assert pts[1] == 1.0
    # Point 2 (original 6002): win 0 and win 1 -> (1.0 + 3.0)/2 = 2.0
    assert pts[2] == 2.0
    # Point 4 (original 6004): win 1 and win 2 -> (3.0 + 5.0)/2 = 4.0
    assert pts[4] == 4.0


def test_empty_windows_returns_default_filled_array():
    scores = np.empty(0)
    indices = np.empty((0, 2), dtype=np.int64)
    pts = windows_to_points(scores, indices, length=5, default=-1.0)
    np.testing.assert_array_equal(pts, np.full(5, -1.0))


def test_scoring_validation_errors():
    with pytest.raises(ValueError, match="scores must be 1D"):
        windows_to_points(np.zeros((2, 2)), np.zeros((2, 2)), length=10)
    with pytest.raises(ValueError, match="indices must have shape"):
        windows_to_points(np.zeros(2), np.zeros(2), length=10)
    with pytest.raises(ValueError, match="Length mismatch"):
        windows_to_points(np.zeros(2), np.zeros((3, 2)), length=10)
    with pytest.raises(ValueError, match="length must be positive"):
        windows_to_points(np.zeros(2), np.zeros((2, 2)), length=0)
    with pytest.raises(ValueError, match="indices exceed bounds"):
        windows_to_points(np.array([1.0]), np.array([[0, 15]]), length=10)
    with pytest.raises(ValueError, match="start >= end"):
        windows_to_points(np.array([1.0]), np.array([[5, 3]]), length=10)
