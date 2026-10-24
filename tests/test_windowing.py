import numpy as np
import pytest

from spectrallens.config import load_config
from spectrallens.data.protocol import Protocol
from spectrallens.data.windowing import Windows, extract_split_windows, slice_windows

CFG = load_config("base", "synthetic")
PROTO = Protocol.from_config(CFG)


def test_slice_windows_toy_example():
    x = np.arange(10, dtype=np.float64)
    # window_size=4, step=2 -> starts: 0, 2, 4, 6. Last window ends at 6 + 4 = 10.
    w = slice_windows(x, window_size=4, step=2)
    assert w.n_windows == 4
    assert w.window_size == 4
    np.testing.assert_array_equal(w.indices, [[0, 4], [2, 6], [4, 8], [6, 10]])
    np.testing.assert_array_equal(
        w.data,
        [
            [0, 1, 2, 3],
            [2, 3, 4, 5],
            [4, 5, 6, 7],
            [6, 7, 8, 9],
        ],
    )


def test_slice_windows_respects_start_and_end_subslice():
    x = np.arange(20, dtype=np.float64)
    # Slice strictly inside [5, 15) with window_size=4, step=3
    # starts inside 5..15: 5, 8, 11 (11+4=15 <= 15).
    w = slice_windows(x, window_size=4, step=3, start=5, end=15, split_name="custom")
    assert w.split_name == "custom"
    assert w.n_windows == 3
    np.testing.assert_array_equal(w.indices, [[5, 9], [8, 12], [11, 15]])
    for start, end in w.indices:
        assert start >= 5 and end <= 15


def test_extract_split_windows_contains_windows_strictly_within_splits():
    x = np.random.default_rng(42).standard_normal(PROTO.series_length)
    split_windows = extract_split_windows(x, PROTO, window_size=128, step=8)

    assert set(split_windows.keys()) == {"fit", "calibration", "test"}

    for name, span in PROTO.splits().items():
        w = split_windows[name]
        start_bound, end_bound = span
        assert w.split_name == name
        assert w.n_windows > 0
        assert w.window_size == 128

        # Verify EVERY window is strictly inside its split
        assert np.all(w.indices[:, 0] >= start_bound)
        assert np.all(w.indices[:, 1] <= end_bound)
        assert np.all(w.indices[:, 1] - w.indices[:, 0] == 128)

        # Verify that sliced data matches x exactly
        for i in range(min(5, w.n_windows)):
            s, e = w.indices[i]
            np.testing.assert_array_equal(w.data[i], x[s:e])


def test_no_windows_straddle_fit_cal_or_cal_test_boundaries():
    x = np.zeros(PROTO.series_length)
    split_windows = extract_split_windows(x, PROTO, window_size=128, step=8)

    fit_max_end = split_windows["fit"].indices[:, 1].max()
    cal_min_start = split_windows["calibration"].indices[:, 0].min()
    assert fit_max_end <= PROTO.fit[1]
    assert cal_min_start >= PROTO.calibration[0]
    assert fit_max_end <= cal_min_start

    cal_max_end = split_windows["calibration"].indices[:, 1].max()
    test_min_start = split_windows["test"].indices[:, 0].min()
    assert cal_max_end <= PROTO.calibration[1]
    assert test_min_start >= PROTO.test[0]
    assert cal_max_end <= test_min_start


def test_window_size_larger_than_span_returns_empty_windows():
    x = np.arange(10, dtype=np.float64)
    w = slice_windows(x, window_size=15, step=2)
    assert w.n_windows == 0
    assert w.data.shape == (0, 15)
    assert w.indices.shape == (0, 2)


def test_invalid_parameters_raise_errors():
    x = np.arange(10, dtype=np.float64)
    with pytest.raises(ValueError, match="window_size must be positive"):
        slice_windows(x, window_size=0, step=2)
    with pytest.raises(ValueError, match="step must be positive"):
        slice_windows(x, window_size=4, step=0)
    with pytest.raises(ValueError, match="Input series x must be 1D"):
        slice_windows(x.reshape(5, 2), window_size=2, step=1)
    with pytest.raises(ValueError, match="Invalid range"):
        slice_windows(x, window_size=4, step=2, start=8, end=5)

    # Windows dataclass validation
    with pytest.raises(ValueError, match="Windows data must be 2D"):
        Windows(data=np.zeros(5), indices=np.zeros((5, 2)))
    with pytest.raises(ValueError, match="Mismatch between number of windows"):
        Windows(data=np.zeros((5, 4)), indices=np.zeros((4, 2)))
