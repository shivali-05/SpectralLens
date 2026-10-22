import pytest

from spectrallens.config import load_config
from spectrallens.data.protocol import Protocol


def _proto():
    return Protocol.from_config(load_config("base", "synthetic"))


def test_masks_are_disjoint_and_cover_expected_ranges():
    p = _proto()
    m = p.masks()
    assert not (m["fit"] & m["calibration"]).any()
    assert not (m["calibration"] & m["test"]).any()
    assert m["fit"].sum() == 4000 and m["calibration"].sum() == 2000 and m["test"].sum() == 6000


def test_start_bounds_respect_margin():
    p = _proto()
    lo, hi = p.start_bounds(200)
    assert lo == 6000 + 600
    assert hi == 12000 - 600 - 200


def test_overlapping_or_unordered_splits_rejected():
    with pytest.raises(ValueError):
        Protocol(series_length=100, fit=(0, 60), calibration=(50, 80), test=(80, 100),
                 anomaly_margin=1)
    with pytest.raises(ValueError):
        Protocol(series_length=100, fit=(0, 40), calibration=(40, 60), test=(60, 120),
                 anomaly_margin=1)


def test_margin_too_large_rejected():
    with pytest.raises(ValueError):
        Protocol(series_length=100, fit=(0, 40), calibration=(40, 60), test=(60, 100),
                 anomaly_margin=25)


def test_duration_too_long_rejected():
    with pytest.raises(ValueError):
        _proto().start_bounds(10_000)
