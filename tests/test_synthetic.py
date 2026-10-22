import numpy as np
import pytest

from spectrallens.config import load_config
from spectrallens.data.protocol import Protocol
from spectrallens.data.synthetic import (
    ANOMALY_TYPES,
    data_hash,
    generate_dataset,
    generate_series,
    load_series,
    save_series,
)

CFG = load_config("base", "synthetic")
PROTO = Protocol.from_config(CFG)
SIGNALS = ["single_sinusoid", "multi_sinusoid", "trend_seasonal"]


def test_same_config_and_seed_give_identical_series():
    for sig in SIGNALS:
        for anom in (None, *ANOMALY_TYPES):
            a = generate_series(CFG, sig, anom, 17)
            b = generate_series(CFG, sig, anom, 17)
            assert data_hash(a.x) == data_hash(b.x)
            np.testing.assert_array_equal(a.labels, b.labels)
            assert a.metadata == b.metadata


def test_different_index_or_global_seed_changes_data():
    a = generate_series(CFG, "single_sinusoid", "frequency_shift", 1)
    b = generate_series(CFG, "single_sinusoid", "frequency_shift", 2)
    other = load_config("base", "synthetic", overrides={"seed": 999})
    c = generate_series(other, "single_sinusoid", "frequency_shift", 1)
    assert data_hash(a.x) != data_hash(b.x)
    assert data_hash(a.x) != data_hash(c.x)


def test_anomaly_range_matches_labels_and_stays_inside_test_margins():
    lo_edge = PROTO.test[0] + PROTO.anomaly_margin
    hi_edge = PROTO.test[1] - PROTO.anomaly_margin
    for sig in SIGNALS:
        for anom in ANOMALY_TYPES:
            for i in range(25):
                s = generate_series(CFG, sig, anom, i)
                a = s.metadata["anomaly"]
                assert a["end"] - a["start"] == a["duration"]
                assert s.labels.sum() == a["duration"]
                assert s.labels[a["start"]:a["end"]].all()
                assert a["start"] >= lo_edge and a["end"] <= hi_edge
                assert not s.labels[: PROTO.test[0]].any()   # no label in fit/calibration


def test_null_series_has_no_anomaly():
    s = generate_series(CFG, "single_sinusoid", None, 3)
    assert s.metadata["anomaly"] is None
    assert s.labels.sum() == 0
    assert "null" in s.series_id


def test_noise_is_paired_across_anomaly_types_and_levels():
    z_ref = None
    for anom in (None, *ANOMALY_TYPES):
        lo = generate_series(CFG, "multi_sinusoid", anom, 5, noise_std=0.1)
        hi = generate_series(CFG, "multi_sinusoid", anom, 5, noise_std=0.3)
        z = (hi.x - lo.x) / 0.2
        if z_ref is None:
            z_ref = z
        np.testing.assert_allclose(z, z_ref, atol=1e-9)
    assert abs(np.std(z_ref) - 1.0) < 0.05


def test_series_equals_the_null_series_outside_the_labeled_range():
    # Regression test: a frequency shift used to leave a permanent phase offset after the anomaly
    # for multi_sinusoid, so post-anomaly windows looked different from training windows.
    for sig in SIGNALS:
        null = generate_series(CFG, sig, None, 4, noise_std=0.0)
        for anom in ANOMALY_TYPES:
            for i in range(5):
                null = generate_series(CFG, sig, None, i, noise_std=0.0)
                s = generate_series(CFG, sig, anom, i, noise_std=0.0)
                a = s.metadata["anomaly"]
                np.testing.assert_allclose(s.x[: a["start"]], null.x[: a["start"]], atol=1e-9)
                np.testing.assert_allclose(s.x[a["end"]:], null.x[a["end"]:], atol=1e-9)


def test_point_spike_has_the_configured_magnitude():
    null = generate_series(CFG, "single_sinusoid", None, 6, noise_std=0.0)
    s = generate_series(CFG, "single_sinusoid", "point_spike", 6, noise_std=0.0)
    diff = s.x - null.x
    nz = np.flatnonzero(diff)
    assert len(nz) == 1 and nz[0] == s.metadata["anomaly"]["start"]
    expected = 5.0 * np.std(null.x)     # null.x is the clean periodic signal when noise = 0
    assert abs(abs(diff[nz[0]]) - expected) < 1e-9


def test_amplitude_change_scales_interior_and_is_continuous_at_edges():
    null = generate_series(CFG, "single_sinusoid", None, 8, noise_std=0.0)
    s = generate_series(CFG, "single_sinusoid", "amplitude_change", 8, noise_std=0.0)
    a = s.metadata["anomaly"]
    ramp = a["params"]["ramp"]
    inner = slice(a["start"] + ramp, a["end"] - ramp)
    np.testing.assert_allclose(s.x[inner], 1.5 * null.x[inner], atol=1e-9)
    np.testing.assert_allclose(s.x[a["end"]:], null.x[a["end"]:], atol=1e-12)


def test_amplitude_change_eases_in_and_out():
    # Without the raised-cosine ramp the first/last labeled sample differs from the null series
    # by (factor - 1) * |wave| (up to 0.5 here). With it the difference is at most about 0.01.
    # Checked over several series because the wave's phase at the edge varies with the seed.
    for sig in ("single_sinusoid", "multi_sinusoid"):
        for i in range(10):
            null = generate_series(CFG, sig, None, i, noise_std=0.0)
            s = generate_series(CFG, sig, "amplitude_change", i, noise_std=0.0)
            a = s.metadata["anomaly"]
            for idx in (a["start"], a["end"] - 1):
                assert abs(s.x[idx] - null.x[idx]) < 0.05


def test_frequency_shift_starts_continuously_and_returns_to_baseline():
    # Phase continuity: at the first labeled sample the shifted wave equals the baseline exactly.
    # Blend-back: at the last labeled sample it is back within about 2 percent of the baseline.
    # Checked on multi_sinusoid too, where leftover phase offset used to show up.
    for sig in ("single_sinusoid", "multi_sinusoid"):
        for i in range(10):
            null = generate_series(CFG, sig, None, i, noise_std=0.0)
            s = generate_series(CFG, sig, "frequency_shift", i, noise_std=0.0)
            a = s.metadata["anomaly"]
            assert abs(s.x[a["start"]] - null.x[a["start"]]) < 1e-9
            assert abs(s.x[a["end"] - 1] - null.x[a["end"] - 1]) < 0.05


def test_frequency_shift_is_phase_continuous_with_the_right_frequency():
    s = generate_series(CFG, "single_sinusoid", "frequency_shift", 9, noise_std=0.0)
    null = generate_series(CFG, "single_sinusoid", None, 9, noise_std=0.0)
    a = s.metadata["anomaly"]
    max_clean_step = np.max(np.abs(np.diff(null.x)))
    for edge in (a["start"], a["end"]):
        assert np.max(np.abs(np.diff(s.x[edge - 3: edge + 4]))) < 1.8 * max_clean_step
    # dominant frequency inside the segment is 5 * 1.6 = 8 Hz
    seg = s.x[a["start"]:a["end"]] * np.hanning(a["duration"])
    spec = np.abs(np.fft.rfft(seg, n=8192))
    f_peak = np.fft.rfftfreq(8192, d=1 / 100)[np.argmax(spec)]
    assert abs(f_peak - 8.0) < 0.3


def test_frequency_insertion_is_additive_local_and_at_the_configured_frequency():
    null = generate_series(CFG, "single_sinusoid", None, 10, noise_std=0.0)
    s = generate_series(CFG, "single_sinusoid", "local_frequency_insertion", 10, noise_std=0.0)
    a = s.metadata["anomaly"]
    diff = s.x - null.x
    assert np.all(diff[: a["start"]] == 0) and np.all(diff[a["end"]:] == 0)
    spec = np.abs(np.fft.rfft(diff[a["start"]:a["end"]], n=8192))
    f_peak = np.fft.rfftfreq(8192, d=1 / 100)[np.argmax(spec)]
    assert abs(f_peak - 20.0) < 0.5


def test_trend_signal_actually_drifts():
    s = generate_series(CFG, "trend_seasonal", None, 0, noise_std=0.0)
    first, last = s.x[:500].mean(), s.x[-500:].mean()
    assert last - first > 0.8      # total drift is 1.0 * amplitude


def test_difficulty_levels_change_magnitude_not_the_base_signal():
    easy = generate_series(CFG, "single_sinusoid", "amplitude_change", 2, difficulty="easy")
    hard = generate_series(CFG, "single_sinusoid", "amplitude_change", 2, difficulty="hard")
    assert easy.metadata["anomaly"]["params"]["factor"] == 2.0
    assert hard.metadata["anomaly"]["params"]["factor"] == 1.25
    assert easy.metadata["anomaly"]["duration"] == hard.metadata["anomaly"]["duration"]


def test_duration_override_and_invalid_inputs():
    s = generate_series(CFG, "single_sinusoid", "frequency_shift", 1, duration=400)
    assert s.metadata["anomaly"]["duration"] == 400
    with pytest.raises(ValueError):
        generate_series(CFG, "single_sinusoid", "point_spike", 1, duration=5)
    with pytest.raises(ValueError):
        generate_series(CFG, "single_sinusoid", "nonsense", 1)
    with pytest.raises(ValueError):
        generate_series(CFG, "nonsense", None, 1)
    with pytest.raises(ValueError):
        generate_series(CFG, "single_sinusoid", "frequency_shift", 1, difficulty="impossible")


def test_metadata_has_required_fields():
    s = generate_series(CFG, "single_sinusoid", "frequency_shift", 17)
    required = {"schema_version", "generator_version", "series_id", "index", "global_seed",
                "signal_type", "signal_components", "sampling_rate", "noise_std",
                "series_length", "split", "anomaly", "config_hash", "data_sha256"}
    assert required <= set(s.metadata)
    assert set(s.metadata["split"]) == {"fit", "calibration", "test"}
    assert {"type", "start", "end", "duration", "difficulty", "params"} <= set(
        s.metadata["anomaly"]
    )


def test_save_load_roundtrip_and_tamper_detection(tmp_path):
    s = generate_series(CFG, "multi_sinusoid", "amplitude_change", 11)
    save_series(s, tmp_path)
    loaded = load_series(tmp_path, s.series_id)
    np.testing.assert_array_equal(loaded.x, s.x)
    np.testing.assert_array_equal(loaded.labels, s.labels)
    assert loaded.metadata == s.metadata
    # corrupt the stored data
    with np.load(tmp_path / f"{s.series_id}.npz") as arrays:
        x, labels = arrays["x"].copy(), arrays["labels"]
    x[0] += 1.0
    np.savez_compressed(tmp_path / f"{s.series_id}.npz", x=x, labels=labels)
    with pytest.raises(ValueError, match="hash mismatch"):
        load_series(tmp_path, s.series_id)


def test_generate_dataset_writes_manifest_and_is_reproducible(tmp_path):
    kw = dict(signals=["single_sinusoid"], anomalies=[None, "frequency_shift"], n_per_cell=2)
    ids1 = generate_dataset(CFG, tmp_path / "a", **kw)
    ids2 = generate_dataset(CFG, tmp_path / "b", **kw)
    assert ids1 == ids2 and len(ids1) == 4
    assert (tmp_path / "a" / "manifest.json").exists()
    for sid in ids1:
        xa = load_series(tmp_path / "a", sid).x
        xb = load_series(tmp_path / "b", sid).x
        np.testing.assert_array_equal(xa, xb)
