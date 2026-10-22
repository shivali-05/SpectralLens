import pytest

from spectrallens.config import (
    DEFAULT_CONFIG_DIR,
    config_hash,
    deep_merge,
    load_config,
    load_experiment,
)


def test_deep_merge_overrides_nested_and_replaces_lists():
    base = {"a": {"b": 1, "c": [1, 2]}, "d": 5}
    over = {"a": {"c": [9]}, "e": 1}
    merged = deep_merge(base, over)
    assert merged == {"a": {"b": 1, "c": [9]}, "d": 5, "e": 1}
    assert base == {"a": {"b": 1, "c": [1, 2]}, "d": 5}   # inputs untouched


def test_config_hash_ignores_key_order_but_not_values():
    assert config_hash({"a": 1, "b": 2}) == config_hash({"b": 2, "a": 1})
    assert config_hash({"a": 1}) != config_hash({"a": 2})
    assert len(config_hash({"a": 1}, length=12)) == 12


def test_load_config_merges_files():
    cfg = load_config("base", "synthetic")
    assert cfg["sampling"]["fs"] == 100
    assert "single_sinusoid" in cfg["signals"]
    assert cfg["protocol"]["series_length"] == 12000


def test_missing_config_raises():
    with pytest.raises(FileNotFoundError):
        load_config("does_not_exist")


def test_experiment_inherit_resolves():
    cfg = load_experiment(DEFAULT_CONFIG_DIR / "experiments" / "rq1_representation_v1.yaml")
    assert cfg["experiment_id"] == "rq1_representation_v1"
    assert cfg["detector"]["name"] == "isolation_forest"      # from base
    assert "frequency_shift" in cfg["anomalies"]              # from synthetic
    assert "inherit" not in cfg
