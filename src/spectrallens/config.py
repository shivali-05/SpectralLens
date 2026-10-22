"""YAML configuration loading, inheritance and hashing."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge ``override`` into a copy of ``base``. Lists and scalars are replaced."""
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_yaml(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return data or {}


def _resolve(name: str | Path, config_dir: Path) -> Path:
    candidate = Path(name)
    if candidate.suffix in {".yaml", ".yml"} and candidate.exists():
        return candidate
    path = config_dir / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Config '{name}' not found (looked for {path})")
    return path


def load_config(
    *names: str | Path,
    config_dir: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> dict:
    """Load and merge configs in order, e.g. ``load_config("base", "synthetic")``."""
    cdir = Path(config_dir) if config_dir else DEFAULT_CONFIG_DIR
    cfg: dict = {}
    for name in names:
        cfg = deep_merge(cfg, load_yaml(_resolve(name, cdir)))
    if overrides:
        cfg = deep_merge(cfg, overrides)
    return cfg


def load_experiment(path: str | Path, config_dir: str | Path | None = None) -> dict:
    """Load an experiment spec and resolve its ``inherit`` list underneath it."""
    spec = load_yaml(path)
    inherit = spec.pop("inherit", [])
    base = load_config(*inherit, config_dir=config_dir)
    return deep_merge(base, spec)


def config_hash(cfg: dict, length: int = 8) -> str:
    """Stable short hash of a config (key order does not matter)."""
    blob = json.dumps(cfg, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:length]
