"""Deterministic, process-independent seeding.

Python's built-in ``hash()`` is salted per process for strings, so it must never be used to
derive seeds. Everything here goes through SHA-256 so the same keys give the same stream on
every run, machine and process.
"""

from __future__ import annotations

import hashlib

import numpy as np


def stable_int(key: object) -> int:
    """Map any key to a stable 64-bit integer via SHA-256 of its string form."""
    digest = hashlib.sha256(str(key).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little")


def seed_sequence(global_seed: int, *keys: object) -> np.random.SeedSequence:
    """SeedSequence from a global seed plus any number of identifying keys."""
    return np.random.SeedSequence([int(global_seed)] + [stable_int(k) for k in keys])


def make_rng(global_seed: int, *keys: object) -> np.random.Generator:
    """Independent random generator for the stream identified by ``keys``."""
    return np.random.default_rng(seed_sequence(global_seed, *keys))


def derive_seed(global_seed: int, *keys: object) -> int:
    """32-bit integer seed for APIs that need one (e.g. scikit-learn ``random_state``)."""
    return int(seed_sequence(global_seed, *keys).generate_state(1, dtype=np.uint32)[0])
