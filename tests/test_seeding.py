import numpy as np

from spectrallens.seeding import derive_seed, make_rng, stable_int


def test_stable_int_is_pinned():
    # Pinned constants: if these change, every previously generated dataset changes too.
    assert stable_int("abc") == 16919744041952114874
    assert derive_seed(1, "a", "b") == 2654635943


def test_same_keys_same_stream():
    a = make_rng(7, "noise", "sig", 3).standard_normal(100)
    b = make_rng(7, "noise", "sig", 3).standard_normal(100)
    np.testing.assert_array_equal(a, b)


def test_different_keys_or_seed_differ():
    base = make_rng(7, "noise", "sig", 3).standard_normal(50)
    assert not np.array_equal(base, make_rng(7, "noise", "sig", 4).standard_normal(50))
    assert not np.array_equal(base, make_rng(8, "noise", "sig", 3).standard_normal(50))
    assert not np.array_equal(base, make_rng(7, "phase", "sig", 3).standard_normal(50))


def test_key_boundaries_matter():
    # ("ab", "c") must not collide with ("a", "bc")
    assert derive_seed(1, "ab", "c") != derive_seed(1, "a", "bc")
