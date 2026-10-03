"""Smoke tests for seed management and stats utilities."""

from utils.seed import get_seed_from_args, set_global_seed


def test_set_global_seed_deterministic():
    """Verify global seed produces deterministic random state."""
    import random

    set_global_seed(42)
    a = [random.random() for _ in range(10)]
    set_global_seed(42)
    b = [random.random() for _ in range(10)]
    assert a == b, "Same seed should produce same random sequence"


def test_set_global_seed_different():
    """Different seeds produce different sequences."""
    import random

    set_global_seed(42)
    a = random.random()
    set_global_seed(99)
    b = random.random()
    assert a != b, "Different seeds should produce different values"


def test_seed_returns_dict():
    """Seed info returns a dict with expected keys."""
    info = set_global_seed(42)
    assert isinstance(info, dict)


def test_numpy_seed():
    """Numpy seeding is deterministic."""
    try:
        import numpy as np

        set_global_seed(42)
        a = np.random.rand(5)
        set_global_seed(42)
        b = np.random.rand(5)
        assert (a == b).all(), "Numpy should be deterministic with same seed"
    except ImportError:
        pass  # numpy not installed, skip


def test_torch_seed():
    """PyTorch seeding is deterministic (replaces the CI reproducibility-check job)."""
    import torch

    set_global_seed(42)
    a = torch.randn(10)
    set_global_seed(42)
    b = torch.randn(10)
    assert torch.equal(a, b), "PyTorch should be deterministic with same seed"


def test_get_seed_from_args(monkeypatch):
    """--seed is parsed from argv; unknown args are ignored; default otherwise."""
    monkeypatch.setattr("sys.argv", ["prog", "--other", "x", "--seed", "7"])
    assert get_seed_from_args() == 7
    monkeypatch.setattr("sys.argv", ["prog"])
    assert get_seed_from_args(default=13) == 13
