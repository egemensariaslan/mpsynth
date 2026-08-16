from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(20240816)


def random_unitary(dim: int, rng: np.random.Generator) -> np.ndarray:
    """Haar-random unitary via QR with the phase ambiguity fixed."""
    z = (rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))) / np.sqrt(2)
    q, r = np.linalg.qr(z)
    return q * (np.diag(r) / np.abs(np.diag(r)))


def random_state(dim: int, rng: np.random.Generator) -> np.ndarray:
    v = rng.normal(size=dim) + 1j * rng.normal(size=dim)
    return v / np.linalg.norm(v)


def assert_equal_up_to_phase(a: np.ndarray, b: np.ndarray, atol: float = 1e-9) -> None:
    idx = np.unravel_index(np.argmax(np.abs(b)), b.shape)
    phase = a[idx] / b[idx]
    assert abs(abs(phase) - 1.0) < 1e-8, "not equal up to a *unit* phase"
    np.testing.assert_allclose(a, phase * b, atol=atol)
