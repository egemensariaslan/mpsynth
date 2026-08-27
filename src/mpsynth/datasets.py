"""Built-in sample vectors, for demos, benchmarks and the CLI.

These are chosen to span the compressibility spectrum: smooth analytic functions
compress to tiny bond dimension, structured combinatorial states sit at bond 2
exactly, and i.i.d. random noise is incompressible -- the worst case.
"""

from __future__ import annotations

import numpy as np

__all__ = ["GENERATORS", "generate"]


def _gaussian(n: int, rng: np.random.Generator) -> np.ndarray:
    x = np.linspace(-4.0, 4.0, 2**n)
    return np.exp(-(x**2) / 2.0)


def _lognormal(n: int, rng: np.random.Generator) -> np.ndarray:
    x = np.linspace(0.02, 5.0, 2**n)
    return np.exp(-((np.log(x)) ** 2) / 0.5) / x


def _damped_oscillation(n: int, rng: np.random.Generator) -> np.ndarray:
    x = np.linspace(0.0, 8.0, 2**n)
    return np.exp(-0.4 * x) * np.cos(4.0 * x)


def _lorentzian(n: int, rng: np.random.Generator) -> np.ndarray:
    x = np.linspace(-6.0, 6.0, 2**n)
    return 1.0 / (1.0 + x**2)


def _heavy_tail(n: int, rng: np.random.Generator) -> np.ndarray:
    x = np.linspace(1.0, 64.0, 2**n)
    return x**-1.5


def _w_state(n: int, rng: np.random.Generator) -> np.ndarray:
    v = np.zeros(2**n)
    for k in range(n):
        v[1 << k] = 1.0
    return v


def _ghz(n: int, rng: np.random.Generator) -> np.ndarray:
    v = np.zeros(2**n)
    v[0] = v[-1] = 1.0
    return v


def _product(n: int, rng: np.random.Generator) -> np.ndarray:
    v: np.ndarray = np.ones(1)
    for _ in range(n):
        v = np.kron(v, rng.normal(size=2))
    return v


def _random(n: int, rng: np.random.Generator) -> np.ndarray:
    return rng.normal(size=2**n)


def _random_complex(n: int, rng: np.random.Generator) -> np.ndarray:
    return rng.normal(size=2**n) + 1j * rng.normal(size=2**n)


def _bimodal(n: int, rng: np.random.Generator) -> np.ndarray:
    x = np.linspace(-6.0, 6.0, 2**n)
    return np.exp(-((x + 2.0) ** 2) / 0.8) + 0.6 * np.exp(-((x - 2.5) ** 2) / 1.5)


GENERATORS = {
    "gaussian": _gaussian,
    "lognormal": _lognormal,
    "bimodal": _bimodal,
    "damped": _damped_oscillation,
    "lorentzian": _lorentzian,
    "heavytail": _heavy_tail,
    "w": _w_state,
    "ghz": _ghz,
    "product": _product,
    "random": _random,
    "randomcomplex": _random_complex,
}


def generate(name: str, n_qubits: int, seed: int = 0) -> np.ndarray:
    """Build a sample vector of length ``2**n_qubits``."""
    try:
        fn = GENERATORS[name]
    except KeyError:
        raise ValueError(f"unknown dataset {name!r}; choose from {sorted(GENERATORS)}") from None
    if not 1 <= n_qubits <= 24:
        raise ValueError("n_qubits must be between 1 and 24")
    return np.asarray(fn(n_qubits, np.random.default_rng(seed)))
