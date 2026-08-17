"""Regenerate the sample vectors committed under examples/.

    python examples/gen_examples.py

They are plain CSV so they stay readable and diffable in git, and they are the same
data the README's numbers were measured on.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

HERE = Path(__file__).parent


def lognormal(n_qubits: int) -> np.ndarray:
    """Risk-neutral asset price density -- the payload of a quantum option pricer."""
    s = np.linspace(0.02, 4.0, 2**n_qubits)
    mu, sigma = 0.0, 0.6
    return np.exp(-((np.log(s) - mu) ** 2) / (2 * sigma**2)) / (s * sigma)


def gaussian(n_qubits: int) -> np.ndarray:
    x = np.linspace(-4.0, 4.0, 2**n_qubits)
    return np.exp(-(x**2) / 2.0)


def chirp(n_qubits: int) -> np.ndarray:
    """A damped swept-frequency signal -- the deliberately *harder* sample.

    Oscillation across the register creates entanglement that a density does not, so
    this one needs several layers where the smooth samples need one.  It is here so the
    examples span the real range of difficulty, not just the flattering end.
    """
    t = np.linspace(0.0, 1.0, 2**n_qubits)
    return np.exp(-2.0 * t) * np.sin(2 * np.pi * (2.0 + 6.0 * t) * t)


SAMPLES = {
    "lognormal_4096.csv": lognormal(12),
    "gaussian_1024.csv": gaussian(10),
    "chirp_1024.csv": chirp(10),
}


def main() -> None:
    for name, values in SAMPLES.items():
        path = HERE / name
        path.write_text("\n".join(f"{v:.12g}" for v in values) + "\n")
        print(f"wrote {path.name:<22} {values.size:>5} values  {path.stat().st_size:>7,} bytes")


if __name__ == "__main__":
    main()
