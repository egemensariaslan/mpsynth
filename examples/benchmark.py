"""Benchmark MPSynth across the built-in datasets and register sizes.

    python examples/benchmark.py

Prints two tables: cost at a fixed fidelity target across input types, and the
scaling of CNOT count with register size for a fixed smooth input.
"""

from __future__ import annotations

import time

from mpsynth import synthesize
from mpsynth.datasets import generate

TARGET = 0.99
DESCRIPTIONS = {
    "gaussian": "smooth analytic",
    "lognormal": "skewed density",
    "bimodal": "two-peak density",
    "damped": "damped oscillation",
    "lorentzian": "heavy shoulders",
    "heavytail": "power law",
    "w": "W state",
    "ghz": "GHZ state",
    "product": "product state",
    "random": "i.i.d. noise (real)",
    "randomcomplex": "i.i.d. noise (complex)",
}


def cost_by_input(n: int = 10, max_layers: int = 12) -> None:
    exact = 2**n - 2
    print(f"\n{n} qubits, fidelity target {TARGET}, exact-encoding baseline ~{exact} CNOTs\n")
    header = f"{'input':<22} {'kind':<20} {'chi':>4} {'L':>3} {'CNOT':>6} {'2q-depth':>9} {'fidelity':>10} {'speedup':>8}"
    print(header)
    print("-" * len(header))
    for name, description in DESCRIPTIONS.items():
        vector = generate(name, n, seed=1)
        result = synthesize(vector, fidelity=TARGET, max_layers=max_layers)
        m = result.circuit.metrics()
        speedup = exact / max(m["cnot"], 1)
        flag = "" if result.reached_target else "  (target not reached)"
        print(
            f"{name:<22} {description:<20} {result.input_bond_dimension:>4} "
            f"{result.layers:>3} {m['cnot']:>6} {m['two_qubit_depth']:>9} "
            f"{result.fidelity:>10.6f} {speedup:>7.0f}x{flag}"
        )


def scaling(max_layers: int = 4) -> None:
    print(f"\nscaling on a smooth input (gaussian), {max_layers} layers\n")
    header = f"{'qubits':>6} {'amplitudes':>12} {'CNOT':>6} {'2q-depth':>9} {'fidelity':>10} {'exact CNOTs':>13} {'time':>7}"
    print(header)
    print("-" * len(header))
    for n in [6, 8, 10, 12, 14, 16, 18]:
        vector = generate("gaussian", n)
        start = time.perf_counter()
        result = synthesize(vector, fidelity=1.0, max_layers=max_layers)
        elapsed = time.perf_counter() - start
        m = result.circuit.metrics()
        print(
            f"{n:>6} {2**n:>12} {m['cnot']:>6} {m['two_qubit_depth']:>9} "
            f"{result.fidelity:>10.6f} {2**n - 2:>13} {elapsed:>6.2f}s"
        )


if __name__ == "__main__":
    cost_by_input()
    scaling()
