"""End-to-end tour of MPSynth.

    python examples/quickstart.py

Synthesises a circuit for a smooth density, prints the trade-off curve, exports to
every supported target, and checks the prepared amplitudes against the input data.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from mpsynth import export, profile, synthesize
from mpsynth.exporters import EXTENSIONS, FORMATS

OUT = Path(__file__).parent / "generated"


def main() -> None:
    # ---------------------------------------------------------------- the data
    x = np.linspace(-6.0, 6.0, 1024)
    data = np.exp(-((x + 1.5) ** 2) / 1.2) + 0.7 * np.exp(-((x - 2.0) ** 2) / 2.5)
    print(f"input: {data.size} real amplitudes -> {int(np.log2(data.size))} qubits\n")

    # ------------------------------------------------------------- synthesise
    result = synthesize(data, fidelity=0.999, max_layers=8)
    print(result.report())

    exact_baseline = 2**result.n_qubits - 2
    print(
        f"\nexact amplitude encoding would need ~{exact_baseline} CNOTs; "
        f"this uses {result.circuit.cnot_count} "
        f"({exact_baseline / max(result.circuit.cnot_count, 1):.0f}x fewer)"
    )

    # ------------------------------------------------- does it prepare the data?
    produced = result.amplitudes()
    overlap = abs(np.vdot(data / np.linalg.norm(data), produced / np.linalg.norm(produced))) ** 2
    worst = np.abs(np.abs(produced) - np.abs(data)).max() / np.abs(data).max()
    print(f"verified overlap with the input: {overlap:.8f}")
    print(f"largest pointwise amplitude error: {worst:.2%} of peak\n")

    # ---------------------------------------------------------- trade-off curve
    curve = profile(data, max_layers=6)
    print(curve.table(fidelity_target=0.999))

    # ----------------------------------------------------------------- exports
    OUT.mkdir(exist_ok=True)
    print(f"\nexporting to {OUT}:")
    for fmt in sorted(FORMATS):
        path = OUT / f"prepare_{fmt}{EXTENSIONS[fmt]}"
        path.write_text(export(result.circuit, fmt))
        print(f"  {fmt:<10} -> {path.name:<24} ({path.stat().st_size:,} bytes)")

    try:
        curve.plot(str(OUT / "tradeoff.png"), title="MPSynth trade-off -- bimodal density")
        print(f"  {'plot':<10} -> tradeoff.png")
    except ImportError:
        print("  plot skipped (matplotlib not installed)")


if __name__ == "__main__":
    main()
