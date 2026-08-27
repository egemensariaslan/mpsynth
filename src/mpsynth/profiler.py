"""Fidelity-versus-depth trade-off profiling.

One synthesis run already produces every shallower circuit as a by-product: the
``L``-layer circuit is built from the first ``L`` staircases the disentangler emits.
So the whole trade-off curve costs one pass plus one verification per point.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np

from .circuit import Circuit
from .mps import MPS
from .synthesis import (
    DENSE_SIMULATION_LIMIT,
    _layers_to_circuit,
    prepare_vector,
    synthesize_mps,
)

__all__ = ["TradeoffPoint", "TradeoffProfile", "profile", "profile_mps"]


@dataclass(frozen=True)
class TradeoffPoint:
    """One point on the depth/fidelity curve."""

    layers: int
    fidelity: float
    depth: int
    two_qubit_depth: int
    cnot: int
    one_qubit: int
    total_gates: int

    @property
    def infidelity(self) -> float:
        return 1.0 - self.fidelity

    def as_dict(self) -> dict[str, float | int]:
        return {
            "layers": self.layers,
            "fidelity": self.fidelity,
            "infidelity": self.infidelity,
            "depth": self.depth,
            "two_qubit_depth": self.two_qubit_depth,
            "cnot": self.cnot,
            "one_qubit": self.one_qubit,
            "total_gates": self.total_gates,
        }


@dataclass
class TradeoffProfile:
    """The full curve plus the metadata needed to interpret it."""

    n_qubits: int
    points: list[TradeoffPoint]
    input_bond_dimension: int
    fidelity_exact: bool = True
    circuits: list[Circuit] = field(default_factory=list, repr=False)

    @property
    def exact_cnot_estimate(self) -> int:
        """CNOT count of a textbook exact amplitude-encoding circuit.

        The standard Mottonen/Shende-Bullock-Markov construction uses
        ``2**n - 2`` CNOTs for an ``n``-qubit real-amplitude state and twice that
        with phases; ``2**n - 2`` is the honest lower-bound-flavoured baseline.
        """
        return 2**self.n_qubits - 2

    def best_for(self, fidelity: float) -> TradeoffPoint | None:
        """Shallowest point meeting a fidelity target, or ``None`` if unreachable."""
        for point in self.points:
            if point.fidelity >= fidelity:
                return point
        return None

    def circuit_for(self, layers: int) -> Circuit:
        """The circuit corresponding to a given layer count."""
        for point, circuit in zip(self.points, self.circuits, strict=True):
            if point.layers == layers:
                return circuit
        raise KeyError(f"no profiled circuit with {layers} layers")

    # ------------------------------------------------------------------ renderers

    _BAR_WIDTH = 18

    def _nines(self, point: TradeoffPoint) -> float:
        """Decades of accuracy, i.e. ``-log10(1 - F)`` -- how many nines of fidelity."""
        if point.infidelity <= 1e-16:
            return 16.0
        return float(-np.log10(point.infidelity))

    def _bar_scale(self) -> float:
        """Full-bar anchor: the next whole nine above the best point on the curve."""
        return max(1.0, float(np.ceil(max(self._nines(p) for p in self.points))))

    def _bar(self, point: TradeoffPoint, scale: float) -> str:
        """Inline bar on ``-log10(1 - F)``.

        A *linear* fidelity bar is useless here -- every usable circuit rounds to a
        full bar.  Accuracy is won in decades of infidelity, so that is what the bar
        measures, and the eye can read diminishing returns straight off it.
        """
        fraction = np.clip(self._nines(point) / scale, 0.0, 1.0)
        filled = int(round(fraction * self._BAR_WIDTH))
        return "#" * filled + "." * (self._BAR_WIDTH - filled)

    def table(self, fidelity_target: float | None = None) -> str:
        """Human-readable trade-off table with an inline log-infidelity bar."""
        scale = self._bar_scale()
        header = (
            f"{'layers':>6} {'CNOT':>6} {'depth':>6} {'2q-depth':>9} "
            f"{'fidelity':>10} {'infidelity':>11}  accuracy (full bar = "
            f"{scale:.0f} nine{'' if scale == 1 else 's'})"
        )
        rows = [header, "-" * (len(header) + 4)]
        for p in self.points:
            mark = ""
            if fidelity_target is not None and p.fidelity >= fidelity_target:
                mark = "  <-- meets target"
            rows.append(
                f"{p.layers:>6} {p.cnot:>6} {p.depth:>6} {p.two_qubit_depth:>9} "
                f"{p.fidelity:>10.6f} {p.infidelity:>11.3e}  {self._bar(p, scale)}{mark}"
            )
        rows.append("")
        rows.append(
            f"exact amplitude encoding baseline: ~{self.exact_cnot_estimate} CNOTs "
            f"({self.n_qubits} qubits, bond dimension {self.input_bond_dimension})"
        )
        if not self.fidelity_exact:
            rows.append("fidelities are MPS estimates (register too large to simulate densely)")
        return "\n".join(rows)

    def to_markdown(self) -> str:
        lines = [
            "| layers | CNOT | depth | 2q-depth | fidelity | infidelity |",
            "| ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for p in self.points:
            lines.append(
                f"| {p.layers} | {p.cnot} | {p.depth} | {p.two_qubit_depth} "
                f"| {p.fidelity:.6f} | {p.infidelity:.3e} |"
            )
        return "\n".join(lines)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(
            {
                "n_qubits": self.n_qubits,
                "input_bond_dimension": self.input_bond_dimension,
                "fidelity_exact": self.fidelity_exact,
                "exact_cnot_estimate": self.exact_cnot_estimate,
                "points": [p.as_dict() for p in self.points],
            },
            indent=indent,
        )

    def plot(self, path: str, title: str | None = None) -> str:
        """Write a fidelity-vs-CNOT / infidelity-vs-depth figure. Needs matplotlib."""
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "plotting needs matplotlib; install with `pip install mpsynth[plot]`"
            ) from exc

        cnots = [p.cnot for p in self.points]
        fids = [p.fidelity for p in self.points]
        infids = [max(p.infidelity, 1e-16) for p in self.points]
        depths = [p.two_qubit_depth for p in self.points]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
        ax1.plot(cnots, fids, "o-", color="#2b6cb0")
        for p in self.points:
            ax1.annotate(
                f"L={p.layers}",
                (p.cnot, p.fidelity),
                textcoords="offset points",
                xytext=(4, -10),
                fontsize=8,
            )
        ax1.set_xlabel("CNOT count")
        ax1.set_ylabel(r"fidelity  $|\langle\psi_{exact}|\psi_{approx}\rangle|^2$")
        ax1.grid(alpha=0.3)

        ax2.semilogy(depths, infids, "s-", color="#c05621")
        ax2.set_xlabel("two-qubit depth")
        ax2.set_ylabel("infidelity  $1 - F$")
        ax2.grid(alpha=0.3, which="both")

        fig.suptitle(title or f"MPSynth trade-off, {self.n_qubits} qubits")
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path


# --------------------------------------------------------------------- profiling


def profile_mps(
    target: MPS,
    max_layers: int = 8,
    residual_chi: int | None = None,
    tol: float = 0.0,
    optimize: bool = True,
    verify_chi: int | None = None,
) -> TradeoffProfile:
    """Profile every layer count from 1 to ``max_layers`` for an MPS target."""
    work = target.copy()
    work.canonicalize()
    work.normalize()

    # fidelity=1.0 keeps the disentangler from stopping early, so we get the
    # complete ladder of staircases in one pass.
    outcome = synthesize_mps(
        work,
        fidelity=1.0,
        max_layers=max_layers,
        residual_chi=residual_chi,
        tol=tol,
        optimize=optimize,
        verify_chi=verify_chi,
        collect_layers=True,
    )
    assert isinstance(outcome, tuple)
    _, layers = outcome

    dense = work.n_sites <= DENSE_SIMULATION_LIMIT
    reference = work.to_statevector() if dense else None
    chi = verify_chi if verify_chi is not None else max(32, 4 * work.max_bond())

    points: list[TradeoffPoint] = []
    circuits: list[Circuit] = []
    for count in range(1, len(layers) + 1):
        circuit = _layers_to_circuit(layers[:count], work.n_sites, optimize)
        fid = _fidelity_of(circuit, work, reference, chi)
        m = circuit.metrics()
        points.append(
            TradeoffPoint(
                layers=count,
                fidelity=fid,
                depth=int(m["depth"]),
                two_qubit_depth=int(m["two_qubit_depth"]),
                cnot=int(m["cnot"]),
                one_qubit=int(m["one_qubit"]),
                total_gates=int(m["total_gates"]),
            )
        )
        circuits.append(circuit)

    return TradeoffProfile(
        n_qubits=work.n_sites,
        points=points,
        input_bond_dimension=work.max_bond(),
        fidelity_exact=dense,
        circuits=circuits,
    )


def _fidelity_of(circuit: Circuit, target: MPS, reference: np.ndarray | None, chi: int) -> float:
    if reference is not None:
        return float(abs(np.vdot(reference, circuit.statevector())) ** 2)
    produced, _ = circuit.to_mps(chi_max=chi)
    produced.canonicalize(chi_max=chi)
    produced.normalize()
    return float(abs(target.overlap(produced)) ** 2)


def profile(
    vector: np.ndarray,
    max_layers: int = 8,
    chi_max: int | None = None,
    residual_chi: int | None = None,
    tol: float = 0.0,
    pad: bool = True,
    qubit_order: str = "big",
    optimize: bool = True,
) -> TradeoffProfile:
    """Profile the depth/fidelity trade-off for a raw classical vector.

    Arguments mirror :func:`mpsynth.synthesize`.
    """
    psi, _, _, _ = prepare_vector(vector, pad=pad, qubit_order=qubit_order)
    target, _ = MPS.from_statevector(psi, chi_max=chi_max, tol=tol)
    return profile_mps(
        target,
        max_layers=max_layers,
        residual_chi=residual_chi,
        tol=tol,
        optimize=optimize,
    )
