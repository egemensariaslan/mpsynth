"""Backend-independent circuit IR.

MPSynth synthesises into a deliberately tiny gate set -- ``rz``, ``ry`` and ``cx``
plus a tracked global phase.  Every export target supports these natively, so no
exporter has to re-transpile, and gate/depth metrics mean the same thing across
targets.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field, replace

import numpy as np

from .gates import CX01, ry, rz, wrap_angle, zyz_decomposition

__all__ = ["Gate", "Circuit"]

ONE_QUBIT_GATES = ("rz", "ry")
TWO_QUBIT_GATES = ("cx",)


@dataclass(frozen=True)
class Gate:
    """A single instruction: gate name, target qubits, rotation parameters."""

    name: str
    qubits: tuple[int, ...]
    params: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if self.name in ONE_QUBIT_GATES:
            expected_q, expected_p = 1, 1
        elif self.name in TWO_QUBIT_GATES:
            expected_q, expected_p = 2, 0
        else:
            raise ValueError(f"unknown gate {self.name!r}")
        if len(self.qubits) != expected_q or len(self.params) != expected_p:
            raise ValueError(
                f"{self.name} takes {expected_q} qubit(s) and {expected_p} param(s), "
                f"got {len(self.qubits)} and {len(self.params)}"
            )

    @property
    def matrix(self) -> np.ndarray:
        if self.name == "rz":
            return rz(self.params[0])
        if self.name == "ry":
            return ry(self.params[0])
        return CX01

    def inverse(self) -> "Gate":
        if self.name == "cx":
            return self
        return replace(self, params=(-self.params[0],))


@dataclass
class Circuit:
    """An ordered list of gates on ``n_qubits`` qubits, plus a global phase.

    ``amplitude_order`` records which statevector index convention the circuit was
    synthesised for, so exported artefacts can say so in their own header:

    * ``"big"``   -- index bit 0 is the *most* significant; qubit 0 is the leading
      bit of the basis label.  This is what PennyLane's ``qml.state()`` returns and
      the usual textbook/OpenQASM reading.
    * ``"little"`` -- qubit 0 is the *least* significant bit, which is what Qiskit's
      ``Statevector`` and Aer return.

    It is metadata only: it never changes how gates are applied.
    """

    n_qubits: int
    gates: list[Gate] = field(default_factory=list)
    global_phase: float = 0.0
    amplitude_order: str = "big"

    # ----------------------------------------------------------------- builders

    def _check(self, *qubits: int) -> None:
        for q in qubits:
            if not 0 <= q < self.n_qubits:
                raise IndexError(f"qubit {q} out of range for {self.n_qubits} qubits")

    def rz(self, theta: float, q: int) -> "Circuit":
        self._check(q)
        self.gates.append(Gate("rz", (q,), (float(theta),)))
        return self

    def ry(self, theta: float, q: int) -> "Circuit":
        self._check(q)
        self.gates.append(Gate("ry", (q,), (float(theta),)))
        return self

    def cx(self, control: int, target: int) -> "Circuit":
        self._check(control, target)
        if control == target:
            raise ValueError("cx needs distinct control and target")
        self.gates.append(Gate("cx", (control, target)))
        return self

    def append_1q(self, matrix: np.ndarray, q: int, atol: float = 1e-12) -> "Circuit":
        """Append an arbitrary 2x2 unitary as an RZ-RY-RZ triple."""
        gates, phase = _zyz_gates(matrix, q, atol=atol)
        self.gates.extend(gates)
        self.global_phase += phase
        return self

    def extend(self, other: "Circuit") -> "Circuit":
        """Append ``other``'s gates (applied after this circuit's)."""
        if other.n_qubits != self.n_qubits:
            raise ValueError("cannot compose circuits over different qubit counts")
        self.gates.extend(other.gates)
        self.global_phase += other.global_phase
        return self

    def copy(self) -> "Circuit":
        return Circuit(
            self.n_qubits, list(self.gates), self.global_phase, self.amplitude_order
        )

    def inverse(self) -> "Circuit":
        return Circuit(
            self.n_qubits,
            [g.inverse() for g in reversed(self.gates)],
            -self.global_phase,
            self.amplitude_order,
        )

    # ------------------------------------------------------------------ metrics

    def __len__(self) -> int:
        return len(self.gates)

    def gate_counts(self) -> dict[str, int]:
        return dict(Counter(g.name for g in self.gates))

    @property
    def cnot_count(self) -> int:
        return sum(1 for g in self.gates if g.name == "cx")

    @property
    def one_qubit_count(self) -> int:
        return sum(1 for g in self.gates if g.name in ONE_QUBIT_GATES)

    @property
    def depth(self) -> int:
        """Circuit depth under as-soon-as-possible scheduling."""
        frontier = [0] * self.n_qubits
        for g in self.gates:
            t = max(frontier[q] for q in g.qubits) + 1
            for q in g.qubits:
                frontier[q] = t
        return max(frontier, default=0)

    @property
    def two_qubit_depth(self) -> int:
        """Depth counting only entangling layers -- the coherence-budget metric."""
        frontier = [0] * self.n_qubits
        for g in self.gates:
            if len(g.qubits) < 2:
                continue
            t = max(frontier[q] for q in g.qubits) + 1
            for q in g.qubits:
                frontier[q] = t
        return max(frontier, default=0)

    def metrics(self) -> dict[str, int]:
        counts = self.gate_counts()
        return {
            "n_qubits": self.n_qubits,
            "depth": self.depth,
            "two_qubit_depth": self.two_qubit_depth,
            "cnot": self.cnot_count,
            "one_qubit": self.one_qubit_count,
            "total_gates": len(self.gates),
            **{f"n_{k}": v for k, v in sorted(counts.items())},
        }

    # --------------------------------------------------------------- simulation

    def statevector(self, max_qubits: int = 26) -> np.ndarray:
        """Dense state produced by applying this circuit to |0...0>."""
        if self.n_qubits > max_qubits:
            raise ValueError(
                f"refusing to simulate {self.n_qubits} qubits densely "
                f"(limit {max_qubits}); use Circuit.to_mps instead"
            )
        psi = np.zeros((2,) * self.n_qubits, dtype=complex)
        psi[(0,) * self.n_qubits] = 1.0
        for g in self.gates:
            if len(g.qubits) == 1:
                (q,) = g.qubits
                psi = np.moveaxis(np.tensordot(g.matrix, psi, axes=([1], [q])), 0, q)
            else:
                c, t = g.qubits
                op = g.matrix.reshape(2, 2, 2, 2)
                psi = np.moveaxis(
                    np.tensordot(op, psi, axes=([2, 3], [c, t])), [0, 1], [c, t]
                )
        return np.exp(1j * self.global_phase) * psi.reshape(-1)

    def unitary(self, max_qubits: int = 12) -> np.ndarray:
        """Dense unitary of the circuit (small circuits only; for tests)."""
        if self.n_qubits > max_qubits:
            raise ValueError(f"refusing to build a 2**{self.n_qubits} unitary")
        dim = 2**self.n_qubits
        out = np.zeros((dim, dim), dtype=complex)
        for j in range(dim):
            psi = np.zeros((2,) * self.n_qubits, dtype=complex)
            psi[np.unravel_index(j, (2,) * self.n_qubits)] = 1.0
            for g in self.gates:
                if len(g.qubits) == 1:
                    (q,) = g.qubits
                    psi = np.moveaxis(np.tensordot(g.matrix, psi, axes=([1], [q])), 0, q)
                else:
                    c, t = g.qubits
                    op = g.matrix.reshape(2, 2, 2, 2)
                    psi = np.moveaxis(
                        np.tensordot(op, psi, axes=([2, 3], [c, t])), [0, 1], [c, t]
                    )
            out[:, j] = psi.reshape(-1)
        return np.exp(1j * self.global_phase) * out

    def to_mps(self, chi_max: int | None = None, tol: float = 0.0):
        """Apply this circuit to |0...0> as an MPS. Returns ``(mps, discarded)``."""
        from .mps import MPS  # local import keeps circuit.py dependency-free

        state = MPS.zero_state(self.n_qubits)
        discarded = 0.0
        for g in self.gates:
            if len(g.qubits) == 1:
                state.apply_1q(g.matrix, g.qubits[0])
                continue
            c, t = g.qubits
            if abs(c - t) != 1:
                raise ValueError(
                    f"MPS simulation needs nearest-neighbour gates, got cx {c},{t}"
                )
            site = min(c, t)
            mat = g.matrix if c < t else _swap_gate_qubits(g.matrix)
            discarded += state.apply_2q(mat, site, chi_max=chi_max, tol=tol)
        return state, discarded

    # ------------------------------------------------------------ optimisation

    def optimized(self, atol: float = 1e-12, drop_leading_phase: bool = True) -> "Circuit":
        """Fuse maximal single-qubit runs and drop no-op rotations.

        A fused run is only kept when it is no longer than the run it replaces, so
        the gate count never goes up: re-expressing a lone ``RY`` as a canonical
        RZ-RY-RZ triple would otherwise be a regression.

        With ``drop_leading_phase`` set (the default), leading diagonal rotations
        acting on a qubit still in |0> are removed and folded into the global phase.
        They cannot affect the prepared state, so ``statevector()`` is preserved but
        ``unitary()`` is not; pass ``False`` to keep the circuit's full unitary.
        """
        out = Circuit(self.n_qubits, [], self.global_phase, self.amplitude_order)
        pending: dict[int, list[Gate]] = {}
        touched = [False] * self.n_qubits

        def flush(q: int) -> None:
            run = pending.pop(q, None)
            if not run:
                return
            matrix = np.eye(2, dtype=complex)
            for gate in run:
                matrix = gate.matrix @ matrix
            fused, phase = _zyz_gates(
                matrix, q, atol=atol, at_start=drop_leading_phase and not touched[q]
            )
            if len(fused) <= len(run):
                out.gates.extend(fused)
                out.global_phase += phase
            else:
                out.gates.extend(run)
            touched[q] = True

        for g in self.gates:
            if len(g.qubits) == 1:
                pending.setdefault(g.qubits[0], []).append(g)
                continue
            for q in g.qubits:
                flush(q)
                touched[q] = True
            out.gates.append(g)
        for q in sorted(pending):
            flush(q)
        # Phases accumulate additively across hundreds of gates; fold the result back
        # into (-pi, pi].  exp(i phi) is unchanged, but the emitted number stays legible
        # and does not bleed precision into its low bits.
        out.global_phase = float(
            (out.global_phase + np.pi) % (2 * np.pi) - np.pi
        )
        return out


def _swap_gate_qubits(mat: np.ndarray) -> np.ndarray:
    """Reverse the qubit order of a 4x4 gate matrix."""
    t = mat.reshape(2, 2, 2, 2).transpose(1, 0, 3, 2)
    return t.reshape(4, 4)


def _zyz_gates(
    matrix: np.ndarray, q: int, atol: float = 1e-12, at_start: bool = False
) -> tuple[list[Gate], float]:
    """Lower a 2x2 unitary to RZ-RY-RZ gates on qubit ``q`` plus a global phase.

    With ``at_start`` the qubit is known to be in |0>, so leading diagonal rotations
    contribute nothing but a phase and are dropped.
    """
    phase, alpha, beta, gamma = zyz_decomposition(matrix)
    angles = [("rz", gamma), ("ry", beta), ("rz", alpha)]

    if at_start:
        phase -= gamma / 2.0  # RZ(gamma)|0> == exp(-i gamma / 2) |0>
        angles = angles[1:]
        if abs(wrap_angle(beta)[0]) <= atol:
            # No RY either, so the surviving RZ also only contributes a phase.
            phase -= alpha / 2.0
            angles = []

    gates: list[Gate] = []
    for name, angle in angles:
        wrapped, extra = wrap_angle(angle)
        phase += extra
        if abs(wrapped) > atol:
            gates.append(Gate(name, (q,), (wrapped,)))
    return gates, phase
