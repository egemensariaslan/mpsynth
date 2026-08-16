"""Q# emitter."""

from __future__ import annotations

from ..circuit import Circuit
from ._header import header_lines

__all__ = ["to_qsharp"]


def _angle(value: float) -> str:
    return f"{value:.17g}"


def to_qsharp(
    circuit: Circuit,
    namespace: str = "MPSynth",
    operation: str = "PrepareState",
) -> str:
    """Emit a Q# operation preparing the state on a caller-supplied register.

    Q#'s ``Rz``/``Ry`` use the same ``exp(-i theta P / 2)`` convention as MPSynth, so
    angles carry over unchanged.  The global phase is applied with
    ``R(PauliI, -2 phi, ...)``, which is unobservable on its own but keeps the
    operation correct under ``Controlled``.
    """
    body: list[str] = []
    for gate in circuit.gates:
        if gate.name == "cx":
            c, t = gate.qubits
            body.append(f"        CNOT(qs[{c}], qs[{t}]);")
        else:
            (q,) = gate.qubits
            name = "Rz" if gate.name == "rz" else "Ry"
            body.append(f"        {name}({_angle(gate.params[0])}, qs[{q}]);")
    if circuit.global_phase:
        body.append(f"        R(PauliI, {_angle(-2.0 * circuit.global_phase)}, qs[0]);")

    n = circuit.n_qubits
    return "\n".join(
        [
            *(f"// {line}" for line in header_lines(circuit)),
            f"namespace {namespace} {{",
            "",
            "    open Microsoft.Quantum.Intrinsic;",
            "    open Microsoft.Quantum.Diagnostics;",
            "",
            "    /// # Summary",
            f"    /// Prepares the synthesised state on `qs`, which must hold {n}"
            " qubits in the |0> state.",
            f"    operation {operation}(qs : Qubit[]) : Unit is Adj + Ctl {{",
            f'        Fact(Length(qs) == {n}, "expected {n} qubits");',
            *body,
            "    }",
            "}",
        ]
    ) + "\n"
