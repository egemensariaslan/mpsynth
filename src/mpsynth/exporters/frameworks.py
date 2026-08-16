"""Python-framework emitters: PennyLane and Qiskit.

Each target gets a source emitter (always available, no dependency) plus a live
object builder that imports the framework lazily.
"""

from __future__ import annotations

from typing import Any

from ..circuit import Circuit
from ._header import header_lines

__all__ = ["to_pennylane", "to_qiskit", "build_pennylane", "build_qiskit"]


def _angle(value: float) -> str:
    return repr(float(value))


# ---------------------------------------------------------------------- PennyLane


def to_pennylane(circuit: Circuit, function_name: str = "prepare_state") -> str:
    """Emit a PennyLane quantum function.

    ``qml.GlobalPhase(phi)`` applies ``exp(-i phi)``, so the tracked phase is
    negated on the way out.
    """
    body: list[str] = []
    for gate in circuit.gates:
        if gate.name == "cx":
            c, t = gate.qubits
            body.append(f"    qml.CNOT(wires=[wires[{c}], wires[{t}]])")
        else:
            (q,) = gate.qubits
            name = "RZ" if gate.name == "rz" else "RY"
            body.append(f"    qml.{name}({_angle(gate.params[0])}, wires=wires[{q}])")
    if circuit.global_phase:
        body.append(f"    qml.GlobalPhase({_angle(-circuit.global_phase)})")
    if not body:
        body.append("    pass")

    return "\n".join(
        [
            '"""' + header_lines(circuit)[0] + '\n\n' +
            "\n".join(header_lines(circuit)[1:]) + '\n"""',
            "",
            "import pennylane as qml",
            "",
            f"N_QUBITS = {circuit.n_qubits}",
            "",
            "",
            f"def {function_name}(wires=None):",
            '    """Apply the state-preparation circuit to `wires` (default: 0..N_QUBITS-1)."""',
            "    if wires is None:",
            "        wires = list(range(N_QUBITS))",
            *body,
            "",
        ]
    )


def build_pennylane(circuit: Circuit, wires: list[Any] | None = None):
    """Return a callable applying this circuit with live PennyLane operations."""
    import pennylane as qml  # noqa: PLC0415 - optional dependency, imported on demand

    targets = list(range(circuit.n_qubits)) if wires is None else list(wires)

    def prepare_state() -> None:
        for gate in circuit.gates:
            if gate.name == "cx":
                c, t = gate.qubits
                qml.CNOT(wires=[targets[c], targets[t]])
            elif gate.name == "rz":
                qml.RZ(gate.params[0], wires=targets[gate.qubits[0]])
            else:
                qml.RY(gate.params[0], wires=targets[gate.qubits[0]])
        if circuit.global_phase:
            qml.GlobalPhase(-circuit.global_phase)

    return prepare_state


# ------------------------------------------------------------------------- Qiskit


def to_qiskit(circuit: Circuit, variable: str = "qc") -> str:
    """Emit Python source that rebuilds this circuit with Qiskit.

    Qiskit reads statevector index bits little-endian.  Synthesise with
    ``qubit_order="little"`` if you want ``Statevector(qc)`` to line up
    element-for-element with your input vector.
    """
    body: list[str] = []
    for gate in circuit.gates:
        if gate.name == "cx":
            c, t = gate.qubits
            body.append(f"{variable}.cx({c}, {t})")
        else:
            (q,) = gate.qubits
            body.append(f"{variable}.{gate.name}({_angle(gate.params[0])}, {q})")
    if circuit.global_phase:
        body.append(f"{variable}.global_phase = {_angle(circuit.global_phase)}")

    return "\n".join(
        [
            '"""' + header_lines(circuit)[0] + '\n\n' +
            "\n".join(header_lines(circuit)[1:]) + '\n"""',
            "",
            "from qiskit import QuantumCircuit",
            "",
            f"{variable} = QuantumCircuit({circuit.n_qubits})",
            *body,
            "",
        ]
    )


def build_qiskit(circuit: Circuit):
    """Return a live ``qiskit.QuantumCircuit`` equivalent to this circuit."""
    from qiskit import QuantumCircuit  # noqa: PLC0415 - optional dependency

    qc = QuantumCircuit(circuit.n_qubits)
    for gate in circuit.gates:
        if gate.name == "cx":
            qc.cx(*gate.qubits)
        elif gate.name == "rz":
            qc.rz(gate.params[0], gate.qubits[0])
        else:
            qc.ry(gate.params[0], gate.qubits[0])
    qc.global_phase = circuit.global_phase
    return qc
