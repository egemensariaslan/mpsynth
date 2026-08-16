"""OpenQASM 2.0 and 3.0 emitters."""

from __future__ import annotations

from ..circuit import Circuit
from ._header import header_lines

__all__ = ["to_qasm2", "to_qasm3"]


def _angle(value: float) -> str:
    """Full double precision, so a round-trip does not lose fidelity."""
    return f"{value:.17g}"


def _body(circuit: Circuit, register: str) -> list[str]:
    lines = []
    for gate in circuit.gates:
        if gate.name == "cx":
            c, t = gate.qubits
            lines.append(f"cx {register}[{c}],{register}[{t}];")
        else:
            (q,) = gate.qubits
            lines.append(f"{gate.name}({_angle(gate.params[0])}) {register}[{q}];")
    return lines


def to_qasm2(circuit: Circuit, register: str = "q", include_header: bool = True) -> str:
    """Emit OpenQASM 2.0.

    The language has no global-phase instruction, so the tracked phase is recorded
    as a comment.  It is unobservable for state preparation but matters if the
    circuit is later used as a controlled subroutine.
    """
    lines: list[str] = []
    if include_header:
        lines += [
            *(f"// {line}" for line in header_lines(circuit)),
            f"// global phase: {_angle(circuit.global_phase)} rad"
            " (not representable in OpenQASM 2.0)",
            "OPENQASM 2.0;",
            'include "qelib1.inc";',
            "",
            f"qreg {register}[{circuit.n_qubits}];",
            "",
        ]
    lines += _body(circuit, register)
    return "\n".join(lines) + "\n"


def to_qasm3(circuit: Circuit, register: str = "q", include_header: bool = True) -> str:
    """Emit OpenQASM 3.0, including the global phase via ``gphase``."""
    lines: list[str] = []
    if include_header:
        lines += [
            *(f"// {line}" for line in header_lines(circuit)),
            "OPENQASM 3.0;",
            'include "stdgates.inc";',
            "",
            f"qubit[{circuit.n_qubits}] {register};",
            "",
        ]
    lines += _body(circuit, register)
    if circuit.global_phase:
        lines.append(f"gphase({_angle(circuit.global_phase)});")
    return "\n".join(lines) + "\n"
