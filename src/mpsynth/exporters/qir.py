"""QIR (Quantum Intermediate Representation) emitter.

Produces a textual LLVM module against the QIR base profile: static qubit ids, no
branching on measurement, one ``main`` entry point.  Angles are written as LLVM hex
float literals so no precision is lost through decimal rounding.
"""

from __future__ import annotations

import struct

from ..circuit import Circuit
from ._header import header_lines

__all__ = ["to_qir"]


def _double(value: float) -> str:
    """LLVM hex literal for an IEEE-754 double -- exact, unlike decimal text."""
    return "0x" + struct.pack(">d", float(value)).hex().upper()


def _qubit(index: int) -> str:
    return f"%Qubit* inttoptr (i64 {index} to %Qubit*)"


def _result(index: int) -> str:
    return f"%Result* inttoptr (i64 {index} to %Result*)"


def to_qir(
    circuit: Circuit,
    entry_point: str = "main",
    measure: bool = False,
    module_id: str = "mpsynth",
) -> str:
    """Emit a QIR base-profile LLVM module.

    Args:
        circuit: the synthesised circuit.
        entry_point: name of the entry-point function.
        measure: append a Z-basis measurement and output recording for every qubit.
            Leave off when the circuit is a state-preparation subroutine.
        module_id: LLVM module identifier.
    """
    n = circuit.n_qubits
    body: list[str] = []
    used = {"rz": False, "ry": False, "cnot": False}

    for gate in circuit.gates:
        if gate.name == "cx":
            c, t = gate.qubits
            body.append(f"  call void @__quantum__qis__cnot__body({_qubit(c)}, {_qubit(t)})")
            used["cnot"] = True
        else:
            (q,) = gate.qubits
            body.append(
                f"  call void @__quantum__qis__{gate.name}__body("
                f"double {_double(gate.params[0])}, {_qubit(q)})"
            )
            used[gate.name] = True

    n_results = 0
    if measure:
        for q in range(n):
            body.append(f"  call void @__quantum__qis__mz__body({_qubit(q)}, {_result(q)})")
        body.append("")
        for q in range(n):
            body.append(f"  call void @__quantum__rt__result_record_output({_result(q)}, i8* null)")
        n_results = n

    declarations = [
        "declare void @__quantum__rt__initialize(i8*)",
    ]
    if used["rz"]:
        declarations.append("declare void @__quantum__qis__rz__body(double, %Qubit*)")
    if used["ry"]:
        declarations.append("declare void @__quantum__qis__ry__body(double, %Qubit*)")
    if used["cnot"]:
        declarations.append("declare void @__quantum__qis__cnot__body(%Qubit*, %Qubit*)")
    if measure:
        declarations += [
            "declare void @__quantum__qis__mz__body(%Qubit*, %Result*) #1",
            "declare void @__quantum__rt__result_record_output(%Result*, i8*)",
        ]

    header = [
        f"; ModuleID = '{module_id}'",
        f'source_filename = "{module_id}"',
        *(f"; {line}" for line in header_lines(circuit)),
        f"; global phase: {circuit.global_phase!r} rad (unobservable; not emitted)",
        "",
        "%Qubit = type opaque",
        "%Result = type opaque",
        "",
        f"define void @{entry_point}() #0 {{",
        "entry:",
        "  call void @__quantum__rt__initialize(i8* null)",
    ]
    footer = [
        "  ret void",
        "}",
        "",
        *declarations,
        "",
        'attributes #0 = { "entry_point" "output_labeling_schema"'
        ' "qir_profiles"="base_profile"'
        f' "required_num_qubits"="{n}" "required_num_results"="{n_results}" }}',
        'attributes #1 = { "irreversible" }' if measure else "",
        "",
        "!llvm.module.flags = !{!0, !1, !2, !3}",
        "",
        '!0 = !{i32 1, !"qir_major_version", i32 1}',
        '!1 = !{i32 7, !"qir_minor_version", i32 0}',
        '!2 = !{i32 1, !"dynamic_qubit_management", i1 false}',
        '!3 = !{i32 1, !"dynamic_result_management", i1 false}',
    ]
    lines = [*header, *body, *footer]
    return "\n".join(line for line in lines if line is not None) + "\n"
