"""Multi-target circuit exporters.

All targets consume the same ``{rz, ry, cx}`` IR, so gate counts and depth reported
by MPSynth describe every emitted artefact identically -- no target re-transpiles.

    >>> from mpsynth import synthesize, export
    >>> result = synthesize([0.1, 0.4, 0.2, 0.9])
    >>> qasm = export(result.circuit, "qasm3")
"""

from __future__ import annotations

from ..circuit import Circuit
from .frameworks import build_pennylane, build_qiskit, to_pennylane, to_qiskit
from .qasm import to_qasm2, to_qasm3
from .qir import to_qir
from .qsharp import to_qsharp

__all__ = [
    "FORMATS",
    "EXTENSIONS",
    "export",
    "to_qasm2",
    "to_qasm3",
    "to_qir",
    "to_qsharp",
    "to_pennylane",
    "to_qiskit",
    "build_pennylane",
    "build_qiskit",
]

FORMATS = {
    "qasm2": to_qasm2,
    "qasm3": to_qasm3,
    "qir": to_qir,
    "qsharp": to_qsharp,
    "pennylane": to_pennylane,
    "qiskit": to_qiskit,
}

#: Conventional file extension per target, used by the CLI when naming outputs.
EXTENSIONS = {
    "qasm2": ".qasm",
    "qasm3": ".qasm",
    "qir": ".ll",
    "qsharp": ".qs",
    "pennylane": ".py",
    "qiskit": ".py",
}

_ALIASES = {
    "qasm": "qasm2",
    "openqasm": "qasm2",
    "openqasm2": "qasm2",
    "openqasm3": "qasm3",
    "qs": "qsharp",
    "q#": "qsharp",
    "llvm": "qir",
    "pl": "pennylane",
}


def export(circuit: Circuit, fmt: str, **kwargs) -> str:
    """Render ``circuit`` in the named target format.

    Args:
        circuit: circuit to export.
        fmt: one of :data:`FORMATS` (aliases such as ``"qasm"`` and ``"q#"`` work too).
        **kwargs: forwarded to the individual emitter.
    """
    key = _ALIASES.get(fmt.lower(), fmt.lower())
    try:
        emitter = FORMATS[key]
    except KeyError:
        raise ValueError(
            f"unknown export format {fmt!r}; choose from {sorted(FORMATS)}"
        ) from None
    return emitter(circuit, **kwargs)
