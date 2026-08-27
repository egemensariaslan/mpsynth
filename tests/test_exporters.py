"""Exporter tests.

Text-shape assertions are cheap but weak, so wherever the real framework is
installed the exported artefact is *executed* and its statevector compared against
MPSynth's own simulation.  Those tests skip cleanly when the framework is absent.
"""

from __future__ import annotations

import numpy as np
import pytest

from mpsynth import export, synthesize
from mpsynth.circuit import Circuit
from mpsynth.datasets import generate
from mpsynth.exporters import EXTENSIONS, FORMATS, to_qir


@pytest.fixture(scope="module")
def result():
    return synthesize(generate("damped", 5), fidelity=0.99, max_layers=4)


@pytest.fixture(scope="module")
def little_endian_result():
    return synthesize(generate("damped", 5), fidelity=0.99, max_layers=4, qubit_order="little")


# ---------------------------------------------------------------------- registry


@pytest.mark.parametrize("fmt", sorted(FORMATS))
def test_every_format_emits_non_empty_text(fmt, result):
    text = export(result.circuit, fmt)
    assert isinstance(text, str) and text.strip()
    assert fmt in EXTENSIONS


@pytest.mark.parametrize("alias,canonical", [("qasm", "qasm2"), ("q#", "qsharp"), ("llvm", "qir")])
def test_aliases(alias, canonical, result):
    assert export(result.circuit, alias) == export(result.circuit, canonical)


def test_unknown_format_is_rejected(result):
    with pytest.raises(ValueError, match="unknown export format"):
        export(result.circuit, "brainfuck")


@pytest.mark.parametrize("fmt", sorted(FORMATS))
def test_header_records_the_amplitude_convention(fmt, result, little_endian_result):
    assert "BIG-endian" in export(result.circuit, fmt)
    assert "LITTLE-endian" in export(little_endian_result.circuit, fmt)


@pytest.mark.parametrize("fmt", sorted(FORMATS))
def test_empty_circuit_still_emits_valid_text(fmt):
    text = export(Circuit(2), fmt)
    assert isinstance(text, str) and text.strip()


# -------------------------------------------------------------------- OpenQASM


def test_qasm2_shape(result):
    text = export(result.circuit, "qasm2")
    assert "OPENQASM 2.0;" in text
    assert 'include "qelib1.inc";' in text
    assert f"qreg q[{result.n_qubits}];" in text
    assert text.count("cx ") == result.circuit.cnot_count
    assert "global phase" in text  # unrepresentable, so it must be documented


def test_qasm3_shape(result):
    text = export(result.circuit, "qasm3")
    assert "OPENQASM 3.0;" in text
    assert f"qubit[{result.n_qubits}] q;" in text
    if result.circuit.global_phase:
        assert "gphase(" in text


def test_qasm_angles_keep_full_precision():
    circuit = Circuit(1)
    circuit.rz(0.1234567890123456789, 0)
    text = export(circuit, "qasm3")
    assert "0.12345678901234568" in text


def test_qasm2_and_qasm3_bodies_agree(result):
    body2 = [
        ln
        for ln in export(result.circuit, "qasm2").splitlines()
        if ln and not ln.startswith(("//", "OPENQASM", "include", "qreg"))
    ]
    body3 = [
        ln
        for ln in export(result.circuit, "qasm3").splitlines()
        if ln and not ln.startswith(("//", "OPENQASM", "include", "qubit", "gphase"))
    ]
    assert body2 == body3


# --------------------------------------------------------------------------- QIR


def test_qir_shape(result):
    text = to_qir(result.circuit)
    assert "%Qubit = type opaque" in text
    assert "define void @main()" in text
    assert f'"required_num_qubits"="{result.n_qubits}"' in text
    assert '"entry_point"' in text
    assert (
        text.count("__quantum__qis__cnot__body") == result.circuit.cnot_count + 1
    )  # + declaration
    assert "!llvm.module.flags" in text


def test_qir_angles_are_exact_hex_doubles():
    circuit = Circuit(1)
    circuit.rz(0.1, 0)
    text = to_qir(circuit)
    assert "double 0x3FB999999999999A" in text, "0.1 must round-trip bit-exactly"


def test_qir_only_declares_the_intrinsics_it_uses():
    circuit = Circuit(2)
    circuit.rz(0.3, 0)
    text = to_qir(circuit)
    assert "__quantum__qis__rz__body" in text
    assert "__quantum__qis__ry__body" not in text
    assert "__quantum__qis__cnot__body" not in text


def test_qir_measurement_option(result):
    text = to_qir(result.circuit, measure=True)
    assert text.count("__quantum__qis__mz__body") == result.n_qubits + 1
    assert "__quantum__rt__result_record_output" in text
    assert f'"required_num_results"="{result.n_qubits}"' in text
    assert '"required_num_results"="0"' in to_qir(result.circuit)


# ---------------------------------------------------------------------------- Q#


def test_qsharp_shape(result):
    text = export(result.circuit, "qsharp")
    assert "namespace MPSynth {" in text
    assert "operation PrepareState(qs : Qubit[]) : Unit is Adj + Ctl {" in text
    assert text.count("CNOT(") == result.circuit.cnot_count
    assert f"Length(qs) == {result.n_qubits}" in text
    assert text.count("{") == text.count("}")


def test_qsharp_custom_names(result):
    text = export(result.circuit, "qsharp", namespace="Foo.Bar", operation="LoadData")
    assert "namespace Foo.Bar {" in text
    assert "operation LoadData(" in text


# ------------------------------------------------ live framework round-trips


def test_pennylane_matches_mpsynth_simulation(result):
    qml = pytest.importorskip("pennylane")
    from mpsynth.exporters import build_pennylane

    dev = qml.device("default.qubit", wires=result.n_qubits)
    apply_prep = build_pennylane(result.circuit)

    @qml.qnode(dev)
    def circuit():
        apply_prep()
        return qml.state()

    np.testing.assert_allclose(np.asarray(circuit()), result.circuit.statevector(), atol=1e-10)


def test_exported_pennylane_source_runs_and_matches(result):
    qml = pytest.importorskip("pennylane")
    import types

    module = types.ModuleType("generated")
    exec(compile(export(result.circuit, "pennylane"), "generated.py", "exec"), module.__dict__)
    assert module.N_QUBITS == result.n_qubits

    dev = qml.device("default.qubit", wires=result.n_qubits)

    @qml.qnode(dev)
    def circuit():
        module.prepare_state()
        return qml.state()

    np.testing.assert_allclose(np.asarray(circuit()), result.circuit.statevector(), atol=1e-10)


def test_qiskit_matches_mpsynth_simulation(little_endian_result):
    pytest.importorskip("qiskit")
    from qiskit.quantum_info import Statevector

    from mpsynth.exporters import build_qiskit

    produced = np.asarray(Statevector(build_qiskit(little_endian_result.circuit)))
    # Qiskit reads little-endian, so it should reproduce the *input* vector directly.
    target = generate("damped", 5).astype(complex)
    target /= np.linalg.norm(target)
    assert abs(np.vdot(target, produced)) ** 2 == pytest.approx(
        little_endian_result.fidelity, abs=1e-9
    )


def test_exported_qiskit_source_runs_and_matches(little_endian_result):
    pytest.importorskip("qiskit")
    import types

    from qiskit.quantum_info import Statevector

    from mpsynth.exporters import build_qiskit

    module = types.ModuleType("generated")
    exec(
        compile(export(little_endian_result.circuit, "qiskit"), "generated.py", "exec"),
        module.__dict__,
    )
    np.testing.assert_allclose(
        np.asarray(Statevector(module.qc)),
        np.asarray(Statevector(build_qiskit(little_endian_result.circuit))),
        atol=1e-10,
    )


def test_qasm2_reloads_into_qiskit(little_endian_result):
    pytest.importorskip("qiskit")
    from qiskit.qasm2 import loads
    from qiskit.quantum_info import Statevector

    from mpsynth.exporters import build_qiskit

    reloaded = np.asarray(Statevector(loads(export(little_endian_result.circuit, "qasm2"))))
    reference = np.asarray(Statevector(build_qiskit(little_endian_result.circuit)))
    # OpenQASM 2 cannot carry the global phase, so compare up to it.
    np.testing.assert_allclose(np.abs(reloaded), np.abs(reference), atol=1e-10)


def test_qasm3_reloads_into_qiskit_including_global_phase(little_endian_result):
    pytest.importorskip("qiskit")
    pytest.importorskip("qiskit_qasm3_import")
    from qiskit.qasm3 import loads
    from qiskit.quantum_info import Statevector

    from mpsynth.exporters import build_qiskit

    reloaded = np.asarray(Statevector(loads(export(little_endian_result.circuit, "qasm3"))))
    reference = np.asarray(Statevector(build_qiskit(little_endian_result.circuit)))
    np.testing.assert_allclose(reloaded, reference, atol=1e-10)


def test_qir_parses_with_llvm(tmp_path, result):
    """Validate the emitted module with a real LLVM IR parser, when one is available."""
    import shutil
    import subprocess

    clang = shutil.which("clang") or shutil.which("llvm-as")
    if clang is None:
        pytest.skip("no LLVM IR parser available")

    path = tmp_path / "module.ll"
    path.write_text(to_qir(result.circuit, measure=True))
    if clang.endswith("llvm-as"):
        cmd = [clang, str(path), "-o", str(tmp_path / "module.bc")]
    else:
        cmd = [clang, "-x", "ir", "-S", "-emit-llvm", "-o", "/dev/null", str(path)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr


def test_global_phase_is_folded_into_a_readable_range():
    from mpsynth.datasets import generate

    deep = synthesize(generate("bimodal", 8), fidelity=1.0, max_layers=10)
    assert -np.pi - 1e-9 <= deep.circuit.global_phase <= np.pi + 1e-9
