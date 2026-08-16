"""Tests for the circuit IR, its metrics, simulation and optimiser."""

from __future__ import annotations

import numpy as np
import pytest

from conftest import assert_equal_up_to_phase, random_unitary
from mpsynth.circuit import Circuit, Gate
from mpsynth.gates import CX01


def test_gate_arity_validation():
    with pytest.raises(ValueError, match="unknown gate"):
        Gate("rx", (0,), (0.1,))
    with pytest.raises(ValueError, match="takes"):
        Gate("cx", (0,), ())
    with pytest.raises(ValueError, match="takes"):
        Gate("rz", (0,), ())


def test_qubit_bounds_are_checked():
    circuit = Circuit(2)
    with pytest.raises(IndexError):
        circuit.rz(0.1, 5)
    with pytest.raises(ValueError, match="distinct"):
        circuit.cx(1, 1)


def test_metrics():
    circuit = Circuit(3)
    circuit.ry(0.1, 0).cx(0, 1).ry(0.2, 2).cx(1, 2).rz(0.3, 2)
    assert circuit.cnot_count == 2
    assert circuit.one_qubit_count == 3
    assert circuit.gate_counts() == {"ry": 2, "cx": 2, "rz": 1}
    # ry(0)/ry(2) run in parallel, then cx(0,1), then cx(1,2), then rz.
    assert circuit.depth == 4
    assert circuit.two_qubit_depth == 2
    assert circuit.metrics()["total_gates"] == 5


def test_empty_circuit_metrics():
    circuit = Circuit(3)
    assert circuit.depth == 0
    assert circuit.two_qubit_depth == 0
    np.testing.assert_allclose(circuit.statevector(), np.eye(8)[0], atol=1e-15)


def test_statevector_matches_unitary_first_column(rng):
    circuit = Circuit(3)
    circuit.ry(0.7, 0).cx(0, 1).rz(-1.1, 1).cx(2, 1).ry(0.3, 2)
    circuit.global_phase = 0.31
    np.testing.assert_allclose(
        circuit.statevector(), circuit.unitary()[:, 0], atol=1e-12
    )


def test_cx_matches_the_reference_matrix():
    circuit = Circuit(2)
    circuit.cx(0, 1)
    np.testing.assert_allclose(circuit.unitary(), CX01, atol=1e-15)

    reversed_circuit = Circuit(2)
    reversed_circuit.cx(1, 0)
    expected = np.array(
        [[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]], dtype=complex
    )
    np.testing.assert_allclose(reversed_circuit.unitary(), expected, atol=1e-15)


def test_append_1q_reproduces_arbitrary_unitaries(rng):
    for _ in range(50):
        u = random_unitary(2, rng)
        circuit = Circuit(1)
        circuit.append_1q(u, 0)
        np.testing.assert_allclose(circuit.unitary(), u, atol=1e-12)


def test_inverse_undoes_the_circuit(rng):
    circuit = Circuit(3)
    circuit.ry(0.7, 0).cx(0, 1).rz(-1.1, 1).cx(2, 1).ry(0.3, 2)
    circuit.global_phase = 0.4
    combined = circuit.copy().extend(circuit.inverse())
    np.testing.assert_allclose(combined.unitary(), np.eye(8), atol=1e-12)


def test_extend_rejects_mismatched_widths():
    with pytest.raises(ValueError, match="different qubit counts"):
        Circuit(2).extend(Circuit(3))


# ------------------------------------------------------------------- optimisation


def test_optimize_preserves_the_prepared_state(rng):
    for _ in range(25):
        circuit = Circuit(4)
        for _ in range(30):
            kind = rng.integers(0, 3)
            if kind == 2:
                a, b = rng.choice(4, size=2, replace=False)
                circuit.cx(int(a), int(b))
            elif kind == 1:
                circuit.rz(float(rng.uniform(-4, 4)), int(rng.integers(0, 4)))
            else:
                circuit.ry(float(rng.uniform(-4, 4)), int(rng.integers(0, 4)))
        optimized = circuit.optimized()
        np.testing.assert_allclose(
            optimized.statevector(), circuit.statevector(), atol=1e-11
        )
        assert len(optimized) <= len(circuit)


def test_optimize_preserves_the_unitary_when_leading_phase_is_kept(rng):
    circuit = Circuit(3)
    for _ in range(40):
        circuit.rz(float(rng.uniform(-4, 4)), int(rng.integers(0, 3)))
        circuit.ry(float(rng.uniform(-4, 4)), int(rng.integers(0, 3)))
        a, b = rng.choice(3, size=2, replace=False)
        circuit.cx(int(a), int(b))
    optimized = circuit.optimized(drop_leading_phase=False)
    np.testing.assert_allclose(optimized.unitary(), circuit.unitary(), atol=1e-11)


def test_optimize_fuses_adjacent_single_qubit_gates():
    circuit = Circuit(2)
    for _ in range(6):
        circuit.rz(0.3, 0).ry(0.2, 0)
    circuit.cx(0, 1)
    optimized = circuit.optimized()
    assert optimized.one_qubit_count <= 3
    assert optimized.cnot_count == 1


def test_optimize_removes_no_op_rotations():
    circuit = Circuit(2)
    circuit.rz(0.0, 0).ry(0.0, 1).cx(0, 1)
    optimized = circuit.optimized()
    assert optimized.one_qubit_count == 0
    assert optimized.cnot_count == 1


def test_optimize_drops_leading_diagonal_rotations():
    """RZ on a qubit still in |0> is a phase and cannot affect the prepared state."""
    circuit = Circuit(2)
    circuit.rz(1.3, 0).ry(0.5, 0).cx(0, 1)
    optimized = circuit.optimized()
    assert optimized.gate_counts().get("rz", 0) == 0
    np.testing.assert_allclose(optimized.statevector(), circuit.statevector(), atol=1e-12)


def test_optimize_is_idempotent(rng):
    circuit = Circuit(3)
    for _ in range(20):
        circuit.ry(float(rng.uniform(-4, 4)), int(rng.integers(0, 3)))
        a, b = rng.choice(3, size=2, replace=False)
        circuit.cx(int(a), int(b))
    once = circuit.optimized()
    twice = once.optimized()
    assert len(twice) == len(once)
    np.testing.assert_allclose(twice.statevector(), once.statevector(), atol=1e-12)


def test_amplitude_order_survives_transformations():
    circuit = Circuit(2, amplitude_order="little")
    circuit.ry(0.4, 0).cx(0, 1)
    assert circuit.copy().amplitude_order == "little"
    assert circuit.optimized().amplitude_order == "little"
    assert circuit.inverse().amplitude_order == "little"


# -------------------------------------------------------------------- MPS backend


def test_to_mps_matches_dense_simulation(rng):
    circuit = Circuit(6)
    for _ in range(40):
        circuit.ry(float(rng.uniform(-4, 4)), int(rng.integers(0, 6)))
        site = int(rng.integers(0, 5))
        circuit.cx(site, site + 1)
    circuit.global_phase = 0.77
    state, discarded = circuit.to_mps()
    assert discarded == pytest.approx(0.0, abs=1e-12)
    # to_mps does not carry the global phase; compare up to it.
    assert_equal_up_to_phase(state.to_statevector(), circuit.statevector(), atol=1e-10)


def test_to_mps_rejects_long_range_gates():
    circuit = Circuit(4)
    circuit.cx(0, 3)
    with pytest.raises(ValueError, match="nearest-neighbour"):
        circuit.to_mps()


def test_simulation_guards_against_huge_registers():
    with pytest.raises(ValueError, match="refusing"):
        Circuit(40).statevector()
    with pytest.raises(ValueError, match="refusing"):
        Circuit(20).unitary()
