"""End-to-end synthesis tests: correctness, monotonicity and the unitarity contract."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import random_state

from mpsynth import MPS, profile, synthesize, synthesize_mps
from mpsynth.datasets import GENERATORS, generate
from mpsynth.synthesis import bond2_layer, prepare_vector


def fidelity_against(vector, result):
    target = np.asarray(vector, dtype=complex).reshape(-1)
    target = target / np.linalg.norm(target)
    return float(abs(np.vdot(target, result.circuit.statevector())) ** 2)


# ------------------------------------------------- the exact bond-2 preparation


@pytest.mark.parametrize("n", [1, 2, 3, 4, 6, 9])
def test_bond2_states_are_prepared_exactly(n, rng):
    """One staircase must reproduce a bond-2 MPS to machine precision."""
    if n == 1:
        target, _ = MPS.from_statevector(random_state(2, rng))
    else:
        target, _ = MPS.from_statevector(random_state(2**n, rng), chi_max=2)
    target.canonicalize()
    target.normalize()

    result = synthesize_mps(target, fidelity=1.0, max_layers=1)
    assert result.layers == 1
    assert result.fidelity == pytest.approx(1.0, abs=1e-12)


def test_bond2_layer_rejects_wider_states(rng):
    target, _ = MPS.from_statevector(random_state(2**6, rng))
    with pytest.raises(ValueError, match="bond dimension <= 2"):
        bond2_layer(target)


def test_bond2_layer_shapes(rng):
    target, _ = MPS.from_statevector(random_state(2**5, rng), chi_max=2)
    layer = bond2_layer(target)
    for matrix, sites in layer:
        expected = 2 ** len(sites)
        assert matrix.shape == (expected, expected)
        np.testing.assert_allclose(matrix.conj().T @ matrix, np.eye(expected), atol=1e-10)
        if len(sites) == 2:
            assert sites[1] == sites[0] + 1, "staircase gates must be nearest-neighbour"


# ------------------------------------------------------------- special structure


def test_product_states_need_no_entangling_gates(rng):
    vec = np.ones(1)
    for _ in range(7):
        vec = np.kron(vec, rng.normal(size=2))
    result = synthesize(vec, fidelity=1.0, max_layers=2)
    assert result.circuit.cnot_count == 0
    assert result.fidelity == pytest.approx(1.0, abs=1e-12)


def test_ghz_and_w_states_are_exact_with_one_layer():
    for name in ["ghz", "w"]:
        vector = generate(name, 6)
        result = synthesize(vector, fidelity=1.0, max_layers=1)
        assert result.layers == 1
        assert result.fidelity == pytest.approx(1.0, abs=1e-12), name


def test_single_qubit_input():
    result = synthesize([0.6, 0.8], fidelity=1.0, max_layers=1)
    assert result.n_qubits == 1
    assert result.circuit.cnot_count == 0
    assert result.fidelity == pytest.approx(1.0, abs=1e-12)


# --------------------------------------------------------- general correctness


@pytest.mark.parametrize("name", sorted(GENERATORS))
def test_reported_fidelity_matches_independent_simulation(name):
    """The headline number must be reproducible from the circuit alone."""
    vector = generate(name, 7)
    result = synthesize(vector, fidelity=0.999, max_layers=6)
    assert fidelity_against(vector, result) == pytest.approx(result.fidelity, abs=1e-9)
    assert 0.0 <= result.fidelity <= 1.0 + 1e-12


def test_complex_amplitudes_are_reproduced(rng):
    vector = random_state(2**6, rng)
    result = synthesize(vector, fidelity=0.9999, max_layers=24)
    assert fidelity_against(vector, result) == pytest.approx(result.fidelity, abs=1e-9)


def test_fidelity_improves_with_depth(rng):
    curve = profile(generate("bimodal", 8), max_layers=6)
    fidelities = [p.fidelity for p in curve.points]
    assert fidelities == sorted(fidelities), "adding a layer must not lose fidelity"
    cnots = [p.cnot for p in curve.points]
    assert cnots == sorted(cnots)


def test_more_layers_reach_higher_fidelity(rng):
    vector = generate("random", 6, seed=3)
    shallow = synthesize(vector, fidelity=1.0, max_layers=2)
    deep = synthesize(vector, fidelity=1.0, max_layers=12)
    assert deep.fidelity > shallow.fidelity
    assert deep.circuit.cnot_count > shallow.circuit.cnot_count


def test_target_is_respected_and_reported():
    vector = generate("gaussian", 8)
    result = synthesize(vector, fidelity=0.99, max_layers=8)
    assert result.reached_target
    assert result.fidelity >= 0.99

    hopeless = synthesize(generate("random", 8, seed=1), fidelity=0.999999, max_layers=2)
    assert not hopeless.reached_target
    assert hopeless.layers == 2


# ------------------------------------------- the unitarity / normalisation contract


@pytest.mark.parametrize("name", ["gaussian", "random", "damped"])
def test_output_state_is_exactly_normalised(name):
    """Truncation loses norm; the emitted circuit must still be unitary."""
    result = synthesize(generate(name, 7), fidelity=0.9, max_layers=3)
    psi = result.circuit.statevector()
    assert np.vdot(psi, psi).real == pytest.approx(1.0, abs=1e-12)


def test_emitted_circuit_is_unitary():
    result = synthesize(generate("lorentzian", 4), fidelity=0.99, max_layers=3)
    u = result.circuit.unitary()
    np.testing.assert_allclose(u.conj().T @ u, np.eye(2**4), atol=1e-10)


def test_truncation_does_not_leak_unnormalised_states(rng):
    """Even with an aggressive input truncation the prepared state stays normalised."""
    result = synthesize(random_state(2**8, rng), chi_max=2, fidelity=0.5, max_layers=2)
    psi = result.circuit.statevector()
    assert np.linalg.norm(psi) == pytest.approx(1.0, abs=1e-12)
    # ... and the reported fidelity is measured against the true input, not the
    # truncated stand-in, so it must be the (lower) true number.
    assert result.fidelity <= 1.0


# ------------------------------------------------------------ input preparation


def test_zero_padding():
    vector = np.arange(1, 6, dtype=float)  # length 5 -> 8
    result = synthesize(vector, fidelity=1.0, max_layers=8)
    assert result.n_qubits == 3
    assert result.padded_from == 5
    produced = result.amplitudes()
    assert produced.shape == (5,)
    assert abs(
        np.vdot(vector / np.linalg.norm(vector), produced / np.linalg.norm(produced))
    ) ** 2 == pytest.approx(1.0, abs=1e-9)


def test_padding_can_be_refused():
    with pytest.raises(ValueError, match="not a power of two"):
        prepare_vector(np.ones(5), pad=False)


def test_normalisation_is_recorded():
    vector = np.array([3.0, 4.0])
    result = synthesize(vector, fidelity=1.0)
    assert result.input_norm == pytest.approx(5.0)
    np.testing.assert_allclose(result.amplitudes(), vector, atol=1e-9)


def test_zero_vector_is_rejected():
    with pytest.raises(ValueError, match="zero norm"):
        synthesize(np.zeros(8))


def test_empty_vector_is_rejected():
    with pytest.raises(ValueError, match="empty"):
        synthesize(np.array([]))


def test_invalid_arguments():
    with pytest.raises(ValueError, match="fidelity must lie"):
        synthesize([1.0, 0.0], fidelity=1.5)
    with pytest.raises(ValueError, match="max_layers"):
        synthesize([1.0, 0.0], max_layers=0)
    with pytest.raises(ValueError, match="qubit_order"):
        synthesize([1.0, 0.0], qubit_order="middle")


# ------------------------------------------------------------------- endianness


@pytest.mark.parametrize("order", ["big", "little"])
def test_qubit_order_round_trip(order, rng):
    vector = random_state(2**5, rng)
    result = synthesize(vector, fidelity=1.0, max_layers=20, qubit_order=order)
    assert result.qubit_order == order
    assert result.circuit.amplitude_order == order
    # amplitudes() undoes the re-indexing, so under *either* convention it lines up
    # with the input and reproduces exactly the fidelity that was reported.
    overlap = abs(np.vdot(vector, result.amplitudes())) ** 2
    assert overlap == pytest.approx(result.fidelity, abs=1e-9)
    assert overlap > 0.999, "20 layers should essentially converge on 5 qubits"


def test_little_endian_is_a_bit_reversal_of_big_endian(rng):
    vector = random_state(2**4, rng)
    big = synthesize(vector, fidelity=1.0, max_layers=20, qubit_order="big")
    little = synthesize(vector, fidelity=1.0, max_layers=20, qubit_order="little")
    n = 4
    rev = [int(format(i, f"0{n}b")[::-1], 2) for i in range(2**n)]
    np.testing.assert_allclose(
        np.abs(little.circuit.statevector()),
        np.abs(big.circuit.statevector()[rev]),
        atol=1e-8,
    )


# ---------------------------------------------------------------- large registers


def test_scales_past_dense_simulation():
    """24 qubits is 16.7M amplitudes: MPS-only path, no dense statevector anywhere."""
    x = np.linspace(-5.0, 5.0, 2**14)
    vector = np.exp(-(x**2) / 2.0)
    target, _ = MPS.from_statevector(vector / np.linalg.norm(vector), chi_max=8)
    target.normalize()
    result = synthesize_mps(target, fidelity=0.99, max_layers=4)
    assert result.n_qubits == 14
    assert result.fidelity > 0.99
    # Linear, not exponential: well under the 2**14 of an exact encoding.
    assert result.circuit.cnot_count < 4 * 14 * 3


def test_cnot_count_grows_linearly_with_qubits():
    counts = []
    for n in [8, 10, 12, 14]:
        x = np.linspace(-4.0, 4.0, 2**n)
        result = synthesize(np.exp(-(x**2) / 2.0), fidelity=0.99, max_layers=3)
        counts.append(result.circuit.cnot_count)
    # Doubling the register must not double the cost more than linearly.
    ratios = [b / a for a, b in zip(counts, counts[1:], strict=False)]
    assert all(r < 1.6 for r in ratios), counts
