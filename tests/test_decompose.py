"""Tests for the one- and two-qubit decompositions.

The canonical-gate identities are re-derived here from scratch rather than trusted:
if any of them were wrong, every synthesised circuit would be wrong.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import random_unitary

from mpsynth.decompose import (
    MAGIC,
    _ops_matrix,
    canonical_gate,
    canonical_ops,
    kak_decomposition,
    two_qubit_ops,
)
from mpsynth.gates import (
    CX01,
    I2,
    SWAP,
    H,
    X,
    Y,
    Z,
    rx,
    ry,
    rz,
    u_zyz,
    wrap_angle,
    zyz_decomposition,
)

kron = np.kron


def n_cx(ops) -> int:
    return sum(1 for op in ops if op[0] == "cx")


# ------------------------------------------------------------------- single qubit


def test_zyz_reconstructs_random_unitaries(rng):
    for _ in range(200):
        u = random_unitary(2, rng)
        phase, alpha, beta, gamma = zyz_decomposition(u)
        np.testing.assert_allclose(u_zyz(alpha, beta, gamma, phase), u, atol=1e-12)
        assert -1e-12 <= beta <= np.pi + 1e-12


@pytest.mark.parametrize(
    "matrix",
    [np.eye(2), X, Y, Z, H, rz(0.7), ry(-1.3), rx(2.1), np.diag([1, 1j])],
)
def test_zyz_handles_degenerate_beta(matrix):
    phase, alpha, beta, gamma = zyz_decomposition(np.asarray(matrix, dtype=complex))
    np.testing.assert_allclose(u_zyz(alpha, beta, gamma, phase), matrix, atol=1e-12)


def test_wrap_angle_preserves_the_rotation_up_to_phase():
    for theta in [0.0, 3.0, -3.0, 7.0, -7.0, 2 * np.pi, -2 * np.pi, 100.0]:
        wrapped, phase = wrap_angle(theta)
        assert -np.pi < wrapped <= np.pi + 1e-12
        np.testing.assert_allclose(rz(theta), np.exp(1j * phase) * rz(wrapped), atol=1e-12)
        np.testing.assert_allclose(ry(theta), np.exp(1j * phase) * ry(wrapped), atol=1e-12)


# --------------------------------------------------- the Bell-basis identities used


def test_bell_transform_diagonalises_the_canonical_generators():
    """B = CX (H (x) I) is what makes the canonical constructions work."""
    b = CX01 @ kron(H, I2)
    np.testing.assert_allclose(b.conj().T @ kron(X, X) @ b, kron(Z, I2), atol=1e-12)
    np.testing.assert_allclose(b.conj().T @ kron(Y, Y) @ b, -kron(Z, Z), atol=1e-12)
    np.testing.assert_allclose(b.conj().T @ kron(Z, Z) @ b, kron(I2, Z), atol=1e-12)


def test_magic_basis_maps_local_gates_to_so4(rng):
    for _ in range(50):
        a, b = random_unitary(2, rng), random_unitary(2, rng)
        a, b = a / np.sqrt(np.linalg.det(a)), b / np.sqrt(np.linalg.det(b))
        m = MAGIC.conj().T @ kron(a, b) @ MAGIC
        np.testing.assert_allclose(m.imag, 0.0, atol=1e-10)
        np.testing.assert_allclose(m @ m.T, np.eye(4), atol=1e-10)


def test_canonical_gate_matches_the_matrix_exponential(rng):
    """canonical_gate() is built from eigenphases; check against a direct expm."""
    scipy_linalg = pytest.importorskip("scipy.linalg")
    for _ in range(50):
        a, b, c = rng.uniform(-np.pi, np.pi, 3)
        generator = a * kron(X, X) + b * kron(Y, Y) + c * kron(Z, Z)
        np.testing.assert_allclose(
            canonical_gate(a, b, c), scipy_linalg.expm(1j * generator), atol=1e-10
        )


# ---------------------------------------------------------------- canonical gates


def test_canonical_ops_reproduce_n_exactly(rng):
    for _ in range(300):
        a, b, c = rng.uniform(-np.pi, np.pi, 3)
        ops, phase = canonical_ops(a, b, c)
        np.testing.assert_allclose(
            np.exp(1j * phase) * _ops_matrix(ops), canonical_gate(a, b, c), atol=1e-10
        )
        assert n_cx(ops) <= 3


def test_canonical_ops_use_cheap_constructions_when_available():
    assert n_cx(canonical_ops(0.0, 0.0, 0.0)[0]) == 0
    assert n_cx(canonical_ops(np.pi / 4, 0.0, 0.0)[0]) == 1
    assert n_cx(canonical_ops(0.31, 0.17, 0.0)[0]) == 2
    assert n_cx(canonical_ops(0.31, 0.17, 0.09)[0]) == 3


# -------------------------------------------------------------------------- KAK


def test_kak_reconstructs_random_unitaries(rng):
    for _ in range(200):
        u = random_unitary(4, rng)
        np.testing.assert_allclose(kak_decomposition(u).matrix(), u, atol=1e-9)


def test_kak_lands_in_the_weyl_chamber(rng):
    for _ in range(200):
        a, b, c = kak_decomposition(random_unitary(4, rng)).coefficients
        assert a <= np.pi / 4 + 1e-7
        assert a >= b - 1e-7
        assert b >= abs(c) - 1e-7
        assert b >= -1e-7


def test_kak_factors_are_local_unitaries(rng):
    for _ in range(50):
        k = kak_decomposition(random_unitary(4, rng))
        for factor in (k.k1a, k.k1b, k.k2a, k.k2b):
            np.testing.assert_allclose(factor.conj().T @ factor, np.eye(2), atol=1e-9)


def test_kak_rejects_non_unitary_input():
    with pytest.raises(ValueError):
        kak_decomposition(np.ones((4, 4)))


# ---------------------------------------------------- full two-qubit gate lowering


def test_two_qubit_ops_reproduce_random_unitaries(rng):
    for _ in range(200):
        u = random_unitary(4, rng)
        ops, phase = two_qubit_ops(u)
        np.testing.assert_allclose(np.exp(1j * phase) * _ops_matrix(ops), u, atol=1e-9)
        assert n_cx(ops) == 3, "a generic U(4) needs exactly three CNOTs"


def test_two_qubit_ops_hit_the_optimal_cnot_count_for_known_classes(rng):
    a, b = random_unitary(2, rng), random_unitary(2, rng)
    iswap = np.array([[1, 0, 0, 0], [0, 0, 1j, 0], [0, 1j, 0, 0], [0, 0, 0, 1]], dtype=complex)
    w, v = np.linalg.eigh(SWAP)
    sqrt_swap = (v * np.exp(0.5j * np.angle(w.astype(complex)))) @ v.conj().T

    expected = {
        0: [np.eye(4, dtype=complex), kron(a, b)],
        1: [CX01, np.diag([1, 1, 1, -1]).astype(complex)],
        2: [iswap, canonical_gate(0.3, 0.2, 0.0)],
        3: [SWAP, sqrt_swap],
    }
    for count, matrices in expected.items():
        for matrix in matrices:
            ops, phase = two_qubit_ops(matrix)
            np.testing.assert_allclose(np.exp(1j * phase) * _ops_matrix(ops), matrix, atol=1e-9)
            assert n_cx(ops) == count


def test_two_qubit_ops_survive_nearly_degenerate_inputs(rng):
    """Small perturbations of local gates stress the eigenvector branch selection."""
    for scale in [1e-1, 1e-3, 1e-6, 1e-9]:
        for _ in range(20):
            local = kron(random_unitary(2, rng), random_unitary(2, rng))
            perturbation = canonical_gate(*(scale * rng.normal(size=3)))
            u = local @ perturbation
            ops, phase = two_qubit_ops(u)
            np.testing.assert_allclose(np.exp(1j * phase) * _ops_matrix(ops), u, atol=1e-8)
