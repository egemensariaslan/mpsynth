"""Tests for the MPS / tensor-train core."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import random_state, random_unitary

from mpsynth.gates import CX01, ry, rz
from mpsynth.linalg import complete_to_unitary, truncated_svd
from mpsynth.mps import MPS


def dense_apply_2q(psi, gate, q0, q1, n):
    t = psi.reshape((2,) * n)
    op = gate.reshape(2, 2, 2, 2)
    t = np.moveaxis(np.tensordot(op, t, axes=([2, 3], [q0, q1])), [0, 1], [q0, q1])
    return t.reshape(-1)


# --------------------------------------------------------------- decomposition


@pytest.mark.parametrize("n", [1, 2, 3, 5, 8])
def test_exact_roundtrip(n, rng):
    psi = random_state(2**n, rng)
    mps, discarded = MPS.from_statevector(psi)
    np.testing.assert_allclose(mps.to_statevector(), psi, atol=1e-12)
    assert discarded == pytest.approx(0.0, abs=1e-12)


def test_bond_dimensions_follow_the_schmidt_rank(rng):
    n = 8
    mps, _ = MPS.from_statevector(random_state(2**n, rng))
    expected = [min(2**k, 2 ** (n - k)) for k in range(1, n)]
    assert mps.bond_dimensions() == expected


def test_product_state_has_bond_dimension_one(rng):
    vec = np.ones(1)
    for _ in range(6):
        block = random_state(2, rng)
        vec = np.kron(vec, block)
    mps, _ = MPS.from_statevector(vec)
    assert mps.max_bond() == 1


def test_rejects_non_power_of_two():
    with pytest.raises(ValueError, match="power of two"):
        MPS.from_statevector(np.ones(6))


# ----------------------------------------------------------------- canonical form


def test_moving_the_centre_preserves_the_state(rng):
    n = 7
    psi = random_state(2**n, rng)
    mps, _ = MPS.from_statevector(psi)
    for target in [0, n - 1, 3, 5, 0, n - 1]:
        mps.move_center(target)
        assert mps.center == target
        np.testing.assert_allclose(mps.to_statevector(), psi, atol=1e-11)


def test_canonicalize_is_lossless_without_truncation(rng):
    psi = random_state(2**6, rng)
    mps, _ = MPS.from_statevector(psi)
    discarded = mps.canonicalize()
    assert discarded == pytest.approx(0.0, abs=1e-12)
    np.testing.assert_allclose(mps.to_statevector(), psi, atol=1e-11)


def test_right_canonical_tensors_are_isometries(rng):
    mps, _ = MPS.from_statevector(random_state(2**6, rng))
    mps.canonicalize()
    mps.normalize()
    for k, t in enumerate(mps.tensors):
        dl, _, dr = t.shape
        mat = t.reshape(dl, 2 * dr)
        np.testing.assert_allclose(
            mat @ mat.conj().T, np.eye(dl), atol=1e-10, err_msg=f"site {k} is not right-canonical"
        )


def test_move_center_from_unknown_gauge(rng):
    psi = random_state(2**5, rng)
    mps, _ = MPS.from_statevector(psi)
    mps.center = None
    mps.move_center(2)
    np.testing.assert_allclose(mps.to_statevector(), psi, atol=1e-11)


# --------------------------------------------------------------------- truncation


def test_truncation_reduces_bond_and_reports_lost_weight(rng):
    n = 8
    psi = random_state(2**n, rng)
    exact, _ = MPS.from_statevector(psi)
    approx, discarded = MPS.from_statevector(psi, chi_max=4)
    assert approx.max_bond() <= 4
    assert 0.0 < discarded < 1.0
    approx.normalize()
    # The discarded weight bounds the infidelity that truncation can cause.
    assert 1.0 - approx.fidelity(exact) <= discarded + 1e-9


def test_truncation_is_exact_for_a_low_rank_state(rng):
    """A state that genuinely has bond dimension 2 must survive chi_max=2 untouched."""
    n = 6
    exact, _ = MPS.from_statevector(random_state(2**n, rng), chi_max=2)
    exact.normalize()
    again, discarded = MPS.from_statevector(exact.to_statevector(), chi_max=2)
    assert discarded == pytest.approx(0.0, abs=1e-12)
    again.normalize()
    assert again.fidelity(exact) == pytest.approx(1.0, abs=1e-12)


def test_tolerance_based_truncation():
    """A smooth vector has fast-decaying singular values, so `tol` should bite."""
    x = np.linspace(-4.0, 4.0, 2**8)
    psi = np.exp(-(x**2) / 2.0)
    psi /= np.linalg.norm(psi)

    exact, _ = MPS.from_statevector(psi)
    loose, discarded = MPS.from_statevector(psi, tol=1e-4)
    assert loose.max_bond() < exact.max_bond()
    assert discarded > 0
    loose.normalize()
    assert 1.0 - loose.fidelity(exact) <= discarded + 1e-9

    tight, tight_discarded = MPS.from_statevector(psi, tol=1e-12)
    assert tight.max_bond() >= loose.max_bond()
    assert tight_discarded <= discarded


def test_normalize_restores_unit_norm(rng):
    approx, _ = MPS.from_statevector(random_state(2**7, rng), chi_max=3)
    assert approx.norm() < 1.0  # truncation lost weight
    approx.normalize()
    assert approx.norm() == pytest.approx(1.0, abs=1e-12)
    assert abs(approx.overlap(approx)) == pytest.approx(1.0, abs=1e-12)


# ------------------------------------------------------------------ gate actions


def test_two_qubit_gate_matches_dense_simulation(rng):
    n = 7
    psi = random_state(2**n, rng)
    gate = random_unitary(4, rng)
    for site in range(n - 1):
        mps, _ = MPS.from_statevector(psi)
        mps.apply_2q(gate, site)
        np.testing.assert_allclose(
            mps.to_statevector(), dense_apply_2q(psi, gate, site, site + 1, n), atol=1e-11
        )


def test_one_qubit_gate_matches_dense_simulation(rng):
    n = 6
    psi = random_state(2**n, rng)
    gate = ry(0.83) @ rz(-1.2)
    for site in range(n):
        mps, _ = MPS.from_statevector(psi)
        mps.apply_1q(gate, site)
        t = psi.reshape((2,) * n)
        t = np.moveaxis(np.tensordot(gate, t, axes=([1], [site])), 0, site)
        np.testing.assert_allclose(mps.to_statevector(), t.reshape(-1), atol=1e-11)


def test_one_qubit_gate_preserves_canonical_form(rng):
    mps, _ = MPS.from_statevector(random_state(2**5, rng))
    mps.canonicalize()
    center = mps.center
    mps.apply_1q(ry(0.4), 3)
    assert mps.center == center
    for t in mps.tensors[1:]:
        dl, _, dr = t.shape
        mat = t.reshape(dl, 2 * dr)
        np.testing.assert_allclose(mat @ mat.conj().T, np.eye(dl), atol=1e-10)


def test_applying_a_gate_then_its_inverse_is_identity(rng):
    psi = random_state(2**6, rng)
    mps, _ = MPS.from_statevector(psi)
    gate = np.kron(ry(0.4), rz(1.1)) @ CX01
    mps.apply_2q(gate, 2)
    mps.apply_2q(gate.conj().T, 2)
    np.testing.assert_allclose(mps.to_statevector(), psi, atol=1e-11)


# ------------------------------------------------------------------------ scalars


def test_overlap_matches_dense_inner_product(rng):
    a, b = random_state(2**6, rng), random_state(2**6, rng)
    ma, _ = MPS.from_statevector(a)
    mb, _ = MPS.from_statevector(b)
    assert ma.overlap(mb) == pytest.approx(np.vdot(a, b), abs=1e-11)


def test_amplitude_matches_the_dense_entry(rng):
    n = 6
    psi = random_state(2**n, rng)
    mps, _ = MPS.from_statevector(psi)
    for index in [0, 1, 17, 2**n - 1]:
        bits = format(index, f"0{n}b")
        assert mps.amplitude(bits) == pytest.approx(psi[index], abs=1e-12)


def test_zero_state():
    mps = MPS.zero_state(5)
    assert mps.max_bond() == 1
    expected = np.zeros(32)
    expected[0] = 1.0
    np.testing.assert_allclose(mps.to_statevector(), expected, atol=1e-15)


def test_structural_validation():
    with pytest.raises(ValueError, match="at least one site"):
        MPS([])
    with pytest.raises(ValueError, match="shape"):
        MPS([np.zeros((1, 3, 1))])
    with pytest.raises(ValueError, match="boundary"):
        MPS([np.zeros((2, 2, 1))])
    with pytest.raises(ValueError, match="bond mismatch"):
        MPS([np.zeros((1, 2, 3)), np.zeros((2, 2, 1))])


# --------------------------------------------------------------- linalg helpers


def test_truncated_svd_reports_the_relative_discarded_weight(rng):
    mat = rng.normal(size=(16, 16)) + 1j * rng.normal(size=(16, 16))
    s = np.linalg.svd(mat, compute_uv=False)
    for keep in [1, 4, 8, 16]:
        res = truncated_svd(mat, chi_max=keep)
        assert len(res.s) == keep
        expected = np.sum(s[keep:] ** 2) / np.sum(s**2)
        assert res.discarded_weight == pytest.approx(expected, abs=1e-12)


def test_truncated_svd_keeps_at_least_one_value(rng):
    res = truncated_svd(rng.normal(size=(8, 8)), tol=1.0)
    assert len(res.s) >= 1


def test_complete_to_unitary(rng):
    for m, k in [(4, 1), (4, 2), (4, 3), (4, 4), (2, 1)]:
        iso = np.linalg.qr(rng.normal(size=(m, m)) + 1j * rng.normal(size=(m, m)))[0][:, :k]
        full = complete_to_unitary(iso)
        np.testing.assert_allclose(full[:, :k], iso, atol=1e-12)
        np.testing.assert_allclose(full.conj().T @ full, np.eye(m), atol=1e-12)


def test_complete_to_unitary_rejects_wide_input():
    with pytest.raises(ValueError, match="more columns than rows"):
        complete_to_unitary(np.zeros((2, 4)))
