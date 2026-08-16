"""Elementary gate matrices and single-qubit Euler decomposition.

Rotation conventions match OpenQASM, Qiskit, PennyLane and Q#::

    RZ(t) = exp(-i t Z / 2)     RY(t) = exp(-i t Y / 2)     RX(t) = exp(-i t X / 2)

Two-qubit matrices use :func:`numpy.kron` ordering: for a gate on ``(qa, qb)`` the
row/column index is ``2 * s_qa + s_qb``, i.e. ``qa`` is the more significant bit.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "I2", "X", "Y", "Z", "H", "S", "SDG", "CX01", "CX10", "SWAP",
    "rz", "ry", "rx", "u_zyz", "zyz_decomposition", "wrap_angle",
]

I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
S = np.array([[1, 0], [0, 1j]], dtype=complex)
SDG = S.conj().T

CX01 = np.array(
    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex
)
CX10 = np.array(
    [[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]], dtype=complex
)
SWAP = np.array(
    [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=complex
)


def rz(theta: float) -> np.ndarray:
    """``exp(-i theta Z / 2)``."""
    return np.array(
        [[np.exp(-0.5j * theta), 0.0], [0.0, np.exp(0.5j * theta)]], dtype=complex
    )


def ry(theta: float) -> np.ndarray:
    """``exp(-i theta Y / 2)``."""
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)


def rx(theta: float) -> np.ndarray:
    """``exp(-i theta X / 2)``."""
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)


def u_zyz(alpha: float, beta: float, gamma: float, phase: float = 0.0) -> np.ndarray:
    """``exp(i phase) RZ(alpha) RY(beta) RZ(gamma)``."""
    return np.exp(1j * phase) * (rz(alpha) @ ry(beta) @ rz(gamma))


def zyz_decomposition(u: np.ndarray, atol: float = 1e-12) -> tuple[float, float, float, float]:
    """Factor a 2x2 unitary as ``exp(i phase) RZ(alpha) RY(beta) RZ(gamma)``.

    Returns ``(phase, alpha, beta, gamma)`` with ``beta`` in ``[0, pi]``.  Degenerate
    cases (``beta == 0`` or ``beta == pi``) only fix ``alpha ± gamma``; the free
    parameter is resolved by setting ``gamma = 0``.
    """
    u = np.asarray(u, dtype=complex)
    if u.shape != (2, 2):
        raise ValueError(f"expected a 2x2 matrix, got {u.shape}")

    det = u[0, 0] * u[1, 1] - u[0, 1] * u[1, 0]
    if abs(det) < atol:
        raise ValueError("matrix is singular, cannot be unitary")
    phase = float(np.angle(det) / 2.0)
    v = u * np.exp(-1j * phase)  # now in SU(2)

    cos_half = abs(v[0, 0])
    sin_half = abs(v[1, 0])
    beta = float(2.0 * np.arctan2(sin_half, cos_half))

    if cos_half > 1e-9 and sin_half > 1e-9:
        plus = float(np.angle(v[1, 1]))   # (alpha + gamma) / 2
        minus = float(np.angle(v[1, 0]))  # (alpha - gamma) / 2
        alpha, gamma = plus + minus, plus - minus
    elif cos_half > 1e-9:  # beta == 0, only alpha + gamma matters
        alpha, gamma = float(2.0 * np.angle(v[1, 1])), 0.0
    else:  # beta == pi, only alpha - gamma matters
        alpha, gamma = float(2.0 * np.angle(v[1, 0])), 0.0

    return phase, alpha, beta, gamma


def wrap_angle(theta: float) -> tuple[float, float]:
    """Fold a rotation angle into ``(-pi, pi]``.

    Returns ``(wrapped, phase)`` such that ``R(theta) == exp(i phase) R(wrapped)``
    for any of RX/RY/RZ, since ``R(theta + 2 pi) == -R(theta)``.
    """
    turns = np.floor((theta + np.pi) / (2 * np.pi))
    wrapped = theta - 2 * np.pi * turns
    if wrapped <= -np.pi:  # keep the half-open convention exact at the boundary
        wrapped += 2 * np.pi
        turns -= 1
    phase = np.pi * turns  # (-1)**turns
    return float(wrapped), float(phase)
