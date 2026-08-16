"""Exact decomposition of arbitrary one- and two-qubit unitaries into {RZ, RY, CX}.

The two-qubit routine is a Cartan (KAK) decomposition::

    U = exp(i phi) (k1a (x) k1b) . N(a, b, c) . (k2a (x) k2b)
    N(a, b, c)  =  exp(i (a XX + b YY + c ZZ))

The canonical factor ``N(a, b, c)`` is emitted with the fewest CX gates its Weyl
chamber coordinates allow: 0 for local gates, 1 for CX-equivalent gates, 2 when one
coordinate vanishes, and 3 in general -- which is optimal for a generic U(4).

Every construction in this module was derived from the Bell-basis diagonalisation of
{XX, YY, ZZ} and is checked numerically against the target before being returned, so
a synthesis run can never silently emit a wrong circuit.
"""

from __future__ import annotations

import numpy as np

from .gates import CX01, H, I2, S, SDG, Y, Z, rx, rz
from .linalg import is_product, split_product

__all__ = [
    "KakDecomposition",
    "kak_decomposition",
    "canonical_gate",
    "canonical_ops",
    "two_qubit_ops",
]

# Op stream shared by every construction here.  Entries are either
#   ("1q", local_qubit, 2x2 matrix)   or   ("cx", control, target)
# listed in *time* order (first element applied first).
Op = tuple

_PI_4 = np.pi / 4

# Magic basis: columns are the Bell states with phases chosen so that
# M^dagger (SU(2) (x) SU(2)) M == SO(4) and M^dagger N(a,b,c) M is diagonal.
MAGIC = np.array(
    [
        [1, 1j, 0, 0],
        [0, 0, 1j, 1],
        [0, 0, 1j, -1],
        [1, -1j, 0, 0],
    ],
    dtype=complex,
) / np.sqrt(2)

# V swaps the Y and Z axes (and flips X), mapping YY <-> ZZ under conjugation.
_V = (Y + Z) / np.sqrt(2)


def canonical_gate(a: float, b: float, c: float) -> np.ndarray:
    """``exp(i (a XX + b YY + c ZZ))`` built from its Bell-basis eigenphases."""
    phases = np.exp(1j * np.array([a - b + c, -a + b + c, a + b - c, -a - b - c]))
    return MAGIC @ np.diag(phases) @ MAGIC.conj().T


# --------------------------------------------------------------------- KAK core


class KakDecomposition:
    """Result of :func:`kak_decomposition`."""

    __slots__ = ("global_phase", "k1a", "k1b", "coefficients", "k2a", "k2b")

    def __init__(self, global_phase, k1a, k1b, coefficients, k2a, k2b):
        self.global_phase = float(global_phase)
        self.k1a = k1a
        self.k1b = k1b
        self.coefficients = tuple(float(x) for x in coefficients)
        self.k2a = k2a
        self.k2b = k2b

    def matrix(self) -> np.ndarray:
        """Reassemble the original unitary."""
        return (
            np.exp(1j * self.global_phase)
            * np.kron(self.k1a, self.k1b)
            @ canonical_gate(*self.coefficients)
            @ np.kron(self.k2a, self.k2b)
        )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        a, b, c = self.coefficients
        return f"KakDecomposition(a={a:.6f}, b={b:.6f}, c={c:.6f})"


def _simultaneous_real_eigenvectors(g: np.ndarray, atol: float = 1e-8) -> np.ndarray:
    """Real orthogonal ``P`` diagonalising a complex *symmetric unitary* ``g``.

    For symmetric unitary ``g = A + iB`` the real symmetric parts ``A`` and ``B``
    commute, so a single random real combination ``A + lambda B`` generically has
    a non-degenerate spectrum whose eigenvectors diagonalise both.
    """
    re, im = np.real(g), np.imag(g)
    rng = np.random.default_rng(0xC0FFEE)
    for lam in [0.0, 1.0, *rng.uniform(-3.0, 3.0, 24)]:
        _, p = np.linalg.eigh(re + lam * im)
        off = p.T @ g @ p
        off = off - np.diag(np.diag(off))
        if np.max(np.abs(off)) < atol:
            return p
    raise np.linalg.LinAlgError(
        "failed to simultaneously diagonalise the KAK Gram matrix"
    )


def _coefficients_from_phases(theta: np.ndarray) -> tuple[float, float, float]:
    """Invert ``diag(N) = exp(i[a-b+c, -a+b+c, a+b-c, -a-b-c])``."""
    a = (theta[0] + theta[2]) / 2.0
    b = (theta[1] + theta[2]) / 2.0
    c = (theta[0] + theta[1]) / 2.0
    return float(a), float(b), float(c)


def _chamber_penalty(a, b, c):
    """How far ``(a, b, c)`` is from the canonical Weyl chamber pi/4 >= a >= b >= |c|.

    Vectorised over numpy arrays; zero exactly inside the chamber, and the trailing
    ``a + b + |c|`` term breaks ties toward the least entangling representative.
    """
    eps = 1e-9
    zero = np.zeros_like(np.asarray(a, dtype=float))
    pen = zero.copy()
    for x in (a, b, c):
        pen = pen + 10.0 * np.maximum(0.0, np.abs(x) - _PI_4 - eps)
    pen = pen + 10.0 * (np.maximum(0.0, -a) + np.maximum(0.0, -b))
    pen = pen + 10.0 * (np.maximum(0.0, b - a) + np.maximum(0.0, np.abs(c) - b))
    return pen + a + b + np.abs(c)


_PERMUTATIONS = [
    (i, j, k, l)
    for i in range(4)
    for j in range(4)
    for k in range(4)
    for l in range(4)
    if len({i, j, k, l}) == 4
]
#: Sign patterns with product +1.  A sign flip shifts that eigenphase by pi; flips must
#: pair up so that det(A) stays +1 and K1 stays in SO(4).
_EVEN_SIGNS = [
    s
    for s in [
        (a, b, c, d)
        for a in (1, -1)
        for b in (1, -1)
        for c in (1, -1)
        for d in (1, -1)
    ]
    if s[0] * s[1] * s[2] * s[3] == 1
]

_PERM_INDEX = np.array(_PERMUTATIONS, dtype=int)                      # (24, 4)
_SIGN_SHIFT = np.where(np.array(_EVEN_SIGNS) < 0, np.pi, 0.0)          # (8, 4)
_PERM_PARITY = np.array(
    [
        (-1) ** sum(1 for i in range(4) for j in range(i + 1, 4) if p[i] > p[j])
        for p in _PERMUTATIONS
    ]
)
#: The fourth root in `det(u)**(1/4)` has four branches.  Branch `k` multiplies every
#: eigenphase by `exp(-i k pi/2)` and adds `k pi/2` to the global phase, leaving K1 and
#: K2 untouched -- and it is the move that lets `(a, b, c)` reach the Weyl chamber.
_BRANCHES = np.arange(4)


def kak_decomposition(u: np.ndarray, atol: float = 1e-8) -> KakDecomposition:
    """Cartan decomposition of a 4x4 unitary, reduced to the Weyl chamber."""
    u = np.asarray(u, dtype=complex)
    if u.shape != (4, 4):
        raise ValueError(f"expected a 4x4 matrix, got {u.shape}")

    det = np.linalg.det(u)
    if abs(abs(det) - 1.0) > 1e-6:
        raise ValueError("kak_decomposition requires a unitary matrix")
    base_phase = float(np.angle(det) / 4.0)
    su = u * np.exp(-1j * base_phase)  # det == 1

    um = MAGIC.conj().T @ su @ MAGIC
    gram = um.T @ um  # complex symmetric and unitary
    p0 = _simultaneous_real_eigenvectors(gram)
    if np.linalg.det(p0) < 0:
        p0 = p0.copy()
        p0[:, 0] *= -1.0

    d2 = np.diag(p0.T @ gram @ p0)
    d0 = np.sqrt(d2.astype(complex))
    if np.real(np.prod(d0)) < 0:  # det(A) must be +1 to keep K1 in SO(4)
        d0 = d0.copy()
        d0[0] *= -1.0

    # Search the residual gauge freedom -- column permutations, paired sign flips, and
    # the four branches of det**(1/4) -- for the representative in the Weyl chamber.
    # Only the branch index touches the global phase; K1 and K2 are the same for all.
    theta0 = np.angle(d0)
    theta = (
        theta0[_PERM_INDEX][:, None, None, :]          # (24, 1, 1, 4) permutations
        + _SIGN_SHIFT[None, :, None, :]                # ( 1, 8, 1, 4) pi shifts
        - (_BRANCHES * (np.pi / 2))[None, None, :, None]  # (1, 1, 4, 1) phase branch
    )
    a = (theta[..., 0] + theta[..., 2]) / 2.0
    b = (theta[..., 1] + theta[..., 2]) / 2.0
    c = (theta[..., 0] + theta[..., 1]) / 2.0
    scores = _chamber_penalty(a, b, c)
    pi, si, bi = np.unravel_index(int(np.argmin(scores)), scores.shape)

    coeffs = (float(a[pi, si, bi]), float(b[pi, si, bi]), float(c[pi, si, bi]))
    global_phase = base_phase + float(_BRANCHES[bi]) * np.pi / 2.0

    order = _PERM_INDEX[pi]
    d = d0[order] * np.array(_EVEN_SIGNS[si])
    p = p0[:, order]
    if _PERM_PARITY[pi] < 0:  # keep det(P) = +1 so K1, K2 stay in SO(4)
        p = p.copy()
        p[:, 0] *= -1.0

    k1 = MAGIC @ (um @ p @ np.diag(1.0 / d)) @ MAGIC.conj().T
    k2 = MAGIC @ p.T @ MAGIC.conj().T
    if not (is_product(k1, atol=1e-7) and is_product(k2, atol=1e-7)):
        raise np.linalg.LinAlgError("KAK produced non-local factors; input not unitary?")

    k1a, k1b = split_product(k1)
    k2a, k2b = split_product(k2)

    result = KakDecomposition(global_phase, k1a, k1b, coeffs, k2a, k2b)
    if not np.allclose(result.matrix(), u, rtol=0.0, atol=max(atol, 1e-8)):
        raise np.linalg.LinAlgError("KAK reconstruction failed")
    return result


# ------------------------------------------------------- canonical gate circuits
#
# All three constructions come from  B = CX . (H (x) I), which satisfies
#     B^dag XX B = Z(x)I      B^dag YY B = -Z(x)Z      B^dag ZZ B = I(x)Z
# so N(a,b,c) is diagonal in the Bell basis.  See tests/test_decompose.py, which
# re-derives and re-checks each identity.

_U0 = rz(np.pi / 2) @ H @ S
_U1 = H @ S @ H
_V0 = H @ rz(np.pi / 2)


def _ops_one_cnot() -> tuple[list[Op], float]:
    """exp(i pi/4 XX) -- the CX-equivalent class."""
    ops: list[Op] = [
        ("1q", 0, H),
        ("cx", 0, 1),
        ("1q", 0, H @ SDG),
        ("1q", 1, H @ SDG @ H),
    ]
    return ops, np.pi / 4


def _ops_two_cnot(a: float, b: float) -> tuple[list[Op], float]:
    """N(a, b, 0), using CX . (RX (x) RZ) . CX == N(a, 0, b) conjugated by V (x) V."""
    ops: list[Op] = [
        ("1q", 0, _V),
        ("1q", 1, _V),
        ("cx", 0, 1),
        ("1q", 0, rx(-2.0 * a)),
        ("1q", 1, rz(-2.0 * b)),
        ("cx", 0, 1),
        ("1q", 0, _V),
        ("1q", 1, _V),
    ]
    return ops, 0.0


def _ops_three_cnot(a: float, b: float, c: float) -> tuple[list[Op], float]:
    """General N(a, b, c) in three CX gates."""
    ops: list[Op] = [
        ("1q", 0, _V0),
        ("cx", 0, 1),
        ("1q", 0, _U0),
        ("1q", 1, rz(2.0 * b) @ _U1),
        ("cx", 0, 1),
        ("1q", 0, H @ rz(-2.0 * a)),
        ("1q", 1, rz(-2.0 * c)),
        ("cx", 0, 1),
    ]
    return ops, np.pi / 4


def canonical_ops(a: float, b: float, c: float, atol: float = 1e-9) -> tuple[list[Op], float]:
    """Time-ordered ops and phase with ``prod(ops) * exp(i phase) == N(a, b, c)``.

    Picks the cheapest construction valid for the given coordinates and verifies
    it against :func:`canonical_gate` before returning.
    """
    target = canonical_gate(a, b, c)
    small = atol * 10.0

    candidates: list[tuple[list[Op], float]] = []
    if max(abs(a), abs(b), abs(c)) < small:
        candidates.append(([], 0.0))
    if abs(a - _PI_4) < small and abs(b) < small and abs(c) < small:
        candidates.append(_ops_one_cnot())
    if abs(c) < small:
        candidates.append(_ops_two_cnot(a, b))
    candidates.append(_ops_three_cnot(a, b, c))

    for ops, phase in candidates:
        if np.allclose(np.exp(1j * phase) * _ops_matrix(ops), target, rtol=0.0, atol=1e-9):
            return ops, phase
    raise np.linalg.LinAlgError(
        f"no canonical construction reproduced N({a}, {b}, {c})"
    )


def _ops_matrix(ops: list[Op]) -> np.ndarray:
    """Dense 4x4 matrix of a time-ordered op list."""
    mat = np.eye(4, dtype=complex)
    for op in ops:
        if op[0] == "1q":
            _, q, m = op
            step = np.kron(m, I2) if q == 0 else np.kron(I2, m)
        else:
            step = CX01
        mat = step @ mat
    return mat


def two_qubit_ops(u: np.ndarray, atol: float = 1e-8) -> tuple[list[Op], float]:
    """Decompose a 4x4 unitary into time-ordered ``{1q, cx}`` ops plus a phase.

    Returns ``(ops, phase)`` with ``exp(i phase) * prod(ops) == u`` to ``atol``.
    """
    u = np.asarray(u, dtype=complex)

    # Local gates need no entangling resource at all.
    if is_product(u, atol=1e-9):
        a, b = split_product(u)
        ops: list[Op] = [("1q", 0, a), ("1q", 1, b)]
        if np.allclose(_ops_matrix(ops), u, rtol=0.0, atol=atol):
            return ops, 0.0

    kak = kak_decomposition(u, atol=atol)
    mid, phase = canonical_ops(*kak.coefficients, atol=atol)
    ops = [("1q", 0, kak.k2a), ("1q", 1, kak.k2b), *mid, ("1q", 0, kak.k1a), ("1q", 1, kak.k1b)]
    phase += kak.global_phase

    if not np.allclose(
        np.exp(1j * phase) * _ops_matrix(ops), u, rtol=0.0, atol=max(atol, 1e-8)
    ):
        raise np.linalg.LinAlgError("two-qubit decomposition failed verification")
    return ops, phase
