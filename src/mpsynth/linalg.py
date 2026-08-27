"""Small numerical-linear-algebra helpers shared by the tensor and synthesis layers.

Everything here is plain NumPy: MPSynth deliberately has no SciPy runtime dependency.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "TruncationResult",
    "truncated_svd",
    "complete_to_unitary",
    "is_unitary",
    "is_product",
    "split_product",
]


@dataclass(frozen=True)
class TruncationResult:
    """Outcome of one truncated SVD.

    Attributes:
        u, s, vh: the retained factors, ``mat ~= u @ diag(s) @ vh``.
        discarded_weight: sum of squared *discarded* singular values divided by the
            sum of all squared singular values.  This is exactly the relative loss of
            squared norm caused by the truncation, i.e. ``1 - F`` for the local step.
        full_rank: rank before truncation (number of non-negligible singular values).
    """

    u: np.ndarray
    s: np.ndarray
    vh: np.ndarray
    discarded_weight: float
    full_rank: int


def truncated_svd(
    mat: np.ndarray,
    chi_max: int | None = None,
    tol: float = 0.0,
    abs_cutoff: float = 1e-14,
) -> TruncationResult:
    """SVD of ``mat`` truncated to bond dimension ``chi_max`` and relative weight ``tol``.

    Singular values are dropped from the tail while the accumulated relative
    discarded weight stays at or below ``tol``, and the retained count never
    exceeds ``chi_max``.  At least one singular value is always retained so the
    bond dimension never collapses to zero.

    Args:
        mat: matrix to decompose.
        chi_max: hard cap on the retained bond dimension (``None`` for no cap).
        tol: maximum relative discarded weight, ``sum(s_dropped**2)/sum(s**2)``.
        abs_cutoff: singular values below this absolute value are always dropped.
    """
    try:
        u, s, vh = np.linalg.svd(mat, full_matrices=False)
    except np.linalg.LinAlgError:  # pragma: no cover - rare LAPACK non-convergence
        # Jitter and retry; gesdd occasionally fails on pathological inputs.
        noise = np.finfo(mat.dtype).eps * np.linalg.norm(mat)
        u, s, vh = np.linalg.svd(
            mat + noise * np.eye(*mat.shape, dtype=mat.dtype), full_matrices=False
        )

    total = float(np.sum(s**2))
    if total <= 0.0:
        return TruncationResult(u[:, :1], s[:1], vh[:1], 0.0, 0)

    full_rank = int(np.count_nonzero(s > abs_cutoff * s[0]))

    # Tail sums: tail[k] == sum(s[k:]**2), so tail[k]/total is the loss if we keep k.
    tail = np.concatenate([np.cumsum((s**2)[::-1])[::-1], [0.0]])
    keep = int(np.searchsorted(-tail / total, -tol, side="left"))
    keep = max(keep, 1)
    keep = min(keep, len(s))
    if chi_max is not None:
        keep = min(keep, max(int(chi_max), 1))
    # Never keep numerically-zero directions.
    keep = max(1, min(keep, max(full_rank, 1)))

    discarded = float(tail[keep] / total)
    return TruncationResult(u[:, :keep], s[:keep], vh[:keep], discarded, full_rank)


def complete_to_unitary(iso: np.ndarray) -> np.ndarray:
    """Extend an isometry to a full unitary.

    Args:
        iso: ``(m, k)`` array with orthonormal columns, ``m >= k``.

    Returns:
        ``(m, m)`` unitary whose first ``k`` columns are exactly ``iso``.
    """
    m, k = iso.shape
    if k > m:
        raise ValueError(f"cannot complete a {m}x{k} isometry: more columns than rows")
    if k == m:
        return np.array(iso, dtype=complex)

    # Left singular vectors of `iso` span the same space as its columns, so the
    # trailing ones form an orthonormal basis of the orthogonal complement.
    u, _, _ = np.linalg.svd(iso, full_matrices=True)
    return np.concatenate([iso, u[:, k:]], axis=1).astype(complex)


def is_unitary(mat: np.ndarray, atol: float = 1e-9) -> bool:
    """True if ``mat`` satisfies ``U^dagger U == I`` to within ``atol``."""
    mat = np.asarray(mat)
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        return False
    return bool(np.allclose(mat.conj().T @ mat, np.eye(mat.shape[0]), atol=atol))


def _product_reshape(mat: np.ndarray) -> np.ndarray:
    """Rearrange a 4x4 operator so that a tensor product becomes a rank-1 matrix."""
    t = mat.reshape(2, 2, 2, 2)  # (i, j, k, l) for <i j| U |k l>
    return t.transpose(0, 2, 1, 3).reshape(4, 4)  # ((i,k), (j,l))


def is_product(mat: np.ndarray, atol: float = 1e-10) -> bool:
    """True if the 4x4 operator ``mat`` factorises as ``A (x) B``."""
    s = np.linalg.svd(_product_reshape(mat), compute_uv=False)
    return bool(s[1] <= atol * max(s[0], 1.0))


def split_product(mat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split ``mat == A (x) B`` into unitary factors ``A`` and ``B``.

    The global phase is distributed so that both factors have unit determinant
    magnitude; the caller is responsible for tracking any residual phase (use
    :func:`is_product` first if the factorisation is not guaranteed).
    """
    u, s, vh = np.linalg.svd(_product_reshape(mat))
    a = (u[:, 0] * np.sqrt(s[0])).reshape(2, 2)
    b = (vh[0, :] * np.sqrt(s[0])).reshape(2, 2)
    # `a` and `b` are only defined up to reciprocal scalars; renormalise each to
    # be unitary and push the leftover phase onto `a`.
    da = np.linalg.det(a)
    db = np.linalg.det(b)
    if abs(da) > 1e-12:
        a = a / np.sqrt(da)
    if abs(db) > 1e-12:
        b = b / np.sqrt(db)
    # Fix the residual scalar by matching one non-negligible entry of the product.
    prod = np.kron(a, b)
    idx = np.unravel_index(np.argmax(np.abs(prod)), prod.shape)
    a = a * (mat[idx] / prod[idx])
    return a, b
