"""Matrix Product State (Tensor Train) core.

Site tensors are stored as ``(D_left, 2, D_right)`` arrays with ``D_left == 1`` on
the first site and ``D_right == 1`` on the last, so that

    psi[s_0, ..., s_{n-1}] = A[0][:, s_0, :] @ A[1][:, s_1, :] @ ... @ A[n-1][:, s_{n-1}, :]

Index convention
----------------
Site ``k`` is qubit ``k``.  When an MPS is flattened to a dense statevector, index
``i`` is read in *big-endian* order: the most significant bit of ``i`` is qubit 0.
:func:`mpsynth.synthesis.synthesize` exposes a ``qubit_order`` switch for callers
who need the little-endian (Qiskit ``Statevector``) convention instead.

Canonical form
--------------
An MPS carries an orthogonality centre.  Tensors strictly left of the centre are
left-canonical, tensors strictly right of it are right-canonical, and the centre
tensor carries the norm.  Truncation is only variationally optimal at the centre,
so every routine that truncates moves the centre first.
"""

from __future__ import annotations

import numpy as np

from .linalg import truncated_svd

__all__ = ["MPS"]


class MPS:
    """A finite, open-boundary matrix product state over qubits."""

    def __init__(self, tensors: list[np.ndarray], center: int | None = None):
        if not tensors:
            raise ValueError("an MPS needs at least one site tensor")
        self.tensors = [np.asarray(t, dtype=complex) for t in tensors]
        for k, t in enumerate(self.tensors):
            if t.ndim != 3 or t.shape[1] != 2:
                raise ValueError(f"site {k}: expected shape (Dl, 2, Dr), got {t.shape}")
        if self.tensors[0].shape[0] != 1 or self.tensors[-1].shape[2] != 1:
            raise ValueError("open boundary conditions require trivial edge bonds")
        for k in range(len(self.tensors) - 1):
            if self.tensors[k].shape[2] != self.tensors[k + 1].shape[0]:
                raise ValueError(f"bond mismatch between sites {k} and {k + 1}")
        self.center = center

    # ------------------------------------------------------------------ basics

    @property
    def n_sites(self) -> int:
        return len(self.tensors)

    def bond_dimensions(self) -> list[int]:
        """Interior bond dimensions, length ``n_sites - 1``."""
        return [t.shape[2] for t in self.tensors[:-1]]

    def max_bond(self) -> int:
        dims = self.bond_dimensions()
        return max(dims) if dims else 1

    def copy(self) -> "MPS":
        return MPS([t.copy() for t in self.tensors], self.center)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"MPS(n_sites={self.n_sites}, max_bond={self.max_bond()}, center={self.center})"

    # ------------------------------------------------------------ constructors

    @classmethod
    def zero_state(cls, n_sites: int) -> "MPS":
        """The computational basis state |0...0>."""
        t = np.zeros((1, 2, 1), dtype=complex)
        t[0, 0, 0] = 1.0
        return cls([t.copy() for _ in range(n_sites)], center=0)

    @classmethod
    def from_statevector(
        cls,
        psi: np.ndarray,
        chi_max: int | None = None,
        tol: float = 0.0,
    ) -> tuple["MPS", float]:
        """Decompose a dense statevector by a left-to-right sweep of SVDs.

        Args:
            psi: length ``2**n`` amplitude vector (big-endian, see module docstring).
            chi_max: cap on bond dimension.
            tol: per-bond relative discarded weight budget.

        Returns:
            ``(mps, discarded)`` where ``mps`` is left-canonical with the centre on
            the last site and ``discarded`` is the total relative weight thrown away
            (summed over bonds; an upper bound on ``1 - fidelity``).
        """
        psi = np.asarray(psi, dtype=complex).reshape(-1)
        size = psi.size
        n = int(round(np.log2(size)))
        if 2**n != size:
            raise ValueError(f"statevector length {size} is not a power of two")
        if n == 0:
            raise ValueError("need at least one qubit")

        tensors: list[np.ndarray] = []
        total_discarded = 0.0
        left = 1
        rest = psi.reshape(1, -1)
        for _ in range(n - 1):
            mat = rest.reshape(left * 2, -1)
            res = truncated_svd(mat, chi_max=chi_max, tol=tol)
            tensors.append(res.u.reshape(left, 2, -1))
            total_discarded += res.discarded_weight
            rest = res.s[:, None] * res.vh
            left = res.u.shape[1]
        tensors.append(rest.reshape(left, 2, 1))
        return cls(tensors, center=n - 1), total_discarded

    def to_statevector(self) -> np.ndarray:
        """Contract to a dense statevector (only sane for modest site counts)."""
        if self.n_sites > 26:
            raise ValueError(f"refusing to build a 2**{self.n_sites} statevector")
        psi = self.tensors[0]
        for t in self.tensors[1:]:
            psi = np.tensordot(psi, t, axes=([-1], [0]))
        return psi.reshape(-1)

    # ---------------------------------------------------------------- geometry

    def _shift_center_right(self, k: int) -> None:
        """Make site ``k`` left-canonical, pushing its weight onto site ``k+1``."""
        t = self.tensors[k]
        dl, _, dr = t.shape
        q, r = np.linalg.qr(t.reshape(dl * 2, dr))
        self.tensors[k] = q.reshape(dl, 2, -1)
        self.tensors[k + 1] = np.tensordot(r, self.tensors[k + 1], axes=([1], [0]))

    def _shift_center_left(self, k: int) -> None:
        """Make site ``k`` right-canonical, pushing its weight onto site ``k-1``."""
        t = self.tensors[k]
        dl, _, dr = t.shape
        # RQ via QR of the transpose: M = R @ Q with Q having orthonormal rows.
        q1, r1 = np.linalg.qr(t.reshape(dl, 2 * dr).T)
        self.tensors[k] = q1.T.reshape(-1, 2, dr)
        self.tensors[k - 1] = np.tensordot(self.tensors[k - 1], r1.T, axes=([2], [0]))

    def move_center(self, target: int) -> None:
        """Move the orthogonality centre to ``target`` without truncating."""
        if not 0 <= target < self.n_sites:
            raise IndexError(f"site {target} out of range for {self.n_sites} sites")
        if self.center is None:
            # Unknown gauge: sweep all the way in from the far edge.
            for k in range(self.n_sites - 1):
                self._shift_center_right(k)
            self.center = self.n_sites - 1
        while self.center < target:
            self._shift_center_right(self.center)
            self.center += 1
        while self.center > target:
            self._shift_center_left(self.center)
            self.center -= 1

    def canonicalize(self, chi_max: int | None = None, tol: float = 0.0) -> float:
        """Sweep right then left, truncating on the way back.

        Returns the total relative discarded weight.  The state is left with its
        centre on site 0 and every other tensor right-canonical.
        """
        self.move_center(self.n_sites - 1)
        discarded = 0.0
        for k in range(self.n_sites - 1, 0, -1):
            t = self.tensors[k]
            dl, _, dr = t.shape
            res = truncated_svd(t.reshape(dl, 2 * dr), chi_max=chi_max, tol=tol)
            discarded += res.discarded_weight
            self.tensors[k] = res.vh.reshape(-1, 2, dr)
            self.tensors[k - 1] = np.tensordot(
                self.tensors[k - 1], res.u * res.s[None, :], axes=([2], [0])
            )
        self.center = 0
        return discarded

    def compress(self, chi_max: int | None = None, tol: float = 0.0) -> float:
        """Alias for :meth:`canonicalize` that reads better at call sites."""
        return self.canonicalize(chi_max=chi_max, tol=tol)

    # ------------------------------------------------------------------- norms

    def overlap(self, other: "MPS") -> complex:
        """``<self|other>`` by transfer-matrix contraction."""
        if self.n_sites != other.n_sites:
            raise ValueError("overlap needs matching site counts")
        env = np.ones((1, 1), dtype=complex)
        for a, b in zip(self.tensors, other.tensors):
            env = np.einsum("ab,asc,bsd->cd", env, a.conj(), b, optimize=True)
        return complex(env[0, 0])

    def norm(self) -> float:
        if self.center is not None:
            t = self.tensors[self.center]
            return float(np.sqrt(np.real(np.vdot(t, t))))
        return float(np.sqrt(max(np.real(self.overlap(self)), 0.0)))

    def normalize(self) -> float:
        """Rescale to unit norm in place; returns the norm that was removed.

        This is the renormalisation that must follow every truncation: discarding
        singular values shrinks ``sum |c_i|^2`` below 1, and a quantum state must
        satisfy ``sum |c_i|^2 == 1``.
        """
        nrm = self.norm()
        if nrm <= 0:
            raise ValueError("cannot normalize a zero state")
        idx = self.center if self.center is not None else 0
        self.tensors[idx] = self.tensors[idx] / nrm
        return nrm

    def fidelity(self, other: "MPS") -> float:
        """``|<self|other>|^2`` for normalised states."""
        return float(abs(self.overlap(other)) ** 2)

    # ------------------------------------------------------ entanglement structure

    def schmidt_values(self) -> list[np.ndarray]:
        """Schmidt coefficients across every bipartition ``(0..k | k+1..n-1)``.

        These are what decide whether the state compresses at all: a bond whose
        spectrum decays fast can be truncated cheaply, one that is flat cannot.
        Returned in descending order per bond, normalised to unit 2-norm.
        """
        work = self.copy()
        work.canonicalize()
        work.normalize()

        spectra: list[np.ndarray] = []
        for k in range(work.n_sites - 1):
            t = work.tensors[k]
            dl, _, dr = t.shape
            u, s, vh = np.linalg.svd(t.reshape(dl * 2, dr), full_matrices=False)
            norm = np.linalg.norm(s)
            spectra.append(s / norm if norm > 0 else s)
            work.tensors[k] = u.reshape(dl, 2, -1)
            work.tensors[k + 1] = np.tensordot(
                np.diag(s) @ vh, work.tensors[k + 1], axes=([1], [0])
            )
            work.center = k + 1
        return spectra

    def entanglement_entropy(self, base: float = 2.0) -> np.ndarray:
        """Von Neumann entropy ``S(k) = -sum p log p`` across each bipartition.

        In units of qubits (``base=2``).  A product state gives 0 everywhere; a
        Haar-random state approaches the maximum ``min(k+1, n-k-1)``, which is exactly
        the regime where no shallow circuit can help.
        """
        out = []
        for s in self.schmidt_values():
            p = s**2
            p = p[p > 1e-15]
            entropy = -np.sum(p * np.log(p)) / np.log(base) if p.size else 0.0
            out.append(max(0.0, float(entropy)))  # clamp -0.0 from rounding
        return np.array(out)

    def amplitude(self, bits: str | list[int]) -> complex:
        """Amplitude of one computational basis state, given qubit-0-first bits."""
        if len(bits) != self.n_sites:
            raise ValueError("bit string length must equal the number of sites")
        vec = np.ones((1,), dtype=complex)
        for t, s in zip(self.tensors, bits):
            vec = vec @ t[:, int(s), :]
        return complex(vec[0])

    # ------------------------------------------------------------ gate actions

    def apply_1q(self, gate: np.ndarray, site: int) -> None:
        """Apply a 2x2 unitary in place.

        A unitary acting on a single site preserves both left- and right-canonical
        form, so the orthogonality centre does not move.
        """
        gate = np.asarray(gate, dtype=complex)
        if gate.shape != (2, 2):
            raise ValueError(f"expected a 2x2 gate, got {gate.shape}")
        self.tensors[site] = np.einsum("st, atb->asb", gate, self.tensors[site], optimize=True)

    def apply_2q(
        self,
        gate: np.ndarray,
        site: int,
        chi_max: int | None = None,
        tol: float = 0.0,
    ) -> float:
        """Apply a 4x4 unitary to sites ``(site, site+1)``; returns discarded weight.

        The gate's index order is ``(q_site, q_site+1)`` with ``q_site`` the more
        significant bit, matching :func:`numpy.kron` ordering.
        """
        gate = np.asarray(gate, dtype=complex)
        if gate.shape != (4, 4):
            raise ValueError(f"expected a 4x4 gate, got {gate.shape}")
        if not 0 <= site < self.n_sites - 1:
            raise IndexError(f"two-site gate at {site} out of range")

        self.move_center(site)
        left = self.tensors[site]
        right = self.tensors[site + 1]
        dl = left.shape[0]
        dr = right.shape[2]

        theta = np.tensordot(left, right, axes=([2], [0]))  # (dl, 2, 2, dr)
        theta = np.einsum("ij,ajb->aib", gate, theta.reshape(dl, 4, dr), optimize=True)

        res = truncated_svd(theta.reshape(dl * 2, 2 * dr), chi_max=chi_max, tol=tol)
        self.tensors[site] = res.u.reshape(dl, 2, -1)
        self.tensors[site + 1] = (res.s[:, None] * res.vh).reshape(-1, 2, dr)
        self.center = site + 1
        return res.discarded_weight
