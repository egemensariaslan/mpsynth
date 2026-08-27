"""MPS -> quantum circuit synthesis.

Two facts drive the whole engine.

1. *A bond-dimension-2 MPS can be prepared exactly by one staircase of
   nearest-neighbour two-qubit gates.*  Put the state in right-canonical form; each
   site tensor is then an isometry ``C^{D_left} -> C^2_physical (x) C^{D_right}``.
   Embed it in a two-qubit unitary that consumes the bond carried by qubit ``k`` plus
   a fresh ``|0>`` on qubit ``k+1``, and emits the physical value on ``k`` and the
   next bond on ``k+1``.  Sweeping left to right costs ``n-1`` two-qubit gates and one
   single-qubit gate -- depth ``O(n)``, not ``O(2^n)``.

2. *Higher bond dimension is recovered by stacking those staircases.*  Approximate the
   target by its best bond-2 MPS, synthesise that layer exactly as ``U_1``, then apply
   ``U_1^dagger`` to the target and repeat on what is left.  After ``L`` layers

       |psi>  ~=  U_1 U_2 ... U_L |0...0>

   and the residual ``U_L^dagger ... U_1^dagger |psi>`` converges to ``|0...0>``.  ``L``
   is the depth knob traded against fidelity.

This is the disentangling scheme of Ran, *Phys. Rev. A* **101**, 032310 (2020).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .circuit import Circuit
from .decompose import two_qubit_ops
from .gates import CX01, I2, ry
from .linalg import complete_to_unitary
from .mps import MPS

__all__ = [
    "SynthesisResult",
    "Layer",
    "bond2_layer",
    "synthesize",
    "synthesize_mps",
    "prepare_vector",
]

# One entangling staircase: (matrix, sites) in time order.  `sites` is a 1-tuple for
# the trailing single-qubit gate and a 2-tuple (k, k+1) for the staircase gates.
Layer = list[tuple[np.ndarray, tuple[int, ...]]]

DENSE_SIMULATION_LIMIT = 22


# ------------------------------------------------------------- exact bond-2 layer


def bond2_layer(state: MPS, atol: float = 1e-8) -> Layer:
    """Exact preparation staircase for a normalised MPS of bond dimension <= 2.

    The state is right-canonicalised in place first; see the module docstring for
    why that makes every site tensor an isometry.
    """
    if state.max_bond() > 2:
        raise ValueError(f"bond2_layer needs bond dimension <= 2, got {state.max_bond()}")
    state.canonicalize()
    state.normalize()

    n = state.n_sites
    layer: Layer = []

    for k in range(n - 1):
        t = state.tensors[k]  # (dl, 2, dr) with dl, dr in {1, 2}
        dl, _, dr = t.shape

        if dr == 1:
            # The outgoing bond is trivial, so qubit k+1 stays in |0>: this site is a
            # single-qubit rotation and needs no entangling gate at all.
            iso = np.array([[t[al, s, 0] for al in range(dl)] for s in range(2)], dtype=complex)
            _check_isometry(iso, f"site {k}", atol)
            layer.append((complete_to_unitary(iso), (k,)))
            continue

        # Columns of the isometry, padded into the two-qubit space:
        #   row = 2 * s_k + alpha_k   (qubit k is the more significant bit)
        #   col = 2 * alpha_{k-1}     (fresh qubit k+1 starts in |0>)
        iso = np.zeros((4, dl), dtype=complex)
        for al in range(dl):
            for s in range(2):
                for ar in range(dr):
                    iso[2 * s + ar, al] = t[al, s, ar]
        _check_isometry(iso, f"site {k}", atol)

        if dl == 1:
            # Only |00> is constrained, so we are free to pick the cheapest unitary
            # that maps it to the required two-qubit state -- always one CNOT.
            layer.append((_state_preparation_gate(iso[:, 0]), (k, k + 1)))
            continue

        # complete_to_unitary appends the complement after the given columns, but the
        # incoming bond alpha_{k-1} lives on qubit k, so column 2*alpha is the one
        # the isometry must occupy.  Permute the free columns into the gaps.
        full = complete_to_unitary(iso)
        used = {2 * al for al in range(dl)}
        columns = [2 * al for al in range(dl)] + [c for c in range(4) if c not in used]
        gate = np.zeros((4, 4), dtype=complex)
        gate[:, columns] = full
        layer.append((gate, (k, k + 1)))

    # Final site: the remaining bond maps straight onto the last physical index.
    last = state.tensors[n - 1]  # (dl, 2, 1)
    dl = last.shape[0]
    iso = np.array([[last[al, s, 0] for al in range(dl)] for s in range(2)], dtype=complex)
    _check_isometry(iso, "last site", atol)
    layer.append((complete_to_unitary(iso), (n - 1,)))

    return layer


def _check_isometry(iso: np.ndarray, where: str, atol: float) -> None:
    if not np.allclose(iso.conj().T @ iso, np.eye(iso.shape[1]), rtol=0.0, atol=atol):
        raise ValueError(f"{where} is not right-canonical; cannot embed as a unitary")


def _state_preparation_gate(psi: np.ndarray) -> np.ndarray:
    """Cheapest two-qubit unitary mapping |00> to the two-qubit state ``psi``.

    Via the Schmidt decomposition ``psi = (A (x) B)(cos t |00> + sin t |11>)``, and
    ``cos t |00> + sin t |11> = CX . (RY(2t) (x) I) |00>`` -- so one CNOT suffices for
    any two-qubit state.
    """
    a, sigma, bh = np.linalg.svd(psi.reshape(2, 2))
    theta = float(np.arctan2(sigma[1], sigma[0]))
    return np.kron(a, bh.T) @ CX01 @ np.kron(ry(2.0 * theta), I2)


def _apply_layer_dagger(state: MPS, layer: Layer, chi_max: int | None, tol: float) -> float:
    """Apply ``U^dagger`` for a staircase, in reverse time order."""
    discarded = 0.0
    for matrix, sites in reversed(layer):
        if len(sites) == 1:
            state.apply_1q(matrix.conj().T, sites[0])
        else:
            discarded += state.apply_2q(matrix.conj().T, sites[0], chi_max=chi_max, tol=tol)
    return discarded


def _layers_to_circuit(layers: list[Layer], n_qubits: int, optimize: bool) -> Circuit:
    """Assemble ``U_1 U_2 ... U_L |0>``: last layer synthesised runs first in time."""
    circuit = Circuit(n_qubits)
    for layer in reversed(layers):
        for matrix, sites in layer:
            if len(sites) == 1:
                circuit.append_1q(matrix, sites[0])
                continue
            wires = sites
            ops, phase = two_qubit_ops(matrix)
            circuit.global_phase += phase
            for op in ops:
                if op[0] == "1q":
                    circuit.append_1q(op[2], wires[op[1]])
                else:
                    circuit.cx(wires[op[1]], wires[op[2]])
    return circuit.optimized() if optimize else circuit


# ------------------------------------------------------------------- entry points


@dataclass
class SynthesisResult:
    """A synthesised state-preparation circuit and everything measured about it."""

    circuit: Circuit
    n_qubits: int
    fidelity: float
    layers: int
    fidelity_history: list[float] = field(default_factory=list)
    input_bond_dimension: int = 1
    residual_chi: int = 0
    truncation_discarded: float = 0.0
    input_norm: float = 1.0
    padded_from: int | None = None
    qubit_order: str = "big"
    reached_target: bool = True
    fidelity_exact: bool = True

    @property
    def infidelity(self) -> float:
        return 1.0 - self.fidelity

    def amplitudes(self) -> np.ndarray:
        """The prepared state, re-indexed to line up element-for-element with the input.

        Undoes both the ``qubit_order`` re-indexing and the zero padding, and rescales
        by the original input norm, so ``result.amplitudes()`` is directly comparable
        with the vector that was passed to :func:`synthesize`.
        """
        psi = self.circuit.statevector()
        if self.qubit_order == "little":
            psi = _bit_reverse(psi, self.n_qubits)
        if self.padded_from is not None:
            psi = psi[: self.padded_from]
        return psi * self.input_norm

    def metrics(self) -> dict[str, float | int | str]:
        return {
            **self.circuit.metrics(),
            "layers": self.layers,
            "fidelity": self.fidelity,
            "infidelity": self.infidelity,
            "qubit_order": self.qubit_order,
        }

    def report(self) -> str:
        m = self.circuit.metrics()
        lines = [
            f"qubits            {self.n_qubits}",
            f"layers            {self.layers}",
            f"fidelity          {self.fidelity:.6f}"
            + ("" if self.fidelity_exact else "  (MPS estimate)"),
            f"infidelity        {self.infidelity:.3e}",
            f"depth             {m['depth']}",
            f"2-qubit depth     {m['two_qubit_depth']}",
            f"CNOT count        {m['cnot']}",
            f"1-qubit gates     {m['one_qubit']}",
            f"total gates       {m['total_gates']}",
            f"input bond dim    {self.input_bond_dimension}",
        ]
        if self.padded_from is not None:
            lines.append(f"zero-padded from  {self.padded_from} -> {2**self.n_qubits}")
        if not self.reached_target:
            lines.append("note              fidelity target not reached within max_layers")
        return "\n".join(lines)


def _bit_reverse(vector: np.ndarray, n: int) -> np.ndarray:
    """Reorder amplitudes between big-endian and little-endian qubit conventions."""
    idx = np.arange(vector.size)
    rev = np.zeros_like(idx)
    for bit in range(n):
        rev |= ((idx >> bit) & 1) << (n - 1 - bit)
    return vector[rev]


def prepare_vector(
    vector: np.ndarray,
    pad: bool = True,
    qubit_order: str = "big",
) -> tuple[np.ndarray, int, float, int | None]:
    """Normalise, zero-pad and (optionally) re-index a raw classical vector.

    Returns ``(psi, n_qubits, original_norm, padded_from)``.
    """
    vec = np.asarray(vector, dtype=complex).reshape(-1)
    if vec.size == 0:
        raise ValueError("input vector is empty")
    norm = float(np.linalg.norm(vec))
    if norm <= 0:
        raise ValueError("input vector has zero norm; nothing to encode")

    size = vec.size
    n = max(1, int(np.ceil(np.log2(size))))
    padded_from = None
    if 2**n != size:
        if not pad:
            raise ValueError(
                f"input length {size} is not a power of two (pass pad=True to zero-pad)"
            )
        padded = np.zeros(2**n, dtype=complex)
        padded[:size] = vec
        vec, padded_from = padded, size

    psi = vec / np.linalg.norm(vec)
    if qubit_order == "little":
        psi = _bit_reverse(psi, n)
    elif qubit_order != "big":
        raise ValueError("qubit_order must be 'big' or 'little'")
    return psi, n, norm, padded_from


def synthesize_mps(
    target: MPS,
    fidelity: float = 0.98,
    max_layers: int = 16,
    residual_chi: int | None = None,
    tol: float = 0.0,
    optimize: bool = True,
    verify_chi: int | None = None,
    collect_layers: bool = False,
) -> SynthesisResult | tuple[SynthesisResult, list[Layer]]:
    """Synthesise a preparation circuit for an MPS target.

    Args:
        target: normalised target state (not modified).
        fidelity: stop once the estimated fidelity reaches this value.
        max_layers: hard cap on the number of entangling staircases.
        residual_chi: bond-dimension cap while propagating the residual state.
            Defaults to twice the target's bond dimension, which is enough for the
            disentangler to work without the residual blowing up.
        tol: extra relative-weight truncation budget applied per bond.
        optimize: fuse single-qubit runs in the emitted circuit.
        verify_chi: bond cap used for the final fidelity check when the register is
            too large to simulate densely.
        collect_layers: also return the raw staircases (used by the profiler).
    """
    if not 0.0 < fidelity <= 1.0:
        raise ValueError("fidelity must lie in (0, 1]")
    if max_layers < 1:
        raise ValueError("max_layers must be at least 1")

    n = target.n_sites
    work = target.copy()
    work.canonicalize()
    work.normalize()
    input_bond = work.max_bond()

    if residual_chi is None:
        residual_chi = int(min(max(4, 2 * input_bond), 2 ** (n // 2)))

    zeros = "0" * n
    residual = work.copy()
    layers: list[Layer] = []
    history: list[float] = []
    discarded = 0.0
    reached = False

    for _ in range(max_layers):
        approx = residual.copy()
        approx.canonicalize(chi_max=2)
        approx.normalize()
        layer = bond2_layer(approx)
        layers.append(layer)

        discarded += _apply_layer_dagger(residual, layer, chi_max=residual_chi, tol=tol)
        discarded += residual.canonicalize(chi_max=residual_chi, tol=tol)
        residual.normalize()

        history.append(float(abs(residual.amplitude(zeros)) ** 2))
        if history[-1] >= fidelity:
            reached = True
            break

    circuit = _layers_to_circuit(layers, n, optimize)
    achieved, exact = _verify_fidelity(circuit, work, verify_chi)

    result = SynthesisResult(
        circuit=circuit,
        n_qubits=n,
        fidelity=achieved,
        layers=len(layers),
        fidelity_history=history,
        input_bond_dimension=input_bond,
        residual_chi=residual_chi,
        truncation_discarded=discarded,
        reached_target=reached or achieved >= fidelity,
        fidelity_exact=exact,
    )
    return (result, layers) if collect_layers else result


def _verify_fidelity(circuit: Circuit, target: MPS, verify_chi: int | None) -> tuple[float, bool]:
    """Fidelity of the circuit output against the target: exact when affordable."""
    n = target.n_sites
    if n <= DENSE_SIMULATION_LIMIT:
        produced = circuit.statevector()
        return float(abs(np.vdot(target.to_statevector(), produced)) ** 2), True
    chi = verify_chi if verify_chi is not None else max(32, 4 * target.max_bond())
    produced, _ = circuit.to_mps(chi_max=chi)
    produced.canonicalize(chi_max=chi)
    produced.normalize()
    return float(abs(target.overlap(produced)) ** 2), False


def synthesize(
    vector: np.ndarray,
    fidelity: float = 0.98,
    max_layers: int = 16,
    chi_max: int | None = None,
    residual_chi: int | None = None,
    tol: float = 0.0,
    pad: bool = True,
    qubit_order: str = "big",
    optimize: bool = True,
) -> SynthesisResult:
    """Synthesise a shallow approximate state-preparation circuit for a raw vector.

    Args:
        vector: classical data; normalised internally and zero-padded to ``2**n``.
        fidelity: target ``|<psi_exact|psi_approx>|^2``.
        max_layers: cap on entangling staircases (each costs ~``n-1`` CNOTs).
        chi_max: bond cap when decomposing the *input* into an MPS.  ``None`` keeps
            the decomposition exact, so the reported fidelity is measured against the
            true input rather than a pre-truncated stand-in.
        residual_chi: bond cap for the disentangling residual (see
            :func:`synthesize_mps`).
        tol: relative discarded-weight budget per bond.
        pad: zero-pad inputs whose length is not a power of two.
        qubit_order: ``"big"`` (qubit 0 is the most significant index bit, the
            OpenQASM/textbook reading) or ``"little"`` (Qiskit ``Statevector``).
        optimize: fuse single-qubit runs after synthesis.
    """
    psi, n, norm, padded_from = prepare_vector(vector, pad=pad, qubit_order=qubit_order)
    target, input_discarded = MPS.from_statevector(psi, chi_max=chi_max, tol=tol)

    result = synthesize_mps(
        target,
        fidelity=fidelity,
        max_layers=max_layers,
        residual_chi=residual_chi,
        tol=tol,
        optimize=optimize,
    )
    assert isinstance(result, SynthesisResult)

    # Re-measure against the *true* input when the MPS step itself truncated.
    if input_discarded > 0 and n <= DENSE_SIMULATION_LIMIT:
        produced = result.circuit.statevector()
        result.fidelity = float(abs(np.vdot(psi, produced)) ** 2)
        result.reached_target = result.fidelity >= fidelity

    result.truncation_discarded += input_discarded
    result.input_norm = norm
    result.padded_from = padded_from
    result.qubit_order = qubit_order
    result.circuit.amplitude_order = qubit_order
    return result
