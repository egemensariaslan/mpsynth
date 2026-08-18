"""Assemble JSON-serialisable analysis payloads for the UI and static reports.

Everything here is measurement, not presentation: the same numbers back the web UI,
the exported report and the tests.  Nothing is rounded or smoothed on the way out --
if a figure looks good it is because the circuit is good.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import __version__
from .circuit import Circuit
from .mps import MPS
from .profiler import TradeoffProfile, profile_mps
from .synthesis import DENSE_SIMULATION_LIMIT, prepare_vector

__all__ = ["Analysis", "analyze", "layer_detail", "verify_circuit", "MAX_PLOT_POINTS"]

#: Amplitude series are decimated to this many points before leaving the backend.
MAX_PLOT_POINTS = 1024


def _decimate(values: np.ndarray, limit: int = MAX_PLOT_POINTS) -> tuple[list[float], int]:
    """Subsample for plotting, reporting the stride so the axis can stay honest."""
    stride = max(1, int(np.ceil(values.size / limit)))
    return [float(v) for v in values[::stride]], stride


@dataclass
class Analysis:
    """A completed analysis, cached so per-layer detail is cheap to fetch."""

    label: str
    psi: np.ndarray          # normalised target, in synthesis (qubit_order) indexing
    raw: np.ndarray          # the user's vector, normalised, original indexing
    target: MPS
    curve: TradeoffProfile
    n_qubits: int
    qubit_order: str
    padded_from: int | None
    input_norm: float

    def summary(self) -> dict:
        """Everything the UI needs before a specific layer count is chosen."""
        spectra = self.target.schmidt_values()
        entropy = self.target.entanglement_entropy()
        n = self.n_qubits
        max_entropy = [float(min(k + 1, n - k - 1)) for k in range(n - 1)]

        amplitudes, stride = _decimate(np.real(self.raw))
        imaginary = np.imag(self.raw)
        has_phase = bool(np.max(np.abs(imaginary)) > 1e-12)

        return {
            "meta": {
                "version": __version__,
                "label": self.label,
                "n_qubits": n,
                "n_amplitudes": int(self.raw.size),
                "padded_from": self.padded_from,
                "qubit_order": self.qubit_order,
                "input_norm": self.input_norm,
                "bond_dimension": int(self.target.max_bond()),
                "fidelity_exact": self.curve.fidelity_exact,
            },
            "input": {
                "values": amplitudes,
                "imag": _decimate(imaginary)[0] if has_phase else None,
                "stride": stride,
                "has_phase": has_phase,
            },
            "entanglement": {
                # Schmidt spectra for every cut; the UI shows the middle one by default.
                "spectra": [[float(x) for x in s[:64]] for s in spectra],
                "entropy": [float(x) for x in entropy],
                "max_entropy": max_entropy,
                "bond_dims": [int(d) for d in self.target.bond_dimensions()],
                "saturation": float(
                    np.mean(entropy / np.maximum(np.array(max_entropy), 1e-12))
                ),
            },
            "tradeoff": [p.as_dict() for p in self.curve.points],
            "baseline": {
                "exact_cnot": self.curve.exact_cnot_estimate,
                "n_amplitudes": 2**n,
            },
        }


def analyze(
    vector: np.ndarray,
    label: str = "input",
    max_layers: int = 8,
    chi_max: int | None = None,
    residual_chi: int | None = None,
    tol: float = 0.0,
    qubit_order: str = "big",
) -> Analysis:
    """Decompose, profile and package a classical vector for display."""
    psi, n, norm, padded_from = prepare_vector(vector, pad=True, qubit_order=qubit_order)
    target, _ = MPS.from_statevector(psi, chi_max=chi_max, tol=tol)

    raw = np.asarray(vector, dtype=complex).reshape(-1)
    raw = raw / np.linalg.norm(raw)

    curve = profile_mps(
        target, max_layers=max_layers, residual_chi=residual_chi, tol=tol
    )
    return Analysis(
        label=label,
        psi=psi,
        raw=raw,
        target=target,
        curve=curve,
        n_qubits=n,
        qubit_order=qubit_order,
        padded_from=padded_from,
        input_norm=norm,
    )


def _reindex_to_input(psi: np.ndarray, analysis: Analysis) -> np.ndarray:
    """Undo the qubit-order re-indexing and padding so a series lines up with the input."""
    from .synthesis import _bit_reverse

    if analysis.qubit_order == "little":
        psi = _bit_reverse(psi, analysis.n_qubits)
    if analysis.padded_from is not None:
        psi = psi[: analysis.padded_from]
    return psi


def layer_detail(analysis: Analysis, layers: int) -> dict:
    """Prepared amplitudes, circuit and verification for one point on the curve."""
    circuit = analysis.curve.circuit_for(layers)
    point = next(p for p in analysis.curve.points if p.layers == layers)

    detail: dict = {
        "layers": layers,
        "metrics": {k: v for k, v in circuit.metrics().items()},
        "point": point.as_dict(),
        "circuit": _circuit_payload(circuit),
        "verification": verify_circuit(circuit, analysis),
    }

    if analysis.n_qubits <= DENSE_SIMULATION_LIMIT:
        produced = _reindex_to_input(circuit.statevector(), analysis)
        stride = _decimate(np.real(produced))[1]
        # Align the global phase before comparing, since it is unobservable.
        overlap = np.vdot(analysis.raw, produced)
        if abs(overlap) > 1e-15:
            produced = produced * np.exp(-1j * np.angle(overlap))
        # The chart plots real parts, so the plotted residual must be the difference of
        # those same real parts or it would not explain the curve above it.  The
        # reported error metrics use the full complex difference, which is the honest
        # figure -- a purely imaginary error is still an error.
        plotted_residual = np.real(produced) - np.real(analysis.raw)
        complex_error = np.abs(produced - analysis.raw)
        detail["amplitudes"] = {
            "produced": _decimate(np.real(produced))[0],
            "target": _decimate(np.real(analysis.raw))[0],
            "residual": _decimate(plotted_residual)[0],
            "stride": stride,
            "max_abs_error": float(np.max(complex_error)),
            "rms_error": float(np.sqrt(np.mean(complex_error**2))),
        }
    return detail


def _circuit_payload(circuit: Circuit, max_gates: int = 4000) -> dict:
    """Gate list with an ASAP column index, ready for an SVG circuit diagram."""
    frontier = [0] * circuit.n_qubits
    gates = []
    for gate in circuit.gates[:max_gates]:
        column = max(frontier[q] for q in gate.qubits)
        for q in gate.qubits:
            frontier[q] = column + 1
        gates.append(
            {
                "name": gate.name,
                "qubits": list(gate.qubits),
                "params": [round(float(p), 6) for p in gate.params],
                "column": column,
            }
        )
    return {
        "n_qubits": circuit.n_qubits,
        "gates": gates,
        "columns": max(frontier, default=0),
        "truncated": len(circuit.gates) > max_gates,
        "total_gates": len(circuit.gates),
        "global_phase": float(circuit.global_phase),
    }


def verify_circuit(circuit: Circuit, analysis: Analysis) -> dict:
    """Independent checks on the emitted circuit, reported pass/fail with residuals.

    These are deliberately *not* the checks synthesis already made internally: the
    state is re-simulated by a second, structurally different simulator (a dense
    statevector pass versus the tensor-network path) and the two are compared.
    """
    checks: list[dict] = []
    TOL = 1e-9

    def record(name: str, detail: str, residual: float, limit: float, bound: str) -> None:
        checks.append(
            {
                "name": name,
                "detail": detail,
                "residual": float(residual),
                "limit": float(limit),
                "bound": bound,
                "ok": bool(residual < limit),
            }
        )

    dense_ok = analysis.n_qubits <= DENSE_SIMULATION_LIMIT
    dense = circuit.statevector() if dense_ok else None

    if dense is not None:
        norm = float(np.linalg.norm(dense))
        record(
            "|‖ψ‖ − 1|",
            "truncation loses norm; the emitted circuit must still be unitary",
            abs(norm - 1.0), TOL, "< 10⁻⁹",
        )

        fidelity = float(abs(np.vdot(analysis.target.to_statevector(), dense)) ** 2)
        matching = [p.fidelity for p in analysis.curve.points if p.cnot == circuit.cnot_count]
        reported = matching[0] if matching else fidelity
        record(
            "|F_reported − F_measured|",
            "fidelity recomputed from the emitted gate list alone",
            abs(fidelity - reported), TOL, "< 10⁻⁹",
        )

        # Cross-check against the tensor-network simulator: two different algorithms.
        mps_state, _ = circuit.to_mps(chi_max=2 ** (analysis.n_qubits // 2))
        mps_state.normalize()
        agreement = float(abs(mps_state.overlap(MPS.from_statevector(dense)[0])) ** 2)
        record(
            "1 − |⟨dense|tensor⟩|²",
            "dense statevector against tensor-network contraction",
            abs(agreement - 1.0), TOL, "< 10⁻⁹",
        )

    if circuit.n_qubits <= 10:
        u = circuit.unitary(max_qubits=10)
        deviation = float(np.max(np.abs(u.conj().T @ u - np.eye(2**circuit.n_qubits))))
        record(
            "max |U†U − I|",
            "over the full 2ⁿ Hilbert space",
            deviation, TOL, "< 10⁻⁹",
        )

    entropy = analysis.target.entanglement_entropy()
    max_entropy = np.array(
        [min(k + 1, analysis.n_qubits - k - 1) for k in range(analysis.n_qubits - 1)]
    )
    saturation = float(np.mean(entropy / np.maximum(max_entropy, 1e-12)))
    record(
        "S̄ / S_max",
        "mean entanglement against the ceiling; above 0.7 no shallow circuit can help",
        saturation, 0.7, "< 0.70",
    )

    return {
        "checks": checks,
        "all_passed": all(c["ok"] for c in checks),
        "dense_simulated": dense_ok,
    }
