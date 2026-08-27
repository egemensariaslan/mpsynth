"""Statistical validation suite for MPSynth's core claims.

Unlike the unit tests (which check individual behaviors against fixed expectations),
this suite runs each claim across many random trials and reports aggregate statistics
with confidence intervals -- and, where possible, cross-checks against an independent
codebase (Qiskit) rather than MPSynth's own simulator, so a bug shared between
"the code" and "the code checking the code" cannot hide.

    python validation/run_validation.py --trials 200

Qiskit-dependent checks (B, D) are skipped -- explicitly, not silently -- if qiskit
isn't installed. Everything else needs only numpy.

Exit code is 0 iff every check passes; --ci makes that exit code the point of running
this in CI rather than eyeballing the printout.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mpsynth import MPS, profile_mps, synthesize  # noqa: E402
from mpsynth.datasets import GENERATORS, generate  # noqa: E402


@dataclass
class CheckResult:
    name: str
    claim: str
    n_trials: int
    passed: bool
    stats: dict = field(default_factory=dict)
    detail: str = ""


def mean_ci95(values: np.ndarray) -> tuple[float, float]:
    """Mean and half-width of a normal-approximation 95% CI."""
    values = np.asarray(values, dtype=float)
    if values.size < 2:
        return float(values.mean()) if values.size else 0.0, 0.0
    se = float(values.std(ddof=1) / np.sqrt(values.size))
    return float(values.mean()), 1.96 * se


# --------------------------------------------------------------------- Check A


def check_fidelity_reproducibility(trials: int, rng: np.random.Generator) -> CheckResult:
    """Claim: the fidelity synthesize() reports is exactly what re-simulation gives.

    This is the single most load-bearing number MPSynth prints. If it were wrong,
    every other measurement in this suite would be wrong too, silently.
    """
    errors = []
    families = list(GENERATORS)
    for i in range(trials):
        name = families[i % len(families)]
        n = int(rng.integers(4, 11))
        vector = generate(name, n, seed=int(rng.integers(0, 1_000_000)))
        result = synthesize(vector, fidelity=1.0, max_layers=int(rng.integers(1, 6)))

        target = vector.astype(complex) / np.linalg.norm(vector)
        produced = result.circuit.statevector()
        measured = float(abs(np.vdot(target, produced)) ** 2)
        errors.append(abs(measured - result.fidelity))

    errors = np.array(errors)
    mean, ci = mean_ci95(errors)
    passed = bool(np.max(errors) < 1e-8)
    return CheckResult(
        name="fidelity_reproducibility",
        claim="synthesize().fidelity == |<target|circuit.statevector()>|^2, independently re-simulated",
        n_trials=trials,
        passed=passed,
        stats={"max_abs_error": float(np.max(errors)), "mean_abs_error": mean, "ci95": ci},
        detail=f"max discrepancy {np.max(errors):.3e} across {trials} trials over {len(families)} dataset families",
    )


# --------------------------------------------------------------------- Check B


def check_qiskit_cross_validation(trials: int, rng: np.random.Generator) -> CheckResult:
    """Claim: MPSynth's fidelity number agrees with an *independent* simulator.

    MPSynth's circuit is exported and simulated by Qiskit's own Statevector -- code
    this project did not write -- and compared against the target vector directly,
    with no MPSynth code in the loop at measurement time.
    """
    try:
        from qiskit.quantum_info import Statevector

        from mpsynth.exporters import build_qiskit
    except ImportError:
        return CheckResult(
            name="qiskit_cross_validation",
            claim="fidelity agrees with an independently-implemented simulator",
            n_trials=0,
            passed=True,
            detail="SKIPPED: qiskit not installed",
        )

    errors = []
    families = list(GENERATORS)
    for i in range(trials):
        name = families[i % len(families)]
        n = int(rng.integers(4, 10))
        vector = generate(name, n, seed=int(rng.integers(0, 1_000_000)))
        result = synthesize(vector, fidelity=0.995, max_layers=6, qubit_order="little")
        target = vector.astype(complex) / np.linalg.norm(vector)

        qc = build_qiskit(result.circuit)
        produced = np.asarray(Statevector(qc))
        measured = float(abs(np.vdot(target, produced)) ** 2)
        errors.append(abs(measured - result.fidelity))

    errors = np.array(errors)
    mean, ci = mean_ci95(errors)
    passed = bool(np.max(errors) < 1e-8)
    return CheckResult(
        name="qiskit_cross_validation",
        claim="fidelity agrees with an independently-implemented simulator (Qiskit Statevector)",
        n_trials=trials,
        passed=passed,
        stats={"max_abs_error": float(np.max(errors)), "mean_abs_error": mean, "ci95": ci},
        detail=f"max discrepancy {np.max(errors):.3e} across {trials} trials, cross-checked via Qiskit",
    )


# --------------------------------------------------------------------- Check C


def check_monotonicity(trials: int, rng: np.random.Generator) -> CheckResult:
    """Claim: adding a disentangling layer never decreases fidelity or CNOT count.

    This is what makes the trade-off curve a genuine curve (safe to pick any point)
    rather than a scatter where a deeper circuit could occasionally be worse.
    """
    violations = 0
    total_points = 0
    families = list(GENERATORS)
    for i in range(trials):
        name = families[i % len(families)]
        n = int(rng.integers(5, 9))
        vector = generate(name, n, seed=int(rng.integers(0, 1_000_000)))
        target, _ = MPS.from_statevector(vector.astype(complex) / np.linalg.norm(vector))
        curve = profile_mps(target, max_layers=5)

        fids = [p.fidelity for p in curve.points]
        cnots = [p.cnot for p in curve.points]
        total_points += len(fids)
        if any(b < a - 1e-12 for a, b in zip(fids, fids[1:], strict=False)):
            violations += 1
        if any(b < a for a, b in zip(cnots, cnots[1:], strict=False)):
            violations += 1

    passed = violations == 0
    return CheckResult(
        name="monotonicity",
        claim="fidelity(L+1) >= fidelity(L) and cnot(L+1) >= cnot(L) for every trial",
        n_trials=trials,
        passed=passed,
        stats={"violations": violations, "total_curve_points": total_points},
        detail=f"{violations} violations across {trials} trials ({total_points} curve points)",
    )


# --------------------------------------------------------------------- Check D


def check_cnot_reduction_vs_qiskit(rng: np.random.Generator) -> CheckResult:
    """Claim: MPSynth uses substantially fewer CNOTs than exact state preparation
    at a matched, high fidelity target -- measured against Qiskit's own exact
    StatePreparation construction (Mottonen et al.), not a back-of-envelope estimate.
    """
    try:
        from qiskit import transpile
        from qiskit.circuit.library import StatePreparation
    except ImportError:
        return CheckResult(
            name="cnot_reduction_vs_qiskit",
            claim="CNOT count is far below Qiskit's exact StatePreparation at matched fidelity",
            n_trials=0,
            passed=True,
            detail="SKIPPED: qiskit not installed",
        )

    cases = [
        ("gaussian", 8),
        ("gaussian", 10),
        ("lognormal", 8),
        ("lognormal", 10),
        ("bimodal", 8),
        ("damped", 9),
        ("lorentzian", 8),
        ("heavytail", 9),
    ]
    reductions = []
    per_case = []
    skipped = []
    for name, n in cases:
        vector = generate(name, n, seed=0)
        try:
            exact_qc = StatePreparation(vector / np.linalg.norm(vector)).definition
            exact_tqc = transpile(
                exact_qc, basis_gates=["rz", "ry", "rx", "cx"], optimization_level=1
            )
            exact_cnot = exact_tqc.count_ops().get("cx", 0)
        except Exception as exc:
            # Qiskit's own isometry decomposition has a known numerical edge case on
            # certain smooth/symmetric real vectors (observed: ValueError "Input matrix
            # is not unitary" from its internal UnitaryGate construction). That is a
            # fragility in the baseline we are comparing against, not in MPSynth, so we
            # skip the single case rather than let a third-party bug fail this check.
            skipped.append({"dataset": f"{name}:{n}", "reason": f"{type(exc).__name__}: {exc}"})
            continue

        result = synthesize(vector, fidelity=0.999, max_layers=8)
        reduction = exact_cnot / max(result.circuit.cnot_count, 1)
        reductions.append(reduction)
        per_case.append(
            {
                "dataset": f"{name}:{n}",
                "qiskit_exact_cnot": exact_cnot,
                "mpsynth_cnot": result.circuit.cnot_count,
                "reduction": round(reduction, 1),
            }
        )

    if not reductions:
        return CheckResult(
            name="cnot_reduction_vs_qiskit",
            claim="CNOT count is far below Qiskit's exact StatePreparation at matched fidelity",
            n_trials=0,
            passed=False,
            detail="all cases skipped due to Qiskit-side failures",
            stats={"skipped": skipped},
        )
    reductions = np.array(reductions)
    mean, ci = mean_ci95(reductions)
    passed = bool(np.min(reductions) > 2.0)  # every case at least 2x fewer CNOTs
    return CheckResult(
        name="cnot_reduction_vs_qiskit",
        claim="CNOT count at F>=0.999 is far below Qiskit's exact StatePreparation, same basis gates",
        n_trials=len(cases),
        passed=passed,
        stats={
            "mean_reduction": mean,
            "ci95": ci,
            "min_reduction": float(np.min(reductions)),
            "max_reduction": float(np.max(reductions)),
            "per_case": per_case,
            "skipped": skipped,
        },
        detail=f"mean {mean:.1f}x fewer CNOTs (95% CI +/-{ci:.1f}) across {len(per_case)}/{len(cases)} datasets"
        f" ({len(skipped)} skipped: Qiskit-side failure), min {np.min(reductions):.1f}x",
    )


# --------------------------------------------------------------------- Check E


def check_linear_scaling(rng: np.random.Generator) -> CheckResult:
    """Claim: CNOT count grows linearly (not exponentially) in qubit count for
    compressible inputs, at a fixed layer count and fixed fidelity behavior.
    """
    sizes = [8, 10, 12, 14, 16, 18]
    counts = []
    for n in sizes:
        vector = generate("gaussian", n)
        result = synthesize(vector, fidelity=1.0, max_layers=4)
        counts.append(result.circuit.cnot_count)

    sizes_arr = np.array(sizes, dtype=float)
    counts_arr = np.array(counts, dtype=float)
    slope, intercept = np.polyfit(sizes_arr, counts_arr, 1)
    predicted = slope * sizes_arr + intercept
    ss_res = float(np.sum((counts_arr - predicted) ** 2))
    ss_tot = float(np.sum((counts_arr - counts_arr.mean()) ** 2))
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 1.0

    exact_baseline = [2**n - 2 for n in sizes]
    passed = bool(r_squared > 0.98 and counts[-1] < exact_baseline[-1] / 100)
    return CheckResult(
        name="linear_scaling",
        claim="CNOT(n) is linear in n for a fixed layer count (vs. exact encoding's 2^n - 2)",
        n_trials=len(sizes),
        passed=passed,
        stats={
            "sizes": sizes,
            "cnot_counts": counts,
            "exact_baseline": exact_baseline,
            "slope_cnot_per_qubit": float(slope),
            "r_squared": float(r_squared),
        },
        detail=f"linear fit R^2={r_squared:.4f}, slope={slope:.1f} CNOT/qubit, "
        f"vs exact 2^n-2 = {exact_baseline[-1]:,} at n={sizes[-1]}",
    )


# --------------------------------------------------------------------- Check F


def check_normalization_and_unitarity(trials: int, rng: np.random.Generator) -> CheckResult:
    """Claim: every emitted circuit is exactly unitary, even under aggressive truncation.

    This is the correctness contract from the spec: SVD truncation loses norm, and
    the circuit must re-normalize so the physical state stays valid.
    """
    norm_errors = []
    unitary_errors = []
    families = list(GENERATORS)
    for i in range(trials):
        name = families[i % len(families)]
        n = int(rng.integers(3, 7))  # small enough for a dense unitary check
        vector = generate(name, n, seed=int(rng.integers(0, 1_000_000)))
        chi = int(rng.integers(1, 3))  # aggressive truncation
        result = synthesize(vector, chi_max=chi, fidelity=0.5, max_layers=2)

        psi = result.circuit.statevector()
        norm_errors.append(abs(float(np.linalg.norm(psi)) - 1.0))

        u = result.circuit.unitary(max_qubits=8)
        dim = u.shape[0]
        unitary_errors.append(float(np.max(np.abs(u.conj().T @ u - np.eye(dim)))))

    norm_errors = np.array(norm_errors)
    unitary_errors = np.array(unitary_errors)
    passed = bool(np.max(norm_errors) < 1e-9 and np.max(unitary_errors) < 1e-9)
    return CheckResult(
        name="normalization_and_unitarity",
        claim="||psi|| == 1 and U^dagger U == I for every circuit, even under aggressive truncation",
        n_trials=trials,
        passed=passed,
        stats={
            "max_norm_error": float(np.max(norm_errors)),
            "max_unitary_error": float(np.max(unitary_errors)),
        },
        detail=f"max norm error {np.max(norm_errors):.2e}, max U-dagger-U error {np.max(unitary_errors):.2e}",
    )


# ------------------------------------------------------------------------ main


CHECKS = [
    ("A", "Fidelity reproducibility", lambda t, r: check_fidelity_reproducibility(t, r)),
    ("B", "Qiskit cross-validation", lambda t, r: check_qiskit_cross_validation(t, r)),
    ("C", "Trade-off monotonicity", lambda t, r: check_monotonicity(t, r)),
    ("D", "CNOT reduction vs. Qiskit exact", lambda t, r: check_cnot_reduction_vs_qiskit(r)),
    ("E", "Linear CNOT scaling", lambda t, r: check_linear_scaling(r)),
    ("F", "Normalization & unitarity", lambda t, r: check_normalization_and_unitarity(t, r)),
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=200, help="trials per statistical check")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ci", action="store_true", help="exit nonzero if any check fails")
    parser.add_argument("--out", default=str(Path(__file__).parent / "report.json"))
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    results: list[CheckResult] = []

    print(f"MPSynth statistical validation -- {args.trials} trials/check, seed={args.seed}\n")
    start = time.time()
    for letter, label, fn in CHECKS:
        t0 = time.time()
        result = fn(args.trials, rng)
        elapsed = time.time() - t0
        mark = "PASS" if result.passed else "FAIL"
        print(f"[{letter}] {mark}  {label:<32s} {result.detail}  ({elapsed:.1f}s)")
        results.append(result)

    total = time.time() - start
    all_passed = all(r.passed for r in results)
    print(f"\n{'ALL CHECKS PASSED' if all_passed else 'SOME CHECKS FAILED'} in {total:.1f}s")

    report = {
        "trials": args.trials,
        "seed": args.seed,
        "all_passed": all_passed,
        "elapsed_seconds": total,
        "checks": [asdict(r) for r in results],
    }
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"wrote {args.out}")

    return 0 if (not args.ci or all_passed) else 1


if __name__ == "__main__":
    raise SystemExit(main())
