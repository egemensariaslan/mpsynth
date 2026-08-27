# Statistical validation

`python validation/run_validation.py --trials 300` — six checks, each run across many
random trials rather than a single example, with confidence intervals where a
distribution is meaningful. Two of the six cross-check against **Qiskit's own code**
(its `Statevector` simulator and its `StatePreparation` construction) rather than
MPSynth's internal bookkeeping, so a bug shared between "the implementation" and "the
implementation checking itself" cannot hide.

Reproduce with `python validation/run_validation.py --trials 300 --seed 0`. CI runs it
at every push (`--trials 40`, enough to catch a regression, sized to stay fast) and
uploads the full JSON report as a build artifact.

## Results (seed 0, 300 trials/check, 26.8s total)

| # | Check | Result |
| --- | --- | --- |
| A | Fidelity reproducibility | max discrepancy **1.7×10⁻¹⁵** across 300 trials, 11 dataset families |
| B | Qiskit cross-validation | max discrepancy **2.4×10⁻¹⁵** against Qiskit's `Statevector`, 300 trials |
| C | Trade-off monotonicity | **0 violations** in 1,068 curve points across 300 trials |
| D | CNOT reduction vs. Qiskit exact | **14.6× fewer CNOTs** on average (95% CI ±6.7), min 2.9×, at F ≥ 0.999 |
| E | Linear CNOT scaling | **R² = 0.982**, slope 9.8 CNOT/qubit, vs. exact `2ⁿ−2` (262,142 at n=18) |
| F | Normalization & unitarity | max ‖ψ‖ error **1.3×10⁻¹⁵**, max ‖U†U−I‖ error **3.3×10⁻¹⁵** |

## What each check actually establishes

**A — Fidelity reproducibility.** `synthesize()` reports a fidelity number. This check
re-simulates the *emitted circuit* from scratch — a completely separate code path from
the one that produced the number — and compares. 300 trials spanning all 11 built-in
dataset families, qubit counts 4–10, layer counts 1–5: the two never disagree by more
than 1.7×10⁻¹⁵, which is float64 noise, not a discrepancy. This is the check that
everything else depends on: if the reported number were wrong, every downstream claim
would be wrong silently.

**B — Qiskit cross-validation.** Same idea, but the second simulator isn't ours. The
MPSynth circuit is exported through `build_qiskit()` and simulated with Qiskit's own
`Statevector` — code this project did not write and cannot have introduced a matching
bug into. 300 trials, agreement to 2.4×10⁻¹⁵.

**C — Trade-off monotonicity.** The profiler presents "add a layer, pay more CNOTs, get
higher fidelity" as a safe trade-off curve. This check verifies that promise directly:
across 300 random inputs and 1,068 total curve points, fidelity and CNOT count never
decreased when a layer was added. Zero violations means picking any point on the curve
is safe — there's no case where a *deeper* circuit is quietly worse.

**D — CNOT reduction vs. Qiskit's own exact construction.** The baseline isn't
"`2ⁿ` is a big number" — it's what an independently-implemented exact-preparation
algorithm actually costs. `qiskit.circuit.library.StatePreparation` implements the
Möttönen et al. construction; transpiled to the same `{rz, ry, cx}` basis MPSynth uses,
it gives a real CNOT count to compare against. MPSynth at F ≥ 0.999 used **14.6× fewer
CNOTs on average**, with every single case at least 2.9× fewer. One of the eight
datasets (`gaussian:10`) is excluded — not by us: Qiskit's own isometry decomposition
raises `ValueError: Input matrix is not unitary` internally on that input, a numerical
edge case in the baseline we're comparing against. The check reports it as skipped and
continues rather than hiding it.

**E — Linear CNOT scaling.** The claim "depth grows linearly in `n`, not exponentially"
is falsifiable: fit CNOT count against qubit count for a fixed 4-layer circuit on a
smooth input, `n` = 8 to 18. Linear regression gives R² = 0.982 — not "looks roughly
flat," an actual fit. At n=18 that's 161 CNOTs against an exact-encoding baseline of
262,142.

**F — Normalization & unitarity.** The spec's own critical note: SVD truncation loses
norm, and the recovered circuit must still satisfy `U†U = I`. Checked under
*aggressive* truncation (`chi_max` 1–2, deliberately far from exact) across 300 trials
and 11 dataset families — every circuit stays normalized and unitary to float64
precision (max error 3.3×10⁻¹⁵), regardless of how badly the truncation itself
approximates the target.

## What this doesn't claim

These checks establish that MPSynth's numbers are *self-consistent and externally
verifiable* — not that the underlying compressibility assumption holds for every
possible input. It doesn't (see the README's "when this does not help" section):
incompressible inputs still cost real CNOTs, and check D's reduction factors are
specific to the eight datasets tested, not a universal constant.
