# Changelog

All notable changes to this project are documented here. Format loosely follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.1.0] - 2026-08-27

Initial release.

### Added

- Core synthesis engine: exact bond-2 MPS staircase preparation, sequential
  disentangling (Ran, *Phys. Rev. A* **101**, 032310, 2020), and a Cartan (KAK)
  decomposition that lowers every two-qubit gate to the minimum CX count its
  Weyl-chamber coordinates allow (0/1/2/3).
- `mpsynth.mps` — Matrix Product State core: truncated SVD with an explicit
  discarded-weight budget, canonical forms, Schmidt spectra, entanglement entropy.
- `mpsynth.circuit` — backend-independent `{RZ, RY, CX}` IR with metrics, dense
  simulation, an MPS-backed simulator for large registers, and a peephole optimizer
  that never increases gate count.
- Multi-target export: OpenQASM 2.0/3.0, QIR (LLVM base profile), Q#, PennyLane,
  Qiskit — verified by executing the emitted code in the real frameworks, not just
  pattern-matching the text.
- `mpsynth.profiler` — fidelity/depth trade-off curves computed as a by-product of
  a single synthesis pass.
- CLI (`mpsynth synth|profile|show|ui`) and a zero-install repo-root launcher
  (`./mpsynth`) that self-heals a missing numpy via `uv run --with numpy`.
- Local web workbench (`mpsynth ui`): stdlib `http.server` backend, hand-written
  HTML/CSS/SVG frontend, zero external dependencies. Entanglement entropy and
  Schmidt-spectrum visualization, an interactive fidelity/cost curve, a linked
  amplitude/residual cursor, and per-circuit verification against a second,
  independent simulator.
- Standalone offline report export (`mpsynth ui` → "report"): a single
  self-contained HTML file with every layer's circuit, checks and every export
  format pre-embedded, fully interactive with no server.
- Statistical validation suite (`validation/run_validation.py`): six checks run
  across hundreds of random trials, two of which cross-validate against Qiskit's
  own `Statevector` simulator and `StatePreparation` construction rather than
  MPSynth's internal bookkeeping. See `validation/VALIDATION.md`.
- 233+ unit/integration tests plus the validation suite; CI matrix across
  Python 3.10-3.13 on Linux/macOS/Windows, mypy (strict, zero errors), ruff.

### Known limitations

- Incompressible inputs (i.i.d. random amplitudes) do not compress — this is
  information-theoretic, not an implementation gap, and the tool reports the
  resulting low fidelity rather than hiding it.
- Qiskit's own `StatePreparation` has a numerical edge case (`ValueError: Input
  matrix is not unitary`) on certain smooth, symmetric real inputs at n=10 in this
  version of Qiskit (2.5.2); the validation suite detects and skips the affected
  case rather than failing outright. Tracked as a third-party issue, not an
  MPSynth bug.
