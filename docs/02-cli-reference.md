# CLI and API reference

## Command line

```console
./mpsynth examples/gaussian_1024.csv          # smooth density, 10 qubits
./mpsynth examples/chirp_1024.csv -L 6        # oscillatory signal -- deliberately harder
./mpsynth gaussian:16                         # built-in generator, no data file
./mpsynth random:10                           # an input that does NOT compress
./mpsynth data.csv -o out.py --format qiskit  # qasm2 | qasm3 | qir | qsharp | pennylane | qiskit
./mpsynth synth data.csv -f 0.99 -o out.qasm  # stop at the target instead of profiling
./mpsynth ui                                  # interactive workbench
./mpsynth show                                # datasets and export targets
```

Inputs may be `.csv`, `.txt`, `.npy`, `.npz`, `.json`, or a built-in generator spec such
as `gaussian:12` (name : qubits [: seed]). Vectors are normalised and zero-padded to a
power of two automatically. With no subcommand, `./mpsynth <input>` profiles the
trade-off; `-o` also writes the cheapest circuit that meets `-f`.

Nothing needs installing — `./mpsynth` in a clone is fully functional. Install only if
you want `mpsynth` on your PATH without the clone path, or the library in your own code:

```console
uv tool install .           # or: pipx install .
pip install -e ".[dev]"     # into an active virtualenv, + pytest/scipy/matplotlib
```

## Library API

```python
synthesize(vector, fidelity=0.98, max_layers=16, chi_max=None,
           residual_chi=None, tol=0.0, pad=True,
           qubit_order="big", optimize=True) -> SynthesisResult
```

| argument | meaning |
| --- | --- |
| `fidelity` | target `\|⟨ψ_exact\|ψ_approx⟩\|²`; synthesis stops once it is reached |
| `max_layers` | cap on entangling staircases; each costs about `n-1` two-qubit gates |
| `chi_max` | bond cap when decomposing the *input*. `None` keeps it exact, so the reported fidelity is measured against your true data |
| `residual_chi` | bond cap for the disentangling residual |
| `tol` | relative discarded-weight budget per bond |
| `optimize` | fuse single-qubit runs (never increases the gate count) |

`SynthesisResult` carries `.circuit`, `.fidelity`, `.layers`, `.report()`,
`.metrics()`, `.amplitudes()`, `.reached_target`, `.input_bond_dimension`.

Already have a tensor network? Skip the dense vector entirely — `synthesize_mps(mps, ...)`
and `profile_mps(mps, ...)` take an `MPS` directly, so nothing ever materialises `2ⁿ`
amplitudes.

## Export targets

```python
export(result.circuit, "qasm2")      # OpenQASM 2.0     .qasm
export(result.circuit, "qasm3")      # OpenQASM 3.0     .qasm   (carries global phase)
export(result.circuit, "qir")        # QIR / LLVM IR    .ll     (base profile)
export(result.circuit, "qsharp")     # Q#               .qs
export(result.circuit, "pennylane")  # PennyLane        .py
export(result.circuit, "qiskit")     # Qiskit           .py
```

Everything is synthesised into one small gate set — `RZ`, `RY`, `CX` plus a tracked
global phase — so no exporter re-transpiles and the reported depth and gate counts
describe every emitted artefact identically. Live objects are available too, importing
the framework lazily:

```python
from mpsynth.exporters import build_qiskit, build_pennylane
qc = build_qiskit(result.circuit)
```

Notes on fidelity of the artefacts themselves: OpenQASM 2.0 has no global-phase
instruction, so the phase is recorded as a comment (unobservable for state preparation,
but it matters if you use the circuit as a *controlled* subroutine — use OpenQASM 3 or
Q# there). QIR angles are emitted as LLVM hex-float literals so nothing is lost to
decimal rounding.
