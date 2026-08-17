# MPSynth

**Shallow, approximate quantum state-preparation circuits from classical vectors, via Matrix Product States.**

Loading an `N`-dimensional classical vector into an `n = log₂N` qubit register exactly
(amplitude encoding) costs **O(2ⁿ)** gates. On current hardware and simulators that
spends the entire coherence budget before any actual computation starts — the data is
destroyed by noise on the way in.

MPSynth compresses the vector into a Matrix Product State, truncates its bond dimension
to a fidelity tolerance *you* choose, and synthesises the result as a stack of
nearest-neighbour two-qubit staircases. Depth grows **linearly** in `n`, not
exponentially.

```
18 qubits (262,144 amplitudes), smooth input, fidelity 0.9999:
   exact amplitude encoding   ~262,142 CNOTs
   MPSynth, 4 layers               161 CNOTs      1,600x fewer
```
## Quick start

Two commands. No install, no virtualenv, no `pip install -e .`:

```console
git clone https://github.com/YOUR-ORG/mpsynth && cd mpsynth

./mpsynth examples/lognormal_4096.csv -f 0.999 -o prepare.qasm
```

```
mpsynth 0.1.0  examples/lognormal_4096.csv  (fidelity >= 0.999)

layers   CNOT  depth  2q-depth   fidelity  infidelity  accuracy (full bar = 5 nines)
----------------------------------------------------------------------------------------
     1     30    109        30   0.998335   1.665e-03  ##########........
     2     60    133        36   0.999494   5.057e-04  ############......  <-- meets target
     3     89    151        41   0.999749   2.507e-04  #############.....  <-- meets target
     4    118    176        48   0.999899   1.013e-04  ##############....  <-- meets target
     5    144    195        54   0.999961   3.884e-05  ################..  <-- meets target
     6    169    216        60   0.999971   2.882e-05  ################..  <-- meets target
     7    197    237        66   0.999979   2.077e-05  #################.  <-- meets target
     8    224    256        71   0.999984   1.584e-05  #################.  <-- meets target

exact amplitude encoding baseline: ~4094 CNOTs (12 qubits, bond dimension 48)

cheapest circuit at fidelity >= 0.999: 2 layer(s), 60 CNOTs, 2q-depth 36 (68.2x fewer CNOTs than exact encoding)
wrote prepare.qasm (qasm3, 2 layers, 60 CNOTs)
```

That is the whole setup. `./mpsynth` works from any directory
(`/path/to/mpsynth/mpsynth data.csv -o out.qasm`), and on Windows as
`python mpsynth data.csv -o out.qasm`.

**60 CNOTs instead of ~4094**, at 99.95% fidelity — a lognormal asset-price density,
the payload of a quantum option pricer. The bar is log-scaled on `1 - F`, so diminishing
returns read straight off it: layers 7-8 buy less than one extra nine for 164 more CNOTs.

**One dependency: numpy.** MPSynth's core is a singular value decomposition, so unlike a
pure-stdlib tool there is no honest way around it. If numpy is missing but you have
[`uv`](https://docs.astral.sh/uv/), `./mpsynth` re-runs itself through
`uv run --with numpy` and you never notice; otherwise it tells you the one line to run.

### More

```console
./mpsynth examples/gaussian_1024.csv          # smooth density, 10 qubits
./mpsynth examples/chirp_1024.csv -L 6        # oscillatory signal -- deliberately harder
./mpsynth gaussian:16                         # built-in generator, no data file
./mpsynth random:10                           # an input that does NOT compress
./mpsynth data.csv -o out.py --format qiskit  # qasm2 | qasm3 | qir | qsharp | pennylane | qiskit
./mpsynth synth data.csv -f 0.99 -o out.qasm  # stop at the target instead of profiling
./mpsynth show                                # datasets and export targets
```

Inputs may be `.csv`, `.txt`, `.npy`, `.npz`, `.json`, or a built-in generator spec such
as `gaussian:12` (name : qubits [: seed]). Vectors are normalised and zero-padded to a
power of two automatically. With no subcommand, `./mpsynth <input>` profiles the
trade-off; `-o` also writes the cheapest circuit that meets `-f`.

---

## Install (optional)

Nothing here is required — `./mpsynth` in a clone is fully functional. Install only if
you want `mpsynth` on your PATH without the clone path, or the library in your own code:

```console
uv tool install .           # or: pipx install .
pip install -e ".[dev]"     # into an active virtualenv, + pytest/scipy/matplotlib
```

## Use it as a library

```python
import numpy as np
from mpsynth import synthesize, export

x = np.linspace(-4, 4, 1024)
data = np.exp(-x**2 / 2)                      # any classical vector

result = synthesize(data, fidelity=0.99)      # <- your tolerance, not ours
print(result.report())

open("prepare.qasm", "w").write(export(result.circuit, "qasm3"))
```

```
qubits            10
layers            1
fidelity          0.998696
infidelity        1.304e-03
depth             84
2-qubit depth     24
CNOT count        24
1-qubit gates     125
total gates       149
input bond dim    11
```

24 CNOTs instead of ~1022. Single-qubit gates are cheap and fast on real hardware;
`cnot` and `two_qubit_depth` are the numbers that decide whether the state survives.

---

## How it works

Two facts carry the whole engine.

**1. A bond-dimension-2 MPS can be prepared *exactly* by one staircase.**
Put the state in right-canonical form. Each site tensor is then an isometry
`ℂ^{D_left} → ℂ²_physical ⊗ ℂ^{D_right}`. Embed it in a two-qubit unitary that consumes
the bond carried on qubit `k` plus a fresh `|0⟩` on qubit `k+1`, and emits the physical
value on `k` and the next bond on `k+1`. Sweeping left to right costs `n-1` two-qubit
gates and one single-qubit gate — depth **O(n)**.

```
q0 ──┤G₀├────────────────────
     └┬─┘
q1 ───┴──┤G₁├───────────────      each Gₖ consumes the bond on qubit k
         └┬─┘                     and hands the next one to qubit k+1
q2 ───────┴──┤G₂├───────────
             └┬─┘
q3 ───────────┴──────┤ V ├──
```

**2. Higher bond dimension is recovered by stacking staircases.**
Approximate the target by its best bond-2 MPS, synthesise that exactly as `U₁`, apply
`U₁†` to the target, and repeat on what is left. After `L` layers

```
|ψ⟩  ≈  U₁ U₂ … U_L |0…0⟩
```

and the residual `U_L†…U₁†|ψ⟩` converges to `|0…0⟩`. `L` is the knob you trade against
fidelity. This is the disentangling scheme of Ran, *Phys. Rev. A* **101**, 032310 (2020).

Each two-qubit gate is then lowered to `{RZ, RY, CX}` by an exact Cartan (KAK)
decomposition, using the fewest CX gates its Weyl-chamber coordinates allow.

### Modules

| Module | Role |
| --- | --- |
| `mpsynth.mps` | MPS / tensor-train core: SVD decomposition, canonical forms, gate application |
| `mpsynth.linalg` | Truncated SVD with an explicit discarded-weight budget; isometry completion |
| `mpsynth.decompose` | KAK decomposition; canonical-gate circuits at 0/1/2/3 CX |
| `mpsynth.synthesis` | Bond-2 staircase and the iterative disentangler |
| `mpsynth.circuit` | Backend-independent IR, metrics, simulation, peephole optimiser |
| `mpsynth.exporters` | OpenQASM 2/3, QIR, Q#, PennyLane, Qiskit |
| `mpsynth.profiler` | Fidelity-vs-depth trade-off curves (text, markdown, JSON, PNG) |

---

## The trade-off profiler

One synthesis run produces every shallower circuit as a by-product, so the whole curve
costs one pass.

```bash
$ ./mpsynth gaussian:10 -L 6 -f 0.999
```

```
layers   CNOT  depth  2q-depth   fidelity  infidelity  accuracy (full bar = 5 nines)
----------------------------------------------------------------------------------------
     1     24     85        24   0.998696   1.304e-03  ##########........
     2     46    102        29   0.999329   6.713e-04  ###########.......  <-- meets target
     3     68    126        34   0.999841   1.587e-04  ##############....  <-- meets target
     4     89    142        39   0.999916   8.363e-05  ###############...  <-- meets target
     5    110    168        46   0.999952   4.781e-05  ################..  <-- meets target
     6    130    187        51   0.999969   3.112e-05  ################..  <-- meets target

exact amplitude encoding baseline: ~1022 CNOTs (10 qubits, bond dimension 11)

cheapest circuit at fidelity >= 0.999: 2 layer(s), 46 CNOTs, 2q-depth 29 (22.2x fewer CNOTs than exact encoding)
```

Add `-o prepare.qasm` and it also writes the cheapest qualifying circuit, so profiling
and synthesis are a single command.

```python
from mpsynth import profile
curve = profile(data, max_layers=8)
best = curve.best_for(0.999)          # shallowest circuit meeting the target
circuit = curve.circuit_for(best.layers)
curve.plot("tradeoff.png")            # needs matplotlib
print(curve.to_markdown(), curve.to_json())
```

---

## Measured results

`python examples/benchmark.py`. **10 qubits, fidelity target 0.99**, baseline ~1022 CNOTs:

| input | kind | bond dim | layers | CNOT | 2q-depth | fidelity | vs exact |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `product` | product state | 1 | 1 | **0** | 0 | 1.000000 | — |
| `ghz` | GHZ state | 2 | 1 | 17 | 17 | 1.000000 | 60x |
| `w` | W state | 2 | 1 | 21 | 21 | 1.000000 | 49x |
| `damped` | damped oscillation | 2 | 1 | 23 | 23 | 1.000000 | 44x |
| `gaussian` | smooth analytic | 11 | 1 | 24 | 24 | 0.998696 | 43x |
| `lognormal` | skewed density | 9 | 1 | 24 | 24 | 0.997502 | 43x |
| `lorentzian` | heavy shoulders | 12 | 1 | 24 | 24 | 0.999357 | 43x |
| `heavytail` | power law | 9 | 1 | 24 | 24 | 0.999937 | 43x |
| `bimodal` | two-peak density | 13 | 3 | 70 | 35 | 0.994648 | 15x |
| `random` | i.i.d. noise | 32 | 12 | 242 | 79 | **0.330** | *target not reached* |

**Scaling** on a smooth input, 4 layers — cost is essentially flat while the exact
baseline doubles every qubit:

| qubits | amplitudes | CNOT | 2q-depth | fidelity | exact CNOTs | time |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10 | 1,024 | 89 | 39 | 0.999916 | 1,022 | 0.04s |
| 12 | 4,096 | 113 | 45 | 0.999917 | 4,094 | 0.06s |
| 14 | 16,384 | 134 | 51 | 0.999917 | 16,382 | 0.10s |
| 16 | 65,536 | 149 | 57 | 0.999917 | 65,534 | 0.21s |
| 18 | 262,144 | 161 | 63 | 0.999917 | 262,142 | 0.69s |

### When this does *not* help

**MPSynth compresses structure, and i.i.d. random noise has none.** A Haar-random
vector has near-maximal entanglement across every cut, its MPS bond dimension is the
full `2^{n/2}`, and no shallow circuit can reproduce it — that is information-theoretic,
not a limitation of the implementation. In the table above the random inputs stall
around F ≈ 0.33 and the tool says so rather than quietly reporting a good-looking
number. Real data — densities, signals, images, smooth functions, financial
distributions — is the regime this is built for.

---

## Correctness

Approximation is the *point* of this tool, so everything that is not the approximation
is held to machine precision. 233 tests, `pytest -q`.

- **The gate decompositions are re-derived, not trusted.** Every canonical-gate identity
  is proved numerically in `tests/test_decompose.py` from the Bell-basis diagonalisation
  of `{XX, YY, ZZ}`, and each construction is verified against its target *at synthesis
  time* before being emitted — a wrong circuit cannot leave the library silently.
- **Optimal CX counts.** Verified against every known equivalence class: local gates 0,
  CX/CZ 1, iSWAP 2, SWAP and generic U(4) 3.
- **Unitarity and normalisation.** SVD truncation loses norm; every truncation is
  followed by renormalisation, and tests assert `⟨ψ|ψ⟩ = 1` to 1e-12 and `U†U = I` on the
  emitted circuit — including under aggressive truncation.
- **Exports are executed, not just pattern-matched.** The Qiskit and PennyLane artefacts
  are run in the real frameworks and their statevectors compared against MPSynth's own
  simulation; OpenQASM 2/3 are reloaded through Qiskit's parsers.
- **The reported fidelity is reproducible from the emitted circuit alone**, and is
  measured against the *true* input vector rather than the truncated MPS stand-in.

```bash
pytest -q          # 233 passed
```

---

## Qubit ordering — read this once

Frameworks disagree about which end of a basis label qubit 0 lives on. Getting it wrong
gives you a bit-reversed state at a plausible-looking fidelity, which is the easiest way
to misuse a state-preparation tool. So MPSynth makes it explicit:

| `qubit_order` | qubit 0 is | verified against |
| --- | --- | --- |
| `"big"` (default) | the **most** significant index bit | PennyLane `qml.state()`, textbook/OpenQASM reading |
| `"little"` | the **least** significant index bit | Qiskit `Statevector`, Aer |

```python
result = synthesize(data, qubit_order="little")   # now Statevector(qc) == your data
```

Every exported artefact stamps its convention into its own header comment. And
`result.amplitudes()` always returns the prepared state re-indexed to line up
element-for-element with the vector you passed in — padding and ordering undone,
original norm restored:

```python
np.allclose(result.amplitudes(), data, atol=1e-3)
```

---

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

---

## API

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

---

## License

Apache-2.0.
