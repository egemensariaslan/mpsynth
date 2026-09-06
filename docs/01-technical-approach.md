# Technical approach

How MPSynth turns a classical vector into a shallow circuit, and the convention every
framework disagrees on if you don't pin it down.

## Two facts carry the whole engine

**1. A bond-dimension-2 Matrix Product State can be prepared *exactly* by one staircase.**
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
fidelity. This is the disentangling scheme of Ran, *Phys. Rev. A* **101**, 032310 (2020)
(full citation in the [README references](../README.md#10-references)).

Each two-qubit gate is then lowered to `{RZ, RY, CX}` by an exact Cartan (KAK)
decomposition, using the fewest CX gates its Weyl-chamber coordinates allow: 0 for local
gates, 1 for the CX-equivalence class, 2 when one coordinate vanishes, 3 in general.

## Modules

| Module | Role |
| --- | --- |
| `mpsynth.mps` | MPS / tensor-train core: SVD decomposition, canonical forms, gate application |
| `mpsynth.linalg` | Truncated SVD with an explicit discarded-weight budget; isometry completion |
| `mpsynth.decompose` | KAK decomposition; canonical-gate circuits at 0/1/2/3 CX |
| `mpsynth.synthesis` | Bond-2 staircase and the iterative disentangler |
| `mpsynth.circuit` | Backend-independent IR, metrics, simulation, peephole optimiser |
| `mpsynth.exporters` | OpenQASM 2/3, QIR, Q#, PennyLane, Qiskit |
| `mpsynth.profiler` | Fidelity-vs-depth trade-off curves (text, markdown, JSON, PNG) |
| `mpsynth.analysis` | Entanglement spectra, verification, JSON payloads for the UI |
| `mpsynth.ui` | Local workbench: stdlib HTTP server + hand-written HTML/CSS/SVG |

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
