"""MPSynth -- shallow approximate quantum state preparation via Matrix Product States.

Loading an ``N``-dimensional classical vector into an ``n = log2(N)`` qubit register
exactly costs ``O(2^n)`` gates, which spends the whole coherence budget before any
computation starts.  MPSynth compresses the vector into a Matrix Product State,
truncates its bond dimension to a user-chosen fidelity tolerance, and synthesises the
result as a stack of nearest-neighbour two-qubit staircases whose depth grows *linearly*
in ``n`` rather than exponentially.

    >>> import numpy as np
    >>> from mpsynth import synthesize, export
    >>> x = np.linspace(-4, 4, 256)
    >>> result = synthesize(np.exp(-x**2 / 2), fidelity=0.99)
    >>> result.fidelity > 0.99
    True
    >>> qasm = export(result.circuit, "qasm3")
"""

from __future__ import annotations

from .circuit import Circuit, Gate
from .decompose import kak_decomposition, two_qubit_ops
from .exporters import EXTENSIONS, FORMATS, export
from .mps import MPS
from .profiler import TradeoffPoint, TradeoffProfile, profile, profile_mps
from .synthesis import SynthesisResult, bond2_layer, synthesize, synthesize_mps

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # core data types
    "MPS",
    "Circuit",
    "Gate",
    # synthesis
    "synthesize",
    "synthesize_mps",
    "SynthesisResult",
    "bond2_layer",
    # decomposition
    "kak_decomposition",
    "two_qubit_ops",
    # profiling
    "profile",
    "profile_mps",
    "TradeoffProfile",
    "TradeoffPoint",
    # export
    "export",
    "FORMATS",
    "EXTENSIONS",
]
