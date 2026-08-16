"""Command-line interface: ``mpsynth synth``, ``mpsynth profile``, ``mpsynth show``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from . import __version__
from .datasets import GENERATORS, generate
from .exporters import EXTENSIONS, FORMATS, export
from .profiler import profile
from .synthesis import synthesize

__all__ = ["main"]


# ------------------------------------------------------------------ input loading


def load_vector(spec: str) -> tuple[np.ndarray, str]:
    """Load a vector from a file path or a ``name:qubits[:seed]`` generator spec.

    Supported files: ``.npy``, ``.npz`` (first array), ``.json``, and any
    whitespace/comma separated text (``.csv``, ``.txt``, ``.dat``).
    """
    if ":" in spec and not Path(spec).exists():
        parts = spec.split(":")
        name = parts[0]
        if name not in GENERATORS:
            raise SystemExit(
                f"unknown dataset {name!r}; choose from {', '.join(sorted(GENERATORS))}"
            )
        n = int(parts[1]) if len(parts) > 1 else 8
        seed = int(parts[2]) if len(parts) > 2 else 0
        return generate(name, n, seed), f"{name} ({n} qubits)"

    path = Path(spec)
    if not path.exists():
        raise SystemExit(f"input not found: {spec}")

    suffix = path.suffix.lower()
    if suffix == ".npy":
        data = np.load(path)
    elif suffix == ".npz":
        with np.load(path) as archive:
            data = archive[archive.files[0]]
    elif suffix == ".json":
        raw = json.loads(path.read_text())
        data = np.asarray(
            [complex(x["re"], x["im"]) if isinstance(x, dict) else x for x in raw]
        )
    else:
        text = path.read_text()
        delimiter = "," if "," in text else None
        data = np.loadtxt(path, delimiter=delimiter, dtype=complex)

    return np.asarray(data).reshape(-1), str(path)


# ---------------------------------------------------------------------- commands


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "input",
        help="path to .npy/.npz/.json/.csv/.txt, or a generator spec like 'gaussian:10'",
    )
    parser.add_argument(
        "--chi", type=int, default=None,
        help="bond-dimension cap when decomposing the input (default: exact)",
    )
    parser.add_argument(
        "--residual-chi", type=int, default=None,
        help="bond-dimension cap for the disentangling residual",
    )
    parser.add_argument(
        "--tol", type=float, default=0.0,
        help="relative discarded-weight budget per bond (default: 0)",
    )
    parser.add_argument(
        "--qubit-order", choices=["big", "little"], default="big",
        help="big: qubit 0 is the most significant index bit (OpenQASM reading); "
             "little: Qiskit Statevector reading",
    )
    parser.add_argument(
        "--no-optimize", action="store_true",
        help="skip single-qubit gate fusion",
    )


def cmd_synth(args: argparse.Namespace) -> int:
    vector, label = load_vector(args.input)
    result = synthesize(
        vector,
        fidelity=args.fidelity,
        max_layers=args.max_layers,
        chi_max=args.chi,
        residual_chi=args.residual_chi,
        tol=args.tol,
        qubit_order=args.qubit_order,
        optimize=not args.no_optimize,
    )

    print(f"input             {label}", file=sys.stderr)
    print(result.report(), file=sys.stderr)
    if not result.reached_target:
        print(
            f"\nwarning: fidelity {result.fidelity:.6f} < target {args.fidelity}; "
            f"raise --max-layers or lower --fidelity",
            file=sys.stderr,
        )

    text = export(result.circuit, args.format)
    if args.output:
        out = Path(args.output)
        out.write_text(text)
        print(f"\nwrote {out}", file=sys.stderr)
    else:
        sys.stdout.write(text)
    return 0 if result.reached_target else 1


def cmd_profile(args: argparse.Namespace) -> int:
    vector, label = load_vector(args.input)
    # Banner on stderr so stdout stays pure data: --markdown output must be
    # pasteable as-is, and `mpsynth data.csv > table.txt` should capture only the table.
    print(
        f"mpsynth {__version__}  {label}  (fidelity >= {args.fidelity})\n",
        file=sys.stderr,
    )
    curve = profile(
        vector,
        max_layers=args.max_layers,
        chi_max=args.chi,
        residual_chi=args.residual_chi,
        tol=args.tol,
        qubit_order=args.qubit_order,
        optimize=not args.no_optimize,
    )

    if args.json:
        Path(args.json).write_text(curve.to_json())
    if args.markdown:
        print(curve.to_markdown())
    else:
        print(curve.table(fidelity_target=args.fidelity))

    best = curve.best_for(args.fidelity)
    print()
    if best is None:
        print(
            f"no profiled circuit reaches fidelity {args.fidelity}; "
            f"best is {curve.points[-1].fidelity:.6f} at {curve.points[-1].cnot} CNOTs"
        )
    else:
        saving = curve.exact_cnot_estimate / max(best.cnot, 1)
        print(
            f"cheapest circuit at fidelity >= {args.fidelity}: {best.layers} layer(s), "
            f"{best.cnot} CNOTs, 2q-depth {best.two_qubit_depth} "
            f"({saving:.1f}x fewer CNOTs than exact encoding)"
        )

    # One command should be able to finish the job: emit the circuit it just picked.
    if args.output:
        if best is None:
            print(f"nothing written to {args.output}: no circuit met the target")
            return 1
        out = Path(args.output)
        out.write_text(export(curve.circuit_for(best.layers), args.format))
        print(f"wrote {out} ({args.format}, {best.layers} layers, {best.cnot} CNOTs)")

    if args.plot:
        try:
            curve.plot(args.plot, title=f"MPSynth trade-off -- {label}")
            print(f"wrote {args.plot}")
        except ImportError as exc:
            print(f"plot skipped: {exc}", file=sys.stderr)
    if args.json:
        print(f"wrote {args.json}")
    return 0


def cmd_ui(args: argparse.Namespace) -> int:
    """Serve the interactive workbench."""
    from .ui import serve

    return serve(port=args.port, open_browser=not args.no_browser)


def cmd_show(args: argparse.Namespace) -> int:
    """List datasets and export formats."""
    print("datasets (use as 'name:qubits[:seed]'):")
    for name in sorted(GENERATORS):
        print(f"  {name}")
    print("\nexport formats:")
    for name in sorted(FORMATS):
        print(f"  {name:<11s} -> {EXTENSIONS[name]}")
    return 0


# ------------------------------------------------------------------------- parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mpsynth",
        description=(
            "Synthesise shallow approximate quantum state-preparation circuits "
            "from classical vectors via Matrix Product States."
        ),
    )
    parser.add_argument("--version", action="version", version=f"mpsynth {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    synth = sub.add_parser("synth", help="synthesise a circuit and export it")
    _add_common(synth)
    synth.add_argument("-f", "--fidelity", type=float, default=0.98,
                       help="target |<psi_exact|psi_approx>|^2 (default: 0.98)")
    synth.add_argument("-L", "--max-layers", type=int, default=16,
                       help="maximum entangling staircases (default: 16)")
    synth.add_argument("--format", choices=sorted(FORMATS), default="qasm3",
                       help="export target (default: qasm3)")
    synth.add_argument("-o", "--output", help="write to a file instead of stdout")
    synth.set_defaults(func=cmd_synth)

    prof = sub.add_parser("profile", help="report the fidelity/depth trade-off curve")
    _add_common(prof)
    prof.add_argument("-f", "--fidelity", type=float, default=0.98,
                      help="fidelity target to highlight (default: 0.98)")
    prof.add_argument("-L", "--max-layers", type=int, default=8,
                      help="profile 1..L layers (default: 8)")
    prof.add_argument("-o", "--output",
                      help="also export the cheapest circuit that meets --fidelity")
    prof.add_argument("--format", choices=sorted(FORMATS), default="qasm3",
                      help="export target for --output (default: qasm3)")
    prof.add_argument("--json", help="also write the curve as JSON")
    prof.add_argument("--plot", help="also write a PNG figure (needs matplotlib)")
    prof.add_argument("--markdown", action="store_true", help="emit a markdown table")
    prof.set_defaults(func=cmd_profile)

    ui = sub.add_parser("ui", help="open the interactive workbench in a browser")
    ui.add_argument("--port", type=int, default=8765, help="port to serve on (default: 8765)")
    ui.add_argument("--no-browser", action="store_true", help="do not open a browser")
    ui.set_defaults(func=cmd_ui)

    show = sub.add_parser("show", help="list built-in datasets and export formats")
    show.set_defaults(func=cmd_show)

    return parser


#: Subcommand names, so a bare input path can be told apart from an explicit verb.
COMMANDS = ("synth", "profile", "show", "ui")

#: What `mpsynth <input> ...` means with no verb given: profile the trade-off and, with
#: -o, export the cheapest circuit that meets the target.
DEFAULT_COMMAND = "profile"


def _insert_default_command(argv: list[str]) -> list[str]:
    """Allow ``mpsynth data.npy -f 0.99 -o out.qasm`` with no subcommand."""
    if not argv:
        return argv
    first = argv[0]
    if first in COMMANDS or first in ("-h", "--help", "--version"):
        return argv
    return [DEFAULT_COMMAND, *argv]


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(_insert_default_command(argv))
    try:
        return int(args.func(args))
    except (ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
