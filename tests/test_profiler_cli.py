"""Tests for the trade-off profiler and the command-line interface."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

from mpsynth import profile, synthesize
from mpsynth.cli import build_parser, load_vector, main
from mpsynth.datasets import GENERATORS, generate
from mpsynth.profiler import profile_mps
from mpsynth.mps import MPS

# ------------------------------------------------------------------- datasets


@pytest.mark.parametrize("name", sorted(GENERATORS))
def test_every_dataset_is_usable(name):
    vector = generate(name, 5)
    assert vector.shape == (32,)
    assert np.linalg.norm(vector) > 0
    assert np.all(np.isfinite(vector))


def test_datasets_are_deterministic():
    np.testing.assert_array_equal(generate("random", 6, seed=7), generate("random", 6, seed=7))
    assert not np.array_equal(generate("random", 6, seed=1), generate("random", 6, seed=2))


def test_dataset_validation():
    with pytest.raises(ValueError, match="unknown dataset"):
        generate("nope", 4)
    with pytest.raises(ValueError, match="n_qubits"):
        generate("gaussian", 0)


# ------------------------------------------------------------------- profiler


@pytest.fixture(scope="module")
def curve():
    return profile(generate("bimodal", 7), max_layers=5)


def test_curve_covers_every_layer_count(curve):
    assert [p.layers for p in curve.points] == [1, 2, 3, 4, 5]
    assert curve.n_qubits == 7
    assert curve.fidelity_exact


def test_curve_is_monotone(curve):
    assert [p.fidelity for p in curve.points] == sorted(p.fidelity for p in curve.points)
    assert [p.cnot for p in curve.points] == sorted(p.cnot for p in curve.points)


def test_profiled_circuits_reproduce_their_fidelities(curve):
    target = generate("bimodal", 7).astype(complex)
    target /= np.linalg.norm(target)
    for point in curve.points:
        circuit = curve.circuit_for(point.layers)
        measured = abs(np.vdot(target, circuit.statevector())) ** 2
        assert measured == pytest.approx(point.fidelity, abs=1e-9)
        assert circuit.cnot_count == point.cnot


def test_best_for(curve):
    assert curve.best_for(0.0) is curve.points[0]
    assert curve.best_for(1.0 + 1e-9) is None
    target = curve.points[-1].fidelity
    chosen = curve.best_for(target)
    assert chosen is not None and chosen.fidelity >= target


def test_circuit_for_rejects_unknown_layer_counts(curve):
    with pytest.raises(KeyError):
        curve.circuit_for(99)


def test_renderers(curve):
    table = curve.table(fidelity_target=0.99)
    assert "layers" in table and "CNOT" in table and "exact amplitude encoding" in table

    markdown = curve.to_markdown()
    assert markdown.startswith("| layers |")
    assert len(markdown.splitlines()) == len(curve.points) + 2

    payload = json.loads(curve.to_json())
    assert payload["n_qubits"] == 7
    assert len(payload["points"]) == len(curve.points)
    assert payload["points"][0]["infidelity"] == pytest.approx(
        1.0 - payload["points"][0]["fidelity"]
    )


def test_exact_baseline(curve):
    assert curve.exact_cnot_estimate == 2**7 - 2
    assert curve.points[0].cnot < curve.exact_cnot_estimate


def test_plot(tmp_path, curve):
    pytest.importorskip("matplotlib")
    path = curve.plot(str(tmp_path / "curve.png"))
    assert (tmp_path / "curve.png").exists()
    assert (tmp_path / "curve.png").stat().st_size > 0
    assert path.endswith("curve.png")


def test_profile_mps_entry_point():
    target, _ = MPS.from_statevector(generate("gaussian", 6) / np.linalg.norm(generate("gaussian", 6)))
    curve = profile_mps(target, max_layers=3)
    assert len(curve.points) == 3
    assert curve.points[-1].fidelity > 0.99


# ------------------------------------------------------------------------ CLI


def test_load_generator_spec():
    vector, label = load_vector("gaussian:6")
    assert vector.shape == (64,)
    assert "gaussian" in label
    assert np.array_equal(load_vector("random:5:3")[0], generate("random", 5, 3))


def test_load_unknown_generator():
    with pytest.raises(SystemExit, match="unknown dataset"):
        load_vector("nonsense:6")


def test_load_missing_file():
    with pytest.raises(SystemExit, match="input not found"):
        load_vector("/definitely/not/here.npy")


@pytest.mark.parametrize("suffix", [".npy", ".csv", ".txt", ".json"])
def test_load_file_formats(tmp_path, suffix):
    data = np.linspace(0.1, 1.0, 8)
    path = tmp_path / f"data{suffix}"
    if suffix == ".npy":
        np.save(path, data)
    elif suffix == ".json":
        path.write_text(json.dumps(data.tolist()))
    elif suffix == ".csv":
        path.write_text(",".join(repr(float(x)) for x in data))
    else:
        path.write_text("\n".join(repr(float(x)) for x in data))
    loaded, label = load_vector(str(path))
    np.testing.assert_allclose(np.real(loaded), data, atol=1e-12)
    assert str(path) in label


def test_load_npz(tmp_path):
    data = np.linspace(0.1, 1.0, 8)
    path = tmp_path / "data.npz"
    np.savez(path, values=data)
    np.testing.assert_allclose(np.real(load_vector(str(path))[0]), data, atol=1e-12)


def test_cli_synth_writes_a_file(tmp_path, capsys):
    out = tmp_path / "circuit.qasm"
    code = main(["synth", "gaussian:6", "-f", "0.99", "--format", "qasm3", "-o", str(out)])
    assert code == 0
    text = out.read_text()
    assert "OPENQASM 3.0;" in text
    assert "qubit[6] q;" in text
    assert "fidelity" in capsys.readouterr().err


def test_cli_synth_writes_to_stdout(capsys):
    assert main(["synth", "ghz:4", "-f", "0.99", "--format", "qasm2"]) == 0
    assert "OPENQASM 2.0;" in capsys.readouterr().out


@pytest.mark.parametrize("fmt", ["qasm2", "qasm3", "qir", "qsharp", "pennylane", "qiskit"])
def test_cli_supports_every_format(fmt, capsys):
    assert main(["synth", "w:4", "-f", "0.99", "--format", fmt]) == 0
    assert capsys.readouterr().out.strip()


def test_cli_reports_failure_to_reach_target(capsys):
    code = main(["synth", "random:7:5", "-f", "0.999999", "-L", "1", "--format", "qasm3"])
    assert code == 1
    assert "warning" in capsys.readouterr().err


def test_cli_profile(capsys):
    assert main(["profile", "lorentzian:6", "-L", "3", "-f", "0.99"]) == 0
    out = capsys.readouterr().out
    assert "layers" in out and "cheapest circuit" in out


def test_cli_profile_markdown_and_json(tmp_path, capsys):
    payload = tmp_path / "curve.json"
    assert main(["profile", "gaussian:6", "-L", "2", "--markdown", "--json", str(payload)]) == 0
    assert capsys.readouterr().out.lstrip().startswith("| layers |")
    assert len(json.loads(payload.read_text())["points"]) == 2


def test_cli_profile_reports_unreachable_target(capsys):
    assert main(["profile", "random:6:9", "-L", "2", "-f", "0.99999"]) == 0
    assert "no profiled circuit reaches" in capsys.readouterr().out


def test_cli_show(capsys):
    assert main(["show"]) == 0
    out = capsys.readouterr().out
    assert "gaussian" in out and "qasm3" in out


def test_cli_qubit_order_flag(capsys):
    assert main(["synth", "gaussian:5", "--qubit-order", "little", "--format", "qasm3"]) == 0
    assert "LITTLE-endian" in capsys.readouterr().out


def test_cli_requires_a_command():
    with pytest.raises(SystemExit):
        build_parser().parse_args([])


# ------------------------------------------- one-command workflow (profile -o)


def test_profile_exports_the_cheapest_qualifying_circuit(tmp_path, capsys):
    """`mpsynth profile ... -o` must do the whole job in one command."""
    out = tmp_path / "prepare.qasm"
    assert main(["profile", "gaussian:8", "-L", "4", "-f", "0.999", "-o", str(out)]) == 0
    printed = capsys.readouterr().out
    assert f"wrote {out}" in printed

    text = out.read_text()
    assert "OPENQASM 3.0;" in text and "qubit[8] q;" in text

    # The exported circuit must be exactly the cheapest point meeting the target.
    curve = profile(generate("gaussian", 8), max_layers=4)
    best = curve.best_for(0.999)
    assert best is not None
    assert text.count("cx ") == best.cnot
    assert f"{best.layers} layers, {best.cnot} CNOTs" in printed


def test_profile_export_honours_the_format_flag(tmp_path, capsys):
    out = tmp_path / "prepare.qs"
    assert main(["profile", "ghz:5", "-L", "2", "-f", "0.99", "-o", str(out),
                 "--format", "qsharp"]) == 0
    assert "namespace MPSynth {" in out.read_text()


def test_profile_export_refuses_when_no_circuit_meets_the_target(tmp_path, capsys):
    out = tmp_path / "prepare.qasm"
    code = main(["profile", "random:6:9", "-L", "2", "-f", "0.99999", "-o", str(out)])
    assert code == 1
    assert not out.exists(), "must not write a circuit that misses the target"
    assert "nothing written" in capsys.readouterr().out


# ------------------------------------------------------- the log-accuracy bar


def test_bar_distinguishes_rows_that_linear_fidelity_would_not(curve):
    """Every row rounds to F ~ 1.0; the bar must still separate them."""
    bars = [curve._bar(p, curve._bar_scale()) for p in curve.points]
    assert len(set(bars)) > 1, "a linear fidelity bar would collapse to one value"
    filled = [b.count("#") for b in bars]
    assert filled == sorted(filled), "more layers must never shrink the bar"
    assert all(len(b) == curve._BAR_WIDTH for b in bars)


def test_bar_scale_is_anchored_above_the_best_point(curve):
    scale = curve._bar_scale()
    assert scale >= max(curve._nines(p) for p in curve.points)
    assert curve.table().splitlines()[0].endswith("nines)")


def test_bar_handles_a_perfect_circuit():
    exact = profile(generate("ghz", 5), max_layers=2)
    assert exact.points[0].fidelity == pytest.approx(1.0, abs=1e-12)
    assert exact._bar(exact.points[0], exact._bar_scale()).endswith("#")


def test_bar_handles_a_hopeless_circuit():
    stuck = profile(generate("random", 6, seed=4), max_layers=2)
    bars = [stuck._bar(p, stuck._bar_scale()) for p in stuck.points]
    assert all(b.count("#") <= stuck._BAR_WIDTH for b in bars)
    assert "nine)" in stuck.table().splitlines()[0] or "nines)" in stuck.table().splitlines()[0]


# --------------------------------------- flat invocation: `mpsynth <input> [flags]`


def test_bare_input_defaults_to_profile(tmp_path, capsys):
    """`mpsynth data.csv -o out.qasm` must work with no subcommand, qizil-style."""
    out = tmp_path / "prepare.qasm"
    assert main(["gaussian:8", "-L", "3", "-f", "0.999", "-o", str(out)]) == 0
    captured = capsys.readouterr()
    assert "layers" in captured.out and "cheapest circuit" in captured.out
    assert "OPENQASM 3.0;" in out.read_text()


def test_bare_input_matches_the_explicit_subcommand(capsys):
    main(["gaussian:7", "-L", "2"])
    implicit = capsys.readouterr().out
    main(["profile", "gaussian:7", "-L", "2"])
    assert implicit == capsys.readouterr().out


def test_explicit_subcommands_still_win_over_the_default(capsys):
    assert main(["show"]) == 0
    assert "datasets" in capsys.readouterr().out


def test_default_command_insertion():
    from mpsynth.cli import _insert_default_command

    assert _insert_default_command(["data.npy", "-f", "0.9"]) == ["profile", "data.npy", "-f", "0.9"]
    assert _insert_default_command(["profile", "x"]) == ["profile", "x"]
    assert _insert_default_command(["synth", "x"]) == ["synth", "x"]
    assert _insert_default_command(["show"]) == ["show"]
    assert _insert_default_command(["--help"]) == ["--help"]
    assert _insert_default_command([]) == []


def test_banner_goes_to_stderr_so_stdout_stays_pasteable(capsys):
    main(["profile", "gaussian:6", "-L", "2", "--markdown"])
    captured = capsys.readouterr()
    assert captured.out.startswith("| layers |")
    assert "mpsynth" in captured.err


# ------------------------------------------------ running straight from a clone


def test_repo_launcher_is_executable_and_runs(tmp_path):
    """The `./mpsynth` launcher must work with nothing installed."""
    import os
    import subprocess

    root = Path(__file__).resolve().parent.parent
    launcher = root / "mpsynth"
    assert launcher.is_file(), "repo-root launcher is missing"
    assert os.access(launcher, os.X_OK), "launcher is not executable (chmod +x mpsynth)"

    out = tmp_path / "prepare.qasm"
    proc = subprocess.run(
        [sys.executable, str(launcher), "gaussian:8", "-L", "2", "-o", str(out)],
        capture_output=True, text=True, cwd=tmp_path,
    )
    assert proc.returncode == 0, proc.stderr
    assert "cheapest circuit" in proc.stdout
    assert "OPENQASM 3.0;" in out.read_text()


def test_python_dash_m_entry_point(tmp_path):
    import subprocess

    root = Path(__file__).resolve().parent.parent
    proc = subprocess.run(
        [sys.executable, "-m", "mpsynth", "show"],
        capture_output=True, text=True, cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(root / "src")},
    )
    assert proc.returncode == 0, proc.stderr
    assert "gaussian" in proc.stdout


def test_shipped_examples_are_loadable_and_synthesisable():
    root = Path(__file__).resolve().parent.parent
    files = sorted((root / "examples").glob("*.csv"))
    assert files, "no example vectors are committed"
    for path in files:
        vector, label = load_vector(str(path))
        assert vector.size in (1024, 4096)
        assert np.all(np.isfinite(vector))
        shallow = synthesize(vector, fidelity=1.0, max_layers=1)
        deeper = synthesize(vector, fidelity=1.0, max_layers=6)
        assert deeper.fidelity >= shallow.fidelity, path.name
        assert deeper.fidelity > 0.95, f"{path.name}: {deeper.fidelity}"
