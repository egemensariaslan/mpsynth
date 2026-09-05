"""Tests for the analysis payloads and the local UI server."""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import numpy as np
import pytest
from conftest import random_state

from mpsynth.analysis import MAX_PLOT_POINTS, analyze, layer_detail
from mpsynth.datasets import generate
from mpsynth.mps import MPS
from mpsynth.ui.server import _State, build_handler

# ------------------------------------------------------- entanglement primitives


def test_ghz_carries_exactly_one_bit_across_every_cut():
    target, _ = MPS.from_statevector(generate("ghz", 6) / np.linalg.norm(generate("ghz", 6)))
    np.testing.assert_allclose(target.entanglement_entropy(), np.ones(5), atol=1e-12)


def test_product_state_has_zero_entropy(rng):
    vec = np.ones(1)
    for _ in range(6):
        vec = np.kron(vec, rng.normal(size=2))
    target, _ = MPS.from_statevector(vec / np.linalg.norm(vec))
    entropy = target.entanglement_entropy()
    np.testing.assert_allclose(entropy, 0.0, atol=1e-10)
    assert np.all(entropy >= 0.0), "entropy must never be reported negative"


def test_random_state_approaches_the_maximum(rng):
    n = 8
    target, _ = MPS.from_statevector(random_state(2**n, rng))
    entropy = target.entanglement_entropy()
    bound = np.array([min(k + 1, n - k - 1) for k in range(n - 1)])
    assert np.all(entropy <= bound + 1e-9), "entropy cannot exceed the Page bound"
    assert entropy[n // 2 - 1] > 0.7 * bound[n // 2 - 1]


def test_schmidt_values_are_normalised_and_descending(rng):
    target, _ = MPS.from_statevector(random_state(2**7, rng))
    for s in target.schmidt_values():
        assert np.linalg.norm(s) == pytest.approx(1.0, abs=1e-10)
        assert np.all(np.diff(s) <= 1e-12), "Schmidt values must be descending"


def test_schmidt_values_do_not_mutate_the_state(rng):
    psi = random_state(2**6, rng)
    target, _ = MPS.from_statevector(psi)
    target.schmidt_values()
    np.testing.assert_allclose(target.to_statevector(), psi, atol=1e-11)


# -------------------------------------------------------------- analysis payload


@pytest.fixture(scope="module")
def analysis():
    return analyze(generate("lognormal", 10), label="lognormal:10", max_layers=4)


def test_summary_is_json_serialisable(analysis):
    payload = json.loads(json.dumps(analysis.summary()))
    assert payload["meta"]["n_qubits"] == 10
    assert len(payload["tradeoff"]) == 4
    assert len(payload["entanglement"]["entropy"]) == 9
    assert len(payload["entanglement"]["max_entropy"]) == 9


def test_summary_reports_the_decimation_stride():
    """Plot series are subsampled; the stride must travel with them or axes lie."""
    big = analyze(generate("gaussian", 13), max_layers=1)
    payload = big.summary()
    stride = payload["input"]["stride"]
    assert stride > 1, "a 8192-point vector must be decimated"
    assert len(payload["input"]["values"]) <= MAX_PLOT_POINTS
    # stride * plotted points must recover the true length, to within one stride
    assert abs(len(payload["input"]["values"]) * stride - 2**13) < stride


def test_detail_amplitudes_line_up_with_the_input(analysis):
    detail = layer_detail(analysis, 2)
    amps = detail["amplitudes"]
    assert len(amps["produced"]) == len(amps["target"]) == len(amps["residual"])
    assert amps["max_abs_error"] >= amps["rms_error"] >= 0
    # The plotted residual must be exactly the difference of the two plotted series,
    # otherwise the lower chart does not explain the upper one.
    diff = np.array(amps["produced"]) - np.array(amps["target"])
    np.testing.assert_allclose(diff, amps["residual"], atol=1e-12)
    # ...while the headline error is the full complex difference, which can only be
    # larger than anything visible in the real-part plot.
    assert amps["max_abs_error"] >= np.max(np.abs(diff)) - 1e-12


def test_detail_circuit_payload_matches_the_circuit(analysis):
    detail = layer_detail(analysis, 2)
    circuit = analysis.curve.circuit_for(2)
    payload = detail["circuit"]
    assert payload["total_gates"] == len(circuit.gates)
    assert payload["n_qubits"] == circuit.n_qubits
    assert payload["columns"] == circuit.depth, "ASAP columns must equal circuit depth"
    for gate in payload["gates"]:
        assert 0 <= min(gate["qubits"]) and max(gate["qubits"]) < circuit.n_qubits


def test_verification_passes_for_a_compressible_input(analysis):
    detail = layer_detail(analysis, 2)
    verification = detail["verification"]
    assert verification["all_passed"], [c for c in verification["checks"] if not c["ok"]]
    names = {c["name"] for c in verification["checks"]}
    assert "1 − |⟨dense|tensor⟩|²" in names
    assert "max |U†U − I|" in names
    for check in verification["checks"]:
        # every row must carry its own bound, so the table shows evidence not a badge
        assert check["bound"] and check["limit"] > 0
        assert check["ok"] == (check["residual"] < check["limit"])


def test_verification_flags_an_incompressible_input():
    """The UI must say so rather than showing a flattering number."""
    hopeless = analyze(generate("random", 8, seed=2), max_layers=2)
    detail = layer_detail(hopeless, 2)
    checks = {c["name"]: c for c in detail["verification"]["checks"]}
    saturation = checks["S̄ / S_max"]
    assert not saturation["ok"] and saturation["residual"] > 0.7
    # ...while the circuit itself is still perfectly valid
    assert checks["|‖ψ‖ − 1|"]["ok"]
    assert checks["max |U†U − I|"]["ok"]
    assert not detail["verification"]["all_passed"]


def test_reported_fidelity_matches_the_curve(analysis):
    for point in analysis.curve.points:
        detail = layer_detail(analysis, point.layers)
        assert detail["point"]["fidelity"] == pytest.approx(point.fidelity)
        assert detail["metrics"]["cnot"] == point.cnot


# --------------------------------------------------------------------- UI server


@pytest.fixture(scope="module")
def server():
    state = _State()
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), build_handler(state))
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_port}"
    httpd.shutdown()
    httpd.server_close()


# Tests must never depend on the host's ambient proxy configuration: a system
# HTTP proxy (VPN client, corporate MITM tool, another dev tool) can intercept
# even 127.0.0.1 traffic and hang or misroute it. Build an opener that always
# talks to the loopback server directly, regardless of environment/system proxy
# settings.
_DIRECT = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def get(base, path):
    with _DIRECT.open(base + path, timeout=10) as response:
        return response.status, json.loads(response.read())


def post(base, path, payload):
    request = urllib.request.Request(
        base + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with _DIRECT.open(request, timeout=10) as response:
        return response.status, json.loads(response.read())


@pytest.mark.parametrize(
    "path,needle",
    [
        ("/", b"MPSynth"),
        ("/app.css", b"--cursor"),
        ("/app.js", b"plotTradeoff"),
    ],
)
def test_static_assets_are_served(server, path, needle):
    with urllib.request.urlopen(server + path) as response:
        assert response.status == 200
        assert needle in response.read()


def test_datasets_endpoint(server):
    status, payload = get(server, "/api/datasets")
    assert status == 200
    assert "gaussian" in payload["datasets"]
    assert "qasm3" in payload["formats"]


def test_analyze_then_detail_then_export(server):
    status, summary = post(server, "/api/analyze", {"source": "gaussian:8", "max_layers": 3})
    assert status == 200
    assert summary["meta"]["n_qubits"] == 8
    assert len(summary["tradeoff"]) == 3

    status, detail = get(server, "/api/detail?layers=2")
    assert status == 200
    assert detail["layers"] == 2
    assert detail["metrics"]["cnot"] == summary["tradeoff"][1]["cnot"]

    status, exported = get(server, "/api/export?layers=2&format=qasm3")
    assert status == 200
    assert "OPENQASM 3.0;" in exported["text"]
    assert exported["extension"] == ".qasm"


def test_uploaded_values_are_accepted(server):
    values = list(np.exp(-(np.linspace(-3, 3, 256) ** 2)))
    status, summary = post(
        server, "/api/analyze", {"values": values, "name": "mine.csv", "max_layers": 2}
    )
    assert status == 200
    assert summary["meta"]["n_qubits"] == 8
    assert "mine.csv" in summary["meta"]["label"]


def test_bad_dataset_is_a_client_error(server):
    with pytest.raises(urllib.error.HTTPError) as excinfo:
        post(server, "/api/analyze", {"source": "not_a_dataset:8"})
    assert excinfo.value.code == 400
    assert "unknown dataset" in json.loads(excinfo.value.read())["error"]


def test_oversized_request_is_rejected(server):
    with pytest.raises(urllib.error.HTTPError) as excinfo:
        post(server, "/api/analyze", {"source": "gaussian:24"})
    assert excinfo.value.code == 400


def test_unknown_route_is_404(server):
    with pytest.raises(urllib.error.HTTPError) as excinfo:
        get(server, "/api/nope")
    assert excinfo.value.code == 404


@pytest.mark.filterwarnings("ignore::ResourceWarning")
def test_every_curve_layer_has_detail_and_every_export_format(server):
    """The standalone-report feature embeds detail + every export format for every
    layer on the curve; if any layer or format ever raised, the report silently
    would too (client-side, hard to notice). Guard it here instead.

    Covers every layer at least once and every format at least once (round-robin)
    rather than the full layers x formats cartesian product: the client-side
    report builder itself is exercised end-to-end separately (see the UI), so this
    is a defensive regression guard, not the only line of coverage, and does not
    need to open dozens of fresh connections in a tight loop to do its job.
    """
    status, summary = post(server, "/api/analyze", {"source": "gaussian:8", "max_layers": 5})
    assert status == 200
    status, meta = get(server, "/api/datasets")
    assert status == 200
    formats = meta["formats"]

    for i, point in enumerate(summary["tradeoff"]):
        status, detail = get(server, f"/api/detail?layers={point['layers']}")
        assert status == 200
        assert detail["metrics"]["cnot"] == point["cnot"]
        fmt = formats[i % len(formats)]
        status, exported = get(server, f"/api/export?layers={point['layers']}&format={fmt}")
        assert status == 200
        assert exported["text"].strip()

    # Confirm every format works too, just not against every layer.
    for fmt in formats:
        status, exported = get(
            server, f"/api/export?layers={summary['tradeoff'][0]['layers']}&format={fmt}"
        )
        assert status == 200
        assert exported["text"].strip()
