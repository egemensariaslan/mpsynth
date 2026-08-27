"""A local web UI for MPSynth, served from the standard library.

No framework, no CDN, no build step: the front end is three static files and the
backend is ``http.server``.  Everything it displays comes from
:mod:`mpsynth.analysis`, so the UI cannot show a number the library does not measure.
"""

from __future__ import annotations

import json
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

from .. import __version__
from ..analysis import Analysis, analyze, layer_detail
from ..datasets import GENERATORS, generate
from ..exporters import EXTENSIONS, FORMATS, export

__all__ = ["serve", "build_handler"]

STATIC = Path(__file__).parent / "static"
CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".svg": "image/svg+xml",
}


class _State:
    """The most recent analysis, so per-layer detail does not recompute the sweep."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.analysis: Analysis | None = None


def _load_source(payload: dict) -> tuple[np.ndarray, str]:
    """Resolve a request into a vector: uploaded values, dataset spec, or file path."""
    values = payload.get("values")
    if values:
        name = payload.get("name") or "uploaded"
        return np.asarray(values, dtype=float), f"{name} ({len(values)} values)"

    source = (payload.get("source") or "gaussian:10").strip()
    if ":" in source and not Path(source).exists():
        parts = source.split(":")
        name = parts[0]
        if name not in GENERATORS:
            raise ValueError(f"unknown dataset {name!r}")
        n = int(parts[1]) if len(parts) > 1 else 10
        seed = int(parts[2]) if len(parts) > 2 else 0
        if not 1 <= n <= 20:
            raise ValueError("qubit count must be between 1 and 20 in the UI")
        return generate(name, n, seed), f"{name} ({n} qubits)"

    from ..cli import load_vector

    return load_vector(source)


def build_handler(state: _State) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = f"mpsynth/{__version__}"

        # ------------------------------------------------------------- plumbing

        def log_message(self, fmt: str, *args) -> None:  # noqa: A003
            pass  # keep the console clean; errors still surface in responses

        def _send(self, code: int, body: bytes, content_type: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, payload: dict, code: int = 200) -> None:
            self._send(code, json.dumps(payload).encode(), "application/json")

        def _error(self, exc: Exception, code: int = 400) -> None:
            self._json({"error": str(exc), "type": type(exc).__name__}, code)

        def _static(self, name: str) -> None:
            path = (STATIC / name).resolve()
            if not path.is_file() or STATIC.resolve() not in path.parents:
                self._send(404, b"not found", "text/plain")
                return
            self._send(200, path.read_bytes(), CONTENT_TYPES.get(path.suffix, "text/plain"))

        # --------------------------------------------------------------- routes

        def do_GET(self) -> None:  # noqa: N802
            url = urlparse(self.path)
            route = url.path
            try:
                if route in ("/", "/index.html"):
                    return self._static("index.html")
                if route in ("/app.css", "/app.js"):
                    return self._static(route.lstrip("/"))
                if route == "/api/datasets":
                    return self._json(
                        {
                            "datasets": sorted(GENERATORS),
                            "formats": sorted(FORMATS),
                            "extensions": EXTENSIONS,
                            "version": __version__,
                        }
                    )
                if route == "/api/detail":
                    return self._detail(parse_qs(url.query))
                if route == "/api/export":
                    return self._export(parse_qs(url.query))
                self._send(404, b"not found", "text/plain")
            except ValueError as exc:  # bad input is the caller's fault, not ours
                self._error(exc, 400)
            except Exception as exc:  # surface failures in the UI, never a blank page
                self._error(exc, 500)

        def do_POST(self) -> None:  # noqa: N802
            url = urlparse(self.path)
            try:
                length = int(self.headers.get("Content-Length", 0))
                payload = json.loads(self.rfile.read(length) or b"{}")
                if url.path == "/api/analyze":
                    return self._analyze(payload)
                self._send(404, b"not found", "text/plain")
            except ValueError as exc:
                self._error(exc, 400)
            except Exception as exc:
                self._error(exc, 500)

        # -------------------------------------------------------------- handlers

        def _analyze(self, payload: dict) -> None:
            vector, label = _load_source(payload)
            analysis = analyze(
                vector,
                label=label,
                max_layers=int(payload.get("max_layers", 8)),
                chi_max=payload.get("chi") or None,
                qubit_order=payload.get("qubit_order", "big"),
            )
            with state.lock:
                state.analysis = analysis
            self._json(analysis.summary())

        def _current(self) -> Analysis:
            with state.lock:
                if state.analysis is None:
                    raise ValueError("no analysis yet; run one first")
                return state.analysis

        def _detail(self, query: dict) -> None:
            analysis = self._current()
            layers = int(query.get("layers", ["1"])[0])
            self._json(layer_detail(analysis, layers))

        def _export(self, query: dict) -> None:
            analysis = self._current()
            layers = int(query.get("layers", ["1"])[0])
            fmt = query.get("format", ["qasm3"])[0]
            circuit = analysis.curve.circuit_for(layers)
            self._json(
                {
                    "format": fmt,
                    "extension": EXTENSIONS.get(fmt, ".txt"),
                    "text": export(circuit, fmt),
                }
            )

    return Handler


def serve(port: int = 8765, open_browser: bool = True, host: str = "127.0.0.1") -> int:
    """Run the UI until interrupted. Returns a process exit code."""
    state = _State()
    handler = build_handler(state)

    try:
        httpd = ThreadingHTTPServer((host, port), handler)
    except OSError as exc:
        print(f"error: cannot bind {host}:{port} ({exc}); try --port {port + 1}")
        return 1

    url = f"http://{host}:{httpd.server_port}"
    print(f"mpsynth {__version__} UI  ->  {url}")
    print("press Ctrl-C to stop")
    if open_browser:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        httpd.server_close()
    return 0
