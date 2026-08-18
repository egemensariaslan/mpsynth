/* MPSynth — instrument panel.
 *
 * Plots are SVG paths only; gridlines, gutters and the cursor rules are CSS. The
 * renderer reads its margins back out of the stylesheet (--pad-*), so the drawn
 * axis and the CSS ruler are guaranteed to agree.
 */

const NS = "http://www.w3.org/2000/svg";
const $ = (id) => document.getElementById(id);

const app = {
  summary: null,
  detail: null,
  layers: null,
  uploaded: null,
  exported: null,
  extensions: {},
};

/* ── helpers ──────────────────────────────────────────────────────────── */

function svg(tag, attrs = {}, text = null) {
  const node = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
  if (text !== null) node.textContent = text;
  return node;
}

/** Plot gutters live in CSS; read them back so nothing can drift. */
function gutters(host) {
  const cs = getComputedStyle(host);
  const px = (name) => parseFloat(cs.getPropertyValue(name)) || 0;
  return { l: px("--pad-l"), r: px("--pad-r"), t: px("--pad-t"), b: px("--pad-b") };
}

const SUP = { "-": "⁻", 0: "⁰", 1: "¹", 2: "²", 3: "³",
  4: "⁴", 5: "⁵", 6: "⁶", 7: "⁷", 8: "⁸", 9: "⁹" };

const superscript = (n) => String(n).split("").map((c) => SUP[c] || c).join("");

function decade(v) {
  const e = Math.round(Math.log10(v));
  return e === 0 ? "1" : "10" + superscript(e);
}

function eng(x, digits = 2) {
  if (x === 0) return "0";
  const [m, e] = x.toExponential(digits).split("e");
  return `${m}×10${superscript(parseInt(e, 10))}`;
}

const si = (x) =>
  Math.abs(x) >= 1000 ? (x / 1000).toFixed(x >= 10000 ? 0 : 1) + "k" : String(x);

function ticks(lo, hi, n = 4) {
  if (!isFinite(lo) || !isFinite(hi) || lo === hi) return [lo];
  const step0 = (hi - lo) / n;
  const mag = 10 ** Math.floor(Math.log10(step0));
  const k = step0 / mag;
  const step = (k >= 5 ? 10 : k >= 2 ? 5 : k >= 1 ? 2 : 1) * mag;
  const out = [];
  for (let t = Math.ceil(lo / step) * step; t <= hi + step * 1e-9; t += step) out.push(t);
  return out;
}

function decades(lo, hi, max = 5) {
  const a = Math.floor(Math.log10(lo)), b = Math.ceil(Math.log10(hi));
  const step = Math.max(1, Math.ceil((b - a + 1) / max));
  const out = [];
  for (let e = a; e <= b; e += step) out.push(10 ** e);
  return out.filter((t) => t >= lo * 0.999 && t <= hi * 1.001);
}

const d = (points) =>
  points.map((p, i) => `${i ? "L" : "M"}${p[0].toFixed(2)},${p[1].toFixed(2)}`).join("");

/** An SVG sized to its host, using the gutters CSS declares. */
function canvas(host) {
  host.querySelectorAll("svg").forEach((n) => n.remove());
  const box = host.getBoundingClientRect();
  const W = Math.max(300, box.width), H = Math.max(70, box.height);
  const g = gutters(host);
  const root = svg("svg", { viewBox: `0 0 ${W} ${H}`, preserveAspectRatio: "none" });
  host.appendChild(root);
  return {
    root, W, H, g,
    px: (fx) => g.l + fx * (W - g.l - g.r),
    py: (fy) => g.t + fy * (H - g.t - g.b),
  };
}

function yLabels(c, items, format) {
  for (const { value, y } of items) {
    c.root.appendChild(
      svg("text", { class: "tick", x: c.g.l - 8, y: y + 3, "text-anchor": "end" }, format(value))
    );
  }
}

function xLabels(c, items, format, y) {
  for (const item of items) {
    c.root.appendChild(
      svg("text", { class: "tick", x: item.x, y, "text-anchor": "middle" }, format(item.value))
    );
  }
}

/* ── plots ────────────────────────────────────────────────────────────── */

function plotTraces(hostId, series, { zero = false, axisName } = {}) {
  const c = canvas($(hostId));
  const all = series.flatMap((s) => s.values);
  if (!all.length) return;

  let lo = Math.min(...all), hi = Math.max(...all);
  if (zero) { lo = Math.min(lo, 0); hi = Math.max(hi, 0); }
  if (lo === hi) { lo -= 1; hi += 1; }
  const pad = (hi - lo) * 0.08;
  lo -= pad; hi += pad;

  const n = series[0].values.length;
  const X = (i) => c.px(n <= 1 ? 0.5 : i / (n - 1));
  const Y = (v) => c.py(1 - (v - lo) / (hi - lo));

  yLabels(c, ticks(lo, hi, 3).map((v) => ({ value: v, y: Y(v) })),
    (v) => (Math.abs(v) >= 0.01 || v === 0 ? v.toFixed(2) : eng(v, 0)));

  if (zero) {
    c.root.appendChild(
      svg("line", { class: "baseline", x1: c.g.l, x2: c.W - c.g.r, y1: Y(0), y2: Y(0) })
    );
  }
  for (const s of series) {
    c.root.appendChild(svg("path", { class: s.klass, d: d(s.values.map((v, i) => [X(i), Y(v)])) }));
  }
  if (axisName) {
    c.root.appendChild(
      svg("text", { class: "axis-name", x: c.W - c.g.r, y: c.g.t + 8, "text-anchor": "end" }, axisName)
    );
  }
}

function plotEntropy(hostId, entropy, ceiling) {
  const c = canvas($(hostId));
  const hi = Math.max(...ceiling) * 1.1;
  const n = entropy.length;
  const X = (i) => c.px(n <= 1 ? 0.5 : i / (n - 1));
  const Y = (v) => c.py(1 - v / hi);

  yLabels(c, ticks(0, hi, 3).map((v) => ({ value: v, y: Y(v) })), (v) => v.toFixed(1));
  xLabels(c, [0, n - 1].map((i) => ({ value: i + 1, x: X(i) })), String, c.H - 1);

  const top = ceiling.map((v, i) => [X(i), Y(v)]);
  c.root.appendChild(
    svg("path", { class: "ceiling", d: d(top.concat([[X(n - 1), Y(0)], [X(0), Y(0)]])) + "Z" })
  );
  const pts = entropy.map((v, i) => [X(i), Y(v)]);
  c.root.appendChild(
    svg("path", { class: "fill", d: d(pts.concat([[X(n - 1), Y(0)], [X(0), Y(0)]])) + "Z" })
  );
  c.root.appendChild(svg("path", { class: "trace", d: d(pts) }));
  c.root.appendChild(
    svg("text", { class: "axis-name", x: c.W - c.g.r, y: c.g.t + 8, "text-anchor": "end" }, "S(k) bits")
  );
}

function plotSpectrum(hostId, values) {
  const c = canvas($(hostId));
  const v = values.filter((x) => x > 1e-15);
  if (!v.length) return;
  const hi = Math.max(...v), lo = Math.max(Math.min(...v), hi * 1e-13);
  const X = (i) => c.px(v.length <= 1 ? 0.5 : i / (v.length - 1));
  const Y = (x) =>
    c.py(1 - (Math.log10(Math.max(x, lo)) - Math.log10(lo)) / (Math.log10(hi) - Math.log10(lo)));

  yLabels(c, decades(lo, hi, 4).map((x) => ({ value: x, y: Y(x) })), decade);
  xLabels(c, [0, v.length - 1].map((i) => ({ value: i + 1, x: X(i) })), String, c.H - 1);
  c.root.appendChild(svg("path", { class: "trace-2", d: d(v.map((x, i) => [X(i), Y(x)])) }));
  c.root.appendChild(
    svg("text", { class: "axis-name", x: c.W - c.g.r, y: c.g.t + 8, "text-anchor": "end" }, "σ")
  );
}

function plotTradeoff(hostId, points, targetF, current, onPick) {
  const c = canvas($(hostId));
  const floor = 1e-13;
  const inf = points.map((p) => Math.max(p.infidelity, floor));
  const tgt = Math.max(1 - targetF, floor);
  const hi = Math.max(...inf, tgt) * 3, lo = Math.min(...inf, tgt) / 3;
  const xMax = Math.max(...points.map((p) => p.cnot)) * 1.1 || 1;

  const X = (v) => c.px(v / xMax);
  const Y = (v) =>
    c.py(1 - (Math.log10(Math.max(v, lo)) - Math.log10(lo)) / (Math.log10(hi) - Math.log10(lo)));

  yLabels(c, decades(lo, hi, 5).map((v) => ({ value: v, y: Y(v) })), decade);
  xLabels(c, ticks(0, xMax, 4).map((v) => ({ value: v, x: X(v) })), (v) => si(Math.round(v)), c.H - 1);
  c.root.appendChild(
    svg("text", { class: "axis-name", x: c.W - c.g.r, y: c.H - 1, "text-anchor": "end" }, "cnot")
  );
  c.root.appendChild(
    svg("text", { class: "axis-name", x: c.W - c.g.r, y: c.g.t + 8, "text-anchor": "end" }, "1 − F")
  );

  const ty = Y(tgt);
  c.root.appendChild(svg("line", { class: "limit", x1: c.g.l, x2: c.W - c.g.r, y1: ty, y2: ty }));
  c.root.appendChild(svg("text", { class: "limit-tag", x: c.g.l + 4, y: ty - 5 }, `F = ${targetF}`));

  const pts = points.map((p, i) => [X(p.cnot), Y(inf[i])]);
  c.root.appendChild(svg("path", { class: "trace", d: d(pts) }));

  points.forEach((p, i) => {
    const node = svg("circle", {
      class: "node", cx: pts[i][0], cy: pts[i][1], r: 5,
      "data-meets": p.fidelity >= targetF ? 1 : 0,
      "data-current": p.layers === current ? 1 : 0,
    });
    node.appendChild(
      svg("title", {}, `${p.layers}L · ${p.cnot} cnot · depth ${p.depth} · F ${p.fidelity.toFixed(6)}`)
    );
    node.addEventListener("click", () => onPick(p.layers));
    c.root.appendChild(node);
    c.root.appendChild(
      svg("text", { class: "node-tag", x: pts[i][0], y: pts[i][1] - 11, "text-anchor": "middle" }, p.layers)
    );
  });
}

function drawCircuit(hostId, circuit) {
  const host = $(hostId);
  host.innerHTML = "";
  const n = circuit.n_qubits;
  const cw = 20, rh = 24, padL = 30, padT = 10;
  const W = padL + Math.max(circuit.columns, 1) * cw + 14;
  const H = padT * 2 + n * rh;

  const root = svg("svg", { viewBox: `0 0 ${W} ${H}`, width: W, height: H });
  root.style.width = W + "px";
  root.style.height = H + "px";
  const Y = (q) => padT + q * rh + rh / 2;
  const X = (col) => padL + col * cw + cw / 2;

  for (let q = 0; q < n; q++) {
    root.appendChild(svg("line", { class: "wire", x1: padL - 6, x2: W - 8, y1: Y(q), y2: Y(q) }));
    root.appendChild(svg("text", { class: "wire-tag", x: 0, y: Y(q) + 3 }, q));
  }
  for (const gate of circuit.gates) {
    const x = X(gate.column);
    if (gate.name === "cx") {
      const [ctrl, targ] = gate.qubits;
      root.appendChild(svg("line", { class: "link", x1: x, x2: x, y1: Y(ctrl), y2: Y(targ) }));
      root.appendChild(svg("circle", { class: "ctrl", cx: x, cy: Y(ctrl), r: 2.6 }));
      root.appendChild(svg("circle", { class: "targ", cx: x, cy: Y(targ), r: 5 }));
      root.appendChild(svg("line", { class: "targ", x1: x - 5, x2: x + 5, y1: Y(targ), y2: Y(targ) }));
    } else {
      const q = gate.qubits[0];
      const box = svg("rect", { class: `op ${gate.name}`, x: x - 7.5, y: Y(q) - 7, width: 15, height: 14 });
      box.appendChild(svg("title", {}, `${gate.name}(${gate.params[0]}) q${q}`));
      root.appendChild(box);
      root.appendChild(svg("text", { class: "op-tag", x, y: Y(q) }, gate.name === "rz" ? "z" : "y"));
    }
  }
  host.appendChild(root);
}

/* ── linked cursor ────────────────────────────────────────────────────── */

/* One pointer handler writes one custom property. CSS positions every rule from
   it; JS only supplies the numbers that happen to lie under the cursor. */
function wireCursor() {
  const strips = $("strips");
  const out = $("cursor-readout");
  const idle = "<span>hover to probe</span>";
  out.innerHTML = idle;

  strips.addEventListener("pointermove", (event) => {
    const probe = strips.querySelector(".plot");
    if (!probe || !app.detail || !app.detail.amplitudes) return;
    const box = probe.getBoundingClientRect();
    const g = gutters(probe);
    const f = Math.min(1, Math.max(0, (event.clientX - box.left - g.l) / (box.width - g.l - g.r)));
    strips.style.setProperty("--cursor", f.toFixed(5));
    strips.dataset.live = "1";

    const a = app.detail.amplitudes;
    const i = Math.round(f * (a.target.length - 1));
    out.innerHTML =
      `<span>i</span> ${si(i * (a.stride || 1))}` +
      `   <span>target</span> ${a.target[i].toExponential(3)}` +
      `   <span>prepared</span> ${a.produced[i].toExponential(3)}` +
      `   <span>Δ</span> ${a.residual[i].toExponential(2)}`;
  });

  strips.addEventListener("pointerleave", () => {
    strips.dataset.live = "0";
    out.innerHTML = idle;
  });
}

/* ── render ───────────────────────────────────────────────────────────── */

function ticker(text, state = "idle") {
  $("ticker").dataset.state = state;
  $("ticker-text").textContent = text;
}

const count = (id, value) => $(id).style.setProperty("--n", Math.round(value));

function renderSummary(s) {
  const e = s.entanglement;
  $("ent-meta").textContent =
    `χ ${s.meta.bond_dimension} · ${(e.saturation * 100).toFixed(0)}% of ceiling`;
  plotEntropy("plot-entropy", e.entropy, e.max_entropy);
  plotSpectrum("plot-spectrum", e.spectra[Math.floor(e.spectra.length / 2)] || []);
}

function renderStripAxis(points, stride) {
  const host = $("strip-axis");
  host.innerHTML = "";
  const probe = $("strips").querySelector(".plot");
  const g = gutters(probe);
  const box = host.getBoundingClientRect();
  const root = svg("svg", { viewBox: `0 0 ${box.width} 26`, preserveAspectRatio: "none" });
  root.setAttribute("style", "position:absolute;inset:0;width:100%;height:100%");
  const last = (points - 1) * stride;
  for (const t of ticks(0, last, 5)) {
    const x = g.l + (t / last) * (box.width - g.l - g.r);
    root.appendChild(svg("text", { class: "tick", x, y: 11, "text-anchor": "middle" }, si(Math.round(t))));
  }
  root.appendChild(svg("text", { class: "axis-name", x: 0, y: 11 }, "index"));
  host.appendChild(root);
}

function renderDetail(detail, summary) {
  const m = detail.metrics;
  const p = detail.point;

  count("r-cnot", m.cnot);
  count("r-2qd", m.two_qubit_depth);
  count("r-depth", m.depth);
  count("r-1q", m.one_qubit);
  count("r-layers", detail.layers);
  count("r-ratio", summary.baseline.exact_cnot / Math.max(m.cnot, 1));
  $("r-fid").textContent = p.fidelity.toFixed(6);
  $("r-infid").textContent = eng(p.infidelity);

  if (detail.amplitudes) {
    const a = detail.amplitudes;
    plotTraces("plot-signal", [
      { values: a.target, klass: "trace-ghost" },
      { values: a.produced, klass: "trace" },
    ], { axisName: "amplitude" });
    plotTraces("plot-residual", [{ values: a.residual, klass: "trace-2" }],
      { zero: true, axisName: "Δ" });
    renderStripAxis(a.target.length, a.stride || 1);
  }

  const body = $("checks").querySelector("tbody");
  body.innerHTML = "";
  for (const check of detail.verification.checks) {
    const tr = document.createElement("tr");
    tr.dataset.ok = check.ok;
    tr.innerHTML =
      `<td class="q">${check.name}<small>${check.detail}</small></td>` +
      `<td class="num">${check.residual === null ? "" : eng(check.residual)}</td>` +
      `<td class="num">${check.bound || ""}</td>` +
      `<td class="verdict">${check.ok ? "ok" : "over"}</td>`;
    body.appendChild(tr);
  }

  $("circuit-meta").textContent =
    `${si(detail.circuit.total_gates)} ops · ${detail.circuit.columns} columns`;
  drawCircuit("circuit", detail.circuit);
}

/* ── flow ─────────────────────────────────────────────────────────────── */

async function api(path, options) {
  const res = await fetch(path, options);
  const data = await res.json();
  if (!res.ok || data.error) throw new Error(data.error || res.statusText);
  return data;
}

function target() {
  const v = parseFloat($("fidelity").value);
  return isFinite(v) && v > 0 && v < 1 ? v : 0.999;
}

async function select(layers) {
  app.layers = layers;
  plotTradeoff("plot-tradeoff", app.summary.tradeoff, target(), layers, select);
  ticker(`simulating ${layers}-layer circuit`, "busy");
  app.detail = await api(`/api/detail?layers=${layers}`);
  renderDetail(app.detail, app.summary);
  await emit();
  const over = app.detail.verification.checks.filter((c) => !c.ok).length;
  ticker(
    `${app.summary.meta.label} · ${layers} layer${layers === 1 ? "" : "s"}` +
    (over ? ` · ${over} out of bound` : ""),
    over ? "error" : "idle"
  );
}

function cheapest() {
  const t = target();
  const hit = app.summary.tradeoff.find((p) => p.fidelity >= t);
  return (hit || app.summary.tradeoff.at(-1)).layers;
}

async function emit() {
  if (app.layers === null) return;
  app.exported = await api(`/api/export?layers=${app.layers}&format=${$("format").value}`);
  $("export").firstChild.textContent = app.exported.text;
}

async function run(event) {
  if (event) event.preventDefault();
  $("run").disabled = true;
  try {
    ticker("decomposing · synthesising · profiling", "busy");
    const body = {
      source: $("source").value,
      max_layers: parseInt($("layers").value, 10),
      qubit_order: $("order").value,
    };
    if (app.uploaded) { body.values = app.uploaded.values; body.name = app.uploaded.name; }
    app.summary = await api("/api/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    $("workspace").hidden = false;
    renderSummary(app.summary);
    await select(cheapest());
  } catch (err) {
    ticker(err.message, "error");
  }
  $("run").disabled = false;
}

function redraw() {
  if (!app.summary) return;
  renderSummary(app.summary);
  plotTradeoff("plot-tradeoff", app.summary.tradeoff, target(), app.layers, select);
  if (app.detail) renderDetail(app.detail, app.summary);
}

async function init() {
  if (localStorage.getItem("mpsynth-theme") === "light") {
    document.documentElement.dataset.theme = "light";
  }
  $("theme").addEventListener("click", () => {
    const root = document.documentElement;
    root.dataset.theme = root.dataset.theme === "light" ? "dark" : "light";
    localStorage.setItem("mpsynth-theme", root.dataset.theme);
    redraw();
  });

  const meta = await api("/api/datasets");
  $("version").textContent = meta.version;
  app.extensions = meta.extensions;
  for (const name of meta.datasets) {
    const option = document.createElement("option");
    option.value = `${name}:12`;
    $("datasets").appendChild(option);
  }
  for (const name of meta.formats) {
    const option = document.createElement("option");
    option.value = option.textContent = name;
    if (name === "qasm3") option.selected = true;
    $("format").appendChild(option);
  }

  $("console").addEventListener("submit", run);
  $("fidelity").addEventListener("change", () => { if (app.summary) select(cheapest()); });
  $("format").addEventListener("change", emit);
  $("source").addEventListener("input", () => { app.uploaded = null; });
  $("upload-btn").addEventListener("click", () => $("upload").click());
  $("upload").addEventListener("change", async (event) => {
    const file = event.target.files[0];
    if (!file) return;
    const values = (await file.text()).split(/[\s,;]+/).map(Number).filter(isFinite);
    if (values.length < 2) return ticker(`${file.name}: no numeric values`, "error");
    app.uploaded = { values, name: file.name };
    $("source").value = file.name;
    ticker(`${file.name} · ${si(values.length)} values · press run`);
  });
  $("copy").addEventListener("click", async () => {
    await navigator.clipboard.writeText($("export").textContent);
    ticker("copied to clipboard");
  });
  $("download").addEventListener("click", () => {
    if (!app.exported) return;
    const anchor = document.createElement("a");
    anchor.href = URL.createObjectURL(new Blob([app.exported.text], { type: "text/plain" }));
    anchor.download = "prepare" + (app.extensions[app.exported.format] || ".txt");
    anchor.click();
    URL.revokeObjectURL(anchor.href);
  });

  wireCursor();
  let timer;
  window.addEventListener("resize", () => { clearTimeout(timer); timer = setTimeout(redraw, 140); });

  run();
}

init();
