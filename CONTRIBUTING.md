# Contributing

## Setup

```bash
git clone <repo-url> && cd mpsynth
uv pip install -e ".[dev]"          # or: pip install -e ".[dev]"
```

No dev container, no special tooling — the whole stack is numpy plus stdlib.

## Before opening a PR

```bash
pytest -q                                          # 233+ tests, should take <30s
ruff check src tests examples mpsynth validation    # lint
ruff format --check src tests examples mpsynth validation
mypy src/mpsynth --ignore-missing-imports           # zero errors, not advisory
python validation/run_validation.py --trials 100 --ci   # statistical checks
```

All of the above run in CI (`.github/workflows/ci.yml`) on every push and PR,
across Python 3.10-3.13 on Linux, macOS and Windows. A PR that fails any of them
won't merge, so it's faster to run them locally first.

## Ground rules for this codebase specifically

**Every numerical claim is verified before it ships, not just tested.** If you add
a new gate decomposition or a new circuit construction, add a test that
reconstructs the target matrix from the emitted gates and checks it with
`atol=1e-9, rtol=0` (not numpy's default `rtol=1e-5`, which is too loose to catch
a wrong construction — see the git history for what that cost once). If you add a
claim to the README ("N times fewer CNOTs", "linear scaling"), it needs a
corresponding check in `validation/run_validation.py`, not just a one-off number
you measured locally.

**No silent truncation or approximation without reporting it.** Every place that
discards information (SVD truncation, MPS bond capping) must report how much was
discarded, and the reported fidelity must be measured against the *true* input,
never a pre-truncated stand-in.

**The exported circuit is the source of truth.** Tests should verify claims by
re-simulating the actual emitted `{RZ, RY, CX}` gate list, not by trusting internal
state. Prefer `circuit.statevector()` or `circuit.to_mps()` over reaching into
intermediate objects.

**UI changes**: the frontend (`src/mpsynth/ui/static/`) is hand-written HTML/CSS/JS
with zero dependencies and no build step — keep it that way. If you touch a chart,
verify it renders correctly in an actual browser (screenshot or manual check)
before committing; a change that only "looks right" in the source is not verified.

## Filing issues

Include the output of `python -c "import mpsynth, sys; print(mpsynth.__version__, sys.version)"`
and, for numerical issues, the smallest input vector that reproduces the problem.
