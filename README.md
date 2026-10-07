# PEtab MCP

[![CI](https://github.com/frohlich-lab/PEtab-MCP/actions/workflows/ci.yml/badge.svg)](https://github.com/frohlich-lab/PEtab-MCP/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## Overview

A local [MCP](https://modelcontextprotocol.io) server that wraps a single
[PEtab v1](https://petab.readthedocs.io) parameter-estimation problem and turns mechanistic
model building into a **disciplined, auditable** workflow for LLM agents — the loop behind
model-building assistants such as [petabgpt](https://github.com/frohlich-lab/petabgpt).

Point an agent at a dataset and it proposes and refines a mechanistic model, one justified
change at a time. The experimental design stays fixed and the model is edited only through
one constrained tool, so an agent cannot silently rewrite the rules of the problem. Every
change is **scored** on the spot (fit and simulated), **versioned** (nothing overwritten,
nothing lost), and **justified** (a rationale attached, optionally checked against PubMed).
This means that instead of a black box just delivering a model, you get the model and
inspectable provenance describing how it developed.

Simulation/objective use [AMICI](https://github.com/AMICI-dev/AMICI), optimization uses
[pyPESTO](https://github.com/ICB-DCM/pyPESTO) with the
[Fides](https://github.com/fides-dev/fides) trust-region optimizer. All are **required** —
fitting is not optional machinery.

The agent harness that drives this server — the analyst panel, the judge, the zero-shot
benchmark examples and the run explorer — lives in
[frohlich-lab/petab_mcp](https://github.com/frohlich-lab/petab_mcp).

## Install

```bash
pip install -e .          # installs AMICI + pyPESTO (a C++ toolchain is required)
```

AMICI compiles generated C++ and needs a toolchain: a C++ compiler, CMake, Ninja, and the
pip `swig` binary on `PATH` (OpenBLAS ships in the AMICI wheel).

Expanding a PySB model into equations (`pysb.bng.generate_equations`) needs BioNetGen's `BNG2.pl`
on `PATH` (or `BNGPATH` set); it is not a pip dependency.

## Run

```bash
petab-mcp --petab-yaml path/to/problem.yaml
# --provenance-dir DIR   --reference-validation {required,best_effort,off}
# --build-root DIR       --skills-dir DIR
# --entity-map PATH      (offers make_changes the `statements` route; see src/petab_mcp/seeding.py)
# --transport {stdio,sse,streamable-http}
# --criterion {bic,caic,aic,aicc,hqc,tic}   (env PETAB_MCP_CRITERION; default bic)
```

Provenance is written to `<problem_dir>/.petab_mcp/` by default and is durable — restarting
resumes the latest version. A ready toy problem lives at `tests/data/toy_v1/`.

## Tools

`make_changes` is the single tool that changes the model or its parameter search space, and
it always fits — see its docstring in [`src/petab_mcp/server.py`](src/petab_mcp/server.py)
for the full argument and result reference. Everything else is read-only: problem/model/
parameter inspection (`describe_problem`, `get_model`, `get_parameters`, ...), fit
inspection (`get_fit_summary`, `get_fit_diagnosis`, `query_data` — arbitrary SQL over the
fit via DuckDB), and provenance (`get_provenance`, `list_versions`, `restore_version`, ...).
A live server also exposes these as MCP resources under `petab://...`. Each submodule of
[`src/petab_mcp`](src/petab_mcp/) is documented in its own docstrings.

## Design notes

Principles to hold onto when extending this package.

- **No free-form editing.** The model is only ever changed through `make_changes`, and every
  change must carry a justification. Don't add a second way to write to the model.
- **A change and its score never come apart.** `make_changes` always fits and snapshots a
  version — there is no edit without a fit, and no fit without a version.
- **SBML and PEtab stay consistent automatically.** Reconciliation (new symbols, orphan
  pruning, parametric initial conditions) is the server's job, not the agent's — but a few
  cases (an IC of exactly 0, a condition-table override, an assignment rule, a compound
  expression) are deliberately left untouched because minting there would be wrong, not just
  unnecessary. Read [`src/petab_mcp/reconcile.py`](src/petab_mcp/reconcile.py) before
  "fixing" one of these.
- **Every operation is versioned and logged.** Provenance under `.petab_mcp/` is append-only
  and durable; extend what a version records rather than bypassing it.
- **Iteration is cheap; results are not cached.** Structural edits recompile, parameter
  edits don't — but an identical or already-seen structure is refused a wasted refit, never
  served from a fit-result cache (there isn't one).
- **Fit inspection is SQL, not a growing menu.** A new way to look at a fit should be a
  `query_data` query over the DuckDB tables, not a new bespoke tool.
- **Comparisons must be valid before they're reported.** A commit that isn't nested in what
  it's compared against is flagged, not silently ranked by score.
- **One score, named, on one scale.** Models are ranked by a complexity-penalised score of the
  form `2·nll + penalty` — `bic` by default; `caic`, `aic`, `aicc`, `hqc` or `tic` (Takeuchi,
  charging the effective parameter count `k_eff = tr(J·H⁻¹)` the server computes) via
  `--criterion`. Every fit reports every score plus `criterion`/`criterion_value`, and every
  gate — new best, plateau stop, de-dup, merge price, the harness judge — reads that one
  value, so they cannot disagree; canonical BIC is recorded on every run regardless, and
  `result.json` carries the whole table. Named penalties and nothing between them: the fixed
  effect-size thresholds carry over when the penalty is swapped, not when it is scaled. See
  [`src/petab_mcp/criteria.py`](src/petab_mcp/criteria.py) before adding one.
- **A hardcoded constant is not free.** `k` is the estimated parameters plus every numeric
  literal in the model's math bar a few structural ones (each distinct value once — see
  [`src/petab_mcp/literals.py`](src/petab_mcp/literals.py)), so freezing a fitted value costs
  what estimating it costs and there is no threshold to round past. A genuinely known constant
  ships as a fixed row: the problem author's call, not the agent's.

## Development

```bash
pip install -e '.[dev]'   # AMICI + pyPESTO are core deps; dev adds pytest + ruff
pytest                    # needs the full install (AMICI/pyPESTO are hard deps); PubMed is stubbed
```

## License

MIT
