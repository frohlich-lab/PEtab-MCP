# PEtab MCP

[![CI](https://github.com/frohlich-lab/PEtab-MCP/actions/workflows/ci.yml/badge.svg)](https://github.com/frohlich-lab/PEtab-MCP/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## Overview

A local [MCP](https://modelcontextprotocol.io) server that wraps a single
[PEtab v2](https://petab.readthedocs.io/en/latest/v2/documentation_data_format.html)
parameter-estimation problem (a PEtab v1 problem is upgraded on load) and turns mechanistic
model building into a **disciplined, auditable** workflow for LLM agents — the loop behind
model-building assistants such as [petabgpt](https://github.com/frohlich-lab/petabgpt).

Point an agent at a dataset and it proposes and refines a mechanistic model, one justified
change at a time. The experimental design stays fixed and the model is edited only through
one constrained tool, so an agent cannot silently rewrite the rules of the problem. Every
change is **scored** on the spot (fit and simulated), **versioned** (nothing overwritten,
nothing lost), and **justified** (a rationale attached, optionally checked against PubMed).
This means that instead of a black box just delivering a model, you get the model and
inspectable provenance describing how it developed.

Simulation/objective use [AMICI](https://github.com/AMICI-dev/AMICI)'s PEtab v2 import,
optimization uses [pyPESTO](https://github.com/ICB-DCM/pyPESTO) with the
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
pip `swig` binary on `PATH`. AMICI ships only a source distribution, so it is built on install
(BLAS comes from the `scipy-openblas64` wheel, or Accelerate on macOS).

A model built from statements (`make_changes`' `statements` route, `petab_mcp.problem_sbml`)
has its rules expanded into reactions by BioNetGen, which the `bionetgen` package ships.
BioNetGen is a Perl program, so `perl` must be on `PATH`. Without it every statements build is
refused with `Expanding the statements into reactions failed: [Errno 2] No such file or
directory: 'perl'`; the Antimony routes do not use it. macOS and most Linux distributions ship
perl, slim container images do not.

## Run

```bash
petab-mcp --petab-yaml path/to/problem.yaml
# --provenance-dir DIR   --reference-validation {required,best_effort,off}
# --build-root DIR       --skills-dir DIR
# --entity-map PATH      (make_changes offers the `statements` route in place of the Antimony
#                         routes, never both; see src/petab_mcp/seeding.py)
# --constraints PATH     (with --entity-map: causal claims every model must make; see src/petab_mcp/constraints.py)
# --constraint-updates {off,any}   (any: offers set_constraints, which replaces them mid-session)
# --transport {stdio,sse,streamable-http}
# --criterion {bic,caic,aic,aicc,hqc,tic}   (env PETAB_MCP_CRITERION; default bic)
# --fit-pool ADDRESS     (env PETAB_MCP_FIT_POOL; see "Fitting on SLURM")
```

Provenance is written to `<problem_dir>/.petab_mcp/` by default and is durable — restarting
resumes the latest version (a store written by the PEtab v1 server is refused). The server's
stderr goes to `stderr.log` there, beside `petab_mcp.log` and the fit log `pypesto.log`. A ready toy
problem lives at `tests/data/toy_v2/`; `tests/data/toy_v1/` is the same problem in v1 form.

## Fitting on SLURM

The fits of many servers, a sweep's runs say, share one pool of SLURM workers. `petab-mcp-pool`
runs the pool as a SLURM job of its own, and a server joins it with `--fit-pool ADDRESS` (or
`PETAB_MCP_FIT_POOL`). Both need the `dask` extra (`pip install -e '.[dask]'`).

```bash
petab-mcp-pool --slurm-max-jobs N --address-file PATH   # --slurm-min-jobs M   --port 8786
petab-mcp --petab-yaml problem.yaml --build-root DIR --fit-pool "$(cat PATH)"
```

The pool's scheduler listens on `--port` of the pool's node, so a pool restarted there comes back
at the same address, and the address goes to `--address-file` once it listens. The pool submits up
to N worker jobs while batches wait for a worker and keeps M (default 0) while idle. On SIGTERM it
removes the address file, waits up to 10 min for the batches running, then cancels its jobs, failing
the fits still waiting. It needs:

- **every server inside a SLURM allocation**, never on the login node: a server compiles every model
  itself, and a worker never does;
- **one `--build-root` on storage the compute nodes share**, the same for every server of the pool,
  not `$TMPDIR`: the workers import the compiled models by the server's path. Servers on two nodes
  building one model wait for each other through a POSIX (`fcntl`) lock there, so the storage's
  locks must hold across nodes, as GPFS's do;
- **one environment for the pool and its servers**: a start is pickled, so a server refuses a pool
  whose `amici`, `pypesto`, `petab` or `petab-mcp` differs from its own, naming both. After a pin
  bump, restart the pool;
- **a job of 2 CPUs and 8 GB for the pool itself**: at 1200 workers and 100 servers its scheduler
  used about one core and 1.4 GB (`python -m petab_mcp.pool_stress`, on NEMO);
- **the worker jobs and the scheduler configured in dask's own config**, not by flags: a YAML file
  in `~/.config/dask/` (or `DASK_*` variables) where the pool runs, for example:

```yaml
jobqueue:
  slurm:
    queue: <partition>
    cores: 32
    processes: 32              # = cores: one single-threaded worker per core, a core per start
    memory: 64GiB
    walltime: "3-00:00:00"     # longer than the pool will run
    death-timeout: 600         # seconds a starting worker waits for the scheduler
    local-directory: $TMPDIR   # the node's own scratch
    log-directory: <shared storage>/dask-logs
distributed:
  scheduler:
    worker-ttl: "10 minutes"   # well above the longest heartbeat gap pool_stress reports
```

A worker job is not replaced when it ends: with several workers to a job, jobqueue's `adapt` never
resubmits it, and the pool stays smaller. Its running batches are rerun on other workers. So give
worker jobs a walltime longer than the pool will run, and restart the pool between sweeps. A worker
`--lifetime` buys nothing: a worker reaching it drops its running batch, as a killed one does. The
default `death-timeout` of 60 s is too short for a full job's workers to import from shared
storage.

A dask task runs as many starts as fill `PETAB_MCP_FIT_TASK_SECONDS` (default 270) at the median
start time of the session's previous fit, or at its per-start cap (by default
`PETAB_MCP_FIT_PER_START_TIMEOUT`) without one; `PETAB_MCP_FIT_STARTS_PER_TASK` fixes the number
instead. A pool lost mid-fit, or a worker that cannot import a model, ends the call with
`infrastructure_error: fit_pool: `, as a full disk does (see "Tools").

## Tools

`make_changes` is the single tool that changes the model or its parameter search space, and
it always fits — see its docstring in [`src/petab_mcp/server.py`](src/petab_mcp/server.py),
and the [`reading-fits`](src/petab_mcp/_skills/reading-fits/SKILL.md) skill for what its
result's fields mean. Everything else is read-only: problem/model/
parameter inspection (`describe_problem`, `list_conditions`, `list_experiments`, `get_model`,
`get_parameters`, ...), fit
inspection (`get_fit_summary`, `get_fit_diagnosis`, `query_data` — arbitrary SQL over the
fit via DuckDB), and provenance (`get_provenance`, `list_versions`, `restore_version`, ...).
A live server also exposes these as MCP resources under `petab://...`. Each submodule of
[`src/petab_mcp`](src/petab_mcp/) is documented in its own docstrings.

A refused call returns a result that says why (`error`, and `error_cause` where the refusal has a
class). A full disk or exceeded quota (ENOSPC, EDQUOT) is not a refusal: nothing the agent can
change fixes it, so wherever it strikes (edit, AMICI build, fit, stored result) the call ends as an
MCP tool error whose text starts with `infrastructure_error: disk_full: `. Any edit it carried is
undone, and a fit it ends does not count against the identical-fit guard. A client should end the
run on it and resubmit once the disk has room
([`errors.InfrastructureError`](src/petab_mcp/errors.py)).

A start still running its session's per-start cap (by default `PETAB_MCP_FIT_PER_START_TIMEOUT`)
plus 600 s after it could have begun (stuck inside one simulation, where Fides never checks its
cap) is stopped and scored as a failed start;
the fit is scored from the rest and reports `n_starts_killed`. A stopped warm start from the
parent's optimum is rerun once. A fit whose starts stop finishing ends the same way as a full disk,
as `infrastructure_error: fit_wedged: `, with its workers stopped and nothing scored: when no
start finishes for a start's cap + 600 s while others wait for a worker, when every start was
stopped, or when a rerun warm start is stopped again ([`multistart`](src/petab_mcp/multistart.py)).
Every 5 minutes a running fit logs how many of its starts are done. While a change is being
fitted, a second `make_changes` on the session waits for nothing: it is refused at once with
`error_cause: fit_in_progress` (text starting `make_changes: fit_in_progress: `), and the
running fit's result becomes the next version: `wait_for_fit` waits for it on the server and
answers with that version's row, or with how far the fit has got. A client that cancels a call (MCP
`notifications/cancelled`) stops its fit, and the edit is undone. A call that carries a progress
token gets an MCP progress notification every 30 s while it runs, naming the session's running
change: its phase (`compiling`, `queued` with none of its starts running, or `running`), the
seconds spent in it and its starts done, so a client can time a call out on silence instead of on a
budget.

A client that renders MCP Apps (Claude Code's desktop app) shows a `make_changes` call as a badge for
the person watching it: the fit's phase, its starts done, running and queued, the time spent and, at
the pace so far, left, and the best start, then the version it committed. The page,
[`fit_status.html`](src/petab_mcp/fit_status.html), reads `wait_for_fit(timeout_s=0)` every 2 s from
when its host draws it: in Claude Code, once the call moves to the background. Only such a client
finds it in `tools/list` and `resources/list`, and no call returns anything new. Claude Code's Code tab
passes the page's calls on only for a server declared in `claude_desktop_config.json`
([anthropics/claude-code#106708](https://github.com/anthropics/claude-code/issues/106708)), and cuts
such a server's call off after about 60 s; for any other server the badge says so and shows the call's
result.

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
  edits don't — but a model an earlier version already fitted (its structure, its bounds, as many
  starts) is refused a wasted refit, never served from a fit-result cache (there isn't one). A
  refit is a `make_changes` without a model route, and a new version.
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
- **The estimation scale is the server's.** PEtab v2 dropped `parameterScale`, so the server keeps
  the column as an extension of the parameter table and hands it to pyPESTO's v2 importer as its
  `parameter_scales`, which puts AMICI's simulator on it too — every pyPESTO problem is built
  through `fitting.pypesto_problem`, never handed to pyPESTO linear. A problem that states a
  prior is refused at load: the objective is the likelihood the criterion scores.
- **A statement says everything its rate reads.** On the `statements` route a row's statement is
  the claim and its binding the maths behind it, the two given apart under the row's prefix
  (`statements=["r1: RAF phosphorylates MEK"], bindings={"r1": "two_step kf=k_on"}`), so a rate (a
  written `rate`, a `scheme`) reads only the agents the statement names: its roles, the partners they are written bound or not
  bound to, and the agents it states regulate its rate without taking part, `inhibited_by`,
  `activated_by` or `modulated_by` ("RAF phosphorylates MEK, inhibited by sorafenib"). A scheme
  reads one by its kind, in a rate or as a molecule its rules bind (sorafenib sequestering RAF),
  on sites named after each other, so the complex is the one the partner notation writes
  (`RAF{!sorafenib}`) and the claim stays on the statement. Each is
  recorded as the kind the reaction's rate shows for every value of its names within their
  bounds: inhibited where it never rises with it, activated where it never falls, modulated where
  neither can be shown. That rate is the row's net rate as the model builds it, read off the
  reactions its rules make in the whole model's expansion: the row's own agents, the states other
  rows give them and its bindings, its intermediates steady and its products absent, since where a
  step sits decides which way its own rate moves the reaction. A direction counts where sympy
  proves it and the rate, read at points across its names' ranges, moves no other way; it is
  signed again whenever a row, a binding or a bound changes. A regulator that rate does not change
  with, acting only on a step it cannot see (a product's degradation, a complex coming apart), is
  refused, pointing to that step's own row: a regulator is a claim about its statement's own
  reaction, so each claim keeps one meaning and a causal graph never puts it on another reaction.
  Products stay absent in that rate, so a regulator that acts only with the product present
  (feedback through it, autoregulation, autocatalysis) does not move it either: its refusal names
  the rows that state it, the product stated as a regulator too, or that step on a row of its own
  so.
  Any other claim the maths does not bear out is recorded as what it does show, beside what was
  stated, never refused: the edit reports it and asks the agent to reconsider, and `get_model` and
  the English mark it ("activated by sorafenib (from its rate)")
  ([`statements`](src/petab_mcp/statements.py), [`regulation`](src/petab_mcp/regulation.py)).
  INDRA has no statement about a
  reaction, so a causal graph gets the nearest one it has, wider than the claim
  (`statements.as_indra`: "inhibited by sorafenib" on RAF's step is `Inhibition(sorafenib, RAF)`).
  A condition input the entity map
  does not map to an agent is no agent, so a row's binding may read it (`kf="k*dose"`) and the
  condition table says what it is, unless a seed of an agent at its id alone makes it that agent.
  Which inputs are agents is the problem author's call, made in the entity map, or the modeller's,
  made by such a seed; one mapped either way is read only as the agent it is.
- **A constraint is a claim the model must make.** A session may start with causal constraints
  (`--constraints`), statements rows without bindings that each end `, directly` (one row's claim
  entails it) or `, indirectly` (the network the rows expand to carries it, with its sign), with
  `not ..., directly` and `X does not affect Y` forbidding what they name; `make_changes` refuses a
  model that breaks one, a parameter edit included, since a bound can turn a regulator. Each claim
  is a row's own, its regulators as their rates record them, so a constraint rests on the maths,
  never on what the agent wrote; `activates` is about the forms the constraints declare active,
  never inferred from the agent's rows. An indirect constraint is traced, as the MSB 2023 paper
  traces a signal: every molecule gets a tag site, and BioNetGen expands the rows' model with the
  subject's molecules tagged and each rule handing a driver's tag to what it changes or makes
  ([`tracing`](src/petab_mcp/tracing.py)). Which way a model moves an object is not decidable from
  its structure once it has feedback, so an indirect constraint holds wherever the trace moves it as
  claimed, whatever else the trace finds: that sign is the fit's to say
  ([`constraints`](src/petab_mcp/constraints.py)). With `--constraint-updates any`,
  `set_constraints` replaces them for whoever drives the session, logged with whether it loosens or
  tightens them; a version a new set breaks keeps the verdicts it was committed under, but is
  flagged, cannot be restored and no longer ranks as the best.
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
