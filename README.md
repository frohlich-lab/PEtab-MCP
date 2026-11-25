# PEtab MCP

[![CI](https://github.com/frohlich-lab/PEtab-MCP/actions/workflows/ci.yml/badge.svg)](https://github.com/frohlich-lab/PEtab-MCP/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## Disclaimer

Research software under active, rapid and heavy development: **use it at your own risk**. Tool
names, schemas and agent-facing behaviour change without notice. Its behaviour has been tuned and
checked on benchmark problems only, and it is not for clinical or regulatory use. It comes
without warranty of any kind; see the [LICENSE](https://github.com/frohlich-lab/PEtab-MCP/blob/main/LICENSE).

## Overview

A local [MCP](https://modelcontextprotocol.io) server that wraps a single
[PEtab v2](https://petab.readthedocs.io/en/latest/v2/documentation_data_format.html)
parameter-estimation problem (a PEtab v1 problem is upgraded on load) and turns mechanistic
model building into a **disciplined, auditable** workflow for LLM agents — the loop behind
model-building agents such as [gemot](https://github.com/frohlich-lab/gemot).

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

## Install

```bash
pip install petab-mcp   # installs AMICI + pyPESTO (a C++ toolchain is required)
```

The `statements` route (a server given `--entity-map`) builds models from INDRA statements as
PySB rules and needs the `statements` extra. No released pysb installs beside AMICI, so name its
commit:

```bash
pip install "petab-mcp[statements]" "pysb @ git+https://github.com/pysb/pysb.git@9fc6bcb118683ad851e0a4d5604abe646f77cae3"
```

From a checkout, the `pins` group (pip >= 25.1) adds the unreleased commits of pysb and petab,
which PyPI refuses as dependencies. The petab commit imports a problem of thousands of
experiments (Froehlich_CellSystems2018):

```bash
pip install -e '.[statements]' --group pins
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
# env PETAB_MCP_FIT_AMICI_N_THREADS=T   (AMICI threads per start, default 1; workers x T <= cores; on a pool, job-cpu = processes x T)
```

Provenance is written to `<problem_dir>/.petab_mcp/` by default and is durable — restarting
resumes the latest version (a store written by the PEtab v1 server is refused). The server's
stderr goes to `stderr.log` there, beside `petab_mcp.log` and the fit log `pypesto.log`. A ready toy
problem lives at `tests/data/toy_v2/`; `tests/data/toy_v1/` is the same problem in v1 form.

## Fitting on SLURM

The fits of many servers, a sweep's runs say, share one pool of SLURM workers. `petab-mcp-pool`
runs the pool as a SLURM job of its own, and a server joins it with `--fit-pool ADDRESS` (or
`PETAB_MCP_FIT_POOL`). Both need the `dask` extra (`pip install -e '.[dask]' --group pins`).

```bash
petab-mcp-pool --slurm-max-jobs N --address-file PATH   # --port 8786
petab-mcp --petab-yaml problem.yaml --build-root DIR --fit-pool "$(cat PATH)"
```

The pool's scheduler listens on `--port` of the pool's node, so a pool restarted there comes back at
the same address, and the address goes to `--address-file` once it listens. The pool submits N
worker jobs as it starts and keeps them, idle or not, until it stops, so start it for a sweep and
stop it after. It does not scale with its queue: dask's `adapt` cancelled each new job before its
workers could take a batch (#250). On SIGTERM it removes the address file, waits up to 10 min for
the batches running, then cancels its jobs, failing the fits still waiting. It needs:

- **every server inside a SLURM allocation**, never on the login node: a server compiles every model
  itself, and a worker never does;
- **one `--build-root` on storage the compute nodes share**, the same for every server of the pool,
  not `$TMPDIR`: the workers import the compiled models by the server's path, and read each fit's
  problem from its `fit_problems/`, where the fit leaves it once for all its starts. Servers on two
  nodes building one model wait for each other through a POSIX (`fcntl`) lock there, so the
  storage's locks must hold across nodes, as GPFS's do;
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

A worker job is not replaced when it ends: with several workers to a job, jobqueue never resubmits
it, and the pool stays smaller. Its running batches are rerun on other workers. So give worker jobs
a walltime longer than the pool will run, and restart the pool between sweeps. A worker `--lifetime`
buys nothing: a worker reaching it drops its running batch, as a killed one does. The default
`death-timeout` of 60 s is too short for a full job's workers to import from shared storage.

A dask task runs as many starts as fill `PETAB_MCP_FIT_TASK_SECONDS` (default 270) at the
90th-percentile start time of the session's previous fit, or at its per-start cap (by default
`PETAB_MCP_FIT_PER_START_TIMEOUT`) without one, but no more than spread the fit over the pool's idle
workers; `PETAB_MCP_FIT_STARTS_PER_TASK` fixes the number instead. A pool lost mid-fit, or a worker
that cannot import a model, ends the call with `infrastructure_error: fit_pool: `, as a full disk
does (see "Tools"). How the pool's parts fit together, and what was measured on NEMO, is in
[docs/fit-pool.md](docs/fit-pool.md).

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

## Design notes

The principles to hold onto when extending this package are in [docs/DESIGN.md](docs/DESIGN.md).

## Development

```bash
pip install -e '.[dev]' --group pins   # AMICI + pyPESTO are core deps; dev adds pytest + ruff
pytest                    # needs the full install (AMICI/pyPESTO are hard deps); PubMed is stubbed
```

## Citation

If you use PEtab MCP, please cite the preprint it was introduced in:

> T. de Pomereu, B. Snelling, F. Fröhlich. GEMOT: Towards Mechanistic World Models for Biology.
> bioRxiv (2026). [doi:10.64898/2026.09.26.754619](https://doi.org/10.64898/2026.09.26.754619)

[CITATION.cff](CITATION.cff) has the same in machine-readable form (GitHub's "Cite this
repository").

## License

MIT
