# Plan: one fit pool for a sweep

**Status:** a proposal for review. Nothing here is implemented. It replaces the shared server that
#187 asks for. frohlich-lab/gemot#89, the client side of #187, changes with it.

## Goal

Let every run of a sweep fit on one pool of SLURM workers. Today each run's array task holds 100
CPUs for its whole run (`--cpus-per-task 100`, `PETAB_MCP_FIT_N_WORKERS=$SLURM_CPUS_PER_TASK`),
idle while the model thinks: 100 CPUs × 34 h for one large problem in the last sweep. With one
pool, the fits of all runs take turns on shared workers. Between fits, a run holds only what its
server needs.

Nothing else about a run changes. It keeps its own server, session, tools and settings.

## Why not one shared server

#187 asks for one `petab-mcp` process that hosts every run's session. A spike on 2026-10-31 showed
that this works on FastMCP 4.1.4, but it costs:

- **A tool surface per connection.** The arms change the schema of `make_changes`, and constraints
  and held-out data add tools. So each session needs a FastMCP of its own, and the server forwards
  each connection's calls to it.
- **Per-session settings.** Every setting that is per process today must become per session: the
  per-start cap (#188), and any setting added later.
- **Session plumbing:** `open_session`, `resume_session` and a durable `SessionSpec`.
- **An older protocol.** Routing by connection needs the handshake protocol (2025-11-25).
  FastMCP's own `Client` defaults to 2026-07-28, in which every call is a new connection.
- **Shared threads.** FastMCP's 40 worker threads would run every run's fits.
- **Shared failure.** One run's crash or wedge would take down every run.

All of that buys one thing: the fits share a pool. A pool can be shared without sharing the server.
`DaskEngine` already reaches its cluster through a `distributed.Client`, and a client can connect
to a scheduler in another process.

## Decisions

- **The pool is a service job, managed like the Phoenix collector.**
  - A long-lived SLURM job hosts the dask scheduler and adds SLURM worker jobs on demand.
  - `slurm/fitpool_ctl.sh start|status|stop` manages it.
  - It publishes its address to a file, which `make_sweep.py` and `array_task.sh` read, as they
    read `phoenix_ctl.sh`'s `endpoint.txt` for traces.
- **PEtab-MCP ships the pool's process,** `petab-mcp-pool`. The contract with the workers is the
  server's: no builds on a worker, AMICI with HDF5, the same environment. So the code that sets up
  the workers lives beside `DaskEngine`.
- **A server joins a pool by address:** `petab-mcp --fit-pool tcp://node:port` (or
  `PETAB_MCP_FIT_POOL`), or `create_server(cluster="tcp://node:port")`.
- **`--slurm-max-jobs` and `--slurm-min-jobs` move from `petab-mcp` to `petab-mcp-pool`.** A server
  never owns a cluster, so there is one way to fit on SLURM, not two. A single run that wants SLURM
  starts a pool.
- **Every run keeps its own server.** Its tools, arm, held-out set, constraints and per-start cap
  stay process settings, as today.
- **Each run's build root is its own directory on shared storage.** Workers import a run's modules
  from it by the same absolute path. No two servers write to one build root, so no lock has to hold
  across nodes.
- **The multi-session server is dropped.** `registry.py` and its errors go, and #188 and the
  shared-server branch are no longer needed.

## Design

### The pool (PEtab-MCP: `petab-mcp-pool`)

- **`petab-mcp-pool --slurm-max-jobs N [--slurm-min-jobs M] [--port 8786] --address-file PATH`:**
  - starts a `dask_jobqueue.SLURMCluster` with its scheduler on `--port` of this node, and calls
    `adapt(minimum_jobs=M, maximum_jobs=N)`. Every other SLURM setting comes from dask's
    `jobqueue.yaml`, as with `--slurm-max-jobs` today (`cli._slurm_pool`, which moves here);
  - registers `_NoBuild` on the workers itself, so a worker never compiles, even before any server
    connects. Each server's `DaskEngine` also registers it, under the same name, and a second
    registration replaces the first;
  - writes `tcp://<host>:<port>` to `--address-file` once the scheduler listens, and removes the
    file on exit;
  - on SIGTERM, removes the address file first, then closes the cluster, which cancels its worker
    jobs.
- **A fixed port** lets a restarted pool come back at the same address, as `PHOENIX_NODELIST` does
  for Phoenix.
- **One thread per worker process** (jobqueue's `processes` equal to `cores`), so each start has a
  core to itself. Otherwise contention would slow starts until the per-start cap stops them, which
  is why `DaskEngine` exists.
- **Adaptive scaling sees every run's load,** because the scheduler holds every run's tasks.

### The server (PEtab-MCP)

- **`--fit-pool ADDRESS`** passes the address to `create_server(cluster=...)`. `DaskEngine` hands
  it to `Client`, which accepts an address as it accepts a cluster. Like `--slurm-max-jobs` now, it
  needs `--build-root`.
- **The server checks the pool's versions when it connects.** It compares the `amici`, `pypesto`,
  `petab` and `petab_mcp` versions the scheduler process imports with its own, and a mismatch stops
  it at startup with both versions named.
  - Starts are pickled, so a pool started from another environment fails every fit, with errors
    that are hard to read.
  - The check catches a pin bump made without restarting the pool.
- **Nothing per fit changes.**
  - Each batch carries its own per-start cap (`DaskEngine.batched`), so runs with different caps
    share one pool: 1200 s for lucarelli, 12000 s for isensee.
  - Batch `j` of every fit is submitted at `priority=-j`, so fits from all runs interleave batch by
    batch, as the fits of one server do now.
- **Queue time is recorded.** `get_run_timing` gains `queue_seconds`: the part of each fit's wall
  time when its batches were pending and none was running. It is read from the scheduler, as
  `_overrun` reads executing times. gemot's `timing_split` already has a `queue` column for it.
- **A lost pool is an infrastructure error.** `_InfrastructureErrors` reports a `FitBackendError`
  caused by the client losing the scheduler as `infrastructure_error: fit_pool: …`, a new cause
  beside `errors.DISK_FULL`. gemot ends a run on any cause reported in that form
  (`infrastructure._SERVER_REPORT`).
  - A worker refusing to build reports the same cause: it means the build root is not shared.
  - A batch that kills its worker (`KilledWorker`, usually memory) stays a failed fit of this run.
- **Progress shows queueing.** `wait_for_fit` (#177, open) reports `starts_queued` beside
  `starts_done`, and its remaining-time estimate counts only running time.

### The sweep (gemot_iclr_2026 `slurm/`)

- **`fitpool_server.sh` and `fitpool_ctl.sh`,** modelled on `phoenix_server.sh` and
  `phoenix_ctl.sh`:
  - `start` submits the service job (1–2 CPUs, walltime past the sweep's) unless one is running.
    `FITPOOL_NODELIST` pins a restart to the old job's node;
  - `status` prints the job and the published address;
  - `stop` removes the address file, waits (bounded) until no task is processing, then sends
    SIGTERM. `stop --force` cancels at once, which fails every fit in flight.
- **`make_sweep.py --fit-pool`** refuses to submit unless the address is published and reachable,
  as `--phoenix` does.
- **`array_task.sh`** reads the address at task start, checks that it connects, and exports
  `PETAB_MCP_FIT_POOL`. It gives the server a build root under the run's directory on shared
  storage.
- **A task asks for what its server uses between fits:** compiling, scoring at nominal values, and
  the simulations for diagnostics and held-out scoring.
  - 8 CPUs, because compiling sets the size. The server compiles every new model version on the
    task's own node, and `array_task.sh` already caps `AMICI_PARALLEL_COMPILE` at 8: compiling is
    memory-bound per process. With fewer CPUs, every compile is slower than today's. The rest
    (simulations, the agent's driver) is mostly single-threaded.
  - Memory is sized again from measured MaxRSS, as `make_sweep.py` does now, without the 100 fit
    processes.
  - `--cpus-per-task 100` and the pinned `PETAB_MCP_FIT_N_WORKERS` go.
- **The QOS budget is split** between the pool's `--slurm-max-jobs` and `throttle ×
  cpus-per-task`. With 2000 CPUs, for example: 12 worker jobs of 100 CPUs, plus 100 runs of 8.

### gemot

- **`runner/server.py` passes `PETAB_MCP_FIT_POOL`** from its environment to
  `create_server(cluster=...)` in the bootstrap, and records it in `fit_settings`.
- **The fit budget stops assuming a private pool.** `fit_time_budget` sizes a fit as
  `ceil(n_starts / FIT_N_WORKERS)` waves. On a pool, a fit's wall time includes waiting behind
  other runs, which no formula bounds. So with a pool:
  - the deepagents driver waits for fits as the CLI drivers do: `make_changes`, then `wait_for_fit`
    (#177), instead of one call sized to the whole fit;
  - the stall watchdog counts each `still_running` answer as progress. `_fit_wedged` stays, for a
    server that stops answering.

  Without a pool, budgets stay as today.
- **A lost pool ends the run as `infrastructure_error: fit_pool`,** which blames the pool, not the
  run. `infrastructure.find` already reads whatever cause the server reports, so this needs no new
  code. `FIT_POOL` is added beside `DISK_FULL` only to give the cause a name. Every other error
  stays the run's own.
- **`queue_seconds`** goes from `get_run_timing` into `result.json`.

### What goes

- `src/petab_mcp/registry.py`, `tests/test_registry.py`, `NoSessionBoundError` and
  `SessionLockedError`. `filelock.py` stays, because the build lock uses it.
- `route()` (#17) and per-session log routing (#16, #18) stay. They cost nothing, and removing them
  would touch every tool.
- #188 (the per-start cap per session) is not needed. Proposed: close it unmerged.
- The shared-server branch `claude/187-shared-server-f5dbde` (local only) is not needed. Proposed:
  discard it.

## Checks

1. **Two servers, one pool.** Two servers on different problems fit at the same time on a
   `LocalCluster` started in another process. Batches of both run interleaved, and each server gets
   back exactly its own starts (PEtab-MCP).
2. **Caps per run.** The two servers run under different `PETAB_MCP_FIT_PER_START_TIMEOUT` values,
   and each batch is stopped at its own server's cap (PEtab-MCP).
3. **A version mismatch is refused** at startup, naming both versions. The test patches the
   versions the scheduler reports (PEtab-MCP).
4. **Same model, two build roots.** A worker runs batches from two runs whose identical models were
   compiled under two build roots, and both import (PEtab-MCP).
5. **A lost pool ends the fit as `fit_pool`.** Killing the scheduler mid-fit gives
   `infrastructure_error: fit_pool`, and a `KilledWorker` gives a failed fit (PEtab-MCP).
6. **A restarted pool at the same address serves the next fit** without restarting the server
   (PEtab-MCP, and on NEMO with `FITPOOL_NODELIST`).
7. **`queue_seconds`** equals the time a fit waited with nothing running, on a pool of one worker
   with two fits (PEtab-MCP).
8. **A queued fit does not end a run.** gemot's watchdog sees `still_running` with every start
   queued for longer than the old budget, and carries on (gemot).
9. **On NEMO:** a smoke sweep of 4 runs on a pool of 2 worker jobs, and a `fitpool_ctl.sh stop`
   with nothing processing that cancels the worker jobs (gemot_iclr_2026).

## Rollout

1. **PEtab-MCP:** `petab-mcp-pool`, `--fit-pool`, the version check, `queue_seconds`, the
   `fit_pool` infrastructure error and `starts_queued`, plus deleting the registry. These go in
   separate PRs of under about 500 lines each.
2. **gemot:** the bootstrap, the fit budget and watchdog under a pool, `FIT_POOL` and
   `queue_seconds`. Its pin moves to step 1, together with the edits #175 and #184 need.
3. **gemot_iclr_2026:** `fitpool_server.sh`, `fitpool_ctl.sh`, `make_sweep.py --fit-pool` and
   `array_task.sh`, and a bump of the gemot pin.
4. **Issues:** #187 and gemot#89 are rewritten to this plan.

PEtabAgentBench does not change.

## Open questions

1. **Share one build root across runs?** Runs of the same model would then compile it once. But the
   build lock (#20) would have to hold across nodes, which needs `flock` to work on NEMO's shared
   filesystem. Proposed: one build root per run first, then measure the duplicate compiles.
2. **Worker lifetime.** Jobqueue's worker `--lifetime` must exceed the largest per-start cap plus
   `FIT_DEADLINE_MARGIN_S` (isensee: 12000 s), or a worker retires in the middle of a batch. Check
   what dask does with a batch running on a retiring worker, and set the lifetime from the sweep's
   largest cap.
3. **Worker memory.** A worker keeps every model module it has imported, across problems, and only
   its lifetime bounds that. Is one memory size per worker enough when isensee and small problems
   share workers, or does a sweep with isensee want a pool of its own?
4. **Reconnecting.** Does `distributed.Client` reconnect by itself to a scheduler restarted at the
   same address, or does `DaskEngine` need a new client after a `fit_pool` error? Check 6 answers
   this.
5. **Who stops the pool** when a sweep ends? Proposed: the operator, as with Phoenix. The
   alternative is `make_sweep.py`, with a dependent job.
6. **Compile on the pool too?** A task would then need only 1–2 CPUs. But workers would then build
   into the runs' build roots, which `_NoBuild` forbids today to keep builds and their races in
   one place, the server. Proposed: not now.
