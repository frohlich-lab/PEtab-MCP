# Plan: one fit pool for a sweep

**Status:** approved. It replaces the shared server that #187 asked for (closed as superseded).
PEtab-MCP's part is on main except Check 6, which #216 finishes. gemot's part is
frohlich-lab/gemot#105, #106 and #107, which replace frohlich-lab/gemot#89.

## Goal

Let every run of a sweep fit on one pool of SLURM workers. Today each run's array task holds 100
CPUs for its whole run (`--cpus-per-task 100`, `PETAB_MCP_FIT_N_WORKERS=$SLURM_CPUS_PER_TASK`),
idle while the model thinks: 100 CPUs × 34 h for one large problem in the last sweep. With one
pool, the fits of all runs take turns on shared workers. Between fits, a run holds only what its
server needs.

Nothing else about a run changes. It keeps its own server, session, tools and settings.

## Why not one shared server

#187 asked for one `petab-mcp` process that hosts every run's session. A spike on 2026-10-31 showed
that this works on FastMCP 4.1.4, but it costs:

- **A tool surface per connection.** The arms change the schema of `make_changes`, and constraints
  and held-out data add tools. So each session needs a FastMCP of its own, and the server forwards
  each connection's calls to it.
- **Every per-session setting first.** Every setting that is per process today would have to become
  per session before one server could host two runs: the per-start cap (#188), and any setting
  added later. Per-session settings are wanted anyway (see "Later"), but sharing a pool should not
  have to wait for all of them.
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
  - A control script starts, inspects and stops it, as `phoenix_ctl.sh` does the collector.
  - It publishes its address to a file, which the sweep driver reads, as it reads the collector's
    `endpoint.txt`.
- **The sweep driver lives in gemot.** Today Perica/Garrido, Fontaine/Briscoe and Vuillemenot each
  carry their own copy of gemot_iclr_2026's `slurm/`. With the pool, one driver lives in gemot, and
  the application repos keep only what is theirs: their problems, arms and settings.
  - A pool serves one environment: the version check refuses a pool from another gemot or server.
    So an application repo on its own pin runs its own pool.
- **gemot_iclr_2026 does not change.** Its pins are frozen with the paper, so its sweeps keep their
  private pools.
- **`petab-mcp-pool` does everything but the SLURM submission:** readiness, the address file and
  the drain on SIGTERM. gemot's control script only submits, finds and signals the job.
- **PEtab-MCP ships the pool's process,** `petab-mcp-pool`. The contract with the workers is the
  server's: no builds on a worker, and the same environment. So the code that sets up the workers
  lives beside `DaskEngine`.
- **A server joins a pool by address:** `petab-mcp --fit-pool tcp://node:port` (or
  `PETAB_MCP_FIT_POOL`), or `create_server(cluster="tcp://node:port")`.
- **`--slurm-max-jobs` and `--slurm-min-jobs` move from `petab-mcp` to `petab-mcp-pool`.** A server
  never owns a cluster, so there is one way to fit on SLURM, not two. A single run that wants SLURM
  starts a pool.
- **Every run keeps its own server,** with its own tools, arm, held-out set, constraints and
  per-start cap. The pool needs no per-session settings. Moving settings onto the session is wanted
  on its own terms, later (see "Later").
- **One build root for the pool, on shared storage.** Runs of the same model compile it once, and
  workers import every module from it by the same absolute path.
  - The build root is already keyed by AMICI build and module hash, so a pool started from another
    environment builds into folders of its own.
  - The build lock (#20) keeps two builds of one structure apart. It must hold across nodes. On
    NEMO's GPFS `flock` does not, so it is a POSIX `lockf` under a per-path mutex (#240), checked
    there across two nodes.
- **Compiling stays on the run's node,** not on the pool. `_NoBuild` keeps builds, and their races,
  in the servers.
- **The operator starts and stops the pool,** as with Phoenix. A sweep never stops it.
- **The multi-session server is not built now,** because the pool does not need it.

## Design

### The pool (PEtab-MCP: `petab-mcp-pool`)

- **`petab-mcp-pool --slurm-max-jobs N [--slurm-min-jobs M] [--port 8786] --address-file PATH`:**
  - starts a `dask_jobqueue.SLURMCluster` with its scheduler on `--port` of this node, and calls
    `adapt(minimum_jobs=M, maximum_jobs=N)`. Every other SLURM setting comes from dask's
    `jobqueue.yaml`, as with `--slurm-max-jobs` today (`cli._slurm_pool`, which moves here);
  - registers `_NoBuild` on the workers itself, so a worker never compiles, even before any server
    connects. Each server's `DaskEngine` registers it too. It is idempotent (#239, distributed
    ≥ 2023.12.0), so a server joining a pool sends nothing to its workers: at 1200 workers, one
    broadcast per server stalled the scheduler for 6–37 s;
  - writes `tcp://<host>:<port>` to `--address-file` once the scheduler listens, and removes the
    file on exit;
  - on SIGTERM, removes the address file first, waits (bounded) until no task is processing, then
    closes the cluster, which cancels its worker jobs.
- **A fixed port** lets a restarted pool come back at the same address, as `PHOENIX_NODELIST` does
  for Phoenix.
- **One thread per worker process** (jobqueue's `processes` equal to `cores`), so each start has a
  core to itself. Otherwise contention would slow starts until the per-start cap stops them, which
  is why `DaskEngine` exists.
- **Adaptive scaling sees every run's load,** because the scheduler holds every run's tasks.
- **One scheduler carries every run's load.** It handles:
  - every worker's heartbeats: up to `--slurm-max-jobs` × processes per job, about 1200 at 12 jobs
    of 100;
  - every batch's state changes;
  - one client per server;
  - one `_overrun` poll (`run_on_scheduler(_executing)`) per running fit.

  If its event loop falls behind, it can miss heartbeats and drop a worker past
  `distributed.scheduler.worker-ttl`. It then reruns that worker's batches elsewhere: the work is
  lost, and their starts' deadlines restart. So:
  - batches stay long (about 4.5 min, `PETAB_MCP_FIT_TASK_SECONDS`), which keeps state changes
    few;
  - `worker-ttl` is set well above the longest stall the stress check (Checks 10) measures;
  - the service job's resources come from that check, not from a guess.

  On NEMO, at 1200 workers and 100 clients, the loop fell behind by at most 2.4 s, no worker went
  more than 9.1 s without a heartbeat, and the scheduler peaked at 1.4 GB. So the service job gets
  2 CPUs and 8 GB, and `worker-ttl` is 10 minutes (README, "Fitting on SLURM").
- **Worker jobs outlive the pool.** With several workers to a job, jobqueue never replaces a job
  that ends, and a worker reaching `--lifetime` drops its running batch. So worker jobs ask for a
  walltime past the pool's life and set no `--lifetime`, and the pool is restarted between sweeps.

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
- **A build another run holds is waited for, not refused.** Today the second builder of a structure
  fails at once with `ModelBuildBusyError` (`engine.build_lock`). Across runs that would be common,
  because every run of a problem scores the same shipped model at startup, and agents converge on
  the same structures.
  - So the lock is taken blocking: the waiter then finds the module finished and imports it.
  - If the holder died, its lock was released with it, and the waiter removes the unfinished folder
    and builds the module itself.
  - The wait is the `compiling` phase of the heartbeat.
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
- **Every call sends a heartbeat.** A middleware beside each tool call (as `_Cancellation` is)
  sends an MCP progress notification every 30 s while the call runs, on the call's own progress
  token (`Context.report_progress`).
  - It reports the session's running change: starts done out of `n_starts`, and its phase
    (`compiling`, `queued` with batches waiting for a worker, or `running`), with the seconds spent
    in that phase.
  - It runs on the event loop, so it holds no thread and changes no tool.
  - A call without a progress token gets no heartbeat.
- **Progress shows queueing.** The change's `Progress` (#177) gains the phase and
  `starts_queued`. `wait_for_fit` reports both, and its remaining-time estimate counts only running
  time.

### The sweep driver (gemot, frohlich-lab/gemot#107)

gemot takes in the sweep driver the application repos copied from gemot_iclr_2026, and adds what
it already does for the Phoenix collector (`phoenix_ctl.sh`, `make_sweep.py --phoenix`).
gemot_iclr_2026 stays as it is (see Decisions).

- **A service job and its control script.** The script only submits, finds and signals the job;
  `petab-mcp-pool` does the rest.
  - `start` submits the pool's job unless one is running. Its resources come from the stress check
    (Checks 10), and its walltime runs past the sweep's. A node pin restarts it on the old job's
    node. On NEMO a job of 6 h or less is moved to QOS `short`, which runs only on cn077–cn088, so
    the pool's job and its worker jobs ask for more;
  - `status` prints the job and the published address;
  - `stop` sends SIGTERM, and `petab-mcp-pool` then removes the address file, waits (bounded) until
    no task is processing, and closes the cluster. `stop --force` cancels at once, which fails
    every fit in flight.
- **Submission is refused** unless the address is published and reachable, as `--phoenix` does.
- **Each task** reads the address when it starts, checks that it connects, and exports
  `PETAB_MCP_FIT_POOL`. It passes the pool's build root as the server's `--build-root`. The operator
  clears that folder; nothing else does.
- **A task asks for what its server uses between fits:** compiling, scoring at nominal values, and
  the simulations for diagnostics and held-out scoring.
  - 8 CPUs, because compiling sets the size. The server compiles every new model version on the
    task's own node, and gemot_iclr_2026's `array_task.sh` caps `AMICI_PARALLEL_COMPILE` at 8:
    compiling is memory-bound per process. With fewer CPUs, every compile is slower than today's.
    The rest (simulations, the agent's driver) is mostly single-threaded.
  - Memory is sized again from measured MaxRSS, without the 100 fit processes.
  - The 100 CPUs per task and the pinned `PETAB_MCP_FIT_N_WORKERS` go.
- **The QOS budget is split** between the pool's `--slurm-max-jobs` and `throttle ×
  cpus-per-task`. With 2000 CPUs, for example: 12 worker jobs of 100 CPUs, plus 100 runs of 8.

### gemot (frohlich-lab/gemot#105, #106)

- **`runner/server.py` passes `PETAB_MCP_FIT_POOL`** from its environment to
  `create_server(cluster=...)` in the bootstrap, and records it in `fit_settings`.
- **A call's timeout becomes an idle timeout.** `fit_time_budget` sizes a fit as
  `ceil(n_starts / FIT_N_WORKERS)` waves of the per-start cap. On a pool, a fit's wall time also
  includes waiting behind other runs, which no formula bounds. So the deepagents driver gives a call
  no fixed timeout and lives on the server's heartbeats instead, with no polling:
  - `_wrap_call_timeout` runs the call in an anyio cancel scope, and each heartbeat pushes the
    scope's deadline out by a window of about 5 min. mcp 2.3 cannot do this itself, because its
    request clock only pauses for human input. The adapter's own progress callback is still called;
  - with no heartbeat inside the window, the call takes today's timeout path (`_fit_wedged`). The
    server is not told to stop the fit (`_ToolCallsRunOn`);
  - a change in `compiling` for longer than `FIT_COMPILE_MARGIN_S` counts as no heartbeat, so a hung
    compile is still caught;
  - the stall watchdog (`_STEP_STALL_TIMEOUT_S`) is fed by the same heartbeats;
  - the run's wall-time budget is the hard ceiling. It leaves out the run's `queue_seconds`, so a
    run that waited for the pool is not left less time to think.

  This applies to every call, with or without a pool. A read waiting behind a fit gets heartbeats
  too. It replaces the fit arithmetic in `_wrap_call_timeout`: the waves × cap budget, and the
  `in_flight` term that held reads queued behind fits.
- **The CLI drivers keep `wait_for_fit`.** Codex and Claude Code cut a call at their own 600 s tool
  timeout, whatever arrives on it. Their model polls `wait_for_fit` (#177), about once per 9 min of
  queueing.
- **A lost pool ends the run as `infrastructure_error: fit_pool`,** which blames the pool, not the
  run. `infrastructure.find` already reads whatever cause the server reports, so this needs no new
  code. `FIT_POOL` is added beside `DISK_FULL` only to give the cause a name. Every other error
  stays the run's own.
- **`queue_seconds`** goes from `get_run_timing` into `result.json`.

### What goes

- **AMICI's HDF5 requirement.** AMICI 1.2.0 pickles a solver as a dict of its settings
  (`restore_solver`), not through HDF5. So these go:
  - `DaskEngine`'s `hdf5_enabled` refusal and its test;
  - the sentence about it in `create_server`'s docstring;
  - CI's `libhdf5-dev` install, and the `hdf5` prefix on its cache key.

  pyPESTO's result file is written by h5py, which brings its own HDF5, so it is not affected.
- **`registry.py`** goes, with `tests/test_registry.py`, `NoSessionBoundError` and
  `SessionLockedError`. `filelock.py` stays, because the build lock uses it. Git keeps the registry
  for a multi-session server later.
- **`route()` (#17) and per-session log routing (#16, #18) stay.** They cost nothing, and removing
  them would touch every tool.

## Checks

Checks 1–5, 7 and 8 have tests on main. Checks 4 and 10 also passed on NEMO on 2026-11-04, Check
10 only after #239. Check 6 is #216's last item. Checks 9 and 11 are gemot's
(frohlich-lab/gemot#106, #107).

1. **Two servers, one pool.** Two servers on different problems fit at the same time on a
   `LocalCluster` started in another process. Batches of both run interleaved, and each server gets
   back exactly its own starts (PEtab-MCP).
2. **Caps per run.** The two servers run under different `PETAB_MCP_FIT_PER_START_TIMEOUT` values,
   and each batch is stopped at its own server's cap (PEtab-MCP).
3. **A version mismatch is refused** at startup, naming both versions. The test patches the
   versions the scheduler reports (PEtab-MCP).
4. **Two runs, one structure.** Two servers build the same model at once. One builds it, and the
   other waits, then imports it. A waiter whose holder is killed builds the module itself
   (PEtab-MCP; across two nodes on NEMO's mount).
5. **A lost pool ends the fit as `fit_pool`.** Killing the scheduler mid-fit gives
   `infrastructure_error: fit_pool`, and a `KilledWorker` gives a failed fit (PEtab-MCP).
6. **A restarted pool at the same address serves the next fit** without restarting the server
   (PEtab-MCP, and on NEMO with `FITPOOL_NODELIST`).
7. **`queue_seconds`** equals the time a fit waited with nothing running, on a pool of one worker
   with two fits (PEtab-MCP).
8. **Heartbeats name the phase.** A call through a fit gets heartbeats at the set interval:
   `compiling`, then `queued` on a pool with no free worker, then `running`. A read waiting on the
   fit's lock gets them too (PEtab-MCP).
   - The test runs on the protocol gemot negotiates, which is 2025-11-25 from its raw
     `ClientSession.initialize()`.
   - `fastmcp.Client` defaults to `mode="auto"`, which gets 2026-07-28, where every request is its
     own connection. So a test through `Client(mcp)` passes `mode="legacy"`.
   - Progress notifications have not yet been tried on either protocol.
9. **A queued fit does not end a run.** A fit queued for longer than the old budget keeps its call
   alive on heartbeats and returns its result. A server that stops sending heartbeats ends the call
   within the window, through `_fit_wedged` (gemot).
10. **The scheduler holds up under a sweep's load.** This runs on NEMO, on the pool at full size
    with every worker job up. As many clients as a sweep has servers (100, say) each submit a fit's
    batches in one burst, then poll as `_overrun` does. The batches are short sleeps, so the
    scheduler is under test, not the fits. It passes when:
    - no worker is removed for missed heartbeats;
    - no batch runs twice (each batch records its runs);
    - the scheduler's event loop stays responsive, with no "Event loop was unresponsive" warning
      past a few seconds.

    It sets the service job's resources and `worker-ttl` (PEtab-MCP, a script beside
    `petab-mcp-pool`).
11. **On NEMO:** a smoke sweep of 4 runs on a pool of 2 worker jobs, and a `stop` with nothing
    processing that cancels the worker jobs (gemot's driver).

## Rollout

1. **PEtab-MCP:**
   - `petab-mcp-pool` and `--fit-pool`;
   - the version check;
   - `queue_seconds`;
   - the `fit_pool` infrastructure error;
   - the heartbeat, and the phase and `starts_queued` in `Progress`;
   - the build lock waiting for another run's build;
   - the scheduler stress script (Checks 10), run before any sweep shares a pool;
   - deleting the HDF5 requirement and the registry.

   These go in separate PRs of under about 500 lines each. The heartbeat does not depend on the
   pool, so it can land first. Done (#188, #221–#226, #239–#241), except Check 6 (#216).
2. **gemot** (frohlich-lab/gemot#105, #106, #107):
   - the bootstrap;
   - the idle timeout in place of the fit budget, and the watchdog fed by heartbeats;
   - the wall-time budget without `queue_seconds`;
   - `FIT_POOL` and `queue_seconds`;
   - the sweep driver, taken in from the application repos' copies, with the pool's control
     script, the submission check and each task's pool address.

   Its pin moves to step 1, together with the edits gemot#99 lists.
3. **Application repos** (Perica/Garrido, Fontaine/Briscoe, Vuillemenot): drop their copied
   `slurm/` for gemot's driver, at the gemot pin bump that brings it.
4. **Issues:** #187 is closed as superseded, and frohlich-lab/gemot#89 is replaced by
   frohlich-lab/gemot#105, #106 and #107.

PEtabAgentBench does not change.

## Later: per-session settings

The pool does not need them, but per-session settings are wanted. A setting should belong to the
session that uses it, not be read from the environment once, at import. That is also what a server
hosting several sessions would need first, if one is ever built.

- **#188** moves the per-start cap onto the session. It is independent of this plan, and is
  reviewed and merged on its own.
- **Next candidates:**
  - the rest of what a sweep sets per problem through the environment, i.e. the fit settings
    `fit_budget.server_fit_settings` records;
  - among them, the AMICI tolerances (`PETAB_MCP_FIT_AMICI_ATOL`, `_RTOL`, `_RTOL_STEADY_STATE`,
    `_ATOL_STEADY_STATE`), which `fitting.py` reads once at import, as it does the cap. gemot
    exports them per problem, as for raimundez. The pool needs nothing from them, because they are
    set on the server's solver, which travels pickled with each start;
  - `reference_mode`.

  The arm's files (`entity_map`, `constraints`) already belong to the session.
- **The shared-server branch** `claude/187-shared-server-f5dbde` (local only) adds these fields to
  `SessionSpec`. It is kept, unpushed, as a reference.

## Open questions

1. **Worker lifetime.** Answered on NEMO: a worker reaching `--lifetime` drops its running batch
   within 20 ms instead of finishing it, and with several workers to a job, jobqueue never replaces
   the job. So worker jobs set no `--lifetime` and ask for a walltime past the pool's life (see
   "The pool").
2. **Worker memory.** A worker keeps every model module it has imported, across problems, and only
   the pool's restart bounds that. Is one memory size per worker enough when isensee and small
   problems share workers, or does a sweep with isensee want a pool of its own?
3. **Reconnecting.** Answered on NEMO: a client reconnects by itself to a pool back within its 30 s
   window (18 s for a restart in place), but not to one resubmitted (about 4 min). After that,
   `DaskEngine` opens a new client: Check 6 (#216).
4. **Do the CLI drivers' clients extend a call on progress?** If Claude Code or Codex reset their
   tool timeout on progress notifications, those arms need no `wait_for_fit` either. Check this
   before relying on it.
5. **Fix #180 first?** With heartbeats, a `get_progress` or `get_run_timing` behind a fit no longer
   times out, but it still waits out the fit, queue included. If those reads stop taking the lock
   the change holds, they answer at once. Being fixed.
