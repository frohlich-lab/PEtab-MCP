# The fit pool

Every run of a sweep fits on one pool of SLURM workers. Each run keeps its own server, session,
tools and settings; only the starts of its fits run on the shared workers. Before the pool, each
run's array task held 100 CPUs for its whole run, idle while the model thought: 100 CPUs × 34 h for
one large problem in one sweep. Now the fits of all runs take turns on the pool, and between fits a
run holds only what its server needs.

How to run one is in the README, "Fitting on SLURM". This document says how the parts fit together,
what was measured, and what is still open. It replaces the plan approved in #189; #187 asked for a
shared server instead and was closed as superseded.

## The parts

- **`petab-mcp-pool`** (`pool.py`) is the pool: a long-lived SLURM job whose process hosts a
  `dask_jobqueue.SLURMCluster` scheduler on `--port` of its node.
  - It submits `--slurm-max-jobs` worker jobs as it starts and keeps them until it stops. It does
    not `adapt`: a worker joins before it can take a batch (a cold `import petab_mcp`), so adapt saw
    idle workers beside queued batches and cancelled each new job (#250).
  - It writes its address to `--address-file` once the scheduler listens, and removes it on exit.
  - On SIGTERM it removes the address file, waits up to 10 min for the batches running, then closes
    the cluster, which cancels its jobs.
  - Every other SLURM setting comes from dask's own `jobqueue.slurm` config.
- **A server joins by address:** `petab-mcp --fit-pool ADDRESS`, `PETAB_MCP_FIT_POOL`, or
  `create_server(cluster=ADDRESS)`. It needs `--build-root` on storage the compute nodes share. A
  server never owns a cluster, so there is one way to fit on SLURM: a single run that wants one
  starts a pool.
- **`DaskEngine`** (`dask_engine.py`) is the server's side: a pyPESTO engine that sends a fit's
  starts to the pool and waits for them (`multistart.watch`).
- **`python -m petab_mcp.pool_stress`** loads a pool's scheduler as a sweep's servers would, and
  says whether it holds up (see "Measured on NEMO").
- **The sweep driver is gemot's** (frohlich-lab/gemot#107): it submits, finds and signals the
  pool's job, refuses a sweep whose pool is not published and reachable, and gives each task the
  pool's address and build root.

## What a server and the pool share

- **One environment.** A start is pickled, so the server compares the `amici`, `pypesto`, `petab`
  and `petab_mcp` versions the pool imports with its own, when it connects and at every fit, and
  refuses a mismatch naming both. After a pin bump, restart the pool.
- **One build root.** Workers never compile: `_NoBuild` refuses a build on a worker from the
  moment it starts, as the `fit_pool` error. They import each module from the server's build root at
  the same absolute path, and read each fit's problem from `fit_problems/` there, where the fit
  leaves it once for all its starts (#280).
  - `_NoBuild` is idempotent (#239, `distributed` ≥ 2023.12.0), so a server joining sends nothing
    to the workers. Without that, at 1200 workers each join stalled the scheduler for 6–37 s.
  - The build root is keyed by AMICI build and module hash, so runs of one model compile it once.
- **One build at a time per structure.** Servers compile on their own nodes. Two that reach one
  structure take its build lock, a POSIX `lockf` under a per-path mutex (#240): GPFS's `flock` is
  node-local, `lockf` holds across nodes. The second waits, then imports the finished module. A
  waiter whose holder died removes the unfinished folder and builds it itself.

## A fit on the pool

- **Batches.** A fit sends its starts in dask tasks of `k` starts each, run one after another on one
  worker. `k` fills `PETAB_MCP_FIT_TASK_SECONDS` (270 s) at the 90th-percentile start time of the
  session's previous fit, or at its per-start cap without one, but is at most
  `ceil(n_starts / idle workers)`, so a fit on a quiet pool spreads out (#254).
  `PETAB_MCP_FIT_STARTS_PER_TASK` fixes `k` instead. Long batches keep the shared scheduler's state
  changes few.
- **Turns.** Batch `j` of every fit goes in at `priority=-j`, so the fits of all runs interleave
  batch by batch.
- **Caps per run.** Each batch carries its own session's per-start cap (#188), so runs with
  different caps share one pool. A batch still running its caps plus 600 s after it began is
  cancelled, and its starts are scored as failed. Its worker runs on until the batch returns: dask
  cannot stop a running task.
- **Threads.** A worker runs one batch at a time. With `PETAB_MCP_FIT_AMICI_N_THREADS` = T, AMICI
  simulates a start's experiments on T threads (#258): keep jobqueue's `cores` equal to `processes`
  and book `job-cpu` = processes × T.
- **Queueing is visible.**
  - `get_run_timing` reports `queue_seconds`: the part of each fit when none of its batches was
    running.
  - The change's phase is `compiling`, `queued` or `running`. `wait_for_fit` reports it with
    `starts_queued` and `remaining_s`, which counts running time only.
  - Every tool call sends an MCP progress notification every 30 s while it runs (`_Heartbeat`),
    with the phase and the starts done. gemot extends a call's timeout on each one instead of
    budgeting a fit's wall time (frohlich-lab/gemot#106).
- **Failures.**
  - A pool lost mid-fit ends the call as `infrastructure_error: fit_pool`, which gemot reads as the
    pool's failure, not the run's.
  - The client reconnects by itself to a pool back within 30 s. After that, the next fit opens a new
    client (#244), so a pool restarted at the same address serves it without restarting the server.
  - A worker killed under a batch (`KilledWorker`: memory, a crash, or its job ending) fails the
    fit with a message that names the pool, not the model; nothing is scored.

## Measured on NEMO

On 2026-11-04, with `pool_stress` at 1200 workers and 100 clients each sending a fit's batches in
one burst:

- the scheduler's event loop fell behind by at most 2.4 s, no worker went more than 9.1 s without a
  heartbeat, and the scheduler peaked at 1.4 GB. So the pool's own job is 2 CPUs and 8 GB, and
  `worker-ttl` is 10 min;
- it passed only once `_NoBuild` was idempotent.

Also found there:

- **Worker jobs are not replaced.** With several workers to a job, jobqueue never resubmits one
  that ends, and a worker reaching `--lifetime` drops its running batch. So worker jobs ask for a
  walltime past the pool's life, set no `--lifetime`, and the pool is restarted between sweeps.
- **A full job's workers need time to start.** Importing from GPFS takes 48–98 s cold, so
  `death-timeout` is 600 s, not dask's 60.
- **NEMO moves any job of 6 h or less to QOS `short`,** which runs only on 12 nodes (cn077–cn088)
  that are often full. So the pool's job and its worker jobs ask for more than 6 h.

## Decisions

- **One pool, a server per run,** rather than one server hosting every run's session. A shared
  server would need a tool surface per connection (the arms change `make_changes`' schema), every
  per-process setting moved onto the session first, the older connection-routed MCP protocol, and
  FastMCP's 40 threads shared by every run's fits. One run's crash or wedge would end them all. The
  pool shares the fits alone.
- **The pool is a service, run like the Phoenix collector:** the operator starts and stops it, and a
  sweep never does. A pool serves one environment, so a repository on its own pin runs its own
  pool.
- **Compiling stays on the run's node,** never on a worker.
- **A fit returns when its slowest batch has.** A cutoff that stops the last few starts once most
  are done was considered (2026-11-07) and not adopted. Other runs' batches fill the workers a fit's
  tail leaves idle, and a sweep should run more sessions than the pool can fit at once.
- **One session fits one change at a time.** `make_changes` refuses a second while a fit runs: each
  version builds on its parent and warm-starts from its optima.

## Open

- **The sweep driver** (frohlich-lab/gemot#107, PR frohlich-lab/gemot#114) is not merged. Until it
  is, the application repositories still carry their own copies of `slurm/`, and drop them at the
  gemot pin bump that brings the driver.
- **Worker memory.** A worker keeps every module it has imported, across problems, until the pool
  restarts. Whether a sweep with isensee needs a pool of its own is not measured.
- **The CLI drivers.** Whether Codex and Claude Code extend a tool call on progress notifications
  is not checked. Until it is, their models poll `wait_for_fit`.
- **Per-session settings.** The per-start cap is the session's (#188). The AMICI tolerances and the
  rest of what a sweep sets per problem through the environment are still read once per process;
  the pool needs nothing from them, since they travel with each start on the pickled solver.
