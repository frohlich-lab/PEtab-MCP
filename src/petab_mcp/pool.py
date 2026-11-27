"""``petab-mcp-pool``: the service process that owns a fit pool, which servers join by address
(``petab-mcp --fit-pool ADDRESS``) to run their fits' starts on shared SLURM workers.

It runs a ``dask_jobqueue.SLURMCluster`` whose scheduler lives in this process, on ``--port`` of
this node, so a pool restarted on the same node comes back at the same address. Every SLURM setting
but the number of jobs comes from dask's own ``jobqueue.slurm`` config (README, "Fitting on SLURM").

* **The pool scales with its batches, in whole jobs** (:class:`_Jobs`), from ``--slurm-min-jobs``
  to ``--slurm-max-jobs``: a worker for every batch queued or running, a job none of whose workers
  has run a batch for ``--slurm-idle-minutes`` cancelled, and a job whose workers have all left
  replaced while batches wait.
* **Workers never build** (:class:`~petab_mcp.dask_engine._NoBuild`), from the moment they start,
  not only once a server has connected.
* **The address is published** in ``--address-file`` once the scheduler listens, and removed on
  exit.
* **SIGTERM (or SIGINT) drains.** The address file goes first, so no new server joins. Then the pool
  waits until no batch is processing, at most :data:`DRAIN_S`, and closes the cluster, which
  cancels its jobs and fails the fits still waiting on it.

Submitting this process as a SLURM job, finding it and signalling it is the sweep driver's (gemot).
"""

from __future__ import annotations

import argparse
import math
import signal
import sys
import time
from pathlib import Path
from typing import Any

import dask
from dask.utils import parse_timedelta
from distributed.deploy import Adaptive

#: Seconds a drain waits for the batches processing: about two batches
#: (``fitting.FIT_TASK_SECONDS``). A stop that will not wait is the driver's ``scancel``.
DRAIN_S = 600.0

_SIGNALS = {signal.SIGTERM, signal.SIGINT}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="petab-mcp-pool",
        description="Run a dask fit pool on SLURM that petab-mcp servers join with --fit-pool.",
    )
    parser.add_argument(
        "--slurm-max-jobs",
        type=int,
        required=True,
        metavar="N",
        help="SLURM jobs the pool grows to while batches wait for a worker. The jobs are set up in "
        "dask's jobqueue.slurm config.",
    )
    parser.add_argument(
        "--slurm-min-jobs",
        type=int,
        default=0,
        metavar="M",
        help="SLURM jobs the pool keeps while idle, at most N (default 0); M = N keeps them all.",
    )
    parser.add_argument(
        "--slurm-idle-minutes",
        type=float,
        default=30.0,
        metavar="T",
        help="Minutes a job none of whose workers runs a batch is kept before it is cancelled "
        "(default 30): a job submitted again waits in SLURM's queue again.",
    )
    parser.add_argument(
        "--port", type=int, default=8786, help="The scheduler's port on this node (default 8786)."
    )
    parser.add_argument(
        "--address-file",
        type=Path,
        required=True,
        metavar="PATH",
        help="Where the scheduler's address is written once it listens; removed on exit.",
    )
    return parser


class _Jobs(Adaptive):
    """dask's ``adapt`` in whole SLURM jobs, for a pool each of whose tasks is a batch that holds one
    single-threaded worker for minutes.

    dask's own rules cancelled every job this pool submitted, and no fit got a worker (#250). They
    are replaced, not tuned:

    * **The target is the batches queued or running**, a worker each. ``Scheduler.adaptive_target``
      drops to the workers it would not retire, and it retires an idle one whatever is queued: a
      job's workers are idle from joining until they have imported ``petab_mcp``, 48-98 s cold from
      shared storage. It also counts a batch never seen as 0.5 s against a 5 s goal, a tenth of a worker.
    * **Only a whole job is closed**, one none of whose workers has run a batch for ``idle_s``,
      newest first, its workers retired first so that no batch lands on them meanwhile. dask closes
      single workers, and ``SpecCluster`` cancels the job of each, busy workers included.
    * **A job whose workers have all left** (its walltime, a failed node) is closed after
      ``distributed.deploy.lost-worker-timeout``, so the next poll replaces it while batches wait.
      ``SpecCluster`` does this only for a job of one worker: it matches a worker's name to its
      job's, and a job of several names them ``<job>-<i>``.
    """

    def __init__(self, cluster, idle_s: float, **kwargs) -> None:
        super().__init__(cluster, **kwargs)
        self.idle_s = idle_s
        self.lost_s = parse_timedelta(dask.config.get("distributed.deploy.lost-worker-timeout"))
        self.processes = cluster._dummy_job.worker_processes
        self._idle_since: dict[str, float] = {}
        self._lost_since: dict[str, float] = {}
        #: Each job a worker of which has joined, and its SLURM job id then: ``SpecCluster`` reopens a
        #: job of one worker it has given up on under the same name. Not the ``Job`` itself:
        #: ``SpecCluster`` asserts at close that every ``Job`` still alive is closed, and a
        #: jobqueue ``Job`` never says it is.
        self._joined: dict[str, str | None] = {}

    async def target(self) -> int:
        s = self.cluster.scheduler
        return len(s.queued) + len(s.unrunnable) + _processing(s)

    async def recommendations(self, target: int) -> dict:
        now, jobs = time.monotonic(), list(self.cluster.worker_spec)
        workers: dict[str, list[str]] = {}
        busy = set()
        for ws in self.cluster.scheduler.workers.values():
            job = ws.name if self.processes == 1 else ws.name.rsplit("-", 1)[0]
            workers.setdefault(job, []).append(ws.name)
            if ws.processing:
                busy.add(job)
        for kept in (self._idle_since, self._lost_since, self._joined):
            for job in kept.keys() - set(jobs):
                del kept[job]

        lost = []
        for job in jobs:
            handle = self.cluster.workers.get(job)
            submission = getattr(handle, "job_id", None)
            if job in workers:
                self._joined[job] = submission
            elif job in self._joined and (handle is None or submission == self._joined[job]):
                if now - self._lost_since.setdefault(job, now) >= self.lost_s:
                    lost.append(job)
                continue
            else:  # not joined yet, or reopened under its name
                self._joined.pop(job, None)
            self._lost_since.pop(job, None)
        if lost:
            return {"status": "down", "workers": lost}

        for job in jobs:
            if job in busy:
                self._idle_since.pop(job, None)
            else:
                self._idle_since.setdefault(job, now)
        wanted = math.ceil(target / self.processes)
        if wanted > len(jobs):
            return {"status": "up", "n": wanted * self.processes}
        idle = [j for j in reversed(jobs) if now - self._idle_since.get(j, math.inf) >= self.idle_s]
        closed = idle[: len(jobs) - wanted]
        if not closed:
            return {"status": "same"}
        return {
            "status": "down",
            "workers": closed + [w for j in closed for w in workers.get(j, ())],
        }


def _slurm_cluster(port: int, min_jobs: int, max_jobs: int, idle_s: float) -> Any:
    """A ``SLURMCluster`` whose scheduler listens on ``port``, scaled by :class:`_Jobs` within the two
    bounds; the rest of ``jobqueue.slurm.scheduler-options`` (a host or interface) is kept."""
    from dask_jobqueue import SLURMCluster

    options = {**dask.config.get("jobqueue.slurm.scheduler-options", {}), "port": port}
    cluster = SLURMCluster(scheduler_options=options)
    cluster.adapt(Adaptive=_Jobs, minimum_jobs=min_jobs, maximum_jobs=max_jobs, idle_s=idle_s)
    return cluster


def _processing(dask_scheduler) -> int:
    """The tasks assigned to the cluster's workers (run on the scheduler)."""
    return sum(len(ws.processing) for ws in dask_scheduler.workers.values())


def serve(cluster, address_file: Path) -> None:
    """Publish ``cluster``'s address in ``address_file`` until a signal of :data:`_SIGNALS`, then
    drain and close it (see the module docstring). The signals must be blocked in every thread, the
    cluster's included, so that only this thread takes them."""
    from distributed import Client

    from .dask_engine import _NoBuild

    with cluster, Client(cluster) as client:
        client.register_plugin(_NoBuild())
        written = address_file.with_name(f".{address_file.name}.tmp")
        written.write_text(f"{cluster.scheduler_address}\n", encoding="utf-8")
        written.replace(address_file)  # whole, as a reader finds it
        print(f"petab-mcp-pool: {cluster.scheduler_address} in {address_file}", file=sys.stderr)
        try:
            signal.sigwait(_SIGNALS)
        finally:
            address_file.unlink(missing_ok=True)
        deadline = time.monotonic() + DRAIN_S
        while client.run_on_scheduler(_processing) and time.monotonic() < deadline:
            time.sleep(1.0)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    jobs = 0 <= args.slurm_min_jobs <= args.slurm_max_jobs and args.slurm_max_jobs >= 1
    if not jobs or args.slurm_idle_minutes < 0:
        print(
            "error: the pool needs 0 <= --slurm-min-jobs <= --slurm-max-jobs, 1 <= "
            "--slurm-max-jobs, and 0 <= --slurm-idle-minutes.",
            file=sys.stderr,
        )
        return 2
    # Before the cluster starts its threads, which inherit the mask: `serve` waits for these.
    signal.pthread_sigmask(signal.SIG_BLOCK, _SIGNALS)
    cluster = _slurm_cluster(
        args.port, args.slurm_min_jobs, args.slurm_max_jobs, 60.0 * args.slurm_idle_minutes
    )
    serve(cluster, args.address_file)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
