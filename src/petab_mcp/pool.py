"""``petab-mcp-pool``: the service process that owns a fit pool, which servers join by address
(``petab-mcp --fit-pool ADDRESS``) to run their fits' starts on shared SLURM workers.

It runs a ``dask_jobqueue.SLURMCluster`` whose scheduler lives in this process, on ``--port`` of
this node, so a pool restarted on the same node comes back at the same address. The cluster grows to
``--slurm-max-jobs`` SLURM jobs while batches wait for a worker and shrinks to ``--slurm-min-jobs``
when idle; every other SLURM setting comes from dask's own ``jobqueue.slurm`` config (README,
"Fitting on SLURM").

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
import signal
import sys
import time
from pathlib import Path
from typing import Any

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
        help="SLURM jobs the pool keeps while idle, at most N (default 0).",
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


def _slurm_cluster(port: int, min_jobs: int, max_jobs: int) -> Any:
    """A ``SLURMCluster`` whose scheduler listens on ``port``, bounded in jobs by the two flags; the
    rest of ``jobqueue.slurm.scheduler-options`` (a host or interface) is kept."""
    import dask
    from dask_jobqueue import SLURMCluster

    options = {**dask.config.get("jobqueue.slurm.scheduler-options", {}), "port": port}
    cluster = SLURMCluster(scheduler_options=options)
    cluster.adapt(minimum_jobs=min_jobs, maximum_jobs=max_jobs)
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
    if not 0 <= args.slurm_min_jobs <= args.slurm_max_jobs or args.slurm_max_jobs < 1:
        print(
            "error: the pool needs 0 <= --slurm-min-jobs <= --slurm-max-jobs, and 1 <= "
            "--slurm-max-jobs.",
            file=sys.stderr,
        )
        return 2
    # Before the cluster starts its threads, which inherit the mask: `serve` waits for these.
    signal.pthread_sigmask(signal.SIG_BLOCK, _SIGNALS)
    serve(_slurm_cluster(args.port, args.slurm_min_jobs, args.slurm_max_jobs), args.address_file)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
