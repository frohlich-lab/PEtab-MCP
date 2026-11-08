"""Load a fit pool's dask scheduler as a sweep's servers would, and measure whether it holds up.

One scheduler serves every run of a sweep: each worker's heartbeats, each batch's state changes, one
client per server, and each running fit's ``run_on_scheduler`` poll. If its event loop falls
behind, it misses heartbeats, drops a worker past ``distributed.scheduler.worker-ttl``, and reruns
that worker's batches. :func:`stress` checks for that:

* **The load is a server's own call pattern.** Each of ``clients`` threads is one server: it opens
  a :class:`~petab_mcp.dask_engine.DaskEngine` on the scheduler, and once every one has connected,
  runs one fit through it. The fit's batches go in one burst at ``priority=-j``, then the engine
  watches them and polls the scheduler for overruns, as in a real fit. A batch only sleeps
  ``seconds``, so the scheduler is under test, not the fits. A sweep's batches run about 270 s
  (``PETAB_MCP_FIT_TASK_SECONDS``), so seconds-long ones change state far more often than a sweep
  does.
* **The scheduler measures itself.** A :class:`_Monitor` plugin samples, every ``interval``, how
  late its event loop is, the longest any worker has gone without a heartbeat (what ``worker-ttl``
  is compared with), and the scheduler's RSS. It also records every worker removal, the scheduler's
  own ``worker-ttl-timed-out`` events, and each batch's runs, which a batch logs from its worker as
  it starts.
* **It passes** when no worker is removed, every batch runs once, every fit finishes, and the loop
  is never late by more than ``max_lag``. The default is distributed's own "Event loop was
  unresponsive" threshold (``distributed.admin.tick.limit``).

It is a module, not a script in a checkout, because it runs from the pool's environment: the workers
import :mod:`petab_mcp.dask_engine` from it anyway. The tests run it small, on a ``LocalCluster``.

At full size, on NEMO, it is run by hand before any sweep shares a pool. Start the pool with every
worker job up and none retiring (as many minimum jobs as maximum: 12 jobs of 100 workers here).
Then, from the pool's environment, on a node of its own, since every client runs in this one
process:

.. code-block:: bash

    python -m petab_mcp.pool_stress tcp://NODE:8786 --workers 1200 --clients 100 --batches 60

``--workers`` waits until that many workers have joined. It prints a JSON report and exits 1 when
a check fails. ``event_loop_lag_s`` and
``scheduler_rss_mib`` size the pool's service job. ``heartbeat_gap_s`` sizes ``worker-ttl``,
which should sit well above it. Raise ``--batches`` to hold the load longer.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import tempfile
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from distributed import Client, SchedulerPlugin, get_worker
from distributed.metrics import time as wall_time
from pypesto.engine import Task

from .dask_engine import DaskEngine

#: The monitor's plugin name, and the topic each batch logs its runs under.
_NAME = "petab-mcp-pool-stress"

#: Each batch's cap. It is finite, so each fit polls the scheduler as a capped fit does (an
#: uncapped one never polls), and far above any batch here, so none is stopped.
_CAP_S = 3600.0


class _Batch(Task):
    """One batch: it logs its run to the scheduler, then sleeps ``seconds``."""

    def __init__(self, batch_id: str, seconds: float) -> None:
        super().__init__()
        self.id, self.seconds = batch_id, seconds

    def execute(self) -> str:
        get_worker().log_event(_NAME, self.id)
        time.sleep(self.seconds)
        return self.id


class _Monitor(SchedulerPlugin):
    """Samples the scheduler every ``interval`` seconds, and records what it hears (see the module
    docstring)."""

    name = _NAME

    def __init__(self, interval: float) -> None:
        self.interval = interval

    async def start(self, scheduler) -> None:
        self.scheduler = scheduler
        self.lag = self.gap = 0.0
        self.rss = 0
        self.removed: list[str] = []
        self.timed_out: list[str] = []
        self.runs: Counter[str] = Counter()
        self.sampling = asyncio.create_task(self._sample())

    async def _sample(self) -> None:
        while True:
            before = time.monotonic()
            await asyncio.sleep(self.interval)
            self.lag = max(self.lag, time.monotonic() - before - self.interval)
            now = wall_time()
            seen = [ws.last_seen for ws in self.scheduler.workers.values()]
            self.gap = max([self.gap, *(now - last for last in seen)])
            self.rss = max(self.rss, self.scheduler.monitor.get_process_memory())

    async def close(self) -> None:
        self.sampling.cancel()

    def remove_worker(self, scheduler, worker: str, *, stimulus_id: str, **kwargs) -> None:
        self.removed.append(stimulus_id)

    def log_event(self, topic: str, msg: Any) -> None:
        if topic == _NAME:
            self.runs[msg] += 1
        elif topic == "scheduler" and msg.get("action") == "worker-ttl-timed-out":
            self.timed_out += msg["workers"]


def _finish(dask_scheduler) -> dict[str, Any]:
    """Removes the monitor from the scheduler and returns what it recorded (run on the
    scheduler)."""
    monitor = dask_scheduler.plugins.pop(_NAME)
    monitor.sampling.cancel()
    return {
        "workers": len(dask_scheduler.workers),
        "event_loop_lag_s": round(monitor.lag, 3),
        "heartbeat_gap_s": round(monitor.gap, 3),
        "worker_ttl_s": dask_scheduler.worker_ttl,
        "scheduler_rss_mib": round(monitor.rss / 2**20),
        "workers_removed": monitor.removed,
        "worker_ttl_timeouts": monitor.timed_out,
        "runs": dict(monitor.runs),
    }


def _import_engine() -> None:
    """Imports the engine, as a server's first poll does on a scheduler that has not."""
    import petab_mcp.dask_engine  # noqa: F401


def _server(
    address: str, i: int, batches: int, seconds: float, connected: threading.Barrier
) -> str | None:
    """Server ``i``: connects, waits until every server has, then runs one fit; returns its error,
    if any."""
    engine = None
    try:
        engine = DaskEngine(address, tempfile.gettempdir())  # a batch leaves no problem
        connected.wait()
        fit = [_Batch(f"{i}:{j}", seconds) for j in range(batches)]
        engine.batched(1, _CAP_S).execute(fit)
    except Exception as exc:  # noqa: BLE001 - a failed fit fails the check, reported with the rest
        connected.abort()
        return repr(exc)
    finally:
        if engine is not None:
            engine.client.close()
    return None


def stress(
    address: str,
    clients: int = 100,
    batches: int = 60,
    seconds: float = 5.0,
    workers: int = 0,
    max_lag: float = 3.0,
    interval: float = 0.1,
) -> dict[str, Any]:
    """Run ``clients`` servers' fits of ``batches`` batches of ``seconds`` each on the scheduler at
    ``address``, once it has ``workers`` workers; returns the report, ``passed`` included."""
    connected = threading.Barrier(clients)
    with Client(address) as client:
        if workers:
            client.wait_for_workers(workers)
        # The pool's own processes import the engine before any server connects; a bare scheduler
        # and its workers would import it on their event loops at the first server's poll and
        # plugin, which is not load.
        client.run_on_scheduler(_import_engine)
        DaskEngine(address, tempfile.gettempdir()).client.close()
        client.register_plugin(_Monitor(interval))
        began = time.monotonic()
        try:
            with ThreadPoolExecutor(clients) as pool:
                errors = pool.map(
                    lambda i: _server(address, i, batches, seconds, connected), range(clients)
                )
                failed = [error for error in errors if error]
        finally:
            report = client.run_on_scheduler(_finish)
    runs = report.pop("runs")
    report = {
        "clients": clients,
        "batches": clients * batches,
        "wall_s": round(time.monotonic() - began, 1),
        **report,
        "batches_run": len(runs),
        "batches_run_twice": sorted(batch for batch, n in runs.items() if n > 1),
        "fits_failed": failed,
    }
    report["passed"] = not (
        report["workers_removed"]
        or report["worker_ttl_timeouts"]
        or report["batches_run_twice"]
        or failed
        or report["batches_run"] < report["batches"]
        or report["event_loop_lag_s"] > max_lag
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m petab_mcp.pool_stress",
        description="Load a fit pool's dask scheduler as a sweep's servers would; print a JSON "
        "report, and exit 1 when a check fails.",
    )
    parser.add_argument("address", help="the scheduler's address, tcp://HOST:PORT")
    parser.add_argument("--clients", type=int, default=100, help="servers, one client each")
    parser.add_argument("--batches", type=int, default=60, help="batches in each server's fit")
    parser.add_argument("--seconds", type=float, default=5.0, help="how long a batch sleeps")
    parser.add_argument("--workers", type=int, default=0, help="workers to wait for first")
    parser.add_argument(
        "--max-lag", type=float, default=3.0, help="the most the event loop may fall behind, in s"
    )
    args = parser.parse_args(argv)
    report = stress(
        args.address, args.clients, args.batches, args.seconds, args.workers, args.max_lag
    )
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
