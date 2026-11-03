"""The fit pool's scheduler stress check (``petab_mcp.pool_stress``), small, on ``LocalCluster``s
with process workers."""

from __future__ import annotations

import threading
import time

from distributed import Client, LocalCluster

from petab_mcp import pool_stress


def _cluster() -> LocalCluster:
    return LocalCluster(n_workers=2, threads_per_worker=1, processes=True, dashboard_address=":0")


def _started(dask_scheduler) -> int:
    """How many batches have logged a run to the stress check's monitor."""
    monitor = dask_scheduler.plugins.get(pool_stress._NAME)
    return len(monitor.runs) if monitor else 0


async def _remove_a_busy_worker(dask_scheduler) -> None:
    ws = next(ws for ws in dask_scheduler.workers.values() if ws.processing)
    await dask_scheduler.remove_worker(ws.address, stimulus_id="test-remove")


def test_a_small_load_passes_and_reports_what_sizes_the_pool():
    with _cluster() as cluster:
        report = pool_stress.stress(cluster.scheduler_address, clients=4, batches=3, seconds=0.2)

    assert report["passed"], report
    assert report["batches_run"] == report["batches"] == 12, "every batch logged its one run"
    assert report["workers"] == 2
    assert report["heartbeat_gap_s"] > 0 and report["scheduler_rss_mib"] > 0
    assert 0 <= report["event_loop_lag_s"] < 3


def test_a_worker_removed_mid_batch_fails_the_check():
    """The checks hear the workers and the scheduler: a worker removed while its batch runs is
    recorded, and so is that batch's second run elsewhere. Without this, a check that heard nothing
    would pass every load."""
    report = {}
    with _cluster() as cluster, Client(cluster) as client:
        load = threading.Thread(
            target=lambda: report.update(
                pool_stress.stress(cluster.scheduler_address, clients=1, batches=2, seconds=3)
            )
        )
        load.start()
        deadline = time.monotonic() + 60
        while client.run_on_scheduler(_started) < 2 and time.monotonic() < deadline:
            time.sleep(0.1)
        client.run_on_scheduler(_remove_a_busy_worker)
        load.join()

    assert not report["passed"]
    assert report["workers_removed"] == ["test-remove"]
    assert len(report["batches_run_twice"]) == 1, report
    assert report["fits_failed"] == []
