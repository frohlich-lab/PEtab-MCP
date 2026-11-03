"""The fit pool (``petab_mcp.pool``) and the servers that join it by address, on a pool run in
another process: what crosses that boundary is the contract.

CI has no SLURM, so the pool's process runs ``pool.main`` with a ``LocalCluster`` in place of its
``SLURMCluster`` (:data:`_POOL`), and the SLURM cluster itself is only built, never submits a job.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import shutil
import signal
import socket
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path

import dask
import numpy as np
import pandas as pd
import pytest
from distributed import Client, get_task_stream

from petab_mcp import fitting, pool
from petab_mcp.dask_engine import DaskEngine
from petab_mcp.references import StubReferenceResolver
from petab_mcp.server import create_server

#: A pool whose workers are local processes, one per job.
_POOL = textwrap.dedent("""
    import sys
    from distributed import LocalCluster
    from petab_mcp import pool

    if __name__ == "__main__":
        pool._slurm_cluster = lambda port, min_jobs, max_jobs: LocalCluster(
            n_workers=max_jobs, threads_per_worker=1, scheduler_port=port, dashboard_address=":0"
        )
        raise SystemExit(pool.main(sys.argv[1:]))
""")


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@contextlib.contextmanager
def _pool(tmp_path: Path, workers: int = 2):
    """A pool of ``workers`` in another process; yields its published address, its address file
    and its process. Its process group is killed on exit."""
    (tmp_path / "pool.py").write_text(_POOL, encoding="utf-8")
    published = tmp_path / "address"
    flags = ["--slurm-max-jobs", str(workers), "--port", str(_free_port())]
    flags += ["--address-file", str(published)]
    proc = subprocess.Popen(
        [sys.executable, str(tmp_path / "pool.py"), *flags], start_new_session=True
    )
    try:
        deadline = time.monotonic() + 300
        while not published.exists():
            assert proc.poll() is None and time.monotonic() < deadline, "the pool never published"
            time.sleep(0.2)
        yield published.read_text(encoding="utf-8").strip(), published, proc
    finally:
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()


def _server(yaml: str, build_root, store: Path, cluster=None):
    return create_server(
        yaml,
        build_root=str(build_root),
        provenance_dir=str(store),
        resolver=StubReferenceResolver(),
        cluster=cluster,
    )


def _fit(mcp) -> dict:
    """Fit 4 starts of a widened k1; each start's ``(fval, n_fval)`` by its id."""

    def call(tool, args):
        return asyncio.run(mcp.call_tool(tool, args)).structured_content

    bounds = {"parameter_id": "k1", "lower_bound": 1e-3, "upper_bound": 1e3}
    call("make_changes", {"parameters": [bounds], "justification": "x", "n_starts": 4})
    sql = "SELECT id, fval, n_fval FROM optimization_starts"
    return {r["id"]: (r["fval"], r["n_fval"]) for r in call("query_data", {"sql": sql})["rows"]}


def test_two_servers_on_different_problems_fit_at_once_and_each_gets_its_own_starts(
    yaml_path, amici_build_root, tmp_path, monkeypatch
):
    """Two servers' fits on one pool interleave batch by batch, as one server's fits do, and each
    server gets back exactly the starts it would have run alone: on its own problem, the other's
    data doubled. The x0 of both are drawn from one seed, so each fit's starts are compared with
    the same server's fit run in this process."""
    monkeypatch.setattr(fitting, "FIT_N_WORKERS", 1)  # the fits run here run in this process
    monkeypatch.setattr(fitting, "FIT_STARTS_PER_TASK", 1)
    yamls = {}
    for label, scale in (("a", 1.0), ("b", 2.0)):
        shutil.copytree(Path(yaml_path).parent, tmp_path / label)
        data = pd.read_csv(tmp_path / label / "measurements.tsv", sep="\t")
        data["measurement"] *= scale
        data.to_csv(tmp_path / label / "measurements.tsv", sep="\t", index=False)
        yamls[label] = str(tmp_path / label / "problem.yaml")
    here = {}
    for label, yaml in yamls.items():
        np.random.seed(0)
        here[label] = _fit(_server(yaml, amici_build_root, tmp_path / label / "here"))

    # Each fit draws its x0 from the seed, then waits for the other's before submitting a batch.
    drawn, both, run, arrived = (
        threading.Event(),
        threading.Barrier(2, timeout=300),
        DaskEngine._run,
        [],
    )

    def _run(self, tasks):
        arrived.append(self.client)
        drawn.set()
        both.wait()
        return run(self, tasks)

    submitted, submit = {}, Client.submit

    def spy(self, *args, **kwargs):
        future = submit(self, *args, **kwargs)
        submitted[future.key] = self
        return future

    def fit(label, mcp):
        there[label] = _fit(mcp)

    monkeypatch.setattr(DaskEngine, "_run", _run)
    monkeypatch.setattr(Client, "submit", spy)
    there = {}
    with _pool(tmp_path) as (address, _, _), Client(address) as client:
        servers = {
            label: _server(yaml, amici_build_root, tmp_path / label / "pool", address)
            for label, yaml in yamls.items()
        }
        with get_task_stream(client) as stream:
            fits = [threading.Thread(target=fit, args=item) for item in servers.items()]
            for thread in fits:
                np.random.seed(0)
                thread.start()
                drawn.wait()
                drawn.clear()
            for thread in fits:
                thread.join()

    assert there == here and here["a"] != here["b"]
    began = sorted(stream.data, key=lambda task: task["startstops"][0]["start"])
    order = ["ab"[arrived.index(submitted[task["key"]])] for task in began]
    assert sorted(order) == ["a"] * 4 + ["b"] * 4
    assert sum(x != y for x, y in zip(order, order[1:])) >= 2, order


def test_sigterm_unpublishes_the_pool_then_waits_for_the_batch_running(tmp_path):
    """On SIGTERM the address file goes first, so no new server joins, and the pool stays up until
    the batch it is running has finished, then exits."""
    with _pool(tmp_path, workers=1) as (address, published, proc), Client(address) as client:
        batch = client.submit(time.sleep, 10, pure=False)
        while not any(client.processing().values()):
            time.sleep(0.1)
        proc.send_signal(signal.SIGTERM)
        while published.exists():
            time.sleep(0.1)
        assert proc.poll() is None and not batch.done()
        batch.result(timeout=120)
        assert proc.wait(timeout=120) == 0


def test_the_pool_listens_on_its_port_and_takes_every_slurm_setting_from_dask_config(tmp_path):
    """``_slurm_cluster`` gives ``SLURMCluster`` the port and nothing else, so what the README's
    ``jobqueue.yaml`` sets must reach the job script from dask's config alone: partition, cores,
    one single-threaded worker process per core, memory, walltime, the workers' lifetime, both
    directories, and the scheduler's host. With a minimum of 0 jobs nothing is submitted."""
    logs = tmp_path / "logs"
    config = {
        "jobqueue.slurm.queue": "ncpu",
        "jobqueue.slurm.cores": 4,
        "jobqueue.slurm.processes": 4,
        "jobqueue.slurm.memory": "8GiB",
        "jobqueue.slurm.walltime": "01:00:00",
        "jobqueue.slurm.worker-extra-args": ["--lifetime", "55m"],
        "jobqueue.slurm.local-directory": "$TMPDIR",
        "jobqueue.slurm.log-directory": str(logs),
        "jobqueue.slurm.scheduler-options": {"host": "127.0.0.1"},  # no worker will connect
    }
    port = _free_port()
    # Every job reads the config as it is described, the script's one included.
    with dask.config.set(config), pool._slurm_cluster(port, min_jobs=0, max_jobs=2) as cluster:
        script = cluster.job_script()
        assert cluster.scheduler_address == f"tcp://127.0.0.1:{port}"
    for directive in ("-p ncpu", "--cpus-per-task=4", "--mem=8G", "-t 01:00:00", f"-o {logs}/"):
        assert f"#SBATCH {directive}" in script
    for arg in ("--nworkers 4", "--nthreads 1", "--lifetime 55m", "--local-directory $TMPDIR"):
        assert arg in script


@pytest.mark.parametrize(
    "jobs", [["0"], ["2", "--slurm-min-jobs", "3"], ["2", "--slurm-min-jobs", "-1"]]
)
def test_job_bounds_the_pool_cannot_keep_are_refused_before_it_starts(
    monkeypatch, capsys, tmp_path, jobs
):
    """Bounds outside 0 <= min <= max, max >= 1 would otherwise pass without a word: a pool of no
    jobs serves no fit, and dask clamps the rest."""
    monkeypatch.setattr(pool, "_slurm_cluster", lambda *a: pytest.fail("the pool started"))
    argv = ["--slurm-max-jobs", *jobs, "--address-file", str(tmp_path / "address")]
    assert pool.main(argv) == 2
    assert "0 <= --slurm-min-jobs <= --slurm-max-jobs" in capsys.readouterr().err
