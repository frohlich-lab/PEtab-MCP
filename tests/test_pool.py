"""The fit pool (``petab_mcp.pool``) and the servers that join it by address, on a pool run in
another process: what crosses that boundary is the contract.

CI has no SLURM, so the pool's process runs ``pool.main`` with a ``LocalCluster`` in place of its
``SLURMCluster`` (:data:`_POOL`), and the SLURM cluster itself submits its jobs to an ``sbatch``
that runs them on this machine (:data:`_FAKE_SLURM`).
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
import dask_jobqueue  # noqa: F401 - its config defaults, here, outside every test's dask.config.set
import numpy as np
import pandas as pd
import pytest
from distributed import Client, WorkerPlugin, get_task_stream
from fastmcp.exceptions import ToolError

from petab_mcp import dask_engine, fitting, pool
from petab_mcp.dask_engine import DaskEngine
from petab_mcp.errors import FIT_POOL, InfrastructureError
from petab_mcp.references import StubReferenceResolver
from petab_mcp.server import create_server

#: A pool whose workers are local processes, one per job, whose scheduler reports the versions of
#: this environment but for an ``amici`` of ``argv[1]`` (``-``: this environment's own).
_POOL = textwrap.dedent("""
    import sys
    from distributed import LocalCluster
    from petab_mcp import dask_engine, pool

    if __name__ == "__main__":
        if sys.argv[1] != "-":
            ours = dask_engine.versions
            dask_engine.versions = lambda: {**ours(), "amici": sys.argv[1]}
        pool._slurm_cluster = lambda port, jobs: LocalCluster(
            n_workers=jobs, threads_per_worker=1, scheduler_port=port, dashboard_address=":0"
        )
        raise SystemExit(pool.main(sys.argv[2:]))
""")


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@contextlib.contextmanager
def _pool(tmp_path: Path, workers: int = 2, amici: str = "-", port: int | None = None):
    """A pool of ``workers`` in another process, on ``port`` or a free one; yields its published
    address, its address file and its process. Its process group is killed on exit."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "pool.py").write_text(_POOL, encoding="utf-8")
    published = tmp_path / "address"
    flags = ["--slurm-max-jobs", str(workers), "--port", str(port or _free_port())]
    flags += ["--address-file", str(published)]
    proc = subprocess.Popen(
        [sys.executable, str(tmp_path / "pool.py"), amici, *flags], start_new_session=True
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


def _fit(mcp, upper: float = 1e3) -> dict:
    """Fit 4 starts of k1 widened up to ``upper``; each start's ``(fval, n_fval)`` by its id."""

    def call(tool, args):
        return asyncio.run(mcp.call_tool(tool, args)).structured_content

    bounds = {"parameter_id": "k1", "lower_bound": 1e-3, "upper_bound": upper}
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


def test_a_pool_from_another_environment_is_refused_at_startup_naming_both_versions(
    yaml_path, tmp_path
):
    """A start is pickled, so a pool whose scheduler imports another AMICI would fail every fit
    with errors that are hard to read: the server refuses it before opening, naming both."""
    with _pool(tmp_path, workers=1, amici="0.0.1") as (address, _, _):
        with pytest.raises(InfrastructureError) as refused:
            create_server(
                yaml_path,
                provenance_dir=str(tmp_path / "prov"),
                resolver=StubReferenceResolver(),
                auto_fit=False,
                cluster=address,
            )
    assert refused.value.cause == FIT_POOL
    assert f"amici 0.0.1 there, {dask_engine.versions()['amici']} here" in str(refused.value)
    assert "pypesto" not in str(refused.value)
    assert not (tmp_path / "prov").exists()


def test_a_pool_lost_mid_fit_ends_the_call_as_fit_pool(
    yaml_path, amici_build_root, tmp_path, monkeypatch
):
    """The pool's scheduler killed while a fit waits on it is the pool's failure, not the run's:
    the call ends as the ``fit_pool`` infrastructure error a client ends the run on."""
    with _pool(tmp_path, workers=1) as (address, _, proc):
        mcp = _server(yaml_path, amici_build_root, tmp_path / "prov", address)
        wait = DaskEngine._wait

        def kill_then_wait(self, pending, timeout):
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
            return wait(self, pending, timeout)

        monkeypatch.setattr(DaskEngine, "_wait", kill_then_wait)
        with pytest.raises(
            ToolError, match="^infrastructure_error: fit_pool: the fit's client lost"
        ):
            _fit(mcp)


def test_a_pool_restarted_at_its_address_serves_the_next_fit_unless_from_another_environment(
    yaml_path, amici_build_root, tmp_path, monkeypatch
):
    """A pool resubmitted to SLURM came back 4 min later (NEMO, 2026-11-04), long after its client
    had stopped reconnecting and closed, and every later fit failed as ``fit_pool`` until the server
    restarted (#216). A fit while the pool is away still fails so; the next one, once it is back,
    opens a new client there and runs. A pool back from another environment is refused naming
    both, also when the client reconnected to it by itself."""
    opened, connect = [], dask_engine._connect

    def spy(cluster):
        opened.append(connect(cluster))
        return opened[-1]

    monkeypatch.setattr(dask_engine, "_connect", spy)
    port, window = _free_port(), "distributed.comm.timeouts.connect"
    with _pool(tmp_path / "a", workers=1, port=port) as (address, _, _):
        with dask.config.set({window: "2s"}):
            mcp = _server(yaml_path, amici_build_root, tmp_path / "prov", address)
    deadline = time.monotonic() + 60
    while opened[0].status != "closed":
        assert time.monotonic() < deadline, "the client never stopped reconnecting"
        time.sleep(0.1)
    with dask.config.set({window: "2s"}):
        with pytest.raises(ToolError, match="^infrastructure_error: fit_pool: "):
            _fit(mcp)
    with _pool(tmp_path / "b", workers=1, port=port):
        with dask.config.set({window: "300s"}):  # the new client outlasts pool c's start
            assert len(_fit(mcp)) == 4
    assert len(opened) == 2
    with _pool(tmp_path / "c", workers=1, amici="0.0.1", port=port):
        with pytest.raises(ToolError, match="^infrastructure_error: fit_pool: ") as refused:
            _fit(mcp, upper=1e4)
    assert len(opened) == 2, "the client reconnected to pool c by itself"
    here = dask_engine.versions()["amici"]
    assert f"the pool at {address} runs another environment (amici 0.0.1 there, {here} here)" in (
        str(refused.value)
    )
    for client in opened:
        client.close()


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
    directories, and the scheduler's host. With no job asked for, nothing is submitted."""
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
    with dask.config.set(config), pool._slurm_cluster(port, jobs=0) as cluster:
        script = cluster.job_script()
        assert cluster.scheduler_address == f"tcp://127.0.0.1:{port}"
    for directive in ("-p ncpu", "--cpus-per-task=4", "--mem=8G", "-t 01:00:00", f"-o {logs}/"):
        assert f"#SBATCH {directive}" in script
    for arg in ("--nworkers 4", "--nthreads 1", "--lifetime 55m", "--local-directory $TMPDIR"):
        assert arg in script


#: ``sbatch`` and ``scancel`` that run a job script on this machine, in a process group of its own
#: whose id is the job's, and log each submission and cancellation in ``../ledger``, beside the
#: jobs' output in ``../jobs.out``.
_FAKE_SLURM = {
    "sbatch": """
        out = open(LEDGER.with_name("jobs.out"), "a")
        script = Path(sys.argv[1]).read_text()
        job = subprocess.Popen(["bash", "-c", script], start_new_session=True, stdout=out, stderr=out)
        log("submitted", job.pid)
        print(f"Submitted batch job {job.pid}")
    """,
    "scancel": """
        log("cancelled", sys.argv[1])
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(int(sys.argv[1]), signal.SIGKILL)
    """,
}
_FAKE_SLURM_HEAD = """
    import contextlib, os, signal, subprocess, sys
    from pathlib import Path

    LEDGER = Path(sys.argv[0]).resolve().parent.parent / "ledger"

    def log(event, job):
        with LEDGER.open("a") as ledger:
            ledger.write(f"{event} {job}\\n")
"""


class _UnpickledSlowly(WorkerPlugin):
    """Takes 8 s to unpickle on a worker, as :class:`~petab_mcp.dask_engine._NoBuild` does where
    ``import petab_mcp`` is cold (48-98 s on NEMO)."""

    def __reduce__(self):
        def unpickle():
            time.sleep(8)
            return WorkerPlugin()

        return unpickle, ()


def test_a_fit_of_fewer_starts_than_a_jobs_workers_runs_on_the_job_the_pool_submitted_once(
    tmp_path, monkeypatch
):
    """A worker joins the scheduler, then unpickles its plugins, and takes no batch until it has.
    While a job's workers did, ``adapt`` saw them all idle beside the queued batches, cancelled the
    job and submitted another, and so on: on NEMO five 64-worker jobs came up and were cancelled
    within 15 s each, and the fit ran 0/100 starts in 300 s (#250). Here ``sbatch`` runs the job on
    this machine, and its 2 workers' plugin takes 8 s: the one job submitted runs the batch."""
    for name, body in _FAKE_SLURM.items():
        command = tmp_path / "bin" / name
        command.parent.mkdir(exist_ok=True)
        head, body = textwrap.dedent(_FAKE_SLURM_HEAD), textwrap.dedent(body)
        command.write_text(f"#!{sys.executable}\n{head}\n{body}", encoding="utf-8")
        command.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path / 'bin'}{os.pathsep}{os.environ['PATH']}")
    config = {
        "jobqueue.slurm.cores": 2,
        "jobqueue.slurm.processes": 2,
        "jobqueue.slurm.memory": "2GiB",
        "jobqueue.slurm.scheduler-options": {"host": "127.0.0.1"},
    }
    ledger = tmp_path / "ledger"
    try:
        with (
            dask.config.set(config),
            pool._slurm_cluster(_free_port(), jobs=1) as cluster,
            Client(cluster) as client,
        ):
            client.register_plugin(_UnpickledSlowly(), name="unpickled-slowly")
            batch = client.submit(time.sleep, 1, pure=False)
            with contextlib.suppress(TimeoutError):
                batch.result(timeout=120)
            events = [line.split()[0] for line in ledger.read_text().splitlines()]
            assert events == ["submitted"] and batch.done()
            assert len(client.scheduler_info()["workers"]) == 2
    finally:
        for line in ledger.read_text().splitlines() if ledger.exists() else []:
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(int(line.split()[1]), signal.SIGKILL)


@pytest.mark.parametrize("jobs", ["0", "-1"])
def test_a_pool_of_no_jobs_is_refused_before_it_starts(monkeypatch, capsys, tmp_path, jobs):
    """A pool of no jobs would otherwise start without a word and serve no fit."""
    monkeypatch.setattr(pool, "_slurm_cluster", lambda *a: pytest.fail("the pool started"))
    argv = ["--slurm-max-jobs", jobs, "--address-file", str(tmp_path / "address")]
    assert pool.main(argv) == 2
    assert "--slurm-max-jobs >= 1" in capsys.readouterr().err
