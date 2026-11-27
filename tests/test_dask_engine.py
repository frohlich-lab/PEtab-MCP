"""Fits on a shared dask cluster (``petab_mcp.dask_engine``), on real ``LocalCluster``s with process
workers: the engine's contract is what crosses that boundary."""

from __future__ import annotations

import asyncio
import contextlib
import itertools
import logging
import os
import re
import socket
import statistics
import tempfile
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import dask
import numpy as np
import pytest
from distributed import LocalCluster, WorkerPlugin, default_client, get_task_stream, wait
from fastmcp import Client
from pypesto.engine import Task

from petab_mcp import dask_engine
from petab_mcp import fitting
from petab_mcp import logging_setup
from petab_mcp import multistart
from petab_mcp import server
from petab_mcp.dask_engine import DaskEngine
from petab_mcp.errors import FIT_POOL, FIT_WEDGED, FitBackendError, InfrastructureError
from petab_mcp.references import StubReferenceResolver
from petab_mcp.server import create_server


@contextlib.contextmanager
def _engine(n_workers: int = 2, config: dict | None = None):
    """A :class:`DaskEngine` on a fresh process cluster, with a build root of its own; ``config``
    is dask config for its scheduler. Closed on exit."""
    with dask.config.set(config or {}):
        cluster = LocalCluster(
            n_workers=n_workers, threads_per_worker=1, processes=True, dashboard_address=":0"
        )
    with cluster, tempfile.TemporaryDirectory() as build_root:
        engine = DaskEngine(cluster, build_root)
        with engine.client:
            yield engine


class _Sleep(Task):
    """A start that takes ``seconds`` and returns when it finished."""

    def __init__(self, label: str, i: int, seconds: float) -> None:
        super().__init__()
        self.label, self.i, self.seconds = label, i, seconds
        self.id = f"{label}{i}"

    def execute(self):
        time.sleep(self.seconds)
        return self.label, self.i, time.time()


class _Log(Task):
    """A start that logs through pyPESTO's and Fides' loggers, as an optimizer start does."""

    def __init__(self, text: str) -> None:
        super().__init__()
        self.text = text

    def execute(self):
        logging.getLogger("pypesto.optimize").info("pypesto says %s", self.text)
        logging.getLogger("fides").warning("fides says %s", self.text)
        return self.text


class _Count(Task):
    """A start that counts itself into its problem, which every start of its fit shares here."""

    def __init__(self, problem: list) -> None:
        super().__init__()
        self.problem = problem

    def execute(self):
        self.problem.append(None)
        return len(self.problem)


class _Die(Task):
    """A start that takes its worker process down."""

    def execute(self):
        os._exit(1)


class _Fail(Task):
    """A start that fails on its worker."""

    def execute(self):
        raise RuntimeError("start failed")


class _Touch(Task):
    """A start that leaves ``path`` behind once it has run."""

    def __init__(self, path: str) -> None:
        super().__init__()
        self.path = path

    def execute(self):
        time.sleep(0.5)
        Path(self.path).touch()


class _Build(Task):
    """A start that reaches one model-build entry point on its worker: pyPESTO's v1 or v2 creator
    asked for a model whose module there will not import, or AMICI's PEtab importer."""

    def __init__(self, folder: str, how: str) -> None:
        super().__init__()
        self.folder, self.how = folder, how

    def execute(self):
        from amici.importers.petab import PetabImporter
        from pypesto.petab.objective_creator import (
            AmiciObjectiveCreator,
            AmiciPetabV2ObjectiveCreator,
        )

        if self.how == "amici":
            return PetabImporter._do_import_sbml(SimpleNamespace(output_dir=Path(self.folder)))
        creator = {"v1": AmiciObjectiveCreator, "v2": AmiciPetabV2ObjectiveCreator}[self.how]
        problem = SimpleNamespace(model=SimpleNamespace(type_id="sbml"))
        return creator(problem, output_folder=self.folder, model_name="m").create_model()


def _sharing(tasks: list[Task]) -> list[Task]:
    """``tasks``, holding one problem as the starts of a fit do."""
    problem = {"shared": True}
    for task in tasks:
        task.problem = problem
    return tasks


@pytest.fixture
def left(monkeypatch) -> list[Path]:
    """The file each fit leaves its problem in, in the order the fits left them."""
    paths, leave = [], dask_engine._leave

    def spy(problem, folder):
        paths.append(leave(problem, folder))
        return paths[-1]

    monkeypatch.setattr(dask_engine, "_leave", spy)
    return paths


def test_a_server_given_a_cluster_fits_there_and_each_start_ends_as_it_would_here(
    yaml_path, amici_build_root, tmp_path, monkeypatch
):
    """A server given a cluster runs every start there, and each ends exactly where it ends in this
    process."""
    monkeypatch.setattr(fitting, "FIT_N_WORKERS", 1)  # the local fit runs in this process

    def fit(store, cluster=None):
        mcp = create_server(
            yaml_path,
            build_root=str(amici_build_root),
            provenance_dir=str(tmp_path / store),
            resolver=StubReferenceResolver(),
            cluster=cluster,
        )

        def call(tool, args):
            return asyncio.run(mcp.call_tool(tool, args)).structured_content

        bounds = {"parameter_id": "k1", "lower_bound": 1e-3, "upper_bound": 1e3}
        np.random.seed(0)  # the same fresh starts on both sides
        call("make_changes", {"parameters": [bounds], "justification": "x", "n_starts": 4})
        sql = "SELECT id, fval, n_fval FROM optimization_starts"
        return {r["id"]: (r["fval"], r["n_fval"]) for r in call("query_data", {"sql": sql})["rows"]}

    local = fit("local")
    with _engine() as engine, get_task_stream(engine.client) as stream:
        remote = fit("dask", engine.client.cluster)

    assert len(stream.data) == 4, "every start ran on the cluster"
    # Exact: the toy's starts share one optimum, so only fval's last digits and n_fval tell them apart.
    assert len(local) == 4 and remote == local


def _hold(path: str) -> None:
    """Hold a worker until ``path`` exists, at most a minute."""
    deadline = time.monotonic() + 60
    while not os.path.exists(path) and time.monotonic() < deadline:
        time.sleep(0.05)


class _SlowStarts(WorkerPlugin):
    """Makes each optimizer start on a worker take a second longer, so the scheduler hears that its
    batch runs before it ends."""

    def setup(self, worker) -> None:
        from pypesto.optimize.task import OptimizerTask

        execute = OptimizerTask.execute

        def slow(task):
            time.sleep(1)
            return execute(task)

        OptimizerTask.execute = slow


def test_a_call_through_a_fit_gets_heartbeats_naming_its_phase(
    yaml_path, amici_build_root, tmp_path, monkeypatch
):
    """Once fits queue for a shared pool no budget bounds a call, so gemot times it out on silence
    (#217). On its protocol, 2025-11-25, a make_changes gets a heartbeat every HEARTBEAT_S while its
    model builds, while it waits for the pool's only worker, and while its starts run; the wait is
    its queue_seconds."""
    import pypesto.petab

    monkeypatch.setattr(server, "HEARTBEAT_S", 0.2)
    monkeypatch.setattr(multistart, "POLL_S", 0.1)
    create = pypesto.petab.PetabImporter.create_problem

    def slow(self, *args, **kwargs):  # a build lasts a few heartbeats, its module cached or not
        time.sleep(1)
        return create(self, *args, **kwargs)

    monkeypatch.setattr(pypesto.petab.PetabImporter, "create_problem", slow)
    release, beats = tmp_path / "release", []

    async def heard(progress, total, message):
        beats.append((progress, message))
        if message and message.startswith("queued"):
            release.touch()

    async def change(mcp):
        async with Client(mcp, mode="legacy", progress_handler=heard) as client:
            assert client.protocol_version == "2025-11-25"
            bounds = {"parameter_id": "k1", "lower_bound": 1e-3, "upper_bound": 1e3}
            args = {"parameters": [bounds], "justification": "x", "n_starts": 2}
            out = await client.call_tool("make_changes", args)
            timing = await client.call_tool("get_run_timing", {})
        return out.structured_content, timing.structured_content

    with _engine(n_workers=1) as engine:
        mcp = create_server(
            yaml_path,
            build_root=str(amici_build_root),
            provenance_dir=str(tmp_path / "prov"),
            resolver=StubReferenceResolver(),
            cluster=engine.client.cluster,
        )
        engine.client.register_plugin(_SlowStarts(), name="slow-starts")
        held = engine.client.submit(_hold, str(release), pure=False)  # noqa: F841 - kept queued
        out, timing = asyncio.run(change(mcp))

    assert out["applied"] is True, out
    phases = iter(k for k, _ in itertools.groupby(m.split(" for ")[0] for _, m in beats if m))
    assert all(phase in phases for phase in ("compiling", "queued", "running")), beats
    assert any(re.fullmatch(r"queued for \d+s, 0 of 2 starts done", m or "") for _, m in beats)
    assert any(re.fullmatch(r"running for \d+s, [01] of 2 starts done", m or "") for _, m in beats)
    # Not a bound on the longest gap: the end of a build holds the GIL, the event loop's too, for
    # seconds.
    gaps = [b - a for (a, _), (b, _) in itertools.pairwise(beats)]
    assert 0.15 < min(gaps) and statistics.median(gaps) < 0.3, gaps
    assert 0 < timing["queue_seconds"] < timing["fit_seconds"], timing


class _SolverSettings(Task):
    """A start that reports the tolerances its problem's AMICI solver carries on its worker, and
    the threads its PEtab simulator runs on."""

    def __init__(self, problem) -> None:
        super().__init__()
        self.problem = problem

    def execute(self):
        solver = self.problem.objective.amici_solver
        return (
            solver.get_absolute_tolerance(),
            solver.get_relative_tolerance(),
            solver.get_relative_tolerance_steady_state(),
            solver.get_absolute_tolerance_steady_state(),
            self.problem.objective._petab_simulator.num_threads,
        )


def test_a_fits_amici_settings_reach_its_starts_on_a_worker(fit_session, monkeypatch):
    """The tolerance and thread knobs are read from the SERVER's environment, which a dask or SLURM
    worker does not share. They reach a start because they are set on the solver and the simulator
    before its problem is pickled, and AMICI pickles both with their settings; a start that
    integrated at AMICI's defaults could score such a problem ~0.2 nats off."""
    monkeypatch.setattr(fitting, "FIT_AMICI_RTOL", 1e-7)
    monkeypatch.setattr(fitting, "FIT_AMICI_RTOL_STEADY_STATE", 1e-9)
    monkeypatch.setattr(fitting, "FIT_AMICI_ATOL_STEADY_STATE", 1e-11)
    monkeypatch.setattr(fitting, "FIT_AMICI_N_THREADS", 3)
    problem, _ = fitting.estimation_problem(fit_session, fit_session.engine)

    with _engine(n_workers=1) as engine:
        (on_worker,) = engine.execute([_SolverSettings(problem)])

    assert on_worker == (fitting.FIT_AMICI_ATOL, 1e-7, 1e-9, 1e-11, 3)


def test_a_fits_problem_leaves_the_server_once_and_its_starts_carry_only_themselves(
    fit_session, monkeypatch, left
):
    """Each froehlich start carried its 0.45 GiB problem as its task's argument, which the scheduler
    keeps until the fit ends: a sweep's 160 starts in flight needed 72 GiB of a pool job of 8 GB, and
    the server pickled the problem once per start, at 60 s each (2026-11-07). A fit now leaves
    its problem in one file, gone once it returns, and each start crosses the scheduler in a few KB."""
    sizes, problem_bytes = [], []
    with _engine(n_workers=1) as engine:
        submit = engine.client.submit

        def spy(fn, payloads, problem, **kwargs):
            sizes.extend(len(payload) for payload in payloads)
            problem_bytes.append(problem.stat().st_size)
            return submit(fn, payloads, problem, **kwargs)

        monkeypatch.setattr(engine.client, "submit", spy)
        fit_session.engine.fit_engine = engine
        fitting.fit_and_simulate(fit_session, fit_session.engine, do_fit=True, n_starts=3)
        assert list(engine.problems.iterdir()) == []

    assert len(left) == 1 and len(sizes) == 3
    assert max(sizes) < 10_000 and 10 * max(sizes) < min(problem_bytes), (sizes, problem_bytes)


def test_a_fit_arriving_during_another_takes_turns_with_it():
    """A fit arriving while another runs alternates with it start by start (9 turns here), not
    after all of its starts as at one priority (1 turn)."""
    done = {}
    with _engine(n_workers=1) as engine:

        def fit(label):
            done[label] = engine.execute([_Sleep(label, i, 0.2) for i in range(6)])

        first = threading.Thread(target=fit, args=("a",))
        first.start()
        time.sleep(0.1)  # the first fit's starts are all queued and its start 0 is running
        fit("b")
        first.join()

    assert [i for _, i, _ in done["a"]] == list(range(6)), "results come back in task order"
    finished = [label for label, _, _ in sorted(done["a"] + done["b"], key=lambda r: r[2])]
    assert sum(x != y for x, y in zip(finished, finished[1:])) >= 4, finished


def test_a_fit_waits_on_its_own_client_not_the_latest_one_in_the_process(monkeypatch):
    """``distributed.wait`` runs on the process's default client, the latest one made. With servers
    in one process (as tests/test_pool.py runs them), a fit waited on another server's client and
    failed in ``distributed.utils.sync`` once that server closed it (#219's stress script)."""
    waited_on = []

    def spy(*args, **kwargs):
        waited_on.append(default_client())
        return wait(*args, **kwargs)

    monkeypatch.setattr(dask_engine, "wait", spy)
    with _engine(n_workers=1) as engine, dask_engine.Client(engine.client.cluster):
        engine.execute([_Sleep("a", 0, 0.1)])
    assert waited_on and all(client is engine.client for client in waited_on)


def test_a_starts_records_land_in_the_pypesto_log_of_the_session_that_ran_it(tmp_path):
    """A start's records, logged on a worker, land in the pypesto.log of the session whose fit ran
    it, and in no other."""
    logging_setup.configure_pypesto(tmp_path / "default")
    with _engine() as engine:
        for text in ("alpha", "beta"):
            with logging_setup.use_store(tmp_path / text):
                assert engine.execute([_Log(text)]) == [text]

    for text, other in (("alpha", "beta"), ("beta", "alpha")):
        log = (tmp_path / text / "pypesto.log").read_text(encoding="utf-8")
        assert f"INFO pypesto.optimize: pypesto says {text}" in log
        assert f"WARNING fides: fides says {text}" in log
        assert other not in log


def test_a_killed_worker_fails_the_fit_as_the_pools_failure():
    """A worker that dies under a start is a cluster failure: the fit fails with FitBackendError,
    which says so. DeepSeek raimundez 63782292_12 read a bare ``KilledWorker`` as its Hill form
    crashing (PEtab-MCP#341)."""
    with _engine(n_workers=1, config={"distributed.scheduler.allowed-failures": 0}) as engine:
        with pytest.raises(FitBackendError, match="KilledWorker") as failed:
            engine.execute([_Die()])
    assert "the pool's failure, not the model's" in str(failed.value)


def test_a_failed_start_cancels_the_fits_starts_still_queued(tmp_path):
    """A failed fit leaves none of its other starts queued on the shared pool (at most the one its
    worker had already taken runs)."""
    ran = tmp_path / "ran"
    ran.mkdir()
    with _engine(n_workers=1) as engine:
        with pytest.raises(FitBackendError, match="start failed"):
            engine.execute([_Fail(), *(_Touch(str(ran / str(i))) for i in range(5))])
        time.sleep(3)  # long enough for every start left queued to run
    assert len(list(ran.iterdir())) <= 1


@pytest.fixture
def batch_size(monkeypatch):
    """The starts per task of a fit of ``n_starts`` on a pool with ``free`` idle workers, after a
    fit whose starts took ``times`` (``None``: no previous fit), at a per-start cap of 60 s."""
    monkeypatch.setattr(fitting, "FIT_TASK_SECONDS", 270.0)
    monkeypatch.setattr(fitting, "FIT_STARTS_PER_TASK", 0)

    def k(n_starts, times, free=0):
        shared = SimpleNamespace(free_workers=lambda: free, batched=lambda k, *_: k)
        previous = None if times is None else {"starts": [{"time": t} for t in times]}
        return fitting._fit_engine(n_starts, 60.0, shared, previous)

    return k


def test_the_batch_size_fills_FIT_TASK_SECONDS_unless_FIT_STARTS_PER_TASK_is_set(
    batch_size, monkeypatch
):
    """On a pool with no idle worker, a fit runs as many starts per task as fill FIT_TASK_SECONDS at
    the 90th percentile start time of the session's previous fit (here 9 of its 10 timed starts), or
    at the per-start cap without one, at least one and never more than it has. FIT_STARTS_PER_TASK,
    when set, replaces all of that, idle workers or not."""

    def times(p90):  # the None is a start with no finite time
        return [None, *[p90] * 9, 10 * p90]

    assert [batch_size(20, times(p90)) for p90 in (30.0, 100.0, 400.0)] == [9, 2, 1]
    assert [batch_size(20, None), batch_size(4, times(30.0))] == [4, 4]
    monkeypatch.setattr(fitting, "FIT_STARTS_PER_TASK", 5)
    assert [batch_size(20, times(30.0)), batch_size(20, None)] == [5, 5]
    assert [batch_size(4, times(400.0)), batch_size(20, times(30.0), free=100)] == [4, 5]


def test_a_fast_fit_on_an_idle_pool_runs_about_one_start_per_task(batch_size):
    """A fit with idle workers is spread over them, however many of its fast starts would fill
    FIT_TASK_SECONDS: fujita's starts packed 43 to a task left a 384-worker pool idle while its
    75-start fit took 2,638 s (#253)."""
    fast = [2.0] * 50

    assert [batch_size(75, fast, free=384), batch_size(50, fast, free=50)] == [1, 1]
    assert batch_size(50, fast, free=20) == 3


def test_a_fit_on_a_pool_with_few_idle_workers_keeps_its_time_based_batches(batch_size):
    """Idle workers only ever shorten a fit's tasks: with too few to spread over, it fills
    FIT_TASK_SECONDS as on a busy pool, which keeps the shared scheduler's state changes few (#189's
    Check 10)."""
    thirty = [30.0] * 50

    assert [batch_size(100, thirty, free=free) for free in (0, 2, 11)] == [9, 9, 9]
    assert batch_size(100, thirty, free=25) == 4


def test_a_heavy_tailed_history_keeps_its_tail_out_of_one_task(batch_size):
    """A task runs as long as all its starts together. At the median of a history whose slowest
    fifth took 135 s, a 50-start fit would go in one task of 1,430 s; at its 90th percentile no task
    runs past FIT_TASK_SECONDS at the tail's pace (#253)."""
    tail = [2.0] * 40 + [135.0] * 10

    assert batch_size(50, tail) == 2


def test_a_fit_spreads_over_the_threads_its_pool_has_idle(monkeypatch):
    """The engine reads its pool's idle threads from the scheduler (a thread running a batch is not
    idle), and a fit's batch size is spread over them; a closed client reads none, so its fit packs
    by time until it reconnects."""
    monkeypatch.setattr(fitting, "FIT_TASK_SECONDS", 270.0)
    monkeypatch.setattr(fitting, "FIT_STARTS_PER_TASK", 0)
    fast = {"starts": [{"time": 2.0}] * 10}
    with _engine() as engine:
        assert engine.free_workers() == 2
        assert fitting._fit_engine(4, 60.0, engine, fast).starts_per_task == 2
        held = engine.client.submit(time.sleep, 30, pure=False)
        deadline = time.monotonic() + 10
        while engine.free_workers() != 1 and time.monotonic() < deadline:
            time.sleep(0.1)
        assert engine.free_workers() == 1
        assert fitting._fit_engine(4, 60.0, engine, fast).starts_per_task == 4
        engine.client.cancel(held)
        engine.client.close()
        assert engine.free_workers() == 0
        assert fitting._fit_engine(4, 60.0, engine, fast).starts_per_task == 4


def test_each_fit_on_the_shared_pool_stops_its_batches_at_its_own_cap():
    """Two fits batched off one engine keep their own caps, so a batch is stopped at its own
    session's deadline, not at another's (#187: one pool serves sessions capped at 1200 s and at
    12000 s)."""
    shared = object.__new__(DaskEngine)  # no cluster: only the deadline is read
    ran = multistart.FIT_DEADLINE_MARGIN_S + 100.0
    shared.client = SimpleNamespace(run_on_scheduler=lambda _: {"batch": ran})
    futures = [SimpleNamespace(key="batch")]

    short, long = shared.batched(1, 60.0), shared.batched(1, 120.0)

    assert short._look(futures, [1])([0], None) == ({0}, set())
    assert long._look(futures, [1])([0], None) == (set(), set())


def test_a_batched_fit_runs_k_starts_per_task_and_keeps_their_order_and_records(
    tmp_path, monkeypatch
):
    """A fit batched 3 to a task sends its 7 starts as ceil(7/3) = 3 tasks, batch j at priority -j,
    gets their results back in task order, and its starts' records land in its session's
    pypesto.log."""
    logging_setup.configure_pypesto(tmp_path / "default")
    texts = [f"start{i}" for i in range(7)]
    priorities = []
    with _engine() as engine:
        submit = engine.client.submit

        def spy(*args, priority, **kwargs):
            priorities.append(priority)
            return submit(*args, priority=priority, **kwargs)

        monkeypatch.setattr(engine.client, "submit", spy)
        with logging_setup.use_store(tmp_path / "session"):
            assert engine.batched(3).execute([_Log(text) for text in texts]) == texts

    assert priorities == [0, -1, -2]
    log = (tmp_path / "session" / "pypesto.log").read_text(encoding="utf-8")
    assert all(f"pypesto says {text}" in log for text in texts)


class _Fval(Task):
    """A start whose optimizer result has the objective ``fval``."""

    def __init__(self, fval: float) -> None:
        super().__init__()
        self.fval, self.id = fval, str(fval)

    def execute(self):
        return SimpleNamespace(fval=self.fval)


def test_a_batch_crosses_the_scheduler_once_though_its_progress_is_read(monkeypatch):
    """The progress reader gathers each batch as it finishes, for the best objective so far
    (PEtab-MCP#172). A batch it read is not gathered again, so the scheduler that a sweep's runs
    share relays each result once, as before the reader."""
    fvals = [3.0, 1.0, 2.0, 5.0]
    with _engine() as engine:
        gather, keys = engine.client._gather, []

        async def counted(futures, *args, **kwargs):
            keys.extend(future.key for future in futures)
            return await gather(futures, *args, **kwargs)

        monkeypatch.setattr(engine.client, "_gather", counted)
        progress = multistart.Progress()
        token = multistart.PROGRESS.set(progress)
        try:
            out = engine.batched(2).execute([_Fval(v) for v in fvals])
        finally:
            multistart.PROGRESS.reset(token)

    assert [result.fval for result in out] == fvals
    assert (progress.done, progress.best) == (4, 1.0)
    assert len(keys) == len(set(keys)) == 2


def test_the_starts_of_a_batch_share_nothing_on_their_worker():
    """Each start of a batch runs on its own copy of what it was given, as it would alone: starts
    that share one problem here do not share one objective there."""
    problem = []
    with _engine(n_workers=1) as engine:
        assert engine.batched(3).execute([_Count(problem) for _ in range(3)]) == [1, 1, 1]


def test_a_fits_problem_file_is_removed_however_the_fit_ends(monkeypatch, left):
    """The file a fit left its problem in is gone once the fit returns: finished, failed, or with a
    batch stopped past its caps that still runs on its worker."""
    monkeypatch.setattr(dask_engine, "FIT_DEADLINE_MARGIN_S", 0.5)
    monkeypatch.setattr(multistart, "POLL_S", 0.1)
    with _engine() as engine:
        engine.client.run(_Sleep, "warm", 0, 0)
        engine.batched(2).execute(_sharing([_Sleep("a", i, 0) for i in range(3)]))
        with pytest.raises(FitBackendError, match="start failed"):
            engine.execute(_sharing([_Fail(), _Sleep("b", 1, 0)]))
        _, stopped = engine.batched(2, 0.25)._run(
            _sharing([_Sleep("c", 0, 0), _Sleep("c", 1, 30), _Sleep("c", 2, 0)])
        )
        assert stopped == {0, 1}
        assert len(left) == 3 and list(engine.problems.iterdir()) == []


def test_a_worker_that_cannot_read_its_fits_problem_fails_the_fit_as_fit_pool(monkeypatch):
    """A worker off the server's build root cannot read the file its fit left the problem in: the
    fit fails as ``fit_pool``, naming the file and the host, not with an unpickling error."""
    leave = dask_engine._leave

    def gone(problem, folder):
        path = leave(problem, folder)
        path.unlink()
        return path

    monkeypatch.setattr(dask_engine, "_leave", gone)
    with _engine(n_workers=1) as engine:
        with pytest.raises(InfrastructureError) as refused:
            engine.execute(_sharing([_Sleep("a", 0, 0)]))

    assert refused.value.cause == FIT_POOL
    assert f"is not readable on worker {socket.gethostname()}" in str(refused.value)


def test_a_start_failing_mid_batch_fails_the_fit_and_cancels_its_batches_still_queued(tmp_path):
    """A start failing inside a batch fails the fit with FitBackendError, and none of the fit's
    batches still queued runs (at most the one its worker had already taken)."""
    ran = tmp_path / "ran"
    ran.mkdir()
    touch = [_Touch(str(ran / str(i))) for i in range(5)]
    with _engine(n_workers=1) as engine:
        with pytest.raises(FitBackendError, match="start failed"):
            engine.batched(2).execute([touch[0], _Fail(), *touch[1:]])
        time.sleep(3)  # long enough for every batch left queued to run
    assert len(list(ran.iterdir())) <= 3


def test_a_batch_running_past_its_starts_caps_is_stopped_and_the_others_kept(monkeypatch):
    """A batch whose start never returns is given up on once it has run past its starts' caps and
    the margin, as its worker's heartbeats tell the scheduler: its starts, the one of them that
    finished included, are the stopped ones, and the fit's other batch comes back (#118)."""
    monkeypatch.setattr(dask_engine, "FIT_DEADLINE_MARGIN_S", 0.5)
    monkeypatch.setattr(multistart, "POLL_S", 0.1)
    with _engine() as engine:
        # Every worker imports this module now, not while a batch's clock runs.
        engine.client.run(_Sleep, "warm", 0, 0)
        began = time.monotonic()
        results, stopped = engine.batched(2, 0.25)._run(
            [_Sleep("a", 0, 0), _Sleep("a", 1, 30), _Sleep("a", 2, 0)]
        )
        took = time.monotonic() - began

    assert stopped == {0, 1}
    assert results[:2] == [None, None] and results[2][:2] == ("a", 2)
    assert took < 15, took


def test_a_fit_whose_every_batch_is_stopped_fails_as_wedged(monkeypatch):
    """A fit whose only batch runs past its starts' caps has nothing to score: it fails as wedged
    rather than holding the session."""
    monkeypatch.setattr(dask_engine, "FIT_DEADLINE_MARGIN_S", 0.5)
    monkeypatch.setattr(multistart, "POLL_S", 0.1)
    with _engine(n_workers=1) as engine:
        with pytest.raises(InfrastructureError) as wedged:
            engine.batched(2, 0.25).execute([_Sleep("a", 0, 30), _Sleep("a", 1, 0)])

    assert wedged.value.cause == FIT_WEDGED
    assert "all 2 starts ran past the 0s per-start cap" in str(wedged.value)


def test_a_batch_waiting_behind_another_fits_is_not_wedged(monkeypatch):
    """A batch's deadline counts from when it began running: one that waited its whole deadline
    behind another session's batch, or for the pool's jobs to start, is not wedged."""
    monkeypatch.setattr(dask_engine, "FIT_DEADLINE_MARGIN_S", 0.2)
    monkeypatch.setattr(multistart, "POLL_S", 0.1)
    done = {}
    with _engine(n_workers=1) as engine:
        other = threading.Thread(
            target=lambda: done.update(a=engine.batched(1, 10).execute([_Sleep("a", 0, 3)]))
        )
        other.start()
        time.sleep(0.5)  # the other fit's batch is running
        done["b"] = engine.batched(1, 0.1).execute([_Sleep("b", 0, 0.05)])
        other.join()

    assert [label for label, _, _ in done["a"] + done["b"]] == ["a", "b"]


def test_a_fits_queue_seconds_are_its_wait_with_none_of_its_starts_running(monkeypatch):
    """On a pool of one worker, a fit arriving while another's batch runs waits with none of its own
    running until that batch ends: those seconds are its queue_seconds, and meanwhile its Progress
    says it is queued with all its starts (#217). The fit already running waited for nothing."""
    monkeypatch.setattr(multistart, "POLL_S", 0.1)
    done = {}
    with _engine(n_workers=1) as engine:
        engine.client.run(_Sleep, "warm", 0, 0)
        first, second = engine.batched(1), engine.batched(1)
        other = threading.Thread(target=lambda: done.update(a=first.execute([_Sleep("a", 0, 3)])))
        other.start()
        time.sleep(1)  # the scheduler has heard that the first fit's batch runs
        progress, seen = multistart.Progress(), []
        threading.Timer(1, lambda: seen.append((progress.phase, progress.queued))).start()
        token = multistart.PROGRESS.set(progress)
        submitted = time.time()
        try:
            second.execute([_Sleep("b", 0, 0.1), _Sleep("b", 1, 0.1)])
        finally:
            multistart.PROGRESS.reset(token)
        other.join()

    waited = done["a"][0][2] - submitted
    assert seen == [("queued", 2)]
    assert waited - 0.2 < second.queue_seconds < waited + 1, (second.queue_seconds, waited)
    assert first.queue_seconds < 1


def test_a_lost_cluster_fails_the_fit_and_nothing_reruns_it_here(fit_session, monkeypatch):
    """A client that lost its cluster fails the fit as the pool's failure, ``fit_pool``: `minimize`
    runs once, on the cluster, never again sequentially on the server node."""
    import pypesto.optimize

    engines = []
    minimize = pypesto.optimize.minimize

    def spy(*args, engine=None, **kwargs):
        engines.append(engine)
        return minimize(*args, engine=engine, **kwargs)

    monkeypatch.setattr(pypesto.optimize, "minimize", spy)
    with _engine(n_workers=1) as engine:
        fit_session.engine.fit_engine = engine
    with pytest.raises(InfrastructureError, match="^infrastructure_error: fit_pool: .*closed"):
        fitting.fit_and_simulate(fit_session, fit_session.engine, do_fit=True, n_starts=2)
    assert len(engines) == 1 and engines[0].client is engine.client


def _no_build_plugin(dask_worker) -> int:
    """The identity of the :class:`_NoBuild` instance installed on this worker (run on a worker)."""
    return id(dask_worker.plugins[dask_engine._NoBuild.name])


def test_a_server_joining_a_cluster_that_has_no_build_reaches_no_worker():
    """A server joining a pool, which registered _NoBuild first, leaves every worker's plugin as it
    was: no plugin-add is broadcast. At 1200 workers, 100 servers' broadcasts stalled the
    scheduler past every connect's 30 s deadline (2026-11-04)."""
    with _engine() as engine:
        installed = engine.client.run(_no_build_plugin)
        with DaskEngine(engine.client.cluster, engine.problems.parent).client:
            assert engine.client.run(_no_build_plugin) == installed


def test_a_worker_never_builds_a_model(tmp_path):
    """On a worker, pyPESTO's creators and AMICI's importer refuse to build in the shared build root,
    naming the folder and the host, and leave the folder as it was. The pool failed the run: its
    build root is not the server's."""
    folder = tmp_path / "pypesto_hash"
    folder.mkdir()
    (folder / "half_built").write_text("", encoding="utf-8")
    refusal = re.escape(f"{folder} is not importable on worker {socket.gethostname()}")
    with _engine(n_workers=1) as engine:
        for how in ("v1", "v2", "amici"):
            with pytest.raises(InfrastructureError, match=refusal) as refused:
                engine.execute([_Build(str(folder), how)])
            assert refused.value.cause == FIT_POOL
    assert [p.name for p in folder.iterdir()] == ["half_built"]
