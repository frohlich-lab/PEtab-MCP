"""Fits on a shared dask cluster (``petab_mcp.dask_engine``), on real ``LocalCluster``s with process
workers: the engine's contract is what crosses that boundary."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import re
import socket
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import dask
import numpy as np
import pytest
from distributed import LocalCluster, get_task_stream
from pypesto.engine import Task

from petab_mcp import fitting
from petab_mcp import logging_setup
from petab_mcp.dask_engine import DaskEngine
from petab_mcp.errors import FitBackendError
from petab_mcp.references import StubReferenceResolver
from petab_mcp.server import create_server


@contextlib.contextmanager
def _engine(n_workers: int = 2, config: dict | None = None):
    """A :class:`DaskEngine` on a fresh process cluster; ``config`` is dask config for its
    scheduler. Closed on exit."""
    with dask.config.set(config or {}):
        cluster = LocalCluster(
            n_workers=n_workers, threads_per_worker=1, processes=True, dashboard_address=":0"
        )
    with cluster:
        engine = DaskEngine(cluster)
        with engine.client:
            yield engine


class _Sleep(Task):
    """A start that takes ``seconds`` and returns when it finished."""

    def __init__(self, label: str, i: int, seconds: float) -> None:
        super().__init__()
        self.label, self.i, self.seconds = label, i, seconds

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


def test_a_killed_worker_fails_the_fit():
    """A worker that dies under a start is a cluster failure: the fit fails with FitBackendError."""
    with _engine(n_workers=1, config={"distributed.scheduler.allowed-failures": 0}) as engine:
        with pytest.raises(FitBackendError, match="KilledWorker"):
            engine.execute([_Die()])


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


def test_a_batch_holds_FIT_TASK_SECONDS_of_the_previous_fits_median_start(monkeypatch):
    """A fit on the cluster runs as many starts per task as fill FIT_TASK_SECONDS at the MEDIAN start
    time of the session's previous fit (the mean would be dragged by one slow start), at least one
    and never more than it has; its first fit sends one start per task."""
    monkeypatch.setattr(fitting, "FIT_TASK_SECONDS", 270.0)
    shared = SimpleNamespace(batched=lambda k: k)  # hands back the batch size it is given

    def k(n_starts, median):
        previous = None
        if median is not None:  # the None is a start with no finite time
            previous = {"starts": [{"time": t} for t in (None, median / 3, median, 10 * median)]}
        return fitting._fit_engine(n_starts, shared, previous)

    assert [k(20, median) for median in (30.0, 100.0, 400.0, None)] == [9, 2, 1, 1]
    assert k(4, 30.0) == 4


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


def test_a_lost_cluster_fails_the_fit_and_nothing_reruns_it_here(fit_session, monkeypatch):
    """A cluster failure fails the fit: `minimize` runs once, on the cluster, never again
    sequentially on the server node."""
    import pypesto.optimize

    engines = []
    minimize = pypesto.optimize.minimize

    def spy(*args, engine=None, **kwargs):
        engines.append(engine)
        return minimize(*args, engine=engine, **kwargs)

    monkeypatch.setattr(pypesto.optimize, "minimize", spy)
    with _engine(n_workers=1) as engine:
        fit_session.engine.fit_engine = engine
    with pytest.raises(FitBackendError, match="ClosedClientError"):
        fitting.fit_and_simulate(fit_session, fit_session.engine, do_fit=True, n_starts=2)
    assert len(engines) == 1 and engines[0].client is engine.client


def test_a_worker_never_builds_a_model(tmp_path):
    """On a worker, pyPESTO's creators and AMICI's importer refuse to build in the shared build root,
    naming the folder and the host, and leave the folder as it was."""
    folder = tmp_path / "pypesto_hash"
    folder.mkdir()
    (folder / "half_built").write_text("", encoding="utf-8")
    refusal = re.escape(f"{folder} is not importable on worker {socket.gethostname()}")
    with _engine(n_workers=1) as engine:
        for how in ("v1", "v2", "amici"):
            with pytest.raises(FitBackendError, match=refusal):
                engine.execute([_Build(str(folder), how)])
    assert [p.name for p in folder.iterdir()] == ["half_built"]


def test_an_amici_without_hdf5_refuses_the_cluster_before_connecting(
    yaml_path, tmp_path, monkeypatch
):
    """An AMICI without HDF5 cannot pickle a start, so the server refuses the cluster before
    connecting to it (`object()` is no cluster)."""
    import amici.sim.sundials

    monkeypatch.setattr(amici.sim.sundials, "hdf5_enabled", False)
    with pytest.raises(FitBackendError, match="without HDF5"):
        create_server(
            yaml_path,
            provenance_dir=str(tmp_path / "prov"),
            resolver=StubReferenceResolver(),
            auto_fit=False,
            cluster=object(),
        )
