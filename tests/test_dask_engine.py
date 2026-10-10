"""Fits on a shared dask cluster (``petab_mcp.dask_engine``).

A server given a cluster runs every session's starts there instead of in local processes, so fits
running at once share one worker pool rather than oversubscribing the node, where Fides' per-start
time cap stops a slowed start early. These pin what crosses the process boundary: a fit on the
cluster ends where the local fit ends, start for start; fits take turns; a start's log reaches the
session that ran it; a cluster failure fails the fit and is never rerun on the server; a worker never
compiles; and the starts are today's, drawn from the fit's own RNG. Real ``LocalCluster``s with
process workers, because the contract is about that boundary.
"""

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
    """`create_server(cluster=...)` hands one DaskEngine to every fit of the server. The cluster
    runs the starts the server drew, on the module the server built, so each start ends where it
    ends in this process, not merely the best one."""
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
        call("make_changes", {"parameters": [bounds], "justification": "x", "n_starts": 4})
        sql = "SELECT id, fval, n_fval FROM optimization_starts"
        return {r["id"]: (r["fval"], r["n_fval"]) for r in call("query_data", {"sql": sql})["rows"]}

    local = fit("local")
    with _engine() as engine, get_task_stream(engine.client) as stream:
        remote = fit("dask", engine.client.cluster)

    assert len(stream.data) == 4, "every start ran on the cluster"
    # Exact: on the toy every start reaches one optimum, so only the last digits of `fval` and the
    # evaluation count tell the starts apart.
    assert len(local) == 4 and remote == local


def test_a_fit_arriving_during_another_takes_turns_with_it():
    """Start i of every fit is submitted at priority -i, so a fit that arrives while another runs
    does not queue behind all of the other's starts: on one worker their starts finish alternating
    (9 turns here). At one priority dask runs them first come, first served: the first fit's six,
    then the second's (1 turn)."""
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
    """`get_fitting_log` serves the session's pypesto.log, and a start on a worker logs in another
    process. Its records come back with its result and are re-emitted in the calling thread, so the
    per-store routing files each in the log of the session whose fit ran it, and in no other."""
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


def test_a_lost_cluster_fails_the_fit_and_nothing_reruns_it_here(fit_session, monkeypatch):
    """`_run_fit` reruns a failed parallel multistart sequentially, in this process. For the cluster
    that would quietly run every start on the server node, the oversubscription the cluster exists
    to end, so a cluster failure fails the fit: `minimize` runs once, on the cluster."""
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
    assert engines == [engine]


def test_a_worker_never_builds_a_model(tmp_path):
    """A start reaches a worker by import only. pyPESTO's creators, asked for a module that will not
    import there, would rmtree its folder and recompile it inside the build root every session
    shares; on a worker they and AMICI's importer refuse instead, naming the folder and the host."""
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
    """AMICI pickles a solver's settings through HDF5, so without it no start reaches a worker and
    every fit on the cluster fails, as on CI's AMICI before the workflow installed HDF5. The server
    refuses the cluster instead, before connecting to it (`object()` is no cluster)."""
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


def test_a_fits_own_stream_draws_the_starts_the_global_seed_drew(session, engine):
    """Fresh starts came from numpy's global stream after `np.random.seed(FIT_SEED)`, which two
    fits in two threads interleave. The fit's own RandomState must draw the same starts byte for
    byte, warm starts first, or no stored score is reproducible by a refit; and it must leave the
    global stream alone."""
    from pypesto.startpoint import to_startpoint_method

    problem, _ = fitting.estimation_problem(session, engine)
    n_starts = 8
    problem.x_guesses_full = fitting.warm_start_guesses(
        problem, session.problem.parameter_df, n_starts
    )
    np.random.seed(fitting.FIT_SEED)
    before = problem.startpoint_method(n_starts=n_starts, problem=problem)
    state = np.random.get_state()

    after = to_startpoint_method(fitting._seeded_starts())(n_starts=n_starts, problem=problem)

    assert after.tobytes() == before.tobytes()
    assert np.random.get_state()[2] == state[2] and (np.random.get_state()[1] == state[1]).all()
