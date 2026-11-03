"""A fit ends (``petab_mcp.multistart``): one whose starts stop finishing fails by its deadline, a
change arriving meanwhile is refused at once instead of queued behind it, and a client's cancel
stops it. One fit's starts went quiet after 40 minutes and held its session for 9 h, while its
client resent the same change three times behind it (PEtab-MCP#108). A start stuck past its cap is
stopped and the fit scored from the rest: one hung start of 300 used to fail its whole fit after
11,400 s, discarding the 299 that had finished (PEtab-MCP#118)."""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextvars
import functools
import logging
import math
import multiprocessing
import os
import re
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import anyio
import numpy as np
import pypesto
import pypesto.optimize
import pytest
from fastmcp import Client

from petab_mcp import fitting, logging_setup
from petab_mcp import server
from petab_mcp import multistart as M
from petab_mcp.errors import (
    FIT_IN_PROGRESS,
    FIT_WEDGED,
    FitCancelledError,
    InfrastructureError,
    ModelUpdateError,
)
from petab_mcp.references import StubReferenceResolver
from petab_mcp.server import create_server
from petab_mcp.session import PetabSession

TOY = Path(__file__).parent / "data" / "toy_v2"

_ANTIMONY = """model conversion_reaction
  compartment compartment_ = 1;
  species A in compartment_;
  species B in compartment_;
  A = A_init;
  B = B_init;
  A_init = 1.0;
  B_init = 0.0;
  k1 = 0.5;
  k2 = 0.1;
  k3 = 0.01;
  r1: A => B; k1 * A;
  r2: B => A; k2 * B;
  r3: A => ; k3 * A;
end
"""


def _start(value):
    """A start that returns ``value`` at once; stdlib-only, so a spawned worker unpickles it
    without importing this module."""
    return SimpleNamespace(execute=functools.partial(abs, value))


def _hang():
    """A start that never returns, as a worker deadlocked on a lock it inherited."""
    return SimpleNamespace(id="hung", execute=functools.partial(time.sleep, 3600))


class _Pool:
    """A ``ProcessPoolExecutor`` that runs each start in a thread of this process, at most
    ``n_procs`` at once, and records the workers it is told to kill: a hung worker without a
    process."""

    def __init__(self, n_procs, mp_context=None, initializer=None, initargs=()):
        self.killed = []
        self._processes = {
            i: SimpleNamespace(kill=functools.partial(self.killed.append, i))
            for i in range(n_procs)
        }
        self._free = threading.Semaphore(n_procs)
        _Pool.last = self

    def submit(self, fn, payload):
        future = concurrent.futures.Future()

        def run():
            with self._free:
                future.set_result(fn(payload))

        threading.Thread(target=run, daemon=True).start()
        return future

    def shutdown(self, wait=True, cancel_futures=False):
        pass


def test_a_fit_whose_every_worker_is_stuck_fails_as_wedged_and_kills_them(monkeypatch):
    """Two stuck workers leave a third start queued: once no start has finished for a start's cap
    plus the margin, the fit fails as wedged, says how far it got, and kills its workers."""
    monkeypatch.setattr(M.concurrent.futures, "ProcessPoolExecutor", _Pool)
    monkeypatch.setattr(M, "FIT_DEADLINE_MARGIN_S", 0.2)
    monkeypatch.setattr(M, "POLL_S", 0.02)
    began = time.monotonic()
    with pytest.raises(InfrastructureError) as wedged:
        M.ProcessEngine(2, start_seconds=0.1).execute([_hang(), _hang(), _start(3)])
    took = time.monotonic() - began

    assert wedged.value.cause == FIT_WEDGED
    assert str(wedged.value).startswith("infrastructure_error: fit_wedged: no start had finished")
    assert "0 of 3 starts had finished" in str(wedged.value)
    assert 0.1 + 0.2 <= took < 2, took
    assert _Pool.last.killed == [0, 1]


def test_a_fit_whose_every_start_is_stopped_fails_as_wedged(monkeypatch):
    """A fit with no start left to score fails as wedged rather than as an ill-posed model, and
    kills its workers."""
    monkeypatch.setattr(M.concurrent.futures, "ProcessPoolExecutor", _Pool)
    monkeypatch.setattr(M, "FIT_DEADLINE_MARGIN_S", 0.2)
    monkeypatch.setattr(M, "POLL_S", 0.02)
    with pytest.raises(InfrastructureError) as wedged:
        M.ProcessEngine(2, start_seconds=0.1).execute([_hang(), _hang()])

    assert wedged.value.cause == FIT_WEDGED
    assert "all 2 starts ran past the 0s per-start cap" in str(wedged.value)
    assert _Pool.last.killed == [0, 1]


def _quadratic(x, hang_at, marker):
    """A quadratic's value, which at ``hang_at`` blocks as a start stuck inside one simulation,
    where Fides never checks its cap: always, or, given ``marker``, until that file exists (the
    first call leaves it)."""
    if x[0] == hang_at and not (marker and os.path.exists(marker)):
        if marker:
            Path(marker).touch()
        time.sleep(3600)
    return float(x @ x)


def _gradient(x):
    return 2 * x


def _hessian(x):
    return 2 * np.eye(len(x))


def _fit(engine, marker=None):
    """A six-start Fides multistart of a quadratic on ``engine``, whose start "0", warm at (1, 1),
    blocks in its first evaluation (:func:`_quadratic`)."""
    objective = pypesto.Objective(
        fun=functools.partial(_quadratic, hang_at=1.0, marker=marker),
        grad=_gradient,
        hess=_hessian,
    )
    problem = pypesto.Problem(objective, lb=[-5, -5], ub=[5, 5], x_guesses=[[1.0, 1.0]])
    return pypesto.optimize.minimize(
        problem,
        optimizer=pypesto.optimize.FidesOptimizer(verbose=logging.ERROR),
        n_starts=6,
        engine=engine,
        progress_bar=False,
    )


@pytest.fixture
def stuck(monkeypatch):
    """Caps short enough to watch a start stopped, in forked workers: one started by the fork
    server imports pyPESTO before its start begins, which on a loaded machine outlasts them."""
    monkeypatch.setattr(M, "FIT_DEADLINE_MARGIN_S", 1.0)
    monkeypatch.setattr(M, "POLL_S", 0.05)
    monkeypatch.setattr(M, "START_METHOD", "fork")


def _no_worker_left():
    for _ in range(200):
        if not multiprocessing.active_children():
            return True
        time.sleep(0.05)
    return False


def test_a_start_stuck_inside_a_simulation_is_stopped_and_the_fit_scored_from_the_rest(stuck):
    """REGRESSION (#118): one start of 300 hung inside a simulation, and its fit failed as wedged
    after 11,400 s with its 299 other starts discarded. Now it is killed its cap plus the margin
    after the last start was handed out, and scored as a failed start beside the rest."""
    engine = M.ProcessEngine(2, start_seconds=0.5)
    began = time.monotonic()
    result = _fit(engine)
    took = time.monotonic() - began

    starts = {r.id: r for r in result.optimize_result.list}
    assert engine.killed == ["0"]
    assert starts["0"].fval == np.inf and starts["0"].exitflag == -1 and starts["0"].x is None
    assert starts["0"].message == "stopped by the server: still running past its 0s cap"
    assert list(starts["0"].x0) == [1.0, 1.0]
    assert all(starts[i].fval < 1e-8 for i in "12345")
    assert 0.5 + 1.0 <= took < 30, took
    assert _no_worker_left()


def test_a_stopped_warm_start_from_the_parents_optimum_is_rerun_once(stuck, tmp_path, caplog):
    """The warm start from the parent's optimum keeps a nested edit's fit within reach of its
    parent's, so one stopped past its cap runs once more, in a fresh pool, and counts as run."""
    engine = M.ProcessEngine(2, start_seconds=0.5, rerun=frozenset({"0"}))
    with caplog.at_level(logging.WARNING, logger=M.__name__):
        result = _fit(engine, marker=str(tmp_path / "hung"))

    assert (tmp_path / "hung").exists()
    assert engine.killed == []
    assert all(r.fval < 1e-8 for r in result.optimize_result.list)
    assert "fit: rerunning start(s) ['0'] from the parent's optimum once" in caplog.messages
    assert _no_worker_left()


def test_a_warm_start_stopped_twice_fails_the_fit_as_wedged(stuck):
    """A warm start from the parent's optimum stuck again on its rerun fails the fit: scored
    without it, a nested edit could come out worse than its parent for want of the search."""
    with pytest.raises(InfrastructureError) as wedged:
        _fit(M.ProcessEngine(2, start_seconds=0.5, rerun=frozenset({"0"})))

    assert wedged.value.cause == FIT_WEDGED
    assert "start(s) ['0'], warm from the parent's optimum, ran past" in str(wedged.value)
    assert _no_worker_left()


def test_a_fit_within_its_deadline_returns_its_starts_in_task_order(monkeypatch):
    """The starts of a fit that ends in time come back in the order they were given, through real
    worker processes."""
    monkeypatch.setattr(M, "POLL_S", 0.02)
    starts = [_start(-i) for i in range(5)]
    assert M.ProcessEngine(2, start_seconds=60).execute(starts) == [0, 1, 2, 3, 4]


def test_a_worker_that_was_not_forked_logs_to_the_sessions_pypesto_log(monkeypatch, tmp_path):
    """The workers are started by a fork server, not forked from the threaded server, whose
    forked workers sat at 0 CPU until two fits failed as wedged; so they inherit no log routing
    and must be given it, or every start's optimizer log is lost."""
    monkeypatch.setattr(M, "POLL_S", 0.02)
    assert M.START_METHOD != "fork"
    logging_setup.configure_pypesto(tmp_path)
    start = SimpleNamespace(
        execute=functools.partial(logging.getLogger("pypesto").info, "a start ran")
    )
    with logging_setup.use_store(tmp_path):
        M.ProcessEngine(2, start_seconds=60).execute([start, start])
    assert logging_setup.pypesto_log_path(tmp_path).read_text().count("a start ran") == 2


def test_a_cancelled_fit_kills_its_worker_processes(monkeypatch):
    """A hung worker is a real process the fit must kill, not wait for: none is left once the
    cancelled fit has failed."""
    monkeypatch.setattr(M, "POLL_S", 0.02)
    cancelled = threading.Event()
    raised = []

    def fit():
        M.CANCELLED.set(cancelled)
        try:
            M.ProcessEngine(2, start_seconds=60).execute([_hang(), _hang()])
        except FitCancelledError as exc:
            raised.append(exc)

    thread = threading.Thread(target=contextvars.copy_context().run, args=(fit,))
    thread.start()
    for _ in range(200):
        if len(multiprocessing.active_children()) == 2:
            break
        time.sleep(0.05)
    cancelled.set()
    thread.join(30)

    assert raised and "0 of 2 starts had finished" in str(raised[0])
    for _ in range(200):
        if not multiprocessing.active_children():
            break
        time.sleep(0.05)
    assert multiprocessing.active_children() == []


def test_a_running_fit_logs_how_many_of_its_starts_are_done(monkeypatch, caplog):
    """Every PROGRESS_INTERVAL_S the starts done so far reach the session's log, so a fit that hangs
    can be located; before, nothing was logged between `fit: start` and a `fit: done` that never
    came."""
    monkeypatch.setattr(M, "PROGRESS_INTERVAL_S", 0.0)
    monkeypatch.setattr(M, "POLL_S", 0.02)
    futures = [concurrent.futures.Future() for _ in range(3)]
    futures[0].set_result(None)
    threading.Timer(0.1, futures[1].set_result, (None,)).start()
    threading.Timer(0.2, futures[2].set_result, (None,)).start()
    with caplog.at_level(logging.INFO, logger=M.__name__):
        M.watch(futures, [2, 1, 1], concurrent.futures.wait, lambda *_: ((), ()))

    assert "fit: 2/4 starts done after 0s" in caplog.messages


def test_a_running_fit_reports_its_starts_done_and_its_best_into_the_change(monkeypatch):
    """`wait_for_fit` and the fit_in_progress refusal read how far a fit has got off its change's
    Progress (PEtab-MCP#172); a start that raised has no objective to report."""
    monkeypatch.setattr(M, "POLL_S", 0.02)
    futures = [concurrent.futures.Future() for _ in range(3)]
    futures[0].set_result([3.0, math.inf])
    progress, seen = M.Progress(), []

    def finish():
        seen.append((progress.done, progress.best))
        futures[1].set_exception(RuntimeError("a start that raised"))
        futures[2].set_result([1.5])

    threading.Timer(0.2, finish).start()
    token = M.PROGRESS.set(progress)
    try:
        M.watch(
            futures,
            [2, 1, 1],
            concurrent.futures.wait,
            lambda *_: ((), ()),
            fvals=lambda f: f.result(),
        )
    finally:
        M.PROGRESS.reset(token)

    assert seen == [(2, 3.0)]
    assert (progress.total, progress.done, progress.best) == (4, 4, 1.5)


def test_the_starts_left_are_timed_by_the_pace_of_those_done_while_running():
    """A fit queued for a shared pool's workers is not slower: its time queued is left out."""
    progress = M.Progress()
    assert progress.remaining_s(10.0) is None
    progress.began, progress.total = 0.0, 10
    assert progress.remaining_s(10.0) is None
    progress.done = 2
    assert progress.remaining_s(20.0) == 80.0
    progress.enter("queued", 20.0)
    assert progress.remaining_s(30.0) == 80.0
    progress.enter("running", 30.0)
    assert progress.remaining_s(40.0) == 120.0


def test_a_wedged_fit_is_not_rerun_sequentially(fit_session, monkeypatch):
    """A failure the server raised ends the fit: rerunning a wedged multistart in this thread
    would hold the session for as long again, with nothing to bound it."""
    import pypesto.optimize

    class Wedged:
        n_procs = 2

        def execute(self, tasks, progress_bar=None):
            raise InfrastructureError(FIT_WEDGED, "a stub's starts were not done")

    engines = []
    minimize = pypesto.optimize.minimize

    def spy(*args, engine=None, **kwargs):
        engines.append(engine)
        return minimize(*args, engine=engine, **kwargs)

    monkeypatch.setattr(pypesto.optimize, "minimize", spy)
    monkeypatch.setattr(fitting, "_fit_engine", lambda *a, **k: Wedged())
    with pytest.raises(InfrastructureError, match="fit_wedged"):
        fitting.fit_and_simulate(fit_session, fit_session.engine, do_fit=True, n_starts=2)
    assert len(engines) == 1


class _StopsTheLast(M.CappedEngine):
    """Runs a fit's starts here, all but its last, which comes back stopped past its cap."""

    start_seconds = 60.0

    def _run(self, tasks):
        return [task.execute() for task in tasks[:-1]] + [None], {len(tasks) - 1}


def test_a_stopped_start_reaches_the_result_the_version_and_the_de_dup(fit_session, monkeypatch):
    """A fit scored without a stopped start says so: in its result, with a warning naming its
    session's cap, and in its version, whose fit the de-dup then lets stand only for one of as many
    starts as it completed, so the same model asked for with all three is refit, its warm start
    from that fit's optimum named for a rerun (#118)."""
    reruns = []

    def engine(n_starts, per_start_timeout, shared, previous, rerun):
        reruns.append(rerun)
        return _StopsTheLast()

    monkeypatch.setattr(fitting, "_fit_engine", engine)
    fit_session.fit_per_start_timeout = 42.0
    out = fit_session.fit(n_starts=3)

    assert out["committed"] is True and out["neg_log_likelihood"] is not None
    assert out["n_starts"] == 3 and out["n_starts_killed"] == 1
    assert "Starts stopped past the 42s per-start cap: 1 of 3" in out["warnings"][-1]
    assert fit_session.provenance.get_meta(out["version"])["n_starts_killed"] == 1
    stored = fit_session.provenance.get_version(out["version"])["optimization"]
    assert stored["n_starts"] == 3 and stored["n_starts_killed"] == 1
    assert fit_session.fit(n_starts=2)["no_change"] is True
    again = fit_session.fit(n_starts=3)
    assert again["committed"] is True
    assert any("against its 2" in w for w in again["warnings"])
    assert reruns == [frozenset(), frozenset({"0"})]


def test_two_sessions_on_one_pool_fit_under_their_own_caps(tmp_path, monkeypatch):
    """Each session's fits run at its own per-start cap, not one process-wide value: Fides' MAXTIME,
    the cap the shared pool stops a start at, and the starts batched per task all follow the
    session (#187: one server's sessions need 1200 s on one problem and 12000 s on another)."""
    import fides

    from petab_mcp.errors import PetabMcpError

    class Seen(PetabMcpError):
        pass

    seen = []

    def minimize(problem, *, optimizer, engine, **kwargs):
        cap = optimizer.options[fides.Options.MAXTIME]
        seen.append((cap, engine.start_seconds, engine.starts_per_task))
        raise Seen

    monkeypatch.setattr(pypesto.optimize, "minimize", minimize)
    problem = SimpleNamespace(x_names=["k1"], x_free_indices=[0])
    monkeypatch.setattr(fitting, "estimation_problem", lambda session, engine: (problem, {}))
    monkeypatch.setattr(fitting, "FIT_TASK_SECONDS", 270.0)
    monkeypatch.setattr(fitting, "FIT_STARTS_PER_TASK", 0)
    pool = SimpleNamespace(
        batched=lambda k, cap, rerun: SimpleNamespace(starts_per_task=k, start_seconds=cap)
    )
    engine = SimpleNamespace(
        evaluate_objective=lambda: {"neg_log_likelihood": None}, fit_engine=pool
    )
    for name, cap in (("a", 100.0), ("b", 300.0)):
        session = PetabSession(
            str(TOY / "problem.yaml"),
            provenance_dir=str(tmp_path / name),
            resolver=StubReferenceResolver(),
        )
        session.fit_per_start_timeout = cap
        with pytest.raises(Seen):
            fitting._run_fit(session, engine, n_starts=4)

    assert seen == [(100.0, 100.0, 2), (300.0, 300.0, 1)]


def test_a_fit_whose_starts_mostly_run_out_of_time_says_more_starts_will_not_help(
    fit_session, monkeypatch
):
    """MAXTIME is a per-start cap, so a fit most of whose starts hit it is told that more starts
    would hit it too, and what its session's cap is (#213)."""
    fit_session.fit_per_start_timeout = 1e-9
    out = fit_session.fit(n_starts=3)

    assert out["exitflag_counts"] == {"ExitFlag.MAXTIME": 3}
    assert any(
        "3 of 3 starts ran out of the 0s" in w and "More starts" in w for w in out["warnings"]
    )


# -- the tool boundary --------------------------------------------------------------------------
def _server(tmp_path):
    return create_server(
        str(TOY / "problem.yaml"),
        provenance_dir=str(tmp_path / "prov"),
        resolver=StubReferenceResolver(),
    )


def _call(mcp, name, args):
    return asyncio.run(mcp.call_tool(name, args)).structured_content


_CHANGE = {"antimony": _ANTIMONY, "justification": "add a loss of A"}


def test_a_change_arriving_during_a_fit_is_refused_at_once(tmp_path, monkeypatch):
    """The resent change is refused at once with the cause a client tells its lead, instead of
    timing out again behind the fit it resends; once that fit ends the session takes changes."""
    fitting_, release = threading.Event(), threading.Event()

    def _fit(self, n_starts=None):
        fitting_.set()
        release.wait(30)
        raise ModelUpdateError("the stub's fit ended")

    monkeypatch.setattr(PetabSession, "_auto_fit_and_improvement", _fit)
    mcp = _server(tmp_path)
    first = threading.Thread(target=_call, args=(mcp, "make_changes", _CHANGE))
    first.start()
    assert fitting_.wait(30)
    began = time.monotonic()
    out = _call(mcp, "make_changes", _CHANGE)
    took = time.monotonic() - began
    release.set()
    first.join(30)

    assert took < 5, took
    assert out["applied"] is False and out["error_cause"] == FIT_IN_PROGRESS
    assert out["error"].startswith("make_changes: fit_in_progress: a fit is still running")
    assert _call(mcp, "make_changes", _CHANGE).get("error_cause") != FIT_IN_PROGRESS


def test_a_cancelled_call_stops_its_fit_and_undoes_its_edit(tmp_path, monkeypatch):
    """A client's cancel reaches the fit, which runs in a thread no cancel reaches, through
    multistart.CANCELLED: the edit is undone and the session takes the next change."""
    started, heard = threading.Event(), threading.Event()

    def _fit(self, n_starts=None):
        started.set()
        if M.CANCELLED.get().wait(30):
            heard.set()
        raise FitCancelledError("the client cancelled the call")

    monkeypatch.setattr(PetabSession, "_auto_fit_and_improvement", _fit)
    mcp = _server(tmp_path)

    async def cancel_mid_fit():
        async with anyio.create_task_group() as tasks:
            tasks.start_soon(mcp.call_tool, "make_changes", _CHANGE)
            await anyio.to_thread.run_sync(started.wait, 30)
            tasks.cancel_scope.cancel()

    asyncio.run(cancel_mid_fit())

    assert heard.is_set()
    params = _call(mcp, "get_parameters", {})
    assert "k3" not in {r["parameterId"] for r in params.get("result", params)}
    monkeypatch.setattr(PetabSession, "_auto_fit_and_improvement", _fit_ends)
    assert _call(mcp, "make_changes", _CHANGE).get("error_cause") != FIT_IN_PROGRESS


def _stuck_fit(monkeypatch, **progress):
    """Stub the fit to report ``progress`` and hold its session until the returned event is set;
    returns ``(fitting, release)``."""
    fitting_, release = threading.Event(), threading.Event()

    def _fit(self, n_starts=None):
        for key, value in progress.items():
            setattr(M.PROGRESS.get(), key, value)
        fitting_.set()
        release.wait(30)
        raise ModelUpdateError("the stub's fit ended")

    monkeypatch.setattr(PetabSession, "_auto_fit_and_improvement", _fit)
    return fitting_, release


def test_a_wait_for_the_running_fit_says_how_it_stands_then_how_it_ended(tmp_path, monkeypatch):
    """An agent whose make_changes timed out called list_versions 3-5 times in a row while one fit
    ran, each call a model call re-sending a 250k-token context (PEtab-MCP#172). The refusal names
    wait_for_fit and the time left; the wait is the server's, and answers once the fit has ended."""
    fitting_, release = _stuck_fit(
        monkeypatch,
        began=time.monotonic() - 60,
        total=8,
        done=2,
        best=4.5,
        phase="queued",
        queued=3,
    )
    mcp = _server(tmp_path)
    assert _call(mcp, "wait_for_fit", {}) == {"status": "idle"}
    first = threading.Thread(target=_call, args=(mcp, "make_changes", _CHANGE))
    first.start()
    assert fitting_.wait(30)

    refused = _call(mcp, "make_changes", _CHANGE)["error"]
    assert "2 of its 8 starts are done, the rest at that pace about 3 min" in refused
    assert "call wait_for_fit" in refused
    running = _call(mcp, "wait_for_fit", {"timeout_s": 0.1})
    assert running["status"] == "still_running"
    assert (running["starts_done"], running["n_starts"], running["best_so_far"]) == (2, 8, 4.5)
    assert (running["phase"], running["starts_queued"]) == ("queued", 3)
    assert 175 <= running["remaining_s"] <= 200
    threading.Timer(0.5, release.set).start()
    began = time.monotonic()
    ended = _call(mcp, "wait_for_fit", {"timeout_s": 30})
    took = time.monotonic() - began
    first.join(30)

    assert ended == {"status": "no_version"}
    assert took < 10, took


def test_a_wait_for_a_fit_holds_no_worker_thread(tmp_path, monkeypatch):
    """Sync tools share anyio's worker threads, 40 by default, which running fits already hold: a
    wait holding one more would starve other calls. With one thread, a call made during the wait
    still answers at once."""
    fitting_, release = _stuck_fit(monkeypatch)
    mcp = _server(tmp_path)
    first = threading.Thread(target=_call, args=(mcp, "make_changes", _CHANGE))
    first.start()
    assert fitting_.wait(30)

    async def call_during_a_wait():
        anyio.to_thread.current_default_thread_limiter().total_tokens = 1
        async with anyio.create_task_group() as tasks:
            tasks.start_soon(mcp.call_tool, "wait_for_fit", {"timeout_s": 30})
            await anyio.sleep(0.5)
            began = time.monotonic()
            await mcp.call_tool("get_errors", {})
            took = time.monotonic() - began
            release.set()
        return took

    took = asyncio.run(call_during_a_wait())
    first.join(30)
    assert took < 5, took


def test_a_read_waiting_on_a_fits_lock_gets_heartbeats_naming_the_fits_phase(tmp_path, monkeypatch):
    """A read behind a fit waits out the fit (#180), so on gemot's protocol, 2025-11-25, its client
    hears the running change every HEARTBEAT_S meanwhile, as the fit's own call does (#217)."""
    monkeypatch.setattr(server, "HEARTBEAT_S", 0.1)
    fitting_, release = _stuck_fit(monkeypatch, total=4, done=1, phase="queued")
    mcp = _server(tmp_path)
    first = threading.Thread(target=_call, args=(mcp, "make_changes", _CHANGE))
    first.start()
    assert fitting_.wait(30)
    beats = []

    async def heard(progress, total, message):
        beats.append((progress, message))
        if len(beats) == 3:
            release.set()

    async def read():
        async with Client(mcp, mode="legacy", progress_handler=heard) as client:
            assert client.protocol_version == "2025-11-25"
            return await client.call_tool("get_run_timing", {})

    timing = asyncio.run(read()).structured_content
    first.join(30)

    assert timing["queue_seconds"] == 0.0
    assert len(beats) >= 3
    assert all(re.fullmatch(r"queued for \d+s, 1 of 4 starts done", m) for _, m in beats[:3])
    assert [p for p, _ in beats] == sorted(p for p, _ in beats)


def _fit_ends(self, n_starts=None):
    raise ModelUpdateError("the stub's fit ended")
