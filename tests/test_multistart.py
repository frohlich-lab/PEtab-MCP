"""A fit ends (``petab_mcp.multistart``): one whose starts stop finishing fails by its deadline, a
change arriving meanwhile is refused at once instead of queued behind it, and a client's cancel
stops it. One fit's starts went quiet after 40 minutes and held its session for 9 h, while its
client resent the same change three times behind it (PEtab-MCP#108)."""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextvars
import functools
import logging
import multiprocessing
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import anyio
import pytest

from petab_mcp import fitting
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
    return SimpleNamespace(execute=functools.partial(time.sleep, 3600))


class _Pool:
    """A ``ProcessPoolExecutor`` that runs each start in a thread of this process and records the
    workers it is told to kill: a hung worker without a process."""

    def __init__(self, n_procs, mp_context=None):
        self.killed = []
        self._processes = {
            i: SimpleNamespace(kill=functools.partial(self.killed.append, i))
            for i in range(n_procs)
        }
        _Pool.last = self

    def submit(self, fn, payload):
        future = concurrent.futures.Future()

        def run():
            future.set_result(fn(payload))

        threading.Thread(target=run, daemon=True).start()
        return future

    def shutdown(self, wait=True, cancel_futures=False):
        pass


def test_a_fit_whose_start_never_returns_fails_at_its_deadline_and_kills_its_workers(monkeypatch):
    """Three starts on two workers are two waves of the per-start cap: past those and the margin,
    the fit fails as wedged, says how far it got, and kills its workers, instead of waiting."""
    monkeypatch.setattr(M.concurrent.futures, "ProcessPoolExecutor", _Pool)
    monkeypatch.setattr(M, "FIT_DEADLINE_MARGIN_S", 0.2)
    monkeypatch.setattr(M, "POLL_S", 0.02)
    began = time.monotonic()
    with pytest.raises(InfrastructureError) as wedged:
        M.ProcessEngine(2, start_seconds=0.1).execute([_start(1), _hang(), _start(3)])
    took = time.monotonic() - began

    assert wedged.value.cause == FIT_WEDGED
    assert str(wedged.value).startswith("infrastructure_error: fit_wedged: ")
    assert "after they began (2 waves of the" in str(wedged.value)
    assert "2 of 3 starts had finished" in str(wedged.value)
    assert 2 * 0.1 + 0.2 <= took < 2, took
    assert _Pool.last.killed == [0, 1]


def test_a_fit_within_its_deadline_returns_its_starts_in_task_order(monkeypatch):
    """The starts of a fit that ends in time come back in the order they were given, through real
    worker processes."""
    monkeypatch.setattr(M, "POLL_S", 0.02)
    starts = [_start(-i) for i in range(5)]
    assert M.ProcessEngine(2, start_seconds=60).execute(starts) == [0, 1, 2, 3, 4]


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
        M.watch(futures, [2, 1, 1], concurrent.futures.wait, lambda _: None)

    assert "fit: 2/4 starts done after 0s" in caplog.messages


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


def _fit_ends(self, n_starts=None):
    raise ModelUpdateError("the stub's fit ended")
