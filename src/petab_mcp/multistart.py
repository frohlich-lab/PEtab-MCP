"""Wait for a fit's starts: stop a start that runs past its cap, score the fit from the rest, and
end a fit whose starts stopped finishing.

pyPESTO hands a multistart's starts to an engine and blocks until every one returns. Fides' per-start
cap (the session's ``fit_per_start_timeout``, its ``MAXTIME``) is checked between a start's
iterations, inside the worker, so it bounds a start only while that start keeps iterating: a worker
stuck in one simulation or on a lock, or a pool that stops handing out starts, is never checked
against it, and pyPESTO waits on its pool with no timeout. One fit held its session for 9 h after
its starts went quiet that way, and every later call queued behind it (PEtab-MCP#108). This module
owns that wait, for both pools a fit runs on (:class:`CappedEngine`):

* **A start past its cap is stopped.** Locally (:class:`ProcessEngine`), once no start is left
  queued every start still running began by then, so one still running its cap plus
  :data:`FIT_DEADLINE_MARGIN_S` after that has overrun it, and so has every other start still
  running: the pool is killed and nothing is lost. On a cluster (:mod:`petab_mcp.dask_engine`) a
  batch still running its starts' caps plus the margin after it began is given up on, the starts
  of it that had finished included, since a batch returns its starts together. A stopped start is
  scored as a failed one, as pyPESTO records a start that raised, the fit is scored from the rest,
  and :attr:`CappedEngine.killed` names the stopped starts for the version's record.
* **The warm start from the parent's optimum is rerun.** A stopped start in
  :attr:`CappedEngine.rerun` (``fitting.parent_starts``, of which one keeps a nested edit's fit
  within reach of its parent's) runs once more.
* **A fit whose starts stopped finishing fails.** Locally, no start finishing for a start's cap plus
  the margin while others wait for a worker means every worker is stuck; a fit whose every start
  was stopped has nothing to score; and one whose rerun warm start is stopped again could score a
  nested edit worse than its parent. Each ends the fit with the ``fit_wedged``
  :class:`~petab_mcp.errors.InfrastructureError`: the workers are stopped, nothing is scored, the
  edit is undone, and the identical-fit guard does not count it.
* **Progress.** Every :data:`PROGRESS_INTERVAL_S` the count of starts done goes to the session's
  log, so a fit that hangs can be located, and the change's :class:`Progress` in :data:`PROGRESS`
  is kept current for a caller waiting on the fit (``wait_for_fit``).
* **The client's cancel.** A call its client cancels (MCP ``notifications/cancelled``) sets the
  event in :data:`CANCELLED` (``server._Cancellation``), and its fit stops with
  :class:`~petab_mcp.errors.FitCancelledError`.

A fit missing a stopped start is scored rather than failed. Failing the whole fit for one hung start
of 300 discarded the 299 that had finished, kept 99 of 100 workers idle for 2.7 h, and ended the
client's run with it (PEtab-MCP#118). What failing protected is kept otherwise: the version records
how many starts were stopped (``n_starts_killed``), the de-dup lets it stand only for a fit of no
more starts than it completed (:mod:`dedup`), and the warm start that keeps a nested edit from
scoring worse than its parent is rerun.

A sequential fit (one worker) runs in the tool's own thread, and nothing here bounds it.
"""

from __future__ import annotations

import concurrent.futures
import logging
import math
import multiprocessing
import threading
import time
from collections.abc import Callable, Iterable, Sequence
from contextvars import ContextVar
from typing import Any

import cloudpickle
import numpy as np
from pypesto import OptimizerResult
from pypesto.engine import Engine, Task

from . import logging_setup
from .errors import FIT_WEDGED, FitCancelledError, InfrastructureError

#: Seconds a start gets beyond its cap before it is stopped: a start overruns its cap by up to one
#: optimizer iteration, since Fides checks the cap between iterations, and a local start's clock
#: begins when it is handed out, before its worker has booted. It keeps a fit inside the budget a
#: client times the call out after (gemot's ``fit_budget``: a 600 s compile margin plus the waves,
#: times 1.4), so the server ends a stuck start before its client gives up on the call.
FIT_DEADLINE_MARGIN_S = 600.0

#: How the local pool starts its workers. Deliberately NOT ``fork``, Linux's default before Python
#: 3.14: a worker forked out of the threaded server inherits every lock another thread holds at that
#: moment, and on NEMO all 16 workers of two fits sat at 0 CPU until the fit failed as wedged, while
#: the same fit under ``forkserver`` ran. A worker started this way inherits no log routing, so
#: :func:`logging_setup.init_worker` sets it up again.
START_METHOD = "forkserver"

#: Seconds between two progress lines in the session's log.
PROGRESS_INTERVAL_S = 300.0

#: Seconds between two looks at a fit's starts, for their caps and the client's cancel.
POLL_S = 5.0

#: The current tool call's cancel event, which the server sets when the client cancels the call. A
#: :class:`~contextvars.ContextVar` because FastMCP runs a sync tool in a copy of the caller's context.
CANCELLED: ContextVar[threading.Event | None] = ContextVar("petab_mcp_cancelled", default=None)

_log = logging.getLogger(__name__)


class Progress:
    """A change's fit as :func:`watch` sees it: ``done`` of its ``total`` starts finished since the
    multistart ``began`` (``time.monotonic()``, ``None`` before it has), and ``best``, the lowest
    objective among them (``None`` until one is finite). A sequential fit is not watched, and leaves
    it empty."""

    def __init__(self) -> None:
        self.began: float | None = None
        self.total = 0
        self.done = 0
        self.best: float | None = None

    def remaining_s(self, now: float) -> float | None:
        """Seconds the starts left will take at the rate the finished ones took, ``None`` before
        one has finished."""
        if self.began is None or not self.done:
            return None
        return (now - self.began) * (self.total - self.done) / self.done


#: The current change's :class:`Progress`, which ``PetabSession.change`` sets for its fit to report
#: into. A :class:`~contextvars.ContextVar` for the reason :data:`CANCELLED` is one.
PROGRESS: ContextVar[Progress | None] = ContextVar("petab_mcp_progress", default=None)


def watch(
    futures: Sequence[Any],
    sizes: Sequence[int],
    wait: Callable[[list, float], Any],
    overrun: Callable[[list[int], float], Iterable[int]],
    quiet: float = math.inf,
    fvals: Callable[[Any], Iterable[float]] | None = None,
) -> set[int]:
    """Return once each of ``futures`` is done or overrun, ``futures[i]`` running ``sizes[i]``
    starts; returns the indices of the overrun ones.

    ``wait(pending, timeout)`` blocks until some of ``pending`` is done or ``timeout`` passes, and
    ``overrun(pending, seconds)`` names those of the ``pending`` indices that ran past their caps
    that many seconds after this began. Raises the ``fit_wedged`` error once no start has finished
    for ``quiet`` seconds and none is overrun, and :class:`FitCancelledError` when the call's client
    cancels it; the caller stops the workers. ``fvals(future)`` reads a finished future's
    objectives for :data:`PROGRESS`.
    """
    cancelled, progress = CANCELLED.get(), PROGRESS.get() or Progress()
    total = sum(sizes)
    began = logged = moved = time.monotonic()
    progress.began, progress.total, progress.done = began, total, 0
    finished, stopped, read = 0, set(), set()
    while left := [i for i, f in enumerate(futures) if i not in stopped and not f.done()]:
        now = time.monotonic()
        done = total - sum(sizes[i] for i in [*left, *stopped])
        if done > finished:
            finished, moved = done, now
            progress.done = done
            if fvals is not None:
                _read_best(progress, futures, fvals, read | stopped | set(left), read)
        if cancelled is not None and cancelled.is_set():
            raise FitCancelledError(
                f"the client cancelled the call: {done} of {total} starts had finished, the rest "
                "were stopped and nothing was scored"
            )
        if overdue := set(overrun(left, now - began)):
            stopped |= overdue
            continue
        if now - moved > quiet:
            raise InfrastructureError(
                FIT_WEDGED,
                f"no start had finished for {now - moved:.0f}s, longer than a start may run, while "
                f"others waited for a worker; {done} of {total} starts had finished, the rest were "
                "stopped and nothing was scored",
            )
        if now - logged >= PROGRESS_INTERVAL_S:
            _log.info("fit: %d/%d starts done after %.0fs", done, total, now - began)
            logged = now
        wait([futures[i] for i in left], POLL_S)
    progress.done = total - sum(sizes[i] for i in stopped)
    if fvals is not None:
        _read_best(progress, futures, fvals, read | stopped, read)
    return stopped


def _read_best(
    progress: Progress,
    futures: Sequence[Any],
    fvals: Callable[[Any], Iterable[float]],
    skip: set[int],
    read: set[int],
) -> None:
    """Lower ``progress.best`` to the finite objectives of the ``futures`` not in ``skip``, adding
    each one read to ``read``. A future that raised is passed over: the engine raises its error
    where it collects the results."""
    for i, future in enumerate(futures):
        if i in skip:
            continue
        read.add(i)
        try:
            values = [v for v in fvals(future) if math.isfinite(v)]
        except Exception:  # noqa: BLE001 - see the docstring
            continue
        if values:
            progress.best = min(values if progress.best is None else [progress.best, *values])


def _killed(task: Task, cap: float) -> OptimizerResult:
    """The failed result of ``task``, stopped past its cap of ``cap`` seconds: what pyPESTO records
    for a start that raised."""
    result = OptimizerResult(
        id=task.id,
        x0=task.x0,
        fval=np.inf,
        exitflag=-1,
        message=f"stopped by the server: still running past its {cap:.0f}s cap",
        optimizer=str(task.optimizer),
    )
    result.update_to_full(task.problem)
    return result


class CappedEngine(Engine):
    """A pyPESTO engine whose :meth:`_run` stops the starts that run past their cap of
    ``start_seconds`` (see the module docstring).

    :meth:`execute` scores a stopped start as failed, reruns one in :attr:`rerun` once, and fails
    the fit as wedged when a rerun start is stopped again or no start is left to score.
    """

    #: Each start's cap in seconds.
    start_seconds = math.inf
    #: The ids of the starts to rerun once when stopped.
    rerun: frozenset[str] = frozenset()
    #: The ids of the starts the last :meth:`execute` stopped and scored as failed.
    killed: list[str] = []

    def _run(self, tasks: list[Task]) -> tuple[list[Any], set[int]]:
        """``tasks``' results in task order, ``None`` for each one stopped, and the indices of
        those."""
        raise NotImplementedError

    def execute(self, tasks: list[Task], progress_bar: bool | None = None) -> list[Any]:
        """The tasks' results, in task order."""
        results, stopped = self._run(tasks)
        again = sorted(i for i in stopped if tasks[i].id in self.rerun)
        if again:
            ids = [tasks[i].id for i in again]
            _log.warning("fit: rerunning start(s) %s from the parent's optimum once", ids)
            rerun, still = self._run([tasks[i] for i in again])
            if still:
                raise InfrastructureError(
                    FIT_WEDGED,
                    f"start(s) {[ids[j] for j in sorted(still)]}, warm from the parent's optimum, "
                    f"ran past the {self.start_seconds:.0f}s per-start cap twice; nothing was "
                    "scored",
                )
            for i, result in zip(again, rerun):
                results[i] = result
            stopped -= set(again)
        if len(stopped) == len(tasks):
            raise InfrastructureError(
                FIT_WEDGED,
                f"all {len(tasks)} starts ran past the {self.start_seconds:.0f}s per-start cap and "
                "were stopped; nothing was scored",
            )
        for i in stopped:
            results[i] = _killed(tasks[i], self.start_seconds)
        self.killed = [tasks[i].id for i in sorted(stopped)]
        if stopped:
            _log.warning(
                "fit: %d of %d starts stopped past the %.0fs per-start cap: %s",
                len(stopped),
                len(tasks),
                self.start_seconds,
                self.killed,
            )
        return results


def _work(payload: bytes) -> Any:
    """Run one cloudpickled pyPESTO task in a worker process."""
    return cloudpickle.loads(payload).execute()


def _kill(pool: concurrent.futures.ProcessPoolExecutor) -> None:
    for process in list(pool._processes.values()):  # Python 3.14: pool.kill_workers()
        process.kill()
    pool.shutdown(wait=False, cancel_futures=True)


class ProcessEngine(CappedEngine):
    """pyPESTO's ``MultiProcessEngine`` under :func:`watch`: up to ``n_procs`` worker processes,
    each start capped at ``start_seconds``, every worker killed when the fit fails or a start is
    stopped.

    Not pyPESTO's own: it waits on its pool with no timeout, and leaving its ``with`` joins every
    worker, a hung one included, so it cannot be bounded from outside.
    """

    def __init__(
        self, n_procs: int, start_seconds: float, rerun: frozenset[str] = frozenset()
    ) -> None:
        super().__init__()
        self.n_procs = n_procs
        self.start_seconds = start_seconds
        self.rerun = rerun

    def _run(self, tasks: list[Task]) -> tuple[list[Any], set[int]]:
        n_procs = min(self.n_procs, len(tasks))
        budget = self.start_seconds + FIT_DEADLINE_MARGIN_S
        handed_out = None  # seconds into the wait at which no start was left queued

        def overrun(pending: list[int], seconds: float) -> list[int]:
            nonlocal handed_out
            if handed_out is None and len(pending) <= n_procs:
                handed_out = seconds
            return pending if handed_out is not None and seconds - handed_out > budget else []

        pool = concurrent.futures.ProcessPoolExecutor(
            n_procs,
            mp_context=multiprocessing.get_context(START_METHOD),
            initializer=logging_setup.init_worker,
            initargs=(logging_setup.worker_store(),),
        )
        futures = [pool.submit(_work, cloudpickle.dumps(task)) for task in tasks]
        try:
            stopped = watch(
                futures,
                [1] * len(futures),
                concurrent.futures.wait,
                overrun,
                budget,
                lambda future: [future.result().fval],
            )
        except BaseException:
            _kill(pool)
            raise
        if stopped:
            _kill(pool)
        else:
            pool.shutdown()
        return [None if i in stopped else f.result() for i, f in enumerate(futures)], stopped
