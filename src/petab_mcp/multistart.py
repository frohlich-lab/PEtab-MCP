"""Wait for a fit's starts, and end a fit whose starts stopped finishing.

pyPESTO hands a multistart's starts to an engine and blocks until every one returns. Fides' per-start
cap (``fitting.FIT_PER_START_TIMEOUT_S``, its ``MAXTIME``) is checked between a start's iterations,
inside the worker, so it bounds a start only while that start keeps iterating: a worker stuck in one
simulation or on a lock it inherited when it was forked, or a pool that stops handing out starts, is
never checked against it, and pyPESTO waits on its pool with no timeout. One fit held its session
for 9 h after its starts went quiet that way, and every later call queued behind it (PEtab-MCP#108).
This module owns that wait, for both pools a fit runs on:

* **A deadline.** :class:`ProcessEngine`, the local pool, fails a fit whose starts are not all done
  :data:`FIT_DEADLINE_MARGIN_S` after the waves their caps allow (``⌈starts / workers⌉ × cap``). On
  a cluster (:mod:`petab_mcp.dask_engine`) a batch may wait behind other sessions' for any time, so
  there each batch has the deadline its own starts allow, counted from when it began running.
  Either way the workers are stopped and the fit fails with the ``fit_wedged``
  :class:`~petab_mcp.errors.InfrastructureError`: nothing is scored, the edit is undone, and the
  identical-fit guard does not count it.
* **Progress.** Every :data:`PROGRESS_INTERVAL_S` the count of starts done goes to the session's
  log, so a fit that hangs can be located.
* **The client's cancel.** A call its client cancels (MCP ``notifications/cancelled``) sets the
  event in :data:`CANCELLED` (``server._Cancellation``), and its fit stops with
  :class:`~petab_mcp.errors.FitCancelledError`.

Deliberately NOT a partial result. A wedged fit is rare (one in some 400 fits of the sweep that found
it), and scoring the starts that did finish would record a version whose multistart is smaller than
the one it names, which the de-dup would then take as standing for a full refit, and which may lack
the warm start from the parent's optimum that keeps a nested edit from scoring worse than its parent
(``fitting.WARM_START_FRACTION``). Failing costs one refit.

A sequential fit (one worker) runs in the tool's own thread, and nothing here bounds it.
"""

from __future__ import annotations

import concurrent.futures
import logging
import math
import multiprocessing
import threading
import time
from collections.abc import Callable, Sequence
from contextvars import ContextVar
from typing import Any

import cloudpickle
from pypesto.engine import Engine, Task

from .errors import FIT_WEDGED, FitCancelledError, InfrastructureError

#: Seconds a fit's starts get beyond what their per-start caps allow before the fit is wedged: a
#: start overruns its cap by up to one optimizer iteration, since Fides checks the cap between
#: iterations. It keeps the deadline inside the budget a client times the call out after (gemot's
#: ``fit_budget``: a 600 s compile margin plus the waves, times 1.4), so the server ends a wedged
#: fit before its client gives up on the call.
FIT_DEADLINE_MARGIN_S = 600.0

#: Seconds between two progress lines in the session's log.
PROGRESS_INTERVAL_S = 300.0

#: Seconds between two looks at a fit's starts, for its deadline and its client's cancel.
POLL_S = 5.0

#: The current tool call's cancel event, which the server sets when the client cancels the call. A
#: :class:`~contextvars.ContextVar` because FastMCP runs a sync tool in a copy of the caller's context.
CANCELLED: ContextVar[threading.Event | None] = ContextVar("petab_mcp_cancelled", default=None)

_log = logging.getLogger(__name__)


def watch(
    futures: Sequence[Any],
    sizes: Sequence[int],
    wait: Callable[[list, float], Any],
    overdue: Callable[[float], str | None],
) -> None:
    """Return once every one of ``futures`` is done, ``futures[i]`` running ``sizes[i]`` starts.

    ``wait(pending, timeout)`` blocks until some of ``pending`` is done or ``timeout`` passes, and
    ``overdue(seconds)`` says why the fit is wedged that many seconds after this began, or ``None``.
    Raises the ``fit_wedged`` error when it says so and :class:`FitCancelledError` when the call's
    client cancels it; the caller stops the workers.
    """
    cancelled = CANCELLED.get()
    total = sum(sizes)
    began = logged = time.monotonic()
    while left := [(f, n) for f, n in zip(futures, sizes) if not f.done()]:
        now = time.monotonic()
        done = total - sum(n for _, n in left)
        if cancelled is not None and cancelled.is_set():
            raise FitCancelledError(
                f"the client cancelled the call: {done} of {total} starts had finished, the rest "
                "were stopped and nothing was scored"
            )
        if (why := overdue(now - began)) is not None:
            raise InfrastructureError(
                FIT_WEDGED,
                f"{why}; {done} of {total} starts had finished, the rest were stopped and nothing "
                "was scored",
            )
        if now - logged >= PROGRESS_INTERVAL_S:
            _log.info("fit: %d/%d starts done after %.0fs", done, total, now - began)
            logged = now
        wait([f for f, _ in left], POLL_S)


def _work(payload: bytes) -> Any:
    """Run one cloudpickled pyPESTO task in a worker process."""
    return cloudpickle.loads(payload).execute()


class ProcessEngine(Engine):
    """pyPESTO's ``MultiProcessEngine`` under :func:`watch`: up to ``n_procs`` worker processes,
    each start capped at ``start_seconds``, every worker killed when the fit fails.

    Not pyPESTO's own: it waits on its pool with no timeout, and leaving its ``with`` joins every
    worker, a hung one included, so it cannot be bounded from outside.
    """

    def __init__(self, n_procs: int, start_seconds: float) -> None:
        super().__init__()
        self.n_procs = n_procs
        self.start_seconds = start_seconds

    def execute(self, tasks: list[Task], progress_bar: bool | None = None) -> list[Any]:
        """The tasks' results, in task order."""
        n_procs = min(self.n_procs, len(tasks))
        waves = math.ceil(len(tasks) / n_procs)
        deadline = waves * self.start_seconds + FIT_DEADLINE_MARGIN_S

        def overdue(seconds: float) -> str | None:
            if seconds <= deadline:
                return None
            return (
                f"the fit's starts were not done {deadline:.0f}s after they began ({waves} waves "
                f"of the {self.start_seconds:.0f}s per-start cap, plus {FIT_DEADLINE_MARGIN_S:.0f}s)"
            )

        # The platform's start method, as pyPESTO's: a forked worker (Linux) inherits the session's
        # log routing (see logging_setup).
        pool = concurrent.futures.ProcessPoolExecutor(
            n_procs, mp_context=multiprocessing.get_context()
        )
        futures = [pool.submit(_work, cloudpickle.dumps(task)) for task in tasks]
        try:
            watch(futures, [1] * len(futures), concurrent.futures.wait, overdue)
        except BaseException:
            for process in list(pool._processes.values()):  # Python 3.14: pool.kill_workers()
                process.kill()
            pool.shutdown(wait=False, cancel_futures=True)
            raise
        pool.shutdown()
        return [future.result() for future in futures]
