"""Run a fit's multistart on the server's dask cluster, one worker pool for every session.

Without a cluster, each fit starts up to ``FIT_N_WORKERS`` local processes, so sessions fitting at
once oversubscribe the node, and Fides' per-start time cap then stops slowed starts early:
contention changes results, not only speed. With one, :func:`petab_mcp.server.create_server` builds
one :class:`DaskEngine`, and every fit and held-out refit of that server runs its starts there.

* **Starts go in batches.** A task per start would flood the scheduler, which runs in the server's
  process, with second-long tasks, so a fit sends ``k`` starts per task, run one after another on
  one worker (:meth:`DaskEngine.batched`). Each is unpickled on its own there, so a start's result
  does not depend on the batch it ran in.
* **Fits take turns.** Batch ``j`` is submitted at ``priority=-j``; dask runs higher priorities
  first and equal ones in submission order, so concurrent fits interleave batch by batch.
* **Logs follow the session.** A start's ``pypesto``/``fides`` records come back with its result
  and are re-emitted in the calling thread, so :mod:`petab_mcp.logging_setup` files them in that
  session's ``pypesto.log``.
* **A cluster failure fails the fit** with :class:`~petab_mcp.errors.FitBackendError`, and the
  starts never rerun on the server node (see ``fitting._run_fit``).
* **A batch has a deadline**: its starts' caps plus
  :data:`~petab_mcp.multistart.FIT_DEADLINE_MARGIN_S` from when it began running, as the scheduler
  hears from its worker's heartbeats. One past it is cancelled and its starts are scored as failed
  (:class:`petab_mcp.multistart.CappedEngine`); its worker runs on until it retires. Not from when
  it was submitted: it may wait behind other sessions' batches, or for the pool's jobs to start,
  for any time.
* **Workers never build.** Unpickling a start imports its compiled module from the server's build
  root at the same absolute path. pyPESTO's creator and AMICI's importer would instead compile it
  there, inside the build root every session shares, so :class:`_NoBuild` makes them raise.

The only module that imports ``distributed`` (the ``dask`` extra); a server without a cluster never
imports it.
"""

from __future__ import annotations

import contextlib
import copy
import itertools
import logging
import math
import socket
import threading
from collections.abc import Callable
from typing import Any

import cloudpickle
from distributed import Client, WorkerPlugin, wait
from pypesto.engine import Task

from .errors import FitBackendError, PetabMcpError
from .multistart import FIT_DEADLINE_MARGIN_S, CappedEngine, watch

#: The loggers whose records a start carries back, the ones ``configure_pypesto`` routes.
_LOGGERS = ("pypesto", "fides")


class _Records(logging.Handler):
    """Collects what the current thread logs as ``(name, levelno, message, created)``. A record's
    ``args`` need not pickle, its message does."""

    def __init__(self) -> None:
        super().__init__(logging.DEBUG)
        self.thread = threading.get_ident()
        self.records: list[tuple[str, int, str, float]] = []

    def emit(self, record: logging.LogRecord) -> None:
        # Another thread of the same worker may be running another session's start.
        if record.thread != self.thread:
            return
        try:
            self.records.append((record.name, record.levelno, record.getMessage(), record.created))
        except Exception:  # noqa: BLE001 - a log call must never raise into the code that made it
            self.handleError(record)


def _work(payloads: list[bytes]) -> tuple[list[Any], list[tuple[str, int, str, float]]]:
    """Run cloudpickled pyPESTO tasks one after another on a worker; returns their results and
    their records."""
    capture = _Records()
    loggers = [logging.getLogger(name) for name in _LOGGERS]
    for logger in loggers:
        logger.addHandler(capture)
    try:
        return [cloudpickle.loads(payload).execute() for payload in payloads], capture.records
    finally:
        for logger in loggers:
            logger.removeHandler(capture)


def _executing(dask_scheduler) -> dict[str, float]:
    """For each task running on the cluster, the seconds it has run, by key (run on the
    scheduler)."""
    return {
        ts.key: seconds
        for ws in dask_scheduler.workers.values()
        for ts, seconds in ws.executing.items()
    }


def _refuse_build(builder, *args, **kwargs):
    """Stands in for a model build on a worker; ``builder`` is the pyPESTO creator or AMICI
    importer that was about to compile."""
    path = getattr(builder, "output_folder", None) or builder.output_dir
    raise FitBackendError(
        f"the AMICI module at {path} is not importable on worker {socket.gethostname()}: the "
        "build root must be on a filesystem the workers share, with an identical Python env"
    )


class _NoBuild(WorkerPlugin):
    """Replaces pyPESTO's and AMICI's model-build entry points with :func:`_refuse_build`, in each
    worker process."""

    def setup(self, worker) -> None:
        from amici.importers.petab import PetabImporter
        from pypesto.petab.objective_creator import (
            AmiciObjectiveCreator,
            AmiciPetabV2ObjectiveCreator,
        )

        AmiciObjectiveCreator.compile_model = _refuse_build
        AmiciPetabV2ObjectiveCreator.compile_model = _refuse_build
        PetabImporter._do_import_sbml = _refuse_build


class DaskEngine(CappedEngine):
    """A pyPESTO engine that runs tasks on ``cluster``, ``starts_per_task`` to a dask task, through
    a client of its own.

    Refused before connecting when AMICI was built without HDF5: AMICI pickles a solver's settings
    through HDF5, so no start could reach a worker and every fit on the cluster would fail.
    """

    #: Starts per dask task; a fit sets its own, and their cap, through :meth:`batched`.
    starts_per_task = 1

    def __init__(self, cluster) -> None:
        from amici.sim.sundials import hdf5_enabled

        if not hdf5_enabled:
            raise FitBackendError(
                "this AMICI was built without HDF5, which it needs to pickle a start for a "
                "worker: rebuild AMICI with the HDF5 headers installed to fit on a cluster"
            )
        super().__init__()
        self.client = Client(cluster)
        self.client.register_plugin(_NoBuild(), name="petab-mcp-no-build")

    def batched(
        self, k: int, start_seconds: float = math.inf, rerun: frozenset[str] = frozenset()
    ) -> DaskEngine:
        """This engine with ``k`` starts per dask task, each capped at ``start_seconds`` (by default
        uncapped, so no batch is stopped), rerunning ``rerun`` once when stopped: a copy, as one
        engine serves every session's fits at once."""
        engine = copy.copy(self)
        engine.starts_per_task = k
        engine.start_seconds = start_seconds
        engine.rerun = rerun
        return engine

    def _wait(self, pending: list, timeout: float) -> None:
        """Block until one of ``pending`` finishes or ``timeout`` passes; raise a failed start's
        error at once, so the fit's other batches are cancelled instead of run."""
        with contextlib.suppress(TimeoutError):
            wait(pending, timeout=timeout, return_when="FIRST_COMPLETED")
        for future in pending:
            if future.status == "error":
                future.result()

    def _overrun(self, futures: list, sizes: list[int]) -> Callable[[list[int], float], set[int]]:
        """Names the batches of ``futures`` still pending that have run longer than their starts'
        caps allow."""

        def overrun(pending: list[int], _) -> set[int]:
            if math.isinf(self.start_seconds):
                return set()
            ran = self.client.run_on_scheduler(_executing)
            return {
                i
                for i in pending
                if ran.get(futures[i].key, 0.0)
                > sizes[i] * self.start_seconds + FIT_DEADLINE_MARGIN_S
            }

        return overrun

    def _run(self, tasks: list[Task]) -> tuple[list[Any], set[int]]:
        """A failure cancels the starts still queued, so a failed fit leaves nothing queued on the
        pool (a batch stopped past its caps keeps its worker until that worker retires)."""
        futures, batches = [], list(itertools.batched(range(len(tasks)), self.starts_per_task))
        sizes = [len(batch) for batch in batches]
        try:
            for j, batch in enumerate(batches):
                payloads = [cloudpickle.dumps(tasks[i]) for i in batch]
                futures.append(self.client.submit(_work, payloads, priority=-j, pure=False))
            stopped = watch(futures, sizes, self._wait, self._overrun(futures, sizes))
            if stopped:
                self.client.cancel([futures[j] for j in stopped])
            done = self.client.gather([f for j, f in enumerate(futures) if j not in stopped])
        except Exception as exc:  # noqa: BLE001 - every cluster failure fails the fit
            with contextlib.suppress(Exception):  # the client itself may be what failed
                self.client.cancel(futures)
            if isinstance(exc, PetabMcpError):  # ours: a worker's refusal, a wedge, a cancel
                raise
            raise FitBackendError(f"the fit's dask cluster failed it: {exc!r}") from exc
        for _, records in done:
            for name, levelno, message, created in records:
                record = logging.makeLogRecord(
                    {
                        "name": name,
                        "levelno": levelno,
                        "levelname": logging.getLevelName(levelno),
                        "msg": message,
                        "created": created,
                        "msecs": created % 1 * 1000,
                    }
                )
                logging.getLogger(name).handle(record)
        results = [None] * len(tasks)
        kept = [batch for j, batch in enumerate(batches) if j not in stopped]
        for batch, (out, _) in zip(kept, done, strict=True):
            for i, result in zip(batch, out, strict=True):
                results[i] = result
        return results, {i for j in stopped for i in batches[j]}
