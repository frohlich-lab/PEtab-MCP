"""Run a fit's multistart on the server's dask cluster, one worker pool for every session.

Without a cluster, each fit starts its own local pyPESTO ``MultiProcessEngine`` of up to
``FIT_N_WORKERS`` processes, so sessions fitting at once oversubscribe the node. Fides caps each
start's wall time, so a start slowed by that contention stops early: contention changes results,
not only speed. With a cluster, :func:`petab_mcp.server.create_server` builds one
:class:`DaskEngine`, and every fit and held-out refit of that server runs its starts there.

* **Fits take turns.** Start ``i`` of a fit is submitted at ``priority=-i``. dask runs higher
  priorities first and equal ones in submission order, so concurrent fits interleave start by
  start without a queue of our own.
* **Logs follow the session.** A start's ``pypesto``/``fides`` records come back with its result.
  They are re-emitted in the calling thread, where :mod:`petab_mcp.logging_setup` files them in
  that session's ``pypesto.log``.
* **A cluster failure fails the fit.** A lost scheduler, a killed worker, a cancelled start or a
  task that will not run on a worker raises :class:`~petab_mcp.errors.FitBackendError`, and the
  edit reverts. The starts never rerun on the server node (see ``fitting._run_fit``).
* **Workers never build.** A start's AMICI model reaches a worker by import only: unpickling
  imports the compiled module from the server's build root at the same absolute path and checks
  its checksum. The builder paths, pyPESTO's creator and AMICI's PEtab importer, would instead
  compile the module there, inside the build root every session shares (pyPESTO's after an
  ``rmtree`` of its folder), so :class:`_NoBuild` makes them raise in each worker process.

A separate module because it is the only one that imports ``distributed`` (the ``dask`` extra).
A server without a cluster never imports it.
"""

from __future__ import annotations

import logging
import socket
import threading
from typing import Any

import cloudpickle
from distributed import Client, WorkerPlugin
from pypesto.engine import Engine, Task

from .errors import FitBackendError

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


def _work(payload: bytes) -> tuple[Any, list[tuple[str, int, str, float]]]:
    """Run one cloudpickled pyPESTO task on a worker; returns its result and its records."""
    capture = _Records()
    loggers = [logging.getLogger(name) for name in _LOGGERS]
    for logger in loggers:
        logger.addHandler(capture)
    try:
        return cloudpickle.loads(payload).execute(), capture.records
    finally:
        for logger in loggers:
            logger.removeHandler(capture)


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


class DaskEngine(Engine):
    """A pyPESTO engine that runs each task on ``client``'s cluster."""

    def __init__(self, client: Client) -> None:
        super().__init__()
        self.client = client
        client.register_plugin(_NoBuild(), name="petab-mcp-no-build")

    def execute(self, tasks: list[Task], progress_bar: bool | None = None) -> list[Any]:
        """The tasks' results, in task order."""
        try:
            futures = [
                self.client.submit(_work, cloudpickle.dumps(task), priority=-i, pure=False)
                for i, task in enumerate(tasks)
            ]
            done = self.client.gather(futures)
        except FitBackendError:
            raise
        except Exception as exc:  # noqa: BLE001 - every cluster failure fails the fit
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
        return [result for result, _ in done]
