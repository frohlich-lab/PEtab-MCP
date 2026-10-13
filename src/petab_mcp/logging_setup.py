"""Package logging setup.

Attaches a file handler to the ``petab_mcp`` logger so a human-readable log
(``<store>/petab_mcp.log``) accompanies the structured provenance/error logs, and routes
pyPESTO's own optimizer logging to a separate ``<store>/pypesto.log`` so the (verbose) fit
logs are captured without cluttering the main log or the server's stderr. Idempotent:
calling either repeatedly does not add duplicate handlers.

One process serves many sessions, so each logger group has ONE routing handler that writes a record
into the store of the session it was logged for. The caller chooses that store: code running on a
session's behalf (a tool call, the session's construction) runs inside :func:`use_store`, and a
record logged outside any such scope goes to the store last passed to :func:`configure` /
:func:`configure_pypesto`, which is all single-session use needs. The store is a
:class:`~contextvars.ContextVar`, not a global, because sessions log at the same time from different
threads; FastMCP runs every sync tool call in its own copy of the context
(``anyio.to_thread.run_sync``), so even a plain :func:`use_store` inside a call is scoped to that
call. A forked pyPESTO worker (Linux) inherits the handler and the variable from the tool thread
that forked it; a spawned one (macOS) inherits neither and logs nothing to file.

The open files are a small LRU guarded by the routing handler's own lock, which ``logging``
re-initialises in a forked child. Do not add a :class:`threading.Lock` here (see
:class:`_RateLimitFilter`).

It also RATE-LIMITS AMICI's own warnings (:func:`configure_amici`). AMICI logs one warning per
non-finite value it finds, and the solver re-evaluates a failing right-hand side many times
before giving the step up, so one badly conditioned start emits the same line repeatedly.
"""

from __future__ import annotations

import contextlib
import logging
import os
from collections import OrderedDict
from contextvars import ContextVar
from pathlib import Path

_LOGGER_NAME = "petab_mcp"
_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"

#: Files each routing handler keeps open, across all sessions; past this the least recently written
#: one is closed, and reopened for appending on its next record. The cap is for pytest, which builds
#: hundreds of sessions in one process; 64 is more sessions than one server hosts at once.
_MAX_OPEN = 64

#: The store this context logs to (see :func:`use_store`); ``None`` means the configured default.
_STORE_ROOT: ContextVar[Path | None] = ContextVar("petab_mcp_store_root", default=None)


class _StoreRouter(logging.Handler):
    """Writes each record to ``<store>/<filename>``: the store :data:`_STORE_ROOT` names, else
    ``default_root``, the store this logger group was last configured for.

    ``_files`` holds the open :class:`logging.FileHandler` per store, least recently written first,
    and is touched only under ``self.lock`` (``Handler.handle`` holds it around ``emit``).
    """

    def __init__(self, filename: str) -> None:
        super().__init__()
        self.setFormatter(logging.Formatter(_FORMAT))
        self.filename = filename
        self.default_root: Path | None = None
        self._files: OrderedDict[Path, logging.FileHandler] = OrderedDict()

    def _file(self, root: Path) -> logging.FileHandler:
        handler = self._files.get(root)
        if handler is not None:
            self._files.move_to_end(root)
            return handler
        root.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(root / self.filename, encoding="utf-8")
        handler.setFormatter(self.formatter)
        self._files[root] = handler
        while len(self._files) > _MAX_OPEN:
            self._files.popitem(last=False)[1].close()
        return handler

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._file(_STORE_ROOT.get() or self.default_root).emit(record)  # flushes the record
        except Exception:  # noqa: BLE001 - a log call must never raise into the code that made it
            self.handleError(record)

    def install(
        self, logger_names: tuple[str, ...], store_root: str | Path, level: int, propagate: bool
    ) -> None:
        """Make ``store_root`` the default store, creating its file now, and install this handler on
        ``logger_names`` (``addHandler`` ignores a handler that is already there)."""
        with self.lock:
            self.default_root = Path(store_root).resolve()
            self._file(self.default_root)
        for name in logger_names:
            logger = logging.getLogger(name)
            logger.setLevel(level)
            logger.addHandler(self)
            logger.propagate = propagate


_PETAB_LOG = _StoreRouter("petab_mcp.log")
_PYPESTO_LOG = _StoreRouter("pypesto.log")


def use_store(store_root: str | Path) -> contextlib.ExitStack:
    """Route this context's records to the log files under ``store_root``.

    A plain call holds for the rest of the current context (inside a sync tool call: that call);
    ``with use_store(root):`` puts the previous store back on exit.
    """
    scope = contextlib.ExitStack()
    scope.callback(_STORE_ROOT.reset, _STORE_ROOT.set(Path(store_root).resolve()))
    return scope


def configure(store_root: str | Path, level: int = logging.INFO) -> logging.Logger:
    """Make ``<store_root>/petab_mcp.log`` where the ``petab_mcp`` logger writes outside a
    :func:`use_store` scope.

    petab's warnings (its lint logs through ``petab.v2.lint``, which has no handler of its own) and
    Python warnings land there too: otherwise both print to the server's stderr, where no session's
    log has them.
    """
    _PETAB_LOG.install((_LOGGER_NAME,), store_root, level, propagate=True)
    logging.captureWarnings(True)
    _PETAB_LOG.install(("petab", "py.warnings"), store_root, logging.WARNING, propagate=False)
    return logging.getLogger(_LOGGER_NAME)


def configure_pypesto(store_root: str | Path, level: int = logging.INFO) -> logging.Logger:
    """Route the fitting optimizer's logging to ``<store_root>/pypesto.log`` (outside a
    :func:`use_store` scope).

    The fit runs pyPESTO with the Fides optimizer; both log through their own loggers
    (``pypesto`` and ``fides``). We attach a dedicated file handler and stop propagation so
    that (verbose) optimizer output lands only in ``pypesto.log`` — not in ``petab_mcp.log``
    or the server's stderr. The separate file is the place for the detailed per-start /
    per-iteration fit log without cluttering anything else.

    Stopping propagation is not enough on its own: importing pyPESTO attaches its own stderr
    handler to ``pypesto`` (``pypesto/__init__.py`` calls ``pypesto.logging.log()``), which copies
    every record onto the server's stderr. So pyPESTO is imported here and that handler dropped.
    Importing pyPESTO imports AMICI, whose logger must be ours first (see :func:`configure_amici`).

    INFO, not DEBUG: pyPESTO's DEBUG logs every objective evaluation (three lines per dataset), and
    forked workers would write all of it into the fit log the agent reads.
    """
    configure_amici()
    import pypesto  # noqa: F401 - runs pyPESTO's import-time handler setup now, so it can be undone

    _PYPESTO_LOG.install(("pypesto", "fides"), store_root, level, propagate=False)
    logger = logging.getLogger("pypesto")
    for handler in [h for h in logger.handlers if h is not _PYPESTO_LOG]:
        logger.removeHandler(handler)
    return logger


def pypesto_log_path(store_root: str | Path) -> Path:
    """Path of the pyPESTO/Fides optimizer log under ``store_root`` (see :func:`configure_pypesto`)."""
    return Path(store_root) / "pypesto.log"


def read_pypesto_log(store_root: str | Path, tail_lines: int) -> dict:
    """Read the pyPESTO optimizer log, returning ``{path, exists, n_lines_total, tail_lines, log}``.

    ``tail_lines`` of ``0`` returns the whole file; a positive N the last N lines. Backs the
    ``get_fitting_log`` tool (which validates ``tail_lines`` first)."""
    path = pypesto_log_path(store_root)
    if not path.exists():
        return {
            "path": str(path),
            "exists": False,
            "n_lines_total": 0,
            "tail_lines": tail_lines,
            "log": "",
        }
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    kept = lines if tail_lines == 0 else lines[-tail_lines:]
    return {
        "path": str(path),
        "exists": True,
        "n_lines_total": len(lines),
        "tail_lines": tail_lines,
        "log": "\n".join(kept),
    }


#: Records let through per wall-clock second before the rest of that second is dropped.
_AMICI_PER_SECOND = 4
#: AMICI's logging format
_AMICI_FORMAT = "%(asctime)s.%(msecs).3d - %(name)s - %(levelname)s - %(message)s"
_AMICI_DATEFMT = "%Y-%m-%d %H:%M:%S"

_amici_handler: logging.Handler | None = None


class _RateLimitFilter(logging.Filter):
    """Let through at most :data:`_AMICI_PER_SECOND` records per wall-clock second, drop the rest.

    A solver re-evaluating a failing right-hand side can emit the same warning millions of
    times a second; this caps the cost of that per second rather than per distinct message, so
    it doesn't matter whether the flood is one message or many.

    DELIBERATELY UNLOCKED. The counter is a plain int mutated without synchronisation: reads and
    writes are individually atomic under the GIL, so the worst a race costs is an occasionally
    miscounted record, which a rate limiter does not care about. A :class:`threading.Lock` here
    would be held across ``fork`` — pyPESTO's ``MultiProcessEngine`` forks its workers out of a
    threaded server — and a lock that happened to be held at fork time is held forever in the
    child, deadlocking that worker on its first AMICI warning.
    """

    def __init__(self) -> None:
        super().__init__()
        self._second = -1
        self._count = 0

    def filter(self, record: logging.LogRecord) -> bool:
        second = int(record.created)
        if second != self._second:
            self._second = second
            self._count = 0
        self._count += 1
        return self._count <= _AMICI_PER_SECOND


def _amici_level() -> int:
    """``AMICI_LOG`` env as a level, or ``WARNING``.

    A bad value is IGNORED rather than fatal, so that a bad log level doesn't prevent
    the server from starting.
    """
    raw = os.environ.get("AMICI_LOG")
    if raw is None:
        return logging.WARNING
    try:
        return int(raw)
    except ValueError:
        pass
    named = logging.getLevelNamesMapping().get(raw.strip().upper())
    return named if named is not None else logging.WARNING


def configure_amici(level: int | None = None) -> logging.Logger:
    """Install a rate-limited stderr handler on the ``amici`` logger.

    ``level`` defaults to :func:`_amici_level` — i.e. to ``AMICI_LOG``, the env var AMICI itself
    honours.
    """
    global _amici_handler
    logger = logging.getLogger("amici")
    effective_level = _amici_level() if level is None else level
    logger.setLevel(effective_level)
    if _amici_handler is not None:
        _amici_handler.setLevel(effective_level)
        return logger
    handler = logging.StreamHandler()
    handler.setLevel(effective_level)
    handler.setFormatter(logging.Formatter(_AMICI_FORMAT, datefmt=_AMICI_DATEFMT))
    handler.addFilter(_RateLimitFilter())
    logger.addHandler(handler)
    logger.propagate = False
    _amici_handler = handler
    return logger
