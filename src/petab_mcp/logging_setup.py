"""Package logging setup.

Attaches a file handler to the ``petab_mcp`` logger so a human-readable log
(``<store>/petab_mcp.log``) accompanies the structured provenance/error logs, and routes
pyPESTO's own optimizer logging to a separate ``<store>/pypesto.log`` so the (verbose) fit
logs are captured without cluttering the main log or the server's stderr. Idempotent:
calling either repeatedly for the same directory does not add duplicate handlers.

It also RATE-LIMITS AMICI's own warnings (:func:`configure_amici`). AMICI logs one warning per
non-finite value it finds, and the solver re-evaluates a failing right-hand side many times
before giving the step up, so one badly conditioned start emits the same line repeatedly.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

_LOGGER_NAME = "petab_mcp"
_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"

# The single active file handler installed on each logger group. These loggers are
# process-global, so we keep ONE handler per group and swap it when the store dir changes
# (rather than accumulating a handler — and an open fd — per session/directory).
_petab_handler: logging.FileHandler | None = None
_optimizer_handler: logging.FileHandler | None = None


def _swap_file_handler(
    current: logging.FileHandler | None,
    logger_names: tuple[str, ...],
    log_path: str,
    level: int,
    propagate: bool,
) -> logging.FileHandler:
    """Install a single file handler at ``log_path`` on ``logger_names``, replacing any
    previously installed one so repeated calls (with different dirs) don't accumulate
    handlers or leak file descriptors."""
    for name in logger_names:
        logging.getLogger(name).setLevel(level)
    if current is not None and getattr(current, "baseFilename", None) == log_path:
        return current  # already installed for this path

    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setFormatter(logging.Formatter(_FORMAT))
    for name in logger_names:
        lg = logging.getLogger(name)
        if current is not None and current in lg.handlers:
            lg.removeHandler(current)
        lg.addHandler(handler)
        lg.propagate = propagate
    if current is not None:
        current.close()
    return handler


def configure(store_root: str | Path, level: int = logging.INFO) -> logging.Logger:
    """Ensure the ``petab_mcp`` logger writes to ``<store_root>/petab_mcp.log``."""
    global _petab_handler
    root = Path(store_root)
    root.mkdir(parents=True, exist_ok=True)
    log_path = str((root / "petab_mcp.log").resolve())
    _petab_handler = _swap_file_handler(_petab_handler, (_LOGGER_NAME,), log_path, level, True)
    return logging.getLogger(_LOGGER_NAME)


def configure_pypesto(store_root: str | Path, level: int = logging.DEBUG) -> logging.Logger:
    """Route the fitting optimizer's logging to ``<store_root>/pypesto.log``.

    The fit runs pyPESTO with the Fides optimizer; both log through their own loggers
    (``pypesto`` and ``fides``). We attach a dedicated file handler and stop propagation so
    that (verbose) optimizer output lands only in ``pypesto.log`` — not in ``petab_mcp.log``
    or the server's stderr. The separate file is the place for the detailed per-start /
    per-iteration fit log without cluttering anything else.
    """
    global _optimizer_handler
    root = Path(store_root)
    root.mkdir(parents=True, exist_ok=True)
    log_path = str((root / "pypesto.log").resolve())
    _optimizer_handler = _swap_file_handler(
        _optimizer_handler, ("pypesto", "fides"), log_path, level, False
    )
    return logging.getLogger("pypesto")


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
