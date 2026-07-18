"""Package logging setup.

Attaches a file handler to the ``petab_mcp`` logger so a human-readable log
(``<store>/petab_mcp.log``) accompanies the structured provenance/error logs, and routes
pyPESTO's own optimizer logging to a separate ``<store>/pypesto.log`` so the (verbose) fit
logs are captured without cluttering the main log or the server's stderr. Idempotent:
calling either repeatedly for the same directory does not add duplicate handlers.
"""

from __future__ import annotations

import logging
from pathlib import Path

_LOGGER_NAME = "petab_mcp"
_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"

# The single active file handler installed on each logger group. These loggers are
# process-global, so we keep ONE handler per group and swap it when the store dir changes
# (rather than accumulating a handler — and an open fd — per session/directory).
_petab_handler: logging.FileHandler | None = None
_optimizer_handler: logging.FileHandler | None = None


def get_logger() -> logging.Logger:
    return logging.getLogger(_LOGGER_NAME)


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
        return {"path": str(path), "exists": False, "n_lines_total": 0,
                "tail_lines": tail_lines, "log": ""}
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    kept = lines if tail_lines == 0 else lines[-tail_lines:]
    return {"path": str(path), "exists": True, "n_lines_total": len(lines),
            "tail_lines": tail_lines, "log": "\n".join(kept)}
