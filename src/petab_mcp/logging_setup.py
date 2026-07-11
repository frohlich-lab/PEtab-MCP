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
_configured_paths: set[str] = set()
_pypesto_configured_paths: set[str] = set()

_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def get_logger() -> logging.Logger:
    return logging.getLogger(_LOGGER_NAME)


def configure(store_root: str | Path, level: int = logging.INFO) -> logging.Logger:
    """Ensure the ``petab_mcp`` logger writes to ``<store_root>/petab_mcp.log``."""
    root = Path(store_root)
    root.mkdir(parents=True, exist_ok=True)
    log_path = str((root / "petab_mcp.log").resolve())

    logger = logging.getLogger(_LOGGER_NAME)
    logger.setLevel(level)
    if log_path not in _configured_paths:
        handler = logging.FileHandler(log_path, encoding="utf-8")
        handler.setFormatter(logging.Formatter(_FORMAT))
        logger.addHandler(handler)
        _configured_paths.add(log_path)
    return logger


def configure_pypesto(store_root: str | Path, level: int = logging.DEBUG) -> logging.Logger:
    """Route pyPESTO's own logging to ``<store_root>/pypesto.log``.

    pyPESTO logs its multistart optimization (per-start progress, convergence, history) via
    the ``pypesto`` logger. We attach a dedicated file handler at DEBUG and stop propagation
    so those verbose logs land only in ``pypesto.log`` — not in ``petab_mcp.log`` or the
    server's stderr. DEBUG is intentional: the whole point of the separate file is to hold
    the detailed per-start fit log without cluttering anything else. Idempotent per path.
    """
    root = Path(store_root)
    root.mkdir(parents=True, exist_ok=True)
    log_path = str((root / "pypesto.log").resolve())

    logger = logging.getLogger("pypesto")
    logger.setLevel(level)
    if log_path not in _pypesto_configured_paths:
        handler = logging.FileHandler(log_path, encoding="utf-8")
        handler.setFormatter(logging.Formatter(_FORMAT))
        logger.addHandler(handler)
        logger.propagate = False  # keep pyPESTO's verbose logs to its own file
        _pypesto_configured_paths.add(log_path)
    return logger
