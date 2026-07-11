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
    """Route the fitting optimizer's logging to ``<store_root>/pypesto.log``.

    The fit runs pyPESTO with the Fides optimizer; both log through their own loggers
    (``pypesto`` and ``fides``). We attach a dedicated file handler and stop propagation so
    that (verbose) optimizer output lands only in ``pypesto.log`` — not in ``petab_mcp.log``
    or the server's stderr. The separate file is the place for the detailed per-start /
    per-iteration fit log without cluttering anything else. Idempotent per path.
    """
    root = Path(store_root)
    root.mkdir(parents=True, exist_ok=True)
    log_path = str((root / "pypesto.log").resolve())

    if log_path not in _pypesto_configured_paths:
        handler = logging.FileHandler(log_path, encoding="utf-8")
        handler.setFormatter(logging.Formatter(_FORMAT))
        for name in ("pypesto", "fides"):
            lg = logging.getLogger(name)
            lg.setLevel(level)
            lg.addHandler(handler)
            lg.propagate = False  # keep optimizer logs to their own file
        _pypesto_configured_paths.add(log_path)
    return logging.getLogger("pypesto")
