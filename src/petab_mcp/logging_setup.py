"""Package logging setup.

Attaches a file handler to the ``petab_mcp`` logger so a human-readable log
(``<store>/petab_mcp.log``) accompanies the structured provenance/error logs. Idempotent:
calling it repeatedly for the same directory does not add duplicate handlers.
"""

from __future__ import annotations

import logging
from pathlib import Path

_LOGGER_NAME = "petab_mcp"
_configured_paths: set[str] = set()


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
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        )
        logger.addHandler(handler)
        _configured_paths.add(log_path)
    return logger
