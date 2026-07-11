"""Problem validation via petab.v1 linting.

petab.v1's ``lint_problem(problem)`` returns a bool (``True`` if errors occurred) and
reports the specifics through the ``petab`` logger rather than a structured result.
We capture those log records so the report still carries human-readable messages, and
rely on the same lint to enforce the invariant that the model must satisfy the fixed
tables (an observable/condition/parameter referencing a missing entity is an error).
"""

from __future__ import annotations

import logging
from typing import Any

import petab.v1 as petab


class _CaptureHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


def validate_problem(problem) -> dict[str, Any]:
    """Run petab.v1 linting and return a JSON-friendly report.

    Returns ``valid`` (no errors), ``n_errors``/``n_warnings``, and the captured
    ``issues`` (level/message/check).
    """
    logger = logging.getLogger("petab")
    handler = _CaptureHandler()
    prev_level = logger.level
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    try:
        has_errors = bool(petab.lint_problem(problem))
    finally:
        logger.removeHandler(handler)
        logger.setLevel(prev_level)

    issues = [
        {
            "level": rec.levelname.lower(),
            "message": rec.getMessage(),
            "check": rec.name,
        }
        for rec in handler.records
        if rec.levelno >= logging.WARNING
    ]
    n_errors = sum(1 for r in handler.records if r.levelno >= logging.ERROR)
    n_warnings = sum(1 for r in handler.records if r.levelno == logging.WARNING)
    return {
        "valid": not has_errors,
        "n_errors": n_errors,
        "n_warnings": n_warnings,
        "issues": issues,
    }
