"""Problem validation, reusing petab.v2's own linting.

petab.v2's ``Problem.validate()`` already runs the checks that enforce the
invariant we care about — that the model still provides every entity the fixed
tables (observables, conditions, experiments, measurements) reference, e.g.
``CheckValidConditionTargets``, ``CheckObservablesDoNotShadowModelEntities``,
``CheckMeasuredObservablesDefined``, ``CheckAllParametersPresentInParameterTable``.
So we lean on it rather than re-implementing entity-reference checking.
"""

from __future__ import annotations

from typing import Any

from .formatting import issues_to_records


def validate_problem(problem) -> dict[str, Any]:
    """Run petab.v2 linting and return a JSON-friendly report.

    Returns a dict with ``valid`` (no ERROR/CRITICAL issues), ``issues`` (all
    issues with level/message/check), and convenience ``n_errors``/``n_warnings``.
    """
    result = problem.validate()
    issues = issues_to_records(result)
    n_errors = sum(1 for i in issues if i["level"] in ("error", "critical"))
    n_warnings = sum(1 for i in issues if i["level"] == "warning")
    return {
        "valid": not result.has_errors(),
        "n_errors": n_errors,
        "n_warnings": n_warnings,
        "issues": issues,
    }
