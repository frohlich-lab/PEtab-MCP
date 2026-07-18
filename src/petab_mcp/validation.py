"""Problem validation: petab.v1 linting + libsbml SBML consistency.

petab.v1's ``lint_problem(problem)`` returns a bool (``True`` if errors occurred) and
reports the specifics through the ``petab`` logger rather than a structured result.
We capture those log records so the report still carries human-readable messages, and
rely on the same lint to enforce the invariant that the model must satisfy the fixed
tables (an observable/condition/parameter referencing a missing entity is an error).

petab's lint only checks the PEtab *tables* against the model, not the SBML document
itself, so we ALSO run libsbml's ``checkConsistency`` on the candidate model and fold its
findings into the same ``issues`` list. Structural SBML problems (general/identifier/MathML/
overdetermined consistency) become hard errors that gate the commit — surfaced to the agent
pre-compile with a clear message instead of an opaque AMICI codegen crash. Unit-consistency,
modeling-practice, and SBO checks are turned OFF: these are unit-less mechanistic models, so
those categories would flag essentially every reaction/parameter and drown the real signal.
"""

from __future__ import annotations

import logging
from typing import Any

import petab.v1 as petab

from .reconcile import _model_math_names, observation_model_symbols, parameter_ids

# libsbml consistency categories that are pure noise for unit-less mechanistic models — we
# neither gate nor report them (every species/parameter would otherwise flag a unit warning).
_SBML_SILENCED_CATEGORIES = (
    "LIBSBML_CAT_UNITS_CONSISTENCY",
    "LIBSBML_CAT_MODELING_PRACTICE",
    "LIBSBML_CAT_SBO_CONSISTENCY",
)
_SBML_MSG_CAP = 300  # keep a single SBML message from bloating the (re-sent) tool result


class _CaptureHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


def _sbml_consistency_issues(problem) -> list[dict[str, Any]]:
    """libsbml ``checkConsistency`` on the candidate SBML, as ``issues``.

    ERROR/FATAL findings are hard errors (they gate the commit); WARNING findings are
    advisory. The noisy categories in ``_SBML_SILENCED_CATEGORIES`` are disabled first.
    Best-effort: any failure here returns ``[]`` rather than breaking validation.
    """
    try:
        import libsbml

        sbml_model = getattr(problem.model, "sbml_model", None)
        doc = sbml_model.getSBMLDocument() if sbml_model is not None else None
        if doc is None:
            return []
        for cat in _SBML_SILENCED_CATEGORIES:
            code = getattr(libsbml, cat, None)
            if code is not None:
                doc.setConsistencyChecks(code, False)
        doc.getErrorLog().clearLog()  # so we read only THIS consistency pass
        doc.checkConsistency()

        issues: list[dict[str, Any]] = []
        for i in range(doc.getNumErrors()):
            err = doc.getError(i)
            severity = err.getSeverity()
            if severity >= libsbml.LIBSBML_SEV_ERROR:
                level = "error"
            elif severity == libsbml.LIBSBML_SEV_WARNING:
                level = "warning"
            else:
                continue  # skip INFO
            message = (err.getShortMessage() or err.getMessage() or "").strip()
            if len(message) > _SBML_MSG_CAP:
                message = message[:_SBML_MSG_CAP] + "…"
            issues.append({
                "level": level,
                "message": message,
                "check": f"sbml:{err.getCategoryAsString()}",
            })
        return issues
    except Exception:  # noqa: BLE001 - validation must never crash on the SBML lint
        return []


def _observation_model_reuse_issues(problem) -> list[dict[str, Any]]:
    """Hard-error on a model that reuses a fixed observation-model parameter (an
    observable-scaling factor or noise sigma, resolved through the observable/noise formulas or
    their measurement-level ``observableParameters``/``noiseParameters`` overrides) as a
    mechanistic rate constant inside a reaction, rule, or initial assignment.

    A benchmark sweep caught exactly this: two independent runs wired the Brannmark scenario's
    four fixed observable-scaling factors directly into their reaction rate laws instead of
    introducing the ~14 real kinetic parameters the scenario expects, ending up with a
    severely under-mechanistic model that still passed its own chi2 goodness-of-fit check
    (with help from noise sigmas inflated 20-60x). Gate this at commit time rather than relying
    on an agent to notice: reusing one of these symbols does not "cost" anything on its own
    metrics, so nothing else in the pipeline would catch it.
    """
    sm = problem.model.sbml_model
    reused = (observation_model_symbols(problem) & parameter_ids(sm)) & _model_math_names(sm)
    return [
        {
            "level": "error",
            "message": (
                f"'{pid}' is a fixed observation-model parameter (an observable-scaling factor "
                "or noise sigma, resolved through the measurement table's observableParameters/"
                "noiseParameters or an observable/noise formula) but is ALSO used inside a "
                "reaction, rule, or initial assignment. Reusing it lets the model dodge "
                "introducing a real kinetic parameter, and reusing/inflating a noise sigma can "
                "mask a wrong mechanism behind an artificially loose noise model instead of "
                "being estimated as what it actually is. Introduce a SEPARATE, dedicated "
                "parameter for the reaction instead."
            ),
            "check": "observation_model_reuse",
        }
        for pid in sorted(reused)
    ]


def validate_problem(problem) -> dict[str, Any]:
    """Run petab.v1 linting + libsbml SBML consistency and return a JSON-friendly report.

    Returns ``valid`` (no errors from either check), ``n_errors``/``n_warnings``, and the
    captured ``issues`` (level/message/check) — petab table issues first, then SBML ones.
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

    sbml_issues = _sbml_consistency_issues(problem)
    issues += sbml_issues
    sbml_errors = sum(1 for i in sbml_issues if i["level"] == "error")
    n_errors += sbml_errors
    n_warnings += sum(1 for i in sbml_issues if i["level"] == "warning")

    reuse_issues = _observation_model_reuse_issues(problem)
    issues += reuse_issues
    n_errors += len(reuse_issues)

    return {
        "valid": (not has_errors) and sbml_errors == 0 and not reuse_issues,
        "n_errors": n_errors,
        "n_warnings": n_warnings,
        "issues": issues,
    }
