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
from petab.v1.C import CONDITION_ID, CONDITION_NAME, ESTIMATE

from .reconcile import (
    _model_math_names,
    _reaction_species,
    observation_model_symbols,
    parameter_ids,
    species_ids,
)

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


def _fixed_constant_role_issues(problem, protected_fixed_ids) -> list[dict[str, Any]]:
    """Hard-error when a parameter that ships FIXED and is a mechanistic/structural constant —
    a conservation total, a known rate, a measured concentration — has been REMOVED from the
    model or turned ESTIMATED.

    ``protected_fixed_ids`` is the caller-computed set of such constants (see
    :meth:`session.PetabSession.protected_fixed_parameter_ids`); it already excludes
    initial-condition parameters (removed legitimately with their species) and observation-model
    factors (their own estimation paths). A published problem fixes these for a reason, and:

    * **removing** one drops a piece of the problem's structure (an orphaned conservation total
      or known rate silently pruned when its reaction is dropped/re-authored);
    * **estimating** one lets its value DRIFT off the constant (a benchmark run let three
      conservation totals fixed at 1 wander to 9 / 0.001 / 0.4) and adds to k, inflating BIC.

    Nothing toggles a parameter's estimated/fixed status directly (there is no such tool), and a
    shipped-fixed parameter keeps that status only while it stays declared — so the recourse for
    both is the same: keep the constant declared in the model, as shipped. Passing an empty/None
    set (e.g. a problem with no such constants, or an unreadable v0) yields no issues.
    """
    if not protected_fixed_ids:
        return []
    pdf = getattr(problem, "parameter_df", None)
    if pdf is None:
        return []
    current = {str(pid) for pid in pdf.index}

    def _estimated(pid) -> bool:
        try:
            return ESTIMATE in pdf.columns and float(pdf.loc[pid, ESTIMATE]) != 0
        except (TypeError, ValueError, KeyError):
            return False

    issues: list[dict[str, Any]] = []
    for pid in sorted(protected_fixed_ids):
        if pid not in current:
            issues.append({
                "level": "error",
                "message": (
                    f"'{pid}' ships FIXED (estimate=0) in this problem — a known constant (a "
                    "conservation total, a measured concentration, a fixed rate) — but this edit "
                    "REMOVED it from the model. A shipped-fixed constant is part of the problem's "
                    "structure; dropping it changes the problem. Keep it declared AND used: a "
                    "shipped-fixed constant must be referenced in a rate law, rule, or initial "
                    "assignment (a constant you declare but never use is pruned as an orphan, "
                    "which trips this too). Retain the reaction/expression that uses it."
                ),
                "check": "fixed_constant_removed",
            })
        elif _estimated(pid):
            issues.append({
                "level": "error",
                "message": (
                    f"'{pid}' ships FIXED (estimate=0) in this problem but is now an ESTIMATED "
                    "parameter. A parameter the problem fixes is a known constant — a conservation "
                    "total, a measured concentration, a fixed rate — and estimating it lets its "
                    "value drift off that constant while adding to k (the estimated-parameter "
                    "count), which inflates BIC. There is no tool to toggle a parameter between "
                    "fixed and estimated; a shipped-fixed constant is re-added ESTIMATED only if "
                    "it was dropped and its symbol later reintroduced. Keep the constant declared "
                    "in the model, as shipped, so its fixed status is preserved."
                ),
                "check": "fixed_constant_made_estimated",
            })
    return issues


def _unused_condition_target_issues(problem) -> list[dict[str, Any]]:
    """Hard-error when the model declares a condition-table target but never USES it.

    Each condition-table column (other than ``conditionId``/``conditionName``) sets a model
    quantity per condition — a dose, an input level, a genotype switch, a rate multiplier: the
    very thing that makes the conditions differ. petab requires the target to be a model entity
    (a species or parameter) and reconcile keeps it (a condition column is never pruned), so a
    target the model declares but never references in a rate law, rule, initial assignment, or
    reaction — nor in the observation model — passes the lint yet does NOTHING: every condition
    then simulates identically and the experimental design is silently ignored. A run can then
    "fit" by tuning other parameters while the perturbation it should be responding to is inert.
    Gate it here so the model actually consumes each condition-table quantity.

    Only targets that ARE model entities are checked (a target absent from the model is a
    different error petab lint already reports). Best-effort: no condition table / no model
    yields no issues.
    """
    cdf = getattr(problem, "condition_df", None)
    model = getattr(problem, "model", None)
    sm = model.sbml_model if model is not None else None
    if cdf is None or sm is None:
        return []

    used = _model_math_names(sm) | _reaction_species(sm) | observation_model_symbols(problem)
    model_entities = parameter_ids(sm) | species_ids(sm)
    targets = {str(c) for c in cdf.columns} - {CONDITION_ID, CONDITION_NAME}

    return [
        {
            "level": "error",
            "message": (
                f"The condition table sets '{tgt}' per condition, but the model never uses "
                f"'{tgt}' — it appears in no rate law, rule, initial assignment, reaction, or "
                "observation formula. The conditions therefore don't perturb the model: every "
                "condition simulates identically and the experimental design that distinguishes "
                f"them is ignored. Use '{tgt}' in the mechanism — as an input/driver in a rate "
                "law, or as the initial state of a species that participates in the dynamics."
            ),
            "check": "unused_condition_target",
        }
        for tgt in sorted(targets)
        if tgt in model_entities and tgt not in used
    ]


def validate_problem(problem, protected_fixed_ids=None) -> dict[str, Any]:
    """Run petab.v1 linting + libsbml SBML consistency and return a JSON-friendly report.

    Returns ``valid`` (no errors from either check), ``n_errors``/``n_warnings``, and the
    captured ``issues`` (level/message/check) — petab table issues first, then SBML ones.

    ``protected_fixed_ids`` (the shipped-fixed mechanistic/structural constants, from
    :meth:`session.PetabSession.protected_fixed_parameter_ids`) enables the role check that
    rejects removing or estimating one (see :func:`_fixed_constant_role_issues`); omit it to
    skip that check. The condition-target check (see :func:`_unused_condition_target_issues`)
    always runs — it needs no reference.
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

    role_issues = _fixed_constant_role_issues(problem, protected_fixed_ids)
    issues += role_issues
    n_errors += len(role_issues)

    condition_issues = _unused_condition_target_issues(problem)
    issues += condition_issues
    n_errors += len(condition_issues)

    return {
        "valid": (
            (not has_errors) and sbml_errors == 0
            and not reuse_issues and not role_issues and not condition_issues
        ),
        "n_errors": n_errors,
        "n_warnings": n_warnings,
        "issues": issues,
    }
