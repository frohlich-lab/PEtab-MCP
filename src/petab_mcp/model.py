"""Model inspection and editing.

The model is the primary mutable surface. Editing goes through Antimony (compact,
LLM-friendly) or raw SBML text, converted natively by petab.v2's ``SbmlModel``.
A minimal, extensible programmatic SBML API (libsbml-backed) is also exposed for
targeted edits; it will grow over time.

Every model change is validated against the *fixed* tables before it is committed
(swap-validate-revert): if the new model no longer satisfies the observables /
conditions / parameter table, the edit is rejected and the old model restored.
"""

from __future__ import annotations

from typing import Any

from petab.v2.models.sbml_model import SbmlModel

from .errors import ModelUpdateError, UnknownEntityError, ValidationError
from .parameters import parameter_sync_diff
from .validation import validate_problem

MODEL_FORMATS = ("antimony", "sbml")


def get_model(session, fmt: str = "antimony") -> str:
    """Return the current model as Antimony or SBML text."""
    fmt = fmt.lower()
    if fmt == "antimony":
        return session.model.to_antimony()
    if fmt == "sbml":
        return session.model.to_sbml_str()
    raise ModelUpdateError(f"Unknown model format {fmt!r}; use one of {MODEL_FORMATS}.")


def _build_model(text: str, fmt: str, model_id: str | None) -> SbmlModel:
    fmt = fmt.lower()
    try:
        if fmt == "antimony":
            return SbmlModel.from_antimony(text, model_id=model_id)
        if fmt == "sbml":
            return SbmlModel.from_string(text, model_id=model_id)
    except Exception as exc:  # conversion / parse errors
        raise ModelUpdateError(f"Failed to parse {fmt} model: {exc}") from exc
    raise ModelUpdateError(f"Unknown model format {fmt!r}; use one of {MODEL_FORMATS}.")


def update_model(session, text: str, fmt: str = "antimony") -> dict[str, Any]:
    """Replace the working model, validating against the fixed tables.

    Builds the new model, swaps it in, runs petab linting, and — if any
    ERROR/CRITICAL issue appears — reverts and raises ``ValidationError``. On
    success the change is committed and a parameter-sync diff is returned so the
    caller can reconcile the parameter table with the new model.
    """
    old = session.problem.model
    old_id = getattr(old, "model_id", None) if old is not None else None
    old_rel = getattr(old, "rel_path", None) if old is not None else None

    new_model = _build_model(text, fmt, model_id=old_id)
    new_model.rel_path = old_rel or "model.xml"

    with session.lock:
        session.problem.model = new_model
        try:
            report = validate_problem(session.problem)
            if not report["valid"]:
                raise ValidationError(
                    "Model update rejected: the new model is inconsistent with the "
                    "fixed experimental design or parameter table.",
                    issues=[i["message"] for i in report["issues"]
                            if i["level"] in ("error", "critical")],
                )
            sync = parameter_sync_diff(session.problem)
        except Exception:
            # Any failure (validation raised, sync raised, ...) must not leave a
            # broken model committed — restore the previous model.
            session.problem.model = old
            raise

    return {
        "committed": True,
        "model_id": getattr(new_model, "model_id", None),
        "validation": report,
        "parameter_sync": sync,
    }


# -- minimal programmatic SBML API (extensible) ---------------------------

def model_structure(session) -> dict[str, Any]:
    """Summarize the current model's structure via libsbml."""
    sm = session.model.sbml_model
    species = [sm.getSpecies(i).getId() for i in range(sm.getNumSpecies())]
    return {
        "model_id": session.model.model_id,
        "compartments": [sm.getCompartment(i).getId()
                         for i in range(sm.getNumCompartments())],
        "species": species,
        "parameters": list(session.model.get_parameter_ids()),
        "reactions": [
            {
                "id": sm.getReaction(i).getId(),
                "reversible": sm.getReaction(i).getReversible(),
                "formula": (
                    sm.getReaction(i).getKineticLaw().getFormula()
                    if sm.getReaction(i).getKineticLaw() is not None
                    else None
                ),
            }
            for i in range(sm.getNumReactions())
        ],
    }


def set_parameter_value(session, parameter_id: str, value: float) -> dict[str, Any]:
    """Set an SBML *model* parameter's value in place (invalidates compiled model).

    Note this edits the SBML model parameter (a structural default), which is
    distinct from a PEtab parameter-table ``nominal_value`` (see
    ``parameters.update_parameters``).
    """
    with session.lock:
        sm = session.model.sbml_model
        param = sm.getParameter(parameter_id)
        if param is None:
            raise UnknownEntityError(
                f"No SBML parameter with id {parameter_id!r} in the model."
            )
        param.setValue(float(value))
    return {"parameter_id": parameter_id, "value": float(value)}
