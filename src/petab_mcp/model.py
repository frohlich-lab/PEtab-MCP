"""Model inspection (read-only).

The model is authored and edited only in Antimony, and structural edits go exclusively
through the constrained operations in :mod:`operations`. This module just exposes the
current model as text and a structured summary.
"""

from __future__ import annotations

from typing import Any


from .errors import ModelUpdateError

MODEL_FORMATS = ("antimony", "sbml")


def get_model(session, fmt: str = "antimony") -> str:
    """Return the current model as Antimony (canonical) or SBML (read-only view)."""
    fmt = fmt.lower()
    if fmt == "antimony":
        return session.model.to_antimony()
    if fmt == "sbml":
        return session.model.to_sbml_str()
    raise ModelUpdateError(f"Unknown model format {fmt!r}; use one of {MODEL_FORMATS}.")


def model_structure(session) -> dict[str, Any]:
    """Structured view of the current model via libsbml.

    Also carries the per-species INITIAL-CONDITION account (see
    :func:`reconcile.initial_conditions_report`): whether each species' declared t=0 is used
    (fixed), fitted (estimated), or overridden per-condition. That account used to ride on every
    edit result until the result became just the fit, which left it computed but unreachable —
    and it is the only place two facts appear at all: which IC parameter belongs to which
    species, and which species are driven by a condition target (those have no parameter row, so
    `get_parameters` cannot show them). It lives here, on an on-demand read, rather than on every
    call.
    """
    from . import reconcile as R

    sm = session.model.sbml_model
    return {
        "model_id": session.model.model_id,
        "initial_conditions": R.initial_conditions_report(session.problem),
        "compartments": [sm.getCompartment(i).getId() for i in range(sm.getNumCompartments())],
        "species": [sm.getSpecies(i).getId() for i in range(sm.getNumSpecies())],
        "parameters": list(session.model.get_parameter_ids()),
        "reactions": [
            {
                "id": sm.getReaction(i).getId(),
                "reversible": sm.getReaction(i).getReversible(),
                "rate_law": (
                    sm.getReaction(i).getKineticLaw().getFormula()
                    if sm.getReaction(i).getKineticLaw() is not None
                    else None
                ),
            }
            for i in range(sm.getNumReactions())
        ],
    }
