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
    """Structured view of the current model via libsbml."""
    sm = session.model.sbml_model
    return {
        "model_id": session.model.model_id,
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
