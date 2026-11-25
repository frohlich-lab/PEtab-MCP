"""Model inspection (read-only).

The model is authored and edited only in Antimony, and structural edits go exclusively
through the constrained operations in :mod:`operations`. This module just exposes the
current model as text and a structured summary.
"""

from __future__ import annotations

import re
from typing import Any


from .errors import ModelUpdateError, ValidationError

MODEL_FORMATS = ("antimony", "sbml")


def get_model(session, fmt: str = "antimony") -> str:
    """Return the current model as Antimony (canonical) or SBML (read-only view), at its
    parameter table's values (:func:`model_text`)."""
    fmt = fmt.lower()
    if fmt not in MODEL_FORMATS:
        raise ModelUpdateError(f"Unknown model format {fmt!r}; use one of {MODEL_FORMATS}.")
    return model_text(session.model, session.problem.parameter_df, fmt)


def model_text(model, parameter_df, fmt: str = "antimony") -> str:
    """``model`` as Antimony (or SBML), each parameter with a row in ``parameter_df`` at that
    row's nominal value. The table owns a parameter's value, which a fit sets to the one it
    found; the model's own is whatever it was written with, and progress_off alkan 15 and 18 froze
    those, read off the model text, as if fitted (PEtab-MCP#341). A copy is set and rendered:
    the model's SBML keys AMICI's build (:meth:`problem.WorkingProblem.module_fingerprint`), so a
    fit leaves it as it was."""
    import pandas as pd
    from petab.v2.C import NOMINAL_VALUE
    from petab.v2.models.sbml_model import SbmlModel

    copy = SbmlModel.from_string(model.to_sbml_str())
    sm = copy.sbml_model
    for pid, value in parameter_df[NOMINAL_VALUE].items():
        if (p := sm.getParameter(str(pid))) is not None and pd.notna(value):
            p.setValue(float(value))
    return copy.to_antimony() if fmt == "antimony" else copy.to_sbml_str()


def version_text(provenance, version: int) -> str:
    """Stored version ``version``'s model as :func:`model_text` renders it, at its own table's
    values."""
    from petab.v2.models.sbml_model import SbmlModel

    model = SbmlModel.from_antimony(provenance.get_model_antimony(version))
    return model_text(model, provenance.get_parameter_df(version))


def model_structure(
    session, reactions: str | None = None, initial_conditions: str | None = None
) -> dict[str, Any]:
    """Structured view of the current model via libsbml: its species, parameters and readouts, and
    ``n_reactions``. ``reactions`` and ``initial_conditions`` are regular expressions: the
    reactions whose id or rate law ``reactions`` matches are listed, and the initial-condition
    account of the species whose id ``initial_conditions`` matches; ``""`` lists every one, and
    either is left out when ``None``. A rule-based model's reactions run to thousands
    (PEtab-MCP#171), and the answer is re-sent on every later turn of the agent that asked.

    The per-species INITIAL-CONDITION account is :func:`reconcile.initial_conditions_report`:
    whether each species' declared t=0 is used (fixed), fitted (estimated), overridden
    per-condition, or held by a rule. That account used to ride on every edit result until the
    result became just the fit, which left it computed but unreachable — and it is the only place
    two facts appear at all: which IC parameter belongs to which species, and which species are
    driven by a condition target (those have no parameter row, so `get_parameters` cannot show
    them). It lives here, on an on-demand read, rather than on every call.

    Until a model is authored (``session.model_authored()``) the species, parameters and initial
    conditions are empty: the working model is then the seed, whose species are placeholders for
    the names the observable formulas and condition table use.
    """
    from . import reconcile as R

    sm = session.model.sbml_model
    readouts = R.readouts(session.problem)
    authored = session.model_authored()
    out = {
        "model_id": session.model.model_id,
        "compartments": [sm.getCompartment(i).getId() for i in range(sm.getNumCompartments())],
        "species": (
            [sm.getSpecies(i).getId() for i in range(sm.getNumSpecies())] if authored else []
        ),
        "parameters": (
            [p for p in session.model.get_parameter_ids() if p not in readouts] if authored else []
        ),
        "readouts": readouts,
        "n_reactions": sm.getNumReactions(),
    }
    if initial_conditions is not None:
        match = _search("initial_conditions", initial_conditions)
        report = R.initial_conditions_report(session.problem) if authored else []
        out["initial_conditions"] = [row for row in report if match(row["species"])]
    if reactions is not None:
        match = _search("reactions", reactions)
        out["reactions"] = []
        for i in range(sm.getNumReactions()):
            reaction, law = sm.getReaction(i), sm.getReaction(i).getKineticLaw()
            rate_law = law.getFormula() if law is not None else None
            if match(reaction.getId()) or (rate_law is not None and match(rate_law)):
                out["reactions"].append(
                    {
                        "id": reaction.getId(),
                        "reversible": reaction.getReversible(),
                        "rate_law": rate_law,
                    }
                )
    return out


def _search(argument: str, pattern: str):
    """``pattern``'s ``search``, refusing one that is not a regular expression."""
    try:
        return re.compile(pattern).search
    except re.error as exc:
        raise ValidationError(
            f"get_model_structure: `{argument}` is a regular expression, and {pattern!r} is not "
            f"one ({exc}); '' lists them all."
        ) from None
