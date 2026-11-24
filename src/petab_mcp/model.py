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
    """Return the current model as Antimony (canonical, :func:`at_fitted_values`) or SBML
    (read-only view)."""
    fmt = fmt.lower()
    if fmt == "antimony":
        return at_fitted_values(session.model.to_antimony(), session.problem.parameter_df)
    if fmt == "sbml":
        return session.model.to_sbml_str()
    raise ModelUpdateError(f"Unknown model format {fmt!r}; use one of {MODEL_FORMATS}.")


_NUMBER = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")


def at_fitted_values(text: str, parameter_df) -> str:
    """``text``, a model's Antimony, with each ESTIMATED parameter's ``id = <number>`` at its row's
    nominal value, which a fit sets to the value it found; a species written ``X = 5`` shows its
    estimated ``X_init``'s. The value a model is written with is only where the fit starts:
    progress_off alkan 18 froze `Kc` at a value read off the model text, and alkan 15 froze
    authoring values in place of fitted ones (PEtab-MCP#341).

    Shown to 6 significant digits, more where that would round to 0 or 1, which a species' initial
    value declares fixed (:func:`reconcile.sync_parameter_table`)."""
    from petab.v2.C import NOMINAL_VALUE

    from .model_edits import _value_assignments, assigned_symbols
    from .parameters import estimated_ids

    estimated = set(estimated_ids(parameter_df))
    edits = []
    for sid in assigned_symbols(text):
        row = sid if sid in estimated else f"{sid}_init"
        if row not in estimated:
            continue
        v = float(parameter_df.loc[row, NOMINAL_VALUE])
        for m in _value_assignments(text, sid):
            value = re.search(r"=\s*(.*?)\s*;?\s*$", m[0])
            if _NUMBER.fullmatch(value[1]) and float(value[1]) != v:
                shown = f"{v:.6g}"
                shown = repr(v) if float(shown) in (0.0, 1.0) else shown
                edits.append((m.start() + value.start(1), m.start() + value.end(1), shown))
    for start, end, shown in sorted(edits, reverse=True):
        text = text[:start] + shown + text[end:]
    return text


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
