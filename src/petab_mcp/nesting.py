"""Is a committed variant NESTED in the parent it is being compared against?

For a model NESTED in its parent the maximum-likelihood negative log-likelihood **cannot get
worse when a parameter is added** — the larger model can always reproduce the smaller one by
holding the new parameter at its neutral value. A commit whose estimated-parameter count ROSE
while its nll got WORSE, by more than ``AT_BEST_NLL_TOL`` (the 1 nll within which a start counts
as at the best), is read one of three ways:

* **Nested**: the edit only ADDED reactions and rules, every parent parameter is still estimated,
  and every parent fitted value lies inside the child's bounds. Then the worse nll is an INVALID
  COMPARISON: the optimizer under-converged (or a new term cannot be switched off inside its
  bounds), and reading the worse score as "the extra structure does not help" is a mistake. BIC
  only says the variant lost, which is how a legitimately-unhelpful parameter looks too, so
  nothing else in the payload surfaces it. This is the flag a client's finish check reads
  (``session.invalid_comparison``).
* **Nested, a new parameter at a bound**: its term cannot switch off inside the box, so the child
  cannot reproduce its parent. The line names that parameter and gives no re-fit advice: luna
  alkan 39 re-fitted with 300 starts over a 0.0017 gap that a term pinned at a positive lower
  bound made (#334).
* **Not nested**: a reaction or rule was removed or rewritten, a parent parameter stopped being
  estimated, or a bound excludes the parent's fitted value. Such a model can fit worse
  legitimately, so the line names the first reason found and gives no re-fit advice. Sweep
  63372750 (#263) is why: the advisory fired 39 times in 11 of 15 runs without testing nesting,
  and agents read every one as under-convergence and re-fitted non-nested edits with 200-250
  starts for up to 99 minutes, to the same nll.

An edit that only REMOVED structure and parameters is nested in its parent the other way round,
so it cannot fit better; a worse nll there gets the objective at the parent's optimum
(``objective_before``) beside the fit's, which fontaine 100 read as a missed basin without (#334).

The structural half reads the unified Antimony diff that ``diff_versions`` reports
(:func:`provenance.model_diff`): a parent reaction or rule on a removed line is either absent from
the child (removed) or present under the same id with other text (rewritten). A reaction whose
line only moved, or was renumbered with its text unchanged, counts as kept. A rate law extended
by a factor counts as rewritten too: whether the box holds that factor's neutral value is not
something the diff can see.

ADVISORY ONLY. Exploring a non-nested variant is legitimate, so the commit is never blocked.
"""

from __future__ import annotations

import math
import re
from typing import Any

from .constants import AT_BEST_NLL_TOL
from .formatting import finite_float as _finite
from .progress_signals import signal, signal_or
from .provenance import model_diff

# Agent-facing text, hoisted like the other server-side advisories (see
# ``session._STRUCTURAL_PLATEAU_ADVICE``) so what the agent reads is greppable and reviewable
# without stepping through the logic. What an outstanding invalid comparison means for a
# `finish` is a client's side of this signal, not this server's; a client that says so can quote
# this string verbatim.
#
# Rides on every fit payload that fires it, and is re-sent in the agent's history each turn (PR
# #23), so each stays ONE line: the numbers, the cause, and what NOT to conclude.
NESTING_ADVISORY_TMPL = (
    signal_or(
        "INVALID COMPARISON vs {against}: k rose {parent_k}->{child_k} but neg_log_likelihood got "
        "WORSE by {worse:.6g}, though the edit only added to {against} and its fitted values lie "
        "inside the bounds — a model nested in its parent cannot fit worse. Either the fit "
        "under-converged or a new term cannot be switched off inside its bounds. This does NOT "
        "show the added structure fails to help",
        "This model is nested in {against}, and either its fit under-converged or a new term "
        "cannot be switched off inside its bounds",
    )
    + ": re-fit with more starts before concluding anything from it."
)
# A comparison and nothing else, so cut whole with progress signals off.
NOT_NESTED_TMPL = signal(
    "vs {against}, not nested: {reason}. k rose {parent_k}->{child_k} and neg_log_likelihood got "
    "worse by {worse:.6g}, which a model not nested in its parent can do. It does not show the "
    "added structure fails to help; to test that, add only that structure to {against}."
)
AT_BOUND_TMPL = (
    signal_or(
        "vs {against}: k rose {parent_k}->{child_k} and neg_log_likelihood got worse by "
        "{worse:.6g} with {pid}, new since {against}, at its {side} bound {bound:g}",
        "{pid}, new since {against}, sits at its {side} bound {bound:g}",
    )
    + ": inside its bounds its term cannot switch off, so this model cannot reproduce "
    "{against}'s fit. That is the bound, not an under-converged fit, and more starts will not "
    "change it. Drop the term, or give it a box that reaches the value switching it off."
)
# A comparison and nothing else, so cut whole with progress signals off.
DELETION_TMPL = signal(
    "vs {against}: k fell {parent_k}->{child_k} and neg_log_likelihood rose by {worse:.6g}. At "
    "{against}'s optimum this model scores {before:.6g} ({cost:+.6g} on {against}), and its fit "
    "took it to {nll:.6g}. A model nested in {against} cannot fit better than it, so the rise is "
    "what the removed structure was worth, not a sign of a missed basin."
)

# A reaction (``id: A -> B; rate;``, irreversible ``=>``) and an assignment (``x := ...``) or rate
# (``x' = ...``) rule, each on a line of its own as libantimony writes model.ant.
_RULE = re.compile(r"^\s*(\w+)\s*(?::=|'\s*=)\s*(.*?)\s*$")
_REACTION = re.compile(r"^\s*(\w+)(?:\s+in\s+\w+)?\s*:\s*(.*(?:->|=>).*?)\s*$")


def _model_text(model: Any) -> str | None:
    """Resolve a ``model`` entry, which may be the Antimony source or a zero-arg callable that
    reads it. Lazy because the source is needed only once k rose and the nll got worse, so the
    common (improving) commit never pays the read."""
    return model() if callable(model) else model


def _statement(line: str) -> tuple[str, str, str] | None:
    """``(kind, id, text)`` of a reaction or rule line, else ``None``."""
    if m := _RULE.match(line):
        return "rule", m.group(1), m.group(2)
    if m := _REACTION.match(line):
        return "reaction", m.group(1), m.group(2)
    return None


def _statement_diff(parent_model: str, child_model: str) -> tuple[dict, dict]:
    """``(removed, added)``: the reactions and rules on the diff's removed and added lines, each
    ``{(kind, id): text}``."""
    removed: dict[tuple[str, str], str] = {}
    added: dict[tuple[str, str], str] = {}
    for line in model_diff(parent_model, child_model).splitlines():
        if line.startswith(("---", "+++")):
            continue
        side = {"-": removed, "+": added}.get(line[:1])
        if side is not None and (found := _statement(line[1:])):
            kind, sid, text = found
            side[kind, sid] = text
    return removed, added


def _unmatched(side: dict, other: dict) -> list[tuple[str, str]]:
    """The statements of ``side`` that ``other`` does not hold: none under the same id with the
    same text, nor, for a reaction, under any id (a line that only moved or was renumbered)."""
    texts = {text for (kind, _), text in other.items() if kind == "reaction"}
    return [
        (kind, sid)
        for (kind, sid), text in side.items()
        if other.get((kind, sid)) != text and not (kind == "reaction" and text in texts)
    ]


def _structural_change(parent_model: str, child_model: str) -> str | None:
    """The first parent reaction or rule the edit removed or rewrote, or ``None``."""
    removed, added = _statement_diff(parent_model, child_model)
    for kind, sid in _unmatched(removed, added):
        return f"{kind} {sid} {'rewritten' if (kind, sid) in added else 'removed'}"
    return None


def _new_at_bound(parent: dict[str, Any], child: dict[str, Any]) -> dict[str, Any] | None:
    """The first of ``child``'s pins (``at_bound``, :func:`fitting.parameters_at_bound`) on a
    parameter ``parent`` did not estimate, as the template's ``pid``, ``side`` and ``bound``."""
    new = set(child.get("estimated_ids") or ()) - set(parent.get("estimated_ids") or ())
    for pin in child.get("at_bound") or ():
        if pin.get("parameter_id") in new and _finite(pin.get("bound_value")) is not None:
            return {"pid": pin["parameter_id"], "side": pin["side"], "bound": pin["bound_value"]}
    return None


def _only_removed(
    parent: dict[str, Any], child: dict[str, Any], parent_model: str, child_model: str
) -> bool:
    """Whether ``child`` estimates a strict subset of ``parent``'s parameters and its model adds
    or rewrites no reaction or rule."""
    if not set(child.get("estimated_ids") or ()) < set(parent.get("estimated_ids") or ()):
        return False
    removed, added = _statement_diff(parent_model, child_model)
    return not _unmatched(added, removed)


def _excluded_by_bounds(
    values: dict[str, Any], bounds: dict[str, tuple[Any, Any]], against: str
) -> str | None:
    """The first parent parameter whose fitted value lies outside the child's bounds."""
    for pid in sorted(values):
        value = _finite(values[pid])
        if value is None:
            continue
        lb, ub = (_finite(b) for b in bounds.get(pid, (None, None)))
        if lb is not None and value < lb and not math.isclose(value, lb):
            return f"{pid}'s lower bound {lb:g} excludes {against}'s fitted {value:.6g}"
        if ub is not None and value > ub and not math.isclose(value, ub):
            return f"{pid}'s upper bound {ub:g} excludes {against}'s fitted {value:.6g}"
    return None


def _not_nested_reason(
    parent: dict[str, Any],
    child: dict[str, Any],
    parent_model: str,
    child_model: str,
    against: str,
) -> str | None:
    """Why ``child`` is not nested in ``parent``, the first reason found, or ``None``: a removed
    or rewritten reaction or rule, then a parameter no longer estimated, then a bound that
    excludes the parent's fitted value."""
    structural = _structural_change(parent_model, child_model)
    if structural:
        return structural
    dropped = sorted(set(parent.get("estimated_ids") or ()) - set(child.get("estimated_ids") or ()))
    if dropped:
        return f"{dropped[0]} no longer estimated"
    return _excluded_by_bounds(parent.get("values") or {}, child.get("bounds") or {}, against)


def nesting_advisory(parent: dict[str, Any], child: dict[str, Any]) -> tuple[str, bool] | None:
    """``(line, invalid)`` for the fit payload when ``child`` gained parameters, or only lost
    some, and fits WORSE than ``parent``; ``None`` when the comparison is fine or cannot be made.
    ``invalid`` is True for an INVALID COMPARISON (the child is nested and under-converged), False
    for any other line.

    Both dicts carry ``version``, ``n_estimated``, ``neg_log_likelihood``, ``estimated_ids`` and
    ``model`` — the Antimony source, or a zero-arg callable returning it, resolved only once k
    moved and the nll got worse. ``parent`` also carries ``values`` (estimated id to its fitted
    value); ``child`` carries ``bounds`` (estimated id to ``(lower, upper)``), ``at_bound`` (its
    pins, :func:`fitting.parameters_at_bound`) and ``objective_before`` (its objective at the
    parent's optimum, before its fit). Returns ``None`` whenever a needed number or either model
    is missing, so an unfitted parent produces no line rather than a wrong one.
    """
    parent_k, child_k = parent.get("n_estimated"), child.get("n_estimated")
    parent_nll = _finite(parent.get("neg_log_likelihood"))
    child_nll = _finite(child.get("neg_log_likelihood"))
    if parent_k is None or child_k is None or parent_nll is None or child_nll is None:
        return None
    worse = child_nll - parent_nll
    if int(child_k) == int(parent_k) or worse <= AT_BEST_NLL_TOL:
        return None
    parent_model, child_model = _model_text(parent.get("model")), _model_text(child.get("model"))
    if not parent_model or not child_model:
        return None

    # Name the parent generically rather than emitting "vs vNone" into text the agent reads.
    version = parent.get("version")
    against = f"v{version}" if version is not None else "its parent"
    numbers = {"against": against, "parent_k": int(parent_k), "child_k": int(child_k)}
    if int(child_k) < int(parent_k):
        before = _finite(child.get("objective_before"))
        if before is None or not _only_removed(parent, child, parent_model, child_model):
            return None
        line = DELETION_TMPL.format(
            worse=worse, before=before, cost=before - parent_nll, nll=child_nll, **numbers
        )
        return line, False
    reason = _not_nested_reason(parent, child, parent_model, child_model, against)
    if reason:
        return NOT_NESTED_TMPL.format(reason=reason, worse=worse, **numbers), False
    if pin := _new_at_bound(parent, child):
        return AT_BOUND_TMPL.format(worse=worse, **pin, **numbers), False
    return NESTING_ADVISORY_TMPL.format(worse=worse, **numbers), True


def valid_added_parameter_experiment(parent: dict[str, Any], child: dict[str, Any]) -> bool:
    """True when ``child`` adds parameters to ``parent`` AND its nll did not worsen — i.e. a
    comparison that CAN answer "does adding this structure help?".

    The counterpart to :func:`nesting_advisory`: one of these clears the session's outstanding
    invalid-comparison flag (see ``session.invalid_comparison``), so an agent that redoes the
    experiment properly is no longer told it has none.
    """
    parent_k, child_k = parent.get("n_estimated"), child.get("n_estimated")
    parent_nll = _finite(parent.get("neg_log_likelihood"))
    child_nll = _finite(child.get("neg_log_likelihood"))
    if parent_k is None or child_k is None or parent_nll is None or child_nll is None:
        return False
    return int(child_k) > int(parent_k) and (child_nll - parent_nll) <= AT_BEST_NLL_TOL
