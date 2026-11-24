"""Is a committed variant NESTED in the parent it is being compared against?

For a model NESTED in its parent the maximum-likelihood negative log-likelihood **cannot get
worse when a parameter is added** — the larger model can always reproduce the smaller one by
holding the new parameter at its neutral value. A commit whose estimated-parameter count ROSE
while its nll got WORSE is read one of two ways, depending on whether it is nested:

* **Nested**: the edit only ADDED reactions and rules, every parent parameter is still estimated,
  and every parent fitted value lies inside the child's bounds. Then the worse nll is an INVALID
  COMPARISON: the optimizer under-converged (or a new term cannot be switched off inside its
  bounds), and reading the worse score as "the extra structure does not help" is a mistake. BIC
  only says the variant lost, which is how a legitimately-unhelpful parameter looks too, so
  nothing else in the payload surfaces it. This is the flag a client's finish check reads
  (``session.invalid_comparison``).
* **Not nested**: a reaction or rule was removed or rewritten, a parent parameter stopped being
  estimated, or a bound excludes the parent's fitted value. Such a model can fit worse
  legitimately, so the line names the first reason found and gives no re-fit advice. Sweep
  63372750 (#263) is why: the advisory fired 39 times in 11 of 15 runs without testing nesting,
  and agents read every one as under-convergence and re-fitted non-nested edits with 200-250
  starts for up to 99 minutes, to the same nll.

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

from .formatting import finite_float as _finite
from .progress_signals import signal, signal_or
from .provenance import model_diff

# An nll worsening below this is numerical noise (the same optimum re-reported), not a nesting
# violation worth a line in every fit payload. Any real displacement or under-convergence clears
# it by orders of magnitude.
_NLL_WORSE_TOLERANCE = 1e-3

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


def _structural_change(parent_model: str, child_model: str) -> str | None:
    """The first parent reaction or rule the edit removed or rewrote, or ``None``."""
    removed: dict[tuple[str, str], str] = {}
    added: dict[tuple[str, str], str] = {}
    for line in model_diff(parent_model, child_model).splitlines():
        if line.startswith(("---", "+++")):
            continue
        side = {"-": removed, "+": added}.get(line[:1])
        if side is not None and (found := _statement(line[1:])):
            kind, sid, text = found
            side[kind, sid] = text
    reaction_texts = {text for (kind, _), text in added.items() if kind == "reaction"}
    for (kind, sid), text in removed.items():
        if added.get((kind, sid)) == text or (kind == "reaction" and text in reaction_texts):
            continue
        return f"{kind} {sid} {'rewritten' if (kind, sid) in added else 'removed'}"
    return None


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
    """``(line, invalid)`` for the fit payload when ``child`` gained parameters yet fits WORSE
    than ``parent``, or ``None`` when the comparison is fine or cannot be made. ``invalid`` is
    True for an INVALID COMPARISON (the child is nested), False for a not-nested line.

    Both dicts carry ``version``, ``n_estimated``, ``neg_log_likelihood``, ``estimated_ids`` and
    ``model`` — the Antimony source, or a zero-arg callable returning it, resolved only once k
    rose and the nll got worse. ``parent`` also carries ``values`` (estimated id to its fitted
    value) and ``child`` carries ``bounds`` (estimated id to ``(lower, upper)``). Returns ``None``
    whenever a needed number or either model is missing, so an unfitted parent produces no line
    rather than a wrong one.
    """
    parent_k, child_k = parent.get("n_estimated"), child.get("n_estimated")
    parent_nll = _finite(parent.get("neg_log_likelihood"))
    child_nll = _finite(child.get("neg_log_likelihood"))
    if parent_k is None or child_k is None or parent_nll is None or child_nll is None:
        return None
    if int(child_k) <= int(parent_k):
        return None
    worse = child_nll - parent_nll
    if worse <= _NLL_WORSE_TOLERANCE:
        return None
    parent_model, child_model = _model_text(parent.get("model")), _model_text(child.get("model"))
    if not parent_model or not child_model:
        return None

    # Name the parent generically rather than emitting "vs vNone" into text the agent reads.
    version = parent.get("version")
    against = f"v{version}" if version is not None else "its parent"
    numbers = {"against": against, "parent_k": int(parent_k), "child_k": int(child_k)}
    reason = _not_nested_reason(parent, child, parent_model, child_model, against)
    if reason:
        return NOT_NESTED_TMPL.format(reason=reason, worse=worse, **numbers), False
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
    return int(child_k) > int(parent_k) and (child_nll - parent_nll) <= _NLL_WORSE_TOLERANCE
