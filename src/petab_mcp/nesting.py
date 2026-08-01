"""Is a committed variant actually NESTED in the parent it is being compared against?

For a model NESTED in its parent the maximum-likelihood negative log-likelihood **cannot get
worse when a parameter is added** — the larger model can always reproduce the smaller one by
holding the new parameter at its neutral value. So a commit whose estimated-parameter count
ROSE while its nll ROSE too implies exactly one of:

(a) **the edit was not additive** — an existing term was DISPLACED rather than extended, so the
    child is not a superset of the parent; or
(b) **the optimizer under-converged** on the larger model.

(b) is the rarer of the two now that half of every multistart is warm-started from the previous
fit's optimum (see :func:`fitting.warm_start_guesses`): an additive edit's first start sits
exactly where the child reproduces its parent, so the search has to actively lose ground to end
up worse. Rarer is not impossible — a non-neutral new parameter, or a nominal point with no
finite objective, still leaves the fresh sample to do the work — so both causes stay named.

Either way the comparison the agent is about to reason from is INVALID, and reading the worse
score as "the extra structure does not help" is a mistake. Nothing else in the payload surfaces
it: BIC only says the variant lost, which is exactly how a legitimately-unhelpful parameter
looks. The motivating case (blasi, run 80ccf20e): an edit meant to add a new ``c816`` factor to
``r5_12_16_8`` OVERWROTE the ``c81216`` already in that rate law, so v4 gained a parameter, lost
3.984 nll, and was read as evidence the model was complete — the run's only "does adding
structure help?" experiment, spent on a variant that could not answer the question.

This is ADVISORY ONLY. Exploring a genuinely non-nested variant is legitimate, so the commit is
never blocked; the agent is only told that this particular comparison cannot support the
conclusion it looks like it supports. Where the culprit is cheaply identifiable — a parent
parameter that stopped being estimated, or one that lost a reference in the rate laws — it is
NAMED, so "c81216 no longer appears" is stated outright instead of left as a guess.
"""

from __future__ import annotations

import re
from typing import Any

from .formatting import finite_float as _finite

# An nll worsening below this is numerical noise (the same optimum re-reported), not a nesting
# violation worth a line in every fit payload. Any real displacement or under-convergence clears
# it by orders of magnitude.
_NLL_WORSE_TOLERANCE = 1e-3

# Agent-facing text, hoisted like the other server-side advisories (see
# ``fitting.NOISE_SIGMA_INFLATED_ADVICE``) so what the agent reads is greppable and reviewable
# without stepping through the logic. It cannot live in ``petab_harness.prompts``: that registry
# is the HARNESS's own standing messages, and this server never imports the harness. The harness
# side of this signal — what an outstanding invalid comparison means for a `finish` — does live
# there, as ``FINISH_ARMED_INVALID_COMPARISON_TMPL``, and quotes this string verbatim.
#
# Rides on every fit payload that fires it, and is re-sent in the agent's history each turn (PR
# #23), so it stays ONE line: the numbers, the cause, and what NOT to conclude.
NESTING_ADVISORY_TMPL = (
    "INVALID COMPARISON vs {against}: k rose {parent_k}->{child_k} but neg_log_likelihood got "
    "WORSE by {worse:.6g} — a model nested in its parent cannot fit worse. Either {cause}, or "
    "the fit under-converged. This does NOT show the added structure fails to help: redo the "
    "edit keeping every parent term, or re-fit with more starts, before concluding anything "
    "from it."
)
# The `cause` clause: named when a displaced parameter is cheaply identifiable, generic when the
# rewrite is somewhere the parameter ids and reference counts cannot see.
NESTING_CAUSE_NAMED_TMPL = "the edit was not additive ({note})"
NESTING_CAUSE_UNKNOWN = (
    "the edit was not additive (a parent term was rewritten — see diff_versions)"
)


def _model_text(model: Any) -> str | None:
    """Resolve a ``model`` entry, which may be the Antimony source or a zero-arg callable that
    reads it. Lazy because the source is needed ONLY to name a displaced parameter — i.e. only
    once the advisory has already been decided — so the common (clean) commit never pays the read.
    """
    return model() if callable(model) else model


def _reference_count(model_text: str, pid: str) -> int:
    """How many times ``pid`` appears as a whole symbol in an Antimony model."""
    return len(re.findall(rf"\b{re.escape(pid)}\b", model_text))


def _displacement_note(
    parent_ids: set[str], child_ids: set[str],
    parent_model: Any, child_model: Any,
) -> str | None:
    """Name the displaced parameter(s), or ``None`` when nothing cheap identifies one.

    Two checks, cheapest first. A parent parameter that is no longer ESTIMATED in the child is
    unambiguous. Otherwise the displacement is inside the rate laws — the motivating case, where
    ``c81216`` stayed in the parameter table but lost one of its nine references — so compare
    whole-symbol reference counts in the two Antimony sources. The models are resolved HERE (see
    :func:`_model_text`), so a caller that has to read one off disk only does so on this path.
    """
    dropped = sorted(parent_ids - child_ids)
    if dropped:
        return f"{', '.join(dropped)} no longer estimated"
    parent_model = _model_text(parent_model)
    child_model = _model_text(child_model)
    if not parent_model or not child_model:
        return None
    for pid in sorted(parent_ids & child_ids):
        before = _reference_count(parent_model, pid)
        after = _reference_count(child_model, pid)
        if after < before:
            return f"{pid} now appears in {after} places, was {before}"
    return None


def nesting_advisory(parent: dict[str, Any], child: dict[str, Any]) -> str | None:
    """One line for the fit payload when ``child`` gained parameters yet fits WORSE than
    ``parent`` — or ``None`` when the comparison is fine or cannot be made.

    Both dicts carry ``version``, ``n_estimated``, ``neg_log_likelihood``, ``estimated_ids`` and
    (optionally) ``model`` — the Antimony source (or a zero-arg callable returning it), used only
    to name the displaced parameter and therefore resolved only when the advisory fires.
    Returns ``None`` whenever a needed number is missing or non-finite, so an unfitted/legacy
    parent simply produces no advisory rather than a wrong one.
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

    note = _displacement_note(
        set(parent.get("estimated_ids") or ()), set(child.get("estimated_ids") or ()),
        parent.get("model"), child.get("model"),
    )
    # Every other field degrades gracefully when absent; name the parent generically rather than
    # emitting "vs vNone" into text the agent reads verbatim.
    version = parent.get("version")
    return NESTING_ADVISORY_TMPL.format(
        against=f"v{version}" if version is not None else "its parent",
        parent_k=int(parent_k), child_k=int(child_k), worse=worse,
        cause=(NESTING_CAUSE_NAMED_TMPL.format(note=note) if note else NESTING_CAUSE_UNKNOWN),
    )


def valid_added_parameter_experiment(parent: dict[str, Any], child: dict[str, Any]) -> bool:
    """True when ``child`` adds parameters to ``parent`` AND its nll did not worsen — i.e. a
    comparison that CAN answer "does adding this structure help?".

    The counterpart to :func:`nesting_advisory`: one of these clears the run's outstanding
    invalid-comparison flag (see ``session.invalid_comparison``), so a lead that redoes the
    experiment properly is no longer told it has none.
    """
    parent_k, child_k = parent.get("n_estimated"), child.get("n_estimated")
    parent_nll = _finite(parent.get("neg_log_likelihood"))
    child_nll = _finite(child.get("neg_log_likelihood"))
    if parent_k is None or child_k is None or parent_nll is None or child_nll is None:
        return False
    return int(child_k) > int(parent_k) and (child_nll - parent_nll) <= _NLL_WORSE_TOLERANCE
