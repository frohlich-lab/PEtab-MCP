"""What ``create_server(fix_pointers=False)`` withholds from everything it sends: the NUMBERS that
point the agent at what to change, or at where the fit is off (gemot's ``fix_pointers`` arm).

The server still computes, stores and reads every one of them (a stop, the de-dup or a refusal may
rest on one); only what leaves it is cut, by :func:`strip` in ``server._FixPointers``, so a stored
version read back is cut the same way. Text is left as written, bar the observables and replicated
points a residual verdict's ``summary`` names, and the warning that a tie has no merge hint at its
base (:func:`identifiability.unhinted_ties`), which goes whole. Cut:

* the payoff of an edit: ``merge_hints`` (each with the ops that take it) and
  ``n_merge_hints``; a priced tie's ``dscore`` and ``merge_improves_score``, and
  ``n_merges_improving_score``; the ranked ``candidates``; a data regime's ``support_for_extension``
  and ``delta_bic_favouring_two``;
* where the fit is off: ``per_observable`` (with its ``omitted`` block) and ``worst_points``,
  ``sigma_heterogeneity``, every ``fitted_over_data_noise``, a replicate lack-of-fit's ``points`` and
  ``flagged``, and which observables a verdict flags: ``flagged_observables`` becomes
  ``n_flagged_observables``, its count.

Kept: every score and total, whether a verdict flags anything, ``low_power``, each noise sigma's
``status``, the parameters at a bound, the per-parameter identifiability and each tie's ``dnll``.
"""

from __future__ import annotations

import re
from typing import Any

from .data_access import _MAX_NAMED
from .identifiability import UNHINTED_TIE

_CUT = frozenset(
    {
        "merge_hints",
        "n_merge_hints",
        "dscore",
        "merge_improves_score",
        "n_merges_improving_score",
        "candidates",
        "support_for_extension",
        "delta_bic_favouring_two",
        "per_observable",
        "worst_points",
        "sigma_heterogeneity",
        "n_sigma_heterogeneity",
        "fitted_over_data_noise",
    }
)

#: The worst replicated points a lack-of-fit line names, ``(<label> <z>; ...; Holm p``
#: (``data_access._replicate_lof_advice``).
_LOF_POINTS = re.compile(r"\([^()]*; Holm p ")


def strip(value: Any) -> Any:
    """``value`` (a JSON-like result) without the fields this module cuts, at any depth."""
    if isinstance(value, list):
        return [strip(v) for v in value if not _unhinted(v)]
    if not isinstance(value, dict):
        return value
    out = {k: strip(v) for k, v in value.items() if k not in _CUT}
    if "per_observable" in value:
        out.pop("omitted", None)
    if "min_p_holm" in value:  # a replicate lack-of-fit verdict
        out.pop("points", None)
        out.pop("flagged", None)
    flagged = value.get("flagged_observables")
    if isinstance(flagged, list):
        n = value.get("n_flagged_observables", len(flagged))
        del out["flagged_observables"]
        out["n_flagged_observables"] = n
        if isinstance(out.get("summary"), str):
            out["summary"] = _unnamed(out["summary"], flagged, n)
    for key in ("summary", "advice"):
        if isinstance(out.get(key), str):
            out[key] = _LOF_POINTS.sub("(Holm p ", out[key])
    return out


def _unhinted(item: Any) -> bool:
    """Is ``item`` the :func:`identifiability.unhinted_ties` warning, as text or a message entry?"""
    text = item.get("message") if isinstance(item, dict) else item
    return isinstance(text, str) and text.startswith(UNHINTED_TIE)


def _unnamed(summary: str, flagged: list, n: int) -> str:
    """``summary`` with the observables its verdict flags (``data_access._named``) as their count."""
    names = ", ".join(str(o) for o in flagged[:_MAX_NAMED])
    if n > _MAX_NAMED:
        names += f" (+{n - _MAX_NAMED} more)"
    count = f"{n} observable{'' if n == 1 else 's'}"
    return summary.replace(f"residual structure in {names} (", f"residual structure in {count} (")
