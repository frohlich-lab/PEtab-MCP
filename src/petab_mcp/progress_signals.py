"""The numbers that tell the agent how well it is doing, and the switch that withholds them.

``create_server(progress_signals=False)`` withholds them from the agent: the fields that rank,
compare or judge a fit (:data:`KEYS`), the lines that state the best version or a score
difference against another version ("best is vM", a structure family's score against the best,
the nesting comparison), and the description text naming a withheld field. The rest of the prose
is unchanged. Each fit still reports its own nll, criterion values, k and chi2.

One thing upstream reads the setting: ``make_changes`` refuses a call without ``base_version``
on a HEAD that is not the best only with signals on (``server._implicit_base``), since the refusal
itself would say which version is not the best. Nothing else does. Text is written with its
signal bracketed (:func:`signal`, :func:`signal_or`), and a result's fields are what they always
were; ``server._ProgressSignals`` applies :func:`render` and :func:`strip` to everything the
server sends. So the stored record (version meta, fit events) keeps every field for whoever
analyses the session, and ``get_progress``, the client's early-stop and best-version probe, is
sent whole.
"""

from __future__ import annotations

import re
from typing import Any

_BRACKETS = "⦃⦄⦅⦆"
#: A progress signal, kept with signals on (:func:`signal`), and its stand-in, kept off
#: (:func:`signal_or`). Bodies hold no bracket of these, so each pass of :func:`render` takes the
#: innermost.
_SIGNAL = re.compile(f"⦃([^{_BRACKETS}]*)⦄")
_PLAIN = re.compile(f"⦅([^{_BRACKETS}]*)⦆")

#: Fields that rank, compare or judge a fit, cut at any depth with signals off. A fit result's
#: ``progress`` block and ``improvement`` block go whole; the rest are version-meta, fit-event,
#: ``list_versions`` and ``diff_versions`` fields.
KEYS = frozenset(
    {
        "progress",
        "improvement",
        "is_new_best",
        "new_best_declined",
        "improvement_vs_best_prior",
        "best_prior_version",
        "best_prior_criterion_value",
        "self_improvement",
        "objective_before",
        "chi2_gof_pvalue",
        "reduced_chi2",
        "residual_flagged",
        "neg_log_likelihood_delta",
    }
)

#: Tools whoever runs the session reads, never the agent: sent whole.
HARNESS_TOOLS = frozenset({"get_progress", "evaluate_validation_data"})


def signal(text: str) -> str:
    """``text``, a progress signal, bracketed for :func:`render`."""
    return f"⦃{text}⦄"


def signal_or(text: str, plain: str) -> str:
    """``text`` with progress signals on, ``plain`` (its wording without them) with them off."""
    return f"⦃{text}⦄⦅{plain}⦆"


def render(text: str, signals: bool) -> str:
    """``text`` with its progress signals kept (``signals``) or cut, and no brackets left."""
    if not any(b in text for b in _BRACKETS):
        return text
    keep, cut = (_SIGNAL, _PLAIN) if signals else (_PLAIN, _SIGNAL)
    while True:
        out = cut.sub("", keep.sub(lambda m: m[1], text))
        if out == text:
            return out
        text = out


def render_all(value: Any, signals: bool) -> Any:
    """:func:`render` on every string in a JSON-like ``value`` (dict values, list items)."""
    if isinstance(value, str):
        return render(value, signals)
    if isinstance(value, dict):
        return {k: render_all(v, signals) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [render_all(v, signals) for v in value]
    return value


def _cut(value: Any) -> Any:
    """``value`` rendered with signals off, without :data:`KEYS` or what rendered to nothing: a
    string field, a list item, or a ``messages`` entry."""
    if isinstance(value, str):
        return render(value, False)
    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            if k in KEYS:
                continue
            if (cut := _cut(v)) != "" or not v:
                out[k] = cut
        return out
    if isinstance(value, (list, tuple)):
        cut = [(_cut(v), v) for v in value]
        return [
            c
            for c, v in cut
            if (c != "" or not v)
            and not (isinstance(v, dict) and "message" in v and "message" not in c)
        ]
    return value


def strip(value: Any, signals: bool) -> Any:
    """A tool result as the agent gets it: :func:`render_all` with signals on, else :func:`_cut`."""
    return render_all(value, signals) if signals else _cut(value)
