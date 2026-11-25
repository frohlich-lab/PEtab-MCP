"""The message stack a ``make_changes`` result carries.

One call runs two very different steps — an EDIT (parse, reconcile, lint) and a FIT — and each
can raise, warn, or both. Before this the two halves reported through different channels: the
edit's advisories went into a flat ``warnings: [str]``, a rejection came back as ``error`` plus
an untyped ``issues`` list, and the lint's own WARNING-level findings were dropped entirely. A
caller reading one of those channels could not tell where a message came from, how serious it
was, or whether it had seen all of them.

So every diagnostic a call produces goes onto ONE ordered stack::

    [{"level": "warning", "source": "edit", "message": "..."}, ...]

* ``level`` is ``"error"`` or ``"warning"``. An error is why the call did not do what was asked;
  a warning is advisory and never blocks.
* ``source`` says which step produced it — ``"make_changes"`` (an argument the tool layer itself
  refused), ``"edit"``, ``"validation"`` (the PEtab/SBML lint), or ``"fit"``.
* The stack is in PIPELINE order: arguments, then the edit and its lint, then the fit. So reading
  it top to bottom replays what happened. The FIRST error is the one that stopped the call: only
  one step can fail, and any error after it is that rejection's own structured issue (see
  :func:`errors_from_exception`).
* An edit or lint warning the last committed result carried is not sent again: one line per source
  says how many are unchanged since which version, and the symbols they name
  (``operations._unrepeated``).

The stack is the ONLY channel: a result carries no flat list of warning texts beside it. The one
flat field left is ``error``, and it is not derived from the stack either — it is the message from
the step that refused, copied straight through by whoever built the result.

Deliberately its own module: :mod:`operations` builds the edit half and :mod:`server` builds the
tool-layer and fit halves, and neither should have to import the other to agree on the shape.
"""

from __future__ import annotations

from typing import Any

LEVEL_ERROR = "error"
LEVEL_WARNING = "warning"

#: Where a message came from, in pipeline order (see the module docstring).
SOURCE_TOOL = "make_changes"
SOURCE_EDIT = "edit"
SOURCE_VALIDATION = "validation"
SOURCE_FIT = "fit"


def message(level: str, source: str, text: Any) -> dict[str, str]:
    """One stack entry. ``text`` is stringified, so an exception can be passed directly."""
    return {"level": level, "source": source, "message": str(text)}


def error(source: str, text: Any) -> dict[str, str]:
    return message(LEVEL_ERROR, source, text)


def warning(source: str, text: Any) -> dict[str, str]:
    return message(LEVEL_WARNING, source, text)


def warnings_from(source: str, texts: Any) -> list[dict[str, str]]:
    """``texts`` (a list of strings, possibly ``None``) as warning entries; empties dropped."""
    return [warning(source, t) for t in (texts or []) if t]


def errors_from_exception(source: str, exc: BaseException) -> list[dict[str, str]]:
    """A raised :class:`~petab_mcp.errors.PetabMcpError` as the error that stopped the call,
    followed by each of its structured ``issues``.

    The issues are attributed to ``validation`` rather than to ``source``: they come from the
    lint/guardrail checks, not from the step that happened to be running, and keeping them
    distinguishable is the point of recording a source at all. An exception with no ``issues``
    (the common case) contributes exactly one entry.
    """
    out = [error(source, exc)]
    out += [error(SOURCE_VALIDATION, issue) for issue in (getattr(exc, "issues", None) or [])]
    return out
