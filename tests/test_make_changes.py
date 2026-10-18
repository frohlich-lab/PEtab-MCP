"""``make_changes``: the single editor, and the fit it always runs.

The lead's step budget used to be dominated by a ``set_model -> change_parameters -> fit``
cycle that was almost always one logical action, and each seam was a way to leave work
unscored — an edit nobody fit, an ``n_starts`` on a call that did not fit, a fit of a model
nobody had changed. ``make_changes`` removes the seams: ``antimony`` and/or ``parameters`` in
one transaction, then a fit, in one call. There is no fit-free edit and no standalone fit, so
"applied but never scored" only exists inside a single call, as the failure mode of that call.

The ``antimony``/``parameters`` transaction is tested directly against a ``PetabSession`` (no
engine needed) through the two internal operation functions the tool dispatches to; everything
about the fit is a server-layer concern, so those tests drive the in-process MCP server the way
``test_server.py`` does.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from pathlib import Path

from petab_mcp import operations as O
from petab_mcp import parameters as P
from petab_mcp.constants import DEFAULT_AUTO_FIT_STARTS
from petab_mcp.errors import ModelUpdateError, ParameterError
from petab_mcp.operations import OperationError
from petab_mcp.references import StubReferenceResolver
from petab_mcp.server import create_server
from petab_mcp.session import PetabSession

YAML_PATH = Path(__file__).parent / "data" / "toy_v2" / "problem.yaml"


_BASE = """model conversion_reaction
  compartment compartment_ = 1;
  species A in compartment_;
  species B in compartment_;
  A = A_init;
  B = B_init;
  A_init = 1.0;
  B_init = 0.0;
  k1 = 0.5;
  k2 = 0.1;
  r1: A => B; k1 * A;
  r2: B => A; k2 * B;
{extra}end
"""


def _antimony(*extra_lines: str) -> str:
    return _BASE.format(extra="".join(f"  {line}\n" for line in extra_lines))


def _records(session):
    return {r["parameterId"]: r for r in P.parameters_to_records(session.problem)}


# -- the re-author + parameters transaction ------------------------------------
def test_a_re_author_and_its_parameters_apply_atomically(session):
    """The re-author AND the bound/scale edits land in ONE edit — one provenance event, one
    pending edit — so the pair costs a single call instead of two."""
    res = O.set_model(
        session,
        _antimony("r3: A => ; k3 * A;"),
        justification="add a degradation step and give its rate a wider box",
        parameters=[
            {"parameter_id": "k1", "lower_bound": 1e-3, "upper_bound": 1e3},
            {"parameter_id": "sigma_B", "scale": "log10"},
        ],
    )
    assert res["applied"] is True and res["version"] is None
    assert res["staged_edit"]["operation"] == "make_changes"
    # the EDIT half on its own does not fit; the tool layer adds that (see the server tests below)
    assert "fit" not in res and "committed" not in res

    recs = _records(session)
    assert "k3" in recs  # the re-author took
    assert float(recs["k1"]["lowerBound"]) == 1e-3 and float(recs["k1"]["upperBound"]) == 1e3
    assert recs["sigma_B"]["parameterScale"] == "log10"

    edits = [e for e in session.provenance.recent_events() if e["type"] == "edit"]
    assert len(edits) == 1
    # provenance names the TOOL, not the internal function that ran: it is read by the agent,
    # which has only ever called make_changes.
    assert edits[-1]["operation"] == "make_changes"
    # the event records both halves of the change
    assert edits[-1]["args"]["parameters"][0]["parameter_id"] == "k1"
    assert edits[-1]["args"]["antimony_chars"] > 0


def test_parameters_can_target_a_newly_introduced_parameter(session):
    """The parameter edit runs AFTER the reconcile, so it can bound a symbol this very re-author
    introduced — the whole point of taking both in one call."""
    res = O.set_model(
        session,
        _antimony("r3: A => ; k3 * A;"),
        justification="bound the new degradation rate",
        parameters=[{"parameter_id": "k3", "lower_bound": 1e-4, "upper_bound": 1e1}],
    )
    assert res["applied"] is True
    recs = _records(session)
    assert float(recs["k3"]["lowerBound"]) == 1e-4 and float(recs["k3"]["upperBound"]) == 1e1


def test_a_parameters_rejection_reverts_the_re_author(session):
    """A rejected parameter entry reverts the MODEL too — nothing is half-applied."""
    before_model = session.problem.model.to_sbml_str()
    before_params = _records(session)

    with pytest.raises(OperationError, match="only one bound"):
        O.set_model(
            session,
            _antimony("r3: A => ; k3 * A;"),
            justification="bad bound entry",
            parameters=[{"parameter_id": "k1", "lower_bound": 1e-3}],
        )

    assert session.problem.model.to_sbml_str() == before_model
    assert _records(session) == before_params
    assert not [e for e in session.provenance.recent_events() if e["type"] == "edit"]


def test_a_parameters_entry_for_a_dropped_parameter_is_rejected(session):
    """An entry naming a parameter the re-author no longer declares fails loudly (the table is
    checked post-reconcile) rather than silently doing nothing."""
    with pytest.raises(ParameterError, match="Unknown parameter id 'k3'"):
        O.set_model(
            session,
            _antimony(),  # the base model has no k3
            justification="bound a parameter that does not exist",
            parameters=[{"parameter_id": "k3", "lower_bound": 1e-4, "upper_bound": 1e1}],
        )
    assert "k3" not in _records(session)


def test_a_re_author_without_parameters_edits_only_the_model(session):
    res = O.set_model(session, _antimony(), justification="plain re-author")
    assert res["applied"] is True and res["version"] is None
    edits = [e for e in session.provenance.recent_events() if e["type"] == "edit"]
    assert "parameters" not in edits[-1]["args"]


@pytest.mark.parametrize("empty", [None, []])
def test_empty_parameters_means_no_parameter_edit(session, empty):
    res = O.set_model(session, _antimony(), justification="no parameter edits", parameters=empty)
    assert res["applied"] is True
    assert (
        "parameters"
        not in [e for e in session.provenance.recent_events() if e["type"] == "edit"][-1]["args"]
    )


def test_a_malformed_parameters_argument_is_rejected(session):
    """The rejection names `make_changes`, the only thing the caller can have called — the two
    operation functions are an implementation detail of that one tool."""
    with pytest.raises(OperationError, match="make_changes: 'parameters'"):
        O.set_model(session, _antimony(), justification="x", parameters={"parameter_id": "k1"})


@pytest.mark.parametrize("bad", ["1e-3", None, True, float("nan"), float("inf")])
def test_a_non_numeric_bound_is_a_clean_rejection(session, bad):
    """Bounds are type-checked during normalization, not left to `update_parameters`' bare
    `float(value)`: a ValueError there is not a PetabMcpError, so it would escape the tool's error
    wrapper as an unstructured MCP failure instead of `applied: false`. (`None` trips the
    both-bounds-or-neither rule; the rest trip the numeric/finite checks.)"""
    with pytest.raises(OperationError):
        O.set_model(
            session,
            _antimony(),
            justification="x",
            parameters=[{"parameter_id": "k1", "lower_bound": bad, "upper_bound": 1e3}],
        )
    assert not [e for e in session.provenance.recent_events() if e["type"] == "edit"]


# -- the fit the tool always runs (server layer) -------------------------------
def _server(tmp_path, auto_fit=False):
    return create_server(
        str(YAML_PATH),
        provenance_dir=str(tmp_path / "prov"),
        resolver=StubReferenceResolver(),
        auto_fit=auto_fit,
    )


def _call(mcp, name, args):
    return asyncio.run(
        mcp.call_tool(name, args)
    ).structured_content  # ToolResult.structured_content


def _head(mcp):
    """HEAD version + pending-edit state. There is no `current_version` tool — everything it
    returned lives on `describe_problem` (only the version key is named differently)."""
    return _call(mcp, "describe_problem", {})


def _listed(result):
    """FastMCP wraps a list return in {"result": [...]}."""
    return result["result"] if isinstance(result, dict) and "result" in result else result


def test_one_call_produces_exactly_one_version(tmp_path):
    """Re-author + parameters + fit as ONE call: one edit event, one fit event, one version."""
    mcp = _server(tmp_path, auto_fit=True)
    res = _call(
        mcp,
        "make_changes",
        {
            "antimony": _antimony(),
            "justification": "re-author and retune k1, scored in one call",
            "parameters": [{"parameter_id": "k1", "lower_bound": 1e-3, "upper_bound": 1e3}],
            "n_starts": 2,
        },
    )
    # the fit is the result: flat, not nested, and named as get_fit_summary names it
    assert res["applied"] is True
    assert res["committed"] is True and res["version"] == 0
    assert res["neg_log_likelihood"] is not None and res["bic"] is not None
    assert "fit" not in res
    # and the edit's own bookkeeping is NOT in it — none of it is a score, and all of it would
    # ride on every call into a history nothing prunes
    assert not (
        {
            "reconcile",
            "validation",
            "parameter_diff",
            "initial_conditions",
            "justification",
            "references",
            "operation",
            "staged_edit",
        }
        & set(res)
    )

    head = _head(mcp)
    assert head["current_version"] == 0
    events = _listed(_call(mcp, "get_provenance", {}))
    assert len([e for e in events if e["type"] == "edit"]) == 1
    assert len([e for e in events if e["type"] == "fit"]) == 1
    # the one fit scored the one edit
    assert [e for e in events if e["type"] == "fit"][-1]["n_edits"] == 1
    recs = {r["parameterId"]: r for r in _listed(_call(mcp, "get_parameters", {}))}
    assert float(recs["k1"]["lowerBound"]) == 1e-3


def test_a_parameters_only_call_fits_too(tmp_path):
    """`antimony` is optional: a retuning with no structural change is still a scored version."""
    mcp = _server(tmp_path, auto_fit=True)
    res = _call(
        mcp,
        "make_changes",
        {
            "parameters": [{"parameter_id": "k1", "lower_bound": 1e-3, "upper_bound": 1e3}],
            "justification": "widen k1 and score it",
            "n_starts": 2,
        },
    )
    assert res["applied"] is True and res["committed"] is True and res["version"] == 0
    assert res["bic"] is not None


def test_a_call_that_changes_nothing_is_rejected_before_it_fits(tmp_path):
    """`antimony`, `model_edits` and `parameters` are each optional, but not all three: with none
    there is nothing to score, and the call would still spend a multistart fit."""
    mcp = _server(tmp_path, auto_fit=True)
    res = _call(mcp, "make_changes", {"justification": "nothing to do"})
    assert res["applied"] is False
    assert all(arg in res["error"] for arg in ("model_edits", "antimony", "parameters"))
    assert "fit" not in res and "committed" not in res
    assert _head(mcp)["current_version"] == -1


def test_dedup_skip_reports_no_change(tmp_path):
    """The de-dup still sees the same structure hash: re-authoring the identical structure is a
    no-op, not a second version."""
    mcp = _server(tmp_path, auto_fit=True)
    first = _call(
        mcp, "make_changes", {"antimony": _antimony(), "justification": "initial", "n_starts": 2}
    )
    assert first["committed"] is True and first["version"] == 0

    again = _call(
        mcp,
        "make_changes",
        {"antimony": _antimony(), "justification": "same structure again", "n_starts": 2},
    )
    assert again["applied"] is True  # the edit itself is always cheap
    assert again["committed"] is False
    assert again["no_change"] is True
    assert again["duplicate_of_version"] == 0
    assert again["version"] == 0
    # NOT reverted: the resulting model IS an existing version, byte-identical to it, so there is
    # nothing unscored to undo — which is why the no-op reports applied: true.
    head = _head(mcp)
    assert head["current_version"] == 0


def test_a_fit_failure_reverts_the_edit(tmp_path, monkeypatch):
    """A change is either scored or gone. An edit that applies and then fails to fit is UNDONE —
    both halves of it, model and parameter table — so the working model is always one a version
    records, and there is never an edit waiting for a fit that no tool can run."""

    def _boom(self, *a, **k):
        raise ModelUpdateError("boom")

    monkeypatch.setattr(PetabSession, "fit", _boom)
    mcp = _server(tmp_path)
    before = {r["parameterId"]: r for r in _listed(_call(mcp, "get_parameters", {}))}

    res = _call(
        mcp,
        "make_changes",
        {
            "antimony": _antimony("r3: A => ; k3 * A;"),
            "justification": "an edit whose fit blows up",
            "parameters": [{"parameter_id": "k1", "lower_bound": 1e-3, "upper_bound": 1e3}],
        },
    )
    assert res["applied"] is False  # applied, then undone -> not in place
    assert res["committed"] is False and res["version"] == -1
    assert "boom" in res["error"]
    assert "REVERTED" in res["message"]

    # the model AND the bounds are back where they were
    head = _head(mcp)
    assert head["current_version"] == -1
    after = {r["parameterId"]: r for r in _listed(_call(mcp, "get_parameters", {}))}
    assert "k3" not in after
    assert float(after["k1"]["lowerBound"]) == float(before["k1"]["lowerBound"])
    # the failure is logged like any other fit rejection, and the revert is itself in the trace:
    # the edit really did happen, so the append-only log shows edited-then-reverted rather than
    # quietly losing the attempt.
    assert any("boom" in (e.get("message") or "") for e in _listed(_call(mcp, "get_errors", {})))
    events = _listed(_call(mcp, "get_provenance", {}))
    assert [e["type"] for e in events][-2:] == ["edit", "revert"]
    assert "boom" in events[-1]["reason"]


def test_a_refused_fit_forwards_its_own_error_instead_of_prescribing_a_repair(
    tmp_path, monkeypatch
):
    """The identical-structure loop guard lands in the same "not scored" branch as a broken model,
    but its error says the OPPOSITE (stop re-fitting, make a different change). The top-level message
    must forward that rather than tell the agent to repair and re-fit — the agent reads the message
    first, and "re-fit" walks straight back into the loop the guard exists to break."""

    def _refuse(self, *a, **k):
        raise ModelUpdateError(
            "This same model structure has been fit 4 times now. Stop re-fitting it: "
            "make a genuinely different change."
        )

    monkeypatch.setattr(PetabSession, "fit", _refuse)
    mcp = _server(tmp_path)
    res = _call(
        mcp,
        "make_changes",
        {"antimony": _antimony(), "justification": "re-author a seen structure"},
    )

    assert res["applied"] is False and res["committed"] is False
    assert "genuinely different change" in res["message"]  # the refusal's own instruction survives
    assert "repair the model" not in res["message"]
    assert "REVERTED" in res["message"]  # ... and the state is reported


def test_a_rejected_edit_never_fits(tmp_path):
    """A REJECTED edit must not run a fit — the result is the plain rejection shape, with no `fit`
    key to misread."""
    mcp = _server(tmp_path)
    res = _call(
        mcp,
        "make_changes",
        {
            "antimony": _antimony(),
            "justification": "",  # empty justification is rejected
        },
    )
    assert res["applied"] is False and "fit" not in res and "committed" not in res
    assert _head(mcp)["current_version"] == -1


def test_the_multistart_default_and_the_starts_actually_used_are_reported(tmp_path):
    """`n_starts` REPLACES the server's multistart default, so both numbers are exposed rather
    than left to be inferred: `describe_problem.fit_n_starts_default` before the call, and the fit
    result's own `n_starts` after it."""
    mcp = _server(tmp_path, auto_fit=True)
    default = _call(mcp, "describe_problem", {})["fit_n_starts_default"]
    assert default == DEFAULT_AUTO_FIT_STARTS

    # omitted -> the default was used
    res = _call(
        mcp,
        "make_changes",
        {"antimony": _antimony(), "justification": "score with the default starts"},
    )
    assert res["n_starts"] == default

    # given -> exactly that count, not default + n_starts
    res = _call(
        mcp,
        "make_changes",
        {
            "parameters": [{"parameter_id": "k1", "lower_bound": 1e-3, "upper_bound": 1e3}],
            "justification": "score with an explicit count",
            "n_starts": default + 2,
        },
    )
    assert res["n_starts"] == default + 2


def test_a_harder_fit_of_an_unchanged_model_is_not_skipped(tmp_path):
    """The structure hash and the setup signature both ignore the multistart count, so a de-dup
    skip would silently return the WEAKER stored metrics for a deliberate "fit harder". A fit with
    more starts than HEAD's is a genuinely different search and must run; the same or fewer is the
    redundant refit the de-dup exists to skip. Re-authoring the identical structure is how a
    caller asks for that harder search now — the edit half is cheap, and the fit is the decision."""
    mcp = _server(tmp_path, auto_fit=True)
    same_model = {"antimony": _antimony(), "justification": "fit this structure"}

    first = _call(mcp, "make_changes", {**same_model, "n_starts": 2})
    assert first["committed"] is True and first["n_starts"] == 2

    # same count -> still the no-op, and it reports the starts its metrics came from
    same = _call(mcp, "make_changes", {**same_model, "n_starts": 2})
    assert same["no_change"] is True and same["n_starts"] == 2

    # more starts -> a real fit, with a warning explaining why it wasn't skipped
    harder = _call(mcp, "make_changes", {**same_model, "n_starts": 4})
    assert harder["committed"] is True and harder["version"] == 1
    assert harder["n_starts"] == 4
    assert any(
        "multistarts against its 2" in m["message"]
        for m in harder["messages"]
        if m["level"] == "warning"
    )

    # fewer starts than the new HEAD -> back to a no-op, not a weaker refit
    weaker = _call(mcp, "make_changes", {**same_model, "n_starts": 3})
    assert weaker["no_change"] is True and _head(mcp)["current_version"] == 1


def test_a_bad_n_starts_is_rejected_before_the_edit_applies(tmp_path):
    """A bad n_starts is validated at the tool boundary, not inside session.fit: rejecting it only
    after the edit had applied would leave an unscored edit behind for an argument typo. (A
    NON-integer never gets this far — the tool schema's `int | None` rejects it outright; a
    negative one parses fine and is what this guard is for.)"""
    mcp = _server(tmp_path)
    res = _call(
        mcp,
        "make_changes",
        {
            "antimony": _antimony("r3: A => ; k3 * A;"),
            "justification": "a good edit with a typo'd n_starts",
            "n_starts": -1,
        },
    )

    assert res["applied"] is False and "n_starts" in res["error"]
    assert "fit" not in res  # the fit never ran
    head = _head(mcp)
    assert head["current_version"] == -1
    assert "k3" not in {r["parameterId"] for r in _listed(_call(mcp, "get_parameters", {}))}
    # the rejection is auditable, including the refused VALUE, so the log says which argument was
    # wrong rather than only repeating the message
    logged = [
        e for e in _listed(_call(mcp, "get_errors", {})) if "n_starts" in (e.get("message") or "")
    ]
    assert [e["args"] for e in logged] == [{"n_starts": -1}]


def test_a_fit_that_breaks_after_snapshotting_reports_the_new_version(tmp_path, monkeypatch):
    """The crash path `edit_then_fit` degrades runs PAST the version snapshot, which consumes the
    pending edits and advances HEAD. Reverting there would throw away a real version, and telling
    the agent to send the change again would walk it into a de-dup no-op, so this case keeps the
    change and reports where HEAD landed."""
    real_fit = PetabSession.fit

    def _fit_then_break(self, *a, **k):
        real_fit(self, *a, **k)  # snapshots the version
        raise RuntimeError("late boom")  # ... then breaks, as a result-block bug would

    monkeypatch.setattr(PetabSession, "fit", _fit_then_break)
    mcp = _server(tmp_path, auto_fit=True)

    res = _call(
        mcp,
        "make_changes",
        {"antimony": _antimony(), "justification": "fit breaks after the snapshot", "n_starts": 2},
    )

    assert res["applied"] is True and res["committed"] is False
    assert "late boom" in res["error"]
    # HEAD really did advance, so say so instead of claiming the change is unscored
    assert res["version"] == 0 == _head(mcp)["current_version"]
    assert "do NOT re-send" in res["message"] and "REVERTED" not in res["message"]


def test_a_snapshot_that_lands_before_the_failure_is_kept_not_reverted(tmp_path, monkeypatch):
    """A failure after the snapshot must not throw the version away: HEAD decides whether the
    change was scored, and there is no second signal to disagree with it."""

    def _snapshot_then_break(self, *a, **k):
        self._snapshot_version(meta={"fit_status": "fit"}, sim=None)
        raise RuntimeError("provenance write failed")  # ... after the version is on disk

    monkeypatch.setattr(PetabSession, "fit", _snapshot_then_break)
    mcp = _server(tmp_path)

    res = _call(
        mcp,
        "make_changes",
        {
            "antimony": _antimony("r3: A => ; k3 * A;"),
            "justification": "snapshot lands, event log does not",
        },
    )

    assert res["applied"] is True and res["committed"] is False
    assert res["version"] == 0  # HEAD moved, and that is what decides
    assert "do NOT re-send" in res["message"]
    head = _head(mcp)
    # the change is still there — it is the version — and nothing is left flagged unscored
    assert head["current_version"] == 0
    assert "k3" in {r["parameterId"] for r in _listed(_call(mcp, "get_parameters", {}))}


def test_a_fit_crash_is_degraded_to_a_failed_fit(tmp_path, monkeypatch):
    """An UNEXPECTED exception from the fit must not escape as a raw tool error: `run_fit`
    re-raises anything that is not a PetabMcpError, and letting that out would surface the call as
    a transport-level failure rather than a result the agent can read. It is degraded to a failed
    fit — which then takes the same revert path as any other."""

    def _crash(self, *a, **k):
        raise RuntimeError("kaboom")

    monkeypatch.setattr(PetabSession, "fit", _crash)
    mcp = _server(tmp_path)

    res = _call(
        mcp,
        "make_changes",
        {"antimony": _antimony("r3: A => ; k3 * A;"), "justification": "an edit whose fit crashes"},
    )
    assert res["applied"] is False and res["committed"] is False
    assert "kaboom" in res["error"]
    head = _head(mcp)
    assert head["current_version"] == -1


def test_there_is_no_way_to_edit_without_fitting(tmp_path):
    """The invariant the collapse buys, asserted directly: EVERY call carries a committed verdict
    and leaves the working model EQUAL to a stored version — whether it succeeds, is rejected
    before it applies, or applies and is then reverted. A future `fit=false`-shaped escape hatch —
    an argument, a default, a second tool — brings back the state this removed, an edit nobody
    scored.

    Checked by comparing the LIVE model against HEAD's stored model."""
    mcp = _server(tmp_path)
    for args in (
        {"antimony": _antimony(), "justification": "structural"},
        {
            "parameters": [{"parameter_id": "k1", "lower_bound": 1e-3, "upper_bound": 1e3}],
            "justification": "bounds only",
        },
        {"antimony": "not antimony at all", "justification": "rejected outright"},
        {"justification": "nothing to change"},
    ):
        res = _call(mcp, "make_changes", args)
        assert "applied" in res, args
        if res["applied"]:
            assert "committed" in res, args
        # ...and the working model IS a stored version, not an unscored intermediate state
        head = _call(mcp, "describe_problem", {})["current_version"]
        stored = _call(mcp, "get_version", {"version": head})["model_antimony"]
        assert _call(mcp, "get_model", {})["result"] == stored, args


# -- the message stack ---------------------------------------------------------


def test_the_message_stack_carries_the_edit_and_the_fit_together(tmp_path):
    """One call runs two steps that can each warn, and they used to report through different
    channels — the edit's advisories into a flat `warnings`, a failure into `error`. `messages` is
    the single ordered stack: every entry tagged with its level and the step it came from, and no
    flat copy beside it to drift from it."""
    mcp = _server(tmp_path, auto_fit=True)
    _call(mcp, "make_changes", {"antimony": _antimony(), "justification": "initial", "n_starts": 2})
    # A re-author that drops a previously-estimated parameter warns — the edit half's advisory.
    res = _call(
        mcp,
        "make_changes",
        {
            "antimony": _BASE.format(extra="")
            .replace("  k2 = 0.1;\n", "")
            .replace("  r2: B => A; k2 * B;\n", ""),
            "justification": "drop the reverse reaction",
            "n_starts": 2,
        },
    )

    stack = res["messages"]
    assert all({"level", "source", "message"} <= set(m) for m in stack)
    assert all(m["level"] in ("error", "warning") for m in stack)
    assert any(m["source"] == "edit" and "k2" in m["message"] for m in stack), stack
    # `warnings` is deliberately ABSENT from the payload: the stack is the only channel, so there
    # is no flat copy of the warning texts to drift from it.
    assert "warnings" not in res
    assert [m["message"] for m in stack if m["level"] == "warning"]


def test_a_rejected_change_reports_the_error_on_the_stack(tmp_path):
    """A rejection is an `error` entry, sourced to the step that produced it — so a caller reads
    one place whether the call warned, failed, or both."""
    mcp = _server(tmp_path, auto_fit=True)
    res = _call(mcp, "make_changes", {"justification": "nothing to do"})
    assert res["applied"] is False
    assert [(m["level"], m["source"]) for m in res["messages"]] == [("error", "make_changes")]
    assert res["messages"][0]["message"] == res["error"]


def test_a_failed_fit_puts_its_error_on_the_stack(tmp_path, monkeypatch):
    """The fit half's failure is sourced to `fit`, next to whatever the edit half already said."""

    def _boom(self, *a, **k):
        raise ModelUpdateError("boom")

    monkeypatch.setattr(PetabSession, "fit", _boom)
    mcp = _server(tmp_path)
    res = _call(
        mcp,
        "make_changes",
        {"antimony": _antimony("r3: A => ; k3 * A;"), "justification": "an edit whose fit fails"},
    )
    assert res["applied"] is False
    errors = [m for m in res["messages"] if m["level"] == "error"]
    assert [m["source"] for m in errors] == ["fit"]
    assert "boom" in errors[0]["message"]


# -- empty models --------------------------------------------------------------


def test_a_model_with_no_dynamics_is_rejected(tmp_path):
    """Nothing to simulate is nothing to score — and an empty model segfaults AMICI's importer,
    a crash no caller can catch. Refused before it can be committed."""
    mcp = _server(tmp_path, auto_fit=True)
    res = _call(
        mcp,
        "make_changes",
        {
            "antimony": "model m\n  species A;\n  A = 1;\nend",
            "justification": "declare a species but no reaction",
        },
    )
    assert res["applied"] is False
    assert "no DYNAMICS" in res["error"]
    assert any(m["level"] == "error" for m in res["messages"])
    # ...and the working model is untouched
    head = _head(mcp)
    assert head["current_version"] == -1 and head["model_authored"] is True


def test_the_single_editor_carries_the_whole_change(tmp_path):
    """The prompt promises ONE call can author the model, set bounds, and score the result.

    That used to be a claim about two schemas agreeing — and it was false: `change_parameters`
    took `changes=` while `set_model` took `parameters=`, so a lead following the prompt sent the
    wrong kwarg and lost a step to a tool-arg rejection. Collapsing the editors removes the class
    of bug rather than re-pinning it: there is one schema, so there is nothing to diverge. What is
    left to assert is that the one schema really carries every part of the promise — and that no
    `fit` flag came back, since fitting is unconditional now and an optional flag would let the
    two halves come apart again.
    """
    mcp = create_server(
        str(Path(__file__).parent / "data" / "toy_v2" / "problem.yaml"),
        provenance_dir=str(tmp_path / "prov"),
        resolver=StubReferenceResolver(),
    )
    got = {
        t.name: set(t.parameters.get("properties") or {}) for t in asyncio.run(mcp.list_tools())
    }["make_changes"]
    # The BASELINE promise, always present: author the model, set bounds, justify, cite, score.
    core = {"antimony", "parameters", "justification", "references", "n_starts"}
    assert core <= got, f"the one editor dropped part of its promise: {sorted(core - got)}"
    # Plus the structured model route, and EXACTLY that — a stray optional argument here is how
    # the two halves of the promise came apart before.
    extra = {"model_edits", "base_version"}
    assert got == core | extra, f"unexpected make_changes args: {sorted(got - (core | extra))}"
    # and NO fit flag: fitting is unconditional, and an optional flag would let a change be
    # committed without a score.
    assert not ({"fit", "do_fit", "skip_fit", "refit"} & got)


# -- the `statements` route ------------------------------------------------------
# The toy's two species as the entity map binds them; ``{extra}`` splices more entries.
_BASE_ENTITY_MAP = '{{"A": {{"name": "A"}}, "B": {{"name": "B"}}{extra}}}'


def _statement_row(type_="PopulationTransition", policy="mass_action"):
    return {
        "statement": {"type": type_, "source": {"name": "A"}, "target": {"name": "B"}},
        "policy": policy,
        "prefix": "conv",
    }


def _statements_server(tmp_path, extra=""):
    entity_map = tmp_path / "entities.json"
    entity_map.write_text(_BASE_ENTITY_MAP.format(extra=extra))
    return create_server(
        str(YAML_PATH),
        provenance_dir=str(tmp_path / "prov"),
        resolver=StubReferenceResolver(),
        auto_fit=False,
        entity_map=str(entity_map),
    )


def _make_changes_schema(mcp):
    tools = asyncio.run(mcp.list_tools())
    return next(t for t in tools if t.name == "make_changes").parameters["properties"]


def test_statements_route_exists_only_with_an_entity_map(tmp_path):
    """Without a map nothing binds a statement-built model to the observables, so offering the
    route would offer a call that always fails."""
    assert "statements" not in _make_changes_schema(_server(tmp_path / "plain"))
    assert "statements" in _make_changes_schema(_statements_server(tmp_path))


def test_statements_commit_records_what_was_asserted(tmp_path):
    """The row and its Emission are stored beside the generated source: a later ``revise``
    addresses the row, and neither is recoverable from the expanded model."""
    mcp = _statements_server(tmp_path)
    out = _call(
        mcp, "make_changes", {"justification": "A becomes B", "statements": [_statement_row()]}
    )
    assert out["applied"] is True and out["committed"] is True
    assert "carry no grounding" in out["messages"][0]["message"]
    (stored,) = json.loads((tmp_path / "prov" / "versions" / "v0" / "statements.json").read_text())
    assert stored["emission"]["parameters"] == ["conv_k"]


def test_a_library_gap_is_its_own_rejection(tmp_path):
    """A pair the library does not hold must stay countable apart from a malformed call — it is
    the to-do list for the library."""
    row = _statement_row(policy="no_such_mechanism")
    out = _call(
        _statements_server(tmp_path), "make_changes", {"justification": "x", "statements": [row]}
    )
    assert out["applied"] is False and out["error_cause"] == "library_gap"


def test_an_unreadable_statement_is_refused_not_dropped(tmp_path):
    """INDRA's reader drops what it cannot parse with only a logged warning; a silently shorter
    model would then be fitted and committed."""
    row = _statement_row(type_="NotAStatement")
    out = _call(
        _statements_server(tmp_path), "make_changes", {"justification": "x", "statements": [row]}
    )
    assert out["applied"] is False and out["error_cause"] == "unreadable_statement"


def test_a_repeated_prefix_is_refused_before_assembly(tmp_path):
    """Two rows minting the same parameter names clash inside PySB, whose error is not one the
    tool boundary catches."""
    rows = [_statement_row(), _statement_row()]
    out = _call(
        _statements_server(tmp_path), "make_changes", {"justification": "x", "statements": rows}
    )
    assert out["applied"] is False and out["error_cause"] == "duplicate_prefix"


def test_statement_species_take_the_problems_own_initial_conditions(tmp_path):
    """The first version minted a fixed one-unit initial amount per species, so Crauste's
    populations all started at one cell and its shipped `Naive_init` (8090) went unused. The
    toy ships `A_init` and `B_init` fixed; both must survive, and no other IC row may appear."""
    mcp = _statements_server(tmp_path)
    _call(mcp, "make_changes", {"justification": "A becomes B", "statements": [_statement_row()]})
    ids = {r["parameterId"] for r in _call(mcp, "get_parameters", {})["result"]}
    assert {"A_init", "B_init"} <= ids
    assert not {i for i in ids if i.endswith("_0")}


_SATURATING = {
    "statement_type": "PopulationTransition",
    "policy": "saturating",
    "based_on": "mass_action",
    "rate": "vmax / (K + source)",
    "parameters": ["vmax", "K"],
}


def test_a_mechanism_the_agent_writes_stays_in_the_run_library(tmp_path):
    """The library is not closed: an entry written in one call serves every later call without
    being resent, and the version that used it records its definition."""
    mcp = _statements_server(tmp_path)
    row = _statement_row(policy="saturating")
    args = {"justification": "saturating", "statements": [row], "mechanisms": [_SATURATING]}
    assert _call(mcp, "make_changes", args)["committed"] is True
    (stored,) = json.loads((tmp_path / "prov" / "versions" / "v0" / "statements.json").read_text())
    assert stored["mechanism"]["rate"] == "vmax / (K + source)"
    row["prefix"] = "conv2"  # a different structure, so the de-dup does not refuse the refit
    again = _call(mcp, "make_changes", {"justification": "reuse", "statements": [row]})
    assert again["applied"] is True


def test_a_written_mechanism_is_not_redefined_in_place(tmp_path):
    """Earlier versions were built with the first definition; changing it under the same name
    would change what their records mean."""
    mcp = _statements_server(tmp_path)
    row = _statement_row(policy="saturating")
    _call(
        mcp,
        "make_changes",
        {"justification": "x", "statements": [row], "mechanisms": [_SATURATING]},
    )
    changed = {**_SATURATING, "rate": "vmax"}
    out = _call(
        mcp, "make_changes", {"justification": "y", "statements": [row], "mechanisms": [changed]}
    )
    assert out["applied"] is False and out["error_cause"] == "mechanism_redefined"


def _concurrently(monkeypatch, mcp, *calls):
    """The ``make_changes`` calls, each on its own thread, each pausing after it builds its
    statements until every other has built too (or 3 s pass)."""
    import threading

    barrier = threading.Barrier(len(calls), timeout=3)
    build = O.statements_to_antimony

    def paused(*args, **kwargs):
        built = build(*args, **kwargs)
        try:
            barrier.wait()
        except threading.BrokenBarrierError:  # the other call waits on the session lock
            pass
        return built

    monkeypatch.setattr(O, "statements_to_antimony", paused)
    results = [None] * len(calls)

    def run(i):
        results[i] = _call(mcp, "make_changes", calls[i])

    threads = [threading.Thread(target=run, args=(i,)) for i in range(len(calls))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return results


def test_concurrent_calls_keep_every_mechanism_they_write(tmp_path, monkeypatch):
    """Both calls built from the registry as it was before either committed: of two written
    mechanisms one was lost, and two definitions of one name both committed."""
    mcp = _statements_server(tmp_path)
    other = {**_SATURATING, "policy": "hill", "rate": "vmax * source / (K + source)"}
    calls = [
        {"justification": m["policy"], "statements": [_statement_row(policy=m["policy"])],
         "mechanisms": [m]}
        for m in (_SATURATING, other)
    ]
    assert all(out["applied"] for out in _concurrently(monkeypatch, mcp, *calls))
    rows = [{**_statement_row(policy=p), "prefix": p} for p in ("saturating", "hill")]
    both = _call(mcp, "make_changes", {"justification": "both", "statements": rows})
    assert both["applied"] is True, both.get("error")

    (tmp_path / "redefined").mkdir()
    mcp = _statements_server(tmp_path / "redefined")
    changed = {**_SATURATING, "rate": "vmax"}
    calls = [{**call, "mechanisms": [m]} for call, m in zip(calls, (_SATURATING, changed))]
    calls[1]["statements"] = calls[0]["statements"]
    outs = _concurrently(monkeypatch, mcp, *calls)
    assert sorted(bool(out["applied"]) for out in outs) == [False, True]
    assert [out.get("error_cause") for out in outs if not out["applied"]] == ["mechanism_redefined"]
