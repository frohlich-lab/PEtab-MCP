"""``make_changes``: the single editor, and the fit it always runs.

An agent's step budget used to be dominated by a ``set_model -> change_parameters -> fit``
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
import re
import shutil

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
    # applied: the change gives version 0's model, and version 0 is the working model
    head = _head(mcp)
    assert head["current_version"] == 0


def _ruled(rate: str) -> str:
    """The toy with r1's rate written the way the statements route writes every rate: an
    assignment rule the rate law multiplies, so the rate law itself never changes."""
    return _antimony(f"k_eff := {rate};").replace("r1: A => B; k1 * A;", "r1: A => B; k_eff * A;")


def _set_rate(rate: str) -> dict:
    return {"op": "set_assignment", "id": "k_eff", "antimony": rate}


def test_a_rate_only_edit_is_fitted_not_deduplicated(tmp_path):
    """REGRESSION: the de-dup's fingerprint hashed each rate law's form, and a rate changed through
    the rule that rate law multiplies left every form as it was. The change came back as a duplicate
    of HEAD and was never fitted."""
    mcp = _server(tmp_path, auto_fit=True)
    first = _call(
        mcp, "make_changes", {"antimony": _ruled("k1"), "justification": "linear", "n_starts": 2}
    )
    assert first["committed"] is True
    res = _call(
        mcp,
        "make_changes",
        {"model_edits": [_set_rate("k1 / (1 + A)")], "justification": "saturating", "n_starts": 2},
    )
    assert res["committed"] is True and res["version"] == 1, res["message"]


def test_a_readout_only_edit_is_fitted_not_deduplicated(tmp_path):
    """The same blind spot for a readout: a `readout_*` mapping is an assignment rule, so a model
    that changed only what the data measure fingerprinted as its predecessor."""
    import shutil

    problem = tmp_path / "problem"
    shutil.copytree(YAML_PATH.parent, problem)
    (problem / "observables.tsv").write_text(
        "observableId\tobservableFormula\tnoiseFormula\nobs_B\treadout_B\tsigma_B\n"
    )
    mcp = create_server(
        str(problem / "problem.yaml"),
        provenance_dir=str(tmp_path / "prov"),
        resolver=StubReferenceResolver(),
        auto_fit=True,
    )
    first = _call(
        mcp,
        "make_changes",
        {"antimony": _antimony("readout_B := B;"), "justification": "B", "n_starts": 2},
    )
    assert first["committed"] is True, first["message"]
    res = _call(
        mcp,
        "make_changes",
        {
            "model_edits": [{"op": "set_assignment", "id": "readout_B", "antimony": "A + B"}],
            "justification": "the antibody sees A too",
            "n_starts": 2,
        },
    )
    assert res["committed"] is True and res["version"] == 1, res["message"]


def test_a_duplicate_leaves_the_session_as_it_was(tmp_path):
    """REGRESSION: the de-dup no-op kept the change's text and authored values as the working
    model, though no version holds them, so the next edit built on a model nothing had accepted.
    A duplicate now leaves the session on the version it duplicates, exactly as stored."""
    mcp = _server(tmp_path, auto_fit=True)
    _call(mcp, "make_changes", {"antimony": _antimony(), "justification": "v0", "n_starts": 2})
    model = _listed(_call(mcp, "get_model", {"format": "antimony"}))
    parameters = _listed(_call(mcp, "get_parameters", {}))

    # the same equations under another reaction id and another starting value
    renamed = _antimony().replace("r2:", "back:").replace("k1 = 0.5", "k1 = 0.7")
    res = _call(mcp, "make_changes", {"antimony": renamed, "justification": "dup", "n_starts": 2})
    assert res["no_change"] is True and res["duplicate_of_version"] == 0

    assert _listed(_call(mcp, "get_model", {"format": "antimony"})) == model
    assert _listed(_call(mcp, "get_parameters", {})) == parameters
    # ...and the source the next structured edit applies to is version 0's, not the duplicate's
    stale = _call(
        mcp,
        "make_changes",
        {"model_edits": [{"op": "remove_reaction", "id": "back"}], "justification": "x"},
    )
    assert stale["applied"] is False and "back" in stale["error"]


def test_a_revert_to_an_earlier_model_is_refused_as_that_version(tmp_path):
    """REGRESSION: with rates invisible to the fingerprint, going back to an earlier rate was
    refused as a duplicate of HEAD, which holds a different one. The version it returns to stands
    for it, so it is refused naming that version and HEAD stays put; under new bounds it is a new
    search, fitted and named as a revisit."""
    mcp = _server(tmp_path, auto_fit=True)
    _call(mcp, "make_changes", {"antimony": _ruled("k1"), "justification": "v0", "n_starts": 2})
    _call(
        mcp,
        "make_changes",
        {"model_edits": [_set_rate("k1 / (1 + A)")], "justification": "v1", "n_starts": 2},
    )
    back = {"model_edits": [_set_rate("k1")], "justification": "back", "n_starts": 2}
    refused = _call(mcp, "make_changes", back)
    assert refused["applied"] is False and "identical to v0" in refused["error"]
    assert _head(mcp)["current_version"] == 1

    bounds = [{"parameter_id": "k1", "lower_bound": 1e-4, "upper_bound": 1e3}]
    wider = _call(mcp, "make_changes", {**back, "parameters": bounds})
    assert wider["committed"] is True and wider["version"] == 2, wider["message"]
    assert any("version(s) [0]" in m["message"] for m in wider["messages"])


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
    """An agent is told ONE call can author the model, set bounds, and score the result.

    That used to be a claim about two schemas agreeing — and it was false: `change_parameters`
    took `changes=` while `set_model` took `parameters=`, so an agent acting on it sent the
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


def _statement_row(type_="PopulationTransition"):
    """A becoming B as a JSON row, its prefix `conv`: the statement, its binding given apart."""
    return {
        "statement": {"type": type_, "source": {"name": "A"}, "target": {"name": "B"}},
        "prefix": "conv",
    }


def _statements_server(tmp_path, extra="", yaml_path=YAML_PATH):
    entity_map = tmp_path / "entities.json"
    entity_map.write_text(_BASE_ENTITY_MAP.format(extra=extra))
    return create_server(
        str(yaml_path),
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


def _description(mcp):
    tools = asyncio.run(mcp.list_tools())
    return next(t for t in tools if t.name == "make_changes").description


def test_the_routes_text_names_statements_only_where_it_is_offered(tmp_path):
    """It required one of `antimony` / `model_edits` / `parameters` on a server whose third
    route is `statements`; without the route the description is the one it was before."""
    from petab_mcp.server import _MAKE_CHANGES_DOC

    assert _description(_server(tmp_path / "plain")) == _MAKE_CHANGES_DOC
    required = _description(_statements_server(tmp_path)).split("at least one of ")[1]
    assert "`statements`" in required.split(" is required.")[0]


def test_an_entity_map_documents_and_accepts_no_text_route(tmp_path):
    """A client that builds models from statements alone stripped `antimony`/`model_edits` from
    the schema itself and wrote its own description, so the agent read none of the server's
    (frohlich-lab/PEtab-MCP#40). A server now offers one family of routes, never both: an
    entity map offers `statements` in place of the text routes, with no flag to keep them."""
    mcp = _statements_server(tmp_path)
    args = set(_make_changes_schema(mcp))
    assert "statements" in args and "statement_edits" in args
    assert not {"antimony", "model_edits"} & args
    description = _description(mcp)
    assert "`antimony`" not in description and "`model_edits`" not in description
    assert "`statements` authors the WHOLE model" in description
    with pytest.raises(Exception, match="antimony"):
        asyncio.run(
            mcp.call_tool("make_changes", {"justification": "x", "antimony": _antimony()})
        )


def _agent_facing_text(mcp):
    """Everything a client is shown before it calls anything: the instructions, every tool's
    description and argument schema, and every listed skill in full."""
    import json

    from petab_mcp import skills as skills_mod

    tools = asyncio.run(mcp.list_tools())
    names = {t.name for t in tools}
    skills = skills_mod.load_skills(skills_mod.default_skills_dir()) if "get_skill" in names else []
    return "\n".join(
        [mcp.instructions or ""]
        + [f"{t.description}\n{json.dumps(t.parameters)}" for t in tools]
        + [f"{s.description}\n{s.text}" for s in skills]
    )


def test_no_text_offers_the_statements_route_where_it_does_not_exist(tmp_path):
    """`get_model` offered its 'statements' format on a server without the route, where the
    format exists for no model the server can build."""
    import re

    route = re.compile(r"(?i)\bstatements\b|statement_edits|list_mechanisms|\bindra\b")
    plain = _agent_facing_text(_server(tmp_path / "plain"))
    assert not route.findall(plain), sorted(set(route.findall(plain)))
    tools = asyncio.run(_statements_server(tmp_path).list_tools())
    assert "'statements'" in next(t for t in tools if t.name == "get_model").description
    with pytest.raises(Exception, match="Unknown model format 'statements'"):
        asyncio.run(_server(tmp_path / "p2").call_tool("get_model", {"format": "statements"}))


def test_the_refusals_name_statements_only_where_it_is_offered(tmp_path):
    """Two routes at once and no route at all were each refused naming `model_edits` and
    `antimony` only, where `statements` was the route."""
    plain, offered = _server(tmp_path / "plain"), _statements_server(tmp_path)

    def error(mcp, args):
        return _call(mcp, "make_changes", {"justification": "x", **args})["error"]

    edits = [{"op": "remove_reaction", "id": "r1"}]
    assert "`statements`" not in error(plain, {"antimony": _antimony(), "model_edits": edits})
    both = {"statements": [_statement_row()], "statement_edits": [{"op": "remove_row", "prefix": "conv"}]}
    said = error(offered, both).split(")", 1)[1]  # past the routes given
    assert "`statements`" in said and "`model_edits`" not in said
    assert "`statements`" not in error(plain, {})
    assert "`statements`" in error(offered, {})


def test_the_types_without_indra_default_are_read_off_the_registry(monkeypatch):
    """The paragraph said every INDRA statement type has `indra_default`; 6 of 50 had none, and a
    row trying one came back a library_gap."""
    from petab_mcp import mechanisms as M
    from petab_mcp.server import _make_changes_statements_doc

    def without():
        doc = _make_changes_statements_doc()
        return doc.split(f" has `{M.INDRA_DEFAULT}`")[0].split("Every INDRA statement type")[1]

    assert "Phosphorylation" not in without()
    monkeypatch.delitem(M.REGISTRY, ("Phosphorylation", M.INDRA_DEFAULT))
    assert "Phosphorylation" in without()


def test_statements_commit_records_what_was_asserted(tmp_path):
    """The row and its Emission are stored beside the generated source: a later ``revise``
    addresses the row, and neither is recoverable from the expanded model."""
    mcp = _statements_server(tmp_path)
    out = _call(
        mcp,
        "make_changes",
        {"justification": "A becomes B", "statements": [_statement_row()],
         "bindings": {"conv": "mass_action"}},
    )
    assert out["applied"] is True and out["committed"] is True
    assert "carry no grounding" in out["messages"][0]["message"]
    record = json.loads((tmp_path / "prov" / "versions" / "v0" / "statements.json").read_text())
    (stored,) = record["rows"]
    assert stored["emission"]["parameters"] == ["conv_k"]
    # The row is the statement and its binding the maths, kept apart as they were given.
    assert set(stored) == {"prefix", "statement", "emission"}
    assert record["bindings"] == {"conv": {"policy": "mass_action"}}
    # What the rows expand to, before and after the reduction, reported and kept with the rows.
    size = {"species": 2, "reactions": 1}
    assert out["network"] == record["network"]
    assert record["network"] == {"before_reduction": size, "after_reduction": size, "split": {}}


def test_ungrounded_agents_are_named_once_per_call(tmp_path):
    """Rows sharing ungrounded agents give one warning naming each agent once, not one per row."""
    mcp = _statements_server(tmp_path)
    back = {
        **_statement_row(),
        "statement": {"type": "PopulationTransition", "source": {"name": "B"}, "target": {"name": "A"}},
        "prefix": "back",
    }
    out = _call(
        mcp,
        "make_changes",
        {"justification": "A and B interconvert", "statements": [back, _statement_row()],
         "bindings": {"back": "mass_action", "conv": "mass_action"}},
    )
    assert out["applied"] is True, out
    assert [m["message"] for m in out["messages"] if "grounding" in m["message"]] == [
        "`statements` agents carry no grounding: B, A."
    ]


def _text(mcp, name, args):
    """A tool's text return (FastMCP wraps a str in {"result": ...})."""
    return _listed(_call(mcp, name, args))


def test_a_scheme_mechanism_builds_and_is_listed_as_written(tmp_path):
    """A mechanism written as its own PySB rules joins the run's library: rows use it, the model
    shows it as written, and `list_mechanisms` lists it beside the library's own entries."""
    mcp = _statements_server(tmp_path)
    out = _call(
        mcp,
        "make_changes",
        {
            "justification": "A becomes B by a rule of my own",
            "statements": [_statement_row()],
            "bindings": {"conv": "hop"},
            "mechanisms": [
                {"statement_type": "PopulationTransition", "policy": "hop",
                 "scheme": "source() >> target() @ k_hop", "parameters": ["k_hop"]}
            ],
        },
    )
    assert out["applied"] is True, out
    (stored,) = json.loads(
        (tmp_path / "prov" / "versions" / "v0" / "statements.json").read_text()
    )["rows"]
    assert stored["emission"]["parameters"] == ["conv_k_hop"]
    listed = _text(mcp, "list_mechanisms", {"statement_type": "PopulationTransition"})
    assert (
        "PopulationTransition | hop (written this run) | parameters: k_hop\n"
        "  source() >> target() @ k_hop"
    ) in listed
    assert "PopulationTransition | mass_action | parameters: k\n" in listed
    model = _text(mcp, "get_model", {"format": "statements"})
    assert (
        "mechanism hop: PopulationTransition, scheme, parameters k_hop\n"
        "  source() >> target() @ k_hop"
    ) in model


def test_a_scheme_naming_a_state_its_molecule_lacks_is_refused(tmp_path):
    """Refused at the call, naming the molecule's sites, rather than built into rules that never
    fire."""
    out = _call(
        _statements_server(tmp_path),
        "make_changes",
        {
            "justification": "A becomes B once primed",
            "statements": [_statement_row()],
            "bindings": {"conv": "primed"},
            "mechanisms": [
                {"statement_type": "PopulationTransition", "policy": "primed",
                 "scheme": "source(primed='yes') >> target() @ k", "parameters": ["k"]}
            ],
        },
    )
    assert out["applied"] is False
    assert "adds site 'primed' to A with the one state 'yes'" in out["error"]
    assert "A has no sites" in out["error"]


def test_the_result_names_the_sites_a_scheme_added(tmp_path):
    out = _call(
        _statements_server(tmp_path),
        "make_changes",
        {
            "justification": "A primes, then becomes B",
            "statements": [_statement_row()],
            "bindings": {"conv": "primed"},
            "mechanisms": [
                {"statement_type": "PopulationTransition", "policy": "primed",
                 "scheme": "source(phase='naive') >> source(phase='primed') @ kp\n"
                           "source(phase='primed') >> target() @ k",
                 "parameters": ["kp", "k"]}
            ],
        },
    )
    assert out["applied"] is True, out
    assert any(
        "schemes added sites no row or entity gives: A phase ['naive', 'primed'] (`statements[0]`)"
        in m["message"]
        for m in out["messages"]
    )


def test_list_mechanisms_exists_only_with_the_route(tmp_path):
    names = lambda mcp: {t.name for t in asyncio.run(mcp.list_tools())}  # noqa: E731
    assert "list_mechanisms" not in names(_server(tmp_path / "plain"))
    assert "list_mechanisms" in names(_statements_server(tmp_path))


def test_list_mechanisms_shows_two_step_as_its_rules(tmp_path):
    listed = _text(_statements_server(tmp_path), "list_mechanisms", {"statement_type": "Phosphorylation"})
    assert (
        "Phosphorylation | indra_two_step | parameters: kc, kf, kr\n"
        "  enz(sub=None) + sub(mod='off', enz=None) >> enz(sub=1) % sub(mod='off', enz=1) @ kf"
    ) in listed


def test_a_library_gap_is_its_own_rejection(tmp_path):
    """A pair the library does not hold must stay countable apart from a malformed call — it is
    the to-do list for the library. INDRA's default asked of a type INDRA does not have is one."""
    out = _call(
        _statements_server(tmp_path),
        "make_changes",
        {"justification": "x", "statements": [_statement_row()],
         "bindings": {"conv": "indra_default"}},
    )
    assert out["applied"] is False and out["error_cause"] == "library_gap"


def test_a_malformed_mechanism_is_not_counted_as_a_library_gap(tmp_path):
    """Every MechanismError counted as a gap, so the library's to-do list was mostly typos."""
    bad = {**_SATURATING, "rate": "vmax / (K + nowhere)"}
    out = _call(
        _statements_server(tmp_path),
        "make_changes",
        {"justification": "x", "statements": [_statement_row()],
         "bindings": {"conv": "saturating"}, "mechanisms": [bad]},
    )
    assert out["applied"] is False and out["error_cause"] == "malformed_mechanism"


def test_an_unreadable_statement_is_refused_not_dropped(tmp_path):
    """INDRA's reader drops what it cannot parse with only a logged warning; a silently shorter
    model would then be fitted and committed."""
    row = _statement_row(type_="NotAStatement")
    out = _call(
        _statements_server(tmp_path),
        "make_changes",
        {"justification": "x", "statements": [row], "bindings": {"conv": "mass_action"}},
    )
    assert out["applied"] is False and out["error_cause"] == "unreadable_statement"


def test_a_statement_missing_a_field_is_refused_by_name(tmp_path):
    """A mod without `mod_type` escaped make_changes as a raw AttributeError; it is refused, the
    row and the field named, before INDRA reads anything."""
    row = {
        "statement": {"type": "Phosphorylation", "enz": {"name": "A"}, "sub": {"name": "B", "mods": [{}]}},
        "prefix": "p",
    }
    out = _call(
        _statements_server(tmp_path),
        "make_changes",
        {"justification": "x", "statements": [row], "bindings": {"p": "default"}},
    )
    assert out["applied"] is False and out["error_cause"] == "unreadable_statement"
    assert "`p`: `sub.mods[0]` has no `mod_type`, one of [" in out["error"]


def test_a_translocation_to_and_from_nowhere_is_refused_as_written(tmp_path):
    """`A moves`, every field present, was refused only once it built no rule, and told to give
    the row a policy that builds one; INDRA's own check of a statement refuses it first."""
    out = _call(
        _statements_server(tmp_path),
        "make_changes",
        {"justification": "x", "statements": ["conv: A becomes B", "t: A moves"],
         "bindings": {"conv": "mass_action", "t": "default"}},
    )
    assert out["applied"] is False and out["error_cause"] == "unreadable_statement"
    assert "`t`: Translocation with no locations. NOTHING was applied." in out["error"]


def test_a_type_the_library_lacks_is_a_library_gap(tmp_path):
    """INDRA's own types are statement types whether or not the library has a mechanism for them:
    refused as no statement type, RegulateAmount was neither counted as a gap nor pointed at
    `mechanisms`."""
    row = {
        "statement": {"type": "RegulateAmount", "subj": {"name": "A"}, "obj": {"name": "B"}},
        "prefix": "h",
    }
    out = _call(
        _statements_server(tmp_path),
        "make_changes",
        {"justification": "x", "statements": [row], "bindings": {"h": {"policy": "indra_default"}}},
    )
    assert out["applied"] is False and out["error_cause"] == "library_gap", out


def test_a_crash_building_statements_is_logged_and_raised(tmp_path, monkeypatch):
    """Only refusals were caught there, so a crash building the model escaped make_changes with
    nothing in errors.jsonl, the run's record of crashes; it is still raised, not refused."""

    def crash(*args, **kwargs):
        raise RuntimeError("UNIQUE_CRASH")

    monkeypatch.setattr(O, "statements_to_antimony", crash)
    with pytest.raises(Exception, match="UNIQUE_CRASH"):
        _call(
            _statements_server(tmp_path),
            "make_changes",
            {"justification": "x", "statements": [_statement_row()],
             "bindings": {"conv": "mass_action"}},
        )
    assert "UNIQUE_CRASH" in (tmp_path / "prov" / "errors.jsonl").read_text()


def test_an_activation_without_obj_activity_builds(tmp_path):
    """The default the statement type documents is filled before INDRA reads the row, and stored
    with it, so the model builds and the row renders as it was read."""
    row = {
        "statement": {"type": "Activation", "subj": {"name": "A"}, "obj": {"name": "B"}},
        "prefix": "act",
    }
    out = _call(
        _statements_server(tmp_path),
        "make_changes",
        {"justification": "x", "statements": [row], "bindings": {"act": "default"}},
    )
    assert out["applied"] is True, out
    (stored,) = json.loads(
        (tmp_path / "prov" / "versions" / "v0" / "statements.json").read_text()
    )["rows"]
    assert stored["statement"]["obj_activity"] == "activity"


def test_a_repeated_prefix_is_refused_before_assembly(tmp_path):
    """Two rows minting the same parameter names clash inside PySB, whose error is not one the
    tool boundary catches."""
    rows = [_statement_row(), _statement_row()]
    out = _call(
        _statements_server(tmp_path),
        "make_changes",
        {"justification": "x", "statements": rows, "bindings": {"conv": "mass_action"}},
    )
    assert out["applied"] is False and out["error_cause"] == "duplicate_prefix"


def test_a_misspelt_policy_is_not_counted_as_a_library_gap(tmp_path):
    """Every pair the library lacked was a gap, so a typo (`mass_actoin`) joined the library's
    to-do list. It is malformed, and the refusal names the policy meant."""
    out = _call(
        _statements_server(tmp_path),
        "make_changes",
        {"justification": "x", "statements": [_statement_row()],
         "bindings": {"conv": "mass_actoin"}},
    )
    assert out["applied"] is False and out["error_cause"] == "malformed_mechanism"
    assert "(did you mean 'mass_action'?)" in out["error"]


def test_statement_species_take_the_problems_own_initial_conditions(tmp_path):
    """The first version minted a fixed one-unit initial amount per species, so every population
    started at one unit and the initial amounts the problem ships went unused. The toy ships
    `A_init` and `B_init` fixed; both must survive, and no other IC row may appear."""
    mcp = _statements_server(tmp_path)
    _call(mcp, "make_changes", {"justification": "A becomes B", "statements": [_statement_row()],
                                "bindings": {"conv": "mass_action"}})
    ids = {r["parameterId"] for r in _call(mcp, "get_parameters", {})["result"]}
    assert {"A_init", "B_init"} <= ids
    assert not {i for i in ids if i.endswith("_0")}


def test_a_binding_to_a_start_or_a_species_is_refused_saying_what_reads_it(tmp_path):
    """A binding to a shipped start, `k="A_init"`, met PySB's duplicate name once the start made
    `A_init` again, and one to a readout id, `k="Btot"`, libSBML's duplicate id beside the
    readout: both escaped make_changes raw."""
    mcp = _statements_server(tmp_path, extra=', "Btot": {"name": "B"}')
    for binding, ident, what, instead in (
        ("A_init", "A_init", "a starting amount", "which only `starts` declares"),
        ("Btot", "Btot", "a species", "`Btot` is the amount of the agent the entity map"),
        ("k1*Btot", "Btot", "a species", "by that agent's own name and sites"),
    ):
        out = _call(mcp, "make_changes", {
            "justification": "x", "statements": [_statement_row()],
            "bindings": {"conv": {"policy": "mass_action", "parameters": {"k": binding}}}})
        assert out["error_cause"] == "malformed_mechanism", out
        assert f"row 'conv' binds 'conv_k' over '{ident}', which names {what}," in out["error"]
        assert instead in out["error"]


def test_a_binding_to_a_shipped_parameter_reads_its_row(tmp_path):
    """How a problem id enters a rate: the row's `k` becomes the toy's own `k1`, which keeps its
    row, and nothing is minted beside it."""
    mcp = _statements_server(tmp_path)
    out = _call(mcp, "make_changes", {"justification": "x", "statements": [_statement_row()],
                                      "bindings": {"conv": "mass_action k=k1"}})
    assert out["applied"] is True, out
    rows = {r["parameterId"]: r for r in _call(mcp, "get_parameters", {})["result"]}
    assert "conv_k" not in rows and rows["k1"]["nominalValue"] == 0.5


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
    row = _statement_row()
    args = {"justification": "saturating", "statements": [row],
            "bindings": {"conv": "saturating"}, "mechanisms": [_SATURATING]}
    assert _call(mcp, "make_changes", args)["committed"] is True
    (stored,) = json.loads(
        (tmp_path / "prov" / "versions" / "v0" / "statements.json").read_text()
    )["rows"]
    assert stored["mechanism"]["rate"] == "vmax / (K + source)"
    model = _text(mcp, "get_model", {"format": "statements"})
    assert "like mass_action, rate vmax / (K + source), parameters vmax, K" in model
    row["prefix"] = "conv2"  # a different structure, so the de-dup does not refuse the refit
    again = _call(mcp, "make_changes", {"justification": "reuse", "statements": [row],
                                        "bindings": {"conv2": "saturating"}})
    assert again["applied"] is True


def test_a_written_mechanism_no_row_uses_survives_a_resume(tmp_path):
    """The run keeps a mechanism it wrote whether or not a row uses it. A resume read them back
    off the rows alone, so one no row used was in the library live and gone after a resume."""
    mcp = _statements_server(tmp_path)
    args = {"justification": "x", "statements": [_statement_row()],
            "bindings": {"conv": "mass_action"}, "mechanisms": [_SATURATING]}
    assert _call(mcp, "make_changes", args)["committed"] is True
    row = {**_statement_row(), "prefix": "conv2"}
    resumed = _statements_server(tmp_path)
    out = _call(resumed, "make_changes", {"justification": "y", "statements": [row],
                                          "bindings": {"conv2": "saturating"}})
    assert out["applied"] is True, out.get("error")


def test_a_kind_made_from_nothing_is_said_to_start_empty(tmp_path):
    """A kind made by zero-order synthesis started at 0 and the result said nothing."""
    rows = [_statement_row(), "syn: B is produced"]
    out = _call(_statements_server(tmp_path), "make_changes", {
        "justification": "x", "statements": rows,
        "bindings": {"conv": "mass_action", "syn": "default"}})
    assert out["applied"] is True, out
    assert any("from nothing" in m["message"] and ": B." in m["message"] for m in out["messages"])


def test_a_written_mechanism_is_not_redefined_in_place(tmp_path):
    """Earlier versions were built with the first definition; changing it under the same name
    would change what their records mean."""
    mcp = _statements_server(tmp_path)
    rows = {"statements": [_statement_row()], "bindings": {"conv": "saturating"}}
    _call(
        mcp,
        "make_changes",
        {"justification": "x", **rows, "mechanisms": [_SATURATING]},
    )
    changed = {**_SATURATING, "rate": "vmax"}
    out = _call(
        mcp, "make_changes", {"justification": "y", **rows, "mechanisms": [changed]}
    )
    assert out["applied"] is False and out["error_cause"] == "mechanism_redefined"


def test_a_version_without_new_rows_keeps_its_parents(tmp_path):
    """A bounds-only commit keeps the model its rows built, so it keeps the rows and the
    mechanisms they name. It stored none, so restoring it restored none: the model no longer read
    as built from statements, and a row naming the written mechanism was refused."""
    mcp = _statements_server(tmp_path)
    row = _statement_row()
    args = {"justification": "saturating", "statements": [row],
            "bindings": {"conv": "saturating"}, "mechanisms": [_SATURATING]}
    assert _call(mcp, "make_changes", args)["committed"] is True
    widen = [{"parameter_id": "conv_vmax", "lower_bound": 1e-3, "upper_bound": 100.0}]
    out = _call(mcp, "make_changes", {"justification": "widen", "parameters": widen})
    assert out["committed"] is True
    versions = tmp_path / "prov" / "versions"
    stored = (versions / "v0" / "statements.json").read_text()
    assert (versions / "v1" / "statements.json").read_text() == stored
    # A durable resume restores HEAD, v1: its rows, and the mechanism they were written with.
    resumed = _statements_server(tmp_path)
    model = _call(resumed, "get_model", {"format": "statements"})["result"]
    assert "mechanism saturating" in model
    row["prefix"] = "conv2"
    assert _call(resumed, "make_changes", {"justification": "reuse", "statements": [row],
                                           "bindings": {"conv2": "saturating"}})[
        "applied"
    ] is True
    # Model text without rows changes the model under them: that version records none.
    out = _call(resumed, "make_changes", {"justification": "by hand", "antimony": _antimony()})
    assert out["committed"] is True
    assert not (versions / "v3" / "statements.json").exists()
    # Resumed at v3, which has no rows, the run's written mechanisms still come back: a row names
    # one, and it may not be redefined.
    again = _statements_server(tmp_path)
    row["prefix"] = "conv3"
    rows = {"statements": [row], "bindings": {"conv3": "saturating"}}
    assert _call(again, "make_changes", {"justification": "back", **rows})[
        "applied"
    ] is True
    changed = {**_SATURATING, "rate": "vmax"}
    out = _call(
        again, "make_changes", {"justification": "y", **rows, "mechanisms": [changed]}
    )
    assert out["error_cause"] == "mechanism_redefined"


def test_a_declared_unit_start_is_fixed_not_estimated(tmp_path):
    """`starts`: a population declared to start at 1 takes reconcile's unit
    normalisation, fixed, where the default seed would have estimated an `A_init`: one parameter
    too many."""
    mcp = _statements_server(tmp_path)
    out = _call(
        mcp,
        "make_changes",
        {
            "justification": "A becomes B",
            "statements": [_statement_row()],
            "bindings": {"conv": "mass_action"},
            "starts": [{"agent": {"name": "A"}, "amount": 1, "name": "A_start"}],
        },
    )
    assert out["applied"] is True
    rows = {r["parameterId"]: r for r in _call(mcp, "get_parameters", {})["result"]}
    assert "A_start_init" in rows
    # The toy also SHIPS a fixed `A_init` row, which stays; neither may cost a k.
    assert not any(int(rows[p]["estimate"]) for p in ("A_start_init", "A_init") if p in rows)
    stored = json.loads((tmp_path / "prov" / "versions" / "v0" / "statements.json").read_text())
    assert stored["starts"] == [{"name": "A_start", "agent": {"name": "A"}, "amount": 1}]


def test_a_redeclared_start_moves_its_row(tmp_path):
    """A start declared at 1 stayed FIXED at 1 through every later declaration."""
    mcp = _statements_server(tmp_path)
    rows = []
    for amount in (1, 5, 1):
        start = {"agent": {"name": "A"}, "amount": amount, "name": "A_start"}
        args = {"justification": "A starts", "statements": [_statement_row()],
                "bindings": {"conv": "mass_action"}, "starts": [start]}
        assert _call(mcp, "make_changes", args)["applied"] is True
        params = {r["parameterId"]: r for r in _call(mcp, "get_parameters", {})["result"]}
        row = params["A_start_init"]
        rows.append((row["nominalValue"], int(row["estimate"])))
        structure = _call(mcp, "get_model_structure", {})
        view = {r["species"]: r for r in structure["initial_conditions"]}["A"]
        assert (view["ic_parameter"], view["t0"]) == ("A_start_init", row["nominalValue"])
    assert rows == [(1.0, 0), (5.0, 1), (1.0, 0)]


def test_a_held_input_is_reported_at_its_condition_value(tmp_path, add_condition):
    """The view reported a held input as starting empty, though the condition table held it at
    its dose."""
    shutil.copytree(YAML_PATH.parent, tmp_path / "problem")
    add_condition(tmp_path / "problem" / "problem.yaml", "E_0", 3.0)
    (tmp_path / "problem" / "inputs.tsv").write_text(
        "inputId\tformula\tconditionInputs\nE\tpiecewise(E_0, time <= 10, 0)\tE_0\n"
    )
    mcp = _statements_server(tmp_path, yaml_path=tmp_path / "problem" / "problem.yaml")
    feed = {**_statement_row(), "prefix": "feed"}
    feed["statement"] = {**feed["statement"], "source": {"name": "E"}}
    out = _call(
        mcp,
        "make_changes",
        {"justification": "E feeds B", "statements": [_statement_row(), feed],
         "bindings": {"conv": "mass_action", "feed": "mass_action"}},
    )
    assert out["applied"] is True, out
    ics = {r["species"]: r for r in _call(mcp, "get_model_structure", {})["initial_conditions"]}
    assert (ics["E"]["t0_source"], ics["E"]["t0"]) == ("rule", 3.0)


def test_an_empty_entity_map_still_offers_the_route(tmp_path):
    """Starts travel with the statements, so a problem may need nothing from its map; withheld on
    an EMPTY map, the route did not exist and every statements call was refused."""
    entity_map = tmp_path / "entities.json"
    entity_map.write_text("{}")
    mcp = create_server(
        str(YAML_PATH),
        provenance_dir=str(tmp_path / "prov"),
        resolver=StubReferenceResolver(),
        auto_fit=False,
        entity_map=str(entity_map),
    )
    assert "statements" in _make_changes_schema(mcp)


def _problem_ids_server(tmp_path, add_condition, *, fixed_d=False, dose=False):
    """The toy with a shipped FIXED `d`, or a dose column `EGF_0` that an input's ramp reads."""
    work = tmp_path / "problem"
    shutil.copytree(YAML_PATH.parent, work)
    if fixed_d:
        with (work / "parameters.tsv").open("a") as fh:
            fh.write("d\tlog10\t1e-12\t1000.0\t1.0\tfalse\n")
    if dose:
        add_condition(work / "problem.yaml", "EGF_0", 1.0)
        (work / "inputs.tsv").write_text(
            "inputId\tformula\tconditionInputs\nEGF\tpiecewise(EGF_0, time <= 10, 0)\tEGF_0\n"
        )
    return _statements_server(tmp_path, yaml_path=work / "problem.yaml")


def _hop(mcp, route, rate, parameters=("k",), binding=None):
    """`A becomes B` at ``rate``, written as a `scheme` or as a `rate` on mass action, its row
    bound by ``binding``."""
    definition = {"statement_type": "PopulationTransition", "policy": "hop",
                  "parameters": list(parameters)}
    if route == "scheme":
        definition["scheme"] = f"source() >> target() @ {rate}"
    else:
        definition |= {"based_on": "mass_action", "rate": rate}
    bound = {"policy": "hop"} | ({"parameters": binding} if binding else {})
    return _call(mcp, "make_changes", {"justification": "A becomes B",
                                       "statements": [_statement_row()],
                                       "bindings": {"conv": bound}, "mechanisms": [definition]})


_ROUTES = pytest.mark.parametrize("route", ["scheme", "rate"])


@_ROUTES
def test_a_shipped_fixed_parameter_enters_a_written_rate_through_the_binding(
    tmp_path, add_condition, route
):
    """`d` in a scheme's rate, meaning the shipped FIXED `d`, became a new estimated `conv_d` in
    its place; a `rate` naming `d` was refused with no way in."""
    mcp = _problem_ids_server(tmp_path, add_condition, fixed_d=True)
    out = _hop(mcp, route, "k*d")
    assert out["applied"] is False
    assert "`d` is a parameter the problem ships" in out["error"]
    assert "enters through a row's binding" in out["error"]
    out = _hop(mcp, route, "k*kd", ("k", "kd"), {"kd": "d"})
    assert out["applied"] is True, out.get("error")
    rows = {r["parameterId"]: r for r in _call(mcp, "get_parameters", {})["result"]}
    assert not int(rows["d"]["estimate"]) and "conv_kd" not in rows and "conv_d" not in rows


@_ROUTES
def test_a_condition_input_enters_a_written_rate_through_the_binding(
    tmp_path, add_condition, route
):
    """A dose column, `EGF_0`, in a scheme's rate became a constant `conv_EGF_0` the condition
    table never set."""
    mcp = _problem_ids_server(tmp_path, add_condition, dose=True)
    out = _hop(mcp, route, "k*EGF_0")
    assert out["applied"] is False
    assert "`EGF_0` is a condition input" in out["error"]
    out = _hop(mcp, route, "k*dose", ("k", "dose"), {"dose": "EGF_0"})
    assert out["applied"] is True, out.get("error")
    rows = {r["parameterId"] for r in _call(mcp, "get_parameters", {})["result"]}
    assert "EGF_0" not in rows and "conv_dose" not in rows  # the condition table sets it


@_ROUTES
def test_an_unknown_rate_name_is_refused_with_what_the_rate_may_name(tmp_path, route):
    """`clock`, meaning the simulation time, became a constant `conv_clock`."""
    out = _hop(_statements_server(tmp_path), route, "k*clock")
    assert out["applied"] is False
    assert "it names ['clock']" in out["error"] and "'time'" in out["error"]


def test_a_row_may_be_written_in_the_notation(tmp_path):
    """The same model, sent as one line instead of JSON, its binding apart, and read back the
    same way."""
    mcp = _statements_server(tmp_path)
    out = _call(
        mcp,
        "make_changes",
        {"justification": "A becomes B", "statements": ["conv: A becomes B"],
         "bindings": {"conv": "mass_action"}},
    )
    assert out["applied"] is True
    (stored,) = json.loads(
        (tmp_path / "prov" / "versions" / "v0" / "statements.json").read_text()
    )["rows"]
    assert stored["statement"]["type"] == "PopulationTransition"
    text = _call(mcp, "get_model", {"format": "statements"})["result"]
    assert "conv: A becomes B\nbinding conv: mass_action" in text


# -- `statement_edits`: changing a statements model by its rows ----------------------
_CONV = "conv: A becomes B"
_BACK = "back: B becomes A"
#: Both rows' bindings: each on mass action.
_MASS_ACTION = {"conv": "mass_action", "back": "mass_action"}


def _stored_rows(tmp_path, version):
    path = tmp_path / "prov" / "versions" / f"v{version}" / "statements.json"
    return json.loads(path.read_text())


def _built(tmp_path, rows, bindings):
    mcp = _statements_server(tmp_path)
    out = _call(mcp, "make_changes", {"justification": "build", "statements": rows,
                                      "bindings": bindings})
    assert out["applied"] is True
    return mcp


def test_a_row_and_its_binding_are_given_apart_and_each_needs_the_other(tmp_path):
    """The statement is the claim and its binding the maths: a row with no binding, a binding
    with no row, a row still written with its binding inline (`| mass_action`) and a JSON row
    carrying `policy` are each refused by name, the last two saying how they split."""
    mcp = _statements_server(tmp_path)

    def make(**args):
        return _call(mcp, "make_changes", {"justification": "x", **args})

    out = make(statements=[_CONV, _BACK], bindings={"conv": "mass_action"})
    assert out["error_cause"] == "unbound_row" and "rows ['back'] have no binding" in out["error"]
    out = make(statements=[_CONV], bindings={"conv": "mass_action", "gone": "mass_action"})
    assert out["error_cause"] == "unbound_binding"
    assert "`bindings` ['gone'] name no row; the rows are ['conv']" in out["error"]
    out = make(statements=["conv: A becomes B | mass_action k=k1"])
    assert out["error_cause"] == "inline_binding"
    assert ('`conv: A becomes B` with `bindings={"conv": "mass_action k=k1"}`' in out["error"])
    out = make(statements=[{**_statement_row(), "policy": "mass_action", "parameters": {"k": "k1"}}])
    assert out["error_cause"] == "inline_binding"
    assert "carries ['parameters', 'policy']" in out["error"]
    assert '`conv: A becomes B` with `bindings={"conv": "mass_action k=k1"}`' in out["error"]
    out = make(statements=[_CONV], bindings={"conv": "conv: mass_action"})
    assert out["error_cause"] == "unreadable_binding" and "`bindings.conv`" in out["error"]
    assert not (tmp_path / "prov" / "versions" / "v0").exists()


def test_a_binding_in_json_names_its_policy_as_its_text_does(tmp_path):
    """`{"policy": "default"}` was taken as written, a name no entry has, while the text `default`
    read as INDRA's `indra_default`, the name the field's own description offers."""
    mcp = _statements_server(tmp_path)
    out = _call(mcp, "make_changes", {"justification": "A phosphorylates B",
                                      "statements": ["p: A phosphorylates B"],
                                      "bindings": {"p": {"policy": "default"}}})
    assert out["applied"] is True, out.get("error")
    assert _stored_rows(tmp_path, 0)["bindings"] == {"p": {"policy": "indra_default"}}


def test_an_object_sent_as_a_json_string_is_read_as_the_object(tmp_path):
    """A client that writes an object argument as a JSON string (`amounts` as '{"a": "A"}') was
    refused before the tool ran, and `bindings` travels with every `statements` call: the string
    is read as the object it holds, as a list's is."""
    from petab_mcp.server import _unwrap_objectish

    mcp = _statements_server(tmp_path)
    out = _call(mcp, "make_changes", {"justification": "A becomes B", "statements": [_CONV],
                                      "bindings": '{"conv": "mass_action"}'})
    assert out["applied"] is True, out.get("error")
    assert _unwrap_objectish(' {"a": "A{pS10}"} ') == {"a": "A{pS10}"}
    assert _unwrap_objectish("{not json}") == "{not json}"
    assert _unwrap_objectish("mass_action") == "mass_action"


def test_a_malformed_row_in_an_edit_keeps_the_notations_hint(tmp_path):
    """Read early by its edit, a row that does not read lost the hint the build adds, which spells
    the notation out."""
    mcp = _built(tmp_path, [_CONV], {"conv": "mass_action"})
    out = _call(mcp, "make_changes", {"justification": "x", "statement_edits": [
        {"op": "add_row", "row": "back: B frobnicates A", "binding": "mass_action"}]})
    assert out["error_cause"] == "unreadable_statement"
    assert "unknown verb 'frobnicates'" in out["error"] and "A row is `prefix: <statement>`" in out["error"]


def test_bindings_ride_with_statements(tmp_path):
    """`bindings` give every row's binding with `statements`; under `statement_edits` a binding
    changes by its own edit."""
    (tmp_path / "alone").mkdir()
    widen = [{"parameter_id": "k1", "lower_bound": 0.01, "upper_bound": 10.0}]
    out = _call(_statements_server(tmp_path / "alone"), "make_changes",
                {"justification": "x", "bindings": {"conv": "mass_action"}, "parameters": widen})
    assert out["applied"] is False and "`bindings`, `mechanisms`" in out["error"]
    mcp = _built(tmp_path, [_CONV], {"conv": "mass_action"})
    out = _call(mcp, "make_changes", {
        "justification": "x", "bindings": {"conv": "mass_action k=k1"},
        "statement_edits": [{"op": "replace_row", "row": _CONV}]})
    assert out["applied"] is False and "`rebind`" in out["error"]


@pytest.mark.parametrize(
    "edit, said",
    [
        ({"op": "add_row", "row": _BACK}, "a new row needs its `binding`"),
        ({"op": "add_row", "row": "back: B becomes A | mass_action"},
         '`{"op": "add_row", "row": "back: B becomes A", "binding": "mass_action"}`'),
        ({"op": "replace_row", "row": "conv: A becomes B | mass_action k=k1"},
         '`{"op": "replace_row", "row": "conv: A becomes B", "binding": "mass_action k=k1"}`'),
        ({"op": "replace_row", "row": {**_statement_row(), "policy": "mass_action"}},
         '`{"op": "replace_row", "row": "conv: A becomes B", "binding": "mass_action"}`'),
        ({"op": "rebind", "prefix": "gone", "binding": "mass_action"}, "no row `gone`"),
        ({"op": "rebind", "prefix": "conv"}, "`binding` is the row's policy"),
        ({"op": "add_row", "row": _CONV, "binding": "mass_action"}, "`rebind` its binding"),
    ],
)
def test_an_edit_missing_or_inlining_a_binding_is_refused_by_its_edit(tmp_path, edit, said):
    """A row given with its binding inside an edit was told to use `bindings`, which is refused
    beside `statement_edits`, or `add_row`, which is refused for a row that exists: the refusal
    names the edit itself, with its own `binding`. An edit that leaves a row without a binding, or
    names a row that is not there, is refused by its index."""
    mcp = _built(tmp_path, [_CONV], {"conv": "mass_action"})
    out = _call(mcp, "make_changes", {"justification": "x", "statement_edits": [edit]})
    assert out["applied"] is False and "`statement_edits[0]`" in out["error"]
    assert said in out["error"], out["error"]


def test_no_agent_facing_text_writes_a_row_with_its_binding(tmp_path):
    """One notation: the route's description, its schema, `list_mechanisms` and the skills show a
    row as its statement alone, never `prefix: <statement> | <policy>`, which is refused."""
    import re

    inline = re.compile(r"`[A-Za-z_]\w*: [^`]*? \| [a-z_]+")
    mcp = _statements_server(tmp_path)
    text = _agent_facing_text(mcp) + _text(mcp, "list_mechanisms", {})
    assert not inline.findall(text), inline.findall(text)


def test_a_version_stored_with_its_bindings_on_its_rows_still_loads(tmp_path):
    """A record stored before rows and bindings were given apart kept each row's `policy` and
    `parameters` on the row. A resume, a `base_version` fork and `get_model` read it as the same
    rows and bindings."""
    mcp = _built(tmp_path, [_CONV, _BACK], {"conv": "mass_action", "back": "mass_action k=conv_k"})
    path = tmp_path / "prov" / "versions" / "v0" / "statements.json"
    record = json.loads(path.read_text())
    bindings = record.pop("bindings")
    record["rows"] = [row | bindings[row["prefix"]] for row in record["rows"]]
    path.write_text(json.dumps(record))
    shown = "conv: A becomes B\nback: B becomes A\nbinding conv: mass_action\nbinding back: mass_action k=conv_k"
    assert shown in _text(mcp, "get_model", {"format": "statements", "version": 0})
    resumed = _statements_server(tmp_path)
    assert shown in _text(resumed, "get_model", {"format": "statements"})
    out = _call(resumed, "make_changes", {
        "justification": "the way back on a constant of its own",
        "statement_edits": [{"op": "rebind", "prefix": "back", "binding": "mass_action"}]})
    assert out["applied"] is True, out.get("error")
    assert _stored_rows(tmp_path, 1)["bindings"] == {
        "conv": {"policy": "mass_action"}, "back": {"policy": "mass_action"}}
    out = _call(resumed, "make_changes", {
        "justification": "from v0, with a third row", "base_version": 0,
        "statement_edits": [{"op": "add_row", "row": "back2: B becomes A", "binding": "mass_action"}]})
    assert out["applied"] is True, out.get("error")
    assert _stored_rows(tmp_path, 2)["bindings"] == {
        "conv": {"policy": "mass_action"}, "back": {"policy": "mass_action", "parameters": {"k": "conv_k"}},
        "back2": {"policy": "mass_action"}}


def test_statement_edits_change_the_rows_they_name_and_keep_the_rest(tmp_path):
    """An edit names a row by its prefix; the rows it does not name are rebuilt as they were,
    so a revision no longer means sending the whole model again."""
    mcp = _built(tmp_path, [_CONV, _BACK], _MASS_ACTION)
    out = _call(
        mcp,
        "make_changes",
        {
            "justification": "drop the way back, then give it its own rate again",
            "statement_edits": [
                {"op": "remove_row", "prefix": "back"},
                {"op": "add_row", "row": "back2: B becomes A", "binding": "mass_action"},
                {"op": "set_start", "start": {"agent": "A", "amount": 1, "name": "A_start"}},
            ],
        },
    )
    assert out["applied"] is True
    assert out["statement_edits_applied"] == [
        "remove_row back",
        "add_row back2",
        "set_start A_start",
    ]
    record = _stored_rows(tmp_path, 1)
    assert [r["prefix"] for r in record["rows"]] == ["conv", "back2"]
    assert record["bindings"] == {"conv": {"policy": "mass_action"}, "back2": {"policy": "mass_action"}}
    assert [s["name"] for s in record["starts"]] == ["A_start"]


def test_rebind_changes_a_rows_binding_and_leaves_its_statement(tmp_path):
    """The maths of a row changes without the claim being stated again."""
    mcp = _built(tmp_path, [_CONV, _BACK], _MASS_ACTION)
    out = _call(
        mcp,
        "make_changes",
        {
            "justification": "the way back shares the forward rate",
            "statement_edits": [{"op": "rebind", "prefix": "back", "binding": "mass_action k=conv_k"}],
        },
    )
    assert out["applied"] is True, out.get("error")
    assert out["statement_edits_applied"] == ["rebind back"]
    record = _stored_rows(tmp_path, 1)
    assert [r["prefix"] for r in record["rows"]] == ["conv", "back"]
    assert record["rows"][1]["statement"] == _stored_rows(tmp_path, 0)["rows"][1]["statement"]
    assert record["bindings"]["back"] == {"policy": "mass_action", "parameters": {"k": "conv_k"}}


def test_replace_row_rewrites_the_statement_in_place_keeping_its_binding(tmp_path):
    """The claim changes and the maths stays, unless the edit gives a binding too."""
    mcp = _built(tmp_path, [_CONV, _BACK], {"conv": "mass_action", "back": "mass_action k=conv_k"})
    out = _call(
        mcp,
        "make_changes",
        {
            "justification": "the way back, with its source",
            "statement_edits": [{"op": "replace_row", "row": "back: B becomes A # [curated] Doe"}],
        },
    )
    assert out["applied"] is True, out.get("error")
    record = _stored_rows(tmp_path, 1)
    assert [r["prefix"] for r in record["rows"]] == ["conv", "back"]
    assert record["rows"][1]["statement"]["evidence"] == [{"source_api": "curated", "text": "Doe"}]
    assert record["bindings"]["back"] == {"policy": "mass_action", "parameters": {"k": "conv_k"}}
    out = _call(
        mcp,
        "make_changes",
        {
            "justification": "the way back on its own constant again",
            "statement_edits": [
                {"op": "replace_row", "row": "back: B becomes A", "binding": "mass_action"}
            ],
        },
    )
    assert out["applied"] is True, out.get("error")
    assert _stored_rows(tmp_path, 2)["bindings"]["back"] == {"policy": "mass_action"}


def test_an_edit_that_names_nothing_applies_nothing(tmp_path):
    mcp = _built(tmp_path, [_CONV], {"conv": "mass_action"})
    out = _call(
        mcp,
        "make_changes",
        {
            "justification": "edit a row that is not there",
            "statement_edits": [
                {"op": "add_row", "row": _BACK, "binding": "mass_action"},
                {"op": "replace_row", "row": "gone: B becomes A"},
            ],
        },
    )
    assert out["applied"] is False
    assert "`statement_edits[1]` (replace_row): no row `gone`" in out["error"]
    assert "['back', 'conv']" in out["error"]
    assert not (tmp_path / "prov" / "versions" / "v1").exists()


def test_statement_edits_need_a_model_built_from_statements(tmp_path):
    mcp = _statements_server(tmp_path)
    out = _call(
        mcp,
        "make_changes",
        {"justification": "edit nothing", "statement_edits": [{"op": "add_row", "row": _CONV, "binding": "mass_action"}]},
    )
    assert out["applied"] is False
    assert "give the whole model as `statements`" in out["error"]


def test_whole_sets_ride_with_statements_not_with_edits(tmp_path):
    mcp = _built(tmp_path, [_CONV], {"conv": "mass_action"})
    out = _call(
        mcp,
        "make_changes",
        {
            "justification": "mixed",
            "statement_edits": [{"op": "add_row", "row": _BACK, "binding": "mass_action"}],
            "starts": [{"agent": "A", "amount": 1}],
        },
    )
    assert out["applied"] is False and "`set_start`" in out["error"]


def test_base_version_applies_edits_to_that_versions_rows(tmp_path):
    """The rollback-and-edit path the `model_edits` route has, on the statements route, where it
    was refused before."""
    mcp = _built(tmp_path, [_CONV], {"conv": "mass_action"})
    _call(
        mcp,
        "make_changes",
        {"justification": "v1", "statement_edits": [{"op": "add_row", "row": _BACK, "binding": "mass_action"}]},
    )
    out = _call(
        mcp,
        "make_changes",
        {
            "justification": "from v0 instead",
            "base_version": 0,
            "statement_edits": [
                {"op": "add_row", "row": "back2: B becomes A", "binding": "mass_action"}
            ],
        },
    )
    assert out["applied"] is True
    assert [r["prefix"] for r in _stored_rows(tmp_path, 2)["rows"]] == ["conv", "back2"]


def test_a_restore_brings_back_that_versions_rows(tmp_path):
    """After `restore_version(v)`, the rows the session shows and edits are v's, not those of
    the HEAD it left; a version built from model text holds none."""
    mcp = _built(tmp_path, [_CONV], {"conv": "mass_action"})

    def add_row(row):
        edits = [{"op": "add_row", "row": row, "binding": "mass_action"}]
        return _call(mcp, "make_changes", {"justification": row, "statement_edits": edits})

    assert add_row(_BACK)["applied"] is True
    antimony = {"justification": "by hand", "antimony": _antimony()}
    assert _call(mcp, "make_changes", antimony)["committed"] is True
    _call(mcp, "restore_version", {"version": 0})
    text = _text(mcp, "get_model", {"format": "statements"})
    assert "conv: A becomes B" in text and "back:" not in text
    assert add_row("back2: B becomes A")["applied"] is True
    assert [r["prefix"] for r in _stored_rows(tmp_path, 3)["rows"]] == ["conv", "back2"]
    _call(mcp, "restore_version", {"version": 2})
    out = add_row(_BACK)
    assert out["applied"] is False and "the current model was not" in out["error"]


def test_base_version_takes_whole_statements_too(tmp_path):
    mcp = _built(tmp_path, [_CONV], {"conv": "mass_action"})
    out = _call(
        mcp,
        "make_changes",
        {"justification": "afresh from v0", "base_version": 0, "statements": [_CONV, _BACK],
         "bindings": _MASS_ACTION},
    )
    assert out["applied"] is True


def _two_rates(tmp_path):
    """A fitted statements run: version 0 with the saturating rate, HEAD (version 1) with a
    steeper one. Returns the server and version 0's row."""
    entity_map = tmp_path / "entities.json"
    entity_map.write_text(_BASE_ENTITY_MAP.format(extra=""))
    mcp = create_server(
        str(YAML_PATH),
        provenance_dir=str(tmp_path / "prov"),
        resolver=StubReferenceResolver(),
        auto_fit=True,
        entity_map=str(entity_map),
    )
    row = _statement_row()
    steeper = {**_SATURATING, "policy": "steeper", "rate": "vmax / (K + source * source)"}
    v0 = {"statements": [row], "bindings": {"conv": "saturating"}, "mechanisms": [_SATURATING]}
    _call(mcp, "make_changes", {"justification": "v0", "n_starts": 2, **v0})
    v1 = _call(
        mcp,
        "make_changes",
        {
            "justification": "v1",
            "statement_edits": [{"op": "rebind", "prefix": "conv", "binding": "steeper"}],
            "mechanisms": [steeper],
            "n_starts": 2,
        },
    )
    assert v1["version"] == 1, v1["message"]
    return mcp, row


def test_a_base_version_fork_from_an_earlier_version_is_fitted(tmp_path):
    """A fork from version 0 while HEAD holds another rate is no duplicate of HEAD. The de-dup's
    fingerprint saw no rate (each is an assignment rule the rate law multiplies), so the fork came
    back as HEAD's duplicate and was never fitted. Its rate is one neither version has."""
    mcp, row = _two_rates(tmp_path)
    other = {**_SATURATING, "policy": "other", "rate": "vmax * K / (K + source)"}
    out = _call(
        mcp,
        "make_changes",
        {
            "justification": "from v0, another rate",
            "base_version": 0,
            "statement_edits": [{"op": "rebind", "prefix": "conv", "binding": "other"}],
            "mechanisms": [other],
            "n_starts": 2,
        },
    )
    assert out["committed"] is True and out["version"] == 2, out["message"]


def test_a_base_version_fork_that_changes_nothing_is_refused_as_that_version(tmp_path):
    """Version 0's own rows again, from version 0: version 0 stands for them, so nothing is
    fitted and HEAD stays where it was."""
    mcp, row = _two_rates(tmp_path)
    out = _call(
        mcp,
        "make_changes",
        {"justification": "v0 again", "base_version": 0, "statements": [row],
         "bindings": {"conv": "saturating"}, "n_starts": 2},
    )
    assert out["applied"] is False and "identical to v0" in out["error"], out["message"]
    assert out["version"] == 1


def test_on_the_statements_route_the_refusals_name_its_own_routes(tmp_path):
    """Told "nothing to do — give `model_edits` ... `antimony`", an agent whose model was built
    from statements wrote Antimony."""
    mcp = _built(tmp_path, [_CONV], {"conv": "mass_action"})
    out = _call(mcp, "make_changes", {"justification": "nothing"})
    assert "`statements`" in out["error"] and "`statement_edits`" in out["error"]
    assert "`antimony`" not in out["error"]
    out = _call(
        mcp,
        "make_changes",
        {
            "justification": "both",
            "statements": [_CONV],
            "bindings": {"conv": "mass_action"},
            "statement_edits": [{"op": "add_row", "row": _BACK, "binding": "mass_action"}],
        },
    )
    assert "`statement_edits` for a change to the current rows" in out["error"]
    out = _call(mcp, "make_changes", {"justification": "bare", "base_version": 0})
    assert "`statement_edits` (or the whole revised `statements`)" in out["error"]


def test_the_route_says_nothing_is_bound_when_the_map_is_empty(tmp_path):
    """With an empty map the route's paragraph said the observables and inputs were bound by it,
    and told the agent to name agents as the map does: a map with nothing in it."""
    from petab_mcp.server import _make_changes_statements_doc

    assert "bound by the problem's entity map" in _make_changes_statements_doc(bound=True)
    unbound = _make_changes_statements_doc(bound=False)
    assert "NOTHING is bound" in unbound and "entity map" not in unbound
    entity_map = tmp_path / "entities.json"
    entity_map.write_text("{}")
    mcp = create_server(
        str(YAML_PATH),
        provenance_dir=str(tmp_path / "prov"),
        resolver=StubReferenceResolver(),
        auto_fit=False,
        entity_map=str(entity_map),
    )
    tools = asyncio.run(mcp.list_tools())
    description = next(t for t in tools if t.name == "make_changes").description
    assert "NOTHING is bound" in description and "`statement_edits`" in description
    assert "statement_edits" in _make_changes_schema(mcp)


def test_an_unread_condition_target_is_explained():
    from petab_mcp.operations import _rejection_summary

    text = _rejection_summary(["Condition table contains invalid targets: {'rit'}"])
    assert "{'rit'}" in text and "input the condition table SETS" in text
    assert "SETS" not in _rejection_summary(["something else"])


# -- readouts, bindings and written mechanisms -------------------------------------------
def _readout_server(tmp_path, extra=""):
    """The toy with its observable reading a blinded `readout_B`, and an empty entity map: a
    blinded problem on the statements route. ``extra`` adds observable rows."""
    import shutil

    data = tmp_path / "toy"
    shutil.copytree(YAML_PATH.parent, data)
    (data / "observables.tsv").write_text(
        "observableId\tobservableFormula\tnoiseFormula\nobs_B\treadout_B\tsigma_B\n" + extra
    )
    entity_map = tmp_path / "entities.json"
    entity_map.write_text("{}")
    return create_server(
        str(data / "problem.yaml"),
        provenance_dir=str(tmp_path / "prov"),
        resolver=StubReferenceResolver(),
        auto_fit=False,
        entity_map=str(entity_map),
    )


def test_a_readout_may_read_an_amount_in_place(tmp_path):
    mcp = _readout_server(tmp_path)
    out = _call(
        mcp,
        "make_changes",
        {
            "justification": "A becomes B, measured as B",
            "statements": [_CONV],
            "bindings": {"conv": "mass_action"},
            "readouts": {"readout_B": "scale_B*amount(B)"},
        },
    )
    assert out["applied"] is True
    stored = _stored_rows(tmp_path, 0)
    assert stored["readouts"] == {"readout_B": "scale_B*amount(B)"} and stored["amounts"] == {}


def _set_readout(mcp, readout, formula):
    edit = {"op": "set_readout", "readout": readout, "formula": formula}
    out = _call(mcp, "make_changes", {"justification": formula, "statement_edits": [edit]})
    assert out["applied"] is True, out.get("error")


def _amount_symbols(mcp):
    """The current model's `amount_*` species, its `amount_*` parameters, and the `amount_*`
    rows its parameter table estimates."""
    import libsbml

    model = libsbml.readSBMLFromString(_text(mcp, "get_model", {"format": "sbml"})).getModel()
    rows = _call(mcp, "get_parameters", {})["result"]
    return (
        {s.getId() for s in model.getListOfSpecies() if s.getId().startswith("amount_")},
        {p.getId() for p in model.getListOfParameters() if p.getId().startswith("amount_")},
        {r["parameterId"] for r in rows if r.get("estimate") and r["parameterId"].startswith("amount_")},
    )


def test_an_amount_read_in_place_keeps_its_name_across_edits(tmp_path):
    """Each edit re-writing `amount(B)` minted the next name (amount_B_2, amount_B_3), and the
    record kept them all."""
    mcp = _readout_server(tmp_path)
    out = _call(
        mcp,
        "make_changes",
        {"justification": "v0", "statements": [_CONV], "bindings": {"conv": "mass_action"},
         "readouts": {"readout_B": "scale_B*amount(B)"}},
    )
    assert out["applied"] is True
    _set_readout(mcp, "readout_B", "scale_B*amount(B) + offset")
    _set_readout(mcp, "readout_B", "scale_B*(amount(B) + offset)")
    assert _amount_symbols(mcp) == ({"amount_B"}, set(), set())
    assert _stored_rows(tmp_path, 2)["amounts"] == {}


def test_an_amount_no_readout_reads_is_gone(tmp_path):
    mcp = _readout_server(tmp_path)
    out = _call(
        mcp,
        "make_changes",
        {"justification": "v0", "statements": [_CONV], "bindings": {"conv": "mass_action"},
         "readouts": {"readout_B": "scale_B*amount(B)"}},
    )
    assert out["applied"] is True
    _set_readout(mcp, "readout_B", "scale_B*amount(A)")
    assert _amount_symbols(mcp) == ({"amount_A"}, set(), set())


def test_a_readout_left_alone_still_reads_the_species(tmp_path):
    """Re-writing one of two readouts of a species gave the species a new name, and the other
    readout, still reading the old name, read a parameter fitted in place of the species."""
    mcp = _readout_server(tmp_path, "obs_C\treadout_C\tsigma_B\n")
    out = _call(
        mcp,
        "make_changes",
        {
            "justification": "v0",
            "statements": [_CONV],
            "bindings": {"conv": "mass_action"},
            "readouts": {"readout_B": "scale_B*amount(B)", "readout_C": "amount(B)"},
        },
    )
    assert out["applied"] is True
    _set_readout(mcp, "readout_C", "amount(B)/scale_C")
    assert _amount_symbols(mcp) == ({"amount_B"}, set(), set())


def test_a_readout_that_reads_no_amount_is_refused(tmp_path):
    """A readout of parameters alone measures nothing the model does, and its flat fit could
    still be the best a run reached."""
    mcp = _readout_server(tmp_path)
    out = _call(
        mcp,
        "make_changes",
        {
            "justification": "a constant readout",
            "statements": [_CONV],
            "bindings": {"conv": "mass_action"},
            "readouts": {"readout_B": "k_level/(k_level + 1)"},
        },
    )
    assert out["applied"] is False and "reads no amount" in out["error"]


def test_a_readout_that_is_a_number_is_not_refused(tmp_path):
    """A number is no parameter the fit tunes: Lucarelli's reference reads the SMAD3 in a SMAD2
    pull-down as all doubly phosphorylated, `readout_S2IP_ppS3 := 1`, and its statements were
    refused for it (PEtabAgentBench#23). `time` alone still reads no amount."""
    mcp = _readout_server(tmp_path)
    out = _call(
        mcp,
        "make_changes",
        {"justification": "a constant readout", "statements": [_CONV], "bindings": {"conv": "mass_action"},
         "readouts": {"readout_B": "1"}},
    )
    assert out["applied"] is True, out.get("error")
    out = _call(
        mcp,
        "make_changes",
        {"justification": "a clock", "statements": [_CONV], "bindings": {"conv": "mass_action"},
         "readouts": {"readout_B": "time"}},
    )
    assert out["applied"] is False and "reads no amount" in out["error"]


def test_a_readout_no_observable_reads_is_refused(tmp_path):
    mcp = _readout_server(tmp_path)
    out = _call(
        mcp,
        "make_changes",
        {
            "justification": "an extra readout",
            "statements": [_CONV],
            "bindings": {"conv": "mass_action"},
            "readouts": {"readout_B": "amount(B)", "readout_drugs": "amount(A)"},
        },
    )
    assert out["applied"] is False
    assert "['readout_drugs'] are read by no observable" in out["error"]
    assert "['readout_B']" in out["error"]


def test_a_binding_may_scale_the_parameter_it_replaces(tmp_path):
    """`k=conv_k*slow`, a binding that scales the constant it replaces, recursed without bottom:
    the binding's expression was redirected into itself."""
    mcp = _built(tmp_path, [_CONV], {"conv": "mass_action"})
    out = _call(
        mcp,
        "make_changes",
        {
            "justification": "the conversion slowed by a factor",
            "statement_edits": [
                {"op": "rebind", "prefix": "conv", "binding": "mass_action k=conv_k*slow"}
            ],
        },
    )
    assert out["applied"] is True, out.get("error")
    rows = {r["parameterId"] for r in _call(mcp, "get_parameters", {})["result"]}
    assert {"conv_k", "slow"} <= rows


def test_a_claim_its_scheme_reverses_is_recorded_and_reported(tmp_path):
    """A scheme whose rate rises with what its statement says inhibits it is recorded as an
    activator, not refused, and the result says so and asks the agent to reconsider."""
    out = _call(
        _statements_server(tmp_path),
        "make_changes",
        {
            "justification": "C slows A becoming B",
            "statements": ["conv: A becomes B, inhibited by C"],
            "bindings": {"conv": "inhibited"},
            "mechanisms": [{"statement_type": "PopulationTransition", "policy": "inhibited",
                            "scheme": "source() >> target() @ k*amount(inhibited_by)",
                            "parameters": ["k"]}],
        },
    )
    assert out["applied"] is True, out.get("error")
    assert any(
        "`conv`: you stated `inhibited by C`, but its rate never falls as C rises, so it "
        "reads as `activated by C`. Check that each mechanism does what you meant; if not, "
        "rewrite its rate." in m["message"]
        for m in out["messages"]
    )


def test_a_regulator_enters_through_the_statement(tmp_path):
    """PEtab-MCP#122: `A becomes B` bound `inhibited inh:=C` read C in its rate while its statement,
    and its English, named only A and B. The statement now states C, and the rate reads it."""
    inhibited = {
        "statement_type": "PopulationTransition",
        "policy": "inhibited",
        "based_on": "mass_action",
        "rate": "k*Ki/(Ki + inhibited_by)",
        "parameters": ["k", "Ki"],
    }

    def change(row, binding, name):
        (tmp_path / name).mkdir()
        mcp = _statements_server(tmp_path / name)
        out = _call(
            mcp,
            "make_changes",
            {"justification": "C slows A becoming B", "statements": [row],
             "bindings": {"conv": binding}, "mechanisms": [inhibited]},
        )
        return mcp, out

    _, unstated = change("conv: A becomes B", "inhibited", "unstated")
    assert unstated["applied"] is False
    assert "reads ['inhibited_by'], which its statement does not state" in unstated["error"]
    _, modifier = change("conv: A becomes B", "inhibited inh:=C", "modifier")
    assert "`inh:=` names no regulator" in modifier["error"]
    mcp, stated = change("conv: A becomes B, inhibited by C", "inhibited", "stated")
    assert stated["applied"] is True, stated.get("error")
    rows = _call(mcp, "get_model", {"format": "statements"})["result"]
    assert "conv: A becomes B, inhibited by C\nbinding conv: inhibited" in rows


def test_a_modulator_takes_the_direction_its_rate_proves(tmp_path):
    """`modulated_by` dodged the direction check, so a statement could say less than its maths.
    It is accepted, and the direction its rate proves is kept beside what it states: in the
    version's record and in `get_model`, which reads back as the statement it was."""
    mcp = _statements_server(tmp_path)
    out = _call(
        mcp,
        "make_changes",
        {"justification": "C changes A becoming B", "statements": [
            "conv: A becomes B, modulated by C"], "bindings": {"conv": "slowed"},
         "mechanisms": [{"statement_type": "PopulationTransition", "policy": "slowed",
                         "based_on": "mass_action", "rate": "k*Ki/(Ki + modulated_by)",
                         "parameters": ["k", "Ki"]}]},
    )
    assert out["applied"] is True, out.get("error")
    rows = _call(mcp, "get_model", {"format": "statements"})["result"]
    assert "conv: A becomes B, modulated by C (inhibits, from its rate)\nbinding conv: slowed" in rows
    (row,) = json.loads(
        (tmp_path / "prov" / "versions" / "v0" / "statements.json").read_text()
    )["rows"]
    assert row["statement"]["modulated_by"] == [{"name": "C"}]
    assert row["emission"]["derived"] == {"modulated_by_0": "inhibited_by"}


def test_the_result_names_the_rows_whose_rate_signs_a_modulator(tmp_path):
    """The agent sees each change where it sees what the edit did: one warning per edit, naming
    every row whose rate records a regulator otherwise than stated, and asking it to reconsider."""
    out = _call(
        _statements_server(tmp_path),
        "make_changes",
        {"justification": "C slows both ways", "statements": [
            "conv: A becomes B, modulated by C",
            "back: B becomes A, modulated by C"], "bindings": {"conv": "slowed", "back": "slowed"},
         "mechanisms": [{"statement_type": "PopulationTransition", "policy": "slowed",
                         "based_on": "mass_action", "rate": "k*Ki/(Ki + modulated_by)",
                         "parameters": ["k", "Ki"]}]},
    )
    assert out["applied"] is True, out.get("error")
    signed = [m["message"] for m in out["messages"] if "from its rate" in m["message"]]
    assert len(signed) == 1, signed
    assert signed[0].startswith(
        "`conv, back`: you stated `modulated by C`, and its rate never rises as C rises, so "
        "it reads as `inhibited by C`. Check that each mechanism does what you meant; if not, "
        "rewrite its rate."
    )


#: Never negative whatever `a` (a rate law that can go negative claims no direction).
_LIN = {"statement_type": "PopulationTransition", "policy": "lin", "based_on": "mass_action",
        "rate": "k*exp(a*activated_by)", "parameters": ["k", "a"]}


def _recorded(tmp_path, version):
    (row,) = [r for r in json.loads(
        (tmp_path / "prov" / "versions" / f"v{version}" / "statements.json").read_text()
    )["rows"] if r["prefix"] == "conv"]
    return row["emission"]["derived"]


def test_a_bound_edit_signs_a_rows_regulators_again(tmp_path):
    """review of #128: `k*(1 + a*C)` was signed once, `a` taken positive, so `activated by C`
    stood after an edit let `a` go negative. Each version's kinds are its own bounds' (here
    `k*exp(a*C)`)."""
    mcp = _statements_server(tmp_path)
    out = _call(mcp, "make_changes", {"justification": "C speeds it",
                                      "statements": ["conv: A becomes B, activated by C"],
                                      "bindings": {"conv": "lin"},
                                      "mechanisms": [_LIN]})
    assert out["applied"] is True, out.get("error")
    assert _recorded(tmp_path, 0) == {}
    out = _call(mcp, "make_changes", {"justification": "C slows it after all", "parameters": [
        {"parameter_id": "conv_a", "lower_bound": -1.0, "upper_bound": -0.01, "scale": "lin"}]})
    assert out["applied"] is True, out.get("error")
    assert _recorded(tmp_path, 1) == {"activated_by_0": "inhibited_by"}
    said = "`conv`: you stated `activated by C`, but its rate never rises as C rises"
    assert any(m["message"].startswith(said) for m in out["messages"])
    # In the edit's own event too, where the audit trail reads it.
    events = [json.loads(line) for line in (tmp_path / "prov" / "provenance.jsonl").read_text().splitlines()]
    (edit,) = [e for e in events if e.get("type") == "edit" and e.get("args", {}).get("parameters")]
    assert any(w.startswith(said) for w in edit["warnings"])


def test_an_edit_reports_only_the_rows_it_touches(tmp_path):
    """review of #128, t1: under `statement_edits` every row whose rate records a kind was
    reported again, by its index in the new set, at each edit."""
    slowed = {"statement_type": "PopulationTransition", "policy": "slowed",
              "based_on": "mass_action", "rate": "k*Ki/(Ki + modulated_by)",
              "parameters": ["k", "Ki"]}
    mcp = _statements_server(tmp_path)
    out = _call(mcp, "make_changes", {"justification": "x", "mechanisms": [slowed], "statements": [
        "aa: A becomes B", "conv: A becomes B, modulated by C"],
        "bindings": {"aa": "mass_action", "conv": "slowed"}})
    assert out["applied"] is True, out.get("error")

    def said(out):
        return [m["message"] for m in out["messages"] if "from its rate" in m["message"]]

    assert said(out)[0].startswith("`conv`: you stated `modulated by C`")
    out = _call(mcp, "make_changes", {"justification": "y", "statement_edits": [
        {"op": "add_row", "row": "back: B becomes A", "binding": "mass_action"}]})
    assert out["applied"] is True, out.get("error")
    assert said(out) == []
    out = _call(mcp, "make_changes", {"justification": "z", "statement_edits": [
        {"op": "remove_row", "prefix": "aa"},
        {"op": "add_row", "row": "back2: B becomes A, modulated by C", "binding": "slowed"}]})
    assert out["applied"] is True, out.get("error")
    (message,) = said(out)
    assert message.startswith("`back2`: you stated `modulated by C`")


def test_a_json_row_shows_what_its_rate_records_and_reads_back(tmp_path):
    """review of #128, t4: a row the notation cannot write (evidence beyond one source) was shown
    as JSON without the kinds its rate records."""
    mcp = _statements_server(tmp_path)
    row = {"statement": {"type": "PopulationTransition", "source": {"name": "A"},
                         "target": {"name": "B"}, "modulated_by": [{"name": "C"}],
                         "evidence": [{"source_api": "lit", "pmid": "12345", "text": "C slows it"}]},
           "prefix": "conv"}
    slowed = {"statement_type": "PopulationTransition", "policy": "slowed",
              "based_on": "mass_action", "rate": "k*Ki/(Ki + modulated_by)",
              "parameters": ["k", "Ki"]}
    out = _call(mcp, "make_changes", {"justification": "x", "statements": [row],
                                      "bindings": {"conv": "slowed"}, "mechanisms": [slowed]})
    assert out["applied"] is True, out.get("error")
    model = _call(mcp, "get_model", {"format": "statements"})["result"]
    assert "binding conv: slowed" in model.split("\n")
    shown = [line for line in _call(mcp, "get_model", {"format": "statements"})["result"].split("\n")
             if line.startswith("{")]
    (line,) = shown
    assert json.loads(line)["derived"] == {"modulated_by_0": "inhibited_by"}
    out = _call(mcp, "make_changes", {"justification": "again", "statements": [json.loads(line)],
                                      "bindings": {"conv": "slowed"}})
    assert out["applied"] is True, out.get("error")


@pytest.mark.parametrize(
    "rows, bindings, match",
    [
        # review of #128, a_crossbind: a binding read another row's rate, and with it a regulator
        # this row does not state.
        (["inh: A becomes B, inhibited by C", "act: B becomes A, activated by D"],
         {"inh": "inh k=act_rate", "act": "act"},
         r"binds 'inh_k' over 'act_rate', an expression the model builds"),
        # t10: a binding naming a regulator minted a parameter of that name.
        (["r1: A becomes B"], {"r1": "mass_action k=k0*inhibited_by"},
         r"binds 'r1_k' over 'inhibited_by', a regulator's name"),
    ],
)
def test_a_binding_reads_parameters_only(tmp_path, rows, bindings, match):
    mechanisms = [
        {"statement_type": "PopulationTransition", "policy": "inh", "based_on": "mass_action",
         "rate": "k*Ki/(Ki + inhibited_by)", "parameters": ["k", "Ki"]},
        {"statement_type": "PopulationTransition", "policy": "act", "based_on": "mass_action",
         "rate": "k*activated_by", "parameters": ["k"]},
    ]
    out = _call(_statements_server(tmp_path), "make_changes",
                {"justification": "x", "statements": rows, "bindings": bindings,
                 "mechanisms": mechanisms})
    assert out["applied"] is False
    assert re.search(match, out["error"]), out["error"]


def test_an_unknown_activity_names_the_modification_tags():
    """`A{mod:on}` (a scheme's site, written in a row) built an activity named "mod"."""
    from petab_mcp import notation

    with pytest.raises(notation.NotationError, match="unknown activity 'mod'"):
        notation.parse_agent("A{mod:on}")


def test_an_undefined_readout_is_remedied_in_the_routes_own_terms(tmp_path):
    """On the statements route a readout is given in `readouts`; the assignment-rule remedy sent
    agents to an Antimony op their editor does not take."""
    mcp = _readout_server(tmp_path)
    out = _call(mcp, "make_changes", {"justification": "no readout", "statements": [_CONV],
                                      "bindings": {"conv": "mass_action"}})
    assert out["applied"] is False
    assert "in `readouts`" in out["error"] and "set_assignment" not in out["error"]


def test_a_failed_fit_names_the_points_a_readout_leaves_undefined():
    """`pA/(pA + pB)` with both starting at 0 is 0/0 at t=0; every start of the hierarchical fit
    failed, and the refusal named only two other causes."""
    import types

    import libsbml
    import pandas as pd

    from petab_mcp.fitting import _undefined_points

    doc = libsbml.SBMLDocument(3, 2)
    sm = doc.createModel()
    rule = sm.createAssignmentRule()
    rule.setVariable("readout_r")
    rule.setMath(libsbml.parseL3Formula("pA/(pA + pB)"))
    session = types.SimpleNamespace(
        problem=types.SimpleNamespace(
            observable_df=pd.DataFrame(
                {"observableFormula": ["100*readout_r", "readout_p"]},
                index=pd.Index(["r", "p"], name="observableId"),
            ),
            model=types.SimpleNamespace(sbml_model=sm),
        )
    )
    records = [
        {"observable_id": "r", "time": 0.0, "measurement": 14.7, "simulation": None},
        {"observable_id": "r", "time": 2.5, "measurement": 15.0, "simulation": 0.4},
        {"observable_id": "p", "time": 0.0, "measurement": 1.0, "simulation": 0.0},
    ]
    text = _undefined_points(session, records)
    assert text == "`r` at t=0; it reads `readout_r := pA / (pA + pB)`"
    # Nothing simulated: an integration failure, the other message's case.
    assert _undefined_points(session, [dict(r, simulation=None) for r in records]) == ""


# -- rates, rules and readouts written in the row notation --------------------------
def test_a_scheme_rate_reads_a_molecule_written_in_the_row_notation(tmp_path):
    """A feedback written `amount(A{pS10})` in a scheme's rate did not parse: the notation's
    braces are no Python."""
    mcp = _statements_server(tmp_path)
    out = _call(
        mcp,
        "make_changes",
        {
            "justification": "A becomes B, slowed by its own phosphorylated form",
            "statements": ["conv: A becomes B", "ap: A is phosphorylated at S10"],
            "bindings": {"conv": "fb", "ap": "basal"},
            "mechanisms": [
                {
                    "statement_type": "PopulationTransition",
                    "policy": "fb",
                    "scheme": "source() >> target() @ k/(1 + amount(A{pS10})/K_fb)",
                    "parameters": ["k", "K_fb"],
                }
            ],
        },
    )
    assert out["applied"] is True, out.get("error")
    rows = {r["parameterId"] for r in _call(mcp, "get_parameters", {})["result"]}
    assert {"conv_k", "conv_K_fb"} <= rows


def test_a_rate_that_does_not_read_says_how_one_is_written():
    from petab_mcp import schemes
    from petab_mcp.errors import MechanismError

    with pytest.raises(MechanismError, match="the molecule written as in a row"):
        schemes._rate("k*amount(A{pS10}")


def test_a_rule_that_never_fires_names_the_form_nothing_makes(tmp_path):
    """A row moving a form out of a compartment nothing puts it in never fires, and the refusal
    only said that such forms exist."""
    mcp = _statements_server(tmp_path)
    out = _call(
        mcp,
        "make_changes",
        {
            "justification": "A moves out of the nucleus, which it never enters",
            "statements": ["conv: A becomes B", "out: A{@nucleus} moves from nucleus to cytoplasm"],
            "bindings": {"conv": "mass_action", "out": "default"},
        },
    )
    assert out["applied"] is False
    assert "nothing forms out: A(" in out["error"] and "nucleus" in out["error"]


def test_an_enzyme_less_modification_under_a_written_rate_names_basal(tmp_path):
    mcp = _statements_server(tmp_path)
    out = _call(
        mcp,
        "make_changes",
        {
            "justification": "A phosphorylated with no enzyme, re-rated",
            "statements": ["conv: A becomes B", "ap: A is phosphorylated"],
            "bindings": {"conv": "mass_action", "ap": "slow"},
            "mechanisms": [
                {"statement_type": "Phosphorylation", "policy": "slow", "rate": "k", "parameters": ["k"]}
            ],
        },
    )
    assert out["applied"] is False and "`basal`" in out["error"]


def test_a_readout_given_as_a_bare_pattern_is_its_amount():
    """`"readout_pA": "A{pS10}"` was refused as no formula."""
    from petab_mcp.operations import _readout_amounts

    readouts, amounts = _readout_amounts({"readout_pA": "A{pS10}"}, None)
    assert readouts == {"readout_pA": "amount_A_pS10"}
    assert amounts == {"amount_A_pS10": "A{pS10}"}


def test_amount_names_keep_the_notations_negations():
    from petab_mcp.operations import _readout_amounts

    readouts, amounts = _readout_amounts({"r": "amount(S{pY694}) + amount(S{-pY694})"}, None)
    assert sorted(amounts) == ["amount_S_no_pY694", "amount_S_pY694"]


def test_an_amount_in_place_reads_the_amount_that_holds_its_pattern():
    """A pattern is named by the agent it reads, however it is spelled."""
    from petab_mcp.operations import _readout_amounts

    readouts, amounts = _readout_amounts(
        {"r": "amount(S{@nucleus, pY694}) / amount( S{pY694,@nucleus} )"},
        {"pS_nuc": "S{pY694, @nucleus}"},
    )
    assert readouts == {"r": "pS_nuc / pS_nuc"} and list(amounts) == ["pS_nuc"]


def test_a_server_amount_name_no_amount_holds_is_refused():
    """Read where no `amount(...)` names it, `amount_B` would be a parameter fitted in place of B."""
    from petab_mcp.operations import _readout_amounts

    with pytest.raises(OperationError, match=r"\['amount_B'\]"):
        _readout_amounts({"r": "scale*amount_B + amount(A)"}, None)
    assert _readout_amounts({"r": "amount_B + amount(B)"}, None)[0] == {"r": "amount_B + amount_B"}


def test_translocates_is_moves():
    from petab_mcp import notation

    row = notation.parse("imp: S{pY694} translocates from cytoplasm to nucleus")
    assert row["statement"]["type"] == "Translocation"
    assert row["statement"]["to_location"] == "nucleus"


def test_an_empty_map_names_the_held_inputs_and_binds_nothing_else(tmp_path):
    """With an empty map the server holds an input to its protocol, and the route's paragraph
    says so, without claiming the map binds the observables."""
    shutil.copytree(YAML_PATH.parent, tmp_path / "problem")
    (tmp_path / "problem" / "inputs.tsv").write_text(
        "inputId\tformula\tconditionInputs\nA\tpiecewise(1, time <= 5, 0)\t\n"
    )
    entity_map = tmp_path / "entities.json"
    entity_map.write_text("{}")
    mcp = create_server(
        str(tmp_path / "problem" / "problem.yaml"),
        provenance_dir=str(tmp_path / "prov"),
        resolver=StubReferenceResolver(),
        auto_fit=False,
        entity_map=str(entity_map),
    )
    description = next(t for t in asyncio.run(mcp.list_tools()) if t.name == "make_changes").description
    assert "NOTHING is bound" in description
    assert "`A` is HELD to the experiment's protocol, `A = piecewise(1, time <= 5, 0)`" in description


def _concurrently(monkeypatch, mcp, *calls):
    """The ``make_changes`` calls, each on its own thread, each pausing after it builds its
    statements until every other has built too (or 3 s pass). Each takes the session lock in turn,
    as a call does that arrives just as another takes it: the refusal of a call arriving during
    another's fit (``fit_in_progress``, tests/test_multistart.py) is turned off here."""
    import threading

    monkeypatch.setattr(PetabSession, "change", lambda session: session.lock)
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
        {"justification": m["policy"], "statements": [_statement_row()],
         "bindings": {"conv": m["policy"]}, "mechanisms": [m]}
        for m in (_SATURATING, other)
    ]
    assert all(out["applied"] for out in _concurrently(monkeypatch, mcp, *calls))
    rows = [{**_statement_row(), "prefix": p} for p in ("saturating", "hill")]
    both = _call(mcp, "make_changes", {"justification": "both", "statements": rows,
                                       "bindings": {p: p for p in ("saturating", "hill")}})
    assert both["applied"] is True, both.get("error")

    (tmp_path / "redefined").mkdir()
    mcp = _statements_server(tmp_path / "redefined")
    # Valid on its own: a definition that drops `K` is malformed, refused first when it wins the lock.
    changed = {**_SATURATING, "rate": "2 * vmax / (K + source)"}
    calls = [{**call, "mechanisms": [m]} for call, m in zip(calls, (_SATURATING, changed))]
    calls[1]["statements"], calls[1]["bindings"] = calls[0]["statements"], calls[0]["bindings"]
    outs = _concurrently(monkeypatch, mcp, *calls)
    assert sorted(bool(out["applied"]) for out in outs) == [False, True]
    assert [out.get("error_cause") for out in outs if not out["applied"]] == ["mechanism_redefined"]


def test_concurrent_statement_edits_keep_each_others_rows(tmp_path, monkeypatch):
    """Both edits read HEAD's rows before either committed, so the later commit dropped the
    row the earlier one added."""
    mcp = _built(tmp_path, [_CONV], {"conv": "mass_action"})
    calls = [
        {"justification": p, "statement_edits": [
            {"op": "add_row", "row": f"{p}: B becomes A", "binding": "mass_action"}]}
        for p in ("back", "back2")
    ]
    assert all(out["applied"] for out in _concurrently(monkeypatch, mcp, *calls))
    text = _text(mcp, "get_model", {"format": "statements"})
    assert "conv: A becomes B" in text and all(f"{p}: B becomes A" in text for p in ("back", "back2"))
