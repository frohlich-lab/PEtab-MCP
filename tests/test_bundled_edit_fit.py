"""Optional bundled parameter changes + fit on the editors (issue #114).

The lead's step budget is dominated by the ``set_model -> change_parameters -> fit`` cycle, which
is almost always one logical action. ``set_model(parameters=[...], fit=True)`` (and
``change_parameters(..., fit=True)``) collapse it into one call — WITHOUT re-coupling editing and
fitting: an unbundled call behaves exactly as before (issue #81).

The ``parameters=`` transaction is tested directly against a ``PetabSession`` (no engine needed);
the ``fit=`` bundle is a server-layer concern, so those tests drive the in-process MCP server the
way ``test_server.py`` does.
"""

from __future__ import annotations

import asyncio

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

YAML_PATH = Path(__file__).parent / "data" / "toy_v1" / "problem.yaml"


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


# -- set_model(parameters=[...]): one atomic transaction -----------------------
def test_set_model_with_parameters_applies_both_atomically(session):
    """The re-author AND the bound/scale edits land in ONE edit — one provenance event, one
    pending edit — so the pair costs a single call instead of two."""
    res = O.set_model(
        session, _antimony("r3: A => ; k3 * A;"),
        justification="add a degradation step and give its rate a wider box",
        parameters=[
            {"parameter_id": "k1", "lower_bound": 1e-3, "upper_bound": 1e3},
            {"parameter_id": "sigma_B", "scale": "log10"},
        ],
    )
    assert res["applied"] is True and res["version"] is None
    assert res["pending_fit"] is True and res["n_pending_edits"] == 1
    assert "fit" not in res and "committed" not in res  # still a fit-free edit

    recs = _records(session)
    assert "k3" in recs  # the re-author took
    assert float(recs["k1"]["lowerBound"]) == 1e-3 and float(recs["k1"]["upperBound"]) == 1e3
    assert recs["sigma_B"]["parameterScale"] == "log10"

    edits = [e for e in session.provenance.recent_events() if e["type"] == "edit"]
    assert len(edits) == 1
    assert edits[-1]["operation"] == "set_model"
    # the event records both halves of the change
    assert edits[-1]["args"]["parameters"][0]["parameter_id"] == "k1"
    assert edits[-1]["args"]["antimony_chars"] > 0


def test_bundled_parameters_can_target_a_newly_introduced_parameter(session):
    """The parameter edit runs AFTER the reconcile, so it can bound a symbol this very re-author
    introduced — the whole point of bundling them."""
    res = O.set_model(
        session, _antimony("r3: A => ; k3 * A;"),
        justification="bound the new degradation rate",
        parameters=[{"parameter_id": "k3", "lower_bound": 1e-4, "upper_bound": 1e1}],
    )
    assert res["applied"] is True
    recs = _records(session)
    assert float(recs["k3"]["lowerBound"]) == 1e-4 and float(recs["k3"]["upperBound"]) == 1e1


def test_set_model_parameters_rejection_reverts_the_re_author(session):
    """A rejected parameter entry reverts the MODEL too — nothing is half-applied."""
    before_model = session.problem.model.to_sbml_str()
    before_params = _records(session)

    with pytest.raises(OperationError, match="only one bound"):
        O.set_model(
            session, _antimony("r3: A => ; k3 * A;"),
            justification="bad bound entry",
            parameters=[{"parameter_id": "k1", "lower_bound": 1e-3}],
        )

    assert session.problem.model.to_sbml_str() == before_model
    assert _records(session) == before_params
    assert not session.has_unfit_edits()
    assert not [e for e in session.provenance.recent_events() if e["type"] == "edit"]


def test_set_model_parameters_entry_for_a_dropped_parameter_is_rejected(session):
    """An entry naming a parameter the re-author no longer declares fails loudly (the table is
    checked post-reconcile) rather than silently doing nothing."""
    with pytest.raises(ParameterError, match="Unknown parameter id 'k3'"):
        O.set_model(
            session, _antimony(),  # the base model has no k3
            justification="bound a parameter that does not exist",
            parameters=[{"parameter_id": "k3", "lower_bound": 1e-4, "upper_bound": 1e1}],
        )
    assert "k3" not in _records(session)


def test_set_model_without_parameters_is_unchanged(session):
    """The default path is byte-identical to before: no parameter edit, no fit."""
    res = O.set_model(session, _antimony(), justification="plain re-author")
    assert res["applied"] is True and res["version"] is None and res["pending_fit"] is True
    assert "fit" not in res
    edits = [e for e in session.provenance.recent_events() if e["type"] == "edit"]
    assert "parameters" not in edits[-1]["args"]


@pytest.mark.parametrize("empty", [None, []])
def test_set_model_empty_parameters_means_no_parameter_edit(session, empty):
    res = O.set_model(session, _antimony(), justification="no parameter edits", parameters=empty)
    assert res["applied"] is True
    assert "parameters" not in [
        e for e in session.provenance.recent_events() if e["type"] == "edit"][-1]["args"]


def test_set_model_rejects_a_malformed_parameters_argument(session):
    with pytest.raises(OperationError, match="set_model: 'parameters'"):
        O.set_model(session, _antimony(), justification="x", parameters={"parameter_id": "k1"})


@pytest.mark.parametrize("bad", ["1e-3", None, True, float("nan"), float("inf")])
def test_a_non_numeric_bound_is_a_clean_rejection(session, bad):
    """Bounds are type-checked during normalization, not left to `update_parameters`' bare
    `float(value)`: a ValueError there is not a PetabMcpError, so it would escape the tool's error
    wrapper as an unstructured MCP failure instead of `applied: false`. (`None` trips the
    both-bounds-or-neither rule; the rest trip the numeric/finite checks.)"""
    with pytest.raises(OperationError):
        O.set_model(
            session, _antimony(), justification="x",
            parameters=[{"parameter_id": "k1", "lower_bound": bad, "upper_bound": 1e3}])
    assert not session.has_unfit_edits()


# -- the fit= bundle (server layer) -------------------------------------------
def _server(tmp_path, auto_fit=False):
    return create_server(
        str(YAML_PATH),
        provenance_dir=str(tmp_path / "prov"),
        resolver=StubReferenceResolver(),
        auto_fit=auto_fit,
    )


def _call(mcp, name, args):
    return asyncio.run(mcp.call_tool(name, args))[1]  # (content_blocks, structured_result)


def _head(mcp):
    """HEAD version + pending-edit state. There is no `current_version` tool — everything it
    returned lives on `describe_problem` (only the version key is named differently)."""
    return _call(mcp, "describe_problem", {})


def _listed(result):
    """FastMCP wraps a list return in {"result": [...]}."""
    return result["result"] if isinstance(result, dict) and "result" in result else result


def test_bundled_call_produces_exactly_one_version(tmp_path):
    """set_model + parameters + fit as ONE call: one edit event, one fit event, one version."""
    mcp = _server(tmp_path, auto_fit=True)
    res = _call(mcp, "set_model", {
        "antimony": _antimony(),
        "justification": "re-author and retune k1, scored in one call",
        "parameters": [{"parameter_id": "k1", "lower_bound": 1e-3, "upper_bound": 1e3}],
        "fit": True,
        "n_starts": 2,
    })
    assert res["applied"] is True
    assert res["committed"] is True and res["version"] == 1
    assert res["pending_fit"] is False and res["n_pending_edits"] == 0
    # the whole fit result is nested, untouched
    assert res["fit"]["committed"] is True and res["fit"]["version"] == 1
    assert res["fit"]["neg_log_likelihood"] is not None and res["fit"]["bic"] is not None

    head = _head(mcp)
    assert head["current_version"] == 1 and head["has_unfit_edits"] is False
    events = _listed(_call(mcp, "get_provenance", {}))
    assert len([e for e in events if e["type"] == "edit"]) == 1
    assert len([e for e in events if e["type"] == "fit"]) == 1
    # the one fit scored the one bundled edit
    assert [e for e in events if e["type"] == "fit"][-1]["n_edits"] == 1
    recs = {r["parameterId"]: r for r in _listed(_call(mcp, "get_parameters", {}))}
    assert float(recs["k1"]["lowerBound"]) == 1e-3


def test_change_parameters_bundled_fit(tmp_path):
    """The parameters-only retuning scores in one call too."""
    mcp = _server(tmp_path, auto_fit=True)
    res = _call(mcp, "change_parameters", {
        "parameters": [{"parameter_id": "k1", "lower_bound": 1e-3, "upper_bound": 1e3}],
        "justification": "widen k1 and score it",
        "fit": True,
        "n_starts": 2,
    })
    assert res["applied"] is True and res["committed"] is True and res["version"] == 1
    assert res["fit"]["bic"] is not None
    assert _head(mcp)["has_unfit_edits"] is False


def test_bundled_dedup_skip_reports_no_change(tmp_path):
    """The de-dup still sees the same structure hash through the bundle: re-authoring the
    identical structure and asking for a fit is a no-op, not a second version."""
    mcp = _server(tmp_path, auto_fit=True)
    first = _call(mcp, "set_model", {
        "antimony": _antimony(), "justification": "initial", "fit": True, "n_starts": 2})
    assert first["committed"] is True and first["version"] == 1

    again = _call(mcp, "set_model", {
        "antimony": _antimony(), "justification": "same structure again",
        "fit": True, "n_starts": 2})
    assert again["applied"] is True          # the edit itself is always cheap
    assert again["committed"] is False
    assert again["fit"]["no_change"] is True
    assert again["fit"]["duplicate_of_version"] == 1
    assert again["version"] == 1
    # the no-op consumed the pending edit and created no version
    assert again["pending_fit"] is False
    assert _head(mcp)["current_version"] == 1


def test_bundled_fit_failure_leaves_the_edit_in_place(tmp_path, monkeypatch):
    """The error semantics the issue calls out: a bundled call that edits fine but fails to fit
    keeps the edit (nothing half-applied) and reports the fit failure actionably."""
    def _boom(self, *a, **k):
        raise ModelUpdateError("boom")

    monkeypatch.setattr(PetabSession, "fit", _boom)
    mcp = _server(tmp_path)

    res = _call(mcp, "set_model", {
        "antimony": _antimony("r3: A => ; k3 * A;"),
        "justification": "an edit whose fit blows up",
        "parameters": [{"parameter_id": "k1", "lower_bound": 1e-3, "upper_bound": 1e3}],
        "fit": True,
    })
    assert res["applied"] is True                 # the edit DID apply
    assert res["committed"] is False and res["version"] is None
    assert res["fit"]["committed"] is False and "boom" in res["fit"]["error"]
    assert res["pending_fit"] is True and res["n_pending_edits"] == 1
    assert "still in place" in res["message"]

    # the working model keeps the edit, ready to be repaired and re-fit
    head = _head(mcp)
    assert head["has_unfit_edits"] is True and head["current_version"] == 0
    assert "k3" in {r["parameterId"] for r in _listed(_call(mcp, "get_parameters", {}))}
    # and the failure is logged like any other fit rejection
    assert any("boom" in (e.get("message") or "")
               for e in _listed(_call(mcp, "get_errors", {})))


def test_a_refused_fit_forwards_its_own_error_instead_of_prescribing_a_repair(
        tmp_path, monkeypatch):
    """The identical-structure loop guard lands in the same "not scored" branch as a broken model,
    but its error says the OPPOSITE (stop re-fitting, restore_version(n)). The top-level message
    must forward that rather than tell the agent to repair and re-fit — the agent reads the message
    first, and "re-fit" walks straight back into the loop the guard exists to break."""
    def _refuse(self, *a, **k):
        raise ModelUpdateError(
            "This same model structure has been fit 4 times now. Stop re-fitting it: "
            "go back to it with restore_version(n).")

    monkeypatch.setattr(PetabSession, "fit", _refuse)
    mcp = _server(tmp_path)
    res = _call(mcp, "set_model", {
        "antimony": _antimony(), "justification": "re-author a seen structure", "fit": True})

    assert res["applied"] is True and res["committed"] is False
    assert "restore_version" in res["message"]          # the refusal's own instruction survives
    assert "repair the model, and call fit() again" not in res["message"]
    assert "still in place" in res["message"]           # the edit state is still reported


def test_rejected_edit_never_fits(tmp_path):
    """fit=true on an edit that is REJECTED must not run a fit — the result is the plain
    rejection shape, with no `fit` key to misread."""
    mcp = _server(tmp_path)
    res = _call(mcp, "set_model", {
        "antimony": _antimony(), "justification": "",  # empty justification is rejected
        "fit": True,
    })
    assert res["applied"] is False and "fit" not in res and "committed" not in res
    assert _head(mcp)["current_version"] == 0


def test_n_starts_without_fit_is_rejected_not_ignored(tmp_path):
    """An argument that would be silently ignored is a rejection (same principle as the
    unknown-entry-key guard): n_starts without fit=true means an expected fit that never ran."""
    mcp = _server(tmp_path)
    for name, args in (
        ("set_model", {"antimony": _antimony(), "justification": "x", "n_starts": 4}),
        ("change_parameters", {
            "parameters": [{"parameter_id": "k1", "lower_bound": 1e-3, "upper_bound": 1e3}],
            "justification": "x", "n_starts": 4}),
    ):
        res = _call(mcp, name, args)
        assert res["applied"] is False and "n_starts" in res["error"], name
    # nothing was edited, and the rejections are auditable — including the refused VALUE, so the
    # log says which argument was wrong rather than only repeating the message
    assert _head(mcp)["has_unfit_edits"] is False
    logged = [e for e in _listed(_call(mcp, "get_errors", {}))
              if "n_starts" in (e.get("message") or "")]
    assert len(logged) == 2
    assert all(e["args"] == {"fit": False, "n_starts": 4} for e in logged)


def test_the_multistart_default_and_the_starts_actually_used_are_reported(tmp_path):
    """`n_starts` REPLACES the server's multistart default, so both numbers are exposed rather
    than left to be inferred: `describe_problem.fit_n_starts_default` before the call, and the fit
    result's own `n_starts` after it."""
    mcp = _server(tmp_path, auto_fit=True)
    default = _call(mcp, "describe_problem", {})["fit_n_starts_default"]
    assert default == DEFAULT_AUTO_FIT_STARTS

    # omitted -> the default was used
    res = _call(mcp, "set_model", {
        "antimony": _antimony(), "justification": "score with the default starts", "fit": True})
    assert res["fit"]["n_starts"] == default

    # given -> exactly that count, not default + n_starts
    res = _call(mcp, "change_parameters", {
        "parameters": [{"parameter_id": "k1", "lower_bound": 1e-3, "upper_bound": 1e3}],
        "justification": "score with an explicit count", "fit": True, "n_starts": default + 2})
    assert res["fit"]["n_starts"] == default + 2


def test_a_harder_fit_of_an_unchanged_model_is_not_skipped(tmp_path):
    """The structure hash and the setup signature both ignore the multistart count, so a de-dup
    skip would silently return the WEAKER stored metrics for a deliberate "fit harder". A fit with
    more starts than HEAD's is a genuinely different search and must run; the same or fewer is the
    redundant refit the de-dup exists to skip."""
    mcp = _server(tmp_path, auto_fit=True)
    first = _call(mcp, "fit", {"n_starts": 2})
    assert first["committed"] is True and first["n_starts"] == 2

    # same count -> still the no-op, and it reports the starts its metrics came from
    same = _call(mcp, "fit", {"n_starts": 2})
    assert same["no_change"] is True and same["n_starts"] == 2

    # more starts -> a real fit, with a warning explaining why it wasn't skipped
    harder = _call(mcp, "fit", {"n_starts": 4})
    assert harder["committed"] is True and harder["version"] == 2
    assert harder["n_starts"] == 4
    assert any("multistarts against its 2" in w for w in harder["warnings"])

    # fewer starts than the new HEAD -> back to a no-op, not a weaker refit
    weaker = _call(mcp, "fit", {"n_starts": 3})
    assert weaker["no_change"] is True and _head(mcp)["current_version"] == 2


def test_a_bad_n_starts_is_rejected_before_the_edit_applies(tmp_path):
    """A bad n_starts is validated at the tool boundary, not inside session.fit: rejecting it only
    after the edit had applied would leave an unscored edit behind for an argument typo. (A
    NON-integer never gets this far — the tool schema's `int | None` rejects it outright; a
    negative one parses fine and is what this guard is for.)"""
    mcp = _server(tmp_path)
    res = _call(mcp, "set_model", {
        "antimony": _antimony("r3: A => ; k3 * A;"),
        "justification": "a good edit with a typo'd n_starts", "fit": True, "n_starts": -1})

    assert res["applied"] is False and "n_starts" in res["error"]
    assert "fit" not in res            # the fit never ran
    head = _head(mcp)
    assert head["has_unfit_edits"] is False and head["current_version"] == 0
    assert "k3" not in {r["parameterId"] for r in _listed(_call(mcp, "get_parameters", {}))}


def test_a_fit_that_breaks_after_snapshotting_reports_the_new_version(tmp_path, monkeypatch):
    """The crash path `edit_then_fit` degrades runs PAST the version snapshot, which consumes the
    pending edits and advances HEAD. Reporting "still in place, call fit() again" there would send
    the agent to re-fit a model that is already scored, so this case reports where HEAD landed."""
    real_fit = PetabSession.fit

    def _fit_then_break(self, *a, **k):
        real_fit(self, *a, **k)          # snapshots the version and clears the pending edits
        raise RuntimeError("late boom")  # ... then breaks, as a result-block bug would

    monkeypatch.setattr(PetabSession, "fit", _fit_then_break)
    mcp = _server(tmp_path, auto_fit=True)

    res = _call(mcp, "set_model", {
        "antimony": _antimony(), "justification": "fit breaks after the snapshot",
        "fit": True, "n_starts": 2})

    assert res["applied"] is True and res["committed"] is False
    assert "late boom" in res["fit"]["error"]
    # HEAD really did advance, so say so instead of claiming the edit is unscored
    assert res["pending_fit"] is False
    assert res["version"] == 1 == _head(mcp)["current_version"]
    assert "do NOT re-fit" in res["message"] and "still in place" not in res["message"]


def test_head_advancing_is_detected_even_while_the_edits_stay_pending(tmp_path, monkeypatch):
    """`has_unfit_edits()` is only a PROXY for "a version was created": `session.fit` clears the
    pending edits several statements after it snapshots, so a failure in that window (an event-log
    write, say) leaves them pending with HEAD already advanced. The version comparison must still
    see it, or the agent is told to re-fit an already-scored model."""
    def _snapshot_then_break(self, *a, **k):
        self._snapshot_version(label="fit", kind="fit", meta={"fit_status": "fit"}, sim=None)
        raise RuntimeError("provenance write failed")   # ... before pending_edits is cleared

    monkeypatch.setattr(PetabSession, "fit", _snapshot_then_break)
    mcp = _server(tmp_path)

    res = _call(mcp, "set_model", {
        "antimony": _antimony(), "justification": "snapshot lands, event log does not",
        "fit": True})

    assert res["applied"] is True and res["committed"] is False
    assert res["pending_fit"] is True          # the edits really are still pending — report that
    assert res["version"] == 1                 # ... but HEAD moved, and that is what matters here
    assert "do NOT re-fit" in res["message"]


def test_bundled_fit_crash_is_degraded_to_a_failed_fit(tmp_path, monkeypatch):
    """An UNEXPECTED exception from the fit must not surface as a failed edit: the edit already
    applied, so it is reported as a failed fit with the edit left in place."""
    def _crash(self, *a, **k):
        raise RuntimeError("kaboom")

    monkeypatch.setattr(PetabSession, "fit", _crash)
    mcp = _server(tmp_path)

    res = _call(mcp, "set_model", {
        "antimony": _antimony("r3: A => ; k3 * A;"),
        "justification": "an edit whose fit crashes", "fit": True})
    assert res["applied"] is True and res["committed"] is False
    assert "kaboom" in res["fit"]["error"]
    assert _head(mcp)["has_unfit_edits"] is True


def test_unbundled_result_shape_unchanged(tmp_path):
    """fit defaults to false: both editors return exactly the fit-free edit shape."""
    mcp = _server(tmp_path)
    for name, args in (
        ("set_model", {"antimony": _antimony(), "justification": "plain edit"}),
        ("change_parameters", {
            "parameters": [{"parameter_id": "k1", "lower_bound": 1e-3, "upper_bound": 1e3}],
            "justification": "plain edit"}),
    ):
        res = _call(mcp, name, args)
        assert res["applied"] is True, name
        assert res["version"] is None and res["pending_fit"] is True, name
        assert "fit" not in res and "committed" not in res, name
    assert _head(mcp)["current_version"] == 0
