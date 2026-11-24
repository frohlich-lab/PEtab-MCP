"""The explicit `fit` tool and fit-free edits (issue #81).

Edits mutate the working model but do NOT fit or version; the dedicated ``fit`` method fits the
current working model, snapshots one version, records the early-stop op, and clears the pending
edits. The de-duplication that used to gate ``set_model`` (issue #59) now gates ``fit``.

The engine-based tests compile the toy model with AMICI (like ``test_fitting``/``test_early_stop``
integration), so they are slower; the fit-free bookkeeping tests need no engine.
"""

from __future__ import annotations

import pytest

from petab_mcp import operations as O
from petab_mcp.errors import ModelUpdateError


# -- the batch tools are gone -------------------------------------------------
def test_batch_api_removed(session):
    """begin_batch/commit_batch/abort_batch (and the batch state) are deleted — edits are
    fit-free, so batching is obsolete."""
    for name in ("begin_batch", "commit_batch", "abort_batch", "batch_active", "staged"):
        assert not hasattr(session, name), name
    assert hasattr(session, "fit")
    # ...and so is the pending-edit state that replaced batching: `make_changes` fits on every
    # call, so an edit is never waiting to be scored and there is nothing to track.
    for name in ("pending_edits", "has_unfit_edits", "record_edit", "note_fit_attempt"):
        assert not hasattr(session, name), name


# -- the edit half on its own creates no version ------------------------------
def test_the_edit_half_creates_no_version(session):
    """The edit half mutates the working model and returns what it changed; the fit versions it."""
    res = O.change_parameters(
        session,
        [{"parameter_id": "k1", "lower_bound": 1e-3, "upper_bound": 1e3}],
        justification="widen",
    )
    assert res["applied"] is True and res["version"] is None
    assert res["staged_edit"]["justification"] == "widen"
    assert session.provenance.current_version == -1  # no version created by an edit


# -- the zero-estimated-params guard (was in test_batch) ----------------------
def test_zero_estimated_params_guard(session):
    """A fit of a model with no free parameters is rejected with a clear message BEFORE pyPESTO
    runs. The guard lives in ``_auto_fit_and_improvement`` and fires before the engine is touched,
    so a dummy engine is enough."""
    from petab.v2.C import ESTIMATE

    session.engine = object()  # non-None so the fit path is entered (guard raises before use)
    session.auto_fit = True
    session.problem.parameter_df[ESTIMATE] = False  # nothing left to estimate
    with pytest.raises(ModelUpdateError, match="NO estimated"):
        session._auto_fit_and_improvement(n_starts=1)


def test_a_seeded_problem_always_has_a_model(empty_session):
    """`fit` used to refuse when there was no model. It no longer needs to: a model-less problem
    is seeded with a valid one at construction, so there IS always a model — and a problem that
    cannot be seeded never gets this far, it fails to construct (see test_engine).

    Having a model is not the same as having something to estimate: this fixture ships every
    parameter fixed, so the seed's n_estimated is 0 and a fit of it would be refused by the
    zero-estimated-params guard above — on its own terms, with its own message."""
    assert empty_session.has_model() is True
    assert empty_session.model_authored() is False


# -- fit: fit + snapshot (needs the engine) -----------------------------------
def test_fit_scores_edits_and_snapshots_a_version(fit_session):
    sess = fit_session
    edit = O.change_parameters(
        sess,
        [{"parameter_id": "k1", "lower_bound": 1e-3, "upper_bound": 1e3}],
        justification="widen",
    )

    res = sess.fit(n_starts=2, staged_edits=[edit["staged_edit"]])
    assert res["committed"] is True
    assert res["version"] == 0
    assert res["neg_log_likelihood"] is not None and res["bic"] is not None
    assert "progress" in res
    assert sess.provenance.current_version == 0
    versions = sess.provenance.list_versions()
    assert versions[-1]["kind"] == "fit"
    # the fit event lists the edits it scored
    fit_ev = [e for e in sess.provenance.recent_events() if e["type"] == "fit"][-1]
    assert fit_ev["n_edits"] == 1


def test_fit_is_a_noop_when_model_unchanged(fit_session):
    """Fitting again with no edits since the last fit skips the refit and returns the stored
    metrics (the de-dup relocated from set_model to fit — issue #59/#81)."""
    from petab_mcp import data_access

    sess = fit_session
    first = sess.fit(n_starts=2)
    assert first["committed"] is True and first["version"] == 0

    again = sess.fit(n_starts=2)
    assert again["committed"] is False
    assert again["no_change"] is True
    assert again["duplicate_of_version"] == 0
    assert again["neg_log_likelihood"] == pytest.approx(first["neg_log_likelihood"])
    # the structure verdict comes back on the no-op too: every doc tells the agent to read it off
    # the fit result, so the de-duplicated shape must not be the one that lacks it. The state is
    # byte-identical to the duplicate version — that is what "no_change" means — so it matches.
    assert again["residual_structure"] == first["residual_structure"]
    # ...and so does the NOISE verdict. It must not be omitted: `noise_sigma_advice` is absent
    # when the noise model is plausible, so a no-op carrying none of these keys would read as
    # "plausible" — and the agent may have no get_fit_summary to check against.
    for key in ("noise_sigma_inflated", "noise_sigma_collapsed"):
        assert key in again, key
        assert again[key] == first[key]
    assert again.get("noise_sigma_report") == first.get("noise_sigma_report")
    assert again.get("noise_sigma_advice") == first.get("noise_sigma_advice")
    # plus the duplicate version's stored flag, mirrored like the other scalar meta keys
    assert again["residual_flagged"] == data_access.residual_flag(first["residual_structure"])
    assert again["residual_flagged"] == sess.provenance.get_meta(0)["residual_flagged"]
    # no churn: no new version was created by the no-op fit
    assert sess.provenance.current_version == 0


def test_zero_starts_is_rejected(fit_session):
    """n_starts=0 used to mean "simulate without optimising", which snapshotted a VERSION whose
    parameters were never optimised — it recorded whatever the nominal values happened to be and
    then competed on BIC against versions that were actually fit. Every change is scored now, so
    a count below 1 is refused rather than quietly producing an unscored version."""
    from petab_mcp.errors import ValidationError

    with pytest.raises(ValidationError, match=">= 1"):
        fit_session.fit(n_starts=0)
    assert fit_session.provenance.current_version == -1


# -- the residual-structure verdict rides on the fit result (issue #113) ------
def test_fit_returns_the_compact_residual_structure_verdict(fit_session):
    """The verdict is computed on every commit anyway (the plateau-adequacy gate needs it), so it
    comes back ON the fit result — the agent must not spend a get_fit_diagnosis call to learn
    whether its model's FORM is still incomplete."""
    from petab_mcp import data_access

    sess = fit_session
    res = sess.fit(n_starts=2)

    rs = res["residual_structure"]
    # The toy problem measures each design point ONCE, so the replicate arm has nothing to test and
    # is left off the fit result (#190)...
    assert set(rs) == {"flagged_observables", "low_power", "summary"}
    assert isinstance(rs["flagged_observables"], list)
    assert isinstance(rs["low_power"], bool)
    assert rs["summary"]
    # ...while the full rollup says so rather than contributing a silent pass.
    lof = data_access.residual_structure_rollup(sess)["replicate_lack_of_fit"]
    assert lof["available"] is False and lof["status"] == "no_replicated_points"
    # ...and the diagnosis tool does NOT repeat it: one verdict, one place. The detail behind it
    # (the per-observable rows) is what that call is for.
    assert "residual_structure" not in data_access.fit_diagnosis(sess)


def test_fit_result_omits_the_detailed_diagnosis(fit_session):
    """COMPACT means compact: the per-observable table and worst_points stay behind
    get_fit_diagnosis, so they don't ride on every fit and pile up in the agent's re-sent
    history (an input-token cost paid again on every later turn)."""
    res = fit_session.fit(n_starts=2)
    assert "per_observable" not in res and "worst_points" not in res
    assert "per_observable" not in res["residual_structure"]


# -- the merge hint rides on the fit result too -------------------------------
# Same reasoning as the residual verdict above, for the model's PARAMETERISATION. Measured: across
# three sessions on one problem the identifiability tool was called 3, 0 and 1 times out of 190,
# 149 and 94 tool calls, while the residual verdict — which rides on every fit — was quoted back in
# the agents' own justifications. The session that came CLOSEST (dBIC +11.6 against a <=10 bar)
# never called it and had a merge in its own FIM costing 0.001 nll against a 5.53 penalty.
#: The toy model with k1's STARTING value as a slot — a re-author that moves the SBML hash without
#: moving the structure fingerprint, i.e. the de-dup no-op that drifts.
_TOY_WITH_K1 = """model conversion_reaction
  compartment compartment_ = 1;
  species A in compartment_;
  species B in compartment_;
  A = A_init;
  B = B_init;
  A_init = 1.0;
  B_init = 0.0;
  k1 = {k1};
  k2 = 0.1;
  r1: A => B; k1 * A;
  r2: B => A; k2 * B;
end
"""


def _fim_with_tie(a_b_curvature: float):
    """A 2x2 FIM over the toy problem's (k1, k2): a near-singular difference direction means tying
    them is nearly free, a well-conditioned one means it is expensive."""
    import numpy as np

    return np.array([[1.0, 1.0 - a_b_curvature], [1.0 - a_b_curvature, 1.0]]) * 1e4


def _stub_fim(monkeypatch, curvature, *, counter=None):
    """Substitute the ONE part of the diagnosis that evaluates sensitivities, so the TIE is
    deterministic while everything downstream — the price, the filters, the assembly onto the fit
    result — is the shipped path. It takes the frozen fit RECORD, whose free names are the toy's
    (k1, k2), so the stub needs to return only the matrix."""
    from petab_mcp import identifiability as I

    def _fim(_record):
        if counter is not None:
            counter.append(1)
        return _fim_with_tie(curvature)

    monkeypatch.setattr(I, "_compute_fim", _fim)


def test_a_near_free_tie_reaches_the_fit_result_as_a_merge_hint_row(fit_session, monkeypatch):
    """The result carries a LIST — every tie the BIC charge covers, capped at
    identifiability.MERGE_HINT_LIMIT — so an agent is never shown one member of a redundant group as
    if it were the only merge available. This toy has one priceable pair, so the list has one row.
    """
    from petab_mcp import identifiability as I

    sess = fit_session
    _stub_fim(monkeypatch, 1e-9)
    res = sess.fit(n_starts=2)

    hints = res["merge_hints"]
    assert len(hints) == 1
    hint = hints[0]
    assert hint["parameters"] == ["k1", "k2"]
    assert hint["dscore"] < 0  # the merge lowers the score — why it is here
    assert hint["dnll"] is not None and hint["dnll"] < abs(hint["dscore"])
    # ...and it is the cheapest row of the diagnosis's own priced list, not a second statistic —
    # including `move_scaled`, the field the price has to be read against
    ties = I.identifiability(sess)["merge_candidates"]["ties"]
    fields = ("parameters", "dnll", "dscore", "move_scaled")
    assert {k: ties[0][k] for k in fields} == {k: hint[k] for k in fields}
    # ~110 B on every fit, and the rename that takes it on a text-route server (#283)
    assert set(hint) == {*fields, "take"}
    assert hint["take"] == {"op": "rename", "old": "k2", "new": "k1"}
    # ...and the SCORE read carries the same one, so a caller that did not run the fit (another
    # agent, a client's finish gate, a final scoring) sees the tie the agent was shown.
    from petab_mcp import fitting

    assert fitting.fit_summary(sess, sess.engine)["merge_hints"] == hints


def test_a_change_lists_the_ties_new_since_its_parent_and_counts_the_rest(fit_session, monkeypatch):
    """PEtab-MCP#208: a tie the parent's own fit result already carried is counted in
    `n_merge_hints`, not repeated, however it was re-priced; the score read keeps every one."""
    from petab_mcp import fitting
    from petab_mcp import operations as O

    sess = fit_session

    def widened_fit(upper):
        O.change_parameters(
            sess,
            [{"parameter_id": "k1", "lower_bound": 1e-3, "upper_bound": upper}],
            justification="widen k1",
        )
        return sess.fit(n_starts=2)

    _stub_fim(monkeypatch, 1.0)
    assert "merge_hints" not in sess.fit(n_starts=2)
    _stub_fim(monkeypatch, 1e-9)
    new = widened_fit(1e3)
    assert [h["parameters"] for h in new["merge_hints"]] == [["k1", "k2"]]
    assert "n_merge_hints" not in new
    again = widened_fit(1e4)
    assert again["merge_hints"] == [] and again["n_merge_hints"] == 1
    assert fitting.fit_summary(sess, sess.engine)["merge_hints"][0]["parameters"] == ["k1", "k2"]


def test_no_hints_when_no_tie_beats_the_parameter_charge(fit_session, monkeypatch):
    """ABSENT, not a null verdict — the same convention as noise_sigma_advice and
    nesting_advisory, so a fit with nothing to say costs nothing."""
    from petab_mcp import fitting, identifiability as I

    _stub_fim(monkeypatch, 1.0)
    res = fit_session.fit(n_starts=2)
    assert "merge_hints" not in res
    assert "merge_hints" not in fitting.fit_summary(fit_session, fit_session.engine)
    ties = I.identifiability(fit_session)["merge_candidates"]["ties"]
    assert ties and not any(r["merge_improves_score"] for r in ties)


def test_a_broken_diagnosis_never_costs_the_fit(fit_session, monkeypatch):
    """An advisory must not be able to fail a commit: the hint is dropped, the version stands."""
    from petab_mcp import identifiability as I

    def _boom(*_a, **_k):
        raise RuntimeError("FIM exploded")

    monkeypatch.setattr(I, "identifiability", _boom)
    res = fit_session.fit(n_starts=2)
    assert res["committed"] is True and "merge_hints" not in res
    assert res["residual_structure"]["summary"]


def test_the_no_op_refit_prices_no_tie(fit_session, monkeypatch):
    """A skipped fit priced nothing, so its RESULT claims no tie — the hint belongs to the call that
    ran the optimizer. The score read is a read of the STATE, and "no_change" means the state is
    byte-identical to the fit that priced the hint, so there it stands.

    Also pins that nothing leaks by aliasing: `self.simulation` IS the last fit's payload. That the
    no-op cannot BUILD anything is the contract in tests/test_diagnosis_never_compiles.py."""
    from petab_mcp import fitting

    _stub_fim(monkeypatch, 1e-9)
    sess = fit_session
    first = sess.fit(n_starts=2)
    assert "merge_hints" in first

    fims: list[int] = []
    _stub_fim(monkeypatch, 1e-9, counter=fims)
    noop = sess.fit(n_starts=2)
    assert noop["no_change"] is True
    assert "merge_hints" not in noop and fims == []
    assert fitting.fit_summary(sess, sess.engine)["merge_hints"] == first["merge_hints"]


def test_a_re_authored_no_op_keeps_the_fits_own_verdict(fit_session, monkeypatch):
    """REGRESSION: the no-op that MATTERS — the same structure re-authored with a different starting
    value. The structure fingerprint ignores nominal values, which is why the fit is SKIPPED; but the
    SBML text moves, and staleness used to be judged on the parameter table too, so this state read as
    "another model's diagnosis". That dropped the tie from the score read and switched the parsimony
    finish gate off for a rank deficiency the frozen FIM still reported — for the same structure, at
    HEAD's own fit, which is the only fit there is. The RESULT still claims nothing (this call priced
    nothing); the STATE still reports the fit it de-duplicated against."""
    from petab_mcp import fitting, identifiability as I
    from petab_mcp import operations as O

    _stub_fim(monkeypatch, 1e-9)
    sess = fit_session
    first = sess.fit(n_starts=2)
    assert "merge_hints" in first
    before = sess.sbml_hash()

    fims: list[int] = []
    _stub_fim(monkeypatch, 1e-9, counter=fims)
    O.set_model(sess, _TOY_WITH_K1.format(k1=0.7), justification="same structure, new guess")
    assert sess.sbml_hash() != before
    noop = sess.fit(n_starts=2)

    assert noop["no_change"] is True and "merge_hints" not in noop
    assert fitting.fit_summary(sess, sess.engine)["merge_hints"] == first["merge_hints"]
    assert fims == []  # ...and nothing was re-priced to say it
    # the diagnosis stands, and says the working TABLE has moved rather than claiming another model
    diagnosis = I.identifiability(sess)
    assert "stale" not in diagnosis
    assert diagnosis["working_table_note"] == I.WORKING_TABLE_NOTE


def test_a_fit_that_never_ran_prices_no_tie(fit_session, monkeypatch):
    """REGRESSION: with `auto_fit` off the values are whatever was authored, and the merge price is
    a local expansion at a stationary point — off one it gets monotonically CHEAPER, so it would
    advertise merges that are not there. That path is also documented as compile-free (57 s of
    AMICI build measured on the toy problem before this guard)."""
    fims: list[int] = []
    _stub_fim(monkeypatch, 1e-9, counter=fims)
    fit_session.auto_fit = False
    res = fit_session.fit()
    assert res["fit_status"] == "skipped"
    assert "merge_hints" not in res and fims == []


def test_the_fim_is_evaluated_on_the_fits_own_problem(fit_session, monkeypatch):
    """No pyPESTO problem is CONSTRUCTED for a diagnosis at all — it evaluates the one the optimizer
    minimised, held on the frozen fit record. Construction used to dominate the hint's cost (0.33 s
    of 0.70 s at k=30) and, when the working SBML had drifted, came with a full C++ build."""
    import pypesto.petab
    from petab.v2.C import UPPER_BOUND

    from petab_mcp import identifiability as I

    sess = fit_session
    sess.fit(n_starts=2)  # warms the module, leaves an optimum, and prices the hint
    record = sess.fit_record

    builds: list[int] = []
    original = pypesto.petab.PetabImporter.create_problem

    def _counted(self, *a, **kw):
        builds.append(1)
        return original(self, *a, **kw)

    monkeypatch.setattr(pypesto.petab.PetabImporter, "create_problem", _counted)
    first = I.identifiability(sess)
    second = I.identifiability(sess)
    assert first["fim"] == second["fim"] and first["fim"]["n_estimated"] == 2
    assert builds == []
    # ...and a parameter-table write does NOT send it looking for a new problem: it is the FIT's
    # geometry that is being reported, and the record still holds the problem that produced it.
    sess.problem.parameter_df.loc["k1", UPPER_BOUND] = 1e4
    third = I.identifiability(sess)
    assert builds == [] and third["fim"] == first["fim"]
    assert third["working_table_note"]  # ...and it says the working table has moved
    assert sess.fit_record is record


def test_a_restore_reports_the_restored_versions_own_tie(fit_session, monkeypatch):
    """A restore did not fit, so its RESULT claims no tie — the hint belongs to the call that ran the
    optimizer. The score read afterwards is a read of the STATE, and the state is now that version's
    fit, so it reports THAT fit's tie: the verdict travels with the version (see
    identifiability.fit_diagnosis_block), which is also what lets a resumed process report it."""
    from petab_mcp import fitting

    _stub_fim(monkeypatch, 1e-9)
    sess = fit_session
    first = sess.fit(n_starts=2)
    assert "merge_hints" in first
    assert "merge_hints" not in sess.restore_version(0)
    assert fitting.fit_summary(sess, sess.engine)["merge_hints"] == first["merge_hints"]


def test_a_restore_diagnoses_the_version_it_restored_not_the_last_fit(fit_session, monkeypatch):
    """Two sources of one verdict must not be able to name two fits. After a restore, the in-memory
    record still holds the LAST fit — another structure — while the restored payload carries the fit
    HEAD now IS; the score read serves that one, so the section has to as well, or the agent is
    shown a tie for a model it is not looking at."""
    from petab_mcp import fitting, identifiability as I
    from petab_mcp import operations as O

    _stub_fim(monkeypatch, 1e-9)
    sess = fit_session
    first = sess.fit(n_starts=2)
    O.set_model(
        sess,
        _TOY_WITH_K1.format(k1=0.5).replace("r2: B => A; k2 * B;", "r2: B => A; k2 * B * B;"),
        justification="a second structure",
    )
    sess.fit(n_starts=2)
    assert sess.fit_record is not None

    sess.restore_version(first["version"])
    diagnosis = I.identifiability(sess)
    assert diagnosis["available"] is True and "stale" not in diagnosis
    assert (
        I.merge_hints(diagnosis, rename=True)
        == fitting.fit_summary(sess, sess.engine)["merge_hints"]
    )


def test_fit_writes_the_structure_flag_into_the_version_meta(fit_session):
    """The version meta carries the bare BOOL, not the rollup: list_versions returns every
    version's meta verbatim, so a summary string per row would re-create the token growth this
    change removes."""
    sess = fit_session
    res = sess.fit(n_starts=2)

    meta = sess.provenance.list_versions()[-1]
    assert meta["residual_flagged"] == bool(res["residual_structure"]["flagged_observables"])
    assert "residual_structure" not in meta


def test_restore_version_carries_the_verdict_too(fit_session):
    """rollback/restore share fit_result_block with fit, so the shape is identical everywhere —
    an agent restoring a prior best sees that version's structure verdict without a follow-up
    call."""
    sess = fit_session
    sess.fit(n_starts=2)
    out = sess.restore_version(0)
    assert set(out["residual_structure"]) == {"flagged_observables", "low_power", "summary"}


def test_untested_residuals_are_recorded_as_unknown_not_as_clean(session, monkeypatch):
    """A version whose residual tests could not run must store `residual_flagged: None`, not
    False — recording "untested" as "nothing found" would let a later reader (or a scorer) treat
    an undiagnosed model as structurally sound. Uses the engine-free session, whose `fit` snapshots
    a version without a real simulation, so the tests genuinely have nothing to run on."""
    from petab_mcp import data_access

    session.fit()
    meta = session.provenance.list_versions()[-1]
    assert meta["residual_flagged"] is None
    # ...and that is exactly the "unavailable" verdict, not an unflagged one
    assert data_access.residual_flag(data_access.residual_structure_rollup(session)) is None
    assert "unavailable" in data_access.residual_structure_rollup(session)["summary"]


def test_a_real_fit_populates_the_noise_sigma_verdict(fit_session):
    """Regression: the noise-sigma report must reach the fit payload from the REAL fit path.

    `fit_and_simulate`'s payload never carried a `noise_sigma_report` key, so `PetabSession.fit`
    read None for it on every fit and pinned noise_sigma_estimated/inflated/collapsed to False in
    the event, the best-BIC checkpoint and the plateau-adequacy gate — the implausible-sigma
    checks were inert in production. The unit tests inject the key into a synthetic fitsim and
    call record_fit_progress directly, so only an end-to-end fit catches it. Needs an ESTIMATED
    sigma: with the toy's fixed sigma_B the report is legitimately empty, which cannot tell a
    working path from a dead one.
    """
    from petab.v2.C import ESTIMATE

    sess = fit_session
    sess.problem.parameter_df.loc["sigma_B", ESTIMATE] = True
    res = sess.fit(n_starts=2)

    report = sess.simulation["noise_sigma_report"]
    assert "sigma_B" in report, "the fit must report the estimated sigma"
    # the verdict rides along, so the agent never needs get_fit_summary for it, and the report
    # with it only for the sigmas it flags (#190)
    assert isinstance(res["noise_sigma_inflated"], bool)
    assert isinstance(res["noise_sigma_collapsed"], bool)
    flagged = {pid for pid, e in report.items() if e["status"] in ("inflated", "collapsed")}
    assert set(res.get("noise_sigma_report", {})) == flagged

    # ...and the report reached the session's own bookkeeping, not a None that silently reads as
    # "plausible": the event and the checkpoint must agree with the returned block.
    ev = [e for e in sess.provenance.recent_events() if e["type"] == "fit"][-1]
    assert ev["noise_sigma_estimated"] is True
    assert ev["noise_sigma_inflated"] == res["noise_sigma_inflated"]
    assert ev["noise_sigma_collapsed"] == res["noise_sigma_collapsed"]


def test_get_fit_summary_carries_the_residual_structure_verdict(fit_session):
    """Regression: the finish gate reads `residual_structure` off get_fit_summary, whose key
    whitelist omitted it, so its replicate lack-of-fit refusal could never fire."""
    from petab_mcp import fitting

    sess = fit_session
    fit_res = sess.fit(n_starts=2)
    summary = fitting.fit_summary(sess, sess.engine)

    assert "residual_structure" in summary, (
        "get_fit_summary must carry the residual-structure verdict — a finish gate reads it here"
    )
    rs = summary["residual_structure"]
    assert isinstance(rs, dict) and "summary" in rs
    # The fit result's is a cut of it (#190), so one key name never means two things.
    assert set(fit_res["residual_structure"]) <= set(rs)
    # The replicate arm must be present and self-describing, since that is the one the gate keys
    # on: it reads `.get("available")` and `.get("n_flagged")`.
    assert "replicate_lack_of_fit" in rs
    assert "available" in rs["replicate_lack_of_fit"]


def test_the_noop_carries_an_implausible_noise_verdict(fit_session):
    """The case the omission would have hidden: when the duplicate version's noise model IS
    implausible, the de-duplicated fit must say so — including the advice string, which is the
    only key that distinguishes "implausible" from "plausible" by its presence."""
    from petab.v2.C import ESTIMATE, LOWER_BOUND, UPPER_BOUND

    sess = fit_session
    sess.problem.parameter_df.loc["sigma_B", ESTIMATE] = True
    # FORCE the implausible verdict via the box rather than hoping the fit lands outside the band:
    # once the plausibility gate became dimensionally correct (#118) this fixture's sigma fitted to
    # a perfectly reasonable value and the test went vacuous. Clamping sigma_B many orders below
    # the data's own spread makes "collapsed" deterministic, which is what the test is about.
    sess.problem.parameter_df.loc["sigma_B", LOWER_BOUND] = 1e-9
    sess.problem.parameter_df.loc["sigma_B", UPPER_BOUND] = 1e-8
    first = sess.fit(n_starts=2)
    assert first["noise_sigma_collapsed"] or first["noise_sigma_inflated"], (
        "fixture no longer produces an implausible sigma — this test would be vacuous"
    )

    again = sess.fit(n_starts=2)
    assert again["no_change"] is True
    assert again["noise_sigma_collapsed"] == first["noise_sigma_collapsed"]
    assert again["noise_sigma_inflated"] == first["noise_sigma_inflated"]
    assert again["noise_sigma_advice"] == first["noise_sigma_advice"]


def test_a_restored_fit_reports_parameter_scales(fit_session):
    """Regression: a restored payload carries no scale map, and fit_result_block read it straight
    off the payload — returning linear values beside `parameter_scales: None`."""
    sess = fit_session
    sess.fit(n_starts=2)  # v0: a real fit, payload persisted
    O.change_parameters(
        sess,
        [{"parameter_id": "k1", "lower_bound": 1e-4, "upper_bound": 1e4}],
        justification="widen so the next fit is not a de-dup no-op",
    )
    sess.fit(n_starts=2)  # v1, so v0 is a genuine restore target
    restored = sess.restore_version(0)

    assert restored["fitted_parameters"], "v0's stored fit must come back with values"
    scales = restored["parameter_scales"]
    assert scales, "a restored fit must carry the scale map, not None beside linear values"
    assert set(scales) == set(restored["fitted_parameters"])
    assert all(v in ("lin", "log", "log10") for v in scales.values())


def test_a_version_keeps_its_fitted_values_in_its_parameter_table_only(fit_session):
    """No other text file in the store names them, and a restore reads them back from the table.
    The pyPESTO result stays beside them."""
    import json

    sess = fit_session
    fitted = sess.fit(n_starts=2)["fitted_parameters"]
    root, vdir = sess.provenance.root, sess.provenance.versions_dir / "v0"

    for name in ("meta.json", "simulations.json"):
        assert "fitted_parameters" not in json.loads((vdir / name).read_text())
    assert all("fitted_parameters" not in event for event in sess.provenance.recent_events(0))
    assert [p.name for p in root.rglob("*.tsv")] == ["parameters.tsv"]
    assert (vdir / "pypesto_result.hdf5").is_file()

    sess.simulation = None
    sess.apply_version(0)
    assert sess.simulation["fitted_parameters"] == pytest.approx(fitted)


def test_a_log_normal_observable_keeps_the_fit_joint(session):
    """pyPESTO refuses a log-normal observable that HAS inner parameters, but scores one without
    them as normal on the linear scale, so it beside an inner-solvable sigma would come out wrong:
    such a problem has no inner parameters, so it is fitted jointly and its sigmas keep their
    bounds."""
    import pandas as pd

    from petab_mcp import fitting

    p = session.problem
    observables = pd.concat(
        [
            p.observable_df.assign(
                noiseFormula="noiseParameter1_obs_B", noisePlaceholders="noiseParameter1_obs_B"
            ),
            pd.DataFrame(
                {"observableFormula": ["A"], "noiseFormula": ["0.2"]},
                index=pd.Index(["obs_A"], name="observableId"),
            ).assign(
                noiseDistribution="log-normal", observablePlaceholders="", noisePlaceholders=""
            ),
        ]
    )
    b = p.measurement_df.assign(noiseParameters="sigma_B")
    p = session.problem = p.replace(
        observable_df=observables,
        measurement_df=pd.concat([b, b.assign(observableId="obs_A", noiseParameters="")]),
    )
    p.parameter_df.loc["sigma_B", "estimate"] = True
    assert fitting._reconcile.classify_inner_parameters(p) == {}
    assert fitting._build_hierarchical_problem(session, {}) == (None, {})


def _scale_offset_problem(session, formula):
    """The toy with fontaine_sag's observation model: ``obs_B`` reads ``formula`` over two
    placeholders, which each measurement fills with the estimated ``scale_B`` and ``offset_B``."""
    p = session.problem
    p = session.problem = p.replace(
        observable_df=p.observable_df.assign(
            observableFormula=formula,
            observablePlaceholders="observableScale_obs_B;observableOffset_obs_B",
        ),
        measurement_df=p.measurement_df.assign(observableParameters="scale_B;offset_B"),
    )
    for pid, value in (("scale_B", 1.0), ("offset_B", 0.0)):
        p.parameter_df.loc[pid] = p.parameter_df.loc["k1"]
        p.parameter_df.loc[pid, ["parameterScale", "lowerBound", "upperBound"]] = ["lin", -10, 10]
        p.parameter_df.loc[pid, ["nominalValue", "estimate"]] = [value, True]
    return p


def test_a_scale_and_an_offset_are_solved_as_what_they_are(session):
    """`scale * readout + offset`, both filled per measurement by overrides, as fontaine_sag's
    flow-cytometry means are: the scale is a scaling and the offset an offset. Both used to be
    called scalings, pyPESTO refused two of one type, and every fit fell back to a joint one."""
    from petab_mcp import fitting

    p = _scale_offset_problem(session, "observableScale_obs_B * B + observableOffset_obs_B")
    assert fitting._reconcile.classify_inner_parameters(p) == {
        "scale_B": "scaling",
        "offset_B": "offset",
    }


@pytest.mark.parametrize(
    ("formula", "inner"),
    [
        # pyPESTO would fold the constant into the scaled term (ICB-DCM/pyPESTO#1785).
        ("observableScale_obs_B * B + 1", {}),
        ("observableScale_obs_B * B + observableOffset_obs_B + 1", {"offset_B": "offset"}),
        ("exp(observableScale_obs_B) * B + observableOffset_obs_B", {"offset_B": "offset"}),
        ("observableScale_obs_B * observableOffset_obs_B * B", {}),  # two scalings
        ("B + observableScale_obs_B + observableOffset_obs_B", {}),  # two offsets
        ("observableScale_obs_B * B + 2 * observableOffset_obs_B", {}),
        ("observableScale_obs_B * (B + 1)", {"scale_B": "scaling"}),
    ],
)
def test_an_observable_parameter_in_no_clean_role_stays_outer(session, formula, inner):
    """Only `scaling * expr + offset`, one of each, is what pyPESTO solves; anything else is
    fitted with the outer parameters."""
    from petab_mcp import fitting

    p = _scale_offset_problem(session, formula)
    assert fitting._reconcile.classify_inner_parameters(p) == inner


def test_pypesto_takes_a_scale_and_an_offset_hierarchically(fit_session):
    """The regression itself: pyPESTO builds the hierarchical problem, rather than refusing it
    and leaving the fit joint."""
    from petab_mcp import fitting

    _scale_offset_problem(fit_session, "observableScale_obs_B * B + observableOffset_obs_B")
    kwargs = fit_session.engine.pypesto_importer_kwargs(fit_session.module_hash())
    problem, inner = fitting._build_hierarchical_problem(fit_session, kwargs)
    assert problem is not None
    assert inner == {"scale_B": "scaling", "offset_B": "offset"}
    assert set(problem.objective.calculator.get_interpretable_inner_par_ids()) == set(inner)


def test_every_inner_value_must_come_back_from_the_fit():
    """Every inner parameter the server classified must come back from pyPESTO's result. A missing
    one was skipped silently, leaving its pre-fit value in the table beside a score taken at the
    inner optimum."""
    from types import SimpleNamespace

    from petab_mcp import fitting

    calculator = SimpleNamespace(get_interpretable_inner_par_ids=lambda: ["sigma_B"])
    problem = SimpleNamespace(objective=SimpleNamespace(calculator=calculator))
    inner = {"sigma_B": "sigma"}
    best = {"inner_parameters": [0.2]}
    assert fitting._inner_fitted_values(problem, best, inner) == {"sigma_B": 0.2}
    for lost in ({}, {"inner_parameters": None}, {"inner_parameters": [0.2, 1.0]}):
        with pytest.raises(ModelUpdateError, match="sigma_B"):
            fitting._inner_fitted_values(problem, lost, inner)
    with pytest.raises(ModelUpdateError, match="s_B"):
        fitting._inner_fitted_values(problem, best, {**inner, "s_B": "scaling"})


_DOSE_DRIVEN = """model conversion_reaction
  compartment compartment_ = 1;
  species A in compartment_, B in compartment_;
  A = A_init;
  B = B_init;
  A_init = 1.0;
  B_init = 0.0;
  dose = 0;
  K = 1;
  k1 = 0.5;
  k2 = 0.1;
  drive := dose/(dose + K*B)
  r1: A => B; k1 * drive * A;
  r2: B => A; k2 * B;
end
"""


def test_a_rate_that_is_0_over_0_at_pre_equilibration_is_named(dosed_session, amici_build_root):
    """lun 13 (sweep 63372750): `EGFdrive := EGF/(EGF + kpE*P2dose)` was 0/0 at
    pre-equilibration, where both inputs are 0. Every start failed, the refusal named two other
    causes, and the agent took three failed fits and a bisection to find it (PEtab-MCP#266)."""
    from petab_mcp.engine import AmiciEngine

    O.set_model(dosed_session, _DOSE_DRIVEN, justification="B made at a dose-driven rate")
    dosed_session.engine = AmiciEngine(dosed_session, build_root=str(amici_build_root))
    dosed_session.auto_fit_starts = 2
    with pytest.raises(ModelUpdateError) as refused:
        dosed_session.fit()
    assert (
        "because `drive := dose / (dose + K * B)` is NaN at t=0 in the pre-equilibration of "
        "experiment `e0`, which stops the simulation"
    ) in str(refused.value)


def test_a_fit_most_of_whose_starts_end_not_finite_says_so(fit_session, monkeypatch):
    """#285: bachmann 78 lost 45-98% of its starts NOT_FINITE in every fit, and its structures were
    judged on collapsed fits (v6 98/100, v9 95/100) with nothing in the result to say so."""
    from petab_mcp import fitting

    payload = fitting._optimization_payload

    def collapsed(result):
        out = payload(result)
        out["exitflag_counts"] = {"ExitFlag.NOT_FINITE": 2, "ExitFlag.FTOL": 1}
        return out

    monkeypatch.setattr(fitting, "_optimization_payload", collapsed)
    out = fit_session.fit(n_starts=3)
    assert any("2 of 3 starts ended NOT_FINITE" in w for w in out["warnings"])


_SOURCED = """model conversion_reaction
  compartment compartment_ = 1;
  species A in compartment_, B in compartment_;
  A = 1; B = 0;
  dose = 0; k1 = 0.5; k2 = 0.1; kb = 0.01;
  r1: A => B; k1 * (1 + dose) * A;
  r2: B => A; k2 * B;
  r3: => B; kb * A;
{sink}end
"""


@pytest.mark.parametrize("sink", ["", "  r4: B => ; kb * B;\n"])
def test_an_unscored_fit_under_a_source_that_breaks_a_total_names_it(dosed_session, sink):
    """#285: brannmark 42 wrote `IRS1_bas: -> IRS1p; k_basal*IRS1` under pre-equilibration, so the
    IRS1 total grew without a steady state, and its unscored fit was called a server fault. The
    lead blamed the numerics and dropped the basal flux that runs 41 and 43 kept (written
    `I -> Ip`). A source whose product has a sink leaves a steady state, and the note as it was."""
    from petab_mcp import fitting

    O.set_model(dosed_session, _SOURCED.format(sink=sink), justification="basal flux")
    sim = {"neg_log_likelihood": 12.0, "n_missing_simulations": 0, "n_expected_points": 6}
    nllh, note = fitting._reconcile_objective(dosed_session, {"neg_log_likelihood": 10.0}, sim)
    assert nllh is None
    if sink:
        assert "fault in how the server evaluates" in note
    else:
        assert "pre-equilibration has no steady state to reach: `r3` (`-> B; kb * A`)" in note
        assert "server" not in note


def test_a_fit_with_no_steady_state_to_pre_equilibrate_to_names_the_source(
    dosed_session, amici_build_root
):
    """#285: the same source fails every start where no pre-equilibration converges, which the
    refusal read as an ODE that will not integrate or an undefined readout."""
    from petab_mcp.engine import AmiciEngine

    O.set_model(dosed_session, _SOURCED.format(sink=""), justification="basal flux")
    dosed_session.engine = AmiciEngine(dosed_session, build_root=str(amici_build_root))
    dosed_session.auto_fit_starts = 2
    with pytest.raises(ModelUpdateError, match="no steady state to reach: `r3` "):
        dosed_session.fit()


def test_a_power_of_a_state_to_an_estimated_exponent_is_named(dosed_session):
    """The powers that go NaN when their state dips below 0 (PEtab-MCP#323), or lose their slope
    where a condition sets their input to 0 (lun 109), named for a fit any of whose starts ended
    NOT_FINITE; a number as the exponent is not one."""
    from petab_mcp import fitting

    model = _DOSE_DRIVEN.replace("k1 * drive * A", "k1 * A^n_A + k1 * B^2 + dose^n_A").replace(
        "k2 = 0.1;", "k2 = 0.1;\n  n_A = 2;"
    )
    O.set_model(dosed_session, model, justification="cooperative conversion")
    assert fitting._reconcile.is_parameter_estimated(dosed_session.problem, "n_A")
    named = fitting.state_powers(dosed_session)
    assert "`A^n_A` raises `A` to the estimated `n_A`" in named
    assert "`dose^n_A` raises `dose` to the estimated `n_A`" in named
    assert "B^2" not in named and ". Below 0 a power" in named
    assert fitting.state_powers(dosed_session, {"B": -0.1}) == ""


def test_a_fit_some_of_whose_starts_end_not_finite_says_how_many(fit_session, monkeypatch):
    """#323: fujita lost 6-25% of its starts NOT_FINITE with nothing said; any share is reported,
    with the powers of states that make one (here a stand-in), and only a majority is said to
    leave the score in doubt."""
    from petab_mcp import fitting

    payload = fitting._optimization_payload

    def some(result):
        out = payload(result)
        out["exitflag_counts"] = {"ExitFlag.FTOL": 2, "ExitFlag.NOT_FINITE": 1}
        return out

    monkeypatch.setattr(fitting, "_optimization_payload", some)
    monkeypatch.setattr(fitting, "state_powers", lambda session: "`A^n` raises the state `A`.")
    out = fit_session.fit(n_starts=3)
    assert (
        "1 of 3 starts ended NOT_FINITE: the objective or its gradient was not a number where "
        "they went. `A^n` raises the state `A`."
    ) in out["warnings"]


def test_a_fit_the_pool_failed_does_not_count_toward_the_identical_fit_guard(
    fit_session, monkeypatch
):
    """A worker that died under the fit (dask KilledWorker) scored nothing, so sending the change
    again is not a loop (PEtab-MCP#341)."""
    from petab_mcp.errors import FitBackendError
    from petab_mcp.session import PetabSession

    def died(self, n_starts=None):
        raise FitBackendError("a worker of the fit pool died")

    monkeypatch.setattr(PetabSession, "_auto_fit_and_improvement", died)
    with pytest.raises(FitBackendError):
        fit_session.fit()
    assert not any(fit_session._fit_attempts.values())


def _log_noise(session):
    p = session.problem
    session.problem = p.replace(
        observable_df=p.observable_df.assign(noiseDistribution="log-normal")
    )


def _records(*points):
    return [
        {"observable_id": "obs_B", "time": t, "measurement": 0.5, "simulation": s}
        for t, s in points
    ]


def test_a_readout_at_0_or_below_under_log_noise_is_named(session):
    """#327: blasi 63612603_5's zero-order rate laws drove log-normal readouts below 0, which no
    check named; a readout of 0 at t=0 is the other origin."""
    from petab_mcp import fitting

    assert fitting._log_readouts_not_positive(session, _records((0.0, 0.0), (1.0, -0.2))) == ""
    _log_noise(session)
    text = fitting._log_readouts_not_positive(session, _records((0.0, 0.0), (1.0, 0.3)))
    assert text.startswith("`obs_B` (down to 0, at t=0): under log noise the likelihood")
    assert "A readout of 0 at t=0 reads a species that starts at 0." in text
    text = fitting._log_readouts_not_positive(session, _records((1.0, -0.2), (2.0, -0.5)))
    assert text.startswith("`obs_B` (down to -0.5, at t=1, t=2)")
    assert "One below 0 reads a species a rate consumes without reading it" in text


@pytest.mark.parametrize("shown", [None, "simulation", "readout"])
def test_an_all_failed_fit_names_the_cause_its_starts_show(session, shown):
    """#327: the refusal sent agents to a fitting log that never holds per-start failures. It
    names the cause its re-simulated starts show, else both; the t=0 example only where a
    log-noise observable is measured at t=0 (the toy's `obs_B` is, at 0.0)."""
    from petab_mcp import fitting

    for log in (False, True):
        if log:
            _log_noise(session)
        text = fitting._every_start_failed(session, "WHERE", "STARTS.", shown)
        assert "WHERE — a property of the problem, not the optimizer. STARTS. " in text
        assert "fitting log" not in text
        assert ("(a) the SIMULATION" in text) == (shown is None)
        assert ("first of two causes: the SIMULATION" in text) == (shown == "simulation")
        assert ("second of two causes: the simulation succeeds" in text) == (shown == "readout")
        example = "measured at t=0 gives log(0) = -inf"
        assert (example in text) == (log and shown != "simulation")


class _RData:
    """A ReturnData stand-in: an experiment's status and AMICI's messages for it."""

    def __init__(self, eid, status=0, *messages):
        from types import SimpleNamespace

        self.id, self.status = eid, status
        self.messages = [SimpleNamespace(message=m) for m in messages]


def _all_failed(statuses, records=()):
    """An all-failed multistart of three starts, two of them from the same point, whose objective
    returns one ReturnData per ``statuses``, and an engine reading ``records`` off them."""
    from types import SimpleNamespace

    from pypesto.C import RDATAS

    calls = []

    def objective(x, sensi_orders, return_dict):
        calls.append(list(x))
        return {RDATAS: [_RData(f"e{i}", s, f"failed at t = {i}") for i, s in enumerate(statuses)]}

    starts = [
        {"exitflag": "ExitFlag.NOT_FINITE", "message": "non-finite at initial point.", "x0": x0}
        for x0 in ([0.0, 1.0, 2.0], [0.0, 1.0, 2.0], [0.5, 1.0, 3.0])
    ] + [{"exitflag": -1, "message": "stopped by the server", "x0": None}]
    result = SimpleNamespace(optimize_result=SimpleNamespace(list=starts))
    problem = SimpleNamespace(x_free_indices=[0, 2], dim_full=3, objective=objective)
    engine = SimpleNamespace(_simulation_records=lambda objective, rdatas: list(records))
    return result, problem, engine, calls


def test_amici_failures_names_each_failed_experiment_with_its_status():
    """AMICI logs a failure to stderr, which on a fit pool reached no log (PEtab-MCP#327)."""
    from amici.sim.sundials import AMICI_ERROR, AMICI_SUCCESS

    from petab_mcp import fitting

    rdatas = [_RData("e0", AMICI_SUCCESS), _RData("e1", AMICI_ERROR, "first", "last")]
    assert fitting.amici_failures(rdatas) == ["`e1` ends AMICI_ERROR (last)"]


def test_the_failed_starts_are_re_simulated_where_they_began(session):
    """#327: the refusal carries each start's exit flag and message, and AMICI's status at the
    distinct points the failed starts began from, reduced to the free parameters."""
    from amici.sim.sundials import AMICI_ERROR, AMICI_SUCCESS

    from petab_mcp import fitting

    result, problem, engine, calls = _all_failed([AMICI_SUCCESS, AMICI_ERROR])
    text, shown = fitting._failed_starts(session, engine, problem, result)
    assert calls == [[0.0, 2.0], [0.5, 3.0]]
    assert shown == "simulation"
    assert text == (
        "Its starts: 3 ended NOT_FINITE (non-finite at initial point.); 1 ended raised (stopped "
        "by the server). Re-simulated at 2 of their start points, AMICI fails: `e1` ends "
        "AMICI_ERROR (failed at t = 1)."
    )


def test_a_failed_start_whose_experiments_simulate_names_the_readout(session):
    """Where every experiment simulates at a failed start's point, the readout is what failed."""
    from amici.sim.sundials import AMICI_SUCCESS

    from petab_mcp import fitting

    _log_noise(session)
    result, problem, engine, _ = _all_failed([AMICI_SUCCESS], _records((0.0, 0.0)))
    text, shown = fitting._failed_starts(session, engine, problem, result)
    assert shown == "readout"
    assert (
        "Re-simulated at 2 of their start points, every experiment simulates, but readouts with "
        "log noise are 0 or below where measured: `obs_B` (down to 0, at t=0)"
    ) in text
    result, problem, engine, _ = _all_failed([AMICI_SUCCESS], _records((0.0, 0.3)))
    assert fitting._failed_starts(session, engine, problem, result)[1] is None


def test_every_failed_start_is_re_simulated_and_its_amici_status_named(
    dosed_session, amici_build_root, monkeypatch
):
    """#327: every start failed, and the refusal sent the agent to a fitting log that held nothing
    of them. Without the source check (see the test above), the refusal reaches its general form,
    which now carries AMICI's status where the starts began and names the one cause it shows."""
    from petab_mcp import fitting
    from petab_mcp.engine import AmiciEngine

    O.set_model(dosed_session, _SOURCED.format(sink=""), justification="basal flux")
    dosed_session.engine = AmiciEngine(dosed_session, build_root=str(amici_build_root))
    dosed_session.auto_fit_starts = 2
    monkeypatch.setattr(fitting, "_sources_breaking_totals", lambda problem: [])
    with pytest.raises(ModelUpdateError) as refused:
        dosed_session.fit()
    text = str(refused.value)
    assert "Its starts: 2 ended " in text
    assert "Re-simulated at 2 of their start points, AMICI fails: `e0` ends AMICI_" in text
    assert "That is the first of two causes: the SIMULATION fails" in text
    assert "fitting log" not in text


def test_a_readout_at_0_under_log_noise_at_the_nominal_values_is_named(fit_session):
    """#327: the toy's `B` starts at 0 and is measured at t=0, which under log noise has no
    likelihood at any start."""
    _log_noise(fit_session)
    p = fit_session.problem
    fit_session.problem = p.replace(
        measurement_df=p.measurement_df.assign(measurement=p.measurement_df["measurement"] + 0.1)
    )
    with pytest.raises(ModelUpdateError) as refused:
        fit_session.fit()
    text = str(refused.value)
    assert "`obs_B` (down to 0, at t=0)" in text, text
    assert "A readout of 0 at t=0 reads a species that starts at 0." in text
    assert "Re-simulated" not in text  # the nominal values' check, before any start's
