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
    from petab.v1.C import ESTIMATE

    session.engine = object()  # non-None so the fit path is entered (guard raises before use)
    session.auto_fit = True
    session.problem.parameter_df[ESTIMATE] = 0  # nothing left to estimate
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
    # "plausible" — and the lead has no get_fit_summary to check against.
    for key in ("noise_sigma_report", "noise_sigma_inflated", "noise_sigma_collapsed"):
        assert key in again, key
        assert again[key] == first[key]
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
    assert set(rs) == {"flagged_observables", "low_power", "summary", "replicate_lack_of_fit"}
    assert isinstance(rs["flagged_observables"], list)
    assert isinstance(rs["low_power"], bool)
    assert rs["summary"]
    # The toy problem measures each design point ONCE, so the replicate arm has nothing to test —
    # and must say so rather than contributing a silent pass.
    assert rs["replicate_lack_of_fit"]["available"] is False
    assert rs["replicate_lack_of_fit"]["status"] == "no_replicated_points"
    # ...and the diagnosis tool does NOT repeat it: one verdict, one place. The detail behind it
    # (the per-observable rows) is what that call is for.
    assert "residual_structure" not in data_access.fit_diagnosis(sess)


def test_fit_result_omits_the_detailed_diagnosis(fit_session):
    """COMPACT means compact: the per-observable table and worst_points stay behind
    get_fit_diagnosis, so they don't ride on every fit and pile up in the agent's re-sent
    history (the input-token cost the ORCHESTRATOR_TOOLS curation exists to avoid)."""
    res = fit_session.fit(n_starts=2)
    assert "per_observable" not in res and "worst_points" not in res
    assert "per_observable" not in res["residual_structure"]


# -- the merge hint rides on the fit result too -------------------------------
# Same reasoning as the residual verdict above, for the model's PARAMETERISATION. Measured: across
# three blasi runs on 3491be9 the identifiability tool was called 3, 0 and 1 times out of 190, 149
# and 94 tool calls, while the residual verdict — which rides on every fit — was quoted back in the
# panel's own justifications. The run that came CLOSEST (dBIC +11.6 against a <=10 bar) never called
# it and had a merge in its own FIM costing 0.001 nll against a 5.53 penalty.
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
    identifiability.MERGE_HINT_LIMIT — so a lead is never shown one member of a redundant group as
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
    assert {k: ties[0][k] for k in ("parameters", "dnll", "dscore", "move_scaled")} == hint
    assert set(hint) == {"parameters", "dnll", "dscore", "move_scaled"}  # ~110 B on every fit
    # ...and the SCORE read carries the same one, so a caller that did not run the fit (the
    # analysts, the harness finish gate, the final scoring) sees the tie the lead was shown.
    from petab_mcp import fitting

    assert fitting.fit_summary(sess, sess.engine)["merge_hints"] == hints


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
    of 0.70 s on schwen, k=30) and, when the working SBML had drifted, came with a full C++ build."""
    import pypesto.petab
    from petab.v1.C import UPPER_BOUND

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
    HEAD now IS; the score read serves that one, so the section has to as well, or the lead is shown a
    tie for a model it is not looking at."""
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
    assert I.merge_hints(diagnosis) == fitting.fit_summary(sess, sess.engine)["merge_hints"]


def test_fit_writes_the_structure_flag_into_the_version_meta(fit_session):
    """The version meta carries the bare BOOL, not the rollup: list_versions returns every
    version's meta verbatim, so a summary string per row would re-create the token growth this
    change removes."""
    sess = fit_session
    res = sess.fit(n_starts=2)

    meta = sess.provenance.list_versions()[-1]
    assert meta["residual_flagged"] == bool(res["residual_structure"]["flagged_observables"])
    assert "residual_structure" not in meta
    # the gate's own bookkeeping still reads the same bool off the fit event
    fit_ev = [e for e in sess.provenance.recent_events() if e["type"] == "fit"][-1]
    assert fit_ev["residual_flagged"] == meta["residual_flagged"]


def test_restore_version_carries_the_verdict_too(fit_session):
    """rollback/restore share fit_result_block with fit, so the shape is identical everywhere —
    the lead restoring a prior best sees that version's structure verdict without a follow-up
    call."""
    sess = fit_session
    sess.fit(n_starts=2)
    out = sess.restore_version(0)
    assert set(out["residual_structure"]) == {
        "flagged_observables",
        "low_power",
        "summary",
        "replicate_lack_of_fit",
    }


def test_untested_residuals_are_recorded_as_unknown_not_as_clean(session, monkeypatch):
    """A version whose residual tests could not run must store `residual_flagged: None`, not
    False — recording "untested" as "nothing found" would let a later reader (or the judge) treat
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
    from petab.v1.C import ESTIMATE

    sess = fit_session
    sess.problem.parameter_df.loc["sigma_B", ESTIMATE] = 1
    res = sess.fit(n_starts=2)

    assert res["noise_sigma_report"], "the fit block must carry the estimated sigma's report"
    assert "sigma_B" in res["noise_sigma_report"]
    # the verdict rides along, so the agent never needs get_fit_summary for it
    assert isinstance(res["noise_sigma_inflated"], bool)
    assert isinstance(res["noise_sigma_collapsed"], bool)

    # ...and the SAME report reached the session's own bookkeeping, not a None that silently
    # reads as "plausible": the event and the checkpoint must agree with the returned block.
    assert sess.simulation["noise_sigma_report"] == res["noise_sigma_report"]
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
        "get_fit_summary must carry the residual-structure verdict — the finish gate reads it here"
    )
    rs = summary["residual_structure"]
    assert isinstance(rs, dict) and "summary" in rs
    # Same shape the fit result reports, so one key name never means two things.
    assert set(rs) == set(fit_res["residual_structure"])
    # The replicate arm must be present and self-describing, since that is the one the gate keys
    # on: it reads `.get("available")` and `.get("n_flagged")`.
    assert "replicate_lack_of_fit" in rs
    assert "available" in rs["replicate_lack_of_fit"]


def test_the_noop_carries_an_implausible_noise_verdict(fit_session):
    """The case the omission would have hidden: when the duplicate version's noise model IS
    implausible, the de-duplicated fit must say so — including the advice string, which is the
    only key that distinguishes "implausible" from "plausible" by its presence."""
    from petab.v1.C import ESTIMATE, LOWER_BOUND, UPPER_BOUND

    sess = fit_session
    sess.problem.parameter_df.loc["sigma_B", ESTIMATE] = 1
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
