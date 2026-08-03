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
        session, [{"parameter_id": "k1", "lower_bound": 1e-3, "upper_bound": 1e3}],
        justification="widen")
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
        sess, [{"parameter_id": "k1", "lower_bound": 1e-3, "upper_bound": 1e3}],
        justification="widen")

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
    # It is the SAME rollup get_fit_diagnosis reports — that equality is the whole point: the
    # follow-up call is redundant, not merely similar.
    assert rs == data_access.fit_diagnosis(sess)["residual_structure"]


def test_fit_result_omits_the_detailed_diagnosis(fit_session):
    """COMPACT means compact: the per-observable table and worst_points stay behind
    get_fit_diagnosis, so they don't ride on every fit and pile up in the agent's re-sent
    history (the input-token cost the ORCHESTRATOR_TOOLS curation exists to avoid)."""
    res = fit_session.fit(n_starts=2)
    assert "per_observable" not in res and "worst_points" not in res
    assert "per_observable" not in res["residual_structure"]


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
    assert set(out["residual_structure"]) == {"flagged_observables", "low_power", "summary",
                                             "replicate_lack_of_fit"}


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
        "fixture no longer produces an implausible sigma — this test would be vacuous")

    again = sess.fit(n_starts=2)
    assert again["no_change"] is True
    assert again["noise_sigma_collapsed"] == first["noise_sigma_collapsed"]
    assert again["noise_sigma_inflated"] == first["noise_sigma_inflated"]
    assert again["noise_sigma_advice"] == first["noise_sigma_advice"]


def test_a_restored_fit_reports_parameter_scales(fit_session):
    """Regression: a restored payload carries no scale map, and fit_result_block read it straight
    off the payload — returning linear values beside `parameter_scales: None`."""
    sess = fit_session
    sess.fit(n_starts=2)                                   # v0: a real fit, payload persisted
    O.change_parameters(
        sess, [{"parameter_id": "k1", "lower_bound": 1e-4, "upper_bound": 1e4}],
        justification="widen so the next fit is not a de-dup no-op")
    sess.fit(n_starts=2)                                   # v1, so v0 is a genuine restore target
    restored = sess.restore_version(0)

    assert restored["fitted_parameters"], "v0's stored fit must come back with values"
    scales = restored["parameter_scales"]
    assert scales, "a restored fit must carry the scale map, not None beside linear values"
    assert set(scales) == set(restored["fitted_parameters"])
    assert all(v in ("lin", "log", "log10") for v in scales.values())
