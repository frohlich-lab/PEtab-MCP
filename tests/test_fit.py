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
    assert hasattr(session, "fit") and hasattr(session, "pending_edits")


# -- edits are fit-free -------------------------------------------------------
def test_edit_marks_pending_without_versioning(session):
    assert not session.has_unfit_edits()
    res = O.change_parameters(
        session, [{"parameter_id": "k1", "lower_bound": 1e-3, "upper_bound": 1e3}],
        justification="widen")
    assert res["applied"] is True and res["version"] is None
    assert session.has_unfit_edits() and res["n_pending_edits"] == 1
    assert session.provenance.current_version == 0  # no version created by an edit


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


def test_fit_without_a_model_is_rejected(empty_session):
    with pytest.raises(ModelUpdateError, match="no model to fit"):
        empty_session.fit()


# -- fit: fit + snapshot + clear pending (needs the engine) -------------------
def test_fit_scores_edits_and_snapshots_a_version(fit_session):
    sess = fit_session
    O.change_parameters(
        sess, [{"parameter_id": "k1", "lower_bound": 1e-3, "upper_bound": 1e3}],
        justification="widen")
    assert sess.has_unfit_edits()

    res = sess.fit(n_starts=2)
    assert res["committed"] is True
    assert res["version"] == 1
    assert res["neg_log_likelihood"] is not None and res["bic"] is not None
    assert "progress" in res
    # the pending edit was consumed by the fit
    assert not sess.has_unfit_edits()
    assert sess.provenance.current_version == 1
    versions = sess.provenance.list_versions()
    assert versions[-1]["kind"] == "fit"
    # the fit event lists the edits it scored
    fit_ev = [e for e in sess.provenance.recent_events() if e["type"] == "fit"][-1]
    assert fit_ev["n_edits"] == 1


def test_fit_is_a_noop_when_model_unchanged(fit_session):
    """Fitting again with no edits since the last fit skips the refit and returns the stored
    metrics (the de-dup relocated from set_model to fit — issue #59/#81)."""
    sess = fit_session
    first = sess.fit(n_starts=2)
    assert first["committed"] is True and first["version"] == 1

    again = sess.fit(n_starts=2)
    assert again["committed"] is False
    assert again["no_change"] is True
    assert again["duplicate_of_version"] == 1
    assert again["neg_log_likelihood"] == pytest.approx(first["neg_log_likelihood"])
    # no churn: no new version was created by the no-op fit
    assert sess.provenance.current_version == 1


def test_fit_simulate_only_with_zero_starts(fit_session):
    """n_starts=0 simulates without optimising and still snapshots a version."""
    sess = fit_session
    res = sess.fit(n_starts=0)
    assert res["committed"] is True and res["version"] == 1
    assert res["fit_status"] == "simulate_only"
