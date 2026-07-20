"""Unit tests for the early-stop chain (issue #58):

* the effect-size gate on ``is_new_best`` (:func:`fitting.compute_improvement`), and
* the non-improving-streak cap + plateau + best-BIC checkpoint tracked on the session
  (:meth:`PetabSession.record_fit_progress` / :meth:`PetabSession.progress_report`).

All of this is pure bookkeeping — it reads a fit payload and the provenance log — so it is
tested here with synthetic fit results, no AMICI/pyPESTO fit required.
"""

from __future__ import annotations

import pytest

from petab_mcp import fitting
from petab_mcp.constants import (
    NEW_BEST_MIN_DELTA_BIC,
    NON_IMPROVING_STREAK_CAP,
    PLATEAU_WINDOW,
)


def _fitsim(nll: float, chi2: float, *, n: int = 20, k: int = 2) -> dict:
    """A minimal ``fit_and_simulate``-shaped payload for a genuine fit."""
    return {"status": "fit", "neg_log_likelihood": nll, "chi2": chi2,
            "n_data_points": n, "n_estimated": k}


# --- (a) effect-size gate on is_new_best (tracks BIC) -------------------------------------
def test_is_new_best_tracks_bic_effect_size(session):
    # No checkpoint yet -> the first fit is always a new best; it reports its own BIC.
    fs0 = _fitsim(-50.0, 18.0)
    first = fitting.compute_improvement(
        session, nllh=fs0["neg_log_likelihood"], objective_before=-40.0, fitsim=fs0)
    assert first["is_new_best"] is True
    assert first["best_prior_bic"] is None
    assert first["bic"] is not None

    # Seed the best-BIC checkpoint at that fit's BIC (as record_fit_progress would).
    session.best_bic, session.best_version, session.best_nll = first["bic"], 1, -50.0

    # BIC = k·ln(n) + 2·nll, so at fixed k,n a ΔBIC of X needs a ΔNLL of X/2. A sub-threshold BIC
    # improvement (< NEW_BEST_MIN_DELTA_BIC) is NOT a new best, even though nll strictly dropped.
    d_small = NEW_BEST_MIN_DELTA_BIC / 2
    fs_small = _fitsim(-50.0 - d_small / 2, 18.0)
    small = fitting.compute_improvement(
        session, nllh=fs_small["neg_log_likelihood"], objective_before=-50.0, fitsim=fs_small)
    assert small["improvement_vs_best_prior"] == pytest.approx(d_small)
    assert small["is_new_best"] is False

    # An improvement EXACTLY at the threshold counts (the gate is ">=", not ">").
    fs_edge = _fitsim(-50.0 - NEW_BEST_MIN_DELTA_BIC / 2, 18.0)  # ΔBIC == NEW_BEST_MIN_DELTA_BIC
    edge = fitting.compute_improvement(
        session, nllh=fs_edge["neg_log_likelihood"], objective_before=-50.0, fitsim=fs_edge)
    assert edge["improvement_vs_best_prior"] == pytest.approx(NEW_BEST_MIN_DELTA_BIC)
    assert edge["is_new_best"] is True

    # A BIC improvement comfortably past the threshold does count.
    d_big = NEW_BEST_MIN_DELTA_BIC + 2
    fs_big = _fitsim(-50.0 - d_big / 2, 18.0)
    big = fitting.compute_improvement(
        session, nllh=fs_big["neg_log_likelihood"], objective_before=-50.0, fitsim=fs_big)
    assert big["improvement_vs_best_prior"] == pytest.approx(d_big)
    assert big["is_new_best"] is True


def test_is_new_best_rejects_nll_gain_that_worsens_bic(session):
    """The whole point of gating on BIC not nll: a fit that lowers nll but adds parameters that
    RAISE BIC is not a new best."""
    session.best_bic = fitting.goodness_of_fit(-50.0, 18.0, 2, 20)["bic"]  # checkpoint at k=2
    session.best_version, session.best_nll = 1, -50.0
    # nll improves (-50 -> -51) but k jumps 2 -> 5: BIC = k·ln(20)+2·nll gets WORSE.
    fs = _fitsim(-51.0, 18.0, k=5)
    imp = fitting.compute_improvement(session, nllh=-51.0, objective_before=-50.0, fitsim=fs)
    assert imp["improvement_vs_best_prior"] < 0     # BIC got worse
    assert imp["is_new_best"] is False


# --- (b) streak cap + plateau + (c) best-BIC checkpoint ------------------------------------
def test_first_fitted_op_sets_checkpoint_and_resets_streak(session):
    session.record_fit_progress(1, _fitsim(-50.0, 18.0), {"is_new_best": True})
    p = session.progress_report()
    assert p["n_fitted_ops"] == 1
    assert p["non_improving_streak"] == 0
    assert p["best_version"] == 1
    assert p["best_bic"] is not None
    assert p["should_stop"] is False


def test_streak_cap_stops_a_never_adequate_run(session):
    """The load-bearing lever: a run whose best never reaches adequacy (χ² GoF rejected) still
    stops once the non-improving streak hits the cap — the plateau gate alone can't catch it."""
    # chi2 far above the dof -> statistically rejected, so the plateau branch never fires.
    session.record_fit_progress(1, _fitsim(-50.0, 500.0), {"is_new_best": True})
    for v in range(2, 2 + NON_IMPROVING_STREAK_CAP):
        session.record_fit_progress(v, _fitsim(-50.0, 500.0), {"is_new_best": False})

    p = session.progress_report()
    assert p["best_gof_pvalue"] is not None and p["best_gof_pvalue"] < 0.05  # never adequate
    assert p["non_improving_streak"] == NON_IMPROVING_STREAK_CAP
    assert p["best_version"] == 1  # the checkpoint stayed on the best model
    assert p["should_stop"] is True
    assert "streak" in p["stop_reason"].lower()


def test_plateau_stops_an_adequate_flat_run(session):
    """Once the best model is adequate and the best BIC stops improving over the window, stop —
    before the hard streak cap, and without requiring the streak to reach it."""
    # chi2 == dof -> reduced_chi2 ~ 1, an adequate GoF p-value; best BIC held flat across the ops.
    session.record_fit_progress(1, _fitsim(-50.0, 18.0), {"is_new_best": True})
    for v in range(2, 2 + PLATEAU_WINDOW):
        session.record_fit_progress(v, _fitsim(-50.0, 18.0), {"is_new_best": False})

    p = session.progress_report()
    assert p["n_fitted_ops"] == PLATEAU_WINDOW + 1
    assert p["non_improving_streak"] == PLATEAU_WINDOW < NON_IMPROVING_STREAK_CAP  # cap not hit
    assert p["best_gof_pvalue"] is not None and p["best_gof_pvalue"] >= 0.05
    assert p["bic_improvement_over_last_n"] == pytest.approx(0.0)
    assert p["should_stop"] is True
    assert "plateau" in p["stop_reason"].lower()


def test_steadily_improving_run_never_stops_early(session):
    """A run that keeps landing real new bests keeps the streak at zero and the BIC falling, so
    neither the cap nor the plateau ever fires."""
    for i, v in enumerate(range(1, 10)):
        # each op a genuine new best: nll keeps dropping, so BIC keeps falling well past the gate
        session.record_fit_progress(v, _fitsim(-50.0 - 5.0 * i, 18.0), {"is_new_best": True})

    p = session.progress_report()
    assert p["non_improving_streak"] == 0
    assert p["best_version"] == 9          # the last (lowest-BIC) op
    assert p["bic_improvement_over_last_n"] > 0
    assert p["should_stop"] is False


def test_checkpoint_holds_the_lowest_bic_after_wandering_worse(session):
    """After the best op, a worse op must NOT move the checkpoint (the wander-off-best case)."""
    session.record_fit_progress(1, _fitsim(-80.0, 18.0), {"is_new_best": True})   # the best
    session.record_fit_progress(2, _fitsim(-40.0, 18.0), {"is_new_best": False})  # worse
    p = session.progress_report()
    assert p["best_version"] == 1
    assert p["non_improving_streak"] == 1
