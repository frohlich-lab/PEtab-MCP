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
    NOISE_SIGMA_COLLAPSE_FRACTION,
    NON_IMPROVING_STREAK_CAP,
    PLATEAU_WINDOW,
)


def _checkpoint(*, bic, version, nll, **over):
    """A best-BIC checkpoint seeded directly, as record_fit_progress would adopt one."""
    from petab_mcp.session import Checkpoint

    fields = dict(version=version, bic=bic, nll=nll, gof_pvalue=None, sigma_estimated=False,
                  sigma_inflated=False, sigma_collapsed=False, sigma_assessable=False,
                  residual_flagged=False)
    fields.update(over)
    return Checkpoint(**fields)



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
    session.best = _checkpoint(bic=first["bic"], version=1, nll=-50.0)

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
    session.best = _checkpoint(                       # checkpoint at k=2
        bic=fitting.goodness_of_fit(-50.0, 18.0, 2, 20)["bic"], version=1, nll=-50.0)
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


def _fitsim_inflated_sigma(nll: float, chi2: float, *, n: int = 20, k: int = 2) -> dict:
    """Like ``_fitsim`` but the fit reports an ESTIMATED noise sigma fitted ABOVE its data-driven
    cap — the inflated-sigma case where a passing chi2 GoF is not genuine adequacy."""
    fs = _fitsim(nll, chi2, n=n, k=k)
    fs["noise_sigma_report"] = {
        "sigma": {"fitted": 0.5, "upper_bound_cap": 0.1, "data_std": 0.2,
                  "fraction_of_data_std": 2.5}}
    return fs


def test_inflated_sigma_blocks_the_adequate_plateau_stop(session):
    """A flat run whose good chi2 GoF is only propped up by an inflated estimated sigma is NOT
    adequate for the plateau gate, so it must NOT stop early — the blasi/sneyd underfit case where
    a free sigma absorbs the misfit and fakes chi2/dof ~ 1."""
    session.record_fit_progress(1, _fitsim_inflated_sigma(-50.0, 18.0), {"is_new_best": True})
    for v in range(2, 2 + PLATEAU_WINDOW):
        session.record_fit_progress(v, _fitsim_inflated_sigma(-50.0, 18.0), {"is_new_best": False})
    p = session.progress_report()
    assert p["best_gof_pvalue"] is not None and p["best_gof_pvalue"] >= 0.05  # GoF LOOKS adequate
    assert session.best.sigma_inflated is True                                # but sigma inflated
    assert p["bic_improvement_over_last_n"] == pytest.approx(0.0)             # and BIC is flat
    assert p["should_stop"] is False                                          # -> no plateau stop


def test_estimated_sigma_not_inflated_is_adequate_without_gof(session):
    """With an ESTIMATED, non-inflated sigma the adequacy gate OMITS the (uninformative) chi2 GoF:
    a flat run is adequate and plateau-stops even when its chi2 would FAIL a GoF test."""
    def _fs():
        f = _fitsim(-50.0, 500.0)  # chi2 >> dof -> GoF would reject, if it were consulted
        f["noise_sigma_report"] = {"sigma": {"fitted": 0.05, "upper_bound_cap": 0.1,
                                             "data_std": 0.2, "fraction_of_data_std": 0.25}}
        return f
    session.record_fit_progress(1, _fs(), {"is_new_best": True})
    for v in range(2, 2 + PLATEAU_WINDOW):
        session.record_fit_progress(v, _fs(), {"is_new_best": False})
    p = session.progress_report()
    assert p["best_gof_pvalue"] is not None and p["best_gof_pvalue"] < 0.05  # GoF WOULD reject
    assert session.best.sigma_estimated is True and session.best.sigma_inflated is False
    assert p["should_stop"] is True                                          # adequate anyway
    assert "plateau" in p["stop_reason"].lower() and "chi2 gof" not in p["stop_reason"].lower()


def test_inflated_sigma_streak_cap_reason_carries_sigma_advice(session):
    """When an inflated-sigma run finally trips the streak cap, its stop_reason carries BOTH the
    structural nudge and the noise-side (inflated-sigma) warning so the agent fixes the mechanism
    rather than leaning on the free sigma."""
    session.record_fit_progress(1, _fitsim_inflated_sigma(-50.0, 18.0), {"is_new_best": True})
    for v in range(2, 2 + NON_IMPROVING_STREAK_CAP):
        session.record_fit_progress(v, _fitsim_inflated_sigma(-50.0, 18.0), {"is_new_best": False})
    p = session.progress_report()
    assert p["should_stop"] is True
    reason = p["stop_reason"].lower()
    assert "structural" in reason and "sigma" in reason and "inflated" in reason


def _fitsim_collapsed_sigma(nll: float, chi2: float, *, n: int = 20, k: int = 2) -> dict:
    """Like ``_fitsim`` but the fit reports an ESTIMATED sigma collapsed far below the data std
    (fraction < NOISE_SIGMA_COLLAPSE_FRACTION) — overfitting the noise, the other implausible tail."""
    fs = _fitsim(nll, chi2, n=n, k=k)
    fs["noise_sigma_report"] = {
        "sigma": {"fitted": NOISE_SIGMA_COLLAPSE_FRACTION / 10, "upper_bound_cap": 0.5,
                  "data_std": 1.0,
                  # derived from the constant so a threshold recalibration can't silently make
                  # this fixture stop being "collapsed" (it did, at [0.01, 0.75] — see #118)
                  "fraction_of_data_std": NOISE_SIGMA_COLLAPSE_FRACTION / 10}}
    return fs


def test_sigma_gamed_improvement_does_not_reset_the_streak(session):
    """A sigma-GAMED new best — a lower BIC bought by an inflated estimated sigma — does NOT reset
    the non-improving streak, so an all-gamed run trips the streak cap instead of running to the
    recursion limit. With no plausible best to protect, the gamed model is still recorded as best
    (best_version not None) so the adequacy gate + judge can handle it."""
    # Every op is a genuine new best by BIC (nll strictly drops) but each games the sigma.
    for i, v in enumerate(range(1, 1 + NON_IMPROVING_STREAK_CAP)):
        session.record_fit_progress(v, _fitsim_inflated_sigma(-50.0 - i, 18.0),
                                    {"is_new_best": True})
    p = session.progress_report()
    assert session.best is not None and session.best.sigma_inflated is True  # gamed best kept
    assert p["non_improving_streak"] == NON_IMPROVING_STREAK_CAP  # gamed bests never reset it
    assert p["should_stop"] is True                             # -> trips the streak cap


def test_gamed_fit_does_not_overwrite_a_plausible_best_checkpoint(session):
    """A plausible best-BIC checkpoint is NOT overwritten by a later, strictly-lower-BIC sigma-gamed
    fit: adoption is gated on sigma plausibility, so the judge always scores the best PLAUSIBLE
    model. (Regression guard: extra search runway must not demote a passing checkpoint to a gamed
    one and flip a would-pass to a fail.)"""
    plausible = _fitsim(-50.0, 18.0)
    plausible["noise_sigma_report"] = {"sigma": {"fitted": 0.05, "upper_bound_cap": 0.1,
                                                 "data_std": 0.2, "fraction_of_data_std": 0.25}}
    session.record_fit_progress(1, plausible, {"is_new_best": True})
    best_v, best_bic = session.best.version, session.best.bic
    assert best_v == 1 and session.best.sigma_inflated is False
    # A later fit with STRICTLY LOWER BIC (nll -80 < -50) but an inflated sigma must NOT take over.
    session.record_fit_progress(2, _fitsim_inflated_sigma(-80.0, 18.0), {"is_new_best": True})
    assert session.best.version == best_v            # plausible checkpoint preserved...
    assert session.best.bic == best_bic
    assert session.best.sigma_inflated is False       # ...best is still the plausible model


def test_a_plausible_new_best_after_gaming_resets_the_streak(session):
    """The gate is specific to gamed improvements: a genuine (plausible-sigma) new best still resets
    the streak, so fixing the noise model immediately clears the accumulated penalty."""
    for i, v in enumerate(range(1, 4)):
        session.record_fit_progress(v, _fitsim_inflated_sigma(-50.0 - i, 18.0),
                                    {"is_new_best": True})
    assert session.progress_report()["non_improving_streak"] == 3  # gamed improvements accrued
    plausible = _fitsim(-60.0, 18.0)
    plausible["noise_sigma_report"] = {"sigma": {"fitted": 0.05, "upper_bound_cap": 0.1,
                                                 "data_std": 0.2, "fraction_of_data_std": 0.25}}
    session.record_fit_progress(4, plausible, {"is_new_best": True})
    assert session.progress_report()["non_improving_streak"] == 0  # a real fix clears it


def _fs_plausible_sigma_flagged(flagged: bool):
    """A fit with a PLAUSIBLE estimated sigma (neither inflated nor collapsed) whose residuals are
    structurally flagged iff `flagged` — the sneyd case: underfit that the sigma gate can't catch."""
    f = _fitsim(-50.0, 18.0)
    f["noise_sigma_report"] = {"sigma": {"fitted": 0.05, "upper_bound_cap": 0.1,
                                         "data_std": 0.2, "fraction_of_data_std": 0.25}}
    f["residual_flagged"] = flagged
    return f


def test_flagged_residual_structure_blocks_the_adequate_plateau_stop(session):
    """A best model with a PLAUSIBLE estimated sigma but structurally-flagged residuals (sign/runs)
    is NOT adequate — its form is incomplete — so a flat-BIC run does NOT plateau-stop; it keeps
    searching (sneyd: underfits that used to stop 'adequate' while flagging residuals)."""
    session.record_fit_progress(1, _fs_plausible_sigma_flagged(True), {"is_new_best": True})
    for v in range(2, 2 + PLATEAU_WINDOW):
        session.record_fit_progress(v, _fs_plausible_sigma_flagged(True), {"is_new_best": False})
    p = session.progress_report()
    assert session.best.sigma_estimated is True and session.best.sigma_inflated is False
    assert session.best.residual_flagged is True
    assert p["bic_improvement_over_last_n"] == pytest.approx(0.0)  # BIC is flat...
    assert p["should_stop"] is False                              # ...but residual structure blocks it


def test_unflagged_residual_still_plateau_stops(session):
    """Contrast: the SAME plausible-sigma flat run WITHOUT flagged residuals still plateau-stops —
    the residual gate only bites when the sign/runs tests actually flag structure (no regression)."""
    session.record_fit_progress(1, _fs_plausible_sigma_flagged(False), {"is_new_best": True})
    for v in range(2, 2 + PLATEAU_WINDOW):
        session.record_fit_progress(v, _fs_plausible_sigma_flagged(False), {"is_new_best": False})
    p = session.progress_report()
    assert session.best.residual_flagged is False
    assert p["should_stop"] is True and "plateau" in p["stop_reason"].lower()


def test_flagged_residual_streak_cap_carries_structural_nudge(session):
    """When a residual-flagged run finally trips the streak cap, its stop_reason carries the
    structural nudge (change the FORM) — the run never counted as adequate."""
    session.record_fit_progress(1, _fs_plausible_sigma_flagged(True), {"is_new_best": True})
    for v in range(2, 2 + NON_IMPROVING_STREAK_CAP):
        session.record_fit_progress(v, _fs_plausible_sigma_flagged(True), {"is_new_best": False})
    p = session.progress_report()
    assert p["should_stop"] is True
    assert "structural" in p["stop_reason"].lower()


def test_collapsed_sigma_blocks_the_adequate_plateau_stop(session):
    """A COLLAPSED estimated sigma (overfitting the noise) is NOT adequate either — the agent gate
    must reject it like the judge's success gate does, so a run doesn't plateau-stop 'adequate' and
    then get rejected downstream for the same sigma."""
    session.record_fit_progress(1, _fitsim_collapsed_sigma(-50.0, 18.0), {"is_new_best": True})
    for v in range(2, 2 + PLATEAU_WINDOW):
        session.record_fit_progress(v, _fitsim_collapsed_sigma(-50.0, 18.0), {"is_new_best": False})
    p = session.progress_report()
    assert session.best.sigma_estimated is True and session.best.sigma_collapsed is True
    assert session.best.sigma_inflated is False
    assert p["should_stop"] is False


def test_collapsed_sigma_streak_cap_reason_carries_sigma_advice(session):
    """A collapsed-sigma run that trips the streak cap gets the structural nudge + the collapsed
    (overfitting-the-noise) warning."""
    session.record_fit_progress(1, _fitsim_collapsed_sigma(-50.0, 18.0), {"is_new_best": True})
    for v in range(2, 2 + NON_IMPROVING_STREAK_CAP):
        session.record_fit_progress(v, _fitsim_collapsed_sigma(-50.0, 18.0), {"is_new_best": False})
    reason = session.progress_report()["stop_reason"].lower()
    assert "structural" in reason and "collapsed" in reason and "overfitting" in reason


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


# --- (d) structural-inadequacy nudge folded into the stop reason (issue #82) --------------
def test_streak_cap_stop_reason_carries_structural_nudge_when_inadequate(session):
    """A never-adequate run that hits the streak cap is the structural-inadequacy case: its
    stop_reason must tell the agent to change the model's FORM, not re-fit the same structure."""
    session.record_fit_progress(1, _fitsim(-50.0, 500.0), {"is_new_best": True})  # never adequate
    for v in range(2, 2 + NON_IMPROVING_STREAK_CAP):
        session.record_fit_progress(v, _fitsim(-50.0, 500.0), {"is_new_best": False})

    reason = session.progress_report()["stop_reason"].lower()
    assert "streak" in reason
    assert "structural" in reason and "form" in reason
    assert "make_changes" in reason


def test_adequate_plateau_stop_reason_omits_structural_nudge(session):
    """An adequate-and-flat plateau is a SUCCESS stop, so its stop_reason must NOT append the
    structural nudge — telling an already-adequate model it 'cannot fit the data' is contradictory.
    """
    session.record_fit_progress(1, _fitsim(-50.0, 18.0), {"is_new_best": True})  # adequate
    for v in range(2, 2 + PLATEAU_WINDOW):
        session.record_fit_progress(v, _fitsim(-50.0, 18.0), {"is_new_best": False})

    reason = session.progress_report()["stop_reason"].lower()
    assert "plateau" in reason and "adequate" in reason
    assert "structural" not in reason  # no nudge on a success stop
    assert "cannot fit" not in reason


def test_running_report_has_no_stop_reason_and_no_nudge(session):
    """While the run is still making progress there is no stop and hence no nudge text."""
    for i, v in enumerate(range(1, 6)):
        session.record_fit_progress(v, _fitsim(-50.0 - 5.0 * i, 18.0), {"is_new_best": True})
    p = session.progress_report()
    assert p["should_stop"] is False
    assert p["stop_reason"] is None


# --- (e) non-finite scores must never become the checkpoint --------------------------------
# Regression for the crauste sweep run whose FIRST fit returned a NaN neg-log-likelihood: NaN
# is a float, so it passed the old `bic is not None` guard, became best_bic, and then locked
# the run out for good — every later `bic <= NaN - delta` is False, so no fit could ever be a
# new best again. 16 fits, 73 minutes, and an unscoreable result.
def test_nan_nll_yields_no_bic_at_all():
    """The upstream chokepoint: a non-finite likelihood is a MISSING score, not a NaN score."""
    g = fitting.goodness_of_fit(float("nan"), 145.98, 10, 21)
    assert g["bic"] is None and g["aic"] is None
    assert fitting.goodness_of_fit(float("inf"), 145.98, 10, 21)["bic"] is None
    # a non-finite chi2 likewise yields no chi2-based scores, but the bic still stands
    g2 = fitting.goodness_of_fit(-50.0, float("nan"), 2, 20)
    assert g2["bic"] is not None
    assert g2["reduced_chi2"] is None and g2["chi2_gof_pvalue"] is None


def test_nan_first_fit_does_not_become_the_checkpoint(session):
    """A NaN first fit must leave the checkpoint unset, so the NEXT good fit can claim it."""
    session.record_fit_progress(1, _fitsim(float("nan"), 145.98, n=21, k=10),
                                {"is_new_best": True})
    p = session.progress_report()
    assert p["best_version"] is None
    assert p["best_bic"] is None
    assert p["non_improving_streak"] == 1   # an unscoreable fit is not progress

    # The next real fit is a legitimate new best and claims the checkpoint.
    fs = _fitsim(266.62, 100.0, n=21, k=10)
    imp = fitting.compute_improvement(session, nllh=266.62, objective_before=8377.0, fitsim=fs)
    assert imp["is_new_best"] is True
    session.record_fit_progress(2, fs, imp)
    p = session.progress_report()
    assert p["best_version"] == 2
    assert p["best_bic"] == pytest.approx(fitting.goodness_of_fit(266.62, 100.0, 10, 21)["bic"])
    assert p["non_improving_streak"] == 0


def test_improving_fits_are_never_locked_out_by_a_poisoned_checkpoint(session):
    """A NaN can never become the checkpoint, and fits after an unscoreable one still register.

    This is the exact shape of the lost run — an unscoreable fit followed by fits at BIC 580,
    569 and 519 — none of which could register under the old comparison. The NaN is now refused
    by `Checkpoint` itself, so the poisoned state is unconstructible rather than survivable.
    """
    with pytest.raises(ValueError, match="real number"):
        _checkpoint(bic=float("nan"), version=1, nll=float("nan"))
    # An unscoreable fit leaves no checkpoint at all, which is the state the run recovers from.
    session.best = None
    for bic_target in (580.97, 569.19, 519.86):
        # BIC = k·ln(n) + 2·nll  =>  nll = (bic - k·ln(n)) / 2
        import math
        nll = (bic_target - 10 * math.log(21)) / 2
        fs = _fitsim(nll, 100.0, n=21, k=10)
        imp = fitting.compute_improvement(session, nllh=nll, objective_before=8377.0, fitsim=fs)
        assert imp["is_new_best"] is True, f"BIC {bic_target} must be able to beat a NaN"
        session.record_fit_progress(2, fs, imp)
        assert session.best.bic == pytest.approx(bic_target)


def test_stall_without_any_checkpoint_reports_an_evaluation_failure(session):
    """A stall where nothing ever scored is an EVALUATION problem: the stop_reason must say so
    instead of blaming the model's structure and sending the agent re-authoring."""
    for v in range(1, 2 + NON_IMPROVING_STREAK_CAP):
        session.record_fit_progress(v, _fitsim(float("nan"), 145.98, n=21, k=10),
                                    {"is_new_best": True})
    reason = session.progress_report()["stop_reason"].lower()
    assert "evaluation failure" in reason
    assert "n_missing_simulations" in reason
    assert "structural inadequacy" not in reason
    assert "make_changes" not in reason


# --- fixed-sigma GoF must be TWO-sided (#118) ------------------------------------------------
def _fitsim_fixed_sigma(chi2: float, *, n: int = 20, k: int = 2) -> dict:
    """A fit with NO estimated sigma (empty noise_sigma_report -> the fixed-sigma adequacy
    branch), whose chi2 the caller sets to drive the GoF p-value."""
    fs = _fitsim(-50.0, chi2, n=n, k=k)
    fs["noise_sigma_report"] = {}
    return fs


def test_fixed_sigma_overfit_is_not_adequate(session):
    """chi2 FAR BELOW its dof means the residuals are much smaller than the KNOWN noise — the model
    is fitting the noise, which is not adequacy. A one-sided `p >= 0.05` test scored exactly this as
    adequate (rahman: chi2/n = 0.016, p = 0.9999999), letting an overfit run plateau-stop."""
    # chi2 << dof -> p -> 1
    fs = _fitsim_fixed_sigma(0.3)
    session.record_fit_progress(1, fs, {"is_new_best": True})
    for v in range(2, 2 + PLATEAU_WINDOW):
        session.record_fit_progress(v, fs, {"is_new_best": False})
    p = session.progress_report()
    assert session.best.sigma_estimated is False
    assert p["best_gof_pvalue"] > 0.99          # would pass a one-sided `>= 0.05` test
    assert p["should_stop"] is False            # ...but the two-sided gate blocks the stop


def test_fixed_sigma_well_calibrated_fit_is_still_adequate(session):
    """No regression: chi2 near its dof stays adequate and plateau-stops as before."""
    fs = _fitsim_fixed_sigma(18.0)              # chi2 ~ dof = 18 -> p comfortably mid-range
    session.record_fit_progress(1, fs, {"is_new_best": True})
    for v in range(2, 2 + PLATEAU_WINDOW):
        session.record_fit_progress(v, fs, {"is_new_best": False})
    p = session.progress_report()
    assert 0.05 <= p["best_gof_pvalue"] <= 0.995
    assert p["should_stop"] is True and "plateau" in p["stop_reason"].lower()


# --- (d) NO fitted-op budget: the streak cap is the only progress-based bound ------------------
def test_a_run_that_keeps_finding_new_bests_is_not_cut_off_by_an_op_budget(session):
    """A 20-fitted-op cap was tried and removed. It was not redundant with the streak cap -- that
    resets on every new best, so 73 of 333 clean runs in the Aug 8-10 sweep passed 20 fits anyway --
    but it cost ~1.8% of passes, charged against exactly the models that explore longest, which
    biases the comparison the benchmark exists to make. Pins that a long, still-improving run runs on.
    """
    nll = -50.0
    for v in range(1, 31):                       # well past the old cap of 20
        nll -= NEW_BEST_MIN_DELTA_BIC
        session.record_fit_progress(v, _fitsim(nll, 500.0), {"is_new_best": True})
    p = session.progress_report()
    assert p["n_fitted_ops"] == 30
    assert p["non_improving_streak"] == 0
    assert p["should_stop"] is False              # still improving -> keep going
    assert p["stop_reason"] is None
    assert "fitted_ops_cap" not in p             # the knob is gone from the contract too


def test_a_stalled_long_run_is_still_stopped_by_the_streak_cap(session):
    """Removing the budget must not remove the bound that matters: a run that stops improving is
    still caught, and still gets the structural diagnosis rather than a budget notice."""
    nll = -50.0
    for v in range(1, 11):
        nll -= NEW_BEST_MIN_DELTA_BIC
        session.record_fit_progress(v, _fitsim(nll, 500.0), {"is_new_best": True})
    for v in range(11, 11 + NON_IMPROVING_STREAK_CAP):
        session.record_fit_progress(v, _fitsim(nll, 500.0), {"is_new_best": False})
    p = session.progress_report()
    assert p["should_stop"] is True
    assert "streak" in p["stop_reason"].lower()
    assert "budget" not in p["stop_reason"].lower()
    assert "STRUCTURAL" in p["stop_reason"]


# --- (e) a scale-coupled sigma has no cross-model meaning ------------------------------------
def test_reference_sigma_basis_is_declined_for_a_scale_coupled_sigma():
    """fiedler's noise is `s_<gel> * sigma`, with the gels' `s_*` ESTIMATED over 77.7..999.99. Only
    the product is identified, so a model at different internal units reaches the SAME effective
    noise with a proportionally different sigma. The judge's reference basis divided one model's raw
    sigma by another's and called 19 of 20 fiedler runs implausible, silently failing 17 that had
    beaten the BIC bar (best dBIC -28.8; the one scored pass was +0.50)."""
    from petab_harness.runner.scoring import _reference_sigma_flag

    reference = {"noise_sigma": {"sigma_pMek": 0.0005804511382150003},
                 "noise_sigma_effective": {"sigma_pMek": 0.0005804511382150003},
                 "k": 22, "n": 72}
    # The real numbers from 51526222_124: plausible on its own data-driven metrics, 179x the
    # reference's raw sigma.
    entry = {"fitted": 0.10386657663301647, "effective_sigma": 0.10386657663301647,
             "n_points": 36, "data_std": 0.33907019890637025,
             "fraction_of_data_std": 0.3063276482805788, "status": "ok"}

    coupled = _reference_sigma_flag({"sigma_pMek": {**entry, "scale_coupled": True}},
                                    reference, k=22, n=72)
    assert coupled is None          # declined -> the judge falls through to the data-driven band

    # Without the coupling the basis still applies, and still flags this ratio -- so the fix is
    # scoped to the coupling and does not disable the gate generally.
    plain = _reference_sigma_flag({"sigma_pMek": {**entry, "scale_coupled": False}},
                                  reference, k=22, n=72)
    assert plain is not None and plain["inflated"] is True
    assert plain["max_ratio"] > 100


@pytest.mark.parametrize("gain_estimated,expected", [(1, True), (0, False)])
def test_only_an_estimated_co_factor_breaks_comparability(gain_estimated, expected):
    """Only an ESTIMATED co-factor breaks the ratio. A FIXED one is a declared constant of the
    problem, identical on both sides, so it cancels and the reference basis stays valid.

    Built as a real fiedler-shaped problem rather than probed on the toy session: the toy sigma has
    no composite noiseFormula at all, so `sigma_scale_coupled` returned False at the `len(parts) < 2`
    guard and never reached the `is_parameter_estimated` branch. Inverting that check would have left
    such a test passing while silently re-flagging every fiedler-shaped run.
    """
    import pandas as pd
    import petab.v1 as petab
    from petab.v1.C import (ESTIMATE, LOWER_BOUND, MEASUREMENT, NOISE_FORMULA, NOISE_PARAMETERS,
                            NOMINAL_VALUE, OBSERVABLE_FORMULA, OBSERVABLE_ID, PARAMETER_ID,
                            PARAMETER_SCALE, SIMULATION_CONDITION_ID, TIME, UPPER_BOUND)
    from petab.v1.models.sbml_model import SbmlModel
    from petab_mcp import reconcile as R

    ant = ("model g\n  compartment c = 1;\n  species pMek in c;\n  pMek = 1.0;\n"
           "  kf = 0.5;\n  r: pMek => ; kf * pMek;\nend\n")
    prob = petab.Problem(model=SbmlModel.from_antimony(ant))
    prob.observable_df = pd.DataFrame(
        {OBSERVABLE_ID: ["pMek_obs"], OBSERVABLE_FORMULA: ["pMek * observableParameter1_pMek_obs"],
         NOISE_FORMULA: ["noiseParameter1_pMek_obs * noiseParameter2_pMek_obs"]}
    ).set_index(OBSERVABLE_ID)
    prob.measurement_df = pd.DataFrame({
        OBSERVABLE_ID: ["pMek_obs"] * 4,
        SIMULATION_CONDITION_ID: ["c0"] * 4,
        TIME: [0.0, 1.0, 2.0, 3.0],
        MEASUREMENT: [1.0, 0.6, 0.4, 0.2],
        # The fiedler shape exactly: a per-gel gain paired with the shared sigma.
        NOISE_PARAMETERS: ["s_gel;sigma_pMek"] * 4,
    })
    prob.parameter_df = pd.DataFrame({
        PARAMETER_ID: ["s_gel", "sigma_pMek", "kf"],
        PARAMETER_SCALE: ["log10"] * 3,
        LOWER_BOUND: [1e-5] * 3, UPPER_BOUND: [1e3] * 3,
        NOMINAL_VALUE: [100.0, 0.001, 0.5],
        ESTIMATE: [gain_estimated, 1, 1],
    }).set_index(PARAMETER_ID)

    assert R.sigma_scale_coupled(prob, "sigma_pMek") is expected
    # The gain itself is never the sigma under test, either way.
    assert R.sigma_scale_coupled(prob, "kf") is False


def test_an_unscorable_run_is_diagnosed_by_the_streak_cap_not_the_budget(session):
    """A run that never produces a usable score reaches the STREAK cap first, and must get the
    unscored-stall diagnosis rather than a budget notice.

    Pins the ordering invariant: `record_fit_progress` advances the non-improving streak on every
    non-finite score (regardless of `is_new_best`), so such a run trips the cap at op 15 and the
    op budget is unreachable for it. That is why the budget branch's `best is None` wording is
    documented as defensive. If this test ever fails because the budget claimed the stop, the
    "no best version to keep" wording has become live and needs its own coverage.
    """
    for v in range(1, NON_IMPROVING_STREAK_CAP + 2):
        session.record_fit_progress(
            v, {"status": "fit", "neg_log_likelihood": None, "chi2": None,
                "n_data_points": 20, "n_estimated": 2}, {"is_new_best": True})

    p = session.progress_report()
    assert p["best_version"] is None
    assert p["should_stop"] is True
    reason = p["stop_reason"]
    assert "streak" in reason.lower() and "budget" not in reason.lower()
    # The actionable finding — the fits are failing to EVALUATE, not the form failing to fit.
    assert "No fit in this run produced a" in reason
    assert "best scored version is kept" not in reason
