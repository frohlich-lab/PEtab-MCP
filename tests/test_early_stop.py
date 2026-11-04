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
    NEW_BEST_MIN_DELTA_SCORE,
    NOISE_SIGMA_COLLAPSE_FRACTION,
    NON_IMPROVING_STREAK_CAP,
    PLATEAU_WINDOW,
)


def _checkpoint(*, score, version, nll, **over):
    """A best-score checkpoint seeded directly, as record_fit_progress would adopt one."""
    from petab_mcp.session import Checkpoint

    fields = dict(
        version=version,
        score=score,
        nll=nll,
        gof_pvalue=None,
        sigma_estimated=False,
        sigma_inflated=False,
        sigma_collapsed=False,
        residual_flagged=False,
    )
    fields.update(over)
    return Checkpoint(**fields)


def _fitsim(nll: float, chi2: float, *, n: int = 20, k: int = 2) -> dict:
    """A minimal ``fit_and_simulate``-shaped payload for a genuine fit."""
    return {
        "status": "fit",
        "neg_log_likelihood": nll,
        "chi2": chi2,
        "n_data_points": n,
        "n_estimated": k,
    }


# --- (a) effect-size gate on is_new_best (tracks BIC) -------------------------------------
def test_is_new_best_tracks_bic_effect_size(session):
    # No checkpoint yet -> the first fit is always a new best; it reports its own BIC.
    fs0 = _fitsim(-50.0, 18.0)
    first = fitting.compute_improvement(
        session, nllh=fs0["neg_log_likelihood"], objective_before=-40.0, fitsim=fs0
    )
    assert first["is_new_best"] is True
    assert first["best_prior_criterion_value"] is None
    assert first["bic"] is not None

    # Seed the best-BIC checkpoint at that fit's BIC (as record_fit_progress would).
    session.best = _checkpoint(score=first["criterion_value"], version=1, nll=-50.0)

    # BIC = k·ln(n) + 2·nll, so at fixed k,n a ΔBIC of X needs a ΔNLL of X/2. A sub-threshold BIC
    # improvement (< NEW_BEST_MIN_DELTA_SCORE) is NOT a new best, even though nll strictly dropped.
    d_small = NEW_BEST_MIN_DELTA_SCORE / 2
    fs_small = _fitsim(-50.0 - d_small / 2, 18.0)
    small = fitting.compute_improvement(
        session, nllh=fs_small["neg_log_likelihood"], objective_before=-50.0, fitsim=fs_small
    )
    assert small["improvement_vs_best_prior"] == pytest.approx(d_small)
    assert small["is_new_best"] is False

    # An improvement EXACTLY at the threshold counts (the gate is ">=", not ">").
    fs_edge = _fitsim(
        -50.0 - NEW_BEST_MIN_DELTA_SCORE / 2, 18.0
    )  # ΔBIC == NEW_BEST_MIN_DELTA_SCORE
    edge = fitting.compute_improvement(
        session, nllh=fs_edge["neg_log_likelihood"], objective_before=-50.0, fitsim=fs_edge
    )
    assert edge["improvement_vs_best_prior"] == pytest.approx(NEW_BEST_MIN_DELTA_SCORE)
    assert edge["is_new_best"] is True

    # A BIC improvement comfortably past the threshold does count.
    d_big = NEW_BEST_MIN_DELTA_SCORE + 2
    fs_big = _fitsim(-50.0 - d_big / 2, 18.0)
    big = fitting.compute_improvement(
        session, nllh=fs_big["neg_log_likelihood"], objective_before=-50.0, fitsim=fs_big
    )
    assert big["improvement_vs_best_prior"] == pytest.approx(d_big)
    assert big["is_new_best"] is True


def test_is_new_best_rejects_nll_gain_that_worsens_bic(session):
    """The whole point of gating on BIC not nll: a fit that lowers nll but adds parameters that
    RAISE BIC is not a new best."""
    session.best = _checkpoint(  # checkpoint at k=2
        score=fitting.goodness_of_fit(-50.0, 18.0, 2, 20)["criterion_value"], version=1, nll=-50.0
    )
    # nll improves (-50 -> -51) but k jumps 2 -> 5: BIC = k·ln(20)+2·nll gets WORSE.
    fs = _fitsim(-51.0, 18.0, k=5)
    imp = fitting.compute_improvement(session, nllh=-51.0, objective_before=-50.0, fitsim=fs)
    assert imp["improvement_vs_best_prior"] < 0  # BIC got worse
    assert imp["is_new_best"] is False


def test_a_fit_short_of_a_new_best_says_what_it_needed(session):
    """`is_new_best: false` named no threshold, so a real but sub-threshold gain read as a flag
    that never fires. The fit result names the gain it made over the best, and the `make_changes`
    description states the gate once: it is a constant, and every fit result is re-sent."""
    from petab_mcp.server import _MAKE_CHANGES_DOC

    best = fitting.goodness_of_fit(-50.0, 18.0, 2, 20)["criterion_value"]
    session.best = _checkpoint(score=best, version=1, nll=-50.0)
    fs = _fitsim(-50.0 - NEW_BEST_MIN_DELTA_SCORE / 4, 18.0)  # ΔBIC = half the gate
    imp = fitting.compute_improvement(session, nllh=fs["neg_log_likelihood"], fitsim=fs)
    shown = fitting.fit_result_block(fs, imp)["improvement"]
    assert set(shown) == {"self_improvement", "improvement_vs_best_prior", "is_new_best"}
    assert shown["is_new_best"] is False
    assert shown["improvement_vs_best_prior"] == pytest.approx(NEW_BEST_MIN_DELTA_SCORE / 2)
    assert f"`improvement_vs_best_prior` reaches {NEW_BEST_MIN_DELTA_SCORE:g}" in _MAKE_CHANGES_DOC


# --- (b) streak cap + plateau + (c) best-BIC checkpoint ------------------------------------
def test_first_fitted_op_sets_checkpoint_and_resets_streak(session):
    session.record_fit_progress(1, _fitsim(-50.0, 18.0), {"is_new_best": True})
    p = session.progress_report()
    assert p["n_fitted_ops"] == 1
    assert p["non_improving_streak"] == 0
    assert p["best_version"] == 1
    assert p["best_score"] is not None
    assert p["should_stop"] is False


def test_streak_cap_stops_a_never_adequate_session(session):
    """The load-bearing lever: a session whose best never reaches adequacy (χ² GoF rejected) still
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


def test_plateau_stops_an_adequate_flat_session(session):
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
    assert p["score_improvement_over_last_n"] == pytest.approx(0.0)
    assert p["should_stop"] is True
    assert "plateau" in p["stop_reason"].lower()


def _fitsim_inflated_sigma(nll: float, chi2: float, *, n: int = 20, k: int = 2) -> dict:
    """Like ``_fitsim`` but the fit reports an ESTIMATED noise sigma fitted ABOVE its data-driven
    cap — the inflated-sigma case where a passing chi2 GoF is not genuine adequacy."""
    fs = _fitsim(nll, chi2, n=n, k=k)
    fs["noise_sigma_report"] = {
        "sigma": {
            "fitted": 0.5,
            "upper_bound_cap": 0.1,
            "data_std": 0.2,
            "fraction_of_data_std": 2.5,
        }
    }
    return fs


def test_inflated_sigma_blocks_the_adequate_plateau_stop(session):
    """A flat session whose good chi2 GoF is only propped up by an inflated estimated sigma is NOT
    adequate for the plateau gate, so it must NOT stop early — the underfit case where a free
    sigma absorbs the misfit and fakes chi2/dof ~ 1."""
    session.record_fit_progress(1, _fitsim_inflated_sigma(-50.0, 18.0), {"is_new_best": True})
    for v in range(2, 2 + PLATEAU_WINDOW):
        session.record_fit_progress(v, _fitsim_inflated_sigma(-50.0, 18.0), {"is_new_best": False})
    p = session.progress_report()
    assert p["best_gof_pvalue"] is not None and p["best_gof_pvalue"] >= 0.05  # GoF LOOKS adequate
    assert session.best.sigma_inflated is True  # but sigma inflated
    assert p["score_improvement_over_last_n"] == pytest.approx(0.0)  # and BIC is flat
    assert p["should_stop"] is False  # -> no plateau stop


def _fitsim_absorbed_misfit(nll: float, ratio: float) -> dict:
    """Three ESTIMATED sigmas inside the data-std band, no residual flag, and one of them `ratio`
    times its data-driven noise floor: a misfit absorbed into that sigma at ratio > 1."""
    fs = _fitsim(nll, 20.0)
    fs["noise_sigma_report"] = {
        f"sigma_{obs}": {
            "fitted": 0.2 * r,
            "effective_sigma": 0.2 * r,
            "upper_bound_cap": 0.75,
            "data_std": 1.0,
            "fraction_of_data_std": 0.2 * r,
            "data_noise_estimate": 0.2,
            "fitted_over_data_noise": r,
            "status": "ok",
        }
        for obs, r in (("pSTAT5", ratio), ("tSTAT5", 1.0), ("pEpoR", 1.0))
    }
    fs["residual_flagged"] = False
    return fs


@pytest.mark.parametrize("ratio", [1.6, 1.0])
def test_an_estimated_sigma_model_runs_to_the_streak_cap(session, ratio):
    """#196: bachmann's round-3 run plateau-stopped "adequate" at fit 9, ΔBIC +62, after five
    cleanup ops, its sigmas 1.4-1.8x the published fit's. Neither the absorbed misfit (1.6) nor a
    sigma at its floor (1.0) plateau-stops; both stop at the cap, with no claim either way."""
    session.record_fit_progress(1, _fitsim_absorbed_misfit(-50.0, ratio), {"is_new_best": True})
    for v, nll in zip(range(2, 2 + PLATEAU_WINDOW), (-49.9, -50.0, -49.8, -50.0, -49.9)):
        session.record_fit_progress(v, _fitsim_absorbed_misfit(nll, ratio), {"is_new_best": False})
    p = session.progress_report()
    assert session.best.sigma_estimated is True and session.best.sigma_plausible is True
    assert p["score_improvement_over_last_n"] == pytest.approx(0.0)
    assert p["should_stop"] is False

    for v in range(2 + PLATEAU_WINDOW, 2 + NON_IMPROVING_STREAK_CAP):
        session.record_fit_progress(
            v, _fitsim_absorbed_misfit(-50.0, ratio), {"is_new_best": False}
        )
    p = session.progress_report()
    assert p["should_stop"] is True
    reason = p["stop_reason"]
    assert reason.startswith("non-improving-streak cap reached")
    assert "adequate" not in reason and "STRUCTURAL" not in reason


def test_inflated_sigma_streak_cap_reason_carries_sigma_advice(session):
    """When an inflated-sigma session finally trips the streak cap, its stop_reason carries BOTH the
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
        "sigma": {
            "fitted": NOISE_SIGMA_COLLAPSE_FRACTION / 10,
            "upper_bound_cap": 0.5,
            "data_std": 1.0,
            # derived from the constant so a threshold recalibration can't silently make
            # this fixture stop being "collapsed" (it did, at [0.01, 0.75] — see #118)
            "fraction_of_data_std": NOISE_SIGMA_COLLAPSE_FRACTION / 10,
        }
    }
    return fs


def _fitsim_plausible_sigma(nll: float, chi2: float, *, n: int = 20, k: int = 2) -> dict:
    """Like ``_fitsim`` but with an ESTIMATED sigma comfortably inside the plausibility band —
    proof, when it appears in a session, that the band is satisfiable on this problem."""
    fs = _fitsim(nll, chi2, n=n, k=k)
    fs["noise_sigma_report"] = {
        "sigma": {
            "fitted": 0.05,
            "upper_bound_cap": 0.1,
            "data_std": 0.2,
            "fraction_of_data_std": 0.25,
        }
    }
    return fs


@pytest.mark.parametrize(
    ("gamed", "streak_after"),
    [(_fitsim_inflated_sigma, 0), (_fitsim_collapsed_sigma, 3)],
    ids=["inflated", "collapsed"],
)
def test_a_gamed_new_best_after_a_plausible_version(session, gamed, streak_after):
    """#238: once one version had a plausible sigma, an inflated new best stopped resetting the
    streak, and lucarelli's run hit the cap 475 BIC into its descent, beside six versions saying
    `is_new_best: true`. An inflated new best resets it whatever came before; a collapsed one
    does not.

    The plausible fit is NOT the best: a plausible best declines every later gamed fit, so the
    streak would advance through the not-adopted branch and the test would pass either way."""
    session.record_fit_progress(1, gamed(-60.0, 18.0), {"is_new_best": True})
    session.record_fit_progress(2, _fitsim_plausible_sigma(-40.0, 18.0), {"is_new_best": False})
    session.record_fit_progress(3, gamed(-70.0, 18.0), {"is_new_best": True})
    assert session.best.version == 3
    assert session.non_improving_streak == streak_after


def test_an_unsatisfiable_inflation_band_does_not_convert_the_cap_into_a_fixed_budget(session):
    """The unsatisfiable-band failure (96 of 96 streak-capped sessions on two problems): where no
    fit ever clears the data-std band, "inflated" describes the PROBLEM, not the fit, and blocking
    every streak reset on it froze the search at exactly cap fits with BIC still falling.

    Safe to reset on, because an inflated sigma cannot BUY the improvement: at the ML optimum
    sigma_hat IS the rms residual, so nll rises monotonically with sigma. A fit that is a new best
    despite an inflated sigma got there by fitting better; the inflation is a symptom of the
    residual misfit, not the mechanism of the gain."""
    for i, v in enumerate(range(1, 1 + NON_IMPROVING_STREAK_CAP)):
        session.record_fit_progress(
            v, _fitsim_inflated_sigma(-50.0 - i, 18.0), {"is_new_best": True}
        )
    p = session.progress_report()
    assert p["non_improving_streak"] == 0
    assert p["should_stop"] is False  # not cut off mid-descent


def test_a_collapsed_new_best_advances_the_streak_and_the_cap_says_so(session):
    """The other tail is NOT safe to reset on, which is why the two are split. nll is unbounded
    BELOW as sigma_hat -> 0 (checked only by k*ln n), so a model fitting a small observable group
    nearly exactly gains without limit — a real exploit with no floor. The plateau stop never fires
    on an estimated sigma, so the cap must keep bounding it.

    #238: the cap said "no new best" beside versions saying `is_new_best: true`; it now counts the
    new bests that did not reset the streak, and says why."""
    for i, v in enumerate(range(1, 4)):
        session.record_fit_progress(
            v, _fitsim_collapsed_sigma(-50.0 - i, 18.0), {"is_new_best": True}
        )
    for v in range(4, 1 + NON_IMPROVING_STREAK_CAP):
        session.record_fit_progress(v, _fitsim_collapsed_sigma(-50.0, 18.0), {"is_new_best": False})
    p = session.progress_report()
    assert p["non_improving_streak"] == NON_IMPROVING_STREAK_CAP
    assert p["should_stop"] is True
    assert "3 of them were new bests with a collapsed estimated noise sigma" in p["stop_reason"]


def test_a_non_improving_inflated_fit_still_advances_the_streak(session):
    """The inflated reset applies to new bests only. A genuine plateau must still trip the cap in
    exactly cap fits, and with nothing to explain, the reason says only that."""
    for v in range(1, 1 + NON_IMPROVING_STREAK_CAP):
        session.record_fit_progress(v, _fitsim_inflated_sigma(-50.0, 18.0), {"is_new_best": False})
    p = session.progress_report()
    assert p["non_improving_streak"] == NON_IMPROVING_STREAK_CAP
    assert p["should_stop"] is True
    assert "of them" not in p["stop_reason"]


def test_gamed_fit_does_not_overwrite_a_plausible_best_checkpoint(session):
    """A plausible best-BIC checkpoint is NOT overwritten by a later, strictly-lower-BIC sigma-gamed
    fit: adoption is gated on sigma plausibility, so the best PLAUSIBLE model is always the one
    scored. (Regression guard: extra search runway must not demote a passing checkpoint to a gamed
    one and flip a would-pass to a fail.)"""
    plausible = _fitsim(-50.0, 18.0)
    plausible["noise_sigma_report"] = {
        "sigma": {
            "fitted": 0.05,
            "upper_bound_cap": 0.1,
            "data_std": 0.2,
            "fraction_of_data_std": 0.25,
        }
    }
    session.record_fit_progress(1, plausible, {"is_new_best": True})
    best_v, best_score = session.best.version, session.best.score
    assert best_v == 1 and session.best.sigma_inflated is False
    # A later fit with STRICTLY LOWER BIC (nll -80 < -50) but an inflated sigma must NOT take over.
    session.record_fit_progress(2, _fitsim_inflated_sigma(-80.0, 18.0), {"is_new_best": True})
    assert session.best.version == best_v  # plausible checkpoint preserved...
    assert session.best.score == best_score
    assert session.best.sigma_inflated is False  # ...best is still the plausible model


def test_a_declined_new_best_is_not_stamped_as_one(session, monkeypatch):
    """Regression (#141): `fit` stamped the score's verdict as `is_new_best` before the checkpoint
    declined a gamed fit, so the result and `list_versions` named a new best that `best_version`
    did not. Drives `fit` itself on synthetic fit numbers: the stamp is assembled there, which a
    direct `record_fit_progress` call never reaches."""
    plausible = {"sigma": {"fitted": 0.05, "data_std": 0.2, "fraction_of_data_std": 0.25}}
    reports = iter([plausible, _fitsim_inflated_sigma(0.0, 0.0)["noise_sigma_report"]])
    fits = iter([_fitsim(-50.0, 18.0), _fitsim(-80.0, 18.0)])  # the second scores far better

    def fit_numbers(n_starts=None):
        fs = {**next(fits), "n_starts": n_starts, "fitted_parameters": {}}
        return fs, fitting.compute_improvement(session, fs["neg_log_likelihood"], fitsim=fs)

    monkeypatch.setattr(session, "_auto_fit_and_improvement", fit_numbers)
    monkeypatch.setattr(fitting, "noise_sigma_report", lambda *_: next(reports))
    first = session.fit(n_starts=1)
    assert first["improvement"]["is_new_best"] is True
    res = session.fit(n_starts=2)  # more starts, so the de-dup lets the same structure refit

    imp = res["improvement"]
    assert imp["improvement_vs_best_prior"] >= NEW_BEST_MIN_DELTA_SCORE  # the score cleared it...
    assert imp["is_new_best"] is False  # ...but the checkpoint kept the first, and the flag says so
    assert "inflated" in imp["new_best_declined"]
    assert f"v{first['version']}" in imp["new_best_declined"]
    assert res["progress"]["best_version"] == first["version"]
    meta = session.provenance.get_meta(res["version"])
    event = [e for e in session.provenance.recent_events() if e["type"] == "fit"][-1]
    assert meta["is_new_best"] is False and event["is_new_best"] is False
    assert session.op_history[-1]["is_new_best"] is False


def test_a_fit_result_carries_only_the_progress_counters_that_move(session, monkeypatch):
    """PEtab-MCP#208: the constants are in the `make_changes` description, `criterion` is on the fit
    block, and the best's score and p-value are on the best's own result. `get_progress` is whole."""
    fits = iter([_fitsim(-50.0, 18.0), _fitsim(-50.0, 18.0)])

    def fit_numbers(n_starts=None):
        fs = {**next(fits), "n_starts": n_starts, "fitted_parameters": {}}
        return fs, fitting.compute_improvement(session, fs["neg_log_likelihood"], fitsim=fs)

    monkeypatch.setattr(session, "_auto_fit_and_improvement", fit_numbers)
    first = session.fit(n_starts=1)
    res = session.fit(n_starts=2)
    assert res["progress"] == {
        "n_fitted_ops": 2,
        "non_improving_streak": 1,
        "best_version": first["version"],
        "should_stop": False,
    }
    assert {"streak_cap", "plateau_window", "best_score"} <= set(session.progress_report())


def test_a_plausible_new_best_after_gaming_resets_the_streak(session):
    """Gamed fits a plausible best declines advance the streak; a genuine (plausible-sigma) new best
    still resets it, so fixing the noise model immediately clears the accumulated penalty."""
    session.record_fit_progress(1, _fitsim_plausible_sigma(-45.0, 18.0), {"is_new_best": True})
    for i, v in enumerate(range(2, 5)):
        session.record_fit_progress(
            v, _fitsim_inflated_sigma(-50.0 - i, 18.0), {"is_new_best": True}
        )
    assert session.progress_report()["non_improving_streak"] == 3  # gamed improvements accrued
    session.record_fit_progress(5, _fitsim_plausible_sigma(-60.0, 18.0), {"is_new_best": True})
    assert session.progress_report()["non_improving_streak"] == 0  # a real fix clears it


def test_flagged_residual_structure_blocks_the_adequate_plateau_stop(session):
    """A best model whose fixed-sigma chi2 GoF passes but whose residuals are structurally flagged
    is NOT adequate — its form is incomplete — so a flat-BIC session does NOT plateau-stop."""
    fs = _fitsim(-50.0, 18.0)
    fs["residual_flagged"] = True
    session.record_fit_progress(1, fs, {"is_new_best": True})
    for v in range(2, 2 + PLATEAU_WINDOW):
        session.record_fit_progress(v, fs, {"is_new_best": False})
    p = session.progress_report()
    assert 0.05 <= p["best_gof_pvalue"] <= 0.995  # the GoF alone would pass
    assert p["score_improvement_over_last_n"] == pytest.approx(0.0)  # BIC is flat...
    assert p["should_stop"] is False  # ...but residual structure blocks it


def test_flagged_residual_streak_cap_carries_structural_nudge(session):
    """When a residual-flagged session finally trips the streak cap, its stop_reason carries the
    structural nudge (change the FORM), its plausible estimated sigma notwithstanding."""
    fs = _fitsim_plausible_sigma(-50.0, 18.0)
    fs["residual_flagged"] = True
    session.record_fit_progress(1, fs, {"is_new_best": True})
    for v in range(2, 2 + NON_IMPROVING_STREAK_CAP):
        session.record_fit_progress(v, fs, {"is_new_best": False})
    p = session.progress_report()
    assert p["should_stop"] is True
    assert "structural" in p["stop_reason"].lower()


def test_collapsed_sigma_blocks_the_adequate_plateau_stop(session):
    """A COLLAPSED estimated sigma (overfitting the noise) is NOT adequate either — the agent gate
    must reject it like the sigma success gate does, so a session doesn't plateau-stop 'adequate'
    and then get rejected downstream for the same sigma."""
    session.record_fit_progress(1, _fitsim_collapsed_sigma(-50.0, 18.0), {"is_new_best": True})
    for v in range(2, 2 + PLATEAU_WINDOW):
        session.record_fit_progress(v, _fitsim_collapsed_sigma(-50.0, 18.0), {"is_new_best": False})
    p = session.progress_report()
    assert session.best.sigma_estimated is True and session.best.sigma_collapsed is True
    assert session.best.sigma_inflated is False
    assert p["should_stop"] is False


def test_collapsed_sigma_streak_cap_reason_carries_sigma_advice(session):
    """A collapsed-sigma session that trips the streak cap gets the structural nudge + the collapsed
    (overfitting-the-noise) warning."""
    session.record_fit_progress(1, _fitsim_collapsed_sigma(-50.0, 18.0), {"is_new_best": True})
    for v in range(2, 2 + NON_IMPROVING_STREAK_CAP):
        session.record_fit_progress(v, _fitsim_collapsed_sigma(-50.0, 18.0), {"is_new_best": False})
    reason = session.progress_report()["stop_reason"].lower()
    assert "structural" in reason and "collapsed" in reason and "overfitting" in reason


def test_steadily_improving_session_never_stops_early(session):
    """A session that keeps landing real new bests keeps the streak at zero and the BIC falling, so
    neither the cap nor the plateau ever fires."""
    for i, v in enumerate(range(1, 10)):
        # each op a genuine new best: nll keeps dropping, so BIC keeps falling well past the gate
        session.record_fit_progress(v, _fitsim(-50.0 - 5.0 * i, 18.0), {"is_new_best": True})

    p = session.progress_report()
    assert p["non_improving_streak"] == 0
    assert p["best_version"] == 9  # the last (lowest-BIC) op
    assert p["score_improvement_over_last_n"] > 0
    assert p["should_stop"] is False


def test_checkpoint_holds_the_lowest_bic_after_wandering_worse(session):
    """After the best op, a worse op must NOT move the checkpoint (the wander-off-best case)."""
    session.record_fit_progress(1, _fitsim(-80.0, 18.0), {"is_new_best": True})  # the best
    session.record_fit_progress(2, _fitsim(-40.0, 18.0), {"is_new_best": False})  # worse
    p = session.progress_report()
    assert p["best_version"] == 1
    assert p["non_improving_streak"] == 1


# --- (d) structural-inadequacy nudge folded into the stop reason (issue #82) --------------
def test_streak_cap_stop_reason_carries_structural_nudge_when_inadequate(session):
    """A never-adequate session that hits the streak cap is the structural-inadequacy case: its
    stop_reason must tell the agent to change the model's FORM, not re-fit the same structure."""
    session.record_fit_progress(1, _fitsim(-50.0, 500.0), {"is_new_best": True})  # never adequate
    for v in range(2, 2 + NON_IMPROVING_STREAK_CAP):
        session.record_fit_progress(v, _fitsim(-50.0, 500.0), {"is_new_best": False})

    reason = session.progress_report()["stop_reason"].lower()
    assert "streak" in reason
    assert "structural" in reason and "form" in reason
    assert "make_changes" in reason


def test_a_missing_fixed_sigma_gof_adds_no_structural_nudge(session):
    """A fixed-sigma best whose GoF could not be computed (no chi2) has a finite score but no test
    that failed, so its streak-cap stop claims neither adequacy nor structural inadequacy."""
    session.record_fit_progress(1, _fitsim(-50.0, None), {"is_new_best": True})
    for v in range(2, 2 + NON_IMPROVING_STREAK_CAP):
        session.record_fit_progress(v, _fitsim(-50.0, None), {"is_new_best": False})
    p = session.progress_report()
    assert p["best_version"] == 1 and p["best_gof_pvalue"] is None
    assert p["stop_reason"].startswith("non-improving-streak cap reached")
    assert "STRUCTURAL" not in p["stop_reason"] and "adequate" not in p["stop_reason"]


def test_adequate_plateau_stop_reason_omits_structural_nudge(session):
    """An adequate-and-flat plateau is a SUCCESS stop, so its stop_reason must NOT append the
    structural nudge — telling an already-adequate model it 'cannot fit the data' is contradictory.
    """
    session.record_fit_progress(1, _fitsim(-50.0, 18.0), {"is_new_best": True})  # adequate
    for v in range(2, 2 + PLATEAU_WINDOW):
        session.record_fit_progress(v, _fitsim(-50.0, 18.0), {"is_new_best": False})

    reason = session.progress_report()["stop_reason"].lower()
    assert "plateau" in reason and "adequate" in reason
    # #196: the reason names the check that made the model adequate
    assert "chi2 gof p=" in reason and "fixed noise sigmas" in reason
    assert "structural" not in reason  # no nudge on a success stop
    assert "cannot fit" not in reason


def test_running_report_has_no_stop_reason_and_no_nudge(session):
    """While the session is still making progress there is no stop and hence no nudge text."""
    for i, v in enumerate(range(1, 6)):
        session.record_fit_progress(v, _fitsim(-50.0 - 5.0 * i, 18.0), {"is_new_best": True})
    p = session.progress_report()
    assert p["should_stop"] is False
    assert p["stop_reason"] is None


# --- (e) non-finite scores must never become the checkpoint --------------------------------
# Regression for a session whose FIRST fit returned a NaN neg-log-likelihood: NaN
# is a float, so it passed the old `bic is not None` guard, became best_score, and then locked
# the session out for good — every later `bic <= NaN - delta` is False, so no fit could ever be a
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
    nan_fs = _fitsim(float("nan"), 145.98, n=21, k=10)
    # ...and is not stamped a new best either, which `best_version` would contradict (#141).
    assert fitting.compute_improvement(session, float("nan"), fitsim=nan_fs)["is_new_best"] is False
    session.record_fit_progress(1, nan_fs, {"is_new_best": True})
    p = session.progress_report()
    assert p["best_version"] is None
    assert p["best_score"] is None
    assert p["non_improving_streak"] == 1  # an unscoreable fit is not progress

    # The next real fit is a legitimate new best and claims the checkpoint.
    fs = _fitsim(266.62, 100.0, n=21, k=10)
    imp = fitting.compute_improvement(session, nllh=266.62, objective_before=8377.0, fitsim=fs)
    assert imp["is_new_best"] is True
    session.record_fit_progress(2, fs, imp)
    p = session.progress_report()
    assert p["best_version"] == 2
    assert p["best_score"] == pytest.approx(fitting.goodness_of_fit(266.62, 100.0, 10, 21)["bic"])
    assert p["non_improving_streak"] == 0


def test_improving_fits_are_never_locked_out_by_a_poisoned_checkpoint(session):
    """A NaN can never become the checkpoint, and fits after an unscoreable one still register.

    This is the exact shape of the lost session — an unscoreable fit followed by fits at BIC 580,
    569 and 519 — none of which could register under the old comparison. The NaN is now refused
    by `Checkpoint` itself, so the poisoned state is unconstructible rather than survivable.
    """
    with pytest.raises(ValueError, match="real number"):
        _checkpoint(score=float("nan"), version=1, nll=float("nan"))
    # An unscoreable fit leaves no checkpoint at all, which is the state the session recovers from.
    session.best = None
    for bic_target in (580.97, 569.19, 519.86):
        # BIC = k·ln(n) + 2·nll  =>  nll = (bic - k·ln(n)) / 2
        import math

        nll = (bic_target - 10 * math.log(21)) / 2
        fs = _fitsim(nll, 100.0, n=21, k=10)
        imp = fitting.compute_improvement(session, nllh=nll, objective_before=8377.0, fitsim=fs)
        assert imp["is_new_best"] is True, f"BIC {bic_target} must be able to beat a NaN"
        session.record_fit_progress(2, fs, imp)
        assert session.best.score == pytest.approx(bic_target)


def test_stall_without_any_checkpoint_reports_an_evaluation_failure(session):
    """A stall where nothing ever scored is an EVALUATION problem: the stop_reason must say so
    instead of blaming the model's structure and sending the agent re-authoring."""
    for v in range(1, 2 + NON_IMPROVING_STREAK_CAP):
        session.record_fit_progress(
            v, _fitsim(float("nan"), 145.98, n=21, k=10), {"is_new_best": True}
        )
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
    adequate (chi2/n = 0.016, p = 0.9999999), letting an overfit session plateau-stop."""
    # chi2 << dof -> p -> 1
    fs = _fitsim_fixed_sigma(0.3)
    session.record_fit_progress(1, fs, {"is_new_best": True})
    for v in range(2, 2 + PLATEAU_WINDOW):
        session.record_fit_progress(v, fs, {"is_new_best": False})
    p = session.progress_report()
    assert session.best.sigma_estimated is False
    assert p["best_gof_pvalue"] > 0.99  # would pass a one-sided `>= 0.05` test
    assert p["should_stop"] is False  # ...but the two-sided gate blocks the stop


def test_fixed_sigma_well_calibrated_fit_is_still_adequate(session):
    """No regression: chi2 near its dof stays adequate and plateau-stops as before."""
    fs = _fitsim_fixed_sigma(18.0)  # chi2 ~ dof = 18 -> p comfortably mid-range
    session.record_fit_progress(1, fs, {"is_new_best": True})
    for v in range(2, 2 + PLATEAU_WINDOW):
        session.record_fit_progress(v, fs, {"is_new_best": False})
    p = session.progress_report()
    assert 0.05 <= p["best_gof_pvalue"] <= 0.995
    assert p["should_stop"] is True and "plateau" in p["stop_reason"].lower()


# --- (d) NO fitted-op budget: the streak cap is the only progress-based bound ------------------
def test_a_session_that_keeps_finding_new_bests_is_not_cut_off_by_an_op_budget(session):
    """A 20-fitted-op cap was tried and removed. It was not redundant with the streak cap -- that
    resets on every new best, so 73 of 333 clean sessions passed 20 fits anyway -- but it cost
    ~1.8% of passes, charged against exactly the agents that explore longest. Pins that a long,
    still-improving session runs on.
    """
    nll = -50.0
    for v in range(1, 31):  # well past the old cap of 20
        nll -= NEW_BEST_MIN_DELTA_SCORE
        session.record_fit_progress(v, _fitsim(nll, 500.0), {"is_new_best": True})
    p = session.progress_report()
    assert p["n_fitted_ops"] == 30
    assert p["non_improving_streak"] == 0
    assert p["should_stop"] is False  # still improving -> keep going
    assert p["stop_reason"] is None
    assert "fitted_ops_cap" not in p  # the knob is gone from the contract too


def test_a_stalled_long_session_is_still_stopped_by_the_streak_cap(session):
    """Removing the budget must not remove the bound that matters: a session that stops improving is
    still caught, and still gets the structural diagnosis rather than a budget notice."""
    nll = -50.0
    for v in range(1, 11):
        nll -= NEW_BEST_MIN_DELTA_SCORE
        session.record_fit_progress(v, _fitsim(nll, 500.0), {"is_new_best": True})
    for v in range(11, 11 + NON_IMPROVING_STREAK_CAP):
        session.record_fit_progress(v, _fitsim(nll, 500.0), {"is_new_best": False})
    p = session.progress_report()
    assert p["should_stop"] is True
    assert "streak" in p["stop_reason"].lower()
    assert "budget" not in p["stop_reason"].lower()
    assert "STRUCTURAL" in p["stop_reason"]


# --- (e) a scale-coupled sigma has no cross-model meaning ------------------------------------
@pytest.mark.parametrize("gain_estimated,expected", [(True, True), (False, False)])
def test_only_an_estimated_co_factor_breaks_comparability(gain_estimated, expected):
    """Only an ESTIMATED co-factor breaks the ratio. A FIXED one is a declared constant of the
    problem, identical on both sides, so it cancels and the reference basis stays valid.

    Built as a real problem whose noise formula multiplies a per-gel scale by a shared sigma,
    rather than probed on the toy session: the toy sigma has no composite noiseFormula at all, so
    `sigma_scale_coupled` returned False at the `len(parts) < 2` guard and never reached the
    `is_parameter_estimated` branch. Inverting that check would have left such a test passing
    while silently re-flagging every problem of that shape.
    """
    import pandas as pd
    from petab.v2.C import (
        ESTIMATE,
        EXPERIMENT_ID,
        LOWER_BOUND,
        MEASUREMENT,
        NOISE_FORMULA,
        NOISE_PARAMETERS,
        NOISE_PLACEHOLDERS,
        NOMINAL_VALUE,
        OBSERVABLE_FORMULA,
        OBSERVABLE_ID,
        OBSERVABLE_PLACEHOLDERS,
        PARAMETER_ID,
        TIME,
        UPPER_BOUND,
    )
    from petab.v2.models.sbml_model import SbmlModel
    from petab_mcp import reconcile as R
    from petab_mcp.problem import PARAMETER_SCALE, WorkingProblem

    ant = (
        "model g\n  compartment c = 1;\n  species A in c;\n  A = 1.0;\n"
        "  kf = 0.5;\n  r: A => ; kf * A;\nend\n"
    )
    prob = WorkingProblem(model=SbmlModel.from_antimony(ant))
    prob = prob.replace(
        observable_df=pd.DataFrame(
            {
                OBSERVABLE_ID: ["A_obs"],
                OBSERVABLE_FORMULA: ["A * observableParameter1_A_obs"],
                OBSERVABLE_PLACEHOLDERS: ["observableParameter1_A_obs"],
                NOISE_FORMULA: ["noiseParameter1_A_obs * noiseParameter2_A_obs"],
                NOISE_PLACEHOLDERS: ["noiseParameter1_A_obs;noiseParameter2_A_obs"],
            }
        ).set_index(OBSERVABLE_ID)
    )
    prob = prob.replace(
        measurement_df=pd.DataFrame(
            {
                OBSERVABLE_ID: ["A_obs"] * 4,
                EXPERIMENT_ID: [""] * 4,
                TIME: [0.0, 1.0, 2.0, 3.0],
                MEASUREMENT: [1.0, 0.6, 0.4, 0.2],
                # That shape exactly: a per-gel gain paired with the shared sigma.
                NOISE_PARAMETERS: ["s_gel;sigma_A"] * 4,
            }
        )
    )
    prob.parameter_df = pd.DataFrame(
        {
            PARAMETER_ID: ["s_gel", "sigma_A", "kf"],
            PARAMETER_SCALE: ["log10"] * 3,
            LOWER_BOUND: [1e-5] * 3,
            UPPER_BOUND: [1e3] * 3,
            NOMINAL_VALUE: [100.0, 0.001, 0.5],
            ESTIMATE: [gain_estimated, True, True],
        }
    ).set_index(PARAMETER_ID)

    assert R.sigma_scale_coupled(prob, "sigma_A") is expected
    # The gain itself is never the sigma under test, either way.
    assert R.sigma_scale_coupled(prob, "kf") is False


def test_an_unscorable_session_is_diagnosed_by_the_streak_cap_not_the_budget(session):
    """A session that never produces a usable score reaches the STREAK cap first, and must get the
    unscored-stall diagnosis rather than a budget notice.

    Pins the ordering invariant: `record_fit_progress` advances the non-improving streak on every
    non-finite score (regardless of `is_new_best`), so such a session trips the cap at op 15 and the
    op budget is unreachable for it. That is why the budget branch's `best is None` wording is
    documented as defensive. If this test ever fails because the budget claimed the stop, the
    "no best version to keep" wording has become live and needs its own coverage.
    """
    for v in range(1, NON_IMPROVING_STREAK_CAP + 2):
        session.record_fit_progress(
            v,
            {
                "status": "fit",
                "neg_log_likelihood": None,
                "chi2": None,
                "n_data_points": 20,
                "n_estimated": 2,
            },
            {"is_new_best": True},
        )

    p = session.progress_report()
    assert p["best_version"] is None
    assert p["should_stop"] is True
    reason = p["stop_reason"]
    assert "streak" in reason.lower() and "budget" not in reason.lower()
    # The actionable finding — the fits are failing to EVALUATE, not the form failing to fit.
    assert "No fit in this session produced a" in reason
    assert "best scored version is kept" not in reason


@pytest.mark.parametrize(
    "fits, best",
    [
        # A lower-scoring fit with an inflated sigma is refused over a plausible best.
        (
            [
                _fitsim_plausible_sigma(-50.0, 18.0),
                _fitsim_inflated_sigma(-80.0, 18.0),
                _fitsim_plausible_sigma(-50.0 - NEW_BEST_MIN_DELTA_SCORE / 4, 18.0),
            ],
            0,
        ),
        # With no plausible best to protect, a gamed new best is adopted, collapsed or inflated.
        (
            [
                _fitsim_collapsed_sigma(-50.0, 18.0),
                _fitsim_inflated_sigma(-80.0, 18.0),
                _fitsim_plausible_sigma(-80.0 - NEW_BEST_MIN_DELTA_SCORE / 4, 18.0),
            ],
            1,
        ),
    ],
    ids=["gamed-refused", "gamed-adopted"],
)
def test_a_resume_seeds_the_checkpoint_the_live_run_kept(
    session, yaml_path, monkeypatch, fits, best
):
    """A restart on the same store rebuilds the best-score checkpoint the live run kept, not the
    lowest score in the log (#9). Driven through `fit` (the optimizer stubbed) so the events are
    the ones `fit` writes; the last fit of each script is plausible but short of the new-best bar.
    Each fit prices a hardcoded constant, so the persisted gof p-value must use the same k as the
    live checkpoint's."""
    from petab_mcp.references import StubReferenceResolver
    from petab_mcp.session import PetabSession

    script = iter(fits)
    reports = []

    def _fit(n_starts=None):
        fs = {**next(script), "hardcoded_constants": ["0.0987"], "n_starts": n_starts}
        reports.append(fs.pop("noise_sigma_report"))
        return fs, fitting.compute_improvement(session, fs["neg_log_likelihood"], fitsim=fs)

    monkeypatch.setattr(session, "_auto_fit_and_improvement", _fit)
    monkeypatch.setattr(fitting, "noise_sigma_report", lambda problem, fitted: reports[-1])
    # Three fits of one structure is the de-dup's cap (dedup.MAX_IDENTICAL_REAUTHORS); each runs
    # more starts than the last, or the de-dup skips it as a refit.
    versions = [session.fit(n_starts=n)["version"] for n in (1, 2, 3)]
    assert session.best.version == versions[best]

    resumed = PetabSession(
        yaml_path, provenance_dir=str(session.provenance.root), resolver=StubReferenceResolver()
    )
    assert resumed.best == session.best
