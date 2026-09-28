"""The complexity criterion: the named penalties and the wiring that makes the model-building
loop rank by whichever one is active.

The arithmetic is checked against the textbook formulae written out independently here rather than
against :mod:`petab_mcp.criteria`'s own helpers, so a sign slip or a factor of two cannot be
self-consistent. The wiring tests are the other half and the load-bearing ones: every gate in the
loop is supposed to read ONE score, and a criterion nothing consults would pass every arithmetic
test while changing nothing about what a run prefers.
"""

from __future__ import annotations

import math

import pytest

from petab_mcp import criteria, fitting
from petab_mcp.criteria import Criterion


# -- the penalties themselves ----------------------------------------------
def test_the_default_is_plain_bic():
    """BIC, and nothing else, unless the environment says otherwise."""
    assert criteria.DEFAULT_CRITERION == "bic"
    assert Criterion() == Criterion("bic")
    assert Criterion().name == "bic"


@pytest.mark.parametrize(
    ("name", "penalty"),
    [
        ("bic", lambda k, n, k_eff: k * math.log(n)),
        ("caic", lambda k, n, k_eff: k * (math.log(n) + 1.0)),
        ("aic", lambda k, n, k_eff: 2.0 * k),
        ("aicc", lambda k, n, k_eff: 2.0 * k + 2.0 * k * (k + 1) / (n - k - 1)),
        ("hqc", lambda k, n, k_eff: 2.0 * k * math.log(math.log(n))),
        ("tic", lambda k, n, k_eff: 2.0 * k_eff),
    ],
)
def test_each_criterion_is_two_nll_plus_its_textbook_penalty(name, penalty):
    nll, k, n, k_eff = 12.5, 4, 60, 3.3
    assert Criterion(name).value(nll, k, n, k_eff) == pytest.approx(
        2.0 * nll + penalty(k, n, k_eff)
    )


def test_caic_is_bic_plus_one_unit_per_parameter():
    """The stricter-than-BIC arm, for where BIC still buys structure the data cannot support."""
    assert Criterion("caic").penalty(5, 200) == pytest.approx(Criterion("bic").penalty(5, 200) + 5)


def test_tic_charges_the_effective_count_and_has_no_value_without_one():
    """TIC reads the FIT, not its dimensions: with k_eff it is AIC with k replaced, without one it
    is a MISSING score — never a fall-back to k, which would silently turn it into AIC."""
    assert Criterion("tic").value(10.0, 5, 100, 3.5) == pytest.approx(2 * 10.0 + 2 * 3.5)
    assert Criterion("tic").value(10.0, 5, 100) is None
    assert Criterion("tic").value(10.0, 5, 100, float("nan")) is None
    # ...and every other criterion ignores it.
    for name in ("bic", "caic", "aic", "aicc", "hqc"):
        assert Criterion(name).value(10.0, 5, 100, 3.5) == Criterion(name).value(10.0, 5, 100)


def test_the_penalties_are_ordered_aic_below_hqc_below_bic_only_above_e_to_the_e():
    """Not decoration: it is WHY the choice matters. The same fit is ranked differently, so a run
    under AIC buys structure a run under BIC refuses.

    But the ordering is NOT global. HQC charges 2*ln(ln(n)) per parameter, which crosses AIC's
    flat 2 at n = e**e ~ 15.2: BELOW that HQC is the LOOSEST of the three, not the middle one.
    Both regimes are pinned here because a PEtab design with ten measurements is ordinary, and a
    reader picking a criterion off the module docstring has to be told which side they are on."""
    k = 5
    aic, hqc, bic = (Criterion(c).penalty(k, 200) for c in ("aic", "hqc", "bic"))
    assert aic < hqc < bic
    # ...and the other side of the crossover, where HQC is looser than AIC.
    aic, hqc, bic = (Criterion(c).penalty(k, 10) for c in ("aic", "hqc", "bic"))
    assert hqc < aic < bic


def test_aicc_exceeds_aic_and_converges_to_it_as_data_grows():
    k = 4
    assert Criterion("aicc").penalty(k, 12) > Criterion("aic").penalty(k, 12)
    assert Criterion("aicc").penalty(k, 100_000) == pytest.approx(
        Criterion("aic").penalty(k, 100_000), rel=1e-3
    )


def test_aicc_is_undefined_when_the_data_cannot_support_the_parameters():
    """``n - k - 1 <= 0`` is not a large correction but no correction: AICc diverges and then
    flips sign there, so it must read as a MISSING score — the same ``None`` every other absent
    score uses — rather than as a suspiciously good one."""
    assert Criterion("aicc").value(10.0, 9, 10) is None  # n - k - 1 == 0
    assert Criterion("aicc").value(10.0, 12, 10) is None  # n - k - 1 < 0
    assert Criterion("aicc").value(10.0, 8, 10) is not None


def test_hqc_is_undefined_where_its_penalty_would_pay_for_parameters():
    """``ln(ln(n))`` is negative below e, so HQC would REWARD complexity on a 2-point problem."""
    assert Criterion("hqc").value(10.0, 2, 2) is None
    assert Criterion("hqc").penalty(2, 3) > 0


def test_a_non_finite_likelihood_has_no_score_under_any_criterion():
    """A NaN is a float and passes every ``is not None`` guard, and because NaN compares False
    against everything, one adopted as the best-score checkpoint locks out every later fit."""
    for name in criteria.PENALTIES:
        assert Criterion(name).value(float("nan"), 3, 100) is None
        assert Criterion(name).value(float("inf"), 3, 100) is None


def test_missing_or_impossible_inputs_have_no_score():
    assert Criterion().value(None, 3, 100) is None
    assert Criterion().value(10.0, None, 100) is None
    assert Criterion().value(10.0, 3, None) is None
    assert Criterion().value(10.0, 3, 0) is None
    assert Criterion().value(10.0, -1, 100) is None


# -- naming ----------------------------------------------------------------
def test_an_unknown_criterion_names_the_ones_that_exist():
    with pytest.raises(ValueError, match="unknown complexity criterion"):
        Criterion("aicx")
    with pytest.raises(ValueError, match="aic, aicc, bic, caic, hqc, tic"):
        Criterion("bayes")


def test_the_name_is_normalised_so_a_config_value_need_not_be_exact():
    assert Criterion(" AICc ").name == "aicc"


# -- the price of one parameter --------------------------------------------
def test_the_parameter_price_is_half_the_penalty_step():
    """What a merge has to beat: the score charges ``2*nll``, so a parameter pays for itself
    exactly when it lowers nll by more than half the penalty it adds."""
    n = 500
    assert Criterion("bic").parameter_price(4, n) == pytest.approx(math.log(n) / 2.0)
    assert Criterion("aic").parameter_price(4, n) == pytest.approx(1.0)
    assert Criterion("hqc").parameter_price(4, n) == pytest.approx(math.log(math.log(n)))


def test_the_aicc_price_rises_with_the_parameters_already_spent():
    """The one criterion whose price is not a constant in ``k`` — which is why the price is
    derived from the penalty step rather than hard-coded as ``ln(n)/2``."""
    price = Criterion("aicc").parameter_price
    assert price(2, 40) < price(8, 40)


def test_there_is_no_price_for_a_parameter_that_is_not_there():
    assert Criterion().parameter_price(0, 100) is None
    assert Criterion().parameter_price(None, 100) is None


def test_tic_prices_no_merge():
    """Deliberately None, not AIC's 1: what a deleted parameter takes off k_eff is only known after
    the refit, and for the parameter a merge hint targets — one the data cannot separate from its
    partner — it is about zero, because TIC never charged for it."""
    assert Criterion("tic").parameter_price(4, 500) is None


# -- reading a stored score back -------------------------------------------
def test_a_stored_fit_is_read_by_criterion_name():
    record = {"bic": 5.0, "aic": 3.0, "aicc": 4.0, "hqc": 3.5}
    assert Criterion("bic").score_of(record) == 5.0
    assert Criterion("aicc").score_of(record) == 4.0


def test_a_non_finite_stored_score_is_no_score():
    assert Criterion().score_of({"bic": float("nan")}) is None
    assert Criterion().score_of({}) is None


# -- the reported score table ----------------------------------------------
def test_every_criterion_rides_on_a_fit_beside_the_one_being_ranked():
    """All four, whichever is active — so a finished run can be re-read under a criterion it was
    not run with."""
    scores = fitting.goodness_of_fit(10.0, 18.0, 3, 100, criterion=Criterion("aicc"), k_eff=2.5)
    assert set(criteria.PENALTIES) <= set(scores)
    assert scores["bic"] == pytest.approx(2 * 10.0 + 3 * math.log(100))
    assert scores["tic"] == pytest.approx(2 * 10.0 + 2 * 2.5)
    assert "k_eff" not in scores, "TIC's input is for the harness, not a number the agent sees"
    assert scores["criterion"] == "aicc"
    assert scores["criterion_value"] == pytest.approx(scores["aicc"])


def test_a_fit_without_an_effective_count_has_no_tic_but_every_other_score():
    scores = fitting.goodness_of_fit(10.0, 18.0, 3, 100)
    assert scores["tic"] is None and "k_eff" not in scores
    assert scores["bic"] is not None


def test_an_unscoreable_fit_still_names_the_criterion_it_would_have_been_ranked_by():
    """The early return must not drop the name: a reader that sees no ``criterion`` cannot tell
    "this run ranks by aicc and this fit failed" from "no criterion was configured"."""
    scores = fitting.goodness_of_fit(None, None, 3, 100, criterion=Criterion("aicc"))
    assert scores["criterion"] == "aicc"
    assert scores["criterion_value"] is None
    assert all(scores[name] is None for name in criteria.PENALTIES)


def test_a_bare_score_read_outside_a_session_is_plain_bic():
    """No session, no criterion argument: the constant default, never something a previous caller
    left behind — there is nothing to leave behind."""
    assert fitting.goodness_of_fit(10.0, None, 3, 100)["criterion"] == "bic"


# -- the wiring: what the loop actually ranks by ---------------------------
def _fitsim(nll: float, *, n: int = 40, k: int = 3, k_eff: float | None = 2.5) -> dict:
    return {
        "status": "fit",
        "neg_log_likelihood": nll,
        "chi2": float(n),
        "n_data_points": n,
        "n_estimated": k,
        "k_eff": k_eff,
        "fitted_parameters": {},
    }


def test_is_new_best_is_decided_by_the_active_criterion_not_by_bic(session):
    """The point of the knob. Under AIC a parameter costs 2; under BIC on 40 points it costs
    ln(40) = 3.69. A fit is constructed whose nll gain lands between the two, so the SAME pair of
    fits is an improvement under AIC and a regression under BIC — and ``is_new_best`` has to
    disagree accordingly, or the criterion is decoration.
    """
    n = 40
    # A 3-parameter reference, then a 4-parameter candidate that buys 1.4 nll for its parameter.
    # AIC: 2*1.4 = 2.8 gained vs 2 charged -> better. BIC: 2.8 vs ln(40) = 3.69 -> worse.
    ref, cand = _fitsim(100.0, n=n, k=3), _fitsim(98.6, n=n, k=4)

    verdicts = {}
    for name in ("aic", "bic"):
        session.criterion = Criterion(name)
        session.best = None
        first = fitting.compute_improvement(session, nllh=100.0, fitsim=ref)
        session.record_fit_progress(1, ref, first)
        second = fitting.compute_improvement(session, nllh=98.6, fitsim=cand)
        verdicts[name] = second["improvement_vs_best_prior"]

    assert verdicts["aic"] > 0 > verdicts["bic"], verdicts


def test_the_checkpoint_and_the_progress_report_agree_on_the_active_criterion(session):
    """One score, read in one place: the checkpoint the judge restores, the value the progress
    report shows, and the name naming it must all come from the same criterion."""
    session.criterion = Criterion("aicc")
    fs = _fitsim(100.0)
    imp = fitting.compute_improvement(session, nllh=100.0, fitsim=fs)
    session.record_fit_progress(1, fs, imp)

    expected = Criterion("aicc").value(100.0, 3, 40)
    assert session.best.score == pytest.approx(expected)
    progress = session.progress_report()
    assert progress["criterion"] == "aicc"
    assert progress["best_score"] == pytest.approx(expected)


def test_a_fit_records_the_ranked_pair_so_a_resume_can_pick_it_up(session):
    """A run resumed under the criterion it was written with recovers its checkpoint from the
    `criterion`/`criterion_value` pair the fit event carries."""
    fs = _fitsim(100.0)
    imp = fitting.compute_improvement(session, nllh=100.0, fitsim=fs)
    assert imp["criterion"] == "bic"
    assert Criterion("bic").score_of(imp) == pytest.approx(imp["criterion_value"])


def test_a_snapshot_stores_every_criterion_in_the_version_meta(session):
    """Exercised through `_snapshot_version` itself, not by assembling the record here: the meta
    is where `dedup.build_registry` reads the score from, so under a non-default criterion the
    whole de-dup ranking rests on this fill loop having run. An assertion built from
    `Criterion.values()` would be tautological — it would pass with the loop deleted, while the
    registry silently lost every score and the worse-family warning went quiet."""
    from petab_mcp import dedup

    session.criterion = Criterion("aicc")
    version = session._snapshot_version(meta={}, sim=_fitsim(100.0))
    meta = session.provenance.get_meta(version)

    assert meta["criterion"] == "aicc"
    for name in criteria.PENALTIES:
        assert meta[name] == pytest.approx(Criterion(name).value(100.0, 3, 40, 2.5)), name
    assert "k_eff" not in meta, "list_versions returns the meta; the count is for the harness"
    assert meta["criterion_value"] == pytest.approx(meta["aicc"])
    # ...and that is exactly the number the de-dup registry ranks structure families on.
    entry = next(e for e in dedup.build_registry(session) if e["version"] == version)
    assert entry["score"] == pytest.approx(meta["aicc"])


def test_a_snapshot_keeps_the_scores_the_fit_already_computed(session):
    """The fill loop must not overwrite what `compute_improvement` put in the meta: it derives
    `k` independently (from the problem, when the payload carries none), so a recompute that won
    would let the stored score disagree with the one the run ranked on."""
    fs = _fitsim(100.0)
    imp = fitting.compute_improvement(session, nllh=100.0, fitsim=fs)
    version = session._snapshot_version(meta={**imp, "bic": -1.0}, sim=fs)
    assert session.provenance.get_meta(version)["bic"] == -1.0


def test_the_merge_price_follows_the_criterion(monkeypatch):
    """A tie is worth taking when it costs less nll than the criterion charges for the parameter
    it deletes — so the SAME tie is a win under BIC and a loss under AIC, whose parameter is
    cheaper. The hint has to move with the knob or it prices against a penalty nobody is using.
    """
    import numpy as np

    from petab_mcp import identifiability as I

    # Two parameters the data cannot separate: an identical, weakly-determined pair.
    fim = np.array([[1.0, 0.999], [0.999, 1.0]])
    theta = np.array([0.0, 1.4])
    kwargs = dict(
        names=["a", "b"],
        theta=theta,
        n_data=500,
        roles={"a": "rate", "b": "rate"},
        scales=["log10", "log10"],
        at_bound=[False, False],
    )

    under_bic = I.merge_candidates(fim, criterion=Criterion("bic"), **kwargs)
    under_aic = I.merge_candidates(fim, criterion=Criterion("aic"), **kwargs)

    assert under_bic["criterion"] == "bic" and under_aic["criterion"] == "aic"
    assert under_bic["parameter_price_nll"] == pytest.approx(math.log(500) / 2.0, abs=1e-3)
    assert under_aic["parameter_price_nll"] == pytest.approx(1.0)
    # The cheaper AIC parameter prices the same tie worse: fewer merges clear the bar.
    assert under_aic["n_merges_improving_score"] <= under_bic["n_merges_improving_score"]
    assert under_bic["ties"][0]["dscore"] < under_aic["ties"][0]["dscore"]


# -- configuration ---------------------------------------------------------
def test_the_criterion_comes_from_the_environment(monkeypatch):
    monkeypatch.setenv("PETAB_MCP_CRITERION", "hqc")
    assert criteria.from_env() == Criterion("hqc")


def test_an_unusable_environment_value_falls_back_to_bic_rather_than_failing_import(monkeypatch):
    """A typo in a sweep's env must not take the server down — and the resolved criterion rides
    on the very next fit result, so the fallback is visible immediately."""
    for bad in ("nonsense", "", "   ", "0.5*bic"):
        monkeypatch.setenv("PETAB_MCP_CRITERION", bad)
        assert criteria.from_env() == Criterion(), bad


def test_resolve_takes_the_argument_else_the_environment_else_bic(monkeypatch):
    """What `PetabSession.__init__` calls, once."""
    monkeypatch.delenv("PETAB_MCP_CRITERION", raising=False)
    assert criteria.resolve(None) == Criterion("bic")
    assert criteria.resolve("aicc") == Criterion("aicc")
    monkeypatch.setenv("PETAB_MCP_CRITERION", "hqc")
    assert criteria.resolve(None) == Criterion("hqc")
    assert criteria.resolve("aicc") == Criterion("aicc"), "the argument wins over the environment"
    crit = Criterion("caic")
    assert criteria.resolve(crit) is crit


def test_resolve_refuses_an_unusable_explicit_criterion():
    """An explicit argument is a mistake to report at startup, unlike a typo'd env var."""
    with pytest.raises(ValueError):
        criteria.resolve("aicx")


def test_each_session_holds_its_own_criterion(yaml_path, tmp_path, monkeypatch):
    """The criterion is SESSION state, not module state: two sessions in one process rank by
    different criteria, and building one never changes what another scores by
    (:mod:`petab_mcp.criteria`)."""
    from petab_mcp.session import PetabSession

    monkeypatch.delenv("PETAB_MCP_CRITERION", raising=False)
    a = PetabSession(yaml_path, provenance_dir=str(tmp_path / "a"), criterion="aicc")
    b = PetabSession(yaml_path, provenance_dir=str(tmp_path / "b"))
    assert a.criterion == Criterion("aicc")
    assert b.criterion == Criterion("bic"), "session B picked up session A's criterion"
    # ...and each one scores by its own.
    assert fitting.compute_improvement(a, nllh=100.0, fitsim=_fitsim(100.0))["criterion"] == "aicc"
    assert fitting.compute_improvement(b, nllh=100.0, fitsim=_fitsim(100.0))["criterion"] == "bic"


@pytest.mark.parametrize("name", ["bic", "caic", "aic", "aicc", "hqc", "tic", " AICc "])
def test_a_reported_name_round_trips_through_from_name(name):
    """The name is the IDENTITY of a scoring rule, and the harness judge reconstructs the rule
    from the one the server REPORTS — so it has to parse back."""
    parsed = Criterion.from_name(name)
    assert parsed is not None and parsed.name == name.strip().lower()


@pytest.mark.parametrize("name", ["", "   ", "nonsense", "0.5*bic", None, 7])
def test_an_unrecognised_name_is_none_rather_than_a_guess(name):
    """`None`, so a caller falls back to a criterion it names itself instead of scoring under a
    different one. A weighted label such as `0.5*bic` is not a name `Criterion` can express."""
    assert Criterion.from_name(name) is None


def test_the_cli_accepts_the_criterion_flags():
    from petab_mcp.cli import build_parser

    args = build_parser().parse_args(["--petab-yaml", "p.yaml", "--criterion", "aicc"])
    assert args.criterion == "aicc"


def test_an_unpassed_cli_flag_leaves_the_resolution_to_the_session(monkeypatch):
    """The CLI deliberately does NOT resolve the environment itself: it passes `None` through so
    the session's `criteria.resolve` is the single point that reads the criterion env, which is
    what keeps several reads of the same variable from disagreeing."""
    monkeypatch.setenv("PETAB_MCP_CRITERION", "hqc")
    from petab_mcp.cli import build_parser

    args = build_parser().parse_args(["--petab-yaml", "p.yaml"])
    assert args.criterion is None
    # ...and passing that straight through, as `main` does, lands on the environment's criterion.
    assert criteria.resolve(args.criterion) == Criterion("hqc")
    monkeypatch.delenv("PETAB_MCP_CRITERION")
    assert criteria.resolve(None) == Criterion("bic")
