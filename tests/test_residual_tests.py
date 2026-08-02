"""Unit tests for the sign / Wald–Wolfowitz / replicate-lack-of-fit residual-structure tests.

Pure numeric code over residual lists, so no AMICI/pyPESTO fit is needed. Where an INDEPENDENT
reference exists (scipy for the sign test's binomial tail, a runs-test PMF that must sum to 1), the
test pins to it; the exact runs-test p-values are checked against hand-worked small cases, the
Student-t tail against closed forms (it now calls scipy itself, so a scipy comparison would be an
identity), and the replicate arm against the measured blasi verdicts (see ``test_replicate_lof.py``,
which also covers the PEtab-shaped plumbing and the rollup).
"""

from __future__ import annotations

import math

import pytest

from petab_mcp import residual_tests as rt


def test_sign_test_matches_exact_binomial():
    # 9 of 10 positive: two-sided exact binomial tail at p=0.5.
    res = rt.sign_test([1, 1, 1, 1, 1, 1, 1, 1, 1, -1])
    assert res["n_pos"] == 9 and res["n_neg"] == 1
    expected = 2 * (math.comb(10, 0) + math.comb(10, 1)) / 2**10
    assert res["p"] == pytest.approx(expected)
    assert res["low_power"] is False  # n == 10 >= MIN_POWER_N

    scipy_stats = pytest.importorskip("scipy.stats")
    ref = scipy_stats.binomtest(1, 10, 0.5).pvalue  # 1 minority sign out of 10
    assert res["p"] == pytest.approx(ref)


def test_sign_test_balanced_is_not_significant():
    res = rt.sign_test([1, -1, 1, -1, 1, -1, 1, -1])
    assert res["n_pos"] == 4 and res["n_neg"] == 4
    assert res["p"] == pytest.approx(1.0)


def test_sign_test_drops_zeros_and_nonfinite():
    res = rt.sign_test([1.0, 0.0, -1.0, float("nan"), 2.0])
    assert res["n"] == 3 and res["n_pos"] == 2 and res["n_neg"] == 1


def test_sign_test_empty():
    res = rt.sign_test([0.0, 0.0])
    assert res["n"] == 0 and res["p"] is None and res["low_power"] is True


def test_runs_pmf_sums_to_one():
    for n1, n2 in [(3, 3), (2, 5), (6, 4), (1, 4)]:
        total = sum(rt._runs_pmf(n1, n2).values())
        assert total == pytest.approx(1.0), (n1, n2)


def test_runs_test_detects_a_perfect_trend():
    # All + then all - : the fewest possible runs (2) for a 5/5 split -> systematic drift.
    res = rt.runs_test([1, 1, 1, 1, 1, -1, -1, -1, -1, -1])
    assert res["runs"] == 2
    assert res["direction"] == "few_runs"
    # Exact two-sided p for R=2 with n1=n2=5: only R=2 is that unlikely at the low tail;
    # P(R=2) = 2 / C(10,5) = 2/252, and the symmetric high-tail extreme R=10 has equal mass.
    assert res["p"] == pytest.approx(2 * (2 / math.comb(10, 5)))


def test_runs_test_alternating_is_many_runs():
    # Perfect alternation of only 3+3 points: max runs (6), but exact two-sided p is just 0.2 —
    # too few points to be significant. That is correct low-power behaviour, not a miss.
    res = rt.runs_test([1, -1, 1, -1, 1, -1])
    assert res["runs"] == 6
    assert res["direction"] == "many_runs"
    assert res["p"] == pytest.approx(0.2)

    # A longer perfect alternation IS significant.
    long_alt = [(-1) ** i for i in range(14)]
    res_long = rt.runs_test(long_alt)
    assert res_long["direction"] == "many_runs"
    assert res_long["p"] < 0.05


def test_runs_test_random_looking_series_not_significant():
    res = rt.runs_test([1, 1, -1, 1, -1, -1, 1, -1])
    assert res["p"] is not None and res["p"] > 0.2
    assert res["low_power"] is False  # n == 8


def test_runs_test_single_sign_is_undefined():
    res = rt.runs_test([1, 1, 1, 1])
    assert res["p"] is None and res["direction"] is None and res["low_power"] is True


def test_runs_test_is_scale_invariant():
    """The whole point: multiplying residuals by any positive factor (an inflated sigma) must
    not change the runs-test result, whereas a magnitude score would move."""
    base = [3.0, 2.0, 1.0, -1.0, -2.0, -3.0]
    inflated = [100.0 * x for x in base]
    assert rt.runs_test(base)["p"] == pytest.approx(rt.runs_test(inflated)["p"])
    assert rt.sign_test(base)["p"] == pytest.approx(rt.sign_test(inflated)["p"])


def test_fisher_combine():
    assert rt.fisher_combine([]) is None
    assert rt.fisher_combine([0.3]) == pytest.approx(0.3)  # single series passes through
    # Two independent p=0.5 -> stat = -2(ln.5+ln.5)=2.7726, dof=4.
    scipy_stats = pytest.importorskip("scipy.stats")
    ref = float(scipy_stats.chi2.sf(-2 * 2 * math.log(0.5), 4))
    assert rt.fisher_combine([0.5, 0.5]) == pytest.approx(ref)
    # Two significant series combine to something at least as significant as either.
    assert rt.fisher_combine([0.04, 0.04]) < 0.04


# -- replicate lack-of-fit: the statistic ---------------------------------------------------------
def _point(label, n, ybar, s, prediction):
    return {"label": label, "n": n, "ybar": ybar, "s": s, "prediction": prediction}


def _z_point(label, z, n=18, s=0.1):
    """A design point whose deviation is exactly ``z`` standard errors — the fixture the statistic
    is easiest to reason about, since z is what the test's p-value is a function of."""
    return _point(label, n, z * s / math.sqrt(n), s, 0.0)


def test_student_t_tail_matches_a_hand_computed_value():
    """The only reference with content now that the tail IS ``2 * scipy.stats.t.sf``: closed forms
    scipy is not party to. (A pinned comparison against ``scipy.t.sf`` would compare the
    implementation with itself; the FAR tail this arm decides on is pinned end to end by blasi's
    measured Holm p of 2.55e-11 at df 17 — see ``test_replicate_lof.py``.)"""
    # df = 1 is the Cauchy: P(|T| >= t) = 1 - (2/pi)*arctan(t), exactly.
    for t in (0.5, 1.0, 4.0):
        assert rt.student_t_two_sided_p(t, 1) == pytest.approx(1 - 2 / math.pi * math.atan(t))
    # df = 2 has the closed form P(|T| >= t) = 1 - t/sqrt(2 + t^2), out into the tail.
    for t in (0.5, 2.0, 7.0, 200.0):
        assert rt.student_t_two_sided_p(t, 2) == pytest.approx(1 - t / math.sqrt(2 + t * t))
    assert rt.student_t_two_sided_p(0.0, 9) == pytest.approx(1.0)
    # symmetric in the sign of t, and fails OPEN (None) on unusable input
    assert rt.student_t_two_sided_p(-3.0, 7) == pytest.approx(rt.student_t_two_sided_p(3.0, 7))
    assert rt.student_t_two_sided_p(float("nan"), 7) is None
    assert rt.student_t_two_sided_p(3.0, 0) is None


def test_holm_adjusted_orders_and_stays_monotone():
    ps = [0.001, 0.02, 0.6]
    # (m-k)*p at each rank: 3*0.001, 2*0.02, 1*0.6 — already increasing, so no monotone lift.
    assert rt.holm_adjusted(ps) == pytest.approx([0.003, 0.04, 0.6])
    # INPUT order is preserved (the caller pairs these back onto its own points).
    assert rt.holm_adjusted([0.6, 0.001, 0.02]) == pytest.approx([0.6, 0.003, 0.04])
    # step-down monotonicity: 3*0.02 = 0.06 exceeds 2*0.025 = 0.05, so the second is LIFTED to 0.06
    assert rt.holm_adjusted([0.02, 0.025, 0.9]) == pytest.approx([0.06, 0.06, 0.9])
    # capped at 1 — and the cap propagates through the monotone lift, so nothing below it survives
    assert rt.holm_adjusted([0.5, 0.6]) == pytest.approx([1.0, 1.0])
    assert rt.holm_adjusted([0.4, 0.9]) == pytest.approx([0.8, 0.9])


def test_replicate_pure_error():
    res = rt.replicate_pure_error([1.0, 2.0, 3.0])
    assert res["n"] == 3 and res["df"] == 2
    assert res["mean"] == pytest.approx(2.0)
    assert res["ssq"] == pytest.approx(2.0)
    assert res["sd"] == pytest.approx(1.0)
    # non-finite values are dropped; a point measured once has no scatter of its own
    assert rt.replicate_pure_error([1.0, float("nan"), 3.0])["n"] == 2
    assert rt.replicate_pure_error([1.0]) is None
    assert rt.replicate_pure_error([]) is None


def test_replicate_lack_of_fit_flags_the_deviating_point():
    """A known-z fixture: one point 12 SE out among four that sit on the model."""
    points = [_z_point("far", 12.0)] + [_z_point(f"near_{i}", 0.2) for i in range(3)]
    out = rt.replicate_lack_of_fit(points)
    assert out["available"] is True and out["status"] == "flagged"
    assert out["flagged"] == ["far"]
    assert out["n_points"] == 4
    assert out["max_abs_z"] == pytest.approx(12.0, rel=1e-3)
    # Holm's smallest step is the family size times the raw tail
    assert out["min_p_holm"] == pytest.approx(4 * rt.student_t_two_sided_p(12.0, 17), rel=1e-2)
    assert out["points"][0]["label"] == "far" and out["points"][0]["z"] == pytest.approx(12.0)


def test_replicate_lack_of_fit_reports_only_the_points_it_flagged():
    """``points`` is what the advice NAMES as "where the form is wrong", so a point the model
    already reproduces must never appear there. Filling the list up to the cap by |z| regardless of
    the flag sent the agent after mechanisms for points at z = 0 in any family with fewer than
    ``REPLICATE_LOF_MAX_REPORTED_POINTS`` real flags."""
    points = [_z_point("far", 12.0)] + [_z_point(f"near_{i}", 0.0) for i in range(3)]
    out = rt.replicate_lack_of_fit(points)
    assert out["flagged"] == ["far"]
    assert [p["label"] for p in out["points"]] == ["far"]
    # the FULL family is still what n_points and the Holm correction are over
    assert out["n_points"] == 4


def test_replicate_lack_of_fit_names_the_points_it_tested():
    """``tested`` is how the assembler learns WHICH observables the arm actually covered, so that a
    powered pass is only credited to those (see data_access._rollup_structure). Dropped points must
    not appear in it."""
    points = [_z_point(f"ok{i}", 0.3) for i in range(3)]
    bad = {"label": "no_prediction", "n": 18, "ybar": 1.0, "s": 0.1, "prediction": None}
    out = rt.replicate_lack_of_fit(points + [bad])
    assert sorted(out["tested"]) == ["ok0", "ok1", "ok2"]
    # an arm that could not run tested nothing at all
    assert rt.replicate_lof_unavailable("vacuous")["tested"] == []


def test_replicate_lack_of_fit_passes_a_model_that_sits_inside_the_scatter():
    out = rt.replicate_lack_of_fit([_z_point(f"p{i}", z) for i, z in
                                   enumerate([0.4, -1.1, 2.0, -0.3, 1.2])])
    assert out["available"] is True and out["status"] == "pass"
    assert out["flagged"] == []
    # a pass carries NO point rows: the payload rides on every fit (see the cap test below)
    assert out["points"] == []


def test_replicate_lack_of_fit_is_invariant_to_the_sigma_convention():
    """z_i uses only the point's own replicate sd, so nothing about the model's noise parameter —
    ML or RSS/(n-k) — can move this verdict. That is the whole reason for this arm."""
    points = [_z_point("far", 6.0)] + [_z_point(f"near_{i}", 0.5) for i in range(4)]
    assert rt.replicate_lack_of_fit(points) == rt.replicate_lack_of_fit(points)
    # scaling every point's scatter AND its deviation together (a different noise convention) leaves
    # each z — and so the verdict — untouched
    rescaled = [{**p, "s": p["s"] * 2.5, "ybar": p["ybar"] * 2.5} for p in points]
    assert rt.replicate_lack_of_fit(rescaled)["max_abs_z"] == \
        rt.replicate_lack_of_fit(points)["max_abs_z"]


# -- the guards, each named after the benchmark example that forces it ----------------------------
def test_replicate_lof_unavailable_without_replicates():
    """No repeated design point (13 of the 16 benchmark examples) -> NO EVIDENCE, never a pass."""
    out = rt.replicate_lack_of_fit([])
    assert out["available"] is False and out["status"] == "no_replicated_points"
    assert out["flagged"] == [] and out["max_abs_z"] is None and out["points"] == []


def test_replicate_lof_unavailable_at_one_degree_of_freedom():
    """schwen: 126 replicated points, EVERY n_i = 2. One df cannot be a yardstick, and its pure
    error there is 7.8x smaller than the problem's own reference sigma — so the arm must abstain
    rather than reject schwen's true model."""
    points = [_z_point(f"p{i}", 30.0, n=2) for i in range(126)]
    out = rt.replicate_lack_of_fit(points)
    assert out["available"] is False and out["status"] == "replicates_too_thin"
    assert out["flagged"] == []
    # ...and it is the df, not the count: give the same points one more replicate each and it tests
    assert rt.replicate_lack_of_fit([_z_point(f"p{i}", 30.0, n=3) for i in range(126)])[
        "available"] is True


def test_replicate_lof_unavailable_below_three_points():
    out = rt.replicate_lack_of_fit([_z_point("a", 20.0), _z_point("b", 0.1)])
    assert out["available"] is False and out["status"] == "too_few_usable_points"
    assert out["n_points"] == 2 and out["flagged"] == []


def test_replicate_lof_vacuous_when_the_fit_can_zero_every_point():
    """Saturation is a property of the PROBLEM, not of one noise family, so the guard is applied
    once — by the assembler, to the pooled point count (data_access._merge_lof_verdicts, which is
    the only caller that knows every family).

    armistead is why. Its four families are sd_Cer_obs, sd_Sphinga_obs and sd_Sphingo_obs with 4
    replicated points each (df 3) and sd_S1P_obs with 2 (df 4), against 10 estimated structural
    parameters. Per family, each 4-point arm looked saturated and the whole example reported NO
    EVIDENCE with a summary line claiming as many parameters as replicated points — of 12 pooled
    points against 10, which leaves 2 lack-of-fit directions and a (thin) real test."""
    from petab_mcp import data_access

    def arms(n_points, families=1):
        return [rt.replicate_lack_of_fit([_z_point(f"f{f}p{i}", 0.1) for i in range(n_points)])
                for f in range(families)]

    armistead = data_access._merge_lof_verdicts(arms(4, families=3), {}, n_structural=10)
    assert armistead["available"] is True and armistead["n_points"] == 12
    assert armistead["lof_df"] == 2 and armistead["weak"] is True
    # ...while a family that really is outnumbered by the parameters tests nothing: 4 points, 10
    # structural, no siblings to pool with.
    out = data_access._merge_lof_verdicts(arms(4), {}, n_structural=10)
    assert out["available"] is False and out["status"] == "vacuous" and out["n_points"] == 4
    # It is the COUNT that decides, not the example: one more point than there are structural
    # parameters and the family carries information again (blasi: 15 points, 8-9 structural).
    assert data_access._merge_lof_verdicts(arms(11), {}, n_structural=10)["available"] is True
    # an unknown parameter count cannot make the test vacuous — it makes its POWER unknown
    unknown = data_access._merge_lof_verdicts(arms(4), {}, n_structural=None)
    assert unknown["available"] is True and "lof_df" not in unknown


def test_replicate_lof_fails_open_on_unusable_points():
    """A diagnostic must never manufacture a verdict from bad input, nor raise into the fit."""
    bad = [
        {"label": "no_prediction", "n": 18, "ybar": 1.0, "s": 0.1, "prediction": None},
        {"label": "zero_scatter", "n": 18, "ybar": 1.0, "s": 0.0, "prediction": 0.0},
        {"label": "nan_mean", "n": 18, "ybar": float("nan"), "s": 0.1, "prediction": 0.0},
        {"label": "missing_n", "ybar": 1.0, "s": 0.1, "prediction": 0.0},
    ]
    out = rt.replicate_lack_of_fit(bad)
    # replicated and thick enough, yet not testable — a distinct status from "no replicates" and
    # from "too thin", since each unavailable status has to mean exactly one thing
    assert out["available"] is False and out["status"] == "unavailable" and out["flagged"] == []
    # the usable points still count, and the unusable ones are simply dropped
    mixed = rt.replicate_lack_of_fit(bad + [_z_point(f"ok{i}", 0.3) for i in range(3)])
    assert mixed["available"] is True and mixed["n_points"] == 3


def test_replicate_lof_caps_the_reported_points():
    """TOKEN COST is a first-class constraint: this verdict rides on EVERY fit and is re-sent in
    the agent's history on every later turn, so a 15- (or 126-) row table must never ship."""
    points = [_z_point(f"p{i}", 20.0 + i) for i in range(40)]
    out = rt.replicate_lack_of_fit(points)
    assert len(out["flagged"]) == 40                       # all condemned...
    assert len(out["points"]) == rt.REPLICATE_LOF_MAX_REPORTED_POINTS  # ...but only 5 reported
    # ...and they are the WORST by |z|, largest first
    assert [p["label"] for p in out["points"]] == ["p39", "p38", "p37", "p36", "p35"]
    assert out["points"] == sorted(out["points"], key=lambda p: abs(p["z"]), reverse=True)
    assert len(out["points"]) < len(points)
