"""Unit tests for the sign / Wald–Wolfowitz residual-structure tests.

Pure numeric code over residual lists, so no AMICI/pyPESTO fit is needed. Where a reference
exists (scipy for the sign test's binomial tail and a runs-test PMF that must sum to 1), the
test pins to it; the exact runs-test p-values are checked against hand-worked small cases.
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
