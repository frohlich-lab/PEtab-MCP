"""Unit tests for :meth:`petab_mcp.engine.AmiciEngine._chi2`.

``_chi2`` is a pure static method (it imports numpy only for the ``RES`` fallback and
never touches AMICI), so its chi2-source-selection logic is unit-testable without a
simulation backend. This guards the review fix that a partial noise-weighted sum must not
silently undercount when AMICI supplies a complete ``RES`` vector.
"""

from __future__ import annotations

import pytest

from petab_mcp.engine import AmiciEngine


def _rec(obs, cond, t, m, s, res, w, noise):
    return {
        "observable_id": obs,
        "experiment_id": cond,
        "time": t,
        "measurement": m,
        "simulation": s,
        "residual": res,
        "residual_weighted": w,
        "noise": noise,
    }


def test_chi2_prefers_res_when_weighted_incomplete():
    # all points have weighted residuals -> use them (petab-canonical)
    full = [
        _rec("o", "c", 0.0, 1.0, 0.0, 1.0, 2.0, 0.5),
        _rec("o", "c", 1.0, 2.0, 0.0, 2.0, 1.0, 1.0),
    ]
    assert AmiciEngine._chi2(full, res=[9.0, 9.0]) == pytest.approx(5.0)  # 2^2 + 1^2, RES ignored

    # one point has a residual but no weighted residual -> incomplete -> fall back to RES
    mixed = [
        _rec("o", "c", 0.0, 1.0, 0.0, 1.0, 2.0, 0.5),
        _rec("o", "c", 1.0, 2.0, 0.0, 2.0, None, None),  # noise eval failed
    ]
    assert AmiciEngine._chi2(mixed, res=[1.0, 1.0, 1.0]) == pytest.approx(3.0)  # sum(RES^2)

    # incomplete weighted AND no usable RES -> None, NOT a partial sum: a chi2 over a subset
    # of the data is indistinguishable from a real one and disagrees with n_data_points.
    assert AmiciEngine._chi2(mixed, res=None) is None

    # nothing available
    assert AmiciEngine._chi2([_rec("o", "c", 0.0, 1.0, None, None, None, None)], res=None) is None


def test_chi2_none_when_simulation_incomplete():
    """A measured point the simulation never reached makes chi2 undefined.

    Regression for the fit where AMICI stopped integrating at t=22: the 5 unreached
    points were dropped before the completeness check, so chi2 was summed over the 16
    surviving points and reported against n_data_points=21 — a finite, plausible score for a
    model whose likelihood was in fact NaN.
    """
    with_gap = [
        _rec("o", "c", 0.0, 1.0, 0.0, 1.0, 2.0, 0.5),
        _rec("o", "c", 1.0, 3.0, None, None, None, None),  # measured, never simulated
    ]
    assert AmiciEngine._chi2(with_gap, res=None) is None
    # RES is NaN-contaminated for the same reason, so it cannot rescue the score either.
    assert AmiciEngine._chi2(with_gap, res=[2.0, float("nan")]) is None
    # ...but a complete, finite RES vector still can.
    assert AmiciEngine._chi2(with_gap, res=[2.0, 1.0]) == pytest.approx(5.0)

    assert AmiciEngine.simulation_gap(with_gap) == (2, 1)
    assert AmiciEngine.simulation_gap([]) == (0, 0)


def test_simulation_gap_ignores_unmeasured_rows():
    """Only measured points are expected: a row with no measurement contributes nothing."""
    recs = [
        _rec("o", "c", 0.0, 1.0, 0.0, 1.0, 2.0, 0.5),
        _rec("o", "c", 1.0, None, 0.0, None, None, None),  # no measurement
    ]
    assert AmiciEngine.simulation_gap(recs) == (1, 0)
    assert AmiciEngine._chi2(recs, res=None) == pytest.approx(4.0)


def test_res_fallback_reachable_from_rdatas():
    """The RES fallback must have a real source now that simulate_petab is out of the path.

    chi2 falls back to AMICI's complete residual vector when a point has a residual but no
    petab-weighted one (its noise formula could not be evaluated). That vector used to arrive as
    `result[RES]` from simulate_petab; the objective route has to gather it off the per-condition
    ReturnData instead, and both callers must pass it — otherwise the branch is unreachable and
    chi2 silently degrades to None for every noise-formula failure.
    """

    class _RD:
        def __init__(self, res):
            self.res = res

    assert AmiciEngine._res_from_rdatas([_RD([1.0, 2.0]), _RD([2.0])]) == [1.0, 2.0, 2.0]
    assert AmiciEngine._res_from_rdatas([_RD([1.0]), _RD(None)]) is None  # incomplete -> unusable
    assert AmiciEngine._res_from_rdatas([_RD([])]) is None
    assert AmiciEngine._res_from_rdatas([]) is None

    # ...and that vector rescues the mixed case the record-based sum cannot score.
    mixed = [
        _rec("o", "c", 0.0, 1.0, 0.0, 1.0, 2.0, 0.5),
        _rec("o", "c", 1.0, 2.0, 0.0, 2.0, None, None),  # noise eval failed
    ]
    res = AmiciEngine._res_from_rdatas([_RD([1.0, 1.0]), _RD([1.0])])
    assert AmiciEngine._chi2(mixed, res=res) == pytest.approx(3.0)
