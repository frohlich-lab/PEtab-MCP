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
        "condition_id": cond,
        "time": t,
        "measurement": m,
        "simulation": s,
        "residual": res,
        "residual_weighted": w,
        "noise": noise,
    }


def test_chi2_prefers_res_when_weighted_incomplete():
    # all points have weighted residuals -> use them (petab-canonical)
    full = [_rec("o", "c", 0.0, 1.0, 0.0, 1.0, 2.0, 0.5), _rec("o", "c", 1.0, 2.0, 0.0, 2.0, 1.0, 1.0)]
    assert AmiciEngine._chi2(full, res=[9.0, 9.0]) == pytest.approx(5.0)  # 2^2 + 1^2, RES ignored

    # one point has a residual but no weighted residual -> incomplete -> fall back to RES
    mixed = [
        _rec("o", "c", 0.0, 1.0, 0.0, 1.0, 2.0, 0.5),
        _rec("o", "c", 1.0, 2.0, 0.0, 2.0, None, None),  # noise eval failed
    ]
    assert AmiciEngine._chi2(mixed, res=[1.0, 1.0, 1.0]) == pytest.approx(3.0)  # sum(RES^2)

    # incomplete weighted AND no RES -> last-resort partial weighted sum
    assert AmiciEngine._chi2(mixed, res=None) == pytest.approx(4.0)  # only the 2.0 term

    # unsimulated points (no residual) don't count as "contributing", so a complete
    # weighted set over the simulated points is still used
    with_gap = [
        _rec("o", "c", 0.0, 1.0, 0.0, 1.0, 2.0, 0.5),
        _rec("o", "c", 1.0, 3.0, None, None, None, None),  # not simulated
    ]
    assert AmiciEngine._chi2(with_gap, res=[9.0]) == pytest.approx(4.0)  # 2^2, RES ignored

    # nothing available
    assert AmiciEngine._chi2([_rec("o", "c", 0.0, 1.0, None, None, None, None)], res=None) is None
