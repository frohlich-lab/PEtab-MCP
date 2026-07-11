"""Unit tests for the backend-agnostic fit/residual statistics (:mod:`petab_mcp.fit`).

These exercise the statistics layer with synthetic records, so they run without
AMICI. Producing records from a real simulation is covered in ``test_engine``.
"""

from __future__ import annotations

import math

import pytest

from petab_mcp import fit


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


# residuals: obs_A/c0 -> [1.0, 0.0]; obs_B/c1 -> [-1.0] and one missing point.
RECORDS = [
    _rec("obs_A", "c0", 0.0, 1.0, 0.0, 1.0, 2.0, 0.5),
    _rec("obs_A", "c0", 1.0, 2.0, 2.0, 0.0, 0.0, 0.5),
    _rec("obs_B", "c1", 0.0, 0.0, 1.0, -1.0, -1.0, 1.0),
    _rec("obs_B", "c1", 1.0, 3.0, None, None, None, None),  # unsimulated -> excluded
]


def test_overall_statistics():
    stats = fit.residual_statistics(RECORDS)["overall"]
    assert stats["n"] == 3
    assert stats["n_missing"] == 1
    assert stats["mean_residual"] == pytest.approx(0.0)
    assert stats["sse"] == pytest.approx(2.0)
    assert stats["rmse"] == pytest.approx(math.sqrt(2 / 3))
    assert stats["mean_abs_residual"] == pytest.approx(2 / 3)
    assert stats["max_abs_residual"] == pytest.approx(1.0)
    # weighted / chi2 terms
    assert stats["chi2"] == pytest.approx(5.0)  # 2^2 + 0 + (-1)^2
    assert stats["reduced_chi2"] == pytest.approx(5 / 3)
    assert stats["mean_weighted_residual"] == pytest.approx(1 / 3)
    assert stats["max_abs_weighted_residual"] == pytest.approx(2.0)
    assert stats["bias_z"] == pytest.approx((1 / 3) * math.sqrt(3))


def test_cross_section_by_observable():
    report = fit.residual_statistics(RECORDS)
    by_obs = {e["observable_id"]: e for e in report["by_observable"]}
    assert set(by_obs) == {"obs_A", "obs_B"}
    assert by_obs["obs_A"]["n"] == 2
    assert by_obs["obs_A"]["sse"] == pytest.approx(1.0)
    assert by_obs["obs_B"]["n"] == 1
    assert by_obs["obs_B"]["n_missing"] == 1
    assert by_obs["obs_B"]["chi2"] == pytest.approx(1.0)


def test_cross_section_by_condition_and_time():
    report = fit.residual_statistics(RECORDS)
    by_cond = {e["condition_id"]: e for e in report["by_condition"]}
    assert by_cond["c0"]["n"] == 2 and by_cond["c1"]["n"] == 1

    # timepoints are numeric and returned in ascending order
    times = [e["time"] for e in report["by_time"]]
    assert times == [0.0, 1.0]
    by_time = {e["time"]: e for e in report["by_time"]}
    assert by_time[0.0]["n"] == 2  # obs_A and obs_B both measured at t=0
    assert by_time[1.0]["n_missing"] == 1


def test_selected_cross_section_only():
    report = fit.residual_statistics(RECORDS, cross_sections=["by_observable"])
    assert "by_observable" in report
    assert "by_condition" not in report and "by_time" not in report


def test_unknown_cross_section_raises():
    with pytest.raises(ValueError):
        fit.residual_statistics(RECORDS, cross_sections=["by_nonsense"])


def test_worst_points_ranked_by_weighted_residual():
    worst = fit.worst_points(RECORDS, top_n=2)
    assert len(worst) == 2
    # |weighted|: 2.0 (obs_A,t0) > 1.0 (obs_B,c1,t0); missing point excluded
    assert worst[0]["observable_id"] == "obs_A" and worst[0]["time"] == 0.0
    assert worst[1]["observable_id"] == "obs_B"
    assert all(r["residual_weighted"] is not None for r in worst)


def test_worst_points_falls_back_to_raw_residual():
    recs = [
        _rec("o", "c", 0.0, 1.0, 0.0, 1.0, None, None),
        _rec("o", "c", 1.0, 5.0, 0.0, 5.0, None, None),
    ]
    worst = fit.worst_points(recs, top_n=1)
    assert worst[0]["residual"] == pytest.approx(5.0)


def test_fit_inspection_bundle():
    report = fit.fit_inspection(RECORDS, top_n=3)
    assert report["n_points"] == 4
    assert report["summary"]["n"] == 3
    assert set(report["statistics"]) >= {"overall", "by_observable", "by_condition", "by_time"}
    assert len(report["worst_points"]) == 3
    assert isinstance(report["bias_flags"], list)


def test_bias_flag_detects_systematic_offset():
    # 10 points all over-shooting by ~1 sigma -> mean_weighted ~ 1, bias_z ~ 3.16
    biased = [
        _rec("obs_bias", "c0", float(i), 1.0, 0.5, 0.5, 1.0, 0.5) for i in range(10)
    ]
    report = fit.fit_inspection(biased)
    flags = report["bias_flags"]
    assert any(f["cross_section"] == "by_observable" and f["group"] == "obs_bias" for f in flags)
    flag = next(f for f in flags if f["group"] == "obs_bias")
    assert flag["direction"] == "model_under_predicts"  # residual = meas - sim > 0
    assert abs(flag["bias_z"]) >= fit._BIAS_Z_THRESHOLD


def test_empty_records():
    report = fit.fit_inspection([])
    assert report["n_points"] == 0
    assert report["summary"]["n"] == 0
    assert report["summary"]["mean_residual"] is None
    assert report["worst_points"] == []
    assert report["bias_flags"] == []
