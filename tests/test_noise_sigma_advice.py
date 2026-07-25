"""The agent-facing noise-sigma advice (fitting.noise_sigma_advice): a concise, actionable string
when the fit's ESTIMATED noise model is implausible, else None. Surfaced proactively in
get_fit_summary/get_fit_diagnosis and echoed by the harness finish gate, so the agent learns WHY a
good-looking BIC won't be accepted and fixes the mechanism instead of the noise. The inflated/
collapsed predicates themselves are covered in test_result_fields; this pins the advice wiring.
"""

from __future__ import annotations

import pytest

pytest.importorskip("petab", reason="petab_mcp dependency")

from petab_mcp.constants import NOISE_SIGMA_COLLAPSE_FRACTION  # noqa: E402
from petab_mcp.fitting import (  # noqa: E402
    NOISE_SIGMA_COLLAPSED_ADVICE,
    NOISE_SIGMA_INFLATED_ADVICE,
    _entry_collapsed,
    _entry_inflated,
    noise_sigma_advice,
    noise_sigma_advice_for,
    noise_sigma_report,
)


def _entry(fitted, cap, frac):
    return {"fitted": fitted, "upper_bound_cap": cap, "data_std": 1.0, "fraction_of_data_std": frac}


def test_none_when_plausible_or_empty():
    assert noise_sigma_advice(None) is None
    assert noise_sigma_advice({}) is None
    # below its cap, above the collapse floor -> plausible -> no advice.
    assert noise_sigma_advice({"s": _entry(0.08, 0.1, 0.4)}) is None


def test_inflated_advice():
    a = noise_sigma_advice({"s": _entry(0.5, 0.1, 2.5)})   # fitted above cap
    assert a == NOISE_SIGMA_INFLATED_ADVICE
    assert "inflated" in a and "will NOT be accepted" in a


def test_collapsed_advice():
    frac = NOISE_SIGMA_COLLAPSE_FRACTION / 2
    a = noise_sigma_advice({"s": _entry(frac, 0.5, frac)})  # fraction below floor
    assert a == NOISE_SIGMA_COLLAPSED_ADVICE
    assert "collapsed" in a and "will NOT be accepted" in a


def test_inflation_reported_first_when_both_present():
    # One inflated + one collapsed sigma (different observables): advice leads with inflation, the
    # more common misfit-masking case (matches noise_sigma_advice's documented ordering).
    both = {"hi": _entry(0.5, 0.1, 2.5),
            "lo": _entry(NOISE_SIGMA_COLLAPSE_FRACTION / 2, 0.5, NOISE_SIGMA_COLLAPSE_FRACTION / 2)}
    assert noise_sigma_advice(both) == NOISE_SIGMA_INFLATED_ADVICE


def test_shared_per_entry_predicates():
    """_entry_inflated/_entry_collapsed are the single source the top-level flags AND the report
    status both derive from, so they can't drift. Unassessable (None cap/frac) is neither."""
    assert _entry_inflated(_entry(0.5, 0.1, 2.5)) and not _entry_collapsed(_entry(0.5, 0.1, 2.5))
    lo = _entry(NOISE_SIGMA_COLLAPSE_FRACTION / 2, 0.5, NOISE_SIGMA_COLLAPSE_FRACTION / 2)
    assert _entry_collapsed(lo) and not _entry_inflated(lo)
    assert not _entry_inflated(_entry(0.08, 0.1, 0.4)) and not _entry_collapsed(_entry(0.08, 0.1, 0.4))
    assert not _entry_inflated({"fitted": 0.5, "upper_bound_cap": None})   # cap unassessable
    assert not _entry_collapsed({"fraction_of_data_std": None})            # frac unassessable


def test_advice_for_flags_derives_without_rescanning():
    assert noise_sigma_advice_for(True, False) == NOISE_SIGMA_INFLATED_ADVICE
    assert noise_sigma_advice_for(False, True) == NOISE_SIGMA_COLLAPSED_ADVICE
    assert noise_sigma_advice_for(True, True) == NOISE_SIGMA_INFLATED_ADVICE   # inflation first
    assert noise_sigma_advice_for(False, False) is None


def test_report_status_per_entry_including_unknown(monkeypatch):
    """noise_sigma_report tags each entry ok/inflated/collapsed/unknown. 'unknown' when the
    observable's std can't be computed (data_std None) — reported honestly, not as a reassuring
    'ok'. Reconcile's data-driven helpers are stubbed so this stays a pure unit test."""
    import petab_mcp.reconcile as R
    data_std = {"inf": 0.2, "col": 1.0, "fine": 0.2, "unk": None}
    cap = {"inf": 0.1, "col": 0.5, "fine": 0.1, "unk": None}
    monkeypatch.setattr(R, "noise_sigma_ids", lambda problem: set(data_std))
    monkeypatch.setattr(R, "sigma_data_std", lambda p, pid: data_std[pid])
    monkeypatch.setattr(R, "sigma_upper_bound_cap", lambda p, pid: cap[pid])
    rep = noise_sigma_report(object(), {"inf": 0.5, "col": 0.01, "fine": 0.05, "unk": 0.3})
    assert rep["inf"]["status"] == "inflated"    # 0.5 > cap 0.1
    assert rep["col"]["status"] == "collapsed"   # frac 0.01 < collapse floor
    assert rep["fine"]["status"] == "ok"         # below cap, above floor
    assert rep["unk"]["status"] == "unknown"     # data_std None -> unassessable, not 'ok'
