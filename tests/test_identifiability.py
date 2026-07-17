"""Unit tests for the identifiability statistics core (:func:`fitting._identifiability_stats`).

These exercise the pure math — FIM -> SE / marginal & partial correlations / flat eigen-directions
/ candidates, plus the multistart-spread signal — with hand-built matrices, so no AMICI/pyPESTO
fit is needed. The pyPESTO glue (`_compute_fim`, `identifiability`) is covered by the live
harness run, not here.
"""

from __future__ import annotations

import numpy as np
import pytest

from petab_mcp.fitting import _identifiability_stats


def _meta(*ids, scale="log10", est_scaled=1.0, at_bound=False):
    return [{"id": i, "scale": scale, "estimate": 10.0, "estimate_scaled": est_scaled,
             "at_bound": at_bound} for i in ids]


def test_collinear_pair_flagged_via_singular_fim():
    # a,b perfectly collinear (rank-1 block); c well determined.
    fim = np.array([[1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 4.0]])
    out = _identifiability_stats(_meta("a", "b", "c"), fim, {}, corr_threshold=0.9)

    assert out["available"] and out["method"] == "fim"
    assert out["n_estimated"] == 3
    assert out["fim"]["rank"] == 2 and out["fim"]["rank_deficient"] is True

    # marginal correlation a-b ~ +1
    ab = [c for c in out["correlations"] if {c["a"], c["b"]} == {"a", "b"}]
    assert ab and ab[0]["r"] == pytest.approx(1.0, abs=1e-6)
    # partial correlation a-b ~ -1 (from the precision/FIM directly)
    pab = [p for p in out["partial_correlations"] if {p["a"], p["b"]} == {"a", "b"}]
    assert pab and pab[0]["pr"] == pytest.approx(-1.0, abs=1e-6)

    # the a-b difference is a flat direction
    assert out["flat_directions"]
    flat_members = {p["id"] for d in out["flat_directions"] for p in d["participants"]}
    assert {"a", "b"} <= flat_members

    # a and b are simplification candidates; c is not
    cand = {c["id"] for c in out["candidates"]}
    assert {"a", "b"} <= cand and "c" not in cand
    # collinearity reason names the partner
    reasons_a = next(c["reasons"] for c in out["candidates"] if c["id"] == "a")
    assert any("collinear_with:b" in r for r in reasons_a)


def test_well_identified_has_no_flags():
    fim = np.diag([4.0, 4.0, 4.0])
    out = _identifiability_stats(_meta("a", "b", "c"), fim, {})
    assert out["available"] and out["method"] == "fim"
    assert out["correlations"] == [] and out["partial_correlations"] == []
    assert out["flat_directions"] == []
    assert out["candidates"] == []
    assert out["fim"]["rank"] == 3 and out["fim"]["rank_deficient"] is False
    # SE = sqrt(1/4) = 0.5 on the scaled axis; fold = 10**0.5 for a log10 parameter
    p = {x["id"]: x for x in out["parameters"]}
    assert p["a"]["se_scaled"] == pytest.approx(0.5, abs=1e-6)
    assert p["a"]["fold_uncertainty"] == pytest.approx(10 ** 0.5, rel=1e-3)


def test_multistart_spread_flags_unstable_parameter():
    fim = np.diag([4.0, 4.0])
    spread = {"a": {"std": 0.7, "min": -0.5, "max": 1.5, "n": 5},
              "b": {"std": 0.01, "min": 0.99, "max": 1.01, "n": 5}}
    out = _identifiability_stats(_meta("a", "b"), fim, spread, spread_unstable=0.5)
    assert out["method"] == "fim+multistart"
    p = {x["id"]: x for x in out["parameters"]}
    assert p["a"]["multistart_std"] == 0.7 and p["a"]["multistart_range"] == pytest.approx(2.0)
    cand = {c["id"] for c in out["candidates"]}
    assert "a" in cand and "b" not in cand
    assert any("multistart_unstable" in r for r in
               next(c["reasons"] for c in out["candidates"] if c["id"] == "a"))


def test_spread_only_when_no_fim():
    spread = {"a": {"std": 0.9, "min": 0.0, "max": 2.0, "n": 4}}
    out = _identifiability_stats(_meta("a", "b"), None, spread)
    assert out["method"] == "multistart"
    assert out["fim"] is None
    assert out["correlations"] == [] and out["flat_directions"] == []
    assert {x["id"]: x for x in out["parameters"]}["a"]["se_scaled"] is None
    assert {c["id"] for c in out["candidates"]} == {"a"}


def test_at_bound_is_a_candidate():
    fim = np.diag([4.0, 4.0])
    meta = _meta("a", "b")
    meta[0]["at_bound"] = True
    out = _identifiability_stats(meta, fim, {})
    a = next(c for c in out["candidates"] if c["id"] == "a")
    assert "at_bound" in a["reasons"]
    assert "b" not in {c["id"] for c in out["candidates"]}


def test_no_signal_reports_unavailable():
    out = _identifiability_stats(_meta("a"), None, {})
    assert out["available"] is False and out["method"] == "none"
    assert out["candidates"] == []
