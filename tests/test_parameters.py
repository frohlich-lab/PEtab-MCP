from __future__ import annotations

import pytest

from petab_mcp import parameters as P
from petab_mcp.errors import ParameterError


def test_records(session):
    recs = {r["id"]: r for r in P.parameters_to_records(session.problem)}
    assert set(recs) == {"k1", "k2", "sigma_B"}
    assert recs["k1"]["estimate"] is True
    assert recs["sigma_B"]["estimate"] is False


def test_update_nominal_and_estimate(session):
    P.update_parameters(session.problem, [{"id": "k1", "nominal_value": 0.9}])
    P.update_parameters(session.problem, [{"id": "sigma_B", "estimate": "true"}])
    recs = {r["id"]: r for r in P.parameters_to_records(session.problem)}
    assert float(recs["k1"]["nominal_value"]) == 0.9
    assert recs["sigma_B"]["estimate"] is True


def test_update_unknown_id(session):
    with pytest.raises(ParameterError):
        P.update_parameters(session.problem, [{"id": "ghost", "nominal_value": 1.0}])


def test_update_bad_bounds(session):
    with pytest.raises(ParameterError):
        P.update_parameters(session.problem, [{"id": "k1", "lb": 10.0, "ub": 1.0}])


def test_update_bad_bounds_is_atomic(session):
    """A rejected update must leave the parameter fully unchanged."""
    before = {r["id"]: r for r in P.parameters_to_records(session.problem)}["k1"]
    with pytest.raises(ParameterError):
        P.update_parameters(session.problem, [{"id": "k1", "lb": 10.0, "ub": 1.0}])
    after = {r["id"]: r for r in P.parameters_to_records(session.problem)}["k1"]
    assert after == before  # lb/ub not partially applied


def test_update_coerces_numeric_string(session):
    """Values arriving as numeric strings persist as floats, not raw strings."""
    P.update_parameters(session.problem, [{"id": "k1", "lb": "0.001"}])
    k1 = next(p for p in session.problem.parameters if p.id == "k1")
    assert isinstance(k1.lb, float)
    assert k1.lb == pytest.approx(0.001)


def test_update_non_editable_field(session):
    with pytest.raises(ParameterError):
        P.update_parameters(session.problem, [{"id": "k1", "id_alias": "x"}])


def test_sync_reports_in_sync(session):
    out = P.sync_parameters(session.problem, "report")
    assert out["diff"]["in_sync"] is True
    assert out["diff"]["model_only"] == []
