from __future__ import annotations

import pytest

from petab_mcp import parameters as P
from petab_mcp.errors import ParameterError


def _by_id(session):
    return {r["parameterId"]: r for r in P.parameters_to_records(session.problem)}


def test_records(session):
    recs = _by_id(session)
    assert set(recs) == {"k1", "k2", "sigma_B"}
    assert recs["k1"]["estimate"] is True
    assert recs["sigma_B"]["estimate"] is False
    assert recs["k1"]["parameterScale"] == "log10"
    assert recs["sigma_B"]["parameterScale"] == "lin"


def test_update_nominal_and_estimate(session):
    P.update_parameters(session.problem, [{"id": "k1", "nominalValue": 0.9}])
    P.update_parameters(session.problem, [{"id": "sigma_B", "estimate": "true"}])
    recs = _by_id(session)
    assert float(recs["k1"]["nominalValue"]) == 0.9
    assert recs["sigma_B"]["estimate"] is True


def test_update_scale(session):
    P.update_parameters(session.problem, [{"id": "k1", "parameterScale": "lin"}])
    assert _by_id(session)["k1"]["parameterScale"] == "lin"
    with pytest.raises(ParameterError):
        P.update_parameters(session.problem, [{"id": "k1", "parameterScale": "ln"}])


def test_update_unknown_id(session):
    with pytest.raises(ParameterError):
        P.update_parameters(session.problem, [{"id": "ghost", "nominalValue": 1.0}])


def test_update_bad_bounds(session):
    with pytest.raises(ParameterError):
        P.update_parameters(session.problem, [{"id": "k1", "lowerBound": 10.0, "upperBound": 1.0}])


def test_update_bad_bounds_is_atomic(session):
    """A rejected update must leave the parameter fully unchanged."""
    before = _by_id(session)["k1"]
    with pytest.raises(ParameterError):
        P.update_parameters(session.problem, [{"id": "k1", "lowerBound": 10.0, "upperBound": 1.0}])
    assert _by_id(session)["k1"] == before


def test_update_coerces_numeric_string(session):
    """Numeric-string values persist as floats."""
    P.update_parameters(session.problem, [{"id": "k1", "lowerBound": "0.001"}])
    lb = session.problem.parameter_df.loc["k1", "lowerBound"]
    assert float(lb) == pytest.approx(0.001)


def test_update_non_editable_field(session):
    with pytest.raises(ParameterError):
        P.update_parameters(session.problem, [{"id": "k1", "parameterName": "x"}])


def test_sync_reports_in_sync(session):
    out = P.sync_parameters(session.problem, "report")
    assert out["diff"]["in_sync"] is True
    assert out["diff"]["model_only"] == []
