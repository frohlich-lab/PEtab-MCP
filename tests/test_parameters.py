from __future__ import annotations

import pytest

from petab_mcp import parameters as P
from petab_mcp.errors import ParameterError


def _by_id(session):
    return {r["parameterId"]: r for r in P.parameters_to_records(session.problem)}


def test_problem_fingerprint_depends_on_both_model_and_parameters():
    import pandas as pd

    def _df(rows):
        return pd.DataFrame(rows).set_index("parameterId")

    params = _df([{"parameterId": "k", "lowerBound": 1e-3, "upperBound": 1e3, "estimate": 1}])
    base = P.problem_fingerprint("model m() end", params)
    assert base == P.problem_fingerprint("model m() end", params)  # deterministic
    assert len(base) == 16
    # a different model -> different hash
    assert base != P.problem_fingerprint("model m() k=1 end", params)
    # same model, a changed parameter bound -> different hash (the footgun this guards)
    changed = _df([{"parameterId": "k", "lowerBound": 1e-3, "upperBound": 1e6, "estimate": 1}])
    assert base != P.problem_fingerprint("model m() end", changed)


def test_records(session):
    recs = _by_id(session)
    assert set(recs) == {"k1", "k2", "sigma_B", "A_init", "B_init"}
    assert recs["k1"]["estimate"] is True
    assert recs["sigma_B"]["estimate"] is False
    assert recs["k1"]["parameterScale"] == "log10"
    assert recs["A_init"]["parameterScale"] == "lin"  # IC params are linear


def test_records_include_role_and_status(session):
    recs = _by_id(session)
    # roles name what each parameter IS
    assert recs["k1"]["role"] == "kinetic"
    assert recs["k2"]["role"] == "kinetic"
    assert recs["sigma_B"]["role"] == "noise_sigma"
    assert recs["A_init"]["role"] == "initial_condition"
    assert recs["B_init"]["role"] == "initial_condition"
    # status restates the estimate flag in words
    assert recs["k1"]["status"] == "estimated"
    assert recs["sigma_B"]["status"] == "fixed"
    assert recs["A_init"]["status"] == "fixed"


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


def test_an_unknown_id_is_told_the_ids_nearest_it(session):
    """`Unknown parameter id` listed every id, 17 times in 11 runs of gemot round 5, for guesses
    like `koff_ip3_1` beside `kon_ip3_1` and a mechanism's bare `Ka` for `s6_phos_Ka` (#251)."""
    known = ["kon_ip3_1", "egfr_act_k1", "s6_phos_Ka", "sigma_B", *(f"x{i}" for i in range(30))]
    assert "['kon_ip3_1']" in P._unknown_id("koff_ip3_1", known, ())
    assert "['egfr_act_k1']" in P._unknown_id("egfr_erec_k", known, ())
    said = P._unknown_id("Ka", known, ("s6_phos",))
    assert "['s6_phos_Ka']" in said and "`<prefix>_<name>`: ['s6_phos_Ka']" in said
    with pytest.raises(
        ParameterError,
        match=r"Unknown parameter id 'init'.*`<prefix>_<name>`: "
        r"\['A_init'\]",
    ):
        P.update_parameters(session.problem, [{"id": "init", "lowerBound": 0.1}], ["A"])
