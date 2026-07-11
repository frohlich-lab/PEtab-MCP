from __future__ import annotations

import pytest

pytest.importorskip("amici", reason="AMICI not installed; skipping simulation tests")


def test_evaluate_objective(engine):
    obj = engine.evaluate_objective()
    assert isinstance(obj["neg_log_likelihood"], float)
    assert obj["log_likelihood"] == pytest.approx(-obj["neg_log_likelihood"])
    assert obj["n_data_points"] == 6
    assert len(obj["simulations"]) == 6
    rec = obj["simulations"][0]
    assert {"observable_id", "time", "measurement", "simulation", "residual"} <= set(rec)


def test_simulate(engine):
    out = engine.simulate()
    assert set(out["parameters"]) == {"k1", "k2", "sigma_B"}
    assert len(out["simulations"]) == 6


def test_parameter_change_no_recompile(engine, session):
    from petab_mcp import parameters as P

    obj0 = engine.evaluate_objective()
    h = session.sbml_hash()
    P.update_parameters(session.problem, [{"id": "k1", "nominal_value": 1.0}])
    obj1 = engine.evaluate_objective()
    assert session.sbml_hash() == h  # model unchanged
    assert obj1["neg_log_likelihood"] != obj0["neg_log_likelihood"]


def test_optimize_improves_fit(engine, session):
    from petab_mcp import optimize as O

    before = engine.evaluate_objective()["neg_log_likelihood"]
    result = O.optimize(session, engine, n_starts=3, seed=1)
    assert result["neg_log_likelihood"] <= before + 1e-6
    assert set(result["fitted_parameters"]) == {"k1", "k2"}
