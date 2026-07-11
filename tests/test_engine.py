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
    assert {"observable_id", "condition_id", "time", "measurement", "simulation", "residual"} <= set(rec)


def test_simulate(engine):
    out = engine.simulate()
    assert {"k1", "k2", "sigma_B", "A_init", "B_init"} == set(out["parameters"])
    assert len(out["simulations"]) == 6


def test_parameter_change_no_recompile(engine, session):
    from petab_mcp import parameters as P

    obj0 = engine.evaluate_objective()
    h = session.sbml_hash()
    P.update_parameters(session.problem, [{"id": "k1", "nominalValue": 1.0}])
    obj1 = engine.evaluate_objective()
    assert session.sbml_hash() == h  # model unchanged
    assert obj1["neg_log_likelihood"] != obj0["neg_log_likelihood"]


def test_simulate_after_operation_recompiles(engine, session):
    from petab_mcp import operations as O

    before = engine.evaluate_objective()["neg_log_likelihood"]
    O.add_reaction(session, "J: B -> D; k3*B", justification="B to D", references=["PMID:1"])
    after = engine.evaluate_objective()  # new species/param -> recompiles
    assert after["neg_log_likelihood"] != before
    assert after["n_data_points"] == 6


def test_optimize_improves_fit(engine, session):
    pytest.importorskip("pypesto", reason="pyPESTO not installed")
    from petab_mcp import optimize as O

    before = engine.evaluate_objective()["neg_log_likelihood"]
    result = O.optimize(session, engine, n_starts=3, seed=1)
    assert result["backend"] == "pypesto"
    assert result["neg_log_likelihood"] <= before + 1e-6
    assert {"k1", "k2"} <= set(result["fitted_parameters"])


def test_optimize_tracks_improvement_vs_versions(engine, session):
    pytest.importorskip("pypesto", reason="pyPESTO not installed")
    from petab_mcp import optimize as O

    first = O.optimize(session, engine, n_starts=2, seed=1)["improvement"]
    assert first["objective_before"] is not None
    assert first["self_improvement"] is not None
    assert first["is_new_best"] is True  # nothing to beat yet
    assert first["improvement_vs_best_prior"] is None

    second = O.optimize(session, engine, n_starts=2, seed=2)["improvement"]
    assert second["best_prior_version"] is not None
    assert second["improvement_vs_best_prior"] is not None
