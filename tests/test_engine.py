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


def test_records_carry_weighted_residuals(engine):
    rec = engine.evaluate_objective()["simulations"][0]
    assert {"residual", "residual_weighted", "noise"} <= set(rec)
    # obs_B noise is sigma_B (nominal 0.1); weighted residual = residual / sigma
    assert rec["noise"] == pytest.approx(0.1)
    assert rec["residual_weighted"] == pytest.approx(rec["residual"] / rec["noise"])


def test_residuals_summary_matches_chi2(engine):
    import math

    import numpy as np

    data = engine.residuals()
    assert data["n_data_points"] == 6
    assert {"neg_log_likelihood", "chi2", "records"} <= set(data)

    # chi2 is the sum of squared noise-weighted residuals (petab-canonical)
    weighted = np.array([r["residual_weighted"] for r in data["records"]], dtype=float)
    noise = np.array([r["noise"] for r in data["records"]], dtype=float)
    assert float(np.sum(weighted**2)) == pytest.approx(data["chi2"], rel=1e-9)

    # the Gaussian log-likelihood identity ties chi2 + noise back to AMICI's objective,
    # cross-validating that our weighted residuals reproduce the fitted likelihood
    nllh = 0.5 * (data["chi2"] + float(np.sum(np.log(2 * math.pi * noise**2))))
    assert nllh == pytest.approx(data["neg_log_likelihood"], rel=1e-6)


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


def test_fit_and_simulate_improves_fit(fit_session):
    pytest.importorskip("pypesto", reason="pyPESTO not installed")
    from petab_mcp import fitting as F

    engine = fit_session.engine
    before = engine.evaluate_objective()["neg_log_likelihood"]
    result = F.fit_and_simulate(fit_session, engine, do_fit=True, n_starts=3)
    assert result["status"] == "fit"
    assert result["neg_log_likelihood"] <= before + 1e-6
    assert {"k1", "k2"} <= set(result["fitted_parameters"])
    assert len(result["records"]) == 6
    assert fit_session.simulation is result  # stored on the session


def test_fit_records_optimization_trace(fit_session):
    pytest.importorskip("pypesto", reason="pyPESTO not installed")
    from petab_mcp import fitting as F

    F.fit_and_simulate(fit_session, fit_session.engine, do_fit=True, n_starts=3)
    opt = fit_session.optimization
    assert opt is not None
    # one result row per start; start 0 is the best (lowest fval)
    assert opt["n_starts"] == 3 and len(opt["starts"]) == 3
    fvals = [s["fval"] for s in opt["starts"]]
    assert fvals[0] == min(v for v in fvals if v is not None)
    # a per-iteration trace was recorded for at least the best start
    assert any(t["start"] == 0 for t in opt["trace"])
    assert all({"start", "iteration", "fval"} <= set(t) for t in opt["trace"])

    # a simulate-only pass clears the (non-persisted) trace
    F.fit_and_simulate(fit_session, fit_session.engine, do_fit=False, n_starts=0)
    assert fit_session.optimization is None


def test_fit_skip_reuses_cache(fit_session):
    pytest.importorskip("pypesto", reason="pyPESTO not installed")
    from petab_mcp import fitting as F

    r1 = F.fit_and_simulate(fit_session, fit_session.engine, do_fit=True, n_starts=2)
    assert fit_session.last_fit_cache_hit is False  # first fit: cache miss, ran the optimizer

    # the SAME estimation problem (model/bounds/estimate/n_starts unchanged) is served from
    # the cache — the optimizer is skipped, but the fitted values + objective still match.
    r2 = F.fit_and_simulate(fit_session, fit_session.engine, do_fit=True, n_starts=2)
    assert fit_session.last_fit_cache_hit is True
    for k, v in r1["fitted_parameters"].items():
        assert r2["fitted_parameters"][k] == pytest.approx(v)
    assert r2["neg_log_likelihood"] == pytest.approx(r1["neg_log_likelihood"], rel=1e-6)

    # disabling the cache forces a real re-fit
    fit_session.fit_cache = False
    F.fit_and_simulate(fit_session, fit_session.engine, do_fit=True, n_starts=2)
    assert fit_session.last_fit_cache_hit is False


def test_commit_tracks_improvement_vs_versions(fit_session):
    pytest.importorskip("pypesto", reason="pyPESTO not installed")
    from petab_mcp import operations as O

    # every commit auto-fits; improvement is tracked across committed versions
    first = O.change_parameter_bounds(fit_session, "k1", 1e-3, 1e3, justification="widen k1")
    assert first["fit_status"] == "fit"
    imp1 = first["improvement"]
    assert imp1["objective_before"] is not None
    assert imp1["self_improvement"] is not None
    assert imp1["is_new_best"] is True  # nothing to beat yet
    assert imp1["improvement_vs_best_prior"] is None

    second = O.change_parameter_bounds(fit_session, "k2", 1e-3, 1e3, justification="widen k2")
    imp2 = second["improvement"]
    assert imp2["best_prior_version"] is not None
    assert imp2["improvement_vs_best_prior"] is not None
