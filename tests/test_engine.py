from __future__ import annotations

import pandas as pd
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


def test_replicate_measurements_not_double_counted(engine, session):
    """Replicate measurements must not inflate n_data_points / chi2.

    ``_simulation_records`` used to merge the measurement and simulation tables on
    (observableId, simulationConditionId, time) alone. For replicate rows that share
    a key, that cross-joins each size-r group into r*r rows, so a table with r=2
    replicates reported ``n_data_points == sum(r**2) == 24`` (and a ~2x-inflated chi2)
    instead of the true 12. The merge is now one-to-one with the measurements.
    """
    import numpy as np

    base = session.problem.measurement_df
    # Two replicates per point (distinct measured values, identical key): 6 groups of
    # size 2 => 12 true points, which the buggy merge would have blown up to 24.
    dup = base.copy()
    dup["measurement"] = dup["measurement"] + 0.01
    rep = pd.concat([base, dup], ignore_index=True)
    session.problem.measurement_df = rep

    obj = engine.evaluate_objective()
    # one record per measurement row (Sum r), not the buggy cross-join (Sum r**2)
    assert obj["n_data_points"] == len(rep) == 2 * len(base)
    assert len(obj["simulations"]) == len(rep)

    # every (obs, cond, time) group has exactly its 2 replicates, sharing one simulation
    groups: dict[tuple, list[float]] = {}
    for r in obj["simulations"]:
        groups.setdefault((r["observable_id"], r["condition_id"], r["time"]), []).append(r["simulation"])
    assert {len(v) for v in groups.values()} == {2}
    for sims in groups.values():
        assert sims[0] == pytest.approx(sims[1])  # replicates simulate identically

    # chi2 counts each weighted residual once, not r*r times: it equals the sum over
    # the records and stays consistent with AMICI's likelihood. Derive the noise term
    # from the per-record noise (not a hard-coded sigma) so the identity holds even if
    # the toy problem's noise model or size changes.
    weighted = np.array([r["residual_weighted"] for r in obj["simulations"]], dtype=float)
    noise = np.array([r["noise"] for r in obj["simulations"]], dtype=float)
    assert obj["chi2"] == pytest.approx(float(np.sum(weighted**2)), rel=1e-9)
    nllh = 0.5 * (obj["chi2"] + float(np.sum(np.log(2 * np.pi * noise**2))))
    assert nllh == pytest.approx(obj["neg_log_likelihood"], rel=1e-6)


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


def test_simulate_after_operation_recompiles(engine, session, reauthor_toy):
    before = engine.evaluate_objective()["neg_log_likelihood"]
    reauthor_toy(session, ["J: B => D; k3 * B;"], justification="B to D", references=["PMID:1"])
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
    # fitted_parameters holds ONLY the estimated params (k1, k2) — never the fixed
    # sigma_B/A_init/B_init — so k = len(fitted) equals n_estimated and the scores stay honest.
    assert set(result["fitted_parameters"]) == {"k1", "k2"}
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


def test_fit_tracks_improvement_vs_versions(fit_session):
    pytest.importorskip("pypesto", reason="pyPESTO not installed")
    from petab_mcp import operations as O

    # edits are fit-free; `fit` scores them and tracks improvement across the fitted versions
    O.change_parameter_bounds(fit_session, "k1", 1e-3, 1e3, justification="widen k1")
    first = fit_session.fit()
    assert first["fit_status"] == "fit"
    imp1 = first["improvement"]
    assert imp1["objective_before"] is not None
    assert imp1["self_improvement"] is not None
    assert imp1["is_new_best"] is True  # nothing to beat yet
    assert imp1["improvement_vs_best_prior"] is None

    O.change_parameter_bounds(fit_session, "k2", 1e-3, 1e3, justification="widen k2")
    second = fit_session.fit()
    imp2 = second["improvement"]
    assert imp2["best_prior_version"] is not None
    assert imp2["improvement_vs_best_prior"] is not None


def test_model_less_problem_does_not_segfault(session, engine):
    """A zero-shot problem before any set_model has no species; handing its empty model to
    AMICI segfaults the server (blasi's 0/5 sweep failure). The engine must refuse it and the
    simulate path must degrade gracefully, WITHOUT invoking AMICI at all (so this test needs no
    compile)."""
    from petab_mcp import fitting
    from petab_mcp.errors import ModelUpdateError
    from petab_mcp.model import SbmlModel

    session.problem.model = SbmlModel.from_antimony("model m\nend")
    assert not session.has_model()

    # _get_model refuses rather than handing an empty model to AMICI (which would SIGSEGV).
    with pytest.raises(ModelUpdateError):
        engine._get_model()

    # ensure_simulation short-circuits (no AMICI import) and leaves no simulation.
    assert fitting.ensure_simulation(session, engine) is None

    # fit_summary returns cleanly with None metrics — the judge scores this as "no valid fit".
    fs = fitting.fit_summary(session, engine)
    assert fs.get("neg_log_likelihood") is None
    assert fs.get("bic") is None


def test_fit_objective_and_reported_nll_are_the_same_number(fit_session):
    """The fit's objective and the harness's reported nll must agree exactly.

    They are the same likelihood at the same parameters, but they used to be computed by two
    different AMICI models with two different solvers — the fit through pyPESTO's objective,
    the report by re-simulating via simulate_petab. That let them disagree: a crauste fit
    converged at nll 266.62 while the re-simulation of its own optimum failed to integrate past
    t=22 and reported NaN, which then poisoned the best-BIC checkpoint and cost the run its
    score. Both now go through one shared objective (fitting.estimation_problem), so this is an
    identity, not an approximation.
    """
    from petab_mcp import fitting

    res = fit_session.fit(justification="test")
    reported = res["neg_log_likelihood"]
    assert reported is not None

    # Re-evaluating the committed model must reproduce the number that was reported for it.
    again = fit_session.engine.evaluate_objective()["neg_log_likelihood"]
    assert again == pytest.approx(reported, rel=1e-9)

    # ...and that number is the optimizer's own objective at the fitted point.
    problem, _ = fitting.estimation_problem(fit_session, fit_session.engine)
    x = fitting.problem_x_from_nominal(problem, fit_session.problem.parameter_df)
    assert float(problem.objective(x)) == pytest.approx(reported, rel=1e-9)


def test_one_amici_build_per_structure(fit_session):
    """The fit and the objective evaluation must share ONE compiled model.

    They used to build two (`petab_mcp_<hash>` and `petab_mcp_pp_<hash>`) from the same SBML —
    at ~80s a build on a cluster node, that was ~20 minutes of a 73-minute benchmark run spent
    compiling the same structure twice.
    """
    fit_session.fit(justification="test")
    n_after_first = fit_session.n_compiles
    # A re-fit of the SAME structure must not compile again...
    fit_session.engine.evaluate_objective()
    fit_session.fit(justification="test again")
    assert fit_session.n_compiles == n_after_first
    # ...and the objective's model is the one the engine reports on.
    problem, _ = __import__("petab_mcp.fitting", fromlist=["x"]).estimation_problem(
        fit_session, fit_session.engine)
    assert problem.objective.amici_model is not None
