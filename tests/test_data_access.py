"""Unit tests for the DuckDB data-access layer (:mod:`petab_mcp.data_access`).

These exercise SQL access to the PEtab tables and the simulation dataframe with a real
session (fixed toy problem) but no AMICI: the ``simulations`` table is driven by a
synthetic ``session.simulation`` payload, so residual analysis — the thing the old handcoded
``fit`` module did — is tested here as plain SQL.
"""

from __future__ import annotations

import math

import pytest

from petab_mcp import data_access
from petab_mcp.errors import ValidationError


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


# obs_A/c0 -> weighted [2.0, 0.0]; obs_B/c1 -> weighted [-1.0] and one unsimulated point.
SIM_RECORDS = [
    _rec("obs_A", "c0", 0.0, 1.0, 0.0, 1.0, 2.0, 0.5),
    _rec("obs_A", "c0", 1.0, 2.0, 2.0, 0.0, 0.0, 0.5),
    _rec("obs_B", "c1", 0.0, 0.0, 1.0, -1.0, -1.0, 1.0),
    _rec("obs_B", "c1", 1.0, 3.0, None, None, None, None),
]


def test_static_tables_queryable_without_simulation(session):
    # No simulation set yet: the PEtab design tables are still fully queryable.
    out = data_access.run_query(session, "SELECT * FROM measurements")
    assert out["row_count"] == 6
    assert "measurement" in out["columns"]

    params = data_access.run_query(
        session, "SELECT parameterId FROM parameters ORDER BY parameterId"
    )
    ids = {r["parameterId"] for r in params["rows"]}
    assert ids == {"A_init", "B_init", "k1", "k2", "sigma_B"}


def test_named_index_promoted_to_column(session):
    # observables/conditions are indexed by their id in petab; it must be a queryable column.
    obs = data_access.run_query(session, "SELECT observableId FROM observables")
    assert obs["row_count"] >= 1 and "observableId" in obs["columns"]
    cond = data_access.run_query(session, "SELECT conditionId FROM conditions")
    assert "conditionId" in cond["columns"]


def test_simulations_table_empty_but_typed(session):
    # Cold start: `simulations` exists with its full schema but no rows.
    out = data_access.run_query(session, "SELECT * FROM simulations")
    assert out["row_count"] == 0
    assert set(data_access.SIMULATION_COLUMNS) <= set(out["columns"])


def test_residual_analysis_via_sql(session):
    session.simulation = {"records": SIM_RECORDS}

    # per-observable chi2 = sum of squared weighted residuals — the old fit.fit_inspection
    # cross section, now a GROUP BY.
    out = data_access.run_query(
        session,
        "SELECT observable_id, count(*) AS n, "
        "sum(residual_weighted*residual_weighted) AS chi2 "
        "FROM simulations WHERE residual IS NOT NULL "
        "GROUP BY observable_id ORDER BY observable_id",
    )
    by_obs = {r["observable_id"]: r for r in out["rows"]}
    assert by_obs["obs_A"]["n"] == 2
    assert by_obs["obs_A"]["chi2"] == pytest.approx(4.0)  # 2^2 + 0^2
    assert by_obs["obs_B"]["n"] == 1
    assert by_obs["obs_B"]["chi2"] == pytest.approx(1.0)

    # overall chi2 across the simulated points
    total = data_access.run_query(
        session, "SELECT sum(residual_weighted*residual_weighted) AS chi2 FROM simulations"
    )
    assert total["rows"][0]["chi2"] == pytest.approx(5.0)


def test_worst_points_via_sql(session):
    session.simulation = {"records": SIM_RECORDS}
    out = data_access.run_query(
        session,
        "SELECT observable_id, time FROM simulations WHERE residual_weighted IS NOT NULL "
        "ORDER BY abs(residual_weighted) DESC LIMIT 2",
    )
    assert out["rows"][0]["observable_id"] == "obs_A" and out["rows"][0]["time"] == 0.0
    assert out["rows"][1]["observable_id"] == "obs_B"


def test_join_measurements_against_simulations(session):
    session.simulation = {"records": SIM_RECORDS}
    # residuals live alongside measurements in the simulations table; a join with the
    # observables table shows arbitrary cross-table SQL works.
    out = data_access.run_query(
        session,
        "SELECT s.observable_id, o.observableFormula "
        "FROM simulations s JOIN observables o ON s.observable_id = o.observableId "
        "GROUP BY s.observable_id, o.observableFormula",
    )
    assert out["row_count"] >= 1


def test_max_rows_truncation(session):
    out = data_access.run_query(session, "SELECT * FROM measurements", max_rows=2)
    assert out["row_count"] == 2 and out["truncated"] is True
    out_all = data_access.run_query(session, "SELECT * FROM measurements", max_rows=1000)
    assert out_all["truncated"] is False


def test_bad_sql_raises_validation_error(session):
    with pytest.raises(ValidationError):
        data_access.run_query(session, "SELECT * FROM nonexistent_table")
    with pytest.raises(ValidationError):
        data_access.run_query(session, "this is not sql")


def test_bad_arguments_rejected(session):
    with pytest.raises(ValidationError):
        data_access.run_query(session, "")
    with pytest.raises(ValidationError):
        data_access.run_query(session, "SELECT 1", max_rows=0)


def test_query_runs_on_disposable_copy(session):
    # Arbitrary DML/DDL is allowed because it hits a private copy, not the session.
    df_before = session.problem.measurement_df.copy()

    data_access.run_query(session, "DELETE FROM measurements")
    data_access.run_query(session, "DROP TABLE parameters")
    data_access.run_query(session, "UPDATE observables SET observableFormula = 'x'")

    # the source DataFrame is untouched ...
    assert len(session.problem.measurement_df) == len(df_before)
    assert session.problem.measurement_df.equals(df_before)
    # ... and every call gets a fresh copy, so the tables are all still there and intact.
    assert data_access.run_query(
        session, "SELECT count(*) AS n FROM measurements"
    )["rows"][0]["n"] == 6
    assert data_access.run_query(
        session, "SELECT count(*) AS n FROM parameters"
    )["rows"][0]["n"] == 5


def test_multi_statement_and_temp_tables_allowed(session):
    # The full SQL surface is available; the last statement's result is returned.
    out = data_access.run_query(
        session,
        "CREATE TEMP TABLE t AS SELECT observableId FROM measurements; "
        "SELECT count(*) AS n FROM t",
    )
    assert out["rows"][0]["n"] == 6


def test_external_file_access_blocked(session):
    # Copying isolates the data but not the host, so file/network access is disabled: this
    # must fail (wrapped as a ValidationError) rather than read a local file.
    with pytest.raises(ValidationError):
        data_access.run_query(session, "SELECT * FROM read_csv_auto('/etc/hostname')")


def test_table_schema(session):
    schema = data_access.table_schema(session)
    tables = {t["table"]: t for t in schema["tables"]}
    assert set(tables) == {
        "measurements", "conditions", "observables", "parameters", "simulations",
        "optimization_starts", "optimization_trace",
    }
    assert tables["measurements"]["n_rows"] == 6
    assert tables["simulations"]["n_rows"] == 0  # nothing simulated yet
    assert set(data_access.SIMULATION_COLUMNS) <= set(tables["simulations"]["columns"])
    # optimization tables are empty (no fit yet) but carry their stable schema
    assert tables["optimization_starts"]["n_rows"] == 0
    assert tables["optimization_trace"]["n_rows"] == 0
    assert set(data_access.OPTIMIZATION_START_COLUMNS) <= set(
        tables["optimization_starts"]["columns"]
    )


def test_optimization_tables_empty_without_fit(session):
    starts = data_access.run_query(session, "SELECT * FROM optimization_starts")
    trace = data_access.run_query(session, "SELECT * FROM optimization_trace")
    assert starts["row_count"] == 0 and trace["row_count"] == 0
    assert set(data_access.OPTIMIZATION_TRACE_COLUMNS) <= set(trace["columns"])


def test_optimization_tables_query(session):
    # Inject a synthetic pyPESTO payload (2 starts, a short fval trace each) — the shape
    # fitting._optimization_payload produces — and query it as SQL.
    session.optimization = {
        "n_starts": 2,
        "starts": [
            {"start": 0, "id": "0", "fval": 1.0, "time": 0.5, "n_fval": 20,
             "n_grad": 20, "n_hess": 0, "exitflag": 0, "message": "converged"},
            {"start": 1, "id": "1", "fval": 5.0, "time": 0.4, "n_fval": 15,
             "n_grad": 15, "n_hess": 0, "exitflag": 0, "message": "converged"},
        ],
        "trace": [
            {"start": 0, "iteration": 0, "fval": 9.0, "time": 0.0},
            {"start": 0, "iteration": 1, "fval": 3.0, "time": 0.2},
            {"start": 0, "iteration": 2, "fval": 1.0, "time": 0.5},
            {"start": 1, "iteration": 0, "fval": 9.0, "time": 0.0},
            {"start": 1, "iteration": 1, "fval": 5.0, "time": 0.4},
        ],
    }

    # best start and the multistart spread — a convergence check in SQL
    best = data_access.run_query(
        session, "SELECT min(fval) AS best, count(*) AS n FROM optimization_starts"
    )
    assert best["rows"][0] == {"best": 1.0, "n": 2}

    # monotone objective descent along the best start's trace
    desc = data_access.run_query(
        session,
        "SELECT fval FROM optimization_trace WHERE start = 0 ORDER BY iteration",
    )
    assert [r["fval"] for r in desc["rows"]] == [9.0, 3.0, 1.0]


def test_nan_values_cleaned_to_none(session):
    session.simulation = {"records": [_rec("o", "c", 0.0, 1.0, float("nan"), None, None, None)]}
    out = data_access.run_query(session, "SELECT simulation FROM simulations")
    # NaN is not valid JSON; it must come back as None.
    val = out["rows"][0]["simulation"]
    assert val is None or (isinstance(val, float) and not math.isnan(val))


# -- fit_diagnosis: the compact, bounded residual read for the lead ---------
def test_fit_diagnosis_empty_without_simulation(session):
    out = data_access.fit_diagnosis(session)
    assert out["per_observable"] == [] and out["worst_points"] == []
    assert "no simulation" in out["status"]


def test_fit_diagnosis_compact_and_bounded(session):
    session.simulation = {"records": SIM_RECORDS}
    out = data_access.fit_diagnosis(session, worst_n=2)
    assert out["n_points"] == 3                          # the NULL-residual row is excluded
    assert out["total_chi2"] == pytest.approx(5.0)       # 2^2 + 0^2 + (-1)^2
    per = {r["observable_id"]: r for r in out["per_observable"]}
    assert per["obs_A"]["n"] == 2 and per["obs_A"]["chi2"] == pytest.approx(4.0)
    assert per["obs_A"]["mean_weighted_residual"] == pytest.approx(1.0)
    assert per["obs_A"]["max_abs_weighted_residual"] == pytest.approx(2.0)
    assert per["obs_B"]["mean_weighted_residual"] == pytest.approx(-1.0)  # systematic bias
    # per_observable ranked worst-chi2 first
    assert out["per_observable"][0]["observable_id"] == "obs_A"
    # worst_points bounded to worst_n and ordered by |residual_weighted|
    assert len(out["worst_points"]) == 2
    assert abs(out["worst_points"][0]["residual_weighted"]) == pytest.approx(2.0)
