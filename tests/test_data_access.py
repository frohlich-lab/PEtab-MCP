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


def _rec(obs, exp, t, m, s, res, w, noise):
    return {
        "observable_id": obs,
        "experiment_id": exp,
        "time": t,
        "measurement": m,
        "simulation": s,
        "residual": res,
        "residual_weighted": w,
        "noise": noise,
    }


# obs_A/e0 -> weighted [2.0, 0.0]; obs_B/e1 -> weighted [-1.0] and one unsimulated point.
SIM_RECORDS = [
    _rec("obs_A", "e0", 0.0, 1.0, 0.0, 1.0, 2.0, 0.5),
    _rec("obs_A", "e0", 1.0, 2.0, 2.0, 0.0, 0.0, 0.5),
    _rec("obs_B", "e1", 0.0, 0.0, 1.0, -1.0, -1.0, 1.0),
    _rec("obs_B", "e1", 1.0, 3.0, None, None, None, None),
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


def test_empty_parameter_table_does_not_break_queries(session):
    # A model-less problem ships a header-only parameters.tsv, so the parameter
    # table has zero rows — which turned into a zero-column DataFrame that DuckDB refuses to
    # register, making EVERY query_data call fail (even `SELECT 1`). Regression: with no
    # parameters, `SELECT 1` and the fixed tables must still be queryable.
    session.problem.parameter_df = session.problem.parameter_df.iloc[0:0]
    assert len(session.problem.parameter_df) == 0

    assert data_access.run_query(session, "SELECT 1 AS one")["rows"][0]["one"] == 1

    meas = data_access.run_query(session, "SELECT * FROM measurements")
    assert meas["row_count"] == 6 and "measurement" in meas["columns"]

    # `TABLE measurements` (DuckDB shorthand for SELECT *) works too — the exact call the
    # trace-review found the agent repeatedly proposing and failing.
    assert data_access.run_query(session, "TABLE measurements")["row_count"] == 6

    # the empty parameters table is still there, queryable, with its canonical columns
    params = data_access.run_query(session, "SELECT * FROM parameters")
    assert params["row_count"] == 0
    assert set(data_access.PARAMETER_COLUMNS) <= set(params["columns"])


def test_absent_optional_table_does_not_break_queries(session):
    # An optional PEtab table petab loaded as None (e.g. no condition table) must not break
    # the query surface either: it becomes an empty, canonically-typed table.
    session.problem.condition_df = None
    out = data_access.run_query(session, "SELECT 1 AS one")
    assert out["rows"][0]["one"] == 1
    cond = data_access.run_query(session, "SELECT * FROM conditions")
    assert cond["row_count"] == 0
    assert set(data_access.CONDITION_COLUMNS) <= set(cond["columns"])


def test_empty_parameter_table_schema_keeps_columns(session):
    # list_data_tables must still expose the parameter columns before any parameter exists.
    session.problem.parameter_df = session.problem.parameter_df.iloc[0:0]
    tables = {t["table"]: t for t in data_access.table_schema(session)["tables"]}
    assert tables["parameters"]["n_rows"] == 0
    assert set(data_access.PARAMETER_COLUMNS) <= set(tables["parameters"]["columns"])


def test_sql_error_points_at_list_data_tables_only(session):
    # The SQL error names list_data_tables, query_data's companion, and no other tool.
    with pytest.raises(ValidationError) as excinfo:
        data_access.run_query(session, "SELECT * FROM nonexistent_table")
    msg = str(excinfo.value)
    assert "list_data_tables" in msg and "get_measurements" not in msg


def test_named_index_promoted_to_column(session):
    # observables are indexed by their id; it must be a queryable column. The condition and
    # experiment tables keep their keys as plain columns, so they are queryable even when empty.
    obs = data_access.run_query(session, "SELECT observableId FROM observables")
    assert obs["row_count"] >= 1 and "observableId" in obs["columns"]
    cond = data_access.run_query(session, "SELECT conditionId FROM conditions")
    assert "conditionId" in cond["columns"]
    exps = data_access.run_query(session, "SELECT experimentId FROM experiments")
    assert "experimentId" in exps["columns"]


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
    assert (
        data_access.run_query(session, "SELECT count(*) AS n FROM measurements")["rows"][0]["n"]
        == 6
    )
    assert (
        data_access.run_query(session, "SELECT count(*) AS n FROM parameters")["rows"][0]["n"] == 5
    )


def test_multi_statement_and_temp_tables_allowed(session):
    # The full SQL surface is available; the last statement's result is returned.
    out = data_access.run_query(
        session,
        "CREATE TEMP TABLE t AS SELECT observableId FROM measurements; SELECT count(*) AS n FROM t",
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
        "measurements",
        "conditions",
        "experiments",
        "observables",
        "parameters",
        "simulations",
        "simulations_history",
        "state_trajectories",
        "optimization_starts",
        "optimization_trace",
    }
    assert tables["simulations_history"]["n_rows"] == 0  # no versions stored yet
    assert tables["state_trajectories"]["n_rows"] == 0
    assert tables["state_trajectories"]["columns"] == data_access.STATE_TRAJECTORY_COLUMNS
    assert set(data_access.SIMULATION_HISTORY_COLUMNS) <= set(
        tables["simulations_history"]["columns"]
    )
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
            {
                "start": 0,
                "id": "0",
                "fval": 1.0,
                "time": 0.5,
                "n_fval": 20,
                "n_grad": 20,
                "n_hess": 0,
                "exitflag": 0,
                "message": "converged",
            },
            {
                "start": 1,
                "id": "1",
                "fval": 5.0,
                "time": 0.4,
                "n_fval": 15,
                "n_grad": 15,
                "n_hess": 0,
                "exitflag": 0,
                "message": "converged",
            },
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


# -- fit_diagnosis: the compact, bounded residual read for the agent --------
def _inner_solved_toy(session, *, sigma: bool, offset: bool):
    """The toy on 10 time points with fontaine's observation model, ``obs_B = B + offset_B`` with
    noise ``sigma_B``, either one estimated or fixed, and records as an inner-solved offset and
    sigma leave them: residuals centred, weighted chi2 equal to the point count."""
    import pandas as pd

    p = session.problem
    times = [float(t) for t in range(10)]
    mdf = pd.DataFrame({"observableId": "obs_B", "experimentId": "", "time": times})
    mdf["measurement"] = 1.0
    odf = p.observable_df
    if offset:
        odf = odf.assign(
            observableFormula="B + observableOffset_obs_B",
            observablePlaceholders="observableOffset_obs_B",
        )
        mdf["observableParameters"] = "offset_B"
    p = session.problem = p.replace(observable_df=odf, measurement_df=mdf)
    p.parameter_df.loc["offset_B"] = p.parameter_df.loc["k1"]
    columns = ["parameterScale", "lowerBound", "upperBound", "nominalValue", "estimate"]
    p.parameter_df.loc["offset_B", columns] = ["lin", -10, 10, 0.0, offset]
    p.parameter_df.loc["sigma_B", "estimate"] = sigma
    residuals = [0.1, 0.2, -0.15, 0.05, -0.1, -0.2, 0.3, -0.25, 0.15, -0.1]  # sum 0, 8 runs
    sd = math.sqrt(sum(r * r for r in residuals) / len(residuals))
    session.simulation = {
        "records": [
            _rec("obs_B", "", t, 1.0, 1.0 - r, r, r / sd, sd) for t, r in zip(times, residuals)
        ]
    }


@pytest.mark.parametrize("sigma", [True, False])
@pytest.mark.parametrize("offset", [True, False])
def test_goodness_of_fit_that_holds_by_construction_says_so(session, sigma, offset):
    """#267: with per-batch offsets and inner-solved sigmas every fontaine diagnosis read
    total_chi2 = n (640), mean residual 0 and sign-test p 1.0 with low_power false, and agents cited
    "zero mean bias" against readout changes. An estimated offset marks its observable's row
    low-power and says why; a sigma solved over every point notes total_chi2. Neither is said where
    the parameter is fixed, though the numbers are the same."""
    _inner_solved_toy(session, sigma=sigma, offset=offset)
    out = data_access.fit_diagnosis(session)
    assert out["total_chi2"] == pytest.approx(10.0)
    assert ("total_chi2_note" in out) is sigma
    (row,) = out["per_observable"]
    assert row["mean_weighted_residual"] == pytest.approx(0.0)
    assert ("by_construction" in row) is offset
    assert row["structure_flag"] is False
    # 10 points are enough for both tests, unless an offset centred them
    assert row["sign_test"]["low_power"] is row["structure_low_power"] is offset
    assert data_access.residual_structure_rollup(session)["low_power"] is offset


def test_fit_diagnosis_empty_without_simulation(session):
    out = data_access.fit_diagnosis(session)
    assert out["per_observable"] == [] and out["worst_points"] == []
    assert "no simulation" in out["status"]
    # the rolled-up verdict is NOT here — it rides on the fit result, and the degraded path must
    # not resurrect a second copy of it
    assert "residual_structure" not in out


def test_fit_diagnosis_compact_and_bounded(session):
    session.simulation = {"records": SIM_RECORDS}
    out = data_access.fit_diagnosis(session, worst_n=2)
    assert out["n_points"] == 3  # the NULL-residual row is excluded
    assert out["total_chi2"] == pytest.approx(5.0)  # 2^2 + 0^2 + (-1)^2
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
    # residual-structure tests are attached PER OBSERVABLE — that is the detail this read exists
    # for — while the rolled-up verdict over them stays on the fit result (~0.5 kB not shipped
    # twice to a caller that has both)
    assert "sign_test" in per["obs_A"] and "runs_test" in per["obs_A"]
    assert "residual_structure" not in out
    # too few points here -> the rollup calls the tests underpowered, so "nothing flagged" is not
    # a pass; it is read off the fit result's verdict
    assert data_access.residual_structure_rollup(session)["low_power"] is True


#: Two species and an assignment rule at two times of experiment e0, as the engine stores them.
TRAJECTORIES = {
    "ids": ["A", "B", "total"],
    "points": [["e0", 0.0], ["e0", 1.0]],
    "values": [[1.0, 0.0, 1.0], [0.6, 0.4, 1.0]],
}


def test_state_trajectories_are_one_row_per_value(session):
    """#289: the hidden states a fit simulated are SQL, not reconstructed by hand from readouts."""
    session.simulation = {"records": SIM_RECORDS, "state_trajectories": TRAJECTORIES}
    out = data_access.run_query(
        session,
        "SELECT time, value FROM state_trajectories WHERE entity_id = 'B' ORDER BY time",
    )
    assert [(r["time"], r["value"]) for r in out["rows"]] == [(0.0, 0.0), (1.0, 0.4)]
    tables = {t["table"]: t for t in data_access.table_schema(session)["tables"]}
    assert tables["state_trajectories"]["n_rows"] == 6


def test_state_trajectories_are_kept_with_a_version_and_left_out_of_its_summary(session):
    """A restore reads a version's stored simulation back, so the trajectories must be stored with
    it; a get_version reply too large to send drops them with the records."""
    from petab_mcp import fitting

    stored = fitting.build_sim_payload_for_version(
        {"records": SIM_RECORDS, "state_trajectories": TRAJECTORIES}
    )
    assert stored["state_trajectories"] == TRAJECTORIES
    summary = data_access.version_part_summary("simulation", stored, 1)
    assert "state_trajectories" not in summary and "records" not in summary


def test_fit_diagnosis_flags_systematic_bias(session):
    """A one-sided residual series must be flagged by the sign test through fit_diagnosis."""
    session.simulation = {
        "records": [_rec("obs_A", "e0", float(t), 1.0, 0.0, 1.0, 1.0, 1.0) for t in range(10)]
    }
    out = data_access.fit_diagnosis(session)
    row = out["per_observable"][0]
    assert row["sign_test"]["n_pos"] == 10 and row["sign_test"]["n_neg"] == 0
    assert row["sign_test"]["p"] is not None and row["sign_test"]["p"] < 0.01
    assert row["structure_flag"] is True
    rollup = data_access.residual_structure_rollup(session)
    assert "obs_A" in rollup["flagged_observables"]
    assert rollup["low_power"] is False  # 10 points -> powered


# -- the bare structure boolean the version meta records -------------------------------------
# `PetabSession.fit` derives it as bool(rollup["flagged_observables"]) rather than calling a
# second helper; these pin that expression against fit_diagnosis's own flagged set.
def _flagged(session) -> bool:
    return bool(data_access.residual_structure_rollup(session)["flagged_observables"])


def test_structure_flag_true_on_systematic_bias(session):
    """A one-sided residual series is flagged, and the per-observable rows fit_diagnosis returns
    mark the SAME observable — pinning the two definitions together (they share one builder) so the
    per-commit gate can't drift from the detail read."""
    session.simulation = {
        "records": [_rec("obs_A", "e0", float(t), 1.0, 0.0, 1.0, 1.0, 1.0) for t in range(10)]
    }
    assert _flagged(session) is True
    assert _flagged(session) is any(
        o.get("structure_flag") for o in data_access.fit_diagnosis(session)["per_observable"]
    )


def test_structure_flag_false_without_simulation(session):
    """No committed simulation -> empty frame -> not flagged (the gate must never crash on it)."""
    assert _flagged(session) is False


def test_structure_flag_false_when_no_weighted_residuals(session):
    """Records exist but carry no weighted residual (unfitted / non-evaluable noise) -> not flagged,
    matching fit_diagnosis, which reports no residual structure for the same input."""
    session.simulation = {"records": [_rec("o", "c", 0.0, 1.0, float("nan"), None, None, None)]}
    assert _flagged(session) is False
    # fit_diagnosis short-circuits here with no rows and no verdict; the (unavailable) verdict is
    # the fit result's, and the REPLICATE arm reaches it there — a collapsed/unevaluable sigma must
    # not switch off the one arm that catches it just because this table needs weighted residuals.
    diag = data_access.fit_diagnosis(session)
    assert diag["per_observable"] == [] and "residual_structure" not in diag
    assert not data_access.residual_structure_rollup(session)["flagged_observables"]


def test_structure_flag_agrees_with_fit_diagnosis_on_mixed(session):
    """On the shared mixed record set (few points -> underpowered -> not flagged), the gate's
    boolean equals what fit_diagnosis's own rows mark."""
    session.simulation = {"records": SIM_RECORDS}
    assert _flagged(session) is any(
        o.get("structure_flag") for o in data_access.fit_diagnosis(session)["per_observable"]
    )


# -- residual_structure_rollup: the compact verdict returned ON the fit result (issue #113) ------
def test_residual_structure_rollup_names_the_flagged_observable(session):
    """The rollup is the verdict `fit` returns, so it must NAME what is structured, not just say
    that something is — and agree exactly with fit_diagnosis's own rollup (one key name, one
    shape, one set of rules)."""
    session.simulation = {
        "records": [_rec("obs_A", "e0", float(t), 1.0, 0.0, 1.0, 1.0, 1.0) for t in range(10)]
    }
    out = data_access.residual_structure_rollup(session)
    assert out["flagged_observables"] == ["obs_A"]
    assert out["low_power"] is False
    assert "obs_A" in out["summary"]
    assert [
        o["observable_id"]
        for o in data_access.fit_diagnosis(session)["per_observable"]
        if o.get("structure_flag")
    ] == ["obs_A"]


def test_residual_structure_rollup_reports_low_power_on_the_mixed_set(session):
    """Few points -> nothing flagged but the tests are underpowered, which the summary must say
    outright: "nothing flagged" here is weak evidence, not a pass."""
    session.simulation = {"records": SIM_RECORDS}
    out = data_access.residual_structure_rollup(session)
    assert out["flagged_observables"] == []
    assert out["low_power"] is True
    assert "UNDERPOWERED" in out["summary"]
    assert not any(
        o.get("structure_flag") for o in data_access.fit_diagnosis(session)["per_observable"]
    )


@pytest.mark.parametrize(
    "records",
    [None, [], [_rec("o", "c", 0.0, 1.0, float("nan"), None, None, None)]],
    ids=["no-simulation", "no-records", "no-weighted-residuals"],
)
def test_residual_structure_rollup_degrades_to_a_well_formed_verdict(session, records):
    """No fit to test -> every key still present (the agent reads this off the fit result, so a
    missing key would be a KeyError), flagging nothing but NOT claiming adequacy."""
    session.simulation = None if records is None else {"records": records}
    out = data_access.residual_structure_rollup(session)
    assert out["flagged_observables"] == [] and out["low_power"] is True
    assert "unavailable" in out["summary"]


def test_residual_structure_rollup_never_raises(session, monkeypatch):
    """An advisory stat must never take down the fit that computes it."""
    monkeypatch.setattr(
        data_access, "_frames", lambda _s: (_ for _ in ()).throw(RuntimeError("boom"))
    )
    out = data_access.residual_structure_rollup(session)
    assert out["flagged_observables"] == [] and out["low_power"] is True
    assert _flagged(session) is False


def test_residual_structure_rollup_agrees_with_fit_diagnosis_on_several_flagged(session):
    """The two callers walk the observables in DIFFERENT orders — fit_diagnosis marks a chi2-ranked
    per_observable, the rollup walks groupby order — so with more than one flagged observable the
    two must agree on WHICH observables are flagged, or the fit result's verdict and the rows the
    agent reads next would name different ones."""
    # obs_B carries the larger chi2, so fit_diagnosis ranks it first while groupby yields obs_A
    # first; both one-sided, so both flag.
    records = [_rec("obs_A", "e0", float(t), 1.0, 0.0, 1.0, 1.0, 1.0) for t in range(10)] + [
        _rec("obs_B", "e0", float(t), 3.0, 0.0, 3.0, 3.0, 1.0) for t in range(10)
    ]
    session.simulation = {"records": records}
    diag = data_access.fit_diagnosis(session)
    assert [o["observable_id"] for o in diag["per_observable"]] == ["obs_B", "obs_A"]

    rollup = data_access.residual_structure_rollup(session)
    assert rollup["flagged_observables"] == ["obs_A", "obs_B"]
    assert (
        sorted(o["observable_id"] for o in diag["per_observable"] if o.get("structure_flag"))
        == rollup["flagged_observables"]
    )


def test_a_fit_result_names_three_flagged_observables_and_counts_the_rest():
    """#190: the rollup named every flagged observable twice, in the list and in the summary, and
    carried a replicate arm with nothing to test, on every fit result. get_fit_summary keeps the
    whole list."""
    from petab_mcp import residual_tests as rt

    silent = data_access._lof_payload(rt.replicate_lof_unavailable("no_replicated_points"))
    rollup = data_access._rollup_structure(
        [(f"obs_{i}", {"structure_flag": True}) for i in range(5)], replicate_lof=silent
    )
    assert rollup["flagged_observables"] == [f"obs_{i}" for i in range(5)]
    assert "in obs_0, obs_1, obs_2 (+2 more) (" in rollup["summary"]

    on_fit = data_access.residual_structure_on_fit(rollup)
    assert on_fit["flagged_observables"] == ["obs_0", "obs_1", "obs_2"]
    assert on_fit["n_flagged_observables"] == 5
    assert "replicate_lack_of_fit" not in on_fit
    assert "replicate_lack_of_fit" in rollup  # a projection, not an edit of the rollup


def test_the_two_rollup_callers_produce_IDENTICAL_verdicts(session, monkeypatch):
    """Not merely the same flagged SET: the same ``summary`` and the same ``low_power`` too.

    ``low_power`` is the field that stops "nothing flagged" being read as a pass, and it reaches
    a caller's record from BOTH paths — the per-row ``structure_low_power`` from fit_diagnosis's
    own pass, the rolled-up one from the fit path — so the two diverging is a silent over-claim.
    The verdicts are captured AT the shared builder, so this pins what each caller actually hands
    it (the inputs are what could drift now that the builder is shared) rather than a
    reconstruction of one.
    """
    verdicts: list[dict] = []
    real = data_access._rollup_structure
    monkeypatch.setattr(
        data_access,
        "_rollup_structure",
        lambda *a, **kw: verdicts.append(real(*a, **kw)) or verdicts[-1],
    )

    # Two flagged observables reached in DIFFERENT orders per caller (fit_diagnosis marks a
    # chi2-ranked table, the rollup walks groupby order), one underpowered set, and one with nothing
    # testable — so both the list and the summary string built from it have to agree.
    two_flagged = [_rec("obs_A", "e0", float(t), 1.0, 0.0, 1.0, 1.0, 1.0) for t in range(10)] + [
        _rec("obs_B", "e0", float(t), 3.0, 0.0, 3.0, 3.0, 1.0) for t in range(10)
    ]
    for records in (
        [_rec("obs_A", "e0", float(t), 1.0, 0.0, 1.0, 1.0, 1.0) for t in range(10)],
        two_flagged,
        SIM_RECORDS,
        [_rec("o", "c", 0.0, 1.0, float("nan"), None, None, None)],
    ):
        verdicts.clear()
        session.simulation = {"records": records}  # no stored verdict -> both paths recompute
        data_access.residual_structure_rollup(session)
        data_access.fit_diagnosis(session)
        if len(verdicts) == 2:  # the degenerate frames short-circuit before the builder
            assert verdicts[0] == verdicts[1]
            assert verdicts[0]["summary"] == verdicts[1]["summary"]
            assert verdicts[0]["low_power"] == verdicts[1]["low_power"]


def test_the_stored_verdict_is_reused_rather_than_recomputed(session, monkeypatch):
    """fit_diagnosis marked its rows by rolling the verdict up AGAIN — a whole replicate
    lack-of-fit pass per noise parameter — although `session.fit` already stored that rollup on the
    very frame this reads (`fitsim` IS `session.simulation`). It reads the stored one, and recomputes
    only when the frame carries none (a simulate-only or restored payload)."""
    session.simulation = {
        "records": [_rec("obs_A", "e0", float(t), 1.0, 0.0, 1.0, 1.0, 1.0) for t in range(10)]
    }
    stored = data_access.residual_structure_rollup(session)
    session.simulation["residual_structure"] = stored

    passes: list[int] = []
    real = data_access._replicate_lof
    monkeypatch.setattr(
        data_access, "_replicate_lof", lambda s, df: passes.append(1) or real(s, df)
    )

    diag = data_access.fit_diagnosis(session)
    assert [
        o["observable_id"] for o in diag["per_observable"] if o.get("structure_flag")
    ] == stored["flagged_observables"]
    assert passes == []  # nothing re-tested

    # ...and with no stored verdict it still marks the same rows, by recomputing
    session.simulation.pop("residual_structure")
    again = data_access.fit_diagnosis(session)
    assert [
        o["observable_id"] for o in again["per_observable"] if o.get("structure_flag")
    ] == stored["flagged_observables"]
    assert passes == [1]


def test_residual_flag_separates_untested_from_unflagged(session):
    """`False` must mean "tested, nothing found"; an unavailable verdict is `None`. Recording the
    two the same way would turn "we never ran the tests" into a clean bill of health."""
    session.simulation = None
    assert data_access.residual_flag(data_access.residual_structure_rollup(session)) is None

    session.simulation = {"records": SIM_RECORDS}
    assert data_access.residual_flag(data_access.residual_structure_rollup(session)) is False

    session.simulation = {
        "records": [_rec("obs_A", "e0", float(t), 1.0, 0.0, 1.0, 1.0, 1.0) for t in range(10)]
    }
    assert data_access.residual_flag(data_access.residual_structure_rollup(session)) is True


def test_fit_diagnosis_caps_per_observable_at_worst_12(session):
    """One-row-per-observable stops being compact at scale: on a 65-observable problem this
    table alone was ~6-8 kB per call, re-sent in the caller's history every later turn — the
    agent's history there grew ~15.5k tokens/turn and 3/9 sessions died on the context window. The
    cap keeps the worst-12 by chi2 plus EVERY structure-flagged row (a cap must never hide a
    verdict), and the omission note keeps a truncated table from reading as complete."""
    recs = []
    for i in range(20):
        # descending chi2 with i: obs_00 worst. 4 points each so the structure tests can run.
        for t, r in ((0.0, 2.0), (1.0, -2.0), (2.0, 2.0), (3.0, -2.0)):
            recs.append(
                {
                    "observable_id": f"obs_{i:02d}",
                    "experiment_id": "e0",
                    "time": t,
                    "measurement": 1.0,
                    "simulation": 1.0,
                    "residual_weighted": r * (20 - i) / 20,
                }
            )
    session.simulation = {
        "records": recs,
        # a stored verdict flagging a LOW-chi2 observable, to prove retention
        "residual_structure": {"flagged_observables": ["obs_19"]},
    }
    out = data_access.fit_diagnosis(session)
    ids = [r["observable_id"] for r in out["per_observable"]]
    assert len(ids) == 13  # worst 12 + the flagged straggler
    assert ids[:12] == [f"obs_{i:02d}" for i in range(12)]  # worst-first by chi2
    assert "obs_19" in ids  # flagged row survives the cap
    assert out["omitted"]["n_observables_omitted"] == 7
    assert out["omitted"]["omitted_chi2_total"] > 0
    # small problems are untouched: no omission note, all rows present
    session.simulation = {"records": recs[:16]}  # 4 observables only
    out2 = data_access.fit_diagnosis(session)
    assert len(out2["per_observable"]) == 4 and "omitted" not in out2


def test_fit_diagnosis_cap_is_bounded_even_when_everything_is_flagged(session):
    """The flagged-row escape must not reopen the payload it caps. On a badly-fitting
    65-observable problem the rollup can flag 50+ observables; an unbounded 'keep all flagged'
    rebuilt exactly the table the cap removes, in the regime where context pressure is worst.
    The bias-rank keep also has to be honoured, since a finish gate ranks by bias, not chi2."""
    recs = []
    for i in range(30):
        for t, r in ((0.0, 2.0), (1.0, -2.0), (2.0, 2.0), (3.0, -2.0)):
            recs.append(
                {
                    "observable_id": f"obs_{i:02d}",
                    "experiment_id": "e0",
                    "time": t,
                    "measurement": 1.0,
                    "simulation": 1.0,
                    "residual_weighted": r * (30 - i) / 30,
                }
            )
    # one low-chi2 observable with a LARGE systematic bias — a finish gate's "worst"
    for t in (0.0, 1.0, 2.0, 3.0):
        recs.append(
            {
                "observable_id": "obs_bias",
                "experiment_id": "e0",
                "time": t,
                "measurement": 1.0,
                "simulation": 1.0,
                "residual_weighted": 0.9,
            }
        )
    session.simulation = {
        "records": recs,
        "residual_structure": {"flagged_observables": [f"obs_{i:02d}" for i in range(30)]},
    }
    out = data_access.fit_diagnosis(session)
    ids = [r["observable_id"] for r in out["per_observable"]]
    assert len(ids) <= 21, f"cap unbounded: {len(ids)} rows"  # 12 chi2 + 3 bias + 6 flagged
    assert "obs_bias" in ids  # bias-ranked row retained
    assert out["omitted"]["n_observables_omitted"] > 0
    assert out["omitted"]["n_flagged_omitted"] > 0  # says flagged rows were cut
    assert "TRUNCATED" in out["omitted"]["note"]


# ------------------------------------------------------------- simulations_history


def _snapshot(session, records, label):
    """Store a version carrying ``records`` as its simulation, via the real ProvenanceStore."""
    return session.provenance.snapshot_version(
        model_ant="// test model",
        parameters=session.problem.parameter_df,
        label=label,
        simulation={"records": records} if records is not None else None,
    )


def test_simulations_history_spans_versions(session):
    v1 = _snapshot(session, SIM_RECORDS[:2], "v1")
    # v2 improves obs_A/e0/t0: weighted residual 2.0 -> 0.5
    better = [
        dict(SIM_RECORDS[0], simulation=0.75, residual=0.25, residual_weighted=0.5),
        SIM_RECORDS[1],
    ]
    v2 = _snapshot(session, better, "v2")

    out = data_access.run_query(
        session,
        "SELECT version, count(*) AS n FROM simulations_history GROUP BY version ORDER BY version",
    )
    assert out["rows"] == [{"version": v1, "n": 2}, {"version": v2, "n": 2}]

    # The comparative query agents kept reconstructing by hand: per-point deltas.
    out = data_access.run_query(
        session,
        f"""SELECT observable_id, time,
                   max(CASE WHEN version={v1} THEN residual_weighted END) AS a,
                   max(CASE WHEN version={v2} THEN residual_weighted END) AS b
              FROM simulations_history GROUP BY 1, 2 ORDER BY time""",
    )
    assert out["rows"][0]["a"] == 2.0 and out["rows"][0]["b"] == 0.5
    assert out["rows"][1]["a"] == 0.0 and out["rows"][1]["b"] == 0.0


def test_simulations_history_skips_versions_without_simulation(session):
    _snapshot(session, None, "failed fit")  # stores no simulations.json
    v2 = _snapshot(session, SIM_RECORDS[:1], "ok")
    out = data_access.run_query(
        session, "SELECT DISTINCT version FROM simulations_history ORDER BY version"
    )
    assert [r["version"] for r in out["rows"]] == [v2]


def test_simulations_history_does_not_shadow_head(session):
    # HEAD `simulations` stays exactly the session's own dataframe — the history table is a
    # separate name, not a replacement.
    _snapshot(session, SIM_RECORDS[:1], "v1")
    session.simulation = {"records": SIM_RECORDS}
    out = data_access.run_query(session, "SELECT count(*) AS n FROM simulations")
    assert out["rows"][0]["n"] == len(SIM_RECORDS)


def test_simulations_history_is_lazy(session, monkeypatch):
    # A query that never names the table must not touch the version store.
    def _boom(_session):
        raise AssertionError("simulations_history materialized for an unrelated query")

    monkeypatch.setattr(data_access, "_simulation_history_frame", _boom)
    out = data_access.run_query(session, "SELECT count(*) AS n FROM measurements")
    assert out["rows"][0]["n"] == 6


def test_simulations_history_cache_tracks_new_versions(session):
    _snapshot(session, SIM_RECORDS[:1], "v1")
    out = data_access.run_query(
        session, "SELECT count(DISTINCT version) AS n FROM simulations_history"
    )
    assert out["rows"][0]["n"] == 1
    _snapshot(session, SIM_RECORDS[:2], "v2")  # cache key changes with the store
    out = data_access.run_query(
        session, "SELECT count(DISTINCT version) AS n FROM simulations_history"
    )
    assert out["rows"][0]["n"] == 2


def test_simulations_history_empty_but_typed(session):
    # Cold start: no versions at all — the table exists, is empty, and its columns resolve.
    out = data_access.run_query(
        session, "SELECT version, observable_id, residual_weighted FROM simulations_history"
    )
    assert out["rows"] == []


def test_simulations_history_positive_control(session, monkeypatch):
    # The inverse of the laziness test: a query that DOES name the table must build it — a
    # broken name gate would otherwise leave every history query failing while the suite
    # stays green.
    called = {}
    real = data_access._simulation_history_frame

    def _spy(s):
        called["yes"] = True
        return real(s)

    monkeypatch.setattr(data_access, "_simulation_history_frame", _spy)
    data_access.run_query(session, "SELECT count(*) FROM simulations_history")
    assert called.get("yes")


def test_simulations_history_survives_corrupt_file(session):
    # A snapshot interrupted mid-write (wall-clock kill, disk full) leaves a truncated
    # simulations.json. It must degrade to "this version contributes no rows" — not take down
    # every query_data/list_data_tables call for the rest of the session.
    v1 = _snapshot(session, SIM_RECORDS[:1], "good")
    v2 = _snapshot(session, SIM_RECORDS[:2], "corrupt")
    vdir = session.provenance.versions_dir / f"v{v2}"
    (vdir / "simulations.json").write_text('{"records": [{"obs', encoding="utf-8")

    out = data_access.run_query(
        session, "SELECT DISTINCT version FROM simulations_history ORDER BY version"
    )
    assert [r["version"] for r in out["rows"]] == [v1]
    schema = data_access.table_schema(session)  # must not raise either
    tables = {t["table"]: t for t in schema["tables"]}
    assert tables["simulations_history"]["n_rows"] == 1


def test_simulations_history_non_object_json_contributes_nothing(session):
    v1 = _snapshot(session, SIM_RECORDS[:1], "v1")
    vdir = session.provenance.versions_dir / f"v{v1}"
    (vdir / "simulations.json").write_text("[1, 2, 3]", encoding="utf-8")
    out = data_access.run_query(session, "SELECT count(*) AS n FROM simulations_history")
    assert out["rows"][0]["n"] == 0


def test_simulations_history_mid_commit_window_heals(session):
    # snapshot_version writes meta.json BEFORE simulations.json, and query_data takes no
    # session lock — so a query can see version N listed while its simulation is not on disk
    # yet. That read must not poison the cache: the entry is keyed by the FILE's fingerprint,
    # so it invalidates the moment the file lands.
    import json as _json

    v1 = _snapshot(session, None, "mid-commit")  # meta exists, simulations.json does not
    out = data_access.run_query(session, "SELECT count(*) AS n FROM simulations_history")
    assert out["rows"][0]["n"] == 0  # the empty view, now cached
    vdir = session.provenance.versions_dir / f"v{v1}"
    (vdir / "simulations.json").write_text(
        _json.dumps({"records": SIM_RECORDS[:2]}), encoding="utf-8"
    )
    out = data_access.run_query(session, "SELECT count(*) AS n FROM simulations_history")
    assert out["rows"][0]["n"] == 2  # healed, not permanently missing


def test_simulations_history_survives_bad_record_shape(session):
    # The blob parses as an object (so the store's isinstance guard passes) but `records` is
    # not a list of dicts — a hand-edited file, or an archive from an older schema. Splatting
    # a non-mapping would raise a TypeError that escapes into EVERY query for the rest of the
    # session; the bad entries are skipped instead.
    import json as _json

    v1 = _snapshot(session, SIM_RECORDS[:1], "good")
    v2 = _snapshot(session, SIM_RECORDS[:1], "bad-records")
    (session.provenance.versions_dir / f"v{v2}" / "simulations.json").write_text(
        _json.dumps({"records": [1, 2, 3]}), encoding="utf-8"
    )
    v3 = _snapshot(session, SIM_RECORDS[:1], "records-not-a-list")
    (session.provenance.versions_dir / f"v{v3}" / "simulations.json").write_text(
        _json.dumps({"records": {"a": 1}}), encoding="utf-8"
    )

    out = data_access.run_query(
        session, "SELECT DISTINCT version FROM simulations_history ORDER BY version"
    )
    assert [r["version"] for r in out["rows"]] == [v1]
    data_access.table_schema(session)  # must not raise either


def test_simulations_history_survives_corrupt_meta(session):
    # A version's meta.json is unrelated to its simulation: list_versions skips such a version,
    # so enumerating from meta would drop an INTACT simulation from every comparison.
    v1 = _snapshot(session, SIM_RECORDS[:2], "v1")
    (session.provenance.versions_dir / f"v{v1}" / "meta.json").write_text("{trunc", "utf-8")
    out = data_access.run_query(session, "SELECT count(*) AS n FROM simulations_history")
    assert out["rows"][0]["n"] == 2


def test_record_version_field_cannot_override_the_column(session):
    # A record carrying its own `version` must not re-attribute the row (the fit payload
    # already stamps a version at blob top level, so this is a live shape).
    rec = dict(SIM_RECORDS[0], version=999)
    v1 = _snapshot(session, [rec], "v1")
    out = data_access.run_query(session, "SELECT DISTINCT version FROM simulations_history")
    assert [r["version"] for r in out["rows"]] == [v1]


def test_table_schema_reports_history_without_building_it(session, monkeypatch):
    # list_data_tables is typically an agent's FIRST call: it must report the
    # row count without concatenating a cross-version DataFrame it immediately discards.
    _snapshot(session, SIM_RECORDS[:2], "v1")

    def _boom(_s):
        raise AssertionError("table_schema built the history frame just to count rows")

    monkeypatch.setattr(data_access, "_simulation_history_frame", _boom)
    tables = {t["table"]: t for t in data_access.table_schema(session)["tables"]}
    assert tables["simulations_history"]["n_rows"] == 2
    assert tables["simulations_history"]["columns"] == list(data_access.SIMULATION_HISTORY_COLUMNS)


# ------------------------------------------------------------- paging


def test_query_pages_without_overlap_or_gap(session):
    # Six measurement rows, three pages of two: every row seen exactly once.
    seen, offset, pages = [], 0, 0
    while True:
        out = data_access.run_query(
            session,
            "SELECT time, observableId FROM measurements ORDER BY observableId, time",
            max_rows=2,
            offset=offset,
        )
        pages += 1
        seen += [tuple(r.values()) for r in out["rows"]]
        assert out["offset"] == offset
        if not out["truncated"]:
            assert "next_offset" not in out
            break
        offset = out["next_offset"]
    assert pages == 3
    assert len(seen) == 6 and len(set(seen)) == 6


def test_offset_past_the_end_is_empty_not_an_error(session):
    out = data_access.run_query(session, "SELECT * FROM measurements", max_rows=5, offset=500)
    assert out["rows"] == [] and out["row_count"] == 0 and out["truncated"] is False


def test_max_rows_is_clamped_to_a_hard_ceiling(session):
    # The paging guarantee must be enforced, not merely advised: without a ceiling the caller
    # can ask for the whole table and reproduce the context blow-up paging exists to prevent.
    n_rows, off = data_access.validate_page_args(100_000, 0)
    assert n_rows == data_access.MAX_PAGE_ROWS and off == 0
    out = data_access.run_query(session, "SELECT * FROM measurements", max_rows=100_000)
    assert out["row_count"] == 6  # small table still returns whole


def test_page_args_rejected(session):
    for bad in (0, -1, "x"):
        with pytest.raises(ValidationError):
            data_access.validate_page_args(bad, 0)
    for bad in (-1, "x"):
        with pytest.raises(ValidationError):
            data_access.validate_page_args(10, bad)
    assert data_access.validate_page_args("5", "2") == (5, 2)  # numeric strings coerce


def test_head_simulations_reference_gate():
    # Decides whether query_data pays an AMICI build: HEAD `simulations` reads must, queries
    # touching only the history (or the static tables) must not.
    ref = data_access.query_reads_head_simulation
    assert ref("SELECT * FROM simulations WHERE residual_weighted IS NOT NULL")
    assert ref("select s.time from SIMULATIONS s")
    assert ref("SELECT value FROM state_trajectories WHERE entity_id = 'B'")
    assert not ref("SELECT version FROM simulations_history")
    assert not ref("SELECT count(*) FROM measurements")
