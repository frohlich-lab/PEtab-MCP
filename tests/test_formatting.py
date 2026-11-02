from __future__ import annotations

import io

import pandas as pd

from petab_mcp.formatting import conditions_tsv, df_to_records, df_to_tsv, experiments_tsv


def test_named_index_becomes_id_key(session):
    # parameter_df / observable_df are indexed by a *named* primary key.
    recs = df_to_records(session.problem.parameter_df)
    assert all("parameterId" in r for r in recs)
    obs = df_to_records(session.problem.observable_df)
    assert all("observableId" in r for r in obs)


def test_filtered_measurements_no_spurious_id(session):
    df = session.problem.measurement_df
    # Non-contiguous filter -> pandas downgrades RangeIndex to Int64Index.
    filtered = df[df["time"].isin([0.0, 1.0, 4.0])]
    recs = df_to_records(filtered)
    assert recs
    assert all("id" not in r for r in recs)  # no positional-offset id leaks in


def test_observables_tsv_leads_with_the_named_index(session):
    header, *rows = df_to_tsv(session.problem.observable_df).splitlines()
    assert header.split("\t")[0] == "observableId"
    assert len(rows) == len(session.problem.observable_df)


def test_conditions_tsv_is_the_condition_table_pivoted_without_loss():
    """#191: one row per condition, one column per target, a blank where it sets none."""
    long = pd.DataFrame(
        {
            "conditionId": ["c1", "c1", "c2", "dose"],
            "targetId": ["MKN1", "medium", "MKN1", "EGF"],
            "targetValue": [1.0, 0.0, 0.0, "2*u"],
        }
    )
    text = conditions_tsv(long)
    assert text.splitlines() == [
        "conditionId\tMKN1\tmedium\tEGF",
        "c1\t1.0\t0.0\t",
        "c2\t0.0\t\t",
        "dose\t\t\t2*u",
    ]
    back = pd.read_csv(io.StringIO(text), sep="\t", dtype=str).melt(
        "conditionId", var_name="targetId", value_name="targetValue"
    )
    assert len(back.dropna()) == len(long)


def test_experiments_tsv_is_one_row_per_period():
    long = pd.DataFrame(
        {
            "experimentId": ["e1", "e1", "e1", "e2"],
            "time": [float("-inf"), float("-inf"), 0.0, 0.0],
            "conditionId": ["base", "cell", "dose", None],
        }
    )
    assert experiments_tsv(long).splitlines() == [
        "experimentId\ttime\tconditionId",
        "e1\t-inf\tbase cell",
        "e1\t0.0\tdose",
        "e2\t0.0\t",
    ]


def test_validation_reports_messages(session, broken_model):
    from petab_mcp.validation import validate_problem

    session.problem.model = broken_model
    report = validate_problem(session.problem)
    assert report["valid"] is False
    assert report["n_errors"] >= 1
    assert report["issues"]  # human-readable messages captured from petab logging
