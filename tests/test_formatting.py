from __future__ import annotations

import io

import numpy as np
import pandas as pd
import pytest

from petab_mcp.formatting import (
    EXPERIMENTS_MAX_BYTES,
    CONDITIONS_MAX_BYTES,
    conditions_tsv,
    df_to_records,
    df_to_tsv,
    experiments_tsv,
)


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


def _wide(n_conditions: int, n_targets: int) -> pd.DataFrame:
    """A long condition table in which condition `line<c>` sets `gene<t>` to (c * t mod 997) / 7."""
    c, t = (a.ravel() for a in np.meshgrid(range(n_conditions), range(n_targets), indexing="ij"))
    return pd.DataFrame(
        {
            "conditionId": "line" + pd.Series(c).astype(str),
            "targetId": "gene" + pd.Series(t).astype(str),
            "targetValue": (c * t % 997) / 7,
        }
    )


def test_a_wide_condition_table_comes_back_one_row_per_target_under_the_cap():
    """froehlich's 9169 conditions setting 145 targets came back as 36 MB of TSV, which no lead
    could read (PEtab-MCP#290)."""
    dose = pd.DataFrame({"conditionId": ["line3"], "targetId": ["dose"], "targetValue": ["2*u"]})
    text = conditions_tsv(pd.concat([_wide(9169, 145), dose]))
    assert len(text.encode()) <= CONDITIONS_MAX_BYTES
    head, *table = text.splitlines()
    assert head.startswith("9169 conditions set 146 targets")
    assert "WHERE condition_id = 'line0'" in head
    summary = pd.read_csv(io.StringIO("\n".join(table)), sep="\t", index_col="targetId")
    assert list(summary.index) == [f"gene{t}" for t in range(145)] + ["dose"]
    assert summary.loc["gene1", ["n_conditions", "n_values", "min", "max"]].tolist() == [
        9169,
        997,
        0,
        pytest.approx(996 / 7, rel=1e-5),
    ]
    assert summary.loc["dose", "expressions"] == "2*u" and pd.isna(summary.loc["dose", "min"])


def test_a_summary_counts_the_targets_past_the_cap():
    """A summary of more targets than one reply holds stays under the cap too."""
    text = conditions_tsv(_wide(40, 900), max_bytes=4096)
    assert len(text.encode()) <= 4096
    *_, last = text.splitlines()
    listed = len(text.splitlines()) - 3
    assert (
        last
        == f"(+{900 - listed} more targets: query_data `SELECT DISTINCT target_id FROM conditions`)"
    )


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


def test_a_froehlich_sized_experiment_table_comes_back_one_row_per_layout_under_the_cap():
    """froehlich's 9169 experiments, one per cell line, drug and dose, were 1 MB of TSV, the
    table beside its 36 MB condition table (PEtab-MCP#290). A layout's periods are read in time
    order, which a file need not keep."""
    ids = [
        f"TUMOR_line{i // 40:03d}_cellline_01_01_drug{i % 40:02d}_Conc{i % 7}_5nM"
        for i in range(9169)
    ]
    froehlich = pd.DataFrame(
        {"experimentId": [f"experiment____{c}" for c in ids], "time": 0.0, "conditionId": ids}
    )
    dosed = pd.DataFrame(
        {
            "experimentId": ["pre"] * 3,
            "time": [0.0, float("-inf"), float("-inf")],
            "conditionId": ["egf", "base", "cell"],
        }
    )
    text = experiments_tsv(pd.concat([froehlich, dosed]))
    assert len(text.encode()) <= EXPERIMENTS_MAX_BYTES
    head, *table = text.splitlines()
    assert head.startswith("9170 experiments, too large a table for one reply")
    assert f"WHERE experiment_id = 'experiment____{ids[0]}'" in head
    assert table == [
        "n_experiments\tn_periods\ttimes\tn_conditions\texample",
        f"9169\t1\t0\t1\texperiment____{ids[0]}",
        "1\t2\t-inf 0\t2 1\tpre",
    ]


def test_an_experiment_summary_counts_the_layouts_past_the_cap():
    """A dose given at its own time per experiment is a layout per experiment: more than one reply
    holds, so the summary stays under the cap too."""
    long = pd.DataFrame(
        {
            "experimentId": [f"e{i}" for i in range(500) for _ in range(2)],
            "time": [t for i in range(500) for t in (0.0, i + 1.0)],
            "conditionId": "dose",
        }
    )
    text = experiments_tsv(long, max_bytes=2048)
    assert len(text.encode()) <= 2048
    *_, last = text.splitlines()
    more = 500 - (len(text.splitlines()) - 3)
    assert last == f"(+{more} more layouts, {more} experiments)"


def test_validation_reports_messages(session, broken_model):
    from petab_mcp.validation import validate_problem

    session.problem.model = broken_model
    report = validate_problem(session.problem)
    assert report["valid"] is False
    assert report["n_errors"] >= 1
    assert report["issues"]  # human-readable messages captured from petab logging
