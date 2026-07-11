from __future__ import annotations

from petab_mcp.formatting import df_to_records


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


def test_validation_reports_messages(session):
    from petab_mcp.model import _build_model
    from petab_mcp.validation import validate_problem

    session.problem.model = _build_model(
        "model conversion_reaction\n species A = 1;\n k1 = 0.5;\n"
        " r1: A => ; k1 * A;\nend",
        "antimony",
        "conversion_reaction",
    )
    report = validate_problem(session.problem)
    assert report["valid"] is False
    assert report["n_errors"] >= 1
    assert report["issues"]  # human-readable messages captured from petab logging
