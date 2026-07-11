from __future__ import annotations

from petab_mcp.validation import validate_problem


def test_valid_problem(session):
    report = validate_problem(session.problem)
    assert report["valid"] is True
    assert report["n_errors"] == 0


def test_invalid_after_breaking_model(session):
    from petab_mcp.model import _build_model

    # Swap in a model missing entities the fixed tables need.
    broken = _build_model(
        "model conversion_reaction\n species A = 1;\n k1 = 0.5;\n"
        " r1: A => ; k1 * A;\nend",
        "antimony",
        "conversion_reaction",
    )
    session.problem.model = broken
    report = validate_problem(session.problem)
    assert report["valid"] is False
    assert report["n_errors"] >= 1
