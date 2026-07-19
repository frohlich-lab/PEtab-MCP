from __future__ import annotations

from petab_mcp.validation import (
    _fixed_constant_role_issues,
    _unused_condition_target_issues,
    validate_problem,
)


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


def test_protected_fixed_constant_made_estimated_is_rejected(session):
    # k1 is a protected shipped-fixed constant; the working table has it estimated (as the toy
    # does) -> estimating a protected fixed constant is a hard error.
    issues = _fixed_constant_role_issues(session.problem, {"k1"})
    assert [i["check"] for i in issues] == ["fixed_constant_made_estimated"]
    assert "k1" in issues[0]["message"]

    report = validate_problem(session.problem, protected_fixed_ids={"k1"})
    assert report["valid"] is False
    assert any(i["check"] == "fixed_constant_made_estimated" for i in report["issues"])
    # with no protected set the check is skipped and the problem is valid again
    assert validate_problem(session.problem)["valid"] is True


def test_protected_fixed_constant_removed_is_rejected(session):
    # A protected constant absent from the parameter table = it was removed from the model.
    issues = _fixed_constant_role_issues(session.problem, {"k_removed"})
    assert [i["check"] for i in issues] == ["fixed_constant_removed"]
    assert "k_removed" in issues[0]["message"]


def test_fixed_constant_check_is_noop_without_protected_set(session):
    assert _fixed_constant_role_issues(session.problem, None) == []
    assert _fixed_constant_role_issues(session.problem, set()) == []


def test_protected_fixed_ids_exclude_ics_and_observation_params(session):
    # The toy fixes sigma_B (a noise sigma), A_init and B_init (initial conditions); none is a
    # mechanistic constant, so the protected set is empty and the toy commits freely.
    assert session.protected_fixed_parameter_ids() == set()


def test_unused_condition_target_check_is_noop_without_targets(session):
    # The toy condition table has only conditionId (no target columns), so nothing to check.
    assert _unused_condition_target_issues(session.problem) == []
