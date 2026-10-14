from __future__ import annotations

from petab.v2.models.sbml_model import SbmlModel

from petab_mcp.validation import (
    _fixed_constant_role_issues,
    _unused_condition_target_issues,
    frozen_parametric_stoichiometry_issues,
    validate_problem,
)


def _stoich_issues(antimony: str):
    """Convert Antimony (as set_model does) and run the frozen-stoichiometry check on the SBML."""
    return frozen_parametric_stoichiometry_issues(
        SbmlModel.from_antimony(antimony, model_id=None).sbml_model
    )


def test_valid_problem(session):
    report = validate_problem(session.problem)
    assert report["valid"] is True
    assert report["n_errors"] == 0


def test_invalid_after_breaking_model(session, broken_model):
    # Swap in a model missing entities the fixed tables need.
    session.problem.model = broken_model
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


# --- frozen parametric reaction stoichiometry (issue #89) -----------------------------------
# libantimony freezes a symbol used as a stoichiometry coefficient to a constant, drops the
# parameter, and leaves the dropped symbol as the SpeciesReference id — the signal we detect. A
# literal coefficient leaves the id unset, so it is never flagged.
# The shipped toy base model (tests/data/toy_v2/model.ant) — real reactions with `=>` arrows and
# no coefficients: it must NOT trip the check (a false positive would block every real edit).
_TOY_BASE = """model conversion_reaction
  compartment compartment_ = 1;
  species A in compartment_;
  species B in compartment_;
  A = A_init;
  B = B_init;
  A_init = 1.0;
  B_init = 0.0;
  k1 = 0.5;
  k2 = 0.1;
  r1: A => B; k1 * A;
  r2: B => A; k2 * B;
end
"""


def test_parametric_product_stoichiometry_is_flagged():
    # A parameter used as a product coefficient is the footgun: frozen to a constant, dropped.
    issues = _stoich_issues("X -> N X; c*k*X; N=3; c=1; k=0.1")
    assert [i["check"] for i in issues] == ["symbolic_stoichiometry"]
    assert "N" in issues[0]["message"]
    # The message names the frozen value, points at a literal integer as the simple fix, and gives
    # the product-side estimable form (split off a scaled production).
    msg = issues[0]["message"].lower()
    assert "literal" in msg and "3" in msg
    assert "production" in msg


def test_literal_integer_stoichiometry_is_accepted():
    # Autocatalysis regression guard: a literal integer coefficient is preserved and must stay OK.
    assert _stoich_issues("X -> 2 X; c*k*X; c=1; k=0.1") == []


def test_symbolic_reactant_stoichiometry_is_flagged_with_reactant_guidance():
    issues = _stoich_issues("r1: N A -> B; k*A; N=2; k=0.1")
    assert [i["check"] for i in issues] == ["symbolic_stoichiometry"]
    msg = issues[0]["message"]
    assert "r1" in msg and "N" in msg and "A" in msg
    # Reactant-side guidance talks about consuming the reactant N-fold, not amplifying a product.
    assert "consum" in msg.lower() and "production '->" not in msg


def test_parametric_stoichiometry_flagged_even_when_symbol_also_in_rate_law():
    # N is used in the rate law too, but antimony STILL freezes+drops it from the stoichiometry
    # (leaving a dangling rate-law reference); the SpeciesReference-id fingerprint still catches it.
    assert _stoich_issues("X -> N X; N*k*X; N=3; k=0.1")


def test_multi_product_flags_only_the_symbolic_coefficient():
    issues = _stoich_issues("A -> 2 B + M C; k*A; M=4; k=0.1")
    assert len(issues) == 1
    assert "M" in issues[0]["message"] and "C" in issues[0]["message"]


def test_real_toy_model_has_no_false_positive():
    # The `=>`-arrow toy with plain (coefficient-1) reactions must not be flagged.
    assert _stoich_issues(_TOY_BASE) == []


def test_float_literal_and_degradation_are_accepted():
    assert _stoich_issues("A -> 2.0 B; k*A; k=0.1") == []
    assert _stoich_issues("A => ; k*A; k=0.1") == []


def test_different_symbols_on_both_sides_are_both_flagged():
    # Distinct symbols each get their own species-reference id, so both are detected.
    issues = _stoich_issues("N A -> M B; k*A; N=2; M=3; k=0.1")
    assert len(issues) == 2
    joined = " ".join(i["message"] for i in issues)
    assert "N" in joined and "M" in joined


def test_same_symbol_both_sides_is_flagged():
    # The SAME symbol on both sides converts, but only the first occurrence keeps the frozen
    # coefficient — the rest become `<reaction>_<species>_stoich` assignment rules reading the
    # dropped symbol. The one frozen reference is enough to flag the model.
    issues = _stoich_issues("N A -> N B; k*A; N=2; k=0.1")
    assert [i["check"] for i in issues] == ["symbolic_stoichiometry"]
    assert "N" in issues[0]["message"]
