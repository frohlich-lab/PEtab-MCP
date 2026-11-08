from __future__ import annotations

from pathlib import Path

import pytest
from petab.v2.models.sbml_model import SbmlModel

from petab_mcp import operations as O
from petab_mcp.problem import WorkingProblem, later_changes
from petab_mcp.validation import (
    INPUT_UNREAD,
    _constant_noise_species_issues,
    _fixed_constant_role_issues,
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


def test_protected_fixed_constant_removed_is_a_warning(session):
    """Warned, not refused: refused, agents faked a use of a constant they could not place."""
    # A protected constant absent from the parameter table = it was removed from the model.
    issues = _fixed_constant_role_issues(session.problem, {"k_removed"})
    assert [i["check"] for i in issues] == ["fixed_constant_removed"]
    assert "k_removed" in issues[0]["message"] and issues[0]["level"] == "warning"


def test_fixed_constant_check_is_noop_without_protected_set(session):
    assert _fixed_constant_role_issues(session.problem, None) == []
    assert _fixed_constant_role_issues(session.problem, set()) == []


def test_protected_fixed_ids_exclude_ics_and_observation_params(session):
    # The toy fixes sigma_B (a noise sigma), A_init and B_init (initial conditions); none is a
    # mechanistic constant, so the protected set is empty and the toy commits freely.
    assert session.protected_fixed_parameter_ids() == set()


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


_DOSED = """model dosed
  r1: -> B; {rate}
  r2: B -> ; kd*B
  D = 2*dose
  B = 0; k = 0.1; kd = 0.1; dose = 0
end
"""


@pytest.mark.parametrize(("rate", "warned"), [("k*D", True), ("k*dose", False)])
def test_an_input_read_only_at_the_start_is_named_with_its_later_periods(
    dosed_session, rate, warned
):
    """isensee's seeds read each dose in an initial assignment, which saw only the
    pre-equilibration 0; libpetab's unread-target check passed, and 16 fits ran blind to every
    drug (PEtab-MCP#192). Read in a rate as well, the dose reaches the model."""
    result = O.set_model(dosed_session, _DOSED.format(rate=rate), justification="dose B")
    said = [m["message"] for m in result["messages"] if "initial amounts" in m["message"]]
    assert bool(said) == warned
    if warned:
        assert said[0].startswith("condition target(s) `dose` (e0 at t = 0) change in these")


def test_an_input_whose_only_reader_was_pruned_is_named(dosed_session):
    """The reconcile prunes `D = 2*dose` when no rate reads `D` and leaves `dose` declared, which
    libpetab's invalid-target check passes (PEtab-MCP#205)."""
    result = O.set_model(dosed_session, _DOSED.format(rate="k"), justification="dose B")
    said = [m["message"] for m in result["messages"] if "read by nothing" in m["message"]]
    assert len(said) == 1 and said[0].startswith("condition target(s) `dose` are read by nothing")


def test_an_unread_input_is_said_to_be_allowed(dosed_session):
    """The warning's "For each, read it as …" read as a refusal of every unread target to 3 of 5
    froehlich leads, and isensee's inert control was given a term for it (PEtab-MCP#290)."""
    result = O.set_model(dosed_session, _DOSED.format(rate="k"), justification="dose B")
    (said,) = [m["message"] for m in result["messages"] if "read by nothing" in m["message"]]
    assert said.endswith(INPUT_UNREAD) and "For each" not in said


@pytest.mark.parametrize(("rate", "warned"), [("k", True), ("k*drive", False)])
def test_an_input_read_only_by_a_rule_nothing_reads_is_named(dosed_session, rate, warned):
    """An edit that drops the last reader of `drive := dose` leaves `dose` read by the rule
    alone, which the unread-input check counted as read, so the model lost its input with no
    warning (PEtab-MCP#264). A rate reading `drive` reaches it."""
    driven = _DOSED.replace("D = 2*dose", "drive := dose")
    result = O.set_model(dosed_session, driven.format(rate=rate), justification="dose B")
    said = [m["message"] for m in result["messages"] if "read by nothing" in m["message"]]
    assert bool(said) == warned
    if warned:
        assert said[0].startswith(
            "condition target(s) `dose` are read by nothing in the model, or only by assignment "
            "rules nothing reads, so the model never sees their value."
        )


def test_a_later_period_that_keeps_the_value_changes_nothing(dosed_session):
    """A period that sets an input to the value it already holds is no change to warn about."""
    yaml = dosed_session.yaml_path
    assert later_changes(WorkingProblem.from_yaml(yaml)) == {"dose": [("e0", 0.0)]}
    conditions = Path(yaml).parent / "conditions.tsv"
    conditions.write_text(conditions.read_text().replace("treat\tdose\t1", "treat\tdose\t0"))
    assert later_changes(WorkingProblem.from_yaml(yaml)) == {}


_EGFR = """model egfr
  rE = 1; rD = 0; B = 0
  egfr_int: rE -> {product}; k_intE*rE
  rec: rD -> rE; {recycling}
  deg: rD -> ; k_deg*rD
  act: -> B; k_act*rE
  k_intE = 0.1; k_rec = 0.1; k_back = 0.1; k_deg = 0.1; k_act = 0.1
end
"""


def _sourceless(result) -> list[str]:
    return [m["message"] for m in result["messages"] if "stays 0" in m["message"]]


@pytest.mark.parametrize(
    ("product", "recycling", "warned"),
    [("", "k_rec*rD", True), ("rD", "k_rec*rD", False), ("", "k_rec*rD - k_back*rE", False)],
)
def test_a_species_nothing_produces_is_named_with_its_readers(session, product, recycling, warned):
    """dessauges' `set_rate_law` kept `egfr_int: rE ->`, so `rD` started at 0 with nothing making
    it; its rates' parameters pinned at a bound and the only reply was "1 warning(s) unchanged"
    (PEtab-MCP#270). A product of `egfr_int`, or `rec` running back at `rD` = 0, is a source."""
    antimony = _EGFR.format(product=product, recycling=recycling)
    said = _sourceless(O.set_model(session, antimony, justification="endocytosis"))
    assert said == (
        [
            "species `rD` starts at 0 and stays 0: no reaction produces it, and no condition, event "
            "or rule sets it. The rate laws and rules that read it (`rec`, `deg`) see 0 throughout, "
            "so a parameter acting only through it cannot change the fit. Give it a source: a "
            "reaction that produces it, or a nonzero initial amount."
        ]
        if warned
        else []
    )


def test_a_species_a_period_sets_has_a_source(dosed_session):
    """`dose` starts at 0 and no reaction makes it, but the `treat` period sets it to 1; `X`, read
    the same way and set by nothing, is the one named."""
    antimony = """model dosed
  species dose = 0, X = 0
  r1: -> B; k*dose
  r2: -> B; kx*X
  r3: B -> ; kd*B
  B = 0; k = 0.1; kx = 0.1; kd = 0.1
end
"""
    said = _sourceless(O.set_model(dosed_session, antimony, justification="dose B"))
    assert len(said) == 1 and said[0].startswith("species `X` starts at 0")


@pytest.mark.parametrize(
    ("reaction", "refused"),
    [
        ("r1: A => ; k*A", True),  # B in no reaction
        ("r1: A => ; k*A*B", True),  # B read in a rate, changed by none
        ("r1: A + B => B; k*A*B", True),  # a catalyst: as much B on each side
        ("r1: A => B; k*A", False),
    ],
)
def test_a_noise_that_reads_a_species_no_reaction_changes_is_refused(reaction, refused):
    """#285: armistead's metabolites are observed with noise proportional to each; one no reaction
    changed made AMICI refuse the import ("the sigma of obs_B depends on tcl_B"), which the agent
    was told was an ill-posed model and answered by changing three unrelated things."""
    import types

    import pandas as pd

    antimony = f"model m\n species A, B; A = 1; B = 2; k = 1;\n {reaction}\nend\n"
    model = SbmlModel.from_antimony(antimony)
    observables = pd.DataFrame(
        {"observableFormula": ["B"], "noiseFormula": ["sigma_B * B"]},
        index=pd.Index(["obs_B"], name="observableId"),
    )
    measurements = pd.DataFrame({"observableId": ["obs_B"], "noiseParameters": [float("nan")]})
    problem = types.SimpleNamespace(
        model=model, observable_df=observables, measurement_df=measurements
    )
    issues = _constant_noise_species_issues(problem)
    assert [i["message"] for i in issues] == (
        [
            "the noise of `obs_B` reads species `B`, which no reaction changes: AMICI holds such "
            "a species as a constant total and cannot build a noise model that reads one, so the "
            "fit would fail. Put `B` in a reaction that produces or consumes it."
        ]
        if refused
        else []
    )
