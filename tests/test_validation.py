from __future__ import annotations

import re
from pathlib import Path

import pytest
from petab.v2.models.sbml_model import SbmlModel

from petab_mcp import operations as O
from petab_mcp.problem import WorkingProblem, later_changes
from petab_mcp.validation import (
    INPUT_UNREAD,
    _constant_noise_species_issues,
    _constant_readout_issues,
    _dead_rule_parameter_issues,
    _fixed_constant_role_issues,
    _power_issues,
    _unread_reactant_issues,
    exponents_below_one_at_zero,
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
  species D = 2*dose
  B = 0; k = 0.1; kd = 0.1; dose = 0
end
"""


@pytest.mark.parametrize(("rate", "warned"), [("k*D", True), ("k*dose", False)])
def test_an_input_read_only_at_the_start_is_named_with_its_later_periods(
    dosed_session, rate, warned
):
    """isensee's seeds read each dose in an initial assignment, which saw only the
    pre-equilibration 0; libpetab's unread-target check passed, and 16 fits ran blind to every
    drug (PEtab-MCP#192). Read in a rate as well, the dose reaches the model. `D` is a species, as
    isensee's seeds were: a parameter assigned so takes an estimated row, refused (#286)."""
    result = O.set_model(dosed_session, _DOSED.format(rate=rate), justification="dose B")
    said = [m["message"] for m in result["messages"] if "only in initial amounts" in m["message"]]
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
    driven = _DOSED.replace("species D = 2*dose", "drive := dose")
    result = O.set_model(dosed_session, driven.format(rate=rate), justification="dose B")
    said = [m["message"] for m in result["messages"] if "read by nothing" in m["message"]]
    assert bool(said) == warned
    if warned:
        assert said[0].startswith(
            "condition target(s) `dose` are read by nothing in the model, or only by assignment "
            "rules nothing reads, so the model never sees their value."
        )


def test_an_input_the_edit_leaves_unread_leads_as_newly_unread(dosed_session):
    """froehlich 91's re-author dropped Erlotinib and Lapatinib, and the two sat unmarked among
    about 120 unread names; the agent read the worse fit as a mechanism (PEtab-MCP#332). Unread
    against the model before the edit, they lead; the next edit sends the plain list."""

    def unread(rate: str) -> str:
        result = O.set_model(dosed_session, _DOSED.format(rate=rate), justification="dose B")
        (said,) = [m["message"] for m in result["messages"] if "read by nothing" in m["message"]]
        return said

    O.set_model(dosed_session, _DOSED.format(rate="k*dose"), justification="dose drives B")
    assert unread("k").startswith(
        "NEWLY UNREAD: condition target(s) `dose`, which the model before this edit read, are "
        "read by nothing in the model"
    )
    assert unread("k").startswith("condition target(s) `dose` are read by nothing")


def test_inputs_unread_before_the_edit_too_are_capped():
    """The 12 shown of those the base left unread too follow the newly unread (PEtab-MCP#332)."""
    import pandas as pd
    from petab.v2.C import CONDITION_ID, TARGET_ID, TARGET_VALUE

    from petab_mcp.validation import _unread_input_issues

    ids = [f"u{i:02d}" for i in range(15)]
    decl = "".join(f"  {t} = 0;\n" for t in ids)
    prob = WorkingProblem(
        model=SbmlModel.from_antimony(
            f"model m\n  species X; X = 1; k = 0.1;\n  r: X -> ; k*X;\n{decl}end\n"
        )
    )
    prob.condition_df = pd.DataFrame({CONDITION_ID: "c", TARGET_ID: ids, TARGET_VALUE: 1.0})
    (issue,) = _unread_input_issues(prob, read_before={"u00"})
    shown = ", ".join(f"`{t}`" for t in ids[1:13])
    assert issue["message"].startswith(
        "NEWLY UNREAD: condition target(s) `u00`, which the model before this edit read"
    )
    assert f" Unread before it too: {shown} (+2 more). {INPUT_UNREAD}" in issue["message"]


def test_an_input_the_model_leaves_out_is_declared_at_0(dosed_session):
    """froehlich's ~145 inputs each had to be declared by hand, and dropping an input's last
    reader was refused as an invalid target with no op to declare it again (raimundez 7,
    froehlich 29; PEtab-MCP#332). The reconcile declares it: a constant at 0, with no row."""
    O.set_model(dosed_session, _DOSED.format(rate="k*dose"), justification="dose drives B")
    bare = _DOSED.format(rate="k").replace("  species D = 2*dose\n", "").replace("; dose = 0", "")
    assert not re.search(r"\bdose\b", bare)
    result = O.set_model(dosed_session, bare, justification="B without dose")
    dose = dosed_session.problem.model.sbml_model.getParameter("dose")
    assert (dose.getValue(), dose.getConstant()) == (0.0, True)
    assert "dose" not in dosed_session.problem.parameter_df.index
    assert result["reconcile"]["inputs"] == ["dose"]
    assert any(
        m["message"].startswith("NEWLY UNREAD: condition target(s) `dose`")
        for m in result["messages"]
    )


def test_a_rule_on_an_input_is_refused_at_validation(dosed_session):
    """froehlich 29 assigned `Sorafenib` by a rule; the edit was accepted and AMICI refused the
    model at compile, after the fit was started (PEtab-MCP#332)."""
    from petab_mcp.errors import ValidationError

    ruled = _DOSED.format(rate="k*dose").replace("dose = 0", "dose := 1")
    with pytest.raises(ValidationError, match=r"`dose` are computed by an assignment rule"):
        O.set_model(dosed_session, ruled, justification="dose by a rule")


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


_HILL = """model hill
  species A = 1, B = 0, C = 1
  X := (A + B)/K
  r1: A => B; k*{rate}
  K = 1; k = 1; h = 2
end
"""


@pytest.mark.parametrize(
    ("rate", "estimated", "fix"),
    [
        ("A^h/(K^h + A^h)", "h", "max(A, 0)^h"),  # a Hill term, its power named once
        ("X^h", "h", "max(X, 0)^h"),  # through an assignment rule
        ("pow(A + B, h + 1)", "h", "max(A + B, 0)^(h + 1)"),
        ("max(A, 0)^h", "h", None),
        ("abs(A)^h", "h", None),
        ("A^h", "", None),  # a fixed exponent
        ("A^2.5", "h", None),
        ("C^h*A", "h", None),  # C is in no reaction
    ],
)
def test_a_state_raised_to_an_estimated_power_is_warned_with_its_guard(rate, estimated, fix):
    """#323: fujita's delivered models raised states to estimated powers (`PIP3^h_p`). A state that
    dipped below 0 after a held-out pulse made the power NaN, and 11 of 25 runs scored no held-out
    data, with no warning at training time."""
    import types

    import pandas as pd

    model = SbmlModel.from_antimony(_HILL.format(rate=rate))
    ids = ["h", "k", "K"]
    pdf = pd.DataFrame(
        {"estimate": [i == estimated for i in ids]}, index=pd.Index(ids, name="parameterId")
    )
    issues = _power_issues(types.SimpleNamespace(model=model, parameter_df=pdf))
    said = [i["message"] for i in issues]
    assert len(said) == (fix is not None), said
    if fix is not None:
        assert said[0].startswith("reaction `r1` raises a state to an estimated power, `")
        assert said[0].endswith(f"Write `{fix}`."), said[0]


def test_a_power_of_a_state_reaches_the_edits_messages(session):
    """The lint runs on every edit, beside the other advisories, and says what to write instead."""
    antimony = _TOY_BASE.replace("k1 * A;", "k1 * A^h; h = 2;")
    said = [
        m
        for m in O.set_model(session, antimony, justification="cooperative")["messages"]
        if "estimated power" in m["message"]
    ]
    assert said == [
        {
            "level": "warning",
            "source": "validation",
            "message": "reaction `r1` raises a state to an estimated power, `A^h`: the integrator "
            "can take a state just below 0, as after a pulse, where a non-integer power is NaN "
            "and the simulation fails. Write `max(A, 0)^h`.",
        }
    ]


_ZERO = """model zero
  species P = 0, Q = 0
  r1: -> P; k
  r2: -> Q; k*{rate}
  B := c*exp(s)
  k = 1; h = 2; s = 0.3; c = 1
end
"""


@pytest.mark.parametrize(
    ("rate", "lower", "when"),
    [
        ("B^h", 0.5, "is 0 in condition `ctrl`"),  # an input the condition switches off
        ("P^h", 0.5, "starts at 0"),
        ("max(P, 0)^h", 0.5, "starts at 0"),  # the guard against a dip below 0 does not help
        ("B^h", 1.0, None),
        ("(B + 1)^h", 0.5, None),
    ],
)
def test_a_base_at_zero_raised_to_an_exponent_that_may_go_below_one_is_found(rate, lower, when):
    """#341 (lun 63612603_109): `M2d^hM2`, its base an input the non-MAP2K2 conditions set to 0,
    fitted hM2 to 0.998 and left the gradient NaN at that optimum. Below 1 the power's slope at 0 is
    infinite: in AMICI 1.2 a readout's sensitivities there are NaN, and a rate reading `P^h` with
    P(0) = 0 fails to simulate (AMICI_FIRST_SRHSFUNC_ERR), with or without `max`/`abs` on the base;
    h = 1.5 runs clean. An edit raises such an exponent's lower bound to 1 (#382)."""
    import types

    import pandas as pd

    model = SbmlModel.from_antimony(_ZERO.format(rate=rate))
    pdf = pd.DataFrame(
        {"estimate": [True, True], "lowerBound": [lower, 1e-3]},
        index=pd.Index(["h", "k"], name="parameterId"),
    )
    cdf = pd.DataFrame(
        {"conditionId": ["ctrl", "treat"], "targetId": ["c", "c"], "targetValue": [0.0, 1.0]}
    )
    problem = types.SimpleNamespace(model=model, parameter_df=pdf, condition_df=cdf)
    found = exponents_below_one_at_zero(problem)
    base = rate.rsplit("^", 1)[0]
    assert found == ({"h": f"reaction `r2` raises `{base}`, which {when}"} if when else {})


def _problem(body: str, formula: str = "B", estimated=(), targets=()):
    """A problem over the Antimony ``body``, observing ``formula`` as ``obs_B`` at one time point,
    with ``estimated`` the parameter rows the fit estimates and ``targets`` the condition inputs."""
    import types

    import pandas as pd

    observables = pd.DataFrame(
        {"observableFormula": [formula], "noiseFormula": ["sigma_B"]},
        index=pd.Index(["obs_B"], name="observableId"),
    )
    measurements = pd.DataFrame({"observableId": ["obs_B"], "time": [1.0], "measurement": [0.5]})
    conditions = pd.DataFrame({"targetId": list(targets), "targetValue": [1.0] * len(targets)})
    parameters = pd.DataFrame(
        {"estimate": [True] * len(estimated)}, index=pd.Index(list(estimated), name="parameterId")
    )
    return types.SimpleNamespace(
        model=SbmlModel.from_antimony(f"model m\n{body}\nend\n"),
        observable_df=observables,
        measurement_df=measurements,
        condition_df=conditions,
        parameter_df=parameters,
    )


_SPECIES = "species A, B, E; A = 1; B = 0; E = 1; k = 1; K = 1;"


@pytest.mark.parametrize(
    ("reaction", "warned"),
    [
        ("r1: A => B; k*E", True),  # brannmark 40: reads another species
        ("r1: A => ; k", True),  # a zero-order drain
        ("r1: A => B; k*A", False),
        ("r1: A => B; k*f; f := A/(A + K)", False),  # read through a rule
        ("r1: A + E => B + E; k*A", False),  # E is not consumed
        ("r1: $A => B; k", False),  # a boundary species is not consumed
    ],
)
def test_a_reaction_consuming_a_species_its_rate_never_reads_is_named(reaction, warned):
    """brannmark 40's `ir_autophos: IRi -> IRP; k_act*f_occ*IR` made IRP from an empty IRi pool,
    committed with no word and stayed the best version for about 15 edits (PEtab-MCP#324)."""
    issues = _unread_reactant_issues(_problem(f"{_SPECIES}\n{reaction}"))
    assert [(i["check"], i["level"]) for i in issues] == (
        [("reactant_unread", "warning")] if warned else []
    )
    if reaction == "r1: A => B; k*E":
        assert issues[0]["message"] == (
            "reaction `r1` consumes `A`, but its rate `k * E` never reads it, so it runs on once "
            "`A` is gone and `A` goes negative. Read `A` in the rate (`k*A`; a constant drain that "
            "stops at 0 is `piecewise(v, A > 0, 0)`), or, if the reaction is a source, take `A` "
            "off its reactants."
        )


_REACTION = "species A, B; A = 1; B = 0; k = 1; s = 1; k_B = 1;\nr1: A => B; k*A"


@pytest.mark.parametrize(
    ("readout", "warned"),
    [
        ("readout_B := 0", True),  # crauste 19
        ("readout_B := k_B", True),  # a free constant, fitted to the data's mean
        ("readout_B := s*B", False),
        ("readout_B := s*g; g := B/(B + k)", False),  # read through a rule
        ("readout_B := piecewise(s, time > 2, 0)", False),
        ("readout_B := s*dose; dose = 1", False),  # a condition input
    ],
)
def test_a_measured_readout_that_reads_no_state_is_named(readout, warned):
    """crauste 19's `readout_Naive` set to 0 passed with no word (PEtab-MCP#324)."""
    problem = _problem(f"{_REACTION}\n{readout}", formula="readout_B", targets=["dose"])
    issues = _constant_readout_issues(problem)
    assert [i["message"] for i in issues] == (
        [
            "observable(s) `obs_B` have measurements but read no species, state, condition input "
            "or time, even through the rules they read, so each predicts one value at every time "
            "point and condition. Write each readout over the species it measures."
        ]
        if warned
        else []
    )
    problem.measurement_df["observableId"] = "obs_other"
    assert _constant_readout_issues(problem) == []  # an unmeasured observable is not named


@pytest.mark.parametrize(
    ("body", "estimated", "warned"),
    [
        ("f := k_dead*A", ["k", "k_dead"], True),
        ("f := g*A; g := k_dead", ["k", "k_dead"], True),  # a dead rule read by a dead rule
        ("f := k_dead*A", ["k"], False),  # fixed: charged nothing
        ("f := k_dead*A; r2: B => ; f*B", ["k", "k_dead"], False),
        ("f := k*A", ["k"], False),  # `k` is read by `r1` too
    ],
)
def test_an_estimated_parameter_only_dead_rules_read_is_named(body, estimated, warned):
    """Estimated parameters read only by rules nothing reads were fitted, flat, and charged in k
    (PEtab-MCP#324)."""
    issues = _dead_rule_parameter_issues(_problem(f"{_REACTION}\n{body}", estimated=estimated))
    rule = "g" if "g :=" in body else "f"
    assert [i["message"] for i in issues] == (
        [
            f"estimated parameter(s) `k_dead` are read only by the rule(s) for `{rule}`, which "
            "reach no rate, initial amount, event or observable, so the fit cannot move them and "
            "each is still charged in k. Read those rules where they should act, or remove them."
        ]
        if warned
        else []
    )
