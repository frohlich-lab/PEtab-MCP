"""Reconcile helpers: exponent (Hill-coefficient) detection + tight linear bounds."""

from __future__ import annotations

from petab.v2.models.sbml_model import SbmlModel

from petab_mcp import reconcile as R

# A model whose production rate law uses n_hill as an EXPONENT (a Hill coefficient) and K as a
# base/threshold. alpha and delta are ordinary kinetic constants.
HILL_ANTIMONY = """model hill_test
  compartment c = 1;
  species x1 in c; species x2 in c;
  x1 = 1.0; x2 = 1.0;
  alpha = 10.0; K = 1.0; n_hill = 2.0; delta = 0.1;
  prod: => x1; alpha * K^n_hill / (K^n_hill + x2^n_hill);
  deg: x1 => ; delta * x1;
end
"""


def _sm(antimony):
    return SbmlModel.from_antimony(antimony).sbml_model


def test_exponent_symbols_detects_hill_coefficient_only():
    exps = R._exponent_symbols(_sm(HILL_ANTIMONY))
    assert "n_hill" in exps  # appears in an exponent position
    assert "K" not in exps  # a base/threshold, not an exponent
    assert "alpha" not in exps and "delta" not in exps and "x2" not in exps


def test_exponent_symbols_ignores_exp_function():
    # exp(x) is AST_FUNCTION_EXP, not a cooperativity power -> its argument is NOT an exponent.
    ant = (
        "model e_test\n  compartment c = 1;\n  species s in c;\n  s = 1.0;\n"
        "  tau = 2.0; kk = 0.1;\n  r: s => ; kk * exp(-s/tau);\nend\n"
    )
    exps = R._exponent_symbols(_sm(ant))
    assert "tau" not in exps and "kk" not in exps


def test_exponent_bounds_tight_linear_bracket_nominal():
    assert R._exponent_bounds(2.0) == (1.0, 8.0)  # typical Hill in range
    assert R._exponent_bounds(12.0) == (1.0, 12.0)  # nominal above -> bracket up
    assert R._exponent_bounds(0.5) == (0.5, 8.0)  # nominal below -> bracket down
    assert R._exponent_bounds(None) == (1.0, 8.0)  # no nominal -> plain tight range


# The author parametrises an initial condition under their OWN name (p1 = p1_0), not <sid>_init.
IC_CUSTOM_ANTIMONY = """model ic_test
  compartment c = 1;
  species p1 in c;
  p1 = p1_0;
  p1_0 = 0.8;
  kdeg = 0.1;
  deg: p1 -> ; kdeg * p1;
end
"""

# The author gives a literal IC (no parameter) — the server must still parametrise it.
IC_LITERAL_ANTIMONY = """model ic_lit
  compartment c = 1;
  species q1 in c;
  q1 = 2.5;
  kd = 0.1;
  deg: q1 -> ; kd * q1;
end
"""

# The author parametrises an IC as a COMPOUND expression of parameters (r2 = ini_R1*ini_R2fold),
# as several benchmark models do — it must be preserved, not clobbered with a fresh r2_init=0.
IC_COMPOUND_ANTIMONY = """model ic_compound
  compartment c = 1;
  species r1 in c; species r2 in c;
  r1 = ini_R1;
  r2 = ini_R1 * ini_R2fold;
  ini_R1 = 5.0; ini_R2fold = 3.0;
  koff = 0.1;
  deg1: r1 -> ; koff * r1;
  deg2: r2 -> ; koff * r2;
end
"""


def test_reconcile_respects_authors_own_ic_parameter():
    # The author's p1_0 must be recognised as the IC param — NOT clobbered by a fresh p1_init=0.
    sm = SbmlModel.from_antimony(IC_CUSTOM_ANTIMONY).sbml_model
    ic = R.ensure_parametric_initial_conditions(sm)
    assert "p1_0" in ic  # author's own IC param is kept
    assert sm.getParameter("p1_init") is None  # no redundant *_init created
    assert abs(sm.getParameter("p1_0").getValue() - 0.8) < 1e-9  # intended value preserved


def test_reconcile_parametrises_a_literal_ic():
    # A literal IC still gets the server's <sid>_init parameter (unchanged behaviour).
    sm = SbmlModel.from_antimony(IC_LITERAL_ANTIMONY).sbml_model
    ic = R.ensure_parametric_initial_conditions(sm)
    assert "q1_init" in ic
    assert abs(sm.getParameter("q1_init").getValue() - 2.5) < 1e-9


def test_reconcile_preserves_a_compound_parametric_ic():
    # r2 = ini_R1*ini_R2fold must be KEPT: no r2_init clobber, and both parameters surface as IC
    # parameters (so the parameter-table sync keeps/estimates them rather than dropping the seed).
    sm = SbmlModel.from_antimony(IC_COMPOUND_ANTIMONY).sbml_model
    ic = R.ensure_parametric_initial_conditions(sm)
    assert sm.getParameter("r2_init") is None  # the compound IC is NOT clobbered
    assert {"ini_R1", "ini_R2fold"} <= set(ic)  # both compound-IC params reported
    assert set(R._ic_parameter_ids(sm)) >= {"ini_R1", "ini_R2fold"}
    # the initial assignment still evaluates to the product (expression untouched)
    ia = next(
        sm.getInitialAssignment(i)
        for i in range(sm.getNumInitialAssignments())
        if sm.getInitialAssignment(i).getSymbol() == "r2"
    )
    assert R._ast_names(ia.getMath()) == {"ini_R1", "ini_R2fold"}


def test_sync_parameter_table_uses_linear_for_nonpositive_nominal(session):
    """A rate-law parameter initialised to 0 (or negative) cannot live on a log10 scale;
    reconcile must fall back to a linear scale rather than emit an invalid log10 row with a
    non-positive bound (which the lint would then reject).
    """
    from petab_mcp.problem import LIN, PARAMETER_SCALE

    ant = (
        "model zero_rate\n"
        "  compartment c = 1;\n"
        "  species A in c; species B in c;\n"
        "  A = 1.0; B = 0.0;\n"
        "  k_off = 0;\n"  # a rate constant initialised to zero
        "  r1: A -> B; k_off * A;\n"
        "end\n"
    )
    session._set_model_from_antimony(ant)
    R.reconcile(session.problem)
    pdf = session.problem.parameter_df
    assert pdf.loc["k_off", PARAMETER_SCALE] == LIN  # linear, NOT an invalid log10


# -- always-on IC estimation; fix only the {0, 1} value conventions (#90) -----------------
def _bare_problem(antimony):
    """A bare problem with an empty parameter table. No observable/measurement tables are
    needed — the IC estimate decision is purely value-based (estimated by default; fixed at 0/1)."""
    import pandas as pd
    from petab_mcp.problem import WorkingProblem
    from petab.v2.C import ESTIMATE, LOWER_BOUND, NOMINAL_VALUE, PARAMETER_ID, UPPER_BOUND
    from petab_mcp.problem import PARAMETER_SCALE

    prob = WorkingProblem(model=SbmlModel.from_antimony(antimony))
    prob.parameter_df = pd.DataFrame(
        columns=[PARAMETER_SCALE, LOWER_BOUND, UPPER_BOUND, NOMINAL_VALUE, ESTIMATE]
    ).rename_axis(PARAMETER_ID)
    return prob


def test_ic_estimated_by_default_empty_pool_unminted():
    """ICs are ESTIMATED by default: a positive seed is estimated on log10; an EXPLICIT zero
    (empty pool) gets NO `<sid>_init` at all — the literal 0 is the author's design, and a minted
    row regenerated into the draft TSV on every commit (the draft-arm A/B measured one lead
    re-delegating the deletion of a minted `IPR_S_init` 17 times)."""
    from petab.v2.C import ESTIMATE, NOMINAL_VALUE
    from petab_mcp.problem import LOG10, PARAMETER_SCALE

    ant = (
        "model seeded_and_empty\n"
        "  compartment c = 1;\n"
        "  species A in c; species B in c;\n"
        "  A = 2.5; B = 0.0;\n"  # A seeded (nonzero, != 1); B empty
        "  k1 = 0.3;\n"
        "  r1: A -> B; k1 * A;\n"
        "end\n"
    )
    prob = _bare_problem(ant)
    R.reconcile(prob)
    pdf = prob.parameter_df
    assert pdf.loc["A_init", ESTIMATE] == 1  # estimated by default
    assert pdf.loc["A_init", PARAMETER_SCALE] == LOG10  # positive -> log10
    assert abs(pdf.loc["A_init", NOMINAL_VALUE] - 2.5) < 1e-9
    assert "B_init" not in pdf.index  # explicit zero -> no row minted
    assert prob.model.sbml_model.getParameter("B_init") is None


def test_ic_unit_normalization_stays_fixed():
    """An IC declared exactly 1.0 is a unit normalization (a conserved-total convention) and stays
    FIXED — decided by value, so it names no species and leaks nothing (issue #90)."""
    from petab.v2.C import ESTIMATE
    from petab_mcp.problem import LIN, PARAMETER_SCALE

    ant = (
        "model unit_norm\n"
        "  compartment c = 1;\n"
        "  species R in c; species O in c;\n"
        "  R = 1.0; O = 0.0;\n"  # R is the total-receptor normalization (== 1)
        "  k1 = 0.3;\n"
        "  r1: R -> O; k1 * R;\n"
        "end\n"
    )
    prob = _bare_problem(ant)
    R.reconcile(prob)
    pdf = prob.parameter_df
    assert pdf.loc["R_init", ESTIMATE] == 0  # unit normalization -> FIXED
    assert pdf.loc["R_init", PARAMETER_SCALE] == LIN
    assert "O_init" not in pdf.index  # explicit zero -> no row minted


def test_ic_convention_fixes_a_pre_existing_estimated_row():
    """The {0,1}->FIXED IC convention must reach a `<sid>_init` row that ALREADY EXISTS as an
    estimated seed artifact, not only a freshly-added one.

    A model-less problem is seeded with a content-free model whose species come from the
    observables; reconcile parametrises their ICs into ESTIMATED `<sid>_init` rows (nominal 0.1).
    When the agent then authors the real mechanism with the SAME state names, those rows are
    already in the table, so the add-loop (which only reaches `valid - table_ids`) skips them and
    the value-based convention never runs — leaving a 0/1 IC spuriously ESTIMATED and
    structurally non-identifiable (blasi's 16 steady-state acetylation states; the agent has no
    tool to fix them, since a parameters-arg edit sets bounds/scale only). Reconcile must re-derive
    the FIXED convention for such a row.
    """
    import pandas as pd
    from petab_mcp.problem import WorkingProblem
    from petab.v2.C import ESTIMATE, LOWER_BOUND, NOMINAL_VALUE, PARAMETER_ID, UPPER_BOUND
    from petab_mcp.problem import LIN, LOG10, PARAMETER_SCALE

    ant = (
        "model seed_collision\n"
        "  compartment c = 1;\n"
        "  species R in c; species O in c; species G in c;\n"
        "  R = 1.0; O = 0.0; G = 2.5;\n"  # R unit-norm, O empty pool, G a positive guess
        "  k1 = 0.3;\n"
        "  r1: R -> O; k1 * R;\n"
        "  r2: O -> G; k1 * O;\n"
        "end\n"
    )
    prob = WorkingProblem(model=SbmlModel.from_antimony(ant))
    # Mint the SBML <sid>_init parameters, then PRE-POPULATE the table with estimated seed rows
    # (nominal 0.1, estimate 1) — exactly the leftover a content-free seed of R/O/G would leave.
    R.ensure_parametric_initial_conditions(prob.model.sbml_model)
    rows = {pid: [LOG10, 1e-4, 1e4, 0.1, 1] for pid in ("R_init", "O_init", "G_init")}
    prob.parameter_df = pd.DataFrame.from_dict(
        rows,
        orient="index",
        columns=[PARAMETER_SCALE, LOWER_BOUND, UPPER_BOUND, NOMINAL_VALUE, ESTIMATE],
    ).rename_axis(PARAMETER_ID)

    R.reconcile(prob)
    pdf = prob.parameter_df
    # R == 1.0 (unit normalization): re-derived to FIXED at the model value, NOT left estimated
    # at the seed's 0.1.
    assert pdf.loc["R_init", ESTIMATE] == 0
    assert pdf.loc["R_init", PARAMETER_SCALE] == LIN
    assert abs(pdf.loc["R_init", NOMINAL_VALUE] - 1.0) < 1e-9
    # Bounds are re-derived to BRACKET the fixed nominal (as the add-loop does), NOT left at the
    # stale positive-log10 seed bounds.
    assert pdf.loc["R_init", LOWER_BOUND] <= 1.0 <= pdf.loc["R_init", UPPER_BOUND]
    # O == 0.0 (explicit empty pool): no `<sid>_init` is minted for it any more, so the stale
    # estimated seed row is PRUNED with the parameter — stronger than the old FIXED-at-0 heal
    # (the non-identifiable row is gone from the table and from the draft TSV entirely).
    assert "O_init" not in pdf.index
    # G == 2.5 (a positive guess, not 0/1) stays ESTIMATED — the convention only FIXES 0/1, and a
    # legitimately-estimated IC's fitted nominal must never be clobbered.
    assert pdf.loc["G_init", ESTIMATE] == 1


# A species started at a parametric IC ``P_init`` declared at ``{}``.
_DECLARED_P = (
    "model m\n  compartment c = 1;\n  species P in c; P = P_init; P_init = {};\n"
    "  kd = 0.1;\n  deg: P -> ; kd*P;\nend\n"
)


def test_an_ic_row_follows_each_redeclared_initial():
    """Fujita r2: a start declared at 1 stayed FIXED at 1 through every later declaration. The row
    follows the value the model declares now."""
    from petab.v2.C import ESTIMATE, NOMINAL_VALUE
    from petab_mcp.problem import LIN, LOG10, PARAMETER_SCALE

    prob = _bare_problem(_DECLARED_P.format(1.0))
    rows = []
    for value in (1.0, 5.0, 0.0, 1.0):
        prob.model = SbmlModel.from_antimony(_DECLARED_P.format(value))
        R.reconcile(prob)
        row = prob.parameter_df.loc["P_init"]
        rows.append((row[NOMINAL_VALUE], bool(row[ESTIMATE]), row[PARAMETER_SCALE]))
    assert rows == [(1.0, False, LIN), (5.0, True, LOG10), (0.0, False, LIN), (1.0, False, LIN)]


def test_a_redeclared_guess_keeps_its_fit():
    """Re-declaring an estimated IC at another guess must not discard its fitted value or bounds."""
    from petab.v2.C import ESTIMATE, LOWER_BOUND, NOMINAL_VALUE, UPPER_BOUND

    prob = _bare_problem(_DECLARED_P.format(5.0))
    R.reconcile(prob)
    prob.parameter_df.loc["P_init", [NOMINAL_VALUE, LOWER_BOUND, UPPER_BOUND]] = [7.3, 0.01, 50.0]
    prob.model = SbmlModel.from_antimony(_DECLARED_P.format(3.0))
    R.reconcile(prob)
    row = prob.parameter_df.loc["P_init"]
    assert bool(row[ESTIMATE]) and row[NOMINAL_VALUE] == 7.3
    assert (row[LOWER_BOUND], row[UPPER_BOUND]) == (0.01, 50.0)


def test_a_shipped_ic_row_does_not_follow_the_declaration():
    """A row the problem ships is its author's: it stays FIXED at its value whatever is declared."""
    from petab.v2.C import ESTIMATE, NOMINAL_VALUE

    prob = _bare_problem(_DECLARED_P.format(1.0))
    shipped = prob.parameter_df.copy()
    shipped.loc["P_init"] = ["lin", 0.0, 10.0, 2.0, False]
    for value in (1.0, 5.0):
        prob.model = SbmlModel.from_antimony(_DECLARED_P.format(value))
        R.reconcile(prob, shipped=shipped)
        row = prob.parameter_df.loc["P_init"]
        assert (row[NOMINAL_VALUE], bool(row[ESTIMATE])) == (2.0, False)


def test_ic_convention_does_not_fix_a_compound_expression_participant():
    """The pre-existing-row {0,1}->FIXED pass is restricted to DIRECT ICs (`<sid> = <param>`). A
    participant of a COMPOUND parametric IC (`r2 = a*b`) valued exactly 1 is a fold/normalisation
    seed the author means to ESTIMATE, not a unit-normalisation IC to fix — so it must be left
    estimated even though its value is 1.0 and it is (via the compound expression) an IC param."""
    import pandas as pd
    from petab_mcp.problem import WorkingProblem
    from petab.v2.C import ESTIMATE, LOWER_BOUND, NOMINAL_VALUE, PARAMETER_ID, UPPER_BOUND
    from petab_mcp.problem import LOG10, PARAMETER_SCALE

    ant = (
        "model ic_compound_unit\n"
        "  compartment c = 1;\n"
        "  species r2 in c;\n"
        "  r2 = a * b;\n"  # a, b appear ONLY in a compound IC expression
        "  a = 2.0; b = 1.0;\n"  # b == 1.0 would be a unit-norm IC if it were direct
        "  koff = 0.1;\n"
        "  deg2: r2 -> ; koff * r2;\n"
        "end\n"
    )
    prob = WorkingProblem(model=SbmlModel.from_antimony(ant))
    R.ensure_parametric_initial_conditions(prob.model.sbml_model)
    rows = {pid: [LOG10, 1e-4, 1e4, val, 1] for pid, val in (("a", 2.0), ("b", 1.0))}
    prob.parameter_df = pd.DataFrame.from_dict(
        rows,
        orient="index",
        columns=[PARAMETER_SCALE, LOWER_BOUND, UPPER_BOUND, NOMINAL_VALUE, ESTIMATE],
    ).rename_axis(PARAMETER_ID)

    R.reconcile(prob)
    pdf = prob.parameter_df
    assert pdf.loc["b", ESTIMATE] == 1  # NOT fixed despite value 1.0 (compound part)
    assert pdf.loc["a", ESTIMATE] == 1


def test_condition_target_species_ic_not_parametrised():
    """A species whose initial value is set PER CONDITION by a condition target (a PEtab
    species-initial override, e.g. armistead's per-genotype metabolite baselines) is driven by the
    condition table — reconcile must NOT mint an estimated ``<sid>_init`` for it (the per-condition
    override would silently supersede it, inflating k with a non-identifiable parameter)."""
    import pandas as pd
    from petab_mcp.problem import WorkingProblem
    from petab.v2.C import (
        CONDITION_ID,
        ESTIMATE,
        LOWER_BOUND,
        NOMINAL_VALUE,
        PARAMETER_ID,
        TARGET_ID,
        TARGET_VALUE,
        UPPER_BOUND,
    )
    from petab_mcp.problem import PARAMETER_SCALE

    ant = (
        "model condcol_ic\n"
        "  compartment c = 1;\n"
        "  species M in c; species P in c;\n"
        "  M = 0.988; P = 0.0;\n"  # M's initial is set per condition (a species target)
        "  k1 = 0.3;\n"
        "  r1: M -> P; k1 * M;\n"
        "end\n"
    )
    prob = WorkingProblem(model=SbmlModel.from_antimony(ant))
    prob.parameter_df = pd.DataFrame(
        columns=[PARAMETER_SCALE, LOWER_BOUND, UPPER_BOUND, NOMINAL_VALUE, ESTIMATE]
    ).rename_axis(PARAMETER_ID)
    prob.condition_df = pd.DataFrame(
        {CONDITION_ID: ["wt", "mut"], TARGET_ID: ["M", "M"], TARGET_VALUE: [0.988, 0.822]}
    )
    R.reconcile(prob)
    pdf = prob.parameter_df
    assert "M_init" not in pdf.index  # condition-driven -> NOT parametrised
    assert "P_init" not in pdf.index  # explicit zero -> not parametrised either


def test_ic_estimate_is_value_based_not_observation_based():
    """Always-on: a nonzero, non-1 IC is estimated with no observable/measurement tables present —
    the decision is purely value-based (a species observed only at t>0 is no longer a special case;
    a declared guess is simply never frozen)."""
    from petab.v2.C import ESTIMATE

    ant = (
        "model decay\n"
        "  compartment c = 1;\n"
        "  species A in c; species B in c;\n"
        "  A = 1.0; B = 460.0;\n"  # B a positive guess (!= 1); A a unit normalization
        "  k1 = 0.3;\n"
        "  r1: B -> A; k1 * B;\n"
        "end\n"
    )
    prob = _bare_problem(ant)
    R.reconcile(prob)
    pdf = prob.parameter_df
    assert pdf.loc["B_init", ESTIMATE] == 1  # guessed IC -> estimated, never frozen
    assert pdf.loc["A_init", ESTIMATE] == 0  # A == 1.0 -> unit normalization, fixed


# A model driven by a per-condition input parameter ``dose`` that is ALSO a condition-table target,
# plus a genuine kinetic ``k``. This is how benchmark condition tables carry parametric inputs
# (EGF doses, insulin steps, rate multipliers).
_CONDCOL_ANTIMONY = """model m
  compartment c = 1;
  species X in c; X = 0;
  dose = 1; k = 0.1;
  r1: -> X; c*dose*k;
  r2: X -> ; c*k*X;
end"""


def _condcol_problem(antimony=_CONDCOL_ANTIMONY):
    import pandas as pd
    from petab_mcp.problem import WorkingProblem

    cond = pd.DataFrame(
        {"conditionId": ["c1", "c2"], "targetId": ["dose", "dose"], "targetValue": [1.0, 5.0]}
    )
    exps = pd.DataFrame(
        {"experimentId": ["e1", "e2"], "time": [0.0, 0.0], "conditionId": ["c1", "c2"]}
    )
    obs = pd.DataFrame(
        {"observableId": ["obs_X"], "observableFormula": ["X"], "noiseFormula": ["1"]}
    ).set_index("observableId")
    meas = pd.DataFrame(
        {
            "observableId": ["obs_X", "obs_X"],
            "experimentId": ["e1", "e2"],
            "time": [1.0, 1.0],
            "measurement": [0.3, 1.2],
        }
    )
    par = pd.DataFrame(
        {
            "parameterId": ["k"],
            "parameterScale": ["log10"],
            "lowerBound": [1e-3],
            "upperBound": [1e3],
            "nominalValue": [0.1],
            "estimate": [True],
        }
    ).set_index("parameterId")
    return WorkingProblem(
        model=SbmlModel.from_antimony(antimony),
        condition_df=cond,
        experiment_df=exps,
        observable_df=obs,
        measurement_df=meas,
        parameter_df=par,
    )


def test_condition_target_symbols_are_the_condition_targets():
    prob = _condcol_problem()
    assert R.condition_target_symbols(prob) == {"dose"}


# -- initial-condition report (issue #57) ---------------------------------------------------


def test_initial_conditions_report_fixed_by_default(session):
    """The toy problem's ICs are FIXED: the report says the declared t=0 governs the t=0 state."""
    rep = {r["species"]: r for r in R.initial_conditions_report(session.problem)}
    assert set(rep) == {"A", "B"}
    a = rep["A"]
    assert a["ic_parameter"] == "A_init"
    assert a["declared_t0"] == 1.0
    assert a["estimated"] is False
    assert a["set_by_condition"] is False
    assert a["t0_source"] == "fixed"
    assert a["declared_t0_used"] is True  # declared t=0 survives into the t=0 state
    assert rep["B"]["declared_t0"] == 0.0 and rep["B"]["declared_t0_used"] is True


def test_initial_conditions_report_flags_estimated_ic(session):
    """An ESTIMATED IC is reported as not governing t=0 (the fit sets it), not fixed."""
    from petab.v2.C import ESTIMATE

    session.problem.parameter_df.loc["A_init", ESTIMATE] = True
    a = {r["species"]: r for r in R.initial_conditions_report(session.problem)}["A"]
    assert a["estimated"] is True
    assert a["t0_source"] == "estimated"
    assert a["declared_t0_used"] is False


def test_initial_conditions_report_says_a_synthesised_species_starts_empty():
    """Fujita r2's EGFR, made by zero-order synthesis, was reported as an empty pool with nothing
    to say that its start was a default, not a fact."""
    prob = _bare_problem(
        "model m\n  compartment c = 1;\n  species P in c; species Q in c; P = 0; Q = 0;\n"
        "  ks = 0.1; kd = 0.1;\n  syn: -> P; ks;\n  conv: P -> Q; kd*P;\nend\n"
    )
    R.reconcile(prob)
    rep = {r["species"]: r for r in R.initial_conditions_report(prob)}
    assert "`starts`" in rep["P"]["note"] and "`starts`" not in rep["Q"]["note"]


def test_initial_conditions_report_needs_no_nominal_column(session):
    """`nominalValue` is optional in PEtab v2; read unguarded, its absence raised on every edit."""
    from petab.v2.C import NOMINAL_VALUE

    session.problem.parameter_df = session.problem.parameter_df.drop(columns=NOMINAL_VALUE)
    a = {r["species"]: r for r in R.initial_conditions_report(session.problem)}["A"]
    assert a["t0"] == 1.0


def test_initial_conditions_report_marks_newly_added(session):
    """`added` (the reconcile diff) flags which ICs this op introduced."""
    rep = {r["species"]: r for r in R.initial_conditions_report(session.problem, added={"A_init"})}
    assert rep["A"]["newly_added"] is True
    assert rep["B"]["newly_added"] is False


def test_initial_conditions_report_condition_target_override():
    """A species set per-condition by the condition table: its Antimony t=0 is overridden."""
    prob = _condcol_problem()
    R.ensure_parametric_initial_conditions(prob.model.sbml_model)
    # Retarget the conditions at the species id so the species itself is condition-set.
    prob.condition_df = prob.condition_df.assign(targetId="X")
    x = {r["species"]: r for r in R.initial_conditions_report(prob)}["X"]
    assert x["set_by_condition"] is True
    assert x["t0_source"] == "condition_table"
    assert x["declared_t0_used"] is False
    assert x["t0"] == {"e1": 1.0, "e2": 5.0}


def test_initial_conditions_report_reads_a_held_input_off_its_rule():
    """Fujita r2: an input an assignment rule holds at a condition column was reported as an empty
    pool, from the literal 0 its species keeps and nothing reads."""
    head, _ = _CONDCOL_ANTIMONY.rsplit("end", 1)
    prob = _condcol_problem(head + "  species E in c; E := piecewise(dose, time < 10, 0);\nend")
    R.reconcile(prob)
    e = {r["species"]: r for r in R.initial_conditions_report(prob)}["E"]
    assert e["t0_source"] == "rule" and e["declared_t0"] is None
    assert e["t0"] == {"e1": 1.0, "e2": 5.0}
    assert "piecewise(dose, time < 10, 0)" in e["note"]


def test_initial_conditions_report_reads_a_fixed_t0_off_the_parameter_table(session):
    """A FIXED IC's parameter-table row, not the model text, sets t=0; the view reported the
    declared value as governing it."""
    session.problem.model.sbml_model.getParameter("A_init").setValue(5.0)
    a = {r["species"]: r for r in R.initial_conditions_report(session.problem)}["A"]
    assert (a["declared_t0"], a["t0"], a["declared_t0_used"]) == (5.0, 1.0, False)
    assert "NOT at the declared 5.0" in a["note"]


def test_condition_target_parameter_not_synced_into_parameter_table():
    """A parameter a condition targets is a per-condition PEtab change, so reconcile must NOT write
    it into the parameter table (PEtab forbids it there) while still syncing a real kinetic — so a
    benchmark condition table with parametric inputs ships verbatim."""
    prob = _condcol_problem()
    diff = R.sync_parameter_table(prob)
    assert "dose" not in prob.parameter_df.index  # condition-target param stays out of the table
    assert "dose" not in diff["added"]
    assert "k" in prob.parameter_df.index  # a genuine kinetic is still synced
    errors = [i for i in prob.to_petab().validate() if i.level >= 30]
    assert not errors, errors  # petab accepts the verbatim condition table


# -- observable-connection gating (issue #64) ------------------------------
# A -> B -> D: B is a free state fed by A; D is a pure downstream sink. `sink`'s rate reuses B,
# so D receives mass but drives nothing.
INFLUENCE_ANTIMONY = """model influence_test
  compartment c = 1;
  species A in c; species B in c; species D in c;
  A = 1.0; B = 0.0; D = 0.0;
  k1 = 0.5; k3 = 0.2;
  r1: A => B; k1 * A;
  sink: B => D; k3 * B;
end
"""

# I is a held (const) input; F is driven by an assignment rule (a forcing term); X is a free state.
INPUT_ANTIMONY = """model input_test
  compartment c = 1;
  const species I in c;
  species X in c; species F in c;
  I = 1.0; X = 1.0;
  F := 2 * I;
  k = 0.1;
  r: X => ; k * X;
end
"""


def test_species_influence_graph_is_directional():
    edges = R.species_influence_graph(_sm(INFLUENCE_ANTIMONY))
    assert "B" in edges["A"]  # A drives its product B
    assert "D" in edges["B"]  # B drives its sink D
    assert edges.get("D", set()) == set()  # a pure sink drives nothing downstream


def test_input_species_flags_const_boundary_and_rule_driven():
    inputs = R.input_species(_sm(INPUT_ANTIMONY))
    assert "I" in inputs  # const species -> a held input
    assert "F" in inputs  # assignment-rule-driven -> a forcing term
    assert "X" not in inputs  # a free dynamic state, not an input


def test_symbolic_noise_formulas_are_memoised_on_the_TABLE_TEXT(session):
    """``petab.get_symbolic_noise_formulas`` parses the WHOLE observable table, and the gain
    resolution needs it once per OBSERVABLE — an n-observable table paid n*n antlr parses on every
    call, which dominated what both the replicate lack-of-fit arm and the pooled noise floor spend
    resolving gains (measured per sigma: blasi 35.8 -> 5.8 ms). The cache is keyed on the formulas' TEXT
    and not on the frame's identity, because the agent RE-AUTHORS the observable table mid-session
    and a stale expression would feed a stop gate."""
    odf = session.problem.observable_df.copy()
    first = R._symbolic_noise_formulas(odf)
    assert R._symbolic_noise_formulas(odf) is first  # same text -> same object, no re-parse
    odf.loc["obs_B", "noiseFormula"] = "sigma_B * B"  # re-authored
    rewritten = R._symbolic_noise_formulas(odf)
    assert rewritten is not first
    assert str(rewritten["obs_B"]) != str(first["obs_B"])
    # ...and a table this cache has never seen is parsed rather than borrowed from a stale neighbour
    assert R._symbolic_noise_formulas(odf.assign(noiseFormula="sigma_B"))["obs_B"] == first["obs_B"]


# -- assignment-rule targets must not get a parametric IC (#158 follow-up) -------------------
# Verbatim shape of the construct that produced 69 rejected `make_changes` calls in the Aug 8-10
# sweep (smallest real instance: 51526222_113 seq 16, deepseek-v4-flash on sneyd). The agent's own
# SBML is legal; the server then minted `IPR_A_init` + an initialAssignment on a symbol that already
# carries an assignment rule, which libSBML rejects with "Cannot set a value using both an initial
# assignment and an assignment rule simultaneously" -- and the agent was blamed for it.
RULE_TARGET_ANTIMONY = """model rule_target
  compartment cell = 1;
  species x in cell; species y in cell; species z in cell;
  species IPR_A in cell; species IPR_O in cell;
  x = 0.5; y = 0.3; z = 0.1;
  IPR_A := x*y*(1-z);
  IPR_O := 0.0;
  kf = 1.0;
  decay: x => ; kf * x;
end
"""

# The counter-case: a RATE rule integrates from an initial value, so its target still needs an IC.
RATE_RULE_ANTIMONY = """model rate_target
  compartment cell = 1;
  species w in cell; species v in cell;
  w = 2.0; v = 4.0;
  kf = 1.0;
  v' = -kf * v;
  decay: w => ; kf * w;
end
"""


def _consistency_errors(sm):
    """Hard libSBML errors on the document this model belongs to.

    The error log must be cleared first: libSBML ACCUMULATES across `checkConsistency()` calls on
    one document, so a second check on a repaired model still reports the first check's errors --
    which reads as "the fix did not work" when it did.
    """
    import libsbml

    doc = sm.getSBMLDocument()
    doc.getErrorLog().clearLog()
    doc.checkConsistency()
    return [
        doc.getError(i).getMessage()
        for i in range(doc.getNumErrors())
        if doc.getError(i).getSeverity() >= libsbml.LIBSBML_SEV_ERROR
    ]


def test_assignment_rule_targets_get_no_parametric_ic():
    sm = _sm(RULE_TARGET_ANTIMONY)
    ic = R.ensure_parametric_initial_conditions(sm)
    # No <sid>_init minted for either rule-driven species...
    assert sm.getParameter("IPR_A_init") is None
    assert sm.getParameter("IPR_O_init") is None
    assert "IPR_A_init" not in ic and "IPR_O_init" not in ic
    # ...and no initialAssignment on them either, which is what libSBML actually objects to.
    ia_symbols = {
        sm.getInitialAssignment(i).getSymbol() for i in range(sm.getNumInitialAssignments())
    }
    assert ia_symbols.isdisjoint({"IPR_A", "IPR_O"})
    # The free dynamic species are untouched -- the fix must not stop parametrising ordinary ICs.
    assert {"x_init", "y_init", "z_init"} <= set(ic)


def test_reconciled_rule_target_model_is_valid_sbml():
    # The end-to-end claim: after reconcile the document libSBML sees is CLEAN. This is the test
    # that would have failed before the fix.
    sm = _sm(RULE_TARGET_ANTIMONY)
    assert _consistency_errors(sm) == []  # the agent's model was always fine
    R.ensure_parametric_initial_conditions(sm)
    assert _consistency_errors(sm) == []  # and reconcile no longer breaks it


def test_rate_rule_targets_keep_their_initial_condition():
    # A rate rule is the opposite case: v' = -kf*v integrates from v(0), so v MUST keep an IC.
    # Lumping both rule kinds into the skip would silently drop an estimated initial value.
    sm = _sm(RATE_RULE_ANTIMONY)
    ic = R.ensure_parametric_initial_conditions(sm)
    assert "v_init" in ic
    assert abs(sm.getParameter("v_init").getValue() - 4.0) < 1e-9
    assert R.assignment_rule_targets(sm) == set()  # v is a rate target, not an assignment


def test_reconcile_heals_a_store_that_already_carries_the_illegal_initial_assignment():
    # A version store written before this fix can carry the illegal IA. Re-authoring over it must
    # repair the document rather than inherit the fault.
    sm = _sm(RULE_TARGET_ANTIMONY)
    R._set_parametric_ic(sm, "IPR_A", 0.0)  # simulate the old buggy reconcile
    assert _consistency_errors(sm) != []  # document is now genuinely broken
    R.ensure_parametric_initial_conditions(sm)
    assert _consistency_errors(sm) == []  # healed
    ia_symbols = {
        sm.getInitialAssignment(i).getSymbol() for i in range(sm.getNumInitialAssignments())
    }
    assert "IPR_A" not in ia_symbols  # the illegal IA is gone...
    # ...AND so is the parameter it pointed at. Leaving `IPR_A_init` behind would keep an ESTIMATED
    # row that the assignment rule overrides at t=0 — a free ln(n) of BIC on every subsequent fit
    # with zero effect on the trajectory, which is exactly the artefact
    # `ensure_parametric_initial_conditions` exists to prevent.
    assert sm.getParameter("IPR_A_init") is None


# -- prune/sync bugs the Aug 8-10 sweep exposed ----------------------------------------------
# A parameter that is the TARGET of an initialAssignment, and that nothing else references.
# `_model_math_names` walks IA math (so `a`/`b` are kept) but not IA symbols, so `total` was pruned
# and its IA left dangling -> invalid SBML -> the whole edit rejected (11 calls).
IA_TARGET_ANTIMONY = """model ia_target
  compartment c = 1;
  species s in c;
  s = 1.0;
  a = 2.0; b = 3.0;
  total = 0.0;
  total := a * b;
  kf = 1.0;
  deg: s => ; kf * s;
end
"""


def test_prune_orphans_never_leaves_a_dangling_initial_assignment():
    import libsbml
    from petab_mcp import reconcile as RR

    sm = _sm(IA_TARGET_ANTIMONY)
    # Give `total` a real initialAssignment (not an assignment rule) and no other reference.
    for i in range(sm.getNumRules()):
        if sm.getRule(i).getVariable() == "total":
            sm.removeRule(i)
            break
    ia = sm.createInitialAssignment()
    ia.setSymbol("total")
    ia.setMath(libsbml.parseL3Formula("a * b"))
    assert _consistency_errors(sm) == []

    class _P:
        pass

    prob = _P()
    prob.model = type("M", (), {"sbml_model": sm})()
    prob.parameter_df = None
    prob.condition_df = None
    prob.observable_df = None
    prob.measurement_df = None
    out = RR.prune_orphans(prob)
    # `total` may go, but never as a dangling IA: the document stays valid either way.
    assert _consistency_errors(sm) == []
    ia_symbols = {
        sm.getInitialAssignment(i).getSymbol() for i in range(sm.getNumInitialAssignments())
    }
    if "total" in out["parameters"]:
        assert "total" not in ia_symbols
        assert "total" in out["initial_assignments"]


def test_an_unset_initial_is_the_same_empty_pool_as_an_explicit_zero():
    """`species B;` with no assignment sets neither initialConcentration nor initialAmount, and
    SBML reads it as an empty pool — the same state as `B = 0;`. Both must take the no-mint
    branch, or the two spellings produce different parameter tables and the regenerating
    `<sid>_init` artifact returns one omitted assignment later."""
    ant = (
        "model unset_vs_zero\n"
        "  compartment c = 1;\n"
        "  species A in c; species B in c; species C in c;\n"
        "  A = 1.5; C = 0;\n"  # B's initial is not written at all
        "  k = 0.3;\n"
        "  r1: A -> B; k * A;\n"
        "  r2: B -> C; k * B;\n"
        "end\n"
    )
    sm = SbmlModel.from_antimony(ant).sbml_model
    assert R.ensure_parametric_initial_conditions(sm) == ["A_init"]
    assert sm.getParameter("B_init") is None and sm.getParameter("C_init") is None


def test_a_shipped_ic_row_survives_the_zero_skip():
    """If the PROBLEM ships a `<sid>_init` row, that row is part of the problem definition. The
    zero/unset skip must not delete it: sync_parameter_table would drop the row and
    validate_problem then rejects the commit, with no edit the agent could make to satisfy it
    short of misstating the biology (`protected_fixed_parameter_ids` excludes IC parameters, so
    nothing else catches this)."""
    ant = (
        "model shipped_ic\n"
        "  compartment c = 1;\n"
        "  species A in c; species C in c;\n"
        "  A = 1.5; C = 0;\n"
        "  k = 0.3;\n"
        "  r1: A -> C; k * A;\n"
        "end\n"
    )
    sm = SbmlModel.from_antimony(ant).sbml_model
    ic = R.ensure_parametric_initial_conditions(sm, protect_ids={"C_init"})
    assert "C_init" in ic and sm.getParameter("C_init") is not None
