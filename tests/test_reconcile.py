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

# The author parametrises an IC as a COMPOUND expression of parameters (r2 = r_tot*r2_fold),
# as several published models do — it must be preserved, not clobbered with a fresh r2_init=0.
IC_COMPOUND_ANTIMONY = """model ic_compound
  compartment c = 1;
  species r1 in c; species r2 in c;
  r1 = r_tot;
  r2 = r_tot * r2_fold;
  r_tot = 5.0; r2_fold = 3.0;
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
    # r2 = r_tot*r2_fold must be KEPT: no r2_init clobber, and both parameters surface as IC
    # parameters (so the parameter-table sync keeps/estimates them rather than dropping the seed).
    sm = SbmlModel.from_antimony(IC_COMPOUND_ANTIMONY).sbml_model
    ic = R.ensure_parametric_initial_conditions(sm)
    assert sm.getParameter("r2_init") is None  # the compound IC is NOT clobbered
    assert {"r_tot", "r2_fold"} <= set(ic)  # both compound-IC params reported
    assert set(R._ic_parameter_ids(sm)) >= {"r_tot", "r2_fold"}
    # the initial assignment still evaluates to the product (expression untouched)
    ia = next(
        sm.getInitialAssignment(i)
        for i in range(sm.getNumInitialAssignments())
        if sm.getInitialAssignment(i).getSymbol() == "r2"
    )
    assert R._ast_names(ia.getMath()) == {"r_tot", "r2_fold"}


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
    row regenerated into the draft TSV on every commit (one agent was measured re-requesting the
    deletion of a minted `B_init` 17 times)."""
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
    structurally non-identifiable (e.g. 16 steady-state states that sum to 1; the agent has no
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
    """A start declared at 1 stayed FIXED at 1 through every later declaration. The row follows
    the value the model declares now."""
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
    species-initial override, e.g. per-genotype baselines) is driven by the
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
# plus a genuine kinetic ``k``. This is how published condition tables carry parametric inputs
# (ligand doses, stimulus steps, rate multipliers).
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
    """A species made by zero-order synthesis was reported as an empty pool with nothing to say
    that its start was a default, not a fact."""
    prob = _bare_problem(
        "model m\n  compartment c = 1;\n  species P in c; species Q in c; P = 0; Q = 0;\n"
        "  ks = 0.1; kd = 0.1;\n  syn: -> P; ks;\n  conv: P -> Q; kd*P;\nend\n"
    )
    R.reconcile(prob)
    rep = {r["species"]: r for r in R.initial_conditions_report(prob)}
    assert "`seeds`" in rep["P"]["note"] and "`seeds`" not in rep["Q"]["note"]


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
    """An input an assignment rule holds at a condition column was reported as an empty pool,
    from the literal 0 its species keeps and nothing reads."""
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
    shipped condition table with parametric inputs ships verbatim."""
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


def test_a_rate_factor_rule_carries_its_species_influence():
    """A statements model's catalyst enters its reaction only through the rate factor the reaction
    reads (``reduction``); read off the rate law alone, a kinase that is never itself modified
    influenced nothing, and the edit warned it was a disconnected hidden species."""
    edges = R.species_influence_graph(
        _sm(
            """model rate_factor
  compartment c = 1;
  species K in c; species S in c; species pS in c;
  K = 1; S = 1; pS = 0; k = 1;
  k_catalysed := k * K;
  r: S => pS; k_catalysed * S;
end
"""
        )
    )
    assert edges["K"] == {"S", "pS"}


def test_input_species_flags_const_boundary_and_rule_driven():
    inputs = R.input_species(_sm(INPUT_ANTIMONY))
    assert "I" in inputs  # const species -> a held input
    assert "F" in inputs  # assignment-rule-driven -> a forcing term
    assert "X" not in inputs  # a free dynamic state, not an input


def test_symbolic_noise_formulas_are_memoised_on_the_TABLE_TEXT(session):
    """``petab.get_symbolic_noise_formulas`` parses the WHOLE observable table, and the gain
    resolution needs it once per OBSERVABLE — an n-observable table paid n*n antlr parses on every
    call, which dominated what both the replicate lack-of-fit arm and the pooled noise floor spend
    resolving gains (measured per sigma on 16 observables: 35.8 -> 5.8 ms). The cache is keyed on
    the formulas' TEXT and not on the frame's identity, because the agent RE-AUTHORS the observable
    table mid-session and a stale expression would feed a stop gate."""
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
# Verbatim shape of the construct that produced 69 rejected `make_changes` calls in one
# measurement (taken from its smallest real instance). The agent's own SBML is legal; the server
# then minted `R_A_init` + an initialAssignment on a symbol that already carries an assignment rule,
# which libSBML rejects with "Cannot set a value using both an initial assignment and an assignment
# rule simultaneously" -- and the agent was blamed for it.
RULE_TARGET_ANTIMONY = """model rule_target
  compartment cell = 1;
  species x in cell; species y in cell; species z in cell;
  species R_A in cell; species R_O in cell;
  x = 0.5; y = 0.3; z = 0.1;
  R_A := x*y*(1-z);
  R_O := 0.0;
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
    assert sm.getParameter("R_A_init") is None
    assert sm.getParameter("R_O_init") is None
    assert "R_A_init" not in ic and "R_O_init" not in ic
    # ...and no initialAssignment on them either, which is what libSBML actually objects to.
    ia_symbols = {
        sm.getInitialAssignment(i).getSymbol() for i in range(sm.getNumInitialAssignments())
    }
    assert ia_symbols.isdisjoint({"R_A", "R_O"})
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
    R._set_parametric_ic(sm, "R_A", 0.0)  # simulate the old buggy reconcile
    assert _consistency_errors(sm) != []  # document is now genuinely broken
    R.ensure_parametric_initial_conditions(sm)
    assert _consistency_errors(sm) == []  # healed
    ia_symbols = {
        sm.getInitialAssignment(i).getSymbol() for i in range(sm.getNumInitialAssignments())
    }
    assert "R_A" not in ia_symbols  # the illegal IA is gone...
    # ...AND so is the parameter it pointed at. Leaving `R_A_init` behind would keep an ESTIMATED
    # row that the assignment rule overrides at t=0 — a free ln(n) of BIC on every subsequent fit
    # with zero effect on the trajectory, which is exactly the artefact
    # `ensure_parametric_initial_conditions` exists to prevent.
    assert sm.getParameter("R_A_init") is None


# -- prune/sync bugs measured sessions exposed ------------------------------------------------
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


# -- a parameter the model never assigns still gets a value (#38) -------------------------
# The shape a wholesale `antimony` edit with a `parameters` entry produced: `k_deg` and the
# exponent `n` appear in a rate law and are never assigned. `k_third` has no value either, but an
# initial assignment instead.
_UNASSIGNED = (
    "model unassigned\n  compartment c = 1;\n  species A in c; A = 2.5;\n  k_third = 1/3;\n"
    "  deg: A -> ; k_deg*k_third*A^n;\nend\n"
)


def test_an_unassigned_parameter_is_seeded_not_exported_empty():
    """An unassigned parameter kept no SBML value: its row's nominal was NaN and the stored model
    read `k_deg = ;`, so anything simulating the model without the table got NaN (#38)."""
    from petab.v2.C import ESTIMATE, LOWER_BOUND, NOMINAL_VALUE, UPPER_BOUND
    from petab_mcp.constants import EXPONENT_LOWER_BOUND, EXPONENT_UPPER_BOUND
    from petab_mcp.problem import LOG10, PARAMETER_SCALE

    prob = _bare_problem(_UNASSIGNED)
    # Named for operations._orphan_warnings (#264); `k_third`'s assignment is its value.
    assert R.reconcile(prob)["parameters"]["seeded"] == ["k_deg", "n"]
    row = prob.parameter_df.loc["k_deg"]
    assert (row[NOMINAL_VALUE], row[PARAMETER_SCALE]) == (R.SEED_INITIAL_VALUE, LOG10)
    assert bool(row[ESTIMATE])
    # An exponent's seed sits inside the exponent bracket rather than widening it.
    row = prob.parameter_df.loc["n"]
    bracket = (EXPONENT_LOWER_BOUND, EXPONENT_UPPER_BOUND)
    assert (row[NOMINAL_VALUE], (row[LOWER_BOUND], row[UPPER_BOUND])) == (bracket[0], bracket)
    sm = prob.model.sbml_model
    assert sm.getParameter("k_deg").getValue() == R.SEED_INITIAL_VALUE
    assert not sm.getParameter("k_third").isSetValue()  # its initial assignment is its value
    ant = prob.model.to_antimony()
    assert "= ;" not in ant and "k_third = 1/3;" in ant


def test_an_unassigned_parameter_takes_its_rows_nominal():
    """Re-authored without its assignment, a parameter takes the value its row already holds (a
    fitted nominal, or the shipped one), not the seed, and the rows stay as they were."""
    from petab.v2.C import NOMINAL_VALUE

    prob = _bare_problem(_UNASSIGNED)
    R.reconcile(prob)
    prob.parameter_df.loc["k_deg", NOMINAL_VALUE] = 0.0019  # a fit's write-back
    shipped = prob.parameter_df.loc[["n"]].copy()
    shipped.loc["n", NOMINAL_VALUE] = 2.0
    prob.parameter_df = prob.parameter_df.drop(index="n")
    prob.model = SbmlModel.from_antimony(_UNASSIGNED)
    assert R.reconcile(prob, shipped=shipped)["parameters"]["seeded"] == []  # a row is a definition
    sm = prob.model.sbml_model
    assert (sm.getParameter("k_deg").getValue(), sm.getParameter("n").getValue()) == (0.0019, 2.0)
    assert prob.parameter_df.loc["k_deg", NOMINAL_VALUE] == 0.0019
    assert prob.parameter_df.loc["n", NOMINAL_VALUE] == 2.0


def test_an_ic_row_keeps_its_convention_when_its_value_is_dropped():
    """Re-authored as `P = P_init;` with no value, a unit-normalisation IC read NaN through the
    {0,1} convention and flipped to ESTIMATED with a NaN nominal (#38)."""
    from petab.v2.C import ESTIMATE, NOMINAL_VALUE

    prob = _bare_problem(_DECLARED_P.format(1.0))
    R.reconcile(prob)
    prob.model = SbmlModel.from_antimony(_DECLARED_P.replace(" P_init = {};", ""))
    R.reconcile(prob)
    row = prob.parameter_df.loc["P_init"]
    assert (row[NOMINAL_VALUE], bool(row[ESTIMATE])) == (1.0, False)
    assert prob.model.sbml_model.getParameter("P_init").getValue() == 1.0


# -- a rate law's local parameters are promoted to global ones (#102) -----------------------------
# Antimony cannot write a local parameter, so each test adds one through libSBML, as the barebones
# null arm's hand-written SBML would carry it.
_RATES = (
    "model rates\n  compartment c = 1;\n  species A in c; species B in c;\n  A = 2.5; B = 0.5;\n"
    "  k = 5; {kf} = 0.3;\n  r1: A -> B; {law};\n  r2: B -> ; k*B;\nend\n"
)


def _add_local(prob, reaction, pid, value):
    lp = prob.model.sbml_model.getReaction(reaction).getKineticLaw().createLocalParameter()
    lp.setId(pid)
    lp.setValue(value)


def _rates(sm):
    """Each reaction's rate at the model's declared values, a local parameter read over a global of
    the same id as SBML scopes it."""
    import libsbml

    values = {p.getId(): p.getValue() for p in sm.getListOfParameters()}
    values |= {s.getId(): s.getInitialConcentration() for s in sm.getListOfSpecies()}
    rates = {}
    for reaction in sm.getListOfReactions():
        law = reaction.getKineticLaw()
        scope = values | {p.getId(): p.getValue() for p in law.getListOfLocalParameters()}
        rates[reaction.getId()] = R._evaluate(libsbml.formulaToL3String(law.getMath()), scope)
    return rates


def test_a_local_parameter_is_charged_as_a_declared_one():
    """A local parameter got no row and was priced as no literal, so a fitted rate constant written
    as one was free (#102). It now costs exactly what the same constant declared globally under the
    promoted id costs: the same estimated row, at the same value."""
    from petab_mcp import literals as L

    declared = _bare_problem(_RATES.format(kf="r1_kf", law="r1_kf*A"))
    local = _bare_problem(_RATES.format(kf="kf", law="kf*A"))
    local.model.sbml_model.removeParameter("kf")
    _add_local(local, "r1", "kf", 0.3)
    for prob in (declared, local):
        R.reconcile(prob)
    assert local.parameter_df.sort_index().equals(declared.parameter_df.sort_index())
    assert "r1_kf" in local.x_free_ids
    assert L.hardcoded_constants(local.model.sbml_model) == []
    assert _rates(local.model.sbml_model) == _rates(declared.model.sbml_model)


def test_a_local_parameter_shadowing_a_global_keeps_its_own_value():
    """r1's local ``k`` shadows the global ``k`` r2 reads, and a global ``r1_k`` already holds the
    id the promotion would take. Each keeps its own value and row, and every rate is unchanged."""
    import pytest
    from petab.v2.C import NOMINAL_VALUE

    prob = _bare_problem(_RATES.format(kf="r1_k", law="k*r1_k*A"))
    _add_local(prob, "r1", "k", 0.7)
    sm = prob.model.sbml_model
    before = _rates(sm)
    assert before == pytest.approx({"r1": 0.7 * 0.3 * 2.5, "r2": 5 * 0.5})
    R.reconcile(prob)
    assert _rates(sm) == pytest.approx(before)
    nominal = prob.parameter_df[NOMINAL_VALUE]
    assert {pid: nominal[pid] for pid in ("k", "r1_k", "r1_k_1")} == {
        "k": 5,
        "r1_k": 0.3,
        "r1_k_1": 0.7,
    }


# -- what the reconcile must not change about the authored model ----------------------------
def _ic_parameter(sm, sid):
    """The one parameter species ``sid``'s initial assignment names, and its value."""
    (pid,) = R._ast_names(sm.getInitialAssignmentBySymbol(sid).getMath()) & R.parameter_ids(sm)
    return pid, sm.getParameter(pid).getValue()


def test_an_orphan_species_leaves_the_ic_parameter_its_rate_law_reads():
    """`E = E_tot` declared but E in no reaction, the rate reading `E_tot` directly: pruning E
    removed `E_tot` with it, and the agent's valid model was rejected for a dangling symbol."""
    prob = _bare_problem(
        "model orphan_ic\n  compartment c = 1;\n  species A in c; species B in c; species E in c;\n"
        "  A = 5; B = 0; E = E_tot; E_tot = 2;\n  kcat = 1; Km = 1;\n"
        "  r1: A -> B; kcat*E_tot*A/(Km + A);\nend\n"
    )
    sm = prob.model.sbml_model
    out = R.reconcile(prob)
    assert out["pruned"]["species"] == ["E"]
    assert _consistency_errors(sm) == []
    assert sm.getParameter("E_tot").getValue() == 2 and "E_tot" in prob.parameter_df.index


def test_an_event_keeps_the_symbols_only_it_reads():
    """The prune walked rate laws, assignments and rules only, so a switch time and level read
    by nothing but an event, and a species only an event assigns, were removed from under it."""
    prob = _bare_problem(
        "model dosed\n  compartment c = 1;\n  species A in c; species B in c; species D in c;\n"
        "  A = 5; B = 0; D = 0;\n  k = 1; t_sw = 2; A_boost = 9;\n  r1: A -> B; k*A;\n"
        "  at time > t_sw: A = A_boost, D = 1;\nend\n"
    )
    sm = prob.model.sbml_model
    out = R.reconcile(prob)
    assert out["pruned"] == {"species": [], "parameters": [], "initial_assignments": []}
    assert _consistency_errors(sm) == []
    assert {"t_sw", "A_boost"} <= set(prob.parameter_df.index)


def test_a_held_init_name_is_refused_not_captured():
    """`A = 5` beside the author's own `A_init = 7`, read by a readout: the mint pointed A at
    that parameter, so A started at 7 (and was FIXED when the value was 1) with nothing saying so.
    Refused, naming the clash and both ways out, rather than minted under another name that the
    `<sid>_init` contract (`protect_ids`, the `*_init` fallbacks) would not recognise."""
    import pytest
    from petab_mcp.errors import ValidationError

    prob = _bare_problem(
        "model captured\n  compartment c = 1;\n  species A in c; species B in c;\n"
        "  A = 5; B = 0; k = 1; A_init = 7;\n  r1: A -> B; k*A;\n  readout_B := B/A_init;\nend\n"
    )
    with pytest.raises(ValidationError, match="holds a parameter under it.*`A = A_init`"):
        R.reconcile(prob)


def _formula(sm, sid):
    import libsbml

    return libsbml.formulaToL3String(sm.getInitialAssignmentBySymbol(sid).getMath())


def test_an_amount_on_a_concentration_species_is_applied_over_its_volume():
    """A species declared by `initialAmount` with hasOnlySubstanceUnits=false in a compartment
    of size 2 is a concentration of 2.5; the mint handed the 5 to an initial assignment, which
    SBML applies as a concentration, so the species started at twice its value. libantimony
    writes `A = 5/c` as exactly this, so the agent's own text reaches it. The parameter holds
    the author's 5 and the assignment divides, so a parametric or zero-sized compartment is the
    same case, and an amount of 2 in a volume of 2 is not read as a unit normalisation."""
    from petab.v2.C import ESTIMATE, NOMINAL_VALUE

    for size in ("2", "V0", "0"):
        prob = _bare_problem(
            f"model amount\n  compartment c = {size}; V0 = 2;\n  species A in c; species B in c;\n"
            "  A = 2/c; B = 0; k = 1;\n  r1: A -> B; k*A;\nend\n"
        )
        sm = prob.model.sbml_model
        assert sm.getSpecies("A").isSetInitialAmount()
        R.reconcile(prob)
        assert _formula(sm, "A") == "A_init / c"
        assert prob.parameter_df.loc["A_init", [NOMINAL_VALUE, ESTIMATE]].tolist() == [2.0, True]
        # ...and a second reconcile preserves the assignment as the author's own compound IC.
        R.reconcile(prob)
        assert _formula(sm, "A") == "A_init / c"


def test_a_concentration_on_a_substance_only_species_is_multiplied_by_its_volume():
    """The mirror shape: an `initialConcentration` on a hasOnlySubstanceUnits=true species, where
    an initial assignment sets an amount."""
    prob = _bare_problem(
        "model conc\n  compartment c = 2;\n  substanceOnly species A in c; species B in c;\n"
        "  A = 5; B = 0; k = 1;\n  r1: A -> B; k*A;\nend\n"
    )
    sm = prob.model.sbml_model
    sp = sm.getSpecies("A")
    sp.unsetInitialAmount()
    sp.setInitialConcentration(2.5)
    R.reconcile(prob)
    assert _formula(sm, "A") == "A_init * c"
    assert sm.getParameter("A_init").getValue() == 2.5


def test_an_assignment_naming_no_parameter_is_kept_over_a_literal():
    """Hand-written SBML may carry both an initialConcentration and an initial assignment, which
    SBML reads over it. `D = A` names no parameter, so the mint read the literal and rewrote the
    assignment to `D = D_init`, starting D at the literal instead of at A."""
    prob = _bare_problem(
        "model both\n  compartment c = 1;\n  species A in c; species B in c; species D in c;\n"
        "  A = 5; B = 0; D = A; k = 1;\n  r1: A -> B; k*A;\n  r2: D -> B; k*D;\nend\n"
    )
    sm = prob.model.sbml_model
    sm.getSpecies("D").setInitialConcentration(7.0)
    R.reconcile(prob)
    assert _formula(sm, "D") == "A" and sm.getParameter("D_init") is None


def test_an_event_target_is_kept_but_not_read_as_mechanism():
    """A flag an event writes and nothing reads survives the prune (removing it breaks the
    event) but is a CONSTANT, not KINETIC: `_model_math_names` is what the math READS, and
    the role and observation-model-reuse checks classify on that."""
    prob = _bare_problem(
        "model flagged\n  compartment c = 1;\n  species A in c; species B in c;\n"
        "  A = 5; B = 0; k = 1; flag = 0;\n  r1: A -> B; k*A;\n  at time > 2: flag = 1;\nend\n"
    )
    R.reconcile(prob)
    assert "flag" in prob.parameter_df.index
    assert R.parameter_roles(prob)["flag"] == R.ROLE_CONSTANT
    assert "flag" not in R._model_math_names(prob.model.sbml_model)


def test_the_report_declares_an_amount_as_the_concentration_it_is():
    """`initial_conditions_report` read the raw initialAmount, so a species left with its
    declaration (a condition target, or before any reconcile) was reported at twice what it
    simulates at in a compartment of size 2."""
    prob = _bare_problem(
        "model amount\n  compartment c = 2;\n  species A in c; species B in c;\n"
        "  A = 5/c; B = 0; k = 1;\n  r1: A -> B; k*A;\nend\n"
    )
    (row,) = [r for r in R.initial_conditions_report(prob) if r["species"] == "A"]
    assert row["declared_t0"] == 2.5


def test_fixed_table_parameters_say_how_the_tables_name_each_row(session):
    """PEtab-MCP#283: what a merge hint may not offer and a rename refuses with its reason. A sigma
    the fit solves analytically says so, as its bound refusal does (#274); an observation scale or
    offset is named only by the observable and measurement tables; a rate only the model reads is
    not named; one the observable reads too is kept by a tie."""
    p = session.problem
    p = session.problem = p.replace(
        observable_df=p.observable_df.assign(
            observableFormula="observableScale_obs_B * B + observableOffset_obs_B",
            observablePlaceholders="observableScale_obs_B;observableOffset_obs_B",
        ),
        measurement_df=p.measurement_df.assign(observableParameters="scale_B;offset_B"),
    )
    for pid, value in (("scale_B", 1.0), ("offset_B", 0.0)):
        p.parameter_df.loc[pid] = p.parameter_df.loc["k1"]
        p.parameter_df.loc[pid, ["parameterScale", "lowerBound", "upperBound"]] = ["lin", -10, 10]
        p.parameter_df.loc[pid, ["nominalValue", "estimate"]] = [value, True]
    assert R.fixed_table_parameters(p) == {
        "sigma_B": "measurement_tables_only",  # shipped fixed: nothing for the fit to solve
        "scale_B": "measurement_tables_only",
        "offset_B": "measurement_tables_only",
    }
    p.parameter_df.loc["sigma_B", "estimate"] = True
    assert R.fixed_table_parameters(p)["sigma_B"] == "solved_analytically"
    p = p.replace(observable_df=p.observable_df.assign(observableFormula="B / k2"))
    assert R.fixed_table_parameters(p)["k2"] == R.READ_BY_TABLES_TOO
