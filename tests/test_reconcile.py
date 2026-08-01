"""Reconcile helpers: exponent (Hill-coefficient) detection + tight linear bounds."""

from __future__ import annotations

import pytest
from petab.v1.models.sbml_model import SbmlModel

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
    assert "n_hill" in exps                    # appears in an exponent position
    assert "K" not in exps                     # a base/threshold, not an exponent
    assert "alpha" not in exps and "delta" not in exps and "x2" not in exps


def test_exponent_symbols_ignores_exp_function():
    # exp(x) is AST_FUNCTION_EXP, not a cooperativity power -> its argument is NOT an exponent.
    ant = ("model e_test\n  compartment c = 1;\n  species s in c;\n  s = 1.0;\n"
           "  tau = 2.0; kk = 0.1;\n  r: s => ; kk * exp(-s/tau);\nend\n")
    exps = R._exponent_symbols(_sm(ant))
    assert "tau" not in exps and "kk" not in exps


def test_exponent_bounds_tight_linear_bracket_nominal():
    assert R._exponent_bounds(2.0) == (1.0, 8.0)          # typical Hill in range
    assert R._exponent_bounds(12.0) == (1.0, 12.0)        # nominal above -> bracket up
    assert R._exponent_bounds(0.5) == (0.5, 8.0)          # nominal below -> bracket down
    assert R._exponent_bounds(None) == (1.0, 8.0)         # no nominal -> plain tight range


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
    assert 'p1_0' in ic                            # author's own IC param is kept
    assert sm.getParameter('p1_init') is None      # no redundant *_init created
    assert abs(sm.getParameter('p1_0').getValue() - 0.8) < 1e-9   # intended value preserved


def test_reconcile_parametrises_a_literal_ic():
    # A literal IC still gets the server's <sid>_init parameter (unchanged behaviour).
    sm = SbmlModel.from_antimony(IC_LITERAL_ANTIMONY).sbml_model
    ic = R.ensure_parametric_initial_conditions(sm)
    assert 'q1_init' in ic
    assert abs(sm.getParameter('q1_init').getValue() - 2.5) < 1e-9


def test_reconcile_preserves_a_compound_parametric_ic():
    # r2 = ini_R1*ini_R2fold must be KEPT: no r2_init clobber, and both parameters surface as IC
    # parameters (so the parameter-table sync keeps/estimates them rather than dropping the seed).
    sm = SbmlModel.from_antimony(IC_COMPOUND_ANTIMONY).sbml_model
    ic = R.ensure_parametric_initial_conditions(sm)
    assert sm.getParameter('r2_init') is None            # the compound IC is NOT clobbered
    assert {'ini_R1', 'ini_R2fold'} <= set(ic)           # both compound-IC params reported
    assert set(R._ic_parameter_ids(sm)) >= {'ini_R1', 'ini_R2fold'}
    # the initial assignment still evaluates to the product (expression untouched)
    ia = next(sm.getInitialAssignment(i) for i in range(sm.getNumInitialAssignments())
              if sm.getInitialAssignment(i).getSymbol() == 'r2')
    assert R._ast_names(ia.getMath()) == {'ini_R1', 'ini_R2fold'}


def test_sync_parameter_table_uses_linear_for_nonpositive_nominal(session):
    """A rate-law parameter initialised to 0 (or negative) cannot live on a log10 scale;
    reconcile must fall back to a linear scale rather than emit an invalid log10 row with a
    non-positive bound (which the lint would then reject).
    """
    from petab.v1.C import LIN, PARAMETER_SCALE

    ant = (
        "model zero_rate\n"
        "  compartment c = 1;\n"
        "  species A in c; species B in c;\n"
        "  A = 1.0; B = 0.0;\n"
        "  k_off = 0;\n"                    # a rate constant initialised to zero
        "  r1: A -> B; k_off * A;\n"
        "end\n"
    )
    session._set_model_from_antimony(ant)
    R.reconcile(session.problem)
    pdf = session.problem.parameter_df
    assert pdf.loc["k_off", PARAMETER_SCALE] == LIN   # linear, NOT an invalid log10


# -- always-on IC estimation; fix only the {0, 1} value conventions (#90) -----------------
def _bare_problem(antimony):
    """A bare petab.Problem with an empty parameter table. No observable/measurement tables are
    needed — the IC estimate decision is purely value-based (estimated by default; fixed at 0/1)."""
    import pandas as pd
    import petab.v1 as petab
    from petab.v1.C import (ESTIMATE, LOWER_BOUND, NOMINAL_VALUE, PARAMETER_ID,
                            PARAMETER_SCALE, UPPER_BOUND)

    prob = petab.Problem(model=SbmlModel.from_antimony(antimony))
    prob.parameter_df = pd.DataFrame(
        columns=[PARAMETER_SCALE, LOWER_BOUND, UPPER_BOUND, NOMINAL_VALUE, ESTIMATE]
    ).rename_axis(PARAMETER_ID)
    return prob


def test_ic_estimated_by_default_empty_pool_fixed():
    """ICs are ESTIMATED by default: a positive seed is estimated on log10; an empty pool
    (t0 == 0) stays fixed at 0."""
    from petab.v1.C import ESTIMATE, LIN, LOG10, NOMINAL_VALUE, PARAMETER_SCALE

    ant = (
        "model seeded_and_empty\n"
        "  compartment c = 1;\n"
        "  species A in c; species B in c;\n"
        "  A = 2.5; B = 0.0;\n"                 # A seeded (nonzero, != 1); B empty
        "  k1 = 0.3;\n"
        "  r1: A -> B; k1 * A;\n"
        "end\n"
    )
    prob = _bare_problem(ant)
    R.reconcile(prob)
    pdf = prob.parameter_df
    assert pdf.loc["A_init", ESTIMATE] == 1              # estimated by default
    assert pdf.loc["A_init", PARAMETER_SCALE] == LOG10   # positive -> log10
    assert abs(pdf.loc["A_init", NOMINAL_VALUE] - 2.5) < 1e-9
    assert pdf.loc["B_init", ESTIMATE] == 0             # empty pool -> FIXED
    assert pdf.loc["B_init", PARAMETER_SCALE] == LIN


def test_ic_unit_normalization_stays_fixed():
    """An IC declared exactly 1.0 is a unit normalization (a conserved-total convention) and stays
    FIXED — decided by value, so it names no species and leaks nothing (issue #90)."""
    from petab.v1.C import ESTIMATE, LIN, PARAMETER_SCALE

    ant = (
        "model unit_norm\n"
        "  compartment c = 1;\n"
        "  species R in c; species O in c;\n"
        "  R = 1.0; O = 0.0;\n"                 # R is the total-receptor normalization (== 1)
        "  k1 = 0.3;\n"
        "  r1: R -> O; k1 * R;\n"
        "end\n"
    )
    prob = _bare_problem(ant)
    R.reconcile(prob)
    pdf = prob.parameter_df
    assert pdf.loc["R_init", ESTIMATE] == 0             # unit normalization -> FIXED
    assert pdf.loc["R_init", PARAMETER_SCALE] == LIN
    assert pdf.loc["O_init", ESTIMATE] == 0             # empty pool -> FIXED


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
    import petab.v1 as petab
    from petab.v1.C import (ESTIMATE, LIN, LOG10, LOWER_BOUND, NOMINAL_VALUE, PARAMETER_ID,
                            PARAMETER_SCALE, UPPER_BOUND)

    ant = (
        "model seed_collision\n"
        "  compartment c = 1;\n"
        "  species R in c; species O in c; species G in c;\n"
        "  R = 1.0; O = 0.0; G = 2.5;\n"        # R unit-norm, O empty pool, G a positive guess
        "  k1 = 0.3;\n"
        "  r1: R -> O; k1 * R;\n"
        "  r2: O -> G; k1 * O;\n"
        "end\n"
    )
    prob = petab.Problem(model=SbmlModel.from_antimony(ant))
    # Mint the SBML <sid>_init parameters, then PRE-POPULATE the table with estimated seed rows
    # (nominal 0.1, estimate 1) — exactly the leftover a content-free seed of R/O/G would leave.
    R.ensure_parametric_initial_conditions(prob.model.sbml_model)
    rows = {pid: [LOG10, 1e-4, 1e4, 0.1, 1] for pid in ("R_init", "O_init", "G_init")}
    prob.parameter_df = pd.DataFrame.from_dict(
        rows, orient="index",
        columns=[PARAMETER_SCALE, LOWER_BOUND, UPPER_BOUND, NOMINAL_VALUE, ESTIMATE],
    ).rename_axis(PARAMETER_ID)

    R.reconcile(prob)
    pdf = prob.parameter_df
    # R == 1.0 (unit normalization) and O == 0.0 (empty pool): both re-derived to FIXED at their
    # model value, NOT left estimated at the seed's 0.1.
    assert pdf.loc["R_init", ESTIMATE] == 0
    assert pdf.loc["R_init", PARAMETER_SCALE] == LIN
    assert abs(pdf.loc["R_init", NOMINAL_VALUE] - 1.0) < 1e-9
    assert pdf.loc["O_init", ESTIMATE] == 0
    assert abs(pdf.loc["O_init", NOMINAL_VALUE] - 0.0) < 1e-9
    # Bounds are re-derived to BRACKET the fixed nominal (as the add-loop does), NOT left at the
    # stale positive-log10 seed bounds — so the empty pool's 0.0 is not below its own lowerBound.
    for pid, v in (("R_init", 1.0), ("O_init", 0.0)):
        assert pdf.loc[pid, LOWER_BOUND] <= v <= pdf.loc[pid, UPPER_BOUND]
    # G == 2.5 (a positive guess, not 0/1) stays ESTIMATED — the convention only FIXES 0/1, and a
    # legitimately-estimated IC's fitted nominal must never be clobbered.
    assert pdf.loc["G_init", ESTIMATE] == 1


def test_ic_convention_does_not_fix_a_compound_expression_participant():
    """The pre-existing-row {0,1}->FIXED pass is restricted to DIRECT ICs (`<sid> = <param>`). A
    participant of a COMPOUND parametric IC (`r2 = a*b`) valued exactly 1 is a fold/normalisation
    seed the author means to ESTIMATE, not a unit-normalisation IC to fix — so it must be left
    estimated even though its value is 1.0 and it is (via the compound expression) an IC param."""
    import pandas as pd
    import petab.v1 as petab
    from petab.v1.C import (ESTIMATE, LOG10, LOWER_BOUND, NOMINAL_VALUE, PARAMETER_ID,
                            PARAMETER_SCALE, UPPER_BOUND)

    ant = (
        "model ic_compound_unit\n"
        "  compartment c = 1;\n"
        "  species r2 in c;\n"
        "  r2 = a * b;\n"                       # a, b appear ONLY in a compound IC expression
        "  a = 2.0; b = 1.0;\n"                 # b == 1.0 would be a unit-norm IC if it were direct
        "  koff = 0.1;\n"
        "  deg2: r2 -> ; koff * r2;\n"
        "end\n"
    )
    prob = petab.Problem(model=SbmlModel.from_antimony(ant))
    R.ensure_parametric_initial_conditions(prob.model.sbml_model)
    rows = {pid: [LOG10, 1e-4, 1e4, val, 1] for pid, val in (("a", 2.0), ("b", 1.0))}
    prob.parameter_df = pd.DataFrame.from_dict(
        rows, orient="index",
        columns=[PARAMETER_SCALE, LOWER_BOUND, UPPER_BOUND, NOMINAL_VALUE, ESTIMATE],
    ).rename_axis(PARAMETER_ID)

    R.reconcile(prob)
    pdf = prob.parameter_df
    assert pdf.loc["b", ESTIMATE] == 1                  # NOT fixed despite value 1.0 (compound part)
    assert pdf.loc["a", ESTIMATE] == 1


def test_condition_column_species_ic_not_parametrised():
    """A species whose initial value is set PER CONDITION by a condition-table column (a PEtab
    species-initial override, e.g. armistead's per-genotype metabolite baselines) is driven by the
    condition table — reconcile must NOT mint an estimated ``<sid>_init`` for it (the per-condition
    override would silently supersede it, inflating k with a non-identifiable parameter)."""
    import pandas as pd
    import petab.v1 as petab
    from petab.v1.C import (CONDITION_ID, ESTIMATE, LOWER_BOUND, NOMINAL_VALUE,
                            PARAMETER_ID, PARAMETER_SCALE, UPPER_BOUND)

    ant = (
        "model condcol_ic\n"
        "  compartment c = 1;\n"
        "  species M in c; species P in c;\n"
        "  M = 0.988; P = 0.0;\n"                # M's initial is set per condition (a species column)
        "  k1 = 0.3;\n"
        "  r1: M -> P; k1 * M;\n"
        "end\n"
    )
    prob = petab.Problem(model=SbmlModel.from_antimony(ant))
    prob.parameter_df = pd.DataFrame(
        columns=[PARAMETER_SCALE, LOWER_BOUND, UPPER_BOUND, NOMINAL_VALUE, ESTIMATE]
    ).rename_axis(PARAMETER_ID)
    prob.condition_df = pd.DataFrame(
        {"M": [0.988, 0.822]}, index=pd.Index(["wt", "mut"], name=CONDITION_ID)
    )
    R.reconcile(prob)
    pdf = prob.parameter_df
    assert "M_init" not in pdf.index                    # condition-driven -> NOT parametrised
    assert pdf.loc["P_init", ESTIMATE] == 0             # empty pool still fixed at 0


def test_ic_estimate_is_value_based_not_observation_based():
    """Always-on: a nonzero, non-1 IC is estimated with no observable/measurement tables present —
    the decision is purely value-based (a species observed only at t>0 is no longer a special case;
    a declared guess is simply never frozen)."""
    from petab.v1.C import ESTIMATE

    ant = (
        "model decay\n"
        "  compartment c = 1;\n"
        "  species A in c; species B in c;\n"
        "  A = 1.0; B = 460.0;\n"               # B a positive guess (!= 1); A a unit normalization
        "  k1 = 0.3;\n"
        "  r1: B -> A; k1 * B;\n"
        "end\n"
    )
    prob = _bare_problem(ant)
    R.reconcile(prob)
    pdf = prob.parameter_df
    assert pdf.loc["B_init", ESTIMATE] == 1             # guessed IC -> estimated, never frozen
    assert pdf.loc["A_init", ESTIMATE] == 0             # A == 1.0 -> unit normalization, fixed


# A model driven by a per-condition input parameter ``dose`` that is ALSO a condition-table column,
# plus a genuine kinetic ``k``. This is how benchmark condition tables carry parametric inputs
# (EGF doses, insulin steps, rate multipliers).
_CONDCOL_ANTIMONY = """model m
  compartment c = 1;
  species X in c; X = 0;
  dose = 1; k = 0.1;
  r1: -> X; c*dose*k;
  r2: X -> ; c*k*X;
end"""


def _condcol_problem():
    import pandas as pd
    import petab.v1 as petab

    cond = pd.DataFrame({"conditionId": ["c1", "c2"], "dose": [1.0, 5.0]}).set_index("conditionId")
    obs = pd.DataFrame({"observableId": ["obs_X"], "observableFormula": ["X"],
                        "noiseFormula": ["1"]}).set_index("observableId")
    meas = pd.DataFrame({"observableId": ["obs_X", "obs_X"], "simulationConditionId": ["c1", "c2"],
                         "time": [1.0, 1.0], "measurement": [0.3, 1.2]})
    par = pd.DataFrame({"parameterId": ["k"], "parameterScale": ["log10"], "lowerBound": [1e-3],
                        "upperBound": [1e3], "nominalValue": [0.1], "estimate": [1]}
                       ).set_index("parameterId")
    return petab.Problem(model=SbmlModel.from_antimony(_CONDCOL_ANTIMONY), condition_df=cond,
                         observable_df=obs, measurement_df=meas, parameter_df=par)


def test_condition_column_symbols_are_the_input_columns():
    prob = _condcol_problem()
    assert R.condition_column_symbols(prob) == {"dose"}  # not conditionId/conditionName


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
    assert a["declared_t0_used"] is True          # declared t=0 survives into the t=0 state
    assert rep["B"]["declared_t0"] == 0.0 and rep["B"]["declared_t0_used"] is True


def test_initial_conditions_report_flags_estimated_ic(session):
    """An ESTIMATED IC is reported as not governing t=0 (the fit sets it), not fixed."""
    from petab.v1.C import ESTIMATE

    session.problem.parameter_df.loc["A_init", ESTIMATE] = 1
    a = {r["species"]: r for r in R.initial_conditions_report(session.problem)}["A"]
    assert a["estimated"] is True
    assert a["t0_source"] == "estimated"
    assert a["declared_t0_used"] is False


def test_initial_conditions_report_marks_newly_added(session):
    """`added` (the reconcile diff) flags which ICs this op introduced."""
    rep = {r["species"]: r for r in
           R.initial_conditions_report(session.problem, added={"A_init"})}
    assert rep["A"]["newly_added"] is True
    assert rep["B"]["newly_added"] is False


def test_initial_conditions_report_condition_column_override():
    """A species set per-condition by a condition-table column: its Antimony t=0 is overridden."""
    prob = _condcol_problem()
    R.ensure_parametric_initial_conditions(prob.model.sbml_model)
    # Rename the condition column to the species id so the species itself is condition-set.
    prob.condition_df = prob.condition_df.rename(columns={"dose": "X"})
    x = {r["species"]: r for r in R.initial_conditions_report(prob)}["X"]
    assert x["set_by_condition"] is True
    assert x["t0_source"] == "condition_table"
    assert x["declared_t0_used"] is False


def test_condition_column_parameter_not_synced_into_parameter_table():
    """A parameter that is a condition-table column is a per-condition PEtab override, so reconcile
    must NOT write it into the parameter table (PEtab forbids it there) while still syncing a real
    kinetic — so a benchmark condition table with parametric input columns ships verbatim."""
    import petab.v1 as petab

    prob = _condcol_problem()
    diff = R.sync_parameter_table(prob)
    assert "dose" not in prob.parameter_df.index      # condition-column param stays out of the table
    assert "dose" not in diff["added"]
    assert "k" in prob.parameter_df.index             # a genuine kinetic is still synced
    assert not petab.lint_problem(prob)               # petab accepts the verbatim condition column


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
    assert "B" in edges["A"]                    # A drives its product B
    assert "D" in edges["B"]                    # B drives its sink D
    assert edges.get("D", set()) == set()       # a pure sink drives nothing downstream


def test_input_species_flags_const_boundary_and_rule_driven():
    inputs = R.input_species(_sm(INPUT_ANTIMONY))
    assert "I" in inputs                        # const species -> a held input
    assert "F" in inputs                        # assignment-rule-driven -> a forcing term
    assert "X" not in inputs                    # a free dynamic state, not an input
