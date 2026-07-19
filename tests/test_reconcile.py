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


def test_ic_bounds_are_nonnegative_and_bracket_nominal():
    # A concentration: lower bound is 0; upper bound is the default, widened to 10x a large value.
    assert R._ic_bounds(3.6) == (0.0, 100.0)      # small nominal -> default upper (1e2)
    assert R._ic_bounds(50.0) == (0.0, 500.0)     # large nominal -> 10x upper
    assert R._ic_bounds(0.0) == (0.0, 100.0)
    assert R._ic_bounds(None) == (0.0, 100.0)     # NaN/None guard -> treated as 0


def test_sync_parameter_table_uses_linear_for_nonpositive_nominal(session):
    """A rate-law parameter initialised to 0 (or negative) cannot live on a log10 scale;
    reconcile must fall back to a linear scale rather than emit an invalid log10 row with a
    non-positive bound (which the lint would then reject). Mirrors parameters.sync_parameters.
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


# -- manual sync_parameters tool (consolidated onto sync_parameter_table) ------------------

def test_sync_parameters_reports_in_sync(session):
    """The toy problem loads already reconciled: 'report' shows nothing to add/remove."""
    out = R.sync_parameters(session.problem, "report")
    assert out["diff"]["in_sync"] is True
    assert out["diff"]["model_only"] == [] and out["diff"]["table_only"] == []
    assert out["added"] == [] and out["removed"] == []


def test_sync_parameters_report_does_not_mutate(session):
    """'report' is a dry run — it must not change the parameter table."""
    session.problem.parameter_df = session.problem.parameter_df.drop(index="k1")  # create drift
    before = set(session.problem.parameter_df.index)
    out = R.sync_parameters(session.problem, "report")
    assert "k1" in out["diff"]["model_only"]              # a sync WOULD add k1
    assert set(session.problem.parameter_df.index) == before  # but report changed nothing


def test_sync_parameters_add_is_full_reconcile(session):
    """'add' runs the full reconcile: re-adds a missing model param AND removes an orphan row
    (not in the model, not referenced by the fixed tables) — matching the commit pipeline."""
    from petab.v1.C import ESTIMATE, LOWER_BOUND, NOMINAL_VALUE, PARAMETER_SCALE, UPPER_BOUND

    df = session.problem.parameter_df
    df.loc["orphan_z", [PARAMETER_SCALE, LOWER_BOUND, UPPER_BOUND, NOMINAL_VALUE, ESTIMATE]] = \
        ["log10", 1e-3, 1e3, 1.0, 1]                       # a stray orphan row
    session.problem.parameter_df = df.drop(index="k1")     # drop a real model param

    out = R.sync_parameters(session.problem, "add")

    assert "k1" in out["added"]           # missing model param re-added
    assert "orphan_z" in out["removed"]   # orphan removed
    assert out["diff"]["in_sync"] is True
    idx = set(session.problem.parameter_df.index)
    assert "k1" in idx and "orphan_z" not in idx


def test_sync_parameters_rejects_unknown_policy(session):
    from petab_mcp.errors import ParameterError

    with pytest.raises(ParameterError):
        R.sync_parameters(session.problem, "frobnicate")


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
