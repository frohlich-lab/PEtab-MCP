"""Reconcile helpers: exponent (Hill-coefficient) detection + tight linear bounds."""

from __future__ import annotations

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
