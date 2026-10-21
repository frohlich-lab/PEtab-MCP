"""The exact reductions of a statements model (:mod:`petab_mcp.reduction`): the reduced model is
built with and without them, and both are integrated at random parameter values. Every species and
readout must agree to solver tolerance, over the same parameters, and the reduced build must
count the unreduced expansion's reactions exactly."""

from __future__ import annotations

import libsbml
import numpy as np
import pytest
import sympy
from indra.statements import (
    Activation,
    ActivityCondition,
    Agent,
    BoundCondition,
    Complex,
    Conversion,
    DecreaseAmount,
    Dephosphorylation,
    Gef,
    IncreaseAmount,
    ModCondition,
    Phosphorylation,
)
from petab.v1.models.sbml_model import SbmlModel
from scipy.integrate import solve_ivp

from petab_mcp import mechanisms as M
from petab_mcp import problem_sbml as PS
from petab_mcp import reduction
from petab_mcp.errors import ModelUpdateError
from petab_mcp.mechanisms import INDRA_DEFAULT


def _p(name: str, *sites: str, **kwargs) -> Agent:
    """``name`` with each of ``sites`` (``Y1``; ``-Y1`` unmodified) phosphorylated."""
    return Agent(
        name,
        mods=[
            ModCondition("phosphorylation", s.lstrip("-")[0], s.lstrip("-")[1:], not s.startswith("-"))
            for s in sites
        ],
        **kwargs,
    )


def _build(rows, entities, **kwargs) -> tuple[str, dict]:
    network: dict = {}
    antimony, _ = PS.to_antimony(rows, entities, network=network, **kwargs)
    return antimony, network


def _both(monkeypatch, rows, entities, **kwargs):
    """The model ``rows`` build, reduced and not, each with its network sizes."""
    reduced = _build(rows, entities, **kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(reduction, "reduce", lambda model: reduction.Reduction())
        full = _build(rows, entities, **kwargs)
    return reduced, full


def _simulate(antimony: str, values: dict[str, float], times) -> dict[str, np.ndarray]:
    """Every species (by its PySB name) and assignment rule of ``antimony`` over ``times``, the
    constants at ``values``: the SBML's own ODEs, integrated with scipy, so no AMICI build."""
    model = SbmlModel.from_antimony(antimony).sbml_model
    symbols = {e.getId(): sympy.Symbol(e.getId()) for e in model.getListOfAllElements() if e.isSetId()}
    symbols["time"] = t = sympy.Symbol("time")

    def read(math) -> sympy.Expr:
        text = libsbml.formulaToL3String(math).replace("^", "**")
        return sympy.sympify(text, locals=symbols)

    constants = {p.getId(): p.getValue() for p in model.getListOfParameters() if p.getConstant()}
    constants |= {c.getId(): c.getSize() for c in model.getListOfCompartments()}
    constants |= values
    rules = {r.getVariable(): read(r.getMath()) for r in model.getListOfRules()}

    def resolve(expr: sympy.Expr) -> sympy.Expr:
        while (new := expr.xreplace({symbols[v]: e for v, e in rules.items()})) != expr:
            expr = new
        return expr.xreplace({symbols[k]: v for k, v in constants.items()})

    species = list(model.getListOfSpecies())
    states = [s.getId() for s in species if s.getId() not in rules]
    start = {s.getId(): s.getInitialAmount() for s in species}
    for assignment in model.getListOfInitialAssignments():
        start[assignment.getSymbol()] = float(resolve(read(assignment.getMath())))
    rhs = {s: sympy.Integer(0) for s in states}
    fixed = {s.getId() for s in species if s.getBoundaryCondition()}
    for rx in model.getListOfReactions():
        rate = resolve(read(rx.getKineticLaw().getMath()))
        for sign, refs in ((-1, rx.getListOfReactants()), (1, rx.getListOfProducts())):
            for ref in refs:
                if ref.getSpecies() not in fixed:
                    rhs[ref.getSpecies()] += sign * ref.getStoichiometry() * rate
    x = [symbols[s] for s in states]
    f = sympy.lambdify([t, x], [rhs[s] for s in states], "numpy")
    solution = solve_ivp(
        lambda time, y: f(time, y),
        (0, times[-1]),
        [start[s] for s in states],
        t_eval=times,
        method="LSODA",
        rtol=1e-10,
        atol=1e-12,
    )
    assert solution.success, solution.message
    out = {}
    named = {s.getId(): f"species {s.getName()}" for s in species}
    for name, expr in [(s, symbols[s]) for s in states] + [(v, symbols[v]) for v in rules]:
        g = sympy.lambdify([t, x], resolve(expr), "numpy")
        out[named.get(name, name)] = np.broadcast_to(
            np.asarray(g(times, solution.y), dtype=float), times.shape
        )
    return out


def _assert_same(monkeypatch, rows, entities, draws: int = 3, **kwargs) -> dict:
    """Build ``rows`` reduced and not, and integrate both at ``draws`` random parameter vectors:
    every species and every rule the unreduced model holds must agree. Returns the reduced
    build's network sizes, checked against the unreduced expansion."""
    (reduced, network), (full, full_network) = _both(monkeypatch, rows, entities, **kwargs)

    def constants(antimony):
        model = SbmlModel.from_antimony(antimony).sbml_model
        ruled = {r.getVariable() for r in model.getListOfRules()}
        return {p.getId() for p in model.getListOfParameters() if p.getConstant()} - ruled

    # k: the reduction adds no parameter and removes none.
    assert constants(reduced) == constants(full)
    assert network["before_reduction"] == full_network["after_reduction"]
    rng = np.random.default_rng(0)
    times = np.linspace(0, 5, 11)
    for _ in range(draws):
        values = {p: 10 ** rng.uniform(-1, 1) for p in sorted(constants(full))}
        a, b = _simulate(reduced, values, times), _simulate(full, values, times)
        shared = {k for k in b if not k.startswith("__obs")}
        assert shared <= set(a)
        for key in sorted(shared):
            np.testing.assert_allclose(a[key], b[key], rtol=1e-6, atol=1e-9, err_msg=key)
    return network


def test_a_kinase_context_becomes_a_rate_factor(monkeypatch):
    """A multisite kinase acting on itself, as an agent writes it (perica_egf's EGFR): every
    unmentioned form of the kinase was a reaction of its own, and the reduction writes one
    reaction per substrate form at ``k·Obs(kinase)`` instead."""
    rows = [
        (Phosphorylation(Agent("E"), Agent("E"), "Y", "1"), INDRA_DEFAULT, "y1"),
        (Phosphorylation(_p("E", "Y1"), Agent("E"), "S", "2"), INDRA_DEFAULT, "s2"),
        (Phosphorylation(_p("E", "Y1"), Agent("E"), "S", "3"), INDRA_DEFAULT, "s3"),
        (Phosphorylation(_p("E", "S2"), Agent("K"), "T", "4"), INDRA_DEFAULT, "t4"),
        (Dephosphorylation(Agent("P"), Agent("E"), "Y", "1"), INDRA_DEFAULT, "dy1"),
        (Dephosphorylation(Agent("P"), Agent("E"), "S", "2"), INDRA_DEFAULT, "ds2"),
        (Dephosphorylation(Agent("P"), Agent("E"), "S", "3"), INDRA_DEFAULT, "ds3"),
        (Dephosphorylation(Agent("P"), Agent("K"), "T", "4"), INDRA_DEFAULT, "dt4"),
    ]
    entities = {"pE": _p("E", "Y1", "S3"), "pK": _p("K", "T4")}
    network = _assert_same(monkeypatch, rows, entities)
    assert network["after_reduction"]["species"] == network["before_reduction"]["species"]
    assert network["after_reduction"]["reactions"] < network["before_reduction"]["reactions"] / 4


@pytest.mark.parametrize(
    "rows, entities",
    [
        pytest.param(
            [
                (Activation(Agent("E", activity=ActivityCondition("kinase", True)), Agent("S")),
                 INDRA_DEFAULT, "act"),
                (Phosphorylation(Agent("K"), Agent("E"), "Y", "1"), INDRA_DEFAULT, "y1"),
                (Activation(_p("E", "Y1"), Agent("E"), "kinase"), INDRA_DEFAULT, "self"),
            ],
            {"aS": Agent("S", activity=ActivityCondition("activity", True))},
            id="activation",
        ),
        pytest.param(
            [
                (Gef(_p("G", "Y1"), Agent("R")), INDRA_DEFAULT, "gef"),
                (Phosphorylation(Agent("K"), Agent("G"), "Y", "1"), INDRA_DEFAULT, "y1"),
            ],
            {"R": Agent("R")},
            id="gef",
        ),
        pytest.param(
            [
                (IncreaseAmount(_p("E", "Y1"), Agent("X")), INDRA_DEFAULT, "syn"),
                (IncreaseAmount(_p("E", "S2"), Agent("Y")), "mass_action", "syn2"),
                (DecreaseAmount(_p("E", "Y1"), Agent("X")), INDRA_DEFAULT, "deg"),
                (DecreaseAmount(_p("E", "S2"), Agent("Y")), "release", "deg2"),
                (Phosphorylation(Agent("K"), Agent("E"), "Y", "1"), INDRA_DEFAULT, "y1"),
                (Phosphorylation(Agent("K"), Agent("E"), "S", "2"), INDRA_DEFAULT, "s2"),
            ],
            {"X": Agent("X"), "Y": Agent("Y")},
            id="synthesis-and-degradation",
        ),
        pytest.param(
            [
                (Conversion(_p("E", "Y1"), [Agent("A")], [Agent("B")]), "mass_action", "conv"),
                (Phosphorylation(Agent("K"), Agent("E"), "Y", "1"), INDRA_DEFAULT, "y1"),
                (Phosphorylation(Agent("K"), Agent("E"), "S", "2"), INDRA_DEFAULT, "s2"),
            ],
            {"B": Agent("B")},
            id="conversion",
        ),
        pytest.param(
            [
                (Phosphorylation(_p("E", "Y1"), Agent("S"), "S", "2"), "indra_michaelis_menten", "mm"),
                (Phosphorylation(Agent("K"), Agent("E"), "Y", "1"), INDRA_DEFAULT, "y1"),
                (Phosphorylation(Agent("K"), Agent("E"), "S", "3"), INDRA_DEFAULT, "s3"),
                (Dephosphorylation(Agent("P"), Agent("S"), "S", "2"), INDRA_DEFAULT, "ds2"),
            ],
            {"pS": _p("S", "S2")},
            id="michaelis-menten",
        ),
        pytest.param(
            [
                (Complex([Agent("E"), Agent("F")]), INDRA_DEFAULT, "ef"),
                (Complex([Agent("S"), Agent("T")]), INDRA_DEFAULT, "st"),
                (Phosphorylation(Agent("E", bound_conditions=[BoundCondition(Agent("F"))]),
                                 Agent("S"), "S", "2"), INDRA_DEFAULT, "s2"),
                (Dephosphorylation(Agent("P"), Agent("S"), "S", "2"), INDRA_DEFAULT, "ds2"),
            ],
            {"pS": _p("S", "S2")},
            id="bound-kinase-and-substrate",
        ),
        pytest.param(
            # A homodimer holds two molecules matching the kinase: BioNetGen counts the species
            # once, which an amount counted in molecules would double.
            [
                (Complex([Agent("E"), Agent("E")]), INDRA_DEFAULT, "ee"),
                (Phosphorylation(Agent("E", bound_conditions=[BoundCondition(Agent("E"))]),
                                 Agent("E"), "Y", "1"), INDRA_DEFAULT, "y1"),
                (Phosphorylation(Agent("E"), Agent("S"), "S", "2"), INDRA_DEFAULT, "s2"),
                (Dephosphorylation(Agent("P"), Agent("E"), "Y", "1"), INDRA_DEFAULT, "dy1"),
            ],
            {"pE": _p("E", "Y1"), "pS": _p("S", "S2")},
            id="dimer-kinase",
        ),
        pytest.param(
            # The kinase written in the very state of the substrate: the two patterns are one.
            [
                (Phosphorylation(_p("E", "-Y1"), Agent("E"), "Y", "1"), INDRA_DEFAULT, "y1"),
                (Phosphorylation(Agent("K"), Agent("E"), "S", "2"), INDRA_DEFAULT, "s2"),
            ],
            {"pE": _p("E", "Y1")},
            id="kinase-in-the-substrates-state",
        ),
    ],
)
def test_every_one_step_shape_reduces_exactly(monkeypatch, rows, entities):
    """Each shape of rule that leaves a complex unchanged: an activation, a GEF, a catalysed
    synthesis and degradation (the last deleting only its molecule), a catalysed conversion, a
    Michaelis-Menten rate that reads its substrate, a kinase bound to a partner acting on a
    substrate that may be bound too."""
    network = _assert_same(monkeypatch, rows, entities)
    antimony, _ = _build(rows, entities)
    assert "_catalysed" in antimony
    assert network["after_reduction"]["reactions"] <= network["before_reduction"]["reactions"]


def test_a_written_rate_keeps_its_modifiers_and_its_catalyst(monkeypatch):
    """A rate the agent wrote, reading a modifier, is multiplied by the catalyst's amount."""
    library = dict(M.REGISTRY)
    library[("Phosphorylation", "inhibited")] = M.written(
        {
            "statement_type": "Phosphorylation",
            "policy": "inhibited",
            "rate": "kc / (1 + inh / Ki)",
            "parameters": ["kc", "Ki"],
            "modifiers": ["inh"],
        },
        library,
    )
    rows = [
        (Phosphorylation(_p("E", "Y1"), Agent("S"), "S", "2"), "inhibited", "s2", None,
         {"inh": Agent("I")}),
        (Phosphorylation(Agent("K"), Agent("E"), "Y", "1"), INDRA_DEFAULT, "y1"),
        (Phosphorylation(Agent("K"), Agent("E"), "S", "3"), INDRA_DEFAULT, "s3"),
        (Phosphorylation(Agent("K"), Agent("I"), "S", "4"), INDRA_DEFAULT, "i4"),
    ]
    _assert_same(monkeypatch, rows, {"pS": _p("S", "S2")}, library=library)


def test_a_two_step_mechanism_is_left_as_written(monkeypatch):
    """A two-step mechanism binds its substrate: no complex of it is left unchanged."""
    rows = [
        (Phosphorylation(Agent("E"), Agent("S"), "S", "2"), "indra_two_step", "s2"),
        (Dephosphorylation(Agent("P"), Agent("S"), "S", "2"), INDRA_DEFAULT, "ds2"),
    ]
    network = _assert_same(monkeypatch, rows, {"pS": _p("S", "S2")})
    model, (two_step, one_step) = M.assemble(rows)
    assert set(reduction.reduce(model).catalysts) == set(one_step.rules)
    assert network["before_reduction"] == network["after_reduction"]


def test_a_kinase_only_its_own_rule_forms_still_never_fires():
    """Rewritten, `pE phosphorylates E` fires on any E whether or not a pE exists, and would form
    the very kinase it needs. A rewritten rule counts only once its catalyst is reached, so the row
    is refused as before, naming the kinase nothing forms."""
    rows = [
        (Phosphorylation(_p("E", "Y1"), Agent("E"), "Y", "1"), INDRA_DEFAULT, "y1"),
        (Phosphorylation(Agent("K"), Agent("S"), "S", "2"), INDRA_DEFAULT, "s2"),
    ]
    with pytest.raises(ModelUpdateError, match=r"rows \['y1'\] build rules that never fire.*Y1="):
        PS.to_antimony(rows, {"pS": _p("S", "S2")})
