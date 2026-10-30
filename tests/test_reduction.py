"""The exact reductions of a statements model (:mod:`petab_mcp.reduction`): the reduced model is
built with and without them, and both are integrated at random parameter values. Every species and
readout must agree to solver tolerance, over the same parameters (a species of a split monomer as
the product of its blocks' species), and the reduced build must count the unreduced expansion
exactly."""

from __future__ import annotations

import asyncio
import json
import re

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
from petab_mcp.protocols import Input


def _p(name: str, *sites: str, **kwargs) -> Agent:
    """``name`` with each of ``sites`` (``Y1``; ``-Y1`` unmodified) phosphorylated."""
    return Agent(
        name,
        mods=[
            ModCondition(
                "phosphorylation", s.lstrip("-")[0], s.lstrip("-")[1:], not s.startswith("-")
            )
            for s in sites
        ],
        **kwargs,
    )


def _build(rows, entities, **kwargs) -> tuple[str, dict]:
    network: dict = {}
    antimony, _ = PS.to_antimony(rows, entities, network=network, **kwargs)
    return antimony, network


def _both(rows, entities, **kwargs):
    """The model ``rows`` build, reduced and not, each with its network sizes."""
    return _build(rows, entities, **kwargs), _build(rows, entities, reduce=False, **kwargs)


def _simulate(antimony: str, values: dict[str, float], times) -> dict[str, np.ndarray]:
    """Every species (by its PySB name) and assignment rule of ``antimony`` over ``times``, the
    constants at ``values``: the SBML's own ODEs, integrated with scipy, so no AMICI build."""
    model = SbmlModel.from_antimony(antimony).sbml_model
    symbols = {
        e.getId(): sympy.Symbol(e.getId()) for e in model.getListOfAllElements() if e.isSetId()
    }
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
        value = np.broadcast_to(np.asarray(g(times, solution.y), dtype=float), times.shape)
        out[named.get(name, name)] = value
        if name in named and not name.startswith("__"):
            out[name] = value  # a readout that names its one species
    return out


def _assert_same(rows, entities, draws: int = 3, **kwargs) -> dict:
    """Build ``rows`` reduced and not, and integrate both at ``draws`` random parameter vectors:
    every species and every rule the unreduced model holds must agree. Returns the reduced
    build's network sizes, checked against the unreduced expansion."""
    (reduced, network), (full, full_network) = _both(rows, entities, **kwargs)

    def constants(antimony):
        model = SbmlModel.from_antimony(antimony).sbml_model
        ruled = {r.getVariable() for r in model.getListOfRules()}
        return {p.getId() for p in model.getListOfParameters() if p.getConstant()} - ruled

    # k: the reduction adds no parameter and removes none.
    assert constants(reduced) == constants(full)
    assert network["before_reduction"] == full_network["after_reduction"]
    split = network.get("split", {})
    rng = np.random.default_rng(0)
    times = np.linspace(0, 5, 11)
    for _ in range(draws):
        values = {p: 10 ** rng.uniform(-1, 1) for p in sorted(constants(full))}
        a, b = _simulate(reduced, values, times), _simulate(full, values, times)
        for key in sorted(k for k in b if not k.startswith("__obs")):
            mine = a[key] if key in a else _product(a, split, key)
            np.testing.assert_allclose(mine, b[key], rtol=1e-6, atol=1e-9, err_msg=key)
    return network


def _product(trajectories: dict, split: dict, key: str) -> np.ndarray:
    """The trajectory of ``key``, a species of a split monomer, from the split model's: the first
    block's amount in that form times each other block's fraction in it."""
    monomer, sites = re.fullmatch(r"species (\w+)\((.*)\)", key).groups()
    form = dict(re.findall(r"(\w+)='(\w+)'", sites))
    value = 1.0
    for block, own in split[monomer].items():
        value = (
            value * trajectories[f"species {block}({', '.join(f'{s}={form[s]!r}' for s in own)})"]
        )
    return value


def test_a_kinase_context_becomes_a_rate_factor():
    """A multisite kinase acting on itself, as an agent writes it (perica_egf's EGFR): every
    unmentioned form of the kinase was a reaction of its own, and the reduction writes one
    reaction per substrate form at ``k·Obs(kinase)`` instead. No rule reads two of its sites on
    the substrate, so each site is tracked apart, and the readout of two is their product."""
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
    network = _assert_same(rows, entities)
    assert network["split"] == {"E": {"E_Y1": ["Y1"], "E_S2": ["S2"], "E_S3": ["S3"]}}
    assert network["after_reduction"]["species"] < network["before_reduction"]["species"]
    assert network["after_reduction"]["reactions"] < network["before_reduction"]["reactions"] / 4


@pytest.mark.parametrize(
    "rows, entities",
    [
        pytest.param(
            [
                (
                    Activation(Agent("E", activity=ActivityCondition("kinase", True)), Agent("S")),
                    INDRA_DEFAULT,
                    "act",
                ),
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
                (
                    Phosphorylation(_p("E", "Y1"), Agent("S"), "S", "2"),
                    "indra_michaelis_menten",
                    "mm",
                ),
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
                (
                    Phosphorylation(
                        Agent("E", bound_conditions=[BoundCondition(Agent("F"))]),
                        Agent("S"),
                        "S",
                        "2",
                    ),
                    INDRA_DEFAULT,
                    "s2",
                ),
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
                (
                    Phosphorylation(
                        Agent("E", bound_conditions=[BoundCondition(Agent("E"))]),
                        Agent("E"),
                        "Y",
                        "1",
                    ),
                    INDRA_DEFAULT,
                    "y1",
                ),
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
def test_every_one_step_shape_reduces_exactly(rows, entities):
    """Each shape of rule that leaves a complex unchanged: an activation, a GEF, a catalysed
    synthesis and degradation (the last deleting only its molecule), a catalysed conversion, a
    Michaelis-Menten rate that reads its substrate, a kinase bound to a partner acting on a
    substrate that may be bound too."""
    network = _assert_same(rows, entities)
    antimony, _ = _build(rows, entities)
    assert "_catalysed" in antimony
    assert network["after_reduction"]["reactions"] <= network["before_reduction"]["reactions"]


def test_a_written_rate_keeps_its_regulator_and_its_catalyst():
    """A rate the agent wrote, reading a regulator the statement states, is multiplied by the
    catalyst's amount."""
    library = dict(M.REGISTRY)
    library[("Phosphorylation", "inhibited")] = M.written(
        {
            "statement_type": "Phosphorylation",
            "policy": "inhibited",
            "rate": "kc / (1 + inhibited_by / Ki)",
            "parameters": ["kc", "Ki"],
        },
        library,
    )
    rows = [
        (
            Phosphorylation(_p("E", "Y1"), Agent("S"), "S", "2"),
            "inhibited",
            "s2",
            None,
            {"inhibited_by": [Agent("I")]},
        ),
        (Phosphorylation(Agent("K"), Agent("E"), "Y", "1"), INDRA_DEFAULT, "y1"),
        (Phosphorylation(Agent("K"), Agent("E"), "S", "3"), INDRA_DEFAULT, "s3"),
        (Phosphorylation(Agent("K"), Agent("I"), "S", "4"), INDRA_DEFAULT, "i4"),
    ]
    _assert_same(rows, {"pS": _p("S", "S2")}, library=library)


def test_a_two_step_mechanism_is_left_as_written():
    """A two-step mechanism binds its substrate: no complex of it is left unchanged."""
    rows = [
        (Phosphorylation(Agent("E"), Agent("S"), "S", "2"), "indra_two_step", "s2"),
        (Dephosphorylation(Agent("P"), Agent("S"), "S", "2"), INDRA_DEFAULT, "ds2"),
    ]
    network = _assert_same(rows, {"pS": _p("S", "S2")})
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


# -- independent sites tracked apart ----------------------------------------------------------
def _sites(*sites: str, substrate: str = "S") -> list[tuple]:
    """K puts on, and P takes off, each of ``sites`` of ``substrate``, one step each."""
    return [
        row
        for site in sites
        for row in (
            (
                Phosphorylation(Agent("K"), Agent(substrate), site[0], site[1:]),
                INDRA_DEFAULT,
                f"{substrate}_{site}",
            ),
            (
                Dephosphorylation(Agent("P"), Agent(substrate), site[0], site[1:]),
                INDRA_DEFAULT,
                f"{substrate}_d{site}",
            ),
        )
    ]


def test_independent_sites_read_together_are_split():
    """No rule reads Y1 and S2 together on one molecule, so each is a monomer of its own, started
    in the declared seed's form at its amount; readouts of both forms are products of the two."""
    entities = {"ppS": _p("S", "Y1", "S2"), "pS": _p("S", "-Y1", "S2")}
    network = _assert_same(_sites("Y1", "S2"), entities, seeds=[("S", _p("S", "Y1"), 1.5)])
    assert network["split"] == {"S": {"S_Y1": ["Y1"], "S_S2": ["S2"]}}


def test_a_rule_reading_a_second_site_keeps_both_in_one_block():
    """Y1 comes off faster once S2 is on: the rule changes Y1 while it reads S2 on the same
    molecule, which couples the two. S3 stays apart."""
    rows = _sites("Y1", "S2", "S3") + [
        (Dephosphorylation(Agent("Q"), _p("S", "S2"), "Y", "1"), INDRA_DEFAULT, "coupled")
    ]
    network = _assert_same(rows, {"pS": _p("S", "Y1", "S3")})
    assert network["split"] == {"S": {"S_Y1_S2": ["Y1", "S2"], "S_S3": ["S3"]}}


def test_a_kinase_read_on_two_sites_couples_neither():
    """MAPK1{pT185, pY187} as a kinase reads both sites on another molecule than the one it
    changes, which couples nothing: MAPK1 is split, and the kinase's rate factor is the product of
    its two blocks."""
    rows = _sites("T185", "Y187", substrate="MAPK1") + [
        (Phosphorylation(_p("MAPK1", "T185", "Y187"), Agent("R"), "T", "359"), INDRA_DEFAULT, "r"),
        (Dephosphorylation(Agent("P"), Agent("R"), "T", "359"), INDRA_DEFAULT, "dr"),
    ]
    network = _assert_same(rows, {"pR": _p("R", "T359")})
    assert network["split"] == {"MAPK1": {"MAPK1_T185": ["T185"], "MAPK1_Y187": ["Y187"]}}


def test_a_readout_across_many_blocks_is_charged_no_constant():
    """A peptide read on six sites at once is a product over six blocks. Written as a power of
    the total, its exponent 5 would be a literal `k` charges; each block's fraction writes none."""
    from petab_mcp.literals import hardcoded_constants

    sites = ("Y1", "S2", "T3", "S4", "Y5", "T6")
    entities = {"peptide": _p("S", "Y1", "-S2", "T3", "-S4", "Y5", "-T6")}
    network = _assert_same(_sites(*sites), entities, draws=1)
    assert list(network["split"]["S"].values()) == [[s] for s in sites]
    antimony, _ = _build(_sites(*sites), entities)
    assert hardcoded_constants(SbmlModel.from_antimony(antimony).sbml_model) == []


@pytest.mark.parametrize(
    "extra, kwargs",
    [
        pytest.param([(IncreaseAmount(None, Agent("S")), "mass_action", "syn")], {}, id="made"),
        pytest.param([(DecreaseAmount(None, Agent("S")), INDRA_DEFAULT, "deg")], {}, id="removed"),
        pytest.param([(Complex([Agent("S"), Agent("B")]), INDRA_DEFAULT, "bind")], {}, id="bound"),
        pytest.param(
            [(Phosphorylation(Agent("K"), Agent("S"), "T", "3"), "indra_two_step", "two")],
            {},
            id="two-step",
        ),
        pytest.param(
            [],
            {"seeds": [("S_u", _p("S", "-Y1", "-S2"), 1.0), ("S_p", _p("S", "Y1", "S2"), 2.0)]},
            id="two-seed-forms",
        ),
        pytest.param([], {"targets": frozenset({"S"})}, id="species-the-conditions-set"),
        pytest.param([], {"inputs": (Input("S", "S_0", ("S_0",)),)}, id="input"),
    ],
)
def test_a_monomer_stays_whole(extra, kwargs):
    """Each condition the product needs, broken: a total rules change, a binding site, a step
    with no catalyst to take out, a start that does not factorise, and a form that the experiment
    sets. S stays whole, and the model is still exact. Without the extra row or argument, S is
    split (`test_independent_sites_read_together_are_split`)."""
    entities = {"S": Agent("S"), "ppS": _p("S", "Y1", "S2")}
    network = _assert_same(_sites("Y1", "S2") + extra, entities, **kwargs)
    assert "S" not in network.get("split", {})


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param({"seeds": [("S", Agent("S"), 0)]}, id="seeds-empty"),
        pytest.param({"seeds": [("S", Agent("S"), "0")]}, id="seeds-empty-as-text"),
        pytest.param({"seeds": [("S", Agent("S"), "2")]}, id="seeds-at-a-number-as-text"),
        pytest.param(
            {"seeds": [("S", Agent("S"), "S_total")], "targets": frozenset({"S_total"})},
            id="seed-the-conditions-set",
        ),
    ],
)
def test_a_total_that_may_be_zero_splits_exactly(kwargs):
    """Every block but the first holds fractions, starting at 1 for all, so an amount across
    blocks divides by nothing and a total of 0 is 0, not NaN: an empty start, one the conditions
    set (a knockout), one written as text. A number written as text is that number, one start
    parameter for the first block alone, so the parameters are the unsplit model's."""
    entities = {"S": Agent("S"), "ppS": _p("S", "Y1", "S2")}
    network = _assert_same(_sites("Y1", "S2"), entities, **kwargs)
    assert network["split"] == {"S": {"S_Y1": ["Y1"], "S_S2": ["S2"]}}


def test_a_total_of_zero_reads_zero():
    """A start parameter at exactly 0, an estimate at a lower bound of 0: before, 0/0 in every
    amount across blocks (the #116 review)."""
    entities = {"S": Agent("S"), "ppS": _p("S", "Y1", "S2")}
    rows = _sites("Y1", "S2")
    (reduced, _), (full, _) = _both(rows, entities)
    values = {"S_init": 0.0, "K_init": 1.0, "P_init": 1.0} | {
        f"S_{p}{s}_kf": 1.0 for p in ("", "d") for s in ("Y1", "S2")
    }
    times = np.linspace(0, 5, 6)
    a, b = _simulate(reduced, values, times), _simulate(full, values, times)
    np.testing.assert_array_equal(a["ppS"], np.zeros_like(times))
    np.testing.assert_array_equal(b["ppS"], np.zeros_like(times))


def test_a_knockout_of_a_monomer_named_as_indra_rewrites_it_reads_zero():
    """`S-1`, PySB's `S_1`, started at `S_wt*(1 - ko_S)` with `ko_S` a condition target: looked up
    by the name as written, the knockout slipped past the start checks and split into 0/0 (the
    #116 review). Every block but the first now holds fractions, so a knocked-out total reads 0,
    and the split model is the unsplit one."""
    rows = [
        (Phosphorylation(Agent("K"), Agent("S-1"), "Y", "1"), INDRA_DEFAULT, "y1"),
        (Phosphorylation(Agent("K"), Agent("S-1"), "S", "2"), INDRA_DEFAULT, "s2"),
        (Dephosphorylation(Agent("P"), Agent("S-1"), "Y", "1"), INDRA_DEFAULT, "dy1"),
        (Dephosphorylation(Agent("P"), Agent("S-1"), "S", "2"), INDRA_DEFAULT, "ds2"),
        (Phosphorylation(_p("S-1", "Y1", "S2"), Agent("X"), "T", "3"), INDRA_DEFAULT, "x"),
        (Dephosphorylation(Agent("P"), Agent("X"), "T", "3"), INDRA_DEFAULT, "dx"),
    ]
    entities = {"ppS": _p("S-1", "Y1", "S2"), "pX": _p("X", "T3")}
    kwargs = {"seeds": [("S", Agent("S-1"), "S_wt*(1 - ko_S)")], "targets": frozenset({"ko_S"})}
    (reduced, network), (full, _) = _both(rows, entities, **kwargs)
    assert network["split"]["S_1"]
    times = np.linspace(0, 5, 6)
    for ko in (0.0, 0.5, 1.0):
        values = {"S_wt": 2.0, "ko_S": ko, "K_init": 1.0, "P_init": 1.0, "X_init": 1.0}
        values |= {f"{r}_kf": 1.0 for r in ("y1", "s2", "dy1", "ds2", "x", "dx")}
        a, b = _simulate(reduced, values, times), _simulate(full, values, times)
        for key in ("ppS", "pX"):
            assert np.isfinite(a[key]).all()
            np.testing.assert_allclose(a[key], b[key], rtol=1e-6, atol=1e-12, err_msg=f"{key} {ko}")


def test_a_refusal_reads_the_same_unreduced():
    """`reduce=False`, the tests' oracle, raised a KeyError where the rows are refused, so a refusal
    could not be compared both ways (the #116 review)."""
    rows = [
        (Phosphorylation(_p("E", "Y1"), Agent("E"), "Y", "1"), INDRA_DEFAULT, "y1"),
        (Phosphorylation(Agent("K"), Agent("S"), "S", "2"), INDRA_DEFAULT, "s2"),
    ]
    messages = []
    for reduce in (True, False):
        with pytest.raises(ModelUpdateError) as refused:
            PS.to_antimony(rows, {"pS": _p("S", "S2")}, reduce=reduce)
        messages.append(str(refused.value))
    assert messages[0] == messages[1]


# -- through the server: the problem's tables, and AMICI ---------------------------------------
_SPLITTING = {
    "statements": [
        "y1: K phosphorylates S at Y1",
        "s2: K phosphorylates S at S2",
        "dy1: P dephosphorylates S at Y1",
        "ds2: P dephosphorylates S at S2",
    ],
    "bindings": {p: "indra_default" for p in ("y1", "s2", "dy1", "ds2")},
}


def _statements_server(tmp_path, entities: dict, **kwargs):
    from conftest import YAML_PATH

    from petab_mcp.references import StubReferenceResolver
    from petab_mcp.server import create_server

    (tmp_path / "entities.json").write_text(json.dumps(entities))
    return create_server(
        str(YAML_PATH),
        provenance_dir=str(tmp_path / "prov"),
        resolver=StubReferenceResolver(),
        entity_map=str(tmp_path / "entities.json"),
        **kwargs,
    )


def _call(mcp, name, args):
    return asyncio.run(mcp.call_tool(name, args)).structured_content


_PY1 = {"mod_type": "phosphorylation", "residue": "Y", "position": "1"}
_PS2 = {"mod_type": "phosphorylation", "residue": "S", "position": "2"}


def test_a_shipped_start_of_a_readout_id_starts_no_block(tmp_path):
    """The toy ships `A_init` = 1. `A` = S{pY1} reads one species once S is split, and named after
    the readout that species took `A_init` for its start, where the unsplit model, whose `A` is a
    sum, starts it empty (the #116 review). A block's species keeps its own name."""
    entities = {"A": {"name": "S", "mods": [_PY1]}, "B": {"name": "S", "mods": [_PS2]}}
    mcp = _statements_server(tmp_path, entities, auto_fit=False)
    out = _call(mcp, "make_changes", {"justification": "S is split", **_SPLITTING})
    assert out["applied"] is True and out["network"]["split"], out
    structure = _call(mcp, "get_model_structure", {"initial_conditions": ""})
    assert "A" not in structure["species"]
    assert "A_init" not in json.dumps(structure["initial_conditions"])


def test_a_split_model_compiles_and_fits(tmp_path):
    """No other test compiles a split model with AMICI (the #116 review): a block starting at the
    `start_fraction` rule, a phosphorylation and its removal merged into one reversible reaction,
    and a readout that is a product of two blocks."""
    entities = {"A": {"name": "S", "mods": [_PY1]}, "B": {"name": "S", "mods": [_PY1, _PS2]}}
    mcp = _statements_server(tmp_path, entities, auto_fit=True)
    out = _call(
        mcp,
        "make_changes",
        {"justification": "S is split", **_SPLITTING, "n_starts": 2},
    )
    assert out["applied"] is True and out["committed"] is True, out
    assert out["network"]["split"] == {"S": {"S_Y1": ["Y1"], "S_S2": ["S2"]}}
    assert np.isfinite(out["neg_log_likelihood"])
