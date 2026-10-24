"""The reduction's regression corpus: every model the #115/#116 reviews built against it, each built
with and without the reductions through :func:`problem_sbml.to_antimony` and integrated at random
parameter values (:func:`test_reduction._assert_same`). The counterexamples the reviews found must
agree now; the shapes they found agreeing must stay so. This, not a check at run time, is where
the reductions are held exact.

A case is rows in the notation, the entity map, and :func:`problem_sbml.to_antimony`'s keywords."""

from __future__ import annotations

import pytest
from indra.statements import Agent, Gap, Gef, stmts_from_json
from test_reduction import _assert_same

from petab_mcp import mechanisms as M
from petab_mcp import notation
from petab_mcp.mechanisms import INDRA_DEFAULT
from petab_mcp.protocols import Input
from petab_mcp.statements import indra_part, regulators, with_defaults


def _rows(lines: list[str]) -> list[tuple]:
    parsed = [notation.parse(line, {}, dict(M.REGISTRY)) for line in lines]
    for row in parsed:
        row["statement"] = with_defaults(row["statement"])
    stmts = stmts_from_json([indra_part(row["statement"]) for row in parsed])
    return [
        (stmt, row["policy"], row["prefix"], row.get("parameters"), regulators(row["statement"]))
        for stmt, row in zip(stmts, parsed)
    ]


def _entities(agents: dict[str, str]) -> dict[str, Agent]:
    return {k: Agent._from_json(notation.parse_agent(v)) for k, v in agents.items()}


_SITES5 = ("Y1", "S2", "T3", "S4", "Y5")

#: The counterexamples: each changed the ODEs, or crashed, before its fix.
WRONG_BEFORE = {
    # make_changes crashed: nothing fires from the first starts, so the expansion held no species
    # for them, and the made monomers are seeded only after.
    "cycle-of-two-enzymes": (
        ["conv: A -> B by E | mass_action", "back: B -> A by F | mass_action"],
        {"A": "A", "B": "B"},
    ),
    # BioNetGen pairs the two E's with the two products' E in order, so both are unchanged and it
    # halves the rate by symmetry; read by place, one E was taken for the catalyst, at full rate.
    "second-identical-catalyst": (["conv: E + A -> B + E by E | mass_action"], {"B": "B"}),
    # BioNetGen keeps the FIRST C8, turned on, with its FADD; read by place, the second (the one
    # written on) looked unchanged, and the rewrite carried its partner over instead.
    "consumed-molecule-written-first": (
        [
            "cl: C8{catalytic:off} + C8{catalytic:on} -> p18 + C8{catalytic:on} | mass_action",
            "disc: C8 binds FADD | indra_default",
            "act: K activates C8{!FADD} as catalytic | indra_default",
        ],
        {"p18": "p18", "aC8": "C8{catalytic:on}", "C8FADD": "C8{!FADD}"},
    ),
    "consumed-first-partner-coupled": (
        [
            "conv: E{pY1} + E{-pY1} -> A + E{-pY1} | mass_action",
            "y1: K phosphorylates E{~F} at Y1 | indra_default",
            "ef: E binds F | indra_default",
        ],
        {"A": "A", "EF": "E{!F}", "F": "F", "pY1": "E{pY1}"},
    ),
    "consumed-first-site-coupled": (
        [
            "conv: E{pY1} + E{-pY1} -> A + E{-pY1} | mass_action",
            "y1: K phosphorylates E at Y1 | indra_default",
            "s2: K phosphorylates E{-pY1} at S2 | indra_default",
        ],
        {"A": "A", "pS2": "E{pS2}", "pY1": "E{pY1}"},
    ),
}

#: Shapes the reviews found agreeing, kept so.
AGREEING = {
    "dimerising-kinase": (
        [
            "dim: E binds E | indra_default",
            "p: E{!E} phosphorylates S at S2 | indra_default",
            "dp: P dephosphorylates S at S2 | indra_default",
        ],
        {"pS": "S{pS2}"},
    ),
    "enzyme-and-substrate-in-one-complex": (
        [
            "es: E binds S | indra_default",
            "p: E phosphorylates S at S2 | indra_default",
            "dp: P dephosphorylates S at S2 | indra_default",
        ],
        {"pS": "S{pS2}"},
    ),
    "self-degradation": (
        ["deg: E decreases E | indra_default", "y1: K phosphorylates E at Y1 | indra_default"],
        {"E": "E", "pE": "E{pY1}"},
    ),
    "self-conversion": (
        ["conv: E -> X by E | mass_action", "y1: K phosphorylates E at Y1 | indra_default"],
        {"E": "E", "X": "X"},
    ),
    "bound-and-modified-catalysts": (
        [
            "ef: E binds F | indra_default",
            "y1: K phosphorylates E at Y1 | indra_default",
            "p: E{!F, pY1} phosphorylates S at S2 | indra_default",
            "dp: P dephosphorylates S at S2 | indra_default",
        ],
        {"pS": "S{pS2}"},
    ),
    "michaelis-menten": (
        [
            "p: K phosphorylates S at S2 | indra_michaelis_menten",
            "dp: P dephosphorylates S at S2 | indra_michaelis_menten",
        ],
        {"pS": "S{pS2}"},
    ),
    "enzyme-made-later": (
        [
            "syn: E is produced | mass_action",
            "p: E phosphorylates S at S2 | indra_default",
            "dp: P dephosphorylates S at S2 | indra_default",
        ],
        {"pS": "S{pS2}"},
    ),
    "five-site-substrate": (
        [
            line
            for s in _SITES5
            for line in (
                f"p{s}: K phosphorylates S at {s} | indra_default",
                f"d{s}: P dephosphorylates S at {s} | indra_default",
            )
        ],
        {"pS_all": "S{" + ", ".join(f"p{s}" for s in _SITES5) + "}", "pS_1": "S{pY1}"},
    ),
    "negated-readouts": (
        [
            "y1: K phosphorylates S at Y1 | indra_default",
            "s2: K phosphorylates S at S2 | indra_default",
            "dy1: P dephosphorylates S at Y1 | indra_default",
            "ds2: P dephosphorylates S at S2 | indra_default",
        ],
        {"peptide": "S{pY1, -pS2}", "none": "S{-pY1, -pS2}"},
    ),
    "three-location-block": (
        [
            "in: S moves from cytoplasm to nucleus | indra_default",
            "out: S moves from nucleus to membrane | indra_default",
            "back: S moves from membrane to cytoplasm | indra_default",
            "p: K phosphorylates S{@nucleus} at S2 | indra_default",
            "dp: P dephosphorylates S at S2 | indra_default",
        ],
        {"pS": "S{pS2}", "nS": "S{@nucleus}"},
    ),
}


@pytest.mark.parametrize("name", [*WRONG_BEFORE, *AGREEING])
def test_the_reduced_model_is_the_unreduced_one(name):
    lines, agents = {**WRONG_BEFORE, **AGREEING}[name]
    _assert_same(_rows(lines), _entities(agents))


def test_gef_and_gap_catalysts():
    """The notation writes no GEF or GAP row: built from INDRA's statements."""
    rows = [
        (Gef(Agent("G"), Agent("R")), INDRA_DEFAULT, "gef"),
        (Gap(Agent("A"), Agent("R")), INDRA_DEFAULT, "gap"),
        *_rows(["y1: K phosphorylates G at Y1 | indra_default"]),
    ]
    _assert_same(rows, _entities({"aR": "R{gtpbound:on}", "R": "R"}))


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param({"inputs": (Input("D", "D_0", ("D_0",)),)}, id="held-to-a-protocol"),
        pytest.param({"targets": frozenset({"D"})}, id="set-by-the-conditions"),
    ],
)
def test_a_catalyst_the_experiment_sets(kwargs):
    """A dose that catalyses: held, as a boundary species or under its protocol."""
    rows = _rows(
        [
            "p: D phosphorylates S at S2 | indra_default",
            "dp: P dephosphorylates S at S2 | indra_default",
        ]
    )
    _assert_same(rows, _entities({"pS": "S{pS2}", "D": "D"}), **kwargs)
