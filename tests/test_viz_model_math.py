"""The Model tab's mathematics (``petab_mcp.viz.model_math``): ODEs, signs and the levels a
model is drawn at, read off a stored model's Antimony."""

from __future__ import annotations

import pandas as pd
import sympy

from petab_mcp.viz import model_math as MM

from conftest import DATA_DIR

#: A statements model as the server stores it: PySB's species patterns and rule names, A binding B
#: before phosphorylating it (copied from a stored version's model.ant).
_TWO_STEP = """
model *statements()
  compartment default;
  substanceOnly species __s0 in default, __s1 in default, __s2 in default, __s3 in default;
  A := __s0 + __s2;
  B := __s1 + __s2 + __s3;
  r0: __s0 + __s1 -> __s2; p_kf*__s0*__s1 - p_kr*__s2;
  r1: __s2 => __s0 + __s3; p_kc*__s2;
  __s0 = A_init; __s1 = B_init; __s2 = 0; __s3 = 0;
  default = 1; A_init = 1; B_init = 1; p_kf = 1; p_kr = 1; p_kc = 1;
  __s0 is "A(b=None)";
  __s1 is "B(phospho='u', a=None)";
  __s2 is "A(b=1) ._br_B(phospho='u', a=1)";
  __s3 is "B(phospho='p', a=None)";
  r0 is "A_phosphorylation_bind_B_phospho + A_dissoc_B";
  r1 is "A_phosphorylation_B_phospho";
end
"""

_ROW = {
    "prefix": "p",
    "statement": {
        "type": "Phosphorylation",
        "enz": {"name": "A"},
        "sub": {"name": "B"},
        "inhibited_by": [{"name": "D"}],
    },
    "emission": {
        "rules": ["A_phosphorylation_bind_B_phospho", "A_phosphorylation_B_phospho", "A_dissoc_B"],
        "derived": {"inhibited_by_0": "modulated_by"},
    },
}

_OBSERVABLES = pd.DataFrame(
    {"observableFormula": ["B"], "noiseFormula": ["sigma_B"]}, index=pd.Index(["obs_B"])
)


def test_the_toys_odes_are_its_mass_action():
    m = MM.from_antimony((DATA_DIR / "model.ant").read_text(), _OBSERVABLES)
    A, B, k1, k2 = sympy.symbols("A B k1 k2")
    assert sympy.simplify(m.odes["A"] - (-k1 * A + k2 * B)) == 0
    assert sympy.simplify(m.odes["B"] - (k1 * A - k2 * B)) == 0
    assert m.observables["obs_B"].reads == ("B",)


def test_an_antimony_models_influences_are_signed_by_its_odes():
    """With no statements to read, an edge's sign is the sign of the ODE's derivative over every
    positive value of the names: the toy's A raises B and B raises A."""
    graph = MM.influence_graph(MM.from_antimony((DATA_DIR / "model.ant").read_text()))
    edges = set(map(tuple, graph.edges[["source", "target", "kind"]].to_numpy()))
    assert edges == {("s:A", "s:B", "activates"), ("s:B", "s:A", "activates")}


def test_a_statements_model_maps_species_to_agents_and_reactions_to_rows():
    """The chain the zoom walks, agent -> species and row -> rules -> reactions, is read off the
    names PySB gave and the rules the row's emission lists; nothing is rebuilt."""
    m = MM.from_antimony(_TWO_STEP, _OBSERVABLES, {"rows": [_ROW], "bindings": {}})
    assert m.species["__s2"].label == "A:B(phospho=u)"
    assert m.species["__s2"].agents == ("A", "B")
    assert {rx.id: rx.rules for rx in m.reactions} == {
        "r0": ("A_phosphorylation_bind_B_phospho", "A_dissoc_B"),
        "r1": ("A_phosphorylation_B_phospho",),
    }
    row = MM.statement_graph(m).nodes.set_index("key").loc["p:p"]
    assert len(row["members"]) == 2
    agents = MM.influence_graph(m).nodes.set_index("key")
    assert set(agents.loc["a:B", "members"]) == {
        f"s:{m.species[s].key}" for s in ("__s1", "__s2", "__s3")
    }
    assert m.observables["obs_B"].reads == ("__s1", "__s2", "__s3")


def test_a_regulator_is_drawn_as_its_rate_records_it():
    """A row stated "inhibited by D" whose rate shows D only modulating it records that
    (emission.derived); the graph draws the recorded kind, as get_model's English does."""
    assert ("D", "B", "modulates") in MM.row_edges(_ROW)
    assert ("A", "B", "modifies") in MM.row_edges(_ROW)


def test_a_coarse_node_sits_where_its_members_are():
    """Zooming in splits a node where it stood: an agent is placed at the centroid of its
    species."""
    m = MM.from_antimony(_TWO_STEP, None, {"rows": [_ROW], "bindings": {}})
    graphs = MM.levels(m)
    assert list(graphs) == ["influence", "statements", "reactions"]
    finest = MM.layout([graphs["reactions"]])
    placed = MM.place(graphs["influence"], finest).set_index("key")
    members = [finest[k] for k in placed.at["a:B", "members"]]
    assert placed.at["a:B", "x"] == sum(p[0] for p in members) / len(members)


def test_a_formula_sympy_cannot_read_leaves_the_rest_of_the_model():
    expr = MM.parse("lambda + 1", {})
    assert isinstance(expr, sympy.Symbol) and "lambda" in expr.name


def test_the_equations_of_a_selection_name_species_by_their_pattern():
    m = MM.from_antimony(_TWO_STEP, None, {"rows": [_ROW], "bindings": {}})
    lines = MM.equations(m, species=["__s3"], values={"p_kc": 2.0})
    assert lines[0].startswith(r"\frac{d\,[\text{B(phospho=p)}]}{dt}")
    assert "2.0" in lines[0] and "p_{kc}" not in lines[0]


def test_a_long_ode_reads_down_the_pane_and_names_print_as_written():
    """Fujita's ODEs ran off the equation pane's edge, and sympy printed `reaction_9_k1` as reaction
    with a subscript 9k1."""
    text = """
model m
  species A = 1, B = 0, C = 0;
  r1: A => B; reaction_9_k1 * A;
  r2: B => A; k2 * B;
  r3: C => B; k3 * C;
end
"""
    lines = MM.equations(MM.from_antimony(text), species=["B"])
    assert len(lines) == 4 and lines[1].startswith(r"&\quad ")
    assert r"\text{reaction\_9\_k1}" in " ".join(lines)
