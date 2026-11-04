"""A catalyst a row takes in every state, the constant amount of it a rate reads, and the readouts no
input reaches, named on the edit from the statements and the model's equations, never a
simulation (PEtab-MCP#232)."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from indra.statements import Activation, Agent, Dephosphorylation, IncreaseAmount, Phosphorylation

from petab_mcp import problem_sbml as PS
from petab_mcp.mechanisms import INDRA_DEFAULT
from petab_mcp.references import StubReferenceResolver
from petab_mcp.server import create_server


def _call(mcp, name, args):
    return asyncio.run(mcp.call_tool(name, args)).structured_content


def _server(tmp_path, empty_yaml):
    """The toy's `obs_B` (`B`, rising 0 -> 0.79) as `S{p}`, measured once `dose` goes from 0 in
    pre-equilibration to 1, with no model until the rows build one."""
    work = Path(empty_yaml).parent
    (work / "conditions.tsv").write_text(
        "conditionId\ttargetId\ttargetValue\npre\tdose\t0\ntreat\tdose\t1\n", encoding="utf-8"
    )
    (work / "experiments.tsv").write_text(
        "experimentId\ttime\tconditionId\ne0\t-inf\tpre\ne0\t0\ttreat\n", encoding="utf-8"
    )
    meas = work / "measurements.tsv"
    meas.write_text(meas.read_text().replace("obs_B\t\t", "obs_B\te0\t"), encoding="utf-8")
    yaml = work / "problem.yaml"
    yaml.write_text(
        yaml.read_text()
        + "condition_files:\n- conditions.tsv\nexperiment_files:\n- experiments.tsv\n"
    )
    (tmp_path / "entities.json").write_text(
        '{"B": {"name": "S", "mods": [{"mod_type": "phosphorylation"}]}}'
    )
    return create_server(
        str(yaml),
        provenance_dir=str(tmp_path / "prov"),
        resolver=StubReferenceResolver(),
        entity_map=str(tmp_path / "entities.json"),
    )


def _build(mcp, enzyme):
    return _call(
        mcp,
        "make_changes",
        {
            "justification": "the dose activates E, which phosphorylates S",
            "statements": [
                "r1: K phosphorylates E",
                f"r2: {enzyme} phosphorylates S",
                "r3: S is dephosphorylated",
            ],
            "bindings": {"r1": "default", "r2": "default", "r3": "basal"},
            "seeds": [{"agent": "K", "amount": "dose"}],
            "n_starts": 2,
        },
    )


def test_a_bare_enzyme_and_its_constant_rate_are_named_on_the_edit(tmp_path, empty_yaml):
    """raimundez r2's `MEK phosphorylates MAPK` read MEK{u} + MEK{p}, a conserved total: pMAPK was
    flat in 13/13 experiments across 18 versions, and no reply said why. Through `make_changes`,
    as the agent meets it."""
    mcp = _server(tmp_path, empty_yaml)
    out = _build(mcp, "E")
    assert out["applied"] is True, out.get("error")
    said = " ".join(m["message"] for m in out["messages"])
    assert "Row `r2` names no state of E, its catalyst" in said
    assert "name its state: `E{p} phosphorylates S`" in said
    assert "No reaction changes the amount of E this rate reads" in said
    assert "No input reaches the readouts `B`" in said

    fixed = _build(mcp, "E{p}")
    assert fixed["applied"] is True, fixed.get("error")
    assert not any(
        "catalyst" in m["message"] or "No input" in m["message"] for m in fixed["messages"]
    )


def _notes(rows):
    notes: list[str] = []
    PS.to_antimony(rows, {"B": Agent("S")}, notes=notes)
    return [n for n in notes if "catalyst" in n]


_SUB = (Dephosphorylation(None, Agent("S")), "basal", "r3")


@pytest.mark.parametrize(
    ("rows", "form", "constant", "policy"),
    [
        # Its activity, not a modification, is what another row changes.
        (
            [(Activation(Agent("K"), Agent("E")), INDRA_DEFAULT, "r1")],
            "E{activity:on}",
            True,
            INDRA_DEFAULT,
        ),
        # The row binds E first: its own binding changes the free E it reads.
        (
            [(Phosphorylation(Agent("K"), Agent("E")), INDRA_DEFAULT, "r1")],
            "E{p}",
            False,
            "indra_two_step",
        ),
        # A row makes E: its total moves, so the rate is not constant, only blind to E's state.
        (
            [
                (Phosphorylation(Agent("K"), Agent("E")), INDRA_DEFAULT, "r1"),
                (IncreaseAmount(None, Agent("E")), INDRA_DEFAULT, "r4"),
            ],
            "E{p}",
            False,
            INDRA_DEFAULT,
        ),
    ],
)
def test_the_state_another_row_gives_the_enzyme_is_the_one_offered(rows, form, constant, policy):
    """The form offered is the state another row's statement puts on the enzyme, and a constant
    amount is claimed only where its derivative in the equations is 0; a synthesis, which makes E
    in one form, changes no state."""
    (note,) = _notes([*rows, (Phosphorylation(Agent("E"), Agent("S")), policy, "r2"), _SUB])
    assert f"`{form} phosphorylates S`" in note and "though `r1` changes it." in note
    assert ("No reaction changes the amount of E" in note) is constant


def test_an_enzyme_no_row_changes_says_nothing():
    """A state site only a seed gives (`E{p}` declared, never put by a row) is not taken in every
    state: no row makes it a different molecule over time."""
    p_e = Agent._from_json({"name": "E", "mods": [{"mod_type": "phosphorylation"}]})
    notes: list[str] = []
    PS.to_antimony(
        [(Phosphorylation(Agent("E"), Agent("S")), INDRA_DEFAULT, "r2"), _SUB],
        {"B": Agent("S")},
        seeds=[("E_p", p_e, 1.0), ("S", Agent("S"), 1.0)],
        notes=notes,
    )
    assert not [n for n in notes if "catalyst" in n]


_P_S = Agent._from_json({"name": "S", "mods": [{"mod_type": "phosphorylation"}]})
_P_E = Agent._from_json({"name": "E", "mods": [{"mod_type": "phosphorylation"}]})


@pytest.mark.parametrize(
    ("rows", "seeds", "reached"),
    [
        # The dose turns E over, but the rate reads E's total.
        (
            [
                (Phosphorylation(Agent("K"), Agent("E")), INDRA_DEFAULT, "r1"),
                (Phosphorylation(Agent("E"), Agent("S")), INDRA_DEFAULT, "r2"),
            ],
            [("K", Agent("K"), "dose")],
            False,
        ),
        # ...and the form it puts on E.
        (
            [
                (Phosphorylation(Agent("K"), Agent("E")), INDRA_DEFAULT, "r1"),
                (Phosphorylation(_P_E, Agent("S")), INDRA_DEFAULT, "r2"),
            ],
            [("K", Agent("K"), "dose")],
            True,
        ),
        # A binding that reads the dose.
        ([(Phosphorylation(None, Agent("S")), "basal", "r2", {"k": "k0*dose"})], [], True),
        # A starting amount that reads it: the total E is read, and set per experiment.
        (
            [(Phosphorylation(Agent("E"), Agent("S")), INDRA_DEFAULT, "r2")],
            [("E", Agent("E"), "2*dose"), ("S", Agent("S"), 1.0)],
            True,
        ),
    ],
)
def test_a_readout_no_input_reaches_is_named(rows, seeds, reached):
    """Traced from the readout through what its time derivative reads, never simulated: a form the
    input changes, a binding and a starting amount each carry it."""
    notes: list[str] = []
    PS.to_antimony(
        [*rows, _SUB], {"B": _P_S}, seeds=seeds, targets=frozenset({"dose"}), notes=notes
    )
    assert any("No input reaches the readouts `B`" in n for n in notes) is not reached


def test_a_problem_without_inputs_names_no_readout():
    """Where no experiment sets anything, every readout reads no input, and saying so tells
    nothing: every toy test without conditions would carry the note."""
    notes: list[str] = []
    PS.to_antimony(
        [(Phosphorylation(None, Agent("S")), "basal", "r2"), _SUB], {"B": _P_S}, notes=notes
    )
    assert not [n for n in notes if "No input" in n]
