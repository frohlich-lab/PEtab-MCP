"""What the statements route holds of an experiment's inputs, and what it leaves a state."""

from __future__ import annotations

import pytest
from indra.statements import Agent, Complex, DecreaseAmount, IncreaseAmount, ModCondition
from petab.v1.models.sbml_model import SbmlModel

from petab_mcp import operations as O
from petab_mcp import problem_sbml as PS
from petab_mcp.errors import ModelUpdateError
from petab_mcp.mechanisms import INDRA_DEFAULT
from petab_mcp.statements import PopulationTransition


def test_a_held_input_that_binds_is_its_free_form():
    """A drug held as an input binds its target. Held as the free form under an anonymous id,
    with `Fsk` the sum over it and the complex, the condition target set nothing."""
    antimony, _ = PS.to_antimony(
        [(Complex([Agent("forskolin"), Agent("ADCY")]), INDRA_DEFAULT, "r1")],
        {"Fsk": Agent("forskolin")},
        targets=frozenset({"Fsk"}),
    )
    sbml = SbmlModel.from_antimony(antimony).sbml_model
    assert sbml.getSpecies("Fsk").getName() == "forskolin(adcy=None)"
    assert sbml.getSpecies("Fsk").getBoundaryCondition()
    assert sbml.getAssignmentRuleByVariable("Fsk") is None


def test_a_condition_target_the_rows_convert_is_a_state_not_an_input():
    """Conditions that set the starting amounts of species the rows convert into each other:
    held on the boundary for being condition targets, they never changed."""
    antimony, _ = PS.to_antimony(
        [(PopulationTransition(Agent("naive"), Agent("effector")), "mass_action", "r1")],
        {"naive": Agent("naive"), "effector": Agent("effector")},
        targets=frozenset({"naive"}),
    )
    sbml = SbmlModel.from_antimony(antimony).sbml_model
    assert not sbml.getSpecies("naive").getBoundaryCondition()


@pytest.mark.parametrize("target", ["naive", "effector"])
def test_a_renamed_condition_target_the_rows_convert_is_a_state_too(target):
    """INDRA names the monomer of `effector-T` `effector_T`. Compared under the agent's own name
    with what the rows make and convert, a target they do was held as well."""
    antimony, _ = PS.to_antimony(
        [(PopulationTransition(Agent("naive-T"), Agent("effector-T")), "mass_action", "r1")],
        {"naive": Agent("naive-T"), "effector": Agent("effector-T")},
        targets=frozenset({target}),
    )
    sbml = SbmlModel.from_antimony(antimony).sbml_model
    assert not sbml.getSpecies(target).getBoundaryCondition()


def test_a_started_state_that_binds_is_its_free_form_too():
    """A ligand a row also degrades is a state the conditions only start; named for the sum over
    it and its complex, its condition target still set nothing."""
    rows = [
        (Complex([Agent("EGF"), Agent("EGFR")]), INDRA_DEFAULT, "r1"),
        (DecreaseAmount(None, Agent("EGF")), INDRA_DEFAULT, "deg"),
    ]
    antimony, _ = PS.to_antimony(rows, {"EGF": Agent("EGF")}, targets=frozenset({"EGF"}))
    sbml = SbmlModel.from_antimony(antimony).sbml_model
    assert sbml.getSpecies("EGF").getName() == "EGF(egfr=None)"
    assert not sbml.getSpecies("EGF").getBoundaryCondition()
    assert sbml.getAssignmentRuleByVariable("EGF") is None


@pytest.mark.parametrize(
    ("degraded", "says"),
    [(False, "Held at what the conditions set"), (True, "Started by the conditions")],
)
def test_the_notes_say_which_condition_targets_are_held(degraded, says):
    """A drug the rows degrade is a state the conditions only start, and nothing but the result
    tells an agent that its degradation row did that."""
    rows = [(Complex([Agent("forskolin"), Agent("ADCY")]), INDRA_DEFAULT, "r1")]
    if degraded:
        rows.append((DecreaseAmount(None, Agent("forskolin")), INDRA_DEFAULT, "deg"))
    notes: list[str] = []
    PS.to_antimony(rows, {"Fsk": Agent("forskolin")}, targets=frozenset({"Fsk"}), notes=notes)
    (note,) = [n for n in notes if "conditions" in n]
    assert note.startswith(says) and note.endswith("Fsk.")


def test_a_started_state_the_rows_make_is_named_too():
    """A species the conditions set and the rows make from another has no start of its own to
    name; looking it up there crashed the build."""
    antimony, _ = PS.to_antimony(
        [(PopulationTransition(Agent("naive"), Agent("effector")), "mass_action", "r1")],
        {"naive": Agent("naive"), "effector": Agent("effector")},
        targets=frozenset({"effector"}),
    )
    sbml = SbmlModel.from_antimony(antimony).sbml_model
    assert sbml.getSpecies("effector").getName() == "effector()"
    assert not sbml.getSpecies("effector").getBoundaryCondition()


@pytest.mark.parametrize("degraded", [False, True])
def test_a_seed_at_a_condition_target_makes_its_agent_that_input(degraded):
    """A seed `forskolin = Fsk` was an initial assignment, which never saw a dose set after
    pre-equilibration (PEtab-MCP#192). At the id alone it is the input, as an entity-map entry
    is: the species takes the id, empty until a period sets it, held while the rows only read it."""
    rows = [(Complex([Agent("forskolin"), Agent("ADCY")]), INDRA_DEFAULT, "r1")]
    if degraded:
        rows.append((DecreaseAmount(None, Agent("forskolin")), INDRA_DEFAULT, "deg"))
    antimony, _ = PS.to_antimony(
        rows, {}, seeds=[("forskolin", Agent("forskolin"), "Fsk")], targets=frozenset({"Fsk"})
    )
    sbml = SbmlModel.from_antimony(antimony).sbml_model
    assert sbml.getSpecies("Fsk").getName() == "forskolin(adcy=None)"
    assert sbml.getSpecies("Fsk").getBoundaryCondition() is not degraded
    assert sbml.getSpecies("Fsk").getInitialAmount() == 0.0
    assert sbml.getInitialAssignmentBySymbol("Fsk") is None
    assert sbml.getParameter("forskolin_init") is None


def test_a_condition_target_names_its_species_before_a_readout_does():
    """With a readout of the same species in the entity map ahead of it, the target became a rule
    over that readout, `dose := D_obs`, which no period can set (review of PEtab-MCP#199)."""
    antimony, _ = PS.to_antimony(
        [(IncreaseAmount(Agent("D"), Agent("B")), INDRA_DEFAULT, "r1")],
        {"D_obs": Agent("D"), "B": Agent("B")},
        seeds=[("D", Agent("D"), "dose")],
        targets=frozenset({"dose"}),
    )
    sbml = SbmlModel.from_antimony(antimony).sbml_model
    assert sbml.getSpecies("dose").getBoundaryCondition()
    assert sbml.getAssignmentRuleByVariable("dose") is None
    assert sbml.getAssignmentRuleByVariable("D_obs").getMath().getName() == "dose"


#: Another state of the drug the input stands for, which a seed may start.
_PHOSPHO = Agent("forskolin", mods=[ModCondition("phosphorylation", "S", "10")])


def test_a_started_input_is_the_state_its_own_seed_writes():
    """Another seeded state of the input's agent, seeded after it, took the target's id: the
    periods dosed the phosphorylated drug instead of the free one (review of PEtab-MCP#199)."""
    antimony, _ = PS.to_antimony(
        [
            (Complex([Agent("forskolin"), Agent("ADCY")]), INDRA_DEFAULT, "r1"),
            (DecreaseAmount(None, Agent("forskolin")), INDRA_DEFAULT, "deg"),
        ],
        {},
        seeds=[("forskolin", Agent("forskolin"), "Fsk"), ("p", _PHOSPHO, 5)],
        targets=frozenset({"Fsk"}),
    )
    sbml = SbmlModel.from_antimony(antimony).sbml_model
    assert sbml.getSpecies("Fsk").getName() == "forskolin(adcy=None, S10='u')"


def test_a_held_input_takes_no_other_seed():
    """A held input is the one state the periods set, as an entity-map input is."""
    with pytest.raises(ModelUpdateError, match="seed 'p': forskolin is an input"):
        PS.to_antimony(
            [(Complex([Agent("forskolin"), Agent("ADCY")]), INDRA_DEFAULT, "r1")],
            {},
            seeds=[("forskolin", Agent("forskolin"), "Fsk"), ("p", _PHOSPHO, 5)],
            targets=frozenset({"Fsk"}),
        )


def test_a_seed_at_an_input_another_agent_is_is_refused():
    with pytest.raises(ModelUpdateError, match="which is already forskolin"):
        PS.to_antimony(
            [(Complex([Agent("FSK"), Agent("ADCY")]), INDRA_DEFAULT, "r1")],
            {"Fsk": Agent("forskolin")},
            seeds=[("FSK", Agent("FSK"), "Fsk")],
            targets=frozenset({"Fsk"}),
        )


_DEGRADED = (DecreaseAmount(None, Agent("B")), INDRA_DEFAULT, "deg")
_MADE_BY_D = (IncreaseAmount(Agent("D"), Agent("B")), INDRA_DEFAULT, "r1")


@pytest.mark.parametrize(
    ("rows", "seeds", "dosed"),
    [
        ([_MADE_BY_D, _DEGRADED], [("D", Agent("D"), "2*dose")], False),
        (
            [(IncreaseAmount(None, Agent("B")), INDRA_DEFAULT, "r1", {"kf": "k*dose"}), _DEGRADED],
            [],
            True,
        ),
        ([_MADE_BY_D, _DEGRADED], [("D", Agent("D"), "dose")], True),
    ],
    ids=["seeded", "bound", "input"],
)
def test_after_pre_equilibration_only_a_rate_or_an_input_sees_the_dose(
    dosed_session, amici_build_root, rows, seeds, dosed
):
    """isensee pre-equilibrates with every drug at 0 and doses at t = 0. libpetab's v2 import sets
    the first period's targets as initial assignments and later ones as events, so a seed reading
    the dose saw 0 throughout, while a rate reading it or a species of its id follows the event
    (PEtab-MCP#192)."""
    from petab_mcp.engine import AmiciEngine

    antimony, _ = PS.to_antimony(rows, {"B": Agent("B")}, seeds=seeds, targets=frozenset({"dose"}))
    O.set_model(dosed_session, antimony, justification="B made at a dosed rate")
    engine = AmiciEngine(dosed_session, build_root=str(amici_build_root))
    simulated = [r["simulation"] for r in engine.evaluate_objective()["simulations"]]
    assert all(s == 0 for s in simulated) is not dosed
