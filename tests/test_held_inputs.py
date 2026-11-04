"""What the statements route makes of an experiment's inputs: the species each period sets."""

from __future__ import annotations

import pytest
from indra.statements import (
    Agent,
    Complex,
    DecreaseAmount,
    IncreaseAmount,
    ModCondition,
    Phosphorylation,
)
from petab.v1.models.sbml_model import SbmlModel

from petab_mcp import operations as O
from petab_mcp import problem_sbml as PS
from petab_mcp.errors import MechanismError, ModelUpdateError
from petab_mcp.mechanisms import INDRA_DEFAULT
from petab_mcp.statements import PopulationTransition


def test_an_input_that_binds_is_its_free_form():
    """A drug the conditions set binds its target. Named as the free form under an anonymous id,
    with `Fsk` the sum over it and the complex, the condition target set nothing. It starts at what
    the periods set, with no `forskolin_init` of its own."""
    antimony, _ = PS.to_antimony(
        [(Complex([Agent("forskolin"), Agent("ADCY")]), INDRA_DEFAULT, "r1")],
        {"Fsk": Agent("forskolin")},
        targets=frozenset({"Fsk"}),
    )
    model = SbmlModel.from_antimony(antimony)
    sbml = model.sbml_model
    assert sbml.getSpecies("Fsk").getName() == "forskolin(adcy=None)"
    assert sbml.getAssignmentRuleByVariable("Fsk") is None
    assert sbml.getInitialAssignmentBySymbol("Fsk") is None
    assert sbml.getParameter("forskolin_init") is None


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


def test_a_seed_at_a_condition_target_makes_its_agent_that_input():
    """A seed `forskolin = Fsk` was an initial assignment, which never saw a dose set after
    pre-equilibration (PEtab-MCP#192). At the id alone it is the input, as an entity-map entry
    is: the species takes the id, empty until a period sets it."""
    antimony, _ = PS.to_antimony(
        [(Complex([Agent("forskolin"), Agent("ADCY")]), INDRA_DEFAULT, "r1")],
        {},
        seeds=[("forskolin", Agent("forskolin"), "Fsk")],
        targets=frozenset({"Fsk"}),
    )
    model = SbmlModel.from_antimony(antimony)
    sbml = model.sbml_model
    assert sbml.getSpecies("Fsk").getName() == "forskolin(adcy=None)"
    assert sbml.getSpecies("Fsk").getInitialAmount() == 0.0
    assert sbml.getInitialAssignmentBySymbol("Fsk") is None
    assert sbml.getParameter("forskolin_init") is None


@pytest.mark.parametrize(
    ("row", "agent", "target"),
    [
        (
            (Phosphorylation(Agent("TGFBR1"), Agent("SMAD2"), "S", "465"), "processive", "S2_phos"),
            Agent("SMAD2"),
            "init_S2",
        ),
        (
            (Complex([Agent("TGFB1"), Agent("TGFBR1")]), "assembly", "rec_act"),
            Agent("TGFB1"),
            "init_TGFb",
        ),
        (
            (Complex([Agent("forskolin"), Agent("ADCY")]), INDRA_DEFAULT, "r1"),
            Agent("forskolin"),
            "Fsk",
        ),
        (
            (Phosphorylation(Agent("A"), Agent("B"), "S", "10"), INDRA_DEFAULT, "r1"),
            Agent("A"),
            "dose",
        ),
    ],
    ids=["phosphorylated", "bound", "bound-reversibly", "catalyst"],
)
def test_no_condition_target_is_held(row, agent, target):
    """lucarelli seeds its SMADs and TGF-beta at condition targets, and its rows phosphorylate and
    bind them. Held on the boundary while no row made or converted them into another monomer,
    they were never depleted: nll 4963.76 where it scored 2379.369 (PEtab-MCP#246). Whatever the
    rows do, the species each period sets is an ordinary one."""
    antimony, _ = PS.to_antimony(
        [row], {}, seeds=[("s", agent, target)], targets=frozenset({target})
    )
    model = SbmlModel.from_antimony(antimony)
    assert not model.sbml_model.getSpecies(target).getBoundaryCondition()


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
    assert sbml.getAssignmentRuleByVariable("dose") is None
    assert sbml.getAssignmentRuleByVariable("D_obs").getMath().getName() == "dose"


#: Another state of the drug the input stands for, which a seed may start.
_PHOSPHO = Agent("forskolin", mods=[ModCondition("phosphorylation", "S", "10")])


def test_a_started_input_is_the_state_its_own_seed_writes():
    """Another seeded state of the input's agent, seeded after it, took the target's id: the
    periods dosed the phosphorylated drug instead of the free one (review of PEtab-MCP#199)."""
    antimony, _ = PS.to_antimony(
        [(Complex([Agent("forskolin"), Agent("ADCY")]), INDRA_DEFAULT, "r1")],
        {},
        seeds=[("forskolin", Agent("forskolin"), "Fsk"), ("p", _PHOSPHO, 5)],
        targets=frozenset({"Fsk"}),
    )
    sbml = SbmlModel.from_antimony(antimony).sbml_model
    assert sbml.getSpecies("Fsk").getName() == "forskolin(adcy=None, S10='u')"


def test_a_seed_at_an_input_another_agent_is_is_refused():
    with pytest.raises(ModelUpdateError, match="which is already forskolin"):
        PS.to_antimony(
            [(Complex([Agent("FSK"), Agent("ADCY")]), INDRA_DEFAULT, "r1")],
            {"Fsk": Agent("forskolin")},
            seeds=[("FSK", Agent("FSK"), "Fsk")],
            targets=frozenset({"Fsk"}),
        )


def test_a_binding_reads_no_input_a_seed_made_a_species():
    """The species is read where its agent is named, and the refusal names the seed that mapped
    it, not the entity map."""
    with pytest.raises(MechanismError, match="the entity map, `amounts` or a seed at its id"):
        PS.to_antimony(
            [
                (Complex([Agent("FSK"), Agent("ADCY")]), INDRA_DEFAULT, "r1"),
                (IncreaseAmount(None, Agent("X")), INDRA_DEFAULT, "r2", {"kf": "k*Fsk"}),
            ],
            {},
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
