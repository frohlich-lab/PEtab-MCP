"""The SBML a set of statements builds for one PEtab problem: readout names, starting amounts,
inputs."""

from __future__ import annotations

import pytest
from indra.statements import Agent, BoundCondition, Phosphorylation
from petab.v1.models.sbml_model import SbmlModel

from petab_mcp import problem_sbml as PS
from petab_mcp.errors import ModelUpdateError
from petab_mcp.mechanisms import INDRA_DEFAULT
from petab_mcp.statements import PopulationTransition

_PHOSPHO = Phosphorylation(Agent("MEK", db_refs={"HGNC": "6840"}), Agent("ERK"), "T", "185")
_TRANSITION = PopulationTransition(Agent("naive"), Agent("effector"))


def test_readouts_name_the_species_the_observables_read():
    """The fixed observable table reads species ids, so the export must define them. A readout
    that is one species names that species, modified forms included: left as a rule over it,
    every species looked disconnected from the data to the edit's reachability warning."""
    p_erk = Agent._from_json(
        {
            "name": "ERK",
            "mods": [{"mod_type": "phosphorylation", "residue": "T", "position": "185"}],
        }
    )
    antimony, _ = PS.to_antimony(
        [(_PHOSPHO, INDRA_DEFAULT, "r1"), (_TRANSITION, "mass_action", "r2")],
        {"naive": Agent("naive"), "pERK": p_erk},
    )
    sbml = SbmlModel.from_antimony(antimony).sbml_model
    assert sbml.getSpecies("naive").getName() == "naive()"
    assert sbml.getSpecies("pERK").getName() == "ERK(T185='p')"
    assert not any(sbml.getAssignmentRuleByVariable(s) for s in ("naive", "pERK"))


def test_an_input_is_held_to_its_protocol():
    """A dose is a fact about the experiment: its species sits under the protocol the entity map
    declares, with no starting amount of its own, and every condition-table column the protocol
    names exists for the table to set — Fujita's warm start was refused for a column `EGF_0` it
    had nowhere to put."""
    egf, akt = Agent("EGF"), Agent("AKT")
    antimony, _ = PS.to_antimony(
        [(Phosphorylation(egf, akt), INDRA_DEFAULT, "r1")],
        {"EGF": egf},
        inputs={"EGF": "piecewise(EGF_0 + EGF_rate*time, time < EGF_end, 0)"},
    )
    sbml = SbmlModel.from_antimony(antimony).sbml_model
    assert sbml.getSpecies("EGF").getBoundaryCondition()
    assert sbml.getAssignmentRuleByVariable("EGF") is not None
    assert sbml.getInitialAssignmentBySymbol("EGF") is None
    assert {"EGF_0", "EGF_rate", "EGF_end"} <= {p.getId() for p in sbml.getListOfParameters()}


def test_what_is_made_starts_empty_and_what_is_modified_starts_present():
    """Crauste's effectors exist only once naive cells convert; a kinase that is only ever
    modified must be there at the start. The modified one's amount is `<monomer>_init`, not
    INDRA's `<monomer>_0`, which skipped EGF outright because Fujita's stimulus column is `EGF_0`."""
    antimony, _ = PS.to_antimony(
        [(_TRANSITION, "mass_action", "r1"), (_PHOSPHO, INDRA_DEFAULT, "r2")],
        {"effector": Agent("effector")},
    )
    sbml = SbmlModel.from_antimony(antimony).sbml_model
    assert sbml.getInitialAssignmentBySymbol("effector") is None
    present = {ia.getMath().getName() for ia in sbml.getListOfInitialAssignments()}
    assert present == {"naive_init", "MEK_init", "ERK_init"}


def test_a_species_two_readouts_read_is_named_once():
    """Renamed once per readout, the species took the second name and left the first defined by
    nothing, so what read the first read a parameter fitted in place of the species (gemot,
    boehm r2: two names for STAT5B)."""
    antimony, _ = PS.to_antimony(
        [(_TRANSITION, "mass_action", "r1")],
        {"naive": Agent("naive"), "naive_total": Agent("naive")},
    )
    sbml = SbmlModel.from_antimony(antimony).sbml_model
    assert sbml.getSpecies("naive") is not None
    assert sbml.getAssignmentRuleByVariable("naive_total").getMath().getName() == "naive"


def test_a_blinded_readout_stays_a_rule():
    """A `readout_*` symbol is a measurement mapping, which the lint accepts only as an assignment
    rule; renamed to the species it reads, Fujita's whole statement model was rejected."""
    antimony, _ = PS.to_antimony(
        [(_TRANSITION, "mass_action", "r1")], {"readout_naive": Agent("naive")}
    )
    sbml = SbmlModel.from_antimony(antimony).sbml_model
    assert sbml.getAssignmentRuleByVariable("readout_naive") is not None


def test_what_the_start_cannot_reach_starts_present():
    """Populations that only interconvert are each made by a rule, but nothing at the start makes
    them; with every made monomer started empty, as a diff of each rule's names had it, the model
    stayed zero. A made monomer the start does reach still starts empty."""
    antimony, _ = PS.to_antimony(
        [
            (_TRANSITION, "mass_action", "r1"),
            (PopulationTransition(Agent("A"), Agent("B")), "mass_action", "r2"),
            (PopulationTransition(Agent("B"), Agent("A")), "mass_action", "r3"),
        ],
        {},
    )
    sbml = SbmlModel.from_antimony(antimony).sbml_model
    present = {ia.getMath().getName() for ia in sbml.getListOfInitialAssignments()}
    assert present == {"naive_init", "A_init", "B_init"}


def test_a_complex_nothing_forms_is_refused():
    """The start holds free monomers only, so a kinase active only while bound to a partner no
    statement binds it to never acts; built anyway, its rate constant was one no fit could move."""
    raf = Agent("RAF", bound_conditions=[BoundCondition(Agent("RAS"))])
    with pytest.raises(ModelUpdateError, match=r"rows \['r1'\].*readouts \['RAF_RAS'\]"):
        PS.to_antimony(
            [(Phosphorylation(raf, Agent("MEK")), INDRA_DEFAULT, "r1")], {"RAF_RAS": raf}
        )
