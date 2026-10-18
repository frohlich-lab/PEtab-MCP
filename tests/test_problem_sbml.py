"""The SBML a set of statements builds for one PEtab problem: readout names, starting amounts,
inputs."""

from __future__ import annotations

import libsbml
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


def test_a_rule_that_never_fires_points_to_starts():
    """The refusal said everything starts free, which `starts` made untrue, and never named it as
    the way to have the missing form present at t=0."""
    raf = Agent("RAF", bound_conditions=[BoundCondition(Agent("RAS"))])
    with pytest.raises(ModelUpdateError, match="declare it in `starts`"):
        PS.to_antimony([(Phosphorylation(raf, Agent("MEK")), INDRA_DEFAULT, "r1")], {})


# -- declared starts (make_changes' `starts`) ----------------------------------------
def _ia(sbml, sid):
    ia = sbml.getInitialAssignmentBySymbol(sid)
    return None if ia is None else libsbml.formulaToL3String(ia.getMath())


def test_a_declared_unit_pool_replaces_the_estimated_seed():
    """Blasi's unacetylated H4 and Sneyd's IPR start at 1, which reconcile fixes as a unit
    normalisation; seeded at an estimated `<monomer>_init` instead, each cost one k for nothing."""
    antimony, _ = PS.to_antimony(
        [(_PHOSPHO, INDRA_DEFAULT, "r1")],
        {"MEK": Agent("MEK", db_refs={"HGNC": "6840"})},
        starts=[("pERK", Agent("ERK"), 1)],
    )
    sbml = SbmlModel.from_antimony(antimony).sbml_model
    params = {p.getId(): p.getValue() for p in sbml.getListOfParameters()}
    assert params["pERK_init"] == 1.0 and "ERK_init" not in params
    assert "MEK_init" in params  # an undeclared monomer is still seeded as before


def test_a_made_monomer_can_be_declared_present():
    """Crauste's effectors start empty because naive cells make them; a population present at the
    start despite being made (Lucarelli's genes at 1) must say so, and then it is present."""
    antimony, _ = PS.to_antimony(
        [(_TRANSITION, "mass_action", "r1")],
        {"effector": Agent("effector")},
        starts=[("effector", Agent("effector"), 5)],
    )
    sbml = SbmlModel.from_antimony(antimony).sbml_model
    assert _ia(sbml, "effector") == "effector_init"
    assert sbml.getParameter("effector_init").getValue() == 5.0


def test_a_declared_start_is_the_state_the_agent_writes():
    """Perelson's virus starts infectious, Isensee's RII in each of its states: the start is the
    Agent's own pattern, every unstated site in its default state."""
    p_erk = Agent._from_json(
        {"name": "ERK", "mods": [{"mod_type": "phosphorylation", "residue": "T", "position": "185"}]}
    )
    antimony, _ = PS.to_antimony(
        [(_PHOSPHO, INDRA_DEFAULT, "r1")],
        {"pERK": p_erk},
        starts=[("pERK", p_erk, 0.5), ("ERK", Agent("ERK"), 2)],
    )
    sbml = SbmlModel.from_antimony(antimony).sbml_model
    names = {sp.getId(): sp.getName() for sp in sbml.getListOfSpecies()}
    assert names["pERK"] == "ERK(T185='p')"
    assert _ia(sbml, "pERK") == "pERK_init"
    assert {_ia(sbml, sid) for sid in names} >= {"pERK_init", "ERK_init"}


def test_a_declared_complex_starts_bound():
    """Bachmann's EpoR:JAK2 exists before any rule runs; the partner is built on the sites INDRA
    names after each other."""
    raf_ras = Agent("RAF", bound_conditions=[BoundCondition(Agent("RAS"))])
    antimony, _ = PS.to_antimony(
        [(Phosphorylation(raf_ras, Agent("MEK")), INDRA_DEFAULT, "r1")],
        {"RAF_RAS": raf_ras},
        starts=[("RAF_RAS", raf_ras, 3)],
    )
    sbml = SbmlModel.from_antimony(antimony).sbml_model
    assert sbml.getSpecies("RAF_RAS").getName().count("._br_") == 1
    assert _ia(sbml, "RAF_RAS") == "RAF_RAS_init"


def test_a_declared_expression_names_parameters_the_table_fills():
    """Lucarelli's SMADs start at a condition column and Fiedler's kinases at a shipped total; the
    names become parameters for the tables to fill, the placeholder goes."""
    antimony, _ = PS.to_antimony(
        [(_TRANSITION, "mass_action", "r1")],
        {"naive": Agent("naive")},
        starts=[("naive", Agent("naive"), "pool*fold")],
    )
    sbml = SbmlModel.from_antimony(antimony).sbml_model
    assert _ia(sbml, "naive") == "pool * fold"
    ids = {p.getId() for p in sbml.getListOfParameters()}
    assert {"pool", "fold"} <= ids and "naive_init" not in ids


def test_an_input_declares_no_start():
    """An input is held to its protocol; a start for it would be overwritten at t=0."""
    egf = Agent("EGF")
    with pytest.raises(ModelUpdateError, match="is an input"):
        PS.to_antimony(
            [(Phosphorylation(egf, Agent("AKT")), INDRA_DEFAULT, "r1")],
            {"EGF": egf},
            inputs={"EGF": "EGF_0"},
            starts=[("EGF", egf, 1)],
        )


def test_a_declared_empty_start_has_no_parameter():
    """Enzyme kinetics' substrate is set by every condition; an `_init` behind it, even a zero one,
    was an estimated row no condition left any effect to."""
    antimony, _ = PS.to_antimony(
        [(_TRANSITION, "mass_action", "r1")], {"naive": Agent("naive")},
        starts=[("naive", Agent("naive"), 0)],
    )
    sbml = SbmlModel.from_antimony(antimony).sbml_model
    assert _ia(sbml, "naive") is None and sbml.getParameter("naive_init") is None
    assert sbml.getSpecies("naive").getInitialAmount() == 0.0


def test_indra_padded_rules_build():
    """INDRA pads one-step Conversion and subject-driven amount rules with None for Kappa; the
    trace read every side as a complex and every such statement failed to build (Bachmann's
    SOCS3-driven degradation, Bruno's conversions, Sneyd's gating)."""
    from indra.statements import Conversion, DecreaseAmount

    antimony, _ = PS.to_antimony(
        [
            (Conversion(None, [Agent("A")], [Agent("B")]), INDRA_DEFAULT, "r1"),
            (DecreaseAmount(Agent("C"), Agent("B")), INDRA_DEFAULT, "r2"),
        ],
        {"B": Agent("B"), "C": Agent("C")},
    )
    assert SbmlModel.from_antimony(antimony).sbml_model.getNumReactions() == 2
