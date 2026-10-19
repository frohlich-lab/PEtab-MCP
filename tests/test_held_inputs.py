"""What the statements route holds of an experiment's inputs, and what it leaves a state."""

from __future__ import annotations

import pytest
from indra.statements import Agent, Complex, DecreaseAmount
from petab.v1.models.sbml_model import SbmlModel

from petab_mcp import problem_sbml as PS
from petab_mcp.mechanisms import INDRA_DEFAULT
from petab_mcp.statements import PopulationTransition


def test_a_held_input_that_binds_is_its_free_form():
    """Isensee's forskolin binds adenylyl cyclase. Held as the free form under an anonymous id,
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
    """armistead's conditions set the starting amounts of its measured metabolites, which convert
    into each other; held on the boundary for being condition targets, they never changed."""
    antimony, _ = PS.to_antimony(
        [(PopulationTransition(Agent("naive"), Agent("effector")), "mass_action", "r1")],
        {"naive": Agent("naive"), "effector": Agent("effector")},
        targets=frozenset({"naive"}),
    )
    sbml = SbmlModel.from_antimony(antimony).sbml_model
    assert not sbml.getSpecies("naive").getBoundaryCondition()


def test_a_started_state_that_binds_is_its_free_form_too():
    """A ligand a row also degrades is a state the conditions only start; named for the sum over
    it and its complex, its condition target still set nothing (raimundez's EGF, before its rows
    stopped degrading it with the receptor)."""
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
