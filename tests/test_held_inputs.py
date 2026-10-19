"""What the statements route holds of an experiment's inputs, and what it leaves a state."""

from __future__ import annotations

from indra.statements import Agent, Complex
from petab.v1.models.sbml_model import SbmlModel

from petab_mcp import problem_sbml as PS
from petab_mcp.mechanisms import INDRA_DEFAULT


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
