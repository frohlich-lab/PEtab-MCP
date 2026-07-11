from __future__ import annotations

import pytest

from petab_mcp import operations as O
from petab_mcp import parameters as P
from petab_mcp.errors import UnknownEntityError, ValidationError
from petab_mcp.operations import OperationError, parse_reaction_line
from petab_mcp.references import ReferenceError


def _ids(session):
    return {r["parameterId"] for r in P.parameters_to_records(session.problem)}


def _species(session):
    sm = session.model.sbml_model
    return {sm.getSpecies(i).getId() for i in range(sm.getNumSpecies())}


# -- parsing ---------------------------------------------------------------
def test_parse_reaction_line():
    spec = parse_reaction_line("J: 2 A + B -> C; k1*A*B")
    assert spec["id"] == "J"
    assert spec["reactants"] == [("A", 2.0), ("B", 1.0)]
    assert spec["products"] == [("C", 1.0)]
    assert spec["rate_law"] == "k1*A*B"


def test_parse_reaction_line_degradation():
    spec = parse_reaction_line("A => ; k_deg*A")
    assert spec["reactants"] == [("A", 1.0)]
    assert spec["products"] == []


def test_parse_reaction_line_no_arrow():
    with pytest.raises(OperationError):
        parse_reaction_line("A + B; k*A")


# -- enforcement -----------------------------------------------------------
def test_requires_justification(session):
    with pytest.raises(OperationError):
        O.add_reaction(session, "J: B -> D; k3*B", justification="  ", references=["PMID:1"])


def test_requires_references(session):
    with pytest.raises(ReferenceError):
        O.add_reaction(session, "J: B -> D; k3*B", justification="ok", references=[])


# -- add / change / remove -------------------------------------------------
def test_add_reaction_auto_adds_species_and_params(session):
    res = O.add_reaction(
        session, "J: B -> D; k3*B", justification="B converts to D", references=["PMID:1"]
    )
    assert res["committed"] is True
    assert "D" in _species(session)
    assert {"D_init", "k3"} <= _ids(session)  # new IC param + rate constant added to table
    assert set(res["reconcile"]["parameters"]["added"]) == {"D_init", "k3"}


def test_change_rate_law(session):
    O.add_reaction(session, "J: B -> D; k3*B", justification="x", references=["PMID:1"])
    res = O.change_rate_law(
        session, "J", "k3*B*B", justification="second order", references=["PMID:2"]
    )
    assert res["committed"] is True
    rxn = session.model.sbml_model.getReaction("J")
    assert "B * B" in rxn.getKineticLaw().getFormula().replace("^", " ") or "B^2" in rxn.getKineticLaw().getFormula()


def test_remove_reaction_prunes_orphans(session):
    O.add_reaction(session, "J: B -> D; k3*B", justification="x", references=["PMID:1"])
    res = O.remove_reaction(session, "J", justification="retracted", references=["PMID:3"])
    assert res["committed"] is True
    assert "D" not in _species(session)
    assert not ({"D_init", "k3"} & _ids(session))  # orphans pruned from the table


def test_remove_unknown_reaction(session):
    with pytest.raises(UnknownEntityError):
        O.remove_reaction(session, "nope", justification="x", references=["PMID:1"])


def test_operation_records_version_and_provenance(session):
    before = session.provenance.current_version
    O.add_reaction(session, "J: B -> D; k3*B", justification="x", references=["PMID:1"])
    assert session.provenance.current_version == before + 1
    events = session.provenance.recent_events()
    alt = [e for e in events if e["type"] == "alteration"][-1]
    assert alt["operation"] == "add_reaction"
    assert alt["justification"] == "x"
    assert alt["references"][0]["input"] == "PMID:1"
