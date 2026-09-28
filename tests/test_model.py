from __future__ import annotations

import pytest

from petab_mcp import model as M
from petab_mcp.errors import ModelUpdateError


def test_get_model_formats(session):
    ant = M.get_model(session, "antimony")
    sbml = M.get_model(session, "sbml")
    assert "conversion_reaction" in ant
    assert "<sbml" in sbml
    with pytest.raises(ModelUpdateError):
        M.get_model(session, "matlab")


def test_model_structure(session):
    s = M.model_structure(session)
    assert set(s["species"]) == {"A", "B"}
    assert {"k1", "k2", "A_init", "B_init"} <= set(s["parameters"])
    assert {r["id"] for r in s["reactions"]} == {"r1", "r2"}
    assert all(r["rate_law"] for r in s["reactions"])


def test_parametric_initial_conditions_present(session):
    # every species has an initial-assignment to an <id>_init parameter
    sm = session.model.sbml_model
    ic_symbols = {
        sm.getInitialAssignment(i).getSymbol() for i in range(sm.getNumInitialAssignments())
    }
    assert {"A", "B"} <= ic_symbols
    assert {"A_init", "B_init"} <= set(session.model.get_parameter_ids())
