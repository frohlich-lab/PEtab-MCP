from __future__ import annotations

import pytest

from petab_mcp import model as M
from petab_mcp.errors import ModelUpdateError, UnknownEntityError, ValidationError


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
    assert set(s["parameters"]) == {"k1", "k2"}
    assert {r["id"] for r in s["reactions"]} == {"r1", "r2"}


def test_update_model_roundtrip(session):
    ant = M.get_model(session, "antimony")
    result = M.update_model(session, ant, "antimony")
    assert result["committed"] is True
    assert result["validation"]["valid"] is True


def test_update_model_guardrail_rejects_dropped_entity(session):
    # Model that drops species B and parameter k2, which the fixed tables need.
    broken = (
        "model conversion_reaction\n"
        "  species A = 1;\n"
        "  k1 = 0.5;\n"
        "  r1: A => ; k1 * A;\n"
        "end"
    )
    with pytest.raises(ValidationError) as exc:
        M.update_model(session, broken, "antimony")
    assert exc.value.issues  # carries the specific lint issues
    # original model still intact
    assert "k2" in M.get_model(session, "antimony")


def test_set_model_parameter(session):
    before = session.sbml_hash()
    out = M.set_parameter_value(session, "k1", 2.5)
    assert out["value"] == 2.5
    assert session.sbml_hash() != before
    with pytest.raises(UnknownEntityError):
        M.set_parameter_value(session, "does_not_exist", 1.0)
