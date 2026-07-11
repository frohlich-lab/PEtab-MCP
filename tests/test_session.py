from __future__ import annotations

import pytest

from petab_mcp.session import PetabSession


def test_load_and_summary(session):
    p = session.problem
    assert p.n_measurements == 6
    assert list(p.get_observable_ids()) == ["obs_B"]
    assert set(p.x_free_ids) == {"k1", "k2"}


def test_sbml_hash_deterministic(session):
    assert session.sbml_hash() == session.sbml_hash()
    assert len(session.sbml_hash()) == 16


def test_missing_yaml_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        PetabSession(tmp_path / "nope.yaml")


def test_reset_restores_model(session):
    from petab_mcp import model as M

    original = M.get_model(session, "antimony")
    original_hash = session.sbml_hash()
    # edit an SBML model parameter, then reset
    M.set_parameter_value(session, "k1", 12.34)
    assert session.sbml_hash() != original_hash
    session.reset()
    assert session.sbml_hash() == original_hash
    assert M.get_model(session, "antimony") == original
