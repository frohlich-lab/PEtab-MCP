from __future__ import annotations

import pytest

from petab_mcp import operations as O
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
        PetabSession(tmp_path / "nope.yaml", provenance_dir=str(tmp_path / "p"))


def test_initial_version_snapshotted(session):
    assert session.provenance.current_version == 0
    versions = session.provenance.list_versions()
    assert versions[0]["kind"] == "initial"


def _species(session):
    sm = session.model.sbml_model
    return {sm.getSpecies(i).getId() for i in range(sm.getNumSpecies())}


def _reactions(session):
    sm = session.model.sbml_model
    return {sm.getReaction(i).getId() for i in range(sm.getNumReactions())}


def test_reset_reverts_to_v0(session):
    O.add_reaction(session, "J: B -> D; k3*B", justification="test", references=["PMID:1"])
    assert "D" in _species(session)
    out = session.reset()
    assert out["reset"] is True
    # structurally identical to the initial model (SBML serialization need not be
    # byte-identical across the Antimony round-trip)
    assert _species(session) == {"A", "B"}
    assert _reactions(session) == {"r1", "r2"}
