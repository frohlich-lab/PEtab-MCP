from __future__ import annotations

from petab_mcp import operations as O
from petab_mcp.provenance import ProvenanceStore


def test_snapshot_and_get(tmp_path):
    store = ProvenanceStore(tmp_path / ".petab_mcp")
    v = store.snapshot_version("model m() end", [{"parameterId": "k", "estimate": 1}], label="a")
    assert v == 0
    assert store.current_version == 0
    snap = store.get_version(0)
    assert snap["model_antimony"].startswith("model m")
    assert snap["parameters"][0]["parameterId"] == "k"


def test_events_appended_and_tailed(tmp_path):
    store = ProvenanceStore(tmp_path / ".petab_mcp")
    for i in range(5):
        store.append_event({"type": "x", "i": i})
    ev = store.recent_events(3)
    assert [e["i"] for e in ev] == [2, 3, 4]
    assert all("timestamp" in e for e in ev)


def test_versioning_diff_and_rollback(session):
    O.add_reaction(session, "J: B -> D; k3*B", justification="x", references=["PMID:1"])
    assert session.provenance.current_version == 1

    diff = session.provenance.diff_versions(0, 1)
    assert set(diff["parameter_diff"]["added"]) == {"D_init", "k3"}
    assert "k3" in diff["model_diff"] or "J" in diff["model_diff"]

    # rollback to v0 restores the original model, as a new appended version
    out = session.rollback_to(0)
    assert out["rolled_back_to"] == 0
    assert session.provenance.current_version == 2
    species = {
        session.model.sbml_model.getSpecies(i).getId()
        for i in range(session.model.sbml_model.getNumSpecies())
    }
    assert species == {"A", "B"}
    kinds = [v["kind"] for v in session.provenance.list_versions()]
    assert kinds == ["initial", "alteration", "rollback"]


def test_durable_resume(yaml_path, tmp_path):
    """A second session on the same provenance dir resumes from HEAD."""
    from petab_mcp.references import StubReferenceResolver
    from petab_mcp.session import PetabSession

    prov = str(tmp_path / "prov")
    s1 = PetabSession(yaml_path, provenance_dir=prov, resolver=StubReferenceResolver())
    O.add_reaction(s1, "J: B -> D; k3*B", justification="x", references=["PMID:1"])
    assert s1.provenance.current_version == 1

    s2 = PetabSession(yaml_path, provenance_dir=prov, resolver=StubReferenceResolver())
    species = {
        s2.model.sbml_model.getSpecies(i).getId()
        for i in range(s2.model.sbml_model.getNumSpecies())
    }
    assert "D" in species  # resumed the altered model, not the pristine YAML
