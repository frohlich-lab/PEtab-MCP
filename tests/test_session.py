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


def test_problem_hash_tracks_parameter_table(session):
    # A parameter-only edit leaves the model (SBML) identical but must change the problem
    # identity — otherwise the fit cache / provenance would treat two different fits as one.
    sbml0, prob0 = session.sbml_hash(), session.problem_hash()
    res = O.change_parameter_bounds(session, "k1", 1e-4, 1e4, justification="widen k1")
    assert res["applied"] is True
    assert session.sbml_hash() == sbml0        # model structure unchanged
    assert session.problem_hash() != prob0     # parameter table changed -> problem changed

    # and the edit event makes that visible: model hash steady, problem hash moved
    ev = next(e for e in reversed(session.provenance.recent_events(0))
              if e.get("operation") == "change_parameter_bounds")
    assert ev["model_hash_before"] == ev["model_hash_after"]
    assert ev["problem_hash_before"] != ev["problem_hash_after"]


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


def test_rollback_restores_and_copies_optimization(session):
    # Snapshot a "fitted" v1 carrying optimization traces + a (stand-in) pyPESTO result file,
    # the way a real fitting commit would. (No engine here, so build it via the store.)
    root = session.provenance.root
    result_file = root / "cache" / "pypesto_result_xyz.hdf5"
    result_file.parent.mkdir(parents=True, exist_ok=True)
    result_file.write_bytes(b"fake pypesto result")
    session.provenance.snapshot_version(
        model_ant=session.model.to_antimony(),
        parameters=session.problem.parameter_df,
        label="fit", kind="alteration",
        optimization={
            "n_starts": 2,
            "starts": [{"start": 0, "fval": 1.0}, {"start": 1, "fval": 4.0}],
            "trace": [{"start": 0, "iteration": 0, "fval": 9.0},
                      {"start": 0, "iteration": 1, "fval": 1.0}],
            "result_file": str(result_file),
        },
    )
    assert session.provenance.current_version == 1

    # Rolling back to v1 restores its optimization traces onto the session ...
    session.rollback_to(1)
    opt = session.optimization
    assert opt is not None and opt["n_starts"] == 2
    assert [t["fval"] for t in opt["trace"]] == [9.0, 1.0]

    # ... and the pyPESTO result file is copied forward into the new rollback version.
    head = session.provenance.current_version
    assert (session.provenance.versions_dir / f"v{head}" / "pypesto_result.hdf5").is_file()

    # rolling back to the unfitted initial version clears the traces
    session.rollback_to(0)
    assert session.optimization is None


def test_reset_reverts_to_v0(session, reauthor_toy):
    reauthor_toy(session, ["J: B => D; k3 * B;"], justification="test", references=["PMID:1"])
    assert "D" in _species(session)
    out = session.reset()
    assert out["reset"] is True
    # structurally identical to the initial model (SBML serialization need not be
    # byte-identical across the Antimony round-trip)
    assert _species(session) == {"A", "B"}
    assert _reactions(session) == {"r1", "r2"}
