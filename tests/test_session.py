from __future__ import annotations

import pytest

from petab_mcp import operations as O
from petab_mcp.session import PetabSession


def test_load_and_summary(session):
    p = session.problem
    assert len(p.measurement_df) == 6
    assert list(p.observable_df.index) == ["obs_B"]
    assert set(p.x_free_ids) == {"k1", "k2"}


def test_sbml_hash_deterministic(session):
    assert session.sbml_hash() == session.sbml_hash()
    assert len(session.sbml_hash()) == 16


def test_problem_hash_tracks_parameter_table(session):
    # A parameter-only edit leaves the model (SBML) identical but must change the problem
    # identity — otherwise the fit cache / provenance would treat two different fits as one.
    sbml0, prob0 = session.sbml_hash(), session.problem_hash()
    res = O.change_parameters(
        session,
        [{"parameter_id": "k1", "lower_bound": 1e-4, "upper_bound": 1e4}],
        justification="widen k1",
    )
    assert res["applied"] is True
    assert session.sbml_hash() == sbml0  # model structure unchanged
    assert session.problem_hash() != prob0  # parameter table changed -> problem changed

    # ...and the edit is on the audit trail. It no longer carries before/after hashes of its own:
    # nothing read them, and the model is stored per version, so any before/after comparison is
    # recoverable from the models themselves.
    ev = next(
        e
        for e in reversed(session.provenance.recent_events(0))
        if e.get("operation") == "make_changes"
    )
    assert ev["type"] == "edit" and ev["justification"] == "widen k1"


def test_missing_yaml_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        PetabSession(tmp_path / "nope.yaml", provenance_dir=str(tmp_path / "p"))


def test_the_initial_model_is_not_a_version(session):
    """Nothing is fit at construction, so there are no versions yet."""
    assert session.provenance.current_version == -1
    assert session.provenance.list_versions() == []
    assert session.has_model() is True


def _species(session):
    sm = session.model.sbml_model
    return {sm.getSpecies(i).getId() for i in range(sm.getNumSpecies())}


def _reactions(session):
    sm = session.model.sbml_model
    return {sm.getReaction(i).getId() for i in range(sm.getNumReactions())}


def test_rollback_restores_and_copies_optimization(session):
    # Snapshot a trace-free v0, then a "fitted" v1 carrying optimization traces + a (stand-in)
    # pyPESTO result file, the way a real fitting commit would. (No engine here, so build them
    # via the store.)
    session.provenance.snapshot_version(
        model_ant=session.model.to_antimony(),
        parameters=session.problem.parameter_df,
        label="fit",
        kind="fit",
    )
    root = session.provenance.root
    result_file = root / "cache" / "pypesto_result_xyz.hdf5"
    result_file.parent.mkdir(parents=True, exist_ok=True)
    result_file.write_bytes(b"fake pypesto result")
    session.provenance.snapshot_version(
        model_ant=session.model.to_antimony(),
        parameters=session.problem.parameter_df,
        label="fit",
        kind="alteration",
        optimization={
            "n_starts": 2,
            "starts": [{"start": 0, "fval": 1.0}, {"start": 1, "fval": 4.0}],
            "trace": [
                {"start": 0, "iteration": 0, "fval": 9.0},
                {"start": 0, "iteration": 1, "fval": 1.0},
            ],
            "result_file": str(result_file),
        },
    )
    assert session.provenance.current_version == 1

    # Rolling back to the version carrying traces restores them onto the session ...
    session.rollback_to(1)
    opt = session.optimization
    assert opt is not None and opt["n_starts"] == 2
    assert [t["fval"] for t in opt["trace"]] == [9.0, 1.0]

    # ... and the pyPESTO result file is copied forward into the new rollback version.
    head = session.provenance.current_version
    assert (session.provenance.versions_dir / f"v{head}" / "pypesto_result.hdf5").is_file()

    # rolling back to a version with no traces clears them
    session.rollback_to(0)
    assert session.optimization is None


def test_describe_names_the_condition_targets(dosed_session):
    """froehlich's analysts spent rounds listing its 145 targets out of a 36 MB `list_conditions`
    (PEtab-MCP#290)."""
    assert dosed_session.describe()["condition_targets"] == ["dose"]
