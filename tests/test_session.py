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


def test_describe_says_which_measurements_are_at_steady_state(session):
    """blasi's 252 measurements are all at t=inf (PEtab-MCP#339)."""
    assert "steady_state" not in session.describe()
    df = session.problem.measurement_df
    df["time"] = df["time"].astype(float)
    df.loc[df.index[:2], "time"] = float("inf")
    line = session.describe()["steady_state"]
    assert line.startswith(f"2 of the {len(df)} measurements are at time inf")


_FLATTEN_OBSERVABLES = (
    "observableId\tobservableFormula\tnoiseFormula\tobservablePlaceholders\n"
    "obs_B\tsign_obs_B * B\tsigma_B\tsign_obs_B\n"
)


def _flattening_problem(tmp_path, overrides):
    """The toy problem with one observable placeholder whose override varies between
    timepoints of one experiment, so it needs flattening."""
    import shutil

    from conftest import DATA_DIR

    root = tmp_path / "problem"
    shutil.copytree(DATA_DIR, root)
    (root / "observables.tsv").write_text(_FLATTEN_OBSERVABLES)
    rows = [
        f"obs_B\t\t{t}\t{m}\t{o}"
        for t, m, o in zip((0.0, 1.0, 2.0, 4.0), (0.0, 0.3, 0.48, 0.66), overrides, strict=True)
    ]
    (root / "measurements.tsv").write_text(
        "observableId\texperimentId\ttime\tmeasurement\tobservableParameters\n"
        + "\n".join(rows)
        + "\n"
    )
    return root / "problem.yaml"


def test_a_problem_whose_flattened_ids_are_invalid_is_refused_at_load(tmp_path):
    """An override of `-1` flattens to the observable id `obs_B__-1`, which PEtab rejects.
    It used to load cleanly and then fail every fit as "ill-posed for estimation", so an agent
    rewrote a model that was fine until its budget ran out (stop_reason: no_model_built after
    2.8 h). The tables it comes from are fixed, so it is refused once, naming the culprit."""
    from petab_mcp.errors import ValidationError

    yaml_path = _flattening_problem(tmp_path, ("1", "-1", "1", "1"))
    with pytest.raises(ValidationError) as exc:
        PetabSession(yaml_path, provenance_dir=str(tmp_path / "p"))
    message = str(exc.value)
    assert "obs_B__-1" in message and "'-1'" in message
    assert exc.value.issues == ["invalid flattened observable id 'obs_B__-1'"]


def test_a_problem_that_flattens_to_valid_ids_still_loads(tmp_path):
    """The check refuses only an unflattenable problem, not every one that needs flattening:
    named overrides flatten to valid ids, and the copy the compile path builds is the one the
    check tried."""
    from petab_mcp.fitting import amici_facing_problem, needs_flattening

    yaml_path = _flattening_problem(tmp_path, ("s_pos", "s_neg", "s_pos", "s_pos"))
    params = yaml_path.parent / "parameters.tsv"
    params.write_text(
        params.read_text() + "s_pos\tlin\t-10\t10\t1.0\tfalse\ns_neg\tlin\t-10\t10\t-1.0\tfalse\n"
    )
    s = PetabSession(yaml_path, provenance_dir=str(tmp_path / "p"))
    assert needs_flattening(s.problem) is True
    flat, flattened = amici_facing_problem(s)
    assert flattened and set(flat.observable_df.index) == {"obs_B__s_pos", "obs_B__s_neg"}
