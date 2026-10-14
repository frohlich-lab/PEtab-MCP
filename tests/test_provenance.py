from __future__ import annotations

import pandas as pd
import pytest

from petab_mcp.provenance import ProvenanceStore


def _pdf(records):
    """Build a parameter-table DataFrame (index = parameterId) from record dicts."""
    df = pd.DataFrame(records)
    if "parameterId" in df.columns:
        df = df.set_index("parameterId")
    else:
        df.index.name = "parameterId"
    return df


def test_diff_versions_reports_fit_delta(tmp_path):
    store = ProvenanceStore(tmp_path / ".petab_mcp")
    store.snapshot_version(
        "model m() end",
        _pdf([{"parameterId": "k", "estimate": 1}]),
        label="v0",
        meta={"neg_log_likelihood": -6.6, "chi2": 3.4},
    )
    store.snapshot_version(
        "model m() k=1 end",
        _pdf([{"parameterId": "k", "estimate": 1}]),
        label="v1",
        meta={"neg_log_likelihood": -8.3, "chi2": 0.04},
    )
    fit = store.diff_versions(0, 1)["fit"]
    assert fit["from_neg_log_likelihood"] == -6.6
    assert fit["to_neg_log_likelihood"] == -8.3
    assert fit["neg_log_likelihood_delta"] == pytest.approx(1.7)  # positive => v1 better
    assert fit["from_chi2"] == 3.4 and fit["to_chi2"] == 0.04

    # a version without stored fit metrics yields a None delta (not a crash)
    store.snapshot_version("model m() end", _pdf([]), label="v2")
    assert store.diff_versions(0, 2)["fit"]["neg_log_likelihood_delta"] is None


def test_snapshot_and_get(tmp_path):
    store = ProvenanceStore(tmp_path / ".petab_mcp")
    v = store.snapshot_version(
        "model m() end", _pdf([{"parameterId": "k", "estimate": 1}]), label="a"
    )
    assert v == 0
    assert store.current_version == 0
    snap = store.get_version(0)
    assert snap["model_antimony"].startswith("model m")
    assert snap["parameters"][0]["parameterId"] == "k"
    # the parameter table round-trips as a real DataFrame, not JSON records
    df = store.get_parameter_df(0)
    assert list(df.index) == ["k"] and df.index.name == "parameterId"
    # stored as a TSV, not parameters.json
    assert (store.versions_dir / "v0" / "parameters.tsv").is_file()
    assert not (store.versions_dir / "v0" / "parameters.json").exists()


def test_snapshot_persists_optimization_and_copies_result_file(tmp_path):
    store = ProvenanceStore(tmp_path / ".petab_mcp")
    # a stand-in pyPESTO result file (the fit would write a real HDF5 here)
    result_file = tmp_path / "cache" / "pypesto_result_abc.hdf5"
    result_file.parent.mkdir(parents=True)
    result_file.write_bytes(b"\x89HDF\r\n\x1a\n fake result")

    optimization = {
        "n_starts": 2,
        "starts": [{"start": 0, "fval": 1.0}, {"start": 1, "fval": 5.0}],
        "trace": [
            {"start": 0, "iteration": 0, "fval": 9.0},
            {"start": 0, "iteration": 1, "fval": 1.0},
        ],
        "result_file": str(result_file),
    }
    v = store.snapshot_version(
        "model m() end",
        _pdf([{"parameterId": "k", "estimate": 1}]),
        label="fit",
        optimization=optimization,
    )

    vdir = store.versions_dir / f"v{v}"
    assert (vdir / "optimization.json").is_file()
    assert (vdir / "pypesto_result.hdf5").is_file()  # the result file was copied in

    snap = store.get_version(v)
    opt = snap["optimization"]
    assert opt["n_starts"] == 2 and len(opt["starts"]) == 2
    assert [t["fval"] for t in opt["trace"]] == [9.0, 1.0]
    # result_file now points at this version's own copy
    assert opt["result_file"] == str(vdir / "pypesto_result.hdf5")

    # a version without optimization simply carries none
    v2 = store.snapshot_version("model m() end", _pdf([]), label="plain")
    assert store.get_version(v2)["optimization"] is None


def test_events_appended_and_tailed(tmp_path):
    store = ProvenanceStore(tmp_path / ".petab_mcp")
    for i in range(5):
        store.append_event({"type": "x", "i": i})
    ev = store.recent_events(3)
    assert [e["i"] for e in ev] == [2, 3, 4]
    assert all("timestamp" in e for e in ev)


def test_versioning_diff_and_rollback(session, reauthor_toy):
    reauthor_toy(session, ["J: B => D; k3 * B;"], justification="x", references=["PMID:1"])
    assert session.provenance.current_version == 0

    reauthor_toy(session, ["J: B => E; k4 * B;"], justification="y")
    diff = session.provenance.diff_versions(0, 1)
    # D and E are both created empty, so neither gets a `<sid>_init` (see the IC convention in
    # reconcile.ensure_parametric_initial_conditions) — only the rate constants move.
    assert set(diff["parameter_diff"]["added"]) == {"k4"}
    assert set(diff["parameter_diff"]["removed"]) == {"k3"}

    # rollback to v0 makes it the working state and moves HEAD back onto it — no new version
    out = session.rollback_to(0)
    assert out["rolled_back_to"] == 0
    assert session.provenance.current_version == 0
    assert len(session.provenance.list_versions()) == 2
    species = {
        session.model.sbml_model.getSpecies(i).getId()
        for i in range(session.model.sbml_model.getNumSpecies())
    }
    assert species == {"A", "B", "D"}
    # ...and the next fit does not collide with the version HEAD skipped back over
    reauthor_toy(session, ["J: B => F; k5 * B;"], justification="z")
    assert session.provenance.current_version == 2


def test_durable_resume(yaml_path, tmp_path, reauthor_toy):
    """A second session on the same provenance dir resumes from HEAD."""
    from petab_mcp.references import StubReferenceResolver
    from petab_mcp.session import PetabSession

    prov = str(tmp_path / "prov")
    s1 = PetabSession(yaml_path, provenance_dir=prov, resolver=StubReferenceResolver())
    reauthor_toy(s1, ["J: B => D; k3 * B;"], justification="x", references=["PMID:1"])
    assert s1.provenance.current_version == 0

    s2 = PetabSession(yaml_path, provenance_dir=prov, resolver=StubReferenceResolver())
    species = {
        s2.model.sbml_model.getSpecies(i).getId()
        for i in range(s2.model.sbml_model.getNumSpecies())
    }
    assert "D" in species  # resumed the altered model, not the pristine YAML


def test_one_unreadable_meta_does_not_empty_the_version_listing(session, reauthor_toy):
    """Regression: callers degrade a raise here to an empty list, so one corrupt meta silently
    switched off de-duplication and the repeated-widening backstop."""
    reauthor_toy(session, ["J: B => D; k3 * B;"], justification="x")
    reauthor_toy(session, ["J: B => E; k4 * B;"], justification="y")
    before = session.provenance.list_versions()
    assert len(before) >= 2

    (session.provenance.versions_dir / "v0" / "meta.json").write_text("{ not json")

    after = session.provenance.list_versions()
    assert len(after) == len(before) - 1, "only the corrupt version should drop out"
    assert 0 not in [m.get("version") for m in after]
    # ...and the survivors are intact, so de-dup and the widening walk still have history
    assert [m.get("version") for m in after] == [
        v for v in (m.get("version") for m in before) if v != 0
    ]


def test_a_meta_that_is_not_an_object_is_excluded_too(session, reauthor_toy):
    """Valid JSON that is not a dict must be skipped, not returned: every caller does `.get` on
    each element without guarding, so a `None` in the list escapes into the commit path."""
    from petab_mcp import dedup as D

    reauthor_toy(session, ["J: B => D; k3 * B;"], justification="x")
    reauthor_toy(session, ["J: B => E; k4 * B;"], justification="y")
    (session.provenance.versions_dir / "v0" / "meta.json").write_text("null")

    metas = session.provenance.list_versions()
    assert all(isinstance(m, dict) for m in metas)
    assert [m["version"] for m in metas] == [1]
    D.build_registry(session)  # must not raise


def test_every_version_is_a_fit(session, reauthor_toy):
    """A version exists because a fit produced it: the initial model is working state, and a
    restore moves HEAD rather than appending an unfit duplicate."""
    reauthor_toy(session, ["J: B => D; k3 * B;"], justification="x")
    reauthor_toy(session, ["J: B => E; k4 * B;"], justification="y")
    session.restore_version(0)

    metas = session.provenance.list_versions()
    assert len(metas) == 2
    assert {m["kind"] for m in metas} == {"fit"}
    # every one of them competes on bic; none is excluded as unscored
    from petab_mcp import dedup as D

    assert {r["version"] for r in D.build_registry(session)} == {m["version"] for m in metas}


def test_a_store_written_by_the_v1_server_is_refused(tmp_path):
    """Its versions hold v1 parameter tables and condition-keyed simulations; an empty store and
    a v2 store open as before."""
    from petab_mcp.errors import PetabMcpError

    root = tmp_path / ".petab_mcp"
    ProvenanceStore(root).snapshot_version("model m() end", _pdf([]), label="v0")
    ProvenanceStore(root)  # its own FORMAT file
    (root / "FORMAT").unlink()
    with pytest.raises(PetabMcpError, match="fresh --provenance-dir"):
        ProvenanceStore(root)
    ProvenanceStore(tmp_path / "empty")
