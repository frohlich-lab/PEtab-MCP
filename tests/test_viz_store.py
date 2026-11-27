"""The viewer's store reader (``petab_mcp.viz.store``): it reads a store as the server leaves it
and as gemot archives it, and never writes to either."""

from __future__ import annotations

import json

import pytest

from petab_mcp.errors import ValidationError
from petab_mcp.provenance import FIT_PROGRESS
from petab_mcp.viz.store import Store


def _files(root):
    return {p: p.stat().st_mtime_ns for p in root.rglob("*")}


def test_a_store_is_read_without_a_file_written(viz_store):
    """ProvenanceStore writes FORMAT and makes directories as it opens, so the viewer reads the
    files itself: an archived run must come out as it went in."""
    root, _ = viz_store
    before = _files(root)
    store = Store(root)
    store.table(), store.starts(1), store.simulations(0), store.trace(1), store.fit_progress()
    assert _files(root) == before


def test_an_archived_versions_gzipped_optimization_is_read(viz_store):
    """gemot's archive keeps optimization.json gzipped and drops pypesto_result.hdf5: the starts
    are read all the same, with how each ended left unknown."""
    store = Store(viz_store[0])
    starts = store.starts(1)
    assert list(starts["fval"]) == [4.0, 4.1, 4.5, 30.0]
    assert starts["exitflag"].isna().all()
    assert store.optimization(1)["exitflag_counts"] == {"ExitFlag.FTOL": 4}


def test_a_store_an_earlier_server_wrote_is_refused(viz_store):
    root, _ = viz_store
    (root / "FORMAT").write_text("petab-v1")
    with pytest.raises(ValidationError, match="petab-v1"):
        Store(root)


def test_the_best_is_the_servers_checkpoint_not_the_lowest_score(viz_store):
    """The server can keep a best over a lower score (a collapsed noise sigma), and stamps the
    versions it adopts is_new_best; the viewer rings what the server ranks."""
    root, _ = viz_store
    meta = root / "versions" / "v1" / "meta.json"
    meta.write_text(json.dumps({**json.loads(meta.read_text()), "is_new_best": False}))
    store = Store(root)
    assert store.best() == 0
    assert (
        store.table().set_index("version").at[1, "justification"] == "is the back reaction needed?"
    )


def test_the_running_fit_is_read_while_the_server_writes_it(viz_store):
    root, _ = viz_store
    store = Store(root)
    assert store.fit_progress() is None
    before = store.signature()
    (root / FIT_PROGRESS).write_text(json.dumps({"phase": "running", "done": 1, "total": 4}))
    assert store.fit_progress()["done"] == 1
    assert store.signature() != before


def test_a_file_read_before_the_server_wrote_it_is_read_again(viz_store):
    """The server writes a version's meta.json before its simulation; a viewer polling between the
    two must not keep "no simulation" for that version."""
    root, _ = viz_store
    store = Store(root)
    sim = root / "versions" / "v1" / "simulations.json"
    held = sim.read_text()
    sim.unlink()
    assert store.simulations(1) is None
    sim.write_text(held)
    assert store.simulations(1) is not None


def test_how_each_start_ended_is_read_from_the_pypesto_result_file(viz_store):
    store = Store(viz_store[0])
    starts = store.starts(0)
    assert list(starts["exitflag"]) == [1, 3, -1]
    assert starts["message"].iloc[2].startswith("stopped by the server")
    assert starts["x0"].iloc[0] == [-0.5, -1.5, 0.1, 1.0, 0.0]
