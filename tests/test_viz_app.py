"""The viewer assembled (``petab_mcp.viz.app``): every tab of every version builds and renders to
Bokeh without a server, and the app follows the store as the server writes it. The parts' own
tests do not reach the assembly, which is where widgets, streams and plot options meet."""

from __future__ import annotations

import json
import shutil

import holoviews as hv
import panel as pn
import pytest
from bokeh.document import Document

from petab_mcp.problem import WorkingProblem
from petab_mcp.provenance import FIT_PROGRESS
from petab_mcp.viz.app import VizApp
from petab_mcp.viz.store import Store

hv.extension("bokeh")


@pytest.fixture
def app(viz_store):
    root, yaml = viz_store
    return VizApp(Store(root), WorkingProblem.from_yaml(yaml), poll_seconds=0)


def test_every_tab_of_every_version_renders(app):
    for version in app.store.versions():
        app.version_pick.value = version
        for tab in (app.multistart, app.model, app.fit):
            pn.panel(tab.view()).get_root(Document())
    app.fit.view_kind.value = "Residuals"
    pn.panel(app.fit.view()).get_root(Document())
    for level in app.model.level.options:
        app.model.level.value = level
        pn.panel(app.model.view()).get_root(Document())
    app.view().server_doc(Document())


def test_the_app_opens_on_head_and_compares_with_its_parent(app):
    assert (app.version, app.compare) == (1, 0)
    app.compare_pick.value = "nothing"
    assert app.compare is None


def test_a_new_version_moves_a_viewer_following_head(app):
    root = app.store.root
    shutil.copytree(root / "versions" / "v1", root / "versions" / "v2")
    meta = root / "versions" / "v2" / "meta.json"
    meta.write_text(json.dumps({**json.loads(meta.read_text()), "version": 2, "parent": 1}))
    (root / "HEAD").write_text("2")
    app.poll()
    assert (app.version, app.compare) == (2, 1)


def test_a_running_fit_shows_without_a_new_version(app):
    tick = app.tick
    (app.store.root / FIT_PROGRESS).write_text(
        json.dumps(
            {
                "phase": "running",
                "since": 0,
                "began": 0,
                "waited": 0,
                "total": 4,
                "done": 2,
                "queued": 0,
                "best": 1.0,
                "fvals": [1.0, 2.0],
            }
        )
    )
    app.poll()
    assert app.tick == tick + 1 and app.version == 1
    pn.panel(app.multistart.view()).get_root(Document())


def test_a_starts_exit_is_named_as_fides_names_it():
    """Fides' -1 is MAXITER, and the server also writes -1 for a start it stopped past its cap: a
    hand-written table had named every MAXITER start "failed / stopped"."""
    from petab_mcp.viz.multistart_tab import STOPPED, _exit

    assert _exit(-1, "Exceeded maximum number of iterations") == "MAXITER"
    assert _exit(-2, None) == "MAXTIME"
    assert _exit(-5, None) == "DELTA_TOO_SMALL"
    assert _exit(-1, "stopped by the server: still running past its 180s cap") == STOPPED
    assert _exit(None, None) == "unknown"


def test_the_viewer_never_simulates(viz_store):
    """The viewer shows what the fits stored and must never simulate the model (user, 2026-11-26):
    building every tab of every version, the pyPESTO result file included, loads neither AMICI nor
    pyPESTO, so nothing it does can simulate or compile. Run in a fresh interpreter, which no other
    test has imported either into."""
    import subprocess
    import sys

    root, yaml = viz_store
    script = f"""
import sys
import holoviews as hv, panel as pn
from bokeh.document import Document
from petab_mcp.problem import WorkingProblem
from petab_mcp.viz.app import VizApp
from petab_mcp.viz.store import Store
hv.extension("bokeh")
app = VizApp(Store({str(root)!r}), WorkingProblem.from_yaml({str(yaml)!r}), poll_seconds=0)
for version in app.store.versions():
    app.version_pick.value = version
    for tab in (app.multistart, app.model, app.fit):
        pn.panel(tab.view()).get_root(Document())
assert app.store.starts(0)["exitflag"].notna().all()
print(sorted({{m.split(".")[0] for m in sys.modules}} & {{"amici", "pypesto"}}))
"""
    out = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=True)
    assert out.stdout.strip().splitlines()[-1] == "[]", out.stderr[-2000:]


def test_picking_an_older_version_stops_following_head(app):
    """Following HEAD, a pick of an older version was undone by the next fit's version."""
    app.version_pick.value = 0
    assert app.follow.value is False


def test_the_cli_says_what_is_missing_instead_of_serving_a_traceback(tmp_path, viz_store):
    from petab_mcp.viz.cli import main

    with pytest.raises(SystemExit, match="no problem at"):
        main(["--petab-yaml", str(tmp_path / "nope.yaml"), "--no-show"])
    with pytest.raises(SystemExit, match="not a version store.*--provenance-dir"):
        main(["--petab-yaml", str(viz_store[1]), "--no-show"])


def test_a_problem_yaml_is_named_after_its_example():
    from petab_mcp.viz.app import problem_name

    assert problem_name("examples/fujita/problem/problem.yaml") == "fujita"
    assert problem_name("examples/fujita/work/problem.yaml") == "fujita"
    assert problem_name("Fujita_SciSignal2010/Fujita_SciSignal2010.yaml") == "Fujita_SciSignal2010"


def test_a_store_before_its_first_version_shows_its_first_fit(tmp_path, viz_store):
    """The viewer is opened as a session starts, before anything is stored: every tab says so, and
    the first fit's starts show as they finish; its first version is then followed to."""
    root = tmp_path / "fresh"
    (root / "versions").mkdir(parents=True)
    (root / FIT_PROGRESS).write_text(
        json.dumps(
            {
                "phase": "running",
                "since": 0,
                "began": 0,
                "waited": 0,
                "total": 4,
                "done": 1,
                "queued": 0,
                "best": 3.0,
                "fvals": [3.0],
            }
        )
    )
    app = VizApp(Store(root), WorkingProblem.from_yaml(viz_store[1]), poll_seconds=0)
    assert app.version is None
    for tab in (app.multistart, app.model, app.fit):
        pn.panel(tab.view()).get_root(Document())
    app.view().server_doc(Document())
    shutil.copytree(viz_store[0] / "versions" / "v0", root / "versions" / "v0")
    (root / "HEAD").write_text("0")
    (root / FIT_PROGRESS).unlink()
    app.poll()
    assert app.version == 0
