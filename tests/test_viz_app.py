"""The viewer assembled (``petab_mcp.viz.app``): every tab of every version builds and renders to
Bokeh without a server, and the app follows the store as the server writes it. The parts' own
tests do not reach the assembly, which is where widgets, streams and plot options meet."""

from __future__ import annotations

import json
import shutil

import holoviews as hv
import pandas as pd
import panel as pn
import pytest
from bokeh.document import Document

from petab_mcp.problem import WorkingProblem
from petab_mcp.provenance import FIT_PROGRESS
from petab_mcp.viz.app import VizApp
from petab_mcp.viz.key import Key
from petab_mcp.viz.store import Store

hv.extension("bokeh")


@pytest.fixture
def app(viz_store):
    root, yaml = viz_store
    return VizApp(Store(root), WorkingProblem.from_yaml(yaml), poll_seconds=0)


def test_every_tab_of_every_version_renders(app):
    for version in app.store.versions():
        app.version_pick.value = version
        for tab in (app.versions, app.multistart, app.model, app.fit):
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
from petab_mcp.viz.key import Key
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


def test_a_failed_starts_exit_is_explained_though_it_is_not_plotted(viz_store):
    """Fujita v0's two NOT_FINITE starts were counted in the summary but missing from the legend,
    which read only the waterfall's points, and a failed start has no objective to plot."""

    root, yaml = viz_store
    opt_path = root / "versions" / "v0" / "optimization.json"
    opt = json.loads(opt_path.read_text())
    opt["starts"][2]["fval"] = None  # the start the server stopped (pypesto_result.hdf5)
    opt_path.write_text(json.dumps(opt))
    app = VizApp(Store(root), WorkingProblem.from_yaml(yaml), poll_seconds=0)
    view = app.multistart._version(0)
    summary = " ".join(str(o.object) for o in view.select(pn.pane.Markdown))
    legend = " ".join(e["label"] + e["about"] for k in view.select(Key) for e in k.entries)
    assert "1 failed" in summary
    assert "stopped by the server" in legend and "no objective, so they are not drawn" in legend


def test_a_harness_adds_a_tab_that_follows_the_version(viz_store):
    """gemot's explorer is the viewer plus the agents' conversation, a tab only it can build."""
    root, yaml = viz_store
    seen = []

    def conversation(viz):
        return pn.bind(
            lambda version: seen.append(version) or f"turns of v{version}", viz.param.version
        )

    app = VizApp(
        Store(root),
        WorkingProblem.from_yaml(yaml),
        poll_seconds=0,
        extra_tabs=[("Conversation", conversation)],
    )
    page = app.view()
    page.server_doc(Document())
    app.version_pick.value = 0
    assert seen[-1] == 0


def test_only_clusters_many_starts_returned_to_get_a_colour():
    """A waterfall cluster is a run of starts each within AT_BEST_NLL_TOL of the one before; one
    or two starts on a step are not a fit the optimiser keeps finding, so they stay grey."""
    from petab_mcp.viz.multistart_tab import OTHERS, cluster_colors, clusters

    fval = [0.0, 0.4, 0.9, 1.5, 5.0, 5.2, 9.0, 9.1, 9.3, 20.0]
    found = pd.Series(clusters(fval))
    assert list(found) == [1, 1, 1, 1, 2, 2, 3, 3, 3, 4]
    colours = cluster_colors(found)
    assert set(colours) == {1, 3} and OTHERS not in colours.values()
    # Among 300 starts a cluster needs 2 in 100 of them, 6, except the best, which needs 3.
    many = pd.Series([1] * 3 + [2] * 5 + [3] * 6 + [4] * 286)
    assert set(cluster_colors(many)) == {1, 3, 4}


def test_the_exit_names_are_fides_own():
    """Written out so the page need not import fides (half a second of its first load)."""
    from fides import ExitFlag

    from petab_mcp.viz.multistart_tab import EXIT_NAMES

    assert EXIT_NAMES == {f.value: f.name for f in ExitFlag}


def test_versions_branch_onto_lanes_and_a_versions_line_runs_back_to_the_first():
    """A change from an earlier version (base_version) is a fork: its own lane, so the tree shows
    where the session went back."""
    from petab_mcp.viz import versions_tab as T

    table = pd.DataFrame(
        {
            "version": [0, 1, 2, 3, 4],
            "parent": [-1, 0, 1, 1, 0],
            "criterion_value": [10.0, 8.0, 9.0, 7.0, 12.0],
            "justification": [""] * 5,
            "kind": ["fit"] * 5,
            "k": [2] * 5,
            "n_starts": [10] * 5,
            "criterion": ["bic"] * 5,
        }
    )
    assert T.lanes(table) == {0: 0, 1: 0, 2: 0, 3: 1, 4: 2}
    # a lane whose line ended before the fork is reused
    reuse = pd.DataFrame({"version": [0, 1, 2, 3, 4, 5], "parent": [-1, 0, 1, 0, 3, 4]})
    assert T.lanes(reuse) == {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1}
    later = pd.DataFrame({"version": [0, 1, 2, 3, 4], "parent": [-1, 0, 0, 2, 2]})
    assert T.lanes(later)[4] == 0  # lane 0's line ended at v1, before v4's fork from v2
    assert T.lineage(table, 3) == [3, 1, 0]
    assert list(T.frame(table)["change"]) == pytest.approx(
        [float("nan"), -2.0, 1.0, -1.0, 2.0], nan_ok=True
    )


def test_the_versions_table_and_tree_carry_what_each_is_for():
    """The table names the version, its parent, k and why it was made; the tree's hover the score,
    never the justification, which is the table's."""
    from petab_mcp.viz import versions_tab as T

    table = pd.DataFrame(
        {
            "version": [0, 1, 2],
            "parent": [-1, 0, 0],
            "criterion_value": [10.0, 8.0, 9.5],
            "justification": ["start", "add a feedback", "try a delay"],
            "kind": ["fit"] * 3,
            "k": [2, 3, 3],
            "n_starts": [10] * 3,
            "criterion": ["bic"] * 3,
        }
    )
    drawn = T.tree(T.frame(table), best=1, head=2, picked=2)
    assert set(drawn["rows"][0]) == {"version", "parent", "k", "justification"}
    assert drawn["rows"][2] == {"version": 2, "parent": 0, "k": 3, "justification": "try a delay"}
    tips = [n["tip"] for n in drawn["nodes"]]
    assert tips[2] == "v2 · bic 9.5 · -0.5 on v0 · k 3"
    assert not any("delay" in t for t in tips)
    assert [n["on_path"] for n in drawn["nodes"]] == [True, False, True]
    assert [(e["from"], e["to"], e["on_path"]) for e in drawn["edges"]] == [
        (0, 1, False),
        (0, 2, True),
    ]
