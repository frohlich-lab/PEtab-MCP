"""The app: a sidebar to pick a version and what to compare it with, and the three tabs.

The sidebar's timeline plots each version's criterion value, the best ringed and HEAD marked, each
joined to its parent; clicking one picks it. *Compare with* defaults to the version's parent and
feeds the Model tab's diff and the Fit tab's dashed line. *Follow HEAD* moves the pick to each new
version as the server writes it.

Each browser session gets its own :class:`VizApp`, which polls the store (:meth:`Store.signature`)
every ``poll_seconds`` and redraws what changed: a new version, or the running fit's progress.

A harness adds what it alone knows as ``extra_tabs`` (gemot, the agents' conversation): each a
title and a function of the app, whose ``version`` and ``tick`` its view can follow.

Only the tab shown is computed: the others are ``lazy`` panes, built when first opened. Built with
the page, as ``pn.Tabs(dynamic=True)`` alone leaves them, Fujita's Model and Fit tabs added 0.7 s to
a page that showed neither, and a harness's conversation tab a Phoenix fetch.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

import holoviews as hv
import pandas as pd
import panel as pn
import param

from ..problem import WorkingProblem
from . import key as K
from . import style
from .fit_tab import FitTab
from .model_tab import ModelTab
from .multistart_tab import MultistartTab
from .store import Store
from .versions_tab import VersionsTab

#: The *Compare with* choice that follows the picked version's parent.
PARENT = "parent"
NONE = "nothing"

TIMELINE = """
Each version's score ({criterion}, lower is better) against its number, in the order the agent
made them. A grey line joins each version to the one it was edited from, so a step down is a change
that helped and a step up one that did not.
"""


class VizApp(param.Parameterized):
    version = param.Integer(default=None, allow_None=True)
    compare = param.Integer(default=None, allow_None=True)
    #: Bumped whenever the store changes, for views that read what is not a version (the fit).
    tick = param.Integer(default=0)

    def __init__(
        self,
        store: Store,
        problem: WorkingProblem,
        poll_seconds: float = 3.0,
        title: str = "PEtab-MCP",
        extra_tabs: Sequence[tuple[str, Callable[[VizApp], pn.viewable.Viewable]]] = (),
        inputs: Sequence = (),
    ) -> None:
        super().__init__()
        self.store, self.problem, self.title = store, problem, title
        self.extra_tabs = list(extra_tabs)
        #: The problem's ``inputs.tsv`` protocols (:mod:`petab_mcp.protocols`), for the Fit tab.
        self.inputs = tuple(inputs)
        self.poll_seconds = poll_seconds
        self._signature = store.signature()
        self.version_pick = pn.widgets.Select(name="Version", sizing_mode="stretch_width")
        self.compare_pick = pn.widgets.Select(name="Compare with", sizing_mode="stretch_width")
        self.follow = pn.widgets.Checkbox(name="Follow HEAD", value=True)
        self.version_pick.param.watch(self._picked, "value")
        self.compare_pick.param.watch(self._compare_picked, "value")
        self._refresh_choices(move=True)
        self.versions = VersionsTab(self)
        self.multistart, self.model, self.fit = MultistartTab(self), ModelTab(self), FitTab(self)

    # -- choices -------------------------------------------------------------------------------
    def _label(self, row) -> str:
        score = (
            "" if pd.isna(row.criterion_value) else f" · {row.criterion} {row.criterion_value:.4g}"
        )
        return f"v{row.version} · {row.kind}{score}"

    def _refresh_choices(self, move: bool) -> None:
        table = self.store.table()
        options = {self._label(r): int(r.version) for r in table.itertuples()}
        self.version_pick.options = options
        head = self.store.head()
        if move or self.version is None:
            self.version_pick.value = (
                head if head in options.values() else (next(iter(options.values()), None))
            )
        compare = {"its parent": PARENT, "nothing": NONE, **options}
        current = self.compare_pick.value
        self.compare_pick.options = compare
        self.compare_pick.value = current if current in compare.values() else PARENT
        self._compare_picked()

    def _picked(self, event) -> None:
        self.version = event.new
        self._compare_picked()
        # A pick of any version but HEAD is a pick to stay on: the next fit must not move it.
        if event.new != self.store.head():
            self.follow.value = False

    def _compare_picked(self, *_) -> None:
        choice = self.compare_pick.value
        if choice == NONE or self.version is None:
            self.compare = None
        elif choice == PARENT:
            parent = self.store.meta(self.version).get("parent")
            self.compare = (
                parent
                if isinstance(parent, int) and parent >= 0 and parent in set(self.store.versions())
                else None
            )
        else:
            self.compare = None if choice == self.version else choice

    def poll(self) -> None:
        signature = self.store.signature()
        if signature == self._signature:
            return
        new_version = signature[-1] != self._signature[-1] or signature[0] != self._signature[0]
        self._signature = signature
        if new_version:
            self._refresh_choices(move=self.follow.value)
        self.tick += 1

    # -- sidebar -------------------------------------------------------------------------------
    def timeline(self) -> pn.viewable.Viewable:
        def plot(_tick, version):
            df = self.store.table().dropna(subset=["criterion_value"])
            if df.empty:
                return pn.pane.Markdown("No version is scored yet.")
            best, head = self.store.best(), self.store.head()
            df = df.assign(picked=df["version"] == version)
            links = [
                ((p, df.set_index("version").at[p, "criterion_value"]), (v, c))
                for v, p, c in zip(df["version"], df["parent"], df["criterion_value"])
                if isinstance(p, (int, float)) and p in set(df["version"])
            ]
            path = hv.Path([list(link) for link in links]).opts(color="#cccccc", line_width=1)
            points = hv.Points(
                df,
                ["version", "criterion_value"],
                ["kind", "justification", "is_new_best", "picked"],
            ).opts(
                color=hv.dim("picked").categorize(
                    {True: style.PALETTE[3]}, default=style.PALETTE[0]
                ),
                size=9,
                tools=["hover", "tap"],
                line_color="white",
            )
            marks = []
            if best is not None and best in set(df["version"]):
                b = df[df["version"] == best]
                marks.append(
                    hv.Points(b, ["version", "criterion_value"]).opts(
                        marker="circle",
                        fill_alpha=0,
                        line_color=style.PALETTE[2],
                        line_width=2,
                        size=17,
                    )
                )
            tap = hv.streams.Tap(source=points)
            tap.add_subscriber(lambda x, y: self._tapped(x))
            criterion = df["criterion"].iloc[-1]
            overlay = (path * points * hv.Overlay(marks)).opts(
                height=230,
                responsive=True,
                xlabel="version",
                ylabel=criterion,
                toolbar=None,
            )
            key = K.key(
                f"HEAD v{head}" + (f" · best v{best}" if best is not None else ""),
                TIMELINE.format(criterion=criterion),
                [
                    K.entry("a version", "Click one to show it in every tab.", style.PALETTE[0]),
                    K.entry("shown", "The version the tabs show.", style.PALETTE[3]),
                    K.entry(
                        "the best", "The version the server judged best.", style.PALETTE[2], "◯"
                    ),
                ],
            )
            return pn.Column(
                key,
                pn.pane.HoloViews(overlay, linked_axes=False, sizing_mode="stretch_width"),
                sizing_mode="stretch_width",
            )

        return pn.panel(pn.bind(plot, self.param.tick, self.param.version))

    def _tapped(self, x) -> None:
        if x is None:
            return
        versions = self.store.versions()
        if versions:
            self.version_pick.value = min(versions, key=lambda v: abs(v - x))

    def about(self) -> pn.viewable.Viewable:
        def text(version, compare, _tick):
            if version is None:
                return "No version."
            meta = self.store.meta(version)
            why = (
                self.store.table()
                .set_index("version")
                .get("justification", pd.Series())
                .get(version)
            )
            lines = [
                f"**v{version}** · {meta.get('kind', '?')} · parent "
                f"{meta.get('parent') if meta.get('parent', -1) >= 0 else '—'}"
            ]
            if meta.get("criterion_value") is not None:
                lines.append(
                    f"{meta.get('criterion', 'score')} {meta['criterion_value']:.6g} · "
                    f"nll {meta.get('neg_log_likelihood', float('nan')):.6g} · "
                    f"k {meta.get('k', '?')}"
                )
            if why:
                lines.append(f"> {why}")
            if compare is not None:
                lines.append(f"compared with v{compare}")
            if self.store.fit_progress():
                lines.append(f"<span style='color:{style.PALETTE[1]}'>● a fit is running</span>")
            return "\n\n".join(lines)

        return pn.pane.Markdown(
            pn.bind(text, self.param.version, self.param.compare, self.param.tick),
            sizing_mode="stretch_width",
        )

    # -- page ----------------------------------------------------------------------------------
    def view(self) -> pn.template.FastListTemplate:
        # A spinner over whatever is recomputing: a large model's first layout takes seconds.
        pn.config.loading_indicator = True
        tabs = pn.Tabs(
            ("Versions", self.versions.view()),
            ("Multistart", self.multistart.view()),
            ("Model", self.model.view()),
            ("Fit", self.fit.view()),
            *[(name, view(self)) for name, view in self.extra_tabs],
            dynamic=True,
            sizing_mode="stretch_width",
        )
        template = pn.template.FastListTemplate(
            title=self.title,
            sidebar=[
                self.version_pick,
                self.compare_pick,
                self.follow,
                self.timeline(),
                self.about(),
            ],
            main=[tabs],
            sidebar_width=330,
            accent=style.PALETTE[0],
        )
        if self.poll_seconds:
            pn.state.add_periodic_callback(self.poll, period=int(self.poll_seconds * 1000))
        return template


def build(
    store_root: str | Path,
    petab_yaml: str | Path,
    poll_seconds: float = 3.0,
    extra_tabs: Sequence[tuple[str, Callable[[VizApp], pn.viewable.Viewable]]] = (),
    title: str | None = None,
) -> pn.viewable.Viewable:
    """The page for the store at ``store_root`` of the problem at ``petab_yaml``, for ``pn.serve``
    or a notebook."""
    from ..protocols import ProtocolError, read

    hv.extension("bokeh")
    store = Store(store_root)
    problem = WorkingProblem.from_yaml(petab_yaml)
    try:
        inputs = read(Path(petab_yaml).parent)
    except ProtocolError:  # the server refuses such a problem; the viewer shows it without them
        inputs = ()
    title = title or f"PEtab-MCP · {problem_name(petab_yaml)}"
    return VizApp(
        store, problem, poll_seconds, title=title, extra_tabs=extra_tabs, inputs=inputs
    ).view()


def problem_name(petab_yaml: str | Path) -> str:
    """What the page calls the problem: its YAML's name, or, for the ``problem.yaml`` every gemot
    example and run has, the directory that names the example (``fujita/problem/problem.yaml`` is
    fujita, ``fujita/work/problem.yaml`` too)."""
    path = Path(petab_yaml).resolve()
    if path.stem != "problem":
        return path.stem
    folder = path.parent
    return folder.parent.name if folder.name in ("problem", "work") else folder.name
