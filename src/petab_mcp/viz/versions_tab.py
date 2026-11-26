"""The Versions tab: a session's versions as a tree, each joined to the version it was made from.

A change is made from HEAD or, with ``base_version``, from an earlier version, so a session's
versions branch wherever a change went back. The tree is drawn as a commit graph: versions left to
right in the order they were made, a version on its parent's lane when it is that parent's first
child and on a new lane below otherwise, so each lane is a line of work and a new lane a fork. A
node's colour is its score (the session's criterion, lower is better); the best version is ringed,
HEAD marked, and the picked version's line back to the first version drawn bold.

Under the tree, every version in a table: its parent, kind, score and the change from its parent's,
k, starts and the justification its change was made with. Clicking a node or a row picks that
version for every tab.
"""

from __future__ import annotations

import math

import holoviews as hv
import numpy as np
import pandas as pd
import panel as pn

from . import style


def _parent(value) -> int | None:
    try:
        parent = int(value)
    except (TypeError, ValueError):
        return None
    return parent if parent >= 0 else None


def lanes(table: pd.DataFrame) -> dict[int, int]:
    """Each version's lane: its parent's for the parent's first child; for any other child, or a
    version with no parent in the table, the lowest lane whose line ended before the parent, so a
    lane is reused once its line of work is over (a new lane each fork staircased Fujita's 28
    versions over a dozen lanes)."""
    known = set(table["version"])
    lane: dict[int, int] = {}
    last: dict[int, int] = {}  # each lane's latest version
    continued: set[int] = set()
    for version, parent in zip(table["version"], table["parent"]):
        parent = _parent(parent)
        if parent in known and parent in lane and parent not in continued:
            lane[version] = lane[parent]
            continued.add(parent)
        else:
            since = parent if parent in lane else version
            free = [n for n, v in last.items() if v < since and n != lane.get(parent)]
            lane[version] = min(free) if free else len(last)
        last[lane[version]] = version
    return lane


def lineage(table: pd.DataFrame, version: int | None) -> list[int]:
    """``version`` and its ancestors, back to the first."""
    parents = {v: _parent(p) for v, p in zip(table["version"], table["parent"])}
    out, seen = [], set()
    while version is not None and version in parents and version not in seen:
        out.append(version)
        seen.add(version)
        version = parents[version]
    return out


def frame(table: pd.DataFrame) -> pd.DataFrame:
    """The table with each version's lane, and its score's change from its parent's."""
    df = table.sort_values("version").reset_index(drop=True)
    lane = lanes(df)
    score = dict(zip(df["version"], df["criterion_value"]))
    df["lane"] = [lane[v] for v in df["version"]]
    df["parent_version"] = [_parent(p) for p in df["parent"]]
    df["change"] = [
        score[v] - score[p]
        if p in score
        and score[v] is not None
        and score[p] is not None
        and not (math.isnan(score[v]) or math.isnan(score[p]))
        else np.nan
        for v, p in zip(df["version"], df["parent_version"])
    ]
    df["why"] = [str(j)[:160] if isinstance(j, str) else "" for j in df["justification"]]
    return df


def tree(df: pd.DataFrame, best: int | None, head: int | None, picked: int | None) -> tuple:
    """The tree as an overlay, and its nodes, which a tap stream reads."""
    xy = {v: (v, -lane) for v, lane in zip(df["version"], df["lane"])}
    path = set(lineage(df, picked))
    edges = [
        (*xy[p], *xy[v], v in path and p in path)
        for v, p in zip(df["version"], df["parent_version"])
        if p in xy
    ]
    segments = pd.DataFrame(edges, columns=["x0", "y0", "x1", "y1", "on_path"])
    kdims = ["x0", "y0", "x1", "y1"]
    layers = []
    if not segments.empty:  # a first version alone has no edge
        on_path = segments["on_path"].astype(bool)
        layers += [
            hv.Segments(segments[~on_path], kdims).opts(color=style.MUTED, line_width=1.5),
            hv.Segments(segments[on_path], kdims).opts(color=style.PALETTE[3], line_width=3),
        ]
    nodes = df.assign(x=df["version"], y=-df["lane"], score=df["criterion_value"])
    points = hv.Points(
        nodes, ["x", "y"], ["version", "kind", "score", "change", "k", "n_starts", "why"]
    ).opts(
        color="score",
        cmap="viridis_r",
        colorbar=True,
        clabel=str(df["criterion"].dropna().iloc[-1]) if df["criterion"].notna().any() else "",
        size=13,
        line_color="white",
        tools=["hover", "tap"],
    )
    layers.append(points)
    if best in xy:
        layers.append(
            hv.Points([xy[best]]).opts(
                marker="circle", fill_alpha=0, line_color=style.ADDED, line_width=3, size=22
            )
        )
    if picked in xy:
        layers.append(
            hv.Points([xy[picked]]).opts(
                marker="circle", fill_alpha=0, line_color=style.PALETTE[3], line_width=3, size=19
            )
        )
    if head in xy:
        layers.append(
            hv.Labels([(*xy[head], "HEAD")], ["x", "y"], "text").opts(
                yoffset=0.35, text_font_size="9pt", text_color=style.PALETTE[1]
            )
        )
    n_lanes = int(df["lane"].max()) + 1 if len(df) else 1
    overlay = hv.Overlay(layers).opts(
        responsive=True,
        height=max(220, 60 * n_lanes + 120),
        yaxis=None,
        xlabel="version",
        ylim=(-n_lanes + 0.4, 0.8),
        title="Versions, each joined to the one it was made from (ringed: best; bold: the picked "
        "version's line)",
        show_legend=False,
        toolbar="above",
    )
    return overlay, points


class VersionsTab:
    """The tab, reading the app's ``version`` and ``tick`` (a new version)."""

    def __init__(self, app) -> None:
        self.app = app

    def view(self) -> pn.viewable.Viewable:
        return pn.panel(
            pn.bind(self._render, self.app.param.version, self.app.param.tick),
            lazy=True,
            sizing_mode="stretch_width",
        )

    def _render(self, version, _tick):
        store = self.app.store
        table = store.table()
        if table.empty:
            return pn.pane.Markdown("No version yet.")
        df = frame(table)
        overlay, points = tree(df, store.best(), store.head(), version)
        tap = hv.streams.Tap(source=points)

        def picked(x, y):
            if x is None:
                return
            d = (df["version"] - x) ** 2 + (-df["lane"] - y) ** 2
            self.app.version_pick.value = int(df.loc[d.idxmin(), "version"])

        tap.add_subscriber(picked)
        shown = df[
            [
                "version",
                "parent_version",
                "kind",
                "criterion_value",
                "change",
                "k",
                "n_starts",
                "why",
            ]
        ].rename(
            columns={
                "parent_version": "parent",
                "criterion_value": "score",
                "change": "change from parent",
                "n_starts": "starts",
                "why": "justification",
            }
        )
        shown = shown.astype({"parent": "Int64", "k": "Int64", "starts": "Int64"})
        rows = pn.widgets.Tabulator(
            shown.round({"score": 2, "change from parent": 2}),
            disabled=True,
            show_index=False,
            selectable=1,
            sizing_mode="stretch_width",
            height=min(420, 32 * len(shown) + 40),
            widths={"justification": 520},
        )

        def selected(event):
            if event.new:
                self.app.version_pick.value = int(shown["version"].iloc[event.new[0]])

        rows.param.watch(selected, "selection")
        return pn.Column(
            pn.pane.HoloViews(overlay, linked_axes=False, sizing_mode="stretch_width"),
            rows,
            sizing_mode="stretch_width",
        )
