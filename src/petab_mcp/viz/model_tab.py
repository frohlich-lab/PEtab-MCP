"""The Model tab: one graph whose detail follows the zoom, and the equations of what is clicked.

The levels (:data:`petab_mcp.viz.model_math.LEVELS`), coarse to fine:

* **influence**: who moves whom and which way. Agents and the signed edges their statements make
  for a model built from statements; species and the sign of each ODE's derivative otherwise;
* **statements** (a model built from statements only): each row a node between its agents;
* **reactions**: species and reactions, each reaction joined to what it consumes, makes and reads.

Observables hang off what they read at every level. All levels share one layout, computed on the
finest graph (of the version and the one compared with, together), and a coarse node sits at the
centroid of what it stands for: zooming in splits it where it stood instead of moving the picture.
With the level on *Auto* the visible share of the graph's width picks it; the buttons fix one.

Clicking a node fills the equation pane: an agent's species' ODEs, a row's statement, binding and
reactions, a reaction's rate, a species' ODE and initial value, an observable's formula and noise.
With nothing clicked it shows the whole model. *Fitted values* puts the version's parameter table
values in place of the names. Against the compared version, new nodes are ringed green, nodes
that are only in the other one are drawn faint, and a row or reaction whose text changed is ringed
orange.
"""

from __future__ import annotations

import math

import holoviews as hv
import numpy as np
import pandas as pd
import panel as pn

from . import model_math as MM
from . import style

#: Visible share of the graph's width above which Auto shows the coarsest level, and above which it
#: shows statements (where the model has them) rather than reactions.
COARSE, MIDDLE = 0.6, 0.3

#: Nodes in view up to which every one is labelled, and reactions up to which each shows its rate.
LABELLED, RATES = 80, 8

#: Where along an edge its arrowhead sits.
HEAD_AT = 0.82

_DIRECTED = {
    "activates",
    "inhibits",
    "modulates",
    "modifies",
    "converts",
    "reads",
    "consumes",
    "produces",
    "role",
}


def pick_level(available: list[str], share: float) -> str:
    """The level Auto shows when ``share`` of the graph's width is in view."""
    if share > COARSE or len(available) == 1:
        return available[0]
    if "statements" in available and share > MIDDLE:
        return "statements"
    return available[-1]


def _segments(nodes: pd.DataFrame, edges: pd.DataFrame) -> pd.DataFrame:
    xy = nodes.set_index("key")[["x", "y"]]
    df = edges.join(xy, on="source").join(xy, on="target", rsuffix="1")
    df = df.rename(columns={"x": "x0", "y": "y0"}).dropna(subset=["x0", "x1"])
    return df[(df["x0"] != df["x1"]) | (df["y0"] != df["y1"])]


def _heads(segments: pd.DataFrame) -> pd.DataFrame:
    df = segments[segments["kind"].isin(_DIRECTED)].copy()
    dx, dy = df["x1"] - df["x0"], df["y1"] - df["y0"]
    df["x"] = df["x0"] + HEAD_AT * dx
    df["y"] = df["y0"] + HEAD_AT * dy
    df["angle"] = np.arctan2(dy, dx) - math.pi / 2
    return df


def draw(
    graph: MM.Graph,
    positions: dict[str, tuple[float, float]],
    other: MM.Graph | None,
    view: tuple[tuple[float, float], tuple[float, float]] | None,
) -> tuple[hv.Overlay, pd.DataFrame]:
    """The ``graph`` (``other``: the compared version's graph at this level) as an overlay, and its
    placed nodes. ``view`` is the visible x and y range, which decides what is labelled."""
    nodes = MM.place(graph, positions)
    nodes["status"] = "same"
    ghosts = pd.DataFrame(columns=nodes.columns)
    if other is not None:
        theirs = MM.place(other, positions)
        info = dict(zip(theirs["key"], theirs["info"]))
        nodes.loc[~nodes["key"].isin(info), "status"] = "added"
        changed = (
            nodes["key"].isin(info)
            & nodes["kind"].isin(["row", "reaction"])
            & (nodes["info"] != nodes["key"].map(info))
        )
        nodes.loc[changed, "status"] = "changed"
        ghosts = theirs[~theirs["key"].isin(set(nodes["key"]))].assign(status="removed")
    segments = _segments(nodes, graph.edges)
    layers: list = []
    if other is not None and not ghosts.empty:
        gone = _segments(pd.concat([nodes, ghosts]), other.edges)
        gone = gone[gone["source"].isin(ghosts["key"]) | gone["target"].isin(ghosts["key"])]
        if not gone.empty:
            layers.append(
                hv.Segments(gone, ["x0", "y0", "x1", "y1"]).opts(
                    color=style.REMOVED, line_dash="dotted", line_width=1
                )
            )
    for kind, sub in segments.groupby("kind"):
        color, dash = style.EDGES.get(kind, (style.MUTED, "solid"))
        layers.append(
            hv.Segments(sub, ["x0", "y0", "x1", "y1"], ["kind"], label=kind).opts(
                color=color,
                line_dash=dash,
                line_width=1.6 if kind in ("activates", "inhibits") else 1.1,
                alpha=0.85,
            )
        )
    heads = _heads(segments)
    for kind, sub in heads.groupby("kind"):
        color, _ = style.EDGES.get(kind, (style.MUTED, "solid"))
        layers.append(
            hv.Points(sub, ["x", "y"], ["angle"]).opts(
                marker="dash" if kind == "inhibits" else "triangle",
                angle=hv.dim("angle"),
                color=color,
                size=12 if kind == "inhibits" else 7,
                line_width=2,
            )
        )
    if not ghosts.empty:
        layers.append(
            hv.Points(ghosts, ["x", "y"], ["label", "kind", "info"]).opts(
                color=style.REMOVED, size=10, alpha=0.6, tools=["hover"]
            )
        )
    for kind, sub in nodes.groupby("kind"):
        marker, color, size = style.NODES.get(kind, ("circle", style.MUTED, 10))
        ring = (
            sub["status"].map({"added": style.ADDED, "changed": style.PALETTE[1]}).fillna("white")
        )
        layers.append(
            hv.Points(
                sub.assign(ring=list(ring)),
                ["x", "y"],
                ["label", "kind", "info", "status", "ring"],
                label=kind,
            ).opts(
                marker=marker,
                color=color,
                size=size,
                line_color="ring",
                line_width=hv.dim("status").categorize({"same": 1}, default=3),
                tools=["hover"],
            )
        )
    shown = nodes
    if view is not None and None not in (view[0][0], view[1][0]):
        (x0, x1), (y0, y1) = view
        shown = nodes[nodes["x"].between(x0, x1) & nodes["y"].between(y0, y1)]
    if len(shown) <= LABELLED:
        labels = shown.copy()
        reactions = labels["kind"] == "reaction"
        if reactions.sum() <= RATES:
            labels.loc[reactions, "label"] = [
                f"{label}: {info[:48]}"
                for label, info in zip(
                    labels.loc[reactions, "label"], labels.loc[reactions, "info"]
                )
            ]
        layers.append(
            hv.Labels(labels, ["x", "y"], "label").opts(text_font_size="8pt", yoffset=0.12)
        )
    return hv.Overlay(layers), nodes


def legend(levels: dict[str, MM.Graph]) -> str:
    """The edge kinds the model's levels draw, and the diff's marks."""
    present = {k for graph in levels.values() for k in graph.edges["kind"]}
    present |= {"consumes"} if {"consumes", "produces"} & present else set()
    kinds = [k for k in (*MM.EDGE_KINDS, "consumes") if k in present and k in style.EDGES]
    spans = " ".join(
        f"<span style='color:{style.EDGES[k][0]}'>━ {'consumes / produces' if k == 'consumes' else k}</span>"
        for k in kinds
    )
    kinds_present = {k for graph in levels.values() for k in graph.nodes["kind"]}
    shapes = {
        "agent": "● agent",
        "species": "● species",
        "row": "■ statement",
        "reaction": "▪ reaction",
        "observable": "◆ observable",
    }
    nodes = " ".join(
        f"<span style='color:{style.NODES[k][1]}'>{label}</span>"
        for k, label in shapes.items()
        if k in kinds_present
    )
    return (
        f"{nodes} &nbsp;·&nbsp; {spans} &nbsp;·&nbsp; "
        f"<span style='color:{style.ADDED}'>◯ new</span> "
        f"<span style='color:{style.PALETTE[1]}'>◯ changed</span> "
        f"<span style='color:{style.REMOVED}'>● only in the compared version</span>"
    )


#: How to drive the graph, above it.
HINT = (
    "Scroll to zoom: the graph shows more detail the further in you go ({levels}); the buttons "
    "fix one level. Click a node for its equations."
)


class ModelTab:
    """The tab, reading the app's ``version`` and ``compare``."""

    def __init__(self, app) -> None:
        self.app = app
        self.level = pn.widgets.RadioButtonGroup(
            options=["Auto"], value="Auto", button_type="light"
        )
        self.fitted = pn.widgets.Checkbox(name="fitted values", value=False)
        self._math: dict[int, MM.ModelMath] = {}
        self._levels: dict[int, dict[str, MM.Graph]] = {}
        self._layouts: dict[tuple, dict] = {}

    # -- per version, built once -------------------------------------------------------------
    def math(self, version: int) -> MM.ModelMath | None:
        if version not in self._math:
            text = self.app.store.antimony(version)
            if text is None:
                return None
            self._math[version] = MM.from_antimony(
                text, self.app.problem.observable_df, self.app.store.statements(version)
            )
            self._levels[version] = MM.levels(self._math[version])
        return self._math[version]

    def positions(self, version: int, compare: int | None) -> dict[str, tuple[float, float]]:
        key = (version, compare)
        if key not in self._layouts:
            graphs = [self._levels[version]["reactions"]]
            if compare is not None and compare in self._levels:
                graphs.append(self._levels[compare]["reactions"])
            self._layouts[key] = MM.layout(graphs)
        return self._layouts[key]

    # -- the view ------------------------------------------------------------------------------
    def view(self) -> pn.viewable.Viewable:
        return pn.bind(self._version, self.app.param.version, self.app.param.compare)

    def _version(self, version, compare):
        if version is None:
            return pn.pane.Markdown("No version yet.")
        math_ = self.math(version)
        if math_ is None:
            return pn.pane.Markdown(f"v{version} stored no model.")
        if compare is not None and self.math(compare) is None:
            compare = None
        levels = self._levels[version]
        available = list(levels)
        self.level.options = ["Auto", *[name.capitalize() for name in available]]
        if self.level.value not in self.level.options:
            self.level.value = "Auto"
        positions = self.positions(version, compare)
        xs = [p[0] for p in positions.values()] or [0.0]
        ys = [p[1] for p in positions.values()] or [0.0]
        pad = 0.08 * max(max(xs) - min(xs), max(ys) - min(ys), 1.0)
        full = (min(xs) - pad, max(xs) + pad), (min(ys) - pad, max(ys) + pad)
        width = full[0][1] - full[0][0]
        drawn: dict[str, pd.DataFrame] = {}
        shown_level: dict[str, str] = {}

        def callback(x_range, y_range, level):
            share = 1.0 if x_range is None else (x_range[1] - x_range[0]) / width
            name = pick_level(available, share) if level == "Auto" else level.lower()
            other = self._levels.get(compare, {}).get(name) if compare is not None else None
            overlay, nodes = draw(
                levels[name], positions, other, (x_range, y_range) if x_range else None
            )
            drawn["nodes"], shown_level["name"] = nodes, name
            return overlay.opts(
                hv.opts.Overlay(
                    xlim=full[0],
                    ylim=full[1],
                    framewise=False,
                    xaxis=None,
                    yaxis=None,
                    responsive=True,
                    height=620,
                    tools=["tap"],
                    active_tools=["wheel_zoom"],
                    show_legend=False,
                    title=f"{name} level"
                    + (f" · against v{compare}" if compare is not None else ""),
                ),
            )

        graph = hv.DynamicMap(
            callback,
            streams=[
                hv.streams.RangeXY(),
                hv.streams.Params(self.level, ["value"], rename={"value": "level"}),
            ],
        )
        tap = hv.streams.Tap(source=graph, x=None, y=None)
        values = self._values(version)

        def pane(x, y, fitted):
            return self._equations(math_, drawn.get("nodes"), x, y, values if fitted else None)

        side = pn.Column(
            self.fitted,
            pn.bind(pane, tap.param.x, tap.param.y, self.fitted),
            width=420,
            scroll=True,
            max_height=720,
        )
        parts = [
            self.level,
            pn.pane.Markdown(
                HINT.format(levels=" → ".join(available)),
                sizing_mode="stretch_width",
                styles={"color": style.MUTED, "font-size": "12px"},
                margin=(0, 10),
            ),
            pn.pane.HTML(
                legend(levels),
                sizing_mode="stretch_width",
                styles={"font-size": "12px", "white-space": "normal"},
            ),
            pn.Row(
                pn.pane.HoloViews(
                    graph, linked_axes=False, sizing_mode="stretch_width", min_width=420, height=640
                ),
                side,
                sizing_mode="stretch_width",
            ),
        ]
        if math_.rows:
            parts.append(
                pn.Card(
                    pn.pane.Markdown(self._statements_text(version)),
                    title="Statements and bindings",
                    collapsed=True,
                    sizing_mode="stretch_width",
                )
            )
        return pn.Column(*parts, sizing_mode="stretch_width")

    def _values(self, version: int) -> dict[str, float]:
        table = self.app.store.parameters(version)
        if table is None or "nominalValue" not in table:
            return {}
        return {str(k): float(v) for k, v in table["nominalValue"].items() if pd.notna(v)}

    def _statements_text(self, version: int) -> str:
        record = self.app.store.statements(version) or {}
        lines = [MM.row_text(row) for row in record.get("rows") or []]
        for prefix, binding in (record.get("bindings") or {}).items():
            policy = binding.get("policy") if isinstance(binding, dict) else binding
            extra = (
                " ".join(f"{k}={v}" for k, v in (binding.get("parameters") or {}).items())
                if (isinstance(binding, dict))
                else ""
            )
            lines.append(f"binding {prefix}: {policy} {extra}".rstrip())
        return "```\n" + "\n".join(lines) + "\n```"

    def _equations(self, math_: MM.ModelMath, nodes: pd.DataFrame | None, x, y, values):
        node = None
        if x is not None and nodes is not None and not nodes.empty:
            d = (nodes["x"] - x) ** 2 + (nodes["y"] - y) ** 2
            i = d.idxmin()
            span = max(
                nodes["x"].max() - nodes["x"].min(), nodes["y"].max() - nodes["y"].min(), 1.0
            )
            if d[i] ** 0.5 < 0.06 * span:
                node = nodes.loc[i]
        header, kwargs = "**The whole model.** Click a node for its own equations.", {}
        if node is not None:
            kwargs, header = self._selection(math_, node)
        lines = MM.equations(math_, values=values, **kwargs)
        body = (r"$$\begin{aligned}" + r" \\ ".join(lines) + r"\end{aligned}$$") if lines else ""
        return pn.Column(
            pn.pane.Markdown(header),
            pn.pane.LaTeX(
                body,
                renderer="katex",
                sizing_mode="stretch_width",
                styles={"font-size": "13px", "overflow-x": "auto"},
            ),
        )

    def _selection(self, math_: MM.ModelMath, node) -> tuple[dict, str]:
        key, kind = node["key"], node["kind"]
        by_species_key = {sp.key: sid for sid, sp in math_.species.items()}
        by_reaction_key = {rx.key: rx.id for rx in math_.reactions}
        name = key.split(":", 1)[1]
        if kind == "species":
            return {"species": [by_species_key[name]]}, f"**Species** `{node['label']}`"
        if kind == "reaction":
            return {"reactions": [by_reaction_key[name]]}, f"**Reaction** `{node['label']}`"
        if kind == "observable":
            return {"observables": [name]}, f"**Observable** `{name}`"
        if kind == "agent":
            species = [
                by_species_key[m.split(":", 1)[1]]
                for m in node["members"]
                if m.split(":", 1)[1] in by_species_key
            ]
            return {
                "species": species
            }, f"**Agent** `{name}`: the ODEs of its {len(species)} species"
        row = next((r for r in math_.rows if str(r.get("prefix")) == name), {})
        reactions = [
            by_reaction_key[m.split(":", 1)[1]]
            for m in node["members"]
            if m.split(":", 1)[1] in by_reaction_key
        ]
        english = MM.row_english(row)
        binding = math_.bindings.get(name)
        policy = binding.get("policy") if isinstance(binding, dict) else binding
        text = (
            f"**Row** `{MM.row_text(row)}`"
            + (f"\n\n{english}" if english else "")
            + (f"\n\nbinding: `{policy}`" if policy else "")
        )
        return {"reactions": reactions, "species": []}, text
