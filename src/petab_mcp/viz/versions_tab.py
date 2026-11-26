"""The Versions tab: a session's versions as a tree, each joined to the version it was made from.

A change is made from HEAD or, with ``base_version``, from an earlier version, so a session's
versions branch wherever a change went back. The tree is drawn as a commit graph: versions left to
right in the order they were made, a version on its parent's lane when it is that parent's first
child and on a new lane below otherwise, so each lane is a line of work and a new lane a fork. A
node's colour is its score (the session's criterion, lower is better); the best version is ringed,
HEAD marked, and the picked version's line back to the first version drawn bold.

Under the tree, every version in a table: its parent, k and the justification its change was made
with. Hovering a node or a row highlights both, and clicking either picks that version for every
tab. The two are one component drawn in the browser (:class:`VersionTree`): neither Bokeh nor
Panel's table reports a hover to the other, and a round trip through the server per mouse move
would lag behind the pointer.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import panel as pn
import param

from . import key as K
from . import style

VERSIONS = """
Every model the agent made in this session, as a family tree. Each dot is a version, numbered in
the order it was made, left to right; a line joins it to the version it was edited from. When the
agent went back to an earlier version and edited that instead, the new branch starts a row of its
own, so each row is a line of work.

Colour is each version's score: lower is better. The score rewards fitting the data and charges
for every parameter, so a version that fits no better with more parameters scores worse.

The table below lists the same versions with the agent's reason for each change. Hover a dot or a
row to find the other; click either to show that version in every tab.
"""


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


#: Viridis, sampled: a version's colour runs from its lowest score (yellow) to its highest (purple).
_VIRIDIS = ("#fde725", "#5ec962", "#21918c", "#3b528b", "#440154")


def color(score: float | None, low: float, high: float) -> str:
    """``score``'s colour between ``low`` and ``high``; grey without one."""
    if score is None or not math.isfinite(score):
        return style.MUTED
    f = 0.0 if high <= low else min(max((score - low) / (high - low), 0.0), 1.0)
    i = min(int(f * (len(_VIRIDIS) - 1)), len(_VIRIDIS) - 2)
    a, b = _VIRIDIS[i], _VIRIDIS[i + 1]
    t = f * (len(_VIRIDIS) - 1) - i
    mix = [round(int(a[k : k + 2], 16) * (1 - t) + int(b[k : k + 2], 16) * t) for k in (1, 3, 5)]
    return "#" + "".join(f"{c:02x}" for c in mix)


def _num(value) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def tree(df: pd.DataFrame, best: int | None, head: int | None, picked: int | None) -> dict:
    """What :class:`VersionTree` draws: a node per version (its lane, colour, marks and hover
    text), an edge per parent, and a table row per version."""
    path = set(lineage(df, picked))
    scores = [s for s in map(_num, df["criterion_value"]) if s is not None]
    low, high = (min(scores), max(scores)) if scores else (0.0, 0.0)
    criterion = str(df["criterion"].dropna().iloc[-1]) if df["criterion"].notna().any() else "score"
    nodes, rows = [], []
    parents = [_parent(_num(p)) for p in df["parent_version"]]  # a missing parent is NaN here
    for r, parent in zip(df.itertuples(), parents):
        score, change = _num(r.criterion_value), _num(r.change)
        tip = [f"v{r.version}"]
        if score is not None:
            tip.append(f"{criterion} {score:.4g}")
        if change is not None:
            tip.append(f"{change:+.3g} on v{parent}")
        if _num(r.k) is not None:
            tip.append(f"k {int(r.k)}")
        nodes.append(
            {
                "version": int(r.version),
                "lane": int(r.lane),
                "color": color(score, low, high),
                "best": r.version == best,
                "head": r.version == head,
                "picked": r.version == picked,
                "on_path": r.version in path,
                "tip": " · ".join(tip),
            }
        )
        rows.append(
            {
                "version": int(r.version),
                "parent": "" if parent is None else parent,
                "k": "" if _num(r.k) is None else int(r.k),
                "justification": r.justification if isinstance(r.justification, str) else "",
            }
        )
    known = set(df["version"])
    edges = [
        {"from": p, "to": int(v), "on_path": v in path and p in path}
        for v, p in zip(df["version"], parents)
        if p is not None and p in known
    ]
    return {"nodes": nodes, "edges": edges, "rows": rows, "legend": legend(criterion, scores)}


def legend(criterion: str, scores: list[float]) -> list[dict]:
    """The tree's legend entries."""
    about = (
        f"Each version's {criterion}, lower is better: yellow the lowest, purple the highest, "
        "grey a version with no score (a refused change or a failed fit)."
    )
    out = []
    if scores:
        out += [
            K.entry(f"{criterion} {min(scores):.4g}, the lowest", about, _VIRIDIS[0]),
            K.entry(f"{criterion} {max(scores):.4g}, the highest", about, _VIRIDIS[-1]),
        ]
    return out + [
        K.entry(
            "the best",
            "Ringed: the version the server judged best so far.",
            style.PALETTE[2],
            "◯",
        ),
        K.entry(
            "HEAD",
            "The version the agent is working on: its next change will be made from it.",
            style.PALETTE[1],
            "▴",
        ),
        K.entry(
            "the picked version's history",
            "Bold: the edits that led from the first version to the one shown in the other tabs.",
            style.PALETTE[3],
            "━",
        ),
    ]


class VersionTree(pn.custom.JSComponent):
    """The tree over the table, drawn in the browser; a hover on either highlights both, and a
    click sets :attr:`picked`."""

    nodes = param.List(default=[])
    edges = param.List(default=[])
    rows = param.List(default=[])
    picked = param.Integer(default=None, allow_None=True)

    _stylesheets = [
        """
    .vt { font-family: inherit; color: inherit; }
    .vt svg { display: block; width: 100%; }
    .vt .node { cursor: pointer; }
    .vt .node.hot circle.dot { stroke: #D55E00; stroke-width: 3; }
    .vt .tip { position: absolute; pointer-events: none; background: rgba(40,40,40,.92);
               color: #fff; font-size: 12px; padding: 3px 7px; border-radius: 4px;
               white-space: nowrap; display: none; z-index: 10; }
    .vt .rows { max-height: 420px; overflow-y: auto; margin-top: 8px;
                border-top: 1px solid rgba(128,128,128,.35); }
    .vt table { border-collapse: collapse; width: 100%; font-size: 13px; }
    .vt th { text-align: left; position: sticky; top: 0; background: var(--panel-background-color,
             var(--background-color, #fff)); padding: 4px 8px; font-weight: 600; }
    .vt td { padding: 3px 8px; vertical-align: top; border-top: 1px solid rgba(128,128,128,.2); }
    .vt td.num { white-space: nowrap; width: 1%; }
    .vt tr { cursor: pointer; }
    .vt tr.hot td { background: rgba(213,94,0,.16); }
    .vt tr.picked td { font-weight: 600; }
    """
    ]

    _esm = """
    const NS = "http://www.w3.org/2000/svg";
    function svg(tag, attrs) {
      const e = document.createElementNS(NS, tag);
      for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
      return e;
    }
    export function render({ model, el }) {
      const root = document.createElement("div"); root.className = "vt";
      root.style.position = "relative";
      const plot = svg("svg", {});
      const tip = document.createElement("div"); tip.className = "tip";
      const scroll = document.createElement("div"); scroll.className = "rows";
      root.append(plot, tip, scroll);
      let marks = {}, rowsBy = {};

      function hot(v, on, fromTable) {
        if (marks[v]) marks[v].classList.toggle("hot", on);
        if (rowsBy[v]) {
          rowsBy[v].classList.toggle("hot", on);
          if (on && !fromTable) {  // the table's own scroll only: scrollIntoView moved the page
            const row = rowsBy[v], head = row.closest("table").tHead.offsetHeight;
            if (row.offsetTop - head < scroll.scrollTop) scroll.scrollTop = row.offsetTop - head;
            else if (row.offsetTop + row.offsetHeight > scroll.scrollTop + scroll.clientHeight)
              scroll.scrollTop = row.offsetTop + row.offsetHeight - scroll.clientHeight;
          }
        }
        if (on && marks[v]) {
          const c = marks[v].querySelector("circle.dot");
          const box = root.getBoundingClientRect(), r = c.getBoundingClientRect();
          tip.textContent = marks[v].dataset.tip;
          tip.style.display = "block";
          const right = r.left - box.left + 14;  // left of the node where it would overflow
          tip.style.left = (right + tip.offsetWidth > box.width
            ? r.left - box.left - tip.offsetWidth - 6 : right) + "px";
          tip.style.top = (r.top - box.top - 26) + "px";
        } else tip.style.display = "none";
      }

      function draw() {
        const nodes = model.nodes, width = Math.max(root.clientWidth || el.clientWidth || 800, 300);
        const lanes = Math.max(1, ...nodes.map(n => n.lane + 1));
        const maxv = Math.max(1, ...nodes.map(n => n.version));
        const left = 24, right = 24, top = 26, step = 46, height = top + lanes * step;
        const x = v => left + (width - left - right) * v / maxv, y = l => top + l * step;
        plot.innerHTML = ""; marks = {};
        plot.setAttribute("viewBox", `0 0 ${width} ${height}`);
        plot.setAttribute("height", height);
        const at = {}; nodes.forEach(n => at[n.version] = n);
        for (const e of model.edges) {
          const a = at[e.from], b = at[e.to]; if (!a || !b) continue;
          const d = a.lane === b.lane ? `M${x(a.version)},${y(a.lane)} L${x(b.version)},${y(b.lane)}`
            : `M${x(a.version)},${y(a.lane)} C${x(a.version) + 18},${y(b.lane)} ` +
              `${x(b.version) - 18},${y(b.lane)} ${x(b.version)},${y(b.lane)}`;
          plot.append(svg("path", { d, fill: "none", stroke: e.on_path ? "#D55E00" : "#8c8c8c",
            "stroke-width": e.on_path ? 3 : 1.5, opacity: e.on_path ? 1 : 0.8 }));
        }
        for (const n of nodes) {
          const g = svg("g", { class: "node" }); g.dataset.tip = n.tip;
          const cx = x(n.version), cy = y(n.lane);
          if (n.best) g.append(svg("circle", { cx, cy, r: 12, fill: "none", stroke: "#009E73",
                                                 "stroke-width": 3 }));
          g.append(svg("circle", { class: "dot", cx, cy, r: 7, fill: n.color,
            stroke: n.picked ? "#D55E00" : "#fff", "stroke-width": n.picked ? 3 : 1.5 }));
          const label = svg("text", { x: cx, y: cy + 21, "text-anchor": "middle",
                                       "font-size": 10, fill: "currentColor", opacity: 0.75 });
          label.textContent = n.version; g.append(label);
          if (n.head) {
            const h = svg("text", { x: cx, y: cy - 13, "text-anchor": "middle", "font-size": 10,
                                     fill: "#E69F00", "font-weight": 600 });
            h.textContent = "HEAD"; g.append(h);
          }
          g.addEventListener("mouseenter", () => hot(n.version, true, false));
          g.addEventListener("mouseleave", () => hot(n.version, false, false));
          g.addEventListener("click", () => { model.picked = n.version; });
          plot.append(g); marks[n.version] = g;
        }
      }

      function table() {
        scroll.innerHTML = ""; rowsBy = {};
        const t = document.createElement("table");
        t.innerHTML = "<thead><tr><th>version</th><th>parent</th><th>k</th>" +
                      "<th>justification</th></tr></thead>";
        const body = document.createElement("tbody");
        const picked = new Set(model.nodes.filter(n => n.picked).map(n => n.version));
        for (const r of model.rows) {
          const tr = document.createElement("tr");
          if (picked.has(r.version)) tr.className = "picked";
          for (const [k, v] of [["num", r.version], ["num", r.parent], ["num", r.k],
                                ["", r.justification]]) {
            const td = document.createElement("td"); td.className = k; td.textContent = v;
            tr.append(td);
          }
          tr.addEventListener("mouseenter", () => hot(r.version, true, true));
          tr.addEventListener("mouseleave", () => hot(r.version, false, true));
          tr.addEventListener("click", () => { model.picked = r.version; });
          body.append(tr); rowsBy[r.version] = tr;
        }
        t.append(body); scroll.append(t);
      }

      model.on("nodes", () => { draw(); table(); });
      new ResizeObserver(draw).observe(root);
      setTimeout(() => { draw(); table(); }, 0);
      return root;
    }
    """


class VersionsTab:
    """The tab, reading the app's ``version`` and ``tick`` (a new version)."""

    def __init__(self, app) -> None:
        self.app = app
        self.component = VersionTree(sizing_mode="stretch_width")
        self.component.param.watch(self._picked, "picked")

    def _picked(self, event) -> None:
        if event.new is not None and event.new != self.app.version:
            self.app.version_pick.value = event.new

    def view(self) -> pn.viewable.Viewable:
        return pn.Column(
            pn.panel(pn.bind(self._update, self.app.param.version, self.app.param.tick)),
            self.component,
            sizing_mode="stretch_width",
        )

    def _update(self, version, _tick):
        """Redraw the component for ``version``; the component itself is the view."""
        store = self.app.store
        table = store.table()
        if table.empty:
            return pn.pane.Markdown("No version yet.")
        drawn = tree(frame(table), store.best(), store.head(), version)
        entries = drawn.pop("legend")
        self.component.param.update(picked=version, **drawn)
        return K.key("Versions", VERSIONS, entries)
