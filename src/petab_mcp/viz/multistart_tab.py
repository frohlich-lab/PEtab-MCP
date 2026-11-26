"""The Multistart tab: how one version's fit went, start by start, and the running fit.

* a line of counts (starts, at the best, stopped, how they ended, how long);
* the waterfall, each start's final objective above the best, by rank, coloured by how it ended;
* each start's objective over its iterations or its wall time;
* where each start ended in each estimated parameter's bounds, on the estimation scale, so a
  parameter the best starts disagree on, or one at a bound, stands out.

Box-selecting starts in the waterfall picks them out in the other two. "At the best" is the
server's own count, within :data:`petab_mcp.constants.AT_BEST_NLL_TOL` of the lowest objective.
How each start ended is in the version's ``pypesto_result.hdf5``; a store without it (an archive)
shows the version's exit counts only.

While a change is being fitted its finished starts are drawn as they come in, from the store's
:data:`petab_mcp.provenance.FIT_PROGRESS`; nothing else about a running fit is on disk.
"""

from __future__ import annotations

import math
import time

import holoviews as hv
import numpy as np
import pandas as pd
import panel as pn

from ..constants import AT_BEST_NLL_TOL
from . import style

#: What a start the server stopped past its cap says (``multistart._killed``): its exit flag, -1,
#: is Fides' MAXITER, so the message is what tells the two apart.
STOPPED = "stopped by the server"

#: Each way a start ends, by Fides' ``ExitFlag`` name (as the version's ``exitflag_counts`` names
#: it), its own colour in every version: converged in blues and greens, stopped short in warm
#: colours.
EXIT_COLORS = {
    "FTOL": style.PALETTE[0],
    "GTOL": style.PALETTE[2],
    "XTOL": style.PALETTE[5],
    "DELTA_TOO_SMALL": style.PALETTE[1],
    "MAXTIME": style.PALETTE[3],
    "MAXITER": style.PALETTE[4],
    "NOT_FINITE": style.PALETTE[6],
    "EXCEEDED_BOUNDARY": style.PALETTE[6],
    STOPPED: style.PALETTE[7],
    "DID_NOT_RUN": style.MUTED,
    "unknown": style.MUTED,
}

#: The starts the trace plot draws without a selection: the best this many.
TRACES_SHOWN = 50

#: The offset the waterfall adds before its log axis, so the best start (zero above itself) shows.
OFFSET = 1.0


def _exit(flag, message) -> str:
    """How a start ended, by Fides' name for its exit flag."""
    from fides import ExitFlag

    if isinstance(message, str) and message.startswith(STOPPED):
        return STOPPED
    if flag is None or (isinstance(flag, float) and math.isnan(flag)):
        return "unknown"
    try:
        return ExitFlag(int(flag)).name
    except ValueError:
        return str(int(flag))


def summary(meta: dict, opt: dict, starts: pd.DataFrame) -> str:
    """The tab's line of counts, as Markdown."""
    finite = starts["fval"].dropna()
    best = finite.min() if len(finite) else None
    at_best = int((finite <= best + AT_BEST_NLL_TOL).sum()) if best is not None else 0
    parts = [
        f"**{len(starts)} starts**",
        f"{at_best} reached the best (objective within {AT_BEST_NLL_TOL:g} of it)",
    ]
    if best is not None:
        parts.append(f"best objective (negative log-likelihood) {best:.6g}")
    if killed := meta.get("n_starts_killed") or opt.get("n_starts_killed"):
        parts.append(f"{killed} stopped past their cap")
    if failed := int(starts["fval"].isna().sum()):
        parts.append(f"{failed} failed")
    counts = opt.get("exitflag_counts") or {}
    if counts:
        parts.append(
            "ended: "
            + ", ".join(
                f"{k.removeprefix('ExitFlag.')} {v}"
                for k, v in sorted(counts.items(), key=lambda kv: -kv[1])
            )
        )
    if meta.get("wall_seconds") is not None:
        parts.append(f"{meta['wall_seconds']} s")
    return " · ".join(parts)


#: What each way a start ends means, for the caption under the waterfall.
EXIT_MEANINGS = {
    "FTOL": "converged: the objective stopped changing",
    "GTOL": "converged: the gradient vanished",
    "XTOL": "converged: the parameters stopped changing",
    "DELTA_TOO_SMALL": "stalled: the trust region shrank to nothing, often a flat or rough objective",
    "MAXITER": "ran out of iterations",
    "MAXTIME": "ran out of time (the per-start cap)",
    "NOT_FINITE": "hit a non-finite objective or gradient",
    "EXCEEDED_BOUNDARY": "left the parameter bounds",
    STOPPED: "stuck past its cap, stopped and scored as failed",
    "DID_NOT_RUN": "never ran",
}


def exit_caption(ended) -> str:
    """The meaning of each way the shown starts ended, the commonest first."""
    counts = pd.Series(list(ended)).value_counts()
    return " · ".join(f"**{e}** {EXIT_MEANINGS[e]}" for e in counts.index if e in EXIT_MEANINGS)


def waterfall_frame(starts: pd.DataFrame, n_warm: int = 0) -> pd.DataFrame:
    df = starts.copy()
    best = df["fval"].min()
    df["rank"] = np.arange(len(df))
    df["above_best"] = df["fval"] - best + OFFSET
    df["ended"] = [_exit(f, m) for f, m in zip(df["exitflag"], df["message"])]
    df["warm"] = [str(i).isdigit() and int(i) < n_warm for i in df["id"]]
    return df.dropna(subset=["fval"])


def waterfall(df: pd.DataFrame) -> tuple[hv.Overlay, hv.Points]:
    """The waterfall, and its points, which a selection stream reads."""
    colors = {e: EXIT_COLORS.get(e, style.MUTED) for e in df["ended"].unique()}
    points = hv.Points(
        df,
        ["rank", "above_best"],
        ["id", "fval", "ended", "time", "n_fval", "warm"],
    ).opts(
        color="ended",
        cmap=colors,
        size=8,
        line_color="white",
        logy=True,
        tools=["hover", "box_select", "tap"],
        active_tools=["box_select"],
        nonselection_alpha=0.3,
        show_legend=True,
        legend_position="right",
        xlabel="start rank (best first)",
        ylabel=f"objective − best + {OFFSET:g}",
        responsive=True,
        height=320,
        ylim=(OFFSET / 2, 3 * max(float(df["above_best"].max()), 3 * OFFSET)),
    )
    warm = hv.Points(df[df["warm"]], ["rank", "above_best"]).opts(
        marker="circle",
        fill_alpha=0,
        line_color="black",
        size=13,
        logy=True,
    )
    band = hv.HLine(OFFSET + AT_BEST_NLL_TOL).opts(
        color=style.MUTED, line_dash="dashed", line_width=1
    )
    return (band * warm * points).opts(title="Waterfall (ringed: warm starts)"), points


def traces(trace: pd.DataFrame, ranks: dict[str, int], best: float, chosen: list[int], x: str):
    """Each start's objective above the best over ``x`` (``iteration`` or ``time``): the chosen
    ranks coloured, the rest grey."""
    if trace is None or trace.empty:
        return hv.Curve([]).opts(title="No per-iteration trace stored", responsive=True, height=300)
    trace = trace.assign(rank=trace["start"].map(lambda s: ranks.get(str(s), s)))
    shown = set(chosen) or set(range(min(TRACES_SHOWN, len(ranks))))
    grey, picked = [], []
    for rank, sub in trace.groupby("rank"):
        sub = sub.sort_values(x)
        y = np.maximum(sub["fval"].to_numpy(dtype=float) - best + OFFSET, OFFSET * 1e-3)
        path = {x: sub[x].to_numpy(dtype=float), "above_best": y, "rank": rank}
        (picked if rank in shown else grey).append(path)
    lines = hv.Path(grey, [x, "above_best"], vdims="rank").opts(color=style.MUTED, alpha=0.25)
    colored = hv.Path(picked, [x, "above_best"], vdims="rank").opts(
        color="rank", cmap="viridis", line_width=1.5, colorbar=False
    )
    return (lines * colored).opts(
        hv.opts.Path(
            logy=True, responsive=True, height=300, tools=["hover"], ylim=(OFFSET * 1e-3, None)
        ),
        hv.opts.Overlay(
            title="Objective over the run", xlabel=x, ylabel=f"objective − best + {OFFSET:g}"
        ),
    )


def _scaled(value: float, scale: str) -> float:
    if scale == "log10":
        return math.log10(value) if value > 0 else -math.inf
    if scale == "log":
        return math.log(value) if value > 0 else -math.inf
    return value


def spread_frame(starts: pd.DataFrame, opt: dict, parameters: pd.DataFrame | None) -> pd.DataFrame:
    """One row per (start, estimated parameter): where the start ended within the parameter's
    bounds on its estimation scale, 0 at the lower bound and 1 at the upper."""
    names = list(opt.get("x_names") or [])
    scales = list(opt.get("x_scales") or ["lin"] * len(names))
    if not names or parameters is None:
        return pd.DataFrame()
    estimated = {p for p, e in parameters["estimate"].items() if str(e).lower() in ("true", "1")}
    best = starts["fval"].min()
    rows = []
    for rank, start in starts.reset_index(drop=True).iterrows():
        x, x0 = start.get("x"), start.get("x0")
        if not isinstance(x, list) or len(x) != len(names):
            continue
        for i, name in enumerate(names):
            if name not in estimated or name not in parameters.index:
                continue
            lb = _scaled(float(parameters.at[name, "lowerBound"]), scales[i])
            ub = _scaled(float(parameters.at[name, "upperBound"]), scales[i])
            if not (math.isfinite(lb) and math.isfinite(ub)) or ub <= lb:
                continue
            row = {
                "parameter": name,
                "rank": rank,
                "id": start["id"],
                "position": (x[i] - lb) / (ub - lb),
                "value": x[i],
                "scale": scales[i],
                "above_best": (start["fval"] - best) if start["fval"] is not None else np.nan,
            }
            if isinstance(x0, list) and len(x0) == len(names):
                row["start_position"] = (x0[i] - lb) / (ub - lb)
            rows.append(row)
    return pd.DataFrame(rows)


def spread(df: pd.DataFrame, chosen: list[int], show_x0: bool):
    if df.empty:
        return hv.Points([]).opts(
            title="No per-start parameter vectors stored", responsive=True, height=320
        )
    at_best = df["above_best"] <= AT_BEST_NLL_TOL
    df = df.assign(group=np.where(at_best, "at the best", "other starts"))
    order = (
        df[at_best]
        .groupby("parameter")["position"]
        .agg(lambda s: s.max() - s.min())
        .reindex(df["parameter"].unique())
        .fillna(0)
        .sort_values(ascending=False)
        .index
    )
    df["parameter"] = pd.Categorical(df["parameter"], list(order))
    df = df.sort_values("parameter")
    width = max(500, 32 * len(order))
    base = hv.Scatter(
        df, "parameter", ["position", "group", "id", "value", "scale", "above_best"]
    ).opts(
        jitter=0.35,
        color="group",
        cmap={"at the best": style.PALETTE[0], "other starts": "#c7c7c7"},
        size=6,
        tools=["hover"],
        xrotation=60,
        width=width,
        height=340,
        ylim=(-0.05, 1.05),
        xlabel="estimated parameter",
        ylabel="position in bounds",
        show_legend=True,
        legend_position="right",
    )
    layers = [
        hv.HLine(0).opts(color=style.MUTED, line_width=1),
        hv.HLine(1).opts(color=style.MUTED, line_width=1),
        base,
    ]
    if show_x0 and "start_position" in df:
        layers.insert(
            2,
            hv.Scatter(df, "parameter", "start_position").opts(
                jitter=0.35, marker="x", color="#999999", size=5
            ),
        )
    if chosen:
        sel = df[df["rank"].isin(chosen)]
        layers.append(
            hv.Scatter(sel, "parameter", ["position", "id"]).opts(
                color=style.PALETTE[3], size=9, marker="triangle"
            )
        )
    return hv.Overlay(layers).opts(title="Where each start ended, per estimated parameter")


def live_view(
    progress: dict | None, reference: tuple[int, float] | None = None
) -> pn.viewable.Viewable | None:
    """The running fit, or ``None`` when none runs. ``reference``, HEAD's version and best
    objective, says whether the fit is beating what is stored."""
    if not progress:
        return None
    now = time.time()
    phase, total, done = (
        progress.get("phase"),
        progress.get("total") or 0,
        progress.get("done") or 0,
    )
    line = [f"**Fitting now** · {phase} for {now - (progress.get('since') or now):.0f} s"]
    if total:
        line.append(f"{done}/{total} starts done, {progress.get('queued') or 0} not begun")
    if progress.get("best") is not None:
        line.append(f"best so far {progress['best']:.6g}")
        if reference is not None:
            line.append(f"v{reference[0]}'s best was {reference[1]:.6g}")
    began = progress.get("began")
    if began and done and done < total:
        waited = (progress.get("waited") or 0.0) + (
            now - progress["since"] if phase == "queued" else 0.0
        )
        line.append(f"about {(now - began - waited) * (total - done) / done:.0f} s left")
    items = [pn.pane.Markdown(" · ".join(line))]
    fvals = sorted(progress.get("fvals") or [])
    if fvals:
        # Dimensions of their own: HoloViews links the ranges of same-named dimensions across the
        # page, which stretched the stored version's waterfall to the running fit's starts.
        df = pd.DataFrame(
            {
                "finished": range(len(fvals)),
                "running_above_best": np.array(fvals) - fvals[0] + OFFSET,
                "fval": fvals,
            }
        )
        items.append(
            pn.pane.HoloViews(
                hv.Points(df, ["finished", "running_above_best"], "fval").opts(
                    xlabel="finished start (best first)",
                    logy=True,
                    color=style.PALETTE[1],
                    size=7,
                    tools=["hover"],
                    responsive=True,
                    height=200,
                    title="Finished starts so far",
                    ylabel=f"objective − best + {OFFSET:g}",
                ),
                sizing_mode="stretch_width",
            )
        )
    return pn.Card(
        *items,
        title="Running fit",
        collapsible=False,
        sizing_mode="stretch_width",
        styles={"border-left": f"4px solid {style.PALETTE[1]}"},
    )


class MultistartTab:
    """The tab, reading the app's ``version`` and ``tick`` (a store change)."""

    def __init__(self, app) -> None:
        self.app = app
        self.axis = pn.widgets.RadioButtonGroup(
            options=["iteration", "time"], value="iteration", button_type="light"
        )
        self.show_x0 = pn.widgets.Checkbox(name="show where each start began", value=False)

    def view(self) -> pn.viewable.Viewable:
        return pn.Column(
            pn.bind(self._live, self.app.param.tick),
            pn.bind(self._version, self.app.param.version),
            sizing_mode="stretch_width",
        )

    def _live(self, _tick):
        store, head = self.app.store, self.app.store.head()
        starts = store.starts(head) if head is not None else None
        reference = None
        if starts is not None and starts["fval"].notna().any():
            reference = (head, float(starts["fval"].min()))
        return live_view(store.fit_progress(), reference) or pn.Spacer(height=0)

    def _version(self, version):
        store = self.app.store
        if version is None:
            return pn.pane.Markdown("No version yet.")
        starts = store.starts(version)
        if starts is None or starts.empty:
            return pn.pane.Markdown(
                f"v{version} stored no multistart: it was a restore, a refused change or a "
                "simulate-only pass."
            )
        meta, opt = store.meta(version), store.optimization(version) or {}
        df = waterfall_frame(starts, int(meta.get("n_warm_starts") or 0))
        fall, points = waterfall(df)
        picked = hv.streams.Selection1D(source=points)
        ranks = {str(i): r for i, r in zip(df["id"], df["rank"])}
        best = float(df["fval"].min())
        trace = store.trace(version)
        spread_df = spread_frame(starts, opt, store.parameters(version))

        def chosen(index):
            return [int(df["rank"].iloc[i]) for i in index or [] if i < len(df)]

        trace_map = hv.DynamicMap(
            pn.bind(lambda index, x: traces(trace, ranks, best, chosen(index), x), x=self.axis),
            streams=[picked],
        )
        spread_map = hv.DynamicMap(
            pn.bind(lambda index, x0: spread(spread_df, chosen(index), x0), x0=self.show_x0),
            streams=[picked],
        )
        return pn.Column(
            pn.pane.Markdown(summary(meta, opt, starts)),
            pn.Row(
                pn.pane.HoloViews(fall, linked_axes=False, sizing_mode="stretch_width"),
                pn.Column(
                    self.axis,
                    pn.pane.HoloViews(trace_map, linked_axes=False, sizing_mode="stretch_width"),
                    sizing_mode="stretch_width",
                ),
                sizing_mode="stretch_width",
            ),
            pn.pane.Markdown(
                exit_caption(df["ended"])
                + "  \nBox-select starts in the waterfall to pick them out in the trace and below.",
                styles={"color": style.MUTED, "font-size": "12px"},
            ),
            pn.pane.Markdown(
                "Each column is an estimated parameter, 0 its lower bound and 1 its upper, on the "
                "scale it is estimated on. Where the starts that reached the best are spread out "
                "along a column, the data do not pin that parameter down; one at 0 or 1 sits on a "
                "bound.",
                styles={"color": style.MUTED, "font-size": "12px"},
            ),
            self.show_x0,
            pn.pane.HoloViews(spread_map, linked_axes=False),
            sizing_mode="stretch_width",
        )
