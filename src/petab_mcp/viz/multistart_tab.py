"""The Multistart tab: how one version's fit went, start by start, and the running fit.

* a line of counts (starts, at the best, stopped, how they ended, how long);
* the waterfall, each start's final objective above the best, by rank, coloured by how it ended
  or by its waterfall cluster (:func:`clusters`);
* each start's objective over its iterations or its wall time, both axes logarithmic;
* where each start ended in each estimated parameter's bounds, on the estimation scale, so a
  parameter the best starts disagree on, or one at a bound, stands out; optionally each start a
  line through the parameters.

One colouring, how each start ended or its waterfall cluster, applies to all three plots, under
one legend. Box-selecting starts in the waterfall picks them out in the other two. "At the best" is the
server's own count, within :data:`petab_mcp.constants.AT_BEST_NLL_TOL` of the lowest objective;
a waterfall cluster is a run of starts each within that of the one before.
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
from . import key as K
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

#: The offset the waterfall adds before its log axis, so the best start (zero above itself) shows.
OFFSET = 1.0


#: Fides' ``ExitFlag`` names by value, written out: importing fides to look them up cost half a
#: second of the page's first load (it imports scipy). A test holds the table to fides' own enum.
EXIT_NAMES = {
    0: "DID_NOT_RUN",
    -1: "MAXITER",
    -2: "MAXTIME",
    -3: "NOT_FINITE",
    -4: "EXCEEDED_BOUNDARY",
    -5: "DELTA_TOO_SMALL",
    1: "FTOL",
    2: "XTOL",
    3: "GTOL",
}


def _exit(flag, message) -> str:
    """How a start ended, by Fides' name for its exit flag."""
    if isinstance(message, str) and message.startswith(STOPPED):
        return STOPPED
    if flag is None or (isinstance(flag, float) and math.isnan(flag)):
        return "unknown"
    return EXIT_NAMES.get(int(flag), str(int(flag)))


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


#: What each way a start ends means, for its legend entry.
EXIT_MEANINGS = {
    "FTOL": "Converged: its fit stopped improving.",
    "GTOL": "Converged: it reached a point where no small change of the parameters improves the "
    "fit.",
    "XTOL": "Converged: the parameters stopped changing.",
    "DELTA_TOO_SMALL": "Stalled: the optimiser's steps shrank to nothing before it converged, "
    "often on a flat or rough stretch of the objective.",
    "MAXITER": "Ran out of iterations before it converged.",
    "MAXTIME": "Ran out of time before it converged (the per-start cap).",
    "NOT_FINITE": "Failed: the model could not be simulated at the parameters it tried.",
    "EXCEEDED_BOUNDARY": "Failed: it left the parameter bounds.",
    STOPPED: "Stuck past its time cap: the server stopped it and scored it as failed.",
    "DID_NOT_RUN": "Never ran.",
    "unknown": "Not stored: the version kept no record of how its starts ended.",
}

#: At most this many waterfall clusters get a colour of their own, best first.
CLUSTERS_COLOURED = 7

#: The fewest starts, and the least share of them, a waterfall cluster needs for a colour: one
#: that few found is not a fit the optimiser keeps returning to. The best cluster needs only the
#: fewest: how often the best fit was found is what the waterfall is read for.
CLUSTER_MIN, CLUSTER_SHARE = 3, 0.02

#: The grey of the starts that are context: the uncoloured clusters, the starts not picked.
OTHERS = "#c7c7c7"

#: The tab's two colourings, each applied to all three plots.
BY_EXIT, BY_CLUSTER = "how each ended", "waterfall cluster"

WATERFALL = """
Fitting runs an optimiser from many random guesses of the parameters and lets each improve its fit
until it stops. Each dot is one of these starts, sorted from the best fit on the left to the worst
on the right. Up is how much worse its fit is than the best start's: the objective (a negative
log-likelihood, lower is better) above the best one, plus 1 so the best start shows on the log
scale.

A flat step of many dots at the bottom left, under the dashed line, means many starts found the
same best fit from different guesses: it is very likely the best this model can do. Steps further
right are other fits the optimiser got stuck in (local optima). If only one or two dots reach the
bottom, the best fit was found by luck, and a better one may exist that more starts would find.

Colour, the same in all three plots, says how each start ended or which step (cluster) it is on;
hover a legend entry for what it means. Box-select dots to follow those starts in the other two
plots; press Esc to clear.
"""

TRACES = """
How each start's fit improved while the optimiser worked on it: one line per start, from its random
guess on the left to where it stopped on the right. Across is the optimiser's step (the first is 1)
or the seconds it had run; up is how much worse than the best start's final fit, plus 1. Both axes
are logarithmic, so the big early gains and the small late ones both show.

A line that flattens out converged. One still falling at its right end was cut off before it
finished. Many lines dropping onto the same floor is the best fit being found again and again. A
line that runs far longer than the rest is a start where the model was slow or hard to simulate.
Box-select starts in the waterfall to show only those in colour.
"""

SPREAD = """
Where each start's fitted parameters ended. Each column is one estimated parameter; up is where its
value ended between the lowest (0) and highest (1) value it was allowed, on the scale it is
estimated on (usually log10, so equal steps are equal fold changes). Within a column the dots are
moved sideways only so they do not hide each other.

Look at the large dots, the starts that reached the best fit. Where they sit together, the data
pin that parameter down. Where they spread along the column, very different values fit the data
equally well: the data cannot tell you that parameter's value (it is not identifiable). Large dots
at 0 or 1 sit on a bound: the fit wants a value outside the range allowed. Columns are sorted most
spread out first.

With "connect each start's points" on, each start is one line through the columns. Between two
neighbouring columns, lines that all cross in an X mean one parameter goes up as the other goes
down: the data fix a combination of the two, such as their product or ratio, not each one.
"""

RUNNING = """
A fit is running now. Each dot is a start that has finished, sorted from the best fit on the left;
up is how much worse its objective is than the best so far, plus 1, on a log scale. A flat step of
dots at the bottom left means the best fit is being found again and again; the stored version's
plots below show how the last fit went.
"""


def clusters(fval) -> np.ndarray:
    """Each start's waterfall cluster, 1 the best, for objectives sorted best first: a new cluster
    wherever the objective rises by more than ``AT_BEST_NLL_TOL`` over the start before it."""
    fval = np.asarray(fval, dtype=float)
    if not len(fval):
        return np.array([], dtype=int)
    return 1 + np.concatenate([[0], np.cumsum(np.diff(fval) > AT_BEST_NLL_TOL)])


def least_starts(n_starts: int) -> int:
    """How many starts a cluster other than the best needs for a colour."""
    return max(CLUSTER_MIN, math.ceil(CLUSTER_SHARE * n_starts))


def cluster_colors(cluster: pd.Series) -> dict[int, str]:
    """A colour for each of the first :data:`CLUSTERS_COLOURED` clusters that hold enough of the
    starts (:func:`least_starts`; the best, :data:`CLUSTER_MIN`)."""
    sizes = cluster.value_counts()
    least = least_starts(len(cluster))
    shared = [c for c in sorted(sizes.index) if sizes[c] >= (CLUSTER_MIN if c == 1 else least)][
        :CLUSTERS_COLOURED
    ]
    return {c: style.PALETTE[i] for i, c in enumerate(shared)}


def waterfall_frame(starts: pd.DataFrame, n_warm: int = 0) -> pd.DataFrame:
    df = starts.copy()
    best = df["fval"].min()
    df["rank"] = np.arange(len(df))
    df["above_best"] = df["fval"] - best + OFFSET
    df["ended"] = [_exit(f, m) for f, m in zip(df["exitflag"], df["message"])]
    df["warm"] = [str(i).isdigit() and int(i) < n_warm for i in df["id"]]
    df = df.dropna(subset=["fval"])
    df["cluster"] = clusters(df["fval"])
    df["cluster_color"] = df["cluster"].map(cluster_colors(df["cluster"])).fillna(OTHERS)
    df["exit_color"] = df["ended"].map(EXIT_COLORS).fillna(style.MUTED)
    return df


def colors(df: pd.DataFrame, by: str) -> dict[int, str]:
    """Each start's colour by its rank, ``by`` either colouring."""
    return dict(zip(df["rank"], df["cluster_color" if by == BY_CLUSTER else "exit_color"]))


def legend(
    df: pd.DataFrame, starts: pd.DataFrame, by: str, show_x0: bool, chosen: list[int]
) -> list[dict]:
    """The tab's one legend, for all three plots. How the starts ended counts every start, a
    failed one too: it has no objective to plot, but the summary counts its exit, and the legend
    must say what it means."""
    out = []
    if by == BY_CLUSTER:
        for c, color in cluster_colors(df["cluster"]).items():
            sub = df[df["cluster"] == c]
            low, high = f"{sub['fval'].min():.6g}", f"{sub['fval'].max():.6g}"
            out.append(
                K.entry(
                    f"cluster {c} · {len(sub)}",
                    f"{len(sub)} starts whose objectives lie within {AT_BEST_NLL_TOL:g} of the "
                    "next, "
                    + (f"all at {low}" if low == high else f"from {low} to {high}")
                    + ": most likely one fit, found again and again."
                    + (" The best one." if c == 1 else ""),
                    color,
                )
            )
        if rest := int((df["cluster_color"] == OTHERS).sum()):
            out.append(
                K.entry(
                    f"the others · {rest}",
                    f"Starts in clusters too small for a colour (fewer than {least_starts(len(df))} "
                    f"starts), or past the first {CLUSTERS_COLOURED}: fits the optimiser rarely "
                    "returned to.",
                    OTHERS,
                )
            )
    else:
        ended = pd.Series([_exit(f, m) for f, m in zip(starts["exitflag"], starts["message"])])
        failed = starts["fval"].isna().to_numpy()
        for e, n in ended.value_counts().items():
            about = EXIT_MEANINGS.get(e, "")
            if n_failed := int(failed[(ended == e).to_numpy()].sum()):
                about += f" {n_failed} of these have no objective, so they are not drawn."
            out.append(K.entry(f"{e} · {n}", about, EXIT_COLORS.get(e, style.MUTED)))
    out.append(
        K.entry(
            "reached the best",
            f"A start whose objective ended within {AT_BEST_NLL_TOL:g} of the best: under the "
            "waterfall's dashed line, and drawn large in the parameter plot. It counts as having "
            "found the best fit.",
            style.MUTED,
            "╌",
        )
    )
    if n_warm := int(df["warm"].sum()):
        out.append(
            K.entry(
                f"warm start · {n_warm}",
                "Ringed in the waterfall: a start that began from an earlier version's best "
                "parameters instead of a random guess.",
                "currentColor",
                "◯",
            )
        )
    if show_x0:
        out.append(
            K.entry(
                "where it began",
                "In the parameter plot: the random guess each start began from.",
                "#999999",
                "×",
            )
        )
    if chosen:
        out.append(
            K.entry(
                f"picked · {len(chosen)}",
                "The starts box-selected in the waterfall: the only ones in colour in the "
                "objective plot, marked in the parameter plot.",
                style.PALETTE[3],
                "▲",
            )
        )
    return out


def waterfall_points(df: pd.DataFrame, by: str = BY_EXIT) -> hv.Points:
    """The waterfall's points, which a selection stream reads."""
    return hv.Points(
        df,
        ["rank", "above_best"],
        ["id", "fval", "ended", "cluster", "time", "n_fval", "exit_color", "cluster_color"],
    ).opts(
        color=hv.dim("cluster_color" if by == BY_CLUSTER else "exit_color"),
        size=8,
        line_color="white",
        logy=True,
        tools=["hover", "box_select", "tap"],
        hover_tooltips=["id", "fval", "ended", "cluster", "time", "n_fval"],
        active_tools=["box_select"],
        nonselection_alpha=0.3,
        xlabel="start rank (best first)",
        ylabel=f"objective − best + {OFFSET:g}",
        responsive=True,
        height=370,
        ylim=(OFFSET / 2, 3 * max(float(df["above_best"].max()), 3 * OFFSET)),
    )


def waterfall_marks(df: pd.DataFrame) -> hv.Overlay:
    """The waterfall's dashed line at the best and its rings round the warm starts."""
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
    return band * warm


def on_top(df: pd.DataFrame) -> pd.DataFrame:
    """``df``'s rows in drawing order: grey first, then the worst start to the best, so the best
    are drawn over the rest."""
    return (
        df.assign(_grey=df["color"] == OTHERS)
        .sort_values(["_grey", "rank"], ascending=[False, False], kind="stable")
        .drop(columns="_grey")
    )


def traces(
    trace: pd.DataFrame,
    ranks: dict[str, int],
    best: float,
    chosen: list[int],
    x: str,
    color: dict[int, str],
):
    """Each start's objective above the best over ``x`` (``iteration`` or ``time``), both axes
    logarithmic: the chosen ranks, or every one, in their ``color``, the rest grey. Every start's
    trace is rows of one segments table: a path per start made HoloViews build a dataset for each,
    7.8 s of Fujita's 300 starts."""
    if trace is None or trace.empty:
        return hv.Curve([]).opts(responsive=True, height=300)
    df = trace.assign(rank=trace["start"].map(lambda s: ranks.get(str(s), s)))
    df = df.sort_values(["rank", x], kind="stable")
    rank = df["rank"].to_numpy()
    xs = df[x].to_numpy(dtype=float)
    if x == "iteration":
        xs = xs + 1  # the first iteration is 0, which a log axis cannot show
    else:
        xs = np.maximum(xs, np.nanmin(xs[xs > 0]) if (xs > 0).any() else 1e-3)
    ys = np.maximum(df["fval"].to_numpy(dtype=float) - best + OFFSET, OFFSET * 1e-3)
    joined = rank[1:] == rank[:-1]
    segments = pd.DataFrame(
        {"x0": xs[:-1], "y0": ys[:-1], "x1": xs[1:], "y1": ys[1:], "rank": rank[:-1]}
    )[joined]
    segments["color"] = segments["rank"].map(color).fillna(OTHERS)
    picked = segments["rank"].isin(chosen) if chosen else np.full(len(segments), True)
    kdims = ["x0", "y0", "x1", "y1"]
    low = min(OFFSET / 2, float(np.nanmin(ys)) / 2) if len(ys) else OFFSET / 2
    return (
        hv.Segments(segments[~picked], kdims, "rank").opts(color=style.MUTED, alpha=0.2)
        * hv.Segments(on_top(segments[picked]), kdims, ["rank", "color"]).opts(
            color=hv.dim("color"), line_width=1.2, alpha=0.8
        )
    ).opts(
        hv.opts.Segments(logx=True, logy=True, responsive=True, height=300, ylim=(low, None)),
        hv.opts.Overlay(
            xlabel="iteration" if x == "iteration" else "time (s)",
            ylabel=f"objective − best + {OFFSET:g}",
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
    cols, lbs, ubs = [], [], []
    for i, name in enumerate(names):
        if name not in estimated or name not in parameters.index:
            continue
        lb = _scaled(float(parameters.at[name, "lowerBound"]), scales[i])
        ub = _scaled(float(parameters.at[name, "upperBound"]), scales[i])
        if math.isfinite(lb) and math.isfinite(ub) and ub > lb:
            cols.append(i), lbs.append(lb), ubs.append(ub)
    starts = starts.reset_index(drop=True)

    def matrix(column: str) -> np.ndarray:
        rows = [
            v if isinstance(v, list) and len(v) == len(names) else [np.nan] * len(names)
            for v in starts.get(column, pd.Series([None] * len(starts)))
        ]
        return np.asarray(rows, dtype=float).reshape(len(starts), len(names))[:, cols]

    if not cols or starts.empty:
        return pd.DataFrame()
    lb, width = np.asarray(lbs), np.asarray(ubs) - np.asarray(lbs)
    x, x0 = matrix("x"), matrix("x0")
    n_starts, n_cols = x.shape
    fval = starts["fval"].astype(float).to_numpy()
    df = pd.DataFrame(
        {
            "parameter": np.tile([names[i] for i in cols], n_starts),
            "rank": np.repeat(np.arange(n_starts), n_cols),
            "id": np.repeat(starts["id"].to_numpy(), n_cols),
            "position": ((x - lb) / width).ravel(),
            "value": x.ravel(),
            "scale": np.tile([scales[i] for i in cols], n_starts),
            "above_best": np.repeat(fval - np.nanmin(fval), n_cols),
            "start_position": ((x0 - lb) / width).ravel(),
        }
    )
    df = df[np.isfinite(df["value"])]
    return df if df["start_position"].notna().any() else df.drop(columns="start_position")


#: How far either side of its column a start's dots move, as a share of the columns' spacing.
SIDEWAYS = 0.3


def spread(
    df: pd.DataFrame,
    chosen: list[int],
    show_x0: bool,
    connect: bool,
    color: dict[int, str],
):
    """Where each start ended in each parameter's bounds, in its ``color``: the starts at the best
    large, the rest small and faint. A start keeps one sideways offset in every column, so
    ``connect`` can draw it as one line through the columns."""
    if df.empty:
        return hv.Points([]).opts(responsive=True, height=320)
    at_best = df["above_best"] <= AT_BEST_NLL_TOL
    order = list(
        df[at_best]
        .groupby("parameter")["position"]
        .agg(lambda s: s.max() - s.min())
        .reindex(df["parameter"].unique())
        .fillna(0)
        .sort_values(ascending=False)
        .index
    )
    column = {p: i for i, p in enumerate(order)}
    # The golden ratio's fractions spread the starts evenly across the offsets, the same in
    # every column.
    offset = ((df["rank"].to_numpy() * 0.618034) % 1 - 0.5) * 2 * SIDEWAYS
    df = on_top(
        df.assign(
            column=df["parameter"].map(column).to_numpy() + offset,
            color=df["rank"].map(color).fillna(OTHERS).to_numpy(),
            best=at_best.to_numpy(),
        )
    )
    col = hv.Dimension("column", label="estimated parameter")
    layers = [
        hv.HLine(0).opts(color=style.MUTED, line_width=1),
        hv.HLine(1).opts(color=style.MUTED, line_width=1),
    ]
    if connect:
        path = df.sort_values(["rank", "column"])
        same = path["rank"].to_numpy()[1:] == path["rank"].to_numpy()[:-1]
        lines = on_top(
            pd.DataFrame(
                {
                    "x0": path["column"].to_numpy()[:-1],
                    "y0": path["position"].to_numpy()[:-1],
                    "x1": path["column"].to_numpy()[1:],
                    "y1": path["position"].to_numpy()[1:],
                    "color": path["color"].to_numpy()[:-1],
                    "rank": path["rank"].to_numpy()[:-1],
                    "best": path["best"].to_numpy()[:-1],
                }
            )[same]
        )
        layers.append(
            hv.Segments(lines, ["x0", "y0", "x1", "y1"], ["color", "best"]).opts(
                color=hv.dim("color"),
                alpha=hv.dim("best").categorize({True: 0.6}, default=0.15),
                line_width=1,
            )
        )
    if show_x0 and "start_position" in df:
        layers.append(
            hv.Scatter(df, col, "start_position").opts(marker="x", color="#999999", size=5)
        )
    layers.append(
        hv.Scatter(
            df,
            col,
            ["position", "color", "best", "parameter", "id", "value", "scale", "above_best"],
        ).opts(
            color=hv.dim("color"),
            size=hv.dim("best").categorize({True: 8}, default=4),
            alpha=hv.dim("best").categorize({True: 1.0}, default=0.45),
            tools=["hover"],
            hover_tooltips=["parameter", "id", "value", "scale", "above_best"],
        )
    )
    if chosen:
        sel = df[df["rank"].isin(chosen)]
        layers.append(
            hv.Scatter(sel, col, ["position", "id"]).opts(
                color=style.PALETTE[3], size=9, marker="triangle"
            )
        )
    return hv.Overlay(layers).opts(
        hv.opts.Scatter(
            xticks=list(enumerate(order)),
            xrotation=60,
            xlim=(-0.6, len(order) - 0.4),
            ylim=(-0.05, 1.05),
            width=max(500, 32 * len(order)),
            height=340,
            xlabel="estimated parameter",
            ylabel="position in bounds",
        )
    )


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
        items += [
            K.key("Finished starts so far", RUNNING),
            pn.pane.HoloViews(
                hv.Points(df, ["finished", "running_above_best"], "fval").opts(
                    xlabel="finished start (best first)",
                    logy=True,
                    color=style.PALETTE[1],
                    size=7,
                    tools=["hover"],
                    responsive=True,
                    height=200,
                    ylabel=f"objective − best + {OFFSET:g}",
                ),
                sizing_mode="stretch_width",
            ),
        ]
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
        self.colour = pn.widgets.RadioButtonGroup(
            options=[BY_EXIT, BY_CLUSTER], value=BY_EXIT, button_type="light", align="center"
        )
        self.show_x0 = pn.widgets.Checkbox(name="show where each start began", value=False)
        self.connect = pn.widgets.Checkbox(name="connect each start's points", value=False)

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
        points = hv.DynamicMap(pn.bind(lambda by: waterfall_points(df, by), self.colour))
        picked = hv.streams.Selection1D(source=points)
        ranks = {str(i): r for i, r in zip(df["id"], df["rank"])}
        best = float(df["fval"].min())
        trace = store.trace(version)
        spread_df = spread_frame(starts, opt, store.parameters(version))

        def chosen(index):
            return [int(df["rank"].iloc[i]) for i in index or [] if i < len(df)]

        trace_map = hv.DynamicMap(
            pn.bind(
                lambda index, x, by: traces(trace, ranks, best, chosen(index), x, colors(df, by)),
                x=self.axis,
                by=self.colour,
            ),
            streams=[picked],
        )
        spread_map = hv.DynamicMap(
            pn.bind(
                lambda index, x0, line, by: spread(
                    spread_df, chosen(index), x0, line, colors(df, by)
                ),
                x0=self.show_x0,
                line=self.connect,
                by=self.colour,
            ),
            streams=[picked],
        )
        key = pn.bind(
            lambda index, by, x0: K.Key(
                entries=legend(df, starts, by, x0, chosen(index)),
                sizing_mode="stretch_width",
                margin=(0, 10),
            ),
            picked.param.index,
            self.colour,
            self.show_x0,
        )

        def plot(title: str, about: str, *body, basis: int = 420) -> pn.Column:
            return pn.Column(
                K.key(title, about),
                *body,
                sizing_mode="stretch_width",
                min_width=basis,
                styles={"flex": f"1 1 {basis}px"},
            )

        return pn.Column(
            pn.pane.Markdown(summary(meta, opt, starts)),
            pn.Row(
                pn.pane.HTML("<b>Colour by</b>", align="center", margin=(0, 4, 0, 10)),
                self.colour,
            ),
            key,
            # Side by side where there is room, one above the other where there is not.
            pn.FlexBox(
                # A flex basis each: stretch_width alone claims the whole row, so they always
                # wrapped, wide window or not.
                plot(
                    "Waterfall",
                    WATERFALL,
                    pn.pane.HoloViews(
                        waterfall_marks(df) * points, linked_axes=False, sizing_mode="stretch_width"
                    ),
                ),
                plot(
                    "Objective over the run",
                    TRACES,
                    self.axis,
                    pn.pane.HoloViews(trace_map, linked_axes=False, sizing_mode="stretch_width"),
                ),
                flex_wrap="wrap",
                sizing_mode="stretch_width",
            ),
            K.key("Where each start ended, per estimated parameter", SPREAD),
            pn.Row(self.show_x0, self.connect),
            pn.pane.HoloViews(spread_map, linked_axes=False),
            sizing_mode="stretch_width",
        )
