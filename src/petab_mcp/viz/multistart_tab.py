"""The Multistart tab: how one version's fit went, start by start, and the running fit.

* a line of counts (starts, at the best, stopped, how they ended, how long);
* the waterfall, each start's final objective above the best, by rank, coloured by how it ended
  or by its waterfall cluster (:func:`clusters`);
* each start's objective over its iterations or its wall time, both axes logarithmic;
* where each start ended in each estimated parameter's bounds, on the estimation scale, so a
  parameter the best starts disagree on, or one at a bound, stands out; optionally each start a
  line through the parameters.

One colouring, how each start ended, its waterfall cluster or whether it was a warm start,
applies to all three plots, under one legend. Box-selecting starts in the waterfall picks them out in the other two. "At the best" is the
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
import warnings

import holoviews as hv
import numpy as np
from bokeh.models import CustomJS, HoverTool
import pandas as pd
import panel as pn

from ..constants import AT_BEST_NLL_TOL, WARM_START_FRACTION
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

#: The y axis the waterfall and the objective plot share: Panel gives the plots of two panes that
#: allow it (``linked_axes``) one range for a dimension of one name, so a zoom in either is a zoom
#: in both. No other pane on the page allows it.
ABOVE = hv.Dimension("above_best", label=f"objective − best + {OFFSET:g}")

#: The height of the waterfall's and the objective plot's plotting areas: the same, and their
#: controls under them, so the y axis they share lines up side by side.
FRAME = 320


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

#: The tab's colourings, each applied to all three plots.
BY_EXIT, BY_CLUSTER, BY_START = "how each ended", "waterfall cluster", "warm or cold start"

#: A start's colour by where it began (:func:`warm_starts`), and what that means.
NOT_RECORDED = "not recorded"
START_COLORS = {"warm": style.PALETTE[1], "cold": style.PALETTE[0], NOT_RECORDED: style.MUTED}
START_MEANINGS = {
    "warm": "Began from the optima of the version this one was made from (its best, and its other "
    "optima) instead of a random guess.",
    "cold": "Began from a random guess within the parameters' bounds.",
    NOT_RECORDED: "This fit did not record which of its starts were warm (the server records it "
    "since 25 November 2026), and its start points do not tell: none begins where the fit of the "
    "version it was made from ended, as a warm start would.",
}

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

Colour, the same in all three plots, says how each start ended, which step (cluster) it is on, or
whether it began from an earlier version's best (warm) or a random guess (cold).
Box-select dots to follow those starts in the other two plots; press Esc to clear.
"""

LEGEND = """
One colour per group of starts, the same in all three plots: how each start ended, which
waterfall cluster it is in, or whether it was a warm or a cold start. Untick a group to grey it out in the waterfall and leave it out of the
other two plots. Hover an entry for what it means.
"""

TRACES = """
How each start's fit improved while the optimiser worked on it: one line per start, from its random
guess on the left to where it stopped on the right. Across is how many times the optimiser had
computed the model's fit (the first is 1), or the seconds it had run; up is how much worse than the
best start's final fit, plus 1.

Each line shows only the steps the optimiser kept, each better than the last; "rejected steps" adds
the trial steps it computed and threw away, which jump up. "Median per colour" draws, instead of
each start, the median of the starts of one colour, a band where the middle half of them lie, and
a fainter band from the lowest of them to the highest.
"Log axes" puts both axes on a log scale, so the big early gains and the small late ones both show.

A line that flattens out converged. One still falling at its right end was cut off before it
finished. Many lines dropping onto the same floor is the best fit being found again and again.
Box-select starts in the waterfall to show only those.
"""

SPREAD = """
Where each start's fitted parameters ended. Each column is one estimated parameter; up is where its
value ended between the lowest (0) and highest (1) value it was allowed, on the scale it is
estimated on, named over each column (log₁₀ for most, where equal steps are equal fold changes;
hover the legend for each). With "values" on, up is the
value itself on that scale, and grey bars mark each parameter's bounds. Within a column the dots
are moved sideways only so they do not hide each other.

Look at the large dots, the starts that reached the best fit. Where they sit together, the data
pin that parameter down. Where they spread along the column, very different values fit the data
equally well: the data cannot tell you that parameter's value (it is not identifiable). Large dots
at 0 or 1 sit on a bound: the fit wants a value outside the range allowed. Columns are sorted most
spread out first.

With "connect each start's points" on, each start is one line through the columns. Between two
neighbouring columns, lines that all cross in an X mean one parameter goes up as the other goes
down: the data fix a combination of the two, such as their product or ratio, not each one.

With "where each start began" on, an arrow runs from the random guess each start began from to
where it ended.
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


def _matches(x: np.ndarray, y: np.ndarray) -> bool:
    return bool(np.all(np.abs(x - y) <= 1e-9 * np.maximum(1.0, np.abs(y))))


def warm_starts(store, version: int, problem) -> int | None:
    """How many of ``version``'s starts were warm, the first so many ids: its fit's record says,
    since 25 November 2026 (``n_warm_starts``); before, its start points do.

    pyPESTO numbers a problem's guesses first and its random samples after them
    (``pypesto.startpoint.CheckedStartpoints``; the server's problems never re-sort them), and the
    server seeds ``WARM_START_FRACTION`` of the starts as guesses whenever it warm-starts
    (``fitting.warm_start_guesses``): the first at the pre-fit parameter table, which holds the
    parent's fitted values, the others at the parent's other optima. A random start begins at
    neither, so one that does makes the fit warm-started. ``None`` where none does."""
    opt = store.optimization(version) or {}
    if opt.get("n_warm_starts") is not None:
        return int(opt["n_warm_starts"])
    starts = store.starts(version)
    names, scales = list(opt.get("x_names") or []), list(opt.get("x_scales") or [])
    if starts is None or not names:
        return None
    began = {
        str(i): np.asarray(x0, dtype=float)
        for i, x0 in zip(starts["id"], starts["x0"])
        if isinstance(x0, list) and len(x0) == len(names)
    }
    n_warm = int(len(starts) * WARM_START_FRACTION)
    parent = store.meta(version).get("parent")
    has_parent = isinstance(parent, int) and parent >= 0
    table = store.parameters(parent) if has_parent else problem.parameter_df
    if "0" in began and table is not None:
        at = [
            (i, _scaled(float(table.at[n, "nominalValue"]), scales[i]))
            for i, n in enumerate(names)
            if n in table.index and str(table.at[n, "estimate"]).lower() in ("1", "true")
        ]
        at = [(i, v) for i, v in at if math.isfinite(v)]
        if at and _matches(began["0"][[i for i, _ in at]], np.array([v for _, v in at])):
            return n_warm
    if (
        has_parent
        and (popt := store.optimization(parent))
        and (pstarts := store.starts(parent)) is not None
    ):
        pnames = list(popt.get("x_names") or [])
        shared = [n for n in names if n in pnames]
        optima = np.array(
            [
                np.asarray(x, dtype=float)[[pnames.index(n) for n in shared]]
                for x in pstarts["x"]
                if isinstance(x, list) and len(x) == len(pnames)
            ]
        )
        warm = np.array(
            [
                began[str(i)][[names.index(n) for n in shared]]
                for i in range(n_warm)
                if str(i) in began
            ]
        )
        if shared and optima.size and warm.size:
            close = np.abs(warm[:, None, :] - optima[None]) <= 1e-9 * np.maximum(
                1.0, np.abs(optima)
            )
            if close.all(axis=2).any():
                return n_warm
    return None


def waterfall_frame(starts: pd.DataFrame, n_warm: int | None = None) -> pd.DataFrame:
    """The waterfall's starts. ``n_warm``, the fit's warm starts, numbered first; ``None`` where
    the fit did not record them."""
    df = starts.copy()
    best = df["fval"].min()
    df["rank"] = np.arange(len(df))
    df["above_best"] = df["fval"] - best + OFFSET
    df["ended"] = [_exit(f, m) for f, m in zip(df["exitflag"], df["message"])]
    df["warm"] = [n_warm is not None and str(i).isdigit() and int(i) < n_warm for i in df["id"]]
    df = df.dropna(subset=["fval"])
    df["cluster"] = clusters(df["fval"])
    df["cluster_color"] = df["cluster"].map(cluster_colors(df["cluster"])).fillna(OTHERS)
    df["exit_color"] = df["ended"].map(EXIT_COLORS).fillna(style.MUTED)
    df["began"] = NOT_RECORDED if n_warm is None else np.where(df["warm"], "warm", "cold")
    df["start_color"] = df["began"].map(START_COLORS)
    return df


def groups(df: pd.DataFrame, by: str) -> pd.Series:
    """Each start's colour group, the toggle of its legend entry: how it ended, its cluster (all
    the uncoloured clusters one group), or where it began."""
    if by == BY_START:
        return "start:" + df["began"]
    if by == BY_CLUSTER:
        return np.where(
            df["cluster_color"] == OTHERS, "cluster:others", "cluster:" + df["cluster"].astype(str)
        )
    return "exit:" + df["ended"]


def colors(df: pd.DataFrame, by: str, hidden=()) -> dict[int, str]:
    """Each start's colour by its rank, ``by`` any colouring; the starts of a ``hidden`` group
    left out."""
    shown = ~pd.Series(groups(df, by), index=df.index).isin(list(hidden))
    column = {BY_CLUSTER: "cluster_color", BY_START: "start_color"}.get(by, "exit_color")
    return dict(zip(df.loc[shown, "rank"], df.loc[shown, column]))


def legend(df: pd.DataFrame, starts: pd.DataFrame, by: str, chosen: list[int]) -> list[dict]:
    """The tab's one legend, for all three plots, each colour with a checkbox that hides its
    starts (:func:`groups`). How the starts ended counts every start, a failed one too: it has no
    objective to plot, but the summary counts its exit, and the legend must say what it means."""
    out = []
    if by == BY_START:
        for began, n in df["began"].value_counts().reindex(list(START_COLORS)).dropna().items():
            out.append(
                K.entry(
                    f"{began} · {int(n)}",
                    START_MEANINGS[began],
                    START_COLORS[began],
                    toggle=f"start:{began}",
                )
            )
    elif by == BY_CLUSTER:
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
                    toggle=f"cluster:{c}",
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
                    toggle="cluster:others",
                )
            )
    else:
        ended = pd.Series([_exit(f, m) for f, m in zip(starts["exitflag"], starts["message"])])
        failed = starts["fval"].isna().to_numpy()
        for e, n in ended.value_counts().items():
            about = EXIT_MEANINGS.get(e, "")
            if n_failed := int(failed[(ended == e).to_numpy()].sum()):
                about += f" {n_failed} of these have no objective, so they are not drawn."
            out.append(
                K.entry(f"{e} · {n}", about, EXIT_COLORS.get(e, style.MUTED), toggle=f"exit:{e}")
            )
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
    if (n_warm := int(df["warm"].sum())) and by != BY_START:
        out.append(
            K.entry(
                f"warm start · {n_warm}",
                "Ringed in the waterfall: a start that began from an earlier version's best "
                "parameters instead of a random guess.",
                "currentColor",
                "◯",
            )
        )
    if chosen:
        out.append(
            K.entry(
                f"picked · {len(chosen)}",
                "The starts box-selected in the waterfall: the only ones drawn in the objective "
                "plot, marked in the parameter plot.",
                style.PALETTE[3],
                "▲",
            )
        )
    return out


#: A hidden group's starts in the waterfall: kept, so the waterfall keeps its shape and its
#: selection its indices, but without their colour.
HIDDEN = "#e3e3e3"


def y_range(df: pd.DataFrame, tf: pd.DataFrame | None) -> tuple[float, float]:
    """The range of :data:`ABOVE`, the same in both plots that share it: every start's final
    objective and every evaluation of the trace, a rejected trial's too."""
    high = float(df["above_best"].max())
    if tf is not None and not tf.empty:
        high = max(high, float((tf["fval"] - df["fval"].min() + OFFSET).max()))
    return OFFSET / 2, 3 * max(high, 3 * OFFSET)


#: The waterfall's hover.
WATERFALL_HOVER = [
    ("start", "@{id}"),
    ("objective", "@{fval}{0.[000]}"),
    ("ended", "@{ended}"),
    ("cluster", "@{cluster}"),
    ("began", "@{began}"),
    ("time (s)", "@{time}{0.0}"),
    ("evaluations", "@{n_fval}"),
]

#: Where dots overlap the hover lists each of them; this puts a line between one and the next.
#: Bokeh drops a ``<style>`` from a tooltip template, and the tooltip is a shadow DOM of its own
#: that the page's styles do not reach, so the hover adds the rule to it as it shows.
SEPARATE = """
for (const host of document.querySelectorAll(".bk-Tooltip")) {
  const root = host.shadowRoot;
  if (root && !root.querySelector("style[data-separate]")) {
    const style = document.createElement("style");
    style.dataset.separate = "";
    style.textContent = ".bk-tooltip-content > div > div:not(:first-child)" +
      " { border-top: 1px solid #999; margin-top: 4px; padding-top: 4px; }";
    root.append(style);
  }
}
"""


def _separate(plot, _element) -> None:
    for tool in plot.state.tools:
        if isinstance(tool, HoverTool):
            tool.callback = CustomJS(code=SEPARATE)


def waterfall_points(
    df: pd.DataFrame, by: str = BY_EXIT, hidden=(), ylim: tuple | None = None
) -> hv.Points:
    """The waterfall's points, which a selection stream reads; a ``hidden`` group's grey."""
    shown = colors(df, by, hidden)
    df = df.assign(color=[shown.get(r, HIDDEN) for r in df["rank"]])
    return hv.Points(
        df,
        ["rank", ABOVE],
        ["id", "fval", "ended", "cluster", "began", "time", "n_fval", "color"],
    ).opts(
        color=hv.dim("color"),
        size=8,
        line_color="white",
        logy=True,
        tools=["hover", "box_select", "tap"],
        hover_tooltips=WATERFALL_HOVER,
        hooks=[_separate],
        active_tools=["box_select"],
        nonselection_alpha=0.3,
        xlabel="start rank (best first)",
        ylabel=ABOVE.label,
        responsive=True,
        frame_height=FRAME,
        ylim=ylim or y_range(df, None),
    )


def waterfall_marks(df: pd.DataFrame) -> hv.Overlay:
    """The waterfall's dashed line at the best and its rings round the warm starts."""
    warm = hv.Points(df.loc[df["warm"], ["rank", "above_best"]], ["rank", ABOVE]).opts(
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


def trace_frame(trace: pd.DataFrame | None, df: pd.DataFrame) -> pd.DataFrame | None:
    """The stored trace, a row per evaluation of a start's objective, with the start's waterfall
    ``rank`` and whether the optimiser kept the step (``accepted``). The trace numbers a start by
    its place in pyPESTO's result list (``start``), not by its id. It records every evaluation, a
    rejected trial step too; Fides keeps a step only where it lowers the objective, so a step
    counts as kept where it lowers the start's best so far."""
    if trace is None or trace.empty:
        return None
    place = df["start"] if "start" in df else df["rank"]
    rank = dict(zip(place.astype(int), df["rank"]))
    t = trace.assign(rank=trace["start"].map(rank)).dropna(subset=["rank", "fval"])
    t = t.assign(rank=t["rank"].astype(int)).sort_values(["rank", "iteration"], kind="stable")
    before = t.groupby("rank")["fval"].cummin().groupby(t["rank"]).shift(1)
    return t.assign(accepted=before.isna() | (t["fval"] < before))


def medians(t: pd.DataFrame, color: dict[int, str], log: bool) -> list:
    """For each colour, the median over its starts of their best so far at each ``_x``, a band
    where the middle half of them lie, and a fainter one from the lowest to the highest: grey
    first, then the worst colour to the best."""
    xs = t["_x"].to_numpy()
    high = float(np.nanmax(xs))
    low = float(np.nanmin(xs[xs > 0])) if (xs > 0).any() else high / 1e3
    grid = np.geomspace(low, high, 200) if log and high > low else np.linspace(low, high, 200)
    by = t.assign(_c=t["rank"].map(color))
    first = by.groupby("_c")["rank"].min()
    order = sorted(first.index, key=lambda c: (c != OTHERS, -first[c]))
    layers = []
    for c in order:
        rows = []
        for _, s in by[by["_c"] == c].groupby("rank"):
            at = np.searchsorted(s["_x"].to_numpy(), grid, side="right") - 1
            rows.append(np.where(at >= 0, s["_y"].to_numpy()[np.clip(at, 0, None)], np.nan))
        with warnings.catch_warnings():  # a grid point no start of the colour has reached
            warnings.simplefilter("ignore", RuntimeWarning)
            q0, q25, q50, q75, q100 = np.nanpercentile(
                np.vstack(rows), [0, 25, 50, 75, 100], axis=0
            )
        ok = ~np.isnan(q50)
        layers += [
            hv.Area((grid[ok], bottom[ok], top[ok]), "x0", [ABOVE, "above_best_high"]).opts(
                color=c, alpha=alpha, line_alpha=0
            )
            for bottom, top, alpha in ((q0, q100, 0.08), (q25, q75, 0.2))
        ] + [
            hv.Curve((grid[ok], q50[ok]), "x0", ABOVE).opts(color=c, line_width=2.5),
        ]
    return layers


def traces(
    tf: pd.DataFrame | None,
    best: float,
    chosen: list[int],
    x: str,
    color: dict[int, str],
    rejected: bool = False,
    median: bool = False,
    log: bool = True,
    ylim: tuple | None = None,
):
    """Each start's objective above the best over ``x`` (``iteration``, counted from 1, or
    ``time``): the starts ``color`` holds, the ``chosen`` ones only where there are any, in their
    colour. Only the steps the optimiser kept unless ``rejected``; with ``median``, each colour's
    median and middle half (:func:`medians`) instead of its starts. On log axes, its y axis is
    :data:`ABOVE` over ``ylim``, the waterfall's. Every start's trace is rows of one segments
    table: a path per start made HoloViews build a dataset for each, 7.8 s of
    Fujita's 300 starts."""
    if tf is None:
        return hv.Curve([]).opts(responsive=True, frame_height=FRAME)
    shown = set(color) & set(chosen) if chosen else set(color)
    t = tf[tf["rank"].isin(shown) & (tf["accepted"] | (rejected and not median))]
    xs = t[x].to_numpy(dtype=float) + (1 if x == "iteration" else 0)
    if log and x == "time":
        xs = np.maximum(xs, np.nanmin(xs[xs > 0]) if (xs > 0).any() else 1e-3)
    ys = t["fval"].to_numpy(dtype=float) - best + OFFSET
    t = t.assign(_x=xs, _y=ys)
    if t.empty:
        layers = [hv.Curve([], "x0", ABOVE)]
    elif median:
        layers = medians(t, color, log)
    else:
        rank = t["rank"].to_numpy()
        joined = rank[1:] == rank[:-1]
        segments = pd.DataFrame(
            {
                "x0": xs[:-1],
                "above_best": ys[:-1],
                "x1": xs[1:],
                "above_best_end": ys[1:],
                "rank": rank[:-1],
            }
        )[joined]
        segments["color"] = segments["rank"].map(color)
        layers = [
            hv.Segments(
                on_top(segments), ["x0", ABOVE, "x1", "above_best_end"], ["rank", "color"]
            ).opts(color=hv.dim("color"), line_width=1.2, alpha=0.8)
        ]
    return hv.Overlay(layers).opts(
        hv.opts.Segments(logx=log, logy=log),
        hv.opts.Curve(logx=log, logy=log),
        hv.opts.Area(logx=log, logy=log),
        hv.opts.Overlay(
            responsive=True,
            frame_height=FRAME,
            ylim=(ylim or (OFFSET / 2, None)) if log else (None, None),
            xlabel="evaluation" if x == "iteration" else "time (s)",
            ylabel=ABOVE.label,
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
    bounds on its estimation scale, 0 at the lower bound and 1 at the upper; its value and the
    bounds on that scale."""
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
            "lower": np.tile(lb, n_starts),
            "upper": np.tile(lb + width, n_starts),
            "above_best": np.repeat(fval - np.nanmin(fval), n_cols),
            "start_position": ((x0 - lb) / width).ravel(),
        }
    )
    df = df[np.isfinite(df["value"])]
    return df if df["start_position"].notna().any() else df.drop(columns="start_position")


#: How far either side of its column a start's dots move, as a share of the columns' spacing.
SIDEWAYS = 0.3

#: Each estimation scale's name above its columns, and what a step along a column means on it.
SCALE_NAMES = {"log10": "log₁₀", "log": "ln", "lin": "lin"}
SCALE_ABOUT = {
    "log10": "Estimated as the log₁₀ of its value: one step up is ten times the value, so a spread "
    "along the column is a spread in fold changes.",
    "log": "Estimated as the natural log of its value: one step up is e (2.7) times the value.",
    "lin": "Estimated as its value itself: equal steps are equal differences, not fold changes.",
}


def scale_entries(df: pd.DataFrame) -> list[dict]:
    """The parameter plot's legend: each estimation scale its columns are on."""
    if df.empty:
        return []
    counts = df.groupby("parameter")["scale"].first().value_counts()
    return [
        K.entry(
            f"{SCALE_NAMES.get(scale, scale)} · {n} parameter{'s' if n > 1 else ''}",
            SCALE_ABOUT.get(scale, ""),
            style.MUTED,
            "",
        )
        for scale, n in counts.items()
    ]


def spread(
    df: pd.DataFrame,
    chosen: list[int],
    show_x0: bool,
    connect: bool,
    color: dict[int, str],
    raw: bool = False,
):
    """Where each start ``color`` holds ended in each parameter's bounds, or, ``raw``, at what
    value on its estimation scale; in its colour, the starts at the best large, the rest small and
    faint. A start keeps one sideways offset in every column, so ``connect`` can draw it as one
    line through the columns, and ``show_x0`` an arrow from where it began to where it ended."""
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
    bounds = df.groupby("parameter")[["lower", "upper"]].first().reindex(order)
    scales = df.groupby("parameter")["scale"].first().reindex(order)
    # The golden ratio's fractions spread the starts evenly across the offsets, the same in
    # every column.
    offset = ((df["rank"].to_numpy() * 0.618034) % 1 - 0.5) * 2 * SIDEWAYS
    df = df.assign(
        column=df["parameter"].map(column).to_numpy() + offset,
        color=df["rank"].map(color).to_numpy(),
        best=at_best.to_numpy(),
    )
    df = on_top(df[df["rank"].isin(color)])
    y = "value" if raw else "position"
    if "start_position" in df:
        df["began"] = (
            df["lower"] + df["start_position"] * (df["upper"] - df["lower"])
            if raw
            else df["start_position"]
        )
    col = hv.Dimension("column", label="estimated parameter")
    if raw:
        edges = np.concatenate([bounds["lower"].to_numpy(), bounds["upper"].to_numpy()])
        middle = np.tile(np.arange(len(order)), 2)
        ticks = pd.DataFrame({"x0": middle - 0.45, "y0": edges, "x1": middle + 0.45, "y1": edges})
        layers = [
            hv.Segments(ticks, ["x0", "y0", "x1", "y1"]).opts(color=style.MUTED, line_width=2)
        ]
        pad = 0.05 * float((bounds["upper"] - bounds["lower"]).max())
        ylim = (float(bounds["lower"].min()) - pad, float(bounds["upper"].max()) + 3 * pad)
        named = float(bounds["upper"].max()) + 1.8 * pad
    else:
        layers = [hv.HLine(edge).opts(color=style.MUTED, line_width=1) for edge in (0, 1)]
        ylim, named = (-0.05, 1.15), 1.09
    # Each column's estimation scale over it (scale_entries says what each means).
    layers.append(
        hv.Labels(
            pd.DataFrame(
                {
                    "column": np.arange(len(order)),
                    "named": named,
                    "scale": [SCALE_NAMES.get(sc, sc) for sc in scales],
                }
            ),
            [col, "named"],
            "scale",
        ).opts(text_font_size="8pt", text_color=style.MUTED)
    )

    def lines(x0, y0, x1, y1, frame) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "x0": np.asarray(x0),
                "y0": np.asarray(y0),
                "x1": np.asarray(x1),
                "y1": np.asarray(y1),
                "color": frame["color"].to_numpy(),
                "rank": frame["rank"].to_numpy(),
                "best": frame["best"].to_numpy(),
            }
        )

    if connect:
        path = df.sort_values(["rank", "column"])
        same = path["rank"].to_numpy()[1:] == path["rank"].to_numpy()[:-1]
        xs, ys = path["column"].to_numpy(), path[y].to_numpy()
        joined = on_top(lines(xs[:-1], ys[:-1], xs[1:], ys[1:], path.iloc[:-1])[same])
        layers.append(
            hv.Segments(joined, ["x0", "y0", "x1", "y1"], ["color", "best"]).opts(
                color=hv.dim("color"),
                alpha=hv.dim("best").categorize({True: 0.6}, default=0.15),
                line_width=1,
            )
        )
    if show_x0 and "began" in df:
        moved = df[df["began"].notna()]
        arrows = on_top(lines(moved["column"], moved["began"], moved["column"], moved[y], moved))
        heads = arrows.assign(y=arrows["y0"] + 0.8 * (arrows["y1"] - arrows["y0"]))
        layers.append(
            hv.Segments(arrows, ["x0", "y0", "x1", "y1"], ["color", "best"]).opts(
                color=hv.dim("color"),
                alpha=hv.dim("best").categorize({True: 0.6}, default=0.2),
                line_width=1,
            )
        )
        for marker, up in (("triangle", True), ("inverted_triangle", False)):
            layers.append(
                hv.Points(heads[(heads["y1"] > heads["y0"]) == up], ["x0", "y"], "color").opts(
                    marker=marker, color=hv.dim("color"), size=6, alpha=0.6
                )
            )
    layers.append(
        hv.Scatter(
            df,
            col,
            [y, "color", "best", "parameter", "id", "value", "position", "scale", "above_best"],
        ).opts(
            color=hv.dim("color"),
            size=hv.dim("best").categorize({True: 8}, default=4),
            alpha=hv.dim("best").categorize({True: 1.0}, default=0.45),
            tools=["hover"],
            hover_tooltips=["parameter", "id", "value", "position", "scale", "above_best"],
        )
    )
    if chosen:
        layers.append(
            hv.Scatter(df[df["rank"].isin(chosen)], col, [y, "id"]).opts(
                color=style.PALETTE[3], size=9, marker="triangle"
            )
        )
    return hv.Overlay(layers).opts(
        hv.opts.Scatter(
            xticks=list(enumerate(order)),
            xrotation=60,
            xlim=(-0.6, len(order) - 0.4),
            ylim=ylim,
            framewise=True,
            width=max(500, 32 * len(order)),
            height=340,
            xlabel="estimated parameter",
            ylabel="value (on its estimation scale)" if raw else "position in bounds",
        )
    )


#: A running fit's phases, in words.
PHASES = {
    "compiling": "compiling the model",
    "queued": "waiting for a worker",
    "running": "running its starts",
}


def duration(seconds: float) -> str:
    """``seconds`` as a person reads them."""
    s = max(int(round(seconds)), 0)
    if s < 60:
        return f"{s} s"
    if s < 3600:
        return f"{s // 60} min {s % 60:02d} s"
    return f"{s // 3600} h {s % 3600 // 60:02d} min"


def live_text(progress: dict, reference: tuple[int, float] | None, now: float) -> str:
    """The running fit in words, as of ``now``: what it is doing and how long the change has run,
    its starts done, running and not begun, its best against ``reference`` (HEAD's version and
    best objective), and how long it has left at the rate its finished starts took."""
    phase = progress.get("phase")
    total, done = progress.get("total") or 0, progress.get("done") or 0
    queued = progress.get("queued") or 0
    # A progress file from before the server published `started`: the phase's start instead.
    started = progress.get("started") or progress.get("since") or now
    lines = [
        f"**Fitting now** · {PHASES.get(phase, phase)} · "
        f"{duration(now - started)} since the change began"
    ]
    if total:
        lines.append(
            f"{done} of {total} starts done · {max(total - done - queued, 0)} running · "
            f"{queued} not begun"
        )
    if progress.get("best") is not None:
        best = f"best so far {progress['best']:.6g}"
        if reference is not None:
            best += f" · v{reference[0]}'s best was {reference[1]:.6g}"
        lines.append(best)
    began = progress.get("began")
    if began and done and done < total:
        waited = (progress.get("waited") or 0.0) + (
            now - progress["since"] if phase == "queued" else 0.0
        )
        left = (now - began - waited) * (total - done) / done
        lines.append(f"about {duration(left)} left · about {duration(now - started + left)} in all")
    return "  \n".join(lines)


def live_view(progress: dict, text: pn.viewable.Viewable) -> pn.Card:
    """The running fit: ``text`` (:func:`live_text`, kept current by the caller), its starts done
    as a bar, and the objectives of those finished so far."""
    total, done = progress.get("total") or 0, progress.get("done") or 0
    items = [text]
    if total:
        items.append(
            pn.indicators.Progress(
                value=done, max=total, bar_color="warning", sizing_mode="stretch_width", height=12
            )
        )
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
        self.colour = pn.widgets.RadioButtonGroup(
            options=[BY_EXIT, BY_CLUSTER, BY_START],
            value=BY_EXIT,
            button_type="light",
            align="center",
        )
        #: The one legend, kept across versions so a colour unticked stays unticked.
        self.legend = K.key("Colours", LEGEND)
        self.axis = pn.widgets.RadioButtonGroup(
            options={"evaluations": "iteration", "time": "time"},
            value="iteration",
            button_type="light",
        )
        self.rejected = pn.widgets.Checkbox(name="rejected steps", value=False)
        self.median = pn.widgets.Checkbox(name="median per colour", value=False)
        self.log = pn.widgets.Checkbox(name="log axes", value=True)
        self.show_x0 = pn.widgets.Checkbox(name="where each start began", value=False)
        self.connect = pn.widgets.Checkbox(name="connect each start's points", value=False)
        self.raw = pn.widgets.Checkbox(name="values, not position in bounds", value=False)
        # A median is over the steps kept; the rejected ones have no median to add to.
        self.median.param.watch(lambda e: setattr(self.rejected, "disabled", e.new), "value")
        self._watching: list = []

    def view(self) -> pn.viewable.Viewable:
        return pn.Column(
            pn.bind(self._live, self.app.param.tick),
            pn.bind(self._version, self.app.param.version),
            sizing_mode="stretch_width",
        )

    def _live(self, _tick):
        """The running fit, drawn anew when its progress file changes; its words also at each
        poll in between (the app's ``clock``), so its times move while no start finishes."""
        store, head = self.app.store, self.app.store.head()
        progress = store.fit_progress()
        if not progress:
            return pn.Spacer(height=0)
        starts = store.starts(head) if head is not None else None
        reference = None
        if starts is not None and starts["fval"].notna().any():
            reference = (head, float(starts["fval"].min()))

        def text(_clock):
            return live_text(store.fit_progress() or progress, reference, time.time())

        return live_view(
            progress,
            pn.pane.Markdown(pn.bind(text, self.app.param.clock), sizing_mode="stretch_width"),
        )

    def _version(self, version):
        store = self.app.store
        for widget, watcher in self._watching:  # the version shown before
            widget.param.unwatch(watcher)
        self._watching = []
        if version is None:
            return pn.pane.Markdown("No version yet.")
        starts = store.starts(version)
        if starts is None or starts.empty:
            return pn.pane.Markdown(
                f"v{version} stored no multistart: it was a restore, a refused change or a "
                "simulate-only pass."
            )
        meta, opt = store.meta(version), store.optimization(version) or {}
        df = waterfall_frame(starts, warm_starts(store, version, self.app.problem))
        hidden = self.legend.param.hidden
        tf = trace_frame(store.trace(version), df)
        ylim = y_range(df, tf)
        points = hv.DynamicMap(
            pn.bind(lambda by, off: waterfall_points(df, by, off, ylim), self.colour, hidden)
        )
        picked = hv.streams.Selection1D(source=points)
        best = float(df["fval"].min())
        spread_df = spread_frame(starts, opt, store.parameters(version))

        def chosen(index):
            return [int(df["rank"].iloc[i]) for i in index or [] if i < len(df)]

        def relegend(*_):
            self.legend.entries = legend(df, starts, self.colour.value, chosen(picked.index))

        relegend()
        self._watching = [(self.colour, self.colour.param.watch(relegend, "value"))]
        picked.param.watch(relegend, "index")

        # A plot keeps the axes it was first drawn with and its layers: what changes either draws
        # the plot anew, the rest redraws it in place. The objective plot keeps its ranges too, so
        # picking starts does not rescale the waterfall, whose y axis it shares (ABOVE).
        def trace_plot(log, median, x):
            return pn.pane.HoloViews(
                hv.DynamicMap(
                    pn.bind(
                        lambda index, by, off, rejected: traces(
                            tf,
                            best,
                            chosen(index),
                            x,
                            colors(df, by, off),
                            rejected,
                            median,
                            log,
                            ylim,
                        ),
                        by=self.colour,
                        off=hidden,
                        rejected=self.rejected,
                    ),
                    streams=[picked],
                ),
                sizing_mode="stretch_width",
            )

        def spread_plot(raw):
            return pn.pane.HoloViews(
                hv.DynamicMap(
                    pn.bind(
                        lambda index, x0, line, by, off: spread(
                            spread_df, chosen(index), x0, line, colors(df, by, off), raw
                        ),
                        x0=self.show_x0,
                        line=self.connect,
                        by=self.colour,
                        off=hidden,
                    ),
                    streams=[picked],
                ),
                linked_axes=False,
            )

        def plot(title: str, about: str, *body) -> pn.Column:
            return pn.Column(
                K.key(title, about),
                *body,
                sizing_mode="stretch_width",
                min_width=420,
                styles={"flex": "1 1 420px"},
            )

        return pn.Column(
            pn.pane.Markdown(summary(meta, opt, starts)),
            pn.Row(
                pn.pane.HTML("<b>Colour by</b>", align="center", margin=(0, 4, 0, 10)),
                self.colour,
            ),
            self.legend,
            # Side by side where there is room, one above the other where there is not.
            pn.FlexBox(
                # A flex basis each: stretch_width alone claims the whole row, so they always
                # wrapped, wide window or not.
                plot(
                    "Waterfall",
                    WATERFALL,
                    pn.pane.HoloViews(waterfall_marks(df) * points, sizing_mode="stretch_width"),
                ),
                plot(
                    "Objective over the run",
                    TRACES,
                    pn.bind(trace_plot, self.log, self.median, self.axis),
                    pn.FlexBox(
                        self.axis, self.rejected, self.median, self.log, align_items="center"
                    ),
                ),
                flex_wrap="wrap",
                sizing_mode="stretch_width",
            ),
            K.key(
                "Where each start ended, per estimated parameter", SPREAD, scale_entries(spread_df)
            ),
            pn.Row(self.show_x0, self.connect, self.raw),
            pn.bind(spread_plot, self.raw),
            sizing_mode="stretch_width",
        )
