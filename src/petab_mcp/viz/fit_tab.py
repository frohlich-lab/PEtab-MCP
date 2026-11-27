"""The Fit tab: the data against a version's simulation, and its residuals.

A version stores its simulation at the measured points only (``simulations.json``), so the line
through them is drawn straight between measurement times: it is the fit as scored, not the model's
trajectory between the points.

**What each experiment does, in its plots.** A PEtab v2 experiment is a sequence of periods,
each applying conditions that set targets (a dose, an amount) from its start; a pre-equilibration
period runs to steady state first; and a problem's ``inputs.tsv`` declares inputs that change within
a period (a ramp), set per experiment by condition targets (:mod:`petab_mcp.protocols`). Each
experiment is a card, titled by the inputs that set it apart, its chi2 and its pre-equilibration,
holding its observables on one time axis, each plot's background shaded period by period
(:func:`period_bands`): hovering a band says what the experiment sets and changes in it, and the
formula of each input it sets. Experiments are shown worst-fitted first, a few at a time, each with
its worst-fitted observables: a problem with thousands of experiments or dozens of observables
shows a handful, and the pickers the rest.
"""

from __future__ import annotations

import html
import math
from statistics import NormalDist

import holoviews as hv
import numpy as np
import pandas as pd
import panel as pn
from bokeh.models import ColumnDataSource, DataRange1d, HoverTool, Whisker
from bokeh.palettes import Viridis256
from bokeh.plotting import figure

from ..problem import experiment_inputs
from . import key as K
from . import style

#: Experiments the tab shows at first.
PAGE = 3

#: Observables a card plots, its worst-fitted first; the picker adds the rest.
TRACKS = 6

#: Inputs a card title names before "+N more".
TITLE_INPUTS = 3

#: The height of an observable's plot in a card, and the width below which the next wraps under it.
HEIGHT, WIDTH = 190, 260

#: The Simulations view's groupings: a card per experiment, or a plot per observable over them all.
BY_EXPERIMENT, BY_OBSERVABLE = "experiment", "observable"

#: Its x axes besides an input that sets the experiments apart.
TIME, SIMULATED = "time", "simulated value"

#: Measurement times a dose-response plot names in its legend before naming only the first and last.
TIMES_NAMED = 10

#: Residual weights beyond which a point is more than two noise sigmas off.
BAND = 2.0

FIT = """
One card per experiment, the worst-fitted first; in each, one plot per measured quantity
(observable). Dots are the measurements and their bars the noise the fit assumes for them (σ, one
standard deviation). The line is what the model predicts with its fitted parameters, drawn through
its values at the measured times, so it can cut corners between them.

The χ² in each title adds up, over the plot's n measurements, the square of each dot's distance
from the line in units of its σ. About n is a fit as good as the noise allows; far more means the
model misses the data.

The background is shaded period by period: a new shade where the experiment changes what it does,
such as adding a drug, grey before the first period while it settles to steady state. Hover a band
for what the experiment sets and changes in it; "periods" turns the shading off.

"Group by observable" draws one plot per observable with the experiments picked overlaid, each in
its colour. "x axis" puts the simulated value across instead of time (a perfect fit lies on the
diagonal), or, grouped by observable, an input that differs between the experiments, such as a
dose: a dose-response curve per measurement time.
"""

CHI2_OBSERVABLE = """
How well each measured quantity is fitted, against what measurement noise alone would give. A
quantity's χ² adds up the squared distances of its n measurements from the model's prediction, in
units of their noise σ. Pure noise gives a χ² of about n, with a known spread, so each χ² is shown
as a z-score of that spread: 0 is a typical fit for noise alone, beyond ±2 is rare for it.

The bars are how the quantities fall; the dashed curve is where they would fall if the model and
its noise were right (the fitted parameters absorb a little of the noise, so a right model sits a
little to the left). The furthest out on either side are named; hover a mark for its χ² and n.

Far right: a quantity the model misses. Far left: one fitted closer than its noise allows, so its
σ is likely too large, or the model bends to fit the noise.
"""

CHI2_EXPERIMENT = """
The same per experiment, all its measurements counted together. Far right: an experiment the model
misses; pick it in the list above to see its data and the prediction side by side.
"""

#: How many of the χ² furthest out on either side a distribution names; where there are fewer than
#: ten, how many.
NAMED, NAMED_FEW = 3, 2

RESIDUAL_TIME = """
Each dot is one measurement's distance from the model's prediction in units of its noise σ (a
weighted residual), against the time it was taken, coloured by what was measured; the experiments
picked above only. If the model and its noise are right the dots scatter evenly around 0, about 95
in 100 between the dashed lines at ±2. A run of dots on one side of 0, or a trend over time, is a
feature of the data the model misses at those times.
"""

QQ = """
Whether the misfit looks like the noise the fit assumes. Each dot is a weighted residual, sorted,
against where the same number of sorted draws from a standard normal distribution would lie.

On the dashed diagonal, the noise model holds. Steeper than the diagonal, the misfit is larger than
σ allows: the model misses the data, or σ is too small. Flatter, σ is larger than the misfit needs.
Dots that leave the diagonal only at its ends are a few outliers.
"""


def _fmt(value) -> str:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{v:g}"


def chi2(sim: pd.DataFrame, by: str) -> pd.Series:
    return (
        (sim["residual_weighted"].astype(float) ** 2)
        .groupby(sim[by])
        .sum()
        .sort_values(ascending=False)
    )


def periods(problem, experiment_id: str, t_end: float) -> tuple[pd.DataFrame, str | None]:
    """The periods of ``experiment_id`` after pre-equilibration: ``start``, ``end``, ``condition``
    (the conditions it applies, joined by ``+``), ``sets`` (each target it sets) and ``changed``
    (those whose value differs from the period before, ``target: old → new``). Also the
    pre-equilibration's conditions, ``None`` without one."""
    edf, cdf = problem.experiment_df, problem.condition_df
    empty = pd.DataFrame(columns=["start", "end", "condition", "sets", "changed"])
    if edf is None or not len(edf) or not experiment_id:
        return empty, None
    rows = edf[edf["experimentId"].astype(str) == experiment_id]
    targets: dict[str, list[tuple[str, object]]] = {}
    if cdf is not None and len(cdf):
        for cid, sub in cdf.groupby("conditionId", sort=False):
            targets[str(cid)] = list(zip(sub["targetId"].astype(str), sub["targetValue"]))
    applied: dict[str, object] = {}
    out, preeq = [], None
    times = sorted({float(t) for t in rows["time"]})
    for t in times:
        conditions = [
            str(c) for c in rows.loc[rows["time"].astype(float) == t, "conditionId"].dropna()
        ]
        sets, changed = [], []
        for cid in conditions:
            for target, value in targets.get(cid, []):
                sets.append(f"{target} = {_fmt(value)}")
                if target not in applied:
                    changed.append(f"{target} = {_fmt(value)}")
                elif str(applied[target]) != str(value):
                    changed.append(f"{target}: {_fmt(applied[target])} → {_fmt(value)}")
                applied[target] = value
        if math.isinf(t) and t < 0:
            preeq = " + ".join(conditions) or "(no condition)"
            continue
        out.append(
            {
                "start": t,
                "condition": " + ".join(conditions) or "(none)",
                "sets": "; ".join(sets) or "nothing",
                "changed": "; ".join(changed) or "nothing",
            }
        )
    df = pd.DataFrame(out, columns=["start", "condition", "sets", "changed"])
    df["end"] = list(df["start"].iloc[1:]) + [
        max(t_end, df["start"].iloc[-1] if len(df) else 0) + 1e-9
    ]
    return df, preeq


def distinguishing_inputs(problem) -> dict[str, str]:
    """Per experiment, the inputs that set it apart, as ``target=value`` text: a target's value
    where it differs from the value most experiments give that target (Brannmark's dose-response
    experiments name their dose, the two-step one its dose, second dose and switch time)."""
    try:
        wide = experiment_inputs(problem)
    except Exception:  # noqa: BLE001 - a design the helper cannot read gets plain titles
        return {}
    if wide is None or wide.empty:
        return {}
    text = wide.astype(str)
    usual = {
        c: (m.index[0] if len(m) and m.iloc[0] > 1 else None)
        for c in text.columns
        for m in [text[c].value_counts()]
    }
    out = {}
    for eid, row in wide.iterrows():
        given = [f"{c}={_fmt(v)}" for c, v in row.items() if pd.notna(v) and str(v) != usual[c]]
        extra = f" +{len(given) - TITLE_INPUTS} more" if len(given) > TITLE_INPUTS else ""
        out[str(eid)] = ", ".join(given[:TITLE_INPUTS]) + extra
    return out


def period_bands(table: pd.DataFrame, preeq: str | None, inputs: list[str]) -> pd.DataFrame:
    """The bands that shade an experiment's periods (:func:`periods`' ``table``) and its
    pre-equilibration (``preeq``, a stub before the first period), each with its shade and what a
    hover says (``info``, HTML): its span, condition, what it changes and sets, and the formula of
    each of the experiment's ``inputs``."""
    if table.empty:
        return pd.DataFrame(columns=["left", "right", "fill", "alpha", "info"])
    first, last = float(table["start"].iloc[0]), float(table["end"].iloc[-1])
    formulas = "".join(f"<br>{html.escape(i)}" for i in inputs)

    def lines(text: str) -> str:
        return "<br>&nbsp;&nbsp;".join(html.escape(part) for part in text.split("; "))

    rows = []
    if preeq is not None:
        rows.append(
            {
                "left": first - 0.04 * ((last - first) or 1.0),
                "right": first,
                "fill": style.MUTED,
                "alpha": 0.25,
                "info": f"<b>pre-equilibration</b>, condition {html.escape(preeq)}: run to steady "
                "state before the first period",
            }
        )
    n = len(table)
    for i, r in enumerate(table.itertuples()):
        rows.append(
            {
                "left": float(r.start),
                "right": float(r.end),
                "fill": style.PALETTE[2],
                "alpha": 0.06 if i % 2 == 0 else 0.16,
                "info": f"<b>period {i + 1} of {n}</b>, time {_fmt(r.start)} to {_fmt(r.end)}, "
                f"condition {html.escape(r.condition)}"
                # A first period changes all it sets: once is enough.
                + (f"<br>changes: {lines(r.changed)}" if r.changed != r.sets else "")
                + f"<br>sets: {lines(r.sets)}{formulas}",
            }
        )
    return pd.DataFrame(rows)


def _figure(title: str, x_range, y_label: str, height: int = HEIGHT) -> figure:
    p = figure(
        title=title,
        x_range=x_range,
        height=height,
        sizing_mode="stretch_width",
        tools="pan,wheel_zoom,box_zoom,reset,save",
        x_axis_label="time",
        y_axis_label=y_label,
    )
    p.title.text_font_size = "9pt"
    p.xaxis.ticker.desired_num_ticks = 4
    p.toolbar.autohide = True
    return p


#: What hovering a period's band shows: its ``info`` (:func:`period_bands`), HTML.
BAND_HOVER = '<div style="max-width: 380px; font-size: 12px">@info{safe}</div>'


def observable_figure(
    obs: str,
    sim: pd.DataFrame,
    compare: pd.DataFrame | None,
    bands: pd.DataFrame | None,
    error_bars: bool,
    x_range,
) -> figure:
    """An observable's measurements in one experiment, ± their noise, and the version's
    simulation through them, the compared version's dashed, over the experiment's period
    ``bands`` (:func:`period_bands`)."""
    points = sim.sort_values("time")
    noise = points["noise"].astype(float).fillna(0.0).to_numpy()
    measured = points["measurement"].astype(float).to_numpy()
    source = ColumnDataSource(
        {
            "time": points["time"].astype(float).to_numpy(),
            "measurement": measured,
            "residual": points["residual_weighted"].astype(float).to_numpy(),
            "lower": measured - noise,
            "upper": measured + noise,
        }
    )
    total = float((points["residual_weighted"].astype(float) ** 2).sum())
    p = _figure(f"{obs} · χ² {total:.3g} (n={len(points)})", x_range, obs)
    if bands is not None and not bands.empty:
        # Far past the data either way, and left out of the y range: a band is the plot's height.
        low, high = np.nanmin(source.data["lower"]), np.nanmax(source.data["upper"])
        reach = 10 * ((high - low) or 1.0)
        shade = p.quad(
            left="left",
            right="right",
            bottom=low - reach,
            top=high + reach,
            fill_color="fill",
            fill_alpha="alpha",
            line_alpha=0,
            source=ColumnDataSource(bands),
        )
        p.add_tools(HoverTool(renderers=[shade], tooltips=BAND_HOVER))
    data = []
    if error_bars and points["noise"].notna().any():
        p.add_layout(
            Whisker(source=source, base="time", lower="lower", upper="upper", line_color="#7f7f7f")
        )
        data += [p.scatter("time", y, source=source, alpha=0) for y in ("lower", "upper")]
    dots = p.scatter("time", "measurement", source=source, size=6, color=style.PALETTE[0])
    data.append(dots)
    p.add_tools(
        HoverTool(
            renderers=[dots],
            tooltips=[
                ("time", "@time{0.[00]}"),
                ("measured", "@measurement{0.[0000]}"),
                ("weighted residual", "@residual{0.00}"),
            ],
        )
    )
    line = points.groupby("time")["simulation"].mean()
    data.append(
        p.line(line.index.to_numpy(), line.to_numpy(), color=style.PALETTE[3], line_width=2)
    )
    if compare is not None and not compare.empty:
        theirs = compare.groupby("time")["simulation"].mean()
        p.line(
            theirs.index.to_numpy(),
            theirs.to_numpy(),
            color=style.PALETTE[3],
            line_dash="dashed",
            line_width=1.5,
            alpha=0.7,
        )
    p.y_range = DataRange1d(renderers=data)
    return p


def numeric_inputs(problem, experiments: list[str]) -> pd.DataFrame:
    """The inputs that take more than one numeric value across ``experiments``, one column each
    (:func:`petab_mcp.problem.experiment_inputs`), NaN where an experiment's is not a number."""
    try:
        wide = experiment_inputs(problem)
    except Exception:  # noqa: BLE001 - a design the helper cannot read has no inputs to plot on
        return pd.DataFrame()
    if wide is None or wide.empty:
        return pd.DataFrame()
    wide = wide.reindex([e for e in experiments if e in wide.index])
    wide = wide.apply(pd.to_numeric, errors="coerce")
    return wide.loc[:, [c for c in wide.columns if wide[c].nunique(dropna=True) > 1]]


def _points(sim: pd.DataFrame, color: str | pd.Series) -> ColumnDataSource:
    """A plot's measurements, with their noise interval, simulation and hover fields."""
    noise = sim["noise"].astype(float).fillna(0.0).to_numpy()
    measured = sim["measurement"].astype(float).to_numpy()
    return ColumnDataSource(
        {
            "time": sim["time"].astype(float).to_numpy(),
            "measurement": measured,
            "simulation": sim["simulation"].astype(float).to_numpy(),
            "residual": sim["residual_weighted"].astype(float).to_numpy(),
            "lower": measured - noise,
            "upper": measured + noise,
            "experiment": sim["experiment_id"].astype(str).to_numpy(),
            "color": np.broadcast_to(np.asarray(color, dtype=object), len(sim)).tolist(),
        }
    )


#: What hovering a measurement shows in the plots over several experiments.
POINT_HOVER = [
    ("experiment", "@experiment"),
    ("time", "@time{0.[00]}"),
    ("measured", "@measurement{0.[0000]}"),
    ("simulated", "@simulation{0.[0000]}"),
    ("weighted residual", "@residual{0.00}"),
]


def agreement_figure(obs: str, sim: pd.DataFrame, colors: dict[str, str], error_bars: bool):
    """An observable's measurements against the simulation at the same points, each experiment in
    its colour: on the diagonal where the model matches the measurement."""
    sim = sim.assign(_c=sim["experiment_id"].map(colors).fillna(style.MUTED))
    source = _points(sim, sim["_c"])
    total = float((sim["residual_weighted"].astype(float) ** 2).sum())
    p = figure(
        title=f"{obs} · χ² {total:.3g} (n={len(sim)})",
        height=HEIGHT,
        sizing_mode="stretch_width",
        tools="pan,wheel_zoom,box_zoom,reset,save",
        x_axis_label=f"simulated {obs}",
        y_axis_label=f"measured {obs}",
    )
    p.title.text_font_size = "9pt"
    p.toolbar.autohide = True
    values = np.concatenate([source.data["simulation"], source.data["measurement"]])
    values = values[np.isfinite(values)]
    if values.size:
        low, high = float(values.min()), float(values.max())
        p.line([low, high], [low, high], color=style.MUTED, line_dash="dashed")
    if error_bars and sim["noise"].notna().any():
        p.add_layout(
            Whisker(
                source=source, base="simulation", lower="lower", upper="upper", line_color="#7f7f7f"
            )
        )
    dots = p.scatter("simulation", "measurement", source=source, size=6, color="color")
    p.add_tools(HoverTool(renderers=[dots], tooltips=POINT_HOVER))
    return p


def overlay_figure(
    obs: str,
    sim: pd.DataFrame,
    compare: pd.DataFrame | None,
    colors: dict[str, str],
    error_bars: bool,
):
    """An observable over time in several experiments, each in its colour: measured as dots, its
    simulation as a line, the compared version's dashed."""
    p = _figure(
        f"{obs} · χ² {float((sim['residual_weighted'].astype(float) ** 2).sum()):.3g}",
        DataRange1d(),
        obs,
    )
    dots = []
    for eid, color in colors.items():
        sub = sim[sim["experiment_id"] == eid].sort_values("time")
        if sub.empty:
            continue
        source = _points(sub, color)
        if error_bars and sub["noise"].notna().any():
            p.add_layout(
                Whisker(source=source, base="time", lower="lower", upper="upper", line_color=color)
            )
        line = sub.groupby("time")["simulation"].mean()
        p.line(line.index.to_numpy(), line.to_numpy(), color=color, line_width=2)
        if compare is not None:
            theirs = compare[compare["experiment_id"] == eid].groupby("time")["simulation"].mean()
            if not theirs.empty:
                p.line(
                    theirs.index.to_numpy(),
                    theirs.to_numpy(),
                    color=color,
                    line_dash="dashed",
                    alpha=0.7,
                )
        dots.append(p.scatter("time", "measurement", source=source, size=6, color=color))
    p.add_tools(HoverTool(renderers=dots, tooltips=POINT_HOVER))
    return p


def dose_figure(obs: str, sim: pd.DataFrame, dose: pd.Series, name: str, error_bars: bool):
    """An observable against an input (``dose``, its value by experiment), a curve per measurement
    time in a sequential palette, early dark to late yellow: measured as dots, the simulation as a
    line. The axis is logarithmic where the input spans two decades or more."""
    sim = sim.assign(x=sim["experiment_id"].map(dose)).dropna(subset=["x"])
    xs = sim["x"].to_numpy(dtype=float)
    log = bool(xs.size) and xs.min() > 0 and xs.max() >= 100 * xs.min()
    p = figure(
        title=f"{obs} against {name}",
        height=HEIGHT,
        sizing_mode="stretch_width",
        tools="pan,wheel_zoom,box_zoom,reset,save",
        x_axis_type="log" if log else "linear",
        x_axis_label=name,
        y_axis_label=obs,
    )
    p.title.text_font_size = "9pt"
    p.toolbar.autohide = True
    dots = []
    for t, color in time_colors(sim["time"]).items():
        sub = sim[sim["time"] == t].sort_values("x")
        source = _points(sub, color)
        source.data["x"] = sub["x"].to_numpy(dtype=float)
        if error_bars and sub["noise"].notna().any():
            p.add_layout(
                Whisker(source=source, base="x", lower="lower", upper="upper", line_color=color)
            )
        line = sub.groupby("x")["simulation"].mean()
        p.line(line.index.to_numpy(), line.to_numpy(), color=color, line_width=1.5)
        dots.append(p.scatter("x", "measurement", source=source, size=6, color=color))
    p.add_tools(HoverTool(renderers=dots, tooltips=[(name, "@x"), *POINT_HOVER]))
    return p


def time_colors(times: pd.Series) -> dict[float, str]:
    """Each measurement time's colour, early dark to late yellow (viridis)."""
    unique = sorted(times.astype(float).unique())
    picks = np.linspace(0, len(Viridis256) - 1, max(len(unique), 1)).astype(int)
    return {t: Viridis256[i] for t, i in zip(unique, picks)}


def _normal_pdf(x):
    """The standard normal's density: scipy's import cost the residuals view a second."""
    return np.exp(-np.square(x) / 2) / math.sqrt(2 * math.pi)


def chi2_z(chi2, n) -> np.ndarray:
    """Each χ² as a z-score of the χ² distribution with ``n`` degrees of freedom (Wilson and
    Hilferty's cube-root approximation): about standard normal where the model and its noise are
    right, whatever ``n``. ``n`` is the number of measurements, the fitted parameters not taken
    off: no observable or experiment owns one alone."""
    chi2, n = np.asarray(chi2, dtype=float), np.asarray(n, dtype=float)
    c = 2 / (9 * n)
    return ((chi2 / n) ** (1 / 3) - (1 - c)) / np.sqrt(c)


def chi2_distribution(frame: pd.DataFrame, name: str, color: str) -> hv.Overlay:
    """``frame``'s χ² (columns ``name``, ``chi2`` and ``n``) as z-scores (:func:`chi2_z`): their
    histogram, the standard normal they would follow, a mark for each, and the furthest out on
    either side named."""
    df = frame.assign(z=chi2_z(frame["chi2"], frame["n"])).sort_values("z")
    low, high = min(float(df["z"].min()), -3.5), max(float(df["z"].max()), 3.5)
    edges = np.linspace(low, high, int(np.clip(2 * np.sqrt(len(df)), 8, 40)) + 1)
    density, _ = np.histogram(df["z"], bins=edges, density=True)
    xs = np.linspace(low, high, 300)
    top = max(float(density.max()), _normal_pdf(0.0))
    k = NAMED if len(df) >= 10 else NAMED_FEW
    named = df if len(df) <= 2 * k else pd.concat([df.head(k), df.tail(k)])
    # Stacked above the plot, each a level of its own on its side, so long names do not overlap.
    level = named.groupby(named["z"] >= 0).cumcount().to_numpy()
    named = named.assign(y=top * (1.12 + 0.16 * level), y0=0.0, z1=named["z"])
    z = hv.Dimension("z", label="χ² as a z-score (0: what noise alone gives)")
    layers = [
        hv.Histogram((edges, density), kdims=[z], vdims=["density"]).opts(
            color=color, alpha=0.45, line_color=None
        ),
        hv.Curve((xs, _normal_pdf(xs)), z, "density").opts(
            color=style.MUTED, line_dash="dashed", line_width=2
        ),
        hv.Segments(named, [z, "y0", "z1", "y"]).opts(
            color=style.MUTED, line_dash="dotted", line_width=1
        ),
        hv.Scatter(df.assign(mark=-0.04 * top), z, ["mark", name, "chi2", "n"]).opts(
            color=color, marker="square", size=7, tools=["hover"]
        ),
    ]
    for right in (False, True):
        side = named[(named["z"] >= 0) == right]
        layers.append(
            hv.Labels(side, [z, "y"], name).opts(
                text_font_size="8pt",
                text_align="right" if right else "left",
                text_color=style.MUTED,
            )
        )
    return hv.Overlay(layers).opts(
        hv.opts.Histogram(
            responsive=True,
            height=300,
            ylim=(-0.08 * top, top * (1.3 + 0.16 * k)),
            xlim=(low - 0.3, high + 0.3),
            ylabel="density",
        )
    )


def residual_views(sim: pd.DataFrame, chosen: list[str]) -> pn.viewable.Viewable:
    def per(by: str) -> pd.DataFrame:
        out = chi2(sim, by).reset_index().rename(columns={"residual_weighted": "chi2"})
        return out.assign(n=out[by].map(sim.groupby(by).size()))

    by_obs, by_exp = per("observable_id"), per("experiment_id")
    by_exp["experiment_id"] = by_exp["experiment_id"].replace("", "(no experiment)")
    shown = sim[sim["experiment_id"].isin(chosen)] if chosen else sim
    colors = style.categorical(sorted(sim["observable_id"].unique()))
    scatter = hv.Scatter(
        shown,
        "time",
        ["residual_weighted", "observable_id", "experiment_id", "measurement", "simulation"],
    ).opts(
        color="observable_id",
        cmap=colors,
        size=6,
        tools=["hover"],
        responsive=True,
        height=300,
        show_legend=False,
        ylabel="weighted residual",
    )
    band = (hv.HLine(BAND) * hv.HLine(-BAND) * hv.HLine(0)).opts(
        hv.opts.HLine(color=style.MUTED, line_dash="dashed", line_width=1)
    )
    r = np.sort(sim["residual_weighted"].dropna().astype(float).to_numpy())
    if len(r):
        normal = NormalDist()
        q = np.array([normal.inv_cdf((i + 0.5) / len(r)) for i in range(len(r))])
        lim = float(max(abs(r).max(), abs(q).max()))
        qq = (
            hv.Scatter((q, r), "normal quantile", "weighted residual").opts(
                color=style.PALETTE[0], size=5
            )
            * hv.Curve([(-lim, -lim), (lim, lim)], "normal quantile", "weighted residual").opts(
                color=style.MUTED, line_dash="dashed"
            )
        ).opts(responsive=True, height=300)
    else:
        qq = hv.Curve([]).opts(responsive=True, height=300)

    def keyed(header: K.Key, plot) -> pn.Column:
        return pn.Column(
            header,
            pn.pane.HoloViews(plot, linked_axes=False, sizing_mode="stretch_width"),
            sizing_mode="stretch_width",
        )

    def expected(what: str, colour: int) -> list[dict]:
        return [
            K.entry(
                what,
                f"How the {what}' χ² fall: one mark each under the axis, bars for how many.",
                style.PALETTE[colour],
                "■",
            ),
            K.entry(
                "if the model were right",
                "The standard normal: where the z-scores would fall if the model and its noise "
                "were right.",
                style.MUTED,
                "╍",
            ),
        ]

    observables = [K.entry(o, color=c) for o, c in colors.items()]
    return pn.Column(
        pn.Row(
            keyed(
                K.key("χ² per observable", CHI2_OBSERVABLE, expected("observables", 0)),
                chi2_distribution(by_obs, "observable_id", style.PALETTE[0]),
            ),
            keyed(
                K.key("χ² per experiment", CHI2_EXPERIMENT, expected("experiments", 1)),
                chi2_distribution(by_exp, "experiment_id", style.PALETTE[1]),
            ),
            sizing_mode="stretch_width",
        ),
        pn.Row(
            keyed(K.key("Weighted residual over time", RESIDUAL_TIME, observables), scatter * band),
            keyed(
                K.key(
                    "Residuals against a standard normal",
                    QQ,
                    [
                        K.entry("weighted residual", color=style.PALETTE[0]),
                        K.entry("where noise alone puts them", mark="╍"),
                    ],
                ),
                qq,
            ),
            sizing_mode="stretch_width",
        ),
        sizing_mode="stretch_width",
    )


def fit_entries(version, compare, error_bars: bool, show_periods: bool) -> list[dict]:
    out = [
        K.entry(
            "measured" + (" ± σ" if error_bars else ""),
            "One measurement" + ("; its bar is ± its noise σ." if error_bars else "."),
            style.PALETTE[0],
        ),
        K.entry(
            f"v{version}",
            "The model's prediction at the measured times, with this version's fitted parameters.",
            style.PALETTE[3],
            "━",
        ),
    ]
    if compare is not None:
        out.append(
            K.entry(
                f"v{compare}", "The version compared with, for reference.", style.PALETTE[3], "╍"
            )
        )
    if show_periods:
        out += [
            K.entry(
                "a period",
                "The experiment's periods, shaded in turn: a new shade where it changes what it "
                "does, such as adding a drug. Hover a band for what it sets and changes there.",
                style.PALETTE[2],
                "▮",
            ),
            K.entry(
                "pre-equilibration",
                "Before the first period, the system run to steady state under the experiment's "
                "pre-equilibration condition.",
                style.MUTED,
                "▮",
            ),
        ]
    return out


def scheme_entries(
    group: str, x: str, colors: dict[str, str], labels: dict[str, str], times: pd.Series | None
) -> list[dict]:
    """The legend of the views other than an experiment's data over time (:func:`fit_entries`)."""
    out = []
    if group == BY_OBSERVABLE and x in (TIME, SIMULATED):
        out += [K.entry(labels.get(e, e) or "(no experiment)", e, c) for e, c in colors.items()]
    if x == SIMULATED:
        out.append(
            K.entry(
                "where they agree",
                "The diagonal: a measurement on it is exactly what the model predicts; above it "
                "the model predicts too little, below it too much.",
                style.MUTED,
                "╌",
            )
        )
    elif x not in (TIME, SIMULATED) and times is not None:
        shown = time_colors(times)
        named = list(shown.items())
        if len(named) > TIMES_NAMED:
            named = [named[0], named[-1]]
        out += [
            K.entry(f"t = {_fmt(t)}", "Measured at this time (dots) and simulated (line).", c)
            for t, c in named
        ]
    return out


def design_table(problem, chosen: list[str], ends: dict[str, float]) -> pd.DataFrame:
    rows = []
    for eid in chosen:
        df, preeq = periods(problem, eid, ends.get(eid, 0.0))
        if preeq:
            rows.append(
                {
                    "experiment": eid,
                    "period start": "pre-equilibration",
                    "condition": preeq,
                    "sets": "",
                    "changed": "",
                }
            )
        rows += [
            {
                "experiment": eid,
                "period start": _fmt(r.start),
                "condition": r.condition,
                "sets": r.sets,
                "changed": r.changed,
            }
            for r in df.itertuples()
        ]
    return pd.DataFrame(
        rows, columns=["experiment", "period start", "condition", "sets", "changed"]
    )


def _wrapped(plots: list) -> pn.FlexBox:
    """Bokeh's own figures, side by side as the window allows: built through HoloViews, a card's
    four plots took most of the tab's load."""
    return pn.FlexBox(
        *[
            pn.Column(
                pn.pane.Bokeh(plot, sizing_mode="stretch_width"),
                sizing_mode="stretch_width",
                min_width=WIDTH,
                styles={"flex": f"1 1 {WIDTH}px"},
            )
            for plot in plots
        ],
        flex_wrap="wrap",
        sizing_mode="stretch_width",
    )


class FitTab:
    """The tab, reading the app's ``version`` and ``compare``."""

    def __init__(self, app) -> None:
        self.app = app
        self.observables = pn.widgets.MultiChoice(
            name="Observables", options=[], value=[], sizing_mode="stretch_width"
        )
        self.experiments = pn.widgets.MultiChoice(
            name="Experiments (worst-fitted first)",
            options=[],
            value=[],
            sizing_mode="stretch_width",
        )
        self.show_periods = pn.widgets.Checkbox(name="periods", value=True)
        self.error_bars = pn.widgets.Checkbox(name="noise σ as error bars", value=True)
        self.group = pn.widgets.RadioButtonGroup(
            options=[BY_EXPERIMENT, BY_OBSERVABLE], value=BY_EXPERIMENT, button_type="light"
        )
        self.x_axis = pn.widgets.Select(options=[TIME, SIMULATED], value=TIME, width=200)
        self.group.param.watch(self._axes, "value")
        self.experiments.param.watch(self._axes, "value")
        self.x_axis.param.watch(self._periods_apply, "value")
        self.group.param.watch(self._periods_apply, "value")
        self._version_seen: int | None = None
        self.app.param.watch(self._reset, "version")
        self._reset()

    def _reset(self, *_):
        """Fill the pickers for the version: every observable, the worst-fitted experiments."""
        version = self.app.version
        sim = self.app.store.simulations(version) if version is not None else None
        if sim is None:
            self.observables.options, self.experiments.options = [], []
            return
        totals = chi2(sim, "experiment_id")
        exps = list(totals.index)
        obs = sorted(sim["observable_id"].unique())
        inputs = distinguishing_inputs(self.app.problem)
        labels = {
            " · ".join(
                filter(None, [e or "(no experiment)", inputs.get(e), f"χ² {totals[e]:.3g}"])
            ): e
            for e in exps
        }
        keep = [v for v in self.experiments.value if v in labels.values()]
        self.experiments.options = labels
        self.experiments.value = keep or exps[:PAGE]
        self.observables.options = obs
        self.observables.value = [o for o in self.observables.value if o in obs] or obs

    def _axes(self, *_):
        """The x axes the grouping and the experiments picked allow: an input only over several
        experiments, where it differs."""
        options = [TIME, SIMULATED]
        if self.group.value == BY_OBSERVABLE:
            options += list(numeric_inputs(self.app.problem, self.experiments.value).columns)
        self.x_axis.options = options
        if self.x_axis.value not in options:
            self.x_axis.value = TIME

    def _periods_apply(self, *_):
        """Periods shade only an experiment's data over time."""
        self.show_periods.disabled = not (
            self.group.value == BY_EXPERIMENT and self.x_axis.value == TIME
        )

    def view(self) -> pn.viewable.Viewable:
        simulations = pn.bind(
            self._simulations,
            self.app.param.version,
            self.app.param.compare,
            self.experiments,
            self.observables,
            self.show_periods,
            self.error_bars,
            self.group,
            self.x_axis,
        )
        residuals = pn.bind(self._residuals, self.app.param.version, self.experiments)
        # lazy: built when the tab is first shown, not with the page (see app.VizApp.view); the
        # sub-tab not shown is not built either.
        return pn.Column(
            pn.Row(self.experiments, self.observables, sizing_mode="stretch_width"),
            pn.Tabs(
                (
                    "Simulations",
                    pn.Column(
                        pn.Row(
                            pn.pane.HTML("<b>Group by</b>", align="center", margin=(0, 4, 0, 10)),
                            self.group,
                            pn.pane.HTML("<b>x axis</b>", align="center", margin=(0, 4, 0, 16)),
                            self.x_axis,
                            self.show_periods,
                            self.error_bars,
                            align="center",
                        ),
                        pn.panel(simulations, lazy=True, sizing_mode="stretch_width"),
                        sizing_mode="stretch_width",
                    ),
                ),
                ("Residuals", pn.panel(residuals, lazy=True, sizing_mode="stretch_width")),
                dynamic=True,
                sizing_mode="stretch_width",
            ),
            sizing_mode="stretch_width",
        )

    def _stored(self, version) -> pd.DataFrame | pn.pane.Markdown:
        sim = self.app.store.simulations(version) if version is not None else None
        if sim is None:
            return pn.pane.Markdown(
                f"v{version} stored no simulation: its fit failed or it was never fitted."
            )
        return sim

    def _residuals(self, version, experiments):
        sim = self._stored(version)
        return residual_views(sim, experiments) if isinstance(sim, pd.DataFrame) else sim

    def _simulations(
        self, version, compare, experiments, observables, show_periods, error_bars, group, x
    ):
        store, problem = self.app.store, self.app.problem
        sim = self._stored(version)
        if not isinstance(sim, pd.DataFrame):
            return sim
        other = store.simulations(compare) if compare is not None else None
        ends = sim.groupby("experiment_id")["time"].max().to_dict()
        titles = distinguishing_inputs(problem)
        show_periods = show_periods and x == TIME
        if group == BY_OBSERVABLE:
            body = self._by_observable(sim, other, experiments, observables, error_bars, x)
        else:
            body = self._by_experiment(
                sim, other, experiments, observables, show_periods, error_bars, x, titles, ends
            )
        if x == TIME and group == BY_EXPERIMENT:
            entries = fit_entries(version, compare, error_bars, show_periods)
        else:
            colors = style.categorical(experiments)
            shown = sim[sim["experiment_id"].isin(experiments)]
            entries = scheme_entries(group, x, colors, titles, shown["time"])
        observable_info = self._observable_table(sim)
        design = design_table(problem, experiments, ends)
        details = [
            pn.Card(
                pn.pane.DataFrame(observable_info, index=False, sizing_mode="stretch_width"),
                title="Observables",
                collapsed=True,
                sizing_mode="stretch_width",
            )
        ]
        if not design.empty:
            details.append(
                pn.Card(
                    pn.pane.DataFrame(design, index=False, sizing_mode="stretch_width"),
                    title="Design of the experiments shown, period by period",
                    collapsed=True,
                    sizing_mode="stretch_width",
                )
            )
        return pn.Column(
            K.key("Data and simulation", FIT, entries),
            *body,
            *details,
            sizing_mode="stretch_width",
        )

    def _by_observable(self, sim, other, experiments, observables, error_bars, x):
        """A plot per observable over the experiments picked."""
        colors = style.categorical(experiments)
        shown = sim[sim["experiment_id"].isin(experiments)]
        theirs = other[other["experiment_id"].isin(experiments)] if other is not None else None
        dose = numeric_inputs(self.app.problem, experiments)
        plots = []
        for obs in [o for o in chi2(shown, "observable_id").index if o in set(observables)]:
            points = shown[shown["observable_id"] == obs]
            if x == SIMULATED:
                plots.append(agreement_figure(obs, points, colors, error_bars))
            elif x in dose.columns:
                plots.append(dose_figure(obs, points, dose[x], x, error_bars))
            else:
                compare = theirs[theirs["observable_id"] == obs] if theirs is not None else None
                plots.append(overlay_figure(obs, points, compare, colors, error_bars))
        if not plots:
            return [
                pn.pane.Markdown("Nothing measured for the experiments and observables picked.")
            ]
        return [_wrapped(plots)]

    def _by_experiment(
        self, sim, other, experiments, observables, show_periods, error_bars, x, titles, ends
    ):
        """A card per experiment, a plot per observable in it."""
        problem = self.app.problem
        cards = []
        for eid in experiments:
            sub = sim[sim["experiment_id"] == eid]
            table, preeq = periods(problem, eid, float(ends.get(eid, 0.0)))
            bands = None
            if show_periods:
                # An input's formula where the experiment sets every target it reads.
                targets = {
                    item.split(" = ", 1)[0] for text in table["sets"] for item in text.split("; ")
                }
                formulas = [
                    f"{i.id}(t) = {i.formula}" for i in self.app.inputs if set(i.targets) <= targets
                ]
                bands = period_bands(table, preeq, formulas)
            measured = [o for o in chi2(sub, "observable_id").index if o in set(observables)]
            # One time axis for the card's plots, so they zoom and pan together.
            x_range = DataRange1d()
            plots = []
            for obs in measured[:TRACKS]:
                points = sub[sub["observable_id"] == obs]
                theirs = None
                if other is not None:
                    theirs = other[
                        (other["experiment_id"] == eid) & (other["observable_id"] == obs)
                    ]
                if x == SIMULATED:
                    plots.append(agreement_figure(obs, points, {eid: style.PALETTE[0]}, error_bars))
                else:
                    plots.append(observable_figure(obs, points, theirs, bands, error_bars, x_range))
            if not plots:
                continue
            total = float((sub["residual_weighted"].astype(float) ** 2).sum())
            title = " · ".join(
                filter(
                    None,
                    [
                        eid or "(no experiment)",
                        titles.get(eid),
                        f"χ² {total:.3g}",
                        f"pre-equilibrated in {preeq}" if preeq else None,
                        f"{len(table)} periods" if len(table) > 1 else None,
                    ],
                )
            )
            items = [_wrapped(plots)]
            if len(measured) > TRACKS:
                items.append(
                    pn.pane.Markdown(
                        f"{len(measured) - TRACKS} more observables measured here, better fitted: "
                        "pick them above.",
                        styles={"font-size": "12px"},
                        sizing_mode="stretch_width",
                    )
                )
            cards.append(pn.Card(*items, title=title, sizing_mode="stretch_width", margin=(5, 0)))
        if not cards:
            cards.append(
                pn.pane.Markdown("Nothing measured for the experiments and observables picked.")
            )
        return cards

    def _observable_table(self, sim: pd.DataFrame) -> pd.DataFrame:
        odf = self.app.problem.observable_df
        stats = (
            sim.assign(chi2=sim["residual_weighted"].astype(float) ** 2)
            .groupby("observable_id")
            .agg(points=("chi2", "size"), chi2=("chi2", "sum"))
        )
        rows = []
        for oid in stats.index:
            row = odf.loc[oid] if odf is not None and oid in odf.index else {}
            rows.append(
                {
                    "observable": oid,
                    "formula": row.get("observableFormula", ""),
                    "noise": row.get("noiseFormula", ""),
                    "distribution": row.get("noiseDistribution", "normal"),
                    "points": int(stats.at[oid, "points"]),
                    "χ²": round(stats.at[oid, "chi2"], 3),
                }
            )
        return pd.DataFrame(rows)
