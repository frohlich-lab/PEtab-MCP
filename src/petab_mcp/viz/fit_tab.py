"""The Fit tab: the data against a version's simulation, and its residuals.

A version stores its simulation at the measured points only (``simulations.json``), so the line
through them is drawn straight between measurement times: it is the fit as scored, not the model's
trajectory between the points.

**What each experiment does, under its data.** A PEtab v2 experiment is a sequence of periods,
each applying conditions that set targets (a dose, an amount) from its start; a pre-equilibration
period runs to steady state first; and a problem's ``inputs.tsv`` declares inputs that change within
a period (a ramp), set per experiment by condition targets (:mod:`petab_mcp.protocols`). Each
experiment is a card, titled by the inputs that set it apart, its chi2 and its pre-equilibration,
holding its observables on one time axis and, directly under them on the same axis:

* a plot of each quantity that changes over time: a target that steps between periods, its
  pre-equilibration value a dashed stub before the first period, or an ``inputs.tsv`` input as its
  formula gives it from the experiment's targets (the design's input, not the model simulated);
* one line of what the experiment sets and never changes, what sets it apart in bold.

Faint dashed lines mark where each later period starts, in every plot of the card. Experiments are
shown worst-fitted first, a few at a time, each with its worst-fitted observables: a problem with
thousands of experiments or dozens of observables shows a handful, and the pickers the rest.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import holoviews as hv
import numpy as np
import pandas as pd
import panel as pn

from ..problem import experiment_inputs
from . import key as K
from . import style

#: Experiments the tab shows at first.
PAGE = 3

#: Observables a card plots, its worst-fitted first; the picker adds the rest.
TRACKS = 6

#: Quantities that change over time a card plots under its data.
PROTOCOLS = 3

#: Settings the line under a card names before "+N more".
SETTINGS = 10

#: Inputs a card title names before "+N more".
TITLE_INPUTS = 3

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

Dashed vertical lines mark where the experiment changes what it does, such as adding a drug. The
plots under the data show what it sets over time, and the card's last line what it holds fixed.
"""

CHI2_OBSERVABLE = """
How badly each measured quantity is fitted, over all its measurements: each adds the square of its
distance from the model's prediction, in units of its noise σ. A quantity measured n times fits as
well as its noise allows at about χ² = n (hover a bar for its n). A bar far longer than its n is a
quantity the model misses.
"""

CHI2_EXPERIMENT = """
The same per experiment, for the 25 worst-fitted. Pick an experiment in the list above to see its
data and the model's prediction side by side.
"""

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


def _number(value) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


@dataclass
class Protocol:
    """What an experiment sets. ``curves``: each quantity that changes over time, by name, as
    ``(time, value, steps)`` arrays (``steps`` for a value held from each period's start), with its
    pre-equilibration value in ``preeq`` where it has one; ``constant``: each target set and never
    changed, ``target = value``; ``starts``: the later periods' start times."""

    curves: dict[str, tuple[np.ndarray, np.ndarray, bool]] = field(default_factory=dict)
    preeq: dict[str, float] = field(default_factory=dict)
    #: What a curve is made of, for its title: an input's formula and the targets it reads.
    notes: dict[str, str] = field(default_factory=dict)
    constant: dict[str, str] = field(default_factory=dict)
    starts: list[float] = field(default_factory=list)


def protocol(problem, inputs, experiment_id: str, t_end: float) -> Protocol:
    """``experiment_id``'s :class:`Protocol` up to ``t_end``."""
    edf, cdf = problem.experiment_df, problem.condition_df
    out = Protocol()
    if edf is None or not len(edf) or not experiment_id:
        return out
    rows = edf[edf["experimentId"].astype(str) == experiment_id]
    sets: dict[str, list[tuple[str, object]]] = {}
    if cdf is not None and len(cdf):
        for cid, sub in cdf.groupby("conditionId", sort=False):
            sets[str(cid)] = list(zip(sub["targetId"].astype(str), sub["targetValue"]))
    times = sorted({float(t) for t in rows["time"]})
    finite = [t for t in times if math.isfinite(t)]
    t0 = finite[0] if finite else 0.0
    out.starts = finite[1:]
    end = max(t_end, finite[-1] if finite else t0)
    end = end if end > t0 else t0 + 1.0
    applied: dict[str, object] = {}
    by_period: list[tuple[float, dict[str, object]]] = []  # each period's targets in effect
    preeq: dict[str, object] = {}
    for t in times:
        for cid in rows.loc[rows["time"].astype(float) == t, "conditionId"].dropna():
            applied.update(sets.get(str(cid), []))
        if math.isinf(t) and t < 0:
            preeq = dict(applied)
        else:
            by_period.append((t, dict(applied)))
    targets = list(dict.fromkeys(k for _, values in by_period for k in values))

    def steps(target: str) -> tuple[np.ndarray, np.ndarray] | None:
        values = [_number(v.get(target)) for _, v in by_period]
        if any(v is None for v in values):
            return None
        ts = [t for t, _ in by_period] + [end]
        return np.asarray(ts, float), np.asarray(values + values[-1:], float)

    explained: set[str] = set()
    for item in inputs:
        if not set(item.targets) <= set(targets):
            continue
        grid = np.unique(np.concatenate([np.linspace(t0, end, 400), finite]))
        held = {}
        for target in item.targets:
            stepped = steps(target)
            if stepped is None:
                break
            ts, vs = stepped
            held[target] = vs[np.searchsorted(ts, grid, side="right") - 1]
        else:
            out.curves[f"{item.id}(t)"] = (grid, item.evaluate(held, grid), False)
            first = by_period[0][1]
            out.notes[f"{item.id}(t)"] = f"{item.formula}: " + ", ".join(
                f"{t}={_fmt(first.get(t))}" for t in item.targets
            )
            explained |= set(item.targets)
    for target in targets:
        if target in explained:
            continue
        values = [v.get(target) for _, v in by_period]
        first = values[0]
        changes = any(str(v) != str(first) for v in values) or (
            target in preeq and str(preeq[target]) != str(first)
        )
        stepped = steps(target) if changes else None
        if stepped is not None:
            out.curves[target] = (*stepped, True)
            if target in preeq and (value := _number(preeq[target])) is not None:
                out.preeq[target] = value
        elif changes:
            shown = [_fmt(preeq[target])] if target in preeq else []
            out.constant[target] = " → ".join(shown + [_fmt(v) for v in values])
        else:
            out.constant[target] = _fmt(first)
    return out


def protocol_plots(proto: Protocol) -> list:
    """A plot of each quantity that changes over time, to sit under a card's data."""
    t = hv.Dimension("time")
    plots = []
    for name, (ts, vs, stepped) in list(proto.curves.items())[:PROTOCOLS]:
        y = hv.Dimension(f"u_{name}", label=name)
        layers = [
            hv.Curve((ts, vs), t, y).opts(
                color=style.PALETTE[2],
                line_width=2,
                interpolation="steps-post" if stepped else "linear",
            )
        ]
        if name in proto.preeq:
            span = float(ts[-1] - ts[0]) or 1.0
            stub = proto.preeq[name]
            layers.append(
                hv.Curve(([ts[0] - 0.04 * span, ts[0]], [stub, stub]), t, y).opts(
                    color=style.PALETTE[2], line_dash="dashed", line_width=2
                )
            )
        layers += [
            hv.VLine(s).opts(color=style.MUTED, line_dash="dashed", line_width=1)
            for s in proto.starts
        ]
        # From zero, with room above: a dose held constant had a range of nothing, its ticks piled.
        values = [*vs, proto.preeq.get(name, vs[0])]
        low, high = min(0.0, float(np.min(values))), float(np.max(values))
        pad = 0.15 * (high - low) if high > low else max(abs(high), 1.0) * 0.15
        note = proto.notes.get(name)
        plots.append(
            hv.Overlay(layers).opts(
                title=f"{name} = {note}" if note else f"{name}, as the experiment sets it",
                responsive=True,
                height=150,
                ylim=(low - (pad if low < 0 else 0), high + pad),
                show_legend=False,
            )
        )
    return plots


def settings_line(proto: Protocol, apart: str | None) -> str:
    """The targets an experiment sets and never changes, what sets it apart (``apart``, as
    :func:`distinguishing_inputs` names it) in bold."""
    marked = {item.split("=", 1)[0].strip() for item in (apart or "").split(",") if "=" in item}
    items = [f"**{k} = {v}**" if k in marked else f"{k} = {v}" for k, v in proto.constant.items()]
    items.sort(key=lambda text: not text.startswith("**"))
    more = f", +{len(items) - SETTINGS} more" if len(items) > SETTINGS else ""
    if not items:
        return ""
    return "Set throughout: " + ", ".join(items[:SETTINGS]) + more


def observable_plot(
    obs: str,
    sim: pd.DataFrame,
    compare: pd.DataFrame | None,
    starts: list[float],
    error_bars: bool,
    compare_label: str | None,
) -> hv.Overlay:
    y = hv.Dimension(f"y_{obs}", label=obs)
    t = hv.Dimension("time")
    points = sim.sort_values("time")
    layers = []
    if error_bars and points["noise"].notna().any():
        layers.append(
            hv.ErrorBars(
                (points["time"], points["measurement"], points["noise"].fillna(0)),
                kdims=[t],
                vdims=[y, "noise"],
            ).opts(color="#7f7f7f", line_width=1)
        )
    layers.append(
        hv.Scatter(
            points.rename(columns={"measurement": y.name}),
            t,
            [y, "residual_weighted"],
        ).opts(color=style.PALETTE[0], size=6, tools=["hover"])
    )
    line = points.groupby("time", as_index=False)["simulation"].mean()
    layers.append(
        hv.Curve((line["time"], line["simulation"]), t, y, label="simulation").opts(
            color=style.PALETTE[3], line_width=2
        )
    )
    if compare is not None and not compare.empty:
        cline = compare.groupby("time", as_index=False)["simulation"].mean()
        layers.append(
            hv.Curve((cline["time"], cline["simulation"]), t, y, label=compare_label).opts(
                color=style.PALETTE[3], line_dash="dashed", line_width=1.5, alpha=0.7
            )
        )
    layers += [
        hv.VLine(start).opts(color=style.MUTED, line_dash="dashed", line_width=1)
        for start in starts
    ]
    n = len(points)
    total = float((points["residual_weighted"].astype(float) ** 2).sum())
    return hv.Overlay(layers).opts(
        title=f"{obs} · χ² {total:.3g} (n={n})",
        responsive=True,
        height=210,
        show_legend=False,
        framewise=True,
    )


def residual_views(sim: pd.DataFrame, chosen: list[str]) -> pn.viewable.Viewable:
    def per(by: str) -> pd.DataFrame:
        out = chi2(sim, by).reset_index().rename(columns={"residual_weighted": "chi2"})
        return out.assign(n=out[by].map(sim.groupby(by).size()))

    by_obs, by_exp = per("observable_id"), per("experiment_id")
    by_exp["experiment_id"] = by_exp["experiment_id"].replace("", "(no experiment)")
    bars_obs = hv.Bars(by_obs.iloc[::-1], "observable_id", ["chi2", "n"]).opts(
        invert_axes=True,
        responsive=True,
        height=max(180, 26 * len(by_obs) + 80),
        color=style.PALETTE[0],
        tools=["hover"],
        xlabel="",
        ylabel="χ²",
    )
    worst = by_exp.head(25).iloc[::-1]
    bars_exp = hv.Bars(worst, "experiment_id", ["chi2", "n"]).opts(
        invert_axes=True,
        responsive=True,
        height=max(180, 22 * len(worst) + 80),
        color=style.PALETTE[1],
        tools=["hover"],
        xlabel="",
        ylabel="χ²",
    )
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
        from scipy.stats import norm

        q = norm.ppf((np.arange(1, len(r) + 1) - 0.5) / len(r))
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

    observables = [K.entry(o, color=c) for o, c in colors.items()]
    return pn.Column(
        pn.Row(
            keyed(K.key("χ² per observable", CHI2_OBSERVABLE), bars_obs),
            keyed(K.key("χ² per experiment (the worst 25)", CHI2_EXPERIMENT), bars_exp),
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
                "a period starts",
                "The experiment changes what it does here, such as adding or removing a drug.",
                style.MUTED,
                "┆",
            ),
            K.entry(
                "what the experiment sets",
                "Under the data: a quantity the experiment controls over time, such as a drug's "
                "concentration. Dashed before the first period: the value held while the system "
                "settled to steady state before the experiment began (pre-equilibration).",
                style.PALETTE[2],
                "━",
            ),
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
        self.show_periods = pn.widgets.Checkbox(name="what each experiment sets", value=True)
        self.error_bars = pn.widgets.Checkbox(name="noise σ as error bars", value=True)
        self.view_kind = pn.widgets.RadioButtonGroup(
            options=["Data and simulation", "Residuals"],
            value="Data and simulation",
            button_type="light",
        )
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

    def view(self) -> pn.viewable.Viewable:
        controls = pn.Row(
            pn.Column(self.experiments, self.observables, sizing_mode="stretch_width"),
            pn.Column(self.view_kind, self.show_periods, self.error_bars, width=260),
            sizing_mode="stretch_width",
        )
        body = pn.bind(
            self._body,
            self.app.param.version,
            self.app.param.compare,
            self.experiments,
            self.observables,
            self.show_periods,
            self.error_bars,
            self.view_kind,
        )
        # lazy: built when the tab is first shown, not with the page (see app.VizApp.view).
        return pn.Column(
            controls,
            pn.panel(body, lazy=True, sizing_mode="stretch_width"),
            sizing_mode="stretch_width",
        )

    def _body(self, version, compare, experiments, observables, show_periods, error_bars, kind):
        store, problem = self.app.store, self.app.problem
        sim = store.simulations(version) if version is not None else None
        if sim is None:
            return pn.pane.Markdown(
                f"v{version} stored no simulation: its fit failed or it was never fitted."
            )
        if kind == "Residuals":
            return residual_views(sim, experiments)
        other = store.simulations(compare) if compare is not None else None
        compare_label = f"v{compare}" if compare is not None else None
        ends = sim.groupby("experiment_id")["time"].max().to_dict()
        titles = distinguishing_inputs(problem)
        cards = []
        for eid in experiments:
            sub = sim[sim["experiment_id"] == eid]
            proto = protocol(problem, self.app.inputs, eid, float(ends.get(eid, 0.0)))
            _, preeq = periods(problem, eid, float(ends.get(eid, 0.0)))
            starts = proto.starts if show_periods else []
            measured = [o for o in chi2(sub, "observable_id").index if o in set(observables)]
            plots = []
            for obs in measured[:TRACKS]:
                points = sub[sub["observable_id"] == obs]
                theirs = None
                if other is not None:
                    theirs = other[
                        (other["experiment_id"] == eid) & (other["observable_id"] == obs)
                    ]
                plots.append(
                    observable_plot(obs, points, theirs, starts, error_bars, compare_label)
                )
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
                        f"{len(proto.starts) + 1} periods" if proto.starts else None,
                    ],
                )
            )
            if show_periods:
                plots += protocol_plots(proto)
            items = [
                pn.pane.HoloViews(
                    hv.Layout(plots).cols(1).opts(shared_axes=True),
                    linked_axes=False,
                    sizing_mode="stretch_width",
                )
            ]
            notes = []
            if show_periods and (line := settings_line(proto, titles.get(eid))):
                notes.append(line)
            if len(measured) > TRACKS:
                notes.append(
                    f"{len(measured) - TRACKS} more observables measured here, better fitted: "
                    "pick them above."
                )
            if notes:
                items.append(
                    pn.pane.Markdown(
                        "  \n".join(notes),
                        styles={"font-size": "12px"},
                        sizing_mode="stretch_width",
                    )
                )
            cards.append(pn.Card(*items, title=title, sizing_mode="stretch_width"))
        if not cards:
            cards.append(
                pn.pane.Markdown("Nothing measured for the experiments and observables picked.")
            )
        cards.insert(
            0,
            K.key(
                "Data and simulation", FIT, fit_entries(version, compare, error_bars, show_periods)
            ),
        )
        observable_info = self._observable_table(sim)
        design = design_table(problem, experiments, ends)
        details = [
            pn.Card(
                pn.widgets.Tabulator(
                    observable_info, disabled=True, show_index=False, sizing_mode="stretch_width"
                ),
                title="Observables",
                collapsed=True,
                sizing_mode="stretch_width",
            )
        ]
        if not design.empty:
            details.append(
                pn.Card(
                    pn.widgets.Tabulator(
                        design, disabled=True, show_index=False, sizing_mode="stretch_width"
                    ),
                    title="Design of the experiments shown, period by period",
                    collapsed=True,
                    sizing_mode="stretch_width",
                )
            )
        return pn.Column(*cards, *details, sizing_mode="stretch_width")

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
