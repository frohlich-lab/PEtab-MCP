"""The Fit tab: the data against a version's simulation, and its residuals.

A version stores its simulation at the measured points only (``simulations.json``), so the line
through them is drawn straight between measurement times: it is the fit as scored, not the model's
trajectory between the points.

**Experiments and their periods, without a wall of text.** A PEtab v2 experiment is a sequence of
periods, each applying one or more conditions, which set targets (a dose, an amount) from its start
time on; a pre-equilibration period runs to steady state first. Most of that is the same in every
experiment, so the tab says only what tells them apart, in layers:

* each experiment is a card whose title names the inputs that set it apart (a target most
  experiments set alike is not named), its chi2, and whether it is pre-equilibrated;
* inside, its observables are tracks stacked on one time axis, zoomed and panned together, under a
  slim track with one segment per period coloured by its condition (one colour per condition
  across every card), whose hover says what the condition sets and what changed from the period
  before;
* faint dashed lines mark where each later period starts in every observable's track;
* the whole design, period by period, and the observables' formulas sit in collapsed cards.

An experiment with one period and no pre-equilibration has no strip at all. Experiments are shown
worst-fitted first, a page at a time: a problem with thousands of experiments shows a handful.
"""

from __future__ import annotations

import math

import holoviews as hv
import numpy as np
import pandas as pd
import panel as pn

from ..problem import experiment_inputs
from . import style

#: Experiments a page shows at first.
PAGE = 4

#: Inputs a card title names before "+N more".
TITLE_INPUTS = 3

#: Residual weights beyond which a point is more than two noise sigmas off.
BAND = 2.0


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


def period_strip(df: pd.DataFrame, colors: dict[str, str]) -> hv.Overlay:
    frame = df.assign(y0=0.0, y1=1.0).rename(columns={"start": "time", "end": "time_end"})
    rects = hv.Rectangles(
        frame[["time", "y0", "time_end", "y1", "condition", "changed", "sets"]],
        kdims=["time", "y0", "time_end", "y1"],
        vdims=["condition", "changed", "sets"],
    ).opts(
        color="condition",
        cmap=colors,
        line_color="white",
        line_width=2,
        alpha=0.85,
        tools=["hover"],
        height=50,
        responsive=True,
        yaxis=None,
        xaxis=None,
        toolbar=None,
        show_legend=False,
        title="",
    )
    # A segment says what it changes where it has room, its condition where it has a little, and
    # nothing where it has none; the hover has it all.
    share = (df["end"] - df["start"]) / max(float(df["end"].max() - df["start"].min()), 1e-12)
    text = [
        (f"{c}: {ch}" if ch != "nothing" else c)[:60] if w > 0.25 else c if w > 0.08 else ""
        for c, ch, w in zip(df["condition"], df["changed"], share)
    ]
    labels = hv.Labels(
        df.assign(x=(df["start"] + df["end"]) / 2, y=0.5, text=text), ["x", "y"], "text"
    ).opts(text_font_size="9pt", text_color="white")
    return rects * labels


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
    by_obs = chi2(sim, "observable_id").reset_index().rename(columns={"residual_weighted": "chi2"})
    by_exp = chi2(sim, "experiment_id").reset_index().rename(columns={"residual_weighted": "chi2"})
    by_exp["experiment_id"] = by_exp["experiment_id"].replace("", "(no experiment)")
    bars_obs = hv.Bars(by_obs.iloc[::-1], "observable_id", "chi2").opts(
        title="χ² per observable",
        invert_axes=True,
        responsive=True,
        height=max(180, 26 * len(by_obs) + 80),
        color=style.PALETTE[0],
        tools=["hover"],
        xlabel="",
        ylabel="χ²",
    )
    worst = by_exp.head(25).iloc[::-1]
    bars_exp = hv.Bars(worst, "experiment_id", "chi2").opts(
        title="χ² per experiment (worst 25)",
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
        title="Weighted residual over time (the experiments shown)",
        show_legend=True,
        legend_position="right",
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
        ).opts(title="Weighted residuals against a standard normal", responsive=True, height=300)
    else:
        qq = hv.Curve([]).opts(title="No residuals")
    return pn.Column(
        pn.Row(
            pn.pane.HoloViews(bars_obs, linked_axes=False, sizing_mode="stretch_width"),
            pn.pane.HoloViews(bars_exp, linked_axes=False, sizing_mode="stretch_width"),
            sizing_mode="stretch_width",
        ),
        pn.Row(
            pn.pane.HoloViews(scatter * band, linked_axes=False, sizing_mode="stretch_width"),
            pn.pane.HoloViews(qq, linked_axes=False, sizing_mode="stretch_width"),
            sizing_mode="stretch_width",
        ),
        sizing_mode="stretch_width",
    )


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
        self.show_periods = pn.widgets.Checkbox(name="periods", value=True)
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
        return pn.Column(controls, body, sizing_mode="stretch_width")

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
        condition_colors = style.categorical(
            []
            if problem.experiment_df is None
            else problem.experiment_df["conditionId"].astype(str)
        )
        titles = distinguishing_inputs(problem)
        cards = []
        for eid in experiments:
            sub = sim[sim["experiment_id"] == eid]
            table, preeq = periods(problem, eid, float(ends.get(eid, 0.0)))
            starts = list(table["start"].iloc[1:]) if show_periods else []
            plots = []
            for obs in observables:
                points = sub[sub["observable_id"] == obs]
                if points.empty:
                    continue
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
                        f"{len(table)} periods" if len(table) > 1 else None,
                    ],
                )
            )
            if show_periods and (len(table) > 1 or preeq):
                plots.insert(0, period_strip(table, condition_colors))
            tracks = hv.Layout(plots).cols(1).opts(shared_axes=True)
            cards.append(
                pn.Card(
                    pn.pane.HoloViews(tracks, linked_axes=False, sizing_mode="stretch_width"),
                    title=title,
                    sizing_mode="stretch_width",
                    collapsed=len(cards) >= PAGE,
                )
            )
        if not cards:
            cards.append(
                pn.pane.Markdown("Nothing measured for the experiments and observables picked.")
            )
        key = (
            f"<span style='color:{style.PALETTE[0]}'>●</span> measured"
            + (" ± noise σ" if error_bars else "")
            + f" &nbsp; <span style='color:{style.PALETTE[3]}'>━</span> v{version}, simulated at"
            " the measured times"
            + (
                f" &nbsp; <span style='color:{style.PALETTE[3]}'>╍</span> v{compare}"
                if compare is not None
                else ""
            )
            + (" &nbsp; ┆ a period starts" if show_periods else "")
        )
        cards.insert(0, pn.pane.HTML(key, styles={"font-size": "12px", "color": "#555"}))
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
