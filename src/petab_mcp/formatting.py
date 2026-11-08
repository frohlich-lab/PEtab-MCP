"""Helpers to turn PEtab/pandas objects into compact, JSON-serializable records or TSV text.

MCP tool results must be plain JSON. The problem's tables are pandas DataFrames (``None``
when a table is absent), and numpy/sympy/enum values need stringifying. These helpers
centralize that conversion so every tool returns clean data.
"""

from __future__ import annotations

import math
from typing import Any

import pandas as pd
from petab.v2.C import (
    CONDITION_ID,
    EXPERIMENT_ID,
    MEASUREMENT,
    OBSERVABLE_ID,
    TARGET_ID,
    TARGET_VALUE,
    TIME,
)


def finite_float(value: Any) -> float | None:
    """Coerce ``value`` to a finite ``float``, or ``None`` (missing / NaN / Inf).

    The single place that turns a possibly-missing numeric cell into a JSON-safe number:
    JSON has no NaN/Inf, and downstream stats treat them as absent. Shared by :func:`_clean`
    and the per-point record builders so the coercion rule lives in one spot.
    """
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return None if (math.isnan(f) or math.isinf(f)) else f


def _clean(value: Any) -> Any:
    """Coerce a single cell into a JSON-friendly value."""
    if value is None:
        return None
    if isinstance(value, float):
        # JSON has no NaN/Inf; represent them as None so results stay valid.
        return finite_float(value)
    if isinstance(value, (str, int, bool)):
        return value
    # pandas NA, sympy exprs, enums, numpy scalars, ...
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(value, "item"):  # numpy scalar
        try:
            return _clean(value.item())
        except (ValueError, TypeError):
            pass
    return str(value)


def df_to_records(df: "pd.DataFrame | None") -> list[dict[str, Any]]:
    """Convert a DataFrame (or ``None``) to a list of cleaned record dicts.

    A NAMED index is preserved under its own name (the observable and parameter tables are
    indexed by their primary-key id — ``observableId``, ``parameterId``). An unnamed index
    carries only positional offsets and is dropped, so boolean-filtering a measurement table
    cannot leak a spurious id field.
    """
    if df is None or len(df) == 0:
        return []
    records: list[dict[str, Any]] = []
    index_name = df.index.name
    # Only surface the index when it is a meaningful, *named* primary key (the
    # observable/parameter tables). Unnamed indexes — including the Int64Index that
    # boolean-filtering leaves behind on a measurement table — carry only positional
    # offsets and must not become a spurious `id` field.
    use_index = index_name is not None
    for idx, row in df.iterrows():
        record: dict[str, Any] = {}
        if use_index:
            record[index_name] = _clean(idx)
        for col, val in row.items():
            record[str(col)] = _clean(val)
        records.append(record)
    return records


def df_to_tsv(df: "pd.DataFrame | None") -> str:
    """A table as TSV text, a missing cell empty and a named index its first column, as in
    :func:`df_to_records`; ``"(none)"`` for an absent or empty table."""
    if df is None or len(df) == 0:
        return "(none)"
    return df.to_csv(sep="\t", index=df.index.name is not None, lineterminator="\n")


#: Most bytes :func:`conditions_tsv` returns. froehlich's table, 145 targets set per cell line,
#: drug and dose, was 36 MB, which no lead could read and a harness re-sent with every later
#: result (PEtab-MCP#290); isensee's, the widest of the rest, is 13 KB.
CONDITIONS_MAX_BYTES = 32 * 1024

#: Expressions named per target in :func:`conditions_summary`; the rest are counted.
_EXPRESSIONS_SHOWN = 3


def conditions_tsv(df: "pd.DataFrame | None", max_bytes: int = CONDITIONS_MAX_BYTES) -> str:
    """The v2 condition table, one row per (condition, target), as TSV with one row per condition
    and one column per target, blank where the condition does not set it; :func:`conditions_summary`
    where that would pass ``max_bytes``. Each cell takes at least its tab, so a table with more
    cells than ``max_bytes`` is summarised without being pivoted."""
    if df is None or len(df) == 0:
        return df_to_tsv(df)
    conditions, targets = df[CONDITION_ID].unique(), df[TARGET_ID].unique()
    if len(conditions) * len(targets) <= max_bytes:
        wide = df.pivot(index=CONDITION_ID, columns=TARGET_ID, values=TARGET_VALUE)
        text = df_to_tsv(wide.reindex(index=conditions, columns=targets))
        if len(text.encode()) <= max_bytes:
            return text
    return conditions_summary(df, max_bytes)


def conditions_summary(df: "pd.DataFrame", max_bytes: int = CONDITIONS_MAX_BYTES) -> str:
    """A condition table too large for one reply, as TSV with one row per target in the table's
    order: how many conditions set it, how many distinct values it takes, the range of the numeric
    ones and the first few expressions, under a line that says where a condition's own values are.
    Targets past ``max_bytes`` are counted, not listed."""
    numeric = pd.to_numeric(df[TARGET_VALUE], errors="coerce")
    by = df.assign(_numeric=numeric).groupby(TARGET_ID, sort=False)
    table = pd.DataFrame(
        {
            "n_conditions": by[CONDITION_ID].nunique(),
            "n_values": by[TARGET_VALUE].nunique(),
            "min": by["_numeric"].min(),
            "max": by["_numeric"].max(),
        }
    )
    expressions = df[numeric.isna() & df[TARGET_VALUE].notna()]
    if len(expressions):

        def named(values: pd.Series) -> str:
            distinct = values.astype(str).unique()
            more = len(distinct) - _EXPRESSIONS_SHOWN
            return ", ".join(distinct[:_EXPRESSIONS_SHOWN]) + (f" (+{more})" if more > 0 else "")

        table["expressions"] = expressions.groupby(TARGET_ID, sort=False)[TARGET_VALUE].agg(named)
    head = (
        f"{df[CONDITION_ID].nunique()} conditions set {len(table)} targets, too large a table for "
        "one reply: one row per target instead. A condition's own values are query_data `SELECT "
        f"targetId, targetValue FROM conditions WHERE conditionId = '{df[CONDITION_ID].iloc[0]}'`.\n"
    )
    header, *rows = table.to_csv(sep="\t", float_format="%.6g", lineterminator="\n").splitlines(
        keepends=True
    )
    tail = "(+{} more targets: query_data `SELECT DISTINCT targetId FROM conditions`)\n"
    budget = max_bytes - len((head + header + tail.format(len(rows))).encode())
    kept = []
    for row in rows:
        budget -= len(row.encode())
        if budget < 0:
            break
        kept.append(row)
    more = len(rows) - len(kept)
    return head + header + "".join(kept) + (tail.format(more) if more else "")


#: Most bytes :func:`experiments_tsv` returns. froehlich's table, one experiment per cell line,
#: drug and dose, is 1 MB, which a harness re-sends with every later result as it did the 36 MB
#: condition table (PEtab-MCP#290); lun's, the largest of the rest, is 42 KB.
EXPERIMENTS_MAX_BYTES = 32 * 1024


def experiments_tsv(df: "pd.DataFrame | None", max_bytes: int = EXPERIMENTS_MAX_BYTES) -> str:
    """The v2 experiment table, one row per (experiment, time, condition), as TSV with one row per
    period, its conditions space-separated in ``conditionId``; :func:`experiments_summary` where
    that would pass ``max_bytes``."""
    if df is None or len(df) == 0:
        return df_to_tsv(df)
    periods = (
        df.groupby([EXPERIMENT_ID, TIME], sort=False)[CONDITION_ID]
        .agg(lambda ids: " ".join(filter(None, ids.dropna())))
        .reset_index()
    )
    text = df_to_tsv(periods)
    return text if len(text.encode()) <= max_bytes else experiments_summary(periods, max_bytes)


def experiments_summary(periods: pd.DataFrame, max_bytes: int = EXPERIMENTS_MAX_BYTES) -> str:
    """An experiment table too large for one reply, given as :func:`experiments_tsv`'s one row per
    period, as TSV with one row per period layout, most shared first: how many experiments share
    it, their periods' start times in order, how many conditions each period applies and the first
    such experiment, under a line that says where an experiment's own periods are. Layouts past
    ``max_bytes`` are counted, not listed."""
    ordered = periods.assign(
        _experiment=pd.factorize(periods[EXPERIMENT_ID])[0],
        times=periods[TIME].map("{:g}".format),
        n_conditions=periods[CONDITION_ID].str.split().str.len().astype(str),
    ).sort_values(["_experiment", TIME], kind="stable")
    experiments = ordered.groupby(EXPERIMENT_ID, sort=False)[["times", "n_conditions"]].agg(
        " ".join
    )
    layouts = experiments.reset_index().groupby(["times", "n_conditions"], sort=False)
    table = (
        pd.DataFrame({"n_experiments": layouts.size(), "example": layouts[EXPERIMENT_ID].first()})
        .reset_index()
        .sort_values("n_experiments", ascending=False, kind="stable")
    )
    table["n_periods"] = table["times"].str.count(" ") + 1
    head = (
        f"{len(experiments)} experiments, too large a table for one reply: one row per period "
        "layout instead, `n_conditions` the conditions each of its `times` applies. An "
        "experiment's own periods are query_data `SELECT time, conditionId FROM experiments WHERE "
        f"experimentId = '{periods[EXPERIMENT_ID].iloc[0]}'`.\n"
    )
    columns = ["n_experiments", "n_periods", "times", "n_conditions", "example"]
    header, *rows = (
        table[columns].to_csv(sep="\t", index=False, lineterminator="\n").splitlines(keepends=True)
    )
    tail = "(+{} more layouts, {} experiments)\n"
    budget = max_bytes - len((head + header + tail.format(len(rows), len(experiments))).encode())
    kept = []
    for row in rows:
        budget -= len(row.encode())
        if budget < 0:
            break
        kept.append(row)
    rest = table["n_experiments"].iloc[len(kept) :].sum()
    more = tail.format(len(rows) - len(kept), rest) if len(kept) < len(rows) else ""
    return head + header + "".join(kept) + more


def summarize_measurements(df: "pd.DataFrame | None", *, preview: int = 4) -> dict[str, Any]:
    """Compact summary of a measurement table: per observable×experiment, the point count,
    time span, value min/max/mean, and a small evenly-spaced ``preview`` of points.

    Returned by ``get_measurements`` by default so the agent sees the data's SHAPE (dynamics,
    ranges, sampling) without every raw point re-entering the LLM context on every turn — a
    full measurement dump is 35-81k tokens on problems of a few hundred to two thousand rows
    (measured: 541 rows/35k, 687/73k, 1755/81k) and compounds as history, which
    is why the raw read is PAGED. Raw records come from ``get_measurements(summarize=False)``
    one page at a time, or from SQL over the ``measurements`` table.
    """
    if df is None or len(df) == 0:
        # Same key set as the non-empty return, so consumers never special-case empty.
        return {
            "n_points": 0,
            "n_groups": 0,
            "columns": [],
            "groups": [],
            "note": "no measurements",
        }
    id_cols = [c for c in (OBSERVABLE_ID, EXPERIMENT_ID) if c in df.columns]
    tcol = TIME if TIME in df.columns else None
    vcol = MEASUREMENT if MEASUREMENT in df.columns else None
    groups: list[dict[str, Any]] = []
    grouper = df.groupby(id_cols, sort=False) if id_cols else [((), df)]
    for key, sub in grouper:
        key = key if isinstance(key, tuple) else (key,)
        g: dict[str, Any] = {c: _clean(v) for c, v in zip(id_cols, key)}
        g["n"] = int(len(sub))
        if tcol is not None:
            g["time"] = {"min": _clean(sub[tcol].min()), "max": _clean(sub[tcol].max())}
        if vcol is not None:
            m = pd.to_numeric(sub[vcol], errors="coerce")
            g["measurement"] = {
                "min": _clean(m.min()),
                "max": _clean(m.max()),
                "mean": _clean(round(float(m.mean()), 6)) if len(m) else None,
            }
            if tcol is not None:
                ss = sub.sort_values(tcol)
                n = len(ss)
                idx = (
                    [0]
                    if n <= 1 or preview <= 1
                    else sorted(
                        {int(round(i * (n - 1) / (preview - 1))) for i in range(min(preview, n))}
                    )
                )
                g["preview"] = [
                    {"time": _clean(ss.iloc[i][tcol]), "measurement": _clean(ss.iloc[i][vcol])}
                    for i in idx
                ]
        groups.append(g)
    return {
        "n_points": int(len(df)),
        "n_groups": len(groups),
        "columns": [str(c) for c in df.columns],
        "groups": groups,
        # Says PAGED, not "all points": the raw read caps at max_rows and returns
        # truncated/next_offset. An agent told this returns everything would reason over a
        # silently partial view — concluding, say, that an observable has no late timepoints
        # because they fell past the first page.
        "note": "Summary view (per observable×experiment: n, time span, value min/max/mean, and a "
        "few preview points). get_measurements(summarize=False) returns raw points "
        "PAGED (max_rows per call, with total_rows/truncated/next_offset) — page it, or "
        "narrow with experiment_id/observable_id.",
    }
