"""Helpers to turn PEtab/pandas objects into compact, JSON-serializable records.

MCP tool results must be plain JSON. petab.v1 tables come back either as pandas
DataFrames (which may be ``None`` when empty) or as lists of pydantic objects, and
sympy/enum values need stringifying. These helpers centralize that conversion so
every tool returns clean data.
"""

from __future__ import annotations

import math
from typing import Any

import pandas as pd
from petab.v1.C import (
    MEASUREMENT,
    OBSERVABLE_ID,
    PREEQUILIBRATION_CONDITION_ID,
    SIMULATION_CONDITION_ID,
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

    A NAMED index is preserved under its own name (petab tables are indexed by their
    primary-key id — ``observableId``, ``conditionId``, ``parameterId``). An unnamed index
    carries only positional offsets and is dropped, so boolean-filtering a measurement table
    cannot leak a spurious id field.
    """
    if df is None or len(df) == 0:
        return []
    records: list[dict[str, Any]] = []
    index_name = df.index.name
    # Only surface the index when it is a meaningful, *named* primary key (petab's
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


def summarize_measurements(df: "pd.DataFrame | None", *, preview: int = 4) -> dict[str, Any]:
    """Compact summary of a measurement table: per observable×condition, the point count,
    time span, value min/max/mean, and a small evenly-spaced ``preview`` of points.

    Returned by ``get_measurements`` by default so the panel sees the data's SHAPE (dynamics,
    ranges, sampling) without every raw point re-entering the LLM context on every turn — a
    full measurement dump is 35-81k tokens on the shipped benchmark problems (measured:
    bachmann 541 rows/35k, isensee 687/73k, lucarelli 1755/81k) and compounds as history, which
    is why the raw read is PAGED. Raw records come from ``get_measurements(summarize=False)``
    one page at a time, or from SQL over the ``measurements`` table.
    """
    if df is None or len(df) == 0:
        # Same key set as the non-empty return, so consumers never special-case empty.
        return {"n_points": 0, "n_groups": 0, "columns": [], "groups": [], "note": "no measurements"}
    id_cols = [c for c in (OBSERVABLE_ID, SIMULATION_CONDITION_ID,
                           PREEQUILIBRATION_CONDITION_ID) if c in df.columns]
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
            g["measurement"] = {"min": _clean(m.min()), "max": _clean(m.max()),
                                "mean": _clean(round(float(m.mean()), 6)) if len(m) else None}
            if tcol is not None:
                ss = sub.sort_values(tcol)
                n = len(ss)
                idx = ([0] if n <= 1 or preview <= 1 else
                       sorted({int(round(i * (n - 1) / (preview - 1))) for i in range(min(preview, n))}))
                g["preview"] = [{"time": _clean(ss.iloc[i][tcol]),
                                 "measurement": _clean(ss.iloc[i][vcol])} for i in idx]
        groups.append(g)
    return {
        "n_points": int(len(df)), "n_groups": len(groups),
        "columns": [str(c) for c in df.columns], "groups": groups,
        # Says PAGED, not "all points": the raw read caps at max_rows and returns
        # truncated/next_offset. An agent told this returns everything would reason over a
        # silently partial view — concluding, say, that an observable has no late timepoints
        # because they fell past the first page.
        "note": "Summary view (per observable×condition: n, time span, value min/max/mean, and a "
                "few preview points). get_measurements(summarize=False) returns raw points "
                "PAGED (max_rows per call, with total_rows/truncated/next_offset) — page it, or "
                "narrow with condition_id/observable_id, or use query_data for SQL over the "
                "`measurements` table.",
    }
