"""Helpers to turn PEtab/pandas objects into compact, JSON-serializable records.

MCP tool results must be plain JSON. petab.v2 tables come back either as pandas
DataFrames (which may be ``None`` when empty) or as lists of pydantic objects, and
sympy/enum values need stringifying. These helpers centralize that conversion so
every tool returns clean data.
"""

from __future__ import annotations

import math
from typing import Any

import pandas as pd


def _clean(value: Any) -> Any:
    """Coerce a single cell into a JSON-friendly value."""
    if value is None:
        return None
    if isinstance(value, float):
        # JSON has no NaN/Inf; represent them as None so results stay valid.
        if math.isnan(value) or math.isinf(value):
            return None
        return value
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

    The DataFrame index is preserved under an ``id`` key when it is named or
    carries meaningful labels (petab tables are typically indexed by their
    primary-key id).
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
            record[index_name or "id"] = _clean(idx)
        for col, val in row.items():
            record[str(col)] = _clean(val)
        records.append(record)
    return records


def issues_to_records(validation_result) -> list[dict[str, Any]]:
    """Convert a petab.v2 ``ValidationResultList`` to a list of issue dicts."""
    out: list[dict[str, Any]] = []
    for issue in validation_result:
        level = getattr(issue, "level", None)
        # `task` is the check's name (a string), not an object — use it directly.
        task = getattr(issue, "task", None)
        out.append(
            {
                "level": getattr(level, "name", str(level)).lower(),
                "message": str(getattr(issue, "message", issue)),
                "check": str(task) if task else None,
            }
        )
    return out
