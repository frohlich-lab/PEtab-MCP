"""Parameter-table inspection, editing, and model reconciliation (PEtab v1).

In PEtab v1 the parameter table is a pandas DataFrame indexed by ``parameterId``
with columns ``parameterScale`` (lin/log/log10), ``lowerBound``, ``upperBound``,
``nominalValue``, ``estimate`` (0/1), and optional priors. Edits operate directly on
``problem.parameter_df``.
"""

from __future__ import annotations

from typing import Any

import petab.v1 as petab
from petab.v1.C import (
    ESTIMATE,
    LIN,
    LOG,
    LOG10,
    LOWER_BOUND,
    NOMINAL_VALUE,
    OBJECTIVE_PRIOR_PARAMETERS,
    OBJECTIVE_PRIOR_TYPE,
    PARAMETER_ID,
    PARAMETER_SCALE,
    UPPER_BOUND,
)

from .errors import ParameterError

_SCALES = {LIN, LOG, LOG10}
_NUMERIC_FIELDS = {LOWER_BOUND, UPPER_BOUND, NOMINAL_VALUE}
_EDITABLE_FIELDS = (
    NOMINAL_VALUE,
    LOWER_BOUND,
    UPPER_BOUND,
    ESTIMATE,
    PARAMETER_SCALE,
    OBJECTIVE_PRIOR_TYPE,
    OBJECTIVE_PRIOR_PARAMETERS,
)

# Defaults used when auto-adding a model parameter that is missing from the table.
_DEFAULT_LB = 1e-5
_DEFAULT_UB = 1e5


def _default_bounds(nominal) -> tuple[float, float]:
    """Bounds for an auto-added parameter that always bracket its nominal value."""
    lb, ub = _DEFAULT_LB, _DEFAULT_UB
    if nominal is None:
        return lb, ub
    nominal = float(nominal)
    if nominal > 0:
        return min(lb, nominal * 1e-3), max(ub, nominal * 1e3)
    return min(lb, nominal - 1.0), max(ub, nominal + 1.0)


def _clean(value: Any) -> Any:
    import math

    if value is None:
        return None
    if isinstance(value, float):
        return None if (math.isnan(value) or math.isinf(value)) else value
    if isinstance(value, (str, bool, int)):
        return value
    try:
        import pandas as pd

        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, TypeError):
            pass
    return str(value)


def _param_record(problem, pid: str) -> dict[str, Any]:
    row = problem.parameter_df.loc[pid]
    rec: dict[str, Any] = {PARAMETER_ID: pid}
    for col in problem.parameter_df.columns:
        rec[col] = _clean(row[col])
    if ESTIMATE in rec and rec[ESTIMATE] is not None:
        rec[ESTIMATE] = bool(int(float(rec[ESTIMATE])))
    return rec


def parameters_to_records(problem) -> list[dict[str, Any]]:
    """Return the parameter table as a list of JSON-friendly records."""
    return [_param_record(problem, pid) for pid in problem.parameter_df.index]


def _coerce_estimate_int(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return 1 if value else 0
    if isinstance(value, str):
        low = value.strip().lower()
        if low in ("true", "1", "yes"):
            return 1
        if low in ("false", "0", "no"):
            return 0
    raise ParameterError(f"Cannot interpret estimate value {value!r} as 0/1.")


def update_parameters(problem, updates: list[dict[str, Any]]) -> dict[str, Any]:
    """Apply a batch of parameter-table edits, transactionally.

    Each update is a dict with an ``id`` (or ``parameterId``) plus any editable
    PEtab column (``nominalValue``, ``lowerBound``, ``upperBound``, ``estimate``,
    ``parameterScale``, prior columns). Unknown ids, non-editable columns, bad
    bounds, or an invalid scale raise ``ParameterError`` and leave the table
    unchanged.
    """
    if not isinstance(updates, list):
        raise ParameterError("`updates` must be a list of {id, ...field} objects.")
    df = problem.parameter_df

    resolved: list[tuple[str, dict[str, Any]]] = []
    for upd in updates:
        pid = upd.get("id", upd.get(PARAMETER_ID))
        if pid is None:
            raise ParameterError("Each update must include an 'id'.")
        if pid not in df.index:
            raise ParameterError(
                f"Unknown parameter id {pid!r}. Known ids: {sorted(df.index)}"
            )
        fields: dict[str, Any] = {}
        for field, value in upd.items():
            if field in ("id", PARAMETER_ID):
                continue
            if field not in _EDITABLE_FIELDS:
                raise ParameterError(
                    f"Field {field!r} is not editable. Editable: {_EDITABLE_FIELDS}"
                )
            fields[field] = value
        resolved.append((pid, fields))

    touched = [pid for pid, _ in resolved]
    snapshot = df.loc[touched].copy(deep=True)

    try:
        for pid, fields in resolved:
            for col, value in fields.items():
                if col in _NUMERIC_FIELDS:
                    value = float(value)
                elif col == ESTIMATE:
                    value = _coerce_estimate_int(value)
                elif col == PARAMETER_SCALE:
                    value = str(value)
                    if value not in _SCALES:
                        raise ParameterError(
                            f"Invalid parameterScale {value!r}; use one of {sorted(_SCALES)}."
                        )
                df.loc[pid, col] = value
            lb = df.loc[pid, LOWER_BOUND] if LOWER_BOUND in df.columns else None
            ub = df.loc[pid, UPPER_BOUND] if UPPER_BOUND in df.columns else None
            if lb is not None and ub is not None and float(lb) > float(ub):
                raise ParameterError(
                    f"Parameter {pid!r}: lower bound {lb} exceeds upper bound {ub}."
                )
    except ParameterError:
        df.loc[touched] = snapshot
        raise
    except Exception as exc:
        df.loc[touched] = snapshot
        raise ParameterError(f"Invalid parameter update: {exc}") from exc

    return {
        "updated": touched,
        "parameters": [_param_record(problem, pid) for pid in touched],
    }


def parameter_sync_diff(problem) -> dict[str, Any]:
    """Compare the model's parameters with the parameter table.

    ``model_only`` are model parameters missing from the table; ``table_only`` are
    table entries not backed by a model parameter (usually noise/observable params).
    """
    model_params = set(problem.get_model_parameters())
    table_ids = set(problem.parameter_df.index)
    return {
        "in_sync": model_params.issubset(table_ids),
        "model_only": sorted(model_params - table_ids),
        "table_only": sorted(table_ids - model_params),
    }


def sync_parameters(problem, policy: str = "report") -> dict[str, Any]:
    """Reconcile the parameter table with the model ('report' or 'add')."""
    diff = parameter_sync_diff(problem)
    added: list[str] = []
    if policy == "add":
        for pid in diff["model_only"]:
            nominal = problem.model.get_parameter_value(pid)
            lb, ub = _default_bounds(nominal)
            scale = LOG10 if (nominal is not None and float(nominal) > 0) else LIN
            problem.parameter_df.loc[pid, PARAMETER_SCALE] = scale
            problem.parameter_df.loc[pid, LOWER_BOUND] = lb
            problem.parameter_df.loc[pid, UPPER_BOUND] = ub
            problem.parameter_df.loc[pid, NOMINAL_VALUE] = nominal
            problem.parameter_df.loc[pid, ESTIMATE] = 1
            added.append(pid)
        diff = parameter_sync_diff(problem)
    elif policy != "report":
        raise ParameterError(f"Unknown sync policy {policy!r}; use 'report' or 'add'.")
    return {"policy": policy, "added": added, "diff": diff}
