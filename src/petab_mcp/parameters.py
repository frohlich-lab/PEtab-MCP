"""Parameter-table inspection, editing, and model reconciliation.

The parameter table is mutable. In PEtab v2 a parameter has the fields
``id, lb, ub, nominal_value, estimate, prior_distribution, prior_parameters``
(there is no ``parameterScale`` column — scale is expressed through the prior).
"""

from __future__ import annotations

from typing import Any

from .errors import ParameterError

_EDITABLE_FIELDS = (
    "nominal_value",
    "lb",
    "ub",
    "estimate",
    "prior_distribution",
    "prior_parameters",
)

# Defaults used when auto-adding a model parameter that is missing from the table.
_DEFAULT_LB = 1e-5
_DEFAULT_UB = 1e5


def _param_map(problem) -> dict[str, Any]:
    return {p.id: p for p in problem.parameters}


def _param_record(p) -> dict[str, Any]:
    def s(v):
        return None if v is None else (v if isinstance(v, (int, float, bool, str)) else str(v))

    return {
        "id": p.id,
        "lb": s(p.lb),
        "ub": s(p.ub),
        "nominal_value": s(p.nominal_value),
        "estimate": bool(p.estimate),
        "prior_distribution": s(p.prior_distribution),
        "prior_parameters": s(p.prior_parameters),
    }


def parameters_to_records(problem) -> list[dict[str, Any]]:
    """Return the parameter table as a list of JSON-friendly records."""
    return [_param_record(p) for p in problem.parameters]


def _coerce_estimate(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        low = value.strip().lower()
        if low in ("true", "1", "yes"):
            return True
        if low in ("false", "0", "no"):
            return False
    raise ParameterError(f"Cannot interpret estimate value {value!r} as boolean.")


def update_parameters(problem, updates: list[dict[str, Any]]) -> dict[str, Any]:
    """Apply a batch of parameter-table edits, transactionally.

    Each update is a dict with an ``id`` plus any of the editable fields. Unknown
    ids, non-editable fields, or bounds violations raise ``ParameterError`` and
    leave the table unchanged (all-or-nothing). Returns the updated records.

    The apply/rollback dance is needed because petab's ``Parameter`` validates on
    assignment (``lb <= ub``): setting fields one at a time could fail mid-update
    on a transient invalid state, or leave a parameter half-modified.
    """
    if not isinstance(updates, list):
        raise ParameterError("`updates` must be a list of {id, ...field} objects.")
    pmap = _param_map(problem)

    # First pass: validate structure and resolve the fields to change per id.
    resolved: list[tuple[str, dict[str, Any]]] = []
    for upd in updates:
        pid = upd.get("id")
        if pid is None:
            raise ParameterError("Each update must include an 'id'.")
        if pid not in pmap:
            raise ParameterError(
                f"Unknown parameter id {pid!r}. Known ids: {sorted(pmap)}"
            )
        fields: dict[str, Any] = {}
        for field, value in upd.items():
            if field == "id":
                continue
            if field not in _EDITABLE_FIELDS:
                raise ParameterError(
                    f"Field {field!r} is not editable. Editable: {_EDITABLE_FIELDS}"
                )
            fields[field] = _coerce_estimate(value) if field == "estimate" else value
        resolved.append((pid, fields))

    touched = {pid for pid, _ in resolved}
    snapshots = {
        pid: {f: getattr(pmap[pid], f) for f in _EDITABLE_FIELDS} for pid in touched
    }

    def _rollback() -> None:
        for pid, snap in snapshots.items():
            for f, v in snap.items():
                object.__setattr__(pmap[pid], f, v)

    try:
        # Apply all fields without per-assignment validation (avoids transient
        # invalid states), then validate each touched parameter as a whole.
        for pid, fields in resolved:
            for field, value in fields.items():
                object.__setattr__(pmap[pid], field, value)
        for pid in touched:
            param = pmap[pid]
            type(param).model_validate(param.model_dump())
            lb, ub = param.lb, param.ub
            if lb is not None and ub is not None and float(lb) > float(ub):
                raise ParameterError(
                    f"Parameter {pid!r}: lower bound {lb} exceeds upper bound {ub}."
                )
    except ParameterError:
        _rollback()
        raise
    except Exception as exc:  # pydantic validation, coercion, ...
        _rollback()
        raise ParameterError(f"Invalid parameter update: {exc}") from exc

    changed_ids = [pid for pid, _ in resolved]
    return {
        "updated": changed_ids,
        "parameters": [_param_record(pmap[pid]) for pid in changed_ids],
    }


def parameter_sync_diff(problem) -> dict[str, Any]:
    """Compare the model's parameters with the parameter table.

    ``model_only`` are model parameters eligible for (but missing from) the table.
    ``table_only`` are table entries not backed by a model parameter — usually
    legitimate noise/observable parameters, occasionally stale after a model edit
    (petab linting flags the truly extraneous ones).
    """
    valid = set(problem.model.get_valid_parameters_for_parameter_table())
    table_ids = {p.id for p in problem.parameters}
    return {
        "in_sync": valid.issubset(table_ids),
        "model_only": sorted(valid - table_ids),
        "table_only": sorted(table_ids - valid),
    }


def sync_parameters(problem, policy: str = "report") -> dict[str, Any]:
    """Reconcile the parameter table with the model.

    ``policy='report'`` (default) just returns the diff; ``policy='add'`` also
    appends any model-only parameters to the table with sensible defaults.
    """
    diff = parameter_sync_diff(problem)
    added: list[str] = []
    if policy == "add":
        for pid in diff["model_only"]:
            nominal = problem.model.get_parameter_value(pid)
            problem.add_parameter(
                pid,
                estimate=True,
                nominal_value=nominal,
                lb=_DEFAULT_LB,
                ub=_DEFAULT_UB,
            )
            added.append(pid)
        diff = parameter_sync_diff(problem)
    elif policy != "report":
        raise ParameterError(f"Unknown sync policy {policy!r}; use 'report' or 'add'.")
    return {"policy": policy, "added": added, "diff": diff}
