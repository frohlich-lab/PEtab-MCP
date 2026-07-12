"""Fit inspection and residual statistics (backend-agnostic).

Given per-data-point residual records — as produced by
:meth:`petab_mcp.engine.AmiciEngine.residuals` — this module aggregates residual
statistics across the natural cross sections of a PEtab problem
(**observables**, **conditions**, **timepoints**) and assembles a
"fit inspection" report that surfaces the worst-fit points and any systematic
bias.

The functions here are deliberately free of AMICI/pandas-heavy machinery: they
consume plain record dicts and return plain JSON-friendly dicts, so the whole
statistics layer is unit-testable without a simulation backend. Producing the
records (which needs AMICI) lives in :mod:`petab_mcp.engine`.

Record schema (one per measurement point)::

    {
        "observable_id": str,
        "condition_id": str,
        "time": float | None,
        "measurement": float | None,
        "simulation": float | None,
        "residual": float | None,           # measurement - simulation (natural units)
        "residual_weighted": float | None,  # (measurement - simulation) / noise, on the
                                            # observable scale — the term entering chi2
        "noise": float | None,              # evaluated noise standard deviation (sigma)
    }

Points whose ``simulation`` (hence ``residual``) is missing are counted under
``n_missing`` but excluded from the statistics.
"""

from __future__ import annotations

import math
from typing import Any, Iterable

# The cross sections a report is grouped by. Each maps a human-facing block name
# to the record key that identifies the group.
CROSS_SECTIONS: dict[str, str] = {
    "by_observable": "observable_id",
    "by_condition": "condition_id",
    "by_time": "time",
}

# |z| beyond which a group's mean weighted residual is flagged as systematic bias.
_BIAS_Z_THRESHOLD = 2.0

# |corr(time, weighted residual)| beyond which a group is flagged as having a systematic
# time trend — a dynamics/asymptote error the mean-based bias flag misses (the residuals
# drift from one sign to the other over time, averaging to ~0). Needs a few timepoints.
_TREND_R_THRESHOLD = 0.8
_TREND_MIN_POINTS = 4


def _finite(value: Any) -> float | None:
    """Return ``value`` as a finite float, or ``None`` if missing/non-finite."""
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return f


def _mean(xs: list[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def _std(xs: list[float], mean: float | None) -> float | None:
    """Sample standard deviation (ddof=1); ``None`` for fewer than two points."""
    if mean is None or len(xs) < 2:
        return None
    var = sum((x - mean) ** 2 for x in xs) / (len(xs) - 1)
    return math.sqrt(var)


def _round(value: float | None, ndigits: int = 6) -> float | None:
    if value is None:
        return None
    return round(value, ndigits)


def _group_statistics(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate residual statistics over a set of records.

    ``residual`` statistics are in the observable's natural units; the
    ``*_weighted`` / ``chi2`` statistics use noise-normalized residuals (over the
    ``n_weighted`` points whose noise could be evaluated) and are comparable across
    observables. ``bias_z`` is the z-score of the mean weighted residual
    (``mean_weighted * sqrt(n_weighted)``); its magnitude exceeding
    ``_BIAS_Z_THRESHOLD`` indicates a systematic over/under-shoot.

    The key set is fixed regardless of the data: raw-residual keys are ``None`` when
    the group is empty, and weighted keys are ``None`` (with ``n_weighted == 0``) when
    no weighted residual could be computed — so downstream consumers see a stable
    schema and never need membership checks.
    """
    residuals = [r for r in (_finite(rec.get("residual")) for rec in records) if r is not None]
    weighted = [
        w for w in (_finite(rec.get("residual_weighted")) for rec in records) if w is not None
    ]
    n = len(residuals)
    n_missing = len(records) - n

    stats: dict[str, Any] = {"n": n, "n_missing": n_missing}

    if n:
        mean = _mean(residuals)
        abs_res = [abs(r) for r in residuals]
        sse = sum(r * r for r in residuals)
        stats.update(
            {
                "mean_residual": _round(mean),
                "std_residual": _round(_std(residuals, mean)),
                "mean_abs_residual": _round(_mean(abs_res)),
                "rmse": _round(math.sqrt(sse / n)),
                "sse": _round(sse),
                "max_abs_residual": _round(max(abs_res)),
            }
        )
    else:
        stats.update(
            {
                "mean_residual": None,
                "std_residual": None,
                "mean_abs_residual": None,
                "rmse": None,
                "sse": None,
                "max_abs_residual": None,
            }
        )

    # Noise-weighted statistics (the chi2 terms). These keys are ALWAYS present —
    # set to None when no weighted residual could be computed for the group — so the
    # output schema does not vary with whether the noise model was evaluable.
    _WEIGHTED_KEYS = (
        "mean_weighted_residual",
        "std_weighted_residual",
        "rmse_weighted",
        "chi2",
        "reduced_chi2",
        "max_abs_weighted_residual",
        "bias_z",
    )
    if weighted:
        nw = len(weighted)
        mean_w = _mean(weighted)
        chi2 = sum(w * w for w in weighted)
        stats.update(
            {
                "n_weighted": nw,
                "mean_weighted_residual": _round(mean_w),
                "std_weighted_residual": _round(_std(weighted, mean_w)),
                "rmse_weighted": _round(math.sqrt(chi2 / nw)),
                "chi2": _round(chi2),
                "reduced_chi2": _round(chi2 / nw),
                "max_abs_weighted_residual": _round(max(abs(w) for w in weighted)),
                "bias_z": _round(mean_w * math.sqrt(nw)),
            }
        )
    else:
        stats["n_weighted"] = 0
        stats.update(dict.fromkeys(_WEIGHTED_KEYS, None))
    return stats


def _group_by(records: list[dict[str, Any]], key: str) -> list[tuple[Any, list[dict[str, Any]]]]:
    """Group records by ``record[key]``, preserving first-seen order of the keys."""
    order: list[Any] = []
    groups: dict[Any, list[dict[str, Any]]] = {}
    for rec in records:
        gk = rec.get(key)
        if gk not in groups:
            groups[gk] = []
            order.append(gk)
        groups[gk].append(rec)
    return [(gk, groups[gk]) for gk in order]


def _sort_key(group_value: Any) -> tuple[int, Any]:
    """Order groups: numeric values ascending, then everything else by string."""
    num = _finite(group_value)
    if num is not None:
        return (0, num)
    return (1, str(group_value))


def residual_statistics(
    records: Iterable[dict[str, Any]],
    cross_sections: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Residual statistics overall and across the requested cross sections.

    Parameters
    ----------
    records:
        Per-point residual records (see the module docstring).
    cross_sections:
        Which cross-section blocks to compute; any of ``CROSS_SECTIONS`` keys
        (``"by_observable"``, ``"by_condition"``, ``"by_time"``). Defaults to all.

    Returns a dict with an ``overall`` block plus one list per requested cross
    section, each entry being that group's id and statistics.
    """
    records = list(records)
    requested = list(cross_sections) if cross_sections is not None else list(CROSS_SECTIONS)

    report: dict[str, Any] = {"overall": _group_statistics(records)}
    for block in requested:
        if block not in CROSS_SECTIONS:
            raise ValueError(
                f"Unknown cross section {block!r}; valid: {sorted(CROSS_SECTIONS)}"
            )
        key = CROSS_SECTIONS[block]
        label = key  # e.g. "observable_id", "condition_id", "time"
        entries = [
            {label: group_value, **_group_statistics(group)}
            for group_value, group in _group_by(records, key)
        ]
        entries.sort(key=lambda e: _sort_key(e[label]))
        report[block] = entries
    return report


def _rank_value(rec: dict[str, Any]) -> float:
    """Ranking magnitude for 'worst fit': |weighted residual|, else |residual|."""
    w = _finite(rec.get("residual_weighted"))
    if w is not None:
        return abs(w)
    r = _finite(rec.get("residual"))
    return abs(r) if r is not None else -1.0


def worst_points(records: Iterable[dict[str, Any]], top_n: int = 10) -> list[dict[str, Any]]:
    """The ``top_n`` worst-fit points, ranked by |weighted residual| (else |residual|)."""
    scored = [rec for rec in records if _rank_value(rec) >= 0.0]
    scored.sort(key=_rank_value, reverse=True)
    return scored[: max(0, top_n)]


def _bias_flags(report: dict[str, Any]) -> list[dict[str, Any]]:
    """Cross-section groups whose mean weighted residual is significantly non-zero."""
    flags: list[dict[str, Any]] = []
    for block, key in CROSS_SECTIONS.items():
        for entry in report.get(block, []):
            z = entry.get("bias_z")
            if z is not None and abs(z) >= _BIAS_Z_THRESHOLD and entry.get("n", 0) > 0:
                flags.append(
                    {
                        "cross_section": block,
                        "group": entry.get(key),
                        "n": entry["n"],
                        "mean_weighted_residual": entry.get("mean_weighted_residual"),
                        "bias_z": z,
                        "direction": "model_under_predicts"
                        if z > 0
                        else "model_over_predicts",
                    }
                )
    flags.sort(key=lambda f: abs(f["bias_z"]), reverse=True)
    return flags


def _corr(xs: list[float], ys: list[float]) -> float | None:
    """Pearson correlation of two equal-length sequences, or ``None`` if undefined."""
    n = len(xs)
    if n < 2:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx <= 0 or syy <= 0:
        return None
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return sxy / math.sqrt(sxx * syy)


def _trend_flags(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per (observable, condition) groups whose residuals trend systematically with time.

    A strong correlation between ``time`` and the (signed) weighted residual means the model
    has the wrong dynamics/asymptote — it drifts from under- to over-prediction (or vice
    versa) across time, so the *mean* residual (and hence ``bias_z``) can be ~0 while the fit
    is clearly wrong. This is the signal a plain bias flag misses.
    """
    groups: dict[tuple[Any, Any], list[dict[str, Any]]] = {}
    order: list[tuple[Any, Any]] = []
    for rec in records:
        gk = (rec.get("observable_id"), rec.get("condition_id"))
        if gk not in groups:
            groups[gk] = []
            order.append(gk)
        groups[gk].append(rec)

    flags: list[dict[str, Any]] = []
    for obs, cond in order:
        pairs = [
            (t, w)
            for t, w in (
                (_finite(r.get("time")), _finite(r.get("residual_weighted")))
                for r in groups[(obs, cond)]
            )
            if t is not None and w is not None
        ]
        if len(pairs) < _TREND_MIN_POINTS:
            continue
        r = _corr([t for t, _ in pairs], [w for _, w in pairs])
        if r is None or abs(r) < _TREND_R_THRESHOLD:
            continue
        flags.append(
            {
                "observable_id": obs,
                "condition_id": cond,
                "n": len(pairs),
                "time_residual_corr": _round(r),
                # residual falls with time (r<0) => simulation rises past the data late
                "pattern": "over_predicts_over_time" if r < 0 else "under_predicts_over_time",
            }
        )
    flags.sort(key=lambda f: abs(f["time_residual_corr"]), reverse=True)
    return flags


def fit_inspection(
    records: Iterable[dict[str, Any]],
    top_n: int = 10,
) -> dict[str, Any]:
    """Assemble a fit-inspection report from per-point residual records.

    The report bundles the overall fit summary, residual statistics across every
    cross section (observable/condition/time), the ``top_n`` worst-fit points,
    systematic-bias flags, and time-trend flags (dynamics/asymptote errors) —
    everything an assistant needs to diagnose *where* a model fits the data poorly.
    """
    records = list(records)
    stats = residual_statistics(records)
    return {
        "n_points": len(records),
        "summary": stats["overall"],
        "statistics": stats,
        "worst_points": worst_points(records, top_n),
        "bias_flags": _bias_flags(stats),
        "trend_flags": _trend_flags(records),
    }
