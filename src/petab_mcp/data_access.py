"""SQL access to the PEtab tables and the stored simulation dataframe via DuckDB.

Instead of a fixed menu of handcoded residual-statistics tools, the server exposes the
underlying data as SQL tables the agent can query directly. Everything the model-building
loop produces is queryable in one place:

* the fixed experimental design — ``measurements``, ``conditions``, ``observables``;
* the ``parameters`` table (bounds, scale, fitted ``nominalValue``, ``estimate``, priors,
  plus each parameter's ``role`` and ``status``);
* the ``simulations`` dataframe from the last commit — one row per measurement point,
  carrying the ``simulation`` value and the **residuals** (``residual``,
  ``residual_weighted`` — the chi2 term — and the evaluated ``noise``) next to the
  ``measurement``;
* the last fit's pyPESTO multistart — ``optimization_starts`` (one row per start) and
  ``optimization_trace`` (one row per start/iteration, the objective over the run).

Residual statistics, worst-fit points, per-observable / per-condition / per-timepoint
breakdowns, and bias or time-trend diagnostics are all just SQL over these tables now: the
agent writes the query it needs rather than picking from a fixed set of aggregations.

Isolation is by *copy*, not by restriction: each query runs on its own throwaway in-memory
DuckDB database into which the current tables are copied. The full SQL surface is therefore
available — CTEs, temp tables, even ``INSERT``/``UPDATE``/``DROP`` — and none of it can touch
the session: the copy is discarded when the query returns, and the source DataFrames are
never written back. The connection additionally has host file/network access disabled
(``read_csv``, ``ATTACH``, ``COPY``, ``INSTALL``/``LOAD``), which copying the data does not by
itself prevent. Results come back as plain JSON records.
"""

from __future__ import annotations

from typing import Any

from petab.v1.C import (
    CONDITION_ID,
    ESTIMATE,
    LOWER_BOUND,
    MEASUREMENT,
    NOISE_FORMULA,
    NOMINAL_VALUE,
    OBSERVABLE_FORMULA,
    OBSERVABLE_ID,
    PARAMETER_ID,
    PARAMETER_SCALE,
    SIMULATION_CONDITION_ID,
    TIME,
    UPPER_BOUND,
)

from .errors import MissingDependencyError, ValidationError
from .formatting import _clean

# Canonical (minimal) column schema for each FIXED PEtab table. Used only as a FALLBACK when
# the live table is absent (an optional table petab loaded as ``None``) or empty with no
# columns at all — e.g. a model-less problem that ships a header-only ``parameters.tsv``, whose
# parameter table has zero rows and (once turned into JSON records) zero columns. DuckDB refuses
# to register a zero-column DataFrame ("Need a DataFrame with at least one column"), and because
# every table is copied in BEFORE the query runs, one such frame made EVERY ``query_data`` call
# fail — even ``SELECT 1``. Giving the empty table its real column names keeps it queryable
# (it just returns no rows) and keeps the whole query surface working.
MEASUREMENT_COLUMNS = [OBSERVABLE_ID, SIMULATION_CONDITION_ID, TIME, MEASUREMENT]
CONDITION_COLUMNS = [CONDITION_ID]
OBSERVABLE_COLUMNS = [OBSERVABLE_ID, OBSERVABLE_FORMULA, NOISE_FORMULA]
PARAMETER_COLUMNS = [
    PARAMETER_ID, PARAMETER_SCALE, LOWER_BOUND, UPPER_BOUND, NOMINAL_VALUE, ESTIMATE,
]

# Canonical column order for the per-point simulation dataframe. Declaring it up front means
# the ``simulations`` table has a stable schema (and named columns resolve in SQL) even at
# cold start, before any commit has produced records — the table is simply empty then.
SIMULATION_COLUMNS = [
    "observable_id",
    "condition_id",
    "time",
    "measurement",
    "simulation",
    "residual",
    "residual_weighted",
    "noise",
]

# Columns of the pyPESTO optimization tables (from the last fit). Fixing them keeps the
# empty-table schema stable before/without a fit, same as SIMULATION_COLUMNS.
OPTIMIZATION_START_COLUMNS = [
    "start", "id", "fval", "time", "n_fval", "n_grad", "n_hess", "exitflag", "message",
]
OPTIMIZATION_TRACE_COLUMNS = ["start", "iteration", "fval", "time"]

# The queryable tables and a one-line description of each (surfaced by ``list_data_tables``).
TABLE_DESCRIPTIONS: dict[str, str] = {
    "measurements": "Fixed measurement data — one row per measured point.",
    "conditions": "Fixed experimental conditions and the model changes they apply.",
    "observables": "Fixed observable definitions (formula, noise model, transformation).",
    "parameters": (
        "Parameter table: bounds, scale, nominalValue (fit-only), estimate, priors, plus "
        "role (kinetic/noise_sigma/observable_scaling/initial_condition/exponent/constant) "
        "and status (estimated/fixed)."
    ),
    "simulations": (
        "Per-point simulation dataframe from the last commit: measurement vs simulation "
        "with residual, residual_weighted (the chi2 term), and noise (sigma). Empty until "
        "the first commit/fit."
    ),
    "optimization_starts": (
        "pyPESTO multistart results from the last fit: one row per start (start=0 is best) "
        "with fval, time, n_fval/n_grad/n_hess, exitflag, message. Empty on a simulate-only "
        "pass or before the first fit."
    ),
    "optimization_trace": (
        "pyPESTO per-iteration optimization trace from the last fit: one row per "
        "(start, iteration) with the objective fval and cumulative time."
    ),
}


def _duckdb():
    """Import DuckDB, raising a clear, actionable error if it is not installed."""
    try:
        import duckdb
    except ImportError as exc:  # pragma: no cover - duckdb is a hard dependency
        raise MissingDependencyError("SQL data access", "duckdb", "duckdb") from exc
    return duckdb


def _promote_index(df):
    """Return a copy of ``df`` with a *named* index promoted to a regular column.

    petab's observable/condition/parameter tables are indexed by their primary-key id
    (``observableId`` etc.); promoting it makes that id queryable as a column. An unnamed
    positional index (e.g. the measurement table's RangeIndex) carries no meaning and is
    left out.
    """
    if df is None:
        return None
    named = df.index.name is not None or any(n is not None for n in df.index.names)
    return df.reset_index() if named else df.copy()


def _with_schema(df, columns):
    """Guarantee a queryable schema so the frame can be registered into DuckDB.

    A fixed PEtab table can be absent (``None`` — an optional table petab did not load) or
    present-but-empty with no columns at all (the parameter table of a model-less problem,
    once turned into zero JSON records). DuckDB cannot register a zero-column DataFrame, and
    since the tables are copied in before the query runs, one such frame breaks EVERY query.
    Fall back to an empty frame carrying the table's canonical column NAMES in that case (the
    columns are ``object``-dtyped — no rows to infer types from — which DuckDB registers fine);
    otherwise pass the real frame through untouched (its own columns are authoritative).
    """
    import pandas as pd

    if df is None or df.shape[1] == 0:
        return pd.DataFrame(columns=columns)
    return df


def _frames(session) -> dict[str, Any]:
    """Build the ``{table_name: DataFrame}`` mapping registered for a query.

    Reads the *current* problem tables and the stored simulation dataframe
    (``session.simulation``); the ``simulations`` frame is empty (but well-typed) when no
    commit has produced records yet. Every frame is guaranteed at least one column (see
    :func:`_with_schema`) so registering it into DuckDB cannot fail on an empty table.
    """
    import pandas as pd

    from .parameters import parameters_to_records

    problem = session.problem
    frames: dict[str, Any] = {
        "measurements": _with_schema(_promote_index(problem.measurement_df), MEASUREMENT_COLUMNS),
        "conditions": _with_schema(_promote_index(problem.condition_df), CONDITION_COLUMNS),
        "observables": _with_schema(_promote_index(problem.observable_df), OBSERVABLE_COLUMNS),
        "parameters": _with_schema(
            pd.DataFrame(parameters_to_records(problem)), PARAMETER_COLUMNS
        ),
    }

    sim = getattr(session, "simulation", None) or {}
    records = sim.get("records") or []
    frames["simulations"] = pd.DataFrame(records, columns=SIMULATION_COLUMNS)

    opt = getattr(session, "optimization", None) or {}
    frames["optimization_starts"] = pd.DataFrame(
        opt.get("starts") or [], columns=OPTIMIZATION_START_COLUMNS
    )
    frames["optimization_trace"] = pd.DataFrame(
        opt.get("trace") or [], columns=OPTIMIZATION_TRACE_COLUMNS
    )
    return frames


def validate_query_args(sql: Any, max_rows: Any) -> int:
    """Validate ``query_data`` arguments cheaply (no DuckDB, no simulation).

    Returns the normalized ``max_rows``. Raises :class:`ValidationError` on bad input so the
    tool boundary can reject it *before* generating a simulation dataframe.
    """
    if not isinstance(sql, str) or not sql.strip():
        raise ValidationError("`sql` must be a non-empty SQL query string.")
    try:
        max_rows = int(max_rows)
    except (TypeError, ValueError):
        raise ValidationError(f"max_rows must be an integer >= 1, got {max_rows!r}.")
    if max_rows < 1:
        raise ValidationError(f"max_rows must be an integer >= 1, got {max_rows}.")
    return max_rows


def table_schema(session) -> dict[str, Any]:
    """Describe the queryable tables: name, description, columns, and current row count."""
    frames = _frames(session)
    tables = [
        {
            "table": name,
            "description": TABLE_DESCRIPTIONS.get(name, ""),
            "columns": [str(c) for c in df.columns],
            "n_rows": int(len(df)),
        }
        for name, df in frames.items()
    ]
    return {"tables": tables}


def run_query(session, sql: str, max_rows: int = 1000) -> dict[str, Any]:
    """Run ``sql`` over the PEtab tables + simulation dataframe and return the rows.

    Copies every queryable table (``measurements``, ``conditions``, ``observables``,
    ``parameters``, ``simulations``, ``optimization_starts``, ``optimization_trace``) into a
    fresh in-memory DuckDB database, executes ``sql``, and returns
    ``{columns, rows, row_count, truncated}`` with at most ``max_rows`` records (one extra
    row is fetched to set ``truncated``). Because the tables are a disposable copy, the query
    may use any SQL it likes without affecting the session; a SQL error is surfaced as a
    :class:`ValidationError` (a client-facing 4xx), not an internal crash.
    """
    max_rows = validate_query_args(sql, max_rows)
    duckdb = _duckdb()
    frames = _frames(session)

    con = duckdb.connect()
    try:
        # Copy each frame into a real in-memory table (not a view over the DataFrame), so the
        # query has a private, disposable copy it can freely mutate — nothing is written back
        # to the session, and the whole database is discarded when the connection closes.
        # Table names are fixed internal constants, so the CREATE is not an injection point.
        # A failure HERE (e.g. an un-registerable frame) happens before the user's SQL runs, so
        # surface it as a clean, actionable client error — pointing at the working alternatives —
        # rather than letting a raw exception crash the tool and get blindly re-proposed.
        try:
            for name, df in frames.items():
                con.register("_src", df)
                try:
                    con.execute(f'CREATE TABLE "{name}" AS SELECT * FROM _src')
                finally:
                    con.unregister("_src")  # always drop the alias, even if the CREATE raised
        except duckdb.Error as exc:
            raise ValidationError(
                f"Could not prepare the data tables for the query ({exc}). Use "
                "`get_measurements` for the raw measurement data, or `list_data_tables` for "
                "the available tables and their columns."
            ) from exc
        # Copying isolates the *data*, but not the host: a query could still read/write local
        # files or reach the network (read_csv/parquet, ATTACH, COPY, INSTALL/LOAD, httpfs).
        # Disable that and lock the setting so the query can't turn it back on.
        con.execute("SET enable_external_access=false")
        con.execute("SET lock_configuration=true")
        try:
            rel = con.execute(sql)
            columns = [d[0] for d in rel.description] if rel.description else []
            rows = rel.fetchmany(max_rows + 1)
        except duckdb.Error as exc:
            raise ValidationError(
                f"SQL error: {exc}. Call `list_data_tables` for the exact table/column names, "
                "or use `get_measurements` for the raw measurement data."
            ) from exc
    finally:
        con.close()

    truncated = len(rows) > max_rows
    rows = rows[:max_rows]
    records = [dict(zip(columns, (_clean(v) for v in row))) for row in rows]
    return {
        "columns": columns,
        "rows": records,
        "row_count": len(records),
        "truncated": truncated,
    }


def fit_diagnosis(session, worst_n: int = 8) -> dict[str, Any]:
    """A COMPACT, size-bounded residual diagnosis of the current fit.

    Unlike :func:`run_query` (arbitrary SQL, can return the whole dataset), this returns fixed
    aggregations with capped output — one row per observable (weighted χ², mean weighted
    residual = bias, worst |residual|) plus the ``worst_n`` largest-residual points — so a
    caller can see WHERE and HOW a fit is off without pulling the raw per-point data into its
    context. Computed from the last commit's simulation dataframe.
    """
    # The early returns carry residual_structure too (as the "tests unavailable" verdict): the key
    # is part of this tool's contract and the fit result's, so a caller reading it must never have
    # to guard for its absence just because there is nothing to diagnose yet.
    df = _frames(session)["simulations"]
    if df is None or df.empty:
        return {"status": "no simulation yet — commit a model first",
                "per_observable": [], "worst_points": [],
                "residual_structure": _unavailable_structure()}
    d = df[df["residual_weighted"].notna()].copy()
    if d.empty:
        return {"status": "no weighted residuals available (unfitted or non-evaluable noise)",
                "per_observable": [], "worst_points": [],
                "residual_structure": _unavailable_structure()}
    d["_rw"] = d["residual_weighted"].astype(float)

    per_observable = []
    for obs, g in d.groupby("observable_id"):
        rw = g["_rw"]
        per_observable.append({
            "observable_id": _clean(obs),
            "n": int(len(g)),
            "chi2": round(float((rw * rw).sum()), 3),
            "mean_weighted_residual": round(float(rw.mean()), 3),   # far from 0 => systematic bias
            "max_abs_weighted_residual": round(float(rw.abs().max()), 3),
            # Sign-based structure tests (see _residual_structure): these survive estimated
            # noise sigmas, which flatten the magnitude scores above. Isolated so a failure in
            # the added stats degrades to "tests unavailable" rather than taking down the whole
            # (pre-existing, robust) magnitude diagnosis for this observable.
            **_safe_residual_structure(g),
        })
    per_observable.sort(key=lambda r: r["chi2"], reverse=True)

    worst = d.reindex(d["_rw"].abs().sort_values(ascending=False).index).head(int(worst_n))
    worst_points = [{
        "observable_id": _clean(r["observable_id"]), "condition_id": _clean(r["condition_id"]),
        "time": _clean(r["time"]), "measurement": round(float(r["measurement"]), 4),
        "simulation": round(float(r["simulation"]), 4),
        "residual_weighted": round(float(r["_rw"]), 3),
    } for _, r in worst.iterrows()]

    # Rolled up from the per-observable verdicts ALREADY computed above — no second pass of the
    # sign/runs tests. Shares its builder with residual_structure_rollup (which the fit path calls
    # directly), so the two can never drift apart.
    structure = _rollup_structure((o["observable_id"], o) for o in per_observable)

    # Noise-model plausibility, echoed here too (the agent reads this tool to see WHERE the fit is
    # off): an inflated/collapsed ESTIMATED sigma means a good BIC is not genuine adequacy — the fix
    # is mechanistic, not noise-side. Best-effort; advisory only, never breaks the diagnosis (the
    # judge calls this too). Empty {} when the problem has no estimated sigma or no fit yet.
    noise_sigma: dict[str, Any] = {}
    try:
        from . import fitting
        _fit = (getattr(session, "simulation", None) or {}).get("fitted_parameters")
        _rep = fitting.noise_sigma_report(session.problem, _fit)
        if _rep:
            noise_sigma = {"report": _rep, "advice": fitting.noise_sigma_advice(_rep)}
    except Exception:  # noqa: BLE001 - advisory; never break the diagnosis
        noise_sigma = {}

    return {
        "n_points": int(len(d)),
        "total_chi2": round(float((d["_rw"] * d["_rw"]).sum()), 3),
        "per_observable": per_observable,   # bounded: one row per observable
        "worst_points": worst_points,       # bounded: <= worst_n rows
        "noise_sigma": noise_sigma,         # {} if no estimated sigma/fit; else {report, advice} (advice None when plausible)
        # The residual-structure verdict, rolled up: which observables show a significant
        # sign-test bias or runs-test time trend (p < 0.05), and whether the tests are
        # underpowered at this sample size (so "nothing flagged" is weak, not reassuring).
        # The `fit` result carries this SAME rollup (see residual_structure_rollup), so calling
        # this tool is only needed for the per-observable table and worst_points below.
        "residual_structure": structure,
        "hint": ("mean_weighted_residual far from 0 = a systematic over/under-shoot for that "
                 "observable; worst_points show the largest model-vs-data deviations. "
                 "sign_test / runs_test detect bias and time trend from residual SIGNS, so they "
                 "still work when estimated noise sigmas flatten chi2/reduced_chi2 — but are "
                 "underpowered on few points (low_power), so read a non-significant result as "
                 "weak, not as a pass. For a full breakdown or the optimizer log, delegate to "
                 "an analyst."),
    }


_STRUCTURE_UNAVAILABLE = "residual-structure tests unavailable (no weighted residuals yet)."


def residual_structure_rollup(session) -> dict[str, Any]:
    """The current fit's residual-structure verdict, rolled up — the SAME
    ``{flagged_observables, low_power, summary}`` shape :func:`fit_diagnosis` returns under its
    ``residual_structure`` key, so one key name never means two shapes.

    Which observables show significant residual STRUCTURE (sign/runs tests) — a
    form-incompleteness signal that SURVIVES estimated noise sigmas (unlike chi2 GoF, which a free
    sigma rescales to look adequate). Computed on every commit to feed the session's
    plateau-adequacy gate, and returned on the ``fit`` result so the agent gets the verdict without
    spending a separate ``get_fit_diagnosis`` call (see issue #113).

    Computes ONLY the per-observable sign/runs verdict — it skips fit_diagnosis's magnitude
    aggregations and worst-points sort, since this runs on every fit. Best-effort: no simulation,
    no weighted residuals, or any diagnostic failure degrades to a well-formed
    "tests unavailable" dict, so the gate is never blocked and the agent never hits a missing key.
    """
    try:
        df = _frames(session)["simulations"]
        if df is None or df.empty:
            return _unavailable_structure()
        d = df[df["residual_weighted"].notna()]
        if d.empty:
            return _unavailable_structure()
        verdicts = [(_clean(obs), _safe_residual_structure(g)) for obs, g in
                    d.groupby("observable_id")]
    except Exception:  # noqa: BLE001 - advisory; a diagnostic failure must not affect the fit/gate
        return _unavailable_structure()
    return _rollup_structure(verdicts)


def _rollup_structure(verdicts) -> dict[str, Any]:
    """Roll ``(observable_id, verdict)`` pairs up into the reported verdict. The single builder
    behind BOTH :func:`fit_diagnosis`'s ``residual_structure`` key and
    :func:`residual_structure_rollup`, so the shape and the rules stay identical; each caller
    passes the per-observable verdicts it has already computed."""
    pairs = list(verdicts)
    # SORTED, so the verdict does not depend on the caller's iteration order: fit_diagnosis rolls
    # up a chi2-ranked per_observable while residual_structure_rollup walks groupby order, and the
    # two must agree exactly (both the list and the summary string built from it) whenever more
    # than one observable is flagged. The chi2 RANKING lives in per_observable, not here.
    flagged = sorted(obs for obs, v in pairs if v.get("structure_flag"))
    # Low power is only a concern where nothing was flagged: a flagged observable already found
    # its structure, so its tests being underpowered elsewhere (e.g. a runs test made undefined by
    # an all-one-sign series that the sign test still flagged) is moot. This flag exists to warn
    # that "nothing flagged" is weak evidence, so it must reflect only the unflagged,
    # underpowered observables.
    low_power = any(v.get("structure_low_power") and not v.get("structure_flag")
                    for _obs, v in pairs)
    return {"flagged_observables": flagged, "low_power": low_power,
            "summary": _structure_summary(flagged, low_power)}


def _unavailable_structure() -> dict[str, Any]:
    """The degraded rollup: well-formed, and honest that nothing was tested (NOT a clean bill)."""
    return {"flagged_observables": [], "low_power": True,
            "summary": _STRUCTURE_UNAVAILABLE}


def residual_flag(structure: dict[str, Any]) -> bool | None:
    """The bare structure flag of a rollup, or ``None`` when the tests could not run at all.

    The distinction matters wherever the flag is RECORDED (the version meta, the fit event): an
    unavailable verdict flags nothing, but recording that as ``False`` would turn "untested" into a
    clean bill of health. Consumers that only gate on it (the plateau-adequacy check) coerce the
    ``None`` to falsy anyway, so they are unaffected."""
    if structure.get("summary") == _STRUCTURE_UNAVAILABLE:
        return None
    return bool(structure.get("flagged_observables"))


# p-value at/below which a residual-structure test is called significant in the rolled-up
# verdict (the per-test p-values are always reported raw, so a caller can judge for itself).
_STRUCTURE_ALPHA = 0.05


def _safe_residual_structure(group) -> dict[str, Any]:
    """:func:`_residual_structure`, but any failure in the added stats degrades to a
    tests-unavailable stub instead of taking down the whole diagnosis (which the magnitude
    aggregations above compute robustly). fit_diagnosis is called by the agent's
    get_fit_diagnosis AND by the judge, so a raise here must not break either."""
    try:
        return _residual_structure(group)
    except Exception:  # noqa: BLE001 - advisory stats; never fail the diagnosis on them
        return {"sign_test": None, "runs_test": None,
                "structure_flag": False, "structure_low_power": True}


def _residual_structure(group) -> dict[str, Any]:
    """Sign-test + runs-test summary for one observable's residuals (see
    :mod:`petab_mcp.residual_tests`).

    BOTH tests are per (condition) SERIES, then Fisher-combined across conditions into one
    p-value each. The runs test must be per-series (a runs test across conditions would invent
    spurious runs at the boundaries); the SIGN test is per-series too, so that opposite biases
    in different conditions — an over-shoot in one, an under-shoot in another — do not cancel
    into a spuriously balanced pooled count and hide a real systematic bias. Replicate points
    sharing a (condition, time) are AVERAGED to one residual first: replicates are not an
    independent draw of the residual's sign, and leaving them in would let a high-replicate
    timepoint dominate.

    ``group`` is the observable's rows from :func:`fit_diagnosis`, already restricted to points
    with a weighted residual; the raw ``residual`` (same sign, sigma being positive) is the sign
    source. Conditions are iterated once, collecting both per-series p-values and the usable
    (non-zero) point count that gauges power.
    """
    from . import residual_tests as rt

    g = group.copy()
    g["_r"] = g["residual"].astype(float)
    # Average replicates to one residual per (condition, time), then order by time per series.
    collapsed = (g.groupby(["condition_id", "time"], dropna=False)["_r"].mean()
                 .reset_index())

    sign_ps: list[float] = []
    runs_ps: list[float] = []
    n_pos = n_neg = 0
    usable = 0  # total non-zero, finite points across series — the real power denominator
    for _cond, s in collapsed.groupby("condition_id", dropna=False):
        vals = s.sort_values("time")["_r"].tolist()
        st = rt.sign_test(vals)
        n_pos += st["n_pos"]
        n_neg += st["n_neg"]
        usable += st["n"]
        if st["p"] is not None:
            sign_ps.append(st["p"])
        rres = rt.runs_test(vals)
        if rres["p"] is not None:
            runs_ps.append(rres["p"])

    sign_p = rt.fisher_combine(sign_ps)
    runs_p = rt.fisher_combine(runs_ps)
    # Power is the usable-point count, not raw len (which counts exact-zero timepoints the tests
    # drop). A runs test additionally needs a computable series (both signs present somewhere).
    low_power = usable < rt.MIN_POWER_N
    sig_bias = sign_p is not None and sign_p < _STRUCTURE_ALPHA
    sig_trend = runs_p is not None and runs_p < _STRUCTURE_ALPHA
    return {
        "sign_test": {"n_pos": n_pos, "n_neg": n_neg, "p": _round_p(sign_p),
                      "n_series": len(sign_ps), "low_power": low_power},
        "runs_test": {"p": _round_p(runs_p), "n_series": len(runs_ps),
                      "low_power": low_power or not runs_ps},
        "structure_flag": bool(sig_bias or sig_trend),
        "structure_low_power": bool(low_power or not runs_ps),
    }


def _round_p(p: float | None) -> float | None:
    return None if p is None else round(float(p), 4)


def _structure_summary(flagged: list[str], low_power: bool) -> str:
    if flagged:
        return (f"systematic residual structure in {', '.join(flagged)} "
                f"(sign-test bias or runs-test time trend, p < {_STRUCTURE_ALPHA}) — "
                "the model's FORM is missing something for those observables.")
    if low_power:
        return ("no significant residual structure, but the tests are UNDERPOWERED at this "
                "sample size — this is weak evidence of adequacy, not a pass.")
    return "no significant residual structure detected (residuals look sign-random)."
