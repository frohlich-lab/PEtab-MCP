"""SQL access to the PEtab tables and the stored simulation dataframe via DuckDB.

Instead of a fixed menu of handcoded residual-statistics tools, the server exposes the
underlying data as SQL tables the agent can query directly. Everything the model-building
loop produces is queryable in one place:

* the fixed experimental design — ``measurements``, ``conditions``, ``observables``;
* the ``parameters`` table (bounds, scale, fitted ``nominalValue``, ``estimate``, priors);
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

from .errors import MissingDependencyError, ValidationError
from .formatting import _clean

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
        "Parameter table: bounds, scale, nominalValue (fit-only), estimate, priors."
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


def _frames(session) -> dict[str, Any]:
    """Build the ``{table_name: DataFrame}`` mapping registered for a query.

    Reads the *current* problem tables and the stored simulation dataframe
    (``session.simulation``); the ``simulations`` frame is empty (but well-typed) when no
    commit has produced records yet.
    """
    import pandas as pd

    from .parameters import parameters_to_records

    problem = session.problem
    frames: dict[str, Any] = {
        "measurements": _promote_index(problem.measurement_df),
        "conditions": _promote_index(problem.condition_df),
        "observables": _promote_index(problem.observable_df),
        "parameters": pd.DataFrame(parameters_to_records(problem)),
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

    # Never register ``None`` (an absent optional table) — use an empty frame instead.
    for name, df in list(frames.items()):
        if df is None:
            frames[name] = pd.DataFrame()
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
        for name, df in frames.items():
            con.register("_src", df)
            con.execute(f'CREATE TABLE "{name}" AS SELECT * FROM _src')
            con.unregister("_src")
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
            raise ValidationError(f"SQL error: {exc}") from exc
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
