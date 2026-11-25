"""SQL access to the PEtab tables and the stored simulation dataframe via DuckDB.

Instead of a fixed menu of handcoded residual-statistics tools, the server exposes the
underlying data as SQL tables the agent can query directly. Everything the model-building
loop produces is queryable in one place:

* the fixed experimental design — ``measurements``, ``conditions``, ``experiments``,
  ``observables``;
* the ``parameters`` table (bounds, scale, fitted ``nominalValue``, ``estimate``, plus each
  parameter's ``role`` and ``status``);
* the ``simulations`` dataframe from the last commit — one row per measurement point,
  carrying the ``simulation`` value and the **residuals** (``residual``,
  ``residual_weighted`` — the chi2 term — and the evaluated ``noise``) next to the
  ``measurement``;
* ``simulations_history`` — those same per-point rows for EVERY stored version, with a
  ``version`` column, so model versions can be compared point-by-point in one query
  (assembled lazily from the version store; see :func:`_simulation_history_frame`);
* ``state_trajectories`` — the last commit's species and assignment rules at the measurement
  times (see :meth:`engine.AmiciEngine._state_trajectories`), one row per value;
* the last fit's pyPESTO multistart — ``optimization_starts`` (one row per start) and
  ``optimization_trace`` (one row per start/iteration, the objective over the run).

Residual statistics, worst-fit points, per-observable / per-experiment / per-timepoint
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

import logging
import math
import re
from collections import Counter
from typing import Any

from petab.v2.C import (
    CONDITION_ID,
    ESTIMATE,
    EXPERIMENT_ID,
    LOWER_BOUND,
    MEASUREMENT,
    NOISE_FORMULA,
    NOISE_PARAMETERS,
    NOMINAL_VALUE,
    OBSERVABLE_FORMULA,
    OBSERVABLE_ID,
    OBSERVABLE_PARAMETERS,
    PARAMETER_ID,
    TARGET_ID,
    TARGET_VALUE,
    TIME,
    UPPER_BOUND,
)

from .errors import MissingDependencyError, ValidationError
from .formatting import _clean
from .problem import PARAMETER_SCALE
from .progress_signals import signal

# Canonical (minimal) column schema for each FIXED PEtab table. Used only as a FALLBACK when
# the live table is absent (an optional table petab loaded as ``None``) or empty with no
# columns at all — e.g. a model-less problem that ships a header-only ``parameters.tsv``, whose
# parameter table has zero rows and (once turned into JSON records) zero columns. DuckDB refuses
# to register a zero-column DataFrame ("Need a DataFrame with at least one column"), and because
# every table is copied in BEFORE the query runs, one such frame made EVERY ``query_data`` call
# fail — even ``SELECT 1``. Giving the empty table its real column names keeps it queryable
# (it just returns no rows) and keeps the whole query surface working.
MEASUREMENT_COLUMNS = [OBSERVABLE_ID, EXPERIMENT_ID, TIME, MEASUREMENT]
CONDITION_COLUMNS = [CONDITION_ID, TARGET_ID, TARGET_VALUE]
EXPERIMENT_COLUMNS = [EXPERIMENT_ID, TIME, CONDITION_ID]
OBSERVABLE_COLUMNS = [OBSERVABLE_ID, OBSERVABLE_FORMULA, NOISE_FORMULA]
PARAMETER_COLUMNS = [
    PARAMETER_ID,
    PARAMETER_SCALE,
    LOWER_BOUND,
    UPPER_BOUND,
    NOMINAL_VALUE,
    ESTIMATE,
]

# Canonical column order for the per-point simulation dataframe. Declaring it up front means
# the ``simulations`` table has a stable schema (and named columns resolve in SQL) even at
# cold start, before any commit has produced records — the table is simply empty then.
SIMULATION_COLUMNS = [
    "observable_id",
    "experiment_id",
    "time",
    "measurement",
    "simulation",
    "residual",
    "residual_weighted",
    "noise",
    "noise_parameters",
]

# The cross-version copy of the simulation dataframe: SIMULATION_COLUMNS plus which stored
# version each row belongs to. Same stable-schema rationale as SIMULATION_COLUMNS.
SIMULATION_HISTORY_COLUMNS = ["version", *SIMULATION_COLUMNS]

#: The ``state_trajectories`` table: long, so its schema does not depend on the model's ids.
STATE_TRAJECTORY_COLUMNS = ["experiment_id", "time", "entity_id", "value"]

#: Rows discarded per fetch when skipping to a page ``offset``. Bounds the peak allocation of
#: the skip itself, which would otherwise be O(offset) in one go.
_OFFSET_SKIP_CHUNK = 1000

#: Hard ceiling on any single page, whatever the caller asks for. Sized from the measured cost
#: of the reads it bounds: the full raw measurement table is 35-81k tokens on problems of 541,
#: 687 and 1755 rows, so 1000 rows keeps the worst single tool result to roughly a quarter of the
#: smallest of those while still returning whole small tables in one call. Callers page past it
#: with ``next_offset``.
MAX_PAGE_ROWS = 1000

# Columns of the pyPESTO optimization tables (from the last fit). Fixing them keeps the
# empty-table schema stable before/without a fit, same as SIMULATION_COLUMNS.
OPTIMIZATION_START_COLUMNS = [
    "start",
    "id",
    "fval",
    "time",
    "n_fval",
    "n_grad",
    "n_hess",
]
OPTIMIZATION_TRACE_COLUMNS = ["start", "iteration", "fval", "time"]

# The queryable tables and a one-line description of each (surfaced by ``list_data_tables``).
TABLE_DESCRIPTIONS: dict[str, str] = {
    "measurements": (
        "Fixed measurement data — one row per measured point, keyed by the experiment it was "
        "measured in (experimentId '' = the model simulated as is)."
    ),
    "conditions": (
        "Fixed experimental conditions — one row per change a condition makes: conditionId, "
        "targetId, targetValue."
    ),
    "experiments": (
        "Fixed experiments — one row per (experiment, period start time, condition applied "
        "then); time -inf is pre-equilibration."
    ),
    "observables": "Fixed observable definitions (formula, noise formula, noise distribution).",
    "parameters": (
        "Parameter table: bounds, scale, nominalValue (fit-only), estimate, plus "
        "role (kinetic/noise_sigma/observable_scaling/initial_condition/exponent/constant) "
        "and status (estimated/fixed)."
    ),
    "simulations": (
        "Per-point simulation dataframe from the last commit: measurement vs simulation "
        "with residual, residual_weighted (the chi2 term), and noise (sigma). Empty until "
        "the first commit/fit."
    ),
    "simulations_history": (
        "The same per-point rows for EVERY stored version, with a `version` column — compare "
        "residuals across model versions in one query (e.g. pivot residual_weighted on "
        "version to see per-point where v4 got worse than v3). A version whose fit failed "
        "stores no simulation and contributes no rows."
    ),
    "state_trajectories": (
        "Every species and assignment rule of the last commit's model (entity_id) at each time its "
        "experiment was measured, in model units: one row per value. AMICI simulates no other "
        "times. Empty until the first fit."
    ),
    "optimization_starts": (
        "pyPESTO multistart results from the last fit: one row per start (start=0 is best) "
        "with fval, time, n_fval/n_grad/n_hess. Per-start exit outcomes are aggregated into "
        "the fit result's exitflag_counts histogram instead of stored per row. Empty on a "
        "simulate-only pass or before the first fit."
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

    The observable and parameter tables are indexed by their primary-key id
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
        "experiments": _with_schema(_promote_index(problem.experiment_df), EXPERIMENT_COLUMNS),
        "observables": _with_schema(_promote_index(problem.observable_df), OBSERVABLE_COLUMNS),
        "parameters": _with_schema(pd.DataFrame(parameters_to_records(problem)), PARAMETER_COLUMNS),
    }

    sim = session.simulation or {}
    records = sim.get("records") or []
    frames["simulations"] = pd.DataFrame(records, columns=SIMULATION_COLUMNS)

    opt = session.optimization or {}
    frames["optimization_starts"] = pd.DataFrame(
        opt.get("starts") or [], columns=OPTIMIZATION_START_COLUMNS
    )
    frames["optimization_trace"] = pd.DataFrame(
        opt.get("trace") or [], columns=OPTIMIZATION_TRACE_COLUMNS
    )
    return frames


def _state_trajectories(session) -> dict[str, Any]:
    """The stored ``{"ids", "points", "values"}`` of the last commit, ``{}`` where it has none."""
    return (session.simulation or {}).get("state_trajectories") or {}


def _state_trajectory_frame(session):
    """The ``state_trajectories`` table: one row per (point, id) of :func:`_state_trajectories`.
    Built only for a query that names it, like ``simulations_history``: it is ids × points rows."""
    import pandas as pd

    traj = _state_trajectories(session)
    rows = [
        (eid, t, entity, value)
        for (eid, t), values in zip(traj.get("points") or [], traj.get("values") or [])
        for entity, value in zip(traj.get("ids") or [], values)
    ]
    return pd.DataFrame(rows, columns=STATE_TRAJECTORY_COLUMNS)


def _simulation_history_frame(session):
    """One row per (stored version, measurement point): the ``simulations_history`` table.

    Assembled from each version's ``simulations.json`` via
    :meth:`ProvenanceStore.get_version_simulation` (the light read), with a PER-VERSION cache
    on the session: each entry is keyed by that version's file fingerprint (size, mtime — see
    :meth:`ProvenanceStore.simulation_fingerprint`), so a rebuild re-reads only versions whose
    file changed or appeared. Two properties hang off that key choice:

    * **rebuilds are O(new versions), not O(all)** — the store only grows, and re-parsing every
      earlier blob after each commit would make a session's total history I/O quadratic in its
      version count;
    * **the mid-commit window cannot poison the cache** — ``snapshot_version`` writes
      ``meta.json`` before ``simulations.json`` and ``query_data`` takes no session lock, so a
      query can see version N listed while its simulation is not on disk yet. That read caches
      ``(fingerprint=None, no rows)`` for N, and the entry invalidates the moment the file
      lands. (A rollback appends nothing and rewrites nothing, so it needs no invalidation.)

    Versions come from :meth:`ProvenanceStore.stored_version_numbers` (directory names), NOT
    ``list_versions``: the latter parses every ``meta.json`` on every call — pure waste when
    only the numbers are wanted — and skips versions whose meta is unreadable, which would drop
    an intact simulation from the history over an unrelated corrupt file.

    Cost, stated honestly: on a COLD cache both callers parse every version's
    ``simulations.json`` — ``table_schema`` skips only the DataFrame concat, not the parse, so
    the first ``list_data_tables`` of a session with N committed versions reads N files. What the
    fingerprint cache buys is every call AFTER that: a schema listing costs one stat per
    version, and a commit costs parsing only the version it added. Versions whose fit failed or
    was skipped store no simulation and contribute no rows.
    """
    import pandas as pd

    per_version = _simulation_history_rows(session)
    # Explicit columns=, like the HEAD `simulations` frame above: the promised schema holds
    # even if some version's records miss a key (that column is then null, not absent).
    return pd.DataFrame(
        [r for n in sorted(per_version) for r in per_version[n][1]],
        columns=SIMULATION_HISTORY_COLUMNS,
    )


def _simulation_history_rows(session) -> "dict[int, tuple[Any, list]]":
    """``{version: (fingerprint, rows)}`` for every stored version — the cache itself.

    Split from the frame builder so ``table_schema`` can report ``n_rows`` by summing lengths
    instead of materializing (and immediately discarding) a full cross-version DataFrame on
    every ``list_data_tables`` — typically an agent's FIRST call, on a store where
    a single ``simulations.json`` can be hundreds of kB.
    """
    prov = getattr(session, "provenance", None)
    if prov is None:
        return {}
    cache: dict = getattr(session, "_sim_history_cache", None) or {}
    fresh: dict = {}
    for n in prov.stored_version_numbers():
        fp = prov.simulation_fingerprint(n)
        hit = cache.get(n)
        if hit is not None and hit[0] == fp:
            fresh[n] = hit
            continue
        rows: list[dict[str, Any]] = []
        if fp is not None:
            records = (prov.get_version_simulation(n) or {}).get("records")
            # Shape-check per record: a blob can parse as an object (so the store's own
            # isinstance guard passes) and still hold `records: [1, 2, 3]` — a hand-edited
            # file, or an archive from an older schema. Splatting a non-mapping raises a
            # TypeError that would escape into every query_data AND list_data_tables call for
            # the rest of the session, which is exactly the poisoning this path exists to avoid.
            for r in records if isinstance(records, list) else []:
                if isinstance(r, dict):
                    # `version` LAST so the column is authoritative: a record carrying its own
                    # `version` field (the fit payload already stamps one at blob top level)
                    # must not silently re-attribute the row to another version.
                    rows.append({**r, "version": n})
        fresh[n] = (fp, rows)
    session._sim_history_cache = fresh
    return fresh


def query_references(sql: Any, table: str) -> bool:
    """Does ``sql`` name ``table``? A word-boundary scan of the raw query text.

    Two callers gate expensive work on this: the server skips generating the HEAD simulation
    dataframe (an AMICI build+simulate — minutes on a cold session) when a query reads neither
    ``simulations`` nor ``state_trajectories``, and :func:`run_query` skips assembling
    ``simulations_history`` or ``state_trajectories`` when a query does not read it. ``\\b`` does
    not match between ``simulations`` and ``_`` (both word characters), so the two names do not
    alias each other.

    Deliberately a text scan, with the failure modes stated: a false positive (the bare name
    inside a comment or a string literal) costs one unnecessary build, and a name reached only
    through the catalog (``SHOW TABLES``) is missed. A false negative is impossible for a query
    that actually reads the table by name, since DuckDB resolves the bare identifier the same
    way. Non-string ``sql`` answers True — validation rejects it moments later, and guessing
    "no" there would skip work a valid query needed.
    """
    if not isinstance(sql, str):
        return True
    return bool(re.search(rf"\b{re.escape(table)}\b", sql, re.IGNORECASE))


def query_reads_head_simulation(sql: Any) -> bool:
    """Does ``sql`` read a table of the HEAD simulation, ``simulations`` or
    ``state_trajectories`` (as opposed to only the history)?"""
    return query_references(sql, "simulations") or query_references(sql, "state_trajectories")


def validate_page_args(max_rows: Any, offset: Any = 0) -> "tuple[int, int]":
    """Normalize a ``(max_rows, offset)`` page window, or raise :class:`ValidationError`.

    Shared by every paged read so one bad argument is rejected identically wherever it lands —
    ``query_data``'s SQL page and ``get_measurements``' raw page have no reason to disagree
    about what a valid window is.

    ``max_rows`` is CLAMPED to :data:`MAX_PAGE_ROWS`. Paging is a guarantee, not advice: without
    a ceiling the caller can simply ask for the whole table (``max_rows=100000`` returns all
    1755 rows of a large problem, ~81k tokens) and reproduce the context blow-up paging exists to
    prevent — a tool description telling the model not to is not a bound. Clamped rather than
    rejected because the reply already reports ``truncated``/``next_offset``, so an over-large
    request degrades into a correct first page instead of an error the model must recover from.
    """
    try:
        max_rows = int(max_rows)
    except (TypeError, ValueError):
        raise ValidationError(f"max_rows must be an integer >= 1, got {max_rows!r}.")
    if max_rows < 1:
        raise ValidationError(f"max_rows must be an integer >= 1, got {max_rows}.")
    max_rows = min(max_rows, MAX_PAGE_ROWS)
    try:
        offset = int(offset)
    except (TypeError, ValueError):
        raise ValidationError(f"offset must be an integer >= 0, got {offset!r}.")
    if offset < 0:
        raise ValidationError(f"offset must be an integer >= 0, got {offset}.")
    return max_rows, offset


def validate_query_args(sql: Any, max_rows: Any, offset: Any = 0) -> "tuple[int, int]":
    """Validate ``query_data`` arguments cheaply (no DuckDB, no simulation).

    Returns the normalized ``(max_rows, offset)``. Raises :class:`ValidationError` on bad input
    so the tool boundary can reject it *before* generating a simulation dataframe.
    """
    if not isinstance(sql, str) or not sql.strip():
        raise ValidationError("`sql` must be a non-empty SQL query string.")
    return validate_page_args(max_rows, offset)


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
    # The history is listed with a real row count — so an agent deciding whether
    # cross-version comparison is even possible sees "0 rows" vs "N rows" here rather than
    # discovering it through an empty query result — but WITHOUT building the frame: the
    # columns are the declared schema and the count is a sum over the cache. This is the one
    # caller that touches the history without the query-names-it gate, and it is also the
    # most frequent one (an agent typically calls list_data_tables FIRST), so it must
    # not concatenate a cross-version DataFrame it would immediately discard.
    # Guarded: this is the one entry that touches DISK, and list_data_tables was previously
    # incapable of failing. A stale NFS handle or a permissions change on the version store
    # must not break the session's opening move — degrade to an unknown count and let the caller
    # still learn every table's columns.
    try:
        n_history = sum(len(rows) for _, rows in _simulation_history_rows(session).values())
    except Exception:
        logging.getLogger("petab_mcp").warning(
            "could not read the version store for simulations_history; reporting an unknown "
            "row count",
            exc_info=True,
        )
        n_history = None
    tables.append(
        {
            "table": "simulations_history",
            "description": TABLE_DESCRIPTIONS.get("simulations_history", ""),
            "columns": list(SIMULATION_HISTORY_COLUMNS),
            "n_rows": n_history,
        }
    )
    traj = _state_trajectories(session)
    tables.append(
        {
            "table": "state_trajectories",
            "description": TABLE_DESCRIPTIONS["state_trajectories"],
            "columns": list(STATE_TRAJECTORY_COLUMNS),
            "n_rows": len(traj.get("ids") or []) * len(traj.get("points") or []),
        }
    )
    return {"tables": tables}


def run_query(session, sql: str, max_rows: int = 1000, offset: int = 0) -> dict[str, Any]:
    """Run ``sql`` over the PEtab tables + simulation dataframe and return the rows.

    Copies the queryable tables (``measurements``, ``conditions``, ``experiments``, ``observables``,
    ``parameters``, ``simulations``, ``optimization_starts``, ``optimization_trace`` — plus
    ``simulations_history`` and ``state_trajectories``, each materialized only when the query
    names it) into a fresh in-memory DuckDB database, executes ``sql``, and returns
    ``{columns, rows, row_count, truncated}`` with at most ``max_rows`` records (one extra
    row is fetched to set ``truncated``). Because the tables are a disposable copy, the query
    may use any SQL it likes without affecting the session; a SQL error is surfaced as a
    :class:`ValidationError` (a client-facing 4xx), not an internal crash.
    """
    max_rows, offset = validate_query_args(sql, max_rows, offset)
    duckdb = _duckdb()
    frames = _frames(session)
    # Materialized ONLY when the query names it (see query_references for the trade-off):
    # assembling it parses every stored version's simulations.json on a cold cache, and the
    # overwhelmingly common HEAD-only queries must not pay that.
    if query_references(sql, "simulations_history"):
        # Guarded like table_schema's read, and for the same reason: the version store lives on
        # shared storage, and an ESTALE/EIO from the glob must not surface as a raw exception —
        # ctx.run re-raises anything that is not a PetabMcpError, the circuit breaker counts it,
        # and three in a row would disable query_data for the rest of the session over a transient
        # filesystem hiccup on a table the query merely mentioned.
        try:
            frames["simulations_history"] = _simulation_history_frame(session)
        except Exception as exc:
            raise ValidationError(
                f"Could not read the version store for `simulations_history` ({exc}). The other "
                "tables are unaffected — query `simulations` for the current model's residuals."
            ) from exc
    if query_references(sql, "state_trajectories"):
        frames["state_trajectories"] = _state_trajectory_frame(session)

    con = duckdb.connect()
    try:
        # Copy each frame into a real in-memory table (not a view over the DataFrame), so the
        # query has a private, disposable copy it can freely mutate — nothing is written back
        # to the session, and the whole database is discarded when the connection closes.
        # Table names are fixed internal constants, so the CREATE is not an injection point.
        # A failure HERE (e.g. an un-registerable frame) happens before the user's SQL runs, so
        # surface it as a clean, actionable client error — pointing at `list_data_tables` —
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
                f"Could not prepare the data tables for the query ({exc}). Call "
                "`list_data_tables` for the exact table/column names."
            ) from exc
        # Copying isolates the *data*, but not the host: a query could still read/write local
        # files or reach the network (read_csv/parquet, ATTACH, COPY, INSTALL/LOAD, httpfs).
        # Disable that and lock the setting so the query can't turn it back on.
        con.execute("SET enable_external_access=false")
        con.execute("SET lock_configuration=true")
        try:
            rel = con.execute(sql)
            columns = [d[0] for d in rel.description] if rel.description else []
            # Skip `offset` rows on the streaming cursor rather than re-running the query with
            # a LIMIT/OFFSET wrapper: the SQL surface deliberately allows statements with side
            # effects (INSERT/UPDATE/DROP over the disposable copy), so executing the caller's
            # text twice could produce two different results. Same reason no total row count is
            # reported — obtaining one means either exhausting the cursor (unbounded memory,
            # which is what paging exists to avoid) or a second execution.
            # Skipped in bounded chunks, not one fetchmany(offset): a single call would
            # materialize `offset` tuples at once, which is the unbounded-memory objection the
            # paragraph above raises against reporting a total. Peak stays O(chunk).
            remaining = offset
            while remaining > 0:
                skipped = rel.fetchmany(min(remaining, _OFFSET_SKIP_CHUNK))
                if not skipped:
                    break  # fewer rows than the offset — the page below is legitimately empty
                remaining -= len(skipped)
            rows = rel.fetchmany(max_rows + 1)
        except duckdb.Error as exc:
            raise ValidationError(
                f"SQL error: {exc}. Call `list_data_tables` for the exact table/column names."
            ) from exc
    finally:
        con.close()

    truncated = len(rows) > max_rows
    rows = rows[:max_rows]
    records = [dict(zip(columns, (_clean(v) for v in row))) for row in rows]
    out = {
        "columns": columns,
        "rows": records,
        "row_count": len(records),
        "offset": offset,
        "truncated": truncated,
    }
    if truncated:
        # The exact argument for the next page, so a caller that hit the cap does not have to
        # infer it (or rewrite its SQL with LIMIT/OFFSET and risk a different row order).
        out["next_offset"] = offset + len(records)
    return out


def version_part_summary(part: str, blob: dict[str, Any], version: int) -> dict[str, Any]:
    """Version ``version``'s stored ``part`` (:data:`provenance.VERSION_PARTS`) without its rows,
    for a ``get_version`` reply the rows would make too large to send.

    ``simulation`` keeps everything but ``records`` (the scores, ``k_eff``, the noise report and the
    parameter diagnosis a client's judge reads) and adds ``chi2_by_observable``; ``optimization``
    keeps everything but ``starts`` and ``trace`` and adds ``fvals``, each start's objective, best
    first. ``omitted`` says what was dropped and where its rows are.
    """
    if part == "simulation":
        records = blob.get("records") or []
        by_observable: dict[str, dict[str, Any]] = {}
        for r in records:
            rw = r.get("residual_weighted")
            if rw is not None:
                entry = by_observable.setdefault(r["observable_id"], {"n": 0, "chi2": 0.0})
                entry["n"] += 1
                entry["chi2"] += rw * rw
        for entry in by_observable.values():
            entry["chi2"] = round(entry["chi2"], 3)
        return {k: v for k, v in blob.items() if k != "records"} | {
            "chi2_by_observable": by_observable,
            "omitted": f"its {len(records)} records, more than one reply carries: "
            f"simulations_history WHERE version = {version} holds them",
        }
    starts = blob.get("starts") or []
    return {k: v for k, v in blob.items() if k not in ("starts", "trace")} | {
        "fvals": [s.get("fval") for s in starts],
        "omitted": f"its {len(starts)} starts and {len(blob.get('trace') or [])} trace rows, more "
        "than one reply carries: optimization_starts and optimization_trace hold the "
        "head version's",
    }


#: Why the residual section is empty: there are no per-point records for this model and a READ does
#: not compile one to make them (see :func:`fitting.ensure_simulation`).
_NO_SIMULATION_STATUS = (
    "no simulation to diagnose — no fit of this model has produced one, and a read does not compile "
    "a model to make one."
)

#: Why ``get_fit_diagnosis(version=N)`` has no residual section for that version.
NO_STORED_RECORDS_STATUS = (
    "version {version} stored no per-point records (its fit produced none), so it has no "
    "residuals to diagnose. A read never refits to make them."
)


def fit_diagnosis(
    session,
    worst_n: int = 8,
    max_observables: int | None = 12,
    *,
    no_records: str = _NO_SIMULATION_STATUS,
) -> dict[str, Any]:
    """A COMPACT, size-bounded residual diagnosis of the current fit — the ``residuals`` section of
    ``get_fit_diagnosis``. ``session`` may be a stored version's fit (:meth:`PetabSession.stored_fit`)
    instead, and ``no_records`` is the ``status`` said where there are no per-point records.

    Unlike :func:`run_query` (arbitrary SQL, can return the whole dataset), this returns fixed
    aggregations with capped output — one row per observable (weighted χ², mean weighted
    residual = bias, worst |residual|) plus the ``worst_n`` largest-residual points — so a
    caller can see WHERE and HOW a fit is off without pulling the raw per-point data into its
    context. Computed from the last commit's simulation dataframe.

    It does NOT return the rolled-up ``residual_structure`` verdict: that rides on every fit result
    already (:func:`residual_structure_rollup`), and repeating it here shipped the same ~0.5 kB
    twice to a caller that has both. The per-observable rows still carry their own sign/runs
    verdicts — the detail this section exists for — and the observables the rollup flags are marked
    ``structure_flag`` here too.

    Empty-but-typed, with ``status`` saying why, when there are no per-point records: the model has
    no simulation and one is not MADE here if that would mean compiling it (see
    :func:`fitting.ensure_simulation`). An empty ``per_observable`` must not read as a clean bill of
    health, so the reason travels with it.
    """
    df = _frames(session)["simulations"]
    if df is None or df.empty:
        return {"status": no_records, "per_observable": [], "worst_points": []}
    d = df[df["residual_weighted"].notna()].copy()
    if d.empty:
        return {
            "status": "no weighted residuals available (unfitted or non-evaluable noise)",
            "per_observable": [],
            "worst_points": [],
        }
    d["_rw"] = d["residual_weighted"].astype(float)

    centred = _offset_centred_observables(session.problem)
    per_observable = []
    for obs, g in d.groupby("observable_id"):
        rw = g["_rw"]
        per_observable.append(
            {
                "observable_id": _clean(obs),
                "n": int(len(g)),
                "chi2": round(float((rw * rw).sum()), 3),
                "mean_weighted_residual": round(
                    float(rw.mean()), 3
                ),  # far from 0 => systematic bias
                "max_abs_weighted_residual": round(float(rw.abs().max()), 3),
                # Sign-based structure tests (see _residual_structure): no sigma rescales these,
                # unlike the magnitude scores above. Isolated so a failure in
                # the added stats degrades to "tests unavailable" rather than taking down the whole
                # (pre-existing, robust) magnitude diagnosis for this observable.
                **_safe_residual_structure(g, centred=_clean(obs) in centred),
                **({"by_construction": _offset_note(g)} if _clean(obs) in centred else {}),
            }
        )
    per_observable.sort(key=lambda r: r["chi2"], reverse=True)

    worst = d.reindex(d["_rw"].abs().sort_values(ascending=False).index).head(int(worst_n))
    worst_points = [
        {
            "observable_id": _clean(r["observable_id"]),
            "experiment_id": _clean(r["experiment_id"]),
            "time": _clean(r["time"]),
            "measurement": round(float(r["measurement"]), 4),
            "simulation": round(float(r["simulation"]), 4),
            "residual_weighted": round(float(r["_rw"]), 3),
        }
        for _, r in worst.iterrows()
    ]

    # Keep the per-observable rows consistent with the rolled-up verdict. The replicate arm flags a
    # DESIGN POINT, not a series, so an observable it condemns would otherwise read
    # structure_flag: false here while appearing in the verdict's flagged list.
    #
    # READ off the verdict the fit already stored on this very frame (session.fit stashes it on the
    # simulation payload — see residual_structure_rollup), rather than rolling it up again: the
    # replicate arm re-runs a whole lack-of-fit pass per noise parameter, and this read exists to
    # report the DETAIL, not to recompute the verdict. Recomputed only when the frame carries none
    # (a simulate-only or restored payload), and that fallback goes through the SAME builder the fit
    # path uses, so the two cannot disagree about which observables are flagged.
    stored = (session.simulation or {}).get("residual_structure")
    if isinstance(stored, dict):
        flagged = set(stored.get("flagged_observables") or ())
    else:
        flagged = set(
            _rollup_structure(
                ((o["observable_id"], o) for o in per_observable),
                replicate_lof=_replicate_lof(session, df),
            )["flagged_observables"]
        )
    for o in per_observable:
        if o["observable_id"] in flagged:
            o["structure_flag"] = True

    # SIZE CAP, worst-first, applied AFTER the structure flags exist. "One row per observable"
    # stops being compact at scale: on a 65-observable problem this table alone was ~6-8 kB
    # per call, and it rides on payloads the caller re-sends in history on every later turn —
    # the agent's history there grew 15.5k tokens/TURN and three of nine sessions died on the
    # context window before their fit budget. Worst-K keeps the rows the caller acts on (sorted by
    # chi2, and every structure-FLAGGED row is retained regardless of rank, so the cap can never
    # hide a verdict); the omission note says what was dropped, so a truncated table cannot read
    # as a complete one.
    # Selection keeps the rows a caller ACTS on, under a HARD ceiling. Three ranks are unioned
    # because three different consumers each have their own notion of "worst" and a cap that
    # knows only one of them silently changes their answers: chi2 (this table's own sort),
    # |mean_weighted_residual| (a client's finish gate picks its headline observable by bias), and
    # the structure flags (the rolled-up verdict must never be contradicted by a truncated table).
    # The flagged set is CAPPED too — on a badly-fitting 65-observable problem the rollup can
    # flag 50+, and an unbounded "keep all flagged" escape reproduces exactly the payload the cap
    # exists to remove, in precisely the regime where context pressure is worst.
    # `max_observables=None` (or 0) means NO cap — for the OFFLINE readers. The cap exists for
    # the agent's context window; an offline scorer has no such pressure and what it records is
    # permanent, so truncating there loses rows for every future analysis of the session. Same
    # function, different budget, chosen by the caller.
    _CAP, _BIAS_KEEP, _FLAG_KEEP = (max_observables or 0), 3, 6
    omitted_note = None
    if _CAP and len(per_observable) > _CAP:
        keep, seen = [], set()

        def _take(rows):
            for r in rows:
                if id(r) not in seen:
                    seen.add(id(r))
                    keep.append(r)

        _take(per_observable[:_CAP])  # worst by chi2
        _take(
            sorted(
                per_observable,
                key=lambda r: abs(r.get("mean_weighted_residual") or 0.0),
                reverse=True,
            )[:_BIAS_KEEP]
        )  # worst by BIAS
        _take([r for r in per_observable if r.get("structure_flag")][:_FLAG_KEEP])
        omitted = [r for r in per_observable if id(r) not in seen]
        n_flagged_omitted = sum(1 for r in omitted if r.get("structure_flag"))
        per_observable = [r for r in per_observable if id(r) in seen]  # keep chi2 order
        omitted_note = {
            "n_observables_omitted": len(omitted),
            "omitted_chi2_total": round(sum(r["chi2"] for r in omitted), 3),
            "n_flagged_omitted": n_flagged_omitted,
            "note": (
                f"TRUNCATED: showing the worst {_CAP} by chi2, the worst {_BIAS_KEEP} by "
                f"|mean_weighted_residual| and up to {_FLAG_KEEP} structure-flagged rows. "
                "Aggregates over these rows are NOT whole-problem aggregates. The rolled-up "
                "residual_structure verdict on the fit result covers every observable."
            ),
        }

    # Noise-model plausibility, echoed here too (the agent reads this tool to see WHERE the fit is
    # off): an inflated/collapsed ESTIMATED sigma means a good BIC is not genuine adequacy — the fix
    # is mechanistic, not noise-side. Best-effort; advisory only, never breaks the diagnosis (a
    # scorer outside the session calls this too). Empty {} when the problem has no estimated sigma
    # or no fit yet.
    noise_sigma: dict[str, Any] = {}
    _rep = None
    try:
        from . import fitting

        _fit = (session.simulation or {}).get("fitted_parameters")
        _rep = fitting.noise_sigma_report(session.problem, _fit)
        if _rep:
            residual = {"flagged_observables": sorted(flagged)}
            advice = fitting.noise_sigma_advice(_rep, session.problem, residual)
            noise_sigma = {"report": _rep, "advice": advice}
    except Exception:  # noqa: BLE001 - advisory; never break the diagnosis
        noise_sigma = {}
    signals = sigma_misfit_signals(session, report=_rep, sims=df)
    if signals["fitted_over_data_noise"]:
        noise_sigma["fitted_over_data_noise"] = signals["fitted_over_data_noise"][:_SIGNALS_KEEP]
    caveat = structure_tests_caveat(session.problem)
    total_chi2 = float((d["_rw"] * d["_rw"]).sum())
    chi2_note = _chi2_pinned(session.problem, d)

    return {
        "n_points": int(len(d)),
        "total_chi2": round(total_chi2, 3),
        **({"total_chi2_note": _CHI2_BY_CONSTRUCTION} if chi2_note else {}),
        "per_observable": per_observable,  # bounded: worst-12 by chi2 + all flagged rows
        **({"omitted": omitted_note} if omitted_note else {}),
        "worst_points": worst_points,  # bounded: <= worst_n rows
        # {} if no estimated sigma/fit, else {report, advice (None when plausible),
        # fitted_over_data_noise}
        "noise_sigma": noise_sigma,
        "sigma_heterogeneity": [
            {**h, "worst": h["worst"][:3]} for h in signals["heterogeneity"][:_SIGNALS_KEEP]
        ],
        **({"structure_tests_caveat": caveat} if caveat else {}),
        "hint": (
            "mean_weighted_residual far from 0 = a systematic over/under-shoot for that "
            "observable; worst_points show the largest model-vs-data deviations. "
            "sign_test / runs_test detect bias and time trend from residual SIGNS, which no "
            "sigma rescales — but are underpowered on few points (low_power), so read a "
            "non-significant result as "
            "weak, not as a pass. The rolled-up verdict over these tests — including the "
            "replicate_lack_of_fit arm, whose PASS is worth only its lof_df (replicated points "
            "MINUS your estimated structural parameters): weak: true, or no lof_df, certifies "
            "nothing — is the residual_structure block on the fit result, not repeated here. "
            "A full breakdown and the optimizer's log are separate reads."
        ),
    }


#: How many sigmas each ranked list in :func:`fit_diagnosis` keeps, worst first.
_SIGNALS_KEEP = 5

#: :func:`fit_diagnosis`'s ``total_chi2_note`` and a fit result's ``chi2_note``, where
#: :func:`_chi2_pinned` holds.
_CHI2_BY_CONSTRUCTION = (
    "chi2 equals the number of points by construction: the fit estimates every noise sigma, and at "
    "its optimum an estimated sigma sets the chi2 of the points it covers to their count, so chi2 "
    "says nothing about the model" + signal(", and neither does its goodness-of-fit p") + "."
)

#: A ``per_observable`` row's ``by_construction``, on an observable of
#: :func:`_offset_centred_observables` whose points share one noise sd.
_OFFSET_CENTRED = (
    "an estimated offset centres these residuals: mean_weighted_residual is ~0 and sign_test "
    "cannot see a bias, whatever the model."
)

#: ...and on one whose noise sd varies by point (SEMs per point, say).
_OFFSET_CENTRED_PRECISION_WEIGHTED = (
    "an estimated offset zeroes the precision-weighted mean of these residuals (weights "
    "1/sigma^2), not mean_weighted_residual: sigma varies by point here, so mean_weighted_residual "
    "and sign_test can still show a bias, unless sigma is one value across the points each offset "
    "covers."
)


def _offset_note(group) -> str:
    """The ``by_construction`` note of an offset-centred observable's rows ``group``. The offset
    solves ``sum(residual / sigma^2) = 0`` over the points it covers, which zeroes the mean of
    their weighted residuals only where they share one sigma (PEtab-MCP#285)."""
    noise = group["noise"].dropna().astype(float)
    varies = len(noise) > 1 and not math.isclose(noise.min(), noise.max(), rel_tol=1e-6)
    return _OFFSET_CENTRED_PRECISION_WEIGHTED if varies else _OFFSET_CENTRED


def chi2_by_construction_note(problem, records) -> str | None:
    """:data:`_CHI2_BY_CONSTRUCTION` where a fit's per-point ``records`` hold :func:`_chi2_pinned`,
    else ``None``, also on any failure. Read by every fit result and ``get_fit_summary``."""
    import pandas as pd

    try:
        df = pd.DataFrame(records or [], columns=SIMULATION_COLUMNS)
        d = df[df["residual_weighted"].notna()]
        return _CHI2_BY_CONSTRUCTION if len(d) and _chi2_pinned(problem, d) else None
    except Exception:  # noqa: BLE001 - advisory; never break a fit result
        return None


def _chi2_pinned(problem, d) -> bool:
    """Whether the chi2 of ``d`` (weighted records) equals its row count because
    :func:`_sigma_estimated_everywhere` holds."""
    rw = d["residual_weighted"].astype(float)
    return math.isclose(float((rw * rw).sum()), len(d), rel_tol=1e-3) and (
        _sigma_estimated_everywhere(problem, d)
    )


def _sigma_estimated_everywhere(problem, d) -> bool:
    """Whether every row of ``d`` (:func:`fit_diagnosis`'s weighted records) has an estimated noise
    sigma, solved analytically (:func:`reconcile.classify_inner_parameters`) or optimized with the
    rest. Where the noise sd scales with its sigmas, the likelihood is stationary in them only where
    the weighted chi2 of their points is their count, so an outer sigma pins it as an inner one does
    (PEtab-MCP#285). ``False`` on any failure."""
    from . import reconcile as R

    try:
        estimated = {s for s in R.noise_sigma_ids(problem) if R.is_parameter_estimated(problem, s)}
        frame = d.rename(
            columns={
                "observable_id": OBSERVABLE_ID,
                "experiment_id": EXPERIMENT_ID,
                "noise_parameters": NOISE_PARAMETERS,
            }
        )
        covered: set = set()
        for pid in estimated:
            rows = R._measurement_rows_for_sigma(problem, pid, frame)
            if rows is not None:
                covered |= set(rows.index)
        return bool(covered) and covered >= set(d.index)
    except Exception:  # noqa: BLE001 - advisory; never break the diagnosis
        return False


def _offset_centred_observables(problem) -> set[str]:
    """Observables every measurement of which reads an offset the fit solves analytically
    (:func:`reconcile.classify_inner_parameters`), in its formula or through its
    ``observableParameters``. That offset is the mean of its points' residuals, so it puts their
    mean at 0 and takes the sign test's power on bias. Empty on any failure."""
    from . import reconcile as R
    from .problem import split_overrides

    try:
        inner = R.classify_inner_parameters(problem)
        offsets = {pid for pid, kind in inner.items() if kind == R.INNER_TYPE_OFFSET}
        if not offsets:
            return set()
        odf, mdf = problem.observable_df, problem.measurement_df
        out = set()
        for obs_id, formula in odf[OBSERVABLE_FORMULA].items():
            if R.formula_symbols(formula) & offsets:
                out.add(str(obs_id))
                continue
            if OBSERVABLE_PARAMETERS not in mdf.columns:
                continue
            cells = mdf.loc[mdf[OBSERVABLE_ID] == obs_id, OBSERVABLE_PARAMETERS]
            if len(cells) and all(
                isinstance(c, str) and set(map(str, split_overrides(c))) & offsets for c in cells
            ):
                out.add(str(obs_id))
        return out
    except Exception:  # noqa: BLE001 - advisory; never break the diagnosis
        return set()


#: :func:`structure_tests_caveat`'s line; ``{}`` is what is estimated.
_ABSORBED_MISFIT_CAVEAT = (
    "Estimated {} absorb misfit, so the sign/runs tests have little power here and no structure "
    "flag is NOT adequacy. sigma_heterogeneity and fitted_over_data_noise rank where to look."
)


def sigma_misfit_signals(session, report=None, sims=None) -> dict[str, list[dict[str, Any]]]:
    """Where an estimated noise sigma absorbed a misfit: two ranked lists, uncapped, worst first.

    * ``heterogeneity``: per estimated noise sigma, :func:`residual_tests.sigma_heterogeneity`
      across the (observable, experiment) series it covers, as ``{parameter_id, p, n_series,
      worst}``, smallest ``p`` first. A series is labelled by its experiment, or as
      ``observable[experiment]`` where the sigma covers several observables.
    * ``fitted_over_data_noise``: ``{parameter_id, fitted_over_data_noise, n_points, low_power}``
      off ``report`` (:func:`fitting.noise_sigma_report`, computed when ``None``), largest first;
      ``low_power`` below ``NOISE_SIGMA_MIN_POINTS`` measurements, where the report's ``status`` is
      ``unknown``.

    ``sims`` is the simulations frame when the caller has one. Both lists are empty on any failure.
    """
    from . import fitting

    try:
        if report is None:
            report = fitting.noise_sigma_report(
                session.problem, (session.simulation or {}).get("fitted_parameters")
            )
        ratios = _ranked_noise_ratios(report)
    except Exception:  # noqa: BLE001 - advisory; never break the diagnosis
        ratios = []
    try:
        if sims is None:
            sims = _frames(session)["simulations"]
        heterogeneity = _sigma_heterogeneity(session.problem, sims)
    except Exception:  # noqa: BLE001 - advisory; never break the diagnosis
        heterogeneity = []
    return {"heterogeneity": heterogeneity, "fitted_over_data_noise": ratios}


def _ranked_noise_ratios(report) -> list[dict[str, Any]]:
    from .constants import NOISE_SIGMA_MIN_POINTS

    out = []
    for pid, entry in (report or {}).items():
        ratio = entry.get("fitted_over_data_noise")
        if ratio is None or not math.isfinite(ratio):
            continue
        n = entry.get("n_points")
        out.append(
            {
                "parameter_id": pid,
                "fitted_over_data_noise": round(float(ratio), 3),
                "n_points": n,
                "low_power": n is None or n < NOISE_SIGMA_MIN_POINTS,
            }
        )
    return sorted(out, key=lambda r: r["fitted_over_data_noise"], reverse=True)


def _sigma_heterogeneity(problem, sims) -> list[dict[str, Any]]:
    from . import reconcile as R
    from . import residual_tests as rt

    if sims is None or sims.empty:
        return []
    # A sigma's own records, picked as its measurement rows are: renamed to PEtab's columns.
    frame = sims[sims["residual_weighted"].notna()].rename(
        columns={
            "observable_id": OBSERVABLE_ID,
            "experiment_id": EXPERIMENT_ID,
            "noise_parameters": NOISE_PARAMETERS,
        }
    )
    out = []
    for pid in _noise_parameter_ids(problem):
        if not R.is_parameter_estimated(problem, pid):
            continue
        sub = R._measurement_rows_for_sigma(problem, pid, frame)
        if sub is None:
            continue
        single = sub[OBSERVABLE_ID].nunique() == 1
        series = {}
        for (obs, exp), g in sub.groupby([OBSERVABLE_ID, EXPERIMENT_ID], sort=False, dropna=False):
            label = (str(exp) if single else f"{obs}[{exp}]") if exp else str(obs)
            series[label] = g["residual_weighted"].tolist()
        verdict = rt.sigma_heterogeneity(series)
        if verdict:
            out.append({"parameter_id": pid, **verdict})
    return sorted(out, key=lambda v: (v["p"] is None, v["p"] or 0.0))


def structure_tests_caveat(problem) -> str | None:
    """The line :func:`fit_diagnosis` carries while noise sigmas or observable scales/offsets are
    estimated, and :func:`residual_structure_rollup` while it also flags no observable, or
    ``None``."""
    from . import reconcile as R

    df = getattr(problem, "parameter_df", None)
    table = set() if df is None else {str(pid) for pid in df.index}

    def estimated(ids) -> bool:
        return any(R.is_parameter_estimated(problem, pid) for pid in ids & table)

    what = [
        name
        for name, ids in (
            ("noise sigmas", R.plausibility_sigma_ids(problem)),
            ("observable scales/offsets", R.observable_scaling_ids(problem)),
        )
        if estimated(ids)
    ]
    return _ABSORBED_MISFIT_CAVEAT.format(" and ".join(what)) if what else None


_STRUCTURE_UNAVAILABLE = "residual-structure tests unavailable (no weighted residuals yet)."


def residual_structure_rollup(session, sims=None) -> dict[str, Any]:
    """The current fit's residual-structure verdict, rolled up:
    ``{flagged_observables, low_power, summary, replicate_lack_of_fit}``, plus the pointers
    :func:`_absorbed_misfit` adds, which flag nothing. THE authority on where
    this verdict is reported: on every fit result and on ``get_fit_summary`` (both score payloads
    for one fit, assembled by :func:`fitting.fit_result_block`/:func:`fitting.fit_summary`) —
    and NOT by ``get_fit_diagnosis``, which returns the per-observable detail behind it.

    Which observables show significant residual STRUCTURE — a signal no sigma rescales (unlike chi2
    GoF), though estimated scales and offsets take most of its power
    (:func:`structure_tests_caveat`). THREE tests, all feeding the one ``flagged_observables`` list,
    which no stop condition reads:

    * the sign test and the runs test, per observable, over the points WITHIN a residual series;
    * the replicate lack-of-fit test, per replicated design point, ACROSS that point's own
      replicates (:func:`_replicate_lof`) — the axis the sign tests cannot reach when a design point
      IS an observable (one steady-state condition, so every series has length 1).

    Computed on every commit, and returned on the ``fit`` result so the agent gets the verdict
    without spending a ``get_fit_diagnosis`` call at all (see issue #113). Computes ONLY the
    verdict — it skips fit_diagnosis's magnitude aggregations and worst-points sort. ``sims`` is the
    simulations frame when the caller already built one. Best-effort: no simulation, or any
    diagnostic failure, degrades to a well-formed "tests unavailable" dict, so a fit is never
    blocked and the agent never hits a missing key.
    """
    try:
        df = _frames(session)["simulations"] if sims is None else sims
        if df is None or df.empty:
            return _unavailable_structure()
    except Exception:  # noqa: BLE001 - advisory; a diagnostic failure must not affect the fit
        return _unavailable_structure()
    # The replicate arm runs FIRST and OUTSIDE the residual_weighted gate below. z_i is built from
    # the pooled replicate scatter and the raw prediction, so it needs no fitted sigma at all —
    # and gating it behind the sigma-DERIVED residual_weighted column switched it off in exactly the
    # cases it exists for: engine._weighted_residual leaves the whole column NaN both when the noise
    # formula will not evaluate and when the evaluated sd is 0.0, a COLLAPSED sigma.
    lof = _replicate_lof(session, df)
    try:
        d = df[df["residual_weighted"].notna()]
        # The sign/runs arm keeps taking its rows from the SAME filter fit_diagnosis's magnitude
        # table uses, so the two builders cannot roll up different inputs.
        centred = _offset_centred_observables(session.problem)
        verdicts = (
            []
            if d.empty
            else [
                (_clean(obs), _safe_residual_structure(g, centred=_clean(obs) in centred))
                for obs, g in d.groupby("observable_id")
            ]
        )
    except Exception:  # noqa: BLE001 - advisory; never fail the fit on a diagnostic
        verdicts = []
    if not verdicts and not lof.get("available"):
        return _unavailable_structure()  # nothing was tested at all — NOT a clean bill
    out = _rollup_structure(verdicts, replicate_lof=lof)
    return {**out, **_absorbed_misfit(session, df, flagged=bool(out["flagged_observables"]))}


def _absorbed_misfit(session, sims, flagged: bool) -> dict[str, Any]:
    """The rollup's pointers to where an estimated sigma absorbed a misfit (#195), each key absent
    when empty, ``{}`` on any failure.

    ``sigma_heterogeneity``: the sigmas whose :func:`residual_tests.sigma_heterogeneity` ``p`` stays
    below ``_STRUCTURE_ALPHA`` after Holm across the sigmas tested, smallest first, each with that
    ``p``, its ``worst_series`` and its ``fitted_over_data_noise``. Holm, where :func:`fit_diagnosis`
    ranks raw ``p``, because these ride on every fit: across bachmann's ~20 sigmas the passing fit's
    pEpoR (p = 0.006) would be named on every fit, the failing fit's (9e-5) is named either way. The
    ratio rides only on a named sigma, as it has no threshold of its own (``fitting._entry_inflated``).

    ``structure_tests_caveat`` while no observable is ``flagged``: where one is, the caveat does
    not change how the verdict reads.
    """
    from . import fitting
    from . import residual_tests as rt

    out: dict[str, Any] = {}
    try:
        tested = [h for h in _sigma_heterogeneity(session.problem, sims) if h["p"] is not None]
        adjusted = rt.holm_adjusted([h["p"] for h in tested])
        named = [h for h, p in zip(tested, adjusted) if p < _STRUCTURE_ALPHA]
        if named:
            fitted = (session.simulation or {}).get("fitted_parameters")
            ratios = {
                r["parameter_id"]: r["fitted_over_data_noise"]
                for r in _ranked_noise_ratios(fitting.noise_sigma_report(session.problem, fitted))
            }
            out["sigma_heterogeneity"] = [
                {
                    "parameter_id": h["parameter_id"],
                    "p": h["p"],
                    "worst_series": h["worst"][0]["series"],
                    "fitted_over_data_noise": ratios.get(h["parameter_id"]),
                }
                for h in named
            ]
        caveat = None if flagged else structure_tests_caveat(session.problem)
        if caveat:
            out["structure_tests_caveat"] = caveat
    except Exception:  # noqa: BLE001 - advisory; never fail the fit on a diagnostic
        return {}
    return out


def _rollup_structure(verdicts, replicate_lof: dict[str, Any] | None = None) -> dict[str, Any]:
    """Roll ``(observable_id, verdict)`` pairs plus the replicate lack-of-fit arm up into the
    reported verdict. The single builder behind BOTH :func:`residual_structure_rollup` (which
    reports it) and :func:`fit_diagnosis` (which uses it only to mark its own rows), so the two
    cannot disagree about which observables are flagged; each caller passes the per-observable
    verdicts (and the one replicate arm) it has already computed."""
    from . import residual_tests as rt

    pairs = list(verdicts)
    lof = replicate_lof or _lof_payload(rt.replicate_lof_unavailable())
    # SORTED, so the verdict does not depend on the caller's iteration order: fit_diagnosis rolls
    # up a chi2-ranked per_observable while residual_structure_rollup walks groupby order, and the
    # two must agree exactly (both the list and the summary string built from it) whenever more
    # than one observable is flagged. The chi2 RANKING lives in per_observable, not here.
    # UNION over the three tests: the replicate test names the design points it condemns, mapped back to
    # their observables, so a lack-of-fit finding lands in the SAME list as a sign/runs one.
    flagged = sorted(
        {obs for obs, v in pairs if v.get("structure_flag")}
        | set(lof.get("flagged_observables") or ())
    )
    # Low power is only a concern where nothing was flagged: a flagged observable already found
    # its structure, so its tests being underpowered elsewhere (e.g. a runs test made undefined by
    # an all-one-sign series that the sign test still flagged) is moot. This flag exists to warn
    # that "nothing flagged" is weak evidence, so it must reflect only the unflagged,
    # underpowered observables.
    #
    # ...and it is cleared for an observable the replicate arm COVERED: there the deviation was
    # tested against its noise sigma's pooled replicate scatter over >= 3 points, with lack-of-fit
    # directions to spare (`_merge_lof_verdicts` credits coverage only from a POWERED band), so an
    # unflagged verdict is a genuine pass rather than "we could not tell". Only for the covered
    # ones — clearing it globally as soon as ANY arm ran asserted sign-randomness for observables
    # with no replicates at all, and that over-claim reaches every record of the verdict and any
    # client's finish gate keyed on it.
    #
    # KNOWN LIMIT, accepted: ``covered`` is per OBSERVABLE while the sign/runs tests are per SERIES,
    # so an observable with one replicated condition and one single-measurement condition is credited
    # on the replicated one. Bounded, and the summary now names which observables the sign/runs arm
    # was underpowered for and which points the replicate arm actually tested, so the two claims can
    # be told apart. Not reached on any problem it was measured on (none mixes a replicated and a
    # non-replicated condition within one observable).
    # An arm with too few lack-of-fit directions to certify anything on its own axis credits NO
    # coverage, so it cannot stand in for the sign/runs tests being underpowered — decided once, in
    # `_merge_lof_verdicts`, where the df and the threshold live (an empty `covered_observables`
    # arrives here). Re-testing `weak` here instead would have missed the FLAGGED low-df arm, whose
    # `weak` is False by construction.
    covered = set(lof.get("covered_observables") or ())
    weak = sorted(
        obs
        for obs, v in pairs
        if v.get("structure_low_power") and not v.get("structure_flag") and obs not in covered
    )
    low_power = bool(weak)
    return {
        "flagged_observables": flagged,
        "low_power": low_power,
        "summary": _structure_summary(
            flagged,
            low_power,
            lof,
            sign_runs_tested=bool(pairs),
            # NAMED only where the replicate arm claims a pass for
            # OTHER observables, which is the only case where the two
            # halves of the summary contradict each other. With nothing
            # covered there is no competing claim to disambiguate, and
            # naming every underpowered observable would put a list of
            # them on every fit (22 on one problem: a 134 -> 1053 char
            # summary, forever, to say what `low_power` already says).
            low_power_observables=weak if covered else [],
        ),
        "replicate_lack_of_fit": _lof_reported(lof),
    }


def _unavailable_structure() -> dict[str, Any]:
    """The degraded rollup: well-formed, and honest that nothing was tested (NOT a clean bill)."""
    from . import residual_tests as rt

    return {
        "flagged_observables": [],
        "low_power": True,
        "summary": _STRUCTURE_UNAVAILABLE,
        "replicate_lack_of_fit": _lof_reported(_lof_payload(rt.replicate_lof_unavailable())),
    }


# -- the replicate lack-of-fit arm ---------------------------------------------------------------
# Assembly only: the STATISTIC lives in residual_tests.replicate_lack_of_fit and the PEtab-shaped
# plumbing (which rows are one design point, and the coordinate their noise is additive on) in
# reconcile.replicate_pure_error_points. This layer joins the two, per noise parameter, and turns
# the result into the compact agent-facing verdict.


#: Keys the ASSEMBLER consumes that must never ride on a fit result: the unbounded per-point and
#: per-observable label lists, and the advice line the summary already carries. Carrying any of them
#: twice would cost ~700 bytes on every fit — and the fit result is re-sent in the agent's history on
#: every later turn (PR #23 cut ~40% of the per-call prompt by trimming exactly this duplication).
#: Named in ONE place, and applied at both projections, so a key added to the verdict cannot leak
#: into the payload by being forgotten at the second.
_LOF_ROLLUP_ONLY_KEYS = (
    "flagged",
    "tested",
    "flagged_observables",
    "covered_observables",
    "advice",
)

#: ...and the keys DROPPED from a fit payload when they have nothing to say, so that the same
#: question is not settled per key. `lof_df` and `weak` are absent together wherever the df could not
#: be counted (an UNKNOWN structural count) and `weak` alone wherever there is nothing to qualify, so
#: absence never has to be told apart from a null: every consumer reads them with ``.get()``.
#: `lof_df` is never 0 on an available arm (df <= 0 is `vacuous`, which is not available), so dropping
#: on falsy cannot hide a real count.
#: Measured, so the next reader does not have to guess whether this pays: the arm's payload on a fit
#: is 15-31 bytes smaller than carrying the nulls (184 -> 153 on the 13 of 16 measured problems that
#: cannot run it, which is the case that repeats on every fit forever — PR #23), against ~27 more in
#: the summary where the arm PASSES, because the powered sentence now names the df it rests on.
_LOF_OMIT_WHEN_EMPTY_KEYS = ("lof_df", "weak", "untested_status")


def _lof_payload(
    verdict: dict[str, Any], observables=(), advice: str | None = None, covered=()
) -> dict[str, Any]:
    """The replicate arm's verdict: the statistic's own keys with the (unbounded) label lists replaced
    by a count, plus the observables it condemns, the observables it COVERED, and — only when it
    flagged — the advice line. ONE projection for every path, so no caller can build a shape of its
    own. The extra keys are consumed by :func:`_rollup_structure` (into the rollup's own list, its
    ``low_power`` decision and its summary) and stripped from what rides on the fit result.

    The shape is a fixed CORE plus named optional keys, which is a different policy from the
    statistic's (every key on every path — that verdict is the assembler's contract, and rebuilding
    it from a stale key list is how ``lof_df``/``weak`` were dropped last time). The core is what the
    statistic emits minus ``_LOF_ROLLUP_ONLY_KEYS``; the optional keys are
    ``_LOF_OMIT_WHEN_EMPTY_KEYS``, dropped when they have nothing to say, plus ``advice``, which is
    not part of the verdict at all — the assembler passes it in, and only on a flag. Two names, one
    rule each, and the shape test pins the result of both.

    Observable ids are coerced to ``str`` as a belt-and-braces invariant, not as the fix for the
    TypeError ``sorted({None, 'obs_A'})`` used to raise: the merge already substitutes a point's own
    label for a missing observableId, so nothing None-bearing can reach here through it. This keeps
    the guarantee local for the other two call sites and for any future one."""
    out = {
        k: v
        for k, v in verdict.items()
        if k not in _LOF_ROLLUP_ONLY_KEYS and not (k in _LOF_OMIT_WHEN_EMPTY_KEYS and not v)
    }
    out["n_flagged"] = len(verdict.get("flagged") or ())
    out["flagged_observables"] = sorted(str(obs) for obs in observables)
    out["covered_observables"] = sorted(str(obs) for obs in covered)
    if advice:
        out["advice"] = advice
    return out


def _lof_reported(lof: dict[str, Any]) -> dict[str, Any]:
    """The part of the arm that RIDES ON A FIT: the numbers plus at most
    ``REPLICATE_LOF_MAX_REPORTED_POINTS`` named points, never a full table (see
    ``constants.REPLICATE_LOF_MAX_REPORTED_POINTS`` and ``_LOF_ROLLUP_ONLY_KEYS``)."""
    return {k: v for k, v in lof.items() if k not in _LOF_ROLLUP_ONLY_KEYS}


def _replicate_lof(session, sims=None) -> dict[str, Any]:
    """The replicate lack-of-fit verdict for the current fit, across every noise parameter.

    Per noise parameter (never across — each has its own coordinate and its own replicate scatter,
    and pooling two families would also inflate the multiple-testing correction), the replicated
    design points come from :func:`reconcile.replicate_pure_error_points` with the simulation
    records supplying the prediction, and :func:`residual_tests.replicate_lack_of_fit` supplies the
    verdict. The arms are then merged into one.

    Best-effort throughout, in this module's style: anything unresolvable fails OPEN to a
    NO-EVIDENCE verdict (never a pass, and never an exception): a diagnostic must not be able to
    block a fit.

    ``sims`` is the ``simulations`` frame when the caller already built one (both callers have), so
    this does not rebuild every PEtab frame a second time on a path that runs on every commit.
    """
    from . import reconcile as R
    from . import residual_tests as rt

    try:
        problem = session.problem
        if sims is None:
            sims = _frames(session)["simulations"]
        if sims is None or sims.empty:
            return _lof_payload(rt.replicate_lof_unavailable())
        # The records use this module's snake_case column names; reconcile speaks PEtab's.
        frame = sims.rename(
            columns={
                "observable_id": OBSERVABLE_ID,
                "experiment_id": EXPERIMENT_ID,
                "time": TIME,
            }
        )
        fitted = (session.simulation or {}).get("fitted_parameters")
        n_structural = _n_estimated_structural(problem)
        verdicts: list[dict[str, Any]] = []
        label_observable: dict[str, Any] = {}
        # Each family's points are resolved first so the LABELS can be decided across all of them.
        # Deciding them within a family made two DISTINCT design points of one observable, owned by
        # different noise parameters (a per-row noiseParameters split), each the only point of its own
        # family — so both printed as the bare observable id and the agent was shown one name twice
        # with nothing to tell the points apart.
        #
        # This is also where the per-design-point pure error is computed a second time per fit: the
        # pooled noise floor (fitting.noise_sigma_report) calls the same function on the MEASUREMENT
        # table while this call groups the simulation subset, so there is no one result to share. The
        # duplicated prelude was dominated by resolving the noise-formula gains, which reconcile now
        # memoises on the observable table's TEXT (measured per sigma, gain resolution: 35.8 ->
        # 5.8 ms on a 16-observable problem, 54.3 -> 10.9 ms on another; the whole call on the first
        # 40.8 -> 11.2 ms). What is left to duplicate is single-digit ms per sigma against a
        # multistart fit's seconds, so the remaining redundancy is not worth a (problem, pid,
        # fitted) cache: the measurement table is rewritten between the two calls whenever the agent
        # re-authors, and a stale row set in a fit's verdict is a worse failure than the cost.
        resolved = [
            (pid, R.replicate_pure_error_points(problem, pid, fitted, simulations=frame))
            for pid in _noise_parameter_ids(problem)
        ]
        per_observable = Counter(p.get("observable_id") for _, pts in resolved if pts for p in pts)
        for pid, points in resolved:
            if points is None:
                # Unresolvable rows/coordinate/gain for this sigma. Recorded as an explicit
                # NO-EVIDENCE arm rather than skipped, so that if a SIBLING sigma passes the merge
                # still reports that this family was never tested (see _merge_lof_verdicts).
                verdicts.append(rt.replicate_lof_unavailable())
                continue
            labelled = _labelled_points(points, per_observable)
            label_observable.update({p["label"]: p["observable_id"] for p in labelled})
            verdicts.append(rt.replicate_lack_of_fit(labelled))
        # n_structural goes to the MERGE, not to each arm: it is one whole-model count against the
        # POOLED points of every family (rt.lof_power).
        return _merge_lof_verdicts(verdicts, label_observable, n_structural=n_structural)
    except Exception:  # noqa: BLE001 - advisory; a diagnostic failure must not affect the fit
        return _lof_payload(rt.replicate_lof_unavailable())


def _noise_parameter_ids(problem) -> list[str]:
    """The REAL noise parameters to scope the arm by, sorted.

    ``plausibility_sigma_ids`` also yields the ``noiseParameter1_<obsId>`` PLACEHOLDER symbols of
    every noiseFormula (16 of them beside the one real ``sigma`` on a 16-observable problem), and
    each placeholder resolves to exactly one observable — so scoping by them would split its
    single 15-point family into 16 one-point families, none of which can be tested. Intersecting
    with the parameter table is what ``fitting.noise_sigma_report`` does (against the fitted
    values) for the same reason."""
    from . import reconcile as R

    ids = R.plausibility_sigma_ids(problem)
    df = getattr(problem, "parameter_df", None)
    if df is not None:
        ids = ids & {str(pid) for pid in df.index}
    return sorted(ids)


def _labelled_points(
    points: list[dict[str, Any]], per_observable: "Counter[Any] | None" = None
) -> list[dict[str, Any]]:
    """Give each replicated design point a short unique label for the agent-facing payload: the bare
    observable id where that observable has only one replicated point (e.g. steady-state
    observables measured once each), qualified by experiment (where it has one) and time otherwise.

    ``per_observable`` is the point count per observable across EVERY noise family, so a label is
    unique in the merged verdict and not merely within its own family (:func:`_replicate_lof` builds
    it); it falls back to this family's own counts."""
    per_observable = (
        Counter(p.get("observable_id") for p in points)
        if per_observable is None
        else per_observable
    )
    out = []
    for p in points:
        obs = p.get("observable_id")
        exp = _clean(p.get("experiment_id"))
        where = f"{exp}, t={_clean(p.get('time'))}" if exp else f"t={_clean(p.get('time'))}"
        out.append({**p, "label": str(obs) if per_observable[obs] == 1 else f"{obs}[{where}]"})
    return out


#: The two NO-EVIDENCE statuses whose absence is unremarkable and PERMANENT — a property of the
#: problem's tables, not of the fit, so prose about them would be a per-fit tax forever (13 of 16
#: problems measured are one of these: 9 with no repeated design point at all, 4 whose
#: noiseFormula is a placeholder over NUMERIC noiseParameters cells and which also have 0 replicated
#: design points). ONE source of truth for both places the policy applies: the summary line stays
#: silent for them, and the merge never hedges a sibling's verdict against them.
_LOF_SILENT_STATUSES = ("no_replicated_points", "no_noise_parameter")

#: Unavailable statuses, most informative first — a merge keeps the one that says the most about WHY
#: the arm could not run (a vacuous saturation over "no replicates at all").
#: The catch-all ``unavailable`` outranks BOTH silent statuses even though it names the least: it is
#: the only status that reports a real, fixable failure on a problem that DOES have replicates, and
#: ranking it last let a benign sibling silence it — one sigma resolving to zero replicated points
#: hid another whose observables mix lin and log, and the merged verdict said nothing at all.
_LOF_STATUS_PRIORITY = (
    "vacuous",
    "too_few_usable_points",
    "replicates_too_thin",
    "unavailable",
) + _LOF_SILENT_STATUSES


def _merge_lof_verdicts(
    verdicts: list[dict[str, Any]],
    label_observable: dict[str, Any],
    n_structural: int | None = None,
    min_lof_df: int | None = None,
) -> dict[str, Any]:
    """Merge the per-noise-parameter arms into one verdict. Flagged points union (each mapped back
    to its observable, so they join ``flagged_observables``), and the reported worst points are the
    worst ACROSS arms — still capped.

    THIS is where the lack-of-fit degrees of freedom are decided, because this is the only place that
    knows the POOLED point count: ``n_structural`` (the whole-model estimated-structural count from
    :func:`_n_estimated_structural`) is subtracted ONCE from the total the arms tested, and
    ``rt.lof_power`` turns that into the band. Per family it was neither — total points minus F times
    the shared parameters — which understated the df of a multi-sigma problem and refused three
    4-point families one at a time as vacuous, on a pooled count that leaves 2 directions.
    ``min_lof_df`` overrides the threshold for a caller that wants a different bar; the default
    lives in ``rt.lof_power`` alone.

    An arm that could NOT run is not dropped: its status survives as ``untested_status``, because a
    merge that kept only the available arms reported a bare "pass" for a problem where one family was
    tested and another was never looked at (seen on every fit of one problem: three families of 4
    points pass while a fourth family's 2 points are too few to test). Only for a status the summary
    is willing to SPEAK about, though (see ``_LOF_SILENT_STATUSES``): hedging a real pass against a
    family that has no replicated design points at all says nothing — there is nothing there that
    could not be tested."""
    from . import residual_tests as rt
    from .constants import REPLICATE_LOF_MAX_REPORTED_POINTS

    def _worst_status(vs):
        return min(
            vs,
            key=lambda v: (
                _LOF_STATUS_PRIORITY.index(v["status"])
                if v["status"] in _LOF_STATUS_PRIORITY
                else len(_LOF_STATUS_PRIORITY)
            ),
        )

    available = [v for v in verdicts if v.get("available")]
    untested = [v for v in verdicts if not v.get("available")]
    if not available:
        if not verdicts:
            # No noise family to scope the arm by at all — the whole loop never ran. Its own status,
            # not the catch-all: a numeric noiseFormula is fixed DATA, so there is nothing for a
            # re-author to resolve, and with a FIXED sigma the chi2 GoF this arm exists to backstop
            # is not blind in the first place (the four such problems measured also have no
            # replicated design point between them).
            return _lof_payload(rt.replicate_lof_unavailable("no_noise_parameter"))
        worst = _worst_status(untested)
        return _lof_payload(
            rt.replicate_lof_unavailable(worst["status"], n_points=worst["n_points"])
        )
    n_points = sum(v["n_points"] for v in available)
    lof_df, power = rt.lof_power(n_points, n_structural, min_lof_df)
    if power == "vacuous":
        # Saturation, on the POOLED count: no direction left to disagree in, so nothing was tested —
        # including the flags, which a fit that flexible could have placed anywhere. NOT TESTED is
        # the fail-open reading, and the one the arm gave before the df moved here.
        return _lof_payload(
            rt.replicate_lof_unavailable("vacuous", available[0]["alpha"], n_points=n_points)
        )
    flagged = [label for v in available for label in v["flagged"]]
    points = sorted(
        (p for v in available for p in v["points"]), key=lambda p: abs(p["z"]), reverse=True
    )[:REPLICATE_LOF_MAX_REPORTED_POINTS]
    merged = {
        "available": True,
        "status": "flagged" if flagged else "pass",
        "n_points": n_points,
        "alpha": available[0]["alpha"],
        "flagged": flagged,
        "tested": [label for v in available for label in v["tested"]],
        "max_abs_z": max(v["max_abs_z"] for v in available),
        "min_p_holm": min(v["min_p_holm"] for v in available),
        "points": points,
        # Carrying these two through the merge is what makes the weak wording reach the agent: an
        # assembler that rebuilds the verdict from a fixed key list drops them silently, and the
        # "this arm IS powered" line then fires on a df = 1 pass.
        #
        # WHERE THE df BAND GOES, so nobody has to guess: the summary prose (_replicate_lof_line),
        # the coverage credit just below (which keys on the BAND, not on `weak` — see there), and
        # whatever a caller records off the fit payload. NO stop condition reads either, nor the
        # flag itself: the arm is advisory, and the agent is an LLM that reads the prose, so the
        # wording IS the mechanism.
        "lof_df": lof_df,
        # A pass carried by too few lack-of-fit directions to mean much; never set on a FLAG, which is
        # conclusive however few directions remain. Absent (not False) when the df is UNKNOWN — see
        # _replicate_lof_line, which claims no power there either.
        "weak": bool(not flagged and power == "weak"),
        # The most informative reason among the arms that were NOT TESTED, so a partial pass
        # cannot read as a whole one. None when every arm ran — and also when the only arms that did
        # not run are the silent ones, which have no untested points to hedge about.
        "untested_status": _untested_status(untested, _worst_status),
    }
    # `or label` (not a two-argument get): a point whose rows carry no observableId maps to None, and
    # falling back to its LABEL gives the agent a real string to act on instead of the literal
    # "None" — and keeps the flag, which sorting a {None, str} set used to destroy.
    observables = {label_observable.get(label) or label for label in flagged}
    # COVERAGE is credited only by a POWERED arm, whether or not it flagged — the band, not `weak`.
    # Keying it on `weak` (which is forced False as soon as anything flagged) let a df = 1 arm that
    # flagged obs_A certify obs_B as sign-random on the strength of the same arm the summary calls too
    # flexible to mean anything. An UNKNOWN df certifies nothing either: this credit CLEARS
    # `low_power` for the observables it names, which is a positive claim, and "we could not count the
    # directions" is not evidence for it.
    covered = (
        {label_observable.get(label) or label for label in merged["tested"]}
        if power == "powered"
        else set()
    )
    return _lof_payload(
        merged,
        observables,
        advice=_replicate_lof_advice(merged) if flagged else None,
        covered=covered,
    )


def _untested_status(untested: list[dict[str, Any]], worst_status) -> str | None:
    """The reason a SIBLING family went untested, or ``None`` when there is nothing to say.

    Only statuses the summary line speaks about qualify: a family whose status is one of
    ``_LOF_SILENT_STATUSES`` has no replicated design points (or no sigma) in it, so "a further
    family could NOT be tested" would hedge a real pass against an empty set."""
    speaking = [v for v in untested if v["status"] not in _LOF_SILENT_STATUSES]
    return worst_status(speaking)["status"] if speaking else None


def _replicate_lof_advice(verdict: dict[str, Any]) -> str:
    """The agent-facing line. Names the worst points and their deviation in standard errors of the
    replicate scatter pooled over their noise sigma, and what that does and does not show.
    Deliberately says nothing about the reference model — no delta_bic, no parameter-count target —
    since the agent must not be able to read the answer off a diagnostic."""
    named = "; ".join(f"{p['label']} {p['z']:+.1f}" for p in verdict["points"][:3])
    return (
        f"Replicate lack-of-fit: {len(verdict['flagged'])} of {verdict['n_points']} replicated "
        f"design points sit further from the model than replicate scatter allows, in standard "
        f"errors of the scatter pooled over each point's noise sigma ({named}; Holm p "
        f"{verdict['min_p_holm']:.3g} at alpha {verdict['alpha']}). "
        "A larger sigma cannot close that gap, since the yardstick is the replicates themselves. "
        "A mechanism those points need might, but replicates tighter than the spread between "
        "experiments flag the same way, so it is not proof that one exists."
    )


def _n_estimated_structural(problem) -> int | None:
    """Estimated parameters MINUS the noise sigmas: the structural degrees of freedom the replicate
    arm's df guard subtracts ONCE from the POOLED replicated-point count of every noise family
    (``rt.lof_power``; e.g. 10 structural against 3 families of 4 points, so 2 lack-of-fit
    directions survive the fit). ``None`` when the parameter table cannot answer — which does not
    turn the guard off but makes the arm's power UNKNOWN: no coverage credit, and the summary says
    the pass could not be weighed rather than calling it powered.

    Subtracts the SAME set :func:`_noise_parameter_ids` scopes the arm by — ``plausibility_sigma_ids``,
    i.e. the noise symbols that are not ALSO observable-scaling factors. Subtracting the wider
    ``noise_sigma_ids`` here left the dual-role symbols (e.g. eight per-gel scales ``s_gel*``, in
    both noiseParameters and observableParameters) in NEITHER set: excluded from the noise families
    the arm iterates AND from the structural count, though they move the PREDICTION, which is
    exactly the flexibility this guard weighs."""
    from . import reconcile as R

    df = getattr(problem, "parameter_df", None)
    if df is None or ESTIMATE not in getattr(df, "columns", []):
        return None
    try:
        sigmas = R.plausibility_sigma_ids(problem)
        return sum(
            1
            for pid in df.index
            if str(pid) not in sigmas and R.is_parameter_estimated(problem, str(pid))
        )
    except Exception:  # noqa: BLE001 - advisory; leave the guard off rather than fail the diagnosis
        return None


def residual_flag(structure: dict[str, Any]) -> bool | None:
    """The bare structure flag of a rollup, or ``None`` when NEITHER arm could run — not merely the
    sign/runs tests, which a collapsed sigma silences on its own while the replicate arm still runs.

    The distinction matters where the flag is RECORDED (the version meta, so ``list_versions``):
    an unavailable verdict flags nothing, but recording that as ``False`` would turn "untested"
    into a clean bill of health."""
    if structure.get("summary") == _STRUCTURE_UNAVAILABLE:
        return None
    return bool(structure.get("flagged_observables"))


# p-value at/below which a residual-structure test is called significant in the rolled-up
# verdict (the per-test p-values are always reported raw, so a caller can judge for itself).
_STRUCTURE_ALPHA = 0.05

# At most this many flagged or underpowered observables are NAMED in the summary, and flagged
# observables or sigmas in a fit result's ``flagged_observables`` / ``sigma_heterogeneity`` (the
# rest as a count): they ride on every fit, and get_fit_summary lists every one.
_MAX_NAMED = 3


def _named(ids: list[str]) -> str:
    """The first ``_MAX_NAMED`` of ``ids``, comma-joined, with a count of the rest."""
    rest = len(ids) - _MAX_NAMED
    return ", ".join(ids[:_MAX_NAMED]) + (f" (+{rest} more)" if rest > 0 else "")


def residual_structure_on_fit(rollup: dict[str, Any] | None) -> dict[str, Any] | None:
    """The rollup as a fit result carries it: ``flagged_observables`` and ``sigma_heterogeneity``
    cut to the first ``_MAX_NAMED``, with ``n_flagged_observables`` / ``n_sigma_heterogeneity``
    counting all of them when any is cut, and no ``replicate_lack_of_fit`` where the design leaves
    that arm nothing to test (``_LOF_SILENT_STATUSES``, 13 of 16 problems measured)."""
    if not isinstance(rollup, dict):
        return rollup
    out = dict(rollup)
    for key in ("flagged_observables", "sigma_heterogeneity"):
        named = out.get(key) or []
        if len(named) > _MAX_NAMED:
            out[key] = named[:_MAX_NAMED]
            out[f"n_{key}"] = len(named)
    if (out.get("replicate_lack_of_fit") or {}).get("status") in _LOF_SILENT_STATUSES:
        del out["replicate_lack_of_fit"]
    return out


def _safe_residual_structure(group, centred: bool = False) -> dict[str, Any]:
    """:func:`_residual_structure`, but any failure in the added stats degrades to a
    tests-unavailable stub instead of taking down the whole diagnosis (which the magnitude
    aggregations above compute robustly). fit_diagnosis is called by the agent's
    get_fit_diagnosis AND by offline scorers, so a raise here must not break either."""
    try:
        return _residual_structure(group, centred=centred)
    except Exception:  # noqa: BLE001 - advisory stats; never fail the diagnosis on them
        return {
            "sign_test": None,
            "runs_test": None,
            "structure_flag": False,
            "structure_low_power": True,
        }


def _residual_structure(group, centred: bool = False) -> dict[str, Any]:
    """Sign-test + runs-test summary for one observable's residuals (see
    :mod:`petab_mcp.residual_tests`).

    Each test is per (experiment) SERIES, Fisher-combined across the series that can reach
    ``_STRUCTURE_ALPHA`` on their own (``min_p``): a series that cannot always gives p near 1 and
    only dilutes the combination (a one-point series gives exactly 1). The runs test must be
    per-series, since runs across experiments would invent spurious runs at the boundaries. The
    SIGN test is stratified: a series long enough is its own stratum, so that opposite biases in
    different experiments do not cancel, and the shorter series are pooled into one stratum,
    where a common bias shows (one-point series on steady-state data, PEtab-MCP#329). Replicate
    points sharing an (experiment, time) are AVERAGED to one residual first: replicates are not an
    independent draw of the residual's sign.

    ``group`` is the observable's rows from :func:`fit_diagnosis`, already restricted to points
    with a weighted residual; the raw ``residual`` (same sign, sigma being positive) is the sign
    source. ``n_series`` counts the series a test's ``p`` rests on, and ``low_power`` is set below
    ``residual_tests.MIN_POWER_N`` usable points among them. ``centred`` (an observable of
    :func:`_offset_centred_observables`) marks the sign test low-power whatever its count.
    """
    from . import residual_tests as rt

    def reaches(res: dict[str, Any]) -> bool:
        return res["p"] is not None and res["min_p"] < _STRUCTURE_ALPHA

    g = group.copy()
    g["_r"] = g["residual"].astype(float)
    # Average replicates to one residual per (experiment, time), then order by time per series.
    collapsed = g.groupby(["experiment_id", "time"], dropna=False)["_r"].mean().reset_index()

    sign: dict[str, Any] = {"ps": [], "n": 0, "series": 0}
    runs: dict[str, Any] = {"ps": [], "n": 0, "series": 0}
    short: list[float] = []
    short_series = 0
    n_pos = n_neg = 0
    for _exp, s in collapsed.groupby("experiment_id", dropna=False):
        vals = s.sort_values("time")["_r"].tolist()
        st = rt.sign_test(vals)
        n_pos += st["n_pos"]
        n_neg += st["n_neg"]
        for res, acc in ((st, sign), (rt.runs_test(vals), runs)):
            if reaches(res):
                acc["ps"].append(res["p"])
                acc["n"] += res["n"]
                acc["series"] += 1
        if st["n"] and not reaches(st):
            short.extend(vals)
            short_series += 1
    pooled = rt.sign_test(short)
    if reaches(pooled):
        sign["ps"].append(pooled["p"])
        sign["n"] += pooled["n"]
        sign["series"] += short_series

    sign_p = rt.fisher_combine(sign["ps"])
    runs_p = rt.fisher_combine(runs["ps"])
    sign_low = sign["n"] < rt.MIN_POWER_N or centred
    runs_low = runs["n"] < rt.MIN_POWER_N
    sig_bias = sign_p is not None and sign_p < _STRUCTURE_ALPHA
    sig_trend = runs_p is not None and runs_p < _STRUCTURE_ALPHA
    return {
        "sign_test": {
            "n_pos": n_pos,
            "n_neg": n_neg,
            "p": _round_p(sign_p),
            "n_series": sign["series"],
            "low_power": sign_low,
        },
        "runs_test": {
            "p": _round_p(runs_p),
            "n_series": runs["series"],
            "low_power": runs_low,
        },
        "structure_flag": bool(sig_bias or sig_trend),
        "structure_low_power": bool(sign_low or runs_low),
    }


def _round_p(p: float | None) -> float | None:
    return None if p is None else round(float(p), 4)


def _structure_summary(
    flagged: list[str],
    low_power: bool,
    lof: dict[str, Any] | None = None,
    sign_runs_tested: bool = True,
    low_power_observables: list[str] | None = None,
) -> str:
    if flagged:
        # The parenthetical names only tests that actually RAN: quoting a p < 0.05 threshold for the
        # sign/runs tests while the same summary says they could not run (a collapsed sigma leaves no
        # weighted residual, and the replicate arm flags anyway) attributes the finding to a test
        # that produced no p at all.
        how = (
            "replicate lack-of-fit"
            if not sign_runs_tested
            else f"sign-test bias or runs-test time trend, p < {_STRUCTURE_ALPHA}, or replicate "
            "lack-of-fit"
        )
        head = (
            f"residual structure in {_named(flagged)} ({how}): where the fit is off, not proof "
            "that a better form exists. A flag may be chance or a misfit no model clears, and "
            "the score already charges it."
        )
    elif not sign_runs_tested:
        # Reached when the residuals carry no weighted value at all (an unevaluable or collapsed
        # sigma) while the replicate arm still ran. Saying "residuals look sign-random" there would
        # assert a test that never happened.
        head = (
            "the sign and runs tests could not run (no weighted residuals), so nothing is known "
            "about bias or time trend."
        )
    elif low_power:
        # NAMED, and attributed to the sign/runs arm alone, because the replicate line that follows
        # may report a genuine pass for the observables it covered: a blanket "the tests are
        # UNDERPOWERED at this sample size" beside "this arm IS powered (N lack-of-fit degrees of
        # freedom remain), so that is a pass" is a flat contradiction the agent cannot resolve
        # (seen on every fit of one problem: three observables covered, a fourth's family too small
        # to test).
        where = f" for {_named(low_power_observables)}" if low_power_observables else ""
        head = (
            f"no significant residual structure, but the sign/runs tests are UNDERPOWERED{where} "
            f"at this sample size — that is weak evidence of adequacy{' there' if where else ''}"
            ", not a pass."
        )
    else:
        head = "no significant residual structure detected (residuals look sign-random)."
    tail = _replicate_lof_line(lof or {})
    return f"{head} {tail}" if tail else head


def _replicate_lof_line(lof: dict[str, Any]) -> str:
    """The replicate arm's one line for the summary, or ``""``.

    Silent for the two statuses whose absence is unremarkable and PERMANENT — no repeated design
    point at all, and no noise parameter to scope the arm by (``_LOF_SILENT_STATUSES``; 13 of 16
    problems measured, where the structured keys already carry ``available: false`` plus the reason
    and prose would be a per-fit tax on every one of them, forever). Every other status speaks: a
    real pass, a flag, or the specific reason the arm could not be run — including the catch-all
    ``unavailable``, which used to be silent too and so hid real failures on problems that DO have
    replicates.
    """
    status = lof.get("status")
    if status in _LOF_SILENT_STATUSES:
        return ""
    # Whatever the arm DID test, it may not be read as a whole verdict while another family of the
    # same problem went untested. Seen on every fit of one problem: three noise families of 4
    # replicated points are tested, a fourth family's 2 points are `too_few_usable_points`, and
    # without this the 12-point pass read as a whole-problem pass.
    partial = ""
    if lof.get("untested_status"):
        partial = (
            f" A further family of replicated design points could NOT be tested "
            f"({lof['untested_status']}), so this says nothing about those points."
        )
    # STATUS first, and only then `available`: `advice` is one of the rollup-only keys stripped from
    # what rides on a fit (_LOF_ROLLUP_ONLY_KEYS), so dispatching on its presence turned a stripped
    # FLAGGED verdict into "that is a pass" — inverting the finding for any caller that summarises
    # the projection rather than the merge.
    if status == "flagged":
        return (
            lof.get("advice")
            or (
                f"replicate lack-of-fit: {lof.get('n_flagged', 0)} of {lof['n_points']} replicated "
                f"design points sit further from the model than replicate scatter allows "
                f"(worst {lof['max_abs_z']} SE, Holm p {lof['min_p_holm']}), which no sigma can "
                "absorb."
            )
        ) + partial
    if lof.get("available"):
        head = (
            f"replicate lack-of-fit: none over {lof['n_points']} replicated design points "
            f"(worst {lof['max_abs_z']} SE of replicate scatter)"
        )
        # How much that pass is worth depends on how many directions were left to fail in. A fit
        # with almost as many structural parameters as replicated points can place its predictions
        # on the replicate means in all but a few directions whatever its form, so small z there is
        # mostly FLEXIBILITY, and calling it powered endorses a model on the strength of its own
        # flexibility (see constants.REPLICATE_LOF_MIN_LOF_DF).
        if lof.get("weak"):
            return (
                f"{head}, but only {lof.get('lof_df')} lack-of-fit degree(s) of freedom remain "
                "— nearly as many estimated structural parameters as replicated points, so this "
                "is WEAK evidence of form rather than a pass: a fit that flexible can reach "
                "those points whatever its mechanism." + partial
            )
        # ...and where the df could not be counted at all, the pass CANNOT be called powered either:
        # `lof_df` is absent on an available arm only when the parameter table would not yield a
        # structural count (_n_estimated_structural), and an uncounted direction is not a spare one.
        # Claiming power here is the same over-claim as claiming it at df = 1, one step further from
        # the evidence.
        if lof.get("lof_df") is None:
            return (
                f"{head}, but the estimated structural parameters could not be counted, so how "
                "many lack-of-fit directions this pass rests on is UNKNOWN — read it as weak "
                "evidence for the points it tested." + partial
            )
        return (
            f"{head} — this arm IS powered ({lof['lof_df']} lack-of-fit degrees of freedom "
            "remain), so that is a pass rather than weak evidence for the points it tested."
            + partial
        )
    if status == "replicates_too_thin":
        why = (
            "the repeated design points have only one degree of freedom each, too thin to be a "
            "yardstick."
        )
    elif status == "too_few_usable_points":
        why = "too few usable replicated design points to test."
    elif status == "vacuous":
        # "at least as many", and counted over ALL the replicated design points this problem has:
        # said of one noise family at a time this sentence was simply false for a problem whose
        # families add up (e.g. 12 points against 10 parameters).
        why = (
            "this model has at least as many estimated structural parameters as it has replicated "
            f"design points ({lof.get('n_points')}), so the test would measure its flexibility, "
            "not its form."
        )
    else:
        # The catch-all, and it no longer covers "there is no noise parameter at all" — that has
        # its own SILENT status. What is left is a sigma that exists and still would not resolve:
        # its observables mix lin and log so there is no single coordinate, a gain will not reduce
        # to a number, no prediction was constant across a point's replicates, or the diagnostic
        # itself raised. Every one of those is a real, fixable failure on a problem that may well
        # HAVE replicates, so it speaks: an untested arm read as a pass is the more expensive
        # mistake.
        why = (
            "the arm could not be run for this problem's noise parameter (no single noise "
            "coordinate for its observables, an unresolvable amplitude, or no constant prediction "
            "across a design point's replicates)."
        )
    return f"replicate lack-of-fit: NOT TESTED ({status or 'unavailable'}) — {why}"
