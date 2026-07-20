"""The MCP server: FastMCP tools and resources over a :class:`PetabSession`.

Tool groups:

* read-only (fixed design): ``describe_problem``, ``list_observables``,
  ``list_conditions``, ``get_measurements``
* model (constrained operations): ``add_reaction``, ``change_rate_law``,
  ``remove_reaction``, ``change_parameter_scale``, ``change_parameter_bounds`` (each
  requires a justification; references optional), plus read-only ``get_model``,
  ``get_model_structure``, ``validate_problem``
* parameters (read-only + reconcile): ``get_parameters``, ``sync_parameters``
* batching: ``begin_batch``, ``commit_batch``, ``abort_batch``
* fit + simulation: **automatic on every commit** — each commit fits the free
  parameters and simulates, storing a simulation dataframe on the new version; read the
  score with ``get_fit_summary`` (metrics only), query the data (PEtab tables + simulation
  dataframe + residuals + pyPESTO traces) with SQL via ``query_data`` / ``list_data_tables``,
  and inspect the optimizer numerics with ``get_fitting_log`` (the pyPESTO/Fides log)
* provenance/versions/errors: ``get_provenance``, ``get_errors``, ``list_versions``,
  ``get_version``, ``diff_versions``, ``rollback_to``, ``current_version``
* session: ``reset``
* skills (tool-usage documentation, bundled under ``_skills/``): ``list_skills``,
  ``get_skill``

No tool mutates the measurement/condition/observable tables — that invariant is
enforced structurally (there is simply no such tool) and re-checked by the operation
pipeline's lint (see ``operations``/``validation``). Parameter *values* are set only by
the automatic fit; bounds and scale are changed through the versioned operations.

``create_server`` wires the session/engine and delegates the (large) tool + resource
surface to the ``_register_*`` groups below, each closing over a small :class:`_Ctx`
that carries the session, engine, and the error-wrapping helpers. The tools stay thin:
the real work lives in the ``session`` / ``operations`` / ``fitting`` / ``data_access`` /
``reconcile`` modules.
"""

from __future__ import annotations

import json
from typing import Any

from mcp.server.fastmcp import FastMCP
from petab.v1.C import OBSERVABLE_ID, SIMULATION_CONDITION_ID

from . import data_access as data_mod
from . import fitting as fitting_mod
from . import identifiability as identifiability_mod
from . import logging_setup
from . import model as model_mod
from . import operations as ops_mod
from . import parameters as param_mod
from . import reconcile as reconcile_mod
from . import skills as skills_mod
from .constants import DEFAULT_AUTO_FIT_STARTS
from .engine import AmiciEngine
from .errors import PetabMcpError, ValidationError
from .formatting import df_to_records, summarize_measurements
from .session import PetabSession
from .validation import validate_problem as _validate_problem

SERVER_INSTRUCTIONS = (
    "Interact with a fixed PEtab v1 estimation problem. The experimental design "
    "(measurements, conditions, observables) is read-only. If the problem ships WITHOUT a "
    "model (`describe_problem` reports `model_initialized: false`), author the whole "
    "model from Antimony with `set_model` (justification + references). `set_model` also "
    "re-authors an existing model wholesale — the clean way to make a coupled/structural "
    "change in one step; for smaller, local changes the model is edited "
    "through the constrained, justified operations — `add_reaction`, "
    "`change_rate_law`, `remove_reaction` — each of which requires a non-empty "
    "justification (references are optional, PubMed-validated if given) and automatically "
    "keeps the SBML model and PEtab parameter table consistent (parametric initial "
    "conditions). Every species gets a parametric initial condition, FIXED at its declared t=0 "
    "by default; each commit returns an `initial_conditions` block reporting whether every "
    "species' t=0 is fixed (declared value used), estimated (the fit sets it), or set per "
    "condition. Antimony cannot declare estimate status, so re-naming an initial condition in the "
    "model never changes whether it is fitted — read the report instead of guessing. Parameter "
    "scales and "
    "bounds are changed through versioned, justified operations "
    "(`change_parameter_scale`, `change_parameter_bounds`); parameter *values* are set "
    "only by the automatic fit, not by hand. EVERY commit automatically fits the free "
    "parameters and simulates, then snapshots a single version carrying the fitted "
    "parameters, the fit metrics (neg_log_likelihood, chi2, improvement vs prior "
    "versions), and a simulation dataframe. Pass `n_starts` on a commit to fit harder "
    "(or 0 to simulate without fitting). Chain several edits into one version — fitted "
    "once, on commit — with `begin_batch`/`commit_batch`/`abort_batch`. Check the score "
    "cheaply with `get_fit_summary` (metrics only). Diagnose the fit with SQL: `query_data` "
    "runs DuckDB queries over a disposable copy of the PEtab tables (`measurements`, `conditions`, "
    "`observables`, `parameters`), the `simulations` dataframe (measurement vs simulation "
    "with per-point `residual`, `residual_weighted`, and `noise`), and the last fit's pyPESTO "
    "multistart (`optimization_starts`, `optimization_trace`) — compute any residual "
    "breakdown, worst-fit points, bias/time-trend check, or convergence diagnostic you need "
    "(call `list_data_tables` for the schema). `get_fitting_log` exposes the pyPESTO/Fides "
    "optimizer log (multistart convergence, parameters pinned at bounds, ill-conditioning) for "
    "numerical diagnosis. Every change is versioned and logged: rank versions by fit with "
    "`list_versions` / `diff_versions` (which reports the neg-log-likelihood delta); inspect "
    "with `get_provenance`; errors via `get_errors`; revert with `rollback_to` or `reset`. To go "
    "BACK to an earlier model, use `restore_version(n)` — it restores that version's stored fit "
    "without recompiling/refitting; do NOT re-author an earlier model with `set_model` (that "
    "just refits to the same result and appends a duplicate — `set_model` warns on, then refuses, "
    "a re-authored structure it has already seen)."
)

# Appended to the instructions only when skills are actually registered, so clients
# are never told to call `list_skills`/`get_skill` when those tools are absent.
_SKILLS_INSTRUCTION = (
    " New to this server? Call `list_skills` and read "
    "`get_skill('petab-model-building')` for the full tool-usage workflow."
)


class _Ctx:
    """Shared state + error-wrapping helpers the tool groups close over.

    Bundles the :class:`PetabSession` and :class:`AmiciEngine` with the two dispatch
    wrappers that turn a raised :class:`PetabMcpError` into a structured tool result (and
    log it), so every ``_register_*`` group builds its tools against the same helpers.
    """

    def __init__(self, session: PetabSession, engine: AmiciEngine) -> None:
        self.session = session
        self.engine = engine

    def log_error(self, tool: str, args: dict[str, Any], exc: Exception) -> None:
        self.session.provenance.log_error(
            {
                "tool": tool,
                "args": args,
                "error_type": type(exc).__name__,
                "message": str(exc),
                "issues": getattr(exc, "issues", None),
            }
        )

    @staticmethod
    def short(value: Any, limit: int = 200) -> Any:
        """Truncate long strings before they go into the (persisted) error log, so a large
        argument — e.g. a whole Antimony model passed to `set_model` — cannot bloat the
        provenance error log or dump the full model contents into it."""
        if isinstance(value, str) and len(value) > limit:
            return value[:limit] + f"… (+{len(value) - limit} chars)"
        return value

    def run_operation(self, op_name: str, fn, *args) -> dict[str, Any]:
        """Run an operation; on a known error, log it and return committed=False."""
        try:
            return fn(*args)
        except PetabMcpError as exc:
            self.log_error(op_name, {"args": [self.short(a) for a in args if isinstance(a, (str, int, float, list))]}, exc)
            self.session.logger.warning("operation %s rejected: %s", op_name, exc)
            out = {"committed": False, "error": str(exc)}
            if getattr(exc, "issues", None):
                out["issues"] = exc.issues
            return out
        except Exception as exc:
            self.log_error(op_name, {}, exc)
            self.session.logger.exception("operation %s crashed", op_name)
            raise

    def run(self, tool: str, fn, args: dict[str, Any] | None = None):
        """Run a non-operation tool; log failures (return {ok:false} or re-raise)."""
        try:
            return fn()
        except PetabMcpError as exc:
            self.log_error(tool, args or {}, exc)
            self.session.logger.warning("%s failed: %s", tool, exc)
            out = {"ok": False, "error": str(exc)}
            if getattr(exc, "issues", None):
                out["issues"] = exc.issues
            return out
        except Exception as exc:
            self.log_error(tool, args or {}, exc)
            self.session.logger.exception("%s crashed", tool)
            raise

    def current_simulation(self) -> dict[str, Any]:
        """The stored simulation dataframe, generated on demand if not present yet."""
        return fitting_mod.ensure_simulation(self.session, self.engine)


# ------------------------------------------------------------------- read tools
def _register_read_tools(mcp: FastMCP, ctx: _Ctx) -> None:
    session = ctx.session

    @mcp.tool()
    def describe_problem() -> dict[str, Any]:
        """Summarize the problem: model size, tables, and free parameters."""
        return session.describe()

    @mcp.tool()
    def list_observables() -> list[dict[str, Any]]:
        """List the (fixed) observables and their formulas/noise models."""
        return df_to_records(session.problem.observable_df)

    @mcp.tool()
    def list_conditions() -> list[dict[str, Any]]:
        """List the (fixed) conditions and the model changes they apply."""
        return df_to_records(session.problem.condition_df)

    @mcp.tool()
    def get_measurements(
        condition_id: str | None = None, observable_id: str | None = None,
        summarize: bool = True,
    ) -> "list[dict[str, Any]] | dict[str, Any]":
        """Return the (fixed) measurement data, optionally filtered by condition/observable.

        By default (``summarize=True``) returns a COMPACT SUMMARY — per observable×condition:
        point count, time span, value min/max/mean, and a few preview points — so the data's
        shape is visible without every raw point (a full dump is ~6k tokens and re-enters the
        context each turn). Pass ``summarize=False`` for the full per-point records, or use
        ``query_data`` for targeted SQL over the ``measurements`` table.
        """
        df = session.problem.measurement_df
        if df is not None:
            if condition_id is not None:
                df = df[df[SIMULATION_CONDITION_ID] == condition_id]
            if observable_id is not None:
                df = df[df[OBSERVABLE_ID] == observable_id]
        return summarize_measurements(df) if summarize else df_to_records(df)


# ------------------------------------------------------------------ model tools
def _register_model_tools(mcp: FastMCP, ctx: _Ctx) -> None:
    session = ctx.session

    @mcp.tool()
    def get_model(format: str = "antimony") -> str:
        """Return the current model as text. `format` is 'antimony' (canonical) or 'sbml'."""
        return model_mod.get_model(session, format)

    @mcp.tool()
    def get_model_structure() -> dict[str, Any]:
        """Structured view of the model: species, parameters, reactions."""
        return model_mod.model_structure(session)

    @mcp.tool()
    def set_model(
        antimony: str, justification: str,
        references: list[str] | None = None, n_starts: int | None = None,
    ) -> dict[str, Any]:
        """Author the WHOLE model from an Antimony string — the single whole-model operation.

        Use it to BUILD the first model when the problem ships model-less (`describe_problem`
        reports `model_initialized: false`) AND to re-author/replace an existing model
        wholesale. Reach for it to make a STRUCTURAL change as one coherent step — a coupled or
        symmetric rate-law change, or adding/removing several reactions at once — instead of a
        sequence of incremental edits that leave intermediate models inconsistent (the
        `add_reaction` / `change_rate_law` / `remove_reaction` operations remain for local
        refinement). It reconciles the SBML model with the PEtab parameter table (parametric
        initial conditions, parameter-table sync), lints, and fits on commit, exactly like the
        other model ops; any failure reverts, so a bad re-author changes nothing. `justification`
        (non-empty) is required; `references` are optional; `n_starts` overrides the fit default
        (0 = simulate only). Because the whole model is replaced, the result `warnings` flag any
        previously-ESTIMATED parameter the re-author no longer declares (its fitted value is
        dropped); fixed noise/initial-condition parameters and anything the observable/condition
        tables reference survive. A re-author that drops (or reintroduces as estimated) a
        shipped-FIXED mechanistic constant — a conservation total, a known rate — is REJECTED,
        not warned: keep such constants declared and build the mechanism on them as shipped.

        A genuinely NEW structure always commits. Re-authoring a structure already seen this run
        is de-duplicated (it would refit to the same result): re-authoring the CURRENT model is a
        no-op (`committed: false`, `no_change: true` — no fit, no new version); re-authoring an
        EARLIER model commits but warns to use `restore_version(n)` instead; and an identical
        structure re-authored repeatedly is refused, to break an unproductive loop. To go back to
        a prior model, prefer `restore_version(n)`.
        """
        return ctx.run_operation("set_model", ops_mod.set_model, session, antimony, justification, references, n_starts)

    @mcp.tool()
    def add_reaction(
        reaction: str, justification: str, references: list[str] | None = None,
        n_starts: int | None = None,
    ) -> dict[str, Any]:
        """Add a reaction, given as an Antimony line, e.g. "J: A + B -> C; k_f*A*B".

        New species/parameters and their parametric initial conditions are added
        automatically to keep the SBML model and PEtab tables consistent. `justification`
        (non-empty) is required; `references` are optional (validated if given). On commit
        the free parameters are fitted and the model simulated; `n_starts` overrides the
        server default (0 = simulate without fitting). The result includes any parsimony
        `warnings` (e.g. an added unobserved species).
        """
        return ctx.run_operation("add_reaction", ops_mod.add_reaction, session, reaction, justification, references, n_starts)

    @mcp.tool()
    def change_rate_law(
        reaction_id: str, rate_law: str, justification: str,
        references: list[str] | None = None, n_starts: int | None = None,
    ) -> dict[str, Any]:
        """Replace a reaction's rate law (Antimony-like expression). Requires a
        justification (references optional); parameters are reconciled automatically. Fits +
        simulates on commit (`n_starts` overrides the default; 0 = simulate only)."""
        return ctx.run_operation(
            "change_rate_law", ops_mod.change_rate_law, session, reaction_id, rate_law, justification, references, n_starts
        )

    @mcp.tool()
    def remove_reaction(
        reaction_id: str, justification: str, references: list[str] | None = None,
        n_starts: int | None = None,
    ) -> dict[str, Any]:
        """Remove a reaction entirely and prune any orphaned species/parameters.
        Requires a justification (references optional). Fits + simulates on commit
        (`n_starts` overrides the default; 0 = simulate only)."""
        return ctx.run_operation("remove_reaction", ops_mod.remove_reaction, session, reaction_id, justification, references, n_starts)

    @mcp.tool()
    def change_parameter_scale(
        parameter_id: str, scale: str, justification: str,
        references: list[str] | None = None, n_starts: int | None = None,
    ) -> dict[str, Any]:
        """Change a parameter's scale ('lin'|'log'|'log10'). Versioned; requires a
        justification (references optional). Fits + simulates on commit (`n_starts`
        overrides the default; 0 = simulate only)."""
        return ctx.run_operation(
            "change_parameter_scale", ops_mod.change_parameter_scale,
            session, parameter_id, scale, justification, references, n_starts,
        )

    @mcp.tool()
    def change_parameter_bounds(
        parameter_id: str, lower_bound: float, upper_bound: float,
        justification: str, references: list[str] | None = None, n_starts: int | None = None,
    ) -> dict[str, Any]:
        """Change a parameter's lower/upper bounds. Versioned; requires a justification
        (references optional). Fits + simulates on commit (`n_starts` overrides the
        default; 0 = simulate only). REJECTS a widening past an ESTIMATED noise-sigma's
        data-driven cap, and REJECTS a THIRD widening of a still-pinned parameter whose last
        widening bought a negligible BIC gain — a parameter re-pinning at a bound with a flat/
        collinear FIM direction (see `get_identifiability`) is unidentifiable, not
        under-explored, so remove its reaction or use its limiting (irreversible/instantaneous)
        form instead of widening again; if neither is possible, leave it pinned (do not
        hard-code a value — there is no value/estimate tool, and a literal is a magic number)."""
        return ctx.run_operation(
            "change_parameter_bounds", ops_mod.change_parameter_bounds,
            session, parameter_id, lower_bound, upper_bound, justification, references, n_starts,
        )

    @mcp.tool()
    def validate_problem() -> dict[str, Any]:
        """Lint the current problem (model + tables) and report issues."""
        try:
            protected_fixed = session.protected_fixed_parameter_ids()
        except Exception:  # noqa: BLE001 - the role check is best-effort; lint still runs
            protected_fixed = None
        return _validate_problem(session.problem, protected_fixed_ids=protected_fixed)


# -------------------------------------------------------------- parameter tools
def _register_parameter_tools(mcp: FastMCP, ctx: _Ctx) -> None:
    session = ctx.session

    @mcp.tool()
    def get_parameters() -> list[dict[str, Any]]:
        """Return the parameter table (id, bounds, nominal_value, estimate, prior), each row
        annotated with its `role` and `status`.

        `role` names what the parameter IS — `kinetic` (a mechanistic rate constant used in a
        rate law), `noise_sigma` or `observable_scaling` (fixed observation-model factors, never
        to be wired into a reaction), `initial_condition`, `exponent` (a Hill coefficient), or
        `constant` (an otherwise-unreferenced constant). `status` is `"estimated"` or `"fixed"`.
        Use these to keep roles straight: don't repurpose a scaling/offset factor as a rate
        constant, and don't spend an estimated parameter on a value you already know — a parameter
        you introduce is added estimated and counts toward k (inflating BIC) however tight its
        bounds. A parameter's estimated/fixed status is not something you set here: a symbol you
        introduce is added estimated, and a shipped-fixed constant stays fixed only while your
        model keeps declaring it — so build on the constants the problem ships fixed rather than
        introducing your own.

        Values (`nominalValue`) are set by the automatic fit on commit; adjust bounds and
        scale through `change_parameter_bounds` / `change_parameter_scale`.
        """
        return param_mod.parameters_to_records(session.problem)

    @mcp.tool()
    def sync_parameters(policy: str = "report") -> dict[str, Any]:
        """Reconcile the parameter table with the model ('report' or 'add').

        `report` lists model params missing from the table and orphaned table rows without
        changing anything; `add` runs the same full reconcile the commit pipeline does —
        adding the missing params (with initial-condition/exponent/scale handling) and
        removing the orphans.
        """
        def run() -> dict[str, Any]:
            with session.lock:
                return reconcile_mod.sync_parameters(
                    session.problem, policy,
                    estimate_ics=bool(getattr(session, "estimate_initial_conditions", False)),
                )

        return ctx.run("sync_parameters", run, {"policy": policy})


# ------------------------------------------------------- fit + simulation tools
def _register_fit_tools(mcp: FastMCP, ctx: _Ctx) -> None:
    session = ctx.session
    engine = ctx.engine

    @mcp.tool()
    def get_fit_summary() -> dict[str, Any]:
        """Return just the current fit metrics — no per-point records (a cheap read).

        `{status, neg_log_likelihood, chi2, n_data_points, fitted_parameters,
        parameter_scales, version}` plus the complexity-penalised scores `{bic, aic, chi2_dof,
        reduced_chi2, chi2_gof_pvalue}` (lower bic/aic is better — the score to rank models by).
        `reduced_chi2`/`chi2_gof_pvalue` are only meaningful with a KNOWN noise level: when the
        observable noise sigmas are ESTIMATED, chi2 pins near `n_data_points` by construction, so
        reduced_chi2 ≈ 1 and its p-value are uninformative — do not use them as a fit-adequacy
        gate (judge adequacy from residual STRUCTURE instead, e.g. `get_fit_diagnosis`).
        `fitted_parameters` are LINEAR
        (unscaled) values; `parameter_scales` gives each one's estimation scale (lin/log10) —
        a fitted value is the linear number, never its log10 coordinate. `parameters_at_bound`
        lists any fitted parameter sitting within ~1% of its lower/upper bound (on its own
        estimation scale) — `[]` when none do. Widen at most once; the SAME parameter
        re-pinning after a widening means it is UNIDENTIFIABLE (a flat/collinear FIM direction —
        confirm with `get_identifiability`), not under-explored, so the optimum is unbounded
        (often an exponent trying to fake a mechanism the model doesn't have). Don't widen or
        flip its scale again — remove its reaction or use its limiting (irreversible/
        instantaneous) form; if neither is possible, leave it pinned (don't hard-code a value —
        values come only from the fit). `noise_sigma_report`
        lists every ESTIMATED noise-sigma parameter's fitted value against a data-driven
        reference (a fraction of its
        observable's own raw measurement std) regardless of whether it's at bound — a sigma
        approaching that reference is quietly doing the same thing as an at-bound parameter
        (masking a mechanism problem behind inflated noise); `change_parameter_bounds` rejects
        widening a sigma past it. Use this to check the score without pulling the dataframe;
        use `query_data` to query the measured-vs-simulated points and residuals with SQL.
        Refreshed on every commit; generated on demand if none exists.
        """
        return ctx.run("get_fit_summary", lambda: fitting_mod.fit_summary(session, engine))

    @mcp.tool()
    def get_run_timing() -> dict[str, Any]:
        """Cumulative server-side timing for this run, split by cost kind.

        `{compile_seconds, fit_seconds, n_compiles, n_fits}`. `compile_seconds` is the AMICI
        model-BUILD time (the one-time C++ compile per unique structure — the engine's build
        plus pyPESTO's own); `fit_seconds` is the pyPESTO multistart OPTIMIZATION time. Both
        accumulate only when work actually happens, so a compile/fit cache hit adds nothing.
        These are server-side wall times (a breakdown of the heavy compute), distinct from the
        client's `compute_seconds` tool-busy measure, which also covers reads/SQL.
        """
        def run() -> dict[str, Any]:
            # Read under the lock so the four accumulators are a consistent snapshot even if a
            # fit is mid-update on another thread (the panel dispatches tool calls concurrently).
            with session.lock:
                return {
                    "compile_seconds": round(float(session.compile_seconds), 3),
                    "fit_seconds": round(float(session.fit_seconds), 3),
                    "n_compiles": int(session.n_compiles),
                    "n_fits": int(session.n_fits),
                }

        return ctx.run("get_run_timing", run)

    @mcp.tool()
    def list_data_tables() -> dict[str, Any]:
        """List the SQL tables `query_data` exposes: name, description, columns, row count.

        The tables are the PEtab design tables (`measurements`, `conditions`,
        `observables`), the `parameters` table, the `simulations` dataframe (one row per
        measurement point, carrying `simulation` and the residuals), and the last fit's
        pyPESTO multistart (`optimization_starts`, `optimization_trace`). Use it to learn the
        exact column names before writing a `query_data` query.
        """
        return ctx.run("list_data_tables", lambda: data_mod.table_schema(session))

    @mcp.tool()
    def get_fitting_log(tail_lines: int = 200) -> dict[str, Any]:
        """Return the tail of the pyPESTO/Fides optimizer log for the fits run so far.

        Every commit fits the free parameters with a pyPESTO multistart (the Fides
        trust-region optimizer); its per-start objective values, the multistart summary,
        convergence messages, and any ill-conditioning / gradient warnings are written to
        `<store>/pypesto.log`. This tool surfaces that log so the *numerical* behaviour of
        the fit can be inspected directly — multistart spread (did starts agree on one
        optimum?), convergence, parameters pinned at their bounds, and non-identifiability
        signatures. Returns `{path, exists, n_lines_total, tail_lines, log}` (the last
        `tail_lines` lines of the file; pass a larger value for more history, `0` for all).
        """
        def run() -> dict[str, Any]:
            if not isinstance(tail_lines, int) or tail_lines < 0:
                raise ValidationError(f"tail_lines must be an integer >= 0, got {tail_lines!r}.")
            return logging_setup.read_pypesto_log(session.provenance.root, tail_lines)

        return ctx.run("get_fitting_log", run, {"tail_lines": tail_lines})

    @mcp.tool()
    def query_data(sql: str, max_rows: int = 50) -> dict[str, Any]:
        """Run a SQL query (DuckDB) over the PEtab tables and the simulation dataframe.

        The query runs against a private, disposable copy of the data, so any SQL is allowed
        and nothing can affect the session (host file/network access is disabled). Tables:
        `measurements`, `conditions`, `observables`, `parameters`; `simulations` — the
        per-point dataframe from the last commit, with `measurement`, `simulation`, `residual`
        (natural units), `residual_weighted` (the noise-normalized chi2 term), and `noise`
        (sigma); and the last fit's pyPESTO multistart in `optimization_starts` (one row per
        start, `start`=0 is best) and `optimization_trace` (fval per start/iteration). Call
        `list_data_tables()` for the exact columns. Returns
        `{columns, rows, row_count, truncated}` (capped at `max_rows`, default 50 to keep
        results compact — raise it explicitly for a full dump; prefer aggregations); the
        `simulations` dataframe is generated on demand if no commit has produced one yet.

        This replaces the old fixed residual-analysis tools — compute whatever breakdown you
        need in SQL. Filter to `residual_weighted IS NOT NULL` for residual stats (rows with
        an unsimulated point or an un-evaluable noise model have a NULL residual). Per-
        observable chi2 and worst-fit points, for example:
          `SELECT observable_id, count(*) AS n,
                  sum(residual_weighted*residual_weighted) AS chi2
             FROM simulations WHERE residual_weighted IS NOT NULL
             GROUP BY observable_id ORDER BY chi2 DESC`
          `SELECT * FROM simulations WHERE residual_weighted IS NOT NULL
             ORDER BY abs(residual_weighted) DESC LIMIT 10`
        """
        def run() -> dict[str, Any]:
            # Reject bad arguments cheaply before generating a (compile-heavy) simulation.
            data_mod.validate_query_args(sql, max_rows)
            # Populate the stored simulation dataframe on demand so `simulations` is
            # queryable (a no-op once a commit has produced one). A simulation failure must
            # not block querying the static tables, so fall through with an empty table.
            if session.simulation is None or not session.simulation.get("records"):
                try:
                    ctx.current_simulation()
                except Exception:  # pragma: no cover - defensive; static tables still work
                    session.logger.warning(
                        "query_data: could not generate simulation dataframe", exc_info=True
                    )
            return data_mod.run_query(session, sql, max_rows)

        # Log the query (truncated) alongside max_rows so a failed call is reproducible
        # from `get_errors`.
        logged_sql = sql[:1000] if isinstance(sql, str) else sql
        return ctx.run("query_data", run, {"sql": logged_sql, "max_rows": max_rows})

    @mcp.tool()
    def get_fit_diagnosis(worst_n: int = 8) -> dict[str, Any]:
        """A COMPACT, size-bounded residual diagnosis of the current fit — where/how it is off.

        Fixed aggregations (NOT open SQL): one row per observable with its weighted `chi2`,
        `mean_weighted_residual` (bias — far from 0 = a systematic over/under-shoot), and worst
        |residual|; plus the `worst_n` largest-residual points (measurement vs simulation). Use
        this to locate a poor fit cheaply, without pulling the raw per-point data. For a full
        custom breakdown use `query_data`; for the optimizer log use `get_fitting_log`. The
        simulation dataframe is generated on demand if no commit has produced one yet.
        """
        def run() -> dict[str, Any]:
            if session.simulation is None or not session.simulation.get("records"):
                try:
                    ctx.current_simulation()
                except Exception:  # pragma: no cover - defensive; return the empty-fit result
                    session.logger.warning(
                        "get_fit_diagnosis: could not generate simulation dataframe", exc_info=True
                    )
            return data_mod.fit_diagnosis(session, worst_n)

        return ctx.run("get_fit_diagnosis", run, {"worst_n": worst_n})

    @mcp.tool()
    def get_identifiability(top_n: int = 8) -> dict[str, Any]:
        """Parameter-identifiability diagnosis of the current fit — WHICH estimated parameters
        are unconstrained or trade off, i.e. the basis for SIMPLIFYING an over-parameterised
        model (raw nll/bic tell you it's too complex, not which parameters to cut). Combines the
        Fisher Information Matrix at the optimum — per-parameter standard error + fold_uncertainty,
        marginal and partial correlations, and the near-flat eigen-directions (= non-identifiable
        parameter COMBINATIONS) — with the multistart spread (how much each parameter wanders
        across near-optimal starts, a model-free check). Returns a ranked `candidates` list
        (at_bound / poorly_constrained / collinear / flat_direction / multistart_unstable): drop
        its term, take its limiting (irreversible/instantaneous) form, or MERGE a collinear pair
        into one parameter — all model edits (there is no value/`estimate` tool). If none applies
        cleanly, leave the parameter as is rather than hard-code a value. Bounded output; needs a
        fitted model. Read se/correlations together with multistart_std — the FIM is local.
        """
        def run() -> dict[str, Any]:
            return identifiability_mod.identifiability(session, engine, top_n=top_n)

        return ctx.run("get_identifiability", run, {"top_n": top_n})


# --------------------------------------------------------------- batching tools
def _register_batch_tools(mcp: FastMCP, ctx: _Ctx) -> None:
    session = ctx.session

    @mcp.tool()
    def begin_batch(label: str = "") -> dict[str, Any]:
        """Start a batch: subsequent operations apply + log but defer versioning until commit_batch."""
        return ctx.run("begin_batch", lambda: session.begin_batch(label or None), {"label": label})

    @mcp.tool()
    def commit_batch(label: str = "", n_starts: int | None = None) -> dict[str, Any]:
        """Snapshot all staged operations as a single new version, fitting + simulating
        once for the whole batch (`n_starts` overrides the default; 0 = simulate only)."""
        return ctx.run("commit_batch", lambda: session.commit_batch(label or None, n_starts=n_starts),
                       {"label": label, "n_starts": n_starts})

    @mcp.tool()
    def abort_batch() -> dict[str, Any]:
        """Discard staged operations and restore the pre-batch version."""
        return ctx.run("abort_batch", session.abort_batch)


# ------------------------------------------------------ provenance/version tools
def _register_provenance_tools(mcp: FastMCP, ctx: _Ctx) -> None:
    session = ctx.session

    @mcp.tool()
    def get_provenance(limit: int = 50) -> list[dict[str, Any]]:
        """Return the most recent provenance events (alterations, batches, rollbacks, resets)."""
        return session.provenance.recent_events(limit)

    @mcp.tool()
    def get_errors(limit: int = 50) -> list[dict[str, Any]]:
        """Return the most recent logged errors (rejected operations / failed tool calls)."""
        return session.provenance.recent_errors(limit)

    @mcp.tool()
    def list_versions() -> list[dict[str, Any]]:
        """List all model versions (snapshots) with kind/label/timestamp and fit metrics
        (neg_log_likelihood, chi2, bic, is_new_best) — use it to rank versions by BIC and find
        the best-scoring one. `is_new_best` tracks BIC (gated on a minimum effect size)."""
        return session.provenance.list_versions()

    @mcp.tool()
    def get_version(version: int) -> dict[str, Any]:
        """Return a version's Antimony model and parameter table."""
        return ctx.run("get_version", lambda: session.provenance.get_version(version), {"version": version})

    @mcp.tool()
    def diff_versions(from_version: int, to_version: int) -> dict[str, Any]:
        """Diff two versions: Antimony unified diff + parameter add/remove/change + a `fit`
        block with each version's neg_log_likelihood/chi2 and the neg-log-likelihood delta
        (positive = the `to` version fits better)."""
        return ctx.run("diff_versions", lambda: session.provenance.diff_versions(from_version, to_version),
                       {"from": from_version, "to": to_version})

    @mcp.tool()
    def rollback_to(version: int, justification: str = "") -> dict[str, Any]:
        """Revert the working model + parameters to a prior version (appends a new version)."""
        return ctx.run("rollback_to", lambda: session.rollback_to(version, justification or None),
                       {"version": version})

    @mcp.tool()
    def restore_version(version: int, justification: str = "") -> dict[str, Any]:
        """Bring a prior version's model + parameters back as the working state, WITHOUT refitting.

        The cheap, correct way to go back to an earlier model: it restores that version's stored
        fit instead of recompiling and refitting. Prefer this over re-authoring an earlier model
        with `set_model` — a re-author recompiles, refits, and appends a byte-identical duplicate
        version (and `set_model` will warn, then refuse, once it detects the repeat), whereas this
        reuses the existing fit and records a clean restore. Find the target with `list_versions`
        (pick the lowest-bic / is_new_best version), then `restore_version(v)`. Appends a new
        version, like `rollback_to` (of which this is the aptly-named equivalent)."""
        return ctx.run("restore_version",
                       lambda: session.restore_version(version, justification or None),
                       {"version": version})

    @mcp.tool()
    def current_version() -> dict[str, Any]:
        """Return the current (HEAD) version number and batch state."""
        return {"version": session.provenance.current_version, "batch_active": session.batch_active}

    @mcp.tool()
    def get_progress() -> dict[str, Any]:
        """Early-stop-chain progress for this run — a cheap counter read (no fit/simulation).

        `{n_fitted_ops, non_improving_streak, streak_cap, best_version, best_bic,
        best_gof_pvalue, bic_improvement_over_last_n, plateau_window, should_stop, stop_reason}`.
        `best_version` is the lowest-BIC fitted model committed so far (the checkpoint to score);
        `non_improving_streak` counts consecutive fitted ops since the last real new best.
        `should_stop` is set (with a `stop_reason`) once the run has stopped making progress — the
        hard non-improving-streak cap, or a plateau once the best model is statistically adequate.
        """
        return session.progress_report()


# ---------------------------------------------------------------- session tools
def _register_session_tools(mcp: FastMCP, ctx: _Ctx) -> None:
    session = ctx.session

    @mcp.tool()
    def reset() -> dict[str, Any]:
        """Revert model and parameters to the initial version (v0)."""
        return ctx.run("reset", session.reset)


# --------------------------------------------------------------------- resources
def _register_resources(mcp: FastMCP, ctx: _Ctx) -> None:
    session = ctx.session

    @mcp.resource("petab://model/antimony")
    def _res_model_antimony() -> str:
        return model_mod.get_model(session, "antimony")

    @mcp.resource("petab://model/sbml")
    def _res_model_sbml() -> str:
        return model_mod.get_model(session, "sbml")

    @mcp.resource("petab://observables")
    def _res_observables() -> str:
        return json.dumps(df_to_records(session.problem.observable_df), indent=2)

    @mcp.resource("petab://conditions")
    def _res_conditions() -> str:
        return json.dumps(df_to_records(session.problem.condition_df), indent=2)

    @mcp.resource("petab://measurements")
    def _res_measurements() -> str:
        return json.dumps(df_to_records(session.problem.measurement_df), indent=2)

    @mcp.resource("petab://parameters")
    def _res_parameters() -> str:
        return json.dumps(param_mod.parameters_to_records(session.problem), indent=2)

    @mcp.resource("petab://provenance")
    def _res_provenance() -> str:
        return json.dumps(session.provenance.recent_events(200), indent=2)

    @mcp.resource("petab://errors")
    def _res_errors() -> str:
        return json.dumps(session.provenance.recent_errors(200), indent=2)

    @mcp.resource("petab://versions")
    def _res_versions() -> str:
        return json.dumps(session.provenance.list_versions(), indent=2)

    @mcp.resource("petab://simulations")
    def _res_simulations() -> str:
        return json.dumps(ctx.run("simulations", ctx.current_simulation), indent=2)

    @mcp.resource("petab://data/schema")
    def _res_data_schema() -> str:
        return json.dumps(data_mod.table_schema(session), indent=2)

    @mcp.resource("petab://fit/log")
    def _res_fit_log() -> str:
        path = logging_setup.pypesto_log_path(session.provenance.root)
        return path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""


def create_server(
    yaml_path: str,
    build_root: str | None = None,
    provenance_dir: str | None = None,
    reference_mode: str = "best_effort",
    resolver: Any | None = None,
    skills_dir: str | None = None,
    auto_fit: bool = True,
    auto_fit_starts: int = DEFAULT_AUTO_FIT_STARTS,
    estimate_initial_conditions: bool = False,
) -> FastMCP:
    """Build a FastMCP server bound to the PEtab problem at ``yaml_path``.

    ``skills_dir`` overrides where tool-usage skills are loaded from; when omitted
    the ``_skills`` directory bundled with the package is discovered automatically.

    ``auto_fit`` (default on) makes every commit fit + simulate before snapshotting;
    ``auto_fit_starts`` is the default pyPESTO multistart count (overridable per commit via
    ``n_starts``). Set ``auto_fit=False`` to skip the compile-heavy fit hook (tests/dev).

    ``estimate_initial_conditions`` (opt-in, default off): estimate initial-condition parameters
    rather than fixing them at the observed t=0 value. Off keeps a problem's standard ICs.
    """
    session = PetabSession(
        yaml_path,
        provenance_dir=provenance_dir,
        reference_mode=reference_mode,
        resolver=resolver,
    )
    engine = AmiciEngine(session, build_root=build_root)
    # Wire the engine so commits (operations / commit_batch) can fit + simulate.
    session.engine = engine
    session.auto_fit = auto_fit
    session.auto_fit_starts = auto_fit_starts
    # Opt-in: estimate initial-condition parameters instead of fixing them at the observed t=0
    # value (off by default so published problems keep their standard IC treatment).
    session.estimate_initial_conditions = estimate_initial_conditions

    # Load tool-usage skills up front so the server instructions only advertise
    # `list_skills`/`get_skill` when skills are actually available (registered below).
    resolved_skills_dir = skills_dir or skills_mod.default_skills_dir()
    skills = skills_mod.load_skills(resolved_skills_dir)
    instructions = SERVER_INSTRUCTIONS + (_SKILLS_INSTRUCTION if skills else "")
    mcp = FastMCP("petab-mcp", instructions=instructions)

    # Register the tool + resource groups; each closes over the shared context.
    ctx = _Ctx(session, engine)
    _register_read_tools(mcp, ctx)
    _register_model_tools(mcp, ctx)
    _register_parameter_tools(mcp, ctx)
    _register_fit_tools(mcp, ctx)
    _register_batch_tools(mcp, ctx)
    _register_provenance_tools(mcp, ctx)
    _register_session_tools(mcp, ctx)
    _register_resources(mcp, ctx)
    # Register the skills loaded above as `list_skills`/`get_skill` tools plus
    # `petab://skills` resources (no-op when no skills were found).
    skills_mod.register_skills(mcp, skills)

    return mcp
