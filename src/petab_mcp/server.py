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
"""

from __future__ import annotations

import json
from typing import Any

from mcp.server.fastmcp import FastMCP
from petab.v1.C import PARAMETER_SCALE

from . import data_access as data_mod
from . import fitting as fitting_mod
from . import model as model_mod
from . import operations as ops_mod
from . import parameters as param_mod
from . import skills as skills_mod
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
    "conditions). Parameter scales and "
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
    "with `get_provenance`; errors via `get_errors`; revert with `rollback_to` or `reset`."
)

# Appended to the instructions only when skills are actually registered, so clients
# are never told to call `list_skills`/`get_skill` when those tools are absent.
_SKILLS_INSTRUCTION = (
    " New to this server? Call `list_skills` and read "
    "`get_skill('petab-model-building')` for the full tool-usage workflow."
)


def create_server(
    yaml_path: str,
    build_root: str | None = None,
    provenance_dir: str | None = None,
    reference_mode: str = "best_effort",
    resolver: Any | None = None,
    skills_dir: str | None = None,
    auto_fit: bool = True,
    auto_fit_starts: int = 5,
) -> FastMCP:
    """Build a FastMCP server bound to the PEtab problem at ``yaml_path``.

    ``skills_dir`` overrides where tool-usage skills are loaded from; when omitted
    the ``_skills`` directory bundled with the package is discovered automatically.

    ``auto_fit`` (default on) makes every commit fit + simulate before snapshotting;
    ``auto_fit_starts`` is the default pyPESTO multistart count (overridable per commit via
    ``n_starts``). Set ``auto_fit=False`` to skip the compile-heavy fit hook (tests/dev).
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

    # Load tool-usage skills up front so the server instructions only advertise
    # `list_skills`/`get_skill` when skills are actually available (registered below).
    resolved_skills_dir = skills_dir or skills_mod.default_skills_dir()
    skills = skills_mod.load_skills(resolved_skills_dir)
    instructions = SERVER_INSTRUCTIONS + (_SKILLS_INSTRUCTION if skills else "")
    mcp = FastMCP("petab-mcp", instructions=instructions)

    # ------------------------------------------------------------------ read
    @mcp.tool()
    def describe_problem() -> dict[str, Any]:
        """Summarize the problem: model size, tables, and free parameters."""
        p = session.problem
        sm = session.model.sbml_model
        return {
            "model_id": session.model.model_id,
            "model_initialized": session.has_model(),
            "n_species": int(sm.getNumSpecies()),
            "n_reactions": int(sm.getNumReactions()),
            "n_model_parameters": int(sm.getNumParameters()),
            "n_table_parameters": int(len(p.parameter_df)),
            "n_estimated": int(p.n_estimated),
            "free_parameters": [str(x) for x in p.x_free_ids],
            "observables": [str(x) for x in p.get_observable_ids()],
            "n_conditions": int(len(p.condition_df)),
            "n_measurements": int(p.n_measurements),
            "n_reactions_detail": [sm.getReaction(i).getId() for i in range(sm.getNumReactions())],
            "current_version": session.provenance.current_version,
            "batch_active": session.batch_active,
            "fixed_tables": ["measurements", "conditions", "observables"],
            # current fit metrics from the stored simulation (None until the first
            # commit/simulation); read `get_fit_summary`, or query `simulations` via
            # `query_data`, for more.
            "neg_log_likelihood": (session.simulation or {}).get("neg_log_likelihood"),
            "chi2": (session.simulation or {}).get("chi2"),
        }

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
                df = df[df["simulationConditionId"] == condition_id]
            if observable_id is not None:
                df = df[df["observableId"] == observable_id]
        return summarize_measurements(df) if summarize else df_to_records(df)

    # ----------------------------------------------------------------- model
    @mcp.tool()
    def get_model(format: str = "antimony") -> str:
        """Return the current model as text. `format` is 'antimony' (canonical) or 'sbml'."""
        return model_mod.get_model(session, format)

    @mcp.tool()
    def get_model_structure() -> dict[str, Any]:
        """Structured view of the model: species, parameters, reactions."""
        return model_mod.model_structure(session)

    def _log_error(tool: str, args: dict[str, Any], exc: Exception) -> None:
        session.provenance.log_error(
            {
                "tool": tool,
                "args": args,
                "error_type": type(exc).__name__,
                "message": str(exc),
                "issues": getattr(exc, "issues", None),
            }
        )

    def _short(value: Any, limit: int = 200) -> Any:
        """Truncate long strings before they go into the (persisted) error log, so a large
        argument — e.g. a whole Antimony model passed to `set_model` — cannot bloat the
        provenance error log or dump the full model contents into it."""
        if isinstance(value, str) and len(value) > limit:
            return value[:limit] + f"… (+{len(value) - limit} chars)"
        return value

    def _run_operation(op_name, fn, *args) -> dict[str, Any]:
        """Run an operation; on a known error, log it and return committed=False."""
        try:
            return fn(*args)
        except PetabMcpError as exc:
            _log_error(op_name, {"args": [_short(a) for a in args if isinstance(a, (str, int, float, list))]}, exc)
            session.logger.warning("operation %s rejected: %s", op_name, exc)
            out = {"committed": False, "error": str(exc)}
            if getattr(exc, "issues", None):
                out["issues"] = exc.issues
            return out
        except Exception as exc:
            _log_error(op_name, {}, exc)
            session.logger.exception("operation %s crashed", op_name)
            raise

    def _run(tool: str, fn, args: dict[str, Any] | None = None):
        """Run a non-operation tool; log failures (return {ok:false} or re-raise)."""
        try:
            return fn()
        except PetabMcpError as exc:
            _log_error(tool, args or {}, exc)
            session.logger.warning("%s failed: %s", tool, exc)
            out = {"ok": False, "error": str(exc)}
            if getattr(exc, "issues", None):
                out["issues"] = exc.issues
            return out
        except Exception as exc:
            _log_error(tool, args or {}, exc)
            session.logger.exception("%s crashed", tool)
            raise

    @mcp.tool()
    def set_model(
        antimony: str, justification: str,
        references: list[str] | None = None, n_starts: int | None = None,
    ) -> dict[str, Any]:
        """Author the WHOLE model from an Antimony string — the single whole-model operation.

        Use it to BUILD the first model when the problem ships model-less (`describe_problem`
        reports `model_initialized: false`) AND to re-author/replace an existing model
        wholesale — it is never refused. Reach for it to make a STRUCTURAL change as one
        coherent step — a coupled or symmetric rate-law change, or adding/removing several
        reactions at once — instead of a sequence of incremental edits that leave intermediate
        models inconsistent (the `add_reaction` / `change_rate_law` / `remove_reaction`
        operations remain for local refinement). It reconciles the SBML model with the
        PEtab parameter table (parametric initial conditions, parameter-table sync), lints, and
        fits on commit, exactly like the other model ops; any failure reverts, so a bad
        re-author changes nothing. `justification` (non-empty) is required; `references` are
        optional; `n_starts` overrides the fit default (0 = simulate only). Because the whole
        model is replaced, the result `warnings` flag any previously-ESTIMATED parameter the
        re-author no longer declares (its fitted value is dropped); fixed noise/initial-
        condition parameters and anything the observable/condition tables reference survive.
        """
        return _run_operation("set_model", ops_mod.set_model, session, antimony, justification, references, n_starts)

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
        return _run_operation("add_reaction", ops_mod.add_reaction, session, reaction, justification, references, n_starts)

    @mcp.tool()
    def change_rate_law(
        reaction_id: str, rate_law: str, justification: str,
        references: list[str] | None = None, n_starts: int | None = None,
    ) -> dict[str, Any]:
        """Replace a reaction's rate law (Antimony-like expression). Requires a
        justification (references optional); parameters are reconciled automatically. Fits +
        simulates on commit (`n_starts` overrides the default; 0 = simulate only)."""
        return _run_operation(
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
        return _run_operation("remove_reaction", ops_mod.remove_reaction, session, reaction_id, justification, references, n_starts)

    @mcp.tool()
    def change_parameter_scale(
        parameter_id: str, scale: str, justification: str,
        references: list[str] | None = None, n_starts: int | None = None,
    ) -> dict[str, Any]:
        """Change a parameter's scale ('lin'|'log'|'log10'). Versioned; requires a
        justification (references optional). Fits + simulates on commit (`n_starts`
        overrides the default; 0 = simulate only)."""
        return _run_operation(
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
        default; 0 = simulate only)."""
        return _run_operation(
            "change_parameter_bounds", ops_mod.change_parameter_bounds,
            session, parameter_id, lower_bound, upper_bound, justification, references, n_starts,
        )

    @mcp.tool()
    def validate_problem() -> dict[str, Any]:
        """Lint the current problem (model + tables) and report issues."""
        return _validate_problem(session.problem)

    # ------------------------------------------------------------ parameters
    @mcp.tool()
    def get_parameters() -> list[dict[str, Any]]:
        """Return the parameter table (id, bounds, nominal_value, estimate, prior).

        Values (`nominalValue`) are set by the automatic fit on commit; adjust bounds and
        scale through `change_parameter_bounds` / `change_parameter_scale`.
        """
        return param_mod.parameters_to_records(session.problem)

    @mcp.tool()
    def sync_parameters(policy: str = "report") -> dict[str, Any]:
        """Reconcile the parameter table with the model ('report' or 'add')."""
        with session.lock:
            return param_mod.sync_parameters(session.problem, policy)

    # ------------------------------------------------- fit + simulation (read)
    def _current_simulation() -> dict[str, Any]:
        """The stored simulation dataframe, generated on demand if not present yet."""
        if session.simulation is None or not session.simulation.get("records"):
            fitting_mod.ensure_simulation(session, engine)
        return session.simulation

    @mcp.tool()
    def get_fit_summary() -> dict[str, Any]:
        """Return just the current fit metrics — no per-point records (a cheap read).

        `{status, neg_log_likelihood, chi2, n_data_points, fitted_parameters,
        parameter_scales, version}` plus the complexity-penalised scores `{bic, aic, chi2_dof,
        reduced_chi2, chi2_gof_pvalue}` (lower bic/aic is better; reduced_chi2 ≈ 1 and a non-tiny
        chi2_gof_pvalue mean the fit has reached the noise floor). `fitted_parameters` are LINEAR
        (unscaled) values; `parameter_scales` gives each one's estimation scale (lin/log10) —
        a fitted value is the linear number, never its log10 coordinate. Use this to check the
        score without pulling the dataframe; use `query_data` to query the
        measured-vs-simulated points and residuals with SQL. Refreshed on every commit;
        generated on demand if none exists.
        """
        def run() -> dict[str, Any]:
            sim = _current_simulation() or {}
            out = {k: sim.get(k) for k in (
                "status", "neg_log_likelihood", "chi2", "n_data_points",
                "fitted_parameters", "parameter_scales", "version",
            )}
            # Always pair fitted values with their scale (restored/rollback payloads may lack the
            # map), so a linear value is never misread as a log10 coordinate.
            if out.get("fitted_parameters") and not out.get("parameter_scales"):
                pdf = session.problem.parameter_df
                out["parameter_scales"] = {
                    pid: str(pdf.loc[pid, PARAMETER_SCALE])
                    for pid in out["fitted_parameters"] if pid in pdf.index}
            # k is the number of estimated parameters (n_estimated), which is well-defined
            # even on a simulate-only pass where `fitted_parameters` is None (e.g. baseline).
            out.update(fitting_mod.goodness_of_fit(
                sim.get("neg_log_likelihood"), sim.get("chi2"),
                int(session.problem.n_estimated), sim.get("n_data_points"),
            ))
            return out

        return _run("get_fit_summary", run)

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

        return _run("get_run_timing", run)

    @mcp.tool()
    def list_data_tables() -> dict[str, Any]:
        """List the SQL tables `query_data` exposes: name, description, columns, row count.

        The tables are the PEtab design tables (`measurements`, `conditions`,
        `observables`), the `parameters` table, the `simulations` dataframe (one row per
        measurement point, carrying `simulation` and the residuals), and the last fit's
        pyPESTO multistart (`optimization_starts`, `optimization_trace`). Use it to learn the
        exact column names before writing a `query_data` query.
        """
        return _run("list_data_tables", lambda: data_mod.table_schema(session))

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
            path = session.provenance.root / "pypesto.log"
            if not path.exists():
                return {"path": str(path), "exists": False, "n_lines_total": 0,
                        "tail_lines": tail_lines, "log": ""}
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
            kept = lines if tail_lines == 0 else lines[-tail_lines:]
            return {"path": str(path), "exists": True, "n_lines_total": len(lines),
                    "tail_lines": tail_lines, "log": "\n".join(kept)}

        return _run("get_fitting_log", run, {"tail_lines": tail_lines})

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
                    _current_simulation()
                except Exception:  # pragma: no cover - defensive; static tables still work
                    session.logger.warning(
                        "query_data: could not generate simulation dataframe", exc_info=True
                    )
            return data_mod.run_query(session, sql, max_rows)

        # Log the query (truncated) alongside max_rows so a failed call is reproducible
        # from `get_errors`.
        logged_sql = sql[:1000] if isinstance(sql, str) else sql
        return _run("query_data", run, {"sql": logged_sql, "max_rows": max_rows})

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
                    _current_simulation()
                except Exception:  # pragma: no cover - defensive; return the empty-fit result
                    session.logger.warning(
                        "get_fit_diagnosis: could not generate simulation dataframe", exc_info=True
                    )
            return data_mod.fit_diagnosis(session, worst_n)

        return _run("get_fit_diagnosis", run, {"worst_n": worst_n})

    @mcp.tool()
    def get_identifiability(top_n: int = 8) -> dict[str, Any]:
        """Parameter-identifiability diagnosis of the current fit — WHICH estimated parameters
        are unconstrained or trade off, i.e. the basis for SIMPLIFYING an over-parameterised
        model (raw nll/bic tell you it's too complex, not which parameters to cut). Combines the
        Fisher Information Matrix at the optimum — per-parameter standard error + fold_uncertainty,
        marginal and partial correlations, and the near-flat eigen-directions (= non-identifiable
        parameter COMBINATIONS) — with the multistart spread (how much each parameter wanders
        across near-optimal starts, a model-free check). Returns a ranked `candidates` list
        (at_bound / poorly_constrained / collinear / flat_direction / multistart_unstable): fix
        it, drop its term, or MERGE a collinear pair into one parameter. Bounded output; needs a
        fitted model. Read se/correlations together with multistart_std — the FIM is local.
        """
        def run() -> dict[str, Any]:
            return fitting_mod.identifiability(session, engine, top_n=top_n)

        return _run("get_identifiability", run, {"top_n": top_n})

    # --------------------------------------------------------------- batching
    @mcp.tool()
    def begin_batch(label: str = "") -> dict[str, Any]:
        """Start a batch: subsequent operations apply + log but defer versioning until commit_batch."""
        return _run("begin_batch", lambda: session.begin_batch(label or None), {"label": label})

    @mcp.tool()
    def commit_batch(label: str = "", n_starts: int | None = None) -> dict[str, Any]:
        """Snapshot all staged operations as a single new version, fitting + simulating
        once for the whole batch (`n_starts` overrides the default; 0 = simulate only)."""
        return _run("commit_batch", lambda: session.commit_batch(label or None, n_starts=n_starts),
                    {"label": label, "n_starts": n_starts})

    @mcp.tool()
    def abort_batch() -> dict[str, Any]:
        """Discard staged operations and restore the pre-batch version."""
        return _run("abort_batch", session.abort_batch)

    # ------------------------------------------------------ provenance/versions
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
        (neg_log_likelihood, chi2, is_new_best) — use it to rank versions by fit."""
        return session.provenance.list_versions()

    @mcp.tool()
    def get_version(version: int) -> dict[str, Any]:
        """Return a version's Antimony model and parameter table."""
        return _run("get_version", lambda: session.provenance.get_version(version), {"version": version})

    @mcp.tool()
    def diff_versions(from_version: int, to_version: int) -> dict[str, Any]:
        """Diff two versions: Antimony unified diff + parameter add/remove/change + a `fit`
        block with each version's neg_log_likelihood/chi2 and the neg-log-likelihood delta
        (positive = the `to` version fits better)."""
        return _run("diff_versions", lambda: session.provenance.diff_versions(from_version, to_version),
                    {"from": from_version, "to": to_version})

    @mcp.tool()
    def rollback_to(version: int, justification: str = "") -> dict[str, Any]:
        """Revert the working model + parameters to a prior version (appends a new version)."""
        return _run("rollback_to", lambda: session.rollback_to(version, justification or None),
                    {"version": version})

    @mcp.tool()
    def current_version() -> dict[str, Any]:
        """Return the current (HEAD) version number and batch state."""
        return {"version": session.provenance.current_version, "batch_active": session.batch_active}

    # --------------------------------------------------------------- session
    @mcp.tool()
    def reset() -> dict[str, Any]:
        """Revert model and parameters to the initial version (v0)."""
        return _run("reset", session.reset)

    # ------------------------------------------------------------- resources
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
        return json.dumps(_run("simulations", _current_simulation), indent=2)

    @mcp.resource("petab://data/schema")
    def _res_data_schema() -> str:
        return json.dumps(data_mod.table_schema(session), indent=2)

    @mcp.resource("petab://fit/log")
    def _res_fit_log() -> str:
        path = session.provenance.root / "pypesto.log"
        return path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""

    # ---------------------------------------------------------------- skills
    # Register the skills loaded above as `list_skills`/`get_skill` tools plus
    # `petab://skills` resources (no-op when no skills were found).
    skills_mod.register_skills(mcp, skills)

    # keep references alive / silence linters for closure-registered callables
    _ = (PetabMcpError,)
    return mcp
