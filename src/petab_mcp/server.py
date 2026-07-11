"""The MCP server: FastMCP tools and resources over a :class:`PetabSession`.

Tool groups:

* read-only (fixed design): ``describe_problem``, ``list_observables``,
  ``list_conditions``, ``get_measurements``
* model (constrained operations): ``add_reaction``, ``change_rate_law``,
  ``remove_reaction`` (justification + references), ``change_parameter_scale``,
  ``change_parameter_bounds`` (justification; references optional), plus read-only
  ``get_model``, ``get_model_structure``, ``validate_problem``
* parameters (read-only + reconcile): ``get_parameters``, ``sync_parameters``
* batching: ``begin_batch``, ``commit_batch``, ``abort_batch``
* fit + simulation: **automatic on every commit** — each commit fits the free
  parameters and simulates, storing a simulation dataframe on the new version; inspect it
  with ``inspect_fit`` and ``get_simulations``
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

from . import fit as fit_mod
from . import fitting as fitting_mod
from . import model as model_mod
from . import operations as ops_mod
from . import parameters as param_mod
from . import skills as skills_mod
from .engine import AmiciEngine
from .errors import PetabMcpError
from .formatting import df_to_records
from .session import PetabSession
from .validation import validate_problem as _validate_problem

SERVER_INSTRUCTIONS = (
    "Interact with a fixed PEtab v1 estimation problem. The experimental design "
    "(measurements, conditions, observables) is read-only. The model is edited ONLY "
    "through three constrained, justified operations — `add_reaction`, "
    "`change_rate_law`, `remove_reaction` — each of which requires a justification and "
    "PubMed-validated references and automatically keeps the SBML model and PEtab "
    "parameter table consistent (parametric initial conditions). Parameter scales and "
    "bounds are changed through versioned, justified operations "
    "(`change_parameter_scale`, `change_parameter_bounds`); parameter *values* are set "
    "only by the automatic fit, not by hand. EVERY commit automatically fits the free "
    "parameters and simulates, then snapshots a single version carrying the fitted "
    "parameters, the fit metrics (neg_log_likelihood, chi2, improvement vs prior "
    "versions), and a simulation dataframe. Pass `n_starts` on a commit to fit harder "
    "(or 0 to simulate without fitting). Chain several edits into one version — fitted "
    "once, on commit — with `begin_batch`/`commit_batch`/`abort_batch`. Read the stored "
    "simulation dataframe with `get_simulations` and diagnose the fit with `inspect_fit`, "
    "which breaks residuals down across observables, experiments/conditions, and "
    "timepoints and surfaces the worst-fit points. Every change is versioned and logged: "
    "inspect with `list_versions`, `diff_versions`, `get_provenance`; errors via "
    "`get_errors`; revert with `rollback_to` or `reset`."
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
        condition_id: str | None = None, observable_id: str | None = None
    ) -> list[dict[str, Any]]:
        """Return the (fixed) measurement data, optionally filtered by condition."""
        df = session.problem.measurement_df
        if df is not None:
            if condition_id is not None:
                df = df[df["simulationConditionId"] == condition_id]
            if observable_id is not None:
                df = df[df["observableId"] == observable_id]
        return df_to_records(df)

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

    def _run_operation(op_name, fn, *args) -> dict[str, Any]:
        """Run an operation; on a known error, log it and return committed=False."""
        try:
            return fn(*args)
        except PetabMcpError as exc:
            _log_error(op_name, {"args": [a for a in args if isinstance(a, (str, int, float, list))]}, exc)
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
    def add_reaction(
        reaction: str, justification: str, references: list[str], n_starts: int | None = None
    ) -> dict[str, Any]:
        """Add a reaction, given as an Antimony line, e.g. "J: A + B -> C; k_f*A*B".

        New species/parameters and their parametric initial conditions are added
        automatically to keep the SBML model and PEtab tables consistent.
        `justification` (non-empty) and `references` (>=1, PubMed-validated) are required.
        On commit the free parameters are fitted and the model simulated; `n_starts`
        overrides the server default (0 = simulate without fitting).
        """
        return _run_operation("add_reaction", ops_mod.add_reaction, session, reaction, justification, references, n_starts)

    @mcp.tool()
    def change_rate_law(
        reaction_id: str, rate_law: str, justification: str, references: list[str],
        n_starts: int | None = None,
    ) -> dict[str, Any]:
        """Replace a reaction's rate law (Antimony-like expression). Requires
        justification + references; parameters are reconciled automatically. Fits +
        simulates on commit (`n_starts` overrides the default; 0 = simulate only)."""
        return _run_operation(
            "change_rate_law", ops_mod.change_rate_law, session, reaction_id, rate_law, justification, references, n_starts
        )

    @mcp.tool()
    def remove_reaction(
        reaction_id: str, justification: str, references: list[str], n_starts: int | None = None
    ) -> dict[str, Any]:
        """Remove a reaction entirely and prune any orphaned species/parameters.
        Requires justification + references. Fits + simulates on commit (`n_starts`
        overrides the default; 0 = simulate only)."""
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
    def get_simulations() -> dict[str, Any]:
        """Return the stored simulation dataframe: the fit metrics plus per-point records
        (measurement, simulation, residual, noise-weighted residual, noise).

        Refreshed automatically on every commit. If no simulation exists yet (fresh
        problem, no commit) it is generated on demand from the current parameters.
        """
        return _run("get_simulations", _current_simulation)

    @mcp.tool()
    def inspect_fit(top_n: int = 10) -> dict[str, Any]:
        """Inspect the current fit to the fixed data using the stored simulation dataframe.

        Reports residual statistics overall and across each cross section (observables,
        experiments/conditions, timepoints), the `top_n` worst-fit points, and
        systematic-bias flags — so you can see *where* the model fits poorly, not just the
        aggregate objective. Weighted residuals are noise-normalized (the chi2 terms); raw
        residuals are in observable units. Reads the dataframe produced by the last commit
        (no re-simulation).
        """
        def run() -> dict[str, Any]:
            sim = _current_simulation()
            report = fit_mod.fit_inspection(sim["records"], top_n=top_n)
            report["neg_log_likelihood"] = sim.get("neg_log_likelihood")
            report["chi2"] = sim.get("chi2")
            return report

        return _run("inspect_fit", run, {"top_n": top_n})

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
                    {"label": label})

    @mcp.tool()
    def abort_batch() -> dict[str, Any]:
        """Discard staged operations and restore the pre-batch version."""
        return _run("abort_batch", session.abort_batch)

    # ------------------------------------------------------ provenance/versions
    @mcp.tool()
    def get_provenance(limit: int = 50) -> list[dict[str, Any]]:
        """Return the most recent provenance events (alterations, training, rollbacks)."""
        return session.provenance.recent_events(limit)

    @mcp.tool()
    def get_errors(limit: int = 50) -> list[dict[str, Any]]:
        """Return the most recent logged errors (rejected operations / failed tool calls)."""
        return session.provenance.recent_errors(limit)

    @mcp.tool()
    def list_versions() -> list[dict[str, Any]]:
        """List all model versions (snapshots) with their kind/label/timestamp."""
        return session.provenance.list_versions()

    @mcp.tool()
    def get_version(version: int) -> dict[str, Any]:
        """Return a version's Antimony model and parameter table."""
        return _run("get_version", lambda: session.provenance.get_version(version), {"version": version})

    @mcp.tool()
    def diff_versions(from_version: int, to_version: int) -> dict[str, Any]:
        """Diff two versions (Antimony unified diff + parameter add/remove/change)."""
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

    @mcp.resource("petab://fit")
    def _res_fit() -> str:
        def run() -> dict[str, Any]:
            sim = _current_simulation()
            report = fit_mod.fit_inspection(sim["records"], top_n=10)
            report["neg_log_likelihood"] = sim.get("neg_log_likelihood")
            report["chi2"] = sim.get("chi2")
            return report

        return json.dumps(_run("inspect_fit", run), indent=2)

    @mcp.resource("petab://simulations")
    def _res_simulations() -> str:
        return json.dumps(_run("get_simulations", _current_simulation), indent=2)

    # ---------------------------------------------------------------- skills
    # Register the skills loaded above as `list_skills`/`get_skill` tools plus
    # `petab://skills` resources (no-op when no skills were found).
    skills_mod.register_skills(mcp, skills)

    # keep references alive / silence linters for closure-registered callables
    _ = (PetabMcpError,)
    return mcp
