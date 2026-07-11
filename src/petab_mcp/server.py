"""The MCP server: FastMCP tools and resources over a :class:`PetabSession`.

Tool groups:

* read-only (fixed design): ``describe_problem``, ``list_observables``,
  ``list_conditions``, ``get_measurements``
* model (constrained operations): ``add_reaction``, ``change_rate_law``,
  ``remove_reaction`` (each requires justification + references), plus read-only
  ``get_model``, ``get_model_structure``, ``validate_problem``
* parameters (mutable): ``get_parameters``, ``update_parameters``, ``sync_parameters``
* evaluation: ``simulate``, ``evaluate_objective``, ``optimize``
* provenance/versions: ``get_provenance``, ``list_versions``, ``get_version``,
  ``diff_versions``, ``rollback_to``, ``current_version``
* session: ``reset``

No tool mutates the measurement/condition/observable tables — that invariant is
enforced structurally (there is simply no such tool) and re-checked by the operation
pipeline's lint (see ``operations``/``validation``).
"""

from __future__ import annotations

import json
from typing import Any

from mcp.server.fastmcp import FastMCP

from . import model as model_mod
from . import operations as ops_mod
from . import optimize as optimize_mod
from . import parameters as param_mod
from .engine import AmiciEngine
from .errors import PetabMcpError, ValidationError
from .formatting import df_to_records
from .session import PetabSession
from .validation import validate_problem as _validate_problem

SERVER_INSTRUCTIONS = (
    "Interact with a fixed PEtab v1 estimation problem. The experimental design "
    "(measurements, conditions, observables) is read-only. The model is edited ONLY "
    "through three constrained, justified operations — `add_reaction`, "
    "`change_rate_law`, `remove_reaction` — each of which requires a justification and "
    "PubMed-validated references and automatically keeps the SBML model and PEtab "
    "parameter table consistent (parametric initial conditions). Tune parameters with "
    "`update_parameters`, score with `simulate`/`evaluate_objective`, fit with "
    "`optimize`. Every change is versioned and logged: inspect with `list_versions`, "
    "`diff_versions`, `get_provenance`; revert with `rollback_to` or `reset`."
)


def _amici_available() -> bool:
    try:
        import amici  # noqa: F401
        from amici.importers.petab.v1 import import_petab_problem  # noqa: F401

        return True
    except Exception:
        return False


def create_server(
    yaml_path: str,
    build_root: str | None = None,
    provenance_dir: str | None = None,
    reference_mode: str = "best_effort",
    resolver: Any | None = None,
) -> FastMCP:
    """Build a FastMCP server bound to the PEtab problem at ``yaml_path``."""
    session = PetabSession(
        yaml_path,
        provenance_dir=provenance_dir,
        reference_mode=reference_mode,
        resolver=resolver,
    )
    engine = AmiciEngine(session, build_root=build_root)
    mcp = FastMCP("petab-mcp", instructions=SERVER_INSTRUCTIONS)

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
            "fixed_tables": ["measurements", "conditions", "observables"],
            "amici_available": _amici_available(),
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

    def _run_operation(fn, *args) -> dict[str, Any]:
        try:
            return fn(*args)
        except ValidationError as exc:
            return {"committed": False, "error": str(exc), "issues": exc.issues}
        except PetabMcpError as exc:
            return {"committed": False, "error": str(exc)}

    @mcp.tool()
    def add_reaction(
        reaction: str, justification: str, references: list[str]
    ) -> dict[str, Any]:
        """Add a reaction, given as an Antimony line, e.g. "J: A + B -> C; k_f*A*B".

        New species/parameters and their parametric initial conditions are added
        automatically to keep the SBML model and PEtab tables consistent.
        `justification` (non-empty) and `references` (>=1, PubMed-validated) are required.
        """
        return _run_operation(ops_mod.add_reaction, session, reaction, justification, references)

    @mcp.tool()
    def change_rate_law(
        reaction_id: str, rate_law: str, justification: str, references: list[str]
    ) -> dict[str, Any]:
        """Replace a reaction's rate law (Antimony-like expression). Requires
        justification + references; parameters are reconciled automatically."""
        return _run_operation(
            ops_mod.change_rate_law, session, reaction_id, rate_law, justification, references
        )

    @mcp.tool()
    def remove_reaction(
        reaction_id: str, justification: str, references: list[str]
    ) -> dict[str, Any]:
        """Remove a reaction entirely and prune any orphaned species/parameters.
        Requires justification + references."""
        return _run_operation(ops_mod.remove_reaction, session, reaction_id, justification, references)

    @mcp.tool()
    def validate_problem() -> dict[str, Any]:
        """Lint the current problem (model + tables) and report issues."""
        return _validate_problem(session.problem)

    # ------------------------------------------------------------ parameters
    @mcp.tool()
    def get_parameters() -> list[dict[str, Any]]:
        """Return the parameter table (id, bounds, nominal_value, estimate, prior)."""
        return param_mod.parameters_to_records(session.problem)

    @mcp.tool()
    def update_parameters(updates: list[dict]) -> dict[str, Any]:
        """Edit parameters. Each update: {id, nominalValue?, lowerBound?, upperBound?,
        estimate?, parameterScale?}."""
        with session.lock:
            return param_mod.update_parameters(session.problem, updates)

    @mcp.tool()
    def sync_parameters(policy: str = "report") -> dict[str, Any]:
        """Reconcile the parameter table with the model ('report' or 'add')."""
        with session.lock:
            return param_mod.sync_parameters(session.problem, policy)

    # ------------------------------------------------------------ evaluation
    @mcp.tool()
    def simulate() -> dict[str, Any]:
        """Simulate all experiments with the current model + parameter values."""
        return engine.simulate()

    @mcp.tool()
    def evaluate_objective() -> dict[str, Any]:
        """Score the current model/parameters vs the fixed data (neg-log-likelihood)."""
        return engine.evaluate_objective()

    @mcp.tool()
    def optimize(n_starts: int = 5, seed: int = 0) -> dict[str, Any]:
        """Estimate the free parameters; writes the best fit back into the table."""
        with session.lock:
            return optimize_mod.optimize(session, engine, n_starts=n_starts, seed=seed)

    # ------------------------------------------------------ provenance/versions
    @mcp.tool()
    def get_provenance(limit: int = 50) -> list[dict[str, Any]]:
        """Return the most recent provenance events (alterations, training, rollbacks)."""
        return session.provenance.recent_events(limit)

    @mcp.tool()
    def list_versions() -> list[dict[str, Any]]:
        """List all model versions (snapshots) with their kind/label/timestamp."""
        return session.provenance.list_versions()

    @mcp.tool()
    def get_version(version: int) -> dict[str, Any]:
        """Return a version's Antimony model and parameter table."""
        return session.provenance.get_version(version)

    @mcp.tool()
    def diff_versions(from_version: int, to_version: int) -> dict[str, Any]:
        """Diff two versions (Antimony unified diff + parameter add/remove/change)."""
        return session.provenance.diff_versions(from_version, to_version)

    @mcp.tool()
    def rollback_to(version: int, justification: str = "") -> dict[str, Any]:
        """Revert the working model + parameters to a prior version (appends a new version)."""
        return session.rollback_to(version, justification or None)

    @mcp.tool()
    def current_version() -> dict[str, Any]:
        """Return the current (HEAD) version number."""
        return {"version": session.provenance.current_version}

    # --------------------------------------------------------------- session
    @mcp.tool()
    def reset() -> dict[str, Any]:
        """Revert model and parameters to the initial version (v0)."""
        return session.reset()

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

    @mcp.resource("petab://versions")
    def _res_versions() -> str:
        return json.dumps(session.provenance.list_versions(), indent=2)

    # keep references alive / silence linters for closure-registered callables
    _ = (PetabMcpError,)
    return mcp
