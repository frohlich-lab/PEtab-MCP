"""The MCP server: FastMCP tools and resources over a :class:`PetabSession`.

Tool groups:

* read-only (fixed design): ``describe_problem``, ``list_observables``,
  ``list_conditions``, ``get_measurements``
* model (mutable): ``get_model``, ``update_model``, ``get_model_structure``,
  ``set_model_parameter``, ``validate_problem``
* parameters (mutable): ``get_parameters``, ``update_parameters``, ``sync_parameters``
* evaluation: ``simulate``, ``evaluate_objective``, ``optimize``
* session: ``reset``

No tool mutates the measurement/condition/observable tables — that invariant is
enforced structurally (there is simply no such tool) and by the model-update
guardrail (see ``validation``/``model``).
"""

from __future__ import annotations

import json
from typing import Any

from mcp.server.fastmcp import FastMCP

from . import model as model_mod
from . import optimize as optimize_mod
from . import parameters as param_mod
from .engine import AmiciEngine
from .errors import PetabMcpError, ValidationError
from .formatting import df_to_records
from .session import PetabSession
from .validation import validate_problem as _validate_problem

SERVER_INSTRUCTIONS = (
    "Interact with a fixed PEtab v1 estimation problem. The experimental design "
    "(measurements, conditions, observables) is read-only; you may edit the model "
    "(Antimony or SBML) and the parameter table, then `simulate`, "
    "`evaluate_objective`, or `optimize` to score the fit against the fixed data. "
    "Use `reset` to revert model/parameter edits."
)


def _amici_available() -> bool:
    try:
        import amici  # noqa: F401
        from amici.importers.petab.v1 import import_petab_problem  # noqa: F401

        return True
    except Exception:
        return False


def create_server(yaml_path: str, build_root: str | None = None) -> FastMCP:
    """Build a FastMCP server bound to the PEtab problem at ``yaml_path``."""
    session = PetabSession(yaml_path)
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
        """Return the current model as text. `format` is 'antimony' or 'sbml'."""
        return model_mod.get_model(session, format)

    @mcp.tool()
    def update_model(text: str, format: str = "antimony") -> dict[str, Any]:
        """Replace the model from Antimony/SBML text.

        Validated against the fixed design; an edit that breaks a referenced
        observable/condition/parameter is rejected (committed=False + issues).
        """
        try:
            result = model_mod.update_model(session, text, format)
        except ValidationError as exc:
            return {"committed": False, "error": str(exc), "issues": exc.issues}
        engine.invalidate()
        return result

    @mcp.tool()
    def get_model_structure() -> dict[str, Any]:
        """Structured view of the model: species, parameters, reactions."""
        return model_mod.model_structure(session)

    @mcp.tool()
    def set_model_parameter(parameter_id: str, value: float) -> dict[str, Any]:
        """Set an SBML model parameter's value in place (recompiles on next eval)."""
        result = model_mod.set_parameter_value(session, parameter_id, value)
        engine.invalidate()
        return result

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

    # --------------------------------------------------------------- session
    @mcp.tool()
    def reset() -> dict[str, Any]:
        """Revert all model and parameter edits to the on-disk problem."""
        session.reset()
        engine.invalidate()
        return {"reset": True, "model_id": session.model.model_id}

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

    # keep references alive / silence linters for closure-registered callables
    _ = (PetabMcpError,)
    return mcp
