"""The MCP server: FastMCP tools and resources over a :class:`PetabSession`.

Tool groups:

* read-only (fixed design): ``describe_problem``, ``list_observables``,
  ``list_conditions``, ``get_measurements``
* model + parameters (the single constrained operation): ``make_changes`` — author the whole
  model in Antimony and/or set parameter bounds/scale, then fit and snapshot a version. Requires
  a justification (references optional). Plus read-only ``get_model``, ``get_model_structure``,
  ``get_parameters``
* fit reads: ``get_fit_summary`` (metrics only), the data (PEtab tables + simulation
  dataframe + residuals + pyPESTO traces) with SQL via ``query_data`` / ``list_data_tables``,
  and the optimizer numerics with ``get_fitting_log`` (the pyPESTO/Fides log)
* provenance/versions/errors: ``get_provenance``, ``get_errors``, ``list_versions``,
  ``get_version``, ``diff_versions``, ``restore_version``, ``get_progress``
* session: ``reset``
* skills (tool-usage documentation, bundled under ``_skills/``): ``list_skills``,
  ``get_skill``

No tool mutates the measurement/condition/observable tables — that invariant is
enforced structurally (there is simply no such tool) and re-checked by the operation
pipeline's lint (see ``operations``/``validation``). Parameter *values* are set only by the fit
``make_changes`` runs; bounds and scale are changed through that same justified call.

``create_server`` wires the session/engine and delegates the (large) tool + resource
surface to the ``_register_*`` groups below, each closing over a small :class:`_Ctx`
that carries the session, engine, and the error-wrapping helpers. The tools stay thin:
the real work lives in the ``session`` / ``operations`` / ``fitting`` / ``data_access`` /
``reconcile`` modules.
"""

from __future__ import annotations

import json
from typing import Any

from fastmcp import FastMCP
from petab.v1.C import OBSERVABLE_ID, SIMULATION_CONDITION_ID

from . import data_access as data_mod
from . import fitting as fitting_mod
from . import identifiability as identifiability_mod
from . import logging_setup
from . import messages as msg_mod
from . import model as model_mod
from . import operations as ops_mod
from . import parameters as param_mod
from . import skills as skills_mod
from .constants import DEFAULT_AUTO_FIT_STARTS
from .engine import AmiciEngine
from .errors import PetabMcpError, ValidationError
from .formatting import df_to_records, summarize_measurements
from .session import PetabSession

# Deliberately SHORT. These instructions are re-sent to the client on every turn, and each
# tool's own description is the canonical contract for its arguments and returns — restating
# them here bought a third copy to keep in sync, not clarity. This states only what a caller
# cannot learn from any single tool's description: the shape of the problem, the invariants
# that span tools, and where the full workflow lives.
SERVER_INSTRUCTIONS = (
    "Interact with a fixed PEtab v1 estimation problem. The experimental design "
    "(measurements, conditions, observables) is READ-ONLY — no tool mutates it. "
    "`make_changes` is the SINGLE editor: give it `antimony` to author the whole model, "
    "`parameters` to set estimation bounds/scale, or both — at least one. It requires a "
    "non-empty justification (references optional, PubMed-validated if given), reconciles the "
    "SBML model with the PEtab parameter table (new rate-law symbols become estimated "
    "parameters, new species get a parametric initial condition, orphans are pruned), then "
    "FITS and snapshots a version. There is no fit-free edit and no separate fit call: one "
    "call is one coherent change, scored once. Parameter VALUES come only from that fit, never "
    "by hand. Everything is logged: audit with `list_versions` / `diff_versions` / "
    "`get_provenance` / `get_errors`. Every fitted version is kept and the run is scored on the "
    "best-bic one regardless, so re-authoring an earlier structure to go back only recompiles, "
    "refits to the same result, and appends a duplicate."
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
        argument — e.g. a whole Antimony model passed to `make_changes` — cannot bloat the
        provenance error log or dump the full model contents into it."""
        if isinstance(value, str) and len(value) > limit:
            return value[:limit] + f"… (+{len(value) - limit} chars)"
        return value

    def run_operation(self, op_name: str, fn, *args) -> dict[str, Any]:
        """Run the edit half of a change; on a known error, log it and return ``applied: False``.

        The fit is a separate step (:meth:`edit_then_fit`) even though the tool always runs both,
        so a rejected EDIT is distinguishable from a failed FIT: the first changed nothing, the
        second leaves the change in place unscored.
        """
        try:
            return fn(*args)
        except PetabMcpError as exc:
            self.log_error(op_name, {"args": [self.short(a) for a in args if isinstance(a, (str, int, float, list))]}, exc)
            self.session.logger.warning("operation %s rejected: %s", op_name, exc)
            out = {"applied": False, "error": str(exc),
                   "messages": msg_mod.errors_from_exception(msg_mod.SOURCE_EDIT, exc)}
            if getattr(exc, "issues", None):
                out["issues"] = exc.issues
            return out
        except Exception as exc:
            self.log_error(op_name, {}, exc)
            self.session.logger.exception("operation %s crashed", op_name)
            raise

    def run_fit(self, fn, args: dict[str, Any] | None = None) -> dict[str, Any]:
        """The fit half of a change: on a known error, log it and return ``committed: False``.

        Unlike an edit, a fit produces a version (``committed: True``, ``version: N``) or a
        de-dup no-op (``committed: False``, ``no_change: True``); a rejected fit — a model that
        can't compile or has no free parameters — returns ``committed: False`` with the message.
        """
        try:
            return fn()
        except PetabMcpError as exc:
            self.log_error("fit", args or {}, exc)
            self.session.logger.warning("fit rejected: %s", exc)
            out = {"committed": False, "error": str(exc),
                   "messages": msg_mod.errors_from_exception(msg_mod.SOURCE_FIT, exc)}
            if getattr(exc, "issues", None):
                out["issues"] = exc.issues
            return out
        except Exception as exc:
            self.log_error("fit", args or {}, exc)
            self.session.logger.exception("fit crashed")
            raise

    def edit_then_fit(
        self, edit_result: dict[str, Any], n_starts: int | None, justification: str,
        before: dict[str, Any],
    ) -> dict[str, Any]:
        """Score a just-applied edit, reverting it if the fit produced no version.

        The fit goes through :meth:`run_fit`, so a rejected fit is logged and shaped as
        ``committed: false`` + ``error``. An UNEXPECTED exception is caught here too rather than
        propagating: ``run_fit`` re-raises anything that is not a :class:`PetabMcpError` (a crash
        in the snapshot / progress / result-block code, which sits outside ``session.fit``'s own
        wrap), and letting that escape would surface the whole call as a failed *edit* when the
        edit already applied. ``run_fit`` has already logged it either way.

        Then the all-or-nothing rule: if no version was created, the edit is UNDONE (``before`` is
        the :func:`operations.working_state` snapshot taken before it applied). A change is either
        scored or gone — an edit left in place unscored is a working model nothing has verified,
        which the next call then builds on, and which no tool can clear now that there is no
        standalone fit.

        The de-dup no-op is not a failure and is not reverted: the resulting model IS an existing
        version, byte-identical to it, so there is nothing unscored to undo.
        """
        # HEAD before the fit, so we can tell whether a version was actually created instead of
        # inferring it from the pending-edit flag: `session.fit` clears the pending edits several
        # statements after it snapshots, and a failure in that window leaves them pending with
        # HEAD already advanced.
        head_before = self.session.provenance.current_version
        try:
            fit_result = self.run_fit(
                lambda: self.session.fit(n_starts=n_starts, justification=justification or None),
                {"n_starts": n_starts},
            )
        except Exception as exc:  # noqa: BLE001 - already logged by run_fit; see the docstring
            crashed = f"the edit applied but the fit crashed unexpectedly: {exc}"
            fit_result = {
                "committed": False,
                "error": crashed,
                "messages": [msg_mod.error(msg_mod.SOURCE_FIT, crashed)],
            }
        snapshotted = self.session.provenance.current_version != head_before
        scored = (
            bool(fit_result.get("committed")) or bool(fit_result.get("no_change")) or snapshotted
        )
        if not scored:
            ops_mod.revert_working_state(
                self.session, before,
                f"the fit produced no version: {fit_result.get('error') or 'no error message'}",
            )
        elif snapshotted and self.session.has_unfit_edits():
            # `session.fit` clears the pending edits several statements AFTER it snapshots, so a
            # failure in that window (an event-log write, say) leaves them pending against a
            # version that already records them. Clear them rather than revert: the change IS
            # scored, and leaving the flag set would make `describe_problem` report an unscored
            # edit that no call could clear.
            ops_mod.clear_scored_edits(self.session, before)
        return ops_mod.bundle_result(self.session, edit_result, fit_result, head_before)

    def reject_operation(
        self, op_name: str, message: str, args: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Log and return an ``applied: False`` rejection for an argument the tool layer itself
        refuses — the same shape :meth:`run_operation` produces for a rejected edit.

        ``args`` are recorded in the persisted error log (truncated by :meth:`short`, as in
        ``run_operation``) so a tool-layer rejection is as debuggable as one raised inside the
        operation: the log shows WHICH argument value was refused, not just the message.
        """
        exc = ops_mod.OperationError(message)
        self.log_error(op_name, {k: self.short(v) for k, v in (args or {}).items()}, exc)
        self.session.logger.warning("operation %s rejected: %s", op_name, exc)
        return {"applied": False, "error": message,
                "messages": [msg_mod.error(msg_mod.SOURCE_TOOL, message)]}

    def check_n_starts(self, op_name: str, n_starts: int | None) -> dict[str, Any] | None:
        """Validate ``n_starts`` BEFORE the edit runs; a rejection dict or None.

        It has to happen up front: a bad value is only rejected inside ``session.fit``, by which
        point the edit has applied — so an argument typo would leave an unscored change behind and
        a message telling the agent to repair a model that is fine. Rejecting here keeps a
        malformed call a pure no-op.
        """
        if n_starts is None:
            return None
        args = {"n_starts": n_starts}
        default = self.session.auto_fit_starts
        try:
            coerced = self.session._coerce_n_starts(n_starts)
        except PetabMcpError as exc:
            return self.reject_operation(
                op_name, f"{op_name}: {exc} (omit it to use the server default of {default}).",
                args)
        # Not an error — a smaller value is a legitimate (cheaper) search — but say so, since
        # `n_starts` REPLACES the default and a caller aiming to "fit harder" with a number below
        # it would quietly get a weaker fit than omitting the argument.
        if coerced is not None and coerced < default:
            self.session.logger.warning(
                "%s: n_starts=%d is BELOW the server default of %d, so this fit is weaker than "
                "omitting n_starts would be (it overrides the default, it does not add to it).",
                op_name, coerced, default)
        return None

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
        """Summarize the problem: model size, tables, and free parameters.

        Also reports `fit_n_starts_default` — the pyPESTO multistart count a fit uses when
        `n_starts` is omitted. Read it before passing `n_starts` anywhere: that argument REPLACES
        this default rather than adding to it, so a value below it fits less hard.
        """
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
        """Structured view of the model: species, parameters, reactions — plus the per-species
        `initial_conditions` account.

        Each initial-condition entry reports whether that species' declared t=0 actually governs
        the simulated state: `t0_source` is `fixed` (the declared value is used), `estimated` (the
        fit sets it — the DEFAULT, so a declared t=0 is only a starting guess) or
        `condition_table` (a condition column overrides it per condition), with `ic_parameter`
        naming the parameter that carries it. Read it to check whether a t=0 you meant to pin
        actually took, and to see which species are driven per-condition — those have no
        parameter-table row at all, so `get_parameters` cannot show them.
        """
        return model_mod.model_structure(session)

    @mcp.tool()
    def make_changes(
        justification: str,
        antimony: str | None = None,
        parameters: list[dict[str, Any]] | None = None,
        references: list[str] | None = None,
        n_starts: int | None = None,
    ) -> dict[str, Any]:
        """Change the model and/or its parameter search space, then FIT and snapshot a version.

        The single editor. Give `antimony` to author the WHOLE model (it builds the first model
        when the problem ships model-less — `describe_problem` → `model_authored: false` — and
        re-authors/replaces an existing one wholesale, so every structural change is one coherent
        re-author). Give `parameters` to set estimation BOUNDS and/or SCALE: a list of
        `{parameter_id, lower_bound?, upper_bound?, scale?}` entries, applied atomically AFTER the
        reconcile — so an entry may target a parameter this very call introduces. Give BOTH for a
        structural change and the bounds it needs. At least one is required.

        There is no fit-free edit and no separate fit: one call is one coherent change, scored
        once, snapshotted as ONE version. `n_starts` REPLACES the server's configured multistart
        count (`describe_problem` → `fit_n_starts_default`) rather than adding to it, so omit it
        unless you want MORE than that; below it fits LESS hard, and the minimum is 1.

        ALL-OR-NOTHING. The change applies to a copy, reconciles SBML↔PEtab, and lints; a rejected
        `parameters` entry reverts the re-author too, and if the change applies but the FIT then
        produces no version the change is REVERTED as well. So the working model is always a
        scored one, and there is never an edit waiting to be fit.
        The result IS the fit, flat — the same field names `get_fit_summary` uses — plus two
        flags. Read `applied`: `false` → the change is NOT in the model, either rejected before it
        applied or reverted because the fit failed; `error`/`issues` says which. Then `committed`:
        `true` → a version exists, read `bic`; `false` with `no_change` → the change applied but
        this structure already IS a version, so change something else.
        Nothing else rides on the result: the reconcile/validation reports and the parameter diff
        are not a score, so read the reconciled model on demand through `get_model_structure`
        (which also carries the per-species initial-condition account) or `get_parameters`.

        `messages` is the call's whole diagnostic stack, in pipeline order: every warning and the
        error that stopped it, each tagged `level` ("error"/"warning") and `source` —
        "make_changes" (an argument refused before anything ran), "edit", "validation" (the
        PEtab/SBML lint), or "fit". Read it to see EVERYTHING the call had to say; `warnings` is
        just its warning texts, and `error` just the last error.

        REJECTS an EMPTY model (no species and no reactions — nothing to simulate or fit; this
        includes an Antimony whose species all prune away as orphans), a re-author that drops a
        shipped-FIXED constant, or brings one back estimated, or leaves a condition-table target
        unused, or wires an observation-model symbol (a scaling factor / noise sigma) into a rate
        law. Warns if a previously-estimated parameter is dropped. Widening an ESTIMATED noise
        sigma past its data-driven cap is advisory (a warning, not a rejection); a THIRD widening
        of a still-pinned parameter is rejected — such a parameter is unidentifiable, not
        under-explored (see `get_identifiability`).

        Every fitted version is kept and the run is scored on the best-bic one regardless, so
        re-authoring an earlier structure to go back only recompiles and refits to the same result.
        """
        if not antimony and not parameters:
            return ctx.reject_operation(
                "make_changes",
                "make_changes: give `antimony` (to author the model), `parameters` (to set "
                "bounds/scale), or both — a call that changes nothing has nothing to score.",
                {"antimony": antimony, "parameters": parameters},
            )
        rejected = ctx.check_n_starts("make_changes", n_starts)
        if rejected is not None:
            return rejected
        # One critical section for the change and its fit, so nothing can fit in between
        # (`session.lock` is an RLock, so the nested acquisitions inside each call are fine).
        with session.lock:
            # Taken BEFORE the edit: if the fit then produces no version, `edit_then_fit` undoes
            # the edit against this, so the call is all-or-nothing.
            before = ops_mod.working_state(session)
            if antimony:
                result = ctx.run_operation(
                    "make_changes", ops_mod.set_model,
                    session, antimony, justification, references, parameters,
                )
            else:
                result = ctx.run_operation(
                    "make_changes", ops_mod.change_parameters,
                    session, parameters, justification, references,
                )
            if not result.get("applied"):
                return result
            return ctx.edit_then_fit(result, n_starts, justification, before)


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

        `in_model` says whether the parameter is currently DECLARED by your model. A row with
        `in_model: false` is one the PROBLEM ships that your model does not use yet — on a
        zero-shot build that is every shipped kinetic / initial-condition parameter, since there
        is no mechanism to reference them. They are shown because they were handed to you: they
        are the constants you may build on, and a shipped-FIXED one must stay declared or the
        change is rejected. Declare the name in your Antimony and the row comes back exactly as
        shown here — its bounds, scale, estimated/fixed status and any prior — so re-introducing
        a shipped parameter never costs you its definition. (`query_data`'s `parameters` table is
        the LIVE table only, so a count there is the real estimation problem.)

        Values (`nominalValue`) are set by the automatic fit on commit; adjust bounds and
        scale through `make_changes` (one batched call: a `parameters` list whose entries
        each carry `lower_bound`/`upper_bound` and/or `scale` for a parameter).
        """
        return param_mod.parameters_to_records(
            session.problem, shipped=session.shipped_parameter_df)


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
        observable's measurement std, on the scale its noise is additive on) regardless of whether
        it's at bound — a sigma approaching that reference is quietly doing the same thing as an
        at-bound parameter (masking a mechanism problem behind inflated noise); `make_changes`
        WARNS (does not reject) when widening a sigma past it. The decisive, ground-truth-free
        evidence that a sigma is absorbing misfit is structured residuals — check
        `get_fit_diagnosis`'s sign/runs tests. Each entry also reports `data_noise_estimate`, a
        MODEL-FREE noise level from successive measurement differences, and
        `fitted_over_data_noise` — purely ADVISORY (nothing gates on them). That estimate is biased
        HIGH (it counts whatever the mean does between samples as noise), so a fitted sigma far
        ABOVE it is strong evidence of inflation, while one below it is only weak evidence of
        collapse. Each entry carries a `status` (`ok`/`inflated`/`collapsed`, or
        `unknown` when the observable's measurement std can't be computed so it can't be assessed), and
        the top level flags `noise_sigma_inflated`/`noise_sigma_collapsed` with `noise_sigma_advice`
        when either is set: an inflated sigma is absorbing model misfit, a collapsed one is
        overfitting the noise — either way the bic is NOT genuine adequacy and a finish on it is
        REFUSED, so fix the model FORM (not the noise) when you see this. Use this to check the
        score without pulling the dataframe;
        use `query_data` to query the measured-vs-simulated points and residuals with SQL.
        Refreshed on every commit; generated on demand if none exists.
        """
        return ctx.run("get_fit_summary", lambda: fitting_mod.fit_summary(session, engine))

    @mcp.tool()
    def get_run_timing() -> dict[str, Any]:
        """Cumulative server-side timing for this run, split by cost kind.

        `{compile_seconds, fit_seconds, n_compiles, n_fits}`. `compile_seconds` is the AMICI
        model-BUILD time (the one-time C++ compile per unique structure — ONE build, shared by
        the fit and every objective evaluation); `fit_seconds` is the pyPESTO multistart
        OPTIMIZATION time. Both
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

        Compute whatever residual breakdown you
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
        `mean_weighted_residual` (bias — far from 0 = a systematic over/under-shoot), worst
        |residual|, and two SIGN-based structure tests — `sign_test` (bias) and `runs_test`
        (time trend / serial correlation), each with a `p` and a `low_power` flag; plus the
        `worst_n` largest-residual points (measurement vs simulation) and a rolled-up
        `residual_structure` verdict naming any observable with significant structure.

        That `residual_structure` verdict is ALREADY on every `make_changes` result — call this tool when
        you need the detail behind it (which points, how big, which observable carries the chi2),
        not to ask whether the residuals are structured.

        The sign/runs tests exist BECAUSE `chi2`/`reduced_chi2`/`chi2_gof_pvalue` are
        uninformative when the noise sigmas are ESTIMATED (a free sigma inflates to absorb
        misfit, so magnitude scores pin near their null however wrong the FORM is). Reading only
        residual SIGNS, these still detect a systematic bias or drift the magnitude scores miss.
        They have low power on few points, though: treat `low_power` non-significant results as
        weak evidence, NOT a pass, and never as a hard finish gate. Use this to locate a poor
        fit cheaply; for a full custom breakdown use `query_data`, for the optimizer log
        `get_fitting_log`. The simulation dataframe is generated on demand if no commit has
        produced one yet.
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


# ------------------------------------------------------ provenance/version tools
def _register_provenance_tools(mcp: FastMCP, ctx: _Ctx) -> None:
    session = ctx.session

    @mcp.tool()
    def get_provenance(limit: int = 50) -> list[dict[str, Any]]:
        """Return the most recent provenance events (edits, fits, rollbacks, resets)."""
        return session.provenance.recent_events(limit)

    @mcp.tool()
    def get_errors(limit: int = 50) -> list[dict[str, Any]]:
        """Return the most recent logged errors (rejected operations / failed tool calls)."""
        return session.provenance.recent_errors(limit)

    @mcp.tool()
    def list_versions() -> list[dict[str, Any]]:
        """List all model versions (fit snapshots) with kind/label/timestamp and fit metrics
        (neg_log_likelihood, chi2, bic, is_new_best) — use it to rank versions by BIC and find
        the best-scoring one. Every version is a fit snapshot (or the `initial`/`rollback`/`reset`
        snapshots); `is_new_best` tracks BIC (gated on a minimum effect size)."""
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

    # NOTE: `rollback_to` and `current_version` are deliberately NOT exposed as tools.
    # `restore_version` IS `rollback_to` (session.restore_version delegates to it) under the name
    # that reads correctly for the going-back workflow, and `describe_problem` already returns
    # `current_version` / `has_unfit_edits` / `n_pending_edits`. Two tools for one operation cost
    # a JSON schema in every client's context on every turn and a second docstring to keep in
    # sync — which is how they drifted apart in the first place. Both session methods remain.
    @mcp.tool()
    def restore_version(version: int, justification: str = "") -> dict[str, Any]:
        """Bring a prior version's model + parameters back as the working state, WITHOUT refitting.

        The cheap, correct way to go back to an earlier model: it restores that version's stored
        fit instead of recompiling and refitting. Prefer this over re-authoring an earlier model
        with `make_changes` — a re-author recompiles, refits, and appends a byte-identical duplicate
        version (which `make_changes` will warn about, then refuse, once it detects the repeat), whereas this
        reuses the existing fit and records a clean restore. Find the target with `list_versions`
        (pick the lowest-bic / is_new_best version), then `restore_version(v)`. Appends a new
        version rather than erasing history, so a restore is itself auditable."""
        return ctx.run("restore_version",
                       lambda: session.restore_version(version, justification or None),
                       {"version": version})

    @mcp.tool()
    def get_progress() -> dict[str, Any]:
        """Early-stop-chain progress for this run — a cheap counter read (no fit/simulation).

        `{n_fitted_ops, non_improving_streak, streak_cap, best_version, best_bic,
        best_gof_pvalue, bic_improvement_over_last_n, plateau_window, should_stop, stop_reason}`.
        `best_version` is the lowest-BIC fitted model committed so far (the checkpoint to score);
        `non_improving_streak` counts consecutive fitted ops since the last real new best.
        `should_stop` is set (with a `stop_reason`) once the run has stopped making progress — the
        hard non-improving-streak cap, or a plateau once the best model is statistically adequate.
        A plateaued `stop_reason` also carries the structural-inadequacy nudge: when the score
        stops moving and the best model is still not adequate, the misfit is the model's FORM — the
        next model must change the structure (add a latent state, add a feedback reaction, raise
        interaction order), not re-fit or widen the same one.
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
        return json.dumps(param_mod.parameters_to_records(
            session.problem, shipped=session.shipped_parameter_df), indent=2)

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
) -> FastMCP:
    """Build a FastMCP server bound to the PEtab problem at ``yaml_path``.

    ``skills_dir`` overrides where tool-usage skills are loaded from; when omitted
    the ``_skills`` directory bundled with the package is discovered automatically.

    ``auto_fit`` (default on) makes every commit fit + simulate before snapshotting;
    ``auto_fit_starts`` is the default pyPESTO multistart count (overridable per commit via
    ``n_starts``). Set ``auto_fit=False`` to skip the compile-heavy fit hook (tests/dev).

    Initial conditions are ESTIMATED by default (fixed only at 0 = empty pool or 1 = unit
    normalization; see ``reconcile.sync_parameter_table``); to keep another IC fixed, ship it as
    a fixed parameter row.
    """
    session = PetabSession(
        yaml_path,
        provenance_dir=provenance_dir,
        reference_mode=reference_mode,
        resolver=resolver,
    )
    engine = AmiciEngine(session, build_root=build_root)
    # Wire the engine so the fit `make_changes` runs can fit + simulate.
    session.engine = engine
    session.auto_fit = auto_fit
    session.auto_fit_starts = auto_fit_starts

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
    _register_provenance_tools(mcp, ctx)
    _register_session_tools(mcp, ctx)
    _register_resources(mcp, ctx)
    # Register the skills loaded above as `list_skills`/`get_skill` tools plus
    # `petab://skills` resources (no-op when no skills were found).
    skills_mod.register_skills(mcp, skills)

    return mcp
