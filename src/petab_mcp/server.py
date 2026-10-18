"""The MCP server: FastMCP tools and resources over a :class:`PetabSession`.

Tool groups:

* read-only (fixed design): ``describe_problem``, ``list_observables``,
  ``list_conditions``, ``list_experiments``, ``get_measurements``
* model + parameters (the single constrained operation): ``make_changes`` — author the whole
  model in Antimony and/or set parameter bounds/scale, then fit and snapshot a version. Requires
  a justification (references optional). Plus read-only ``get_model``, ``get_model_structure``,
  ``get_parameters``
* fit reads: ``get_fit_summary`` (metrics only), the data (PEtab tables + simulation
  dataframe + residuals + pyPESTO traces) with SQL via ``query_data`` / ``list_data_tables``,
  and the optimizer numerics with ``get_fitting_log`` (the pyPESTO/Fides log)
* provenance/versions/errors: ``get_provenance``, ``get_errors``, ``list_versions``,
  ``get_version``, ``diff_versions``, ``restore_version``, ``get_progress``
* skills (tool-usage documentation, bundled under ``_skills/``): ``list_skills``,
  ``get_skill``

No tool mutates the measurement/condition/experiment/observable tables — that invariant is
enforced structurally (there is simply no such tool) and re-checked by the operation
pipeline's lint (see ``operations``/``validation``). Parameter *values* are set only by the fit
``make_changes`` runs; bounds and scale are changed through that same justified call.

``create_server`` opens the session/engine (:func:`_open_ctx`) and delegates the (large) tool +
resource surface to the ``_register_*`` groups below. On EVERY call, never once at
registration, each tool asks its group's ``route`` for the small :class:`_Ctx` that carries the
session, engine, and the error-wrapping helpers. The tools stay thin: the real work lives in the
``session`` / ``operations`` / ``fitting`` / ``data_access`` / ``reconcile`` modules.
"""

from __future__ import annotations

import json
from typing import Annotated, Any, Callable, Literal

from fastmcp import FastMCP
from pydantic import BaseModel, BeforeValidator, ConfigDict, Field
from petab.v2.C import EXPERIMENT_ID, OBSERVABLE_ID

from . import data_access as data_mod
from . import fitting as fitting_mod
from . import identifiability as identifiability_mod
from . import logging_setup
from . import messages as msg_mod
from . import model as model_mod
from . import model_edits as model_edits_mod
from . import operations as ops_mod
from . import parameters as param_mod
from . import regimes as regimes_mod
from . import skills as skills_mod
from . import validation_data as validation_data_mod
from .constants import DEFAULT_AUTO_FIT_STARTS
from .engine import AmiciEngine
from .errors import PetabMcpError, UnknownEntityError, ValidationError
from .formatting import df_to_records, summarize_measurements
from .session import PetabSession

# Deliberately SHORT. These instructions are re-sent to the client on every turn, and each
# tool's own description is the canonical contract for its arguments and returns — restating
# them here bought a third copy to keep in sync, not clarity. This states only what a caller
# cannot learn from any single tool's description: the shape of the problem, the invariants
# that span tools, and where the full workflow lives.
SERVER_INSTRUCTIONS = (
    "Interact with a fixed PEtab v2 estimation problem. The experimental design "
    "(measurements, conditions, experiments, observables) is READ-ONLY — no tool mutates it. "
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


# A TYPED model rather than `dict[str, Any]`, because the JSON schema is the only part of a tool
# contract a model is actually constrained by — a docstring is prose it may or may not read.
# Advertised as a bare dict this argument named NONE of its keys, and across the August benchmark
# runs 527 of 889 `make_changes` calls were rejected for guessing them: `id` for `parameter_id`
# (484 on its own, the single largest rejection class), plus `lowerBound`, `estimate`,
# `nominalValue`, `parameterScale`. Five unrelated model families made the same class of mistake,
# which is the signature of a contract problem rather than a model one.
#
# `extra="forbid"` is load-bearing twice: it puts `additionalProperties: false` in the schema (so
# tool-calling is constrained up front), and it preserves the existing behaviour that a misspelled
# key is an ERROR rather than silently ignored. `_normalize_parameter_changes` still re-checks all
# of it server-side — this type narrows what a model is likely to send, it does not replace
# validation.
#
# Keep the class docstring SHORT and model-facing: FastMCP copies it into the tool's JSON schema
# as this argument's description, so maintainer rationale belongs here in a comment, not there.
class ParameterChange(BaseModel):
    """One parameter's estimation bounds and/or scale."""

    model_config = ConfigDict(extra="forbid")

    parameter_id: str = Field(
        description="id of the parameter to change — an existing row, or one this same call's "
        "`antimony` introduces."
    )
    lower_bound: float | None = Field(
        default=None, description="new lower bound of the estimation interval."
    )
    upper_bound: float | None = Field(
        default=None, description="new upper bound of the estimation interval."
    )
    scale: Literal["lin", "log", "log10"] | None = Field(
        default=None,
        description="the scale the parameter is ESTIMATED on (log10 for a positive quantity "
        "that may span decades). Exactly these three values.",
    )


# The STRUCTURED edit surface: name the change, not the characters. See petab_mcp.model_edits
# for the measured case (the substring surface's failures were 71% one derived line, and 81% of
# the rest were anchors that existed nowhere). Typed for the same reason ParameterChange is: the
# JSON schema is the part of the contract a model is actually held to, and `extra="forbid"`
# makes a misspelled key an error rather than a silently-ignored one.
class ModelOp(BaseModel):
    """One structured change to the model, addressed by reaction/symbol id."""

    model_config = ConfigDict(extra="forbid")

    op: Literal[
        "add_reaction",
        "remove_reaction",
        "set_rate_law",
        "set_initial",
        "set_assignment",
        "remove_assignment",
        "rename",
    ] = Field(description="which change to make.")
    id: str | None = Field(
        default=None,
        description="the reaction id (remove_reaction, set_rate_law) or symbol id "
        "(set_initial, set_assignment, remove_assignment). add_reaction takes its id from "
        "the statement itself; "
        "repeating the same id is accepted, a differing one is refused.",
    )
    antimony: str | None = Field(
        default=None,
        description="the content, written as Antimony: for add_reaction one NAMED statement "
        "('transfer: X1 => X2; k*u*X1;'), for set_rate_law the rate "
        "law ('k*u*X1'), for set_initial the value ('5', or a symbol like "
        "'X1_init' to keep it estimated), for set_assignment the expression the "
        "symbol equals at ALL times ('X2 + X3') — the op behind readout_* "
        "measurement mappings.",
    )
    old: str | None = Field(default=None, description="rename only: the current symbol name.")
    new: str | None = Field(default=None, description="rename only: the replacement name.")


# The MECHANISM route (petab_mcp.mechanisms), typed for the same reason as ModelOp. `statement` is
# left an open object: INDRA's statement JSON is a family of shapes keyed on `type`, and INDRA's
# own reader validates it server-side (operations.statements_to_antimony).
class StatementRow(BaseModel):
    """One mechanism assertion and the mechanism that expresses it."""

    model_config = ConfigDict(extra="forbid")

    statement: dict[str, Any] = Field(
        description="the assertion as an INDRA statement's JSON: its `type` and agents, e.g. "
        "{type: 'Phosphorylation', enz: {name: 'MAP2K1'}, sub: {name: 'MAPK1'}, residue: 'T', "
        "position: '185'}."
    )
    policy: str = Field(description="the mechanism expressing it, e.g. 'indra_default'.")
    prefix: str = Field(
        description="names this row's parameters and rules ('<prefix>_kcat'); unique per row."
    )
    parameters: dict[str, str | float] | None = Field(
        default=None,
        description="binds a parameter this row makes, by its kind ('kf' for '<prefix>_kf'), to "
        "a SHARED id (another row's, a problem parameter's, or a new name several rows use) or "
        "to a NUMBER, which fixes it — one rate constant across many reactions is one parameter.",
    )
    modifiers: dict[str, dict[str, Any] | str] | None = Field(
        default=None,
        description="the agents (INDRA Agent JSON) filling the modifier slots of a written "
        "mechanism's rate: an inhibitor, a regulating complex, an input it reads.",
    )


class StartRow(BaseModel):
    """One species present at the start, and how much of it."""

    model_config = ConfigDict(extra="forbid")

    agent: dict[str, Any] | str = Field(
        description="the species in the state it starts in: INDRA Agent JSON or the notation, "
        "e.g. 'H4' or 'EGFR{-pY1068}'; a complex through its bound partners ('RAF{!RAS}')."
    )
    amount: float | str = Field(
        description="its starting amount: a number (1 is a fixed unit pool, 0 an empty start, "
        "anything else estimated from there) or an expression over parameter ids (a parameter "
        "the problem ships, a condition-table column, a new name)."
    )
    name: str | None = Field(
        default=None,
        description="names its parameter `<name>_init`; defaults to the agent's name, and must "
        "differ for a second start of the same agent.",
    )


class MechanismDef(BaseModel):
    """A mechanism you write when none in the library fits: its own reactions (`scheme`), or an
    existing mechanism's reaction with your own rate (`rate`)."""

    model_config = ConfigDict(extra="forbid")

    statement_type: str = Field(description="the statement type it expresses, e.g. 'Conversion'.")
    policy: str = Field(description="its name, new; `statements` rows then use it as `policy`.")
    based_on: str = Field(
        default="indra_default",
        description="the existing mechanism whose reaction (what is consumed, what is made) "
        "this keeps; only the rate is yours.",
    )
    scheme: str | None = Field(
        default=None,
        description="the mechanism's own reactions: one PySB rule per line, `reactants >> "
        "products @ rate` (`|` and two rates for both ways), molecules named by the statement "
        "type's ROLES (`enz`, `sub`, `subj`, `obj`, `members_0`...) and their sites `mod` (the "
        "statement's modification, 'off'/'on'), `act` ('inactive'/'active') or a partner role's "
        "name (the binding site for it): `enz(sub=None) + sub(enz=None, mod='off') >> "
        "enz(sub=1) % sub(enz=1, mod='off') @ kf`. `list_mechanisms` shows every library entry "
        "this way: start from one. A rate is a parameter's name or an expression over parameters, "
        "`amount(<pattern>)`, `time` and exp/log/sqrt. Give `scheme` or `rate`, not both.",
    )
    rate: str | None = Field(
        default=None,
        description="(instead of `scheme`) the rate COEFFICIENT, over the statement type's role names (each the amount "
        "of that agent: `enz`, `sub`, `subj`, `source`, ...), your `parameters`, `time`, and "
        "exp/log/sqrt. It is multiplied by every reactant's amount, catalysts included, so a "
        "rate that stays non-negative cannot drive an amount negative: `kcat / (Km + sub)` is "
        "Michaelis-Menten."
    )
    parameters: list[str] = Field(
        default_factory=list,
        description="the parameter names `rate` uses; each becomes `<prefix>_<name>`, estimated "
        "unless a row's `parameters` shares or fixes it.",
    )
    modifiers: list[str] = Field(
        default_factory=list,
        description="names in `rate` for the amount of an agent that takes no part in the "
        "reaction (an inhibitor, a regulator); each row using this mechanism names the agent.",
    )


class StatementEdit(BaseModel):
    """One change to a model built from statements: a row, a start, an amount or a readout."""

    model_config = ConfigDict(extra="forbid")

    op: Literal[
        "add_row",
        "replace_row",
        "remove_row",
        "set_start",
        "remove_start",
        "set_amount",
        "remove_amount",
        "set_readout",
    ] = Field(
        description="`add_row` a new row; `replace_row` the row with that row's prefix (its "
        "statement, policy or bindings); `remove_row` by `prefix`; `set_start` adds or replaces "
        "the start of that name; `set_amount`/`set_readout` add or replace one; "
        "`remove_start`/`remove_amount` by `name`. A readout is never removed: each is one an "
        "observable reads."
    )
    row: StatementRow | str | None = Field(
        default=None, description="add_row/replace_row: the row, one line or JSON."
    )
    prefix: str | None = Field(default=None, description="remove_row: the row's prefix.")
    start: StartRow | None = Field(default=None, description="set_start: the start.")
    name: str | None = Field(
        default=None, description="remove_start, set_amount, remove_amount: its name."
    )
    agent: dict[str, Any] | str | None = Field(
        default=None, description="set_amount: the agent pattern whose amount it is."
    )
    readout: str | None = Field(
        default=None, description="set_readout: the readout id."
    )
    formula: str | None = Field(default=None, description="set_readout: its formula.")


# Provider serialisation artefact: some models emit a list-typed argument wrapped in a
# single-key object -- `{"parameters": {"item": [{...}, {...}]}}` instead of
# `{"parameters": [{...}, {...}]}` -- or as a JSON string containing the array. pydantic then
# rejects the whole call with `Input should be a valid list [type=list_type]`, and because the
# wrapper is how that model serialises lists, the retry is byte-identical. In the Aug 8-10 sweep
# this produced 28 client-side ToolExceptions that KILLED 3 runs outright (0 fits, 0 versions) and
# recursion-limited a 4th; the escalation guard told the model "REPEATING THIS CALL WILL FAIL
# AGAIN" three times and the wrapper came back each time. Prompting cannot fix a serialiser, so
# accept the shape and unwrap it. Applied as a BeforeValidator so the JSON schema the model is
# shown still says `array` -- we are tolerating a malformed encoding, not documenting it as valid.
_LIST_WRAPPER_KEYS = ("item", "items", "value", "values", "list")


def _unwrap_listish(v: Any) -> Any:
    """Coerce the known malformed encodings of a JSON array back to a list; pass anything else
    through untouched so real validation errors still surface as themselves."""
    if isinstance(v, dict) and len(v) == 1:
        ((key, inner),) = v.items()
        if key in _LIST_WRAPPER_KEYS and isinstance(inner, list):
            return inner
    if isinstance(v, str):
        s = v.strip()
        if s.startswith("[") and s.endswith("]"):
            try:
                parsed = json.loads(s)
            except ValueError:
                return v
            if isinstance(parsed, list):
                return parsed
    return v


def _parameter_entries(parameters: Any) -> list[dict[str, Any]] | None:
    """Normalize the tool argument to the plain dicts ``operations`` expects.

    FastMCP hands back :class:`ParameterChange` instances, but tolerate dicts too so a direct
    in-process caller (the tests, another tool) is not forced to construct the model.
    ``exclude_none`` keeps "field omitted" and "field sent as null" indistinguishable, which is
    what the downstream validator already assumes.
    """
    if parameters is None:
        return None
    return [
        p.model_dump(exclude_none=True) if isinstance(p, ParameterChange) else p for p in parameters
    ]


def _shorten(value: Any, limit: int = 200, _depth: int = 0) -> Any:
    """Recursive body of :meth:`_Ctx.short` — module-level so it can call itself.

    (As a bare ``@staticmethod`` the recursive call resolved to a free name that does not exist
    at class scope, so every container argument raised NameError from inside the error logger.)
    """
    if isinstance(value, str):
        return (value[:limit] + f"… (+{len(value) - limit} chars)") if len(value) > limit else value
    if _depth >= 3:  # containers this deep are not worth walking for a log line
        return "…"
    if isinstance(value, (list, tuple)):
        head = [_shorten(v, limit, _depth + 1) for v in value[:5]]
        return head + [f"… (+{len(value) - 5} more)"] if len(value) > 5 else head
    if isinstance(value, dict):
        out = {k: _shorten(v, limit, _depth + 1) for k, v in list(value.items())[:10]}
        if len(value) > 10:
            out["…"] = f"(+{len(value) - 10} more keys)"
        return out
    return value


# Over this many ids the list is truncated. The ceiling is real — lucarelli's 118-row parameter
# table, re-sent in the caller's history every turn thereafter, was a measured top-3 context cost
# — but it sits where a normal problem's table fits WHOLE. The previous cap of 12 elided the
# answer from the message whose entire job was to give it, and runs re-guessed: one burned six
# get_parameters calls in the 2026-08-17 sweep, another four.
_ID_LIST_CAP = model_edits_mod._ID_LIST_CAP


def unknown_ids_message(unknown: list, known_ids: list[str]) -> str:
    """The `get_parameters` unknown-id error text: what was wrong, what you probably meant, and
    what is valid. Built here rather than inline so it can be tested on the string a caller
    actually receives — the first version of this hint was assembled into a local variable and
    never interpolated, and shipped green because the tests around it called the scorer directly
    and grepped the module source.
    """
    hints = []
    for u in unknown:
        close = model_edits_mod.closest_symbols(str(u), known_ids)
        if close:
            hints.append(f"{u} -> {' or '.join(repr(c) for c in close)}")
    suggest = f" Did you mean: {'; '.join(hints)}." if hints else ""
    known = ", ".join(known_ids[:_ID_LIST_CAP])
    more = (
        ""
        if len(known_ids) <= _ID_LIST_CAP
        else f", (+{len(known_ids) - _ID_LIST_CAP} more — call get_parameters() with no "
        f"parameter_ids for the whole table)"
    )
    return f"get_parameters: unknown parameter_ids {unknown}.{suggest} Known ids: [{known}{more}]."


class _Ctx:
    """One session's state + the error-wrapping helpers a tool resolves through ``route``.

    Bundles the :class:`PetabSession`, its :class:`AmiciEngine` and its held-out set (``None``
    without one) with the two dispatch wrappers that turn a raised :class:`PetabMcpError` into a
    structured tool result (and log it), so every ``_register_*`` group builds its tools against
    the same helpers.
    """

    def __init__(
        self,
        session: PetabSession,
        engine: AmiciEngine,
        held_out: validation_data_mod.HeldOut | None = None,
    ) -> None:
        self.session = session
        self.engine = engine
        self.held_out = held_out

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
    def short(value: Any, limit: int = 200, _depth: int = 0) -> Any:
        """Truncate long strings before they go into the (persisted) error log, so a large
        argument — e.g. a whole Antimony model passed to `make_changes` — cannot bloat the
        provenance error log or dump the full model contents into it.

        RECURSES into lists and dicts, because the bulky arguments on this server are mostly
        containers: `parameters` is up to ~130 entries on a large problem. Truncating only top-level strings let those through whole,
        which mattered little while the log was ephemeral and matters now that `archive_run`
        copies errors.jsonl into every run's permanent archive. The element cap is separate from
        the string cap: a list is summarised, not just its members shortened.
        """
        return _shorten(value, limit, _depth)

    def run_operation(self, op_name: str, fn, *args) -> dict[str, Any]:
        """Run the edit half of a change; on a known error, log it and return ``applied: False``.

        The fit is a separate step (:meth:`edit_then_fit`) even though the tool always runs both,
        so a rejected EDIT is distinguishable from a failed FIT: the first changed nothing, the
        second leaves the change in place unscored.
        """
        try:
            return fn(*args)
        except PetabMcpError as exc:
            self.log_error(
                op_name,
                {"args": [self.short(a) for a in args if isinstance(a, (str, int, float, list))]},
                exc,
            )
            self.session.logger.warning("operation %s rejected: %s", op_name, exc)
            out = {
                "applied": False,
                "error": str(exc),
                "messages": msg_mod.errors_from_exception(msg_mod.SOURCE_EDIT, exc),
            }
            if getattr(exc, "issues", None):
                out["issues"] = exc.issues
            # Carry a machine-readable rejection class when the raiser set one, exactly as
            # `reject_operation` does. Without this, a labelled error raised INSIDE the commit
            # pipeline is silently downgraded to the prose classifier the label exists to
            # replace — and the docstring's promise ("when a raiser sets this, the classifier
            # uses it verbatim") would be false for every raiser below the tool layer.
            if getattr(exc, "cause", None):
                out["error_cause"] = exc.cause
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
            out = {
                "committed": False,
                "error": str(exc),
                "messages": msg_mod.errors_from_exception(msg_mod.SOURCE_FIT, exc),
            }
            if getattr(exc, "issues", None):
                out["issues"] = exc.issues
            return out
        except Exception as exc:
            self.log_error("fit", args or {}, exc)
            self.session.logger.exception("fit crashed")
            raise

    def edit_then_fit(
        self,
        edit_result: dict[str, Any],
        n_starts: int | None,
        justification: str,
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
        scored or gone, so the working model is never a state nothing has verified.

        The de-dup no-op is not a failure and is not reverted: the resulting model IS an existing
        version, byte-identical to it, so there is nothing unscored to undo.
        """
        # HEAD before the fit — the only signal for "was a version created" that survives a
        # failure between the snapshot and the end of `session.fit`.
        head_before = self.session.provenance.current_version
        try:
            fit_result = self.run_fit(
                lambda: self.session.fit(
                    n_starts=n_starts,
                    justification=justification or None,
                    staged_edits=[edit_result["staged_edit"]]
                    if edit_result.get("staged_edit")
                    else None,
                ),
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
                self.session,
                before,
                f"the fit produced no version: {fit_result.get('error') or 'no error message'}",
            )
        return ops_mod.bundle_result(self.session, edit_result, fit_result, head_before)

    def reject_operation(
        self,
        op_name: str,
        message: str,
        args: dict[str, Any] | None = None,
        cause: str | None = None,
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
        out = {
            "applied": False,
            "error": message,
            "messages": [msg_mod.error(msg_mod.SOURCE_TOOL, message)],
        }
        if cause:
            # Machine-readable rejection class for the harness's `edit_rejects.by_cause`. The
            # alternative — substring-matching the English message — put three buckets on an
            # unreachable branch in one day, because a needle shared with an earlier entry
            # steals every match. See operations.OperationError.cause.
            out["error_cause"] = cause
        return out

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
                op_name, f"{op_name}: {exc} (omit it to use the server default of {default}).", args
            )
        # Not an error — a smaller value is a legitimate (cheaper) search — but say so, since
        # `n_starts` REPLACES the default and a caller aiming to "fit harder" with a number below
        # it would quietly get a weaker fit than omitting the argument.
        if coerced is not None and coerced < default:
            self.session.logger.warning(
                "%s: n_starts=%d is BELOW the server default of %d, so this fit is weaker than "
                "omitting n_starts would be (it overrides the default, it does not add to it).",
                op_name,
                coerced,
                default,
            )
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


# MCP tool annotations: hints a client may use to auto-approve reads and to confirm writes. A
# "read-only" tool may still log a failed call or cache the on-demand simulation; neither changes
# what any later call returns. The writers are NOT destructive: each appends a version and
# erases nothing (see ``provenance``), and neither is idempotent, since a repeat appends again.
_READ_ONLY = {"readOnlyHint": True, "idempotentHint": True, "openWorldHint": False}
#: ``make_changes`` reaches PubMed when given ``references``, hence open-world.
_MAKE_CHANGES_ANNOTATIONS = {
    "readOnlyHint": False,
    "destructiveHint": False,
    "idempotentHint": False,
    "openWorldHint": True,
}
_RESTORE_ANNOTATIONS = {**_MAKE_CHANGES_ANNOTATIONS, "openWorldHint": False}


# ------------------------------------------------------------------- read tools
def _register_read_tools(mcp: FastMCP, route: Callable[[], _Ctx]) -> None:
    @mcp.tool(annotations=_READ_ONLY)
    def describe_problem() -> dict[str, Any]:
        """Summarize the problem: model size, tables, and free parameters.

        Also reports `fit_n_starts_default` — the pyPESTO multistart count a fit uses when
        `n_starts` is omitted. Read it before passing `n_starts` anywhere: that argument REPLACES
        this default rather than adding to it, so a value below it fits less hard.
        """
        ctx = route()
        session = ctx.session
        return session.describe()

    @mcp.tool(annotations=_READ_ONLY)
    def list_observables() -> list[dict[str, Any]]:
        """List the (fixed) observables and their formulas/noise models.

        A formula symbol named `readout_<name>` is a MEASUREMENT MAPPING you must define in your
        model with an assignment rule (`readout_x := <expression of your species>`, or the
        `set_assignment` edit op): the problem tells you WHAT was measured — `observableName`
        carries the biological description — and your model says what that quantity IS in terms
        of its own states. Several observables may reference the SAME readout (one per
        dataset/gel, differing only in shipped scalings); one assignment rule covers them all.
        Scaling/offset parameters and the noise model are fixed parts of the assay; only the
        readout mapping is yours to define.
        """
        ctx = route()
        session = ctx.session
        return df_to_records(session.problem.observable_df)

    @mcp.tool(annotations=_READ_ONLY)
    def list_conditions() -> list[dict[str, Any]]:
        """List the (fixed) conditions: one row per change a condition makes — it sets
        `targetId` (a model parameter, species or compartment) to `targetValue`. When a condition
        applies is the experiments' business (`list_experiments`)."""
        ctx = route()
        session = ctx.session
        return df_to_records(session.problem.condition_df)

    @mcp.tool(annotations=_READ_ONLY)
    def list_experiments() -> list[dict[str, Any]]:
        """List the (fixed) experiments: one row per period — from `time` on, experiment
        `experimentId` applies condition `conditionId` (none: the model as is). `time` -inf is a
        pre-equilibration to steady state before the experiment starts. Every measurement names
        the experiment it was taken in; one with no experiment simulates the model as is."""
        ctx = route()
        session = ctx.session
        return df_to_records(session.problem.experiment_df)

    @mcp.tool(annotations=_READ_ONLY)
    def get_measurements(
        experiment_id: str | None = None,
        observable_id: str | None = None,
        summarize: bool = True,
        max_rows: int = 200,
        offset: int = 0,
    ) -> "list[dict[str, Any]] | dict[str, Any]":
        """Return the (fixed) measurement data, optionally filtered by experiment/observable.

        By default (``summarize=True``) returns a COMPACT SUMMARY — per observable×experiment:
        point count, time span, value min/max/mean, and a few preview points — so the data's
        shape is visible without every raw point. Prefer it, and prefer ``query_data`` for a
        targeted slice; the raw dump is rarely the right tool.

        ``summarize=False`` returns raw per-point records, PAGED:
        ``{rows, row_count, total_rows, offset, truncated, next_offset?}``. Paging is not
        cosmetic — an unpaged dump measured 73k tokens on isensee and 81k on lucarelli, in ONE
        tool result that then re-enters the context on every later turn; one such call has
        ended a run with the provider's "prompt exceeds max length". Page with ``offset`` (feed
        back ``next_offset``) or, better, narrow with ``experiment_id``/``observable_id`` first.
        Row order is the measurement table's own and is stable across pages.
        """
        ctx = route()
        session = ctx.session

        def run() -> "list[dict[str, Any]] | dict[str, Any]":
            # Validated on BOTH paths, before the summarize branch: a nonsense max_rows/offset
            # must be reported whichever mode the caller asked for, rather than being silently
            # accepted whenever summarize happens to be true.
            n_rows, off = data_mod.validate_page_args(max_rows, offset)
            df = session.problem.measurement_df
            if df is not None:
                if experiment_id is not None:
                    df = df[df[EXPERIMENT_ID] == experiment_id]
                if observable_id is not None:
                    df = df[df[OBSERVABLE_ID] == observable_id]
            if summarize:
                return summarize_measurements(df)
            records = df_to_records(df)
            total = len(records)
            page = records[off : off + n_rows]
            out: dict[str, Any] = {
                "rows": page,
                "row_count": len(page),
                "total_rows": total,
                "offset": off,
                "truncated": off + len(page) < total,
            }
            if out["truncated"]:
                out["next_offset"] = off + len(page)
            return out

        # Through ctx.run like every other validated tool: a bad max_rows/offset must come back
        # as the logged {ok: false, error} shape, not as a raw exception. An unwrapped raise is
        # invisible to get_errors AND counts as a tool error against the circuit breaker, so
        # three argument typos in a row would disable get_measurements for the rest of the run.
        return ctx.run(
            "get_measurements",
            run,
            {
                "experiment_id": experiment_id,
                "observable_id": observable_id,
                "summarize": summarize,
                "max_rows": max_rows,
                "offset": offset,
            },
        )


# ``make_changes``'s tool description, composed from the summary, the model ROUTES and the
# CONTRACT the call keeps whichever route is taken.
_MAKE_CHANGES_SUMMARY = (
    "Change the model and/or its parameter search space, then FIT and snapshot a version."
)

_MAKE_CHANGES_ROUTES_BODY = """\
The single editor, with TWO routes to a model change — give exactly ONE.

`model_edits` is the route to use for every change after the first: a BATCH of
structured operations addressed by ID, not by text. Each entry is one of
`{op: 'add_reaction', antimony: 'transfer: X1 => X2; k*u*X1'}` (the id comes
from the statement; repeating it as `id` is fine, a DIFFERENT `id` is refused),
`{op: 'remove_reaction', id}`, `{op: 'set_rate_law', id, antimony: '<rate law>'}`,
`{op: 'set_initial', id, antimony: '5'}`, `{op: 'set_assignment', id, antimony:
'X2 + X3'}` (an `id := expr` rule holding at ALL times — how a `readout_*` symbol
referenced by an observable formula is defined in terms of YOUR model),
`{op: 'remove_assignment', id}`, `{op: 'rename', old, new}`. Send the WHOLE coherent
change in ONE call — a new mechanism is usually a new reaction plus a rate-law change on
the reaction it feeds — because one call is one fit and one version; splitting it across
calls pays a multistart per operation. Operations apply IN ORDER and the batch is
ALL-OR-NOTHING: a failure names the failing `model_edits[i]`, lists the valid reaction
ids when the id is wrong, and applies nothing. They edit the model SOURCE (what you last
authored or edited), so there is no text to quote and nothing to keep in sync with the
reconcile's re-render.

`antimony` authors the WHOLE model: it builds the first model when the problem ships
model-less (`describe_problem` → `model_authored: false`) and replaces an existing one
wholesale — the right route for the first model and for genuinely starting over.
Anything the operations cannot express — an assignment rule, an event, a compartment —
is a wholesale re-author too.

`base_version` (with `model_edits`) applies the operations to a PRIOR version's
STORED source instead of the current one — the rollback-and-edit path: never retype an earlier
model from memory (retyped restores measured 6-37,000 BIC worse than the stored
original). Nothing to quote: the operations name reaction ids.
Default: the current committed model. It ALSO restores that version's SEARCH SPACE — its
bounds and scales — AND the point the fit starts from, that version's own best fit, so the
fork resumes from where version N got to rather than from the branch you are abandoning.
You never retype any of it. A parameter your operations newly introduce keeps the default
the reconcile mints, and any parameter you name in this call's `parameters` keeps YOUR
bound and scale rather than the restored one.

Give `parameters` to set estimation BOUNDS and/or SCALE: a list of
`{parameter_id, lower_bound?, upper_bound?, scale?}` entries, applied atomically AFTER the
reconcile — so an entry may target a parameter this very call introduces. Combine it with
either model route for a structural change and the bounds it needs. At least one of
"""

_MAKE_CHANGES_ROUTES = (
    _MAKE_CHANGES_ROUTES_BODY + "`antimony` / `model_edits` / `parameters` is required."
)

#: The same text where the `statements` route is offered too.
_MAKE_CHANGES_STATEMENTS_ROUTES = (
    _MAKE_CHANGES_ROUTES_BODY
    + "`antimony` / `model_edits` / `statements` / `statement_edits` / `parameters` is required."
)

_MAKE_CHANGES_CONTRACT = """\
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
`bic` charges `k` = `n_estimated` + `hardcoded_constants`: a numeric literal in your
model's math is a parameter too, bar a few structural ones, so freezing a fitted value
costs what estimating it costs — estimate it. What you pay for is listed.
Nothing else rides on the result: the reconcile/validation reports and the parameter diff
are not a score, so read the reconciled model on demand through `get_model_structure`
(which also carries the per-species initial-condition account) or `get_parameters`.

`messages` is the call's whole diagnostic stack, in pipeline order: every warning and the
error that stopped it, each tagged `level` ("error"/"warning") and `source` —
"make_changes" (an argument refused before anything ran), "edit", "validation" (the
PEtab/SBML lint), or "fit". Read it to see EVERYTHING the call had to say; `error` alone is
the one rejection that stopped it, and `issues` that rejection's structured detail.

REJECTS an EMPTY model (no species and no reactions — nothing to simulate or fit; this
includes an Antimony whose species all prune away as orphans), a re-author that drops a
shipped-FIXED constant, or brings one back estimated, or leaves a condition-table target
unused, or wires an observation-model symbol (a scaling factor / noise sigma) into a rate
law. Warns if a previously-estimated parameter is dropped. Widening an ESTIMATED noise
sigma past its data-driven cap is advisory (a warning, not a rejection); a THIRD widening
of a still-pinned parameter is rejected — such a parameter is unidentifiable, not
under-explored (see `get_fit_diagnosis`).

Every fitted version is kept and the run is scored on the best-bic one regardless, so
re-authoring an earlier structure to go back only recompiles and refits to the same result."""

_MAKE_CHANGES_DOC = "\n\n".join(
    (_MAKE_CHANGES_SUMMARY, _MAKE_CHANGES_ROUTES, _MAKE_CHANGES_CONTRACT)
)


def _make_changes_statements_doc(bound: bool = True, held: dict[str, str] | None = None) -> str:
    """The `statements` route's paragraph, naming the entries the library holds right now.

    ``bound`` says whether the problem's entity map binds species or readouts. One that does not
    binds nothing of the model: the readouts, and how the condition targets enter, are the
    caller's own, and saying the map binds them would send it looking for names that do not exist.
    ``held`` (agent name to protocol) are the inputs such a map does hold, named with their
    protocol (Fujita's EGF ramp)."""
    from . import mechanisms

    from indra.statements import AddModification, RemoveModification, Statement, get_all_descendants

    families = {
        base.__name__: {c.__name__ for c in get_all_descendants(base)}
        for base in (AddModification, RemoveModification)
    }
    defaulted = {cls for cls, policy in mechanisms.REGISTRY if policy == mechanisms.INDRA_DEFAULT}
    without = sorted({c.__name__ for c in get_all_descendants(Statement)} - defaulted)
    by_policy: dict[str, set[str]] = {}
    for cls, policy in sorted(mechanisms.REGISTRY):
        if policy != mechanisms.INDRA_DEFAULT:
            by_policy.setdefault(policy, set()).add(cls)

    def named(classes: set[str]) -> str:
        # A whole family reads as its base (every AddModification), not two dozen types.
        names = []
        for base, members in families.items():
            if members <= classes:
                classes = classes - members
                names.append(f"every {base}")
        return ", ".join(names + sorted(classes - {"Modification"}))

    entries = "; ".join(f"`{policy}` for {named(c)}" for policy, c in sorted(by_policy.items()))
    return (
        "`statements` is a third model route, and still exactly ONE route per call: it authors "
        "the WHOLE model from mechanism assertions rather than text, each an INDRA statement "
        "with the `policy` (mechanism) that turns it into reactions and a `prefix` naming its "
        "parameters. A row may be one line instead of JSON: `r1: EGFR{pY1068} phosphorylates "
        "AKT1 at S473 | two_step kf=k_on`, agents as NAME{pY1068, -pS473, @nucleus, !BOUND, "
        "~UNBOUND} (see `get_model(format='statements')` for a model written that way)."
        "  Every INDRA statement type"
        + (f" but {', '.join(without)}" if without else "")
        + f" has `{mechanisms.INDRA_DEFAULT}`; the library "
        f"also holds {entries}. `list_mechanisms` shows each as the PySB rules it builds. When "
        "none fits, WRITE one in `mechanisms`, either as a `scheme` (its own rules, written like "
        "those) or as a `rate` on an existing entry's reaction, whose rate may read `modifiers`, "
        "agents each row names that take no part in the reaction (an inhibitor, a regulator); "
        "it stays in the library for the rest of the run. A row's `parameters` shares a rate constant with other "
        "rows, or with a parameter the problem ships, or fixes it to a number. "
        + (
            "The species the observables read, and the inputs the experiments set (a dose), are "
            "bound by the problem's entity map: name agents as it does, and do not assert an "
            "input — it is already held to its protocol. "
            if bound
            else "NOTHING is bound: every readout the observables read is yours to define in "
            "`readouts`, and every target the condition table sets is a parameter id your rows "
            "must read, in a binding (`kf=\"k*dose\"`) or a start's amount. "
            + "".join(
                f"But `{name}` is HELD to the experiment's protocol, `{name} = {formula}`: read "
                "it as an agent in your rows; never start, make or degrade it. "
                for name, formula in sorted((held or {}).items())
            )
        )
        + "`starts` declares what is present at the start, in the state its agent writes; "
        "without it every species no rule makes starts at an estimated `<name>_init` and every "
        "made one empty. `readouts` defines a measurement mapping as a formula over `amounts` "
        "(agents whose amounts it reads, or `amount(<pattern>)` in place; every readout reads "
        "at least one)"
        + (", entity-map ids" if bound else "")
        + " and parameter ids.\n\n"
        "`statement_edits` CHANGES the current rows instead of giving them all again: "
        "`add_row`/`replace_row` a row (one line; `replace_row` takes the row with that prefix, "
        "so a new policy or binding is the row rewritten), `remove_row` by prefix, and "
        "`set_start`/`set_amount`/`set_readout` or `remove_start`/`remove_amount`; with them "
        "`mechanisms` you write for the new rows. They apply in "
        "order, all or nothing, and the model is rebuilt from the edited set. `base_version=N` "
        "applies them to version N's rows instead (with `statements`, it is the whole set given "
        "afresh), and starts the fit from version N's search space and best fit. "
        "`get_model(format='statements')` shows the rows. The result is an ordinary model; "
        "`model_edits` would edit it as text, after which it has no rows to edit."
    )


def _mechanisms_text(session, statement_type: str | None) -> str:
    """The library, and the run's written mechanisms, each as its scheme (:mod:`schemes`)."""
    from . import mechanisms, schemes
    from .errors import MechanismError

    library = dict(mechanisms.REGISTRY)
    for key, definition in session.mechanisms.items():
        library[key] = mechanisms.written(definition, library)
    keys = sorted(k for k in library if statement_type in (None, k[0]) and k[0] not in schemes.ABSTRACT)
    if not keys:
        known = sorted({t for t, _ in library} - schemes.ABSTRACT)
        return f"no mechanism for {statement_type!r}; the statement types are {known}"
    blocks = []
    for stmt_type, policy in keys:
        written = " (written this run)" if (stmt_type, policy) in session.mechanisms else ""
        try:
            text = schemes.describe(stmt_type, policy, library)
        except MechanismError as exc:
            text = f"({exc})"
        blocks.append(f"{stmt_type} | {policy}{written}\n" + "\n".join(f"  {line}" for line in text.splitlines()))
    return "\n".join(blocks)


def _statement_record(session, version: int | None) -> dict | None:
    """The statement record of the current model, or of a stored ``version``; ``None`` for a model
    not built from statements."""
    import json

    if version is None:
        return session.model_statements or None
    path = session.provenance.versions_dir / f"v{int(version)}" / "statements.json"
    return (json.loads(path.read_text()) or None) if path.is_file() else None


def _statements_text(session, version: int | None) -> str:
    """A model built from statements, as the notation (:mod:`notation`): its written mechanisms,
    its rows one per line (INDRA JSON where the notation has no lossless form), its starts,
    amounts and readouts. The current model, or a stored ``version``."""
    import json

    from . import mechanisms, notation

    record = _statement_record(session, version)
    if not record:
        return "(this model was not built from statements)"
    library = dict(mechanisms.REGISTRY)
    lines, written = [], {}
    for row in record["rows"]:
        if "mechanism" in row:
            d = row["mechanism"]
            written[(d["statement_type"], d["policy"])] = d
    for d in written.values():
        library[(d["statement_type"], d["policy"])] = None
        if d.get("scheme"):
            lines.append(f"mechanism {d['policy']}: {d['statement_type']}, scheme")
            lines += [f"  {line.strip()}" for line in d["scheme"].strip().splitlines() if line.strip()]
            continue
        reads = f"  reads {', '.join(d['modifiers'])}" if d.get("modifiers") else ""
        lines.append(
            f"mechanism {d['policy']}: {d['statement_type']} like "
            f"{d.get('based_on', mechanisms.INDRA_DEFAULT)}, rate {d['rate']}{reads}"
        )
    for row in record["rows"]:
        bare = {k: row[k] for k in ("statement", "policy", "prefix", "parameters", "modifiers")
                if row.get(k)}
        lines.append(notation.render(bare, library) or json.dumps(bare))
    for start in record.get("starts") or []:
        lines.append(f"start {start['name']}: {notation.agent_text(start['agent'])} = "
                     f"{start['amount']}")
    for name, agent in (record.get("amounts") or {}).items():
        lines.append(f"amount {name} = {notation.agent_text(agent)}")
    for readout, formula in (record.get("readouts") or {}).items():
        lines.append(f"readout {readout} = {formula}")
    return "\n".join(lines)


# ------------------------------------------------------------------ model tools
def _register_model_tools(
    mcp: FastMCP, route: Callable[[], _Ctx], statements: bool = False
) -> None:
    # `make_changes` takes an argument named `statements`; the route flag goes by this name in it.
    statements_route = statements
    if statements:

        @mcp.tool(annotations=_READ_ONLY)
        def list_mechanisms(statement_type: str | None = None) -> str:
            """Every mechanism a `statements` row can use, as the PySB rules it builds: the
            library's own and those written this run, for one `statement_type` or all.

            Each is written over the statement type's ROLES, the way a `scheme` in
            `make_changes(mechanisms=...)` is written, so a mechanism you need is one of these
            edited: `mod` is the statement's own modification ('off'/'on'), `act` an activity
            ('inactive'/'active'), a site named after a role the binding site for it; `@` gives
            the rate (mass action over the reactants; a name is a parameter a row binds as
            `kf=...`).
            """
            return _mechanisms_text(route().session, statement_type)

    @mcp.tool(annotations=_READ_ONLY)
    def get_model(format: str = "antimony", version: int | None = None) -> str:
        """Return a model as text — the CURRENT one, or a prior `version`'s stored source.

        `format` is 'antimony' (canonical), 'sbml', or 'statements' — a model built from
        statements, one line per row in the notation `make_changes` also accepts. `version` reads that version's stored
        model TEXT and nothing else — a plain read of a prior version, and much cheaper than
        `get_version`, which also returns that version's per-point simulation and full
        optimization trace. On a problem with a few hundred measurements that payload is
        large, and it stays in the caller's history afterwards.

        NOTE the trailing `const ...;` line: it is a DERIVED declaration the server regenerates.
        Antimony treats a parameter as constant unless declared `var`, so its contents carry no
        meaning — read past it. No operation addresses it.
        """
        ctx = route()
        session = ctx.session
        if (format or "").lower() == "statements":
            return _statements_text(session, version)
        if version is None:
            # The RECONCILED model — what is actually fitted, and what `test_make_changes` pins
            # as identical to the stored version ("the working model is always a scored one").
            # It need not equal the source that `model_edits` applies to: operations address
            # reactions by ID, and ids survive the re-render, so nothing here is quoted. What
            # differs is the reconcile's own additions, which is exactly what a reader wants to
            # see here rather than in the source.
            text = model_mod.get_model(session, format)
        else:
            if (format or "antimony").lower() != "antimony":
                # RECORDED, for the same reason the unknown-version rejection below is: this tool
                # is registered without `ctx.run`, so a rejection raised straight out of here
                # burns the caller's turns while showing zero errors in `get_errors` and in
                # result.json's error block.
                exc = ValidationError(
                    "get_model: a stored `version` is only available as antimony (SBML is "
                    "derived from it); omit `format` or pass 'antimony'."
                )
                ctx.log_error("get_model", {"version": version, "format": format}, exc)
                raise exc from None
            # The version's own SOURCE when it has one, because this read exists to be paired
            # with `base_version=N`: the operations apply to that stored source, and showing the
            # reconciled render instead would answer a different question than the one the next
            # call acts on (the render carries reconcile-minted declarations the source lacks).
            # Falls back to the render for versions stored before sources were kept.
            # An absent version is a BAD ARGUMENT, not a missing source: `get_model_source`
            # raises for it, so the `or` fallback below can only ever cover a version that exists
            # without a stored source. Say which versions there are, and RECORD it — this tool is
            # registered without `ctx.run`, so a rejection raised straight out of here would burn
            # the caller's turns while showing zero errors in `get_errors` and in result.json's
            # error block. The prompt actively tells the lead to make this call before a
            # `base_version=N` fork, so a wrong N is a reachable, ordinary mistake.
            try:
                text = session.provenance.get_model_source(
                    int(version)
                ) or session.provenance.get_model_antimony(int(version))
            except UnknownEntityError:
                have = [m.get("version") for m in session.provenance.list_versions()]
                exc = ValidationError(
                    f"get_model: no version {int(version)}. Existing versions: "
                    f"{', '.join(str(v) for v in have if v is not None) or '(none yet)'}."
                )
                ctx.log_error("get_model", {"version": version, "format": format}, exc)
                raise exc from None
        return text

    @mcp.tool(annotations=_READ_ONLY)
    def get_model_structure() -> dict[str, Any]:
        """Structured view of the model: species, parameters, reactions — plus the per-species
        `initial_conditions` account.

        Each initial-condition entry reports whether that species' declared t=0 actually governs
        the simulated state: `t0_source` is `fixed` (the declared value is used), `estimated` (the
        fit sets it — the DEFAULT, so a declared t=0 is only a starting guess) or
        `condition_table` (a condition target overrides it per condition), with `ic_parameter`
        naming the parameter that carries it. Read it to check whether a t=0 you meant to pin
        actually took, and to see which species are driven per-condition — those have no
        parameter-table row at all, so `get_parameters` cannot show them.
        """
        ctx = route()
        session = ctx.session
        return model_mod.model_structure(session)

    def make_changes(
        justification: str,
        antimony: str | None = None,
        model_edits: Annotated[list[ModelOp] | None, BeforeValidator(_unwrap_listish)] = None,
        base_version: int | None = None,
        parameters: Annotated[
            list[ParameterChange] | None, BeforeValidator(_unwrap_listish)
        ] = None,
        references: Annotated[list[str] | None, BeforeValidator(_unwrap_listish)] = None,
        n_starts: int | None = None,
        statements: Annotated[
            list[StatementRow | str] | None, BeforeValidator(_unwrap_listish)
        ] = None,
        mechanisms: Annotated[list[MechanismDef] | None, BeforeValidator(_unwrap_listish)] = None,
        starts: Annotated[list[StartRow] | None, BeforeValidator(_unwrap_listish)] = None,
        readouts: dict[str, str] | None = None,
        amounts: dict[str, dict[str, Any] | str] | None = None,
        statement_edits: Annotated[
            list[StatementEdit] | None, BeforeValidator(_unwrap_listish)
        ] = None,
    ) -> dict[str, Any]:
        ctx = route()
        session = ctx.session
        # To plain dicts before anything else, so every path below (including the rejection
        # payload) sees the same shape the operations layer does. ModelOp is tolerated as plain
        # dicts too, for the same reason _parameter_entries does it: an in-process caller (the
        # tests, another tool) should not have to construct the pydantic model.
        parameters = _parameter_entries(parameters)
        ops = [
            o.model_dump(exclude_none=True) if isinstance(o, ModelOp) else o
            for o in (model_edits or [])
        ] or None
        rows = [
            r.model_dump() if isinstance(r, StatementRow) else r for r in (statements or [])
        ] or None
        edits = [
            e.model_dump(exclude_none=True) if isinstance(e, StatementEdit) else e
            for e in (statement_edits or [])
        ] or None
        routes_given = [
            n
            for n, v in (
                ("antimony", antimony),
                ("model_edits", ops),
                ("statements", rows),
                ("statement_edits", edits),
            )
            if v
        ]
        if len(routes_given) > 1:
            return ctx.reject_operation(
                "make_changes",
                f"make_changes: give ONE model route, not {len(routes_given)} "
                f"({', '.join('`' + r + '`' for r in routes_given)}). "
                + (
                    "`statement_edits` for a change to the current rows, `statements` to give "
                    "the whole model as rows. "
                    if statements_route
                    else "`model_edits` for a structured change to an existing model, "
                    "`antimony` to author one wholesale. "
                )
                + "NOTHING was applied.",
                {"routes": routes_given},
                cause="route_conflict",
            )
        if base_version is not None and not (ops or rows or edits):
            # Names NO other tool. The obvious advice here — "to re-fit it unchanged use
            # restore_version" — would send the LEAD to a tool it does not hold (see
            # agent/tools.py: restore_version is deliberately withheld), which is the
            # defection-trigger class the sibling rejections below are written to avoid, and
            # re-fitting an existing structure is a no-op the de-dup refuses anyway.
            return ctx.reject_operation(
                "make_changes",
                "make_changes: `base_version` selects the version whose STORED model the "
                "edits apply to — it does nothing on its own. Send the change you want as "
                + (
                    "`statement_edits` (or the whole revised `statements`)"
                    if statements_route
                    else "`model_edits`"
                )
                + "; a prior version re-committed unchanged is already a version "
                "and would only re-fit to the same result. If you wanted to RETURN to that "
                "version because it was better: you do not need to — every fitted version is "
                "kept and the run is scored on the BEST one, wherever HEAD ends up.",
                {"base_version": base_version},
            )
        if not antimony and not ops and not rows and not edits and not parameters:
            return ctx.reject_operation(
                "make_changes",
                "make_changes: nothing to do — give "
                + (
                    "`statements` (the whole model as rows), `statement_edits` (changes to the "
                    "current rows, starts, amounts and readouts), "
                    if statements_route
                    else "`model_edits` (structured operations on the committed model — the "
                    "usual route), `antimony` (author the model wholesale), "
                )
                + "`parameters` (to set bounds/scale), or a combination.",
                {"antimony": None, "parameters": None},
            )
        rejected = ctx.check_n_starts("make_changes", n_starts)
        if rejected is not None:
            return rejected
        # WHOLESALE authoring must arrive fully addressable, because the very next change will be
        # addressed by id. An unnamed reaction is legal Antimony and unaddressable by construction:
        # `set_rate_law` cannot name it, so the only way to touch it again is another whole-model
        # re-author — the cost `model_edits` exists to remove. Refused HERE, where it is written, in
        # preference to auto-naming, which would invent ids the caller never chose and renumber
        # them on the next re-author.
        if antimony:
            unnamed = model_edits_mod.unnamed_reactions(antimony)
            if unnamed:
                shown = ", ".join(repr(u) for u in unnamed[:3])
                return ctx.reject_operation(
                    "make_changes",
                    f"make_changes: {len(unnamed)} reaction(s) have no id ({shown}"
                    f"{', …' if len(unnamed) > 3 else ''}). Every reaction needs one — it is how "
                    "`model_edits` addresses it later, and an unnamed reaction can only ever be "
                    "changed by re-authoring the whole model. Write 'name: A => B; k*A;' and "
                    "choose names you will recognise (transfer, decay), not r1/r2. "
                    "NOTHING was applied.",
                    {"unnamed": unnamed[:5]},
                    cause="unnamed_reaction",
                )
        statement_record, statement_warnings, written = None, [], None
        if (starts or readouts or amounts) and edits:
            return ctx.reject_operation(
                "make_changes",
                "make_changes: with `statement_edits`, change starts, amounts and readouts by "
                "their own edits (`set_start`, `set_amount`, `set_readout`, `remove_start`, "
                "`remove_amount`), not "
                "`starts`/`amounts`/`readouts`, which give the whole set with `statements`. "
                "NOTHING was applied.",
                {"n_starts": len(starts or [])},
                cause="statement_edit",
            )
        if (mechanisms or starts or readouts or amounts) and not (rows or edits):
            return ctx.reject_operation(
                "make_changes",
                "make_changes: `mechanisms`, `starts`, `readouts` and `amounts` belong to "
                "`statements` rows; send them together"
                + (" (or `mechanisms` with `statement_edits`)" if statements_route else "")
                + ". NOTHING was applied.",
                {"n_mechanisms": len(mechanisms or []), "n_starts": len(starts or [])},
            )
        # One critical section for the change and its fit, so nothing can fit in between
        # (`session.lock` is an RLock, so the nested acquisitions inside each call are fine).
        with session.lock:
            # Inside the lock: the build reads HEAD's statements record and the run's written
            # mechanisms, and the commit below replaces both, so a concurrent call must not
            # build in between.
            edit_summary: list[str] = []
            if edits:
                # The edits apply to a stored record, HEAD's or `base_version`'s, and give the whole
                # set the `statements` route builds from: the model is rebuilt, never patched, so a
                # version built from edits is the same as one given whole.
                base = _statement_record(session, base_version)
                if base is None:
                    return ctx.reject_operation(
                        "make_changes",
                        "make_changes: `statement_edits` change a model built from statements, and "
                        + (
                            f"version {base_version} was not built from statements"
                            if base_version is not None
                            else "the current model was not: give the whole model as `statements`"
                        )
                        + ". NOTHING was applied.",
                        {"base_version": base_version},
                        cause="statement_edit",
                    )
                try:
                    rows, starts, readouts, amounts, edit_summary = ops_mod.apply_statement_edits(
                        base, edits
                    )
                except PetabMcpError as exc:
                    return ctx.reject_operation(
                        "make_changes",
                        str(exc),
                        {"n_edits": len(edits)},
                        cause=getattr(exc, "cause", None),
                    )
            if rows:
                definitions = [
                    m.model_dump() if isinstance(m, MechanismDef) else m for m in (mechanisms or [])
                ]
                try:
                    antimony, statement_record, statement_warnings, written = (
                        ops_mod.statements_to_antimony(
                            session,
                            rows,
                            definitions,
                            [s.model_dump() if isinstance(s, StartRow) else s for s in starts or []],
                            readouts,
                            amounts,
                        )
                    )
                except PetabMcpError as exc:
                    return ctx.reject_operation(
                        "make_changes",
                        str(exc),
                        {"n_statements": len(rows)},
                        cause=getattr(exc, "cause", None),
                    )
            op_summary: list[str] = []
            base_parameter_df = None
            if ops:
                # STRUCTURED edits apply to the model SOURCE — the text the caller last authored
                # or last edited — not to the reconciled re-render. That is the whole point: no
                # anchor to quote, so nothing to keep in sync with a text the server rewrites,
                # and `child.source == parent.source + ops` makes lineage checkable.
                #
                # Resolved INSIDE the lock so the source cannot move between read and edit. A
                # base_version fork takes THAT version's source; falling back to the rendered
                # model covers versions written before sources were stored, and the very first
                # edit of a shipped (not agent-authored) model.
                try:
                    if base_version is not None:
                        base_src = session.provenance.get_model_source(int(base_version))
                        if base_src is None:
                            base_src = session.provenance.get_model_antimony(int(base_version))
                    else:
                        base_src = session.model_source or model_mod.get_model(session, "antimony")
                    antimony, op_summary = model_edits_mod.apply_model_edits(base_src, ops)
                    # Read INSIDE the guard, next to the source read above: a version dir whose
                    # parameters.tsv is truncated raises EmptyDataError and the legacy json path
                    # raises FileNotFoundError, and evaluating this as a bare call argument let
                    # both escape make_changes as an opaque tool error with nothing in
                    # errors.jsonl — the invisible-failure class get_model's handler exists to stop.
                    if base_version is not None:
                        base_parameter_df = session.provenance.get_parameter_df(int(base_version))
                except (PetabMcpError, OSError, ValueError, TypeError) as exc:
                    msg = str(exc)
                    if not msg.startswith("make_changes"):
                        msg = f"make_changes: {msg}"
                    return ctx.reject_operation(
                        "make_changes",
                        msg,
                        {"n_ops": len(ops), "base_version": base_version},
                        cause=getattr(exc, "cause", None),
                    )
            if base_version is not None and not ops:
                # The statements route: the rows are this call's (given whole, or `base_version`'s
                # record edited), and the version's SEARCH SPACE and fitted point come with them,
                # as they do for `model_edits`.
                try:
                    base_parameter_df = session.provenance.get_parameter_df(int(base_version))
                except (PetabMcpError, OSError, ValueError, TypeError) as exc:
                    return ctx.reject_operation(
                        "make_changes",
                        f"make_changes: version {base_version}: {exc}",
                        {"base_version": base_version},
                    )
            before = ops_mod.working_state(session)
            # LINEAGE: a `base_version` commit descends from THAT version, not from HEAD (see
            # session.derived_from). Set for the duration of this call only, and cleared in the
            # `finally` so a later ordinary commit cannot inherit a stale fork parent.
            session.derived_from = int(base_version) if base_version is not None else None
            # PROVENANCE: the text SUBMITTED to the pipeline — the wholesale `antimony`, or
            # the result of applying this call's operations to the base source. Stored beside
            # the committed model as model_source.ant (see session.submitted_source), and read
            # back by `base_version` forks and by `get_model(version=N)`. On BOTH routes this is
            # the agent's own text: operations apply to the source, never to the re-render, so
            # `child.source == parent.source + ops` holds exactly.
            session.submitted_source = antimony
            session.submitted_statements = statement_record
            try:
                if antimony:
                    result = ctx.run_operation(
                        "make_changes",
                        ops_mod.set_model,
                        session,
                        antimony,
                        justification,
                        references,
                        parameters,
                        # A `base_version=N` fork restores that version's SEARCH SPACE too, not
                        # just its model: before this, a fork inherited whatever the abandoned
                        # branch had widened and the prompt had to ask the agent to retype the old
                        # bounds. Tolerant of a changed symbol set, and applied BEFORE this call's
                        # own `parameters` entries so an explicit bound still wins (see
                        # operations._composed_parameter_edit).
                        #
                        # POSITIONAL, not a keyword: `run_operation` is `(op_name, fn, *args)` with
                        # no **kwargs, so a keyword here raises TypeError on EVERY model-changing
                        # call — outside both except paths, so it escapes make_changes entirely and
                        # the run cannot edit the model at all.
                        base_parameter_df,
                    )
                else:
                    result = ctx.run_operation(
                        "make_changes",
                        ops_mod.change_parameters,
                        session,
                        parameters,
                        justification,
                        references,
                    )
                if not result.get("applied"):
                    return result
                result["messages"] = statement_warnings + (result.get("messages") or [])
                out = ctx.edit_then_fit(result, n_starts, justification, before)
            finally:
                session.derived_from = None
                session.submitted_source = None
                session.submitted_statements = None
            if out.get("applied") and antimony:
                # The submitted text becomes this branch's source for the next structured edit.
                # Keyed on APPLIED, not committed: a de-duplicated commit (`applied: true,
                # committed: false, no_change`) still moves the WORKING model, and leaving the
                # source behind would make the next batch of operations silently revert it.
                # A rejected or fit-reverted call leaves the model where it was, and `applied`
                # is false there, so the source correctly stays put.
                session.model_source = antimony
                session.model_statements = statement_record
                if written is not None:
                    session.mechanisms = written
            if edit_summary:
                out["statement_edits_applied"] = edit_summary
            if op_summary:
                # The diff, PRE-CLASSIFIED. Every earlier analysis had to reconstruct this from
                # re-rendered text with id- and regex-keyed heuristics, and four of those
                # reported artifacts as findings; here the operation IS the classification.
                out["model_edits_applied"] = op_summary
            return out

    if statements:
        mcp.tool(
            description="\n\n".join(
                (
                    _MAKE_CHANGES_SUMMARY,
                    _MAKE_CHANGES_STATEMENTS_ROUTES,
                    _make_changes_statements_doc(
                        bound=bool(set(route().session.entities) - set(route().session.inputs)),
                        held={
                            route().session.entities[k].name: v
                            for k, v in route().session.inputs.items()
                        },
                    ),
                    _MAKE_CHANGES_CONTRACT,
                )
            ),
            annotations=_MAKE_CHANGES_ANNOTATIONS,
        )(make_changes)
        return

    # No entity map: nothing binds a statement-built model to the observables, so the route would
    # fail every call. Withheld at the schema like every other route, which also leaves this
    # registration byte-identical to the one before the route existed.
    def make_changes_text(
        justification: str,
        antimony: str | None = None,
        model_edits: Annotated[list[ModelOp] | None, BeforeValidator(_unwrap_listish)] = None,
        base_version: int | None = None,
        parameters: Annotated[
            list[ParameterChange] | None, BeforeValidator(_unwrap_listish)
        ] = None,
        references: Annotated[list[str] | None, BeforeValidator(_unwrap_listish)] = None,
        n_starts: int | None = None,
    ) -> dict[str, Any]:
        return make_changes(
            justification=justification,
            antimony=antimony,
            model_edits=model_edits,
            base_version=base_version,
            parameters=parameters,
            references=references,
            n_starts=n_starts,
        )

    mcp.tool(
        name="make_changes",
        description=_MAKE_CHANGES_DOC,
        annotations=_MAKE_CHANGES_ANNOTATIONS,
    )(make_changes_text)


# -------------------------------------------------------------- parameter tools
def _register_parameter_tools(mcp: FastMCP, route: Callable[[], _Ctx]) -> None:
    @mcp.tool(annotations=_READ_ONLY)
    def get_parameters(
        parameter_ids: Annotated[list[str] | None, BeforeValidator(_unwrap_listish)] = None,
        role: Literal[
            "kinetic",
            "noise_sigma",
            "observable_scaling",
            "initial_condition",
            "exponent",
            "constant",
        ]
        | None = None,
    ) -> list[dict[str, Any]]:
        """Return the parameter table (id, bounds, scale, nominal_value, estimate), each row
        annotated with its `role` and `status`. `parameter_ids` and/or `role` FILTER the rows —
        on large problems (100+ rows, re-sent in your history every later turn) ask for the rows
        you need (e.g. the parameters a fit just changed, or role='noise_sigma') rather than the
        whole table by reflex.

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
        shown here — its bounds, scale and estimated/fixed status — so re-introducing
        a shipped parameter never costs you its definition. (`query_data`'s `parameters` table is
        the LIVE table only, so a count there is the real estimation problem.)

        Values (`nominalValue`) are set by the automatic fit on commit; adjust bounds and
        scale through `make_changes` (one batched call: a `parameters` list whose entries
        each carry `lower_bound`/`upper_bound` and/or `scale` for a parameter).
        """
        ctx = route()
        session = ctx.session
        rows = param_mod.parameters_to_records(
            session.problem, shipped=session.shipped_parameter_df
        )
        # Server-side filters, added when the 118-row lucarelli table (re-sent in the caller's
        # history every turn thereafter) turned out to be a top-3 context cost. Filtering here
        # rather than telling the model to ignore rows: unread tokens still bill.
        if parameter_ids:
            want = set(parameter_ids)
            # An unknown id must not read as "that parameter does not exist": [] is
            # indistinguishable from a real empty table, and the agent acts on the wrong belief.
            unknown = sorted(want - {r.get("parameterId") for r in rows})
            if unknown:
                known_ids = sorted(str(r.get("parameterId")) for r in rows)
                # Name the near-misses, and do not truncate the id list down to uselessness —
                # see `unknown_ids_message`, which owns both rules and is tested on its output.
                # RAISED, not returned. This tool's declared return type is a LIST, so FastMCP
                # validates its structured output against an array schema and a `ctx.run`
                # {ok:false,...} dict fails that validation with a RuntimeError that KILLS THE
                # RUN — observed 6/14 runs in the 2026-08-16 sneyd/fujita sweep, which had to be
                # cancelled. A raise is the shape a list-returning tool can express: FastMCP
                # turns it into a tool error and langchain-mcp-adapters into a ToolMessage the
                # model reads and corrects, exactly like any other rejected argument.
                exc = ValidationError(unknown_ids_message(unknown, known_ids))
                # RECORD it before raising. ctx.run would have done this; raising past it meant
                # a run that burned turns on bad ids showed ZERO errors in get_errors and in
                # result.json's error block — invisible in the post-mortem artefact.
                ctx.log_error("get_parameters", {"parameter_ids": parameter_ids}, exc)
                raise exc
            rows = [r for r in rows if r.get("parameterId") in want]
        if role:
            # `role` is a Literal, so a typo is rejected by the schema before arriving here.
            rows = [r for r in rows if r.get("role") == role]
        return rows


#: What each ``get_fit_diagnosis`` ``sections`` value asks for. Two halves of one diagnosis, and the
#: CHEAP one is the default: the parameter half is several kB on a k=30 model plus a FIM evaluation,
#: and the payload of every tool call is re-sent in the agent's history on every later turn (PR #23
#: cut ~40% of the per-call prompt by removing exactly that kind of duplication). It is one argument
#: away, and the prompt says to ask for it when BIC stalls — which is when it has something to say.
_DIAGNOSIS_SECTIONS = {
    "residuals": frozenset({"residuals"}),
    "parameters": frozenset({"parameters"}),
    "all": frozenset({"residuals", "parameters"}),
}


def _diagnosis_sections(sections: Any) -> frozenset[str]:
    """The requested ``get_fit_diagnosis`` sections, or a :class:`ValidationError` naming the
    choices — a typo must not silently return the default and hide half the diagnosis."""
    key = sections.strip().lower() if isinstance(sections, str) else sections
    try:
        return _DIAGNOSIS_SECTIONS[key]
    except (KeyError, TypeError):
        raise ValidationError(
            f"sections must be one of {sorted(_DIAGNOSIS_SECTIONS)}, got {sections!r}."
        ) from None


# ------------------------------------------------------- fit + simulation tools
def _register_fit_tools(mcp: FastMCP, route: Callable[[], _Ctx]) -> None:
    @mcp.tool(annotations=_READ_ONLY)
    def get_fit_summary() -> dict[str, Any]:
        """Return just the current fit metrics — no per-point records (a cheap read).

        `{status, neg_log_likelihood, chi2, n_data_points, fitted_parameters_changed,
        parameter_scales, version}` plus the complexity-penalised scores `{bic, aic, aicc, caic,
        hqc, tic, criterion, criterion_value, chi2_dof, reduced_chi2, chi2_gof_pvalue}` (lower is
        better). `criterion` names the penalty this run ranks by (`bic` unless configured
        otherwise) and `criterion_value` is that score — the one to rank models by; the named
        scores are always reported at their textbook weight whichever is active. The `k` they
        charge rides with its two addends: `n_estimated` (the estimated parameter rows) and
        `hardcoded_constants` (the literals being charged, one parameter each — see
        `make_changes`).
        `reduced_chi2`/`chi2_gof_pvalue` are only meaningful with a KNOWN noise level: when the
        observable noise sigmas are ESTIMATED, chi2 pins near `n_data_points` by construction, so
        reduced_chi2 ≈ 1 and its p-value are uninformative — do not use them as a fit-adequacy
        gate (judge adequacy from residual STRUCTURE instead — the `residual_structure` verdict
        rides on this result, as it does on every fit result; see
        :func:`data_access.residual_structure_rollup` for where it is and is not reported).
        Fitted values are LINEAR
        (unscaled) values; `parameter_scales` gives each one's estimation scale (lin/log10) —
        a fitted value is the linear number, never its log10 coordinate. `parameters_at_bound`
        lists any fitted parameter sitting within ~1% of its lower/upper bound (on its own
        estimation scale) — `[]` when none do. Widen at most once; the SAME parameter
        re-pinning after a widening means it is UNIDENTIFIABLE (a flat/collinear FIM direction —
        confirm with `get_fit_diagnosis`), not under-explored, so the optimum is unbounded
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
        evidence that a sigma is absorbing misfit is structured residuals — read the
        `residual_structure` verdict on THIS result (`get_fit_diagnosis` carries the per-observable
        table and worst points behind it). Each entry also reports `data_noise_estimate`, a
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
        ctx = route()
        session = ctx.session
        engine = ctx.engine
        return ctx.run("get_fit_summary", lambda: fitting_mod.fit_summary(session, engine))

    @mcp.tool(annotations=_READ_ONLY)
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
        ctx = route()
        session = ctx.session

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

    @mcp.tool(annotations=_READ_ONLY)
    def get_data_regimes(observables: str | None = None) -> dict[str, Any]:
        """What the MEASUREMENTS require of any model — computed from the data alone, no model, no
        fit, no simulation. Valid BEFORE the first model exists.

        SIZE, since it is re-sent in your history every later turn: 1-9 kB on the shipped examples,
        because at most 12 observables are reported. Pass `observables="id1,id2"` to scope the call
        to the ones you are actually reasoning about — on a problem with a hundred observables an
        unscoped call is mostly noise you will keep paying for.

        Per observable, each verdict measured against that observable's own noise floor (estimated
        from replicates where the design has them, else successive differences — the method is in
        the payload):

        - `shape` — does the response RISE AND FALL under a held input? A pulse cannot be produced
          by constant production plus first-order decay at ANY rate constants, so it REQUIRES a
          depletable pool, a second state, or feedback.
        - `timescales` — is a second relaxation timescale distinguishable, by BIC.
          `delta_bic_favouring_two` is POSITIVE when the two-timescale form fits better.
        - `dose_response` — the steepest local log-log slope against a numeric condition input,
          with its uncertainty. Above 1 rules out a single mass-action binding step.
        - `conservation` — is the sum of the observables constant in time? Imposing the total is
          a constraint rather than an addition: it REMOVES a parameter.

        Each entry carries a `requirement` (non-null only when ESTABLISHED — then any model must
        reproduce it) and `support_for_extension`, where `suggestive` means the estimate leans
        toward the richer structure without establishing it, which is a reason to TRY it.

        THESE FINDINGS CAN FORCE STRUCTURE IN; THEY NEVER RULE IT OUT. `testable: false` means the
        DESIGN cannot answer; a null `requirement` means this design did not settle it. Neither is
        permission to simplify — no evidence for a second timescale is not evidence that one
        suffices. `conservation` is the sole entry that genuinely constrains.
        """
        ctx = route()
        session = ctx.session
        return ctx.run(
            "get_data_regimes",
            lambda: regimes_mod.data_regimes(session, observables),
            {"observables": observables},
        )

    @mcp.tool(annotations=_READ_ONLY)
    def list_data_tables() -> dict[str, Any]:
        """List the SQL tables `query_data` exposes: name, description, columns, row count.

        The tables are the PEtab design tables (`measurements`, `conditions`, `experiments`,
        `observables`), the `parameters` table, the `simulations` dataframe (one row per
        measurement point, carrying `simulation` and the residuals), `simulations_history`
        (those same rows for EVERY stored version, with a `version` column), and the last
        fit's pyPESTO multistart (`optimization_starts`, `optimization_trace`). Use it to
        learn the exact column names before writing a `query_data` query.
        """
        ctx = route()
        session = ctx.session
        return ctx.run("list_data_tables", lambda: data_mod.table_schema(session))

    @mcp.tool(annotations=_READ_ONLY)
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
        ctx = route()
        session = ctx.session

        def run() -> dict[str, Any]:
            if not isinstance(tail_lines, int) or tail_lines < 0:
                raise ValidationError(f"tail_lines must be an integer >= 0, got {tail_lines!r}.")
            return logging_setup.read_pypesto_log(session.provenance.root, tail_lines)

        return ctx.run("get_fitting_log", run, {"tail_lines": tail_lines})

    @mcp.tool(annotations=_READ_ONLY)
    def query_data(sql: str, max_rows: int = 50, offset: int = 0) -> dict[str, Any]:
        """Run a SQL query (DuckDB) over the PEtab tables and the simulation dataframe.

        The query runs against a private, disposable copy of the data, so any SQL is allowed
        and nothing can affect the session (host file/network access is disabled). Tables:
        `measurements`, `conditions`, `experiments`, `observables`, `parameters`; `simulations` —
        the per-point dataframe from the last commit, with `measurement`, `simulation`, `residual`
        (natural units), `residual_weighted` (the noise-normalized chi2 term), and `noise`
        (sigma); `simulations_history` — those same per-point rows for EVERY stored version
        with a `version` column (see the CROSS-VERSION paragraph below); and the last fit's
        pyPESTO multistart in `optimization_starts` (one row per
        start, `start`=0 is best) and `optimization_trace` (fval per start/iteration). Call
        `list_data_tables()` for the exact columns. Returns
        `{columns, rows, row_count, offset, truncated}` (capped at `max_rows`, default 50 to
        keep results compact — prefer aggregations over a full dump); the
        `simulations` dataframe is generated on demand if no commit has produced one yet.

        PAGING: when `truncated` is true the result also carries `next_offset` — pass it back
        as `offset` for the next page rather than raising `max_rows` (a large page is a large
        tool result, and it re-enters your context on EVERY later turn). Add an explicit
        `ORDER BY` when you page: without one, SQL row order is not guaranteed stable between
        calls, so pages can overlap or skip rows. No total row count is reported — obtaining
        one would mean running your query a second time.

        TWO NAMING CONVENTIONS, and they do not mix. The PEtab tables keep PEtab's camelCase:
        `measurements(observableId, experimentId, time, measurement, observableParameters,
        noiseParameters)`, `conditions(conditionId, targetId, targetValue)`,
        `experiments(experimentId, time, conditionId)`, `observables(observableId, ...)`,
        `parameters(parameterId, ...)`. The DERIVED tables use snake_case:
        `simulations(observable_id, experiment_id, time, ...)`, `simulations_history` (the same
        plus `version`), and `optimization_starts` / `optimization_trace`. The measurement
        table's key column is `experimentId` — NOT `experiment_id` (that is `simulations`); an
        empty `experimentId` ('') is a measurement of the model simulated as is. Do not carry a
        column list from a `simulations` query onto `measurements`, or the reverse.

        ONE ROW PER GROUP: use DuckDB's aggregates, do not hand-roll them. In a GROUP BY query
        every selected expression must be an aggregate or a grouping key — a window function
        (`... OVER (...)`) is per-row and cannot appear there, and aggregates cannot nest.
        `ORDER BY` goes INSIDE the aggregate's parentheses (`first(measurement ORDER BY time)`);
        `FILTER` goes OUTSIDE, after the closing paren (`count(*) FILTER (WHERE time > 0)`).
        Neither may be hung on a scalar function like `log10`.
          value at earliest / latest time  `arg_min(measurement, time)` / `arg_max(measurement, time)`
          peak and the time it occurs      `max(measurement)` / `arg_max(time, measurement)`
          trend                            `regr_slope(measurement, time)`, `corr(time, measurement)`
          spread                           `stddev_samp` / `var_samp` (there is no `var`)
          `SELECT observableId, experimentId, count(*) AS n,
                  arg_min(measurement, time) AS y_at_t0, arg_max(measurement, time) AS y_at_tend,
                  max(measurement) AS y_peak, arg_max(time, measurement) AS t_peak,
                  regr_slope(measurement, time) AS slope
             FROM measurements GROUP BY observableId, experimentId`

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

        CROSS-VERSION comparison: `simulations_history` holds the same per-point rows for
        every stored version, with a `version` column (snake_case like `simulations`; a
        version whose fit failed contributes no rows). Where did v4 get worse than v3:
          `SELECT observable_id, experiment_id, time,
                  max(CASE WHEN version=3 THEN residual_weighted END) AS v3,
                  max(CASE WHEN version=4 THEN residual_weighted END) AS v4
             FROM simulations_history WHERE version IN (3, 4)
             GROUP BY 1, 2, 3 ORDER BY abs(v4) - abs(v3) DESC LIMIT 10`
        """
        ctx = route()
        session = ctx.session

        def run() -> dict[str, Any]:
            # Reject bad arguments cheaply before generating a (compile-heavy) simulation, and
            # USE what validation returns: it clamps max_rows to MAX_PAGE_ROWS, so discarding
            # the normalized pair would let the boundary check and the value actually queried
            # disagree.
            n_rows, off = data_mod.validate_query_args(sql, max_rows, offset)
            # Populate the stored simulation dataframe on demand so `simulations` is
            # queryable (a no-op once a commit has produced one). A simulation failure must
            # not block querying the static tables, so fall through with an empty table.
            # Generate the HEAD simulation dataframe only when the query actually reads the
            # `simulations` table: current_simulation() is an AMICI build+simulate — minutes
            # of C++ compile on a cold/resumed session — and a query over the static tables or
            # over `simulations_history` (assembled from per-version files on disk, no
            # simulation involved) must not pay it.
            if data_mod.query_references_head_simulations(sql) and (
                session.simulation is None or not session.simulation.get("records")
            ):
                try:
                    ctx.current_simulation()
                except Exception:  # pragma: no cover - defensive; static tables still work
                    session.logger.warning(
                        "query_data: could not generate simulation dataframe", exc_info=True
                    )
            return data_mod.run_query(session, sql, n_rows, off)

        # Log the query (truncated) alongside max_rows so a failed call is reproducible
        # from `get_errors`.
        logged_sql = sql[:1000] if isinstance(sql, str) else sql
        return ctx.run(
            "query_data", run, {"sql": logged_sql, "max_rows": max_rows, "offset": offset}
        )

    @mcp.tool(annotations=_READ_ONLY)
    def get_fit_diagnosis(
        sections: str = "residuals",
        worst_n: int = 8,
        top_n: int = 8,
        max_observables: int = 12,
    ) -> dict[str, Any]:
        """The COMPACT, size-bounded DETAIL behind the verdicts every fit result already carries —
        where the fit is off, and which parameters it cannot pin.

        `per_observable` keeps `max_observables` rows by chi2 (default 12) on problems with more,
        PLUS up to 3 more by |bias| and up to 6 more carrying a `structure_flag` — so the real
        bound is `max_observables + 9`, not `max_observables`. The extras are deliberate: an
        observable that is biased or structurally flagged is the one worth seeing even when its
        chi2 does not make the cut, and silently dropping it would defeat the diagnosis. Stated
        exactly because the old wording promised a bound the union never honoured
        observables than that — the worst by chi2, plus the worst few by bias and a few
        structure-flagged rows, with an `omitted` block saying what was dropped. It is NOT a
        complete table on a large problem: aggregate over it and you are aggregating over a
        worst-first subsample. Pass `max_observables=0` for every row (large payload, re-sent in
        your history on every later turn). `sections` picks which half:
        `"residuals"` (the default), `"parameters"`, or `"all"` — one call, both.

        COSTS, since the choice is yours, and NEITHER half compiles anything. `residuals` is
        ~2.0-7.4 kB and 0.01-0.12 s of aggregation over the stored simulation (no new simulation);
        `parameters` is ~2.4-8.0 kB and at most ONE sensitivity evaluation at the frozen optimum
        (0.004 s on the toy, 0.04 s on a k=15 steady-state model with 252 points, 0.16 s on a k=15
        hierarchical one — and ~0.001 s where the fit's own verdict is re-served). It is the SIZE you
        pay repeatedly: every payload you request is re-sent in your history on every later turn, so
        ask for `parameters` (or `all`) when BIC has stalled and you need to know WHICH parameters to
        cut, not by reflex.

        RESIDUALS — fixed aggregations (NOT open SQL): one row per observable with its weighted
        `chi2`, `mean_weighted_residual` (bias — far from 0 = a systematic over/under-shoot),
        worst |residual|, and two SIGN-based structure tests, `sign_test` (bias) and `runs_test`
        (time trend / serial correlation), each with a `p` and a `low_power` flag; plus the
        `worst_n` largest-residual points (measurement vs simulation) and the noise-sigma report.
        Sign-based tests exist BECAUSE `chi2`/`reduced_chi2`/`chi2_gof_pvalue` are uninformative
        when the noise sigmas are ESTIMATED (a free sigma inflates to absorb misfit, so magnitude
        scores pin near their null however wrong the FORM is); they read only residual SIGNS, and
        have low power on few points, so treat a `low_power` non-significant result as weak
        evidence, NOT a pass. The rolled-up `residual_structure` verdict over them — the one that
        blocks the run's early plateau stop, replicate lack-of-fit arm included — is NOT repeated
        here: it rides on the fit result. Call this for which points, how big, which observable
        carries the chi2 — not to ask WHETHER the residuals are structured.

        PARAMETERS (`identifiability`) — which estimated parameters are unconstrained or trade
        off, i.e. the basis for SIMPLIFYING an over-parameterised model (raw nll/bic say it is too
        complex, not which parameters to cut). From the Fisher Information Matrix at the optimum:
        per-parameter standard error + fold_uncertainty, `at_bound`, marginal and partial
        correlations, and the near-flat eigen-directions (the parameter COMBINATIONS the data
        cannot pin). Plus the multistart spread — how much each parameter wanders across
        near-optimal starts, a model-free check — rolled with those flags into a ranked
        `candidates` list. `merge_candidates` prices the ONE-SHARED-VALUE tie per pair
        (:func:`petab_mcp.identifiability.merge_candidates`); the fit result already flags the
        cheapest few such ties worth taking as `merge_hints`, and this section is where every priced
        pair, and every pair a filter refused (`n_ties_skipped`, `filters_off`), is visible.
        Remedies are MODEL edits (there is no value/`estimate` tool): drop the term, take its
        limiting (irreversible/instantaneous) form, or give a pair one shared value. Read
        se/correlations with multistart_std — the FIM is local.

        For a full custom breakdown use `query_data`, for the optimizer log `get_fitting_log`.
        NEITHER section ever compiles a model: the residual half simulates on demand only for a model
        that is already built (else it reports `status` and no rows), and the parameter half describes
        the LAST FIT — a local expansion around an optimum — so it needs no simulation at all
        (`available: false` before there is a fit, `stale: true` once a DIFFERENT structure is in the
        working state).
        """
        ctx = route()
        session = ctx.session

        def run() -> dict[str, Any]:
            want = _diagnosis_sections(sections)
            out: dict[str, Any] = {}
            if "residuals" in want:
                if session.simulation is None or not session.simulation.get("records"):
                    # On demand, and only where that is an integration rather than a C++ build:
                    # `ensure_simulation` refuses an unbuilt structure, and `fit_diagnosis` then
                    # returns its empty-but-typed payload with the reason in `status`.
                    try:
                        ctx.current_simulation()
                    except Exception:  # pragma: no cover - defensive; return the empty-fit result
                        session.logger.warning(
                            "get_fit_diagnosis: could not generate simulation dataframe",
                            exc_info=True,
                        )
                out.update(
                    data_mod.fit_diagnosis(
                        session, worst_n, max_observables=max_observables or None
                    )
                )
            if "parameters" in want:
                # Guarded, and guarded SEPARATELY: before the two were one tool a FIM failure could
                # not take the residual half with it, and the caller that pays most for this half
                # dropping (the judge's result.json block) needs the other one regardless. The
                # no-fit case is the diagnosis's own answer (`available: false`), not a branch here:
                # it reads the frozen fit record, so there is no live state left to gate on.
                try:
                    out["identifiability"] = identifiability_mod.identifiability(
                        session, top_n=top_n
                    )
                except Exception:  # noqa: BLE001 - one section must not fail the other
                    session.logger.warning(
                        "get_fit_diagnosis: parameter identifiability unavailable", exc_info=True
                    )
                    out["identifiability"] = identifiability_mod.unavailable(
                        identifiability_mod.FAILED_NOTE
                    )
            return out

        return ctx.run(
            "get_fit_diagnosis", run, {"sections": sections, "worst_n": worst_n, "top_n": top_n}
        )


# ------------------------------------------------- held-out validation (harness-only)
def _register_validation_data_tools(mcp: FastMCP, route: Callable[[], _Ctx]) -> None:
    """Register ``evaluate_validation_data`` — the HARNESS's read, never the agent's.

    Registered only when :func:`create_server` was given a held-out measurement file, and
    deliberately ABSENT from ``petab_harness.agent.tools.ALLOWED_TOOLS``, which is a strict
    allowlist: the panel is handed the intersection of that set with the server's tools, so a tool
    missing from it cannot reach any agent's context — not the lead's, not an analyst's. Two things
    have to stay out of the agent's reach for a held-out score to mean anything: the held-out DATA
    (which never enters the problem tree the server is pointed at, because the harness keeps it in
    a sibling directory of ``problem/``) and the SCORE (which would otherwise be a fitness signal the
    agent could optimise against by trial and error, turning the held-out set into training data
    one tool call at a time).

    Registering it as a normal MCP tool anyway, rather than reaching into the server's internals
    from the harness, because the harness has no other way in: the session lives in a separate
    process (see ``petab_harness.runner.server``), and the model + its fitted values exist only
    there.
    """

    @mcp.tool(annotations=_READ_ONLY)
    def evaluate_validation_data(n_starts: int | None = None) -> dict[str, Any]:
        """Score the CURRENT model against HELD-OUT measurements, in the held-out set's own mode.

        Harness-only (see the registrar's docstring): this is the benchmark judge's predictive
        check, not a modelling tool, and it is not offered to the panel.

        Changes nothing. A fixed-θ set is one objective evaluation of the model as it stands, at
        its current fitted values, against data it was not fitted to. A TRANSFER set (one that
        brings its own parameter table) REFITS the model's estimated parameters on the held-out
        system with an ``n_starts`` multistart (the session's default when omitted) and reports the
        optimum; the fitted point is not written back. Returns
        ``{status, mode, nll, k, n, chi2, n_conditions, observables, ...}``; see
        :mod:`petab_mcp.validation_data` for the statuses and the two modes.
        """
        ctx = route()
        session = ctx.session
        return ctx.run(
            "evaluate_validation_data",
            lambda: validation_data_mod.score_session(
                session, ctx.engine, ctx.held_out, n_starts=n_starts
            ),
            {"n_starts": n_starts},
        )


# ------------------------------------------------------ provenance/version tools
def _register_provenance_tools(mcp: FastMCP, route: Callable[[], _Ctx]) -> None:
    @mcp.tool(annotations=_READ_ONLY)
    def get_provenance(limit: int = 50) -> list[dict[str, Any]]:
        """Return the most recent provenance events (edits, fits, rollbacks)."""
        ctx = route()
        session = ctx.session
        return session.provenance.recent_events(limit)

    @mcp.tool(annotations=_READ_ONLY)
    def get_errors(limit: int = 50) -> list[dict[str, Any]]:
        """Return the most recent logged errors (rejected operations / failed tool calls)."""
        ctx = route()
        session = ctx.session
        return session.provenance.recent_errors(limit)

    @mcp.tool(annotations=_READ_ONLY)
    def list_versions() -> list[dict[str, Any]]:
        """List all model versions (fit snapshots) with kind/label/timestamp and fit metrics
        (neg_log_likelihood, chi2, bic/aic/aicc/caic/hqc/tic, criterion_value, is_new_best) — use it to
        rank versions by `criterion_value` and find the best-scoring one. Every version is a fit
        snapshot (or the `rollback` snapshots); `is_new_best` tracks that same score (gated on a
        minimum effect size)."""
        ctx = route()
        session = ctx.session
        return session.provenance.list_versions()

    @mcp.tool(annotations=_READ_ONLY)
    def get_version(version: int) -> dict[str, Any]:
        """Return a version's Antimony model and parameter table."""
        ctx = route()
        session = ctx.session
        return ctx.run(
            "get_version", lambda: session.provenance.get_version(version), {"version": version}
        )

    @mcp.tool(annotations=_READ_ONLY)
    def diff_versions(from_version: int, to_version: int) -> dict[str, Any]:
        """Diff two versions: Antimony unified diff + parameter add/remove/change + a `fit`
        block with each version's neg_log_likelihood/chi2 and the neg-log-likelihood delta
        (positive = the `to` version fits better)."""
        ctx = route()
        session = ctx.session
        return ctx.run(
            "diff_versions",
            lambda: session.provenance.diff_versions(from_version, to_version),
            {"from": from_version, "to": to_version},
        )

    # NOTE: `current_version` is not a tool — `describe_problem` already returns it. A second tool
    # costs a JSON schema in every client's context on every turn and a docstring to keep in sync.
    @mcp.tool(annotations=_RESTORE_ANNOTATIONS)
    def restore_version(version: int, justification: str = "") -> dict[str, Any]:
        """Bring a prior version's model + parameters back as the working state, WITHOUT refitting.

        The cheap, correct way to go back to an earlier model: it restores that version's stored
        fit instead of recompiling and refitting. Prefer this over re-authoring an earlier model
        with `make_changes` — a re-author recompiles, refits, and appends a byte-identical duplicate
        version (which `make_changes` will warn about, then refuse, once it detects the repeat), whereas this
        reuses the existing fit and records a clean restore. Find the target with `list_versions`
        (pick the best-scoring / is_new_best version), then `restore_version(v)`. Appends a new
        version rather than erasing history, so a restore is itself auditable."""
        ctx = route()
        session = ctx.session
        return ctx.run(
            "restore_version",
            lambda: session.restore_version(version, justification or None),
            {"version": version},
        )

    @mcp.tool(annotations=_READ_ONLY)
    def get_progress() -> dict[str, Any]:
        """Early-stop-chain progress for this run — a cheap counter read (no fit/simulation).

        `{n_fitted_ops, non_improving_streak, streak_cap, best_version, criterion, best_score,
        best_gof_pvalue, score_improvement_over_last_n, plateau_window, should_stop, stop_reason}`.
        `best_score` is under `criterion` — the complexity penalty this run ranks by.
        `best_version` is the best-scoring fitted model committed so far (the checkpoint to score);
        `non_improving_streak` counts consecutive fitted ops since the last real new best.
        `should_stop` is set (with a `stop_reason`) once the run has stopped making progress — the
        hard non-improving-streak cap, or a plateau once the best model is statistically adequate.
        There is no cap on the TOTAL number of fits: a run that keeps genuinely improving keeps
        going, and only a stall stops it.
        A plateaued `stop_reason` also carries the structural-inadequacy nudge: when the score
        stops moving and the best model is still not adequate, the misfit is the model's FORM — the
        next model must change the structure (add a latent state, add a feedback reaction, raise
        interaction order), not re-fit or widen the same one.
        """
        ctx = route()
        session = ctx.session
        return session.progress_report()


# --------------------------------------------------------------------- resources
def _register_resources(mcp: FastMCP, route: Callable[[], _Ctx]) -> None:
    @mcp.resource("petab://model/antimony")
    def _res_model_antimony() -> str:
        ctx = route()
        session = ctx.session
        return model_mod.get_model(session, "antimony")

    @mcp.resource("petab://model/sbml")
    def _res_model_sbml() -> str:
        ctx = route()
        session = ctx.session
        return model_mod.get_model(session, "sbml")

    @mcp.resource("petab://observables")
    def _res_observables() -> str:
        ctx = route()
        session = ctx.session
        return json.dumps(df_to_records(session.problem.observable_df), indent=2)

    @mcp.resource("petab://conditions")
    def _res_conditions() -> str:
        ctx = route()
        session = ctx.session
        return json.dumps(df_to_records(session.problem.condition_df), indent=2)

    @mcp.resource("petab://experiments")
    def _res_experiments() -> str:
        ctx = route()
        session = ctx.session
        return json.dumps(df_to_records(session.problem.experiment_df), indent=2)

    @mcp.resource("petab://measurements")
    def _res_measurements() -> str:
        ctx = route()
        session = ctx.session
        return json.dumps(df_to_records(session.problem.measurement_df), indent=2)

    @mcp.resource("petab://parameters")
    def _res_parameters() -> str:
        ctx = route()
        session = ctx.session
        return json.dumps(
            param_mod.parameters_to_records(session.problem, shipped=session.shipped_parameter_df),
            indent=2,
        )

    @mcp.resource("petab://provenance")
    def _res_provenance() -> str:
        ctx = route()
        session = ctx.session
        return json.dumps(session.provenance.recent_events(200), indent=2)

    @mcp.resource("petab://errors")
    def _res_errors() -> str:
        ctx = route()
        session = ctx.session
        return json.dumps(session.provenance.recent_errors(200), indent=2)

    @mcp.resource("petab://versions")
    def _res_versions() -> str:
        ctx = route()
        session = ctx.session
        return json.dumps(session.provenance.list_versions(), indent=2)

    @mcp.resource("petab://simulations")
    def _res_simulations() -> str:
        ctx = route()
        return json.dumps(ctx.run("simulations", ctx.current_simulation), indent=2)

    @mcp.resource("petab://data/schema")
    def _res_data_schema() -> str:
        ctx = route()
        session = ctx.session
        return json.dumps(data_mod.table_schema(session), indent=2)

    @mcp.resource("petab://fit/log")
    def _res_fit_log() -> str:
        ctx = route()
        session = ctx.session
        path = logging_setup.pypesto_log_path(session.provenance.root)
        return path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""


def _open_ctx(
    yaml_path: str,
    *,
    provenance_dir: str | None,
    criterion: str | None,
    auto_fit_starts: int,
    validation_measurements: str | None,
    score_shipped_model: bool,
    build_root: str | None,
    reference_mode: str,
    resolver: Any | None,
    auto_fit: bool,
    entity_map: str | None,
    fit_engine: Any | None,
) -> _Ctx:
    """Open the session on the problem at ``yaml_path``: its :class:`PetabSession`, the
    :class:`AmiciEngine` wired into it, and its held-out set. The arguments are
    :func:`create_server`'s, but for ``fit_engine``: the server's
    :class:`~petab_mcp.dask_engine.DaskEngine`, which the session's fits share, or ``None``."""
    session = PetabSession(
        yaml_path,
        provenance_dir=provenance_dir,
        reference_mode=reference_mode,
        resolver=resolver,
        criterion=criterion,
    )
    engine = AmiciEngine(session, build_root=build_root, fit_engine=fit_engine)
    # Wire the engine so the fit `make_changes` runs can fit + simulate.
    session.engine = engine
    session.auto_fit = auto_fit
    session.auto_fit_starts = auto_fit_starts
    if entity_map:
        from . import seeding as seeding_mod

        session.entities = seeding_mod.load_entity_map(entity_map)
        session.inputs = seeding_mod.load_inputs(entity_map)
    if score_shipped_model and session.model_authored():
        fitting_mod.score_at_nominal(session, engine)
    held_out = None
    if validation_measurements:
        # Read ONCE, as the session opens: a per-call read would make the score depend on a file
        # the run can no longer see (the harness runs from a scratch copy on the cluster) and would
        # let a mid-run edit of the file change the bar under the judge.
        held_out = validation_data_mod.read_held_out(validation_measurements)
    return _Ctx(session, engine, held_out)


def create_server(
    yaml_path: str,
    build_root: str | None = None,
    provenance_dir: str | None = None,
    reference_mode: str = "best_effort",
    resolver: Any | None = None,
    skills_dir: str | None = None,
    auto_fit: bool = True,
    auto_fit_starts: int = DEFAULT_AUTO_FIT_STARTS,
    criterion: str | None = None,
    validation_measurements: str | None = None,
    score_shipped_model: bool = False,
    entity_map: str | None = None,
    cluster: Any | None = None,
) -> FastMCP:
    """Build a FastMCP server bound to the PEtab problem at ``yaml_path``.

    ``skills_dir`` overrides where tool-usage skills are loaded from; when omitted
    the ``_skills`` directory bundled with the package is discovered automatically.

    ``auto_fit`` (default on) makes every commit fit + simulate before snapshotting;
    ``auto_fit_starts`` is the default pyPESTO multistart count (overridable per commit via
    ``n_starts``). Set ``auto_fit=False`` to skip the compile-heavy fit hook (tests/dev).

    ``criterion`` chooses the complexity penalty the run ranks models by — ``bic`` (default),
    ``caic``, ``aic``, ``aicc``, ``hqc`` or ``tic`` (see :mod:`criteria`). ``None`` means "as the
    environment says" (``PETAB_MCP_CRITERION``, else plain BIC). Resolved once by the session
    and held as ``session.criterion``, which every scorer is handed: there is no process state,
    so two servers in one process can rank by different criteria and nothing can change a
    session's criterion under it.

    ``validation_measurements`` is a PEtab measurement TSV of HELD-OUT data — data the model is
    never fitted to — with the set's own condition/parameter tables beside it where it has any
    (:func:`petab_mcp.validation_data.read_held_out`). Giving it registers
    ``evaluate_validation_data``, which the benchmark judge calls to score the finished model on
    that data; it is not offered to the agent (see ``_register_validation_data_tools``). Omit it and
    no such tool exists.

    Initial conditions are ESTIMATED by default; an IC of exactly 1 (a unit normalization) is held
    FIXED and an IC of exactly 0 (an empty pool) gets no parameter at all (see
    ``reconcile.ensure_parametric_initial_conditions``); to keep another IC fixed, ship it as a
    fixed parameter row.

    ``score_shipped_model`` (default off) compiles a problem that ships a model and scores it at
    the table's nominal values before serving, with no fit (:func:`fitting.score_at_nominal`); a
    model-less problem ignores it.

    ``entity_map`` is a JSON file binding the problem's species to INDRA agents
    (:func:`seeding.load_entity_map`) and declaring its experimental inputs
    (:func:`seeding.load_inputs`). Giving it offers ``make_changes`` the ``statements`` route;
    omit it and the route does not exist.

    ``cluster`` (a ``distributed`` cluster, the ``dask`` extra) runs every fit's starts on that
    cluster's workers through one client (:mod:`petab_mcp.dask_engine`); without one, each fit runs
    its starts in local processes. The workers import the compiled models from ``build_root``, so on
    more than one machine it must be on a filesystem they share. An AMICI built without HDF5 cannot
    send a start to a worker, so it refuses the cluster here.
    """
    fit_engine = None
    if cluster is not None:
        from .dask_engine import DaskEngine

        fit_engine = DaskEngine(cluster)
    ctx = _open_ctx(
        yaml_path,
        provenance_dir=provenance_dir,
        criterion=criterion,
        auto_fit_starts=auto_fit_starts,
        validation_measurements=validation_measurements,
        score_shipped_model=score_shipped_model,
        build_root=build_root,
        reference_mode=reference_mode,
        resolver=resolver,
        auto_fit=auto_fit,
        entity_map=entity_map,
        fit_engine=fit_engine,
    )

    # Load tool-usage skills up front so the server instructions only advertise
    # `list_skills`/`get_skill` when skills are actually available (registered below).
    resolved_skills_dir = skills_dir or skills_mod.default_skills_dir()
    skills = skills_mod.load_skills(resolved_skills_dir)
    instructions = SERVER_INSTRUCTIONS + (_SKILLS_INSTRUCTION if skills else "")
    mcp = FastMCP("petab-mcp", instructions=instructions)

    # Register the tool + resource groups; each tool asks `route` for the context per call.
    def route() -> _Ctx:
        return ctx

    _register_read_tools(mcp, route)
    # Offered whenever a map was given, even an empty one: a problem whose observables all read
    # formulas, and whose starts are the statements' own, binds nothing in it (Rahman).
    _register_model_tools(mcp, route, statements=entity_map is not None)
    _register_parameter_tools(mcp, route)
    _register_fit_tools(mcp, route)
    if ctx.held_out is not None:
        _register_validation_data_tools(mcp, route)
    _register_provenance_tools(mcp, route)
    _register_resources(mcp, route)
    # Register the skills loaded above as `list_skills`/`get_skill` tools plus
    # `petab://skills` resources (no-op when no skills were found).
    skills_mod.register_skills(mcp, skills)

    return mcp
