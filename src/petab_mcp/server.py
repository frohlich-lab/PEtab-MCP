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
  ``get_version``, ``diff_versions``, ``restore_version``, ``get_progress``, ``wait_for_fit``
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

import contextlib
import json
import threading
import time
from typing import Annotated, Any, Callable, Literal

import anyio
from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from fastmcp.server.middleware import Middleware
from fastmcp.tools import ToolResult
from pydantic import BaseModel, BeforeValidator, ConfigDict, Field
from pydantic.json_schema import SkipJsonSchema
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
from . import progress_signals as signals_mod
from . import provenance as prov_mod
from . import reconcile as reconcile_mod
from . import regimes as regimes_mod
from . import skills as skills_mod
from . import validation_data as validation_data_mod
from . import withheld
from .constants import (
    DEFAULT_AUTO_FIT_STARTS,
    MAX_FIT_STARTS_FACTOR,
    NON_IMPROVING_STREAK_CAP,
    PLATEAU_WINDOW,
    STREAK_RESET_MIN_SCORE_GAIN,
)
from .engine import AmiciEngine, resolve_build_root
from .errors import (
    FIT_IN_PROGRESS,
    PetabMcpError,
    UnknownEntityError,
    ValidationError,
    disk_full_error,
)
from .formatting import (
    conditions_tsv,
    df_to_records,
    df_to_tsv,
    experiments_tsv,
    summarize_measurements,
)
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
    "`get_provenance` / `get_errors`. Every fitted version is kept and the best one stands "
    "whatever comes after it, so an earlier fitted model re-authored exactly is refused, not "
    "re-fitted: that version already stands for it."
)


# Appended to the instructions only when skills are actually registered, so clients
# are never told to call `list_skills`/`get_skill` when those tools are absent.
_SKILLS_INSTRUCTION = (
    " New to this server? Call `list_skills` and read "
    "`get_skill('petab-model-building')` for the full tool-usage workflow; "
    "`get_skill('reading-fits')` says what a fit result's scores, flags and diagnosis mean, and "
    "how to summarise the fit's tables in SQL."
)
#: Appended where the `statements` route is offered, with the skill only it registers.
_STATEMENTS_SKILL_INSTRUCTION = (
    " Before your first `statements` call, read `get_skill('statements-grammar')`: a row and its "
    "binding, a condition input, regulators, a `rate` beside a `scheme`, and `seeds` beside "
    "`set_amount`, by example."
)


# A TYPED model rather than `dict[str, Any]`, because the JSON schema is the only part of a tool
# contract a model is actually constrained by — a docstring is prose it may or may not read.
# Advertised as a bare dict this argument named NONE of its keys, and 527 of 889 `make_changes`
# calls were rejected for guessing them: `id` for `parameter_id` (484 on its own, the single
# largest rejection class), plus `lowerBound`, `estimate`, `nominalValue`, `parameterScale`. Five
# unrelated clients made the same class of mistake, which is the signature of a contract problem
# rather than a client one.
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
    """One mechanism assertion: the claim, in words. Its mechanism is its binding, apart."""

    model_config = ConfigDict(extra="forbid")

    statement: dict[str, Any] = Field(
        description="the assertion as an INDRA statement's JSON: its `type` and agents, e.g. "
        "{type: 'Phosphorylation', enz: {name: 'A'}, sub: {name: 'B'}, residue: 'S', "
        "position: '10'}; and the agents that regulate its rate without taking part, each a list: "
        "`inhibited_by`, `activated_by`, `modulated_by` ('..., inhibited by C' in the notation). "
        "Each is recorded as the kind its rate shows, which the result reports where it differs."
    )
    prefix: str = Field(
        description="names this row's parameters and rules ('<prefix>_kcat'), and keys its "
        "binding; unique per row."
    )
    derived: dict[str, str] | None = Field(
        default=None,
        description="ignored: what `get_model` shows a row's rate records for its regulators, by "
        "the kind stated and position ({'inhibited_by_0': 'activated_by'}), so a row it shows "
        "reads back as it was.",
    )
    # Not in the schema, and refused when given (operations.carried_binding): a row's binding is
    # apart from it. Taken here rather than forbidden so the refusal says how the row splits and
    # is a typed one, logged with its cause, not a schema error naming two unknown fields.
    policy: SkipJsonSchema[str | None] = None
    parameters: SkipJsonSchema[dict[str, Any] | None] = None


class RowBinding(BaseModel):
    """A row's binding: the mechanism that expresses its statement, and its parameter bindings."""

    model_config = ConfigDict(extra="forbid")

    policy: str = Field(
        description="the mechanism expressing the row's statement, e.g. 'default' (INDRA's own "
        "for every statement type), 'two_step', or one you write in `mechanisms`."
    )
    parameters: dict[str, str | float] | None = Field(
        default=None,
        description="binds a parameter the row makes, by its kind ('kf' for '<prefix>_kf'), to "
        "a SHARED id (another row's, a problem parameter's, or a new name several rows use) or "
        "to a NUMBER, which fixes it — one rate constant across many reactions is one parameter.",
    )


class SeedRow(BaseModel):
    """One species present at t = 0, and how much of it."""

    model_config = ConfigDict(extra="forbid")

    agent: dict[str, Any] | str = Field(
        description="the species in the state it starts in: INDRA Agent JSON or the notation, "
        "e.g. 'A' or 'A{-pS10}'; a complex through its bound partners ('A{!B}')."
    )
    amount: float | str = Field(
        description="its starting amount: a number (1 is a fixed unit pool, 0 an empty seed, "
        "anything else estimated from there) or an expression over parameter ids (a parameter "
        "the problem ships, a condition-table column, a new name)."
    )
    name: str | None = Field(
        default=None,
        description="names its parameter `<name>_init`; defaults to the agent's name and form "
        "('BRAF' for 'BRAF', 'BRAF_S151A' for 'BRAF{mut:S151A}', 'EGFR_pY1068' for "
        "'EGFR{pY1068}'), and must differ from every other seed's.",
    )


class MechanismDef(BaseModel):
    """A mechanism you write when none in the library fits: its own reactions (`scheme`), or an
    existing mechanism's reaction with your own rate (`rate`)."""

    model_config = ConfigDict(extra="forbid")

    statement_type: str = Field(description="the statement type it expresses, e.g. 'Conversion'.")
    policy: str = Field(description="its name, new; a row's binding then names it as its policy.")
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
        "this way, with the `parameters` it declares: start from one. A rate is an expression "
        "over your `parameters`, `amount(<pattern>)`, `amount(inhibited_by)` (a regulator the "
        "statement states, checked on the row's net rate as the model builds it: its "
        "intermediates at steady state, its products absent; one acting only on a step that rate "
        "cannot see, a product's degradation, goes on that step's own row), `time` and "
        "exp/log/sqrt. A molecule is a role; by its name, a "
        "partner the statement writes a role bound or not bound to (`C` of `E{~C}`); or, by its "
        "kind, a regulator the statement states, with binding sites only, each named after the "
        "partner it binds: a drug that sequesters the enzyme is `enz(inhibited_by=None) + "
        "inhibited_by(enz=None) | enz(inhibited_by=1) % inhibited_by(enz=1)`, the catalytic step "
        "on `enz(inhibited_by=None)`, and other rows read the complex as `E{!X}`. A mechanism "
        "reads no agent its statement does not name. Give `scheme` or `rate`, not both.",
    )
    rate: str | None = Field(
        default=None,
        description="(instead of `scheme`) the rate COEFFICIENT, over the statement type's role "
        "names (each the amount of that agent: `enz`, `sub`, `subj`, `source`, ...), the "
        "regulators the statement states (`inhibited_by`, `activated_by`, `modulated_by`; "
        "`inhibited_by_1` by position where it states several), your `parameters`, `time`, "
        "exp/log/sqrt and `piecewise(value, condition, ..., otherwise)` (`piecewise(k_0, time < "
        "t_1, k_1)`). It is multiplied by every reactant's amount, catalysts included, so a rate "
        "that stays non-negative cannot drive an amount negative; a constant `k` alone is mass "
        "action. Each regulator is recorded as the kind the row's net rate shows for every value "
        "of its names (sympy, within the parameters' bounds): inhibited where it never rises, "
        "activated where it never falls, else modulated; one it does not change with is refused."
    )
    parameters: list[str] = Field(
        default_factory=list,
        description="every constant the `scheme` or `rate` names; each becomes "
        "`<prefix>_<name>`, estimated unless the row's binding shares or fixes it (`kf=k_on`). A "
        "parameter the problem ships or a condition input is none of them: it enters through a "
        "row's binding.",
    )


class StatementEdit(BaseModel):
    """One change to a model built from statements: a row, its binding, a seed, an amount or a
    readout."""

    model_config = ConfigDict(extra="forbid")

    op: Literal[
        "add_row",
        "replace_row",
        "rebind",
        "remove_row",
        "set_seed",
        "remove_seed",
        "set_amount",
        "remove_amount",
        "set_readout",
    ] = Field(
        description="`add_row` a new row with its `binding`; `replace_row` rewrites the statement "
        "of the row with that row's prefix, keeping its binding unless it gives one; `rebind` "
        "gives the row of `prefix` a new `binding`, its statement unchanged; `remove_row` by "
        "`prefix`, with its binding; `set_seed` adds or replaces the seed of that name; "
        "`set_amount`/`set_readout` add or replace one; `remove_seed`/`remove_amount` by `name`. "
        "A readout is never removed: each is one an observable reads."
    )
    row: StatementRow | str | None = Field(
        default=None, description="add_row/replace_row: the row, one line or JSON."
    )
    binding: RowBinding | str | None = Field(
        default=None,
        description="add_row, rebind (and replace_row, where the mechanism changes too): the "
        "row's binding, one line ('two_step kf=k_on') or JSON.",
    )
    prefix: str | None = Field(default=None, description="rebind, remove_row: the row's prefix.")
    agent: dict[str, Any] | str | None = Field(
        default=None,
        description="set_seed: the species in the state it starts in, as a seed's `agent`; "
        "set_amount: the agent pattern whose amount it is.",
    )
    amount: float | str | None = Field(
        default=None, description="set_seed only: its starting amount, as a seed's `amount`."
    )
    name: str | None = Field(
        default=None,
        description="remove_seed, set_amount, remove_amount: its name; set_seed: as a seed's "
        "`name`, by default its agent's form.",
    )
    readout: str | None = Field(
        default=None, description="set_readout: the readout id."
    )
    formula: str | None = Field(default=None, description="set_readout: its formula.")


# Provider serialisation artefact: some clients emit a list-typed argument wrapped in a
# single-key object -- `{"parameters": {"item": [{...}, {...}]}}` instead of
# `{"parameters": [{...}, {...}]}` -- or as a JSON string containing the array. pydantic then
# rejects the whole call with `Input should be a valid list [type=list_type]`, and because the
# wrapper is how that client serialises lists, the retry is byte-identical. This once produced 28
# client-side ToolExceptions that ended 3 sessions outright (0 fits, 0 versions) and
# recursion-limited a 4th; an escalation guard told the client "REPEATING THIS CALL WILL FAIL
# AGAIN" three times and the wrapper came back each time. Prompting cannot fix a serialiser, so
# accept the shape and unwrap it. Applied as a BeforeValidator so the JSON schema the client is
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


def _unwrap_objectish(v: Any) -> Any:
    """An object sent as a JSON string, back to the object; anything else untouched. The same
    serialiser artefact as a list sent as a string (``_unwrap_listish``): clients send `amounts`
    as '{"a": "A"}' (gemot's leak repair leaves such a value to the server), and `bindings`
    travels with every `statements` call."""
    if isinstance(v, str):
        s = v.strip()
        if s.startswith("{") and s.endswith("}"):
            try:
                parsed = json.loads(s)
            except ValueError:
                return v
            if isinstance(parsed, dict):
                return parsed
    return v


#: ``make_changes``' list and object arguments, typed once for its three registrations (see
#: ``_register_model_tools``), each tolerating the encodings ``_unwrap_listish`` and
#: ``_unwrap_objectish`` undo.
_ModelOps = Annotated[list[ModelOp] | None, BeforeValidator(_unwrap_listish)]
_ParameterChanges = Annotated[list[ParameterChange] | None, BeforeValidator(_unwrap_listish)]
_References = Annotated[list[str] | None, BeforeValidator(_unwrap_listish)]
_StatementRows = Annotated[list[StatementRow | str] | None, BeforeValidator(_unwrap_listish)]
_Bindings = Annotated[dict[str, RowBinding | str] | None, BeforeValidator(_unwrap_objectish)]
_Readouts = Annotated[dict[str, str] | None, BeforeValidator(_unwrap_objectish)]
_Amounts = Annotated[dict[str, dict[str, Any] | str] | None, BeforeValidator(_unwrap_objectish)]
_MechanismDefs = Annotated[list[MechanismDef] | None, BeforeValidator(_unwrap_listish)]
_SeedRows = Annotated[list[SeedRow] | None, BeforeValidator(_unwrap_listish)]
_StatementEdits = Annotated[list[StatementEdit] | None, BeforeValidator(_unwrap_listish)]


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


# Over this many ids the list is truncated. The ceiling is real — a 118-row parameter table,
# re-sent in the caller's history every turn thereafter, was a measured top-3 context cost — but
# it sits where a normal problem's table fits WHOLE. The previous cap of 12 elided the answer
# from the message whose entire job was to give it, and callers re-guessed: one burned six
# get_parameters calls, another four.
_ID_LIST_CAP = model_edits_mod._ID_LIST_CAP


def unknown_ids_message(
    unknown: list, known_ids: list[str], explained: dict[str, str] | None = None
) -> str:
    """The `get_parameters` unknown-id error text: what was wrong, what you probably meant, and
    what is valid. Built here rather than inline so it can be tested on the string a caller
    actually receives — the first version of this hint was assembled into a local variable and
    never interpolated, and shipped green because the tests around it called the scorer directly
    and grepped the module source.

    ``explained`` holds, per unknown id the problem does use, why it has no row
    (:func:`reconcile.not_a_parameter`); such an id gets that instead of near-miss guesses.
    """
    explained = explained or {}
    hints = []
    for u in unknown:
        if str(u) in explained:
            continue
        close = model_edits_mod.closest_symbols(str(u), known_ids)
        if close:
            hints.append(f"{u} -> {' or '.join(repr(c) for c in close)}")
    suggest = "".join(f" {explained[str(u)]}" for u in unknown if str(u) in explained)
    suggest += f" Did you mean: {'; '.join(hints)}." if hints else ""
    known = ", ".join(known_ids[:_ID_LIST_CAP])
    more = (
        ""
        if len(known_ids) <= _ID_LIST_CAP
        else f", (+{len(known_ids) - _ID_LIST_CAP} more — call get_parameters() with no "
        f"parameter_ids for the whole table)"
    )
    return f"get_parameters: unknown parameter_ids {unknown}.{suggest} Known ids: [{known}{more}]."


class _InfrastructureErrors(Middleware):
    """Ends a tool call that failed for want of disk, or that a fit pool failed (``fit_pool``,
    :mod:`petab_mcp.dask_engine`), with the MCP error a client ends the run on, whichever tool and
    wherever in it (:class:`~petab_mcp.errors.InfrastructureError`).

    The one place every tool's exception passes, so a raw ``OSError`` from a path no :class:`_Ctx`
    wrapper guards reads the same as a fit's. FastMCP has wrapped the exception by then (``Error
    calling tool '<name>': ...``), so the error is found on its chain, and re-raised as a
    ``ToolError``, whose text FastMCP sends as it is. A crash its chain does not explain is checked
    against the disks the server writes to (``folders``): a build whose install step hit the quota
    fails only at the import after it, in whichever tool built it (a held-out refit, a restore).
    """

    def __init__(self, logger, *folders) -> None:
        self.logger = logger
        self.folders = folders

    async def on_call_tool(self, context, call_next):
        try:
            return await call_next(context)
        except Exception as exc:
            # A crash is FastMCP's ToolError around what the tool raised; a refusal of the
            # arguments is not, and is not worth a write.
            crashed = isinstance(exc, ToolError) and exc.__cause__ is not None
            if (infra := disk_full_error(exc, *(self.folders if crashed else ()))) is None:
                raise
            self.logger.error("%s: %s", context.message.name, infra)
            raise ToolError(str(infra)) from exc


class _ProgressSignals(Middleware):
    """Sends every tool result, tool error, tool description and resource with its progress
    signals kept or cut, as ``create_server(progress_signals=...)`` says
    (:func:`progress_signals.strip`). Every tool but :data:`progress_signals.HARNESS_TOOLS`, which
    only whoever runs the session reads. Outermost, so it renders the other middleware's errors
    too."""

    def __init__(self, signals: bool) -> None:
        self.signals = signals

    async def on_call_tool(self, context, call_next):
        try:
            result = await call_next(context)
        except ToolError as exc:
            raise ToolError(signals_mod.render(str(exc), self.signals)) from exc
        signals = self.signals or context.message.name in signals_mod.HARNESS_TOOLS
        structured = result.structured_content
        if structured is None:
            return result.model_copy(
                update={
                    "content": [
                        b.model_copy(update={"text": signals_mod.render(b.text, signals)})
                        if isinstance(getattr(b, "text", None), str)
                        else b
                        for b in result.content
                    ]
                }
            )
        if (structured := signals_mod.strip(structured, signals)) == result.structured_content:
            return result
        wrapped = (result.meta or {}).get("fastmcp", {}).get("wrap_result")
        return ToolResult(
            content=structured["result"] if wrapped else structured,
            structured_content=structured,
            meta=result.meta,
        )

    async def on_list_tools(self, context, call_next):
        return [
            t.model_copy(
                update={
                    "description": t.description and signals_mod.render(t.description, self.signals),
                    "parameters": signals_mod.render_all(t.parameters, self.signals),
                }
            )
            for t in await call_next(context)
        ]

    async def on_read_resource(self, context, call_next):
        result = await call_next(context)
        contents = [
            c.model_copy(update={"content": signals_mod.render(c.content, self.signals)})
            if isinstance(c.content, str)
            else c
            for c in result.contents
        ]
        return result.model_copy(update={"contents": contents})


class _Cancellation(Middleware):
    """Stops the fit of a tool call its client cancels (MCP ``notifications/cancelled``).

    The MCP SDK cancels the call's task, but a sync tool runs in a worker thread no cancel reaches:
    FastMCP awaits that thread shielded, so the tool would run to its end. A task beside it hears
    the cancel instead and sets the call's event in :data:`petab_mcp.multistart.CANCELLED`, which
    the fit's wait polls. Imported here, not at the top: :mod:`~petab_mcp.multistart` imports
    pyPESTO, and with it AMICI, which must not be imported before the session configures its
    logging (``logging_setup.configure_amici``).
    """

    async def on_call_tool(self, context, call_next):
        from .multistart import CANCELLED

        cancelled = threading.Event()

        async def hear() -> None:
            try:
                await anyio.sleep_forever()
            except anyio.get_cancelled_exc_class():
                if not returned:
                    cancelled.set()
                raise

        returned, result, error = False, None, None
        token = CANCELLED.set(cancelled)
        try:
            async with anyio.create_task_group() as tasks:
                tasks.start_soon(hear)
                try:
                    result = await call_next(context)
                except Exception as exc:  # raised below: the group would wrap it in an ExceptionGroup
                    error = exc
                finally:
                    returned = True
                    tasks.cancel_scope.cancel()
        finally:
            CANCELLED.reset(token)
        if error is not None:
            raise error
        return result


#: Seconds between two heartbeats of a tool call (:class:`_Heartbeat`).
HEARTBEAT_S = 30.0


class _Heartbeat(Middleware):
    """Sends a tool call an MCP progress notification every :data:`HEARTBEAT_S` while it runs, on
    the call's own progress token, so its client can time it out on silence rather than on a
    budget no formula bounds once fits queue for a shared pool (gemot's idle timeout).

    ``progress`` is the call's seconds so far, which the protocol requires to increase, and
    ``message`` the session's running change, ``"<phase> for <s>s, <done> of <n> starts done"``
    (:class:`~petab_mcp.multistart.Progress`), the starts left out before its fit has begun; with
    no change running there is none. The change is read without the session's lock, so a call
    waiting on that lock behind a fit gets heartbeats too. A task beside the call sends them, as
    :class:`_Cancellation` listens, so they hold no worker thread. A call without a progress token
    gets none.
    """

    def __init__(self, session: PetabSession) -> None:
        self.session = session

    def _message(self) -> str | None:
        if self.session.change_began is None:
            return None
        progress, now = self.session.change_progress, time.monotonic()
        starts = f", {progress.done} of {progress.total} starts done" if progress.total else ""
        return f"{progress.phase} for {now - progress.since:.0f}s{starts}"

    async def on_call_tool(self, context, call_next):
        ctx = context.fastmcp_context
        request = ctx.request_context if ctx is not None else None
        if request is None or "progressToken" not in (request.meta or {}):
            return await call_next(context)
        began = time.monotonic()

        async def beat() -> None:
            while True:
                await anyio.sleep(HEARTBEAT_S)
                with contextlib.suppress(Exception):  # a heartbeat lost never fails its call
                    await ctx.report_progress(time.monotonic() - began, message=self._message())

        result, error = None, None
        async with anyio.create_task_group() as tasks:
            tasks.start_soon(beat)
            try:
                result = await call_next(context)
            except Exception as exc:  # raised below: the group would wrap it in an ExceptionGroup
                error = exc
            finally:
                tasks.cancel_scope.cancel()
        if error is not None:
            raise error
        return result


class _FixPointers(Middleware):
    """Sends every tool result and resource without the numbers :mod:`withheld` cuts; installed
    only by ``create_server(fix_pointers=False)``. A JSON text block is cut as its structured
    content is, and one that loses nothing is sent as it was."""

    @staticmethod
    def _text(text: str) -> str:
        try:
            value = json.loads(text)
        except ValueError:
            return text
        cut = withheld.strip(value)
        return text if cut == value else json.dumps(cut, ensure_ascii=False)

    async def on_call_tool(self, context, call_next):
        result = await call_next(context)
        content = [
            b.model_copy(update={"text": self._text(b.text)})
            if isinstance(getattr(b, "text", None), str)
            else b
            for b in result.content
        ]
        return result.model_copy(
            update={
                "content": content,
                "structured_content": withheld.strip(result.structured_content),
            }
        )

    async def on_read_resource(self, context, call_next):
        result = await call_next(context)
        contents = [
            c.model_copy(update={"content": self._text(c.content)})
            if isinstance(c.content, str)
            else c
            for c in result.contents
        ]
        return result.model_copy(update={"contents": contents})


#: Seconds ``wait_for_fit`` waits when not told, and at most: the cap is under the 600 s tool
#: timeout of gemot's CLI drivers, so a wait answers before its client gives up on it, and the
#: default leaves room for a client whose timeout is shorter.
WAIT_FOR_FIT_S = 240
WAIT_FOR_FIT_MAX_S = 540

#: Seconds between two looks of ``wait_for_fit`` at whether the change it waits for has ended.
_WAIT_POLL_S = 1.0


def _fit_in_progress(began: float, progress: Any) -> str:
    """The refusal of a ``make_changes`` that arrives while another change's fit, begun at
    ``began`` (``time.monotonic()``) and at ``progress``
    (:class:`~petab_mcp.multistart.Progress`), still holds the session."""
    now = time.monotonic()
    starts = ""
    if progress.total:
        remaining = progress.remaining_s(now)
        starts = f"; {progress.done} of its {progress.total} starts are done, " + (
            "none yet to time the rest by"
            if remaining is None
            else f"the rest at that pace about {max(1, round(remaining / 60))} min"
        )
    return (
        f"make_changes: {FIT_IN_PROGRESS}: a fit is still running on this session (its change "
        f"began {now - began:.0f}s ago{starts}), so NOTHING was applied. Its result becomes the "
        "next version when it ends: call wait_for_fit, which answers once it is committed, rather "
        "than resending this change or polling another tool."
    )


def _implicit_base(session: PetabSession, call: dict[str, Any]) -> str | None:
    """The refusal of a ``make_changes`` without ``base_version`` that would apply to a HEAD that
    is not the best version, or ``None`` to apply it.

    Agents left ``base_version`` out after a regression and lost the fit of a change to the
    rejected HEAD (75 min in one run, #330), and a warning on the result came only after it. So
    the first such call is refused, before anything runs, and the same call (``call``, its
    arguments bar ``justification`` and ``references``) sent again confirms it. A HEAD
    ``restore_version`` moved to was chosen already, and is not asked about."""
    head = session.provenance.current_version
    best = session.best.version if session.best is not None else None
    if best is None or head < 0 or head in (best, session.head_restored):
        return None
    key = (head, json.dumps(call, sort_keys=True, default=_plain))
    if session.implicit_base_asked == key:
        session.implicit_base_asked = None
        return None
    session.implicit_base_asked = key
    return (
        f"make_changes: no `base_version`, so this change would apply to the working model, "
        f"v{head}, which is not the best version; v{best} is. NOTHING was applied. Send "
        f"`base_version={best}` to change the best, or this same call again to change v{head}."
    )


def _plain(value: Any) -> Any:
    """``value`` as JSON can write it: a pydantic model's fields, else its repr."""
    return value.model_dump() if isinstance(value, BaseModel) else repr(value)


def _labelled(
    session: PetabSession, out: dict[str, Any], version: Any, unversioned: str
) -> dict[str, Any]:
    """A read's result ``out`` with ``version``, the version it shows (``None`` for none), and,
    while a change is being fitted on ``session``, ``fit_running``, which says what the read
    shows: that version, or ``unversioned``. Reads take no lock, so a read beside a fit sees that
    change's working model and the last fit's records (#330). A failed read (``ok: false``) is
    returned as it is."""
    if out.get("ok") is False:
        return out
    shown = version if isinstance(version, int) and version >= 0 else None
    out = {**out, "version": shown}
    if session.change_began is not None:
        out["fit_running"] = (
            "a change is being fitted; this read shows "
            + (f"v{shown}, the last version committed" if shown is not None else unversioned)
            + ". The change becomes the next version when its fit ends."
        )
    return out


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
    def raise_if_infrastructure(exc: Exception) -> None:
        """Re-raise ``exc`` as the :class:`~petab_mcp.errors.InfrastructureError` it is, if it is
        one: a full disk, raised as such or wrapped in a refusal (BioNetGen's ``[Errno 28]``), is
        never a rejection, and stays out of ``errors.jsonl``, which the agent reads."""
        infra = disk_full_error(exc)
        if infra is exc:
            raise exc
        if infra is not None:
            raise infra from exc

    @staticmethod
    def short(value: Any, limit: int = 200, _depth: int = 0) -> Any:
        """Truncate long strings before they go into the (persisted) error log, so a large
        argument — e.g. a whole Antimony model passed to `make_changes` — cannot bloat the
        provenance error log or dump the full model contents into it.

        RECURSES into lists and dicts, because the bulky arguments on this server are mostly
        containers: `parameters` is up to ~130 entries on a large problem. Truncating only top-level strings let those through whole,
        which mattered little while the log was ephemeral and matters now that errors.jsonl is
        kept for good. The element cap is separate from the string cap: a list is summarised, not
        just its members shortened.
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
            self.raise_if_infrastructure(exc)
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
            self.raise_if_infrastructure(exc)
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

        The de-dup no-op is undone too, though it is not a failure: the change gives the same model
        as an existing version, so that version's scored state stands for it. The de-dup compares
        equations and estimation setup, not the text, the authored starting values or the rows the
        change was written as; keeping those would leave a working model no version holds, and the
        next edit would build on it.
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
                    # Before `bundle_result` folds the repeated ones into a count.
                    edit_warnings=[
                        m
                        for m in edit_result.get("messages") or []
                        if m.get("level") == msg_mod.LEVEL_WARNING
                    ],
                ),
                {"n_starts": n_starts},
            )
        except Exception as exc:  # noqa: BLE001 - already logged by run_fit; see the docstring
            if (infra := disk_full_error(exc)) is not None:
                # An unscored edit goes, as below; its revert event may not reach a full disk, and
                # the call ends either way.
                if self.session.provenance.current_version == head_before:
                    with contextlib.suppress(OSError):
                        ops_mod.revert_working_state(self.session, before, str(infra))
                raise
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
        elif fit_result.get("no_change"):
            ops_mod.revert_working_state(
                self.session,
                before,
                f"identical to version {fit_result.get('duplicate_of_version')}, which stands for it",
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
            # Machine-readable rejection class, for a client that tallies rejections by cause. The
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

    def run(self, tool: str, fn, args: dict[str, Any] | None = None, *, record: bool = True):
        """Run a non-operation tool; log failures (return {ok:false} or re-raise).

        ``record=False`` keeps a failure out of ``errors.jsonl``, which ``get_errors`` hands the
        agent, for a tool the agent must not read: it still reaches the session's
        ``petab_mcp.log`` and the caller."""
        try:
            return fn()
        except PetabMcpError as exc:
            self.raise_if_infrastructure(exc)
            if record:
                self.log_error(tool, args or {}, exc)
            self.session.logger.warning("%s failed: %s", tool, exc)
            out = {"ok": False, "error": str(exc)}
            if getattr(exc, "issues", None):
                out["issues"] = exc.issues
            return out
        except Exception as exc:
            if record:
                self.log_error(tool, args or {}, exc)
            self.session.logger.exception("%s crashed", tool)
            raise

    def current_simulation(self) -> dict[str, Any]:
        """The stored simulation dataframe, generated on demand if not present yet."""
        return fitting_mod.ensure_simulation(self.session, self.engine)


# MCP tool annotations: hints a client may use to auto-approve reads and to confirm writes. A
# "read-only" tool may still log a failed call or cache the on-demand simulation; neither changes
# what any later call returns. The writers are NOT destructive: ``make_changes`` appends a version
# and ``restore_version`` moves HEAD onto one, and neither erases anything (see ``provenance``).
# Neither is idempotent: a repeat appends another version or another ``rollback`` event.
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
        """Orientation, and cheap: whether a model has been authored (`model_authored`), table and
        entity counts, the current version, `readouts_to_define` (the `readout_*` measurement
        mappings your model does not define yet), `condition_targets` (the ids the condition table
        sets, `list_conditions`), and `fit_n_starts_default`, the multistart count a fit uses when
        `n_starts` is omitted.

        `inputs` are what the experiments impose within a period, each read in your model by its
        id: define it by its `rule` exactly as given, and never estimate the condition targets the
        rule reads.
        """
        ctx = route()
        session = ctx.session
        return session.describe()

    @mcp.tool(annotations=_READ_ONLY)
    def list_observables() -> str:
        """The (fixed) observables as a TSV table: id, formula, noise model, and `observableName`,
        WHAT each one measures. A formula naming a `readout_*` symbol is a measurement mapping YOUR
        model must define (`readout_x := <expression of your states>`); several observables may
        share one readout (one per dataset/gel, differing only in shipped scalings), and one rule
        covers them all. Scalings, offsets and the noise model are fixed parts of the assay.
        """
        ctx = route()
        session = ctx.session
        return df_to_tsv(session.problem.observable_df)

    @mcp.tool(annotations=_READ_ONLY)
    def list_conditions() -> str:
        """The (fixed) conditions as a TSV table, one row per condition and one column per target
        it sets (a model parameter, species or compartment) to that row's value — a dose, an input
        level, a switch; blank where the condition does not set the target. A table too large for
        one reply comes back as one row per target instead: how many conditions set it and the
        range of its values. Your model reads a target only where it has an effect: leaving an
        input unread is allowed, and one the model leaves out is declared for it, at 0. No rule
        may assign a target. A condition-set initial is an experimental input to reproduce, not a
        value to fit. WHEN a condition applies is `list_experiments`."""
        ctx = route()
        session = ctx.session
        return conditions_tsv(session.problem.condition_df)

    @mcp.tool(annotations=_READ_ONLY)
    def list_experiments() -> str:
        """The (fixed) experiments as a TSV table, one row per period: from `time` on, experiment
        `experimentId` applies every condition `conditionId` lists, space-separated (none: the
        model as is). `time` -inf is a pre-equilibration to steady state before the experiment
        starts. Every measurement names the experiment it was taken in; an empty `experimentId`
        simulates the model as is. A table too large for one reply comes back as one row per
        period layout instead: how many experiments share it, their periods' start times and how
        many conditions each period applies, under the `query_data` SQL that reads one
        experiment's periods."""
        ctx = route()
        session = ctx.session
        return experiments_tsv(session.problem.experiment_df)

    @mcp.tool(annotations=_READ_ONLY)
    def get_measurements(
        experiment_id: str | None = None,
        observable_id: str | None = None,
        summarize: bool = True,
        max_rows: int = 200,
        offset: int = 0,
    ) -> "list[dict[str, Any]] | dict[str, Any]":
        """The (fixed) measurements. By default a COMPACT SUMMARY per observable × experiment
        (point count, time span, value min/max/mean, a few preview points): prefer it, and prefer
        the `experiment_id`/`observable_id` filters or `query_data` for a targeted slice.

        `summarize=False` returns raw points, PAGED: `{rows, row_count, total_rows, offset,
        truncated, next_offset?}`; feed `next_offset` back as `offset` (row order is stable across
        pages). An unpaged dump of a large problem is 70-80k tokens that re-enter your context on
        every later turn.
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
        # invisible to get_errors AND counts as a tool error against a client's circuit breaker,
        # so three argument typos in a row could disable get_measurements for the rest of the
        # session.
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


# ``make_changes``'s tool description, composed by ``_make_changes_doc`` from the summary, the
# model ROUTES the registration offers, and the CONTRACT the call keeps whichever route is taken.
_MAKE_CHANGES_SUMMARY = (
    "THE editor of the model and its parameter search space, and it always FITS: one call is one "
    "coherent change, scored once and snapshotted as ONE version. There is no fit-free edit and no "
    "separate fit."
)

_MAKE_CHANGES_TEXT_ROUTES = """\
`antimony` authors the WHOLE model: the first one (`describe_problem` → `model_authored: \
false`), or a later start-over. Name EVERY reaction: an unnamed one cannot be changed afterwards.

`model_edits` is the route for every change after the first: a BATCH of operations, each written \
as Antimony and addressed by ID — `{op: 'add_reaction', antimony: 'transfer: X1 => X2; \
k*u*X1'}`, `{op: 'remove_reaction', id}`, `{op: 'set_rate_law', id, antimony: 'k*u*X1'}`, \
`{op: 'set_initial', id, antimony: '5'}`, `{op: 'set_assignment', id, antimony: 'X2 + X3'}` (an \
`id := expr` rule holding at ALL times — how a `readout_*` symbol an observable formula \
references is defined in terms of YOUR states), `{op: 'remove_assignment', id}`, `{op: 'rename', \
old, new}` (`new` keeps `old`'s fitted value, bounds and scale; onto an existing parameter it \
MERGES the two in one op, keeping `new`'s: a `merge_hints` row's `take`). Nothing to quote; they \
apply in order, all or nothing, and a bad id is rejected with the valid ids listed. Put the WHOLE \
change in ONE call: one call is one fit. What they cannot express (an event, a compartment) is a \
re-author with `antimony`.

With `base_version=N` the operations apply to version N's STORED model, and its search space and \
best fit are restored as the fit's start: fork and change in one call, never retyping an earlier \
model."""

_MAKE_CHANGES_PARAMETERS_ROUTE = (
    "`parameters` sets estimation BOUNDS and/or SCALE, never a value. Its entries apply after the "
    "reconcile, so one may target a parameter the same call introduces: give it beside a model "
    "route when a structural change needs bounds."
)

_MAKE_CHANGES_CONTRACT = f"""\
`n_starts` REPLACES the configured multistart count (`describe_problem` → \
`fit_n_starts_default`) rather than adding to it, and may be at most {MAX_FIT_STARTS_FACTOR}x it: \
omit it unless you want MORE; below it fits LESS hard. `starts_at_best` counts fresh starts \
apart from warm ones, which begin at or around the previous fit's optima: only the fresh count \
says how often the optimum is found. MORE is warranted once, when 0 to 2 fresh starts reach the \
best value; if the raised fit still shows 0 to 2, the optimum is a needle, and the model needs \
reformulating where it is that sharp (a switch-like term), not more starts. Starts multiply the \
fit's wall time, and do not help starts that run out of the per-start time limit \
(`ExitFlag.MAXTIME`).

A REFIT is a call with no model route: `parameters` and/or `n_starts` alone fit the current \
model again (version N's with `base_version=N`) as a new version. One with neither new bounds nor more starts than that model ran is refused.

ALL-OR-NOTHING: a change that is rejected, or whose fit produces no version, is REVERTED, so the \
working model is always a scored one. The result IS the fit, flat, in `get_fit_summary`'s \
fields, with the noise report, the `merge_hints` and the flagged observables cut to what is \
flagged or new since the parent version, and its pins split into `parameters_newly_at_bound` and \
the ids of those the parent held, `parameters_still_at_bound`. Read `applied` first: \
`false` → nothing moved — rejected, reverted because the fit failed, or already an earlier fitted \
version; `error`/`issues` says which, act on THAT. Then `committed`: `true` → a version exists, \
read its score; `false` with `no_change` → this model already IS the current version, so change \
something else. `messages` is everything the call had to say, in pipeline order, each tagged \
`level` and `source`.

`message` names the version the change was applied to (the working model, or `base_version`'s)⦃ \
and the best version. Without `base_version`, a change to a working model that is not the best \
and that you did not restore is refused once, naming the best; the same call sent again applies \
it. `improvement.is_new_best` marks the version that became the best, \
`progress.best_version`: the lowest score so far (`improvement_vs_best_prior` > 0), unless \
`new_best_declined` says why not. `progress` is `get_progress`'s moving part: a new best resets \
`non_improving_streak` only once the best has gained {STREAK_RESET_MIN_SCORE_GAIN:g} since the \
last reset (`streak_not_reset` says when one has not), a fit left unscored leaves the streak as it \
is, a streak of {NON_IMPROVING_STREAK_CAP} sets `should_stop`, and \
`score_improvement_over_last_n` spans the last {PLATEAU_WINDOW} fits⦄.

The score's `k` is `n_estimated` + `hardcoded_constants`: a numeric literal in the model's math \
is charged as a parameter too, bar a few structural ones, so freezing a fitted value costs what \
estimating it costs.

REJECTS an empty model, dropping (or re-estimating) a shipped-FIXED constant, leaving a \
condition-table target unused, an observation-model symbol (a scaling factor or noise sigma) in a \
rate law, a bound on a noise sigma the fit solves analytically (only the model moves it), and a \
THIRD widening of a still-pinned parameter: that is unidentifiability, see \
`get_fit_diagnosis(sections='parameters')`, not a bounds problem. Widening an estimated noise \
sigma past its data-driven cap only WARNS; it hides a missing mechanism behind inflated noise."""


def _make_changes_doc(statements_doc: str | None = None) -> str:
    """``make_changes``'s description for the model routes one registration offers: the text
    routes (``antimony``, ``model_edits``), or the ``statements`` route's paragraphs, never both."""
    routes = (
        ["`statements`", "`statement_edits`"] if statements_doc else ["`antimony`", "`model_edits`"]
    )
    required = (
        f"Give ONE model route per call; at least one of {' / '.join(routes)} / `parameters` / "
        "`n_starts` is required."
    )
    return "\n\n".join(
        p
        for p in (
            _MAKE_CHANGES_SUMMARY,
            None if statements_doc else _MAKE_CHANGES_TEXT_ROUTES,
            statements_doc,
            _MAKE_CHANGES_PARAMETERS_ROUTE + " " + required,
            _MAKE_CHANGES_CONTRACT,
        )
        if p
    )


_MAKE_CHANGES_DOC = _make_changes_doc()


# What the statements route holds, said once for every version of its paragraph.
_SET_SPECIES = (
    "A species the conditions set starts at what each period sets, and your rows change it as any "
    "other: one they only catalyse or regulate with stays there. Bind a dose in excess "
    "`nondepleting`, which leaves its free amount as set."
)
_HELD_INPUTS = _SET_SPECIES + " Each of `describe_problem`'s `inputs` follows its rule."


#: What the `statements` route's paragraph adds on a server holding constraints (:mod:`constraints`).
_MAKE_CHANGES_CONSTRAINTS = (
    "This session holds CONSTRAINTS (`list_constraints`): claims every model must make. A change whose model breaks one is refused, naming each it breaks and why "
    "(`error_cause: constraint_violated`), and NOTHING is applied; a parameter change is checked "
    "too, since a bound can turn a regulator. One ending `, directly` needs one row to claim it: "
    "its reaction, driven the stated way by the constraint's subject as its catalyst or as a "
    "regulator of the kind its rate records (state a regulator on the row it acts on, `..., "
    "inhibited by X`). One ending `, indirectly` needs the network your rows expand to to carry "
    "the subject's change to its object with the stated sign, traced through each reaction: a "
    "catalyst, a regulator as its rate records it, a substrate or a partner hands the change on "
    "to what the reaction makes or changes, and a competitor or an antagonist that takes a "
    "molecule away lowers what that molecule goes on to make. `not ..., directly` forbids a row "
    "that claims it, and `X does not affect Y` any claim (`directly`) or traced change "
    "(`indirectly`) from X to Y. `activates` and "
    "`inhibits` are about the forms the constraints declare active. Agents are matched by name: use "
    "the constraints' names."
)


def _make_changes_statements_doc(bound: bool = True, held: dict[str, Any] | None = None) -> str:
    """The `statements` route's paragraph, naming the entries the library holds right now.

    ``bound`` says whether the problem's entity map binds species or readouts. One that does not
    binds nothing of the model: the readouts, and how the condition targets enter, are the
    caller's own, and saying the map binds them would send it looking for names that do not exist.
    ``held`` (agent name to :class:`protocols.Input`) are the inputs held all the same, named with
    their protocol (e.g. a stimulus ramped within a period)."""
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
        "`statements` authors the WHOLE model from mechanism assertions rather than text: each row an INDRA "
        "statement, the claim in words, under a `prefix` naming its parameters; and, apart, in "
        "`bindings` under the same prefix, the maths behind it: the `policy` (mechanism) that "
        "turns it into reactions, and its parameter bindings. A row may be one line instead of "
        "JSON, and a binding too: `statements=['r1: A{pS10} phosphorylates B at T20'], "
        "bindings={'r1': 'two_step kf=k_on'}`, agents as NAME{pS10, -pT20, @LOCATION, !BOUND, "
        "~UNBOUND}, a statement's evidence a comment on its row (`r1: ... # [source] text`). Each "
        "row has its binding and each binding its row; a row is never written with its binding "
        "(`| two_step` is refused). See `get_model(format='statements')` for a model written "
        "that way.  Every INDRA statement type"
        + (f" but {', '.join(without)}" if without else "")
        + f" has `{mechanisms.INDRA_DEFAULT}`; the library "
        f"also holds {entries}. `list_mechanisms` shows each as the PySB rules it builds. When "
        "none fits, WRITE one in `mechanisms`, either as a `scheme` (its own rules, written like "
        "those, declaring their constants in `parameters`) or as a `rate` on an existing entry's "
        "reaction; it stays in the library for the rest of the run. A rate reads only the agents "
        "its row's statement names, so an agent that changes a reaction's rate without taking "
        "part is stated as its regulator: `r1: E phosphorylates S at T20, inhibited by C` claims "
        "C slows that reaction, and its binding's mechanism (`'r1': 'inhibited Ki=K_C'`) reads C "
        "as `inhibited_by` (`activated_by` speeds it, `modulated_by` claims no direction), or a "
        "scheme's rules bind it as the molecule `inhibited_by`. Each is "
        "recorded as the kind its rate shows within its parameters' bounds, again whenever a row "
        "or a bound changes: where that differs from the statement, the result says so and "
        "`get_model` marks it `(inhibits, from its rate)`. The rate must read every regulator its "
        "statement states, and change with each. A binding's parameters share a rate constant "
        "with other rows (`kf=k_on` on each), or with a parameter the problem ships, or fix it to "
        "a number (`kf=0.5`). "
        + (
            "The species the observables read are bound by the problem's entity map: name agents "
            "as it does. " + _HELD_INPUTS + " "
            if bound
            else "NOTHING is bound: every readout the observables read is yours to define in "
            "`readouts`, and every target the condition table sets is an input your rows must "
            "read: in a binding (`kf=\"k*dose\"`), which follows every period's value, or as an "
            "agent seeded at the target's id alone (`{\"agent\": \"FSK\", \"amount\": "
            "\"Fsk\"}`), whose species then takes that id for every period to set. "
            + _SET_SPECIES
            + " A seed's amount that is any other expression of a target (`2*Fsk`) takes the "
            "first period's value alone, which in a pre-equilibrated experiment is the "
            "pre-equilibration one. "
            + "".join(
                f"But `{name}` is HELD to the experiment's protocol, `{name} = {item.formula}`: "
                "read it as an agent in your rows; never seed, make or degrade it. "
                for name, item in sorted((held or {}).items())
            )
        )
        + "`seeds` declares what is present at t = 0, in the state its agent writes; "
        "without it every species no rule makes is seeded at an estimated `<name>_init` and every "
        "made one empty. `readouts` defines a measurement mapping as a formula over `amounts` "
        "(agents whose amounts it reads, or `amount(<pattern>)` in place; every readout reads "
        "at least one)"
        + (", entity-map ids" if bound else "")
        + " and parameter ids. The model is simulated reduced, with the same solution: a one-step "
        "rule's enzyme, which the step leaves as it was, is a factor of its rate "
        "(`<rule>_catalysed` = k·amount of the enzyme), and a protein whose sites no rule reads "
        "together on one molecule is tracked as blocks of them: the first in amounts, from its "
        "seed, each other in the fraction of the protein in each form (from `start_fraction` = "
        "1), a readout across blocks their product. The result's `network` counts "
        "species and reactions before and after and names the blocks (`split`).\n\n"
        "`statement_edits` CHANGES the current rows instead of giving them all again: `add_row` a "
        "row with its `binding`, `replace_row` the statement of the row with that prefix (its "
        "binding kept, or given anew beside it), `rebind` a row's binding alone (`{op: 'rebind', "
        "prefix: 'r1', binding: 'two_step kf=k_shared'}`: a new policy, or a constant shared), "
        "`remove_row` by prefix, and `set_seed`/`set_amount`/`set_readout` or "
        "`remove_seed`/`remove_amount`; with them `mechanisms` you write for the new bindings. "
        "They apply in "
        "order, all or nothing, and the model is rebuilt from the edited set. `base_version=N` "
        "applies them to version N's rows instead (with `statements`, it is the whole set given "
        "afresh), and starts the fit from version N's search space and best fit. "
        "`get_model(format='statements')` shows the rows."
    )


def _mechanisms_text(session, statement_types: list[str] | None) -> str:
    """The library, and the run's written mechanisms: those of ``statement_types`` each as its
    scheme (:mod:`schemes`), or without them an index, each statement type's mechanisms with their
    parameters, one line for the types whose entries are the same."""
    from . import mechanisms, schemes
    from .errors import MechanismError

    library = dict(mechanisms.REGISTRY)
    for key, definition in session.mechanisms.items():
        library[key] = mechanisms.written(definition, library)
    known = sorted({t for t, _ in library} - schemes.ABSTRACT)
    if unknown := sorted(set(statement_types or ()) - set(known)):
        return f"no mechanism for {', '.join(unknown)}; the statement types are {known}"
    blocks, index = [], {}
    for stmt_type, policy in sorted(k for k in library if k[0] in (statement_types or known)):
        definition = session.mechanisms.get((stmt_type, policy))
        try:
            text = schemes.describe(stmt_type, policy, library)
            names = schemes.rate_parameters(text)
        except MechanismError as exc:
            if definition is None:
                # A built-in that builds no rule for this type (ActiveForm's): no row can use it.
                text, names = f"({exc})", None
            else:
                # A written one reading what only its own statement names (`amount(PTEN)` of a
                # partner), absent from the placeholder: as written, with what it declares.
                text = definition.get("scheme") or f"({exc})"
                names = sorted(definition.get("parameters") or ())
        written = " (written this run)" if definition is not None else ""
        if statement_types:
            header = f"{stmt_type} | {policy}{written}"
            if names:
                header += f" | parameters: {', '.join(names)}"
            blocks.append(header + "\n" + "\n".join(f"  {line}" for line in text.splitlines()))
        elif names is not None:
            index.setdefault(stmt_type, []).append(f"{policy}({', '.join(names)}){written}")
    if statement_types:
        return "\n".join(blocks)

    def roles(stmt_type: str) -> str:
        """`` [roles ...]``, the names a rate reads its statement's agents by, a list role by
        position; empty for a type no placeholder statement stands for."""
        try:
            stmt = schemes.placeholder(stmt_type)
        except MechanismError:
            return ""
        return " [roles " + ", ".join(
            f"{r}_0..." if isinstance(getattr(stmt, r), list) else r for r in stmt._agent_order
        ) + "]"

    types_by_entries: dict[tuple[str, str], list[str]] = {}
    for stmt_type, entries in index.items():
        types_by_entries.setdefault((roles(stmt_type), ", ".join(entries)), []).append(stmt_type)
    return "\n".join(
        f"{', '.join(t)}{r}: {entries}" for (r, entries), t in types_by_entries.items()
    )


def _statement_record(session, version: int | None) -> dict | None:
    """The statement record of the current model, or of a stored ``version``; ``None`` for a model
    not built from statements."""
    import json

    from .provenance import statements_record

    if version is None:
        return session.model_statements or None
    path = session.provenance.versions_dir / f"v{int(version)}" / "statements.json"
    return statements_record(json.loads(path.read_text()) or None) if path.is_file() else None


def _statements_text(session, version: int | None) -> str:
    """A model built from statements, as the notation (:mod:`notation`): its written mechanisms,
    its rows one per line (INDRA JSON where the notation has no lossless form), then each row's
    binding, its seeds, amounts and readouts. The current model, or a stored ``version``."""
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
        declared = f", parameters {', '.join(d['parameters'])}" if d.get("parameters") else ""
        if d.get("scheme"):
            lines.append(f"mechanism {d['policy']}: {d['statement_type']}, scheme{declared}")
            lines += [f"  {line.strip()}" for line in d["scheme"].strip().splitlines() if line.strip()]
            continue
        lines.append(
            f"mechanism {d['policy']}: {d['statement_type']} like "
            f"{d.get('based_on', mechanisms.INDRA_DEFAULT)}, rate {d['rate']}{declared}"
        )
    for row in record["rows"]:
        bare = {"prefix": row["prefix"], "statement": row["statement"]}
        derived = (row.get("emission") or {}).get("derived")
        lines.append(
            notation.render(bare, derived)
            or json.dumps(bare | ({"derived": derived} if derived else {}))
        )
    for row in record["rows"]:
        binding = (record.get("bindings") or {}).get(row["prefix"], {})
        text = notation.render_binding(binding, row["statement"].get("type"), library)
        lines.append(f"binding {row['prefix']}: {text or json.dumps(binding)}")
    for seed in record.get("seeds") or []:
        lines.append(f"seed {seed['name']}: {notation.agent_text(seed['agent'])} = "
                     f"{seed['amount']}")
    for name, agent in (record.get("amounts") or {}).items():
        lines.append(f"amount {name} = {notation.agent_text(agent)}")
    for readout, formula in (record.get("readouts") or {}).items():
        lines.append(f"readout {readout} = {formula}")
    return "\n".join(lines)


def _get_model_doc(statements: bool) -> str:
    """``get_model``'s description, offering the ``statements`` format only where the route is."""
    formats = (
        "'antimony' (the default), 'sbml', or 'statements': a model built from statements as its "
        "rows, then each row's binding (`binding r1: two_step kf=k_on`, which `make_changes` takes "
        "as `bindings={'r1': 'two_step kf=k_on'}`), in the notation `make_changes` takes, where "
        "its antimony is the reduced model it simulates (enzymes as rate factors, independent "
        "sites as blocks)"
        if statements
        else "'antimony' (the default) or 'sbml'"
    )
    return (
        "A model as text: the CURRENT one, or with `version=N` a prior version's stored model — "
        "read that before a `base_version=N` fork, not `get_version`, which also returns the "
        f"version's record and parameter table. `format` is {formats}. For the "
        "structure alone `get_model_structure` is the cheaper read.\n\n"
        "The trailing `const ...;` line is DERIVED (a parameter is constant unless declared "
        "`var`): read past it; no operation addresses it."
    )


# ------------------------------------------------------------------ model tools
def _register_model_tools(
    mcp: FastMCP,
    route: Callable[[], _Ctx],
    statements: bool = False,
    progress_signals: bool = True,
) -> None:
    # `make_changes` takes an argument named `statements`; the route flag goes by this name in it.
    statements_route = statements
    if statements:

        @mcp.tool(annotations=_READ_ONLY)
        def list_mechanisms(
            statement_types: Annotated[list[str] | None, BeforeValidator(_unwrap_listish)] = None,
        ) -> str:
            """The mechanisms a `statements` row can use: the library's own and those written this
            run. Without `statement_types`, an index: each statement type's roles, the names a rate
            reads its agents by, and its mechanisms with the `parameters` each declares,
            `policy(parameters)`, one line for the types whose entries are the same. With them,
            each of those types' mechanisms as the PySB rules it builds. Read the rules of the types your rows use before choosing a row's policy, and to write
            a new one as a `scheme`: copy the nearest entry and edit it. Each is written over the
            statement type's ROLES; `@` gives the rate, and a name in it is one of the `parameters`
            its header lists, which a row's binding binds as `kf=...` (`bindings={'r1': 'two_step
            kf=k_on'}`). A scheme reads no agent its row's statement does not name: a molecule
            other than a role is a partner the statement writes a role bound or not bound to (`C`
            of `enz{~C}`) or a regulator the statement states, which `amount(inhibited_by)` reads
            and the rules bind as the molecule `inhibited_by`.
            """
            return _mechanisms_text(route().session, statement_types)

    @mcp.tool(annotations=_READ_ONLY, description=_get_model_doc(statements))
    def get_model(format: str = "antimony", version: int | None = None) -> str:
        ctx = route()
        session = ctx.session
        # Without the route there is no statements format: it is refused as any unknown one is.
        if statements_route and (format or "").lower() == "statements":
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
                # burns the caller's turns while showing zero errors in `get_errors`.
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
            # the caller's turns while showing zero errors in `get_errors`. This call is the
            # natural read before a `base_version=N` fork, so a wrong N is a reachable, ordinary
            # mistake.
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
    def get_model_structure(
        reactions: str | None = None, initial_conditions: str | None = None
    ) -> dict[str, Any]:
        """The model, structured: species, parameters, `n_reactions`, and `readouts`, each
        `readout_*` measurement mapping with the expression your model defines it by (null while it
        has none). Empty until a model is built.

        `reactions` and `initial_conditions` are regular expressions, and each list comes only when
        asked for: `reactions` lists the reactions whose id or rate law it matches (id,
        reversibility, rate law), `initial_conditions` the species whose id it matches; '' lists
        all. A species' initial condition says what the simulation starts it at (`t0`, one value or
        one per experiment) and what governs it: `t0_source` is `estimated` (the DEFAULT, so a
        declared t=0 is only a starting guess), `fixed`, `condition_table` (set per condition, with
        no parameter row) or `rule`, and `ic_parameter` names the parameter carrying it. Read it to
        check that a t=0 you meant to pin took.
        """
        ctx = route()
        session = ctx.session
        # The working model: HEAD's, or while a change is being fitted, that change's.
        version = None if session.change_began is not None else session.provenance.current_version
        out = ctx.run(
            "get_model_structure",
            lambda: model_mod.model_structure(session, reactions, initial_conditions),
            {"reactions": reactions, "initial_conditions": initial_conditions},
        )
        return _labelled(session, out, version, "that change's model, not a version yet")

    def _make_changes(
        justification: str,
        antimony: str | None = None,
        model_edits: _ModelOps = None,
        base_version: int | None = None,
        parameters: _ParameterChanges = None,
        references: _References = None,
        n_starts: int | None = None,
        statements: _StatementRows = None,
        bindings: _Bindings = None,
        mechanisms: _MechanismDefs = None,
        seeds: _SeedRows = None,
        readouts: _Readouts = None,
        amounts: _Amounts = None,
        statement_edits: _StatementEdits = None,
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
        bindings = {
            prefix: b.model_dump(exclude_none=True) if isinstance(b, RowBinding) else b
            for prefix, b in (bindings or {}).items()
        } or None
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
        if not (antimony or ops or rows or edits or parameters) and n_starts is None:
            # Names NO other tool: advice to `restore_version` would send the agent to a tool its
            # client may withhold, the defection-trigger class the sibling rejections avoid.
            return ctx.reject_operation(
                "make_changes",
                "make_changes: nothing to do — give "
                + (
                    "`statements` (the whole model as rows), `statement_edits` (changes to the "
                    "current rows, seeds, amounts and readouts), "
                    if statements_route
                    else "`model_edits` (structured operations on the committed model — the "
                    "usual route), `antimony` (author the model wholesale), "
                )
                + "`parameters` (to set bounds/scale), `n_starts` (to refit with more starts), "
                "or a combination."
                + (
                    " `base_version` only picks the version they apply to. To RETURN to it "
                    "because it was better you need nothing: every fitted version is kept and "
                    "the BEST one stands, wherever HEAD ends up."
                    if base_version is not None
                    else ""
                ),
                {"base_version": base_version},
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
        if (seeds or readouts or amounts) and edits:
            return ctx.reject_operation(
                "make_changes",
                "make_changes: with `statement_edits`, change seeds, amounts and readouts by "
                "their own edits (`set_seed`, `set_amount`, `set_readout`, `remove_seed`, "
                "`remove_amount`), not "
                "`seeds`/`amounts`/`readouts`, which give the whole set with `statements`. "
                "NOTHING was applied.",
                {"n_seeds": len(seeds or [])},
                cause="statement_edit",
            )
        if bindings and edits:
            return ctx.reject_operation(
                "make_changes",
                "make_changes: with `statement_edits`, a row's binding changes by its own edit "
                "(`add_row` with its `binding`, `rebind`), not `bindings`, which give every row's "
                "binding with `statements`. NOTHING was applied.",
                {"bindings": sorted(bindings)},
                cause="statement_edit",
            )
        if (bindings or mechanisms or seeds or readouts or amounts) and not (rows or edits):
            return ctx.reject_operation(
                "make_changes",
                "make_changes: `bindings`, `mechanisms`, `seeds`, `readouts` and `amounts` belong "
                "to `statements` rows; send them together"
                + (" (or `mechanisms` with `statement_edits`)" if statements_route else "")
                + ". NOTHING was applied.",
                {"n_mechanisms": len(mechanisms or []), "n_seeds": len(seeds or [])},
            )
        # A change waits for no other: a client whose call timed out resends it, and queued behind
        # the fit it is already waiting for, it times out again (PEtab-MCP#108).
        if (began := session.change_began) is not None:
            return ctx.reject_operation(
                "make_changes",
                _fit_in_progress(began, session.change_progress),
                {},
                cause=FIT_IN_PROGRESS,
            )
        # With progress signals off the agent is not told which version is best, and a refusal
        # here would tell it.
        if progress_signals and base_version is None:
            call = {
                "antimony": antimony,
                "model_edits": ops,
                "parameters": parameters,
                "n_starts": n_starts,
                "statements": rows,
                "bindings": bindings,
                "mechanisms": mechanisms,
                "seeds": seeds,
                "readouts": readouts,
                "amounts": amounts,
                "statement_edits": edits,
            }
            if (refusal := _implicit_base(session, call)) is not None:
                return ctx.reject_operation("make_changes", refusal, {}, cause="implicit_base")
        # One critical section for the change and its fit, so nothing can fit in between
        # (`session.lock` is an RLock, so the nested acquisitions inside each call are fine).
        with session.change():
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
                    rows, bindings, seeds, readouts, amounts, edit_summary = (
                        ops_mod.apply_statement_edits(base, edits)
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
                            bindings,
                            definitions,
                            [s.model_dump() if isinstance(s, SeedRow) else s for s in seeds or []],
                            readouts,
                            amounts,
                        )
                    )
                except PetabMcpError as exc:
                    ctx.raise_if_infrastructure(exc)
                    return ctx.reject_operation(
                        "make_changes",
                        str(exc),
                        {"n_statements": len(rows)},
                        cause=getattr(exc, "cause", None),
                    )
                except Exception as exc:
                    # A crash, not a refusal: logged where every tool's crash lands (`_Ctx.run`),
                    # so the provenance error log (`get_errors`) records it, and raised.
                    ctx.log_error("make_changes", {"n_statements": len(rows)}, exc)
                    session.logger.exception("make_changes crashed building the statements' model")
                    raise
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
                    fixed = (
                        reconcile_mod.fixed_table_parameters(session.problem)
                        if any(
                            isinstance(o, dict) and o.get("op") in ("rename", "set_assignment")
                            for o in ops
                        )
                        else None
                    )
                    antimony, op_summary = model_edits_mod.apply_model_edits(
                        base_src, ops, fixed=fixed
                    )
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
                # as they do for `model_edits`. With no model route at all it is a REFIT of that
                # version: its stored source and rows, rebuilt as a fork with no edit.
                try:
                    base_parameter_df = session.provenance.get_parameter_df(int(base_version))
                    if not (antimony or rows):
                        v, prov = int(base_version), session.provenance
                        antimony = prov.get_model_source(v) or prov.get_model_antimony(v)
                        statement_record = prov.get_statements(v)
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
            session.renamed = model_edits_mod.renames(ops or [])
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
                        # branch had widened and the agent had to retype the old bounds. Tolerant
                        # of a changed symbol set, and applied BEFORE this call's own `parameters`
                        # entries so an explicit bound still wins (see
                        # operations._composed_parameter_edit).
                        #
                        # POSITIONAL, not a keyword: `run_operation` is `(op_name, fn, *args)` with
                        # no **kwargs, so a keyword here raises TypeError on EVERY model-changing
                        # call — outside both except paths, so it escapes make_changes entirely and
                        # no session can edit the model at all.
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
                session.renamed = []
                session.submitted_source = None
                session.submitted_statements = None
            if out.get("applied") and not out.get("no_change") and antimony:
                # The submitted text becomes this branch's source for the next structured edit.
                # A rejected or fit-reverted call leaves the model where it was (`applied` false),
                # and a de-dup no-op leaves the session on the version it duplicates (see
                # `edit_then_fit`), so in both the source correctly stays put.
                session.model_source = antimony
                session.model_statements = statement_record
                if written is not None:
                    session.mechanisms = written
            if rows:
                out["network"] = statement_record["network"]
            if edit_summary:
                out["statement_edits_applied"] = edit_summary
            if op_summary:
                # The diff, PRE-CLASSIFIED. Every earlier analysis had to reconstruct this from
                # re-rendered text with id- and regex-keyed heuristics, and four of those
                # reported artifacts as findings; here the operation IS the classification.
                out["model_edits_applied"] = op_summary
            return out

    # One tool, `make_changes`, either way, and one implementation, `_make_changes`. FastMCP builds
    # the input schema from the registered function's signature, so the statements route is offered
    # by registering `make_changes_statements` and the text routes by registering
    # `make_changes_antimony`; each wrapper takes only its routes and passes them on. A server
    # offers one family, never both: `_make_changes` itself is never registered.
    if statements:
        session = route().session
        statements_doc = _make_changes_statements_doc(
            bound=bool(set(session.entities) - {i.id for i in session.inputs}),
            held={getattr(session.entities.get(i.id), "name", i.id): i for i in session.inputs},
        )
        if session.constraints is not None:
            statements_doc += "\n\n" + _MAKE_CHANGES_CONSTRAINTS
            if session.constraint_updates == "any":
                statements_doc += (
                    " They may change during the session (`set_constraints`): `list_constraints` "
                    "shows the set in force."
                )
        def make_changes_statements(
            justification: str,
            base_version: int | None = None,
            parameters: _ParameterChanges = None,
            references: _References = None,
            n_starts: int | None = None,
            statements: _StatementRows = None,
            bindings: _Bindings = None,
            mechanisms: _MechanismDefs = None,
            seeds: _SeedRows = None,
            readouts: _Readouts = None,
            amounts: _Amounts = None,
            statement_edits: _StatementEdits = None,
        ) -> dict[str, Any]:
            return _make_changes(
                justification=justification,
                base_version=base_version,
                parameters=parameters,
                references=references,
                n_starts=n_starts,
                statements=statements,
                bindings=bindings,
                mechanisms=mechanisms,
                seeds=seeds,
                readouts=readouts,
                amounts=amounts,
                statement_edits=statement_edits,
            )

        mcp.tool(
            description=_make_changes_doc(statements_doc=statements_doc),
            name="make_changes",
            annotations=_MAKE_CHANGES_ANNOTATIONS,
        )(make_changes_statements)
        return

    # No entity map: nothing binds a statement-built model to the observables, so the route would
    # fail every call. Withheld at the schema like every other route, which also leaves this
    # registration byte-identical to the one before the route existed.
    def make_changes_antimony(
        justification: str,
        antimony: str | None = None,
        model_edits: _ModelOps = None,
        base_version: int | None = None,
        parameters: _ParameterChanges = None,
        references: _References = None,
        n_starts: int | None = None,
    ) -> dict[str, Any]:
        return _make_changes(
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
    )(make_changes_antimony)


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
        """The parameter table, one row per parameter: id, bounds, scale, `nominalValue` (the fitted
        value, LINEAR whatever the scale), estimate flag, `role` and `status`. Pass
        `parameter_ids` and/or `role` to FILTER: on a large problem the full table is 100+ rows
        re-sent in your history every later turn.

        `role` is what the parameter IS: `kinetic`, `noise_sigma` or `observable_scaling` (the
        OBSERVATION model: never in a rate law, estimated or not), `initial_condition`, `exponent`,
        or `constant`. `status` is `estimated` or `fixed`, and the `fixed` rows are the COMPLETE
        set of constants you may treat as known: every estimated parameter counts toward k however
        tight its bounds, a symbol you introduce is added estimated, and initial conditions are
        estimated by DEFAULT (an IC of exactly 1 stays fixed; one of exactly 0 gets no row). Do not
        assume a `sigma_*` or `*_init` is fixed: read its `status`.

        `in_model: false` marks a parameter the PROBLEM ships that your model does not declare yet.
        Those are yours to build on: declare the name and the row comes back exactly as shown, and
        a shipped-FIXED one must stay declared. Values come only from the fit; bounds and scale
        change through `make_changes`' `parameters`.
        """
        ctx = route()
        session = ctx.session
        rows = param_mod.parameters_to_records(
            session.problem, shipped=session.shipped_parameter_df
        )
        # Server-side filters, added when a 118-row parameter table (re-sent in the caller's
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
                # {ok:false,...} dict fails that validation with a RuntimeError that ENDS THE
                # CLIENT'S SESSION — observed on 6 of 14. A raise is the shape a list-returning
                # tool can express: FastMCP turns it into a tool error, which the client hands
                # back to read and correct, exactly like any other rejected argument.
                explained = {
                    u: why
                    for u in unknown
                    if (why := reconcile_mod.not_a_parameter(session.problem, u, known_ids))
                }
                exc = ValidationError(unknown_ids_message(unknown, known_ids, explained))
                # RECORD it before raising. ctx.run would have done this; raising past it meant
                # a session that burned turns on bad ids showed ZERO errors in get_errors —
                # invisible afterwards too.
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
#: away, for when BIC stalls — which is when it has something to say.
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
        """The current fit's metrics without per-point records, a cheap score check: the same fields
        every `make_changes` result carries, uncut. Models rank by `criterion_value`, the
        complexity-penalised score `criterion` names (`bic` unless configured otherwise; lower is
        better); the other scores are reported beside it, and the `k` they charge is
        `n_estimated` + `hardcoded_constants`.

        The flags to act on: `parameters_at_bound` (a parameter that re-pins after one widening is
        UNIDENTIFIABLE, not under-explored), `noise_sigma_report` with `noise_sigma_inflated` /
        `noise_sigma_collapsed` (an estimated sigma absorbing misfit or overfitting the noise: fix
        the model's form, not the noise), the `residual_structure` verdict, and `merge_hints`
        (parameter pairs the data cannot separate whose merge the criterion would pay for).⦃ With
        estimated sigmas, `reduced_chi2` and `chi2_gof_pvalue` sit near their null whatever the
        fit and say nothing about adequacy.⦄
        """
        ctx = route()
        session = ctx.session
        engine = ctx.engine
        out = ctx.run("get_fit_summary", lambda: fitting_mod.fit_summary(session, engine))
        return _labelled(session, out, out.get("version"), "no committed fit")

    @mcp.tool(annotations=_READ_ONLY)
    def get_run_timing() -> dict[str, Any]:
        """Cumulative server-side timing for this session, split by cost kind.

        `{compile_seconds, fit_seconds, queue_seconds, n_compiles, n_fits}`. `compile_seconds` is
        the AMICI model-BUILD time (the one-time C++ compile per unique structure — ONE build,
        shared by the fit and every objective evaluation); `fit_seconds` is the pyPESTO multistart
        OPTIMIZATION time, of which `queue_seconds` its starts waited for a worker with none of
        them running. They accumulate only when work actually happens, so a compile/fit cache hit
        adds nothing.
        These are server-side wall times (a breakdown of the heavy compute), distinct from the
        client's `compute_seconds` tool-busy measure, which also covers reads/SQL.
        """
        ctx = route()
        session = ctx.session
        return ctx.run("get_run_timing", session.run_timing)

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
        """The SQL tables `query_data` exposes, with their columns and row counts."""
        ctx = route()
        session = ctx.session
        return ctx.run("list_data_tables", lambda: data_mod.table_schema(session))

    @mcp.tool(annotations=_READ_ONLY)
    def get_fitting_log(tail_lines: int = 200) -> dict[str, Any]:
        """The tail of the pyPESTO/Fides multistart optimizer log (`tail_lines`, 0 for all):
        per-start objective values, convergence, parameters pinned at their bounds,
        ill-conditioning."""
        ctx = route()
        session = ctx.session

        def run() -> dict[str, Any]:
            if not isinstance(tail_lines, int) or tail_lines < 0:
                raise ValidationError(f"tail_lines must be an integer >= 0, got {tail_lines!r}.")
            return logging_setup.read_pypesto_log(session.provenance.root, tail_lines)

        return ctx.run("get_fitting_log", run, {"tail_lines": tail_lines})

    @mcp.tool(annotations=_READ_ONLY)
    def query_data(sql: str, max_rows: int = 50, offset: int = 0) -> dict[str, Any]:
        """Run a DuckDB SQL query over the data, on a private copy (nothing it does reaches the
        session).

        TWO NAMING CONVENTIONS, and they do not mix. The PEtab tables keep PEtab's camelCase:
        `measurements(observableId, experimentId, time, measurement, observableParameters,
        noiseParameters)`, `conditions(conditionId, targetId, targetValue)`,
        `experiments(experimentId, time, conditionId)`, `observables(observableId, ...)`,
        `parameters(parameterId, ...)`. The DERIVED tables use snake_case:
        `simulations(observable_id, experiment_id, time, ...)`, `simulations_history` (the same
        plus `version`), `state_trajectories(experiment_id, time, entity_id, value)`, and
        `optimization_starts` / `optimization_trace`. The measurement table's key column is
        `experimentId` — NOT `experiment_id` (that is `simulations`); an empty `experimentId`
        ('') is a measurement of the model simulated as is. Do not carry a column list from a
        `simulations` query onto `measurements`, or the reverse.

        `simulations` has one row per point of the last fit: `measurement`, `simulation`,
        `residual`, `residual_weighted`, `noise`, and the row's `noise_parameters` — residuals
        live there, not on `measurements`. `state_trajectories` has the last fit's hidden states:
        each species and assignment rule (`entity_id`) at the times its experiment was measured.
        `simulations_history` is the only way to compare versions point by point, e.g.
        `SELECT observable_id, experiment_id, time,
        max(CASE WHEN version=3 THEN residual_weighted END) AS v3,
        max(CASE WHEN version=4 THEN residual_weighted END) AS v4
        FROM simulations_history WHERE version IN (3, 4) GROUP BY 1, 2, 3
        ORDER BY abs(v4) - abs(v3) DESC`. A measurement reaches its inputs through its experiment:
        `measurements.experimentId = experiments.experimentId`, then `experiments.conditionId =
        conditions.conditionId`. `list_data_tables` has every table's columns.

        Aggregate only NUMERIC columns, filter `residual_weighted IS NOT NULL` for residual
        statistics, and prefer aggregations: rows are capped at `max_rows`. When `truncated` is
        true the result carries `next_offset`: page with it rather than raising `max_rows`, with an
        explicit `ORDER BY`, or pages can overlap or skip.
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
            # Generate the HEAD simulation dataframe only when the query actually reads
            # `simulations` or `state_trajectories`: current_simulation() is an AMICI
            # build+simulate — minutes of C++ compile on a cold/resumed session — and a query
            # over the static tables or over `simulations_history` (assembled from per-version
            # files on disk, no simulation involved) must not pay it.
            if data_mod.query_reads_head_simulation(sql) and (
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
        out = ctx.run(
            "query_data", run, {"sql": logged_sql, "max_rows": max_rows, "offset": offset}
        )
        # The version of the fit tables (`simulations`, `parameters`, the optimization's).
        version = (session.simulation or {}).get("version")
        return _labelled(session, out, version, "no committed fit")

    @mcp.tool(annotations=_READ_ONLY)
    def get_fit_diagnosis(
        sections: str = "residuals",
        worst_n: int = 8,
        top_n: int = 8,
        max_observables: int = 12,
        version: int | None = None,
    ) -> dict[str, Any]:
        """The size-bounded DETAIL behind the verdicts every fit result already carries: where the
        fit is off, and which parameters it cannot pin. `sections` picks the half: 'residuals'
        (the default), 'parameters', or 'all'. Neither compiles a model, so the cost is the
        payload (2-8 kB each), re-sent in your history on every later turn: ask for 'parameters'
        when the score has stalled and you need to know what to cut, not by reflex.

        It describes HEAD's fit, or with `version=N` stored version N's, read from what that fit
        stored and never refit; the reply's `version` names which.

        RESIDUALS: per observable the weighted `chi2`, `mean_weighted_residual` (bias), the worst
        |residual| and two sign-based structure tests (`sign_test`, `runs_test`, each with `p` and
        `low_power`), plus the `worst_n` worst points (measurement vs simulation) and the
        noise-sigma report. Read it for WHERE and HOW the fit is off, not WHETHER: the rolled-up
        `residual_structure` verdict rides on the fit result. An estimated sigma absorbs misfit, so
        two ranked pointers say where it did: `sigma_heterogeneity` (per sigma, a Levene `p` that
        its series share one spread, and the series with the largest `rms` weighted residual) and
        `noise_sigma.fitted_over_data_noise`, each `low_power` on few points;
        `structure_tests_caveat` says when no structure flag is weak evidence. On a large problem
        `per_observable` is a worst-first subsample (`max_observables` by chi2, plus a few by bias
        and structure flag, with an `omitted` block); `max_observables=0` returns every row.

        PARAMETERS (`identifiability`), from the Fisher information at the optimum and the
        multistart spread: per-parameter standard error and fold uncertainty, marginal and partial
        correlations, the near-flat eigen-directions (the parameter COMBINATIONS the data cannot
        pin), and a ranked `candidates` list. `merge_candidates` prices giving a pair ONE SHARED
        value: `dnll`, cheapest first; `dscore`, NEGATIVE when the merge lowers the score;
        `move_scaled`, how far the tie moves a parameter. Only same-role, same-scale pairs with
        neither member at a bound are priced (`n_ties_skipped` counts the rest), and the fit
        result's `merge_hints` are the cheapest few. Remedies are MODEL edits, in this order: drop
        the term; take its limiting (irreversible or instantaneous) form; merge a pair, into the
        combination the data constrain (`ka*kb` → one `kab`) or one shared value; last, for a pure
        scale degeneracy where every value fits equally well, write one member as 1 in the rate
        law (a 1 is never charged). The expansion is local, so refit and let the score decide
        each.
        """
        ctx = route()
        session = ctx.session

        def run() -> dict[str, Any]:
            want = _diagnosis_sections(sections)
            head = session.provenance.current_version
            if version is not None and version != head:
                # Off what that version's fit stored: its records and the parameter verdict
                # beside them. Neither section simulates or falls back to HEAD.
                fit = session.stored_fit(version)
                out: dict[str, Any] = {"version": version}
                if "residuals" in want:
                    out.update(
                        data_mod.fit_diagnosis(
                            fit,
                            worst_n,
                            max_observables=max_observables or None,
                            no_records=data_mod.NO_STORED_RECORDS_STATUS.format(version=version),
                        )
                    )
                if "parameters" in want:
                    out["identifiability"] = identifiability_mod.version_diagnosis(
                        fit.simulation, version
                    )
                return out
            out = {"version": head if head >= 0 else None}
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
                # not take the residual half with it, and a caller that reads only the residual
                # half must not lose it to a FIM failure. The
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
            "get_fit_diagnosis",
            run,
            {"sections": sections, "worst_n": worst_n, "top_n": top_n, "version": version},
        )


# ------------------------------------------------- held-out validation (not the agent's)
def _register_validation_data_tools(mcp: FastMCP, route: Callable[[], _Ctx]) -> None:
    """Register ``evaluate_validation_data`` — a read for whoever runs the session, never the
    agent's.

    Registered only when :func:`create_server` was given a held-out measurement file. A client that
    puts an agent on this server must leave the tool out of the agent's tool set. Three things have
    to stay out of the agent's reach for a held-out score to mean anything: the held-out DATA (kept
    outside the problem tree the server is pointed at, so no tool reads it), the SCORE (which would
    otherwise be a fitness signal the agent could optimise against by trial and error, turning the
    held-out set into training data one tool call at a time), and the tool's FAILURES, whose
    messages can name held-out experiments, conditions and observables. Those are therefore kept out
    of ``errors.jsonl``, which ``get_errors`` returns (see :meth:`_Ctx.run`).

    Registering it as a normal MCP tool anyway, rather than reaching into the server's internals,
    because a client has no other way in: the session lives in the server's process, and the model
    + its fitted values exist only there.
    """

    @mcp.tool(annotations=_READ_ONLY)
    def evaluate_validation_data(n_starts: int | None = None) -> dict[str, Any]:
        """Score the CURRENT model against HELD-OUT measurements, in the held-out set's own mode.

        For whoever runs the session, not for the agent building the model (see the registrar's
        docstring): a predictive check, not a modelling tool.

        Changes neither the model nor its fit. A fixed-θ set is one objective evaluation of the
        model as it stands, at its current fitted values, against data it was not fitted to. A
        TRANSFER set (one that brings its own parameter table) REFITS the model's estimated
        parameters on the held-out system with an ``n_starts`` multistart (the session's default
        when omitted) and reports the optimum; the fitted point is not written back. Returns
        ``{status, mode, nll, k, n, chi2, n_experiments, observables, records_file, ...}``, where
        ``records_file`` is the head version's ``validation_simulations.json``: the score with one
        record per held-out point. See :mod:`petab_mcp.validation_data` for the statuses and the two
        modes.
        """
        ctx = route()
        session = ctx.session
        return ctx.run(
            "evaluate_validation_data",
            lambda: validation_data_mod.score_session(
                session, ctx.engine, ctx.held_out, n_starts=n_starts
            ),
            {"n_starts": n_starts},
            record=False,
        )


# ------------------------------------------------------ provenance/version tools
#: ``wait_for_fit``'s description. The tool blocks on the server, not a cheap status call to
#: poll, because each poll is a model call that re-sends the agent's whole context (250k+ tokens on
#: perica_egf, PEtab-MCP#172). It is async, so a wait holds none of the worker threads that the
#: sync tools and the running fits share.
_WAIT_FOR_FIT_DOC = (
    "Wait for the change being fitted on this session (a `make_changes` whose call timed out "
    f"while its fit ran on) for up to `timeout_s` seconds, at most {WAIT_FOR_FIT_MAX_S}: keep it "
    "under your client's tool timeout. The wait is the server's, so call this again while the fit "
    "runs, not another tool. `status` is `committed`, with the new version's `list_versions` row; "
    "`still_running`, with `elapsed_s`, `phase` (`compiling` its model, `queued` with none of its "
    "starts running on the shared pool, else `running`), `starts_done` of `n_starts` (0 while the "
    "model compiles), `starts_queued` (not yet begun), `best_so_far` (the lowest "
    "neg_log_likelihood among the starts done) and `remaining_s` (the starts left at the pace of "
    "those done while running; null before one is); `no_version`, when the change "
    "ended without one (refused or its fit failed, which `get_errors` says, or its model is one a "
    "version already scores); or `idle`, when no change has run."
)

#: Most bytes a ``get_version`` reply may take as the MCP result it is sent as. The MCP Python
#: client refuses an SSE event over 1 MiB (``mcp.client.streamable_http.DEFAULT_MAX_SSE_EVENT_SIZE``)
#: and ended a lucarelli run on a 1755-point ``simulation`` (PEtab-MCP#231); the rest is margin for
#: the JSON-RPC envelope and the SSE framing.
_REPLY_MAX_BYTES = 768 * 1024

#: ``get_version``'s description. A part over :data:`_REPLY_MAX_BYTES` is summarised, not paged:
#: ``simulations_history`` already pages a version's records in SQL, and a second pager over the
#: same rows is the growing menu the README's design notes rule out.
_GET_VERSION_DOC = (
    "A prior version's record (every score, ⦃the new-best bookkeeping, ⦄the noise model's report, "
    "and `edit_warnings`: every warning its change and lint gave, in full, where its "
    "`make_changes` result folded a repeated one into a count), Antimony model and parameter "
    "table, to INSPECT it; for its model alone, "
    "`get_model(version=N)`. `include` adds what its fit stored beside it: 'simulation' (one "
    "record per data point) and 'optimization' (every start with its trace). A reply is at most "
    f"{_REPLY_MAX_BYTES // 1024} KiB, under the 1 MiB the MCP Python client takes in one message: "
    "a part that would take it past that comes back without its rows, 'simulation' with each "
    "observable's chi2 and 'optimization' with each start's objective, and the part's `omitted` "
    "says where its rows are. A version's per-point rows are `query_data` on "
    "`simulations_history`, paged, and its fit's diagnosis `get_fit_diagnosis(version=N)`."
)


def _wire_bytes(out: dict[str, Any]) -> int:
    """Bytes of the MCP result FastMCP sends for a tool returning ``out``: its JSON twice, as text
    content and as ``structuredContent``, the text's quotes escaped."""
    result = ToolResult(structured_content=out)
    wire = {
        "content": [block.model_dump(by_alias=True, exclude_none=True) for block in result.content],
        "structuredContent": result.structured_content,
    }
    return len(json.dumps(wire, separators=(",", ":"), ensure_ascii=False).encode())


def _version_rows(session: PetabSession, metas: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Each of ``metas`` as its ``list_versions`` row (:func:`petab_mcp.provenance.summary`), with
    the session's constraints it breaks as ``violates``."""
    broken = session.constraint_violations()
    return [
        prov_mod.summary(meta) | {"violates": broken[meta["version"]]}
        if meta.get("version") in broken
        else prov_mod.summary(meta)
        for meta in metas
    ]


def _fit_outcome(session: PetabSession) -> dict[str, Any]:
    """``wait_for_fit``'s answer: the running change's progress, or how the last one ended."""
    start, began = session.change_from, session.change_began
    if began is not None:
        progress, now = session.change_progress, time.monotonic()
        remaining = progress.remaining_s(now)
        return {
            "status": "still_running",
            "elapsed_s": round(now - began),
            "phase": progress.phase,
            "starts_done": progress.done,
            "n_starts": progress.total,
            "starts_queued": progress.queued,
            "best_so_far": progress.best,
            "remaining_s": None if remaining is None else round(remaining),
        }
    if start is None:
        return {"status": "idle"}
    version = session.provenance.last_version
    if version <= start:
        return {"status": "no_version"}
    (row,) = _version_rows(session, [session.provenance.get_meta(version)])
    return {"status": "committed", **row}


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
    def list_versions(since: int | None = None) -> list[dict[str, Any]]:
        """Every model version (each a fit), one row each: which structures have been tried and
        how they scored. A row is its `version`, the `parent` it was derived from, its
        `neg_log_likelihood`, `criterion_value` (the score this session ranks by)⦃, `k`,
        `is_new_best` and `residual_flagged` (a residual test flagged one of its observables)⦄⦅ and
        `k`⦆. `since=N` lists only the versions after N. ⦃`is_new_best` marks each version that
        became the best model (the last one is `get_progress`'s `best_version`); every⦄⦅Every⦆
        version is kept, and the best one stands whatever comes after it. Where the session holds
        constraints, a version they break (committed before they changed) carries `violates`, their
        ids: it cannot be restored and does not count as the best. A version's whole record, every
        score⦃ and why it did or did not become the best⦄, is `get_version(N)`."""
        ctx = route()
        session = ctx.session
        return _version_rows(session, session.provenance.list_versions(since))

    @mcp.tool(annotations=_READ_ONLY, description=_WAIT_FOR_FIT_DOC)
    async def wait_for_fit(timeout_s: float = WAIT_FOR_FIT_S) -> dict[str, Any]:
        ctx = route()
        session = ctx.session
        deadline = time.monotonic() + min(max(float(timeout_s), 0.0), WAIT_FOR_FIT_MAX_S)
        while session.change_began is not None and time.monotonic() < deadline:
            await anyio.sleep(_WAIT_POLL_S)
        return await anyio.to_thread.run_sync(
            lambda: ctx.run("wait_for_fit", lambda: _fit_outcome(session), {"timeout_s": timeout_s})
        )

    @mcp.tool(annotations=_READ_ONLY, description=_GET_VERSION_DOC)
    def get_version(
        version: int, include: list[Literal["simulation", "optimization"]] | None = None
    ) -> dict[str, Any]:
        ctx = route()
        session = ctx.session
        parts = tuple(include or ())
        omitted = [part for part in prov_mod.VERSION_PARTS if part not in parts]

        def run() -> dict[str, Any]:
            out = session.provenance.get_version(version, parts)
            # ids × points values, which query_data's state_trajectories pages instead.
            (out.get("simulation") or {}).pop("state_trajectories", None)
            if omitted:
                out["omitted"] = (
                    f"{' and '.join(omitted)}: get_version(include=[...]); per point, "
                    "simulations_history"
                )
            # The largest part first, so a smaller one comes back whole where it can.
            for part in sorted(set(parts), key=lambda p: len(json.dumps(out[p])), reverse=True):
                if _wire_bytes(out) <= _REPLY_MAX_BYTES:
                    break
                if out[part] is not None:
                    out[part] = data_mod.version_part_summary(part, out[part], version)
            return out

        return ctx.run("get_version", run, {"version": version, "include": list(parts)})

    @mcp.tool(annotations=_READ_ONLY)
    def diff_versions(from_version: int, to_version: int) -> dict[str, Any]:
        """Diff two versions: Antimony unified diff + parameter add/remove/change + a `fit`
        block with each version's neg_log_likelihood/chi2⦃ and the neg-log-likelihood delta
        (positive = the `to` version fits better)⦄."""
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
        with `make_changes`, which refuses an exact re-author of a fitted version; this reuses the
        existing fit and records a clean restore. Find the target with `list_versions`
        ⦃(the best model is the last `is_new_best` one), ⦄then `restore_version(v)`. HEAD moves back
        to that version: no version is added or erased, and the restore is recorded as a
        `rollback` event in the provenance log."""
        ctx = route()
        session = ctx.session
        return ctx.run(
            "restore_version",
            lambda: session.restore_version(version, justification or None),
            {"version": version},
        )

    @mcp.tool(annotations=_READ_ONLY)
    def get_progress() -> dict[str, Any]:
        """Early-stop-chain progress for this session — a cheap counter read (no fit/simulation).

        `{n_fitted_ops, non_improving_streak, streak_cap, best_version, criterion, best_score,
        best_gof_pvalue, score_improvement_over_last_n, plateau_window, should_stop, stop_reason,
        streak_not_reset}`, the last only where the last fit was a new best that did not reset the
        streak.
        `best_score` is under `criterion` — the complexity penalty this session ranks by.
        `best_version` is the best model so far (the session's result), the lowest-scoring version,
        the last whose fit says `is_new_best`;
        `non_improving_streak` counts consecutive fitted ops since a new best last counted as
        progress (`make_changes` states the gain that takes).
        `should_stop` is set (with a `stop_reason`) once the session has stopped making progress —
        the hard non-improving-streak cap, or a plateau once the best model is statistically
        adequate, which only a chi2 goodness of fit against FIXED noise sigmas can show: with
        estimated sigmas only the streak cap stops.
        There is no cap on the TOTAL number of fits: a session that keeps genuinely improving keeps
        going, and only a stall stops it.
        A streak-cap `stop_reason` whose best model is shown not adequate also carries the
        structural-inadequacy nudge: a refit or an equivalent rewrite cannot beat the optimum its
        structure already has, so the next model must change the structure (add a latent state, add
        a feedback reaction, raise interaction order), not re-fit the same one.
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


# ------------------------------------------------------------------ constraint tools
def _verdicts(session, constraints) -> dict[str, Any]:
    """Each constraint's verdict on the current model (:meth:`PetabSession.constraint_verdicts`): as
    its commit checked it where its record keeps that under these constraints, else traced now."""
    return session.constraint_verdicts(session.provenance.current_version, constraints)


def _register_constraint_tools(
    mcp: FastMCP, route: Callable[[], _Ctx], updates: bool = False
) -> None:
    """``list_constraints``, on a server started with constraints, and ``set_constraints`` where it
    was started with ``constraint_updates='any'`` (``updates``): absent otherwise, not refused
    (:mod:`constraints`). Told rather than read off the session: a group resolves no session while
    it registers."""
    from . import constraints as constraints_mod

    @mcp.tool(annotations=_READ_ONLY)
    def list_constraints() -> dict[str, Any]:
        """The session's CONSTRAINTS, the claims every model must make, as written (`, directly`: one
        row claims it; `, indirectly`: the network your rows expand to carries it, traced from its
        subject), each with what makes it hold on the current model: the `rows` that claim it, or
        whose reactions carry the traced change onto its object; or `why` it does not.
        `active_forms` are the forms `activates` and `inhibits` are about; `updates` says whether
        `set_constraints` may change them (`any`) or not (`off`)."""
        session = route().session
        constraints = session.constraints or constraints_mod.Constraints(())
        verdicts = _verdicts(session, constraints)
        return {
            "constraints": [
                {"id": c.id, "text": c.text}
                | ({"source": c.source} if c.source else {})
                | verdicts[c.id].stored()
                for c in constraints.items
            ],
            "active_forms": [f"{f.id}: {f.text}" for f in constraints.active_forms],
            "updates": session.constraint_updates,
        }

    if not updates:
        return

    @mcp.tool(annotations=_RESTORE_ANNOTATIONS)
    def set_constraints(constraints: list[str], rationale: str) -> dict[str, Any]:
        """Replace the session's CONSTRAINTS, the claims every model must make, with a whole new
        set, each a line as `list_constraints` lists them (`c1: EPO increases SOCS3, indirectly`,
        `c2: not EPO phosphorylates STAT5, directly`, `a1: STAT5{pY694} is active`). Every change
        from here on is checked against them. A stored version they break is flagged in
        `list_versions`, can no longer be restored, and no longer counts as the best. The result
        says whether the update loosened or tightened the claims, and which versions break it,
        HEAD included; the next change must build a model that makes them. A version is checked
        against a constraint it was not committed under by tracing its model, which takes seconds
        to a minute for each version and indirect constraint. `rationale` says why, and is logged
        with the update."""
        ctx = route()
        session = ctx.session
        args = {"n_constraints": len(constraints)}
        if not (rationale or "").strip():
            return ctx.reject_operation(
                "set_constraints",
                "set_constraints: give the `rationale` for the update; it is logged with it. "
                "NOTHING was changed.",
                args,
            )
        if (began := session.change_began) is not None:
            return ctx.reject_operation(
                "set_constraints",
                f"set_constraints: {FIT_IN_PROGRESS}: a change is still being fitted on this session "
                f"(it began {time.monotonic() - began:.0f}s ago), checked against the constraints in "
                "force, so NOTHING was changed: send the update once it ends.",
                args,
                cause=FIT_IN_PROGRESS,
            )
        groundings = {agent.name: dict(agent.db_refs) for agent in session.entities.values()}
        try:
            new = constraints_mod.parse(constraints, groundings)
        except constraints_mod.ConstraintError as exc:
            return ctx.reject_operation(
                "set_constraints",
                f"set_constraints: {exc}. NOTHING was changed.",
                args,
                cause="unreadable_constraints",
            )
        return ctx.run("set_constraints", lambda: session.set_constraints(new, rationale), args)


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
    constraints: str | None = None,
    constraint_updates: str = "off",
) -> _Ctx:
    """Open the session on the problem at ``yaml_path``: its :class:`PetabSession`, the
    :class:`AmiciEngine` wired into it, and its held-out set. The arguments are
    :func:`create_server`'s, but for ``fit_engine``: the server's
    :class:`~petab_mcp.dask_engine.DaskEngine`, which the session's fits share, or ``None``."""
    # Read before anything is built: a refused map leaves no provenance store or build directory.
    entities = None
    if entity_map:
        from . import seeding as seeding_mod

        entities = seeding_mod.load_entity_map(entity_map)
    loaded = None
    if constraints:
        from . import constraints as constraints_mod

        # With the map, before the session: refused if any does not read, as a prior is, and
        # grounded as the rows are, by the map's names.
        groundings = {agent.name: dict(agent.db_refs) for agent in (entities or {}).values()}
        loaded = constraints_mod.read(constraints, groundings)
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
    if entities is not None:
        session.entities = entities
    session.text_route = entity_map is None
    if loaded is not None:
        session.constraints = loaded
        session.constraint_updates = constraint_updates
        # Logged: the claims the session holds models to are part of what its versions mean.
        session.provenance.append_event(
            {"type": "constraints", "source": str(constraints), "constraints": loaded.texts()}
        )
        # A resumed store may hold versions these break: they rank no more than a later set's do.
        if broken := session.constraint_violations():
            session.reseed_best(set(broken))
    if score_shipped_model and session.model_authored():
        fitting_mod.score_at_nominal(session, engine)
    held_out = None
    if validation_measurements:
        # Read ONCE, as the session opens: a per-call read would make the score depend on a file
        # the session may no longer see (a server can run from a scratch copy of the problem) and
        # would let a mid-session edit of the file move the bar the model is held to.
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
    constraints: str | None = None,
    constraint_updates: str = "off",
    cluster: Any | None = None,
    fix_pointers: bool = True,
    progress_signals: bool = True,
) -> FastMCP:
    """Build a FastMCP server bound to the PEtab problem at ``yaml_path``.

    ``skills_dir`` overrides where tool-usage skills are loaded from; when omitted
    the ``_skills`` directory bundled with the package is discovered automatically.

    ``auto_fit`` (default on) makes every commit fit + simulate before snapshotting;
    ``auto_fit_starts`` is the default pyPESTO multistart count (overridable per commit via
    ``n_starts``). Set ``auto_fit=False`` to skip the compile-heavy fit hook (tests/dev).

    ``criterion`` chooses the complexity penalty the session ranks models by — ``bic`` (default),
    ``caic``, ``aic``, ``aicc``, ``hqc`` or ``tic`` (see :mod:`criteria`). ``None`` means "as the
    environment says" (``PETAB_MCP_CRITERION``, else plain BIC). Resolved once by the session
    and held as ``session.criterion``, which every scorer is handed: there is no process state,
    so two servers in one process can rank by different criteria and nothing can change a
    session's criterion under it.

    ``validation_measurements`` is a PEtab measurement TSV of HELD-OUT data — data the model is
    never fitted to — with the set's own condition/parameter tables beside it where it has any
    (:func:`petab_mcp.validation_data.read_held_out`). Giving it registers
    ``evaluate_validation_data``, which scores the session's model on that data for whoever runs
    the session; it is not for the agent (see ``_register_validation_data_tools``). Omit it and no
    such tool exists.

    Initial conditions are ESTIMATED by default; an IC of exactly 1 (a unit normalization) is held
    FIXED and an IC of exactly 0 (an empty pool) gets no parameter at all (see
    ``reconcile.ensure_parametric_initial_conditions``); to keep another IC fixed, ship it as a
    fixed parameter row.

    ``score_shipped_model`` (default off) compiles a problem that ships a model and scores it at
    the table's nominal values before serving, with no fit (:func:`fitting.score_at_nominal`); a
    model-less problem ignores it.

    ``entity_map`` is a JSON file binding the problem's species to INDRA agents
    (:func:`seeding.load_entity_map`). Giving it offers ``make_changes`` the ``statements`` route
    alone, without the ``antimony`` and ``model_edits`` routes, so the model is built from
    statements; omit it and the text routes are offered alone. A server never offers both.

    ``constraints`` is a file of causal claims every model must make (:mod:`constraints`), each
    ending ``, directly`` or ``, indirectly``: ``make_changes`` refuses a model that breaks one,
    naming it. It needs ``entity_map``. A file that does not read is refused here. ``constraint_updates``
    (``off`` by default) set to ``any`` registers ``set_constraints``, for whoever drives the
    session to replace them; it needs ``constraints`` (a file holding none starts with none).

    ``cluster`` (a ``distributed`` cluster or a pool's scheduler address, the ``dask`` extra) runs
    every fit's starts on its workers (:mod:`petab_mcp.dask_engine`); without one, each fit runs
    its starts in local processes. The workers import the compiled models from ``build_root`` and
    read each fit's problem there, so on more than one machine it must be on a filesystem they
    share.

    ``fix_pointers`` (default on) sends every result whole. Off, the server withholds the numbers
    that point the agent at what to change or where the fit is off (the merge hints, the payoff of
    a tie, the per-observable residual tables and outliers, which observables a verdict flags; see
    :mod:`withheld`) and still computes and stores them.

    ``progress_signals`` (default on) keeps the numbers that tell the agent how well it is doing:
    the best version and which fit became it, score differences against the parent or the best,
    and the chi2 goodness of fit. Off withholds them from every tool but ``get_progress`` and
    ``evaluate_validation_data``, which whoever runs the session reads (:mod:`progress_signals`,
    :class:`_ProgressSignals`); each fit's own numbers stay, and the stored record keeps
    everything.
    """
    if constraints and entity_map is None:
        raise ValidationError(
            "constraints need an entity_map: a constraint is a claim the statements route's rows "
            "make, and only that route builds from statements."
        )
    if constraint_updates not in ("off", "any"):
        raise ValidationError(f"constraint_updates is 'off' or 'any', not {constraint_updates!r}.")
    if constraint_updates == "any" and not constraints:
        raise ValidationError(
            "constraint_updates='any' needs constraints: give a file, one holding none to start "
            "with none."
        )
    fit_engine = None
    if cluster is not None:
        from .dask_engine import DaskEngine

        fit_engine = DaskEngine(cluster, resolve_build_root(build_root))
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
        constraints=constraints,
        constraint_updates=constraint_updates,
    )

    # Load tool-usage skills up front so the server instructions only advertise
    # `list_skills`/`get_skill` when skills are actually available (registered below).
    resolved_skills_dir = skills_dir or skills_mod.default_skills_dir()
    # Offered whenever a map was given, even an empty one: a problem whose observables all read
    # formulas, and whose seeds are the statements' own, binds nothing in it.
    statements = entity_map is not None
    skills = [
        s
        for s in skills_mod.load_skills(resolved_skills_dir)
        if s.route in (None, "statements" if statements else "text")
    ]
    instructions = (
        SERVER_INSTRUCTIONS
        + (_SKILLS_INSTRUCTION if skills else "")
        + (_STATEMENTS_SKILL_INSTRUCTION if any(s.name == "statements-grammar" for s in skills) else "")
    )
    mcp = FastMCP(
        "petab-mcp",
        instructions=signals_mod.render(instructions, progress_signals),
        middleware=[
            _ProgressSignals(progress_signals),
            *([] if fix_pointers else [_FixPointers()]),
            _InfrastructureErrors(
                ctx.session.logger, ctx.engine.build_root, ctx.session.provenance.root
            ),
            _Cancellation(),
            _Heartbeat(ctx.session),
        ],
    )

    # Register the tool + resource groups; each tool asks `route` for the context per call.
    def route() -> _Ctx:
        return ctx

    _register_read_tools(mcp, route)
    _register_model_tools(mcp, route, statements=statements, progress_signals=progress_signals)
    _register_parameter_tools(mcp, route)
    _register_fit_tools(mcp, route)
    if ctx.held_out is not None:
        _register_validation_data_tools(mcp, route)
    if ctx.session.constraints is not None:
        _register_constraint_tools(mcp, route, updates=constraint_updates == "any")
    _register_provenance_tools(mcp, route)
    _register_resources(mcp, route)
    # Register the skills loaded above as `list_skills`/`get_skill` tools plus
    # `petab://skills` resources (no-op when no skills were found).
    skills_mod.register_skills(mcp, skills)

    return mcp
