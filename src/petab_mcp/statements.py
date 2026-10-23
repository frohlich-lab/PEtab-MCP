"""The mechanism vocabulary: what one model edit ASSERTS.

This module owns the bindings, which agents each mechanism applies to. Without it a binding
exists only as a reference inside the authored model text, so nothing can be revised across its
instances. Here it is declared by the agent at edit time and stored beside the model, never
recovered from it: inference from finished rate laws is pattern matching run backwards, and the
model does not record which species are modified forms of which.

Statements are ``indra.statements.Statement`` instances. Nearly all are INDRA's own —
Phosphorylation, Complex, Activation, IncreaseAmount, Translocation and the rest cover what a
molecular model asserts. The one type here covers what INDRA's vocabulary does not, and the gap is
not an oversight on INDRA's part: its types describe molecular biochemistry, while much
mechanistic modelling is cell-population and epidemiological dynamics. Confining the agent to
INDRA's set would exclude those models outright.

It is written as an INDRA subclass rather than as a parallel vocabulary so that one dispatch
serves both: the PySB assembler, Agent grounding, ``Preassembler`` deduplication and belief all
key on the base class, not on membership of INDRA's own set. What that does not buy is INDRA's web
services and network viewers, which enumerate the built-in types; nothing on the fitting or
scoring path depends on them.

Grounding is OPTIONAL here, deliberately. A molecular agent carries ``db_refs`` and stays
comparable against the corpus; a population compartment carries none, and neither does an unnamed
mediator — which needs no type of its own, being an ungrounded agent in an ordinary statement.
An absent grounding warns rather than rejects (:func:`ungrounded`).

A row's statement arrives as INDRA's JSON, whose reader skips what it does not know and leaves
what is missing ``None``; :func:`problems` reads it first, so a misspelt or missing field is
refused by name, with what it takes, instead of building a different model or crashing.
"""

from __future__ import annotations

import inspect
from collections.abc import Collection
from typing import Any

from indra.statements import (
    Agent,
    Conversion,
    Modification,
    RegulateActivity,
    RegulateAmount,
    Statement,
    activity_types,
    get_all_descendants,
    get_statement_by_name,
    modtype_conditions,
)
from indra.statements import statements as _indra_statements
from indra.statements.resources import InvalidResidueError, get_valid_residue
from indra.statements.statements import mk_str


class PopulationTransition(Statement):
    """One population becomes another: a compartment, cell state or class moving to the next.

    Not INDRA's ``Conversion``, which converts molecular species with explicit reactant and
    product lists and carries a catalyst slot that has no reading here.
    """

    _agent_order = ["source", "target"]

    def __init__(self, source: Agent, target: Agent, evidence=None) -> None:
        super().__init__(evidence)
        self.source, self.target = source, target

    def matches_key(self) -> str:
        return mk_str((type(self).__name__, *(a.matches_key() for a in self.agent_list())))

    def set_agent_list(self, agent_list: list[Agent]) -> None:
        for name, agent in zip(self._agent_order, agent_list, strict=True):
            setattr(self, name, agent)

    def to_json(self, use_sbo: bool = False, matches_fun=None) -> dict:
        json_dict = super().to_json(use_sbo, matches_fun)
        for name in self._agent_order:
            json_dict[name] = getattr(self, name).to_json()
        return json_dict

    @classmethod
    def _from_json(cls, json_dict: dict) -> PopulationTransition:
        return cls(*(Agent._from_json(json_dict[name]) for name in cls._agent_order))

    def __str__(self) -> str:
        return f"{type(self).__name__}({', '.join(str(a) for a in self.agent_list())})"


# INDRA deserialises by looking the ``type`` field up in its OWN module, and ``stmts_from_json``
# DROPS a statement it cannot resolve with only a logged warning — so without this a stored
# PopulationTransition vanishes on restore rather than failing.
setattr(_indra_statements, PopulationTransition.__name__, PopulationTransition)


#: Fields INDRA's JSON writes for every statement type, besides the type's own.
_COMMON = ("type", "id", "evidence", "belief", "supports", "supported_by", "matches_hash", "sbo")
#: The roles holding a list of agents; every other role holds one.
_LIST_ROLES = ("members", "obj_from", "obj_to")
#: The fields of an agent and of each of its conditions, as INDRA's JSON writes them.
_AGENT = ("name", "db_refs", "mods", "mutations", "bound_conditions", "location", "activity", "sbo")
_CONDITIONS = {
    "mods": ("mod_type", "residue", "position", "is_modified"),
    "mutations": ("position", "residue_from", "residue_to"),
    "bound_conditions": ("agent", "is_bound"),
}
_ACTIVITY = ("activity_type", "is_active")


def problems(statement: Any) -> list[str]:
    """What keeps ``statement`` (INDRA's JSON) from building as written, each with what to write
    instead; empty when nothing. A type INDRA has and the library does not passes, so its row meets
    the library-gap refusal, which counts it and says to write a mechanism.

    INDRA's reader skips a field it does not know and leaves a missing one ``None``, so a misspelt
    role (``enzyme``) built a model without its enzyme, a misspelt agent field (``mod``) an agent
    without its modification, and a mod without a ``mod_type`` crashed the assembly raw."""
    if not isinstance(statement, dict):
        return [f"a statement is INDRA's JSON, an object with its `type`, not {statement!r}"]
    kind = statement.get("type")
    types = {cls.__name__ for cls in get_all_descendants(Statement)}
    if not isinstance(kind, str) or kind not in types:
        return [f"`type` {kind!r} is no statement type{_close(kind, types)}"]
    cls = get_statement_by_name(kind)
    roles = list(cls._agent_order)
    fields = roles + [
        name for name in inspect.signature(cls.__init__).parameters
        if name not in ("self", "evidence", *roles)
    ]
    out = [
        f"{kind} has no field {field!r}{_close(field, fields)}, its fields are {fields}"
        for field in sorted(set(statement) - set(fields) - set(_COMMON))
    ]
    for role in roles:
        value = statement.get(role)
        if role in _LIST_ROLES:
            if role == "members" and not value:
                out.append(f"{kind} needs `members`, a list of agents: [{{\"name\": \"A\"}}, ...]")
            elif value is not None and not isinstance(value, list):
                out.append(f"`{role}` is a list of agents: [{{\"name\": \"A\"}}, ...]")
            else:
                out += [p for i, a in enumerate(value or []) for p in agent_problems(a, f"{role}[{i}]")]
        elif value is not None:
            out += agent_problems(value, role)
        elif not (
            (role == "enz" and issubclass(cls, Modification))
            or (role == "subj" and issubclass(cls, (RegulateAmount, Conversion)))
        ):
            out.append(f"{kind} needs `{role}`, an agent: {{\"name\": \"A\"}}")
    if "residue" in fields:
        out += _residue(statement.get("residue"), "`residue`")
    for field in ("obj_activity", "activity"):
        if field in fields and statement.get(field) is not None:
            out += _activity(statement[field], f"`{field}`")
    return out


def agent_problems(agent: Any, where: str) -> list[str]:
    """What keeps ``agent`` (INDRA's Agent JSON, at ``where``) from building as written."""
    if not isinstance(agent, dict):
        return [f"`{where}` is an agent, an object like {{\"name\": \"A\"}}, not {agent!r}"]
    out = _unknown(agent, _AGENT, f"`{where}`", "an agent")
    if not isinstance(agent.get("name"), str) or not agent["name"]:
        out.append(f"`{where}` has no `name`")
    for field, allowed in _CONDITIONS.items():
        conditions = agent.get(field) or []
        if not isinstance(conditions, list):
            out.append(f"`{where}.{field}` is a list of objects with {list(allowed)}")
            continue
        for i, condition in enumerate(conditions):
            at = f"`{where}.{field}[{i}]`"
            if not isinstance(condition, dict):
                out.append(f"{at} is an object with {list(allowed)}")
                continue
            out += _unknown(condition, allowed, at, "it")
            if field == "mods":
                mod_type = condition.get("mod_type")
                if not isinstance(mod_type, str) or mod_type not in modtype_conditions:
                    known = sorted(modtype_conditions)
                    out.append(
                        f"{at} has no `mod_type`, one of {known}" if mod_type is None else
                        f"{at} has `mod_type` {mod_type!r}{_close(mod_type, known)}, which is "
                        f"none of {known}"
                    )
                out += _residue(condition.get("residue"), f"{at}.residue")
            elif field == "bound_conditions":
                out += agent_problems(condition.get("agent"), f"{where}.{field}[{i}].agent")
    activity = agent.get("activity")
    if activity is not None and not isinstance(activity, dict):
        out.append(f"`{where}.activity` is an object with {list(_ACTIVITY)}")
    elif activity is not None:
        out += _unknown(activity, _ACTIVITY, f"`{where}.activity`", "it")
        if activity.get("activity_type") is not None:
            out += _activity(activity["activity_type"], f"`{where}.activity.activity_type`")
    return out


def with_defaults(statement: dict) -> dict:
    """``statement`` with the field INDRA's constructor defaults but its JSON reader leaves
    ``None``: a regulated activity is ``activity`` unless it names one (``kinase``)."""
    if issubclass(get_statement_by_name(statement["type"]), RegulateActivity):
        if statement.get("obj_activity") is None:
            return {**statement, "obj_activity": "activity"}
    return statement


def _unknown(value: dict, allowed: Collection[str], where: str, what: str) -> list[str]:
    return [
        f"{where} has no field {field!r}{_close(field, allowed)}, {what} has {list(allowed)}"
        for field in sorted(set(value) - set(allowed))
    ]


def _residue(residue: Any, where: str) -> list[str]:
    try:
        get_valid_residue(residue)
    except (InvalidResidueError, AttributeError):
        return [f"{where} {residue!r} is no amino acid, write its one-letter code ('S')"]
    return []


def _activity(value: Any, where: str) -> list[str]:
    """An activity outside INDRA's vocabulary names a site no other statement writes, so the rule
    reading it never fired (the notation refuses one too)."""
    if isinstance(value, str) and value in activity_types:
        return []
    known = sorted(activity_types)
    return [f"{where} {value!r}{_close(value, known)} is none of INDRA's activities {known}"]


def _close(word: Any, known: Collection[str]) -> str:
    from .model_edits import closest_symbols

    close = closest_symbols(str(word), sorted(known)) if isinstance(word, str) else []
    return f" (did you mean {' or '.join(repr(c) for c in close)}?)" if close else ""


def named_agents(stmt: Statement) -> list[Agent]:
    """Every agent ``stmt`` names, as it names it: each role's agent and, within it, each partner
    it is written bound or not bound to (``E{~C}``), theirs in turn. INDRA's English renders them
    all ("E not bound to C phosphorylates S"), and they are all a mechanism's rate may read."""
    out: list[Agent] = []

    def visit(agent: Agent) -> None:
        out.append(agent)
        for condition in agent.bound_conditions:
            visit(condition.agent)

    for agent in stmt.real_agent_list():
        visit(agent)
    return out


def ungrounded(stmt: Statement) -> list[str]:
    """Agent names in ``stmt`` carrying no ``db_refs``, for the warning path.

    ``TEXT`` does not count: it is the string a reader saw, not an identity.
    """
    return [a.name for a in stmt.real_agent_list() if not set(a.db_refs) - {"TEXT"}]
