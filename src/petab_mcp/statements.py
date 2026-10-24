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

A statement may also name the agents that regulate its rate without taking part in it, a field
INDRA does not have: ``inhibited_by``, ``activated_by`` and ``modulated_by`` (:data:`REGULATORS`),
each a list of agents. "RAF phosphorylates MEK, inhibited by sorafenib" is a claim about that one
reaction at the population level: sorafenib slows it, whatever binds what. INDRA can say neither
part of that. A bound condition (``RAF{~sorafenib}``) claims a complex the rule never forms, and
an ``Inhibition(sorafenib, RAF)`` claims every reaction RAF catalyses, since an INDRA statement
names agents and never another statement. The mechanism's rate reads each regulator
(:mod:`mechanisms`), and each is recorded as the kind the row's net rate shows, read off the
model the rows build (:mod:`regulation`): inhibited where it never rises, activated where it never
falls, modulated where neither can be shown. Where that differs from the statement, it is kept
beside it (:attr:`mechanisms.Emission.derived`), reported to the agent, and shown as the rate's.
The net rate is the reaction's initial rate, since where a step sits decides which way its own
rate moves the reaction. A regulator that rate does not change with acts on a step it cannot see,
a product's degradation, and is that step's, stated on its own row: each claim keeps one meaning,
and :func:`as_indra` never puts a regulator on another reaction. INDRA's
reader and assemblers skip the fields, so :func:`english` writes them, :func:`indra_part` is
what INDRA reads, and :func:`as_indra` projects each signed regulator onto a statement INDRA can
read, for a causal graph built from INDRA statements.
"""

from __future__ import annotations

import inspect
import re
from collections.abc import Callable, Collection
from copy import deepcopy
from typing import Any

from indra.statements import (
    Activation,
    ActivityCondition,
    Agent,
    BoundCondition,
    Complex,
    Conversion,
    DecreaseAmount,
    Evidence,
    Gap,
    Gef,
    IncreaseAmount,
    Inhibition,
    ModCondition,
    Modification,
    RegulateActivity,
    RegulateAmount,
    RemoveModification,
    SelfModification,
    Statement,
    Translocation,
    Transphosphorylation,
    activity_types,
    get_all_descendants,
    get_statement_by_name,
    modtype_conditions,
    stmts_from_json,
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


#: The statement fields naming what regulates its rate, beside INDRA's own (module docstring),
#: each with the words the notation and :func:`english` use.
REGULATORS = {"inhibited_by": "inhibited by", "activated_by": "activated by",
              "modulated_by": "modulated by"}
#: The name a mechanism's rate reads a regulator by: its kind, and its position where the statement
#: states several of that kind (``inhibited_by_1``), as a list role is read.
REGULATOR_NAME = re.compile(rf"(?:{'|'.join(REGULATORS)})(?:_\d+)?")
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
        f"{kind} has no field {field!r}{_close(field, [*fields, *REGULATORS])}, its fields are "
        f"{fields}"
        for field in sorted(set(statement) - set(fields) - set(_COMMON) - set(REGULATORS))
    ]
    for field in REGULATORS:
        value = statement.get(field)
        if value is not None and not isinstance(value, list):
            out.append(f"`{field}` is a list of agents: [{{\"name\": \"A\"}}, ...]")
        else:
            out += [
                p for i, a in enumerate(value or []) for p in agent_problems(a, f"{field}[{i}]")
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


def indra_part(statement: dict) -> dict:
    """``statement`` without its :data:`REGULATORS`: what INDRA's reader reads."""
    return {k: v for k, v in statement.items() if k not in REGULATORS}


def regulators(statement: dict) -> dict[str, list[Agent]]:
    """The agents ``statement`` (INDRA's JSON) states regulate its rate, by kind; the kinds it
    states none of left out."""
    return {
        kind: [Agent._from_json(a) for a in statement[kind]]
        for kind in REGULATORS if statement.get(kind)
    }


def signed(statement: dict, derived: dict[str, str] | None = None) -> list[tuple[str, Agent, bool]]:
    """Each regulator ``statement`` states, with the kind it is recorded as and whether its rate
    gave that kind: as stated, or the kind its rate shows instead, from ``derived`` (kind stated and
    position -> kind recorded, a row's :attr:`mechanisms.Emission.derived`)."""
    out = []
    for kind, agents in regulators(statement).items():
        for i, agent in enumerate(agents):
            shown = (derived or {}).get(f"{kind}_{i}")
            out.append((shown or kind, agent, shown is not None))
    return out


def english(statement: dict, derived: dict[str, str] | None = None) -> str:
    """``statement`` (INDRA's JSON) in English: INDRA's own sentence and the regulators it states,
    which INDRA's assembler does not read. "RAF phosphorylates MEK, inhibited by sorafenib." A
    regulator its rate records as another kind (``derived``, :func:`signed`) reads as that kind,
    marked so: "inhibited by sorafenib (from its rate)"."""
    from indra.assemblers.english import EnglishAssembler
    from indra.assemblers.english.assembler import _assemble_agent_str

    def text(agent: Agent) -> str:
        return _assemble_agent_str(agent).agent_str

    (stmt,) = stmts_from_json([with_defaults(indra_part(statement))])
    if isinstance(stmt, PopulationTransition):  # a type INDRA's assembler does not know
        sentence = f"{text(stmt.source)} becomes {text(stmt.target)}"
    else:
        sentence = EnglishAssembler([stmt]).make_model().rstrip(".")
    clauses: dict[tuple[str, bool], list[Agent]] = {}
    for kind, agent, from_rate in signed(statement, derived):
        clauses.setdefault((kind, from_rate), []).append(agent)
    for (kind, from_rate), agents in clauses.items():
        sentence += f", {REGULATORS[kind]} " + " and ".join(map(text, agents))
        sentence += " (from its rate)" if from_rate else ""
    return sentence + "."


def as_indra(statement: dict, derived: dict[str, str] | None = None) -> list[Statement]:
    """``statement`` (a row's, INDRA's JSON) as INDRA statements: its own, then one for each signed
    regulator it states, which a causal graph assembled from INDRA statements would otherwise drop.

    INDRA's statements name agents, never another statement, so each projection claims less than
    the regulator it comes from, and more widely. A regulator of a reaction with a catalyst
    regulates the catalyst's activity there: "RAF phosphorylates MEK, inhibited by sorafenib" gives
    ``Inhibition(sorafenib, RAF, 'kinase')``, which a reader takes to slow RAF's other kinase
    reactions too. One of a reaction without a catalyst changes the amount of what the reaction
    makes, signed by the claim and by what the reaction does to it: "SKI is produced, inhibited by
    X" gives ``DecreaseAmount(X, SKI)``, "SKI is degraded, inhibited by X" ``IncreaseAmount(X,
    SKI)``. ``modulated_by`` claims no direction, and INDRA has no unsigned statement over agents,
    so it is not projected. Each regulator projects as the kind its rate records (``derived``,
    :func:`signed`), so a claim the rate reverses projects reversed and one it cannot sign not at
    all.
    Each projection carries the statement's evidence, annotated with the claim it comes from
    (:func:`english`) under ``regulates``, and one whose kind its rate gave with ``from_rate``."""
    (stmt,) = stmts_from_json([with_defaults(indra_part(statement))])
    claim = english(statement, derived)

    def evidence(from_rate: bool) -> list[Evidence]:
        marked = {"from_rate": True} if from_rate else {}
        return [
            Evidence(source_api=ev.source_api or "petab_mcp", text=ev.text,
                     annotations={**ev.annotations, "regulates": claim, **marked})
            for ev in stmt.evidence or [Evidence()]
        ]

    catalyst, activity = _catalyst(stmt)
    out = [stmt]
    for kind, regulator, from_rate in signed(statement, derived):
        if kind == "modulated_by":
            continue
        sign = -1 if kind == "inhibited_by" else 1
        if catalyst is not None:
            cls = Inhibition if sign < 0 else Activation
            out.append(cls(regulator, catalyst, activity, evidence=evidence(from_rate)))
            continue
        for product, effect in _products(stmt):
            cls = IncreaseAmount if sign * effect > 0 else DecreaseAmount
            out.append(cls(regulator, product, evidence=evidence(from_rate)))
    return out


def _catalyst(stmt: Statement) -> tuple[Agent | None, str]:
    """The agent that drives ``stmt`` without being changed by it, and the activity it does so with:
    an enzyme's implied by its modification (``kinase`` for a phosphorylation), a subject's as
    written; ``None`` for a statement with none."""
    from indra.assemblers.pysb.assembler import mod_acttype_map

    if isinstance(stmt, Modification):
        return getattr(stmt, "enz", None), mod_acttype_map.get(type(stmt), "activity")
    if isinstance(stmt, (Gef, Gap)):
        return (stmt.gef, "gef") if isinstance(stmt, Gef) else (stmt.gap, "gap")
    if isinstance(stmt, (RegulateActivity, RegulateAmount, Conversion)) and stmt.subj is not None:
        activity = stmt.subj.activity
        return stmt.subj, activity.activity_type if activity else "activity"
    return None, "activity"


def _products(stmt: Statement) -> list[tuple[Agent, int]]:
    """What ``stmt`` makes, each with ``1``, or ``-1`` for what a removal takes away: the agent in
    the state the statement leaves it in (:data:`_REACTIONS`). Empty for a statement that makes
    nothing."""
    return _reaction(stmt)[1]


def _reactants(stmt: Statement) -> list[Agent]:
    """What ``stmt`` turns into its :func:`_products`, in the state it finds it (:data:`_REACTIONS`).
    Empty for a statement that consumes nothing (a synthesis)."""
    return _reaction(stmt)[0]


def _reaction(stmt: Statement) -> tuple[list[Agent], list[tuple[Agent, int]]]:
    """:data:`_REACTIONS`' entry for ``stmt``, by net change: an agent it takes and gives back as
    it was (the ``E`` of ``A + E -> B + E``) is a catalyst, neither consumed nor made."""
    for cls in type(stmt).__mro__:
        if cls in _REACTIONS:
            reactants, products = _REACTIONS[cls](stmt)
            reactants, made = list(reactants), []
            for agent, effect in products:
                same = next((a for a in reactants if a.matches(agent)), None)
                if effect > 0 and same is not None:
                    reactants.remove(same)
                else:
                    made.append((agent, effect))
            return reactants, made
    return [], []


def _modification(stmt: Modification) -> tuple[list[Agent], list[tuple[Agent, int]]]:
    if stmt.sub is None:
        return [], []
    added = not isinstance(stmt, RemoveModification)
    mod = stmt._get_mod_condition()
    return [_with_mod(stmt.sub, mod, not added)], [(_with_mod(stmt.sub, mod, added), 1)]


def _self_modification(stmt: SelfModification) -> tuple[list[Agent], list[tuple[Agent, int]]]:
    """An autophosphorylation modifies its enzyme; a transphosphorylation the partner its enzyme is
    written bound to, in that complex."""
    mod = stmt._get_mod_condition()
    if not isinstance(stmt, Transphosphorylation):
        return [_with_mod(stmt.enz, mod, False)], [(_with_mod(stmt.enz, mod, True), 1)]

    def complex_(modified: bool) -> Agent:
        enz = deepcopy(stmt.enz)
        partner = enz.bound_conditions[0]
        enz.bound_conditions = [BoundCondition(_with_mod(partner.agent, mod, modified)),
                                *enz.bound_conditions[1:]]
        return enz

    return [complex_(False)], [(complex_(True), 1)]


def _active(agent: Agent, activity: str, active: bool) -> Agent:
    out = deepcopy(agent)
    out.activity = ActivityCondition(activity, active)
    return out


def _moved(agent: Agent, location: str | None) -> Agent:
    out = deepcopy(agent)
    out.location = location
    return out


def _bound(stmt: Complex) -> tuple[list[Agent], list[tuple[Agent, int]]]:
    first, *rest = (deepcopy(m) for m in stmt.members)
    first.bound_conditions = [*first.bound_conditions, *(BoundCondition(m) for m in rest)]
    return list(stmt.members), [(first, 1)]


#: What each statement type consumes and what it makes, ``(reactants, products)``, each product
#: with ``1``, or ``-1`` for what a removal takes away, in the states the statement finds and leaves
#: them in: what a row's net rate is the rate of (:mod:`regulation`). Looked up along the type's
#: bases. Every type a row may use that makes anything has an entry
#: (``tests/test_regulation.py``); one that makes nothing (``ActiveForm``) has none.
_REACTIONS: dict[type, Callable[[Statement], tuple[list[Agent], list[tuple[Agent, int]]]]] = {
    IncreaseAmount: lambda s: ([], [(s.obj, 1)]),
    DecreaseAmount: lambda s: ([s.obj], [(s.obj, -1)]),
    Conversion: lambda s: (list(s.obj_from or []), [(a, 1) for a in s.obj_to or []]),
    PopulationTransition: lambda s: ([s.source], [(s.target, 1)]),
    Modification: _modification,
    SelfModification: _self_modification,
    RegulateActivity: lambda s: (
        [_active(s.obj, s.obj_activity, not s.is_activation)],
        [(_active(s.obj, s.obj_activity, s.is_activation), 1)],
    ),
    Gef: lambda s: ([_active(s.ras, "gtpbound", False)], [(_active(s.ras, "gtpbound", True), 1)]),
    Gap: lambda s: ([_active(s.ras, "gtpbound", True)], [(_active(s.ras, "gtpbound", False), 1)]),
    Translocation: lambda s: (
        [_moved(s.agent, s.from_location)],
        [(_moved(s.agent, s.to_location), 1)] if s.to_location else [],
    ),
    Complex: _bound,
}


def _with_mod(agent: Agent, mod: ModCondition, modified: bool) -> Agent:
    """``agent`` with ``mod`` written ``modified`` or not."""
    out = deepcopy(agent)
    mod = deepcopy(mod)
    mod.is_modified = modified
    out.mods = [m for m in out.mods if (m.mod_type, m.residue, m.position) !=
                (mod.mod_type, mod.residue, mod.position)] + [mod]
    return out


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
    all ("E not bound to C phosphorylates S"); they and the :func:`regulators` it states are all
    a mechanism's rate may read."""
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
