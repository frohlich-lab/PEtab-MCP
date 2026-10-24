"""Causal constraints: claims the model must make, given when the session starts (PEtab-MCP#134).

A constraint is a ``statements`` row without a binding, ending with what it requires of the model:

    c1: EPO phosphorylates EpoR_JAK2 at Y1007, directly          # [curated] Bachmann et al. 2011
    c2: EpoR_JAK2{pY1007} phosphorylates STAT5 at Y694, inhibited by SOCS3, directly
    c3: EPO increases SOCS3, indirectly
    c4: not EPO phosphorylates STAT5, directly
    c5: STAT5 does not affect EPO, indirectly
    a1: STAT5{pY694} is active

``, directly`` holds where one row's claim entails the constraint: its reaction, read the way
:data:`statements._REACTIONS` reads it, takes what the constraint's takes and makes what it makes,
driven the right way by the constraint's subject as the row's catalyst or one of its regulators.
``, indirectly`` holds where a chain of the rows' claims carries the subject's change to the
constraint's object with its sign. A constraint says one or the other: there is no default.
``not <statement>, directly`` holds where no row entails the statement, and ``X does not affect Y``
where no row's claim (``directly``) or no chain (``indirectly``) connects them, of either sign.
``<agent> is active`` (or ``inactive``, ``as kinase``) declares the forms ``activates`` and
``inhibits`` are about: the model's activity is what the user says it is, never inferred from the
rows the agent writes, and a constraint on the activity of an agent with no declaration is refused
as the file loads.

Every claim is a row's own: its reaction, its catalyst, and each regulator as the kind its rate
records (:attr:`mechanisms.Emission.derived`, signed on the version's bounds by
:func:`regulation.resign`), never the kind the statement states. So a constraint rests on the
maths: a row that states SOCS3 ``activated by`` on a rate that falls with it satisfies "SOCS3
decreases STAT5{pY694}". A regulator its rate cannot sign (``modulated_by``) claims no direction,
so it satisfies no signed constraint, but it is an effect, so it breaks ``does not affect``.

A claim's amounts are read as readouts read them: the sum over the species holding an agent in the
stated form, the total where none is stated, and only a net change counts. A row that moves an agent
between forms inside the set it names changes nothing (``EGF binds EGFR`` leaves total EGFR as it
is), which is also what keeps a chain from running through the forms of a multi-site protein that
a reader lumps. A chain never takes two steps in a row through one row (a binding, then its own
complex coming apart; a kinase lowering its substrate, which then feeds the product). A reactant
lowers another it is consumed with (a binding partner its partner's free form, a co-reactant
another) at any step, so an inhibitor made downstream that acts by binding is a chain like any
other: restricted to a chain's first step, to keep chains from running back up a binding, it
refused isensee's antagonist competing with cAMP for the holoenzyme, and changed no false claim's
verdict on the formulations #134 measured. A binding claims its complex coming apart as well as
forming.

Whether a model moves its object one way or the other is not decidable from its structure once it
has feedback: on the four formulations #134 measured, a chain of the opposite sign also exists for
15 of 26 true indirect claims. So an indirect constraint holds on a chain of its sign, and one of
the other sign is reported, never refused; which way the fitted model moves it is the fit's to
say. There is no signed indirect negative (``not X increases Y, indirectly``) for the same reason:
feedback breaks it on true models, and the way out would be deleting true feedback.

Agents are matched by NAME, the model's identity: two agents of a model may share a grounding
(bachmann's SOCS3 and SOCS3_oe, a gene's nRNA, mRNA and protein), which is why :func:`_reaction`
is :func:`statements._reaction` with its catalyst test by name. The check reads the statements
record alone, never the expansion, so it costs milliseconds and the reduction (:mod:`reduction`)
does not change it.

A constraint is read by :func:`notation.statement`, the row reader's own, with the regulators a
row states; the qualifier, ``not``, ``does not affect`` and ``is active`` are this module's.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict, deque
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import ValidationError

#: The qualifier every constraint ends with, and the kind it names.
QUALIFIERS = {"directly": "direct", "indirectly": "indirect"}

#: A recorded regulator kind's sign; ``modulated_by`` claims none.
_SIGN = {"activated_by": 1, "inhibited_by": -1, "modulated_by": None}

_ID = re.compile(r"\s*([A-Za-z_][A-Za-z0-9_]*)\s*:\s*")
_QUALIFIER = re.compile(r",\s*(directly|indirectly)\s*$")
_ACTIVE = re.compile(r"(.+?)\s+is\s+(active|inactive)(?:\s+as\s+([A-Za-z_]+))?")
_UNAFFECTED = re.compile(r"(.+?)\s+does\s+not\s+affect\s+(.+)")


class ConstraintError(ValidationError):
    """A constraint the server cannot read or check, with which one and why."""


@dataclass(frozen=True)
class Constraint:
    """One constraint: a statement (INDRA's JSON, its regulators included) the model must entail,
    or must not (``negated``), or the two agents of ``X does not affect Y`` (``unaffected``)."""

    id: str
    text: str
    kind: str
    statement: dict | None = None
    negated: bool = False
    unaffected: tuple[dict, dict] | None = None
    source: str | None = None


@dataclass(frozen=True)
class ActiveForm:
    """A form of an agent the user declares active (or inactive) as ``activity``."""

    id: str
    text: str
    agent: dict
    activity: str = "activity"
    active: bool = True
    source: str | None = None


@dataclass(frozen=True)
class Constraints:
    """A session's constraints and the active forms they read."""

    items: tuple[Constraint, ...]
    active_forms: tuple[ActiveForm, ...] = ()

    def describe(self) -> dict[str, Any]:
        """What ``describe_problem`` shows: each constraint and active form as written."""
        return {
            "constraints": [
                {"id": c.id, "text": c.text, "kind": c.kind}
                | ({"source": c.source} if c.source else {})
                for c in self.items
            ],
            "active_forms": [
                {"id": f.id, "text": f.text} | ({"source": f.source} if f.source else {})
                for f in self.active_forms
            ],
        }

    def texts(self) -> list[str]:
        """Every constraint and active form as a line of the notation, id first."""
        return [f"{f.id}: {f.text}" for f in self.active_forms] + [
            f"{c.id}: {c.text}" for c in self.items
        ]


# --------------------------------------------------------------------------------------- read
def read(path: str | Path, groundings: dict[str, dict] | None = None) -> Constraints:
    """The constraints in the file at ``path``: one per line in the notation (``#`` starting a
    line comments it out, after a constraint it is the constraint's source), or a JSON list of such
    lines and of objects (:func:`parse_entry`). Raises :class:`ConstraintError` naming each
    constraint that does not read, or that the server could not check."""
    text = Path(path).read_text(encoding="utf-8")
    if Path(path).suffix == ".json":
        entries = json.loads(text)
        if not isinstance(entries, list):
            raise ConstraintError(f"{path}: a JSON constraints file is a list")
    else:
        entries = [
            line for line in text.splitlines() if line.strip() and not line.lstrip().startswith("#")
        ]
    return parse(entries, groundings)


def parse(entries: list, groundings: dict[str, dict] | None = None) -> Constraints:
    """``entries`` (lines, or :func:`parse_entry` objects) as constraints, every problem named."""
    items, forms, problems = [], [], []
    for entry in entries:
        try:
            got = parse_entry(entry, groundings)
        except ConstraintError as exc:
            problems.append(str(exc))
            continue
        (forms if isinstance(got, ActiveForm) else items).append(got)
    ids = [x.id for x in [*forms, *items]]
    problems += [f"id {i!r} is given twice" for i in sorted({i for i in ids if ids.count(i) > 1})]
    declared = {_name(f.agent["name"]) for f in forms}
    for c in items:
        st = c.statement or {}
        if (
            st.get("type") in ("Activation", "Inhibition")
            and _name(st["obj"]["name"]) not in declared
        ):
            problems.append(
                f"{c.id!r} is about the activity of {st['obj']['name']}, and no line declares which "
                f"form of it is active: declare one (`a1: {st['obj']['name']}{{pS10}} is active`), or "
                f"name the form the constraint is about (`... increases {st['obj']['name']}{{pS10}}`)"
            )
    if problems:
        raise ConstraintError("the constraints do not read: " + "; ".join(problems))
    return Constraints(tuple(items), tuple(forms))


def parse_entry(entry: str | dict, groundings: dict[str, dict] | None = None):
    """One :class:`Constraint` or :class:`ActiveForm`, from a line of the notation or from an
    object: ``{id, statement, kind, not?, source?}`` with the statement in INDRA's JSON (or the
    notation), ``{id, does_not_affect: [subject, object], kind}`` or ``{id, active_form: agent,
    activity?, active?}``, agents in INDRA's JSON or the notation."""
    from . import notation

    if isinstance(entry, dict):
        return _parse_object(entry, groundings or {})
    m = _ID.match(entry)
    if not m:
        raise ConstraintError(f"a constraint starts with its id and ':': {entry!r}")
    cid = m.group(1)
    body, comment = notation._split_comment(entry[m.end() :])
    body, source = body.strip(), comment or None
    try:
        active = _ACTIVE.fullmatch(body)
        if active:
            activity = active.group(3) or "activity"
            if activity not in notation.ACTIVITIES:
                raise ConstraintError(
                    f"{cid!r}: no activity {activity!r}; one of "
                    f"{', '.join(sorted(notation.ACTIVITIES))}"
                )
            agent = notation.parse_agent(active.group(1), groundings)
            return ActiveForm(cid, body, agent, activity, active.group(2) == "active", source)
        q = _QUALIFIER.search(body)
        if not q:
            raise ConstraintError(
                f"{cid!r} says neither `, directly` nor `, indirectly`: write `{body}, directly` "
                f"(a row states it) or `{body}, indirectly` (a chain of the rows' claims does)"
            )
        kind, claim = QUALIFIERS[q.group(1)], body[: q.start()].strip()
        negated = claim.startswith("not ")
        claim = claim[4:].strip() if negated else claim
        unaffected = _UNAFFECTED.fullmatch(claim)
        if unaffected:
            if negated:
                raise ConstraintError(
                    f"{cid!r}: `not ... does not affect` is `... affects`; write the claim it makes"
                )
            pair = (
                notation.parse_agent(unaffected.group(1), groundings),
                notation.parse_agent(unaffected.group(2), groundings),
            )
            return _checked(Constraint(cid, body, kind, None, False, pair, source))
        statement = statement_json(claim, groundings)
    except notation.NotationError as exc:
        raise ConstraintError(f"{cid!r}: {exc}") from None
    return _checked(Constraint(cid, body, kind, statement, negated, None, source))


def statement_json(text: str, groundings: dict[str, dict] | None = None) -> dict:
    """A statement in the notation (:func:`notation.statement`, then the regulators it states) as
    INDRA's JSON. Raises :class:`notation.NotationError`."""
    from . import notation
    from .statements import REGULATORS

    r = notation._Reader(text, groundings or {})
    st = notation.statement(r)
    while r.take(","):
        kind = next((k for k, words in REGULATORS.items() if r.take(words)), None)
        if kind is None or kind in st:
            r.fail(f"expected one of {', '.join(map(repr, REGULATORS.values()))}, each once")
        st[kind] = [r.agent()]
        while r.take("and"):
            st[kind].append(r.agent())
    if not r.done():
        r.fail("unexpected text")
    return st


def _parse_object(entry: dict, groundings: dict) -> Constraint | ActiveForm:
    from . import notation

    def agent(value):
        return notation.parse_agent(value, groundings) if isinstance(value, str) else dict(value)

    cid = entry.get("id")
    if not isinstance(cid, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", cid):
        raise ConstraintError(f"a constraint object needs an `id` that is an identifier: {entry!r}")
    source = entry.get("source")
    try:
        if "active_form" in entry:
            form = agent(entry["active_form"])
            activity = entry.get("activity", "activity")
            active = bool(entry.get("active", True))
            text = f"{notation.agent_text(form)} is {'active' if active else 'inactive'}" + (
                "" if activity == "activity" else f" as {activity}"
            )
            return ActiveForm(cid, text, form, activity, active, source)
        kind = QUALIFIERS.get(entry.get("kind")) or (
            entry.get("kind") if entry.get("kind") in QUALIFIERS.values() else None
        )
        if kind is None:
            raise ConstraintError(f"{cid!r}: `kind` is `direct` or `indirect`")
        word = "directly" if kind == "direct" else "indirectly"
        if "does_not_affect" in entry:
            subject, obj = (agent(a) for a in entry["does_not_affect"])
            text = (
                f"{notation.agent_text(subject)} does not affect {notation.agent_text(obj)}, {word}"
            )
            return _checked(Constraint(cid, text, kind, None, False, (subject, obj), source))
        st = entry.get("statement")
        st = statement_json(st, groundings) if isinstance(st, str) else dict(st or {})
        negated = bool(entry.get("not"))
        try:
            said = notation.statement_text(st)
        except ValueError:
            said = json.dumps(st)
        text = f"{'not ' if negated else ''}{said}, {word}"
        return _checked(Constraint(cid, text, kind, st, negated, None, source))
    except notation.NotationError as exc:
        raise ConstraintError(f"{cid!r}: {exc}") from None


def _checked(c: Constraint) -> Constraint:
    """``c``, refused where the server could not check what it says."""
    from .statements import REGULATORS, problems

    if c.statement is None:
        return c
    unread = problems(c.statement)
    if unread:
        raise ConstraintError(f"{c.id!r}: {'; '.join(unread)}")
    stmt = _stmt(c.statement)
    if stmt is None:
        raise ConstraintError(f"{c.id!r}: not a statement INDRA can read")
    if c.kind == "indirect":
        if c.negated:
            raise ConstraintError(
                f"{c.id!r}: there is no `not ..., indirectly`. In a model with feedback a chain of "
                "each sign usually exists, so it would refuse true models; write `X does not "
                "affect Y, indirectly` (no chain either way), or `not ..., directly`"
            )
        if any(c.statement.get(k) for k in REGULATORS):
            raise ConstraintError(
                f"{c.id!r}: a regulator is a claim about one row's reaction; "
                "state it in a `directly` constraint"
            )
        if _subject(stmt) is None:
            raise ConstraintError(
                f"{c.id!r}: a {type(stmt).__name__} names no cause and effect for a chain to "
                "connect; write it `, directly`"
            )
    return c


# ------------------------------------------------------------------------------------- agents
def _name(name: str) -> str:
    from indra.assemblers.pysb.assembler import _n

    return _n(name)


def _agent(data: dict):
    from indra.statements import Agent

    return Agent._from_json(deepcopy(data))


def _stmt(statement: dict):
    from indra.statements import stmts_from_json

    from .statements import indra_part, with_defaults

    got = stmts_from_json([with_defaults(indra_part(statement))])
    return got[0] if got else None


def compatible(a, b) -> bool:
    """Whether one form of one agent can be both ``a`` and ``b``: the same name, and no condition of
    one contradicts the other's."""
    if _name(a.name) != _name(b.name):
        return False
    for m in a.mods:
        for n in b.mods:
            if (m.mod_type, m.residue, m.position) == (
                n.mod_type,
                n.residue,
                n.position,
            ) and m.is_modified != n.is_modified:
                return False
    for x in a.bound_conditions:
        for y in b.bound_conditions:
            if _name(x.agent.name) == _name(y.agent.name) and x.is_bound != y.is_bound:
                return False
    if a.location and b.location and a.location != b.location:
        return False
    if (
        a.activity
        and b.activity
        and a.activity.activity_type == b.activity.activity_type
        and a.activity.is_active != b.activity.is_active
    ):
        return False
    return True


def refines(a, b) -> bool:
    """Whether ``a`` is ``b`` or a form of it: every condition ``b`` writes, ``a`` writes too (a
    residue or position ``b`` leaves out, any)."""
    if _name(a.name) != _name(b.name):
        return False
    for n in b.mods:
        if not any(
            m.mod_type == n.mod_type
            and m.is_modified == n.is_modified
            and n.residue in (None, m.residue)
            and n.position in (None, m.position)
            for m in a.mods
        ):
            return False
    for y in b.bound_conditions:
        if not any(
            _name(x.agent.name) == _name(y.agent.name) and x.is_bound == y.is_bound
            for x in a.bound_conditions
        ):
            return False
    if b.location and a.location != b.location:
        return False
    if b.activity and not (
        a.activity
        and a.activity.activity_type == b.activity.activity_type
        and a.activity.is_active == b.activity.is_active
    ):
        return False
    return True


def _text(agent) -> str:
    from . import notation

    try:
        return notation.agent_text(agent.to_json())
    except (ValueError, KeyError):
        return agent.name


def _same(a, b) -> bool:
    """One agent in one state, by name (``Agent.matches`` keys on grounding)."""
    return _name(a.name) == _name(b.name) and a.matches(b)


def _reaction(stmt) -> tuple[list, list]:
    """:func:`statements._reaction`, its catalyst test by name: ``Agent.matches`` keys on
    grounding, so a conversion between two agents that share one (``SOCS3_nRNA1 -> SOCS3_nRNA2``)
    read as making nothing."""
    from .statements import _REACTIONS

    for cls in type(stmt).__mro__:
        if cls in _REACTIONS:
            reactants, products = _REACTIONS[cls](stmt)
            reactants, made = list(reactants), []
            for agent, effect in products:
                same = next((a for a in reactants if _same(a, agent)), None)
                if effect > 0 and same is not None:
                    reactants.remove(same)
                else:
                    made.append((agent, effect))
            return reactants, made
    return [], []


def _joined(members) -> tuple[list, list]:
    """The complex of ``members`` as each member bound to the others, and each member free of
    them: what a binding makes and what it takes, whichever member a later row names it by. A
    member's own written condition on a partner (``A{~B} binds B``) gives way to the binding's."""
    from indra.statements import BoundCondition

    whole, free = [], []
    for a in members:
        others = [b for b in members if b is not a]
        names = {_name(b.name) for b in others}
        kept = [bc for bc in a.bound_conditions if _name(bc.agent.name) not in names]
        w, f = deepcopy(a), deepcopy(a)
        w.bound_conditions = [*kept, *(BoundCondition(b) for b in others)]
        f.bound_conditions = [*kept, *(BoundCondition(b, False) for b in others)]
        whole.append(w)
        free.append(f)
    return whole, free


def _subject(stmt):
    """What drives ``stmt``: its catalyst, or the enzyme of a self-modification; ``None``."""
    from indra.statements import SelfModification

    from .statements import _catalyst

    if isinstance(stmt, SelfModification):
        return stmt.enz
    return _catalyst(stmt)[0]


# -------------------------------------------------------------------------------------- claims
@dataclass
class _Step:
    """One reaction a row claims: what drives it, each ``(agent, sign, role)`` (sign ``None`` for a
    regulator its rate does not sign), and what it changes, each ``(agent, change, consumed)``."""

    prefix: str
    drivers: list[tuple[Any, int | None, str]]
    effects: list[tuple[Any, int, bool]]


def steps(record: dict | None) -> list[_Step]:
    """The reactions ``record``'s rows (a statements record) claim, regulators as recorded."""
    from indra.statements import Complex, SelfModification

    from .statements import signed

    out = []
    for row in (record or {}).get("rows", []):
        stmt = _stmt(row["statement"])
        if stmt is None:
            continue
        derived = (row.get("emission") or {}).get("derived") or {}
        reactants, products = _reaction(stmt)
        if isinstance(stmt, Complex):
            # Made from each member free of the others: as written, a member names no partner, so
            # it read as possibly bound already, and the binding as lowering its own complex.
            whole, reactants = _joined(stmt.members)
            products = [(w, 1) for w in whole]
        # An enzyme modifying itself catalyses no other molecule: it drives the reaction as the
        # reactant it is (read as a catalyst, a complex holding a modified one "made" more of the
        # rest).
        subject = None if isinstance(stmt, SelfModification) else _subject(stmt)
        drivers = [(subject, 1, "catalyst")] if subject is not None else []
        drivers += [
            (agent, _SIGN[kind], "regulator")
            for kind, agent, _ in signed(row["statement"], derived)
        ]
        drivers += [(r, 1, "reactant") for r in reactants]
        effects = [(p, e, False) for p, e in products]
        effects += [(r, -1, True) for r in reactants if not any(_same(r, p) for p, _ in products)]
        out.append(_Step(row["prefix"], drivers, effects))
        if isinstance(stmt, Complex):
            # Coming apart: the complex raises each member's form free of the others, and lowers
            # itself, seen from each member (without that, cAMP leaving PKA's regulatory subunit
            # raised total cAMP).
            out.append(
                _Step(
                    row["prefix"],
                    [(p, 1, "complex") for p, _ in products],
                    [(r, 1, False) for r in reactants] + [(p, -1, True) for p, _ in products],
                )
            )
    return out


def _allowed(step: _Step, d: int | None, own: bool = False) -> list[tuple[Any, int]]:
    """The changes of ``step`` driven by its driver ``d`` (``None``: by the reaction itself), but,
    unless ``own``, that driver's own: a driver that rises does not also fall by its own
    consumption, neither for the rows it drives next nor for a target it is part of (read so, a
    target's own decay, "more pS6, more of it dephosphorylated", ended every chain the other way).
    One row's net, read for a direct claim, counts it: binding EGFR leaves its total as it is."""
    if d is None or own:
        return [(e, change) for e, change, _ in step.effects]
    agent = step.drivers[d][0]
    return [(e, change) for e, change, _ in step.effects if not (e is agent or _same(e, agent))]


def _moves(changes: list[tuple[Any, int]], target, strict: bool = False) -> set[int]:
    """Which way ``changes`` move ``target``'s amount: by their net over the forms within it, and,
    unless ``strict``, by their net over the forms that may be partly within it (a product whose
    location the row leaves to the model's default, read by a row that names one). Each part is
    netted on its own: a binding that moves EGFR from free to bound leaves EGFR{pY1068} as it is,
    though each form may hold some."""
    net = sum(c for e, c in changes if refines(e, target))
    out = {1 if net > 0 else -1} if net else set()
    if strict:
        return out
    maybe = sum(c for e, c in changes if not refines(e, target) and compatible(e, target))
    return out | ({1 if maybe > 0 else -1} if maybe else set())


class _Target:
    """What a constraint's chain must reach: an amount (an agent in a stated form, :func:`_moves`),
    or an agent's activity, its declared active forms rising or inactive ones falling, or a row
    whose catalyst is an active form regulated."""

    def __init__(self, agent=None, forms: list | None = None, any_form: bool = False):
        self.agent, self.forms, self.any_form = agent, forms, any_form

    def reached(
        self, step: _Step, d: int | None, strict: bool = False, own: bool = False
    ) -> set[int]:
        """Which ways ``step``, driven by its driver ``d``, moves the target (``strict``:
        :func:`_moves`). For ``any_form`` (what ``does not affect`` forbids), each way it moves any
        form of the agent, net or not."""
        changes = _allowed(step, d, own)
        if self.any_form:
            return {1 if c > 0 else -1 for e, c in changes if compatible(e, self.agent)}
        if self.forms is None:
            return _moves(changes, self.agent, strict)
        out = set()
        for form, active in self.forms:
            out |= {g if active else -g for g in _moves(changes, form, strict)}
        if (
            d is not None
            and step.drivers[d][2] == "regulator"
            and any(
                role == "catalyst"
                and any(compatible(agent, form) for form, active in self.forms if active)
                for agent, _, role in step.drivers
            )
        ):
            out.add(1)
        return out


def _target(c: Constraint, constraints: Constraints):
    """``(subject, target, sign)`` of a causal constraint."""
    from indra.statements import (
        DecreaseAmount,
        ModCondition,
        Modification,
        RegulateActivity,
        RegulateAmount,
        RemoveModification,
        SelfModification,
        modclass_to_modtype,
        modtype_to_inverse,
    )

    if c.unaffected is not None:
        # A prohibition reads strictly: any form of the object, however the row moves it.
        subject, obj = (_agent(a) for a in c.unaffected)
        return subject, _Target(obj, any_form=True), None
    stmt = _stmt(c.statement)
    subject = _subject(stmt)
    if isinstance(stmt, RegulateActivity):
        forms = [
            (_agent(f.agent), f.active)
            for f in constraints.active_forms
            if _name(f.agent["name"]) == _name(stmt.obj.name)
            and stmt.obj_activity in ("activity", f.activity)
            and compatible(_agent(f.agent), stmt.obj)
        ]
        return subject, _Target(forms=forms), 1 if stmt.is_activation else -1
    if isinstance(stmt, RegulateAmount):
        return subject, _Target(stmt.obj), -1 if isinstance(stmt, DecreaseAmount) else 1
    if isinstance(stmt, (Modification, SelfModification)):
        mod_type, sign = modclass_to_modtype[type(stmt)], 1
        if isinstance(stmt, RemoveModification):
            mod_type, sign = modtype_to_inverse[mod_type], -1
        obj = deepcopy(stmt.enz if isinstance(stmt, SelfModification) else stmt.sub)
        obj.mods = [*obj.mods, ModCondition(mod_type, stmt.residue, stmt.position)]
        return subject, _Target(obj), sign
    # The rest (a GEF's, a conversion's): what the statement makes, which its subject raises.
    _, products = _reaction(stmt)
    return subject, _Target(products[0][0]), products[0][1]


# --------------------------------------------------------------------------------------- check
@dataclass
class Verdict:
    """Whether a constraint holds, and on what: the rows that entail it (``directly``), the chain
    that carries it and one of the other sign (``indirectly``), or why it does not."""

    ok: bool
    rows: tuple[str, ...] = ()
    chain: str | None = None
    opposite: str | None = None
    why: str | None = None

    def stored(self) -> dict[str, Any]:
        return {
            k: v
            for k, v in (
                ("ok", self.ok),
                ("rows", list(self.rows)),
                ("chain", self.chain),
                ("opposite", self.opposite),
                ("why", self.why),
            )
            if v or k == "ok"
        }


def check(constraints: Constraints, record: dict | None) -> dict[str, Verdict]:
    """Each constraint's :class:`Verdict` on the model ``record`` (a statements record, ``None``
    for a model not built from statements, which claims nothing)."""
    claimed = steps(record)
    graph = _Graph(claimed)
    return {c.id: _verdict(c, constraints, claimed, graph) for c in constraints.items}


def _verdict(c: Constraint, constraints: Constraints, claimed: list[_Step], graph) -> Verdict:
    if c.unaffected is not None:
        subject, target, _ = _target(c, constraints)
        if c.kind == "direct":
            hits = [
                s.prefix
                for s in claimed
                for d, (agent, _, _) in enumerate(s.drivers)
                if compatible(agent, subject) and target.reached(s, d)
            ]
            return (
                Verdict(True)
                if not hits
                else Verdict(
                    False,
                    tuple(dict.fromkeys(hits)),
                    why=f"row{'s' if len(set(hits)) > 1 else ''} `{'`, `'.join(dict.fromkeys(hits))}` "
                    f"claim{'' if len(set(hits)) > 1 else 's'} {_text(subject)} changes "
                    f"{_text(target.agent)}",
                )
            )
        chain = graph.search(subject, target, None)
        return (
            Verdict(True)
            if chain is None
            else Verdict(False, chain=chain, why=f"a chain of claims connects them: {chain}")
        )
    if c.kind == "direct":
        rows = _entailing(c, constraints, claimed)
        if c.negated:
            return (
                Verdict(True)
                if not rows
                else Verdict(
                    False,
                    tuple(rows),
                    why=f"row{'s' if len(rows) > 1 else ''} `{'`, `'.join(rows)}` "
                    f"state{'' if len(rows) > 1 else 's'} it",
                )
            )
        return (
            Verdict(True, tuple(rows))
            if rows
            else Verdict(False, why=_why_direct(c, constraints, claimed))
        )
    subject, target, sign = _target(c, constraints)
    same = graph.search(subject, target, sign)
    opposite = graph.search(subject, target, -sign)
    if same is not None:
        return Verdict(True, chain=same, opposite=opposite)
    if opposite is not None:
        return Verdict(
            False, opposite=opposite, why=f"its claims connect them only the other way: {opposite}"
        )
    return Verdict(False, why=graph.frontier(subject, target))


def _entailing(c: Constraint, constraints: Constraints, claimed: list[_Step]) -> list[str]:
    """The prefixes of the rows whose one reaction entails the constraint's statement."""
    from indra.statements import Complex, RegulateActivity, SelfModification

    from .statements import regulators

    stmt = _stmt(c.statement)
    stated = [(kind, a) for kind, given in regulators(c.statement).items() for a in given]
    subject = _subject(stmt)
    target = sign = musts = None
    if isinstance(stmt, RegulateActivity):
        _, target, sign = _target(c, constraints)
    else:
        # What the constraint's reaction makes, and takes, each with the way it moves: its
        # subject drives it, so a row entails it where the subject moves each the same way.
        reactants, products = _reaction(stmt)
        if isinstance(stmt, Complex):
            whole, reactants = _joined(stmt.members)
            products = [(w, 1) for w in whole]
        musts = list(products)
        if not isinstance(stmt, SelfModification):
            musts += [(r, -1) for r in reactants if not any(_same(r, p) for p, _ in products)]
    rows = []
    for s in claimed:
        if not all(
            any(
                role == "regulator" and k == _SIGN[kind] and compatible(agent, a)
                for agent, k, role in s.drivers
            )
            for kind, a in stated
        ):
            continue
        if subject is None:
            ways = [(1, None)]
        else:
            # A catalyst, a regulator, or a reactant other than what it becomes (a binding
            # partner raises its partner's bound form; an enzyme modifying itself drives its row).
            ways = [
                (k, d)
                for d, (agent, k, role) in enumerate(s.drivers)
                if k is not None and role != "complex" and compatible(agent, subject)
            ]
        for k, d in ways:
            # Strict: the row's own forms must be the constraint's, not merely overlap them.
            if musts is None:
                ok = k * sign in target.reached(s, d, strict=True, own=True)
            else:
                changes = _allowed(s, d, own=True)
                ok = all(any(k * g == e for g in _moves(changes, p, strict=True)) for p, e in musts)
            if ok:
                rows.append(s.prefix)
                break
    return list(dict.fromkeys(rows))


def _why_direct(c: Constraint, constraints: Constraints, claimed: list[_Step]) -> str:
    """Why no row entails ``c``: the rows that change what it changes, and what each lacks."""
    from indra.statements import RegulateActivity

    from .statements import REGULATORS, regulators

    stmt = _stmt(c.statement)
    subject = _subject(stmt)
    if isinstance(stmt, RegulateActivity):
        _, target, sign = _target(c, constraints)
        goal = next((form for form, active in target.forms if active), None)
        what = f"the activity of {_text(stmt.obj)}"
    else:
        _, products = _reaction(stmt)
        goal, sign = products[0] if products else (None, 1)
        target = _Target(goal)
        what = _text(goal) if goal is not None else "what it makes"
    changers = [s for s in claimed if target.reached(s, None)]
    if not changers:
        return f"no row changes {what}"
    said = []
    for s in changers:
        found = [
            (d, k, role)
            for d, (agent, k, role) in enumerate(s.drivers)
            if subject is not None and role != "complex" and compatible(agent, subject)
        ]
        if not found:
            continue
        if all(k is None for _, k, _ in found):
            said.append(
                f"`{s.prefix}` states {_text(subject)}, but its rate does not sign it "
                "(recorded `modulated_by`)"
            )
            continue
        missing = [
            f"`{REGULATORS[kind]} {_text(a)}`"
            for kind, given in regulators(c.statement).items()
            for a in given
            if not any(
                r == "regulator" and kk == _SIGN[kind] and compatible(agent, a)
                for agent, kk, r in s.drivers
            )
        ]
        if missing:
            said.append(
                f"`{s.prefix}` has {_text(subject)} but not {', '.join(missing)} as its "
                "rate records them"
            )
        elif not any(k is not None and k * sign in target.reached(s, d) for d, k, _ in found):
            said.append(
                f"`{s.prefix}` has {_text(subject)} change {what} the other way"
                + (
                    " (as its rate records it)"
                    if any(r == "regulator" for _, _, r in found)
                    else ""
                )
            )
        else:
            said.append(
                f"`{s.prefix}` has {_text(subject)} make {what}, but not from what the "
                "constraint's reaction takes"
            )
    if said:
        return "; ".join(dict.fromkeys(said))
    names = ", ".join(f"`{s.prefix}`" for s in changers[:6])
    return (
        f"rows {names} change {what}, and none has {_text(subject)} drive it: state it on the row "
        f"it acts on, as its catalyst or a regulator (`..., inhibited by {_text(subject)}`, "
        "`activated by`)"
        if subject is not None
        else f"rows {names} change {what}, and none takes what it takes"
    )


def _bound_as(subject, complex_form) -> bool:
    """Whether ``subject`` is written bound to a partner ``complex_form`` holds: a release of that
    complex then starts from it."""
    partners = {_name(bc.agent.name) for bc in complex_form.bound_conditions if bc.is_bound}
    return any(bc.is_bound and _name(bc.agent.name) in partners for bc in subject.bound_conditions)


class _Graph:
    """The chains the rows' claims make: from a driver of one reaction, through each change it
    makes, to a driver of another row's reaction that change moves."""

    def __init__(self, claimed: list[_Step]):
        self.steps = claimed
        self.by_name: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for i, s in enumerate(claimed):
            for d, (agent, _, _) in enumerate(s.drivers):
                self.by_name[_name(agent.name)].append((i, d))
        self._next: dict[tuple[int, int], list[tuple[int, int, int]]] = {}

    def next(self, i: int, d: int) -> list[tuple[int, int, int]]:
        """The drivers ``(step, driver, way)`` the changes of step ``i`` driven by ``d`` move, each
        way it moves them, in other rows."""
        key = (i, d)
        if key not in self._next:
            changes = _allowed(self.steps[i], d)
            out = set()
            for name in {_name(e.name) for e, _ in changes}:
                for j, dd in self.by_name.get(name, ()):
                    if self.steps[j].prefix == self.steps[i].prefix:
                        continue
                    driver, _, role = self.steps[j].drivers[dd]
                    # A complex comes apart as there is more of IT: a member made bare (a synthesis
                    # makes it free) does not raise the complex, though one form may be both.
                    for g in _moves(changes, driver, strict=role == "complex"):
                        out.add((j, dd, g))
            self._next[key] = sorted(out)
        return self._next[key]

    def search(self, subject, target: _Target, sign: int | None) -> str | None:
        """The shortest chain from ``subject`` to ``target`` whose signs multiply to ``sign``
        (``None``: of either sign, unsigned claims included), written out; ``None`` if none."""
        signed = sign is not None
        parent: dict[tuple, tuple | None] = {}
        todo: deque = deque()
        for i, s in enumerate(self.steps):
            for d, (agent, k, role) in enumerate(s.drivers):
                if k is None and signed:
                    continue
                if role == "complex" and not _bound_as(subject, agent):
                    # A complex comes apart because there is more of it, not more of a member.
                    continue
                if compatible(agent, subject):
                    state = (i, d, 1)
                    if state not in parent:
                        parent[state] = None
                        todo.append(state)
        while todo:
            state = todo.popleft()
            i, d, way = state
            k = self.steps[i].drivers[d][1]
            flux = way * k if (signed and k is not None) else 0
            reached = target.reached(self.steps[i], d)
            if reached and (not signed or any(flux * g == sign for g in reached)):
                return self._written(state, parent, subject, target, sign)
            for j, dd, g in self.next(i, d):
                kk = self.steps[j].drivers[dd][1]
                if signed and kk is None:
                    continue
                nxt = (j, dd, flux * g if signed else 0)
                if nxt not in parent:
                    parent[nxt] = state
                    todo.append(nxt)
        return None

    def _written(self, state, parent, subject, target: _Target, sign: int | None) -> str:
        """The chain ending at ``state``: the subject, then each row and the form it moves, ``→``
        raising it and ``⊣`` lowering it."""
        chain = []
        while state is not None:
            chain.append(state)
            state = parent[state]
        chain.reverse()
        parts = [_text(subject)]
        for n, (i, d, way) in enumerate(chain):
            s = self.steps[i]
            after = chain[n + 1][2] if n + 1 < len(chain) else sign
            arrow = "⊣" if way and after and way * after < 0 else "→"
            if n + 1 < len(chain):
                head = self.steps[chain[n + 1][0]].drivers[chain[n + 1][1]][0]
            elif target.forms is None:
                head = target.agent
            else:
                head = next(
                    (form for form, active in target.forms if active),
                    target.forms[0][0] if target.forms else None,
                )
            parts.append(
                f"{arrow} ({s.prefix}) {_text(head) if head is not None else 'its activity'}"
            )
        return " ".join(parts)

    def frontier(self, subject, target: _Target) -> str:
        """Why no chain connects them: what the claims reach from ``subject``, and the rows that
        change ``target``."""
        reached, todo, seen = [], deque(), set()
        for i, s in enumerate(self.steps):
            for d, (agent, _, _) in enumerate(s.drivers):
                if compatible(agent, subject):
                    todo.append((i, d))
        while todo:
            i, d = todo.popleft()
            if (i, d) in seen:
                continue
            seen.add((i, d))
            reached += [_text(e) for e, _ in _allowed(self.steps[i], d)]
            todo.extend((j, dd) for j, dd, _ in self.next(i, d))
        reached = list(dict.fromkeys(reached))
        makers = sorted(
            {s.prefix for s in self.steps for d in range(len(s.drivers)) if target.reached(s, d)}
        )
        out = "no chain of claims connects them"
        out += (
            (
                f"; from {_text(subject)} the claims reach {', '.join(reached[:8])}"
                + (" ..." if len(reached) > 8 else "")
            )
            if reached
            else (f"; no row's claim is driven by {_text(subject)}")
        )
        if makers:
            out += f"; the rows that change it are `{'`, `'.join(makers[:8])}`"
        return out


# ------------------------------------------------------------------------------------ commit
def enforce(
    constraints: Constraints, record: dict | None, base: dict | None
) -> tuple[dict, str | None]:
    """Check ``record`` (the statements record a change would commit) against ``constraints``.
    Raises :class:`ConstraintError` (cause ``constraint_violated``) naming each constraint it
    breaks. Returns what the record keeps (the constraints as checked and each verdict) and the
    warning for an indirect constraint whose chain of the other sign is new since ``base``, the
    record the change starts from, or ``None``."""
    verdicts = check(constraints, record)
    broken = [c for c in constraints.items if not verdicts[c.id].ok]
    if broken:
        from .operations import OperationError

        detail = "; ".join(f"`{c.id}` ({c.text}): {verdicts[c.id].why}" for c in broken)
        raise OperationError(
            f"make_changes: the model breaks {len(broken)} of the session's constraints: {detail}. "
            + (
                ""
                if record
                else "This model was not built from statements, and a model claims "
                "only what its rows state. "
            )
            + "NOTHING was applied.",
            cause="constraint_violated",
        )
    was = ((base or {}).get("constraints") or {}).get("verdicts") or {}
    new = [
        c
        for c in constraints.items
        if verdicts[c.id].opposite and not (was.get(c.id) or {}).get("opposite")
    ]
    said = "; ".join(
        f"`{c.id}` ({c.text}) holds through {verdicts[c.id].chain}, and a chain of the other sign "
        f"exists too: {verdicts[c.id].opposite}"
        for c in new
    )
    warning = (
        (
            said + ". Which way the model moves each is the fit's to say, not its structure's: "
            "keep both if both are true."
        )
        if said
        else None
    )
    kept = {
        "checked": constraints.texts(),
        "verdicts": {cid: v.stored() for cid, v in verdicts.items()},
    }
    return kept, warning


# -------------------------------------------------------------------------------------- update
def _implies(a: Constraint, b: Constraint) -> bool:
    """Whether ``a`` holding makes ``b`` hold: the same claim, or ``a`` the stronger form of it (a
    row's claim is a chain of one; no chain either way leaves no row's claim either)."""
    if (a.negated, a.statement, a.unaffected) != (b.negated, b.statement, b.unaffected):
        return False
    if a.kind == b.kind:
        return True
    if a.unaffected is not None:
        return a.kind == "indirect"
    return a.kind == "direct" and not a.negated


def change(old: Constraints | None, new: Constraints) -> str:
    """How ``new`` differs from ``old``: ``same``, ``looser`` (each new constraint follows from an
    old one), ``tighter`` (each old one from a new one) or ``both``. A change of the declared active
    forms changes what every activity constraint means, so it is ``both``."""
    old = old or Constraints(())
    if [f.text for f in old.active_forms] != [f.text for f in new.active_forms]:
        return "both"
    looser = all(any(_implies(a, b) for a in old.items) for b in new.items)
    tighter = all(any(_implies(b, a) for b in new.items) for a in old.items)
    return (
        "same" if looser and tighter else "looser" if looser else "tighter" if tighter else "both"
    )


def violations(constraints: Constraints, records: dict[int, dict | None]) -> dict[int, list[str]]:
    """The ids of the constraints each stored version breaks, by version, ``records`` being each
    version's statements record (``None`` for one not built from statements); a version breaking
    none is left out."""
    out = {}
    for version, record in records.items():
        broken = [cid for cid, v in check(constraints, record).items() if not v.ok]
        if broken:
            out[version] = broken
    return out
