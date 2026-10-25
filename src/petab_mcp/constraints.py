"""Causal constraints: claims the model must make, given when the session starts (PEtab-MCP#134).

A constraint is a ``statements`` row without a binding, ending with what it requires of the model:

    c1: EPO phosphorylates EpoR_JAK2 at Y1007, directly          # [curated] Bachmann et al. 2011
    c2: EpoR_JAK2{pY1007} phosphorylates STAT5 at Y694, inhibited by SOCS3, directly
    c3: EPO increases SOCS3, indirectly
    c4: not EPO phosphorylates STAT5, directly
    c5: STAT5 does not affect EPO, indirectly
    a1: STAT5{pY694} is active

Each is read, once, into what it asks: that its subject, raised, move each of some forms one way
(:class:`Constraint`). ``, directly`` asks it of one row's reaction, the subject its catalyst, one
of its regulators or a reactant, and every form the constraint's own reaction changes moving as it
does (a phosphorylation makes the phosphorylated form and takes the unphosphorylated one).
``, indirectly`` asks it of a chain of the rows' reactions, for the form the constraint is about.
There is no default. ``not <statement>, directly`` holds where no row entails the statement, and
``X does not affect Y`` where no row (``directly``) or no chain (``indirectly``) moves any form of Y
from X, either way. ``<agent> is active`` (``inactive``, ``as kinase``) declares the forms
``activates`` and ``inhibits`` are about: activity is what the user says, never inferred from the
agent's rows, and an activity constraint on an undeclared agent is refused as the file loads. A
regulator of a reaction a declared active form catalyses moves that activity, as ``as_indra``
projects it ("IBMX inhibits PDE").

Every claim is a row's own: its reaction (:data:`statements._REACTIONS`), its catalyst, and each
regulator as the kind its rate records (:attr:`mechanisms.Emission.derived`, signed on the
version's bounds by :func:`regulation.resign`), never the kind the statement states: a row stating
SOCS3 ``activated by`` on a rate that falls with it satisfies "SOCS3 decreases STAT5{pY694}". A
regulator its rate cannot sign (``modulated_by``) claims no direction, so it satisfies no signed
constraint, but it is an effect, so it breaks ``does not affect``.

An amount is read as a readout reads it: the sum over the species holding an agent in the stated
form, the total where none is stated, and only a net change counts (:func:`_moves`). A row that
moves an agent between forms inside the set it names changes nothing (``EGF binds EGFR`` leaves
total EGFR as it is), which also keeps a chain from running through the forms of a multi-site
protein that a reader lumps. A chain never takes two steps in a row through one row (a binding,
then its own complex coming apart; a kinase lowering its substrate, which then feeds the product).
A reactant lowers another it is consumed with (a binding partner its partner's free form, a
co-reactant another) at any step: restricted to a chain's first step, to keep chains from running
back up a binding, it refused isensee's antagonist competing with cAMP for the holoenzyme and
changed no false claim's verdict on the formulations #134 measured. A binding claims its complex
coming apart as well as forming, and that step needs more of the complex itself.

Whether a model moves its object one way or the other is not decidable from its structure once it
has feedback: on those formulations a chain of the opposite sign also exists for 15 of 26 true
indirect claims. So an indirect constraint holds on a chain of its sign, and one of the other sign
is reported, never refused; which way the fitted model moves it is the fit's to say. There is no
signed indirect negative (``not X increases Y, indirectly``) for the same reason: feedback breaks
it on true models, and the way out would be deleting true feedback.

Agents are matched by NAME, the model's identity: two agents may share a grounding (bachmann's
SOCS3 and SOCS3_oe, a gene's nRNA, mRNA and protein), which is why :func:`_reaction` is
:func:`statements._reaction` with its catalyst test by name (PEtab-MCP#137). The check reads the
statements record alone, never the expansion, so it costs milliseconds and the reduction
(:mod:`reduction`) does not change it. A constraint's statement is read by
:func:`notation.statement`, the row reader's own; the qualifier, ``not``, ``does not affect`` and
``is active`` are this module's.
"""

from __future__ import annotations

import re
from collections import defaultdict, deque
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NamedTuple

from .errors import ValidationError

#: A recorded regulator kind's sign; ``modulated_by`` claims none.
_SIGN = {"activated_by": 1, "inhibited_by": -1, "modulated_by": None}

_LINE = re.compile(r"\s*([A-Za-z_][A-Za-z0-9_]*)\s*:\s*(.*)", re.S)
_QUALIFIED = re.compile(r"(.*?),\s*(directly|indirectly)\s*", re.S)
_ACTIVE = re.compile(r"(.+?)\s+is\s+(active|inactive)(?:\s+as\s+([A-Za-z_]+))?")
_UNAFFECTED = re.compile(r"(.+?)\s+does\s+not\s+affect\s+(.+)")


class ConstraintError(ValidationError):
    """A constraint the server cannot read or check, with which one and why."""


class _Form(NamedTuple):
    """A form of an agent a constraint is about, the way its subject must move it (``None``: any
    way), and whether it is a declared active form, whose activity a regulator of a reaction it
    catalyses moves."""

    agent: Any
    way: int | None
    active: bool = False


@dataclass
class Constraint:
    """A constraint as written (``text`` without its id, ``claim`` without its qualifier) and what
    it asks: that ``subject`` (``None``: a row's reaction itself), raised, move ``forms``, every one
    of them (``every``) or any, through one row (``direct``) with ``regulators`` (kind, agent)
    recorded on it, or through a chain; ``negated``, that no row does."""

    id: str
    text: str
    claim: str
    direct: bool
    subject: Any
    forms: list[_Form]
    every: bool = False
    regulators: list[tuple[str, Any]] = field(default_factory=list)
    negated: bool = False
    source: str | None = None


@dataclass
class ActiveForm:
    """A form of an agent the user declares active (or inactive) as ``activity``."""

    id: str
    text: str
    agent: Any
    activity: str
    active: bool
    source: str | None = None


@dataclass
class Constraints:
    """A session's constraints and the active forms they read."""

    items: tuple[Constraint, ...]
    active_forms: tuple[ActiveForm, ...] = ()

    def texts(self) -> list[str]:
        """Every constraint and active form as a line of the notation, id first."""
        return [f"{x.id}: {x.text}" for x in (*self.active_forms, *self.items)]


# --------------------------------------------------------------------------------------- read
def read(path: str | Path, groundings: dict[str, dict] | None = None) -> Constraints:
    """The constraints in the file at ``path``, one per line (:func:`parse`); a line that starts
    with ``#`` is a comment, and ``#`` after a constraint gives its source."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return parse([s for s in lines if s.strip() and not s.lstrip().startswith("#")], groundings)


def parse(lines: list[str], groundings: dict[str, dict] | None = None) -> Constraints:
    """``lines`` as constraints, grounded as the rows are by the entity map's names. Raises
    :class:`ConstraintError` naming each that does not read or that the server could not check."""
    from . import notation

    groundings = groundings or {}
    forms, read_later, problems = [], [], []
    for line in lines:
        m = _LINE.fullmatch(line)
        if not m:
            problems.append(f"a constraint starts with its id and ':': {line!r}")
            continue
        cid, (body, source) = m.group(1), notation._split_comment(m.group(2))
        body = body.strip()
        if active := _ACTIVE.fullmatch(body):
            try:
                forms.append(_active_form(cid, body, active, source, groundings))
            except (ConstraintError, notation.NotationError) as exc:
                problems.append(f"{cid!r}: {exc}")
        else:
            read_later.append((cid, body, source))
    items = []
    for cid, body, source in read_later:  # after every active form, which they may read
        try:
            items.append(_constraint(cid, body, source, forms, groundings))
        except (ConstraintError, notation.NotationError) as exc:
            problems.append(f"{cid!r}: {exc}")
    ids = [x.id for x in (*forms, *items)]
    problems += [f"id {i!r} is given twice" for i in sorted({i for i in ids if ids.count(i) > 1})]
    if problems:
        raise ConstraintError("the constraints do not read: " + "; ".join(problems))
    return Constraints(tuple(items), tuple(forms))


def _active_form(cid, body, m, source, groundings) -> ActiveForm:
    from . import notation

    activity = m.group(3) or "activity"
    if activity not in notation.ACTIVITIES:
        raise ConstraintError(
            f"no activity {activity!r}; one of {', '.join(sorted(notation.ACTIVITIES))}"
        )
    agent = _agent(notation.parse_agent(m.group(1), groundings))
    return ActiveForm(cid, body, agent, activity, m.group(2) == "active", source)


def _constraint(cid, body, source, active_forms, groundings) -> Constraint:
    from indra.statements import RegulateActivity, RemoveModification, SelfModification

    from . import notation
    from .statements import problems, regulators

    q = _QUALIFIED.fullmatch(body)
    if not q:
        raise ConstraintError(
            f"says neither `, directly` nor `, indirectly`: write `{body}, directly` (a row states "
            f"it) or `{body}, indirectly` (a chain of the rows' claims does)"
        )
    claim, direct = q.group(1).strip(), q.group(2) == "directly"
    negated = claim.startswith("not ")
    said = claim[4:].strip() if negated else claim
    if pair := _UNAFFECTED.fullmatch(said):
        if negated:
            raise ConstraintError(
                "`not ... does not affect` is `... affects`; write the claim it makes"
            )
        subject, obj = (_agent(notation.parse_agent(a, groundings)) for a in pair.groups())
        return Constraint(cid, body, claim, direct, subject, [_Form(obj, None)], source=source)
    st = statement_json(said, groundings)
    if unread := problems(st):
        raise ConstraintError("; ".join(unread))
    stmt = _stmt(st)
    if stmt is None:
        raise ConstraintError("not a statement INDRA can read")
    regs = [(kind, a) for kind, given in regulators(st).items() for a in given]
    subject = _subject(stmt)
    if isinstance(stmt, RegulateActivity):
        way = 1 if stmt.is_activation else -1
        forms = [
            _Form(f.agent, way if f.active else -way, f.active)
            for f in active_forms
            if _name(f.agent.name) == _name(stmt.obj.name)
            and stmt.obj_activity in ("activity", f.activity)
            and compatible(f.agent, stmt.obj)
        ]
        if not forms:
            raise ConstraintError(
                f"is about the activity of {stmt.obj.name}, and no line declares which form of it "
                f"is active: declare one (`a1: {stmt.obj.name}{{pS10}} is active`), or name the "
                f"form the constraint is about (`... increases {stmt.obj.name}{{pS10}}`)"
            )
        every = False
    else:
        reactants, products = _reaction(stmt)
        forms = [_Form(p, way) for p, way in products]
        if not isinstance(stmt, SelfModification):  # an enzyme modifying itself is its subject
            forms += [_Form(r, -1) for r in reactants]
        if isinstance(stmt, RemoveModification):  # about the modified form it takes away
            forms.reverse()
        every = True
    if not direct:
        if negated:
            raise ConstraintError(
                "there is no `not ..., indirectly`. In a model with feedback a chain of each sign "
                "usually exists, so it would refuse true models; write `X does not affect Y, "
                "indirectly` (no chain either way), or `not ..., directly`"
            )
        if regs:
            raise ConstraintError(
                "a regulator is a claim about one row's reaction; state it in a `directly` constraint"
            )
        if subject is None:
            raise ConstraintError(
                f"a {type(stmt).__name__} names no cause and effect for a chain to connect; write it "
                "`, directly`"
            )
        forms, every = (forms[:1], False) if every else (forms, every)
    return Constraint(cid, body, claim, direct, subject, forms, every, regs, negated, source)


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


def _text(agent) -> str:
    from . import notation

    try:
        return notation.agent_text(agent.to_json())
    except (ValueError, KeyError):
        return agent.name


def compatible(a, b) -> bool:
    """Whether one form of one agent can be both ``a`` and ``b``: the same name, and no condition of
    one contradicts the other's."""
    if _name(a.name) != _name(b.name):
        return False
    if any(
        (m.mod_type, m.residue, m.position, not m.is_modified)
        == (n.mod_type, n.residue, n.position, n.is_modified)
        for m in a.mods
        for n in b.mods
    ):
        return False
    if any(
        _name(x.agent.name) == _name(y.agent.name) and x.is_bound != y.is_bound
        for x in a.bound_conditions
        for y in b.bound_conditions
    ):
        return False
    if a.location and b.location and a.location != b.location:
        return False
    return not (
        a.activity
        and b.activity
        and a.activity.activity_type == b.activity.activity_type
        and a.activity.is_active != b.activity.is_active
    )


def refines(a, b) -> bool:
    """Whether ``a`` is ``b`` or a form of it: every condition ``b`` writes, ``a`` writes too (a
    residue or position ``b`` leaves out, any)."""
    if _name(a.name) != _name(b.name):
        return False
    if not all(
        any(
            m.mod_type == n.mod_type
            and m.is_modified == n.is_modified
            and n.residue in (None, m.residue)
            and n.position in (None, m.position)
            for m in a.mods
        )
        for n in b.mods
    ):
        return False
    if not all(
        any(
            _name(x.agent.name) == _name(y.agent.name) and x.is_bound == y.is_bound
            for x in a.bound_conditions
        )
        for y in b.bound_conditions
    ):
        return False
    if b.location and a.location != b.location:
        return False
    return not b.activity or bool(
        a.activity
        and a.activity.activity_type == b.activity.activity_type
        and a.activity.is_active == b.activity.is_active
    )


def _same(a, b) -> bool:
    """One agent in one state, by name (``Agent.matches`` keys on grounding)."""
    return _name(a.name) == _name(b.name) and a.matches(b)


def _reaction(stmt) -> tuple[list, list]:
    """What ``stmt`` takes and what it makes, each with ``1`` or ``-1`` (:data:`statements._REACTIONS`):
    :func:`statements._reaction` with its catalyst test by name, and a binding as its complex, each
    member bound to the others, made from each member free of them (as written, a member names no
    partner, so it read as possibly bound already; its own written condition on a partner gives
    way to the binding's)."""
    from indra.statements import BoundCondition, Complex

    from .statements import _REACTIONS

    if isinstance(stmt, Complex):
        whole, free = [], []
        for a in stmt.members:
            others = [b for b in stmt.members if b is not a]
            kept = [
                bc
                for bc in a.bound_conditions
                if _name(bc.agent.name) not in {_name(b.name) for b in others}
            ]
            w, f = deepcopy(a), deepcopy(a)
            w.bound_conditions = [*kept, *(BoundCondition(b) for b in others)]
            f.bound_conditions = [*kept, *(BoundCondition(b, False) for b in others)]
            whole.append((w, 1))
            free.append(f)
        return free, whole
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
            # A removal takes what it removes; that is its product's -1, not a second change.
            reactants = [r for r in reactants if not any(_same(r, p) for p, _ in made)]
            return reactants, made
    return [], []


def _subject(stmt):
    """What drives ``stmt``: its catalyst, or the enzyme of a self-modification; ``None``."""
    from indra.statements import SelfModification

    from .statements import _catalyst

    return stmt.enz if isinstance(stmt, SelfModification) else _catalyst(stmt)[0]


# -------------------------------------------------------------------------------------- claims
@dataclass
class _Step:
    """One reaction a row claims: what drives it, each ``(agent, sign, role)`` (sign ``None`` for a
    regulator its rate does not sign), and what it changes, each ``(agent, change)``."""

    prefix: str
    drivers: list[tuple[Any, int | None, str]]
    effects: list[tuple[Any, int]]


def steps(record: dict | None) -> list[_Step]:
    """The reactions ``record``'s rows (a statements record) claim, regulators as recorded."""
    from indra.statements import Complex, SelfModification

    from .statements import regulators

    out = []
    for row in (record or {}).get("rows", []):
        stmt = _stmt(row["statement"])
        if stmt is None:
            continue
        derived = (row.get("emission") or {}).get("derived") or {}
        reactants, products = _reaction(stmt)
        # An enzyme modifying itself catalyses no other molecule: it drives its row as its
        # reactant (read as a catalyst, a complex holding a modified one "made" more of the rest).
        catalyst = None if isinstance(stmt, SelfModification) else _subject(stmt)
        drivers = [(catalyst, 1, "catalyst")] if catalyst is not None else []
        for kind, given in regulators(row["statement"]).items():
            for i, agent in enumerate(given):
                drivers.append((agent, _SIGN[derived.get(f"{kind}_{i}", kind)], "regulator"))
        drivers += [(r, 1, "reactant") for r in reactants]
        out.append(_Step(row["prefix"], drivers, [*products, *((r, -1) for r in reactants)]))
        if isinstance(stmt, Complex):
            # Coming apart: driven by the complex, it raises each member free of the others and
            # lowers the complex, seen from each (else cAMP leaving PKA's R subunit raised total cAMP).
            out.append(
                _Step(
                    row["prefix"],
                    [(p, 1, "complex") for p, _ in products],
                    [*((r, 1) for r in reactants), *((p, -1) for p, _ in products)],
                )
            )
    return out


def _changes(step: _Step, d: int | None, own: bool) -> list[tuple[Any, int]]:
    """The changes of ``step`` driven by its driver ``d`` (``None``: by the reaction itself), but,
    unless ``own``, that driver's own: a driver that rises does not also fall by its own
    consumption, for the rows it drives next nor for a target it is part of (read so, a target's
    own decay, "more pS6, more of it dephosphorylated", ended every chain the other way). One row's
    net, read for a direct claim, counts it: binding EGFR leaves its total as it is."""
    if d is None or own:
        return step.effects
    driver = step.drivers[d][0]
    return [(e, c) for e, c in step.effects if not (e is driver or _same(e, driver))]


def _moves(changes: list[tuple[Any, int]], target, strict: bool = False) -> set[int]:
    """Which ways ``changes`` move ``target``'s amount: by their net over the forms within it, and,
    unless ``strict``, by their net over the forms that may be partly within it (a product whose
    location the row leaves to the model's default, read by a row that names one). Each part is
    netted on its own: a binding that moves EGFR from free to bound leaves EGFR{pY1068} as it is."""
    within = sum(c for e, c in changes if refines(e, target))
    partly = (
        0
        if strict
        else sum(c for e, c in changes if not refines(e, target) and compatible(e, target))
    )
    return {1 if n > 0 else -1 for n in (within, partly) if n}


def _moved(
    step: _Step, d: int | None, form: _Form, strict: bool = False, own: bool = False
) -> set[int]:
    """Which ways ``step``, driven by ``d``, moves ``form``, per unit of its flux. A form any change
    of which counts (``does not affect``) moves with any change of the agent; a declared active
    form's activity moves with its flux where ``d`` regulates a reaction the form catalyses."""
    changes = _changes(step, d, own)
    if form.way is None:
        return {1 if c > 0 else -1 for e, c in changes if compatible(e, form.agent)}
    out = _moves(changes, form.agent, strict)
    if (
        form.active
        and d is not None
        and step.drivers[d][2] == "regulator"
        and any(role == "catalyst" and compatible(a, form.agent) for a, _, role in step.drivers)
    ):
        out.add(1)
    return out


def _holds(step: _Step, d: int | None, flux: int, c: Constraint, want: int | None, **kw) -> bool:
    """Whether ``step``, driven by ``d`` at ``flux``, moves ``c``'s forms the way ``want``s: as
    claimed (``1``), the other way (``-1``) or any way (``None``); ``every`` form, or one."""

    def one(form: _Form) -> bool:
        ways = _moved(step, d, form, **kw)
        return (
            bool(ways)
            if want is None or form.way is None
            else (any(flux * g == want * form.way for g in ways))
        )

    return (all if c.every else any)(one(f) for f in c.forms)


# --------------------------------------------------------------------------------------- check
@dataclass
class Verdict:
    """Whether a constraint holds, and on what: the rows that claim it (``directly``), the chain
    that carries it and one of the other sign (``indirectly``), or why it does not."""

    ok: bool
    rows: tuple[str, ...] = ()
    chain: str | None = None
    opposite: str | None = None
    why: str | None = None

    def stored(self) -> dict[str, Any]:
        return {
            k: list(v) if k == "rows" else v for k, v in self.__dict__.items() if v or k == "ok"
        }


def check(constraints: Constraints, record: dict | None) -> dict[str, Verdict]:
    """Each constraint's :class:`Verdict` on the model ``record`` (a statements record, ``None``
    for a model not built from statements, which claims nothing)."""
    claimed = steps(record)
    graph = _Graph(claimed)
    return {c.id: _verdict(c, claimed, graph) for c in constraints.items}


def _verdict(c: Constraint, claimed: list[_Step], graph: _Graph) -> Verdict:
    unaffected = c.forms[0].way is None
    if c.direct:
        rows = _claiming(c, claimed)
        if c.negated or unaffected:
            who = f"row{'s' if len(rows) > 1 else ''} `{'`, `'.join(rows)}`"
            said = f"{who} claim{'' if len(rows) > 1 else 's'} " + (
                f"{_text(c.subject)} changes {_text(c.forms[0].agent)}" if unaffected else "it"
            )
            return Verdict(not rows, tuple(rows), why=said if rows else None)
        return Verdict(True, tuple(rows)) if rows else Verdict(False, why=_why_direct(c, claimed))
    if unaffected:
        chain = graph.search(c, None)
        return Verdict(
            chain is None, chain=chain, why=chain and f"a chain of claims connects them: {chain}"
        )
    same, opposite = graph.search(c, 1), graph.search(c, -1)
    if same is not None:
        return Verdict(True, chain=same, opposite=opposite)
    return Verdict(
        False,
        opposite=opposite,
        why=(
            f"its claims connect them only the other way: {opposite}"
            if opposite
            else graph.frontier(c)
        ),
    )


def _claiming(c: Constraint, claimed: list[_Step]) -> list[str]:
    """The rows whose one reaction claims what ``c`` asks: driven by its subject (as catalyst,
    regulator or reactant, of the kind its rate records) with ``c``'s regulators recorded on it, its
    own forms being ``c``'s (strict), not merely overlapping them."""
    want = None if c.forms[0].way is None else 1
    rows = []
    for s in claimed:
        if not all(
            any(
                role == "regulator" and k == _SIGN[kind] and compatible(a, agent)
                for a, k, role in s.drivers
            )
            for kind, agent in c.regulators
        ):
            continue
        ways = (
            [(1, None)]
            if c.subject is None
            else [
                (k, d)
                for d, (a, k, role) in enumerate(s.drivers)
                if role != "complex"
                and (k is not None or want is None)
                and compatible(a, c.subject)
            ]
        )
        if any(_holds(s, d, k or 1, c, want, strict=True, own=True) for k, d in ways):
            rows.append(s.prefix)
    return list(dict.fromkeys(rows))


def _why_direct(c: Constraint, claimed: list[_Step]) -> str:
    """Why no row claims ``c``: the rows that change what it is about, and what each lacks."""
    from .statements import REGULATORS

    form = c.forms[0]
    what = _text(form.agent) if not form.active else f"the activity of {form.agent.name}"
    changers = [s for s in claimed if _moved(s, None, form)]
    if not changers:
        return f"no row changes {what}"
    said = []
    for s in changers:
        found = [
            (d, k, role)
            for d, (a, k, role) in enumerate(s.drivers)
            if c.subject is not None and role != "complex" and compatible(a, c.subject)
        ]
        missing = [
            f"`{REGULATORS[kind]} {_text(a)}`"
            for kind, a in c.regulators
            if not any(
                r == "regulator" and kk == _SIGN[kind] and compatible(x, a)
                for x, kk, r in s.drivers
            )
        ]
        if not found:
            continue
        if all(k is None for _, k, _ in found):
            said.append(
                f"`{s.prefix}` states {_text(c.subject)}, but its rate does not sign it "
                "(recorded `modulated_by`)"
            )
        elif missing:
            said.append(
                f"`{s.prefix}` has {_text(c.subject)} but not {', '.join(missing)} as its "
                "rate records them"
            )
        elif not any(  # which way it moves what the constraint is about
            k is not None and k * g == form.way
            for d, k, _ in found
            for g in _moved(s, d, form, own=True)
        ):
            regulated = any(r == "regulator" for _, _, r in found)
            said.append(
                f"`{s.prefix}` has {_text(c.subject)} change {what} the other way"
                + (" (as its rate records it)" if regulated else "")
            )
        else:
            said.append(
                f"`{s.prefix}` has {_text(c.subject)} make {what}, but not from what the "
                "constraint's reaction takes"
            )
    if said:
        return "; ".join(dict.fromkeys(said))
    names = ", ".join(f"`{s.prefix}`" for s in changers[:6])
    if c.subject is None:
        return f"rows {names} change {what}, and none takes what it takes"
    return (
        f"rows {names} change {what}, and none has {_text(c.subject)} drive it: state it on "
        f"the row it acts on, as its catalyst or a regulator (`..., inhibited by "
        f"{_text(c.subject)}`, `activated by`)"
    )


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
        """The drivers ``(step, driver, way)`` of other rows that the changes of step ``i`` driven
        by ``d`` move, each way they move them."""
        if (i, d) not in self._next:
            changes = _changes(self.steps[i], d, own=False)
            out = set()
            for name in {_name(e.name) for e, _ in changes}:
                for j, dd in self.by_name.get(name, ()):
                    driver, _, role = self.steps[j].drivers[dd]
                    if self.steps[j].prefix != self.steps[i].prefix:
                        # A complex comes apart as there is more of IT: a member made bare does not
                        # raise it, though one form may be both.
                        out |= {
                            (j, dd, g) for g in _moves(changes, driver, strict=role == "complex")
                        }
            self._next[(i, d)] = sorted(out)
        return self._next[(i, d)]

    def _starts(self, subject, signed: bool):
        for i, s in enumerate(self.steps):
            for d, (agent, k, role) in enumerate(s.drivers):
                # A complex comes apart because there is more of it, not more of a member.
                bound_as = role != "complex" or {
                    _name(bc.agent.name) for bc in subject.bound_conditions if bc.is_bound
                } & {_name(bc.agent.name) for bc in agent.bound_conditions if bc.is_bound}
                if (k is not None or not signed) and bound_as and compatible(agent, subject):
                    yield i, d

    def search(self, c: Constraint, want: int | None) -> str | None:
        """The shortest chain from ``c``'s subject that moves its forms as claimed (``want`` 1), the
        other way (-1), or any way through any claim, unsigned ones included (``None``), written
        out; ``None`` if there is none."""
        signed = want is not None
        parent: dict[tuple, tuple | None] = {
            (i, d, 1): None for i, d in self._starts(c.subject, signed)
        }
        todo = deque(parent)
        while todo:
            state = i, d, way = todo.popleft()
            k = self.steps[i].drivers[d][1]
            flux = way * k if signed else 1
            if _holds(self.steps[i], d, flux, c, want):
                return self._written(state, parent, c, want)
            for j, dd, g in self.next(i, d):
                nxt = (j, dd, flux * g if signed else 1)
                if (not signed or self.steps[j].drivers[dd][1] is not None) and nxt not in parent:
                    parent[nxt] = state
                    todo.append(nxt)
        return None

    def _written(self, state, parent, c: Constraint, want: int | None) -> str:
        """The chain ending at ``state``: the subject, then each row and the form it moves, ``→``
        raising it and ``⊣`` lowering it."""
        chain = []
        while state is not None:
            chain.append(state)
            state = parent[state]
        chain.reverse()
        last = next((f for f in c.forms if f.active), c.forms[0])
        parts = [_text(c.subject)]
        for n, (i, d, way) in enumerate(chain):
            if n + 1 < len(chain):
                head, after = (
                    self.steps[chain[n + 1][0]].drivers[chain[n + 1][1]][0],
                    chain[n + 1][2],
                )
            else:
                head, after = last.agent, (want or 1) * (last.way or 1)
            parts.append(
                f"{'⊣' if want is not None and way * after < 0 else '→'} "
                f"({self.steps[i].prefix}) {_text(head)}"
            )
        return " ".join(parts)

    def frontier(self, c: Constraint) -> str:
        """Why no chain connects them: what the claims reach from ``c``'s subject, and the rows that
        change what ``c`` is about."""
        seen, todo = set(), deque(self._starts(c.subject, signed=False))
        reached = []
        while todo:
            if (state := todo.popleft()) in seen:
                continue
            seen.add(state)
            reached += [_text(e) for e, _ in _changes(self.steps[state[0]], state[1], own=False)]
            todo.extend((j, dd) for j, dd, _ in self.next(*state))
        reached = list(dict.fromkeys(reached))
        changers = sorted({s.prefix for s in self.steps if _moved(s, None, c.forms[0])})
        out = "no chain of claims connects them; " + (
            f"from {_text(c.subject)} the claims reach {', '.join(reached[:8])}"
            + (" ..." if len(reached) > 8 else "")
            if reached
            else f"no row's claim is driven by {_text(c.subject)}"
        )
        return out + (
            f"; the rows that change it are `{'`, `'.join(changers[:8])}`" if changers else ""
        )


# ------------------------------------------------------------------------------------ commit
def enforce(
    constraints: Constraints, record: dict | None, base: dict | None
) -> tuple[dict, str | None]:
    """Check ``record`` (the statements record a change would commit) against ``constraints``.
    Raises :class:`operations.OperationError` (cause ``constraint_violated``) naming each it breaks.
    Returns what the record keeps (the constraints as checked and each verdict), and a warning
    naming each indirect constraint whose chain of the other sign is new since ``base``, the record
    the change starts from, or ``None``."""
    verdicts = check(constraints, record)
    broken = [c for c in constraints.items if not verdicts[c.id].ok]
    if broken:
        from .operations import OperationError

        detail = "; ".join(f"`{c.id}` ({c.text}): {verdicts[c.id].why}" for c in broken)
        unbuilt = (
            ""
            if record
            else (
                "This model was not built from statements, and a model claims "
                "only what its rows state. "
            )
        )
        raise OperationError(
            f"make_changes: the model breaks {len(broken)} of the session's constraints: {detail}. "
            f"{unbuilt}NOTHING was applied.",
            cause="constraint_violated",
        )
    was = ((base or {}).get("constraints") or {}).get("verdicts") or {}
    said = "; ".join(
        f"`{c.id}` ({c.text}) holds through {verdicts[c.id].chain}, and a chain of the other sign "
        f"exists too: {verdicts[c.id].opposite}"
        for c in constraints.items
        if verdicts[c.id].opposite and not (was.get(c.id) or {}).get("opposite")
    )
    warning = said and (
        said + ". Which way the model moves each is the fit's to say, not its "
        "structure's: keep both if both are true."
    )
    kept = {
        "checked": constraints.texts(),
        "verdicts": {cid: v.stored() for cid, v in verdicts.items()},
    }
    return kept, warning or None
