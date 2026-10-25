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
``, indirectly`` asks it of the network the rows expand to, for the form the constraint is about,
traced from the subject (:mod:`tracing`). There is no default. ``not <statement>, directly``
holds where no row entails the statement, and ``X does not affect Y`` where no row (``directly``)
or no trace (``indirectly``) moves any form of Y from X, either way. ``<agent> is active``
(``inactive``, ``as kinase``) declares the forms ``activates`` and ``inhibits`` are about:
activity is what the user says, never inferred from the agent's rows, and an activity constraint
on an undeclared agent is refused as the file loads. A regulator of a reaction a declared active
form catalyses moves that activity, as ``as_indra`` projects it ("IBMX inhibits PDE").

Every claim is a row's own: its reaction (:data:`statements._REACTIONS`), its catalyst, and each
regulator as the kind its rate records (:attr:`mechanisms.Emission.derived`, signed on the
version's bounds by :func:`regulation.resign`), never the kind the statement states: a row stating
SOCS3 ``activated by`` on a rate that falls with it satisfies "SOCS3 decreases STAT5{pY694}". A
regulator its rate cannot sign (``modulated_by``) claims no direction, so it satisfies no signed
constraint, but it is an effect, so it breaks ``does not affect``.

An amount is read as a readout reads it: the sum over the species holding an agent in the stated
form, the total where none is stated, and only a net change counts (:func:`_moves`). A row that
moves an agent between forms inside the set it names changes nothing (``EGF binds EGFR`` leaves
total EGFR as it is).

Indirect constraints were first checked on chains of the rows' claims, read by hand from each
statement (#134's prototype and #136's first version). Each special case it needed (a binding's
complex coming apart, a link into a complex needing more of the complex itself, a step through the
same row twice, sequestration at any step) was a hand-written reading of mechanics that BioNetGen's
matching already decides, so tracing replaced it: on the formulations #134 measured it holds every
true claim, the one the chains refused included (PRKACA raising cAMP), and one false claim fewer.
The three false claims it holds each have a route of its sign beside the one that wins in
simulation (bachmann's SHP1 and SOCS3 feedbacks, boehm's STAT5A coming back from the
heterodimer). Whether a model moves its object one way or the other is not
decidable from its structure once it has feedback (on those formulations a route of the opposite
sign exists for most true indirect claims), so an indirect constraint holds wherever the trace
moves it as claimed, whatever else the trace also finds; which way the fitted model moves it is the
fit's to say. There is no signed indirect negative (``not X increases Y, indirectly``) for the same
reason: feedback breaks it on true models, and the way out would be deleting true feedback.

Agents are matched by NAME, the model's identity: two agents may share a grounding (bachmann's
SOCS3 and SOCS3_oe, a gene's nRNA, mRNA and protein), which is why :func:`_reaction` is
:func:`statements._reaction` with its catalyst test by name (PEtab-MCP#137). Direct constraints
read the statements record alone and cost milliseconds; an indirect one runs BioNetGen on the
model the rows build, once per subject (seconds to a minute on the formulations measured). A
constraint's statement is read by :func:`notation.statement`, the row reader's own; the
qualifier, ``not``, ``does not affect`` and ``is active`` are this module's.
"""

from __future__ import annotations

import re
from collections.abc import Callable
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
    recorded on it, or traced through the network (:mod:`tracing`); ``negated``, that no row
    does."""

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
            f"it) or `{body}, indirectly` (the network the rows expand to carries it)"
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
                "there is no `not ..., indirectly`. In a model with feedback the network usually "
                "moves it each way, so it would refuse true models; write `X does not affect Y, "
                "indirectly` (no traced change either way), or `not ..., directly`"
            )
        if regs:
            raise ConstraintError(
                "a regulator is a claim about one row's reaction; state it in a `directly` constraint"
            )
        if subject is None:
            raise ConstraintError(
                f"a {type(stmt).__name__} names no cause and effect to trace; write it `, directly`"
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
    from indra.statements import SelfModification

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
    return out


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


def _moved(step: _Step, d: int | None, form: _Form, strict: bool = False) -> set[int]:
    """Which ways ``step``, driven by ``d``, moves ``form``, per unit of its flux. A form any change
    of which counts (``does not affect``) moves with any change of the agent; a declared active
    form's activity moves with its flux where ``d`` regulates a reaction the form catalyses."""
    if form.way is None:
        return {1 if c > 0 else -1 for e, c in step.effects if compatible(e, form.agent)}
    out = _moves(step.effects, form.agent, strict)
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
    claimed (``1``) or any way (``None``); ``every`` form, or one."""

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
    """Whether a constraint holds, and on what: the rows that claim it (``directly``) or whose
    reactions move its form, traced from its subject (``indirectly``); or why it does not; or,
    held, why it could not be checked."""

    ok: bool
    rows: tuple[str, ...] = ()
    why: str | None = None
    unchecked: str | None = None

    def stored(self) -> dict[str, Any]:
        return {
            k: list(v) if k == "rows" else v for k, v in self.__dict__.items() if v or k == "ok"
        }


def check(
    constraints: Constraints, record: dict | None, model: Callable[[], tuple] | None = None
) -> dict[str, Verdict]:
    """Each constraint's :class:`Verdict` on the model ``record`` (a statements record, ``None``
    for a model not built from statements, which claims nothing). ``model`` builds the PySB model
    the record's rows build, seeded, and its emissions (:func:`operations.statements_model`): what
    an indirect constraint is traced on, built only where there is one."""
    claimed = steps(record)
    indirect = [c for c in constraints.items if not c.direct]
    traced = _traced(indirect, record, model) if indirect else {}
    return {c.id: traced[c.id] if c.id in traced else _direct(c, claimed) for c in constraints.items}


def _direct(c: Constraint, claimed: list[_Step]) -> Verdict:
    rows = _claiming(c, claimed)
    if c.negated or c.forms[0].way is None:
        unaffected = c.forms[0].way is None
        who = f"row{'s' if len(rows) > 1 else ''} `{'`, `'.join(rows)}`"
        said = f"{who} claim{'' if len(rows) > 1 else 's'} " + (
            f"{_text(c.subject)} changes {_text(c.forms[0].agent)}" if unaffected else "it"
        )
        return Verdict(not rows, tuple(rows), why=said if rows else None)
    return Verdict(True, tuple(rows)) if rows else Verdict(False, why=_why_direct(c, claimed))


def _traced(
    items: list[Constraint], record: dict | None, model: Callable[[], tuple] | None
) -> dict[str, Verdict]:
    """The indirect constraints' verdicts, one trace per subject (:mod:`tracing`): a model whose
    rows move none of a constraint's forms from its subject, or not as claimed, breaks it. A
    trace that fails is no reason to refuse a model: it holds, unchecked, and says why."""
    import logging

    from . import tracing

    if not (record or {}).get("rows"):
        return {c.id: Verdict(c.forms[0].way is None, why=None if c.forms[0].way is None else
                              "no row claims anything") for c in items}
    try:
        built, emissions = model()
    except Exception as exc:  # noqa: BLE001 - a check that breaks refuses nothing; never a crash
        logging.getLogger(__name__).exception("building the model to trace constraints on failed")
        return {c.id: Verdict(True, unchecked=f"its model could not be built to trace: {exc}")
                for c in items}
    derived = {row["prefix"]: (row.get("emission") or {}).get("derived") or {}
               for row in record["rows"]}
    claimed = steps(record)
    groups: dict[tuple[str, bool], list[Constraint]] = {}
    for c in items:
        groups.setdefault((_text(c.subject), c.forms[0].way is None), []).append(c)
    out = {}
    for (subject_text, unsigned), cs in groups.items():
        subject = cs[0].subject
        targets = {
            (c.id, i): (f.agent, f.active, f.way is None)
            for c in cs
            for i, f in enumerate(c.forms)
            if _formed(built, f.agent)
        }
        if not _formed(built, subject):
            moved, reached = {}, []
        else:
            try:
                found = tracing.trace(built, emissions, derived, subject, targets, unsigned=unsigned)
            except Exception as exc:  # noqa: BLE001 - BioNetGen refusing refuses nothing
                logging.getLogger(__name__).exception("tracing %s failed", subject_text)
                out |= {c.id: Verdict(True, unchecked=f"tracing {subject_text} failed: {exc}")
                        for c in cs}
                continue
            moved, reached = found.moved, found.reached
        for c in cs:
            out[c.id] = _read_trace(c, moved, reached, claimed)
    return out


def _formed(model, agent) -> bool:
    """Whether ``model`` has the form ``agent`` writes: its monomer, with the sites it names."""
    from .mechanisms import agent_pattern

    try:
        agent_pattern(model, agent)
    except Exception:  # noqa: BLE001 - a form the rows never write is formed nowhere
        return False
    return True


def _read_trace(c: Constraint, moved: dict, reached: list[str], claimed: list[_Step]) -> Verdict:
    """``c``'s verdict on what tracing its subject moved: any of its forms the way it claims (any
    way, for ``does not affect``, which holds where none moves)."""
    from .tracing import Moved

    found = [(f, moved.get((c.id, i)) or Moved()) for i, f in enumerate(c.forms)]
    subject = _text(c.subject)
    if c.forms[0].way is None:
        hit = [(f, m) for f, m in found if m.raised or m.lowered]
        if not hit:
            return Verdict(True)
        rows = tuple(dict.fromkeys(r for _, m in hit for r in (*m.rows[1], *m.rows[-1])))
        return Verdict(False, rows, why=f"tracing {subject} moves {_text(hit[0][0].agent)}"
                       + (f" through rows `{'`, `'.join(rows)}`" if rows else ""))
    for f, m in found:
        if (m.raised if f.way > 0 else m.lowered):
            return Verdict(True, tuple(m.rows[f.way]))
    form = c.forms[0]
    what = _text(form.agent) if not form.active else f"the activity of {form.agent.name}"
    way = "raises" if form.way > 0 else "lowers"
    changers = sorted({s.prefix for s in claimed if _moved(s, None, form)})
    traced = [name for name in reached if name != _name(c.subject.name)]
    return Verdict(False, why=(
        f"tracing {subject} through the network its rows expand to never {way} {what}; "
        + (f"it reaches {', '.join(traced[:8])}" + (" ..." if len(traced) > 8 else "")
           if traced else "it reaches no other agent")
        + (f"; the rows that change {what} are `{'`, `'.join(changers[:8])}`" if changers else "")
    ))


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
                for d, (a, k, _) in enumerate(s.drivers)
                if (k is not None or want is None) and compatible(a, c.subject)
            ]
        )
        if any(_holds(s, d, k or 1, c, want, strict=True) for k, d in ways):
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
            if c.subject is not None and compatible(a, c.subject)
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
            for g in _moved(s, d, form)
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


# ------------------------------------------------------------------------------------ commit
def enforce(
    constraints: Constraints, record: dict | None, model: Callable[[], tuple] | None = None
) -> tuple[dict, str | None]:
    """Check ``record`` (the statements record a change would commit, ``model`` building its
    model, :func:`check`) against ``constraints``. Raises :class:`operations.OperationError`
    (cause ``constraint_violated``) naming each it breaks. Returns what the record keeps (the
    constraints as checked and each verdict), and a warning naming each that could not be
    checked, or ``None``."""
    verdicts = check(constraints, record, model)
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
    said = "; ".join(
        f"`{c.id}` ({c.text}) was not checked: {verdicts[c.id].unchecked}"
        for c in constraints.items
        if verdicts[c.id].unchecked
    )
    kept = {
        "checked": constraints.texts(),
        "verdicts": {cid: v.stored() for cid, v in verdicts.items()},
    }
    return kept, said or None
