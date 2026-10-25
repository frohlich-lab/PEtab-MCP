"""Mechanisms as PySB rules over a statement's roles: read off a library entry, or written new.

A *scheme* is one PySB rule per line in PySB's own shorthand, with the molecules named by the
statement type's ROLES rather than by agents, so one scheme applies to every statement of that
type::

    enz(sub=None) + sub(enz=None, mod='off') >> enz(sub=1) % sub(enz=1, mod='off') @ kf
    enz(sub=1) % sub(enz=1, mod='off') >> enz(sub=None) + sub(enz=None, mod='on') @ kc
    enz(sub=1) % sub(enz=1) >> enz(sub=None) + sub(enz=None) @ kr

- A molecule is a role (``enz``, ``sub``, ``subj``, ``obj``; a list role by position,
  ``members_0``, or by its own name when the statement gives it one agent), each the agent the
  row writes there, in the states it writes; or, by its name, another agent the statement names:
  a partner a role is written bound or not bound to (``C`` of ``enz{~C}``), in the states written
  there; or, by its kind, a regulator the statement states (``inhibited_by``, by position where
  it states several, ``inhibited_by_1``), in the states the statement writes on it and with only
  binding sites of the scheme's: ``enz(inhibited_by=None) + inhibited_by(enz=None) |
  enz(inhibited_by=1) % inhibited_by(enz=1)`` binds a drug the enzyme cannot act while bound to.
  A regulator binds on sites named after each other, so the complex is the one the partner
  notation writes (``MEK{!Drug}``: MEK's site named after Drug, Drug's after MEK), which other
  rows read and bind. No other molecule: a mechanism reads only its statement's agents
  (:func:`statements.named_agents`) and the regulators it states, and a row naming another is
  refused (:func:`mechanisms.refuse_unnamed`).
- A site is ``mod``, the statement's own modification (``'off'``, ``'on'``); ``act``, the role's
  activity (``'inactive'``, ``'active'``): the statement's ``obj_activity`` on its object, the
  activity the modification implies on an enzyme (``kinase`` for a phosphorylation); a role's
  name or a regulator's, the binding site for that partner; or any other name, taken as written.
  Written with a state, a site is one its molecule has, in a state it takes (``mod`` and ``act``
  always are), or one the scheme adds with every state it takes, the one it starts in first,
  under a name that is no slip of one the molecule has. A molecule without the site ``act``
  names is refused, naming the activity it has, rather than ``act`` read as that one: a row
  giving the molecule a second activity would then re-point the scheme with no edit to it.
- A site's value is a state, a bond number, ``None`` (free), ``ANY`` (bound to something) or
  ``WILD``; a ``(state, bond)`` pair is both. ``>>`` is one-way, ``|`` both ways, ``%`` joins a
  complex, ``None`` is nothing.
- ``@`` gives the rate: one for ``>>``, two for ``|``. A rate is an expression over the
  definition's ``parameters`` (each minted per row as ``<prefix>_<name>`` and shared or fixed
  like any, ``kf=k_on`` on the row), ``amount(<pattern>)`` (the amount of what the pattern of
  those molecules matches), ``amount(inhibited_by)`` (a regulator the statement states, as it
  states it, by position where it states several, ``amount(inhibited_by_1)``), ``time`` and
  exp/log/sqrt; mass action multiplies it by every reactant's amount, as PySB does. A regulator,
  read in a rate or bound as a molecule, is checked against the claim on the row's net rate
  (:mod:`regulation`), which holds one the rules bind at its free amount, the enzyme's forms bound
  to it among the intermediates. A name a rate does not declare is refused: a parameter the
  problem ships or a condition input enters through a row's binding, as for a written ``rate``.
  ``delete_molecules`` after the rates removes only the molecules that vanish, not their complex.

:func:`describe` writes any library entry this way, from a statement of placeholder agents named
by role, and :func:`rate_parameters` reads the parameters it names; :func:`compile_scheme` turns a
scheme and those parameters into a library entry. The two invert each other: an entry's
description, compiled, builds the same rules (``tests/test_schemes.py``).
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass

import pysb
import sympy
from indra.assemblers.pysb.assembler import _n, get_site_pattern, mod_acttype_map
from indra.assemblers.pysb.sites import get_binding_site_name, get_mod_site_name, states
from indra.statements import (
    Agent,
    BoundCondition,
    Modification,
    RegulateActivity,
    Statement,
    get_statement_by_name,
)
from pysb import ANY, WILD
from pysb.core import ReactionPattern, RuleExpression, time

from .errors import MechanismError
from .statements import REGULATOR_NAME, REGULATORS, by_name, named_agents

MOD, ACT = "mod", "act"
#: ``amount(inhibited_by)`` (or ``amount(inhibited_by())``) in a rate: a regulator, by its name.
_REGULATOR_AMOUNT = re.compile(rf"amount\(\s*({REGULATOR_NAME.pattern})\s*(?:\(\s*\))?\s*\)")
_OFF_ON = ("off", "on")
_ACTIVITY = ("inactive", "active")
_FUNCTIONS = {"exp": sympy.exp, "log": sympy.log, "sqrt": sympy.sqrt}
#: The operators a rate takes (``^`` is a power, as in a written ``rate``).
_ARITHMETIC = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.BitXor)
_SITE_VALUE_NAMES = {"None": None, "ANY": ANY, "WILD": WILD}


# ------------------------------------------------------------------------------- reading
@dataclass(frozen=True)
class Molecule:
    name: str
    sites: tuple[tuple[str, object], ...]  # (site, value); value as PySB takes it


@dataclass(frozen=True)
class Line:
    lhs: tuple[tuple[Molecule, ...], ...]  # complexes, each its molecules
    rhs: tuple[tuple[Molecule, ...], ...]
    reversible: bool
    rates: tuple[str, ...]
    delete_molecules: bool


def parse(text: str) -> list[Line]:
    """The scheme's lines. Raises :class:`MechanismError` naming the line that does not read."""
    lines = []
    for number, raw in enumerate(re.split(r"[\n;]", text), 1):
        raw = raw.strip()
        if not raw or raw.startswith("#"):
            continue
        try:
            lines.append(_line(raw))
        except (SyntaxError, MechanismError, ValueError) as exc:
            raise MechanismError(f"scheme line {number} ({raw!r}): {exc}") from None
    if not lines:
        raise MechanismError("the scheme has no rules")
    return lines


def _line(raw: str) -> Line:
    if "@" not in raw:
        raise MechanismError("give the rate after `@`, e.g. `... >> enz(sub=1) % sub(enz=1) @ kf`")
    rule, rates = raw.split("@", 1)
    rates = rates.strip()
    delete = rates.endswith("delete_molecules")
    if delete:
        rates = rates[: -len("delete_molecules")].rstrip().rstrip(",").rstrip()
    rate_list = [r.strip() for r in _split_top(rates, ",") if r.strip()]
    tree = ast.parse(rule.strip(), mode="eval").body
    if not isinstance(tree, ast.BinOp) or not isinstance(tree.op, (ast.RShift, ast.BitOr)):
        raise MechanismError("a rule is `reactants >> products` or `reactants | products`")
    reversible = isinstance(tree.op, ast.BitOr)
    if len(rate_list) != (2 if reversible else 1):
        raise MechanismError(
            f"a `{'|' if reversible else '>>'}` rule takes {2 if reversible else 1} rate(s) after "
            f"`@`, got {rate_list}"
        )
    lhs, rhs = _side(tree.left), _side(tree.right)
    _refuse_open_bonds(lhs), _refuse_open_bonds(rhs)
    _refuse_regulator_sites(lhs), _refuse_regulator_sites(rhs)
    return Line(lhs, rhs, reversible, tuple(rate_list), delete)


def _split_top(text: str, sep: str) -> list[str]:
    out, depth, cur = [], 0, ""
    for ch in text:
        depth += ch == "("
        depth -= ch == ")"
        if ch == sep and depth == 0:
            out.append(cur)
            cur = ""
        else:
            cur += ch
    return out + [cur]


def _side(node) -> tuple[tuple[Molecule, ...], ...]:
    if isinstance(node, ast.Constant) and node.value is None:
        return ()
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return _side(node.left) + _side(node.right)
    return (_complex(node),)


def _complex(node) -> tuple[Molecule, ...]:
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mod):
        return _complex(node.left) + _complex(node.right)
    return (_molecule(node),)


def _molecule(node) -> Molecule:
    if isinstance(node, ast.Name):
        return Molecule(node.id, ())
    if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)) or node.args:
        raise MechanismError(f"{ast.unparse(node)!r} is no molecule: write `name(site=value, ...)`")
    return Molecule(node.func.id, tuple((kw.arg, _value(kw.value)) for kw in node.keywords))


def _value(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (str, int)) and not isinstance(
        node.value, bool
    ):
        return node.value
    if isinstance(node, ast.Constant) and node.value is None:
        return None
    if isinstance(node, ast.Name) and node.id in _SITE_VALUE_NAMES:
        return _SITE_VALUE_NAMES[node.id]
    if isinstance(node, ast.Tuple) and len(node.elts) == 2:
        return (_value(node.elts[0]), _value(node.elts[1]))
    raise MechanismError(f"site value {ast.unparse(node)!r}: a state, a bond number, None, ANY, WILD")


def _bond(value) -> int | None:
    value = value[1] if isinstance(value, tuple) else value
    return value if isinstance(value, int) else None


def _refuse_open_bonds(side: tuple[tuple[Molecule, ...], ...]) -> None:
    """Refuse a bond number that is not on exactly two sites of its complex, which PySB refuses
    only once a row builds the rule, in an exception of its own."""
    ends = [
        [(f"{m.name}({s}={v!r})", _bond(v)) for m in complex_ for s, v in m.sites
         if _bond(v) is not None]
        for complex_ in side
    ]
    for i, here in enumerate(ends):
        for bond in {b for _, b in here}:
            on = [text for text, b in here if b == bond]
            if len(on) == 2:
                continue
            apart = [text for j, other in enumerate(ends) if j != i for text, b in other if b == bond]
            if len(on) == 1 and len(apart) == 1:
                raise MechanismError(
                    f"bond {bond} joins `{on[0]}` and `{apart[0]}`, which `+` writes as two "
                    f"complexes: join them with `%`, `{on[0]} % {apart[0]}`"
                )
            raise MechanismError(
                f"bond {bond} is on {len(on)} site(s) of its complex ({', '.join(on)}), not two: "
                "a bond number is on the two sites it joins, in one complex, as `A(b=1) % B(a=1)`"
            )


def _refuse_regulator_sites(side: tuple[tuple[Molecule, ...], ...]) -> None:
    """Refuse a regulator's molecule (``inhibited_by``) written with a state, or bound on sites not
    named after each other: a scheme binds a regulator as its statement states it, and the
    complex it forms is the one the partner notation writes, which other rows read and bind."""
    for complex_ in side:
        ends: dict[int, list[tuple[str, str]]] = {}
        for m in complex_:
            for site, value in m.sites:
                state = value[0] if isinstance(value, tuple) else value
                if REGULATOR_NAME.fullmatch(m.name) and (isinstance(state, str) or site in (MOD, ACT)):
                    kind = re.sub(r"_\d+$", "", m.name)
                    raise MechanismError(
                        f"`{m.name}({site}={state!r})` writes a state on a regulator: a scheme "
                        f"binds a regulator as its statement states it, so its molecule "
                        f"takes binding sites only (`{m.name}(enz=None)`), and a state it needs is "
                        f"written on the statement's agent (`{REGULATORS[kind]} X{{pS10}}`)"
                    )
                if _bond(value) is not None:
                    ends.setdefault(_bond(value), []).append((m.name, site))
        for bond, pair in ends.items():
            if len(pair) != 2 or not any(REGULATOR_NAME.fullmatch(name) for name, _ in pair):
                continue
            (a, at_a), (b, at_b) = pair
            if (at_a, at_b) != (b, a):
                raise MechanismError(
                    f"bond {bond} joins `{a}({at_a}={bond})` and `{b}({at_b}={bond})`: a regulator "
                    f"binds on sites named after each other, `{a}({b}={bond}) % {b}({a}={bond})`, "
                    "so the complex is the one the notation writes `E{!X}`, which other rows read; "
                    "name each site so wherever the scheme writes it"
                )


# ------------------------------------------------------------------------------- roles
def roles(stmt_type: str) -> list[str]:
    return list(get_statement_by_name(stmt_type)._agent_order)


def role_agents(stmt: Statement) -> dict[str, Agent]:
    """Each role the statement fills, by name: a list role by position and, holding one agent,
    by its own name too."""
    agents = {}
    for role in stmt._agent_order:
        value = getattr(stmt, role, None)
        if isinstance(value, Agent):
            agents[role] = value
        elif isinstance(value, list):
            agents |= {f"{role}_{i}": a for i, a in enumerate(value)}
            if len(value) == 1:
                agents[role] = value[0]
    return agents


def _by_name(stmt: Statement) -> dict[str, Agent]:
    """Every agent the statement names (:func:`statements.named_agents`), by its molecule name; the
    first where two share one."""
    agents: dict[str, Agent] = {}
    for agent in named_agents(stmt):
        agents.setdefault(_n(agent.name), agent)
    return agents


def _molecule_agent(stmt: Statement, name: str, agents: dict[str, Agent]) -> Agent:
    """The agent a scheme's molecule ``name`` is: a role's, or another the statement names."""
    agent = agents.get(name) or _by_name(stmt).get(name)
    if agent is None:
        raise MechanismError(f"the statement gives no {name}")
    return agent


def _mod_states(stmt: Statement) -> tuple[str, str] | None:
    return tuple(states[stmt._get_mod_condition().mod_type]) if isinstance(stmt, Modification) else None


def _activity_site(stmt: Statement, role: str) -> str | None:
    """The site ``act`` names on ``role``: the statement's object activity, an enzyme's implied
    activity, or the activity a subject is written with (INDRA's default ``activity``)."""
    if isinstance(stmt, RegulateActivity) and role == "obj":
        return stmt.obj_activity
    if isinstance(stmt, Modification) and role == "enz":
        return mod_acttype_map.get(type(stmt), "activity")
    agent = role_agents(stmt).get(role) or _by_name(stmt).get(role)
    if agent is not None:
        return agent.activity.activity_type if agent.activity else "activity"
    return None


def _site_name(stmt: Statement, role: str, site: str, agents: dict[str, Agent]) -> str:
    if site == MOD:
        if not isinstance(stmt, Modification):
            raise MechanismError(f"`{MOD}` is a modification statement's own site")
        return get_mod_site_name(stmt._get_mod_condition())
    if site == ACT:
        name = _activity_site(stmt, role)
        if name is None:
            raise MechanismError(f"`{ACT}` names no activity on {role}")
        return name
    if site in agents:
        return get_binding_site_name(agents[site])
    return site


def _site_value(stmt: Statement, site: str, value):
    """``mod`` and ``act`` states as PySB writes them for this statement."""
    if isinstance(value, tuple):
        return (_site_value(stmt, site, value[0]), value[1])
    if site == MOD and isinstance(value, str):
        if value not in _OFF_ON:
            raise MechanismError(f"`{MOD}` is 'off' or 'on', not {value!r}")
        return _mod_states(stmt)[_OFF_ON.index(value)]
    if site == ACT and isinstance(value, str) and value not in _ACTIVITY:
        raise MechanismError(f"`{ACT}` is 'inactive' or 'active', not {value!r}")
    return value


# ------------------------------------------------------------------------------- compiling
@dataclass(frozen=True)
class _Rate:
    names: tuple[str, ...]  # parameters
    amounts: tuple[str, ...]  # amount(...) patterns, as written
    text: str
    #: amount(...) patterns written in the row notation (`A{pS10}`), each read in ``text`` as
    #: the name `__agent<k>`
    agents: tuple[str, ...] = ()
    #: the regulators read as `amount(inhibited_by)`, by the name the rate reads each by
    regulators: tuple[str, ...] = ()


def _notation_amounts(text: str) -> tuple[str, tuple[str, ...]]:
    """``text`` with every ``amount(<pattern in the row notation>)`` replaced by ``__agent<k>``,
    and those patterns. The notation's braces are no Python, so before this a rate reading
    `amount(A{pS10})` did not parse at all."""
    out, agents, i = [], [], 0
    while (j := text.find("amount(", i)) >= 0:
        depth, k = 0, j + len("amount")
        for k in range(j + len("amount"), len(text)):
            depth += {"(": 1, ")": -1}.get(text[k], 0)
            if depth == 0:
                break
        if depth:  # unbalanced: left as written, for the parse below to refuse
            break
        inner = text[j + len("amount("):k]
        if "{" in inner:
            out.append(text[i:j] + f"__agent{len(agents)}")
            agents.append(inner.strip())
        else:
            out.append(text[i:k + 1])
        i = k + 1
    out.append(text[i:])
    return "".join(out), tuple(agents)


def _rate(text: str) -> _Rate:
    """A rate's parameters (its names outside ``amount()``) and its ``amount()`` patterns."""
    written = text
    text, agents = _notation_amounts(text)
    try:
        tree = ast.parse(text, mode="eval").body
    except SyntaxError:
        raise MechanismError(
            f"rate {written!r} does not read: an expression over the rate's parameters, `time`, "
            "exp/log/sqrt and `amount(<molecule>)`, the molecule written as in a row "
            "(`amount(A{pS10})`) or as in the scheme (`amount(sub(mod='on'))`)"
        ) from None
    names: list[str] = []
    amounts: list[str] = []
    regulators: list[str] = []

    def visit(node) -> None:
        if isinstance(node, ast.Call):
            fn = getattr(node.func, "id", None)
            if fn == "amount":
                if len(node.args) != 1 or node.keywords:
                    raise MechanismError(f"rate {text!r}: amount() takes one pattern")
                arg = node.args[0]
                called = isinstance(arg, ast.Call) and not arg.args and not arg.keywords
                read = getattr(arg.func if called else arg, "id", None)
                bare = called or isinstance(arg, ast.Name)
                if bare and read and REGULATOR_NAME.fullmatch(read):
                    regulators.append(read)
                    return
                try:
                    _refuse_open_bonds((_complex(node.args[0]),))
                    _refuse_regulator_sites((_complex(node.args[0]),))
                except MechanismError as exc:
                    raise MechanismError(f"rate {text!r}: {exc}") from None
                amounts.append(ast.unparse(node.args[0]))
                return  # its names are molecules and sites
            if fn not in _FUNCTIONS:
                raise MechanismError(f"rate {text!r}: no function {fn!r}; amount, exp, log, sqrt")
            if node.keywords or not 1 <= len(node.args) <= (2 if fn == "log" else 1):
                takes = "one argument, or a base too" if fn == "log" else "one argument"
                raise MechanismError(f"rate {text!r}: {fn}() takes {takes}")
            for arg in node.args:
                visit(arg)
        elif isinstance(node, ast.Name):
            if node.id != "time" and not node.id.startswith("__agent"):
                names.append(node.id)
        elif isinstance(node, ast.BinOp) and isinstance(node.op, _ARITHMETIC):
            visit(node.left), visit(node.right)
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            visit(node.operand)
        elif not (isinstance(node, ast.Constant) and isinstance(node.value, (int, float))):
            raise MechanismError(f"rate {text!r}: {ast.unparse(node)!r} is not allowed")

    visit(tree)
    return _Rate(tuple(sorted(set(names))), tuple(amounts), text, agents, tuple(regulators))


def compile_scheme(definition: dict, problem_ids: dict[str, str] | None = None):
    """A library entry from ``{statement_type, policy, scheme, parameters}``: the scheme's rules,
    built for each row from its own agents (see the module docstring). A rate naming anything but
    the declared ``parameters`` is refused as :func:`mechanisms.written` refuses one, with
    ``problem_ids``."""
    from .mechanisms import refuse_rate_names, refuse_unread

    stmt_type, policy = definition["statement_type"], definition["policy"]
    lines = parse(definition["scheme"])
    known_roles = roles(stmt_type)
    rates = [[_rate(r) for r in line.rates] for line in lines]
    declared = list(definition.get("parameters") or [])
    for line, line_rates in zip(lines, rates):
        for text, rate in zip(line.rates, line_rates):
            if unknown := sorted(set(rate.names) - set(declared)):
                also = ", `amount(<pattern>)` and exp/log/sqrt"
                refuse_rate_names(text, policy, unknown, [*declared, "time"], also,
                                  "`amount(<pattern>)`", problem_ids, known_roles)
    refuse_unread(policy, declared, {n for rs in rates for r in rs for n in r.names})
    patterns = [m for line in lines for side in (line.lhs, line.rhs) for c in side for m in c]
    patterns += [_molecule(n) for rs in rates for r in rs for a in r.amounts
                 for n in _complex_nodes(a)]
    molecules = sorted({m.name for m in patterns if REGULATOR_NAME.fullmatch(m.name)})
    read = {
        s.name for rs in rates for r in rs for s in _sympy(r).free_symbols
        if REGULATOR_NAME.fullmatch(s.name)
    } | set(molecules)

    def policy_fn(stmt: Statement, model: pysb.Model, prefix: str, regulators=None):
        from .mechanisms import Emission, _add

        agents = role_agents(stmt) | (regulators or {})
        params: dict[str, pysb.Parameter] = {}
        amounts: dict[str, str] = {}
        rules = []
        for i, (line, line_rates) in enumerate(zip(lines, rates)):
            lhs = _pattern_side(stmt, model, line.lhs, agents, fresh=None)
            carried = {}
            for cp in lhs:
                for mp in cp.monomer_patterns:
                    carried[mp.monomer.name] = carried.get(mp.monomer.name, 0) + 1
            rhs = _pattern_side(stmt, model, line.rhs, agents, fresh=carried)
            values = []
            for j, rate in enumerate(line_rates):
                for name in rate.names:
                    if name not in params:
                        params[name] = pysb.Parameter(f"{prefix}_{name}", 1.0, _export=False)
                        _add(model, params[name])
                values.append(_rate_value(stmt, model, rate, params, agents,
                                          f"{prefix}_rate{i}_{j}", regulators or {}, amounts))
            rule = pysb.Rule(
                f"{prefix}_r{i}",
                RuleExpression(ReactionPattern(lhs), ReactionPattern(rhs), line.reversible),
                *values,
                delete_molecules=line.delete_molecules,
                _export=False,
            )
            _add(model, rule)
            rules.append(rule.name)
        return Emission(
            (stmt_type, policy),
            prefix,
            {r: ",".join(a.name for a in v) if isinstance(v, list) else (v.name if v else "")
             for r in known_roles for v in [getattr(stmt, r, None)]},
            rules=tuple(rules),
            parameters=tuple(p.name for p in params.values()),
            regulator_molecules=tuple(molecules),
            amounts=amounts,
        )

    def scheme_sites(
        stmt: Statement, given: dict, prefix: str, regulators=None
    ) -> list[tuple[Agent, str, list | None]]:
        """The sites the scheme's molecules need on each monomer, for the monomer pass. A state
        it writes is checked against ``given``, each monomer's sites and their states before any
        scheme adds its own (``{name: (sites, site_states)}``; see the module docstring). A
        molecule row ``prefix``'s statement does not name is refused, before it has a monomer,
        saying how a mechanism reads one of the ``regulators`` (kind to agents) the row states,
        and so is a regulator's molecule it does not state."""
        from .mechanisms import refuse_unnamed, refuse_unstated

        refuse_unstated(prefix, (stmt_type, policy), molecules, regulators or {})
        agents = role_agents(stmt) | by_name(regulators or {})
        named = _by_name(stmt)
        need = []
        added: dict[tuple[str, str], list] = {}
        # A molecule a rate reads in the row notation brings the sites its pattern names.
        read = [_notation_agent(a) for rs in rates for r in rs for a in r.agents]
        for agent in read:
            if _n(agent.name) not in named:
                refuse_unnamed(prefix, stmt, f"mechanism {policy!r}", agent, regulators)
        need += [(agent, None, None) for agent in read]
        for mol in patterns:
            if mol.name not in agents and mol.name not in named and mol.name not in known_roles:
                refuse_unnamed(prefix, stmt, f"mechanism {policy!r}", Agent(mol.name), regulators)
            agent = _molecule_agent(stmt, mol.name, agents)
            sites, site_states = given.get(_n(agent.name), ((), {}))
            for site, value in mol.sites:
                value = value[0] if isinstance(value, tuple) else value
                written = f"mechanism {policy!r}: `{mol.name}({site}={value!r})`"
                if site == MOD and not isinstance(stmt, Modification):
                    own = _own_sites(mol.name, site, value, sites, site_states)
                    itself = f"the site itself, as {' or '.join(own)}, or " if own else ""
                    raise MechanismError(
                        f"{written}: `{MOD}` is a modification statement's own site, and "
                        f"{type(stmt).__name__} is none. Write {itself}the modification on the "
                        f"row's agent, which every `{mol.name}` in the scheme carries; "
                        f"{_sites_text(agent.name, sites, site_states)}"
                    )
                name = _site_name(stmt, mol.name, site, agents)
                if isinstance(value, str):
                    state = _site_value(stmt, site, value)
                    if name in sites:
                        if state not in site_states.get(name, ()):
                            raise MechanismError(
                                f"{written} names a state {agent.name}'s site {name!r} does not "
                                f"take; {_sites_text(agent.name, sites, site_states)}"
                            )
                    elif site in (MOD, ACT):
                        own = _own_sites(mol.name, site, value, sites, site_states)
                        raise MechanismError(
                            f"{written} is site {name!r}, which {agent.name} does not have; "
                            f"{_sites_text(agent.name, sites, site_states)}"
                            + (f". Write {' or '.join(own)} instead" if own else "")
                        )
                    elif state not in added.setdefault((agent.name, name), []):
                        added[(agent.name, name)].append(state)
                if site == MOD:
                    need.append((agent, name, list(_mod_states(stmt))))
                elif site == ACT:
                    need.append((agent, name, list(_ACTIVITY)))
                elif isinstance(value, str):
                    need.append((agent, name, [value]))
                else:
                    need.append((agent, name, None))
            if not mol.sites:
                need.append((agent, None, None))
        for (molecule, name), taken in added.items():
            sites, site_states = given.get(_n(molecule), ((), {}))
            held = _sites_text(molecule, sites, site_states)
            others = [*sites, *(n for m, n in added if m == molecule and n != name)]
            close = [s for s in others if _near(name, s)]
            if close:
                raise MechanismError(
                    f"mechanism {policy!r} adds site {name!r} to {molecule}: did you mean "
                    f"{' or '.join(map(repr, close))}? {held}; a site the scheme adds needs a "
                    "name no other site is a slip of"
                )
            if len(taken) < 2:
                raise MechanismError(
                    f"mechanism {policy!r} adds site {name!r} to {molecule} with the one state "
                    f"{taken[0]!r}, so every {molecule} would start in it; write every state it "
                    f"takes, the one it starts in first, or name a site it has: {held}"
                )
        return need

    policy_fn.indra_policy = "default"
    policy_fn.regulators = frozenset(read)
    policy_fn.scheme_sites = scheme_sites
    policy_fn.scheme = definition["scheme"]
    return policy_fn


def _sites_text(molecule: str, sites, site_states) -> str:
    """``molecule``'s sites, each with its states or marked a binding site."""
    listed = ", ".join(
        f"{s} {site_states[s]}" if site_states.get(s) else f"{s} (binding)" for s in sites
    )
    return f"{molecule}'s sites are {listed}" if listed else f"{molecule} has no sites"


def _own_sites(role: str, site: str, value, sites, site_states) -> list[str]:
    """``role``'s own sites of the kind ``site`` names (an activity for ``act``, a modification for
    ``mod``), each written in the state ``value`` asks for: what to write where the molecule lacks
    the site ``site`` names."""
    if site == ACT:
        return [f"`{role}({s}={value!r})`" for s in sites if site_states.get(s) == list(_ACTIVITY)]
    on = _OFF_ON.index(value) if value in _OFF_ON else 1
    return [
        f"`{role}({s}={site_states[s][on]!r})`" for s in sites
        if site_states.get(s) in states.values()
    ]


def _near(a: str, b: str) -> bool:
    """Whether two site names read as one name mistyped: equal but for case and separators, or one
    edit apart (a letter changed, added, dropped or two swapped) when both are four letters or
    more. Names that differ only in their digits are numbered siblings, not slips: residues
    (``T308``, ``T309``), domains (``dom1``, ``dom2``)."""
    a, b = (re.sub(r"[\W_]", "", s.lower()) for s in (a, b))
    if a == b:
        return True
    if min(len(a), len(b)) < 4 or re.sub(r"\d", "", a) == re.sub(r"\d", "", b):
        return False
    if len(a) == len(b):
        diff = [i for i, (x, y) in enumerate(zip(a, b)) if x != y]
        return len(diff) == 1 or (
            len(diff) == 2 and diff[1] == diff[0] + 1 and a[diff[0]] == b[diff[1]]
            and a[diff[1]] == b[diff[0]]
        )
    short, long_ = sorted((a, b), key=len)
    return len(long_) - len(short) == 1 and any(
        long_[:i] + long_[i + 1:] == short for i in range(len(long_))
    )


def _complex_nodes(text: str):
    node = ast.parse(text, mode="eval").body
    out = []

    def walk(n):
        if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Mod):
            walk(n.left), walk(n.right)
        else:
            out.append(n)

    walk(node)
    return out


def _pattern_side(stmt, model, side, agents, fresh):
    """One side's ComplexPatterns. ``fresh`` (right-hand side only) counts the molecules of each
    monomer the left carries; one beyond them is new, and written in full."""
    left = dict(fresh or {})
    out = []
    for complex_ in side:
        mps = []
        for mol in complex_:
            agent = _molecule_agent(stmt, mol.name, agents)
            monomer = model.monomers[_n(agent.name)]
            stated = {
                s: v[0] if isinstance(v, tuple) else v
                for s, v in get_site_pattern(agent).items()
                if v is not ANY
            }
            given = {
                _site_name(stmt, mol.name, s, agents): _site_value(stmt, s, v) for s, v in mol.sites
            }
            state = {**stated, **given}
            if fresh is not None:
                left[monomer.name] = left.get(monomer.name, 0) - 1
                if left[monomer.name] < 0:
                    for site in monomer.sites:
                        if site not in state:
                            state[site] = (
                                monomer.site_states[site][0] if site in monomer.site_states else None
                            )
            unknown = sorted(set(state) - set(monomer.sites))
            if unknown:
                raise MechanismError(f"{agent.name} has no site(s) {unknown}")
            mps.append(monomer(**state))
        out.append(pysb.ComplexPattern(mps, None))
    return out


def _notation_agent(text: str) -> Agent:
    from . import notation

    try:
        return Agent._from_json(notation.parse_agent(text))
    except notation.NotationError as exc:
        raise MechanismError(f"amount({text}): {exc}") from None


def _sympy(rate: _Rate) -> sympy.Expr:
    """``rate`` as one expression over its parameters, its regulators by name and every other
    amount a name of its own: the names a rate reads."""
    text = _REGULATOR_AMOUNT.sub(r"\1", rate.text)
    for k, pattern in enumerate(rate.amounts):
        text = text.replace(f"amount({pattern})", f"__amount{k}", 1)
    names = set(re.findall(r"[A-Za-z_]\w*", text)) - set(_FUNCTIONS)
    return sympy.sympify(
        text.replace("^", "**"), locals={**{n: sympy.Symbol(n) for n in names}, **_FUNCTIONS}
    )


def _rate_value(stmt, model, rate: _Rate, params, agents, name, regulators, amounts):
    """``rate`` as the value of a rule, ``name`` if an expression, each amount it reads besides
    its regulators an observable, recorded in ``amounts`` (its name -> the amount as written)."""
    from indra.assemblers.pysb.assembler import get_monomer_pattern

    from .mechanisms import _add

    if (not rate.amounts and not rate.agents and not rate.regulators
            and re.fullmatch(r"[A-Za-z_]\w*", rate.text)):
        return params[rate.text]
    observables = {}
    text = _REGULATOR_AMOUNT.sub(r"\1", rate.text)
    for regulator in sorted(set(rate.regulators)):
        obs = pysb.Observable(
            f"{name}_{regulator}", get_monomer_pattern(model, regulators[regulator]), _export=False
        )
        _add(model, obs)
        observables[regulator] = obs
    for k, written in enumerate(rate.agents):
        obs = pysb.Observable(
            f"{name}_agent{k}",
            get_monomer_pattern(model, _notation_agent(written)),
            _export=False,
        )
        _add(model, obs)
        observables[f"__agent{k}"] = obs
        amounts[obs.name] = written
    for k, pattern in enumerate(rate.amounts):
        (cp,) = _pattern_side(stmt, model, (tuple(_molecule(n) for n in _complex_nodes(pattern)),),
                              agents, fresh=None)
        obs = pysb.Observable(f"{name}_amount{k}", cp, _export=False)
        _add(model, obs)
        observables[f"__amount{k}"] = obs
        amounts[obs.name] = pattern
        text = text.replace(f"amount({pattern})", f"__amount{k}", 1)
    expr = sympy.sympify(
        text.replace("^", "**"),
        locals={**{p: params[p] for p in rate.names}, **observables, **_FUNCTIONS, "time": time},
    )
    expression = pysb.Expression(name, expr, _export=False)
    _add(model, expression)
    return expression


# ------------------------------------------------------------------------------- describing
#: Statement types no row can use: bases of the concrete types, and Influence, over concepts.
ABSTRACT = frozenset(
    {"AddModification", "RemoveModification", "Modification", "SelfModification",
     "RegulateActivity", "Association", "Influence"}
)
#: What a placeholder of these types takes besides its agents.
_EXTRA = {
    "Translocation": {"from_location": "cytoplasm", "to_location": "nucleus"},
    "ActiveForm": {"activity": "activity", "is_active": True},
}


def placeholder(stmt_type: str, enzyme: bool = True) -> Statement:
    """A statement of ``stmt_type`` whose agents are named by role: what :func:`describe` builds.
    Without ``enzyme``, a modification names none."""
    if stmt_type in ABSTRACT:
        raise MechanismError(f"{stmt_type} is no statement a row can use")
    cls = get_statement_by_name(stmt_type)
    args = dict(_EXTRA.get(stmt_type, {}))
    for role in cls._agent_order:
        if role == "enz" and not enzyme:
            args[role] = None
        elif role == "members":
            args[role] = [Agent("members_0"), Agent("members_1")]
        elif role in ("obj_from", "obj_to"):
            args[role] = [Agent(role)]
        elif role == "enz" and stmt_type == "Transphosphorylation":
            # The substrate is the enzyme's bound partner, a molecule of the same kind.
            args[role] = Agent(role, bound_conditions=[BoundCondition(Agent(role))])
        elif role == "agent" and stmt_type == "ActiveForm":
            args[role] = Agent(role, activity=None)
        else:
            args[role] = Agent(role)
    try:
        return cls(**args)
    except TypeError as exc:
        raise MechanismError(f"{stmt_type} takes more than its agents ({exc}); build it to see") from None


def rate_parameters(scheme: str) -> list[str]:
    """The parameters ``scheme``'s rates name: what its definition declares."""
    return sorted({n for line in parse(scheme) for r in line.rates for n in _rate(r).names})


def describe(stmt_type: str, policy: str, library) -> str:
    """The entry ``(stmt_type, policy)`` of ``library`` as a scheme: built for
    :func:`placeholder`'s statement and read back."""
    stmt, model, emission = _on_placeholder(stmt_type, policy, library)
    agents = role_agents(stmt)
    lines = [_describe_rule(stmt, model.rules[name], agents) for name in emission.rules]
    if not lines:
        raise MechanismError(f"{(stmt_type, policy)} builds no rule for a placeholder statement")
    return "\n".join(lines)


def _on_placeholder(stmt_type: str, policy: str, library):
    """The entry ``(stmt_type, policy)`` of ``library`` built for :func:`placeholder`'s statement:
    the statement, the model and the emission. An entry that reads regulators is given
    placeholder ones, named as its rates read them."""
    from .mechanisms import assemble

    names = sorted(getattr(library.get((stmt_type, policy)), "regulators", ()))
    stated: dict[str, list[Agent]] = {}
    for name in names:
        kind, _, position = name.rpartition("_") if name[-1].isdigit() else (name, "", "")
        stated[kind] = [Agent(f"{kind}_{i}") for i in range(max(len(stated.get(kind, ())),
                                                              int(position or 0) + 1))]
    # One of a kind, read by its kind's own name, is named so.
    stated = {k: [Agent(k)] if len(v) == 1 and k in names else v for k, v in stated.items()}
    stmt = placeholder(stmt_type)
    try:
        model, (emission,) = assemble([(stmt, policy, "r", None, stated)], {}, library)
    except MechanismError as exc:
        if getattr(stmt, "enz", None) is None:
            raise
        # An entry with no place for an enzyme refuses one: `basal`, and what is written on it.
        # Any other refusal is the one to report.
        stmt = placeholder(stmt_type, enzyme=False)
        try:
            model, (emission,) = assemble([(stmt, policy, "r", None, stated)], {}, library)
        except MechanismError:
            raise exc from None
    return stmt, model, emission


def _describe_rule(stmt, rule, agents) -> str:
    expr = rule.rule_expression

    def side(rp: ReactionPattern) -> str:
        cps = [cp for cp in rp.complex_patterns if cp is not None]
        return " + ".join(" % ".join(_describe_mp(stmt, mp, agents) for mp in cp.monomer_patterns)
                          for cp in cps) or "None"

    rates = [rule.rate_forward] + ([rule.rate_reverse] if rule.is_reversible else [])
    text = f"{side(expr.reactant_pattern)} {'|' if rule.is_reversible else '>>'} {side(expr.product_pattern)}"
    text += " @ " + ", ".join(_describe_rate(stmt, r, agents) for r in rates)
    return text + (" delete_molecules" if rule.delete_molecules else "")


def _site_back(stmt, role: str, site: str, agents) -> str:
    """A site of ``role``'s monomer as the scheme names it (:func:`_site_name` inverted)."""
    if isinstance(stmt, Modification) and site == get_mod_site_name(stmt._get_mod_condition()):
        return MOD
    if role in agents and site == _activity_site(stmt, role):
        return ACT
    for name, agent in agents.items():
        if site == get_binding_site_name(agent):
            return name
    return site


def _value_back(stmt, site_token: str, value) -> str:
    if isinstance(value, tuple):
        return f"({_value_back(stmt, site_token, value[0])}, {_value_back(stmt, site_token, value[1])})"
    if value is None:
        return "None"
    if value is ANY:
        return "ANY"
    if value is WILD:
        return "WILD"
    if site_token == MOD and isinstance(value, str):
        return repr(_OFF_ON[_mod_states(stmt).index(value)])
    return repr(value)


def _describe_mp(stmt, mp, agents) -> str:
    name = next((r for r, a in agents.items() if _n(a.name) == mp.monomer.name), mp.monomer.name)
    sites = []
    for site, value in mp.site_conditions.items():
        token = _site_back(stmt, name, site, agents)
        sites.append(f"{token}={_value_back(stmt, token, value)}")
    return f"{name}({', '.join(sites)})"


def _describe_rate(stmt, rate, agents) -> str:
    if isinstance(rate, pysb.Parameter):
        return rate.name.split("_", 1)[1] if "_" in rate.name else rate.name
    subs = {}
    for symbol in rate.expr.free_symbols:
        if isinstance(symbol, pysb.Parameter):
            subs[symbol] = sympy.Symbol(symbol.name.split("_", 1)[1])
        elif isinstance(symbol, pysb.Observable):
            (cp,) = symbol.reaction_pattern.complex_patterns
            pattern = " % ".join(_describe_mp(stmt, mp, agents) for mp in cp.monomer_patterns)
            subs[symbol] = sympy.Symbol(f"amount({pattern})")
        elif isinstance(symbol, pysb.Expression):
            subs[symbol] = sympy.Symbol(_describe_rate(stmt, symbol, agents))
    return str(rate.expr.xreplace(subs))
