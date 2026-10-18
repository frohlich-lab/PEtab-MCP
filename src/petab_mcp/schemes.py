"""Mechanisms as PySB rules over a statement's roles: read off a library entry, or written new.

A *scheme* is one PySB rule per line in PySB's own shorthand, with the molecules named by the
statement type's ROLES rather than by agents, so one scheme applies to every statement of that
type::

    enz(sub=None) + sub(enz=None, mod='off') >> enz(sub=1) % sub(enz=1, mod='off') @ kf
    enz(sub=1) % sub(enz=1, mod='off') >> enz(sub=None) + sub(enz=None, mod='on') @ kc
    enz(sub=1) % sub(enz=1) >> enz(sub=None) + sub(enz=None) @ kr

- A molecule is a role (``enz``, ``sub``, ``subj``, ``obj``; a list role by position,
  ``members_0``, or by its own name when the statement gives it one agent), each the agent the
  row writes there, in the states it writes; or any other name, a molecule of that name the
  statement does not mention (INDRA's ``atp_dependent`` binds an ``ATP``).
- A site is ``mod``, the statement's own modification (``'off'``, ``'on'``); ``act``, the role's
  activity (``'inactive'``, ``'active'``): the statement's ``obj_activity`` on its object, the
  activity the modification implies on an enzyme (``kinase`` for a phosphorylation); a role's
  name, the binding site for that partner; or any other name, taken as written. Written with a
  state, a site is one its molecule has, in a state it takes (``mod`` and ``act`` always are), or
  one the scheme adds with every state it takes, the one it starts in first, under a name that is
  no slip of one the molecule has.
- A site's value is a state, a bond number, ``None`` (free), ``ANY`` (bound to something) or
  ``WILD``; a ``(state, bond)`` pair is both. ``>>`` is one-way, ``|`` both ways, ``%`` joins a
  complex, ``None`` is nothing.
- ``@`` gives the rate: one for ``>>``, two for ``|``. A rate is a parameter's name, minted per
  row as ``<prefix>_<name>`` and shared or fixed like any (``kf=k_on`` on the row), or an
  expression over parameters, ``amount(<pattern>)`` (the amount of what the pattern matches),
  ``time`` and exp/log/sqrt; mass action multiplies it by every reactant's amount, as PySB does.
  ``delete_molecules`` after the rates removes only the molecules that vanish, not their
  complex.

:func:`describe` writes any library entry this way, from a statement of placeholder agents named
by role; :func:`compile_scheme` turns a scheme into a library entry. The two invert each other:
an entry's description, compiled, builds the same rules (``tests/test_schemes.py``).
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

MOD, ACT = "mod", "act"
_OFF_ON = ("off", "on")
_ACTIVITY = ("inactive", "active")
_FUNCTIONS = {"exp": sympy.exp, "log": sympy.log, "sqrt": sympy.sqrt}
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
    return Line(_side(tree.left), _side(tree.right), reversible, tuple(rate_list), delete)


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


def _mod_states(stmt: Statement) -> tuple[str, str] | None:
    return tuple(states[stmt._get_mod_condition().mod_type]) if isinstance(stmt, Modification) else None


def _activity_site(stmt: Statement, role: str) -> str | None:
    """The site ``act`` names on ``role``: the statement's object activity, an enzyme's implied
    activity, or the activity a subject is written with (INDRA's default ``activity``)."""
    if isinstance(stmt, RegulateActivity) and role == "obj":
        return stmt.obj_activity
    if isinstance(stmt, Modification) and role == "enz":
        return mod_acttype_map.get(type(stmt), "activity")
    agent = role_agents(stmt).get(role)
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


def _rate(text: str) -> _Rate:
    """A rate's parameters (its names outside ``amount()``) and its ``amount()`` patterns."""
    try:
        tree = ast.parse(text, mode="eval").body
    except SyntaxError:
        raise MechanismError(f"rate {text!r} does not read") from None
    names: list[str] = []
    amounts: list[str] = []

    def visit(node) -> None:
        if isinstance(node, ast.Call):
            fn = getattr(node.func, "id", None)
            if fn == "amount":
                if len(node.args) != 1 or node.keywords:
                    raise MechanismError(f"rate {text!r}: amount() takes one pattern")
                amounts.append(ast.unparse(node.args[0]))
                return  # its names are molecules and sites
            if fn not in _FUNCTIONS:
                raise MechanismError(f"rate {text!r}: no function {fn!r}; amount, exp, log, sqrt")
            for arg in node.args:
                visit(arg)
        elif isinstance(node, ast.Name):
            if node.id != "time":
                names.append(node.id)
        elif isinstance(node, ast.BinOp):
            visit(node.left), visit(node.right)
        elif isinstance(node, ast.UnaryOp):
            visit(node.operand)
        elif not (isinstance(node, ast.Constant) and isinstance(node.value, (int, float))):
            raise MechanismError(f"rate {text!r}: {ast.unparse(node)!r} is not allowed")

    visit(tree)
    return _Rate(tuple(sorted(set(names))), tuple(amounts), text)


def compile_scheme(definition: dict):
    """A library entry from ``{statement_type, policy, scheme}``: the scheme's rules, built for
    each row from its own agents (see the module docstring)."""
    stmt_type, policy = definition["statement_type"], definition["policy"]
    lines = parse(definition["scheme"])
    known_roles = roles(stmt_type)
    rates = [[_rate(r) for r in line.rates] for line in lines]

    def policy_fn(stmt: Statement, model: pysb.Model, prefix: str):
        from .mechanisms import Emission, _add

        agents = role_agents(stmt)
        params: dict[str, pysb.Parameter] = {}
        rules = []
        for i, (line, line_rates) in enumerate(zip(lines, rates)):
            lhs = _pattern_side(stmt, model, line.lhs, agents, known_roles, fresh=None)
            carried = {}
            for cp in lhs:
                for mp in cp.monomer_patterns:
                    carried[mp.monomer.name] = carried.get(mp.monomer.name, 0) + 1
            rhs = _pattern_side(stmt, model, line.rhs, agents, known_roles, fresh=carried)
            values = []
            for j, rate in enumerate(line_rates):
                for name in rate.names:
                    if name not in params:
                        params[name] = pysb.Parameter(f"{prefix}_{name}", 1.0, _export=False)
                        _add(model, params[name])
                values.append(_rate_value(stmt, model, rate, params, agents, known_roles,
                                          f"{prefix}_rate{i}_{j}"))
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
        )

    def scheme_sites(stmt: Statement, given: dict) -> list[tuple[Agent, str, list | None]]:
        """The sites the scheme's molecules need on each monomer, for the monomer pass. A state
        it writes is checked against ``given``, each monomer's sites and their states before any
        scheme adds its own (``{name: (sites, site_states)}``; see the module docstring)."""
        agents = role_agents(stmt)
        need = []
        added: dict[tuple[str, str], list] = {}
        patterns = [m for line in lines for side in (line.lhs, line.rhs) for c in side for m in c]
        patterns += [_molecule(n) for rs in rates for r in rs for a in r.amounts
                     for n in _complex_nodes(a)]
        for mol in patterns:
            agent = agents.get(mol.name) or Agent(mol.name)
            sites, site_states = given.get(_n(agent.name), ((), {}))
            for site, value in mol.sites:
                name = _site_name(stmt, mol.name, site, agents)
                value = value[0] if isinstance(value, tuple) else value
                if isinstance(value, str):
                    state = _site_value(stmt, site, value)
                    written = f"mechanism {policy!r}: `{mol.name}({site}={value!r})`"
                    if name in sites:
                        if state not in site_states.get(name, ()):
                            raise MechanismError(
                                f"{written} names a state {agent.name}'s site {name!r} does not "
                                f"take; {_sites_text(agent.name, sites, site_states)}"
                            )
                    elif site in (MOD, ACT):
                        raise MechanismError(
                            f"{written} is site {name!r}, which {agent.name} does not have; "
                            f"{_sites_text(agent.name, sites, site_states)}"
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
    policy_fn.modifiers = ()
    policy_fn.scheme_sites = scheme_sites
    policy_fn.scheme = definition["scheme"]
    return policy_fn


def _sites_text(molecule: str, sites, site_states) -> str:
    """``molecule``'s sites, each with its states or marked a binding site."""
    listed = ", ".join(
        f"{s} {site_states[s]}" if site_states.get(s) else f"{s} (binding)" for s in sites
    )
    return f"{molecule}'s sites are {listed}" if listed else f"{molecule} has no sites"


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


def _pattern_side(stmt, model, side, agents, known_roles, fresh):
    """One side's ComplexPatterns. ``fresh`` (right-hand side only) counts the molecules of each
    monomer the left carries; one beyond them is new, and written in full."""
    left = dict(fresh or {})
    out = []
    for complex_ in side:
        mps = []
        for mol in complex_:
            agent = agents.get(mol.name)
            if agent is None and mol.name in known_roles:
                raise MechanismError(f"the statement gives no {mol.name}")
            agent = agent or Agent(mol.name)
            monomer = model.monomers[_n(agent.name)]
            stated = {
                s: v[0] if isinstance(v, tuple) else v
                for s, v in get_site_pattern(agent).items()
                if v is not ANY
            } if mol.name in agents else {}
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


def _rate_value(stmt, model, rate: _Rate, params, agents, known_roles, name):
    from .mechanisms import _add

    if not rate.amounts and re.fullmatch(r"[A-Za-z_]\w*", rate.text):
        return params[rate.text]
    observables = {}
    text = rate.text
    for k, pattern in enumerate(rate.amounts):
        (cp,) = _pattern_side(stmt, model, (tuple(_molecule(n) for n in _complex_nodes(pattern)),),
                              agents, known_roles, fresh=None)
        obs = pysb.Observable(f"{name}_amount{k}", cp, _export=False)
        _add(model, obs)
        observables[f"__amount{k}"] = obs
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


def placeholder(stmt_type: str) -> Statement:
    """A statement of ``stmt_type`` whose agents are named by role: what :func:`describe` builds."""
    if stmt_type in ABSTRACT:
        raise MechanismError(f"{stmt_type} is no statement a row can use")
    cls = get_statement_by_name(stmt_type)
    args = dict(_EXTRA.get(stmt_type, {}))
    for role in cls._agent_order:
        if role == "members":
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


def describe(stmt_type: str, policy: str, library) -> str:
    """The entry ``(stmt_type, policy)`` of ``library`` as a scheme: built for
    :func:`placeholder`'s statement and read back."""
    from .mechanisms import assemble

    stmt = placeholder(stmt_type)
    model, (emission,) = assemble([(stmt, policy, "r")], {}, library)
    agents = role_agents(stmt)
    lines = [_describe_rule(stmt, model.rules[name], agents) for name in emission.rules]
    if not lines:
        raise MechanismError(f"{(stmt_type, policy)} builds no rule for a placeholder statement")
    return "\n".join(lines)


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
