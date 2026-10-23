"""The mechanism library and the assembler that applies it.

A mechanism is a policy: it receives one statement and the model under construction, and emits
the PySB rules and parameters that statement implies. That is the shape of
INDRA's own PySB assembler policies, so this module extends a dispatch that already exists —
INDRA supplies monomer and site construction, we supply the transformations.

Entries are keyed ``(statement class name, policy)``, and the two halves are deliberately
separate. ``Phosphorylation`` with INDRA's mass action and with an agent-written
Michaelis-Menten are two entries making the same biological claim through different mathematics;
collapsing them would leave the library unable to express "the same mechanism, simplified",
which is an axis model revision often moves along.

Why a policy RETURNS what it emitted rather than leaving it to be found: the PEtab parameter
table must list every estimated parameter, and that list is currently recovered by scanning the
model for symbols with no row yet (:mod:`reconcile`). A generated model needs no scanning — the
generator knows what it generated. :class:`Emission` is that record, and it doubles as the
per-instantiation binding that makes a cross-instance ``revise`` addressable at all.

Policies must emit CONCRETE rules. An unbound site pattern is where rule-based models blow up
under ``generate_equations``, and larger models are where that would surface.

Assembly is two passes because a PySB monomer is declared once with ALL its sites, before any
rule names one: :func:`assemble` first runs INDRA's own monomer pass over every row, then the
policies. A policy therefore never creates a monomer, only patterns over existing ones.

The library is not closed. When no entry expresses what the agent needs it WRITES one
(:func:`written`), and that entry joins the run's library for every later call. A missing entry
is therefore a prompt to write one, not a dead end, and :class:`MechanismError` still records it
so the entries agents had to invent can be counted and promoted into this module.
"""

from __future__ import annotations

import keyword
import math
import re
from collections.abc import Callable
from collections import Counter
from copy import copy
from dataclasses import dataclass, field, replace
from itertools import count

import pysb
import sympy
from indra.assemblers.pysb import assembler as _indra_assembler
from indra.assemblers.pysb.assembler import Policy as IndraPolicy
from indra.assemblers.pysb.assembler import (
    PysbAssembler,
    UnknownPolicyError,
    get_annotation,
    get_monomer_pattern,
    statement_whitelist,
)
from indra.assemblers.pysb.base_agents import BaseAgentSet
from indra.assemblers.pysb.sites import get_mod_site_name, states
from indra.statements import (
    AddModification,
    Agent,
    Autophosphorylation,
    Complex,
    Conversion,
    DecreaseAmount,
    Gap,
    Gef,
    IncreaseAmount,
    ModCondition,
    Modification,
    NotAStatementName,
    RegulateActivity,
    RemoveModification,
    Statement,
    Translocation,
    Transphosphorylation,
    get_all_descendants,
    get_statement_by_name,
    modclass_to_modtype,
)
from indra.assemblers.pysb.assembler import _n, get_site_pattern
from indra.assemblers.pysb.sites import get_binding_site_name
from pysb import ANY
from pysb.core import ReactionPattern, RuleExpression, time

from .errors import MechanismError
from .statements import REGULATOR_NAME, PopulationTransition, named_agents


@dataclass(frozen=True)
class Emission:
    """What one instantiation added. The binding record, and the unit ``revise`` addresses."""

    entry: tuple[str, str]
    prefix: str
    roles: dict[str, str]
    rules: tuple[str, ...]
    parameters: tuple[str, ...]
    #: The row's binding (:func:`bind`): kind -> the shared id or the number that replaced
    #: ``<prefix>_<kind>``. Empty for a row that binds nothing.
    bound: dict[str, str | float] = field(default_factory=dict)
    #: The sites the row's scheme (:mod:`schemes`) gave molecules that had none such, and no
    #: earlier row's scheme had: molecule -> site -> its states (none for a binding site).
    sites_added: dict[str, dict[str, list[str]]] = field(default_factory=dict)


Policy = Callable[[Statement, pysb.Model, str], Emission]

#: Registered entries, keyed ``(statement class name, policy)``. The class NAME, not the class,
#: so a provenance record restores without importing the class that wrote it.
REGISTRY: dict[tuple[str, str], Policy] = {}


def mechanism(stmt_cls: type[Statement], policy: str) -> Callable[[Policy], Policy]:
    """Register one entry. Re-registering a key raises; a silent override is how two policies
    that disagree end up producing whichever model imported last."""
    key = (stmt_cls.__name__, policy)

    def register(fn: Policy) -> Policy:
        if key in REGISTRY:
            raise RuntimeError(f"mechanism {key} is already registered")
        REGISTRY[key] = fn
        return fn

    return register


def _misspelt(
    key: tuple[str, str], own: list[str], library: dict[tuple[str, str], Policy]
) -> str:
    """What the refusal of ``key`` says when its policy is named nowhere — no entry of any type
    (INDRA's own policies are entries) — with the names closest to it, the type's ``own`` first;
    empty when the policy exists, and the pair is a library gap. Counted as gaps, a misspelling
    (``indra_defualt``) would read as a missing entry."""
    from .model_edits import closest_symbols

    stmt_type, policy = key
    named = {p for _, p in library}
    if policy in named:
        return ""
    close = closest_symbols(policy, own)
    other = "" if close else f", which {stmt_type} lacks"
    close = close or closest_symbols(policy, sorted(named - set(own)))
    guess = f" (did you mean {' or '.join(map(repr, close))}{other}?)" if close else ""
    return f": no policy is named {policy!r}{guess}"


def instantiate(
    stmt: Statement,
    model: pysb.Model,
    policy: str,
    prefix: str,
    library: dict[tuple[str, str], Policy] = REGISTRY,
    regulators: dict[str, list[Agent]] | None = None,
) -> Emission:
    """Apply one entry of ``library`` (the registered entries plus any the agent wrote).
    ``regulators`` are the agents the statement states regulate its rate, by kind
    (:data:`statements.REGULATORS`), which the entry's rates read (:func:`_read_regulators`).
    Raises :class:`MechanismError` when the pair is not in it — the agent's signal to write one."""
    key = (type(stmt).__name__, policy)
    if key not in library:
        known = sorted(p for c, p in library if c == key[0])
        typo = _misspelt(key, known, library)
        raise MechanismError(
            f"no mechanism {key}{typo}; the library has {known or 'none'} for {key[0]}. If none fits, "
            "write one with `mechanisms`.",
            cause="malformed_mechanism" if typo else "library_gap",
        )
    entry = library[key]
    read = _read_regulators(prefix, key, getattr(entry, "regulators", frozenset()), regulators or {})
    return entry(stmt, model, prefix, regulators=read) if read else entry(stmt, model, prefix)


def _read_regulators(
    prefix: str, key: tuple[str, str], names: frozenset[str], stated: dict[str, list[Agent]]
) -> dict[str, Agent]:
    """The agent each regulator name the rates of ``key`` read stands for, from those row
    ``prefix``'s statement states. Refused both ways: a name the statement states no agent for, and
    an agent the statement states that no rate reads, a claim the maths would not carry."""
    agents: dict[str, Agent] = {}
    for kind, given in stated.items():
        agents |= {f"{kind}_{i}": agent for i, agent in enumerate(given)}
        if len(given) == 1:
            agents[kind] = given[0]
    if missing := sorted(names - set(agents)):
        several = [n for n in missing if n in stated]
        raise MechanismError(
            f"row {prefix!r}: the rate of {key} reads {missing}, which its statement does not state. "
            + (
                f"It states several {several}: read each by position (`{several[0]}_0`)."
                if several else
                "State each regulator in the statement: `<statement>, inhibited by X` (activated "
                "by, modulated by), several of one kind read by position (`inhibited_by_1`)."
            )
        )
    unread = [
        f"{agent.name} ({kind})"
        for kind, given in stated.items()
        for i, agent in enumerate(given)
        if f"{kind}_{i}" not in names and not (len(given) == 1 and kind in names)
    ]
    if unread:
        raise MechanismError(
            f"row {prefix!r}: its statement states {', '.join(unread)}, which the rate of {key} "
            "does not read: the maths would not carry the claim. A regulator is read in its "
            "mechanism's rate, as `inhibited_by` (`inhibited_by_0`... by position; "
            "`amount(inhibited_by)` in a scheme): write a mechanism that reads it, or leave it out "
            "of the statement"
        )
    return {name: agents[name] for name in names}


def _sign(e: sympy.Expr) -> int | None:
    """``1`` when ``e`` is provably never negative, ``-1`` never positive, from the assumptions on
    its names and how products, powers and sums of signed terms combine; ``None`` otherwise."""
    if e.is_nonnegative:
        return 1
    if e.is_nonpositive:
        return -1
    if isinstance(e, sympy.Mul):
        signs = [_sign(a) for a in e.args]
        return None if None in signs else math.prod(signs)
    if isinstance(e, sympy.Pow):
        base, exponent = e.args
        if exponent.is_even:
            return 1
        sign = _sign(base)
        return sign if sign == 1 or exponent.is_odd else None
    if isinstance(e, sympy.Add):
        signs = {_sign(t) for t in sympy.Add.make_args(sympy.expand(e))}
        return signs.pop() if len(signs) == 1 and None not in signs else None
    return None


def _direction(expr: sympy.Expr, symbol: sympy.Symbol) -> int | None:
    """``-1`` when ``expr`` provably never rises with ``symbol``, ``1`` when it never falls, ``0``
    when it does not change with it at all; ``None`` when neither can be shown, which includes a
    ``piecewise`` that switches on ``symbol``, whose jump its derivative does not see."""
    switches = (c for pw in expr.atoms(sympy.Piecewise) for _, c in pw.args)
    if any(symbol in getattr(c, "free_symbols", ()) for c in switches):
        return None
    d = sympy.diff(expr, symbol)
    if sympy.simplify(d) == 0:
        return 0
    numerator, denominator = sympy.fraction(sympy.together(d))
    for n, den in ((numerator, denominator), (sympy.factor(numerator), sympy.factor(denominator))):
        if (sn := _sign(n)) is not None and (sd := _sign(den)) is not None:
            return sn * sd
    return None


def check_regulators(policy: str, rate: str, expr: sympy.Expr) -> None:
    """Refuse ``rate`` of ``policy`` (``expr``, its names signed by :func:`assumed`) when it does
    not fall with each inhibitor it reads and rise with each activator, for every value of every
    name: the direction a statement stating the regulator claims. ``modulated_by`` claims none."""
    for symbol in sorted(expr.free_symbols, key=str):
        if not REGULATOR_NAME.fullmatch(str(symbol)) or str(symbol).startswith("modulated"):
            continue
        kind = str(symbol).split("_by")[0]
        inhibitor = kind == "inhibited"
        direction = _direction(expr, symbol)
        if direction == (-1 if inhibitor else 1):
            continue
        claim, way = ("inhibits", "fall") if inhibitor else ("activates", "rise")
        other = str(symbol).replace(kind, "activated" if inhibitor else "inhibited", 1)
        unsigned = str(symbol).replace(kind, "modulated", 1)
        if direction == 0:
            found = "it does not change with it, a claim the rate does not carry"
        elif direction is not None:
            found = f"it {'rises' if inhibitor else 'falls'} instead: state it `{other}`"
        else:
            found = (
                "that cannot be shown for every value of the rate's names (nor across a piecewise "
                f"switching on it): state it `{unsigned}`, which claims no direction"
            )
        raise MechanismError(
            f"rate {rate!r} of {policy!r} reads `{symbol}`, which a statement states {claim} its "
            f"reaction, so the rate must {way} as `{symbol}` rises; {found}. "
            f"d/d{symbol} = {sympy.simplify(sympy.diff(expr, symbol))}"
        )


def bind(
    model: pysb.Model,
    emission: Emission,
    binding: dict[str, str | float],
    merged: dict[str, object] | None = None,
    species=frozenset(),
) -> Emission:
    """Replace parameters a row minted (``<prefix>_<kind>``) by a SHARED one or a NUMBER.

    Every row mints its own rate constants, so one rate a model uses in many reactions became
    many estimated parameters: one constant shared by many steps of the same kind, or a fixed rate
    the problem ships. A string binds the kind to that id: a parameter an earlier row bound, or a
    new one of that name — which reconcile then takes from the problem's own row when the problem
    ships it. A number pins the kind to that value; it is no longer estimated. Any other string is
    an EXPRESSION over such ids (an off-rate written ``KD*kf_on``), each name resolved as a shared
    id is, so the parameters are the problem's own and keep their rows. Bound for every entry
    alike, INDRA's and written ones, so no entry needs sharing of its own. What a row no longer
    reads stays behind unreferenced, and the SBML export drops it (as it drops a written
    mechanism's base constant).

    ``merged`` carries, across the rows of one model, each name an earlier binding renamed or
    replaced to what now stands for it, so a row that names another row's parameter shares what
    that row reads, whichever order they bind in. ``species`` are the ids the model names amounts
    by, the entity map's and ``amounts``', so no binding reads one (:func:`_not_a_parameter`).
    """
    merged = {} if merged is None else merged
    minted = {name.removeprefix(f"{emission.prefix}_"): name for name in emission.parameters}
    unknown = sorted(set(binding) - set(minted))
    if unknown:
        raise MechanismError(
            f"row {emission.prefix!r} binds {unknown}, but its mechanism makes only "
            f"{sorted(minted)}"
        )
    parameters = []
    for kind, name in minted.items():
        if kind not in binding:
            parameters.append(name)
            continue
        old, target = model.parameters[name], binding[kind]
        if isinstance(target, bool) or not isinstance(target, (str, int, float)):
            raise MechanismError(f"row {emission.prefix!r} binds {kind!r} to {target!r}")
        if isinstance(target, str) and not _IDENTIFIER.fullmatch(target):
            new, names = _bound_expression(
                model, name, target, emission.prefix, merged, species
            )
            # A binding that scales the row's own constant (`kf="r1_kf*(1 + D)"`)
            # keeps that constant, and a row naming it shares the constant, not the scaling.
            if old not in new.expr.free_symbols:
                merged[name] = new
            _redirect(model, old, new)
            parameters += [n for n in names if n not in parameters]
            continue
        if isinstance(target, str):
            _not_a_parameter(model, target, emission.prefix, name, species)
            existing = _held(model, target, merged)
            if existing is None:
                merged[name] = old
                old._export = False  # INDRA builds with PySB's export off; see `_indra`
                old.rename(target)
                # A new shared id starts where any new name does (an entity-map `initial` or
                # `formula` names one at reconcile's seed too), not at the 1.0 the row minted:
                # reconcile fixes a starting-amount parameter that is exactly 1 as a unit
                # normalisation, so a shared rate also read by an `initial` was fixed or
                # estimated depending on which row named it first.
                from .reconcile import SEED_INITIAL_VALUE

                old.value = SEED_INITIAL_VALUE
                parameters.append(target)
                continue
            new = existing
        else:
            new = pysb.Expression(f"{name}_pinned", sympy.Float(target), _export=False)
            _add(model, new)
        _redirect(model, old, new)
        merged[name] = new
        if isinstance(target, str):
            parameters.append(target)
    return replace(emission, parameters=tuple(parameters), bound=dict(binding))


_IDENTIFIER = re.compile(r"[A-Za-z_]\w*")


def _not_a_parameter(
    model: pysb.Model, ident: str, prefix: str, name: str, species=frozenset()
) -> None:
    """Refuse a binding that reads ``ident`` as a parameter when it names something else: a
    molecule or one of ``species``, the ids the model names amounts by (a parameter of its name
    clashed with it in the SBML), a starting amount ``<x>_init`` (the start made it again, a
    duplicate in PySB), a rate function, or ``time`` (which BioNetGen refused as a parameter
    name). A row's own ``<prefix>_init`` constant is shared like any other."""
    held = model.components.get(ident)
    instead = ""
    if isinstance(held, pysb.Monomer):
        what = "a molecule"
    elif ident in species:
        what = "a species"
        instead = (
            f" `{ident}` is the amount of the agent the entity map (or `amounts`) maps it to, "
            "read by that agent's own name and sites, not by this id."
        )
    elif isinstance(held, (pysb.Parameter, pysb.Expression)):
        return
    elif ident.endswith("_init"):
        what = "a starting amount"
        instead = " `<name>_init` is the amount a species starts at, which only `starts` declares."
    elif held is not None:
        what = f"the model's {type(held).__name__.lower()}"
    elif ident == "time":
        what = "time"
    elif ident in _RATE_FUNCTIONS:
        what = "a function"
    else:
        return
    raise MechanismError(
        f"row {prefix!r} binds {name!r} over {ident!r}, which names {what}, not a parameter. A "
        "binding reads parameters: a shared id, a problem parameter, a condition-table input."
        f"{instead} A rate that reads an amount or `time` is a written mechanism (its `rate`, or "
        "a scheme's `amount(<pattern>)`)"
    )


def _held(model: pysb.Model, name: str, merged: dict[str, object]):
    """The component a shared id names now: what an earlier binding renamed or replaced it by,
    followed on, else the parameter or expression of that name; ``None`` when nothing holds it."""
    component = None
    while name in merged and merged[name] is not component:
        component = merged[name]
        name = component.name
    if component is not None:
        return component
    return model.parameters.get(name) or model.expressions.get(name)


def _shared(model: pysb.Model, name: str, merged: dict[str, object]):
    """The component a shared id names, made at reconcile's seed when nothing holds it yet."""
    existing = _held(model, name, merged)
    if existing is None:
        from .reconcile import SEED_INITIAL_VALUE

        existing = pysb.Parameter(name, SEED_INITIAL_VALUE, _export=False)
        _add(model, existing)
    return existing


def _bound_expression(
    model: pysb.Model,
    name: str,
    text: str,
    prefix: str,
    merged: dict[str, object],
    species=frozenset(),
):
    """``text`` as an Expression over shared ids, named after the parameter it replaces. Read
    over placeholder symbols first: sympy raised on `kcat Km` and `2kcat`, and PySB on `kcat,Km`
    (a tuple), out of make_changes unexplained."""
    tokens = _RATE_TOKENS.findall(text)
    names = sorted({t for t in tokens if _IDENTIFIER.fullmatch(t)} - set(_RATE_FUNCTIONS))
    try:
        trial = sympy.sympify(
            text.replace("^", "**"),
            locals={**{n: sympy.Symbol(n) for n in names}, **_RATE_FUNCTIONS},
        )
    except (sympy.SympifyError, SyntaxError, TypeError, ValueError):
        trial = None
    if "".join(tokens) != text or not isinstance(trial, sympy.Expr):
        raise MechanismError(
            f"row {prefix!r} binds {name!r} to {text!r}, which is no expression: a binding is a "
            "shared id (`k_on`), a number, or arithmetic over ids and numbers with "
            "exp/log/sqrt/piecewise (`KD*kf_on`)"
        )
    for n in names:
        _not_a_parameter(model, n, prefix, name, species)
    symbols = {n: _shared(model, n, merged) for n in names}
    value = sympy.sympify(text.replace("^", "**"), locals={**symbols, **_RATE_FUNCTIONS})
    expression = pysb.Expression(f"{name}_bound", value, _export=False)
    _add(model, expression)
    return expression, names


def _redirect(model: pysb.Model, old, new) -> None:
    """Point every rule rate and expression that reads ``old`` at ``new``, except ``new`` itself:
    a binding may read the parameter it replaces (``kf="r1_kf*(1 + D)"`` scales the row's own
    constant), and redirecting that read into ``new`` made it read itself, which PySB met as a
    recursion with no bottom."""
    for rule in model.rules:
        if rule.rate_forward is old:
            rule.rate_forward = new
        if rule.rate_reverse is old:
            rule.rate_reverse = new
    for expression in model.expressions:
        if expression is not new and old in expression.expr.free_symbols:
            expression.expr = expression.expr.xreplace({old: new})


#: What a written rate may contain besides names: numbers, arithmetic, parentheses, commas, and
#: the comparisons a ``piecewise`` condition makes. A rate is parsed by sympy, which evaluates
#: text, so it is checked against this first and every name in it against the ones the mechanism
#: allows.
_RATE_TOKENS = re.compile(
    r"\s+|(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?|[A-Za-z_]\w*|[<>]=?|[-+*/^(),]"
)


def _piecewise(*args) -> sympy.Expr:
    """Antimony's and SBML's ``piecewise(value, condition, ..., otherwise)``: the first value whose
    condition holds, else the last argument (0 without one). A switch the rate itself makes — a
    transmission rate that steps on an estimated day, a term on only while a dose is applied — is
    the modeller's claim like any other rate, which is why it is written here and never held as an
    input (PEtab-MCP#30). Raises ``TypeError`` (refused by :func:`parse_rate` as a rate that does
    not read) for a condition that is no comparison."""
    pairs = [(args[i], args[i + 1]) for i in range(0, len(args) - 1, 2)]
    otherwise = args[-1] if len(args) % 2 else sympy.Integer(0)
    if not pairs or not all(isinstance(c, sympy.core.relational.Relational) for _, c in pairs):
        raise TypeError("piecewise(value, condition, ..., otherwise): conditions compare")
    return sympy.Piecewise(*pairs, (otherwise, True))


_RATE_FUNCTIONS = {"exp": sympy.exp, "log": sympy.log, "sqrt": sympy.sqrt, "piecewise": _piecewise}


#: What an id of the problem is (``problem_ids`` of :func:`written`). A rate reads none of them.
SHIPPED, CONDITION, SPECIES = (
    "a parameter the problem ships",
    "a condition input",
    "a species of the entity map",
)


def _how_ids_enter(names, problem_ids: dict[str, str], amounts: str) -> list[str]:
    """How each of ``names`` that is an id of the problem enters a rate instead. A starting
    amount (``<x>_init``) is not offered the binding: it is read as the amount itself."""
    bound = [
        n for n in names if problem_ids.get(n) in (SHIPPED, CONDITION) and not n.endswith("_init")
    ]
    species = [n for n in names if problem_ids.get(n) == SPECIES]
    return [
        f"{', '.join(f'`{n}` is {problem_ids[n]}' for n in bound)}. A problem parameter or "
        "condition input enters through a row's binding: declare a parameter of your own and "
        f'bind it on the row, `parameters={{"K": "{bound[0]}"}}`, or `kf="k*{bound[0]}"`.'
        if bound else "",
        f"{', '.join(f'`{n}`' for n in species)}: {SPECIES}, read as the amount of the agent it "
        f"maps to, written by that agent's own name and sites, not by this id: {amounts}."
        if species else "",
    ]


def refuse_rate_names(
    rate: str,
    policy: str,
    unknown: list[str],
    allowed: list[str],
    also: str,
    amounts: str,
    problem_ids: dict[str, str] | None = None,
    roles=(),
) -> None:
    """Refuse a rate naming ``unknown``, names other than the ``allowed`` (``also`` says what else
    it may contain), with the closest allowed ones; and for a starting amount, an id of the problem
    or one of the ``roles`` (a scheme reads them only as amounts), how ``amounts`` are read."""
    from .model_edits import closest_symbols

    problem_ids = problem_ids or {}
    starts = [n for n in unknown if n.endswith("_init")]
    new = [n for n in unknown if n not in (*starts, *roles, *problem_ids)]
    close = [
        f"{u} -> {' or '.join(map(repr, c))}"
        for u in new
        if (c := closest_symbols(u, allowed + list(_RATE_FUNCTIONS)))
    ]
    sentences = [
        f"rate {rate!r} of {policy!r} may name only {allowed}{also}; it names {unknown}.",
        f"Did you mean: {'; '.join(close)}?" if close else "",
        f"{starts}: `<name>_init` is the amount a species starts at, and a rate does not read it; "
        f"read the amount itself, {amounts}." if starts else "",
        *(f"`{r}` is a role: read its amount, `amount({r})`." for r in unknown if r in roles),
        *_how_ids_enter(unknown, problem_ids, amounts),
        "Declare each new constant in the mechanism's `parameters`; it is minted per row as "
        "`<prefix>_<name>`." if new else "",
    ]
    raise MechanismError(" ".join(s for s in sentences if s))


#: A name a written mechanism's own parameter may take: ASCII, as SBML ids are.
_PARAMETER_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
#: Names a written mechanism's parameter may not take: what a rate reads otherwise, and the
#: components the mechanism builds beside its parameters (``<prefix>_rate``, the base entry's
#: ``<prefix>_base_<kind>``; a scheme's rules ``<prefix>_r<i>`` and rates ``<prefix>_rate<i>_<j>``).
_TAKEN = re.compile(
    rf"time|amount|exp|log|sqrt|piecewise|rate(?:\d+_\d+\w*)?|r\d+|base_\w*|{REGULATOR_NAME.pattern}"
)


def refuse_declared(policy: str, declared, roles) -> None:
    """Refuse ``declared`` parameters that cannot be new constants: a name no SBML id takes, or
    one the rate reads otherwise (a role, a list role by position) or a component the mechanism
    builds takes."""
    bad = sorted(n for n in declared if not _PARAMETER_NAME.fullmatch(n) or keyword.iskeyword(n))
    if bad:
        raise MechanismError(
            f"{policy!r} declares {bad} in `parameters`: a parameter is named by ASCII letters, "
            "digits and `_`, not starting with a digit and no Python keyword (`k_1`, not `k-1`)"
        )
    role = re.compile(rf"(?:{'|'.join(map(re.escape, roles))})(?:_\d+)?") if roles else None
    taken = sorted(n for n in declared if _TAKEN.fullmatch(n) or (role and role.fullmatch(n)))
    if taken:
        raise MechanismError(
            f"{policy!r} declares {taken} in `parameters`, names taken already: `time`, `amount` "
            "and exp/log/sqrt are the rate's own, a role and a regulator (`inhibited_by`) are "
            "read as amounts, and `rate`, "
            "`r<i>`, `rate<i>_<j>` and `base_...` name what the mechanism builds. Name each "
            f"apart, e.g. `k_{taken[0]}`"
        )


def refuse_unread(policy: str, declared, named) -> None:
    """Refuse ``declared`` parameters the rate never ``named``: each would be a parameter of every
    row that nothing reads."""
    if unread := sorted(set(declared) - set(named)):
        raise MechanismError(
            f"{policy!r} declares {unread} in `parameters`, which its rate never names: each would "
            "be a parameter of every row that nothing reads. Name it in the rate or leave it out"
        )


def parse_rate(text: str) -> sympy.Expr:
    """``text`` as ONE sympy expression, each name in it a placeholder symbol and
    exp/log/sqrt/piecewise the functions: what a rate, or a binding's expression, must read as before its names become
    components. Checked against :data:`_RATE_TOKENS` first, since sympy evaluates the text.
    Raises :class:`MechanismError` saying how to write it."""
    tokens = _RATE_TOKENS.findall(text)
    if "".join(tokens) != text:
        raise MechanismError(
            f"rate {text!r} holds characters no rate takes: names, numbers, `+ - * / ^`, "
            "parentheses, commas only between a function's arguments, and `< <= > >=` only in a "
            "piecewise condition"
        )
    names = {t for t in tokens if _IDENTIFIER.fullmatch(t)} - set(_RATE_FUNCTIONS)
    try:
        value = sympy.sympify(
            text.replace("^", "**"),
            locals={**{n: sympy.Symbol(n) for n in names}, **_RATE_FUNCTIONS},
        )
    except (sympy.SympifyError, SyntaxError, TypeError, ValueError, AttributeError):
        value = None
    if not isinstance(value, sympy.Expr):
        raise MechanismError(
            f"rate {text!r} does not read as one expression: write `*` between factors "
            "(`kcat*Km`, not `kcat Km` or `2kcat`), apply a function to its argument "
            "(`exp(-k*time)`), join terms instead of listing them with commas, and compare only "
            "in a piecewise condition (`piecewise(k_0, time < t_1, k_1)`)"
        )
    return value


def written(
    definition: dict,
    library: dict[tuple[str, str], Policy],
    problem_ids: dict[str, str] | None = None,
) -> Policy:
    """Compile a mechanism the agent wrote: the reaction of an existing entry (``based_on``) with
    a new ``rate``.

    Only the rate is the agent's. The reaction — what is consumed, what is produced — stays the
    base entry's, and the rate is a COEFFICIENT that mass action multiplies by every reactant's
    amount, catalysts included, so a rate that stays non-negative can never drive an amount
    negative (``kcat / (Km + sub)`` is Michaelis-Menten). The rate names the statement type's roles
    (``enz``, ``sub``, ``source``...), each the amount of that agent as the statement writes it —
    a list role by position (``obj_from_0``), or by its own name when it holds one agent; the
    definition's ``parameters``, each minted as ``<prefix>_<name>`` (and shared by a row's
    binding, :func:`bind`); the regulators the row's statement states (``inhibited_by``,
    ``activated_by``, ``modulated_by``, by position where it states several of a kind,
    :data:`statements.REGULATOR_NAME`), each the amount of an agent that takes no part in the
    reaction;
    ``time``; exp, log, sqrt, and ``piecewise(value, condition, ..., otherwise)``, its conditions
    comparisons over the same names (:func:`_piecewise`). A rate must fall with each inhibitor it
    reads and rise with each activator (:func:`check_regulators`), and read every regulator a row
    states (:func:`_read_regulators`). A scheme (:func:`schemes.compile_scheme`) declares the
    constants its rates name the same way.

    ``problem_ids`` (id to :data:`SHIPPED`, :data:`CONDITION` or :data:`SPECIES`) are names no
    rate reads and no declared parameter takes: a refusal naming one says how it enters instead,
    through a row's binding or as an amount.
    """
    stmt_type, policy = definition["statement_type"], definition["policy"]
    base_key = (stmt_type, definition.get("based_on", INDRA_DEFAULT))
    if (stmt_type, policy) in REGISTRY:
        raise MechanismError(f"{(stmt_type, policy)} is a built-in mechanism; name yours apart")
    try:
        roles = list(get_statement_by_name(stmt_type)._agent_order)
    except NotAStatementName:
        raise MechanismError(
            f"{stmt_type!r} is no statement type; `list_mechanisms` names them"
        ) from None
    amounts = (
        "`amount(<pattern>)`" if definition.get("scheme")
        else "a role (`sub`) or a regulator (`inhibited_by`)"
    )
    declared = list(definition.get("parameters") or [])
    refuse_declared(policy, declared, roles)
    if shadow := sorted(set(declared) & set(problem_ids or {})):
        raise MechanismError(
            " ".join(
                [
                    f"{policy!r} declares {shadow} in `parameters`, each a new parameter minted "
                    "per row as `<prefix>_<name>`: name yours apart.",
                    *(h for h in _how_ids_enter(shadow, problem_ids, amounts) if h),
                ]
            )
        )
    if definition.get("scheme"):
        from .schemes import compile_scheme

        if definition.get("rate"):
            raise MechanismError(f"{policy!r} gives a scheme and a rate; a scheme carries its rates")
        return compile_scheme(definition, problem_ids)
    if not definition.get("rate"):
        raise MechanismError(f"{policy!r} gives neither a `scheme` nor a `rate`")
    if base_key not in library:
        raise MechanismError(f"no mechanism {base_key} to base {policy!r} on")
    names = set(declared)
    rate = definition["rate"]
    tokens = _RATE_TOKENS.findall(rate)
    positional = re.compile(rf"^(?:{'|'.join(map(re.escape, roles))})_\d+$")
    unknown = sorted(
        {t for t in tokens if t[:1].isalpha() or t[:1] == "_"}
        - names
        - set(roles)
        - set(_RATE_FUNCTIONS)
        - {"time"}
        - {t for t in tokens if positional.match(t) or REGULATOR_NAME.fullmatch(t)}
    )
    allowed = sorted(names) + roles + ["time"]
    also = (
        " (a list role by position, e.g. obj_from_0), the regulators a statement states "
        "(inhibited_by, activated_by, modulated_by) and exp/log/sqrt/piecewise"
    )
    if unknown:
        refuse_rate_names(rate, policy, unknown, allowed, also, amounts, problem_ids)
    value = assumed(parse_rate(rate))
    check_regulators(policy, rate, value)
    refuse_unread(policy, declared, tokens)
    base = library[base_key]

    def policy_fn(
        stmt: Statement,
        model: pysb.Model,
        prefix: str,
        regulators: dict[str, Agent] | None = None,
    ) -> Emission:
        # The base's own parameters go unused once re-rated; kept apart from the written ones.
        emitted = base(stmt, model, f"{prefix}_base")
        if len(emitted.rules) != 1 or model.rules[emitted.rules[0]].is_reversible:
            if not emitted.rules and isinstance(stmt, Modification) and stmt.enz is None:
                # `X is phosphorylated` under a mechanism based on indra_default: INDRA builds a
                # modification only through its enzyme, so there was nothing to re-rate.
                raise MechanismError(
                    f"row {prefix!r}: {base_key} builds no reaction for a modification without "
                    "an enzyme. Name the enzyme (`E phosphorylates X`), or base the mechanism on "
                    "`basal`, the library's enzyme-less modification"
                )
            raise MechanismError(
                f"{base_key} builds {len(emitted.rules)} reaction(s) for this statement; a written "
                "mechanism re-rates exactly one irreversible reaction"
            )
        params = {n: pysb.Parameter(f"{prefix}_{n}", 1.0, _export=False) for n in sorted(names)}
        agents = {}
        for role in roles:
            value = getattr(stmt, role, None)
            if isinstance(value, Agent):
                agents[role] = value
            elif isinstance(value, list):
                agents |= {f"{role}_{i}": agent for i, agent in enumerate(value)}
                if len(value) == 1:
                    agents[role] = value[0]
        agents |= regulators or {}
        used = {t for t in tokens if t in agents}
        if missing := sorted({t for t in tokens if t in roles or positional.match(t)} - set(agents)):
            raise MechanismError(
                f"row {prefix!r}: the rate of {policy!r} reads {missing}, which its statement "
                "leaves empty; a role holding several agents is read by position (`obj_from_0`). "
                "Give the statement that agent, or use a mechanism whose rate does not read it"
            )
        amounts = {
            name: pysb.Observable(
                f"{prefix}_{name}", get_monomer_pattern(model, agents[name]), _export=False
            )
            for name in sorted(used | (set(agents) & set(roles)))
        }
        _add(model, *params.values(), *amounts.values())
        coefficient = sympy.sympify(
            rate.replace("^", "**"), locals={**params, **amounts, **_RATE_FUNCTIONS, "time": time}
        )
        expression = pysb.Expression(f"{prefix}_rate", coefficient, _export=False)
        _add(model, expression)
        # The base's own rate constant is left unreferenced; the SBML export drops it.
        model.rules[emitted.rules[0]].rate_forward = expression
        return Emission(
            (stmt_type, policy),
            prefix,
            emitted.roles,
            rules=emitted.rules,
            parameters=tuple(p.name for p in params.values()),
        )

    policy_fn.indra_policy = getattr(base, "indra_policy", "default")
    # What the rate depends on, not what it spells: a regulator it cancels carries no claim.
    policy_fn.regulators = frozenset(
        s.name for s in value.free_symbols if REGULATOR_NAME.fullmatch(s.name)
    )
    return policy_fn


def assumed(expr: sympy.Expr) -> sympy.Expr:
    """``expr`` with each of its names a signed symbol: a regulator's amount never negative and
    possibly 0 (no drug), every other name positive."""
    return expr.xreplace({
        s: sympy.Symbol(s.name, nonnegative=True) if REGULATOR_NAME.fullmatch(s.name)
        else sympy.Symbol(s.name, positive=True)
        for s in expr.free_symbols
    })


def refuse_unnamed(prefix: str, stmt: Statement, reads: str, agent: Agent) -> None:
    """Refuse row ``prefix``'s mechanism reading ``agent`` (``reads`` says where) because ``stmt``
    does not name it: the statement is the claim behind the rate, so an agent the rate reads is
    stated there, as a regulator (``statements.REGULATORS``) when it takes no part."""
    from .notation import agent_text, statement_text

    def noted(write, value, otherwise: str) -> str:
        try:
            return write(value.to_json())
        except ValueError:  # what the notation cannot write
            return otherwise

    listed = ", ".join(dict.fromkeys(noted(agent_text, a, a.name) for a in named_agents(stmt)))
    said = noted(statement_text, stmt, "<statement>")
    name = noted(agent_text, agent, agent.name)
    raise MechanismError(
        f"row {prefix!r}: {reads} reads {name}, which its statement does not name. A rate reads "
        f"only the agents its statement names ({listed}) and the regulators it states. State "
        f"{agent.name} as one: `{said}, inhibited by {name}` (activated by, modulated by), read as "
        "`amount(inhibited_by)`; a partner the rules bind is written on its agent (`E{~"
        f"{agent.name}}}`)."
    )


def assemble(
    rows: list[tuple],
    entities: dict[str, Agent] | None = None,
    library: dict[tuple[str, str], Policy] = REGISTRY,
) -> tuple[pysb.Model, list[Emission]]:
    """Build the whole model from ``(statement, policy, prefix)`` rows, each optionally with a
    fourth element, its parameter binding (:func:`bind`), and a fifth, the regulators its
    statement states, by kind (:func:`statements.regulators`), which INDRA's statement cannot hold.

    ``entities`` (species id to ``Agent``, from the problem's entity map) joins the monomer
    pass, so a species the problem observes has its monomer and sites even when no row names it.
    """
    rows = [(*row, *[None] * (5 - len(row))) for row in rows]
    stmts = [row[0] for row in rows]
    # INDRA's monomer pass collects every site its own statement types imply; it skips the
    # vocabulary in :mod:`statements`, whose agents carry only the sites their own conditions
    # name, so those are added directly.
    indra = PysbAssembler(stmts)
    indra.processed_policies = {
        stmt.uuid: IndraPolicy(_monomer_policy(policy, library)) for stmt, policy, *_ in rows
    }
    indra.agent_set = agents = BaseAgentSet()
    indra._monomers()
    for stmt in stmts:
        for agent in stmt.real_agent_list():
            agents.get_create_base_agent(agent)
    for agent in (entities or {}).values():
        agents.get_create_base_agent(agent)
    # INDRA's pass gives a substrate its modification site only through an enzyme; an
    # enzyme-less modification (`basal`) would otherwise name a site its monomer lacks.
    for stmt in stmts:
        if isinstance(stmt, Modification) and stmt.sub is not None:
            agents.get_create_base_agent(stmt.sub).create_mod_site(stmt._get_mod_condition())
    for *_, regulators in rows:
        for agent in (a for given in (regulators or {}).values() for a in given):
            agents.get_create_base_agent(agent)
    # A scheme entry (:mod:`schemes`) names sites INDRA's pass does not give: its bonds, a
    # molecule the statement does not mention. Its states are checked against the sites the
    # molecule has without any scheme's.
    given = {
        name: (list(base.sites), {s: list(v) for s, v in base.site_states.items()})
        for name, base in agents.agents.items()
    }
    added: list[dict[str, list[str]]] = []
    for stmt, policy, prefix, *_ in rows:
        entry = library.get((type(stmt).__name__, policy))
        added.append({})
        if not hasattr(entry, "scheme_sites"):
            continue
        before = {name: set(base.sites) for name, base in agents.agents.items()}
        for agent, site, site_states in entry.scheme_sites(stmt, given, prefix):
            base = agents.get_create_base_agent(agent)
            if site is not None:
                base.create_site(site, site_states)
        added[-1] = {
            name: [s for s in base.sites if s not in before.get(name, ())]
            for name, base in agents.agents.items()
        }

    model = pysb.Model(name="statements", _export=False)
    for base in agents.agents.values():
        monomer = pysb.Monomer(base.name, base.sites, base.site_states, _export=False)
        model.add_component(monomer)
        for db_name, db_ref in base.db_refs.items():
            annotation = get_annotation(monomer, db_name, db_ref)
            if annotation is not None:
                model.add_annotation(annotation)
    emissions = [
        instantiate(stmt, model, policy, prefix, library, regulators)
        for stmt, policy, prefix, _, regulators in rows
    ]
    _refuse_ruleless(rows, emissions)
    _refuse_unchanging(rows, emissions, model)
    # Bound once every row has minted its own: bound as each row was built, a row sharing a LATER
    # row's `back_k` made it a new shared id, and that row's own `back_k` then clashed with it.
    merged: dict[str, object] = {}
    emissions = [
        bind(model, emission, row[3], merged, set(entities or {})) if row[3] else emission
        for emission, row in zip(emissions, rows)
    ]
    return model, [
        replace(e, sites_added={
            name: {site: list(agents.agents[name].site_states.get(site, [])) for site in sites}
            for name, sites in names.items() if sites
        })
        for e, names in zip(emissions, added)
    ]


def _monomer_policy(policy: str, library: dict[tuple[str, str], Policy]) -> str:
    """The INDRA policy whose monomer pass a row needs: its entry's, or for a written entry the
    one it was based on; ``default`` for the entries of this module, which build no sites."""
    entries = [fn for (_, p), fn in library.items() if p == policy]
    return getattr(entries[0], "indra_policy", "default") if entries else "default"


def _refuse_ruleless(rows: list[tuple], emissions: list[Emission]) -> None:
    """Refuse a row whose entry built no rule: the model would not contain what it asserts, and a
    constant it minted would be fitted to nothing. INDRA builds a rule once, so the same statement
    in a second row built nothing beside an orphan ``kf``, and it builds none for an ActiveForm."""
    for i, (row, emission) in enumerate(zip(rows, emissions)):
        if emission.rules:
            continue
        stmt, policy, prefix = row[:3]
        key = _state_unwritten(stmt).matches_key()
        twin = next((r[2] for r in rows[:i] if _state_unwritten(r[0]).matches_key() == key), None)
        if twin is not None:
            raise MechanismError(
                f"row {prefix!r} asserts the statement of row {twin!r} again, and INDRA builds its "
                "rule once, so this row builds nothing: merge the two rows, or give one another "
                "policy"
            )
        raise MechanismError(
            f"row {prefix!r}: {type(stmt).__name__} under {policy!r} builds no rule, so the "
            "model would not contain it. Give it a policy that builds one, or write one with "
            "`mechanisms`"
        )


def _refuse_unchanging(rows: list[tuple], emissions: list[Emission], model: pysb.Model) -> None:
    """Refuse a row that built a rule whose two sides are the same: the model would not contain
    what the row asserts, and the rule's constant would be fitted to nothing. A state written on
    an agent holds on both sides of a rule that sets it (:func:`_state_unwritten`), so the
    message names the agents' written states."""
    from .notation import agent_text

    for row, emission in zip(rows, emissions):
        stmt, policy, prefix = row[:3]
        for name in emission.rules:
            rule = model.rules[name]
            if not _unchanging(rule):
                continue
            written = []
            for agent in stmt.real_agent_list():
                if agent.mods or agent.activity or agent.location or agent.bound_conditions:
                    try:
                        written.append(agent_text(agent.to_json()))
                    except ValueError:
                        written.append(str(agent))
            raise MechanismError(
                f"row {prefix!r}: {type(stmt).__name__} under {policy!r} builds "
                f"`{rule.rule_expression}`, which changes nothing"
                + (
                    f": the row writes {', '.join(written)}, and a state written on an agent holds "
                    "on both sides of a rule that sets it. Leave the state the statement changes "
                    "off its agent, or give the row a mechanism that sets it"
                    if written
                    else ". Give the row a mechanism whose rule changes something"
                )
            )


def _unchanging(rule: pysb.Rule) -> bool:
    """Whether ``rule``'s products are its reactants written alike, complex for complex. Not
    PySB's ``is_equivalent_to``, which refuses a pattern that is not concrete (INDRA's
    transphosphorylation writes its enzyme ``egfr=ANY``)."""
    return Counter(map(str, rule.reactant_pattern.complex_patterns)) == Counter(
        map(str, rule.product_pattern.complex_patterns)
    )


def _add(model: pysb.Model, *components) -> None:
    for component in components:
        model.add_component(component)


def agent_pattern(
    model: pysb.Model, agent: Agent, fresh=lambda name: False, partners: bool = True
) -> list:
    """``agent`` as the molecules it writes: its own stated sites (a modification as its state,
    not INDRA's ``(state, WILD)``, which on a product bonds to nothing), and for each ``is_bound``
    condition the partner, built the same way and bonded on the sites INDRA names after each other.

    ``fresh(monomer name)`` says whether a molecule is NEW — made by the rule rather than carried
    over from a reactant, which BioNetGen maps by type in order — and a new molecule is written in
    full: every unstated site in its default state, every unbound site free. A pattern on the left,
    or a molecule carried over, leaves unstated sites unstated, so they keep what they had.

    With ``partners`` off a bound condition stays "bound to something" and builds no partner:
    BioNetGen's DeleteMolecules deletes every molecule a pattern names, so a release names only
    the molecule it removes.
    """
    bonds = count(1)

    def visit(a: Agent, partner: tuple | None) -> list:
        monomer = model.monomers[_n(a.name)]
        stated = get_site_pattern(a)
        unknown = sorted(set(stated) - set(monomer.sites))
        if unknown:
            raise MechanismError(f"{a.name} states sites {unknown} that no statement gives it")
        state = {
            site: value[0] if isinstance(value, tuple) else value
            for site, value in stated.items()
            if value is not ANY
        }
        if fresh(monomer.name):
            for site in monomer.sites:
                if site in state:
                    continue
                if site in monomer.site_states:
                    states_ = monomer.site_states[site]
                    state[site] = "cytoplasm" if site == "loc" and "cytoplasm" in states_ else states_[0]
                else:
                    state[site] = None
        bound = []
        for condition in a.bound_conditions:
            if condition.is_bound and not partners:
                state[get_binding_site_name(condition.agent)] = ANY
            elif condition.is_bound:
                bond = next(bonds)
                state[get_binding_site_name(condition.agent)] = bond
                bound += visit(condition.agent, (get_binding_site_name(a), bond))
        if partner is not None:
            state[partner[0]] = partner[1]
        return [monomer(**state), *bound]

    return visit(agent, None)


def _sides(model: pysb.Model, before: list[Agent], after: list[Agent], partners: bool = True):
    """The two sides of one rule turning ``before`` into ``after``, each agent a complex (or, with
    ``partners`` off, the reactant molecule alone, :func:`agent_pattern`). A product molecule
    beyond the reactant molecules of its type is new."""
    lhs = [pysb.ComplexPattern(agent_pattern(model, a, partners=partners), None) for a in before]
    carried = Counter(mp.monomer.name for cp in lhs for mp in cp.monomer_patterns)

    def fresh(name: str) -> bool:
        carried[name] -= 1
        return carried[name] < 0

    rhs = [pysb.ComplexPattern(agent_pattern(model, a, fresh), None) for a in after]
    return ReactionPattern(lhs), ReactionPattern(rhs)


def _one_rule(stmt, model, prefix, policy, before, after, name, roles, delete_molecules=False):
    """One irreversible mass-action rule ``before >> after`` at ``<prefix>_k``."""
    k = pysb.Parameter(f"{prefix}_k", 1.0, _export=False)
    lhs, rhs = _sides(model, before, after, partners=not delete_molecules)
    rule = pysb.Rule(
        f"{prefix}_{name}",
        RuleExpression(lhs, rhs, False),
        k,
        delete_molecules=delete_molecules,
        _export=False,
    )
    _add(model, k, rule)
    return Emission(
        (type(stmt).__name__, policy), prefix, roles, rules=(rule.name,), parameters=(k.name,)
    )


#: Mass action written from the statement's own agents, as complexes in the states it writes.
MASS_ACTION = "mass_action"
#: The same rule, with BioNetGen's DeleteMolecules: a consumed molecule leaves its partners free.
RELEASE = "release"


@mechanism(PopulationTransition, MASS_ACTION)
def _transition(stmt: Statement, model: pysb.Model, prefix: str) -> Emission:
    """First-order transfer, ``k·source``."""
    k = pysb.Parameter(f"{prefix}_k", 1.0, _export=False)
    source, target = _sides(model, [stmt.source], [stmt.target])
    rule = pysb.Rule(
        f"{prefix}_transition", RuleExpression(source, target, False), k, _export=False
    )
    _add(model, k, rule)
    return Emission(
        ("PopulationTransition", "mass_action"),
        prefix,
        {"source": stmt.source.name, "target": stmt.target.name},
        rules=(rule.name,),
        parameters=(k.name,),
    )


#: The policy of a modification lost at a steady rate, with no named enzyme.
BASAL = "basal"


def _state_unwritten(stmt: Statement) -> Statement:
    """``stmt`` with the state it changes left off the agent it changes it on: the modified site,
    the regulated activity, the location. For the entries whose sides INDRA's
    ``get_monomer_pattern(..., extra_fields=...)`` builds, which puts the agent's written state
    over the rule's, so "AKT1{pS473} is dephosphorylated at S473" built ``pS473 >> pS473``.
    Schemes and written rates still read the agents as written."""
    stmt = copy(stmt)
    if isinstance(stmt, Modification) and stmt.sub is not None:
        stmt.sub = _without_site(stmt.sub, stmt._get_mod_condition())
    elif isinstance(stmt, Autophosphorylation):
        stmt.enz = _without_site(stmt.enz, stmt._get_mod_condition())
    elif isinstance(stmt, Transphosphorylation) and stmt.enz.bound_conditions:
        first, *rest = stmt.enz.bound_conditions
        first = copy(first)
        first.agent = _without_site(first.agent, stmt._get_mod_condition())
        stmt.enz = copy(stmt.enz)
        stmt.enz.bound_conditions = [first, *rest]
    elif isinstance(stmt, RegulateActivity):
        stmt.obj = _without_activity(stmt.obj, stmt.obj_activity)
    elif isinstance(stmt, (Gef, Gap)):
        stmt.ras = _without_activity(stmt.ras, "gtpbound")
    elif isinstance(stmt, Translocation):
        stmt.agent = copy(stmt.agent)
        stmt.agent.location = None
    return stmt


def _without_site(agent: Agent, mod: ModCondition) -> Agent:
    site = get_mod_site_name(mod)
    agent = copy(agent)
    agent.mods = [m for m in agent.mods if get_mod_site_name(m) != site]
    return agent


def _without_activity(agent: Agent, activity: str) -> Agent:
    agent = copy(agent)
    if agent.activity is not None and agent.activity.activity_type == activity:
        agent.activity = None
    return agent


def _no_enzyme(stmt: Statement, prefix: str) -> None:
    """Refuse a `basal` row whose statement names an enzyme: the rule has no place for one, so the
    enzyme the row asserts would be left out of the model without a word."""
    if stmt.enz is not None:
        others = sorted(p for c, p in REGISTRY if c == type(stmt).__name__ and p != BASAL)
        raise MechanismError(
            f"row {prefix!r}: `{BASAL}` is a modification with no enzyme, and this statement names "
            f"{stmt.enz.name}, which it would leave out of the model. Drop the enzyme from the "
            f"statement, or give the row a policy that uses it: {others}"
        )


def _basal_removal(stmt: Statement, model: pysb.Model, prefix: str) -> Emission:
    """``k·[modified]``: the modification comes off at a steady rate. INDRA's own default builds
    nothing for a removal with no enzyme, so without this a modification can only accumulate —
    and a transient modification could never decay."""
    _no_enzyme(stmt, prefix)
    mod = stmt._get_mod_condition()
    site = get_mod_site_name(mod)
    unmod, modified = states[mod.mod_type]
    k = pysb.Parameter(f"{prefix}_k", 1.0, _export=False)
    sub = _state_unwritten(stmt).sub
    before = get_monomer_pattern(model, sub, extra_fields={site: modified})
    after = get_monomer_pattern(model, sub, extra_fields={site: unmod})
    rule = pysb.Rule(f"{prefix}_removal", before >> after, k, _export=False)
    _add(model, k, rule)
    return Emission(
        (type(stmt).__name__, BASAL),
        prefix,
        {"sub": stmt.sub.name},
        rules=(rule.name,),
        parameters=(k.name,),
    )


def _conversion(policy: str, delete_molecules: bool) -> Policy:
    def apply(stmt: Statement, model: pysb.Model, prefix: str) -> Emission:
        """``k·[subj]·Π[obj_from]``: the reactants, each the complex its agent writes, become the
        products in the states theirs write; ``subj`` catalyses. INDRA's own Conversion takes one
        reactant and makes products in their base state, so a homodimerisation (2 A -> A:A), a
        cleavage into two products, or a product made in a stated form had no route."""
        if not stmt.obj_from:
            raise MechanismError(f"row {prefix!r}: a Conversion needs obj_from")
        catalyst = [stmt.subj] if stmt.subj is not None else []
        return _one_rule(
            stmt,
            model,
            prefix,
            policy,
            catalyst + list(stmt.obj_from),
            catalyst + list(stmt.obj_to or []),
            "conversion",
            {role: _role_names(getattr(stmt, role)) for role in stmt._agent_order},
            delete_molecules,
        )

    return apply


def _removal(stmt: Statement, model: pysb.Model, prefix: str) -> Emission:
    """``k·[subj]·[obj]``, removing only the obj molecule: a partner it was bound to is freed
    (a ligand degraded while bound leaves its receptor behind), where
    INDRA's degradation deletes the whole complex."""
    catalyst = [stmt.subj] if stmt.subj is not None else []
    return _one_rule(
        stmt,
        model,
        prefix,
        RELEASE,
        catalyst + [stmt.obj],
        catalyst,
        "release",
        {"subj": _role_names(stmt.subj), "obj": stmt.obj.name},
        delete_molecules=True,
    )


@mechanism(IncreaseAmount, MASS_ACTION)
def _synthesis(stmt: Statement, model: pysb.Model, prefix: str) -> Emission:
    """``k·[subj]``: obj is made, from nothing or catalysed by ``subj``, in the state the
    statement writes — INDRA's own IncreaseAmount makes it in its base state, so a protein made
    at a location or in a modified form had no route."""
    catalyst = [stmt.subj] if stmt.subj is not None else []
    return _one_rule(
        stmt,
        model,
        prefix,
        MASS_ACTION,
        catalyst,
        catalyst + [stmt.obj],
        "synthesis",
        {"subj": _role_names(stmt.subj), "obj": stmt.obj.name},
    )


mechanism(Conversion, MASS_ACTION)(_conversion(MASS_ACTION, False))
mechanism(Conversion, RELEASE)(_conversion(RELEASE, True))
mechanism(DecreaseAmount, RELEASE)(_removal)


@mechanism(IncreaseAmount, "self_renewal")
def _self_renewal(stmt: Statement, model: pysb.Model, prefix: str) -> Emission:
    """``k·[subj]·[obj]``: obj makes more of itself (a proliferating population). INDRA reads
    "X increases X" as nothing, and IncreaseAmount
    with no subject makes obj from nothing, which also starts it empty."""
    catalyst = [stmt.subj] if stmt.subj is not None else []
    return _one_rule(
        stmt,
        model,
        prefix,
        "self_renewal",
        catalyst + [stmt.obj],
        catalyst + [stmt.obj, stmt.obj],
        "renewal",
        {"subj": _role_names(stmt.subj), "obj": stmt.obj.name},
    )


def _basal_addition(stmt: Statement, model: pysb.Model, prefix: str) -> Emission:
    """``k·[unmodified]``: the modification goes on at a steady rate, with no named enzyme — a
    basal, ligand-independent modification. The counterpart of :func:`_basal_removal`."""
    _no_enzyme(stmt, prefix)
    mod = stmt._get_mod_condition()
    site = get_mod_site_name(mod)
    unmod, modified = states[mod.mod_type]
    k = pysb.Parameter(f"{prefix}_k", 1.0, _export=False)
    sub = _state_unwritten(stmt).sub
    before = get_monomer_pattern(model, sub, extra_fields={site: unmod})
    after = get_monomer_pattern(model, sub, extra_fields={site: modified})
    rule = pysb.Rule(f"{prefix}_addition", before >> after, k, _export=False)
    _add(model, k, rule)
    return Emission(
        (type(stmt).__name__, BASAL),
        prefix,
        {"sub": stmt.sub.name},
        rules=(rule.name,),
        parameters=(k.name,),
    )


for _cls in get_all_descendants(AddModification):
    mechanism(_cls, BASAL)(_basal_addition)


def _plain(pattern: dict) -> dict:
    """A site pattern with INDRA's ``(state, WILD)`` reduced to the state."""
    return {site: value[0] if isinstance(value, tuple) else value for site, value in pattern.items()}


def _binding_sites(monomer: pysb.Monomer) -> list[str]:
    return [site for site in monomer.sites if site not in monomer.site_states]


def _rate_constants(model: pysb.Model, prefix: str, *kinds: str) -> list[pysb.Parameter]:
    parameters = [pysb.Parameter(f"{prefix}_{kind}", 1.0, _export=False) for kind in kinds]
    _add(model, *parameters)
    return parameters


@mechanism(Complex, "stated")
def _stated_binding(stmt: Statement, model: pysb.Model, prefix: str) -> Emission:
    """``kf·[a]·[b] - kr·[a:b]``, both directions in the states the members write. INDRA's own
    dissociation reads unconditioned members, so a complex comes apart at the same rate in every
    state — an internalised receptor–ligand complex dissociating at the surface rate. A
    one-way dissociation is this entry with ``kf`` bound to 0 (:func:`bind`)."""
    if len(stmt.members) != 2:
        raise MechanismError(f"row {prefix!r}: `stated` binds exactly two members; see `assembly`")
    a, b = stmt.members
    pa = model.monomers[_n(a.name)](**_plain(get_site_pattern(a)))
    pb = model.monomers[_n(b.name)](**_plain(get_site_pattern(b)))
    sa, sb = get_binding_site_name(b), get_binding_site_name(a)
    kf, kr = _rate_constants(model, prefix, "kf", "kr")
    rule = pysb.Rule(
        f"{prefix}_binding",
        pa(**{sa: None}) + pb(**{sb: None}) | pa(**{sa: 1}) % pb(**{sb: 1}),
        kf,
        kr,
        _export=False,
    )
    _add(model, rule)
    return Emission(
        ("Complex", "stated"),
        prefix,
        {"members": _role_names(stmt.members)},
        rules=(rule.name,),
        parameters=(kf.name, kr.name),
    )


@mechanism(Complex, "assembly")
def _assembly(stmt: Statement, model: pysb.Model, prefix: str) -> Emission:
    """``kf·Π[member]``: every member, free, joins the whole complex in one irreversible step,
    bonded as a spanning tree — each member to the first earlier one with which it shares a free
    pair of sites. Members may repeat (A with two B), which INDRA's
    ``multi_way`` cannot: it gives a monomer one site per partner type and overwrites it. Note
    BioNetGen's symmetry factor for identical reactants (``A + A`` runs at ``kf/2``)."""
    members = stmt.members
    monomers = [model.monomers[_n(m.name)] for m in members]
    used: list[set] = [set() for _ in members]
    bonds: list[dict] = [{} for _ in members]
    for i in range(1, len(members)):
        for j in range(i):
            si, sj = get_binding_site_name(members[j]), get_binding_site_name(members[i])
            if si in monomers[i].sites and sj in monomers[j].sites and si not in used[i] and sj not in used[j]:
                n = sum(len(b) for b in bonds) // 2 + 1
                used[i].add(si), used[j].add(sj)
                bonds[i][si] = bonds[j][sj] = n
                break
        else:
            raise MechanismError(
                f"row {prefix!r}: {members[i].name} has no free binding site to join the complex by"
            )
    lhs = rhs = None
    for agent, monomer, bond in zip(members, monomers, bonds):
        stated = {**_plain(get_site_pattern(agent)), **{s: None for s in _binding_sites(monomer)}}
        left, right = monomer(**stated), monomer(**{**stated, **bond})
        lhs = left if lhs is None else lhs + left
        rhs = right if rhs is None else rhs % right
    (kf,) = _rate_constants(model, prefix, "kf")
    rule = pysb.Rule(f"{prefix}_assembly", lhs >> rhs, kf, _export=False)
    _add(model, rule)
    return Emission(
        ("Complex", "assembly"),
        prefix,
        {"members": _role_names(members)},
        rules=(rule.name,),
        parameters=(kf.name,),
    )


@mechanism(Complex, "nondepleting")
def _nondepleting(stmt: Statement, model: pysb.Model, prefix: str) -> Emission:
    """The receptor (first member) binds a copy of the ligand (second) at ``kf·[R]·[L]``, the free
    ligand pool unchanged, and the bound copy is removed at ``kr`` from any receptor state, the
    receptor freed: a ligand in excess, its free pool unchanged by binding."""
    if len(stmt.members) != 2:
        raise MechanismError(f"row {prefix!r}: `nondepleting` takes a receptor and a ligand")
    receptor, ligand = stmt.members
    rec = model.monomers[_n(receptor.name)]
    lig = model.monomers[_n(ligand.name)]
    r_site, l_site = get_binding_site_name(ligand), get_binding_site_name(receptor)
    free = lig(**{**_plain(get_site_pattern(ligand)), l_site: None})
    copy = pysb.ComplexPattern(
        agent_pattern(model, Agent(ligand.name, db_refs=ligand.db_refs), fresh=lambda n: True),
        None,
    ).monomer_patterns[0]
    stated = rec(**_plain(get_site_pattern(receptor)))
    kf, kr = _rate_constants(model, prefix, "kf", "kr")
    occupy = pysb.Rule(
        f"{prefix}_occupy",
        free + stated(**{r_site: None})
        >> free + stated(**{r_site: 1}) % copy.monomer(**{**copy.site_conditions, l_site: 1}),
        kf,
        _export=False,
    )
    vacate = pysb.Rule(
        f"{prefix}_vacate",
        lig(**{l_site: ANY}) >> None,
        kr,
        delete_molecules=True,
        _export=False,
    )
    _add(model, occupy, vacate)
    return Emission(
        ("Complex", "nondepleting"),
        prefix,
        {"members": _role_names(stmt.members)},
        rules=(occupy.name, vacate.name),
        parameters=(kf.name, kr.name),
    )


def _processive(stmt: Statement, model: pysb.Model, prefix: str) -> Emission:
    """``kf·[enz]·[sub]``: one step sets the stated site and every other site of the same type
    the substrate writes as unmodified — a multi-site motif modified in one step, which single-site
    steps cannot take from unmodified to fully modified without an intermediate."""
    mod = stmt._get_mod_condition()
    unmod, modified = states[mod.mod_type]
    site = get_mod_site_name(mod)
    sites = [site] + [
        get_mod_site_name(m)
        for m in stmt.sub.mods
        if m.mod_type == mod.mod_type and not m.is_modified and get_mod_site_name(m) != site
    ]
    sub = model.monomers[_n(stmt.sub.name)]
    stated = _plain(get_site_pattern(stmt.sub))
    before = sub(**{**stated, **{s: unmod for s in sites}})
    after = sub(**{**stated, **{s: modified for s in sites}})
    (kf,) = _rate_constants(model, prefix, "kf")
    if stmt.enz is None:
        rule = pysb.Rule(f"{prefix}_processive", before >> after, kf, _export=False)
    else:
        enz = get_monomer_pattern(model, stmt.enz)
        rule = pysb.Rule(f"{prefix}_processive", enz + before >> enz + after, kf, _export=False)
    _add(model, rule)
    return Emission(
        (type(stmt).__name__, "processive"),
        prefix,
        {"enz": _role_names(stmt.enz), "sub": stmt.sub.name},
        rules=(rule.name,),
        parameters=(kf.name,),
    )


for _cls in get_all_descendants(AddModification):
    mechanism(_cls, "processive")(_processive)


def _dissociative(stmt: Statement, model: pysb.Model, prefix: str) -> Emission:
    """``k·[modified]``, with no enzyme, for the free substrate AND for every complex an EARLIER
    row forms around it, which then falls apart into its free members: a modified complex that
    comes apart once the modification is removed. Only complexes earlier rows have built
    are seen, so list this row after them."""
    mod = stmt._get_mod_condition()
    site = get_mod_site_name(mod)
    unmod, modified = states[mod.mod_type]
    monomer = model.monomers[_n(stmt.sub.name)]
    need = {**_plain(get_site_pattern(stmt.sub)), site: modified}
    free = {b: None for b in _binding_sites(monomer)}
    (k,) = _rate_constants(model, prefix, "k")
    rules = [
        pysb.Rule(
            f"{prefix}_free",
            monomer(**{**need, **free}) >> monomer(**{**need, **free, site: unmod}),
            k,
            _export=False,
        )
    ]
    seen: list = []
    for rule in list(model.rules):
        for cp in rule.product_pattern.complex_patterns:
            if cp is None or len(cp.monomer_patterns) < 2 or any(cp.is_equivalent_to(o) for o in seen):
                continue
            if not all(
                isinstance(mp.site_conditions.get(b, "?"), int) or mp.site_conditions.get(b, "?") is None
                for mp in cp.monomer_patterns
                for b in _binding_sites(mp.monomer)
            ):
                continue
            seen.append(cp)
            for i, mp in enumerate(cp.monomer_patterns):
                if mp.monomer is not monomer or any(
                    _plain({s: mp.site_conditions.get(s)}).get(s) != v
                    for s, v in need.items()
                    if s not in free
                ):
                    continue
                products = None
                for j, other in enumerate(cp.monomer_patterns):
                    state = {
                        s: v for s, v in _plain(other.site_conditions).items()
                        if s not in _binding_sites(other.monomer)
                    }
                    if j == i:
                        state[site] = unmod
                    part = other.monomer(**state, **{b: None for b in _binding_sites(other.monomer)})
                    products = part if products is None else products + part
                rules.append(
                    pysb.Rule(f"{prefix}_release{len(rules)}", cp.copy() >> products, k, _export=False)
                )
    _add(model, *rules)
    return Emission(
        (type(stmt).__name__, "dissociative"),
        prefix,
        {"sub": stmt.sub.name},
        rules=tuple(r.name for r in rules),
        parameters=(k.name,),
    )


for _cls in get_all_descendants(RemoveModification):
    mechanism(_cls, "dissociative")(_dissociative)

for _cls in get_all_descendants(RemoveModification):
    mechanism(_cls, BASAL)(_basal_removal)


#: The policy name every INDRA statement type answers to: INDRA's own default assembly, so every
#: statement INDRA can read has at least one entry. It is what an INDRA warm start is built with,
#: and the baseline the library's own entries extend.
INDRA_DEFAULT = "indra_default"


def _indra(name: str) -> Policy:
    """INDRA's own ``name`` policy for a statement; what it added is read off the model, before
    against after. Its parameters are renamed ``<prefix>_<kind>`` (``kf``, ``kr``, ``Km``...):
    INDRA's own names (``kf_ar_phosphorylation_1``) say neither which row minted them nor which
    statement, and an agent widening a bound had to look each one up."""
    entry = INDRA_DEFAULT if name == "default" else f"indra_{name}"

    def apply(stmt: Statement, model: pysb.Model, prefix: str) -> Emission:
        before = {c.name for c in model.all_components()}
        indra = PysbAssembler()
        indra.processed_policies = {stmt.uuid: IndraPolicy(name)}
        try:
            indra._dispatch(_state_unwritten(stmt), "assemble", model, BaseAgentSet(), {})
        except UnknownPolicyError as exc:
            raise MechanismError(
                f"INDRA has no {name} assembly for {type(stmt).__name__}", cause="library_gap"
            ) from exc
        added = [c for c in model.all_components() if c.name not in before]
        for parameter in (c for c in added if isinstance(c, pysb.Parameter)):
            renamed = f"{prefix}_{parameter.name.split('_')[0]}"
            # INDRA builds with PySB's global export off, which `rename` still tries to update.
            parameter._export = False
            parameter.rename(
                renamed
                if renamed not in model.parameters.keys()
                else f"{renamed}_{parameter.name}"
            )
        return Emission(
            (type(stmt).__name__, entry),
            prefix,
            {role: _role_names(getattr(stmt, role)) for role in stmt._agent_order},
            rules=tuple(c.name for c in added if isinstance(c, pysb.Rule)),
            parameters=tuple(c.name for c in added if isinstance(c, pysb.Parameter)),
        )

    # The monomer pass (:func:`assemble`) must run the same policy: ``two_step`` gives the enzyme
    # and substrate the binding site its complex forms on, which the default pass does not.
    apply.indra_policy = name
    return apply


_indra_default = _indra("default")


def _role_names(value) -> str:
    if isinstance(value, list):
        return ",".join(a.name for a in value)
    return "" if value is None else value.name


def _dispatch_name(cls: type[Statement]) -> str:
    """The name INDRA's ``_dispatch`` looks a statement class's policies up by."""
    if issubclass(cls, RemoveModification) and cls in modclass_to_modtype:
        return modclass_to_modtype[cls]
    if issubclass(cls, RegulateActivity):
        return "regulateactivity"
    return cls.__name__.lower()


_INDRA_ASSEMBLY = {n for n, f in vars(_indra_assembler).items() if "_assemble_" in n and callable(f)}

for _cls in {c for base in statement_whitelist for c in (base, *get_all_descendants(base))}:
    mechanism(_cls, INDRA_DEFAULT)(_indra_default)
    # Every OTHER policy INDRA itself implements for this type, as ``indra_<policy>``: two_step
    # (enzyme-substrate complex, then release), michaelis_menten, hill, multi_way,
    # interactions_only. Only the pairs INDRA defines: its dispatch falls back to the default
    # without a word when a policy is missing, which would build one model under another's name.
    # Not atp_dependent, whose rate reads an ATP no statement names (:func:`refuse_unnamed`).
    for _name in sorted(_INDRA_ASSEMBLY):
        _kind, _policy = _name.split("_assemble_")
        if _kind == _dispatch_name(_cls) and _policy not in ("default", "one_step", "atp_dependent"):
            mechanism(_cls, f"indra_{_policy}")(_indra(_policy))
