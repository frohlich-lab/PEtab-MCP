"""The mechanism library (``M``) and the assembler that applies it.

A mechanism is a policy: it receives one statement and the model under construction, and emits
the PySB rules and parameters that statement implies. That is the shape of
INDRA's own PySB assembler policies, so this module extends a dispatch that already exists —
INDRA supplies monomer and site construction, we supply the transformations.

Entries are keyed ``(statement class name, policy)``, and the two halves are deliberately
separate. ``Phosphorylation`` with INDRA's mass action and with an agent-written
Michaelis-Menten are two entries making the same biological claim through different mathematics;
collapsing them would leave the library unable to express "the same mechanism, simplified",
which is the axis several of the reported wins moved along.

Why a policy RETURNS what it emitted rather than leaving it to be found: the PEtab parameter
table must list every estimated parameter, and that list is currently recovered by scanning the
model for symbols with no row yet (:mod:`reconcile`). A generated model needs no scanning — the
generator knows what it generated. :class:`Emission` is that record, and it doubles as the
per-instantiation binding that makes a cross-instance ``revise`` addressable at all.

Policies must emit CONCRETE rules. An unbound site pattern is where rule-based models blow up
under ``generate_equations``, and the larger benchmarks are where that would be discovered.

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
import re
from collections.abc import Callable
from collections import Counter
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
    Complex,
    Conversion,
    DecreaseAmount,
    IncreaseAmount,
    Modification,
    NotAStatementName,
    RegulateActivity,
    RemoveModification,
    Statement,
    get_all_descendants,
    get_statement_by_name,
    modclass_to_modtype,
)
from indra.assemblers.pysb.assembler import _n, get_site_pattern
from indra.assemblers.pysb.sites import get_binding_site_name
from pysb import ANY
from pysb.core import ReactionPattern, RuleExpression, time

from .errors import MechanismError
from .statements import PopulationTransition


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


def instantiate(
    stmt: Statement,
    model: pysb.Model,
    policy: str,
    prefix: str,
    library: dict[tuple[str, str], Policy] = REGISTRY,
    modifiers: dict[str, Agent] | None = None,
) -> Emission:
    """Apply one entry of ``library`` (the registered entries plus any the agent wrote).
    ``modifiers`` fill the slots a written entry's rate reads (:func:`written`). Raises
    :class:`MechanismError` when the pair is not in it — the agent's signal to write one."""
    key = (type(stmt).__name__, policy)
    if key not in library:
        known = sorted(p for c, p in library if c == key[0])
        raise MechanismError(
            f"no mechanism {key}; the library has {known or 'none'} for {key[0]}. If none fits, "
            "write one with `mechanisms`.",
            cause="library_gap",
        )
    entry = library[key]
    if modifiers:
        if not getattr(entry, "modifiers", ()):
            raise MechanismError(
                f"row {prefix!r} names modifiers, but {key} reads none: only a mechanism written "
                "with a `rate` and `modifiers` reads them. A condition input (a drug dose) needs "
                "no modifier: bind the row's rate to an expression over it, "
                f'`kf="{prefix}_kf*(1 - Sorafenib/(K_i + Sorafenib))"`'
            )
        return entry(stmt, model, prefix, modifiers=modifiers)
    return entry(stmt, model, prefix)


def bind(
    model: pysb.Model,
    emission: Emission,
    binding: dict[str, str | float],
    merged: dict[str, object] | None = None,
) -> Emission:
    """Replace parameters a row minted (``<prefix>_<kind>``) by a SHARED one or a NUMBER.

    Every row mints its own rate constants, so one rate a reference uses in many reactions became
    many estimated parameters: Blasi's single ``a_basal`` in 25 acetylations, the 14 constants
    Sneyd's ten gating steps share, the shipped fixed rates Rahman's deaths run at. A string
    binds the kind to that id: a parameter an earlier row bound, or a new one of that name — which
    reconcile then takes from the problem's own row when the problem ships it. A number pins the
    kind to that value; it is no longer estimated. Any other string is an EXPRESSION over such ids
    (Isensee's off-rates ``KD_cAMP*kf_cAMP``), each name resolved as a shared id is, so the
    parameters are the reference's own and keep their rows. Bound for every entry alike, INDRA's
    and written ones, so no entry needs sharing of its own. What a row no longer reads stays behind
    unreferenced, and the SBML export drops it (as it drops a written mechanism's base constant).

    ``merged`` carries, across the rows of one model, each name an earlier binding renamed or
    replaced to what now stands for it, so a row that names another row's parameter shares what
    that row reads, whichever order they bind in.
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
            new, names = _bound_expression(model, name, target, emission.prefix, merged)
            # A binding that scales the row's own constant (`kf="mek_kf*(1 - Sorafenib/K)"`)
            # keeps that constant, and a row naming it shares the constant, not the scaling.
            if old not in new.expr.free_symbols:
                merged[name] = new
            _redirect(model, old, new)
            parameters += [n for n in names if n not in parameters]
            continue
        if isinstance(target, str):
            _not_a_parameter(model, target, emission.prefix, name)
            existing = _held(model, target, merged)
            if existing is None:
                merged[name] = old
                old._export = False  # INDRA builds with PySB's export off; see `_indra`
                old.rename(target)
                # A new shared id starts where any new name does (an entity-map `initial` or
                # `formula` names one at reconcile's seed too), not at the 1.0 the row minted:
                # reconcile fixes a starting-amount parameter that is exactly 1 as a unit
                # normalisation, so a shared rate also read by an `initial` was fixed or
                # estimated depending on which row named it first (Bachmann lost seven).
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


def _not_a_parameter(model: pysb.Model, ident: str, prefix: str, name: str) -> None:
    """Refuse a binding that reads ``ident`` as a parameter when it names something else: a
    molecule (a parameter of its name clashed with it in the SBML), a rate function, or ``time``
    (which BioNetGen refused as a parameter name)."""
    held = model.components.get(ident)
    if held is None and ident not in _RATE_FUNCTIONS and ident != "time":
        return
    if isinstance(held, (pysb.Parameter, pysb.Expression)):
        return
    what = (
        "a molecule" if isinstance(held, pysb.Monomer)
        else f"the model's {type(held).__name__.lower()}" if held is not None
        else "time" if ident == "time"
        else "a function"
    )
    raise MechanismError(
        f"row {prefix!r} binds {name!r} over {ident!r}, which names {what}, not a parameter. A "
        "binding reads parameters: a shared id, a problem parameter, a condition-table input. A "
        "rate that reads an amount or `time` is a written mechanism (its `rate`, or a scheme's "
        f"`amount(<pattern>)`)"
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
    model: pysb.Model, name: str, text: str, prefix: str, merged: dict[str, object]
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
            "shared id (`k_on`), a number, or arithmetic over ids and numbers with exp/log/sqrt "
            "(`KD*kf_on`)"
        )
    for n in names:
        _not_a_parameter(model, n, prefix, name)
    symbols = {n: _shared(model, n, merged) for n in names}
    value = sympy.sympify(text.replace("^", "**"), locals={**symbols, **_RATE_FUNCTIONS})
    expression = pysb.Expression(f"{name}_bound", value, _export=False)
    _add(model, expression)
    return expression, names


def _redirect(model: pysb.Model, old, new) -> None:
    """Point every rule rate and expression that reads ``old`` at ``new``, except ``new`` itself:
    a binding may read the parameter it replaces (``kf="mek_kf*(1 - Sorafenib/K)"`` scales the
    row's own constant), and redirecting that read into ``new`` made it read itself, which PySB
    met as a recursion with no bottom (three refusals in one sweep, none recovered)."""
    for rule in model.rules:
        if rule.rate_forward is old:
            rule.rate_forward = new
        if rule.rate_reverse is old:
            rule.rate_reverse = new
    for expression in model.expressions:
        if expression is not new and old in expression.expr.free_symbols:
            expression.expr = expression.expr.xreplace({old: new})


#: What a written rate may contain besides names: numbers, arithmetic, parentheses, commas. A
#: rate is parsed by sympy, which evaluates text, so it is checked against this first and every
#: name in it against the ones the mechanism allows.
_RATE_TOKENS = re.compile(r"\s+|(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?|[A-Za-z_]\w*|[-+*/^(),]")
_RATE_FUNCTIONS = {"exp": sympy.exp, "log": sympy.log, "sqrt": sympy.sqrt}


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
_TAKEN = re.compile(r"time|amount|exp|log|sqrt|rate(?:\d+_\d+\w*)?|r\d+|base_\w*")


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
            "and exp/log/sqrt are the rate's own, a role is read as its amount, and `rate`, "
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
    """``text`` as ONE sympy expression, each name in it a placeholder symbol and exp/log/sqrt the
    functions: what a rate, or a binding's expression, must read as before its names become
    components. Checked against :data:`_RATE_TOKENS` first, since sympy evaluates the text.
    Raises :class:`MechanismError` saying how to write it."""
    tokens = _RATE_TOKENS.findall(text)
    if "".join(tokens) != text:
        raise MechanismError(
            f"rate {text!r} holds characters no rate takes: names, numbers, `+ - * / ^`, "
            "parentheses, and commas only between a function's arguments"
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
            "(`exp(-k*time)`), and join terms instead of listing them with commas"
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
    binding, :func:`bind`); its ``modifiers``, each the amount of an agent the ROW names, which
    takes no part in the reaction (Fiedler's sorafenib slowing RAF, the SMAD complexes regulating
    Lucarelli's genes, Sneyd's IP3 and calcium); ``time``; exp, log, sqrt. A scheme
    (:func:`schemes.compile_scheme`) declares the constants its rates name the same way.

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
    amounts = "`amount(<pattern>)`" if definition.get("scheme") else "a role (`sub`) or a modifier"
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
        if definition.get("modifiers"):
            # Accepted and then dropped, before: every row naming them was refused with "reads
            # none" (17 refusals in one sweep, none recovered).
            raise MechanismError(
                f"{policy!r} is a scheme with `modifiers`; a scheme reads any agent in its rates "
                "as `amount(<pattern>)` instead, `@ kf/(1 + amount(Inhibitor)/K_i)` with `kf` and "
                "`K_i` in its `parameters`"
            )
        return compile_scheme(definition, problem_ids)
    if not definition.get("rate"):
        raise MechanismError(f"{policy!r} gives neither a `scheme` nor a `rate`")
    if base_key not in library:
        raise MechanismError(f"no mechanism {base_key} to base {policy!r} on")
    names = set(declared)
    slots = list(definition.get("modifiers") or [])
    clash = sorted(set(slots) & (names | set(roles)))
    if clash:
        raise MechanismError(f"{policy!r} names {clash} both as modifiers and otherwise")
    rate = definition["rate"]
    tokens = _RATE_TOKENS.findall(rate)
    positional = re.compile(rf"^(?:{'|'.join(map(re.escape, roles))})_\d+$")
    unknown = sorted(
        {t for t in tokens if t[:1].isalpha() or t[:1] == "_"}
        - names
        - set(roles)
        - set(slots)
        - set(_RATE_FUNCTIONS)
        - {"time"}
        - {t for t in tokens if positional.match(t)}
    )
    allowed = sorted(names) + slots + roles + ["time"]
    also = " (a list role by position, e.g. obj_from_0) and exp/log/sqrt"
    if unknown:
        refuse_rate_names(rate, policy, unknown, allowed, also, amounts, problem_ids)
    parse_rate(rate)
    refuse_unread(policy, declared, tokens)
    base = library[base_key]

    def policy_fn(
        stmt: Statement,
        model: pysb.Model,
        prefix: str,
        modifiers: dict[str, Agent] | None = None,
    ) -> Emission:
        given = modifiers or {}
        if set(given) != set(slots):
            raise MechanismError(
                f"row {prefix!r} uses {policy!r}, whose rate reads modifiers {sorted(slots)}; "
                f"the row names {sorted(given)}"
            )
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
        agents |= given
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
    policy_fn.modifiers = tuple(slots)
    return policy_fn


def assemble(
    rows: list[tuple],
    entities: dict[str, Agent] | None = None,
    library: dict[tuple[str, str], Policy] = REGISTRY,
) -> tuple[pysb.Model, list[Emission]]:
    """Build the whole model from ``(statement, policy, prefix)`` rows, each optionally with a
    fourth element, its parameter binding (:func:`bind`), and a fifth, the agents filling its
    entry's modifier slots (:func:`written`).

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
    for *_, modifiers in rows:
        for agent in (modifiers or {}).values():
            agents.get_create_base_agent(agent)
    # A scheme entry (:mod:`schemes`) names sites INDRA's pass does not give: its bonds, a
    # molecule the statement does not mention. Its states are checked against the sites the
    # molecule has without any scheme's.
    given = {
        name: (list(base.sites), {s: list(v) for s, v in base.site_states.items()})
        for name, base in agents.agents.items()
    }
    added: list[dict[str, list[str]]] = []
    for stmt, policy, *_ in rows:
        entry = library.get((type(stmt).__name__, policy))
        added.append({})
        if not hasattr(entry, "scheme_sites"):
            continue
        before = {name: set(base.sites) for name, base in agents.agents.items()}
        for agent, site, site_states in entry.scheme_sites(stmt, given):
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
        instantiate(stmt, model, policy, prefix, library, modifiers)
        for stmt, policy, prefix, _, modifiers in rows
    ]
    _refuse_ruleless(rows, emissions)
    # Bound once every row has minted its own: bound as each row was built, a row sharing a LATER
    # row's `back_k` made it a new shared id, and that row's own `back_k` then clashed with it.
    merged: dict[str, object] = {}
    emissions = [
        bind(model, emission, row[3], merged) if row[3] else emission
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
        twin = next((r[2] for r in rows[:i] if r[0].matches_key() == stmt.matches_key()), None)
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
    and Fujita's phospho-EGFR peaks at one minute and is gone by the hour."""
    _no_enzyme(stmt, prefix)
    mod = stmt._get_mod_condition()
    site = get_mod_site_name(mod)
    unmod, modified = states[mod.mod_type]
    k = pysb.Parameter(f"{prefix}_k", 1.0, _export=False)
    before = get_monomer_pattern(model, stmt.sub, extra_fields={site: modified})
    after = get_monomer_pattern(model, stmt.sub, extra_fields={site: unmod})
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
        reactant and makes products in their base state, so Boehm's 2 STAT5A -> dimer, Bruno's
        cleavage products and Raimundez's receptor dimer steps had no route."""
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
    (Schwen's internalised insulin, Isensee's hydrolysed cAMP leave their receptor behind), where
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
    at a location or in a modified form (Alkan's per-phase p21 and Wip1) had no route."""
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
    """``k·[subj]·[obj]``: obj makes more of itself (Crauste's effector expansion, the prey
    birth of Rosenzweig-MacArthur). INDRA reads "X increases X" as nothing, and IncreaseAmount
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
    """``k·[unmodified]``: the modification goes on at a steady rate, with no named enzyme —
    Blasi's acetylations, Brannmark's insulin-independent receptor phosphorylation. The
    counterpart of :func:`_basal_removal`."""
    _no_enzyme(stmt, prefix)
    mod = stmt._get_mod_condition()
    site = get_mod_site_name(mod)
    unmod, modified = states[mod.mod_type]
    k = pysb.Parameter(f"{prefix}_k", 1.0, _export=False)
    before = get_monomer_pattern(model, stmt.sub, extra_fields={site: unmod})
    after = get_monomer_pattern(model, stmt.sub, extra_fields={site: modified})
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
    state — Schwen's internalised insulin-receptor complexes dissociating at the surface rate. A
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
    pair of sites. Members may repeat (Lucarelli's SMAD2 with two SMAD3), which INDRA's
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
    receptor freed: a ligand in excess, Isensee's cAMP and its analogues occupying RII."""
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
    the substrate writes as unmodified — Lucarelli's SMAD2/3 C-terminal SxS motif, which single-site
    steps cannot take from S to ppS without an intermediate the reference does not have."""
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
    row forms around it, which then falls apart into its free members: Lucarelli's phospho-SMAD
    trimers, dephosphorylated in the nucleus and released. Only complexes earlier rows have built
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
#: and the floor the library's own entries are measured against.
INDRA_DEFAULT = "indra_default"


def _indra(name: str) -> Policy:
    """INDRA's own ``name`` policy for a statement; what it added is read off the model, before
    against after. Its parameters are renamed ``<prefix>_<kind>`` (``kf``, ``kr``, ``Km``...):
    INDRA's own names (``kf_ar_phosphorylation_1``) say neither which row minted them nor which
    statement, and an agent widening a bound on Fujita had to look each one up."""
    entry = INDRA_DEFAULT if name == "default" else f"indra_{name}"

    def apply(stmt: Statement, model: pysb.Model, prefix: str) -> Emission:
        before = {c.name for c in model.all_components()}
        indra = PysbAssembler()
        indra.processed_policies = {stmt.uuid: IndraPolicy(name)}
        try:
            indra._dispatch(stmt, "assemble", model, BaseAgentSet(), {})
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
    # interactions_only, atp_dependent. Only the pairs INDRA defines: its dispatch falls back to
    # the default without a word when a policy is missing, which would build one model under
    # another's name.
    for _name in sorted(_INDRA_ASSEMBLY):
        _kind, _policy = _name.split("_assemble_")
        if _kind == _dispatch_name(_cls) and _policy not in ("default", "one_step"):
            mechanism(_cls, f"indra_{_policy}")(_indra(_policy))
