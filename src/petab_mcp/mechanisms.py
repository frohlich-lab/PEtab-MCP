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

import re
from collections.abc import Callable
from dataclasses import dataclass

import pysb
import sympy
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
    Agent,
    RemoveModification,
    Statement,
    get_all_descendants,
    get_statement_by_name,
)
from pysb.core import time

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
) -> Emission:
    """Apply one entry of ``library`` (the registered entries plus any the agent wrote). Raises
    :class:`MechanismError` when the pair is not in it — the agent's signal to write one."""
    key = (type(stmt).__name__, policy)
    if key not in library:
        known = sorted(p for c, p in library if c == key[0])
        raise MechanismError(
            f"no mechanism {key}; the library has {known or 'none'} for {key[0]}. If none fits, "
            "write one with `mechanisms`.",
            cause="library_gap",
        )
    return library[key](stmt, model, prefix)


#: What a written rate may contain besides names: numbers, arithmetic, parentheses, commas. A
#: rate is parsed by sympy, which evaluates text, so it is checked against this first and every
#: name in it against the ones the mechanism allows.
_RATE_TOKENS = re.compile(r"\s+|(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?|[A-Za-z_]\w*|[-+*/^(),]")
_RATE_FUNCTIONS = {"exp": sympy.exp, "log": sympy.log, "sqrt": sympy.sqrt}


def written(definition: dict, library: dict[tuple[str, str], Policy]) -> Policy:
    """Compile a mechanism the agent wrote: the reaction of an existing entry (``based_on``) with
    a new ``rate``.

    Only the rate is the agent's. The reaction — what is consumed, what is produced — stays the
    base entry's, and the rate is a COEFFICIENT that mass action multiplies by every reactant's
    amount, catalysts included, so a rate that stays non-negative can never drive an amount
    negative (``kcat / (Km + sub)`` is Michaelis-Menten). The rate names the statement type's roles
    (``enz``, ``sub``, ``source``...), each the amount of that agent as the statement writes it;
    the definition's ``parameters``, each minted as ``<prefix>_<name>``; ``time``; exp, log, sqrt.
    """
    stmt_type, policy = definition["statement_type"], definition["policy"]
    base_key = (stmt_type, definition.get("based_on", INDRA_DEFAULT))
    if (stmt_type, policy) in REGISTRY:
        raise MechanismError(f"{(stmt_type, policy)} is a built-in mechanism; name yours apart")
    if base_key not in library:
        raise MechanismError(f"no mechanism {base_key} to base {policy!r} on")
    roles = list(get_statement_by_name(stmt_type)._agent_order)
    names = set(definition.get("parameters") or [])
    rate = definition["rate"]
    tokens = _RATE_TOKENS.findall(rate)
    unknown = sorted(
        {t for t in tokens if t[:1].isalpha() or t[:1] == "_"}
        - names
        - set(roles)
        - set(_RATE_FUNCTIONS)
        - {"time"}
    )
    if "".join(tokens) != rate or unknown:
        raise MechanismError(
            f"rate {rate!r} of {policy!r} may name only {sorted(names) + roles + ['time']} and "
            f"exp/log/sqrt{f'; it names {unknown}' if unknown else ''}"
        )
    base = library[base_key]

    def policy_fn(stmt: Statement, model: pysb.Model, prefix: str) -> Emission:
        # The base's own parameters go unused once re-rated; kept apart from the written ones.
        emitted = base(stmt, model, f"{prefix}_base")
        if len(emitted.rules) != 1 or model.rules[emitted.rules[0]].is_reversible:
            raise MechanismError(
                f"{base_key} builds {len(emitted.rules)} reaction(s) for this statement; a written "
                "mechanism re-rates exactly one irreversible reaction"
            )
        params = {n: pysb.Parameter(f"{prefix}_{n}", 1.0, _export=False) for n in sorted(names)}
        amounts = {
            role: pysb.Observable(
                f"{prefix}_{role}", get_monomer_pattern(model, getattr(stmt, role)), _export=False
            )
            for role in roles
            if isinstance(getattr(stmt, role, None), Agent)
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

    return policy_fn


def assemble(
    rows: list[tuple[Statement, str, str]],
    entities: dict[str, Agent] | None = None,
    library: dict[tuple[str, str], Policy] = REGISTRY,
) -> tuple[pysb.Model, list[Emission]]:
    """Build the whole model from ``(statement, policy, prefix)`` rows.

    ``entities`` (species id to ``Agent``, from the problem's entity map) joins the monomer
    pass, so a species the problem observes has its monomer and sites even when no row names it.
    """
    stmts = [stmt for stmt, _, _ in rows]
    # INDRA's monomer pass collects every site its own statement types imply; it skips the
    # vocabulary in :mod:`statements`, whose agents carry only the sites their own conditions
    # name, so those are added directly.
    indra = PysbAssembler(stmts)
    indra.processed_policies = indra.process_policies(None)
    indra.agent_set = agents = BaseAgentSet()
    indra._monomers()
    for stmt in stmts:
        for agent in stmt.real_agent_list():
            agents.get_create_base_agent(agent)
    for agent in (entities or {}).values():
        agents.get_create_base_agent(agent)

    model = pysb.Model(name="statements", _export=False)
    for base in agents.agents.values():
        monomer = pysb.Monomer(base.name, base.sites, base.site_states, _export=False)
        model.add_component(monomer)
        for db_name, db_ref in base.db_refs.items():
            annotation = get_annotation(monomer, db_name, db_ref)
            if annotation is not None:
                model.add_annotation(annotation)
    emissions = [
        instantiate(stmt, model, policy, prefix, library) for stmt, policy, prefix in rows
    ]
    _refuse_ruleless(rows, emissions)
    return model, emissions


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


@mechanism(PopulationTransition, "mass_action")
def _transition(stmt: Statement, model: pysb.Model, prefix: str) -> Emission:
    """First-order transfer, ``k·source``."""
    k = pysb.Parameter(f"{prefix}_k", 1.0, _export=False)
    source = get_monomer_pattern(model, stmt.source)
    target = get_monomer_pattern(model, stmt.target)
    rule = pysb.Rule(f"{prefix}_transition", source >> target, k, _export=False)
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


for _cls in get_all_descendants(RemoveModification):
    mechanism(_cls, BASAL)(_basal_removal)


#: The policy name every INDRA statement type answers to: INDRA's own default assembly, so every
#: statement INDRA can read has at least one entry. It is what an INDRA warm start is built with,
#: and the floor the library's own entries are measured against.
INDRA_DEFAULT = "indra_default"


def _indra_default(stmt: Statement, model: pysb.Model, prefix: str) -> Emission:
    """INDRA's default policy for ``stmt``; what it added is read off the model, before against
    after. Its parameters are renamed ``<prefix>_<kind>`` (``kf``, ``kr``, ``Km``...): INDRA's own
    names (``kf_ar_phosphorylation_1``) say neither which row minted them nor which statement, and
    an agent widening a bound on Fujita had to look each one up."""
    before = {c.name for c in model.all_components()}
    indra = PysbAssembler()
    indra.processed_policies = {stmt.uuid: IndraPolicy("default")}
    try:
        indra._dispatch(stmt, "assemble", model, BaseAgentSet(), {})
    except UnknownPolicyError as exc:
        raise MechanismError(
            f"INDRA has no default assembly for {type(stmt).__name__}", cause="library_gap"
        ) from exc
    added = [c for c in model.all_components() if c.name not in before]
    for parameter in (c for c in added if isinstance(c, pysb.Parameter)):
        name = f"{prefix}_{parameter.name.split('_')[0]}"
        # INDRA builds with PySB's global export off, which `rename` still tries to update.
        parameter._export = False
        parameter.rename(
            name if name not in model.parameters.keys() else f"{name}_{parameter.name}"
        )
    return Emission(
        (type(stmt).__name__, INDRA_DEFAULT),
        prefix,
        {role: _role_names(getattr(stmt, role)) for role in stmt._agent_order},
        rules=tuple(c.name for c in added if isinstance(c, pysb.Rule)),
        parameters=tuple(c.name for c in added if isinstance(c, pysb.Parameter)),
    )


def _role_names(value) -> str:
    if isinstance(value, list):
        return ",".join(a.name for a in value)
    return "" if value is None else value.name


for _cls in {c for base in statement_whitelist for c in (base, *get_all_descendants(base))}:
    mechanism(_cls, INDRA_DEFAULT)(_indra_default)
