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
from collections import Counter
from dataclasses import dataclass
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
    Conversion,
    DecreaseAmount,
    IncreaseAmount,
    Modification,
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
    empty when the policy exists, and the pair is a library gap. Counted as gaps, misspellings
    (``indra_defualt``) filled the library's to-do list."""
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
) -> Emission:
    """Apply one entry of ``library`` (the registered entries plus any the agent wrote). Raises
    :class:`MechanismError` when the pair is not in it — the agent's signal to write one."""
    key = (type(stmt).__name__, policy)
    if key not in library:
        known = sorted(p for c, p in library if c == key[0])
        typo = _misspelt(key, known, library)
        raise MechanismError(
            f"no mechanism {key}{typo}; the library has {known or 'none'} for {key[0]}. If none fits, "
            "write one with `mechanisms`.",
            cause="malformed_mechanism" if typo else "library_gap",
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

    policy_fn.indra_policy = getattr(base, "indra_policy", "default")
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
    indra.processed_policies = {
        stmt.uuid: IndraPolicy(_monomer_policy(policy, library)) for stmt, policy, _ in rows
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
