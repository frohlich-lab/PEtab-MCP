"""The model a set of statements builds, as SBML this PEtab problem can score.

INDRA turns statements into a PySB model (:mod:`mechanisms`) and PySB writes SBML in one call.
Neither knows anything about the PEtab problem the model is for, and the SBML they produce cannot
be scored against it as it stands. This module is the difference, and it is all problem-specific:

* **Readouts.** PySB numbers every species (``__s0``, ``__s1`` ...), while the fixed observable
  table reads names: ``A_p``, or a ``readout_<observableId>`` mapping. Each species the entity
  map names is given that name, or for a ``readout_*`` mapping an assignment rule of that name.
* **Seeds.** What is present at t = 0 is traced (below); each present monomer is seeded
  at an estimated ``<monomer>_init``, or at the row the problem ships (a fixed
  ``<monomer>_init`` row) — unless the statements DECLARE what is there (``seeds``, below).
* **Inputs.** A species the experiment sets is held rather than built: one the condition table
  targets (a drug dose), which its periods set, and one the problem's ``inputs.tsv`` holds to a
  protocol (a stimulus ramp, :func:`protocols.hold`).
* **Leftovers.** A rate constant an agent-written mechanism replaced is removed.

What is present at t = 0 rests on two assumptions, and a third check refuses the models they do
not fit:

1. **The seeds are free monomers in INDRA's default state** (``set_base_initial_condition``):
   unbound, unmodified, inactive, wild type, in the cytoplasm when that is a location (else in
   the first of its locations by name), as a rule makes a new molecule
   (:func:`mechanisms.default_state`). No complex and no modified form exists at t = 0.
2. **What the seeds make is absent at t = 0.** Every monomer no rule makes from other kinds is
   present (one that only proliferates included), and BioNetGen expands the rules from those. A
   made monomer that expansion reaches starts empty: a population that exists only once another
   converts into it. One it does not reach traces back to nothing at t = 0 (populations that
   only interconvert), so it is present too. A protein synthesised at a steady rate therefore
   starts at zero, not at its steady state; nothing here tells the two apart.
3. **Every rule fires, and every readout reads a species,** in the expansion from those seeds. A
   rule that needs a complex or modified form nothing forms (a kinase active only while bound to a
   partner no statement binds it to) is refused by its row, rather than built into a model whose
   fit can never move it. So is an amount a rate reads that nothing forms (``amount(ERK{pY187})``
   where no row phosphorylates Y187), which is 0 throughout.

This is MARM2's causal tracing (Fröhlich et al. 2023, MSB 19:e10988) asked statically. There, a
synthetic site set by the rule that makes a molecule, and carried by BioNetGen through every
later step, splits each simulated species by origin. Here the only question is whether an origin
exists, whether a monomer traces back to the seeds at all, and the expansion's reactions answer
it without a site.

A declared seed (``make_changes``' ``seeds``) replaces the trace for the monomers it names. The
species its Agent writes starts at that amount, in exactly the state the Agent writes (mods,
location, activity, and the complex its ``bound_conditions`` form, partners in their default
state), and no monomer named in any declaration is seeded any other way: declare each state it
starts in. A number other than 0 becomes ``<name>_init`` and reconcile treats it as any other
starting value (1 a unit normalisation, fixed; anything else estimated from there), so a unit pool
costs nothing in k. A 0 is an empty seed with no parameter at all — what lets a species the
condition table sets (a substrate the conditions dose) carry no ``_init`` that every condition
overrides. An expression is the starting amount itself, over parameter ids: a row the problem
ships (a fixed total), a condition-table column (``init_X``), or a new name, estimated. It is
evaluated once, so it takes an experiment's first period's value; a seed at a condition target's
id alone instead makes its agent that input, as an entity-map entry would, which every period
sets. A declared monomer may also be one the rules make (e.g. a gene copy that starts at 1).

It is a separate module because none of it is a mechanism: the same library rows would need
none of this under a server that read PySB directly. This one reads SBML everywhere — reconcile,
lint, de-duplication, versions — so a model built from statements enters as SBML and is from then
on an ordinary authored model; the hop costs a BioNetGen call or two per edit, and nothing
downstream has to know where a model came from.
"""

from __future__ import annotations

from copy import deepcopy

import libsbml
import pysb
from indra.assemblers.pysb.assembler import _n, get_monomer_pattern
from indra.statements import Agent
from petab.v1.models.sbml_model import SbmlModel
from pysb import ANY
from pysb.bng import NoInitialConditionsError, generate_equations
from pysb.core import time
from pysb.export import export

from . import mechanisms as M
from . import reduction, regulation
from .errors import MechanismError, ModelUpdateError
from .protocols import Input, hold
from .reconcile import SEED_INITIAL_VALUE


def to_antimony(
    rows: list[tuple],
    entities: dict[str, Agent],
    library: dict[tuple[str, str], M.Policy] = M.REGISTRY,
    inputs: tuple[Input, ...] = (),
    seeds: list[tuple[str, Agent, float | str]] | None = None,
    formulas: dict[str, str] | None = None,
    amounts: dict[str, Agent] | None = None,
    notes: list[str] | None = None,
    *,
    targets: frozenset[str] = frozenset(),
    times: tuple[float, float] | None = None,
    network: dict | None = None,
    reduce: bool = True,
    seeded: bool = False,
) -> tuple[str | pysb.Model, list[M.Emission]]:
    """The model ``rows`` build, as Antimony the ``antimony`` route accepts unchanged.

    Every species in ``entities`` becomes a symbol of that name, the sum over its PySB pattern,
    because that name is what the fixed observable table reads. What is present at t = 0 is traced
    as the module docstring sets out, and :class:`ModelUpdateError` names the rows a model breaking
    it fails on.

    An experiment's INPUT is held instead, with no starting amount of its own: a species of
    ``entities`` among ``targets`` (the problem's condition targets) that no row makes or
    converts, which its periods set, and each of ``inputs`` (the problem's :mod:`protocols`),
    under its protocol. An input the entity map does not name is the agent of its id. A
    condition target the rows make or convert is a state its periods only start.

    ``times``, the problem's (:func:`regulation.times`), are what a rate's ``time`` ranges over
    where its regulators are checked.

    ``seeds`` (``(name, agent, amount)``: a number or an expression over parameter ids) declare
    what is present at t = 0, as the module docstring sets out; each is ``<name>_init``, so
    no two share a name (:func:`operations.seed_name`). One whose amount is a condition target's id
    alone makes its agent that input instead, as if the entity map named it so.

    ``formulas`` (readout id to an expression; ``make_changes``' ``readouts``) define each readout that is not one pattern's amount as an
    assignment rule over ``amounts`` (name to the agent whose amount it is), the entity map's ids
    — each a symbol of the model by then — and parameter ids, a name the model does not hold
    becoming a parameter as in a declared seed.

    ``notes``, when given, collects what t = 0 holds that nothing declared and the caller may
    not expect: kinds the rules make from nothing, which start empty, and a seed's location chosen
    among several; and each catalyst taken in every state (:func:`_every_state`).

    The rules are reduced before they are expanded (:mod:`reduction`), unless ``reduce`` is off
    (the expansion the tests compare it with), and ``network``, when given, receives the
    species and reactions with and without the reduction. ``seeded`` returns the PySB model
    itself, as the rows build it and seeded, before any reduction or expansion (what
    :mod:`tracing` traces), with the emissions.
    """
    seeds = [(name, agent, _numeric(amount)) for name, agent, amount in seeds or []]
    entities = {**{item.id: Agent(item.id) for item in inputs}, **entities}
    # A seed at a condition target's id alone makes its agent that input, as an entity-map entry
    # does: the species takes the id, and it starts empty where no period sets it.
    mapped: dict[str, Agent] = {}
    by_seed: dict[str, str] = {}  # the seed's name to the target it makes its agent
    for name, agent, amount in seeds:
        target = amount.strip() if isinstance(amount, str) else None
        if target in targets:
            if other := entities.get(target) or mapped.get(target):
                raise ModelUpdateError(
                    f"seed {name!r} makes {agent.name} the input `{target}`, which is already "
                    f"{other.name}; an input is one agent"
                )
            mapped[target], by_seed[name] = agent, target
    seeds = [(n, a, 0.0 if n in by_seed else x) for n, a, x in seeds]
    entities = {**entities, **mapped}
    clash = sorted(set(amounts or {}) & set(entities))
    if clash:
        raise ModelUpdateError(f"amounts {clash} are also entity-map or input ids; name them apart")
    # An amount a readout formula reads is named like an entity-map readout (below), for this
    # model only: the map itself is the problem's, and amounts are the agent's.
    entities = {**entities, **(amounts or {})}
    # Seed agents join the monomer pass (they may name a site or partner no row does); they are
    # not readouts, so they stay out of `entities` below.
    model, emissions = M.assemble(
        rows,
        {**entities, **{f"__seed_{name}": agent for name, agent, _ in seeds}},
        library,
        targets - entities.keys(),
    )
    _refuse_chains(model, emissions)
    made, synthesised, converted = set(), set(), set()
    for rule in model.rules:
        reactants = _monomer_names(rule.reactant_pattern)
        products = _monomer_names(rule.product_pattern)
        made |= products - reactants
        if reactants <= products:  # consumes nothing it does not give back
            synthesised |= products - reactants
        converted |= reactants - products
        if rule.is_reversible:
            made |= reactants - products
            converted |= products - reactants
    # A species the conditions set is an input when the rows only read it (a drug that binds, a
    # hormone that activates). One they make or convert is a state the conditions only start:
    # held, such a species would never change.
    protocol = {item.id: item for item in inputs}
    # Each by the name PySB gives its monomer (INDRA's `_n`: `S-1` is `S_1`), as `made` and the
    # model know it.
    set_keys = {s: _n(entities[s].name) for s in entities if s in protocol or s in targets}
    keys = {s for s in set_keys if s in protocol or set_keys[s] not in made | converted}
    held = {set_keys[s]: protocol.get(s) for s in keys}
    # Readouts are named apart from the species they stand for, which may share a monomer's
    # name; the SBML export numbers observables anyway, and the rename below restores them.
    readouts = {f"readout_{species}": species for species in entities}
    patterns = {name: get_monomer_pattern(model, entities[s]) for name, s in readouts.items()}
    declared = _declare(model, seeds, set(held), set(by_seed))
    _seed(
        model,
        [m for m in model.monomers if (m.name not in made or m.name in held) and m.name not in declared],
    )
    if seeded:
        return model, emissions
    # Read before the reduction rewrites the rules: what a row that never fires is told.
    sites = _site_states(model, emissions)
    # What the experiment sets stays whole (:mod:`reduction`): a form the conditions or a protocol
    # set.
    whole = set(set_keys.values())
    if notes is not None:
        notes.extend(_every_state(rows, model, emissions, whole))
    # A row's regulators are checked on its reactions as BioNetGen expands the rules as written
    # (:mod:`regulation`): reduced, a catalyst is out of its rule and a monomer is blocks, no
    # longer the statement's agents.
    # `time` kept itself: BioNetGen's export knows it by identity, and sympy's cache, once full,
    # copies it.
    as_written = (
        deepcopy(model, {id(time): time})
        if reduce and any(row[4:5] and row[4] for row in rows) else None
    )
    if as_written is not None:
        # The copy builds each expression again, evaluated, over parameters sympy takes as
        # positive from their seeded values: a gate's `ko > 0` became True. Each is the original
        # over the copy's components, every comparison as written.
        components = {c.name: c for c in as_written.components}
        for expression in as_written.expressions:
            expression.expr = M.over(model.expressions[expression.name].expr, components)
    reduced = reduction.reduce(model, whole) if reduce else reduction.Reduction()
    # A readout over blocks of a split monomer is their product, so it is made after the split.
    amounts = {
        name: reduction.amount(model, reduced, pattern, name) for name, pattern in patterns.items()
    }
    _expand(model)
    present = reduction.reached(model, reduced)
    reached = {mp.monomer.name for i in present for mp in model.species[i].monomer_patterns}
    unreached = [
        m for m in model.monomers if m.name in made - held.keys() - reached - declared
    ]
    if unreached:
        _seed(model, unreached)
        model.reset_equations()
        _expand(model)
        present = reduction.reached(model, reduced)
    if notes is not None:
        set_by = sorted(s for s in entities if s in targets)
        kept = [s for s in set_by if s in keys]
        started = [s for s in set_by if s not in keys]
        if set_by:
            notes.append(
                " ".join(
                    part
                    for part in (
                        kept
                        and f"Held at what the conditions set, as your rows only read them: "
                        f"{', '.join(kept)}.",
                        started
                        and f"Started by the conditions, then changed by your rows, which make or "
                        f"convert each: {', '.join(started)}.",
                    )
                    if part
                )
            )
        empty = sorted((synthesised & reached) - held.keys() - declared)
        if empty:
            notes.append(
                f"These start at 0, because rules make them from nothing and no `seeds` entry "
                f"declares them: {', '.join(empty)}. Declare each one present at t=0 (a protein "
                "made at a steady rate is) in `seeds`; one declared at 0 stays empty and leaves "
                "this list."
            )
        chosen = []
        # A split monomer starts as its blocks, and the block holding `loc` speaks for it.
        whole_of = {p.name: m for m, split in reduced.splits.items() for p in split.blocks}
        for initial in model.initials:
            pattern = initial.pattern.monomer_patterns[0]
            name = whole_of.get(pattern.monomer.name, pattern.monomer.name)
            locations = pattern.monomer.site_states.get("loc", ())
            if name in declared.union(held) or len(locations) < 2:
                continue
            if "cytoplasm" in locations:
                continue
            chosen.append(f"{name} in {pattern.site_conditions['loc']} of {sorted(locations)}")
        if chosen:
            notes.append(
                f"These start in the first of their locations by name, since no statement puts "
                f"them in the cytoplasm: {'; '.join(chosen)}. To seed one elsewhere, declare it "
                "in `seeds`."
            )
    fired = reduction.fired(model, reduced, present)
    never = []
    stuck = sorted({e.prefix for e in emissions if not fired.issuperset(e.rules)})
    misnamed = {
        e.prefix: hint
        for e in emissions
        if e.prefix in stuck
        and (hint := _misnamed(e.prefix, [r for r in e.rules if r not in fired], *sites))
    }
    if stuck:
        # Named with the form each needs, which nothing forms: "never fire" alone does not say
        # what to add (a translocation row whose cargo must first be modified).
        needs = {}
        for e in emissions:
            for rule in e.rules if e.prefix in stuck else ():
                if rule not in fired and (need := reduction.unformed(model, reduced, rule, present)):
                    needs[e.prefix] = need
                    break
        never.append(
            f"rows {stuck} build rules that never fire"
            + (f" (nothing forms {'; '.join(f'{p}: {n}' for p, n in needs.items())})" if needs else "")
        )
    unread = sorted(
        sp for name, sp in readouts.items() if not reduction.reads(amounts[name], present)
    )
    if unread:
        unformed = "; ".join(f"{sp}: {patterns[f'readout_{sp}']}" for sp in unread)
        never.append(f"readouts {unread} read no species ({unformed})")
    if never:
        generic = (
            "At t = 0 the model holds free, unmodified molecules and what `seeds` declares, and a "
            "rule fires only on what the statements form from there: one that never fires needs a "
            "complex or a modified form that nothing forms, like a kinase active only while bound "
            "to a partner nothing binds it to. Add the statement that forms it, or, if it is "
            "present at t=0, declare it in `seeds`."
        )
        explained = misnamed.keys() == set(stuck) and not unread
        raise ModelUpdateError(
            " ".join(
                [
                    f"In the model these statements build, {' and '.join(never)}.",
                    *misnamed.values(),
                    *([] if explained else [generic]),
                ]
            )
        )
    for e in emissions:
        for name, written in e.amounts.items():
            observable = model.observables[name]
            if not reduction.reads(reduced.amounts.get(observable, observable), present):
                raise MechanismError(_unformed(e, observable, written))
    if network is not None:
        network.update(reduction.sizes(model, reduced, present))
    if as_written is not None:
        if unreached:
            _seed(as_written, [as_written.monomers[m.name] for m in unreached
                               if m.name in as_written.monomers.keys()])
        _expand(as_written)
    emissions = regulation.record(as_written or model, rows, emissions, times)
    sbml = export(model, "sbml")  # from the expansion above; BioNetGen does not run again
    base_names = {
        str(initial.pattern): initial.pattern.monomer_patterns[0].monomer.name
        for initial in model.initials
    }
    document = libsbml.readSBMLFromString(sbml)
    sbml_model = document.getModel()
    for reaction in sbml_model.getListOfReactions():
        # PySB numbers reactions `r0`, `r1`... whatever else the model names so; they hold
        # PySB's internal form until :func:`_number_reactions` numbers them against the rest.
        reaction.setId(f"__{reaction.getId()}")
        # PySB lists every symbol a rate expression reads that is no parameter as a modifier
        # species, `time` too, which no species is: the SBML did not read.
        while reaction.removeModifier("time") is not None:
            pass
    species_named = {sp.getName(): sp for sp in sbml_model.getListOfSpecies()}
    base = {base_names[n]: sp for n, sp in species_named.items() if n in base_names}
    # An input a seed maps is the state that seed writes, whatever other states of it start.
    base |= {
        set_keys[target]: species_named[str(initial.pattern)]
        for initial in model.initials
        if (target := by_seed.get(initial.value.name.removesuffix("_init")))
    }
    for assignment in list(sbml_model.getListOfInitialAssignments()):
        amount = assignment.getMath().getName()
        if amount.removesuffix("_init") in held:
            sbml_model.getSpecies(assignment.getSymbol()).setInitialAmount(0.0)
            sbml_model.removeParameter(amount)
            assignment.removeFromParentAndDelete()
    named = set()
    blocks = {part.name for split in reduced.splits.values() for part in split.blocks}
    # What the conditions set names its species first, so a readout of that species is the rule
    # over it: a condition target left a rule over another readout's name could not be set.
    for parameter in sorted(
        (p for p in sbml_model.getListOfParameters() if p.getName() in readouts),
        key=lambda p: readouts[p.getName()] not in set_keys,
    ):
        old, new = parameter.getId(), readouts[parameter.getName()]
        rule = sbml_model.getAssignmentRuleByVariable(old)
        # A species the conditions or a protocol set is its free form, held or only started: named
        # for the sum over the complexes it binds into, a condition target that binds into
        # complexes would set nothing.
        free = _free_form(model, sbml_model, base, set_keys[new]) if new in set_keys else None
        species = free or sbml_model.getSpecies(rule.getMath().getName() or "")
        if (
            (free is not None or rule.getMath().isName())
            and species is not None
            and species.getId() not in named
            and not new.startswith("readout_")
            # Nor a block of a split monomer (:mod:`reduction`), which is one species where the
            # monomer whole is several: named, it took a shipped `<id>_init` for its seed, which
            # the unreduced model never gives it (the #116 review).
            and species.getName().split("(", 1)[0] not in blocks
        ):
            # A readout that IS one species names that species, rather than leaving a rule over
            # it: the reachability checks in `reconcile` and `operations` follow reactions, not
            # rules, and warned that every species was disconnected from the data. Except a
            # `readout_*` mapping, which `validation`'s measurement-mapping check requires to
            # stay an assignment rule.
            # A species takes one name: each further readout of it stays a rule over that name.
            old, renamed = species.getId(), species
            named.add(new)
            rule.removeFromParentAndDelete()
            parameter.removeFromParentAndDelete()
        else:
            rule.setVariable(new)
            renamed = parameter
        _rename(sbml_model, renamed, new)
    agents = {m.name for m in model.monomers}
    for readout, formula in (formulas or {}).items():
        _readout_formula(sbml_model, readout, formula, agents, set(entities))
    for name, _, amount in seeds:
        if isinstance(amount, str):
            _initial_expression(sbml_model, f"{name}_init", name, amount)
        elif float(amount) == 0.0:
            _initial_empty(sbml_model, f"{name}_init")
    _drop_unreferenced_parameters(sbml_model)
    for monomer, item in held.items():
        if item is None:
            # A species may sit in reactions and be set by a condition, unchanged by them, only
            # on the boundary.
            base[monomer].setBoundaryCondition(True)
        else:
            hold(sbml_model, base[monomer].getId(), item)
    _number_reactions(sbml_model)
    return SbmlModel.from_string(libsbml.writeSBMLToString(document)).to_antimony(), emissions


def _numeric(amount):
    """A seed amount written as a number in text (``"2"``) is that number, as one written as a
    number. Taken as an expression, its Antimony round trip made it a literal that reconcile
    estimated once per species (the #116 review), and a ``"0"`` was no empty seed."""
    if isinstance(amount, str):
        try:
            return float(amount)
        except ValueError:
            return amount
    return amount


def _declare(
    model: pysb.Model,
    seeds: list[tuple[str, Agent, float | str]],
    held: set[str],
    inputs: set[str] = frozenset(),
) -> set[str]:
    """Seed each declared species at ``<name>_init``, in the state its Agent writes; returns the
    monomers the declarations name, which :func:`_seed` then leaves alone. A ``held`` monomer
    takes no seed but the one of ``inputs`` (seed names) that makes it its input."""
    named: set[str] = set()
    for name, agent, amount in seeds:
        if _n(agent.name) in held and name not in inputs:
            raise ModelUpdateError(
                f"seed {name!r}: {agent.name} is an input, held to its protocol, so it has no "
                "seed of its own"
            )
        if not isinstance(amount, (int, float, str)) or isinstance(amount, bool):
            raise ModelUpdateError(
                f"seed {name!r} is {amount!r}; give a number or an expression over parameter ids"
            )
        try:
            molecules = M.agent_pattern(model, agent, fresh=lambda name: True)
        except M.MechanismError as exc:
            raise ModelUpdateError(f"seed {name!r}: {exc}") from exc
        pattern = pysb.ComplexPattern(molecules, None)
        value = float(amount) if not isinstance(amount, str) else SEED_INITIAL_VALUE
        parameter = pysb.Parameter(f"{name}_init", value, _export=False)
        model.add_component(parameter)
        twin = next(
            (i.value.name.removesuffix("_init") for i in model.initials
             if i.pattern.is_equivalent_to(pattern)),
            None,
        )
        if twin is not None:
            raise ModelUpdateError(
                f"seeds {twin!r} and {name!r} both seed {pattern} (a site neither writes is in its "
                "default state), and one species takes one seed: give it one, or write "
                "each agent in the state that tells them apart"
            )
        try:
            model.add_initial(pysb.Initial(pattern, parameter, _export=False))
        except ValueError as exc:  # pysb: not concrete
            raise ModelUpdateError(
                f"seed {name!r} is not a seed PySB accepts: {exc}"
            ) from exc
        named |= {mp.monomer.name for mp in pattern.monomer_patterns}
    return named


def _parse(formula: str):
    """``formula`` as libSBML's L3 parser reads it, Python's ``**`` read as its ``^``. The parser
    has no ``**``, and its ``^`` binds and associates as ``**`` does (``-a^2`` is ``-(a^2)``,
    ``a^b^c`` is ``a^(b^c)``), so the formula means what it says in Python."""
    return libsbml.parseL3Formula(formula.replace("**", "^"))


def _initial_expression(sbml_model, placeholder: str, species: str, formula: str) -> None:
    """Replace the declared species' ``<key>_init`` with the expression its entry gives. A name
    the model does not hold becomes a parameter: a shipped row or a condition-table column fills
    it, and anything else is estimated from reconcile's seed."""
    math = _parse(formula)
    if math is None:
        raise ModelUpdateError(f"seed {species!r} is no formula: {formula!r}")
    _parameters_for_unknown_names(sbml_model, math)
    for assignment in sbml_model.getListOfInitialAssignments():
        if assignment.getMath().isName() and assignment.getMath().getName() == placeholder:
            assignment.setMath(math)


def _readout_formula(
    sbml_model, readout: str, formula: str, agents=frozenset(), states=None
) -> None:
    """Define ``readout`` as ``formula``: an assignment rule, as the lint wants a measurement
    mapping to be, over symbols of the model and parameters made for the names it lacks.

    A name that is one of the model's ``agents`` is refused rather than made a parameter: a
    formula reads an agent's amount only through ``amounts``, and a parameter of the agent's name
    would read a constant, fitted silently in place of the species (``readout_X := A`` would
    measure an estimated constant ``A``, not the species A).

    A formula that reads none of the model's ``states`` (the ``amounts``, and the entity map's ids)
    is refused too: it measures parameters, a constant the fit tunes to the data's mean, with a
    flat trajectory that can look like a good fit. A number is not refused: nothing fits it, and
    :mod:`literals` prices it unless it is structural. It is the claim that an assay reads the same
    whatever the state, which Lucarelli's reference makes of the SMAD3 in a SMAD2 pull-down (all of
    it doubly phosphorylated, ``readout_S2IP_ppS3 := 1``; PEtabAgentBench#23)."""
    if sbml_model.getElementBySId(readout) is not None:
        raise ModelUpdateError(
            f"readout {readout!r} is a formula and also names something else in the model"
        )
    math = _parse(formula)
    if math is None:
        raise ModelUpdateError(f"readout {readout!r} is no formula: {formula!r}")
    if any(node.isName() and node.getName() == readout for node in _ast_nodes(math)):
        raise ModelUpdateError(
            f"readout {readout!r} = {formula!r} reads itself. A readout is not a parameter to "
            "scale: it is what the assay measures of the model's state, a formula over the "
            "amounts it reads."
        )
    unread = sorted(
        {
            node.getName()
            for node in _ast_nodes(math)
            if node.getType() == libsbml.AST_NAME
            and node.getName() in agents
            and sbml_model.getElementBySId(node.getName()) is None
        }
    )
    if unread:
        example = ", ".join(f'"{n}": "{n}"' for n in unread)
        raise ModelUpdateError(
            f"readout {readout!r} reads {unread}, agent(s) of this model, by name. A readout "
            "formula reads an agent's amount only through `amounts`: declare it there "
            f"(`amounts={{{example}}}` for every state of it, or a pattern such as "
            f'`"{unread[0]}{{p}}"` for one state) and read that name. Unread, it would be a '
            "parameter, a constant, fitted in place of the species."
        )
    named = {node.getName() for node in _ast_nodes(math) if node.isName()}
    if states is not None and named and not named & set(states):
        raise ModelUpdateError(
            f"readout {readout!r} = {formula!r} reads no amount, so it is a constant the fit tunes "
            "to the data. A readout is what the assay measures of the model's state: read an "
            "amount, declared in `amounts` or written in place as `amount(<pattern>)`"
            f"{f' (the amounts are {sorted(states)})' if states else ''}."
        )
    _parameters_for_unknown_names(sbml_model, math)
    parameter = sbml_model.createParameter()
    parameter.setId(readout)
    parameter.setConstant(False)
    rule = sbml_model.createAssignmentRule()
    rule.setVariable(readout)
    rule.setMath(math)


def _parameters_for_unknown_names(sbml_model, math) -> None:
    """A parameter for every name in ``math`` the model does not hold: a shipped row or a
    condition-table column fills it, and anything else is estimated from reconcile's seed."""
    for node in _ast_nodes(math):
        if node.getType() == libsbml.AST_NAME and sbml_model.getElementBySId(node.getName()) is None:
            parameter = sbml_model.createParameter()
            parameter.setId(node.getName())
            parameter.setConstant(True)
            parameter.setValue(SEED_INITIAL_VALUE)


def _initial_empty(sbml_model, placeholder: str) -> None:
    """Seed the declared species at a literal 0: no assignment, so no parameter for reconcile to
    estimate or a condition to override."""
    for assignment in list(sbml_model.getListOfInitialAssignments()):
        if assignment.getMath().isName() and assignment.getMath().getName() == placeholder:
            sbml_model.getSpecies(assignment.getSymbol()).setInitialAmount(0.0)
            assignment.removeFromParentAndDelete()


def _free_form(model: pysb.Model, sbml_model, base: dict, monomer: str):
    """The SBML species that is ``monomer`` alone and bound to nothing: its seed, or, for one only
    the rules make (a metabolite made from a precursor), the one such species the expansion
    reached; else ``None``."""
    if monomer in base:
        return base[monomer]
    names = {
        str(cp)
        for cp in model.species
        if len(cp.monomer_patterns) == 1
        and cp.monomer_patterns[0].monomer.name == monomer
        and not any(
            isinstance(v, int) or (isinstance(v, tuple) and isinstance(v[1], int))
            for v in cp.monomer_patterns[0].site_conditions.values()
        )
    }
    found = [sp for sp in sbml_model.getListOfSpecies() if sp.getName() in names]
    return found[0] if len(found) == 1 else None

def _seed(model: pysb.Model, monomers: list[pysb.Monomer]) -> None:
    """Seed each of ``monomers`` present, as assumption 1 of the module docstring sets out."""
    for monomer in monomers:
        # Not INDRA's `set_base_initial_condition` itself, whose state this is but for the location
        # (:func:`mechanisms.default_state`): its `<monomer>_0` silently skips a monomer whose name
        # is taken, and a condition column may already be named `<monomer>_0`.
        state = M.default_state(monomer)
        # Reconcile's own seed value, not 1, which it holds FIXED as a unit normalisation: a seed
        # of 1 would have fixed every population at a unit seed.
        amount = pysb.Parameter(f"{monomer.name}_init", SEED_INITIAL_VALUE, _export=False)
        model.add_component(amount)
        model.add_initial(pysb.Initial(monomer(**state), amount, _export=False))


#: The most molecules one complex may hold. A network whose complexes reach it is taken for one
#: whose complexes grow without bound, which BioNetGen expands forever. BioNetGen builds every
#: complex up to the cap before the check can refuse, so the cap bounds the cost of finding out:
#: reactions grow about with its square for a chain, exponentially for complexes that branch.
MAX_COMPLEX = 20


def _expand(model: pysb.Model) -> None:
    """Fill in BioNetGen's expansion of ``model`` from its seeds. Left empty when nothing is seeded
    (populations that only interconvert), which PySB refuses to expand, or nothing fires, whose
    network BioNetGen refuses to write.

    Expanded with complexes capped at :data:`MAX_COMPLEX` molecules, and refused when one reaches
    it: statements that let complexes chain without end (A and B each dimerising with
    itself and with the other, every molecule then holding a free site for the next) otherwise
    keep BioNetGen expanding until the client gives up, call after call, with nothing said."""
    try:
        generate_equations(model, max_agg=MAX_COMPLEX)
    except NoInitialConditionsError:
        pass
    except Exception as exc:  # BioNetGen fails as a bare Exception or an OSError
        if "no reactions were found" not in str(exc):
            raise ModelUpdateError(
                f"Expanding the statements into reactions failed: {exc}"
            ) from exc
    grown = [sp for sp in model.species if len(sp.monomer_patterns) >= MAX_COMPLEX]
    if grown:
        kinds = sorted({mp.monomer.name for mp in grown[0].monomer_patterns})
        raise ModelUpdateError(
            f"These statements form complexes that grow without bound: complexes of {kinds} reach "
            f"{MAX_COMPLEX} molecules and would keep growing. A molecule that binds several "
            "partners has a free site for each, so bindings chain (A binds A, A binds B and B "
            "binds B make A-A-B-B-A-...). Require each partner to be free of its other partners "
            "where it binds: `A{~B} binds A{~B}`, `A{~A} binds B{~B}`, `B{~A} binds B{~A}`."
        )


def _drop_unreferenced_parameters(sbml_model) -> None:
    """Remove the parameters nothing reads or assigns: the rate constant a written mechanism
    replaced; and the ``__obs`` rules PySB exports for the amounts a written rate reads, which the
    export inlines into the rate, so nothing reads the rule."""
    while True:
        read = set()
        for element in sbml_model.getListOfAllElements():
            math = element.getMath() if hasattr(element, "getMath") else None
            if math is not None:
                read |= {n.getName() for n in _ast_nodes(math) if n.isName()}
        unread = [
            rule
            for rule in sbml_model.getListOfRules()
            if rule.getVariable().startswith("__obs") and rule.getVariable() not in read
        ]
        if not unread:
            break
        for rule in unread:
            variable = rule.getVariable()
            rule.removeFromParentAndDelete()
            sbml_model.removeParameter(variable)
    used = {rule.getVariable() for rule in sbml_model.getListOfRules()}
    for element in sbml_model.getListOfAllElements():
        math = element.getMath() if hasattr(element, "getMath") else None
        if math is not None:
            used |= {math_node.getName() for math_node in _ast_nodes(math) if math_node.isName()}
    for parameter in list(sbml_model.getListOfParameters()):
        if parameter.getId() not in used:
            parameter.removeFromParentAndDelete()


def _ast_nodes(node):
    yield node
    for i in range(node.getNumChildren()):
        yield from _ast_nodes(node.getChild(i))


def _refuse_chains(model: pysb.Model, emissions: list[M.Emission]) -> None:
    """Refuse, before BioNetGen runs, rules whose complexes can chain without end.

    A complex can grow without bound when a cycle of bonds lets each molecule on it be bound on
    one site while binding on another: A and B each dimerising with itself and with the other
    make A-A-B-B-A-... The rules say this at once, where the capped expansion
    (:data:`MAX_COMPLEX`) only finds it out after building every complex up to the cap, which can
    take minutes. The cap stays as the backstop for growth the bonds alone do not show.

    A cycle grows only through a molecule some rule BINDS while it is already bound elsewhere (a
    gain): one whose bonds all formed in one firing is as large as that rule's product. Three
    trimers each assembled from free SMADs (Lucarelli's) close a cycle of bonds that coexist in
    their products, and were refused although none can take a further partner
    (PEtabAgentBench#23); so a cycle is refused only when it passes a gain.
    """
    formed: dict[tuple, set[str]] = {}  # bond, as its two (monomer, site) ends -> rules forming it
    holds: set[tuple[str, str, str]] = set()  # (monomer, s, t): bound on s and t at once
    gains: set[tuple[str, str, str]] = set()  # ... because a rule binds t while s may be bound
    for rule in model.rules:
        sides = [(rule.reactant_pattern, rule.product_pattern)]
        if rule.is_reversible:
            sides.append((rule.product_pattern, rule.reactant_pattern))
        for before, after in sides:
            for molecule in _molecules(before) + _molecules(after):
                bound = sorted(s for s, v in molecule.site_conditions.items() if _is_bond(v))
                holds |= {(molecule.monomer.name, s, t) for s in bound for t in bound if s != t}
            new = _bonds(after)
            for bond in _bonds(before):
                if bond in new:
                    new.remove(bond)
            for bond in new:
                formed.setdefault(bond, set()).add(rule.name)
                for name, site in bond:
                    # The molecule taking the bond holds every bond its pattern does not forbid.
                    for molecule in _molecules(before):
                        conditions = molecule.site_conditions
                        if molecule.monomer.name == name and _is_free(conditions.get(site, ANY)):
                            gains |= {
                                pair
                                for other in molecule.monomer.sites
                                if other != site and not _is_free(conditions.get(other, ANY))
                                for pair in ((name, site, other), (name, other, site))
                            }
    holds |= gains
    # From "bound on s", a molecule that may also bind on t reaches its partner on t's bond.
    steps: dict[tuple[str, str], list[tuple[tuple[str, str], tuple, bool]]] = {}
    for bond in formed:
        for (name, site), partner in ((bond[0], bond[1]), (bond[1], bond[0])):
            for held_name, s, t in holds:
                if held_name == name and t == site:
                    gain = (name, s, t) in gains
                    steps.setdefault((name, s), []).append((partner, bond, gain))
    cycle = _cycle(steps)
    if cycle is None:
        return
    rules = set().union(*(formed[bond] for _, bond in cycle))
    rows = sorted({e.prefix for e in emissions if rules & set(e.rules)})
    chain = "-".join(node[0] for node, _ in cycle) + f"-{cycle[0][0][0]}-..."
    raise ModelUpdateError(
        f"These statements form complexes that grow without bound: rows {rows} bind "
        f"{sorted({node[0] for node, _ in cycle})} into chains ({chain}), each molecule bound on "
        "one site still free to bind on another. Require each partner to be free of its other "
        "partners where it binds: `A{~B} binds A{~B}`, `A{~A} binds B{~B}`, `B{~A} binds B{~A}`."
    )


def _unformed(emission: M.Emission, observable: pysb.Observable, written: str) -> str:
    """Why ``emission``'s row is refused for reading ``observable``, the amount ``written``, which
    no species of the model matches: it is 0 throughout, as a regulator nothing forms is
    (:func:`regulation.record`). INDRA's monomer pass gives a molecule every site an amount in the
    row notation names (``ERK{pY187}``) in both states, so a site no other row gives, or a slip
    (``T20`` beside ``T202``), is built unformed; each molecule's sites are listed as the rows give
    them."""
    from .schemes import _sites_text

    alone, held = [], []
    for mp in _molecules(observable.reaction_pattern):
        added = emission.sites_added.get(mp.monomer.name, {})
        alone += [f"{mp.monomer.name} the site {s}" for s in mp.site_conditions if s in added]
        sites = [s for s in mp.monomer.sites if s not in added]
        held.append(_sites_text(mp.monomer.name, sites, mp.monomer.site_states))
    return (
        f"row {emission.prefix!r}: its rate reads `amount({written})`, but nothing in the model "
        "forms it, so that amount is 0 throughout"
        + (f": no row but this one gives {', '.join(alone)}" if alone else "")
        + f". Read a form the rows make, on the sites they give ({'; '.join(dict.fromkeys(held))}), "
        "add the statement that forms this one, or declare it present in `seeds`."
    )


def _molecules(reaction_pattern) -> list:
    return [
        mp for cp in reaction_pattern.complex_patterns if cp is not None for mp in cp.monomer_patterns
    ]


def _is_free(value) -> bool:
    """Whether a site condition requires the site unbound (``None``, or ``(state, None)``)."""
    return value is None or (isinstance(value, tuple) and value[1] is None)


def _is_bond(value) -> bool:
    if isinstance(value, tuple):
        value = value[1]
    return value is ANY or (isinstance(value, int) and not isinstance(value, bool))


def _bonds(reaction_pattern) -> list[tuple]:
    """Each bond of the pattern as its two ``(monomer, site)`` ends, sorted."""
    bonds = []
    for cp in reaction_pattern.complex_patterns:
        if cp is None:
            continue
        ends: dict[int, list] = {}
        for mp in cp.monomer_patterns:
            for site, value in mp.site_conditions.items():
                number = value[1] if isinstance(value, tuple) else value
                if isinstance(number, int) and not isinstance(number, bool):
                    ends.setdefault(number, []).append((mp.monomer.name, site))
        bonds += [tuple(sorted(pair)) for pair in ends.values() if len(pair) == 2]
    return bonds


def _cycle(steps: dict) -> list | None:
    """A cycle in ``steps`` (node -> ``(partner, bond, gain)``) through at least one gain step, as
    its ``(node, bond)`` steps, or ``None``: each gain step, then the shortest way back from its
    partner to where it started."""
    for start, outgoing in steps.items():
        for first, first_bond, gain in outgoing:
            if not gain:
                continue
            came: dict = {first: None}
            frontier = [first]
            while frontier and start not in came:
                ahead = []
                for node in frontier:
                    for partner, bond, _ in steps.get(node, []):
                        if partner not in came:
                            came[partner] = (node, bond)
                            ahead.append(partner)
                frontier = ahead
            if start in came:
                back, node = [], start
                while came[node] is not None:
                    back.append(came[node])
                    node = came[node][0]
                return [(start, first_bond), *reversed(back)]
    return None

def _rename(sbml_model, element, new: str) -> None:
    """Rename ``element`` and every reference to it. Element by element: the model's own
    ``renameSIdRefs`` reaches neither a reaction's species references nor a rule's target."""
    old = element.getId()
    for other in sbml_model.getListOfAllElements():
        other.renameSIdRefs(old, new)
    element.setId(new)


def _number_reactions(sbml_model) -> None:
    """Number the reactions ``r0``, ``r1``..., or ``r_0``... when another id already has that form:
    a parameter the caller named ``r8`` would otherwise clash with a reaction id."""
    taken = {
        element.getId()
        for element in sbml_model.getListOfAllElements()
        if element.getTypeCode() != libsbml.SBML_REACTION and element.isSetId()
    }
    reactions = list(sbml_model.getListOfReactions())
    prefix = "r"
    while any(f"{prefix}{i}" in taken for i in range(len(reactions))):
        prefix += "_"
    for i, reaction in enumerate(reactions):
        reaction.setId(f"{prefix}{i}")


def _states(patterns) -> set[tuple[str, str, str]]:
    """``(molecule, site, state)`` of each site the complex ``patterns`` give a state."""
    out = set()
    for cp in patterns:
        for mp in cp.monomer_patterns if cp is not None else ():
            for site, state in mp.site_conditions.items():
                state = state[0] if isinstance(state, tuple) else state
                if isinstance(state, str):
                    out.add((mp.monomer.name, site, state))
    return out


def _site_states(model: pysb.Model, emissions) -> tuple[dict, dict]:
    """Each rule's site states it reads, and each site state, to the rows whose rules put it there
    (``None`` for a starting amount)."""
    reads, puts = {}, {}
    row = {rule: e.prefix for e in emissions for rule in e.rules}
    for rule in model.rules:
        before = _states(rule.reactant_pattern.complex_patterns)
        after = _states(rule.product_pattern.complex_patterns)
        reads[rule.name] = before
        for state in (after - before) | ((before - after) if rule.is_reversible else set()):
            puts.setdefault(state, set()).add(row.get(rule.name))
    for initial in model.initials:
        for state in _states([initial.pattern]):
            puts.setdefault(state, set()).add(None)
    return reads, puts


def _every_state(rows, model: pysb.Model, emissions, set_: set[str]) -> list[str]:
    """What each row whose catalyst names no state, on a site other rows change, is told: every
    state of it catalyses, the row as it would name one, and, where no rule makes or removes the
    catalyst and nothing the experiment sets (``set_``) is it, that its rate reads a constant
    total. raimundez's ``MEK phosphorylates MAPK`` read ``MEK{u} + MEK{p}``, which sum to the
    seed: ``pMAPK`` stayed flat in every experiment of 18 versions, and the agent never looked at
    that rate (PEtab-MCP#232)."""
    from .notation import agent_text, statement_text
    from .statements import _catalyst, _products

    changed: dict[tuple[str, str], set[str]] = {}  # (molecule, site) -> the rows changing it
    for e in emissions:
        for rule in map(model.rules.__getitem__, e.rules):
            after = _states(rule.product_pattern.complex_patterns)
            for m, site, state in _states(rule.reactant_pattern.complex_patterns):
                if any((a, b) == (m, site) and c != state for a, b, c in after):
                    changed.setdefault((m, site), set()).add(e.prefix)
    out = []
    for row, e in zip(rows, emissions):
        stmt, enzyme = row[0], _catalyst(row[0])[0]
        if enzyme is None or (name := _n(enzyme.name)) not in model.monomers.keys():
            continue
        sites = model.monomers[name].site_states
        open_ = {
            site
            for rule in e.rules
            for mp in _molecules(model.rules[rule].reactant_pattern)
            if mp.monomer.name == name
            for site in sites
            if site != "loc" and site not in mp.site_conditions
        }
        writers = sorted(set().union(*(changed.get((name, s), ()) for s in open_)) - {e.prefix})
        if not writers:
            continue
        state = next(
            (
                agent
                for other in rows
                for agent, effect in _products(other[0])
                if effect > 0
                and _n(agent.name) == name
                and (agent.mods or agent.activity)
                and all(m.is_modified for m in agent.mods)
                and (agent.activity is None or agent.activity.is_active)
            ),
            None,
        )
        example = ""
        if state is not None:
            named = deepcopy(enzyme)
            named.mods = [*named.mods, *state.mods]
            named.activity = named.activity or state.activity
            written = deepcopy(stmt)
            setattr(written, next(k for k, v in vars(stmt).items() if v is enzyme), named)
            regulators = {k: [a.to_json() for a in v] for k, v in (row[4:5] or [{}])[0].items()}
            try:
                example = f": `{statement_text({**written.to_json(), **regulators})}`"
            except ValueError:
                example = f": `{agent_text(named.to_json())}`"
        states = "; ".join(f"`{site}` {' and '.join(sites[site])}" for site in sorted(open_))
        text = (
            f"Row `{e.prefix}` names no state of {enzyme.name}, its catalyst, so every state of it "
            f"catalyses ({states} alike), though its state changes in "
            f"{', '.join(f'`{w}`' for w in writers)}. To make one form the enzyme, name its "
            f"state{example}."
        )
        if name not in set_ and all(
            sum(mp.monomer.name == name for mp in _molecules(rule.reactant_pattern))
            == sum(mp.monomer.name == name for mp in _molecules(rule.product_pattern))
            for rule in model.rules
        ):
            text += (
                f" No row makes or removes {enzyme.name} and no condition sets it, so its total is "
                f"constant, and so is what this rate reads of it: nothing that changes "
                f"{enzyme.name}'s state moves the rate."
            )
        out.append(text)
    return out


def _misnamed(prefix: str, rules: list[str], reads: dict, puts: dict) -> str:
    """Why row ``prefix``'s ``rules`` never fire where a phosphorylation they read is put on
    another site of the same molecule: a writer and a reader naming the site apart (`at Y1068` and
    `E{p}`). Phosphorylation only: the other modifications share the states 'n'/'y', so a 'y' on
    another site may be another modification."""
    for molecule, site, state in sorted(set().union(*(reads[r] for r in rules))):
        if state != "p" or (molecule, site, state) in puts:
            continue
        for (m, other, s), writers in sorted(puts.items(), key=str):
            if m == molecule and s == state and other != site:
                by = sorted(w for w in writers if w is not None)
                return (
                    f"Row {prefix!r} reads {molecule}'s site `{site}` as '{state}', which nothing "
                    f"puts there, while {f'rows {by}' if by else 'a seed'} put '{state}' on its site "
                    f"`{other}`: a phosphorylation sits on the site its statement names, by "
                    "residue and position, or `phospho` where it names none. Name the same residue "
                    "and position in both rows, or none in either; a seed joins no two sites."
                )
    return ""


def _monomer_names(reaction_pattern) -> set[str]:
    # INDRA pads a rule's sides with None for Kappa (one-step Conversion, IncreaseAmount and
    # DecreaseAmount with a subject): a place holder, not a complex.
    return {
        mp.monomer.name
        for cp in reaction_pattern.complex_patterns
        if cp is not None
        for mp in cp.monomer_patterns
    }
