"""The model a set of statements builds, as SBML this PEtab problem can score.

INDRA turns statements into a PySB model (:mod:`mechanisms`) and PySB writes SBML in one call.
Neither knows anything about the PEtab problem the model is for, and the SBML they produce cannot
be scored against it as it stands. This module is the difference, and it is all problem-specific:

* **Readouts.** PySB numbers every species (``__s0``, ``__s1`` ...), while the fixed observable
  table reads names: ``pERK``, or a blinded ``readout_pS6_tot``. Each species the entity map
  names is given that name, or for a blinded mapping an assignment rule of that name.
* **Starting amounts.** What is present at the start is traced (below); each present monomer
  starts at an estimated ``<monomer>_init``, or at the row the problem ships (Crauste's fixed
  ``Naive_init``) — unless the statements DECLARE what is there (``starts``, below).
* **Inputs.** A species the experiment sets is held rather than built: one the condition table
  targets (Isensee's ``Fsk``), which its periods set, and one the problem's ``inputs.tsv`` holds to
  a protocol (Fujita's EGF ramp, :func:`protocols.hold`).
* **Leftovers.** A rate constant an agent-written mechanism replaced is removed.

The start rests on two assumptions, and a third check refuses the models they do not fit:

1. **The start holds free monomers in INDRA's default state** (``set_base_initial_condition``):
   unbound, unmodified, inactive, wild type, in the cytoplasm when that is a location (else in
   the first of its locations by name). No complex and no modified form exists at the start.
2. **What the start makes is absent from it.** Every monomer no rule makes from other kinds is
   present (one that only proliferates included), and BioNetGen expands the rules from those. A
   made monomer that expansion reaches starts empty: Crauste's effectors exist only once naive
   cells convert. One it does not reach traces back to nothing at the start (populations that
   only interconvert), so it is present too. A protein synthesised at a steady rate therefore
   starts at zero, not at its steady state; nothing here tells the two apart.
3. **Every rule fires, and every readout reads a species,** in the expansion from that start. A
   rule that needs a complex or modified form nothing forms (a kinase active only while bound to a
   partner no statement binds it to) is refused by its row, rather than built into a model whose
   fit can never move it.

This is MARM2's causal tracing (Fröhlich et al. 2023, MSB 19:e10988) asked statically. There, a
synthetic site set by the rule that makes a molecule, and carried by BioNetGen through every
later step, splits each simulated species by origin. Here the only question is whether an origin
exists, whether a monomer traces back to the start at all, and the expansion's reactions answer
it without a site.

A declared start (``make_changes``' ``starts``) replaces the trace for the monomers it names. The
species its Agent writes starts at that amount, in exactly the state the Agent writes (mods, location, activity, and the complex its
``bound_conditions`` form, partners in their default state), and no monomer named in any
declaration is started any other way: declare each state it starts in. A number other than 0
becomes ``<name>_init`` and reconcile treats it as any other starting value (1 a unit normalisation,
fixed; anything else estimated from there), so Blasi's and Sneyd's unit pools cost nothing in k. A
0 is an empty start with no parameter at all — what lets a species the condition table sets
(Enzyme kinetics' substrate) carry no ``_init`` that every condition overrides. An expression is the starting amount itself, over parameter ids: a row the
problem ships (Fiedler's fixed totals), a condition-table column (Lucarelli's ``init_S2``), or a new
name, estimated. A declared monomer may also be one the rules make (Lucarelli's genes start at 1).

It is a separate module because none of it is a mechanism: the same library rows would need
none of this under a server that read PySB directly. This one reads SBML everywhere — reconcile,
lint, de-duplication, versions — so a model built from statements enters as SBML and is from then
on an ordinary authored model; the hop costs a BioNetGen call or two per edit, and nothing
downstream has to know where a model came from.
"""

from __future__ import annotations

import libsbml
import pysb
from indra.assemblers.pysb.assembler import get_monomer_pattern
from indra.statements import Agent, Statement
from petab.v1.models.sbml_model import SbmlModel
from pysb import ANY
from pysb.bng import NoInitialConditionsError, generate_equations
from pysb.export import export

from . import mechanisms as M
from .errors import ModelUpdateError
from .protocols import Input, hold
from .reconcile import SEED_INITIAL_VALUE


def to_antimony(
    rows: list[tuple[Statement, str, str]],
    entities: dict[str, Agent],
    library: dict[tuple[str, str], M.Policy] = M.REGISTRY,
    inputs: tuple[Input, ...] = (),
    starts: list[tuple[str, Agent, float | str]] | None = None,
    *,
    targets: frozenset[str] = frozenset(),
) -> tuple[str, list[M.Emission]]:
    """The model ``rows`` build, as Antimony the ``antimony`` route accepts unchanged.

    Every species in ``entities`` becomes a symbol of that name, the sum over its PySB pattern,
    because that name is what the fixed observable table reads. The start is traced as the module
    docstring sets out, and :class:`ModelUpdateError` names the rows a model breaking it fails on.

    An experiment's INPUT is held instead, with no starting amount of its own: a species of
    ``entities`` among ``targets`` (the problem's condition targets), which its periods set, and
    each of ``inputs`` (the problem's :mod:`protocols`), under its protocol. An input the entity
    map does not name is the agent of its id.

    ``starts`` (``(name, agent, amount)``: a number or an expression over parameter ids) declare
    what is present at the start, as the module docstring sets out; each is ``<name>_init``.
    """
    starts = starts or []
    entities = {**{item.id: Agent(item.id) for item in inputs}, **entities}
    # Start agents join the monomer pass (they may name a site or partner no row does); they are
    # not readouts, so they stay out of `entities` below.
    model, emissions = M.assemble(
        rows, {**entities, **{f"__start_{name}": agent for name, agent, _ in starts}}, library
    )
    protocol = {item.id: item for item in inputs}
    held = {
        entities[s].name: protocol.get(s) for s in entities if s in protocol or s in targets
    }
    _refuse_chains(model, emissions)
    made = set()
    for rule in model.rules:
        reactants = _monomer_names(rule.reactant_pattern)
        products = _monomer_names(rule.product_pattern)
        made |= products - reactants
        if rule.is_reversible:
            made |= reactants - products
    # Readouts are named apart from the species they stand for, which may share a monomer's
    # name; the SBML export numbers observables anyway, and the rename below restores them.
    readouts = {f"readout_{species}": species for species in entities}
    for name, species in readouts.items():
        pattern = get_monomer_pattern(model, entities[species])
        model.add_component(pysb.Observable(name, pattern, _export=False))
    declared = _declare(model, starts, set(held))
    _seed(
        model,
        [m for m in model.monomers if (m.name not in made or m.name in held) and m.name not in declared],
    )
    _expand(model)
    reached = {mp.monomer.name for sp in model.species for mp in sp.monomer_patterns}
    unreached = [
        m for m in model.monomers if m.name in made - held.keys() - reached - declared
    ]
    if unreached:
        _seed(model, unreached)
        model.reset_equations()
        _expand(model)
    fired = {name for reaction in model.reactions for name in reaction["rule"]}
    never = []
    stuck = sorted({e.prefix for e in emissions if not fired.issuperset(e.rules)})
    if stuck:
        never.append(f"rows {stuck} build rules that never fire")
    unread = sorted(sp for name, sp in readouts.items() if not model.observables[name].species)
    if unread:
        never.append(f"readouts {unread} read no species")
    if never:
        raise ModelUpdateError(
            f"In the model these statements build, {' and '.join(never)}. The start holds free, "
            "unmodified molecules and what `starts` declares, and a rule fires only on what the "
            "statements form from there: one that never fires needs a complex or a modified form "
            "that nothing forms, like a kinase active only while bound to a partner nothing binds "
            "it to. Add the statement that forms it, or, if it is present at t=0, declare it in "
            "`starts`."
        )
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
    base = {
        base_names[sp.getName()]: sp
        for sp in sbml_model.getListOfSpecies()
        if sp.getName() in base_names
    }
    for assignment in list(sbml_model.getListOfInitialAssignments()):
        amount = assignment.getMath().getName()
        if amount.removesuffix("_init") in held:
            sbml_model.getSpecies(assignment.getSymbol()).setInitialAmount(0.0)
            sbml_model.removeParameter(amount)
            assignment.removeFromParentAndDelete()
    named = set()
    for parameter in list(sbml_model.getListOfParameters()):
        if parameter.getName() not in readouts:
            continue
        old, new = parameter.getId(), readouts[parameter.getName()]
        rule = sbml_model.getAssignmentRuleByVariable(old)
        species = sbml_model.getSpecies(rule.getMath().getName() or "")
        if (
            rule.getMath().isName()
            and species is not None
            and species.getId() not in named
            and not new.startswith("readout_")
        ):
            # A readout that IS one species names that species, rather than leaving a rule over
            # it: the reachability checks in `reconcile` and `operations` follow reactions, not
            # rules, and warned that every species was disconnected from the data. Except a
            # blinded `readout_*` mapping, which the lint requires to stay a rule
            # (`validation`'s measurement-mapping check refused Fujita's pS6 as a species).
            # A species takes one name: each further readout of it stays a rule over that name.
            old, renamed = species.getId(), species
            named.add(new)
            rule.removeFromParentAndDelete()
            parameter.removeFromParentAndDelete()
        else:
            rule.setVariable(new)
            renamed = parameter
        _rename(sbml_model, renamed, new)
    for name, _, amount in starts:
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


def _declare(
    model: pysb.Model, starts: list[tuple[str, Agent, float | str]], held: set[str]
) -> set[str]:
    """Start each declared species at ``<name>_init``, in the state its Agent writes; returns the
    monomers the declarations name, which :func:`_seed` then leaves alone."""
    named: set[str] = set()
    names = [name for name, _, _ in starts]
    repeated = sorted({n for n in names if names.count(n) > 1})
    if repeated:
        raise ModelUpdateError(
            f"starts {repeated} are named twice; each start is `<name>_init`, so give a second "
            "start of the same agent a name of its own"
        )
    for name, agent, amount in starts:
        if agent.name in held:
            raise ModelUpdateError(
                f"start {name!r}: {agent.name} is an input, held to its protocol, so it has no "
                "start of its own"
            )
        if not isinstance(amount, (int, float, str)) or isinstance(amount, bool):
            raise ModelUpdateError(
                f"start {name!r} is {amount!r}; give a number or an expression over parameter ids"
            )
        try:
            molecules = M.agent_pattern(model, agent, fresh=lambda name: True)
        except M.MechanismError as exc:
            raise ModelUpdateError(f"start {name!r}: {exc}") from exc
        pattern = pysb.ComplexPattern(molecules, None)
        value = float(amount) if not isinstance(amount, str) else SEED_INITIAL_VALUE
        parameter = pysb.Parameter(f"{name}_init", value, _export=False)
        model.add_component(parameter)
        try:
            model.add_initial(pysb.Initial(pattern, parameter, _export=False))
        except ValueError as exc:  # pysb: not concrete, or a duplicate species
            raise ModelUpdateError(
                f"start {name!r} is not a start PySB accepts: {exc}"
            ) from exc
        named |= {mp.monomer.name for mp in pattern.monomer_patterns}
    return named


def _initial_expression(sbml_model, placeholder: str, species: str, formula: str) -> None:
    """Replace the declared species' ``<key>_init`` with the expression its entry gives. A name
    the model does not hold becomes a parameter: a shipped row or a condition-table column fills
    it, and anything else is estimated from reconcile's seed."""
    math = libsbml.parseL3Formula(formula)
    if math is None:
        raise ModelUpdateError(f"start {species!r} is no formula: {formula!r}")
    for node in _ast_nodes(math):
        if node.getType() == libsbml.AST_NAME and sbml_model.getElementBySId(node.getName()) is None:
            parameter = sbml_model.createParameter()
            parameter.setId(node.getName())
            parameter.setConstant(True)
            parameter.setValue(SEED_INITIAL_VALUE)
    for assignment in sbml_model.getListOfInitialAssignments():
        if assignment.getMath().isName() and assignment.getMath().getName() == placeholder:
            assignment.setMath(math)


def _initial_empty(sbml_model, placeholder: str) -> None:
    """Start the declared species at a literal 0: no assignment, so no parameter for reconcile to
    estimate or a condition to override."""
    for assignment in list(sbml_model.getListOfInitialAssignments()):
        if assignment.getMath().isName() and assignment.getMath().getName() == placeholder:
            sbml_model.getSpecies(assignment.getSymbol()).setInitialAmount(0.0)
            assignment.removeFromParentAndDelete()


def _seed(model: pysb.Model, monomers: list[pysb.Monomer]) -> None:
    """Start each of ``monomers`` present, as assumption 1 of the module docstring sets out."""
    for monomer in monomers:
        # Not INDRA's `set_base_initial_condition` itself, whose state this is: its
        # `<monomer>_0` silently skips a monomer whose name is taken, and Fujita's EGF stimulus
        # column is `EGF_0`.
        state = {site: monomer.site_states.get(site, [None])[0] for site in monomer.sites}
        locations = monomer.site_states.get("loc", ())
        if locations:
            # Else the first by name, not the first INDRA met: that was whichever row came first.
            state["loc"] = "cytoplasm" if "cytoplasm" in locations else min(locations)
        # Reconcile's own seed value, not 1, which it holds FIXED as a unit normalisation: the
        # first version started every Crauste population at one cell that way.
        amount = pysb.Parameter(f"{monomer.name}_init", SEED_INITIAL_VALUE, _export=False)
        model.add_component(amount)
        model.add_initial(pysb.Initial(monomer(**state), amount, _export=False))


#: The most molecules one complex may hold. A network whose complexes reach it is taken for one
#: whose complexes grow without bound, which BioNetGen expands forever. BioNetGen builds every
#: complex up to the cap before the check can refuse, so the cap bounds the cost of finding out:
#: reactions grow about with its square for a chain, exponentially for complexes that branch.
MAX_COMPLEX = 20


def _expand(model: pysb.Model) -> None:
    """Fill in BioNetGen's expansion of ``model`` from its start. Left empty when nothing starts
    (populations that only interconvert), which PySB refuses to expand, or nothing fires, whose
    network BioNetGen refuses to write.

    Expanded with complexes capped at :data:`MAX_COMPLEX` molecules, and refused when one reaches
    it: statements that let complexes chain without end (STAT5A and STAT5B each dimerising with
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
    replaced."""
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
    one site while binding on another: STAT5A and STAT5B each dimerising with itself and with the
    other make STAT5A-STAT5A-STAT5B-STAT5B-STAT5A-... The rules say this at once, where the
    capped expansion (:data:`MAX_COMPLEX`) only finds it out after building every complex up to
    the cap, which took 19 s on that pair alone and 251 s on a loaded laptop. The cap stays as
    the backstop for growth the bonds alone do not show.
    """
    formed: dict[tuple, set[str]] = {}  # bond, as its two (monomer, site) ends -> rules forming it
    holds: set[tuple[str, str, str]] = set()  # (monomer, s, t): bound on s and t at once
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
                            holds |= {
                                pair
                                for other in molecule.monomer.sites
                                if other != site and not _is_free(conditions.get(other, ANY))
                                for pair in ((name, site, other), (name, other, site))
                            }
    # From "bound on s", a molecule that may also bind on t reaches its partner on t's bond.
    steps: dict[tuple[str, str], list[tuple[tuple[str, str], tuple]]] = {}
    for bond in formed:
        for (name, site), partner in ((bond[0], bond[1]), (bond[1], bond[0])):
            for held_name, s, t in holds:
                if held_name == name and t == site:
                    steps.setdefault((name, s), []).append((partner, bond))
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
    """A cycle in ``steps`` as its ``(node, bond)`` steps, or ``None``."""
    state: dict = {}
    path: list = []

    def visit(node) -> list | None:
        state[node] = "open"
        for partner, bond in steps.get(node, []):
            path.append((node, bond))
            if state.get(partner) == "open":
                start = next(i for i, (n, _) in enumerate(path) if n == partner)
                return path[start:]
            if partner not in state:
                found = visit(partner)
                if found:
                    return found
            path.pop()
        state[node] = "done"
        return None

    for node in list(steps):
        if node not in state:
            found = visit(node)
            if found:
                return found
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
    a shared rate an agent named ``r8`` made every Blasi model's SBML unreadable."""
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


def _monomer_names(reaction_pattern) -> set[str]:
    # INDRA pads a rule's sides with None for Kappa (one-step Conversion, IncreaseAmount and
    # DecreaseAmount with a subject): a place holder, not a complex.
    return {
        mp.monomer.name
        for cp in reaction_pattern.complex_patterns
        if cp is not None
        for mp in cp.monomer_patterns
    }
