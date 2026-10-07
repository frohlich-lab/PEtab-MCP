"""The model a set of statements builds, as SBML this PEtab problem can score.

INDRA turns statements into a PySB model (:mod:`mechanisms`) and PySB writes SBML in one call.
Neither knows anything about the PEtab problem the model is for, and the SBML they produce cannot
be scored against it as it stands. This module is the difference, and it is all problem-specific:

* **Readouts.** PySB numbers every species (``__s0``, ``__s1`` ...), while the fixed observable
  table reads names: ``pERK``, or a blinded ``readout_pS6_tot``. Each species the entity map
  names is given that name, or for a blinded mapping an assignment rule of that name.
* **Starting amounts.** A protein that is only ever modified must be present at the start, at an
  estimated ``<monomer>_init``; a population something produces starts empty; a row the problem
  ships (Crauste's fixed ``Naive_init``) is used as shipped.
* **Inputs.** A species the experiment sets (Fujita's EGF dose) is held to the protocol formula
  the entity map declares, and the condition-table columns it reads are declared for the table
  to fill.
* **Leftovers.** A rate constant an agent-written mechanism replaced is removed.

It is a separate module because none of it is a mechanism: the same library rows would need
none of this under a server that read PySB directly. This one reads SBML everywhere — reconcile,
lint, de-duplication, versions — so a model built from statements enters as SBML and is from then
on an ordinary authored model; the hop costs a BioNetGen call per edit, and nothing downstream has
to know where a model came from.
"""

from __future__ import annotations

import libsbml
import pysb
from indra.assemblers.pysb.assembler import get_monomer_pattern
from indra.statements import Agent, Statement
from petab.v1.models.sbml_model import SbmlModel
from pysb.export import export

from . import mechanisms as M
from .errors import ModelUpdateError
from .reconcile import SEED_INITIAL_VALUE


def to_antimony(
    rows: list[tuple[Statement, str, str]],
    entities: dict[str, Agent],
    library: dict[tuple[str, str], M.Policy] = M.REGISTRY,
    inputs: dict[str, str] | None = None,
) -> tuple[str, list[M.Emission]]:
    """The model ``rows`` build, as Antimony the ``antimony`` route accepts unchanged.

    Every species in ``entities`` becomes a symbol of that name, the sum over its PySB pattern,
    because that name is what the fixed observable table reads.

    A monomer some rule CREATES starts empty: Crauste's effector populations exist only once
    naive cells convert. Every other monomer is only ever modified, so it must be present at the
    start, and its unmodified state starts at ``<monomer>_init`` — estimated, from reconcile's
    seed value, unless the problem ships that row (Crauste's fixed 8090 naive cells).

    An experiment's INPUT (``inputs``: species id to the protocol formula the problem's entity map
    declares) is held to that formula instead. It has no starting amount of its own, and each
    column the formula names becomes a parameter the condition table sets.
    """
    model, emissions = M.assemble(rows, entities, library)
    created = set()
    for rule in model.rules:
        created |= _monomer_names(rule.product_pattern) - _monomer_names(rule.reactant_pattern)
    held = {entities[species].name: formula for species, formula in (inputs or {}).items()}
    for monomer in model.monomers:
        # Also BioNetGen's seeds: network generation starts from the initial species. Not INDRA's
        # `set_base_initial_condition`: its `<monomer>_0` silently skips a monomer whose name is
        # taken, and Fujita's EGF stimulus column is `EGF_0`.
        unmodified = {site: monomer.site_states.get(site, [None])[0] for site in monomer.sites}
        # Reconcile's own seed value, not 1, which it holds FIXED as a unit normalisation: the
        # first version started every Crauste population at one cell that way.
        amount = pysb.Parameter(f"{monomer.name}_init", SEED_INITIAL_VALUE, _export=False)
        model.add_component(amount)
        model.add_initial(pysb.Initial(monomer(**unmodified), amount, _export=False))
    base_names = {
        str(initial.pattern): initial.pattern.monomer_patterns[0].monomer.name
        for initial in model.initials
    }
    # Readouts are named apart from the species they stand for, which may share a monomer's
    # name; the SBML export numbers observables anyway, and the rename below restores them.
    readouts = {f"readout_{species}": species for species in entities}
    for name, species in readouts.items():
        pattern = get_monomer_pattern(model, entities[species])
        model.add_component(pysb.Observable(name, pattern, _export=False))
    try:
        sbml = export(model, "sbml")
    except Exception as exc:  # BioNetGen fails as a bare Exception or an OSError
        raise ModelUpdateError(
            f"Expanding the statements into reactions failed: {exc}. (The expansion runs "
            "BioNetGen, which needs BNG2.pl on PATH or BNGPATH set.)"
        ) from exc
    document = libsbml.readSBMLFromString(sbml)
    sbml_model = document.getModel()
    base = {
        base_names[sp.getName()]: sp
        for sp in sbml_model.getListOfSpecies()
        if sp.getName() in base_names
    }
    for assignment in list(sbml_model.getListOfInitialAssignments()):
        amount = assignment.getMath().getName()
        if amount.removesuffix("_init") in created | held.keys():
            sbml_model.getSpecies(assignment.getSymbol()).setInitialAmount(0.0)
            sbml_model.removeParameter(amount)
            assignment.removeFromParentAndDelete()
    for parameter in list(sbml_model.getListOfParameters()):
        if parameter.getName() not in readouts:
            continue
        old, new = parameter.getId(), readouts[parameter.getName()]
        rule = sbml_model.getAssignmentRuleByVariable(old)
        species = sbml_model.getSpecies(rule.getMath().getName() or "")
        if rule.getMath().isName() and species is not None and not new.startswith("readout_"):
            # A readout that IS one species names that species, rather than leaving a rule over
            # it: the reachability checks in `reconcile` and `operations` follow reactions, not
            # rules, and warned that every species was disconnected from the data. Except a
            # blinded `readout_*` mapping, which the lint requires to stay a rule
            # (`validation`'s measurement-mapping check refused Fujita's pS6 as a species).
            old, renamed = species.getId(), species
            rule.removeFromParentAndDelete()
            parameter.removeFromParentAndDelete()
        else:
            rule.setVariable(new)
            renamed = parameter
        _rename(sbml_model, renamed, new)
    _drop_unreferenced_parameters(sbml_model)
    for monomer, formula in held.items():
        protocol = libsbml.parseL3Formula(formula)
        if protocol is None:
            raise ModelUpdateError(
                f"the entity map's input for {monomer} is no formula: {formula!r}"
            )
        for node in _ast_nodes(protocol):
            if (
                node.getType() == libsbml.AST_NAME
                and sbml_model.getElementBySId(node.getName()) is None
            ):
                column = sbml_model.createParameter()
                column.setId(node.getName())
                column.setConstant(True)
                column.setValue(0.0)
        # A species may sit in reactions and under a rule only on the boundary.
        species = base[monomer]
        species.setBoundaryCondition(True)
        rule = sbml_model.createAssignmentRule()
        rule.setVariable(species.getId())
        rule.setMath(protocol)
    return SbmlModel.from_string(libsbml.writeSBMLToString(document)).to_antimony(), emissions


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


def _rename(sbml_model, element, new: str) -> None:
    """Rename ``element`` and every reference to it. Element by element: the model's own
    ``renameSIdRefs`` reaches neither a reaction's species references nor a rule's target."""
    old = element.getId()
    for other in sbml_model.getListOfAllElements():
        other.renameSIdRefs(old, new)
    element.setId(new)


def _monomer_names(reaction_pattern) -> set[str]:
    return {
        mp.monomer.name for cp in reaction_pattern.complex_patterns for mp in cp.monomer_patterns
    }
