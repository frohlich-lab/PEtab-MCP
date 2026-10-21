"""Exact reductions of a statements model's rules, before BioNetGen expands them.

BioNetGen expands every rule over every species its patterns match, so a rule written over one
site of a multisite protein becomes one reaction per form of the sites it does not mention. When
those forms do not change what the rule does, the expansion is redundant, and the rules can be
rewritten so that it never happens: the reduced ODEs have the same solution for every parameter
value. Nothing here approximates (no time-scale separation, no lumping, no pruning by flux).

**A catalyst becomes a rate factor.** A rule that leaves one of its complexes unchanged and
forms no bond (``E(ctx) + S(x='u') >> E(ctx) + S(x='p')``, a one-step phosphorylation, an
activation, a catalysed synthesis) expands into one reaction per (catalyst species, substrate
species) pair, ``k·[E_j]·[S_i]``. Summed over the catalyst species that is
``k·Obs(E(ctx))·[S_i]``, so the rule is rewritten without its catalyst (``S(x='u') >> S(x='p')``)
at the rate ``k·Obs(E(ctx))``, an assignment rule ``<rule>_catalysed`` in the SBML: one reaction
per substrate species, the right-hand side identical term by term. ``Obs`` counts
species, not molecules, because BioNetGen counts a catalyst species once however many molecules
in it match: two maps that differ only outside the reaction centre are one reaction. The same
holds when the catalyst and the substrate are one monomer (EGFR on EGFR), which BioNetGen writes
``k·[s]·[s]`` with no symmetry factor. A rule with two unchanged complexes keeps its form (the two
could be the same pattern, which BioNetGen would count by symmetry), as does a binding, a
two-step mechanism, whose steps bind and release, and a ligand that binds its receptor while its
free pool stays as it was (``nondepleting``). The rule keeps its name, so a reaction still says
which rule and row it came from.

**What the start reaches is still decided by the catalyst.** The rewritten rule fires on its
substrate whether or not any catalyst exists, so BioNetGen's expansion may reach species the
original never would. :func:`reached` decides what the start reaches as the original rules would:
a rewritten rule's reaction counts only once its catalyst is reached. Every rule of a model the
statements route accepts fires (:mod:`problem_sbml`, assumption 3), and then the two expansions
are the same species.

Its own module because none of it is about the PEtab problem: it is the same model, simulated
over fewer reactions (Borisov et al. 2005, Biophys J, doi:10.1529/biophysj.105.060533; Conzelmann
et al. 2006, BMC Bioinformatics 7:34).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pysb
from pysb.core import ReactionPattern, RuleExpression


@dataclass
class Reduction:
    """What :func:`reduce` rewrote."""

    #: Each rule whose catalyst became a rate factor -> the catalyst's amount, in species.
    catalysts: dict[str, pysb.Observable] = field(default_factory=dict)


def reduce(model: pysb.Model) -> Reduction:
    """Rewrite ``model``'s rules in place as the module docstring sets out; returns what changed."""
    return Reduction(_catalysts_as_rates(model))


def _catalysts_as_rates(model: pysb.Model) -> dict[str, pysb.Observable]:
    catalysts = {}
    for rule in model.rules:
        rates = [rule.rate_forward] + ([rule.rate_reverse] if rule.is_reversible else [])
        if rule.move_connected or any(isinstance(r, pysb.Expression) and r.is_local for r in rates):
            continue
        # INDRA pads a side with None for Kappa (`E() + None >> E() + X()`): no complex.
        lhs = [cp for cp in rule.reactant_pattern.complex_patterns if cp is not None]
        rhs = [cp for cp in rule.product_pattern.complex_patterns if cp is not None]
        # At the same place on both sides: BioNetGen maps molecules of one type in order.
        same = [i for i in range(min(len(lhs), len(rhs))) if str(lhs[i]) == str(rhs[i])]
        if len(same) != 1:
            continue
        (i,) = same
        if _bonds(rhs) > _bonds(lhs):
            continue
        amount = pysb.Observable(
            _free(model, f"{rule.name}_catalyst"), lhs[i], match="species", _export=False
        )
        factors = [
            pysb.Expression(_free(model, f"{rule.name}_{name}"), rate * amount, _export=False)
            for name, rate in zip(("catalysed", "catalysed_reverse"), rates)
        ]
        try:
            # Built to be checked: PySB refuses a synthesis whose product is not concrete.
            rewritten = pysb.Rule(
                rule.name,
                RuleExpression(
                    ReactionPattern(lhs[:i] + lhs[i + 1 :]),
                    ReactionPattern(rhs[:i] + rhs[i + 1 :]),
                    rule.is_reversible,
                ),
                *factors,
                delete_molecules=rule.delete_molecules,
                _export=False,
            )
        except ValueError:
            continue
        for component in (amount, *factors):
            model.add_component(component)
        rule.rule_expression = rewritten.rule_expression
        rule.reactant_pattern = rewritten.reactant_pattern
        rule.product_pattern = rewritten.product_pattern
        rule.rate_forward, rule.rate_reverse = rewritten.rate_forward, rewritten.rate_reverse
        catalysts[rule.name] = amount
    return catalysts


def _bonds(complexes: list[pysb.ComplexPattern]) -> int:
    """How many bonds ``complexes`` write, each a number on two sites."""
    ends = 0
    for cp in complexes:
        for mp in cp.monomer_patterns:
            for value in mp.site_conditions.values():
                number = value[1] if isinstance(value, tuple) else value
                ends += isinstance(number, int) and not isinstance(number, bool)
    return ends // 2


def _free(model: pysb.Model, name: str) -> str:
    """``name``, or ``name`` with a number appended, whichever no component of ``model`` holds."""
    taken = {c.name for c in model.all_components()}
    candidate, i = name, 1
    while candidate in taken:
        i += 1
        candidate = f"{name}_{i}"
    return candidate


def reached(model: pysb.Model, reduction: Reduction) -> set[int]:
    """The species (indices into ``model.species``) the start reaches when a rewritten rule's
    reaction counts only once a species of its catalyst is reached, as the original rule's would."""
    present = {model.get_species_index(initial.pattern) for initial in model.initials}
    waiting = set(range(len(model.reactions)))
    while True:
        fire = {i for i in waiting if _enabled(model.reactions[i], reduction, present)}
        if not fire:
            return present
        for i in fire:
            present.update(model.reactions[i]["products"])
        waiting -= fire


def fired(model: pysb.Model, reduction: Reduction, present: set[int]) -> set[str]:
    """The rules with a reaction among the ``present`` species (:func:`reached`)."""
    return {
        name
        for rx in model.reactions
        if present.issuperset(rx["reactants"])
        for name in rx["rule"]
        if _catalyst_present(name, reduction, present)
    }


def _enabled(rx: dict, reduction: Reduction, present: set[int]) -> bool:
    return present.issuperset(rx["reactants"]) and any(
        _catalyst_present(name, reduction, present) for name in rx["rule"]
    )


def _catalyst_present(rule: str, reduction: Reduction, present: set[int]) -> bool:
    amount = reduction.catalysts.get(rule)
    return amount is None or not present.isdisjoint(amount.species)


def sizes(model: pysb.Model, reduction: Reduction) -> dict[str, dict[str, int]]:
    """The species and reactions of the expansion with and without the reduction, a reaction and
    its reverse counted once, as the SBML writes them. The original's are read off the reduced
    expansion: each reaction of a rewritten rule is one per species of its catalyst."""
    expanded = set()
    for rx in model.reactions:
        for name in rx["rule"]:
            amount = reduction.catalysts.get(name)
            for extra in [()] if amount is None else [(j,) for j in amount.species]:
                expanded.add(
                    (tuple(sorted(rx["reactants"] + extra)), tuple(sorted(rx["products"] + extra)))
                )
    species = len(model.species)
    return {
        "before_reduction": {
            "species": species,
            "reactions": len({frozenset((key, key[::-1])) for key in expanded}),
        },
        "after_reduction": {"species": species, "reactions": len(model.reactions_bidirectional)},
    }
