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
``k·[s]·[s]`` with no symmetry factor. Unchanged is as BioNetGen reads it, which pairs the
molecules of one type in the order they are written across each side, not complex by complex:
in ``C8(off) + C8(on) >> p18() + C8(on)`` the C8 that stays is the first one, turned on, and the
second is consumed, so neither complex is unchanged; read by place, the second looked like a
catalyst, and the rewrite carried the wrong molecule's partners over, up to 25-fold wrong (the
#115 review). A rule with two unchanged complexes keeps its form (``E + E + A >> E + B + E``:
BioNetGen counts the two by symmetry, at half the rate), as does a binding, a two-step mechanism,
whose steps bind and release, and a ligand that binds its receptor while its free pool stays as
it was (``nondepleting``). The rule keeps its name, so a reaction still says which rule and row
it came from.

**Independent sites are tracked apart.** Once no catalyst is in a rule, a monomer that rules only
turn from one form into another (``M(a='u') >> M(a='p')``) needs its sites together only where a
rule changes one while it reads another on the same molecule. Its sites fall into blocks that no
rule couples, and each block becomes a monomer of its own (``M_a``, ``M_b``...), started at the
whole monomer's total and changed by the rules on its sites. Each block's dynamics read the others
only through global amounts, so ``[M(a, b)] = [M_a(a)]·[M_b(b)] / M_tot`` at all times if it
holds at the start, which a single start form gives: a sum over ``k`` blocks' forms is the
product of the ``k`` block amounts over ``M_tot^(k-1)`` (Borisov et al. 2005; Conzelmann et al.
2006). Every amount of the whole monomer is written so: a readout, a rate's ``amount`` or
modifier, a catalyst's rate factor. The power is written as each further block's fraction of its
own total (``M_a(a)·(M_b(b)/M_b_tot)``, all totals equal), which writes no exponent, a literal
:mod:`literals` would charge as a parameter.

The product holds only for a monomer whose total is constant and whose forms only rules set, so
one stays whole when any of these fails (``whole`` names those the problem sets):

- every rule that mentions it turns one molecule of it into one, alone (no synthesis,
  degradation or conversion; no catalyst a rule could not take out, no second molecule);
- it has no binding site (a domain-oriented split of complexes, Conzelmann et al. 2006, is not
  done here);
- it starts in exactly one form, nonzero (several start forms generally do not factorise, and a
  zero total leaves the product 0/0);
- no condition, input protocol or condition-set starting amount sets any of its forms.

ODEs only: the product is not exact for stochastic simulation (Borisov et al. 2008), which no
route here does.

**What the start reaches is still decided by the catalyst.** A rewritten rule fires on its
substrate whether or not any catalyst exists, so BioNetGen's expansion may reach species the
original never would. :func:`reached` decides what the start reaches as the original rules would:
a rewritten rule's reaction counts only once its catalyst is reached. Every rule of a model the
statements route accepts fires (:mod:`problem_sbml`, assumption 3), and then the two expansions
are the same species, the whole monomer's forms being the products of its blocks'.

Its own module because none of it is about the PEtab problem: it is the same model, simulated
over fewer reactions and species (Borisov et al. 2005, Biophys J,
doi:10.1529/biophysj.105.060533; Conzelmann et al. 2006, BMC Bioinformatics 7:34; Borisov et al.
2008, IET Syst Biol 2:342).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from math import prod

import pysb
import sympy
from pysb.core import ReactionPattern, RuleExpression, as_complex_pattern
from pysb.pattern import match_complex_pattern


@dataclass
class Split:
    """One monomer tracked as independent blocks of its sites."""

    monomer: str
    #: Each block's monomer, in the whole one's site order; the first also holds the sites no rule
    #: mentions.
    blocks: list[pysb.Monomer]
    #: Each block's total, every species of it: each equal to the whole monomer's.
    totals: list[pysb.Observable]


@dataclass
class Amount:
    """The amount of one pattern of a split monomer: for each block the pattern reads, the amount
    of that block's part of it."""

    split: Split
    factors: dict[int, pysb.Observable]

    def value(self) -> sympy.Expr:
        first, *others = sorted(self.factors)
        value = self.factors[first]
        for i in others:
            value = value * self.factors[i] / self.split.totals[i]
        return value


@dataclass
class Reduction:
    """What :func:`reduce` rewrote."""

    #: Each rule's reactants as written, a catalyst taken out of it left out.
    written: dict[str, list[str]] = field(default_factory=dict)
    #: Each rule whose catalyst became a rate factor -> the catalyst as written, and its amount.
    catalysts: dict[str, tuple[str, pysb.Observable | Amount]] = field(default_factory=dict)
    #: Each monomer tracked as blocks -> how.
    splits: dict[str, Split] = field(default_factory=dict)


def reduce(model: pysb.Model, whole=frozenset()) -> Reduction:
    """Rewrite ``model``'s rules in place as the module docstring sets out, leaving the monomers
    named in ``whole`` whole; returns what changed."""
    written = {
        rule.name: [cp for cp in rule.reactant_pattern.complex_patterns if cp is not None]
        for rule in model.rules
    }
    catalysts = _catalysts_as_rates(model)
    splits, amounts = _splits(model, set(whole))
    reduction = Reduction(splits=splits)
    for name, reactants in written.items():
        at, amount = catalysts.get(name, (None, None))
        reduction.written[name] = [str(cp) for i, cp in enumerate(reactants) if i != at]
        if at is not None:
            reduction.catalysts[name] = (str(reactants[at]), amounts.get(amount, amount))
    return reduction


def _catalysts_as_rates(model: pysb.Model) -> dict[str, tuple[int, pysb.Observable]]:
    """Each rule rewritten -> its catalyst's place among its reactants, and the catalyst's
    amount."""
    catalysts = {}
    for rule in model.rules:
        rates = [rule.rate_forward] + ([rule.rate_reverse] if rule.is_reversible else [])
        if rule.move_connected or any(isinstance(r, pysb.Expression) and r.is_local for r in rates):
            continue
        # INDRA pads a side with None for Kappa (`E() + None >> E() + X()`): no complex.
        lhs = [cp for cp in rule.reactant_pattern.complex_patterns if cp is not None]
        rhs = [cp for cp in rule.product_pattern.complex_patterns if cp is not None]
        unchanged = _unchanged(lhs, rhs)
        if len(unchanged) != 1:
            continue
        ((i, j),) = unchanged
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
                    ReactionPattern(rhs[:j] + rhs[j + 1 :]),
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
        _rewrite(rule, rewritten.reactant_pattern, rewritten.product_pattern)
        rule.rate_forward, rule.rate_reverse = rewritten.rate_forward, rewritten.rate_reverse
        catalysts[rule.name] = (i, amount)
    return catalysts


def _rewrite(rule: pysb.Rule, lhs: ReactionPattern, rhs: ReactionPattern) -> None:
    rule.rule_expression = RuleExpression(lhs, rhs, rule.is_reversible)
    rule.reactant_pattern, rule.product_pattern = lhs, rhs


def _unchanged(lhs: list, rhs: list) -> list[tuple[int, int]]:
    """Each (reactant, product) pair of complexes BioNetGen carries over unchanged: it pairs the
    molecules of one type in the order they are written across each side, so a complex is unchanged
    when every one of its molecules is paired with the same molecule of an equal complex. Removing
    such a pair keeps the order, and so the pairing, of every other molecule."""
    written: dict[tuple[str, str], list[tuple[int, int]]] = {}
    for side, complexes in (("lhs", lhs), ("rhs", rhs)):
        for i, cp in enumerate(complexes):
            for m, mp in enumerate(cp.monomer_patterns):
                written.setdefault((side, mp.monomer.name), []).append((i, m))
    paired = {}
    for (side, monomer), places in written.items():
        if side == "lhs":
            paired |= dict(zip(places, written.get(("rhs", monomer), [])))
    return [
        (i, j)
        for i, cp in enumerate(lhs)
        for j, other in enumerate(rhs)
        if str(cp) == str(other)
        and all(paired.get((i, m)) == (j, m) for m in range(len(cp.monomer_patterns)))
    ]


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


def _splits(model: pysb.Model, whole: set[str]):
    """Split each monomer that can be; returns the splits, and the amount each observable of a
    split monomer now is, which every expression reads in its place."""
    splits = {}
    for monomer in list(model.monomers):
        blocks = None if monomer.name in whole else _blocks(model, monomer)
        if blocks is not None and len(blocks[0]) > 1:
            splits[monomer.name] = _split(model, monomer, *blocks)
    amounts = {}
    for obs in list(model.observables):
        mp = _one_molecule(obs.reaction_pattern)
        if mp is not None and mp.monomer.name in splits:
            amounts[obs] = _amount(model, splits[mp.monomer.name], mp, obs.name, obs.match)
    for expression in model.expressions:
        expression.expr = expression.expr.xreplace({o: a.value() for o, a in amounts.items()})
    return splits, amounts


def _blocks(model: pysb.Model, monomer: pysb.Monomer) -> tuple[list[list[str]], set[str]] | None:
    """``monomer``'s sites in the blocks no rule couples, and the sites no rule changes, which
    join the first block, where they add no species; ``None`` when it must stay whole (the module
    docstring says when)."""
    if any(site not in monomer.site_states for site in monomer.sites):
        return None
    starts = [i for i in model.initials if monomer in _monomers(i.pattern)]
    if len(starts) != 1 or len(starts[0].pattern.monomer_patterns) != 1:
        return None
    for obs in model.observables:
        if (
            monomer in _monomers(obs.reaction_pattern)
            and _one_molecule(obs.reaction_pattern) is None
        ):
            return None
    parent = {site: site for site in monomer.sites}

    def root(site: str) -> str:
        while parent[site] != site:
            site = parent[site]
        return site

    changed: set[str] = set()
    for rule in model.rules:
        sides = [_molecules(rule.reactant_pattern), _molecules(rule.product_pattern)]
        if not any(mp.monomer is monomer for side in sides for mp in side):
            continue
        if rule.delete_molecules or any(
            len(side) != 1 or side[0].monomer is not monomer for side in sides
        ):
            return None
        before, after = (side[0].site_conditions for side in sides)
        sites = sorted(set(before) | set(after))
        changed |= {s for s in sites if before.get(s) != after.get(s)}
        for site in sites[1:]:
            parent[root(site)] = root(sites[0])
    groups: dict[str, list[str]] = {}
    for site in monomer.sites:
        groups.setdefault(root(site), []).append(site)
    blocks = [sites for sites in groups.values() if changed.intersection(sites)]
    if not blocks:
        return None
    rest = {site for sites in groups.values() if not changed.intersection(sites) for site in sites}
    blocks[0] = [site for site in monomer.sites if site in rest.union(blocks[0])]
    return blocks, rest


def _split(
    model: pysb.Model, monomer: pysb.Monomer, blocks: list[list[str]], unchanged: set[str]
) -> Split:
    """Declare a monomer per block, named after the sites rules change in it, start each in the
    whole one's start form, and move each rule onto the block that holds the sites it reads."""
    parts = []
    for sites in blocks:
        part = pysb.Monomer(
            _free(model, "_".join([monomer.name, *(s for s in sites if s not in unchanged)])),
            sites,
            {s: monomer.site_states[s] for s in sites},
            _export=False,
        )
        model.add_component(part)
        parts.append(part)
    owner = {site: part for part in parts for site in part.sites}
    for rule in model.rules:
        sides = [_molecules(rule.reactant_pattern), _molecules(rule.product_pattern)]
        if sides[0] and sides[0][0].monomer is monomer:
            _rewrite(
                rule,
                *(
                    ReactionPattern([pysb.ComplexPattern([_on(owner, parts, side[0])], None)])
                    for side in sides
                ),
            )
    (start,) = [i for i in model.initials if monomer in _monomers(i.pattern)]
    model.initials.remove(start)
    state = start.pattern.monomer_patterns[0].site_conditions
    for part in parts:
        model.add_initial(
            pysb.Initial(
                part(**{s: state[s] for s in part.sites}), start.value, start.fixed, _export=False
            )
        )
    totals = []
    for part in parts:
        total = pysb.Observable(_free(model, f"{part.name}_total"), part(), _export=False)
        model.add_component(total)
        totals.append(total)
    return Split(monomer.name, parts, totals)


def _on(owner: dict, parts: list[pysb.Monomer], mp: pysb.MonomerPattern) -> pysb.MonomerPattern:
    """``mp``, a pattern of the whole monomer, as one of the block that holds its sites."""
    (part,) = {owner[s] for s in mp.site_conditions} or {parts[0]}
    return part(**mp.site_conditions)


def _amount(model: pysb.Model, split: Split, mp, name: str, match: str = "molecules") -> Amount:
    """The amount of ``mp``, a pattern of the split monomer: an observable of each block whose
    sites it reads, named after ``name``."""
    factors = {}
    for i, part in enumerate(split.blocks):
        own = {s: v for s, v in mp.site_conditions.items() if s in part.sites}
        if own:
            factor = pysb.Observable(
                _free(model, f"{name}_{part.name}"), part(**own), match=match, _export=False
            )
            model.add_component(factor)
            factors[i] = factor
    return Amount(split, factors or {0: split.totals[0]})


def amount(model: pysb.Model, reduction: Reduction, pattern, name: str):
    """A component ``name`` of ``model``: the amount of ``pattern``, written over the model as
    the rows wrote it. An observable, or across blocks of a split monomer, the expression of the
    product (the module docstring)."""
    mp = _one_molecule(as_complex_pattern(pattern))
    split = reduction.splits.get(mp.monomer.name) if mp is not None else None
    if split is None:
        component = pysb.Observable(name, pattern, _export=False)
    else:
        owners = [part for part in split.blocks if set(mp.site_conditions) & set(part.sites)]
        if len(owners) <= 1:
            part = (owners or split.blocks)[0]
            component = pysb.Observable(name, part(**mp.site_conditions), _export=False)
        else:
            component = pysb.Expression(
                name, _amount(model, split, mp, name).value(), _export=False
            )
    model.add_component(component)
    return component


def reads(amount, present: set[int]) -> bool:
    """Whether ``amount`` (an observable, an expression over observables, an :class:`Amount`)
    reads a ``present`` species: every observable it multiplies does."""
    if isinstance(amount, Amount):
        observables = amount.factors.values()
    elif isinstance(amount, pysb.Expression):
        observables = amount.expr.atoms(pysb.Observable)
    else:
        observables = [amount]
    return all(not present.isdisjoint(o.species) for o in observables)


def reached(model: pysb.Model, reduction: Reduction) -> set[int]:
    """The species (indices into ``model.species``) the start reaches when a rewritten rule's
    reaction counts only once a species of its catalyst is reached, as the original rule's would."""
    # A start the expansion holds no species for (none, when nothing fires from the first starts
    # and the made ones are seeded after) reaches nothing.
    present = {model.get_species_index(initial.pattern) for initial in model.initials} - {None}
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
    return rule not in reduction.catalysts or reads(reduction.catalysts[rule][1], present)


def unformed(model: pysb.Model, reduction: Reduction, rule: str, present: set[int]) -> str | None:
    """A reactant of ``rule``, as written, that no ``present`` species matches: its catalyst
    first; ``None`` when each has one."""
    if rule in reduction.catalysts and not _catalyst_present(rule, reduction, present):
        return reduction.catalysts[rule][0]
    species = [model.species[i] for i in sorted(present)]
    reactants = [cp for cp in model.rules[rule].reactant_pattern.complex_patterns if cp is not None]
    for text, cp in zip(reduction.written[rule], reactants):
        if not any(match_complex_pattern(cp, sp) for sp in species):
            return text
    return None


def sizes(model: pysb.Model, reduction: Reduction, present: set[int]) -> dict:
    """The species and reactions of the expansion with and without the reduction, a reaction and
    its reverse counted once, as the SBML writes them, and each split monomer's blocks.

    The original's are read off the reduced expansion, with no second BioNetGen run: a split
    monomer's forms are the products of its blocks' forms, a reaction of one block stands for one
    per form of the others, and a rewritten rule's for one per species of its catalyst."""
    block_of, forms = {}, {}
    for split in reduction.splits.values():
        forms[split.monomer] = [len(present.intersection(t.species)) for t in split.totals]
        block_of |= {j: (split.monomer, i) for i, t in enumerate(split.totals) for j in t.species}
    groups: dict[frozenset, tuple[dict, set]] = {}
    for rx in model.reactions:
        if not present.issuperset(rx["reactants"]):
            continue
        forward = (tuple(sorted(rx["reactants"])), tuple(sorted(rx["products"])))
        catalysts = groups.setdefault(frozenset((forward, forward[::-1])), (rx, set()))[1]
        for name in rx["rule"]:
            catalysts |= (
                _species(reduction.catalysts[name][1], present)
                if name in reduction.catalysts
                else {None}
            )
    reactions = 0
    for rx, catalysts in groups.values():
        blocks = [block_of[j] for j in rx["reactants"] + rx["products"] if j in block_of]
        others = 1
        if blocks:
            monomer, i = blocks[0]
            others = prod(n for b, n in enumerate(forms[monomer]) if b != i)
        reactions += others * len(catalysts)
    return {
        "before_reduction": {
            "species": len(present - block_of.keys()) + sum(map(prod, forms.values())),
            "reactions": reactions,
        },
        "after_reduction": {
            "species": len(model.species),
            "reactions": len(model.reactions_bidirectional),
        },
    } | (
        {
            "split": {
                name: {part.name: list(part.sites) for part in split.blocks}
                for name, split in reduction.splits.items()
            }
        }
        if reduction.splits
        else {}
    )


def _species(amount, present: set[int]) -> set:
    """The species of the original expansion ``amount`` sums: a split monomer's as the tuple of
    its blocks' species."""
    if isinstance(amount, Amount):
        blocks = [
            sorted(present.intersection(amount.factors.get(i, total).species))
            for i, total in enumerate(amount.split.totals)
        ]
        return {(amount.split.monomer, forms) for forms in product(*blocks)}
    return present.intersection(amount.species)


def _molecules(pattern: ReactionPattern) -> list:
    return [mp for cp in pattern.complex_patterns if cp is not None for mp in cp.monomer_patterns]


def _monomers(pattern) -> set:
    if isinstance(pattern, ReactionPattern):
        return {mp.monomer for mp in _molecules(pattern)}
    return {mp.monomer for mp in pattern.monomer_patterns}


def _one_molecule(pattern):
    """The one molecule ``pattern`` (a reaction or complex pattern) holds, else ``None``."""
    molecules = (
        _molecules(pattern) if isinstance(pattern, ReactionPattern) else pattern.monomer_patterns
    )
    return molecules[0] if len(molecules) == 1 else None
