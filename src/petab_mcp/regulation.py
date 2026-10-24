"""Whether a row's rate moves with each regulator its statement states, as the statement claims.

A regulator ("RAF phosphorylates MEK, inhibited by sorafenib") is a claim about the statement's own
reaction: which way the rate at which it makes its products moves as the regulator rises. That rate
is the row's net rate J (:func:`net_rate`), read off the model the rows build, BioNetGen's expansion
of every row's rules from the start (:func:`problem_sbml.to_antimony`), on the reactions the row's
own rules make. So it is the rate of the row as built: its own agents in the states the model gives
them, its bindings, and every species a regulator's pattern reads, however the mechanism spells
it. Two names for one agent, an agent stated twice, an agent that is also a role: each reads the
same species, so each is one amount in J.

J is the reaction's initial rate, the way an enzymologist measures inhibition: the row's
intermediates at steady state, its products absent. Which species are which is the statement's to
say (:data:`statements._REACTIONS`), not the rules': an enzyme parked in a form that makes nothing
and a product released by a reversible step sit alike, each on the right of a ``|``, and a substrate
a step back makes again looks like an intermediate. So a product is what the statement makes, or
what the row's reactions make and none uses, held at 0; a reactant is what the statement consumes,
or what the reactions use and never make; a regulator's species, and a partner's that holds no
role's agent, are held at their amounts, and read by its pattern as an amount of its own, a
product's too; every other species of the reactions is an intermediate (the enzyme and its forms). A
pattern matching every form of the enzyme reads its conserved total, held as well; one matching only
some reads an amount the steady state solves for, not one held apart from the reaction, so there is
no J. With the intermediates linear in each rate (one intermediate per step, as in every enzyme
mechanism) their steady state is one linear solve, each conserved total a name of its own (King and
Altman's), and J is a product's net rate, or a removed reactant's loss. A step into a dead end, a
step back, a branch: each moves J as it really does, which no rule on a step's own rate can say.

A direction counts as shown where sympy proves it (:func:`mechanisms._direction`) and J, read at
:data:`SAMPLES` points across its names' ranges, moves no other way (:func:`sampled`); a refusal
rests only on what is shown so.
"""

from __future__ import annotations

import logging
import random
import re
from copy import deepcopy
from dataclasses import dataclass, field, replace
from time import monotonic

import numpy as np
import pysb
import sympy
from indra.assemblers.pysb.assembler import _n
from indra.statements import (
    Activation,
    BoundCondition,
    Complex,
    Conversion,
    DecreaseAmount,
    IncreaseAmount,
    Inhibition,
    Modification,
    RegulateActivity,
    modclass_to_inverse,
)
from pysb.core import time

from .errors import MechanismError
from .statements import REGULATOR_NAME, REGULATORS, named_agents

logger = logging.getLogger(__name__)

#: The most terms the intermediates' steady state may have (:func:`_arborescences`): a row past it
#: has no net rate to read.
NET_TERMS = 400

#: The most intermediates and steps a row may have for its steady state to be solved for.
NET_FORMS, NET_STEPS = 16, 64

#: Seconds a row's regulators are signed in, at most: past it, one not yet signed is not shown.
ROW_SECONDS = 30.0

#: How many points :func:`sampled` reads J at.
SAMPLES = 256

#: An observable a row's rate reads a stated regulator by: named after the name the rate reads it
#: by (``r1_inhibited_by``, ``r1_rate0_0_inhibited_by_1``).
_READ = re.compile(rf"_({REGULATOR_NAME.pattern})$")

#: ``time``, never negative.
TIME = sympy.Symbol("time", nonnegative=True)


@dataclass(frozen=True)
class NetRate:
    """A row's net rate J and, for each regulator its statement states (by kind and position,
    ``inhibited_by_0``), the amounts in J its pattern reads: none where nothing in the model forms
    it."""

    value: sympy.Expr
    regulators: dict[str, tuple[sympy.Symbol, ...]]
    #: J over its steps' coefficients, each a name of its own, and what each name is: the small
    #: form :func:`_through_steps` signs J by.
    over_steps: sympy.Expr | None = None
    steps: dict[sympy.Symbol, sympy.Expr] = field(default_factory=dict)
    #: Each name in J that is a parameter, and its id: what a version's table signs it by.
    parameters: dict[str, str] = field(default_factory=dict)


def position(name: str) -> str:
    """The stated position a regulator name stands for: ``inhibited_by``, the one of its kind, is
    ``inhibited_by_0``."""
    return name if name[-1].isdigit() else f"{name}_0"


def net_rate(model: pysb.Model, rules, stmt) -> NetRate:
    """The net rate J of the row whose ``rules`` built ``stmt``, on ``model`` expanded (see the module
    docstring). Raises :class:`MechanismError` saying why there is none: a reaction the row shares
    with another row, an intermediate meeting another in one step, a steady state too large or not
    unique, or products made at rates no constant ratio relates, two reactions rather than one."""

    from .mechanisms import _vanishes
    from .statements import _products, _reactants

    rules = set(rules)
    reactions = [rx for rx in model.reactions if set(rx["rule"]) & rules]
    if not reactions:
        raise MechanismError("its rules make no reaction in the model")
    if any(set(rx["rule"]) - rules for rx in reactions):
        raise MechanismError("BioNetGen merged one of its reactions with another row's")
    read: dict[str, set[pysb.Observable]] = {}
    for rule in (model.rules[name] for name in rules):
        for obs in _observables(rule.rate_forward) | _observables(rule.rate_reverse):
            if found := _READ.search(obs.name):
                read.setdefault(position(found.group(1)), set()).add(obs)
    regulator_species = {i for observables in read.values() for o in observables for i in o.species}
    made = {i for rx in reactions for i in rx["products"]}
    used = {i for rx in reactions for i in rx["reactants"]}

    effects = _effects(model, stmt)
    products = {i for i, e in effects.items() if e > 0} | (made - used)
    # A reactant a step back makes again (`obj_to >> obj_from`) is still the statement's; of the
    # forms an agent written without states matches, those the row makes are on the way (`S` of
    # "E3 decreases S", ubiquitinated before it is degraded).
    stated = set()
    for agent in _reactants(stmt):
        found = _matching(model, agent)
        stated |= found - made if len(found) > 1 and found - made else found
    # Else what the reactions use and never make, however their lines are written.
    reactants = (stated | (used - made)) - products
    # A partner of no role's agent (the `C` of `E{~C}`, binding the enzyme) is a ligand, held at
    # its amount as a regulator is; the intermediates are the forms of the statement's agents.
    roles = {_n(agent.name) for agent in stmt.real_agent_list()}
    ligands = {
        i for i in made & used
        if not {mp.monomer.name for mp in model.species[i].monomer_patterns} & roles
    }
    # A form the row makes and uses on its way (its enzyme, a complex of its own) stays an
    # intermediate though a regulator's pattern matches it: held, it would freeze that form.
    own = (made & used) - products - reactants - ligands
    intermediates = sorted(
        own | ((made | used) - products - reactants - regulator_species - ligands)
    )
    within = sympy.Matrix([
        [rx["products"].count(i) - rx["reactants"].count(i) for rx in reactions]
        for i in intermediates
    ]) if intermediates else sympy.zeros(0, len(reactions))
    conserved = within.T.nullspace() if intermediates else []
    totals = [sympy.Symbol(f"total{k}", positive=True) for k in range(len(conserved))]
    # A regulator reads those forms at their steady state: where they make up conserved totals
    # (`MEK` of "MEK phosphorylates ERK", free and bound), those totals, held; else an amount the
    # net rate solves for, no amount held apart from the reaction, refused.
    own_reads: dict[pysb.Observable, sympy.Expr] = {}
    for stated_at, observables in sorted(read.items()):
        for o in observables:
            weights = [
                sum(c for c, j in zip(o.coefficients, o.species) if j == i) for i in intermediates
            ]
            if not any(weights):
                continue
            if (reads := _as_totals(weights, conserved, totals)) is None:
                inside = sorted(set(o.species) & own)
                raise MechanismError(_own_form(model, stated_at, observables, inside))
            own_reads[o] = reads

    def held(i: int, regulator: bool) -> sympy.Symbol:
        sign = {"nonnegative": True} if regulator else {"positive": True}
        return sympy.Symbol(f"[{model.species[i]}]", **sign)

    amounts: dict[int, sympy.Expr] = {}
    for i in made | used | {i for rate in _all_observables(model, rules) for i in rate.species}:
        if i in products:
            amounts[i] = sympy.Integer(0)
        elif i in intermediates:
            amounts[i] = sympy.Symbol(f"__x{i}")
        else:
            amounts[i] = held(i, i in regulator_species)
    regulator_amounts = {i: held(i, True) for i in regulator_species - own}
    regulator_observables = {o for observables in read.values() for o in observables}
    names = {p: sympy.Symbol(p.name, positive=True) for p in model.parameters}

    def value(rate) -> sympy.Expr:
        while expressions := rate.atoms(pysb.Expression):
            rate = rate.xreplace({e: e.expr for e in expressions})
        rate = rate.xreplace({
            o: own_reads.get(o, 0) + sympy.Add(*(
                c * (regulator_amounts if o in regulator_observables else amounts).get(i, 0)
                for c, i in zip(o.coefficients, o.species)
            ))
            for o in rate.atoms(pysb.Observable)
        })
        species = {sympy.Symbol(f"__s{i}"): amount for i, amount in amounts.items()}
        return rate.xreplace(species).xreplace(names).xreplace({time: TIME})

    if len(intermediates) > NET_FORMS or len(reactions) > NET_STEPS:
        raise MechanismError("its intermediates' steady state is too large to solve symbolically")
    # A removal's way back, a step that makes what it removes and leaves its intermediates as they
    # were (`None >> obj`, synthesis), is silent as a product's way back is with the product
    # absent: a regulator of synthesis is a synthesis statement's, as one of degradation is.
    removing = {i for i, e in effects.items() if e < 0} & reactants
    back = [
        bool(removing)
        and any(rx["products"].count(i) > rx["reactants"].count(i) for i in removing)
        and all(rx["products"].count(i) == rx["reactants"].count(i) for i in intermediates)
        for rx in reactions
    ]
    rates = [sympy.Integer(0) if b else value(rx["rate"]) for rx, b in zip(reactions, back)]
    involved = sorted(made | used)
    unknown = [amounts[i] for i in intermediates]
    # Linear in the intermediates: at most one among a step's reactants, and none in its rate
    # constant. Read off the network rather than expanded: expanding a large rate never ended.
    # Each step's coefficient, its intermediate apart, is solved for as a name of its own: sympy
    # cancelled the rates themselves for minutes (the review of #128 timed 400 s on two steps).
    coefficients, opaque = {}, []
    for k, (rx, rate) in enumerate(zip(reactions, rates)):
        inside = [amounts[i] for i in rx["reactants"] if i in intermediates]
        if rate == 0:
            opaque.append(sympy.Integer(0))
            continue
        coefficient = rate / sympy.Mul(*inside)
        if len(inside) > 1 or coefficient.free_symbols & set(unknown):
            raise MechanismError(
                "two of its intermediates meet in one step, or one rate reads an intermediate's "
                "amount, so their steady state has no closed form"
            )
        coefficients[name := sympy.Symbol(f"__c{k}")] = coefficient
        opaque.append(name * sympy.Mul(*inside))
    if _arborescences(reactions, intermediates) > NET_TERMS:
        raise MechanismError("its intermediates' steady state is too large to solve symbolically")
    change = {i: sympy.Integer(0) for i in involved}
    for rx, rate in zip(reactions, opaque):
        for i in rx["reactants"]:
            change[i] -= rate
        for i in rx["products"]:
            change[i] += rate
    equations = [change[i] for i in intermediates] + [
        sum(c * x for c, x in zip(vector, unknown)) - total
        for vector, total in zip(conserved, totals)
    ]
    solution = list(sympy.linsolve(equations, unknown))
    if unknown and (len(solution) != 1 or any(set(unknown) & v.free_symbols for v in solution[0])):
        raise MechanismError("its intermediates have no unique steady state")
    # Solved for regulators of any amount but 0, where a step only a regulator drives stops: an
    # enzyme it activates for good keeps whatever form it starts in, and J is the start's.
    if unknown and not _unique_without(
        [e.xreplace(coefficients) for e in equations], unknown, set(regulator_amounts.values())
    ):
        raise MechanismError(
            "its intermediates have no unique steady state where a regulator is absent"
        )
    steady = dict(zip(unknown, solution[0])) if unknown else {}
    # Kept as built, its factors apart, for the sign prover; cancelled only to tell zero. A
    # regulator held at its amount moves with no reaction of the row's but the one it is.
    held_regulators = regulator_species - products - set(effects)
    over_steps = {
        i: change[i].xreplace(steady)
        for i in involved if i not in intermediates and i not in held_regulators
    }
    net = {i: rate.xreplace(coefficients) for i, rate in over_steps.items()}
    moving = {i for i, rate in net.items() if not _vanishes(rate)}
    # A stated agent is every form of it the row makes or takes, alone or in a complex (ERK
    # phosphorylated at T185, Y187 either way, bound to DUSP or not): one lump, its net rate the sum
    # over its forms.
    made_lump = _containing(model, [a for a, e in _products(stmt) if e > 0]) & set(net)
    removed_lump = _containing(model, [a for a, e in _products(stmt) if e < 0]) & set(net)
    stated = stated | (_containing(model, _reactants(stmt)) & set(net) - made_lump)
    made_now = sorted(made_lump & moving) or [i for i in sorted(products) if i in moving]
    removed = sorted(removed_lump & moving)
    regulated = {
        stated: tuple(
            regulator_amounts[i] for i in sorted({i for o in observables for i in o.species} - own)
        ) + tuple(sorted(
            {t for o in observables for t in own_reads.get(o, sympy.S.Zero).free_symbols}, key=str
        ))
        for stated, observables in read.items()
    }
    if not made_now and not removed:
        stated_products = {i for i, e in effects.items() if e > 0}
        making = [rx for rx in reactions if set(rx["products"]) & stated_products]
        if making and all(set(rx["reactants"]) & products for rx in making):
            # Every step that makes a product takes one (`A + B >> 2B`): with its products absent
            # the reaction does not run, and J is 0 whatever its regulators.
            return NetRate(sympy.Integer(0), regulated)
        raise MechanismError("it makes nothing from its reactants alone")
    lump = made_lump if made_lump & moving else (
        removed_lump if removed_lump & moving else {made_now[0]})
    sign = 1 if lump is not removed_lump else -1
    j = sign * sympy.Add(*(net[i] for i in sorted(lump)))
    j_over_steps = sign * sympy.Add(*(over_steps[i] for i in sorted(lump)))
    if j.has(sympy.zoo, sympy.nan, sympy.oo, -sympy.oo):
        raise MechanismError("its net rate is not finite with its products absent")
    groups = [sorted(lump), sorted((stated - removed_lump) & set(net))]
    groups += [[i] for i in sorted(moving - lump - stated)]
    for group in (g for g in groups if g and set(g) & moving):
        if not _constant(sympy.Add(*(net[i] for i in group)), j):
            raise MechanismError(
                f"it changes {', '.join(str(model.species[i]) for i in sorted(lump))} and "
                f"{', '.join(str(model.species[i]) for i in group)} at rates no constant ratio "
                "relates: two reactions, with no one net rate"
            )
    return NetRate(
        j, regulated, j_over_steps,
        {c: e for c, e in coefficients.items() if c in j_over_steps.free_symbols},
        {p.name: p.name for p in model.parameters},
    )


def _as_totals(weights: list, conserved: list, totals: list) -> sympy.Expr | None:
    """The amount ``weights`` (over the intermediates) reads at steady state, where it is a sum
    of conserved ``totals`` with no negative weight; ``None`` otherwise."""
    if not conserved:
        return None
    try:
        solution, _ = sympy.Matrix.hstack(*conserved).gauss_jordan_solve(sympy.Matrix(weights))
    except ValueError:  # no sum of them
        return None
    if any(x.is_negative is not False for x in solution):
        return None
    return sympy.Add(*(x * t for x, t in zip(solution, totals)))


def _own_form(model: pysb.Model, stated_at: str, observables, inside: list[int]) -> str:
    """Why a regulator whose pattern matches forms of the row's own (``inside``) but no total of
    them gives no net rate, and how to state one that does."""
    mine = {mp.monomer.name for o in observables for cp in o.reaction_pattern.complex_patterns
            for mp in cp.monomer_patterns}
    partners = sorted({mp.monomer.name for i in inside for mp in model.species[i].monomer_patterns}
                      - mine)
    alone = any(len(model.species[i].monomer_patterns) == 1 for i in inside)
    hint = (
        "state the agent with no state or bond, every form of it" if alone or not partners
        else f"write on it the bond it lacks there (`~{partners[0]}`, not bound to {partners[0]})"
    )
    return (
        f"the regulator stated at `{stated_at}` reads "
        f"{', '.join(str(model.species[i]) for i in inside)}, a form this row makes and uses on "
        "its way, which the net rate solves for rather than holds: a regulator reads an amount "
        "held apart from the reaction, or a total the row's steps conserve (its enzyme, free and "
        f"bound), so state one that reads either ({hint})"
    )


def _matching(model: pysb.Model, agent) -> set[int]:
    """The species that are ``agent`` exactly, its molecules and no more."""
    from pysb.pattern import match_complex_pattern

    from .mechanisms import agent_pattern

    try:
        pattern = pysb.ComplexPattern(agent_pattern(model, agent), None)
    except (MechanismError, KeyError):  # an agent the model never forms
        return set()
    return {
        i for i, species in enumerate(model.species)
        if len(species.monomer_patterns) == len(pattern.monomer_patterns)
        and match_complex_pattern(pattern, species)
    }


def _containing(model: pysb.Model, agents) -> set[int]:
    """The species holding one of ``agents``, alone or with partners."""
    from pysb.pattern import match_complex_pattern

    from .mechanisms import agent_pattern

    found = set()
    for agent in agents:
        try:
            pattern = pysb.ComplexPattern(agent_pattern(model, agent), None)
        except (MechanismError, KeyError):  # an agent the model never forms
            continue
        found |= {i for i, species in enumerate(model.species)
                  if match_complex_pattern(pattern, species)}
    return found


def _effects(model: pysb.Model, stmt) -> dict[int, int]:
    """The species ``stmt`` makes, each ``1``, or takes away, ``-1`` (:func:`statements._products`)."""
    from .statements import _products

    effects = {i: effect for agent, effect in _products(stmt) for i in _matching(model, agent)}
    if isinstance(stmt, Complex):
        # Its members joined in one complex, in a chain or around one of them alike.
        members = sorted(_n(m.name) for m in stmt.members)
        effects |= {
            i: 1 for i, species in enumerate(model.species)
            if sorted(mp.monomer.name for mp in species.monomer_patterns) == members
        }
    return effects


def _arborescences(reactions, intermediates) -> int:
    """How many terms the intermediates' steady state has, by King and Altman's count: the trees
    of steps that reach each intermediate from all the others, counted by the matrix-tree theorem
    on the steps from one intermediate to another. What the solve costs grows with it, the
    expression past any bound (six enzyme forms all joined took over 15 minutes)."""
    index = {i: k for k, i in enumerate(intermediates)}
    laplacian = np.zeros((len(index), len(index)))
    group = list(range(len(index)))

    def find(k: int) -> int:
        while group[k] != k:
            k = group[k]
        return k

    for rx in reactions:
        before = [index[i] for i in rx["reactants"] if i in index]
        for b in before:
            for a in (index[i] for i in rx["products"] if i in index):
                if a != b:
                    laplacian[b, b] += 1
                    laplacian[b, a] -= 1
                    group[find(a)] = find(b)
    total = 0
    for root in range(len(index)):
        rest = [k for k in range(len(index)) if k != root and find(k) == find(root)]
        total += round(abs(np.linalg.det(laplacian[np.ix_(rest, rest)]))) if rest else 1
    return total


def _unique_without(equations, unknown, regulators) -> bool:
    """Whether ``equations``, linear in ``unknown``, still fix them with every one of
    ``regulators`` at 0, read at a point of positive values of every other name."""
    matrix, _ = sympy.linear_eq_to_matrix(equations, unknown)
    absent = matrix.xreplace({r: 0 for r in regulators})
    draw = random.Random(0)
    point = {s: 10 ** draw.uniform(-1, 1) for s in absent.free_symbols}
    values = np.array(absent.xreplace(point).evalf(), dtype=float)
    return np.linalg.matrix_rank(values) == len(unknown)


def _observables(rate) -> set[pysb.Observable]:
    """Every observable ``rate`` (a parameter, an expression, or None) reads, through the expressions
    it reads."""
    found, todo = set(), [rate] if isinstance(rate, pysb.Expression) else []
    while todo:
        for atom in todo.pop().expr.atoms(pysb.Observable, pysb.Expression):
            if isinstance(atom, pysb.Observable):
                found.add(atom)
            else:
                todo.append(atom)
    return found


def _all_observables(model: pysb.Model, rules) -> set[pysb.Observable]:
    return {
        o for name in rules
        for o in _observables(model.rules[name].rate_forward)
        | _observables(model.rules[name].rate_reverse)
    }


def _constant(part: sympy.Expr, whole: sympy.Expr) -> bool:
    """Whether ``part`` is ``whole`` times one number for every value of their names, by their
    values at eight points of positive ones drawn apart for each name, so a ratio of differences of
    names (``exp(a - b)``) is not read at points where they agree; a point where ``whole`` is 0 (a
    piecewise branch) says so only of ``part`` there. Cancelling the ratio to lowest terms took
    seconds on a large rate. Eight float points and a 1e-9 relative tolerance decide, so a ratio
    nearly but not exactly constant passes; only whether a row is one reaction rests on it (the
    two-reactions refusal), never a sign."""
    names = sorted(part.free_symbols | whole.free_symbols, key=str)
    if not names:
        return True
    draw = random.Random(0)
    ratios = []
    for _ in range(8):
        point = {s: 10 ** draw.uniform(-1, 1) for s in names}
        p, w = complex(part.evalf(subs=point)), complex(whole.evalf(subs=point))
        if p != p or w != w:
            return False
        if abs(w) <= 1e-300:
            if abs(p) > 1e-300:
                return False
            continue
        ratios.append(p / w)
    return all(abs(r - ratios[0]) <= 1e-9 * (1 + abs(ratios[0])) for r in ratios)


def direction(rate: NetRate, stated: str) -> int | None:
    """How J moves as the regulator at ``stated`` rises, as :func:`mechanisms._direction` shows it
    for each amount its pattern reads: ``0`` where it reads none J changes with, else the one way
    every one moves it; ``None`` where one cannot be shown or two disagree."""
    from .mechanisms import _direction

    moves = set()
    for symbol in rate.regulators.get(stated, ()):
        found = _direction(rate.value, symbol)
        moves.add(_through_steps(rate, symbol) if found is None else found)
    moves -= {0}
    if None in moves or len(moves) > 1:
        return None
    return moves.pop() if moves else 0


#: The most terms the numerator of :func:`_degree_one`'s derivative may expand to.
CHAIN_TERMS = 4096


def _degree_one(value: sympy.Expr, c: sympy.Symbol) -> int | None:
    """How ``value``, J over its steps' coefficients, moves with the coefficient ``c`` where it is
    of degree one in ``c`` above and below, as King and Altman's form of a network of single steps
    is: with ``value = (N1*c + N0)/(D1*c + D0)``, its derivative has the sign of
    ``N1*D0 - N0*D1``, expanded within :data:`CHAIN_TERMS` and signed term by term. ``None`` where
    it is not of degree one, or too large."""
    from .mechanisms import _sign, _terms, _vanishes

    if _terms(value) > CHAIN_TERMS:
        return None
    numerator, denominator = sympy.fraction(sympy.together(value))
    n1, d1 = sympy.diff(numerator, c), sympy.diff(denominator, c)
    if not (_vanishes(sympy.diff(n1, c)) and _vanishes(sympy.diff(d1, c))):
        return None
    n0, d0 = numerator.xreplace({c: 0}), denominator.xreplace({c: 0})
    if _terms(n1) * _terms(d0) + _terms(n0) * _terms(d1) > CHAIN_TERMS:
        return None
    return _sign(sympy.expand(n1 * d0 - n0 * d1))


def _through_steps(rate: NetRate, x: sympy.Symbol) -> int | None:
    """How J moves with ``x`` by the chain rule through its steps' coefficients: dJ/dx is the sum
    over the steps whose coefficient reads ``x`` of dJ/dc times dc/dx, each signed apart, J over
    its coefficients small (King and Altman's form) and each coefficient one step's rate. Shown
    where every such product has one sign; ``None`` otherwise, or where J reads ``x`` but through
    its coefficients."""
    from .mechanisms import _direction, _sign

    if rate.over_steps is None or x in rate.over_steps.free_symbols:
        return None
    signed = {}
    for c, coefficient in rate.steps.items():
        sign = _sign(coefficient)
        signed[c] = (sympy.Symbol(c.name, positive=True) if sign == 1
                     else sympy.Symbol(c.name, negative=True) if sign == -1 else c)
    over_steps = rate.over_steps.xreplace(signed)
    ways = set()
    for c, coefficient in rate.steps.items():
        if x not in coefficient.free_symbols:
            continue
        inner = _direction(coefficient, x)
        outer = _direction(over_steps, signed[c]) if inner else 0
        if inner and outer is None:
            outer = _degree_one(over_steps, signed[c])
        if inner is None or outer is None:
            return None
        ways.add(inner * outer)
    ways -= {0}
    return ways.pop() if len(ways) == 1 else (None if ways else 0)


def sampled(
    rate: NetRate,
    stated: str,
    domain: dict[str, tuple[float, float]] | None = None,
    confirm: frozenset[int] = frozenset(),
) -> set[int]:
    """The ways J moves as the regulator at ``stated`` rises a little, read at :data:`SAMPLES`
    points: ``1``, ``-1`` and ``0`` (no change, relative to J), each where some point shows it. Each
    name ranges over ``domain`` (name to ``(low, high)``, log-spaced where both are positive) or,
    lacking it, as its assumptions allow: a positive name over 1e-3 to 1e3, one never negative
    from 0 to 1e4, any other over -1e3 to 1e3. A point where J is not a finite real number reads
    nothing, nor one where J is too small for a float to tell apart from 0. A way in ``confirm``
    counts only where J, read again at those points to 50 digits, moves it too: J written as a
    difference of near-equal terms rounds to the wrong sign in floats."""
    symbols = sorted(rate.value.free_symbols, key=str)
    reads = [s for s in rate.regulators.get(stated, ()) if s in rate.value.free_symbols]
    if not reads:
        return {0}
    generator = np.random.default_rng(0)
    base = [_column(s, (domain or {}).get(s.name), generator) for s in symbols]
    f = sympy.lambdify(symbols, rate.value, "numpy")

    def at(columns) -> np.ndarray:
        with np.errstate(all="ignore"):
            try:
                out = np.asarray(f(*columns), dtype=complex) * np.ones(SAMPLES)
            except (ZeroDivisionError, OverflowError, ValueError, TypeError):
                return np.full(SAMPLES, np.nan, dtype=complex)
        return out

    j0 = at(base)
    found: set[int] = set()
    for symbol in reads:
        k = symbols.index(symbol)
        moved = list(base)
        moved[k] = base[k] + 1e-6 * np.maximum(np.abs(base[k]), 1.0)
        j1 = at(moved)
        real = (
            np.isfinite(j0) & np.isfinite(j1)
            & (np.abs(j0.imag) <= 1e-12 * np.maximum(1, np.abs(j0.real)))
            & (np.abs(j1.imag) <= 1e-12 * np.maximum(1, np.abs(j1.real)))
            # Where J underflows, what is left of it is rounding.
            & (np.maximum(np.abs(j0.real), np.abs(j1.real)) > 1e-250)
        )
        step, scale = (j1 - j0).real[real], 1e-10 * np.abs(j0.real[real])
        at_points = np.flatnonzero(real)
        for way, where in ((1, step > scale), (-1, step < -scale), (0, np.abs(step) <= scale)):
            hits = at_points[where]
            if way in confirm:
                hits = [h for h in hits[:8] if _precise_way(rate.value, symbols, base, moved, h) == way]
            if len(hits):
                found.add(way)
    return found


def _precise_way(value: sympy.Expr, symbols, base, moved, h: int) -> int:
    """How ``value`` moves from point ``h`` of ``base`` to that of ``moved``, read to 50 digits."""
    before = value.evalf(50, subs={s: base[k][h] for k, s in enumerate(symbols)})
    after = value.evalf(50, subs={s: moved[k][h] for k, s in enumerate(symbols)})
    change = complex(after - before)
    if change != change or abs(change.imag) > 1e-30 * (1 + abs(complex(before))):
        return 0
    if abs(change.real) <= 1e-30 * (1 + abs(complex(before))):
        return 0
    return 1 if change.real > 0 else -1


def _column(symbol: sympy.Symbol, bounds, generator) -> np.ndarray:
    """:data:`SAMPLES` values of ``symbol``, its bounds' ends among them."""
    if bounds is not None:
        low, high = (float(b) for b in bounds)
        ends = np.array([low, high])
        if low > 0 and np.isfinite(high):
            inner = 10 ** generator.uniform(np.log10(low), np.log10(high), SAMPLES - 2)
        else:
            low, high = max(low, -1e6), min(high, 1e6)
            inner = generator.uniform(low, high, SAMPLES - 2)
        return generator.permutation(np.concatenate([ends.clip(-1e6, 1e6), inner]))
    if symbol.is_positive:
        return 10 ** generator.uniform(-3, 3, SAMPLES)
    if symbol.is_nonnegative:
        values = 10 ** generator.uniform(-3, 4, SAMPLES)
        values[: SAMPLES // 10] = 0.0
        return generator.permutation(values)
    return generator.uniform(-1e3, 1e3, SAMPLES)


def shown(rate: NetRate, stated: str, domain=None) -> int | None:
    """How J moves as the regulator at ``stated`` rises, where sympy proves it and no point of
    :func:`sampled` reads against it: ``1``, ``-1``, ``0``; ``None`` where either fails. A proof the
    points contradict is logged: the prover or the names' ranges are wrong somewhere."""
    proven = direction(rate, stated)
    if proven is None:
        return None
    seen = sampled(rate, stated, domain, frozenset({1, -1} - {proven}))
    against = seen - {proven, 0} if proven else seen - {0}
    if against:
        logger.warning(
            "regulator %s: proven %s, but J moves %s at sampled points: %s",
            stated, proven, sorted(against), _short(rate.value),
        )
        return None
    return proven


#: The kind a regulator is recorded as, by the way J is shown to move with it.
RECORDED = {-1: "inhibited_by", 1: "activated_by", None: "modulated_by"}


def record(model: pysb.Model, rows: list[tuple], emissions) -> list:
    """Each row's emission with the kind its net rate shows for each regulator its statement
    states (``derived``, where it differs from the kind stated: ``inhibited_by`` where J never
    rises with it, ``activated_by`` where it never falls, ``modulated_by`` where neither is shown)
    and J itself (``rate``, :func:`stored`), for :func:`resign` to sign again on a version's
    bounds. Each name ranges as its assumptions allow here: a parameter is positive.

    Refused only on what is shown: a regulator nothing in the model forms, and one J does not
    change with, which acts only on a step J cannot see (:func:`_unseen`). A row with no one net
    rate, or whose J cannot be read, claims no direction: each regulator it states is
    ``modulated_by``. ``rows`` are :func:`mechanisms.assemble`'s, ``model`` expanded."""
    out = []
    for (stmt, policy, prefix, *rest), emission in zip(rows, emissions):
        stated = (rest[1] if len(rest) > 1 else None) or {}
        positions = [f"{kind}_{i}" for kind, agents in stated.items() for i in range(len(agents))]
        if not positions:
            out.append(emission)
            continue
        try:
            rate = net_rate(model, emission.rules, stmt)
        except MechanismError as exc:
            out.append(replace(emission, derived=_unsigned(positions), rate={"none": str(exc)}))
            continue
        except Exception:  # noqa: BLE001 - a check that breaks claims nothing; never a crash
            logger.exception("row %r: reading its net rate failed", prefix)
            out.append(replace(emission, derived=_unsigned(positions),
                               rate={"none": "its net rate could not be read", "failed": True}))
            continue
        for s in positions:
            if not rate.regulators.get(s):
                raise MechanismError(
                    f"row {prefix!r}: its rate reads `{_said(s, stated)}`, but nothing in the model "
                    f"forms {_agent(stated, s)}, so the rate does not change with it. Add the "
                    "statement that forms it, declare it present in `starts`, or leave it out of "
                    "this statement and its rate"
                )
        deadline = monotonic() + ROW_SECONDS
        found = {s: _shown_safely(rate, s, prefix) if monotonic() < deadline else None
                 for s in positions}
        for s, way in found.items():
            if way == 0:
                raise MechanismError(_unseen(model, emission, stmt, policy, prefix, stated, s))
        out.append(replace(emission, derived=_derived(found), rate=stored(rate)))
    return out


def _unsigned(positions) -> dict[str, str]:
    return {s: "modulated_by" for s in positions if not s.startswith("modulated_by")}


def _derived(found: dict[str, int | None]) -> dict[str, str]:
    """The kinds ``found`` (stated position to the way J moves, ``0`` read as not shown) records
    where they differ from the kinds stated."""
    kinds = {s: RECORDED[None if way == 0 else way] for s, way in found.items()}
    return {s: kind for s, kind in kinds.items() if s.rpartition("_")[0] != kind}


def _shown_safely(rate: NetRate, s: str, prefix: str, domain=None) -> int | None:
    try:
        return shown(rate, s, domain)
    except Exception:  # noqa: BLE001 - a sign that breaks is not shown; never a crash
        logger.exception("row %r: signing %s failed", prefix, s)
        return None


# ------------------------------------------------------------------------------- a version's bounds
def stored(rate: NetRate) -> dict:
    """``rate`` as a version's record keeps it: J as text over names of its own (``p0`` a
    parameter, ``a1`` an amount, ``t2`` a conserved total, ``time``), what each name is, the names
    each regulator reads, and J over its steps (``c0`` a step's coefficient) with each step's."""
    over = rate.over_steps.free_symbols - set(rate.steps) if rate.over_steps is not None else set()
    symbols = sorted(
        rate.value.free_symbols | {x for xs in rate.regulators.values() for x in xs} | over
        | {x for coefficient in rate.steps.values() for x in coefficient.free_symbols},
        key=str,
    )
    names, kinds = {}, {}
    for k, symbol in enumerate(symbols):
        if symbol == TIME:
            name, kinds["time"] = "time", {"time": True}
        elif symbol.name in rate.parameters:
            name = f"p{k}"
            kinds[name] = {"parameter": rate.parameters[symbol.name]}
        elif symbol.name.startswith("total"):
            name, kinds[f"t{k}"] = f"t{k}", {"total": symbol.name}
        else:
            name = f"a{k}"
            kinds[name] = {"amount": symbol.name, "nonnegative": not symbol.is_positive}
        names[symbol] = sympy.Symbol(name)
    steps = {c: sympy.Symbol(f"c{k}") for k, c in enumerate(sorted(rate.steps, key=str))}
    return {
        "J": str(rate.value.xreplace(names)),
        "names": kinds,
        "regulators": {s: [names[x].name for x in xs] for s, xs in rate.regulators.items()},
        **({
            "over_steps": str(rate.over_steps.xreplace(names).xreplace(steps)),
            "steps": {steps[c].name: str(e.xreplace(names)) for c, e in rate.steps.items()},
        } if rate.over_steps is not None else {}),
    }


def restored(data: dict) -> NetRate:
    """The net rate :func:`stored` kept, each name signed as it was built."""
    symbols = {}
    for name, kind in data["names"].items():
        sign = {"nonnegative": True} if kind.get("time") or kind.get("nonnegative") else {
            "positive": True}
        symbols[name] = sympy.Symbol(name, **sign)
    steps = {name: sympy.Symbol(name) for name in data.get("steps", {})}

    def read(text: str) -> sympy.Expr:
        return sympy.sympify(text, locals={**symbols, **steps, "Piecewise": sympy.Piecewise})

    return NetRate(
        read(data["J"]),
        {s: tuple(symbols[n] for n in names) for s, names in data["regulators"].items()},
        read(data["over_steps"]) if "over_steps" in data else None,
        {steps[name]: read(text) for name, text in data.get("steps", {}).items()},
        {name: kind["parameter"] for name, kind in data["names"].items() if "parameter" in kind},
    )


def bounded(rate: NetRate, problem) -> tuple[NetRate, dict[str, tuple[float, float]]]:
    """``rate`` with each parameter as ``problem`` holds it, and the range each name is sampled
    over: a fixed parameter at its value; an estimated one within its bounds, positive where its
    lower bound is or its scale is a log, negative where its upper bound is, of either sign where
    neither (:func:`_ranged`); a condition input over the values its conditions set; ``time`` from
    0 to the last measurement. A name ``problem`` does not hold keeps the sign it was built with."""
    import pandas as pd

    from .problem import (
        ESTIMATE, LOWER_BOUND, NOMINAL_VALUE, PARAMETER_ID, PARAMETER_SCALE, TARGET_ID,
        TARGET_VALUE, TIME as MEASURED, UPPER_BOUND,
    )

    table = problem.parameter_df
    if table is not None and PARAMETER_ID in table.columns:
        table = table.set_index(PARAMETER_ID)
    conditions = problem.condition_df
    inputs: dict[str, list] = {}
    if conditions is not None and TARGET_ID in conditions.columns:
        for target, value in zip(conditions[TARGET_ID], conditions[TARGET_VALUE]):
            inputs.setdefault(str(target), []).append(value)
    swap, domain = {}, {}
    for symbol in rate.value.free_symbols:
        pid = rate.parameters.get(symbol.name)
        if pid is None:
            continue
        # A condition input first: its periods set it whatever a parameter row of its id says
        # (PEtab forbids both, and reconcile keeps it out of the table, but nothing here relies
        # on that).
        if pid in inputs:
            values = pd.to_numeric(pd.Series(inputs[pid]), errors="coerce")
            if values.isna().any():
                swap[symbol] = sympy.Symbol(symbol.name, real=True)
                continue
            low, high = float(values.min()), float(values.max())
            if low == high:
                swap[symbol] = sympy.Float(low)
                continue
            swap[symbol] = _ranged(symbol.name, low, high, domain)
        elif table is not None and pid in table.index:
            row = table.loc[pid]
            if not bool(int(row.get(ESTIMATE, 1))):
                swap[symbol] = sympy.Float(float(row[NOMINAL_VALUE]))
                continue
            low, high = float(row[LOWER_BOUND]), float(row[UPPER_BOUND])
            if str(row.get(PARAMETER_SCALE, "lin")) in ("log", "log10"):
                low = max(low, 0.0)
            swap[symbol] = _ranged(symbol.name, low, high, domain)
    measured = getattr(problem, "measurement_df", None)
    if measured is not None and MEASURED in measured.columns and len(measured):
        last = float(pd.to_numeric(measured[MEASURED], errors="coerce").max())
        if last == last and last > 0:
            domain["time"] = (0.0, last)
    return replace(
        rate,
        value=rate.value.xreplace(swap),
        steps={c: coefficient.xreplace(swap) for c, coefficient in rate.steps.items()},
    ), domain


def _ranged(name: str, low: float, high: float, domain: dict) -> sympy.Expr:
    """A name ranging from ``low`` to ``high``, as what sympy can sign: its lower bound plus a
    name never negative where that bound is not, its upper bound less one where that is not
    positive, and a real name of either sign otherwise; ``domain`` gets the range of the name it
    is written in."""
    span = min(high - low, 1e6) if np.isfinite(high - low) else 1e6
    if low >= 0:
        over = sympy.Symbol(f"{name}__over", nonnegative=True)
        domain[over.name] = (0.0, span)
        return low + over
    if high <= 0:
        under = sympy.Symbol(f"{name}__under", nonnegative=True)
        domain[under.name] = (0.0, span)
        return high - under
    domain[name] = (low, high)
    return sympy.Symbol(name, real=True)


def resign(record: dict, problem) -> tuple[dict, int, list[str]]:
    """``record`` (a version's statements record) with each row's regulators signed again on
    ``problem``'s parameter table, conditions and measurements (:func:`bounded`): the kinds a
    version records are its own bounds', so a bound or a scale an edit changes can change them.
    Never refuses: a regulator J is not shown to move with is ``modulated_by``. A row built before
    its net rate was kept with it (provenance older than this) keeps the kinds it had: unsigned,
    not downgraded. Returns the record, how many rows failed to sign (each logged, each claiming no
    direction, the count kept in the record as ``regulator_check_failures`` while there are any),
    and the prefixes of the rows left unsigned."""
    from copy import deepcopy

    from .statements import regulators

    record, failed, unsigned = deepcopy(record), 0, []
    for row in record.get("rows", []):
        emission = row.get("emission") or {}
        stated = regulators(row["statement"])
        positions = [f"{kind}_{i}" for kind, agents in stated.items() for i in range(len(agents))]
        if not positions:
            continue
        data = emission.get("rate") or {}
        if not data:
            unsigned.append(row.get("prefix"))
            continue
        if "J" not in data:
            emission["derived"] = _unsigned(positions)
            failed += bool(data.get("failed"))
            continue
        try:
            rate, domain = bounded(restored(data), problem)
            deadline = monotonic() + ROW_SECONDS
            found = {s: shown(rate, s, domain) if monotonic() < deadline else None
                     for s in positions}
        except Exception:  # noqa: BLE001 - a sign that breaks is not shown; never a crash
            logger.exception("row %r: signing its regulators on the version's bounds failed",
                             row.get("prefix"))
            emission["derived"] = _unsigned(positions)
            failed += 1
            continue
        emission["derived"] = _derived(found)
    if failed:
        record["regulator_check_failures"] = failed
    else:
        record.pop("regulator_check_failures", None)
    return record, failed, unsigned


def _unseen(model, emission, stmt, policy, prefix, stated, s: str) -> str:
    """The refusal of a regulator J does not change with: the steps that read it, and, where one
    step does and it is a degradation or a dissociation, the row that step is."""
    from .mechanisms import _direction

    observables = {
        o for name in emission.rules
        for o in _observables(model.rules[name].rate_forward)
        | _observables(model.rules[name].rate_reverse)
        if (found := _READ.search(o.name)) and position(found.group(1)) == s
    }
    steps = []
    for name in emission.rules:
        rule = model.rules[name]
        for back, rate in ((False, rule.rate_forward), (True, rule.rate_reverse)):
            if _observables(rate) & observables:
                sides = (rule.product_pattern, rule.reactant_pattern) if back else (
                    rule.reactant_pattern, rule.product_pattern)
                steps.append((*sides, rate))
    agent = _agent(stated, s)
    shown_steps = " and ".join(f"`{_side(before)} >> {_side(after)}`" for before, after, _ in steps)
    if (needed := _product_needed(model, emission, stmt, observables)) is not None:
        return (
            f"row {prefix!r}: {policy!r} reads `{_said(s, stated)}` only in {shown_steps}, where "
            f"what it does needs the product present, and its statement's own reaction rate holds "
            f"its products at 0 (absent), so that rate does not change with it. "
            f"{_with_the_product(stmt, stated, s, *needed)}Or leave {agent} out of this "
            "statement and its rate"
        )
    advice = ""
    if len(steps) == 1:
        before, after, rate = steps[0]
        x = sympy.Symbol("__x", nonnegative=True)
        moves = _direction(_step_rate(rate, observables, x), x)
        kind = {1: "activated_by", -1: "inhibited_by"}.get(moves, "modulated_by")
        kind_, i = s.rpartition("_")[0], int(s.rpartition("_")[2])
        found = _step_row(stmt, before, after, kind, stated[kind_][i])
        if found is not None:
            what, row, mechanism = found
            shown_steps += f", {what}"
            advice = f"State it on a row of that step instead, `{row}`: {mechanism}. "
    return (
        f"row {prefix!r}: {policy!r} reads `{_said(s, stated)}` only in {shown_steps}, and its "
        "statement's own reaction rate, its products absent, does not change with it: a "
        f"statement's regulator is about its own reaction. {advice}Or leave {agent} out of this "
        "statement and its rate"
    )


def _product_needed(model, emission, stmt, observables) -> tuple[str, int] | None:
    """How each step of the row whose rate reads ``observables`` needs a product of ``stmt``
    present: ``"amount"`` where its rate reads the product's amount, ``"reactant"`` where it takes
    the product as a reactant and gives it back with more (autocatalysis); and the way the step's
    rate moves with the product. ``None`` where one such step does not, or takes a product away
    (a way back, a degradation: the C cases)."""
    from .mechanisms import _direction

    products = {i for i, effect in _effects(model, stmt).items() if effect > 0}
    how, ways = set(), set()
    for rx in model.reactions:
        if not set(rx["rule"]) & set(emission.rules):
            continue
        rule = model.rules[rx["rule"][0]]
        rate = rule.rate_reverse if rx["reverse"][0] else rule.rate_forward
        if not _observables(rate) & observables:
            continue
        if any(rx["reactants"].count(i) > rx["products"].count(i) for i in products):
            return None
        reading = {o for o in _observables(rate) - observables if set(o.species) & products}
        if set(rx["reactants"]) & products:
            how.add("reactant")
            ways.add(1)
        elif reading:
            how.add("amount")
            x = sympy.Symbol("__product", nonnegative=True)
            ways.add(_direction(_step_rate(rate, reading, x), x))
        else:
            return None
    if len(how) != 1:
        return None
    return how.pop(), ways.pop() if len(ways) == 1 else None


def _with_the_product(stmt, stated: dict, s: str, how: str, way: int | None) -> str:
    """The two rows a regulator that acts only with the product present can be stated on, each
    written out for ``stmt``: the product stated as a regulator too, read by position where the
    step read it; and that step alone on its own row so, its way back a statement of its own."""
    from .notation import statement_text
    from .statements import _products

    product = next(agent for agent, effect in _products(stmt) if effect > 0)
    kind = {1: "activated_by", -1: "inhibited_by"}.get(way, "modulated_by")
    regulators = {k: list(agents) for k, agents in stated.items()}
    regulators.setdefault(kind, []).append(product)
    row = statement_text(stmt.to_json() | {k: [a.to_json() for a in v] for k, v in regulators.items()})
    own_kind, i = s.rpartition("_")[0], int(s.rpartition("_")[2])

    def read(k: str, j: int) -> str:
        return f"amount({k}_{j})" if len(regulators[k]) > 1 else f"amount({k})"

    product_at = read(kind, len(regulators[kind]) - 1)
    where = ("where it reads the product's amount" if how == "amount" else
             "for the product it takes as a reactant, the step written without it")
    back = _way_back(stmt)
    return (
        f"State the product as a regulator too, `{row}`, its rate reading it as `{product_at}` "
        f"{where}, and {_agent(stated, s)} as `{read(own_kind, i)}`. Or put that step on a row of "
        "its own so, a one-rule mechanism (another step of the same reaction a term of its rate), "
        "and its way back, where the mechanism has one, on a row of its own"
        + (f", `{back}`. " if back else ". ")
    )


def _way_back(stmt) -> str | None:
    """``stmt``'s reaction run backwards, as a statement in the notation, where the notation has
    one: a conversion's outputs back to its inputs, what a synthesis makes degraded, a
    modification taken off."""
    from .notation import statement_text

    if isinstance(stmt, Conversion):
        back = Conversion(None, list(stmt.obj_to), list(stmt.obj_from))
    elif isinstance(stmt, IncreaseAmount):
        back = DecreaseAmount(None, stmt.obj)
    elif isinstance(stmt, DecreaseAmount):
        back = IncreaseAmount(None, stmt.obj)
    elif isinstance(stmt, Modification) and stmt.sub is not None:
        back = modclass_to_inverse[type(stmt)](None, stmt.sub, stmt.residue, stmt.position)
    else:
        return None
    try:
        return statement_text(back.to_json())
    except ValueError:  # what the notation cannot write
        return None


def _step_row(stmt, before, after, kind: str, regulator) -> tuple[str, str, str] | None:
    """A step of row ``stmt``'s rules as a row of its own, ``regulator`` stated on it as ``kind``
    (what the step's own rate does with it): what the step is, the row in the notation, and the
    mechanism the row needs. ``None`` for a step no statement makes alone."""
    from indra.assemblers.pysb.sites import get_mod_site_name

    from .notation import agent_text, statement_text

    def agent(mp):
        return next((a for a in named_agents(stmt) if _n(a.name) == mp.monomer.name), None)

    def text(statement) -> str | None:
        try:
            return statement_text(statement.to_json())
        except ValueError:  # what the notation cannot write
            return None

    factor = {"inhibited_by": f"/(1 + {kind}/K)"}.get(kind, f"*(1 + {kind}/K)")
    clause = f", {REGULATORS[kind]} {_agent({kind: [regulator]}, f'{kind}_0')}"
    left = [cp.monomer_patterns for cp in before.complex_patterns if cp is not None]
    right = [cp.monomer_patterns for cp in after.complex_patterns if cp is not None]
    if len(left) == 1 and len(left[0]) == 1 and not right and agent(left[0][0]) is not None:
        gone = agent(left[0][0])
        row = text(DecreaseAmount(None, gone))
        return row and (
            f"{agent_text(gone.to_json())}'s degradation", row + clause,
            f"a DecreaseAmount whose mechanism reads it, a written `rate` `kd{factor}` with "
            "`parameters` [kd, K]",
        )
    if not left and len(right) == 1 and len(right[0]) == 1 and agent(right[0][0]) is not None:
        made = agent(right[0][0])
        row = text(IncreaseAmount(None, made))
        return row and (
            f"{agent_text(made.to_json())}'s synthesis", row + clause,
            f"an IncreaseAmount whose mechanism reads it, a written `rate` `ks{factor}` with "
            "`parameters` [ks, K]",
        )
    if (len(left) == 1 and len(left[0]) == 2 and len(right) == 2 and all(len(c) == 1 for c in right)
            and sorted(m.monomer.name for m in left[0]) == sorted(c[0].monomer.name for c in right)
            and None not in (agent(m) for m in left[0])):
        first, second = (agent(m) for m in left[0])
        bound = deepcopy(first)
        bound.bound_conditions = [*bound.bound_conditions, BoundCondition(second)]
        row = text(Conversion(None, [bound], [first, second]))
        amount = factor.replace(kind, f"amount({kind})")
        return row and (
            "the dissociation", row + clause,
            "a Conversion whose scheme breaks the bond and reads it, `obj_from_0(obj_to_1=1) % "
            "obj_to_1(obj_to_0=1) >> obj_to_0(obj_to_1=None) + obj_to_1(obj_to_0=None) @ "
            f"kr{amount}` with `parameters` [kr, K]",
        )
    if not (len(left) == len(right) == 1 and len(left[0]) == len(right[0]) == 1
            and left[0][0].monomer is right[0][0].monomer):
        return None
    was, now = left[0][0].site_conditions, right[0][0].site_conditions
    changed = {site for site in set(was) | set(now) if was.get(site) != now.get(site)}
    molecule = agent(left[0][0])
    if isinstance(stmt, Modification) and stmt.sub is not None and molecule is stmt.sub:
        if changed == {get_mod_site_name(stmt._get_mod_condition())}:
            inverse = modclass_to_inverse[type(stmt)]
            row = text(inverse(None, stmt.sub, stmt.residue, stmt.position))
            return row and (
                "the way back of its modification", row + clause,
                f"a {inverse.__name__} whose mechanism, based on `basal`, reads it, a written "
                f"`rate` `k{factor}` with `parameters` [k, K]",
            )
    if (isinstance(stmt, RegulateActivity) and molecule is stmt.obj
            and changed == {stmt.obj_activity} and kind == "activated_by"):
        back = Inhibition if stmt.is_activation else Activation
        row = text(back(regulator, stmt.obj, stmt.obj_activity))
        return row and (
            "the way back of its activity", row,
            f"an {back.__name__} with {regulator.name} its subject, `indra_default`",
        )
    return None


def _step_rate(rate, observables, x: sympy.Symbol) -> sympy.Expr:
    """A step's own rate with the regulator read by ``observables`` as ``x``, every other name
    positive."""
    expr = rate.expr
    while expressions := expr.atoms(pysb.Expression):
        expr = expr.xreplace({e: e.expr for e in expressions})
    expr = expr.xreplace({
        o: x if o in observables else sympy.Symbol(o.name, positive=True)
        for o in expr.atoms(pysb.Observable)
    })
    named = {p: sympy.Symbol(p.name, positive=True) for p in expr.atoms(pysb.Parameter)}
    return expr.xreplace(named).xreplace({time: TIME})


def _side(pattern) -> str:
    return " + ".join(
        " % ".join(mp.monomer.name for mp in cp.monomer_patterns)
        for cp in pattern.complex_patterns if cp is not None
    ) or "None"


def _agent(stated: dict, s: str) -> str:
    from .notation import agent_text

    kind, _, i = s.rpartition("_")
    agent = stated[kind][int(i)]
    try:
        return agent_text(agent.to_json())
    except ValueError:
        return agent.name


def _said(s: str, stated: dict) -> str:
    return f"{REGULATORS[s.rpartition('_')[0]]} {_agent(stated, s)}"


def _short(expr: sympy.Expr, width: int = 240) -> str:
    text = str(expr)
    return text if len(text) <= width else text[: width - 3] + "..."
