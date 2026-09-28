"""Hardcoded constants: the numeric literals in a model's math that ``k`` must price.

Every penalised score charged ``k = n_estimated``, so a value written straight into a rate law
was free however it was obtained and a model could replace its parameter vector with numerals
(issue #180). Every literal is charged except :data:`STRUCTURAL_VALUES`, each distinct VALUE
once; :func:`fitting.complexity_k` adds the count to the rows.

That set is ENUMERATED rather than a significant-digit threshold, which is what makes it safe: a
threshold prices a number by its spelling (``0.878`` written ``0.88`` was free under the one that
shipped first), while the nearest free value to a rate constant is ``0``, which deletes the term.
Membership is exact rational, so ``0.5`` is ``1/2`` and free while ``0.333`` is charged.

There is deliberately no route for the AGENT to declare a constant known: the claim cannot be
checked, so it would be a channel for pricing a fitted value at zero. A published constant is the
PROBLEM's to ship as a FIXED row.

Scanned: kinetic laws, assignment and rate rules, initial assignments, function definitions,
event triggers and assignments. NOT scanned: species initial values (the reconcile already turns
a nonzero, non-unit one into an ESTIMATED ``<sid>_init`` row) and parameter values (the table
overrides them).
"""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
from fractions import Fraction
from typing import Iterator

import libsbml

#: The multiplicities a reaction network is BUILT from: the oligomer sizes
#: (dimer/trimer/tetramer/hexamer) that appear as stoichiometric factors and mass-action
#: exponents. Five is absent because nothing here is a pentamer.
_MULTIPLICITIES = (1, 2, 3, 4, 6)
#: Percent. A readout in ``%`` needs a factor of 100 and has nothing to absorb it into (rahman's
#: ``prevalence := (1 - S/N)*100`` is a pure function of states), so charging it would price a
#: UNIT as a parameter with no free route to avoid it.
_UNIT_FACTORS = (100,)
#: Free as literals: zero, and each of the above with its sign and its reciprocal. Enumerated on
#: purpose — see the module docstring on why a list beats a bound.
STRUCTURAL_VALUES = frozenset(
    [Fraction(0)]
    + [
        sign * value
        for n in (*_MULTIPLICITIES, *_UNIT_FACTORS)
        for value in (Fraction(n), Fraction(1, n))
        for sign in (1, -1)
    ]
)


def _value_key(node) -> Fraction | None:
    """The node's EXACT rational value, the key both the free-set test and the de-dup use.

    Exact so that ``2.69e-5`` and ``0.0000269`` are one constant, and so that ``0.5`` IS ``1/2``
    while ``0.333`` is not ``1/3``: every decimal literal is a rational, and comparing rationals
    needs no tolerance to argue about.
    """
    try:
        t = node.getType()
        if t == libsbml.AST_RATIONAL:
            return Fraction(node.getNumerator(), node.getDenominator())
        if t == libsbml.AST_REAL_E:
            return Fraction(Decimal(repr(node.getMantissa())).scaleb(node.getExponent()))
        if t == libsbml.AST_INTEGER:
            return Fraction(node.getInteger())
        # Python's shortest round-trip repr, so the Fraction is the value libsbml holds and not a
        # binary-float artefact: Fraction(0.1) is 3602879701896397/36028797018963968, while
        # Fraction(Decimal("0.1")) is 1/10 and compares equal to the free 1/10.
        return Fraction(Decimal(repr(node.getReal())))
    except (InvalidOperation, ValueError, ZeroDivisionError, OverflowError):
        return None


def _number_nodes(math) -> Iterator:
    if math is None:
        return
    if math.isNumber():
        yield math
    for i in range(math.getNumChildren()):
        yield from _number_nodes(math.getChild(i))


def model_math(sm) -> Iterator:
    """Every math AST in ``sm`` that the rule applies to (see the module docstring)."""
    for i in range(sm.getNumReactions()):
        kl = sm.getReaction(i).getKineticLaw()
        if kl is not None:
            yield kl.getMath()
    for i in range(sm.getNumRules()):
        yield sm.getRule(i).getMath()
    for i in range(sm.getNumInitialAssignments()):
        yield sm.getInitialAssignment(i).getMath()
    for i in range(sm.getNumFunctionDefinitions()):
        yield sm.getFunctionDefinition(i).getMath()
    for i in range(sm.getNumEvents()):
        ev = sm.getEvent(i)
        if ev.getTrigger() is not None:
            yield ev.getTrigger().getMath()
        for j in range(ev.getNumEventAssignments()):
            yield ev.getEventAssignment(j).getMath()


def hardcoded_constants(sm) -> list[str]:
    """The distinct priced literals in ``sm``'s math, each as first written, sorted by value.

    ``len()`` of the result is what :func:`fitting.complexity_k` adds to the estimated-parameter
    count; the strings themselves ride on the fit payload so the agent can see WHICH numbers it
    is paying for.
    """
    found: dict[Fraction | None, str] = {}
    for math in model_math(sm):
        for node in _number_nodes(math):
            key = _value_key(node)
            # An unreadable number node is charged, not skipped: a literal nothing could parse is
            # the one case where "free" must not be decided by a parse failure.
            if key is not None and key in STRUCTURAL_VALUES:
                continue
            if key not in found:
                found[key] = libsbml.formulaToL3String(node)
    return [found[k] for k in sorted(found, key=lambda v: (v is None, v))]
