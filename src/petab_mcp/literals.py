"""Hardcoded constants: the numeric literals in a model's math that ``k`` must price.

Every penalised score charged ``k = n_estimated``, so a value written straight into a rate law
was free however it was obtained and a model could replace its parameter vector with numerals
(issue #180). Every literal is charged except :data:`STRUCTURAL_VALUES`, each distinct VALUE
once; :func:`fitting.complexity_k` adds the count to the rows.

That set is ENUMERATED rather than a significant-digit threshold, which is what makes it safe: a
threshold prices a number by its spelling (``0.878`` written ``0.88`` was free under the one that
shipped first), while the nearest free value to a rate constant is ``0``, which deletes the term.
Membership is exact rational, so ``0.5`` is ``1/2`` and free while ``0.333`` is charged.

For the same reason free values do not COMBINE into a priced one for nothing (issue #338): a value
made only of free literals and the model's own structure is charged when it is not free itself.
That is a run of constants (``2*2*2``, ``6 - 1``, ``exp(-4)``), a factor repeated or raised past
a power (``x*x*x*x*x*x*x*x`` and ``(x^4)^2`` are ``x^8``) and a term repeated past a multiple
(``k*x + k*x + k*x + k*x + k*x``). Where a written literal is already charged, the value it
helps make is not charged again: ``x^0.7*x^0.7`` and ``10^-3`` are one constant each.

A value is charged ONCE however many places write it, because a value two places share is one
constant: estimating it is a merge of those parameters into one, which ``k`` prices at one and
which fits at least as well, so a literal two unrelated rates share is dominated by that merge
exactly as a lone literal is by its parameter. A sign is not a second constant either.

There is deliberately no route for the AGENT to declare a constant known: the claim cannot be
checked, so it would be a channel for pricing a fitted value at zero. A published constant is the
PROBLEM's to ship as a FIXED row. Nor is a literal matched against the values earlier versions
fitted (issue #338 proposed it): the charge already makes a frozen fitted value cost what
estimating it costs at a fit no better, and a match would need a digit tolerance, which is pricing
by spelling again, against hundreds of ratios that a genuine constant can match by chance.

Scanned: kinetic laws, assignment and rate rules, initial assignments, function definitions,
event triggers and assignments. NOT scanned: species initial values (the reconcile already turns
a nonzero, non-unit one into an ESTIMATED ``<sid>_init`` row) and parameter values (the table
overrides them), a rate law's local parameters among them: the reconcile promotes each to a global
parameter with a row.
"""

from __future__ import annotations

import math
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from typing import Iterator

import libsbml

#: The multiplicities a reaction network is BUILT from: the oligomer sizes
#: (dimer/trimer/tetramer/hexamer) that appear as stoichiometric factors and mass-action
#: exponents. Five is absent because nothing here is a pentamer.
_MULTIPLICITIES = (1, 2, 3, 4, 6)
#: Free as literals: zero, and each of the above with its sign and its reciprocal. Enumerated on
#: purpose — see the module docstring on why a list beats a bound.
STRUCTURAL_VALUES = frozenset(
    [Fraction(0)]
    + [
        sign * value
        for n in _MULTIPLICITIES
        for value in (Fraction(n), Fraction(1, n))
        for sign in (1, -1)
    ]
)
#: Percent: free only as a factor of a whole READOUT, an assignment rule nothing in the dynamics
#: reads (``frac := (1 - A/A_tot)*100``). There it has nothing to absorb it into, so charging it
#: would price a UNIT as a parameter; anywhere else it is a magnitude like any other, and as a rate
#: constant it froze a bound-limited rate for free (issue #338).
_READOUT_VALUES = STRUCTURAL_VALUES | {
    v * unit for v in STRUCTURAL_VALUES for unit in (Fraction(100), Fraction(1, 100))
}

_POWER = (libsbml.AST_POWER, libsbml.AST_FUNCTION_POWER)
#: What a run of constants may be built with and still be one constant: arithmetic, and the
#: elementary functions whose value at a free literal is not free (``exp(-4)`` is 0.018).
_CONSTANT_OPS = frozenset(
    (
        libsbml.AST_PLUS,
        libsbml.AST_MINUS,
        libsbml.AST_TIMES,
        libsbml.AST_DIVIDE,
        *_POWER,
        libsbml.AST_FUNCTION_EXP,
        libsbml.AST_FUNCTION_LN,
        libsbml.AST_FUNCTION_LOG,
        libsbml.AST_FUNCTION_ROOT,
    )
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


def _children(node) -> list:
    return [node.getChild(i) for i in range(node.getNumChildren())]


def _number_nodes(math) -> Iterator:
    if math is None:
        return
    if math.isNumber():
        yield math
    for child in _children(math):
        yield from _number_nodes(child)


def _is_constant(node) -> bool:
    """Numbers under :data:`_CONSTANT_OPS` only: no name, no π or e, no call of the model's own."""
    if node.isNumber():
        return True
    children = _children(node)
    return node.getType() in _CONSTANT_OPS and bool(children) and all(map(_is_constant, children))


def _power(base: Fraction, exponent: Fraction) -> Fraction | None:
    """``base**exponent``, exact for a small integer exponent (an exact huge power never ends)."""
    try:
        if exponent.denominator == 1 and abs(exponent) <= 64:
            return base ** int(exponent)
        value = float(base) ** float(exponent)
        return Fraction(value) if isinstance(value, float) and math.isfinite(value) else None
    except (ZeroDivisionError, OverflowError):
        return None


def _evaluate(node) -> Fraction | None:
    """The value of a constant subtree, exact wherever arithmetic keeps it rational."""
    if node.isNumber():
        return _value_key(node)
    values = [_evaluate(child) for child in _children(node)]
    t = node.getType()
    if None in values:
        return None
    try:
        if t == libsbml.AST_PLUS:
            return sum(values, Fraction(0))
        if t == libsbml.AST_MINUS:
            return -values[0] if len(values) == 1 else values[0] - values[1]
        if t == libsbml.AST_TIMES:
            return math.prod(values, start=Fraction(1))
        if t == libsbml.AST_DIVIDE:
            return values[0] / values[1]
        if t in _POWER:
            return _power(*values)
    except ZeroDivisionError:
        return None
    value = libsbml.SBMLTransforms.evaluateASTNode(node)
    return Fraction(value) if math.isfinite(value) else None


def _spell(value: Fraction | None) -> str:
    if value is None:
        return "?"
    if value.denominator == 1:
        return str(value.numerator)
    decimal = Decimal(value.numerator) / Decimal(value.denominator)
    return format(decimal.normalize(), "f") if Fraction(decimal) == value else f"{float(value):.6g}"


def _charged(nodes, value: Fraction | None, free, spelling: str | None = None) -> Iterator[tuple]:
    """What one value made from ``nodes`` costs: its written literals where any is charged,
    else the value itself unless it is free, spelled as written when it is one subtree."""
    written = [
        (None if key is None else abs(key), libsbml.formulaToL3String(n))
        for node in nodes
        for n in _number_nodes(node)
        if (key := _value_key(n)) is None or key not in free
    ]
    if written:
        yield from written
    elif value is None or value not in free:
        if spelling is None:
            spelling = libsbml.formulaToL3String(nodes[0]) if len(nodes) == 1 else _spell(value)
        yield (None if value is None else abs(value)), spelling


def _factors(node, power: Fraction | None, exponents: tuple, out: list) -> None:
    """Flatten ``node**power`` into ``(factor, power, exponent subtrees on its way)``."""
    t = node.getType()
    children = _children(node)
    if _is_constant(node):
        out.append((node, power, exponents))
    elif t == libsbml.AST_TIMES:
        for child in children:
            _factors(child, power, exponents, out)
    elif t == libsbml.AST_DIVIDE and len(children) == 2:
        _factors(children[0], power, exponents, out)
        _factors(children[1], None if power is None else -power, exponents, out)
    elif t in _POWER and len(children) == 2 and _is_constant(children[1]):
        e = _evaluate(children[1])
        new = None if power is None or e is None else power * e
        _factors(children[0], new, (*exponents, children[1]), out)
    else:
        out.append((node, power, exponents))


def _terms(node, sign: int, out: list) -> None:
    """Flatten a sum into ``(term, sign)``."""
    t = node.getType()
    children = _children(node)
    if t == libsbml.AST_PLUS:
        for child in children:
            _terms(child, sign, out)
    elif t == libsbml.AST_MINUS and len(children) in (1, 2):
        if len(children) == 2:
            _terms(children[0], sign, out)
        _terms(children[-1], -sign, out)
    else:
        out.append((node, sign))


def _by_spelling(items) -> dict[str, list]:
    groups: dict[str, list] = {}
    for item in items:
        groups.setdefault(libsbml.formulaToL3String(item[0]), []).append(item)
    return groups


def _priced(node, free=STRUCTURAL_VALUES) -> Iterator[tuple[Fraction | None, str]]:
    """``(value, as written)`` for each constant ``node`` is charged for; ``free`` applies to
    the constant part of ``node`` itself, not to what is nested below it."""
    t = node.getType()
    children = _children(node)
    if _is_constant(node):
        yield from _charged([node], _evaluate(node), free)
    elif t in (libsbml.AST_TIMES, libsbml.AST_DIVIDE) or (
        t in _POWER and len(children) == 2 and _is_constant(children[1])
    ):
        factors: list = []
        _factors(node, Fraction(1), (), factors)
        constants = [f for f in factors if _is_constant(f[0])]
        if constants:
            parts = [
                None if p is None or (v := _evaluate(f)) is None else _power(v, p)
                for f, p, _ in constants
            ]
            value = None if None in parts else math.prod(parts, start=Fraction(1))
            yield from _charged([n for f, _, exps in constants for n in (f, *exps)], value, free)
        rest = (f for f in factors if not _is_constant(f[0]))
        for spelling, members in _by_spelling(rest).items():
            powers = [p for _, p, _ in members]
            total = None if None in powers else sum(powers, Fraction(0))
            exponents = [e for _, _, exps in members for e in exps]
            yield from _charged(exponents, total, STRUCTURAL_VALUES, f"{spelling}^{_spell(total)}")
            yield from _priced(members[0][0])
    elif t in (libsbml.AST_PLUS, libsbml.AST_MINUS):
        terms: list = []
        _terms(node, 1, terms)
        constants = [(term, s) for term, s in terms if _is_constant(term)]
        if constants:
            parts = [None if (v := _evaluate(term)) is None else s * v for term, s in constants]
            value = None if None in parts else sum(parts, Fraction(0))
            yield from _charged([term for term, _ in constants], value, STRUCTURAL_VALUES)
        rest = ((term, s) for term, s in terms if not _is_constant(term))
        for spelling, members in _by_spelling(rest).items():
            count = Fraction(sum(s for _, s in members))
            if abs(count) not in STRUCTURAL_VALUES:
                yield abs(count), f"{_spell(abs(count))}*({spelling})"
            yield from _priced(members[0][0])
    else:
        for child in children:
            yield from _priced(child)


def _names(math) -> Iterator[str]:
    if math is None:
        return
    if math.getType() == libsbml.AST_NAME:
        yield math.getName()
    for child in _children(math):
        yield from _names(child)


def _model_math(sm) -> Iterator[tuple]:
    """``(math, assignment-rule target or None)`` for every math AST the rule applies to (see the
    module docstring)."""
    for i in range(sm.getNumReactions()):
        kl = sm.getReaction(i).getKineticLaw()
        if kl is not None:
            yield kl.getMath(), None
    for i in range(sm.getNumRules()):
        rule = sm.getRule(i)
        yield rule.getMath(), rule.getVariable() if rule.isAssignment() else None
    for i in range(sm.getNumInitialAssignments()):
        yield sm.getInitialAssignment(i).getMath(), None
    for i in range(sm.getNumFunctionDefinitions()):
        yield sm.getFunctionDefinition(i).getMath(), None
    for i in range(sm.getNumEvents()):
        ev = sm.getEvent(i)
        if ev.getTrigger() is not None:
            yield ev.getTrigger().getMath(), None
        for j in range(ev.getNumEventAssignments()):
            yield ev.getEventAssignment(j).getMath(), None


def _readouts(sm) -> set[str]:
    """The assignment-rule targets nothing outside the readouts reads: outputs only."""
    rules = {target: m for m, target in _model_math(sm) if target is not None}
    read = {n for m, target in _model_math(sm) if target is None for n in _names(m)}
    while grown := {n for t in read & rules.keys() for n in _names(rules[t])} - read:
        read |= grown
    return rules.keys() - read


def hardcoded_constants(sm) -> list[str]:
    """The distinct priced literals in ``sm``'s math, each as first written, sorted by value.

    ``len()`` of the result is what :func:`fitting.complexity_k` adds to the estimated-parameter
    count; the strings themselves ride on the fit payload so the agent can see WHICH numbers it
    is paying for. A value made from free literals is spelled by what it is: ``x^8`` for the
    repeated product, ``5*(k * x)`` for the repeated term.
    """
    readouts = _readouts(sm)
    found: dict[Fraction | None, str] = {}
    for math_, target in _model_math(sm):
        if math_ is None:
            continue
        free = _READOUT_VALUES if target in readouts else STRUCTURAL_VALUES
        # An unreadable number node is charged, not skipped: a literal nothing could parse is
        # the one case where "free" must not be decided by a parse failure.
        for key, spelling in _priced(math_, free):
            found.setdefault(key, spelling)
    return [found[k] for k in sorted(found, key=lambda v: (v is None, v))]
