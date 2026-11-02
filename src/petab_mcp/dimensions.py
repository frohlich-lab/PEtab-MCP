"""Each parameter's physical DIMENSION, as the model's own math implies it.

PEtab states no units, and the agent's model declares none, so nothing the server is handed says
that ``k_close`` is a rate and ``K_Cd`` a concentration. The merge price
(:func:`identifiability.merge_candidates`) pairs parameters by numerical coincidence, so without
this it offered ties no modeller would write: in one sweep (gemot#94) a channel-closing rate tied to
a Ca²⁺ midpoint, a feedback threshold tied to a dephosphorylation rate, and a drug EC50 tied to a
carrying capacity, each taken and scored. The math says what the table does not: a symbol added to
a species has that species' dimension, a rate constant times the species it consumes is that
reaction's flux, and the argument of ``exp`` is a pure number.

A dimension is a vector of rational exponents over GENERATORS:

* ``time``, the one fixed base;
* every species, and every symbol the condition table sets that the math uses as a QUANTITY (a dose
  summed with its midpoint, ``K + Ca``, or assigned outright), each its own generator until the math
  equates it with another. Species a reaction converts into one another come out sharing one,
  because the reaction's flux is each participant's dimension per time; a cell count and a drug
  concentration that no reaction or sum connects stay apart. An input the math only multiplies by
  is a switch or a multiplier (a knockdown factor, a cell-line indicator) and is read as a number:
  a dimension of its own would set every rate it scales apart from the same rate unscaled.

Everything else (the parameters) is an unknown, solved from linear constraints by exact Gaussian
elimination over :class:`fractions.Fraction`:

* every term of a sum, and every value of a piecewise, has one dimension;
* products and quotients add and subtract dimensions, a numeric power scales one;
* a rate law is its reaction's flux: each participant's dimension minus ``time``. A rate rule's
  expression is its target's dimension minus ``time``; an assignment, initial assignment, event
  assignment or condition-table entry gives its target its expression's dimension;
* the argument of ``exp``, ``ln``, a trigonometric function, and a symbolic exponent is
  dimensionless;
* a power with a SYMBOLIC exponent (a Hill coefficient) has no linear dimension, so it gets a fresh
  unknown, and two such powers summed under the same exponent (``K^n + Ca^n``) have bases of one
  dimension.

Numbers carry no dimension in a product. In a sum they are taken as dimensionless (``1 + I/Ki``
makes ``Ki`` the inhibitor's dimension), but only AFTER every other constraint, and each is dropped
where it contradicts what is already known, since agents also write numeric offsets onto
dimensional terms (``geneX + 1e-6``). A constraint that would make ``time`` dimensionless (a rate
law with no rate constant, ``A => B; A``) is dropped the same way, so one inconsistent statement
costs its own information and does not poison the rest. A compartment is a number here: these
models write volumes, where they write any, as a conversion factor, not as the dimension the
filter is about.

A parameter the constraints do not pin is UNDETERMINED, not its own generator, and :func:`differ`
answers ``None`` for it: two parameters differ only where their dimensions differ for EVERY value of
what was left undetermined (``k*A`` vs ``K + A`` differ by a time and an amount, whatever ``A``
is). Failing open here and closed on the role/scale filters beside it is deliberate: a role is
read off a table and is either known or not, while a dimension is inferred, and an undetermined one
is the common case, so failing closed would drop the ties that are fine with the ones that are not.
Two first-order rates for different processes have one dimension and stay a candidate; which of
those is one quantity is a biology call this cannot make.

Pure symbolic work over the SBML the fit was built from, no simulation. Split out of
:mod:`identifiability`, whose merge report is its only consumer.
"""

from __future__ import annotations

from fractions import Fraction

import libsbml

#: The one fixed generator. Not an SBML id: an id a model chooses cannot collide with it.
TIME = "<time>"

#: Prefix of an undetermined column in a reported dimension.
UNDETERMINED = "?"

#: The dimension of a pure number (a literal, a compartment, a switch input, the value of ``exp``),
#: which imposes nothing on what it is multiplied with.
_NUMBER = None

_ARGUMENT_DIMENSIONLESS = {
    libsbml.AST_FUNCTION_EXP,
    libsbml.AST_FUNCTION_LN,
    libsbml.AST_FUNCTION_LOG,
    *(
        getattr(libsbml, f"AST_FUNCTION_{name}")
        for name in (
            "SIN COS TAN SEC CSC COT SINH COSH TANH SECH CSCH COTH ARCSIN ARCCOS ARCTAN ARCSEC "
            "ARCCSC ARCCOT ARCSINH ARCCOSH ARCTANH ARCSECH ARCCSCH ARCCOTH FACTORIAL"
        ).split()
    ),
}
_SAME_AS_ARGUMENTS = {
    libsbml.AST_FUNCTION_ABS,
    libsbml.AST_FUNCTION_CEILING,
    libsbml.AST_FUNCTION_FLOOR,
    libsbml.AST_FUNCTION_MAX,
    libsbml.AST_FUNCTION_MIN,
    libsbml.AST_FUNCTION_REM,
}
_RELATIONAL = {
    libsbml.AST_RELATIONAL_EQ,
    libsbml.AST_RELATIONAL_GEQ,
    libsbml.AST_RELATIONAL_GT,
    libsbml.AST_RELATIONAL_LEQ,
    libsbml.AST_RELATIONAL_LT,
    libsbml.AST_RELATIONAL_NEQ,
}
_LOGICAL = {
    libsbml.AST_LOGICAL_AND,
    libsbml.AST_LOGICAL_OR,
    libsbml.AST_LOGICAL_NOT,
    libsbml.AST_LOGICAL_XOR,
}
_NUMERIC = {
    libsbml.AST_INTEGER,
    libsbml.AST_REAL,
    libsbml.AST_REAL_E,
    libsbml.AST_RATIONAL,
    libsbml.AST_CONSTANT_E,
    libsbml.AST_CONSTANT_PI,
    libsbml.AST_CONSTANT_TRUE,
    libsbml.AST_CONSTANT_FALSE,
    libsbml.AST_NAME_AVOGADRO,
}

Form = dict[str, Fraction]


def _add(a: Form, b: Form, scale: Fraction = Fraction(1)) -> Form:
    out = dict(a)
    for col, c in b.items():
        v = out.get(col, Fraction(0)) + scale * c
        if v:
            out[col] = v
        else:
            out.pop(col, None)
    return out


class _System:
    """Linear constraints ``form == 0`` in reduced row-echelon form, built one at a time.

    Three kinds of column, pivoted in this order: the parameters (and the opaque unknowns), which
    are what the system is solved for; the condition inputs, pinned where the math says what they
    are and a generator of their own where it does not; and the species, which the math may equate
    with one another but never with each other per unit time — that is a rate law missing its rate
    constant (``A => B; A``), not a statement about two species. ``time`` is never a pivot. A
    constraint left with nothing it may pin is DROPPED, so one inconsistent statement costs only
    its own information.
    """

    def __init__(self, species: set[str], inputs: set[str]):
        self.species = species
        self.inputs = inputs
        self.generators = {TIME} | species | inputs
        self.rows: dict[str, Form] = {}  # pivot column -> its row, pivot coefficient 1

    def reduce(self, form: Form) -> Form:
        out = dict(form)
        for col in [c for c in out if c in self.rows]:
            if col in out:
                out = _add(out, self.rows[col], -out[col])
        return out

    def _pivot(self, row: Form) -> str | None:
        for kind in (lambda c: c not in self.generators, lambda c: c in self.inputs):
            candidates = [c for c in row if kind(c)]
            if candidates:
                return min(candidates)  # any choice is exact; a fixed one is reproducible
        if TIME in row:
            return None
        return min(row)

    def add(self, form: Form) -> bool:
        """Impose ``form == 0``; ``False`` (and nothing imposed) where it was already implied or
        has nothing it may pin."""
        row = self.reduce(form)
        if not row:
            return False
        pivot = self._pivot(row)
        if pivot is None:
            return False
        row = {c: v / row[pivot] for c, v in row.items()}
        for col, other in list(self.rows.items()):
            if pivot in other:
                self.rows[col] = _add(other, row, -other[pivot])
        self.rows[pivot] = row
        return True

    def pins_a_parameter(self, form: Form) -> bool:
        """Would ``form == 0`` determine a parameter nothing has determined yet?"""
        return any(c not in self.generators for c in self.reduce(form))

    def solve(self, column: str) -> Form:
        """``column`` in terms of the free columns: generators by name, the rest marked
        :data:`UNDETERMINED`."""
        if column in self.rows:
            form = {c: -v for c, v in self.rows[column].items() if c != column}
        else:
            form = {column: Fraction(1)}
        return {(c if c in self.generators else UNDETERMINED + c): v for c, v in form.items()}


class _Walker:
    """Turns SBML math into the linear form of its dimension, posting the constraints it implies.

    A node's dimension is a :data:`Form` over symbol columns, or :data:`_NUMBER` for a pure number.
    """

    def __init__(self, numbers: set[str]):
        self.numbers = numbers  # compartments and switch inputs, read as numbers
        self.local: dict[str, str] = {}  # the current rate law's local parameters
        self.hard: list[Form] = []
        self.soft: list[Form] = []  # numbers summed onto a term: imposed last, dropped on conflict
        self._fresh = 0
        self._powers: dict[tuple, Form] = {}

    def fresh(self, key=None) -> Form:
        """A new unknown column, or for ``key`` the one already made for it: ``A^n`` in a Hill
        numerator and again in its denominator is one quantity, and two columns for it would
        leave the ratio undetermined."""
        if key is not None and key in self._powers:
            return dict(self._powers[key])
        self._fresh += 1
        out = {f"<opaque {self._fresh}>": Fraction(1)}
        if key is not None:
            self._powers[key] = out
        return dict(out)

    def equal(self, a, b) -> None:
        if a is not _NUMBER and b is not _NUMBER:
            self.hard.append(_add(a, b, Fraction(-1)))

    def name(self, node) -> Form | None:
        sid = node.getName()
        if sid in self.local:
            return {self.local[sid]: Fraction(1)}
        if sid in self.numbers:
            return _NUMBER
        return {sid: Fraction(1)}

    def walk(self, node):
        if node is None:
            return _NUMBER
        t = node.getType()
        kids = [node.getChild(i) for i in range(node.getNumChildren())]
        if t in _NUMERIC:
            return _NUMBER
        if t == libsbml.AST_NAME_TIME:
            return {TIME: Fraction(1)}
        if t == libsbml.AST_NAME:
            return self.name(node)
        if t == libsbml.AST_PLUS or (t == libsbml.AST_MINUS and len(kids) > 1):
            return self.sum(kids, numbers_dimensionless=True)
        if t == libsbml.AST_MINUS:
            return self.walk(kids[0]) if kids else _NUMBER
        if t == libsbml.AST_TIMES:
            dims = [self.walk(k) for k in kids]
            forms = [d for d in dims if d is not _NUMBER]
            if not forms:
                return _NUMBER
            out: Form = {}
            for d in forms:
                out = _add(out, d)
            return out
        if t == libsbml.AST_DIVIDE and len(kids) == 2:
            num, den = self.walk(kids[0]), self.walk(kids[1])
            if num is _NUMBER and den is _NUMBER:
                return _NUMBER
            return _add(num or {}, den or {}, Fraction(-1))
        if t in (libsbml.AST_POWER, libsbml.AST_FUNCTION_POWER) and len(kids) == 2:
            return self.power(kids[0], kids[1])[0]
        if t == libsbml.AST_FUNCTION_ROOT and kids:
            degree = _numeric_value(kids[0]) if len(kids) == 2 else Fraction(2)
            base = self.walk(kids[-1])
            if base is _NUMBER:
                return _NUMBER
            if degree:
                return {c: v / degree for c, v in base.items()}
            return self.fresh()
        if t == libsbml.AST_FUNCTION_PIECEWISE:
            values = [k for i, k in enumerate(kids) if i % 2 == 0]
            for i in range(1, len(kids), 2):
                self.walk(kids[i])  # the conditions: their own relations constrain, value aside
            return self.sum(values)
        if t in _RELATIONAL:
            self.sum(kids)
            return _NUMBER
        if t in _LOGICAL:
            for k in kids:
                self.walk(k)
            return _NUMBER
        if t in _ARGUMENT_DIMENSIONLESS:
            for k in kids:
                self.dimensionless(self.walk(k))
            return _NUMBER
        if t in _SAME_AS_ARGUMENTS:
            return self.sum(kids)
        # A function call, a delay, a csymbol: what comes out is not derivable from what goes in.
        return self.fresh()

    def dimensionless(self, dim) -> None:
        if dim is not _NUMBER:
            self.hard.append(dict(dim))

    def power(self, base_node, exp_node):
        """``(dimension, (exponent key, base dimension) or None)`` of ``base ** exponent``."""
        base = self.walk(base_node)
        exponent = _numeric_value(exp_node)
        if exponent is not None:
            if base is _NUMBER:
                return _NUMBER, None
            return {c: v * exponent for c, v in base.items()}, None
        self.dimensionless(self.walk(exp_node))
        if base is _NUMBER:
            return _NUMBER, None
        key = libsbml.formulaToL3String(exp_node)
        return self.fresh((key, tuple(sorted(base.items())))), (key, base)

    def sum(self, kids, *, numbers_dimensionless: bool = False):
        """One dimension for every term. In an arithmetic sum (``numbers_dimensionless``) a number
        among them makes the rest dimensionless, softly; a piecewise ``0`` branch or a comparison
        with a literal says nothing of the kind."""
        dims, hill = [], {}
        for k in kids:
            t = k.getType()
            if numbers_dimensionless and t in (libsbml.AST_POWER, libsbml.AST_FUNCTION_POWER):
                dim, info = self.power(k.getChild(0), k.getChild(1))
                if info is not None:
                    key, base = info
                    if key in hill:
                        self.equal(base, hill[key])
                    hill[key] = base
                dims.append(dim)
            else:
                dims.append(self.walk(k))
        forms = [d for d in dims if d is not _NUMBER]
        if not forms:
            return _NUMBER
        for d in forms[1:]:
            self.equal(forms[0], d)
        if numbers_dimensionless and len(forms) < len(dims):
            self.soft.append(dict(forms[0]))
            for base in hill.values():
                self.soft.append(dict(base))
        return forms[0]


def _numeric_value(node) -> Fraction | None:
    """The exact value of a subtree that names no symbol, or ``None``."""
    t = node.getType()
    kids = [node.getChild(i) for i in range(node.getNumChildren())]
    if t == libsbml.AST_INTEGER:
        return Fraction(node.getInteger())
    if t == libsbml.AST_RATIONAL:
        return Fraction(node.getNumerator(), node.getDenominator())
    if t in (libsbml.AST_REAL, libsbml.AST_REAL_E):
        try:
            return Fraction(node.getReal()).limit_denominator(1000)
        except (OverflowError, ValueError):  # inf, nan
            return None
    vals = [_numeric_value(k) for k in kids]
    if not kids or any(v is None for v in vals):
        return None
    try:
        if t == libsbml.AST_MINUS:
            return -vals[0] if len(vals) == 1 else vals[0] - sum(vals[1:])
        if t == libsbml.AST_PLUS:
            return sum(vals, Fraction(0))
        if t == libsbml.AST_TIMES:
            out = Fraction(1)
            for v in vals:
                out *= v
            return out
        if t == libsbml.AST_DIVIDE and len(vals) == 2:
            return vals[0] / vals[1]
    except ZeroDivisionError:
        return None
    return None


def infer(sm, inputs=(), condition_entries=()) -> dict[str, Form]:
    """``{symbol: dimension}`` for every parameter and species of SBML model ``sm``, each a form
    over the generators (see the module docstring) whose :data:`UNDETERMINED`-prefixed columns are
    what the math left open. Which species names a pool's generator is the solver's choice; compare
    dimensions with :func:`differ`, never by name.

    ``inputs`` are the ids the condition table sets (a generator where used as a quantity), and
    ``condition_entries`` its ``(target, value)`` pairs, each value a formula: they equate a
    target with what it is set to, so a shipped total the condition table assigns to an initial
    amount gets that amount's dimension.
    """
    species = {sm.getSpecies(i).getId() for i in range(sm.getNumSpecies())}
    compartments = {sm.getCompartment(i).getId() for i in range(sm.getNumCompartments())}
    inputs = {str(i) for i in inputs} - species - compartments
    quantities = _quantities(sm, inputs)
    w = _Walker(compartments | (inputs - quantities))  # a switch input is a number

    def target(sid: str):
        return _NUMBER if sid in compartments else {sid: Fraction(1)}

    def rate_of(sid: str):
        return _NUMBER if sid in compartments else {sid: Fraction(1), TIME: Fraction(-1)}

    for i in range(sm.getNumReactions()):
        rxn = sm.getReaction(i)
        law = rxn.getKineticLaw()
        if law is None:
            continue
        w.local = {
            p.getId(): f"{rxn.getId()}.{p.getId()}"
            for p in (*law.getListOfLocalParameters(), *law.getListOfParameters())
        }
        flux = w.walk(law.getMath())
        w.local = {}
        if flux is _NUMBER:  # a bare number as a rate says nothing about anything
            continue
        # The participants share one dimension first, and the flux is that per time: posted the
        # other way round, a law missing its rate constant would relate two species by a time.
        participants = sorted(
            {r.getSpecies() for r in (*rxn.getListOfReactants(), *rxn.getListOfProducts())}
        )
        for sid in participants[1:]:
            w.equal(target(participants[0]), target(sid))
        if participants:
            w.equal(flux, rate_of(participants[0]))
    for i in range(sm.getNumRules()):
        rule = sm.getRule(i)
        dim = w.walk(rule.getMath())
        sid = rule.getVariable()
        if rule.isRate() and sid:
            w.equal(dim, rate_of(sid))
        elif rule.isAssignment() and sid:
            w.equal(dim, target(sid))
    for i in range(sm.getNumInitialAssignments()):
        ia = sm.getInitialAssignment(i)
        w.equal(w.walk(ia.getMath()), target(ia.getSymbol()))
    for i in range(sm.getNumEvents()):
        event = sm.getEvent(i)
        for part in (event.getTrigger(), event.getPriority(), event.getDelay()):
            if part is not None:
                w.walk(part.getMath())
        for ea in event.getListOfEventAssignments():
            w.equal(w.walk(ea.getMath()), target(ea.getVariable()))
    for sid, value in condition_entries:
        math = libsbml.parseL3Formula(str(value))
        if math is not None:
            w.equal(w.walk(math), target(str(sid)))

    system = _System(species, quantities)
    for form in w.hard:
        system.add(form)
    for form in w.soft:
        # A number summed onto a term contradicts the rest of the math as often as not
        # (`geneX + 1e-6` would make a species dimensionless), so it may only pin a parameter
        # nothing else did: never a species or an input, never against what is known.
        if system.pins_a_parameter(form):
            system.add(form)

    params = {sm.getParameter(i).getId() for i in range(sm.getNumParameters())}
    for sid, value in condition_entries:  # a shipped total may appear in no model math at all
        params |= {str(sid)} | _names(libsbml.parseL3Formula(str(value)))
    return {sid: system.solve(sid) for sid in sorted((params | species) - compartments)}


def _math(sm):
    """Every math tree of ``sm``: rate laws, rules, initial assignments and events."""
    for i in range(sm.getNumReactions()):
        law = sm.getReaction(i).getKineticLaw()
        if law is not None:
            yield law.getMath()
    for i in range(sm.getNumRules()):
        yield sm.getRule(i).getMath()
    for i in range(sm.getNumInitialAssignments()):
        yield sm.getInitialAssignment(i).getMath()
    for i in range(sm.getNumEvents()):
        event = sm.getEvent(i)
        for part in (event.getTrigger(), event.getPriority(), event.getDelay()):
            if part is not None:
                yield part.getMath()
        for ea in event.getListOfEventAssignments():
            yield ea.getMath()


def _names(math) -> set[str]:
    out: set[str] = set()
    stack = [math] if math is not None else []
    while stack:
        node = stack.pop()
        if node.getType() == libsbml.AST_NAME:
            out.add(node.getName())
        stack.extend(node.getChild(i) for i in range(node.getNumChildren()))
    return out


def _quantities(sm, inputs: set[str]) -> set[str]:
    """The ``inputs`` used as a quantity: the whole value of an assignment or initial assignment,
    or a term of a sum or a side of a comparison beside another term that is not a number — bare,
    or as the base of a power (``K^n + Ca^n``). ``1 - ActD`` does not count: summed only with a
    number, an input is the switch it looks like."""
    assigned = [
        *(sm.getInitialAssignment(i).getMath() for i in range(sm.getNumInitialAssignments())),
        *(sm.getRule(i).getMath() for i in range(sm.getNumRules()) if sm.getRule(i).isAssignment()),
    ]
    found = {
        m.getName()
        for m in assigned
        if m is not None and m.getType() == libsbml.AST_NAME and m.getName() in inputs
    }
    stack = [m for m in _math(sm) if m is not None]
    while stack:
        node = stack.pop()
        kids = [node.getChild(i) for i in range(node.getNumChildren())]
        stack.extend(kids)
        t = node.getType()
        if not (
            t == libsbml.AST_PLUS or (t == libsbml.AST_MINUS and len(kids) > 1) or t in _RELATIONAL
        ):
            continue
        if sum(1 for k in kids if _numeric_value(k) is None and k.getType() not in _NUMERIC) < 2:
            continue
        for k in kids:
            if k.getType() in (libsbml.AST_POWER, libsbml.AST_FUNCTION_POWER):
                k = k.getChild(0)
            if k is not None and k.getType() == libsbml.AST_NAME and k.getName() in inputs:
                found.add(k.getName())
    return found


def parameter_dimensions(problem) -> dict[str, Form] | None:
    """:func:`infer` for a PEtab problem: its model's SBML, with the condition table's targets as
    inputs. ``None`` when the problem has no SBML model to read."""
    from .reconcile import TARGET_ID, TARGET_VALUE

    model = getattr(problem, "model", None)
    sm = getattr(model, "sbml_model", None)
    if sm is None:
        return None
    cdf = getattr(problem, "condition_df", None)
    entries = []
    if cdf is not None and TARGET_ID in cdf.columns and TARGET_VALUE in cdf.columns:
        entries = [
            (str(t), v) for t, v in zip(cdf[TARGET_ID], cdf[TARGET_VALUE]) if t == t and v == v
        ]
    return infer(sm, inputs={t for t, _ in entries}, condition_entries=entries)


def differ(a: Form | None, b: Form | None) -> bool | None:
    """``True`` where two dimensions differ whatever their undetermined parts turn out to be,
    ``False`` where they are the same, ``None`` where it depends on what the math left open (or
    either is unknown)."""
    if a is None or b is None:
        return None
    diff = _add(a, b, Fraction(-1))
    if not diff:
        return False
    if any(c.startswith(UNDETERMINED) for c in diff):
        return None
    return True


def describe(dim: Form | None) -> str:
    """A dimension as a reader would write it (``Ca·time^-1``), for messages and tests."""
    if dim is None:
        return "unknown"
    if not dim:
        return "1"
    parts = []
    for col, v in sorted(dim.items()):
        name = "time" if col == TIME else col
        parts.append(name if v == 1 else f"{name}^{v}")
    return "·".join(parts)
