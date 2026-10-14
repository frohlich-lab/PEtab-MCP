"""Keep the SBML model and the PEtab parameter table consistent after a model edit.

Enforces the invariants:

* **Parametric initial conditions** — every species ``S`` gets a parametric initial assignment.
  A literal IC is parametrised as a fresh ``S_init`` parameter, but an author's own parametric IC
  is PRESERVED — whether a single parameter (``S = S_0``) or a COMPOUND expression of parameters
  (``Rec2 = ini_R1*ini_R2fold``, ``Exposed = exposed_init_conc*population``). Its parameters are
  the settable/estimatable IC and are synced into the parameter table.
* **Parameter-table sync** — every model parameter (rate constants + IC parameters) has a
  row in the parameter table; parameters no longer in the model are removed, *except*
  those used by the fixed observable/noise formulas (e.g. ``sigma_*``).
* **Orphan pruning** — species/parameters left unreferenced by any reaction, rule,
  initial assignment, or fixed table are dropped.
"""

from __future__ import annotations

import math
import re
from typing import Any

import libsbml
from petab.v2.C import (
    ESTIMATE,
    EXPERIMENT_ID,
    LOWER_BOUND,
    MEASUREMENT,
    NOISE_FORMULA,
    NOISE_PARAMETERS,
    NOMINAL_VALUE,
    OBSERVABLE_FORMULA,
    OBSERVABLE_ID,
    OBSERVABLE_PARAMETERS,
    TARGET_ID,
    TARGET_VALUE,
    UPPER_BOUND,
)

from .constants import (
    EXPONENT_LOWER_BOUND,
    EXPONENT_UPPER_BOUND,
    NOISE_SIGMA_MAX_FRACTION_OF_DATA_STD,
    NOISE_SIGMA_REPLICATE_MIN_DOF,
)
from .parameters import _default_bounds
from .problem import (
    LIN,
    LOG,
    LOG10,
    PARAMETER_SCALE,
    as_bool,
    observable_transformation,
    placeholder_ids,
    placeholders,
    split_overrides,
)

# Leading (?<![\w.]) so a digit-glued run is never read as an identifier: the `e5` in `1e5`
# and the `e` in `2.5e-3` (scientific-notation exponents) stay part of their numeric literal.
_IDENT_RE = re.compile(r"(?<![\w.])[A-Za-z_]\w*")

#: The t=0 value every SEEDED species is declared at (see :func:`seed_model_antimony`).
#:
#: The value is arbitrary and identical for every problem — it carries no information, and the
#: multistart ignores a nominal value anyway — but it may not be 0 or 1. An explicit 0 is the
#: author's-design skip in :func:`ensure_parametric_initial_conditions` (no ``<sid>_init`` is
#: minted at all — the state starts empty), and 1 is the unit-normalization convention
#: :func:`sync_parameter_table` holds an IC FIXED at; reconcile never re-touches a row once it
#: exists. A seed at either value would therefore leave every seeded species' initial condition
#: constant for the rest of the run — silently reinstating the frozen-IC trap of issue #90, on
#: exactly the species the caller is about to build a mechanism around. Any other positive value
#: lands on the ordinary path: estimated, on log10.
SEED_INITIAL_VALUE = 0.1

#: The ``model_id`` a model-less problem's model is created under; the same id gemot gives the
#: empty SBML it ships. Must not be an Antimony keyword.
ZERO_SHOT_MODEL_ID = "zero_shot_problem"

# Antimony GRAMMAR keywords: `species species;` does not parse, so an entity with one of these
# ids cannot be declared. Determined by probing libantimony, not from a spec — and pinned that
# way by test_seed_model.test_the_reserved_word_list_matches_libantimony, so a libantimony
# release that adds or drops a keyword is caught rather than silently mis-filtered.
#
# It covers the grammar ONLY. The other family of unusable ids — MathML's reserved constants
# (`time`, `pi`, `avogadro`, `exponentiale`, `true`/`false`, `inf`/`nan`) — is NOT listed here
# because :func:`_declarable_id` asks libSBML about those instead of keeping a second list that
# could drift out of step with the first.
_ANTIMONY_RESERVED = frozenset(
    {
        "DNA",
        "after",
        "at",
        "compartment",
        "const",
        "delete",
        "end",
        "event",
        "ext",
        "formula",
        "function",
        "gene",
        "has",
        "import",
        "in",
        "is",
        "model",
        "operator",
        "reaction",
        "species",
        "substanceOnly",
        "unit",
        "var",
    }
)


def _declarable_id(sid: str) -> bool:
    """Whether ``sid`` can be DECLARED as a model entity in the seed.

    Two ways an id is unusable, and each is answered by whoever actually knows:

    * it is a MathML reserved constant — libSBML decides. An L3 formula parses a plain
      identifier to ``AST_NAME``; every reserved name gets its own node type instead (``time`` is
      ``AST_NAME_TIME``, ``pi``/``avogadro``/``exponentiale`` are constants, ``true``/``false``
      are booleans, ``inf``/``nan`` are numeric literals). Declaring one is either rejected or
      silently means something other than a species.
    * it is an Antimony grammar keyword — :data:`_ANTIMONY_RESERVED`.

    This replaces a ``keyword.iskeyword`` test, which was the wrong language's reserved list in
    both directions: it rejected ``lambda``, a perfectly legal SBML/Antimony id (so the seed
    would omit a species its observables reference, fail to lint, and silently fall back to the
    broken empty skeleton), while passing ``species`` and ``compartment``, which really are
    unusable.
    """
    if not sid.isidentifier() or sid in _ANTIMONY_RESERVED:
        return False
    ast = libsbml.parseL3Formula(sid)
    return ast is not None and ast.getType() == libsbml.AST_NAME


# -- symbol helpers --------------------------------------------------------
def _ast_names(math) -> set[str]:
    names: set[str] = set()
    if math is None:
        return names
    stack = [math]
    while stack:
        node = stack.pop()
        if node is None:
            continue
        if node.getType() == libsbml.AST_NAME:
            names.add(node.getName())
        for i in range(node.getNumChildren()):
            stack.append(node.getChild(i))
    return names


def _exponent_symbols(sm) -> set[str]:
    """Names that appear anywhere in an EXPONENT position (``base ** exponent``) across the
    model's reaction rate laws, rules, and initial assignments.

    Such a parameter is a cooperativity / Hill exponent. On the default wide ``log10`` scale a
    fitted exponent can run to ~100 — a near-step-function that makes the ODE stiff and the fit
    plateau (a repressilator run fitted a Hill coefficient of 15). Estimating it on a LINEAR
    scale with tight bounds keeps it in the biophysically sane range. Only ``base**exp`` powers
    count — ``exp(x)`` (AST_FUNCTION_EXP) is not a cooperativity exponent and is ignored.
    """
    found: set[str] = set()

    def walk(node):
        if node is None:
            return
        t = node.getType()
        if t in (libsbml.AST_POWER, libsbml.AST_FUNCTION_POWER) and node.getNumChildren() == 2:
            found.update(_ast_names(node.getChild(1)))  # names in the exponent subtree
        for i in range(node.getNumChildren()):
            walk(node.getChild(i))

    for i in range(sm.getNumReactions()):
        kl = sm.getReaction(i).getKineticLaw()
        if kl is not None:
            walk(kl.getMath())
    for i in range(sm.getNumRules()):
        walk(sm.getRule(i).getMath())
    for i in range(sm.getNumInitialAssignments()):
        walk(sm.getInitialAssignment(i).getMath())
    return found


def _exponent_bounds(nominal) -> tuple[float, float]:
    """Tight LINEAR bounds for a cooperativity / Hill exponent (typical range ~1–8), always
    bracketing the nominal value so the fit starts feasible."""
    lb, ub = EXPONENT_LOWER_BOUND, EXPONENT_UPPER_BOUND
    if nominal is not None:
        n = float(nominal)
        lb, ub = min(lb, n), max(ub, n)
    return lb, ub


def formula_symbols(formula: Any) -> set[str]:
    """Best-effort identifier extraction from an observable/noise/condition formula.

    ``None`` and a pandas empty cell (``float('nan')``) yield an empty set — ``str(nan)`` would
    otherwise leak a bogus ``"nan"`` symbol into the keep/sigma/scaling sets. Numeric literals
    (including scientific notation) never contribute identifiers thanks to ``_IDENT_RE``.
    """
    if formula is None:
        return set()
    if not isinstance(formula, str):
        # A non-string cell is a pandas/NumPy scalar. An empty cell (NaN/Inf, or a non-float
        # scalar like numpy.float32(nan) / pandas <NA>) must not leak a bogus symbol via str().
        # Require a finite numeric coercion; anything else contributes no identifiers.
        try:
            if not math.isfinite(float(formula)):
                return set()
        except (TypeError, ValueError):
            return set()
    return set(_IDENT_RE.findall(str(formula)))


def formula_identifiers(formula: Any) -> set[str]:
    """Identifiers a formula REFERENCES, parsed rather than pattern-matched.

    :func:`formula_symbols` is a regex scan, which is right where a superset is harmless (the
    keep/prune sets) but wrong where each name becomes a model ENTITY: it cannot tell the symbol
    ``P`` in ``log10(P)`` from the function name ``log10``, so seeding a model from it would
    declare a species called ``log10``. libSBML's L3 parser resolves that — a function call is an
    ``AST_FUNCTION_*`` node and ``time`` is ``AST_NAME_TIME``, so neither is collected — and an
    unresolvable symbol still parses fine, which is exactly the case here (the names do not exist
    in any model yet).

    Falls back to :func:`formula_symbols` when the formula will not parse, so a malformed cell
    degrades to the old over-approximation instead of silently contributing nothing.
    """
    if formula is None:
        return set()
    text = str(formula) if isinstance(formula, str) else None
    if text is None:
        if not formula_symbols(formula):  # non-string, non-finite cell -> nothing to parse
            return set()
        text = str(formula)
    ast = libsbml.parseL3Formula(text)
    if ast is None:
        return formula_symbols(formula)
    return _ast_names(ast)


def seed_model_antimony(problem) -> str | None:
    """A CONTENT-FREE but VALID initial model for a problem that ships without one, or ``None``
    when the fixed tables name nothing to declare.

    A problem that ships model-less otherwise starts as an empty skeleton, and that
    skeleton is not a neutral placeholder — it is a *broken* PEtab problem (its
    observables reference entities no model defines) that AMICI's SBML import SEGFAULTS on,
    taking the server process with it. Every downstream guard exists to keep that one state
    unreachable. Making the initial model VALID instead removes the state.

    The seed is NOT a version: nothing is snapshotted at construction (``list_versions`` is empty
    until the first commit), so it cannot be restored or scored. What it must be is IMPORTABLE,
    because ``has_model()`` is an asserted invariant — see :func:`ensure_one_species`, which
    carries it with a placeholder so that nothing here has to pretend an observation symbol is a
    state.

    What it declares, and why that leaks nothing: one SPECIES per identifier the observable
    formulas name, plus each condition target (a per-condition target must be a model
    entity or petab's lint rejects the problem). Both tables are read-only and already fully
    visible through ``list_observables`` / ``list_conditions``, so the seed only restates what
    the caller can already read.

    Nothing else. In particular it does NOT declare the problem's other parameter-table rows.
    It used to, because petab lints a row no model declares as an "extraneous parameter" — but
    that put a pile of parameters the caller never authored into ``get_model``, reading as
    structure the seed does not have. They are simply pruned at v0 instead; what made that safe
    is that :func:`sync_parameter_table` RESTORES a shipped row when the caller's model
    reintroduces the parameter, so the shipped bounds/scale/priors come back with it rather than
    being replaced by defaults.

    NO reactions and NO rules, deliberately: the seed has no dynamics, which is what keeps it
    content-free (the mechanism is the thing the caller must author) and what makes
    ``model_authored`` a meaningful flag.

    Species are seeded at :data:`SEED_INITIAL_VALUE`, and the seed is then RECONCILED like any
    other model — same parametric ICs, same table sync (see
    :meth:`session.PetabSession._seed_initial_model` for the one thing that pass does
    differently).

    Each id is declared with an explicit ``species`` statement. Antimony infers a bare
    ``B = 0.1;`` to be a PARAMETER — it only promotes a symbol to a species when a reaction uses
    it, and the seed deliberately has no reactions — which would leave the observable formulas
    referencing a parameter and the whole seed pointless.
    """
    ids: set[str] = set()
    odf = getattr(problem, "observable_df", None)
    if odf is not None and OBSERVABLE_FORMULA in odf.columns:
        for value in odf[OBSERVABLE_FORMULA]:
            ids |= formula_identifiers(value)
    # A condition-table target is a per-condition target; petab requires it to be a model entity,
    # so the seed has to declare it too or v0 does not lint.
    ids |= condition_target_symbols(problem)
    pdf = getattr(problem, "parameter_df", None)
    rows = {str(p) for p in getattr(pdf, "index", [])}
    # The observables' declared PLACEHOLDERS are not entities: petab substitutes each from the
    # measurement table's observableParameters / noiseParameters column at import, and whatever it
    # resolves to (a scaling factor, a sigma) already has its own parameter-table row. Declaring
    # the placeholder itself would invent a state the real model never has.
    placeholder = placeholder_ids(odf)

    def _declarable(sid: str) -> bool:
        return _declarable_id(sid) and sid not in placeholder

    # A symbol with a parameter-table row is a parameter, not a species — the table is the
    # authority on what it IS, and minting a species for it would collide with its own row.
    species = sorted(sid for sid in ids - rows if _declarable(sid))
    if not species:
        return None

    # A `readout_*` identifier is a MEASUREMENT MAPPING the author must later define with an
    # assignment rule — seeding it as a SPECIES taught the wrong lesson: agents copied the
    # declaration into their own models, wired the readout into reactions as a state, and then
    # could not assign it (4 of the first blinded sweep's 27 rejections were exactly this
    # over-determination refusal, all on fujita, whose mappings are non-identity sums). A constant
    # PARAMETER lints and simulates identically at v0 (flat, like a species with no reactions),
    # keeps the seed dynamics-free (a seeded `:=` rule would flip `has_dynamics` and with it
    # `model_authored`), and suggests nothing about being a state.
    #
    # EVERY readout, with no exception: :func:`ensure_one_species` carries the "at least one
    # species" invariant now, so the seed never has to make a measurement mapping look like a
    # state to stay importable. The two earlier attempts both did: first unconditionally (which
    # broke a pure-readout problem outright), then keyed on "does a non-readout entity exist?",
    # which quietly left every readout a species on boehm, blasi, crauste, perelson and rahman —
    # boehm's mappings being as non-identity as fujita's.
    demote = {sid for sid in species if sid.startswith("readout_")}
    body = "\n".join(
        (
            f"  {sid} = {SEED_INITIAL_VALUE:g};"
            if sid in demote
            else f"  species {sid};\n  {sid} = {SEED_INITIAL_VALUE:g};"
        )
        for sid in species
    )
    return f"model petab_mcp_seed\n{body}\nend"


def has_dynamics(sm) -> bool:
    """Whether the model can MOVE: it has at least one reaction, or a rate/assignment rule.

    The single definition of "a mechanism has been authored", shared by three places that must
    agree or the invariant they form comes apart: ``session.model_authored`` (what
    ``describe_problem`` reports), the ``make_changes`` guard that REFUSES to commit a model
    without dynamics, and :func:`validation._unused_condition_target_issues` (which asks a
    question only a mechanism can answer). Because the guard refuses exactly what this returns
    False for, a dynamics-free model can only ever be the content-free seed — never something a
    caller authored.

    Initial assignments deliberately do NOT count: reconcile gives every species a parametric IC,
    so counting them would make any bag of species look like a mechanism.
    """
    return sm.getNumReactions() > 0 or sm.getNumRules() > 0


def species_ids(sm) -> set[str]:
    return {sm.getSpecies(i).getId() for i in range(sm.getNumSpecies())}


def parameter_ids(sm) -> set[str]:
    return {sm.getParameter(i).getId() for i in range(sm.getNumParameters())}


# -- parametric initial conditions -----------------------------------------
def _ic_symbol_for(sm, sid: str) -> str | None:
    """Return the parametric-IC parameter id for species ``sid`` if its initial assignment is
    already a SINGLE parameter — whether the server's own ``<sid>_init`` OR one the author
    supplied (e.g. writing ``p1 = p1_0`` with a ``p1_0`` parameter).

    Accepting the author's own IC parameter is essential: otherwise a model that parametrises
    its initial conditions under a different name gets a fresh ``<sid>_init`` (defaulting to 0)
    forced on top, silently discarding the intended initial value and starting the species at 0.
    """
    for i in range(sm.getNumInitialAssignments()):
        ia = sm.getInitialAssignment(i)
        if ia.getSymbol() == sid:
            names = _ast_names(ia.getMath())
            if len(names) == 1:
                (name,) = names
                if sm.getParameter(name) is not None:  # a plain parameter, not an expression
                    return name
            return None
    return None


def _ic_expression_params(sm, sid: str) -> list[str]:
    """Parameters referenced by species ``sid``'s initial assignment when it is a COMPOUND
    parametric expression (more than a single bare parameter — that single case is handled by
    :func:`_ic_symbol_for`). Returns ``[]`` when the species has no initial assignment or the
    assignment references no model parameter (a bare literal, or an expression of only species).

    Preserving such an assignment keeps a published compound initial condition — e.g. schwen's
    ``Rec2 = ini_R1*ini_R2fold`` or oliveira's ``Exposed = exposed_init_conc*population`` — instead
    of overwriting it with a fresh fixed ``<sid>_init`` and silently discarding the estimated
    seed(s). Its parameters are picked up as IC parameters by :func:`_ic_parameter_ids`.
    """
    for i in range(sm.getNumInitialAssignments()):
        ia = sm.getInitialAssignment(i)
        if ia.getSymbol() == sid:
            names = _ast_names(ia.getMath())
            return sorted(n for n in names if sm.getParameter(n) is not None)
    return []


def _set_parametric_ic(sm, sid: str, value: float) -> str:
    init_id = f"{sid}_init"
    if sm.getParameter(init_id) is None:
        p = sm.createParameter()
        p.setId(init_id)
        p.setConstant(True)
        p.setValue(float(value))
    # Ensure there is exactly one initialAssignment for this species and that it
    # points at the parametric-IC parameter.
    ia = None
    for i in range(sm.getNumInitialAssignments()):
        cand = sm.getInitialAssignment(i)
        if cand.getSymbol() == sid:
            ia = cand
            break
    if ia is None:
        ia = sm.createInitialAssignment()
        ia.setSymbol(sid)
    ia.setMath(libsbml.parseL3Formula(init_id))
    return init_id


def _drop_initial_assignment(sm, sid: str, *, drop_sole_parameter: bool = False) -> bool:
    """Remove any initialAssignment for ``sid``; True if one was there.

    ``drop_sole_parameter`` also removes the IC parameter the assignment pointed at, when that
    parameter is the assignment's SOLE symbol (the ``<sid> = <sid>_init`` shape). Dropping the
    assignment alone leaves that parameter referenced by nothing, and because it is still an SBML
    parameter it stays in the table as an ESTIMATED row that no longer affects the trajectory --
    a free ``ln(n)`` of BIC per fit with zero fit effect, which is precisely the artefact
    :func:`ensure_parametric_initial_conditions` exists to avoid. Not applied to a COMPOUND
    expression (``Rec = ini_R1*ini_R2fold``): those symbols are the author's own estimated seeds
    and may well be used elsewhere.
    """
    for i in range(sm.getNumInitialAssignments()):
        ia = sm.getInitialAssignment(i)
        if ia.getSymbol() != sid:
            continue
        sole = None
        if drop_sole_parameter:
            names = _ast_names(ia.getMath())
            if len(names) == 1:
                (only,) = names
                if sm.getParameter(only) is not None:
                    sole = only
        sm.removeInitialAssignment(i)
        if sole is not None:
            sm.removeParameter(sole)
        return True
    return False


def assignment_rule_targets(sm) -> set[str]:
    """Symbols set by an ASSIGNMENT rule (``X := expr``), excluding rate-rule targets.

    Narrower than :func:`_rule_target_symbols` on purpose. An assignment rule determines its target
    at EVERY time including t=0, and SBML forbids an initialAssignment on such a symbol
    ("Cannot set a value using both an initial assignment and an assignment rule simultaneously").
    A RATE rule (``X' = expr``) is the opposite case: it integrates from an initial value and so
    still needs one. Anything that skips ICs must therefore split the two, not lump them.
    """
    out: set[str] = set()
    for i in range(sm.getNumRules()):
        rule = sm.getRule(i)
        if rule.isSetVariable() and rule.isAssignment():
            out.add(rule.getVariable())
    return out


def ensure_parametric_initial_conditions(
    sm, skip_species: set[str] | None = None, protect_ids: set[str] | None = None
) -> list[str]:
    """Give every species a parametric IC; returns the IC parameter ids created/kept.

    ``skip_species`` are species whose initial condition is set PER CONDITION by a condition-table
    column (a PEtab species-initial override, e.g. armistead's per-genotype metabolite baselines).
    Their IC is neither a free parameter nor a fixed literal — the condition table drives it — so we
    leave the declared initial in place and do NOT mint a ``<sid>_init``. Minting one would add an
    estimated, non-identifiable parameter that the per-condition override silently supersedes
    (inflating k with no fit effect), which is exactly the artefact this avoids.

    Species driven by an ASSIGNMENT rule are skipped for a different and harder reason: minting an
    IA on them builds an SBML document that libSBML rejects, so the agent's own model — which
    validated cleanly on its own — was rejected for a construct the SERVER wrote. That accounted for
    69 rejected ``make_changes`` calls in the Aug 8-10 sweep, and 48% of them were followed by a
    byte-identical retry because the error described something the agent had not done. Verified by
    round-tripping all 152 affected models through libAntimony -> libsbml.checkConsistency():
    152/152 clean, 0 initialAssignment/rule-target collisions before reconcile touched them.
    Rate-rule targets are NOT skipped — see :func:`assignment_rule_targets`."""
    rule_driven = assignment_rule_targets(sm)
    # NOT unioned into `skip`: the `rule_driven` branch below runs first and `continue`s, so a union
    # here could never change a decision -- it would only read as if there were two paths.
    skip = skip_species or set()
    # Ids that must exist however the species is initialised: a SHIPPED-FIXED `<sid>_init`.
    # Without this, the zero/unset skip below silently drops one, `sync_parameter_table` removes
    # its row, and `validate_problem` then REJECTS the commit ("ships FIXED ... but this edit
    # REMOVED it from the model") -- with no edit the agent could make to satisfy it short of
    # misstating the biology. Minting it back is the only outcome that keeps such a problem
    # editable, and a shipped row is the problem author's decision, not this convention's.
    protect = protect_ids or set()
    ic_params: list[str] = []
    for i in range(sm.getNumSpecies()):
        sp = sm.getSpecies(i)
        sid = sp.getId()
        if sid in rule_driven:
            # Skipping is not enough on its own: a store written by an earlier reconcile (before
            # this skip existed) can carry an IA that is illegal alongside the rule. Drop it -- and
            # its now-unreferenced `<sid>_init` with it, or the parameter table keeps estimating a
            # value the assignment rule overrides at t=0. Heals a corrupted version store instead of
            # inheriting the fault from it.
            _drop_initial_assignment(sm, sid, drop_sole_parameter=True)
            continue
        if sid in skip:  # condition-column-driven initial; not a parameter
            continue
        existing = _ic_symbol_for(sm, sid)
        if existing is not None:
            ic_params.append(existing)
            continue
        expr_params = _ic_expression_params(sm, sid)
        if expr_params:
            # A COMPOUND parametric IC (e.g. Rec2 = ini_R1*ini_R2fold): keep the expression as-is;
            # its parameters ARE the settable/estimatable IC (synced by _ic_parameter_ids). Do NOT
            # clobber it with a fresh <sid>_init, which would discard the estimated seed(s).
            ic_params.extend(expr_params)
            continue
        value = None
        if sp.isSetInitialConcentration():
            value = sp.getInitialConcentration()
        elif sp.isSetInitialAmount():
            value = sp.getInitialAmount()
        # `value is None` is the UNSET species (`species B;` with no assignment): libantimony
        # sets neither initialConcentration nor initialAmount, and SBML reads it as an empty
        # pool. It is the same state as an explicit `B = 0;` and must take the same branch —
        # otherwise the two spellings produce different parameter tables and the regenerating
        # `<sid>_init` artifact returns one omitted assignment later.
        if (value is None or value == 0.0) and f"{sid}_init" not in protect:
            # An EXPLICIT zero initial ("IPR_I = 0;") is the author's design — the state starts
            # empty — not a missing value to parametrise. Same class of reason as the
            # condition-column skip above: minting `<sid>_init` here adds an estimated
            # zero-nominal parameter nobody asked for, and because this runs on EVERY commit and
            # the drafts are refreshed from the committed model, the row REGENERATES no matter
            # how often it is edited away. Measured (sneyd/fujita A/B, draft arm): one lead
            # re-delegated the deletion of a minted `IPR_S_init` 17 times, misattributing the
            # server's artifact to its editor subagent; the only two unfixable lint dead-ends in
            # the arm were log10-scale-on-zero-nominal conflicts on minted `_init` rows. A zero
            # nominal also breaks log-scale estimation by construction, which is why these rows
            # kept surfacing as lint issues no requested change could legally fix.
            continue
        ic_params.append(_set_parametric_ic(sm, sid, value if value is not None else 0.0))
    return ic_params


# -- reference set ---------------------------------------------------------
def _reaction_species(sm) -> set[str]:
    used: set[str] = set()
    for i in range(sm.getNumReactions()):
        rxn = sm.getReaction(i)
        for j in range(rxn.getNumReactants()):
            used.add(rxn.getReactant(j).getSpecies())
        for j in range(rxn.getNumProducts()):
            used.add(rxn.getProduct(j).getSpecies())
        for j in range(rxn.getNumModifiers()):
            used.add(rxn.getModifier(j).getSpecies())
    return used


def _model_math_names(sm) -> set[str]:
    names: set[str] = set()
    for i in range(sm.getNumReactions()):
        kl = sm.getReaction(i).getKineticLaw()
        if kl is not None:
            names |= _ast_names(kl.getMath())
    for i in range(sm.getNumInitialAssignments()):
        names |= _ast_names(sm.getInitialAssignment(i).getMath())
    for i in range(sm.getNumRules()):
        names |= _ast_names(sm.getRule(i).getMath())
    return names


def _rule_target_symbols(sm) -> set[str]:
    """Ids assigned by an assignment- or rate-rule — the quantities the MODEL computes.

    PEtab forbids these in the parameter table (they are not free parameters), and mirrors that in
    ``get_valid_parameters_for_parameter_table``. Kept as its own helper because
    :func:`input_species` needs the same set narrowed to species, while the parameter-table sync
    needs it un-narrowed.
    """
    out: set[str] = set()
    for i in range(sm.getNumRules()):
        rule = sm.getRule(i)
        if rule.isSetVariable() and (rule.isAssignment() or rule.isRate()):
            out.add(rule.getVariable())
    return out


def input_species(sm) -> set[str]:
    """Species that are model INPUTS / forcing terms rather than free dynamic states: held
    constant, on the boundary, or driven by an assignment- or rate-rule (a forcing function).

    Their trajectory is imposed, not fitted — like a condition target — so they are
    legitimately absent from every observable and must NOT be flagged as an unconstrained
    hidden state (e.g. a genotype flag ``S_on`` or an input/forcing level ``insulin_level``)."""
    inputs: set[str] = set()
    for i in range(sm.getNumSpecies()):
        sp = sm.getSpecies(i)
        if sp.getConstant() or sp.getBoundaryCondition():
            inputs.add(sp.getId())
    for i in range(sm.getNumRules()):
        rule = sm.getRule(i)
        if rule.isSetVariable() and (rule.isAssignment() or rule.isRate()):
            inputs.add(rule.getVariable())
    return inputs & species_ids(sm)


def species_influence_graph(sm) -> dict[str, set[str]]:
    """Directed 'influences' graph over species: ``X -> Y`` when ``X`` enters the rate at which
    ``Y`` changes — i.e. ``X`` is a reactant/modifier of, or appears in the rate law of, a
    reaction for which ``Y`` is a reactant or product.

    A pure *product* of a reaction does not influence its other species (it is downstream), so a
    dead-end sink ``B -> D`` yields no outgoing edge from ``D``: ``D`` can reach no observed
    species. That reachability is exactly what distinguishes a genuinely disconnected hidden
    state (worth flagging) from a species that feeds — however indirectly — an observable (an
    enzyme, a substrate, an upstream regulator)."""
    species = species_ids(sm)
    edges: dict[str, set[str]] = {}
    for i in range(sm.getNumReactions()):
        rxn = sm.getReaction(i)
        reactants = {rxn.getReactant(j).getSpecies() for j in range(rxn.getNumReactants())}
        products = {rxn.getProduct(j).getSpecies() for j in range(rxn.getNumProducts())}
        modifiers = {rxn.getModifier(j).getSpecies() for j in range(rxn.getNumModifiers())}
        kl = rxn.getKineticLaw()
        rate_syms = _ast_names(kl.getMath()) if kl is not None else set()
        drivers = (reactants | modifiers | rate_syms) & species
        changed = reactants | products
        for x in drivers:
            edges.setdefault(x, set()).update(changed)
    return edges


def _observable_and_measurement_symbols(problem) -> set[str]:
    """Symbols referenced by the observable/noise formulas AND their measurement-level
    ``observableParameters``/``noiseParameters`` overrides — shared by
    :func:`table_referenced_symbols` (which also folds in condition-table symbols) and
    :func:`observation_model_symbols` (which deliberately does NOT)."""
    names: set[str] = set()
    odf = problem.observable_df
    if odf is not None:
        for col in (OBSERVABLE_FORMULA, NOISE_FORMULA):
            if col in odf.columns:
                for val in odf[col]:
                    names |= formula_symbols(val)
    mdf = problem.measurement_df
    if mdf is not None:
        for col in (OBSERVABLE_PARAMETERS, NOISE_PARAMETERS):
            if col in mdf.columns:
                for val in mdf[col]:
                    names |= formula_symbols(val)
    return names


def table_referenced_symbols(problem) -> set[str]:
    """Symbols the *fixed* tables (observables, conditions) depend on."""
    names = _observable_and_measurement_symbols(problem) | condition_target_symbols(problem)
    cdf = problem.condition_df
    if cdf is not None and TARGET_VALUE in cdf.columns:
        for val in cdf[TARGET_VALUE]:
            names |= formula_symbols(val)
    return names


def condition_target_symbols(problem) -> set[str]:
    """Parameters/species the CONDITION TABLE targets (its ``targetId`` column).

    A model parameter a condition targets is set PER CONDITION — a PEtab condition change — so it
    must NOT be written into the parameter table: PEtab rejects it there ("… present in both the
    condition table and the parameter table"). Excluding these from the parameter-table sync lets
    a benchmark condition table with parametric inputs (EGF doses, insulin steps, rate multipliers,
    genotype switches) ship VERBATIM instead of being rewritten onto species. Mirrors petab's own
    lint, which forbids a target in both tables. Target *values* that reference an estimated
    parameter (a shared dose amplitude) are a different thing — those stay, via
    :func:`table_referenced_symbols`."""
    cdf = problem.condition_df
    if cdf is None or TARGET_ID not in cdf.columns:
        return set()
    return {str(t) for t in cdf[TARGET_ID].dropna()}


def observation_model_symbols(problem) -> set[str]:
    """Symbols belonging to the FIXED *observation model* specifically — observable-scaling
    factors and noise sigmas resolved through the observable/noise formulas and their
    measurement-level ``observableParameters``/``noiseParameters`` overrides.

    Deliberately EXCLUDES condition-table symbols (those are legitimate model DRIVING INPUTS —
    e.g. a dose species set by a condition target — not observation-model constants).
    Used to catch a model that reuses one of these as a mechanistic rate constant instead of
    introducing a dedicated parameter for it (see ``validation._observation_model_reuse_issues``):
    a run once "fit" the Brannmark benchmark by wiring its four fixed observable-scaling
    factors directly into the reaction rate laws instead of the ~14 real kinetic parameters the
    scenario expects, and by letting its noise sigmas float to 20-60x their nominal scale — both
    let a fundamentally under-mechanistic model pass its own goodness-of-fit check.
    """
    return _observable_and_measurement_symbols(problem)


def noise_sigma_ids(problem) -> set[str]:
    """Parameter ids that serve specifically as a NOISE sigma — resolved through the
    observable table's ``noiseFormula`` column or the measurement table's ``noiseParameters``
    overrides. Narrower than :func:`observation_model_symbols`, which also includes
    observable-SCALING factors (resolved through ``observableFormula``/``observableParameters``)
    that don't get the data-driven bound treatment below."""
    names: set[str] = set()
    odf = problem.observable_df
    if odf is not None and NOISE_FORMULA in odf.columns:
        for val in odf[NOISE_FORMULA]:
            names |= formula_symbols(val)
    mdf = problem.measurement_df
    if mdf is not None and NOISE_PARAMETERS in mdf.columns:
        for val in mdf[NOISE_PARAMETERS]:
            names |= formula_symbols(val)
    return names


def plausibility_sigma_ids(problem) -> set[str]:
    """:func:`noise_sigma_ids` MINUS the observable-scaling symbols — the set whose fitted values
    are worth judging as a noise sd.

    A symbol that ALSO resolves through the observable side is a scaling factor whose product with
    the real sigma forms the sd (fiedler's per-gel ``s_pMek_*`` appear in both ``noiseParameters``
    and ``observableParameters``). Scoring each as if it WERE the sd flagged them 900x "inflated"
    while the real sigmas read "collapsed", on a scale-degenerate manifold where which coordinate
    looks large is arbitrary.

    Deliberately a SEPARATE function rather than a narrowing of ``noise_sigma_ids`` itself: that
    primitive has other consumers with different needs. In particular
    :func:`classify_inner_parameters` relies on a parameter being in BOTH sets to detect the
    ambiguous dual-role case and leave it OUTER — narrowing the primitive made the two sets
    disjoint by construction, silently reclassifying fiedler's 8 gel factors as hierarchical
    scaling parameters and handing them to an inner solver whose independence assumption they
    violate.
    """
    return noise_sigma_ids(problem) - observable_scaling_ids(problem)


def observable_scaling_ids(problem) -> set[str]:
    """Parameter ids resolved through the observable table's ``observableFormula`` column or
    the measurement table's ``observableParameters`` overrides — i.e. observable-scaling /
    offset factors. The counterpart of :func:`noise_sigma_ids` on the observable side; used to
    classify inner parameters for hierarchical optimization (see
    :func:`classify_inner_parameters`)."""
    names: set[str] = set()
    odf = problem.observable_df
    if odf is not None and OBSERVABLE_FORMULA in odf.columns:
        for val in odf[OBSERVABLE_FORMULA]:
            names |= formula_symbols(val)
    mdf = problem.measurement_df
    if mdf is not None and OBSERVABLE_PARAMETERS in mdf.columns:
        for val in mdf[OBSERVABLE_PARAMETERS]:
            names |= formula_symbols(val)
    return names


# Inner-parameter type tags (pyPESTO's ``parameterType`` column values). Kept as local
# literals so this module stays dependency-light (no pypesto import); fitting.py maps them to
# pyPESTO's ``InnerParameterType`` when it builds the hierarchical objective.
INNER_TYPE_SIGMA = "sigma"
INNER_TYPE_SCALING = "scaling"

# Parameter-role tags surfaced by ``get_parameters`` (see :func:`parameter_roles`). They name
# what a parameter *is* in the problem, so the agent can tell a mechanistic rate constant apart
# from an observation-model factor (a scaling/offset or noise sigma) or a structural constant —
# the distinction the role-enforcement checks in :mod:`validation` are about.
ROLE_NOISE_SIGMA = "noise_sigma"
ROLE_OBSERVABLE_SCALING = "observable_scaling"
ROLE_INITIAL_CONDITION = "initial_condition"
ROLE_EXPONENT = "exponent"
ROLE_KINETIC = "kinetic"
ROLE_CONSTANT = "constant"


def parameter_roles(problem, extra_ids=()) -> dict[str, str]:
    """Best-effort role for every parameter-table id — ``{parameter_id: role}``.

    ``extra_ids`` are classified alongside the table's own rows. Used for the parameters a
    problem SHIPS that the current model does not declare (see
    :func:`parameters.parameters_to_records`): they are absent from the table, but the caller
    still has to be told what they are. Every rule below reads the fixed tables or the model, so
    it applies to an id whether or not it currently has a row.

    The role names what the parameter *is* (in priority order): a NOISE sigma or an
    observable SCALING/offset factor (the fixed observation model, resolved through the
    observable/noise formulas or their measurement-level overrides), an INITIAL-CONDITION
    parameter, a cooperativity/Hill EXPONENT, a mechanistic KINETIC constant (a symbol used in
    a reaction rate law / rule / initial assignment), or an otherwise-unreferenced CONSTANT.

    Surfaced by ``get_parameters`` so a scaling factor or a structural constant is legible as
    such — the two role-enforcement checks (a scaling factor wired into a rate law; a fixed
    constant made estimated) are about exactly this classification. Observation-model roles are
    checked first so a reused parameter still reads as what it fundamentally is.
    """
    model = getattr(problem, "model", None)
    sm = model.sbml_model if model is not None else None
    sigmas = noise_sigma_ids(problem)
    scalings = observable_scaling_ids(problem)
    ic_params = _ic_parameter_ids(sm) if sm is not None else set()
    exp_params = _exponent_symbols(sm) if sm is not None else set()
    model_math = _model_math_names(sm) if sm is not None else set()

    roles: dict[str, str] = {}
    for raw in [*problem.parameter_df.index, *extra_ids]:
        pid = str(raw)
        if pid in sigmas:
            roles[pid] = ROLE_NOISE_SIGMA
        elif pid in scalings:
            roles[pid] = ROLE_OBSERVABLE_SCALING
        elif pid in ic_params:
            roles[pid] = ROLE_INITIAL_CONDITION
        elif pid in exp_params:
            roles[pid] = ROLE_EXPONENT
        elif pid in model_math:
            roles[pid] = ROLE_KINETIC
        else:
            roles[pid] = ROLE_CONSTANT
    return roles


def classify_inner_parameters(problem) -> dict[str, str]:
    """Which ESTIMATED parameters can be solved analytically in a hierarchical (inner) problem,
    and as what type — ``{parameter_id: "sigma" | "scaling"}``.

    A parameter is inner-eligible only if it belongs purely to the fixed OBSERVATION model:
    a noise sigma (via ``noiseFormula`` / measurement ``noiseParameters``) → ``"sigma"``, or an
    observable-scaling factor (via ``observableFormula`` / ``observableParameters``) →
    ``"scaling"``. A parameter that is BOTH (or that also appears in the model's own math — a
    reused observation-model parameter, which ``validation`` already rejects) is left OUT: it is
    not cleanly separable, so it stays an ordinary outer parameter. Only currently-ESTIMATED
    parameters are returned (a fixed sigma/scaling has nothing to solve).

    Offsets are deliberately NOT classified here (detecting an additive offset needs formula
    parsing pyPESTO does itself); such a parameter simply stays outer. Hierarchical optimization
    is best-effort — see ``fitting`` — so anything this misclassifies fails pyPESTO's own
    validation there and falls back to a joint fit rather than corrupting anything.
    """
    sigmas = noise_sigma_ids(problem)
    scalings = observable_scaling_ids(problem)
    model_names = _model_math_names(problem.model.sbml_model)
    out: dict[str, str] = {}
    for pid in sorted((sigmas | scalings)):
        if not is_parameter_estimated(problem, pid):
            continue
        if pid in model_names:
            continue  # reused inside the model's own math — not a clean inner parameter
        in_sigma, in_scaling = pid in sigmas, pid in scalings
        if in_sigma and not in_scaling:
            out[pid] = INNER_TYPE_SIGMA
        elif in_scaling and not in_sigma:
            out[pid] = INNER_TYPE_SCALING
        # both → ambiguous; leave outer
    return out


def _measurement_rows_for_sigma(problem, pid: str):
    """Measurement rows whose noise is modelled by sigma parameter ``pid`` — matched via a
    direct ``noiseParameters`` override, or (for a sigma named directly in the fixed
    ``noiseFormula`` rather than through a declared noise placeholder) via the observable(s)
    whose formula names it. ``None`` if no match/no data."""
    mdf = problem.measurement_df
    if mdf is None or MEASUREMENT not in mdf.columns:
        return None
    if NOISE_PARAMETERS in mdf.columns:
        mask = mdf[NOISE_PARAMETERS].apply(
            lambda cell: cell == cell and pid in {p.strip() for p in str(cell).split(";")}
        )
        if mask.any():
            return mdf[mask]
    odf = problem.observable_df
    if odf is not None and NOISE_FORMULA in odf.columns and OBSERVABLE_ID in mdf.columns:
        obs_ids = [
            oid for oid, formula in odf[NOISE_FORMULA].items() if pid in formula_symbols(formula)
        ]
        if obs_ids:
            return mdf[mdf[OBSERVABLE_ID].isin(obs_ids)]
    return None


def _observable_transformations(problem, rows) -> set[str]:
    """The coordinates (``lin``/``log``, see :func:`problem.observable_transformation`) the
    observables of the given measurement rows have their residuals in. ``{"lin"}`` when nothing
    says otherwise — PEtab's default."""
    odf = problem.observable_df
    if odf is None or OBSERVABLE_ID not in rows.columns:
        return {LIN}
    out = {
        observable_transformation(odf, oid)
        for oid in rows[OBSERVABLE_ID].dropna().unique()
        if oid in odf.index
    }
    return out or {LIN}


def _transform_values(values, transformation: str):
    """Measurement values on the ESTIMATION scale. A ``log``/``log10`` observable's residuals —
    and therefore its noise sigma — live in log space, so the scale a sigma must be judged
    against is the std of the TRANSFORMED data, not of the raw linear values (the two differ by
    orders of magnitude; see issue: the gate was dimensionally invalid). Non-positive values are
    dropped, being undefined under a log transform."""
    import numpy as np

    if transformation == LOG:
        v = values[values > 0]
        return np.log(v) if len(v) else v
    if transformation == LOG10:
        v = values[values > 0]
        return np.log10(v) if len(v) else v
    return values


def sigma_noise_coordinate(problem, pid: str, rows=None) -> str | None:
    """The coordinate in which ``pid``'s noise is ADDITIVE (homoscedastic) — ``lin`` or ``log`` —
    which is the only coordinate where "sigma vs data spread" is a meaningful ratio.

    Three cases, unified:
      * the noise distribution is a log one — residuals and sigma already live there.
      * a PROPORTIONAL ``noiseFormula`` (``noiseParameter1_X*X``) on a ``lin`` observable — sigma =
        cv*y is additive in LOG space with sd ~= cv (since d(ln y) = dy/y), so the parameter is
        directly comparable to the std of ``ln`` data. This is why a bare CV must never be compared
        against a linear std.
      * otherwise ``lin``.
    ``None`` when the covered observables mix transformations (no single comparable coordinate)."""
    if rows is None:
        rows = _measurement_rows_for_sigma(problem, pid)
    if rows is None or rows.empty:
        return None
    transformations = _observable_transformations(problem, rows)
    if len(transformations) != 1:
        return None
    t = next(iter(transformations))
    if t in (LOG, LOG10):
        return t
    # A PROPORTIONAL sd (cv*y) on a lin observable is additive in LOG space with sd ~= cv, so the
    # cv is comparable to the std of ln y. Decided from the symbolic form (see sigma_noise_form),
    # never from the spread of the data. A form we cannot summarise (mixed additive+proportional)
    # returns None so the caller fails open instead of judging on an arbitrary scale.
    form = sigma_noise_form(problem, pid, rows)
    if form == "proportional":
        return LOG
    if form == "additive":
        return t
    return None


def sigma_data_std(problem, pid: str, fitted_values=None) -> float | None:
    """Measurement std of the observable(s) whose noise sigma parameter ``pid`` models — computed
    ON THE OBSERVABLE'S ESTIMATION SCALE, so it is dimensionally comparable to the fitted sigma.

    This is the basis for the bound cap (see :func:`sigma_upper_bound_cap`) and for reporting a
    fitted sigma against its reference (``fitting.noise_sigma_report``). For a ``log``/``log10``
    observable the sigma is a log-space sd, so the std is taken over the TRANSFORMED measurements;
    comparing it against the raw linear std made a correct sigma look collapsed by many orders of
    magnitude. ``None`` if ``pid`` isn't a resolvable sigma, or there's too little data, or the
    observables it covers mix transformations (no single comparable scale — fail open rather than
    judge on the wrong one).

    When an observable carries a per-gel AMPLITUDE (see :func:`_observable_amplitude`), its
    measurements are divided by that gain BEFORE pooling and before any log transform. Pooling the
    raw values instead would count the BETWEEN-GEL gain differences as data spread: on fiedler that
    inflates the pERK std from 0.096 to 0.688 (~7x), which is not noise the sigma has to cover.
    Dividing in linear space is what makes it a pure normalisation — under log it is a per-gel
    offset, so the shift cancels the same way while the within-gel spread is untouched.

    KNOWN TRADE-OFF, do not "fix" one half without the other. Normalising by the FIT's gains is what
    makes the sd/spread ratio invariant under a legitimate gain/sigma re-split (fiedler: gains x10
    with sigma /10 leaves the fraction at 0.469619 exactly), and that invariance is the whole point.
    The price is that the denominator is no longer a property of the measurements alone: gains that
    do NOT match the data's gel structure scatter y/gain and inflate the pooled std, which raises the
    cap. Constructed case — gains {1, 0.001} where the truth is {1, 10} — lifts the std from 1.118 to
    14790 and lets a sigma 10,000x the truth read "ok". It is not a free win for an agent, since
    gains also enter the observable formula, so distorting them wrecks the mean fit and the run
    fails on BIC instead; but the sigma gate alone does not catch it. Using NOMINAL gains would close
    that hole and reopen the bigger one: any honest fit whose gains land away from nominal would then
    be judged in a coordinate its sigma is not expressed in. There is no fit-independent choice that
    keeps the invariance — the normaliser is a fitted quantity by construction.
    """
    rows = _measurement_rows_for_sigma(problem, pid)
    if rows is None or rows.empty:
        return None
    coordinate = sigma_noise_coordinate(problem, pid, rows)
    if coordinate is None:
        return None  # mixed scales -> no single meaningful reference; don't flag
    normalised = _gain_normalised_series(problem, pid, rows, fitted_values)
    if normalised is None:
        return None
    import numpy as np

    values = np.concatenate(normalised) if normalised else np.asarray([], dtype=float)
    values = _transform_values(values, coordinate)
    if len(values) < 2:
        return None
    std = float(values.std(ddof=0))
    return std if std > 0 else None


def _gain_normalised_series(problem, pid: str, rows, fitted_values=None):
    """``[array_of_measurements_per_observable]`` with each observable's per-gel AMPLITUDE divided
    out (see :func:`_observable_amplitude`), or ``None`` if the covered observables cannot be put on
    one coordinate.

    THE single place a gain is resolved for the data side, so every consumer of the measurements
    (:func:`sigma_data_std`, :func:`sigma_difference_noise_estimate`) lands in the same coordinate as
    :func:`sigma_effective_value`. Splitting this across call sites is what let the pooled std and
    the difference-based estimate drift into different coordinates.

    An observable with NO amplitude normalises by 1.0 — it is ALREADY on the common coordinate, and
    conflating that with an amplitude that would not reduce to a number (both "no gain value") made
    a sigma shared by a gained and an ungained observable bail out and take the whole gate dark.
    Only a gain that is PRESENT but unresolvable forces ``None``, because that one really would pool
    two coordinates.
    """
    gains = _gain_normalised_gains(problem, pid, rows, fitted_values)
    if gains is None:
        return None
    if OBSERVABLE_ID not in rows.columns:
        return [rows[MEASUREMENT].dropna().astype(float).to_numpy()]
    return [
        sub[MEASUREMENT].dropna().astype(float).to_numpy() / gains.get(str(oid), 1.0)
        for oid, sub in rows.groupby(OBSERVABLE_ID, sort=False)
    ]


def _gain_normalised_gains(problem, pid: str, rows, fitted_values=None):
    """``{observable_id: gain}`` for the observables ``rows`` covers — 1.0 where there is no
    amplitude — or ``None`` if any PRESENT amplitude will not reduce to a number.

    See :func:`_gain_normalised_series` for why "no amplitude" must map to 1.0 rather than share the
    ``None`` that means "unresolvable".
    """
    if rows is None or rows.empty or OBSERVABLE_ID not in rows.columns:
        return {}
    out: dict[str, float] = {}
    for oid, sub in rows.groupby(OBSERVABLE_ID, sort=False):
        resolved = _noise_expr_and_observable(problem, pid, sub)
        amplitude = (
            None if resolved is None else _observable_amplitude(problem, str(oid), resolved[0])
        )
        if amplitude is None:
            out[str(oid)] = 1.0
            continue
        gain = _amplitude_value(problem, amplitude, fitted_values)
        if gain is None:
            return None
        out[str(oid)] = gain
    return out


def sigma_difference_noise_estimate(problem, pid: str, fitted_values=None) -> float | None:
    """A MODEL-FREE estimate of ``pid``'s noise level, from successive differences of adjacent
    measurements (the Rice 1984 difference-based variance estimator):

        sigma^2 ~= mean((y[i+1] - y[i])^2) / 2

    Differences are taken WITHIN each ``(observable, experiment)`` series ordered by time, on the
    coordinate where the noise is additive (see :func:`sigma_noise_coordinate`). A smooth mean
    function nearly cancels between adjacent samples, so what survives is dominated by noise — which
    makes this an independent yardstick that needs NEITHER replicates NOR a ground-truth model. That
    is what the plausibility band lacks: it can only compare a sigma to the marginal data spread,
    which conflates variance the model explains with variance it can't.

    ADVISORY ONLY, and BIASED UPWARD: whatever the mean function does between two samples is counted
    as noise, so sparse or fast-varying series over-estimate (measured across this benchmark: ~1.9x
    the shipped sigma at the median), and a CUMULATIVE series is pathological (its differences are
    the increments — ~24x on oliveira). Returns ``None`` when there is too little data to difference.

    WHY THIS DETECTS INFLATION BUT NOT COLLAPSE (don't re-derive it):

    * The bias means ``estimate >= sigma_true`` in expectation — an UPPER bound on the noise. So
      ``fitted > estimate`` implies ``fitted > sigma_true``: the inflation inference is valid, and
      the bias only makes it conservative.
    * Collapse asks the opposite question ("is ``fitted`` far BELOW ``sigma_true``?"), which needs a
      LOWER bound. Differencing can only ever over-attribute variation to noise, never under-, so it
      cannot supply one: a correctly-fitted sigma also sits below this estimate, by an unknown margin
      (1x-24x here). Signal and artifact are indistinguishable in that direction.
    * More fundamentally, overfitting is an OUT-OF-SAMPLE concept: this estimator reads the raw data
      and never the residuals, so it returns the same number whether or not the model interpolated
      them. No in-sample statistic can certify overfitting — a flexible enough model matches any
      in-sample criterion. Only held-out/cross-validated likelihood can, at the cost of a refit per
      fold (affordable post-run in a judge; not within an agent's ~11-fit budget).
    * Replicates are the exception: two measurements at the SAME design point differ by noise alone,
      so pure error is UNBIASED and supports a two-sided lack-of-fit test. Only 3 of 16 benchmark
      examples have any.

    So absent validation data, a collapsed sigma on a real problem is not detectable here, and BIC's
    ``k*ln(n)`` — itself a cheap surrogate for out-of-sample error — is the practical guard. It
    leaks exactly when the extra flexibility is not counted in ``k`` (added states, non-identifiable
    directions).
    """
    rows = _measurement_rows_for_sigma(problem, pid)
    if rows is None or rows.empty or MEASUREMENT not in rows.columns:
        return None
    coordinate = sigma_noise_coordinate(problem, pid, rows)
    if coordinate is None:
        return None
    import numpy as np

    # Differences are taken on the SAME amplitude-normalised coordinate the fitted sigma is reported
    # in (see _gain_normalised_series). Differencing the RAW measurements while `effective_sigma` is
    # normalised puts `fitted_over_data_noise` out by the gel gain — on fiedler's sigma_pMek that is
    # ~0.003 instead of ~1, i.e. this model-free yardstick would tell a correct model its noise is
    # ~300x too small, and inflating on that advice then trips the (correctly normalised) gate.
    gains = _gain_normalised_gains(problem, pid, rows, fitted_values)
    if gains is None:
        return None
    group_cols = [c for c in (OBSERVABLE_ID, EXPERIMENT_ID) if c in rows.columns]
    grouped = rows.groupby(group_cols, sort=False) if group_cols else [(None, rows)]
    squares: list[float] = []
    for _key, series in grouped:
        if "time" in series.columns:
            series = series.sort_values("time")
        raw = series[MEASUREMENT].dropna().astype(float)
        oid = (
            series[OBSERVABLE_ID].iloc[0]
            if OBSERVABLE_ID in series.columns and len(series)
            else None
        )
        values = _transform_values(raw / gains.get(str(oid), 1.0), coordinate)
        values = np.asarray(values, dtype=float)
        if len(values) >= 3:  # need at least two differences for a usable mean
            squares.extend(np.diff(values) ** 2)
    if len(squares) < 2:
        return None
    estimate = float(np.sqrt(float(np.mean(squares)) / 2.0))
    return estimate if estimate > 0 else None


def sigma_replicate_noise_estimate(problem, pid: str, fitted_values=None) -> float | None:
    """The PURE-ERROR noise estimate for ``pid``: the pooled standard deviation WITHIN design points
    that were measured more than once (same observable, experiment and time).

        sigma^2 = sum_g sum_i (y_gi - mean_g)^2 / sum_g (n_g - 1)

    Two measurements at the SAME design point differ by noise alone, so this needs NEITHER a model
    NOR the smoothness assumption that biases :func:`sigma_difference_noise_estimate` — it is the
    classical UNBIASED pure error. Computed on the same amplitude-normalised, noise-additive
    coordinate as the fitted sigma (see :func:`_gain_normalised_gains` / :func:`sigma_noise_coordinate`),
    so ``fitted_over_data_noise`` against it is at least dimensionally sound, unlike the biased
    difference-based ratio.

    ADVISORY: the RATIO against this floor gates nothing (the #140 arm that did is gone — see
    ``fitting._entry_inflated``); #144's per-design-point test keeps each point's
    ``n_i - 1`` df instead. Still worth REPORTING: the only model-free yardstick here that is not
    biased upward.

    Returns ``None`` when the data cannot supply it: no time column to identify replicates, or fewer
    than ``NOISE_SIGMA_REPLICATE_MIN_DOF`` pooled replicate residuals (most benchmark problems are
    single-measurement time series). The caller then falls back to the difference-based estimate.
    """
    points = replicate_pure_error_points(problem, pid, fitted_values)
    if points is None:
        return None
    # POOLING the per-point sums of squares is what makes this one number: sum ssq_i / sum df_i. The
    # per-point (n_i, s_i, df_i) it pools are exposed by that function for the lack-of-fit arm, which
    # asks the other question — is the PREDICTION further from a point than that point's own scatter
    # allows (see residual_tests.replicate_lack_of_fit).
    #
    # This is the SAME estimator as before the split, but not bit-for-bit the same number: fsum adds
    # exactly where the old inline loop accumulated sequentially, which moves 4 of the 6 floors the
    # benchmark suite has by 1-2 ULP (blasi 0.2579081404095943 -> 0.25790814040959437). Measured
    # across all 16 examples, and kept: the new arithmetic is the more accurate one, and a 1e-16 shift
    # cannot move a ratio the inflation gate reads at 1.5 (the run that motivated this arm sat 0.0024
    # away). Anything comparing recorded floors across that boundary should compare with a tolerance.
    ssq = math.fsum(p["ssq"] for p in points)
    dof = sum(p["df"] for p in points)
    if dof < NOISE_SIGMA_REPLICATE_MIN_DOF:
        return None
    estimate = math.sqrt(ssq / dof)
    return estimate if estimate > 0 else None


def replicate_pure_error_points(problem, pid: str, fitted_values=None, simulations=None):
    """The PER-DESIGN-POINT pure error behind :func:`sigma_replicate_noise_estimate`'s single pooled
    number — and, when ``simulations`` is given, the model's prediction at each of those points on
    the SAME coordinate.

    One entry per design point measured MORE THAN ONCE (rows sharing observable, experiment and
    time), each ``{observable_id, experiment_id, time, n, df, ybar, s, ssq, prediction}`` on the
    noise-additive, amplitude-normalised coordinate (see :func:`sigma_noise_coordinate` /
    :func:`_gain_normalised_gains`) — so ``(ybar - prediction) / (s / sqrt(n))`` is a deviation in
    units of that point's OWN replicate scatter and never touches the fitted sigma at all. Points
    measured once carry no scatter and are omitted; a problem without repeats yields ``[]`` (only 3
    of the 16 benchmark examples have any — see BENCHMARK_NOTES).

    ``simulations`` is the session's per-measurement simulation frame renamed to the PEtab column
    names (``data_access._replicate_lof`` does that); it carries one row per MEASUREMENT, so
    the replicate grouping is identical either way. ``prediction`` is ``None`` when no frame is
    passed, or when the simulated value is not CONSTANT across a point's replicates — the one thing
    that would mean the group is not a single design point after all (a ``time`` the records could
    not represent, say, since they store only finite floats).

    ``None`` — not ``[]`` — when the rows, the coordinate or the gains cannot be resolved: that is
    "unknown", not "no replicates", and every caller must fail open on it.
    """
    rows = _measurement_rows_for_sigma(problem, pid)
    if rows is None or rows.empty or MEASUREMENT not in rows.columns or "time" not in rows.columns:
        return None
    coordinate = sigma_noise_coordinate(problem, pid, rows)
    if coordinate is None:
        return None
    gains = _gain_normalised_gains(problem, pid, rows, fitted_values)
    if gains is None:
        return None
    if simulations is not None:
        # Restrict the simulation records to the DESIGN POINTS `pid` actually covers: the noise
        # scoping is per PARAMETER, and mixing two sigmas' points into one family would both cross
        # coordinates and inflate the multiple-testing correction. Scoping by observable ID alone was
        # not enough — `_measurement_rows_for_sigma` also scopes by the per-ROW `noiseParameters`
        # override, so two sigmas governing different rows of ONE observable each got all of that
        # observable's points: every count doubled, the reported points came back duplicated, and
        # each arm applied its own gain and coordinate to rows the other sigma governs.
        if (
            OBSERVABLE_ID not in getattr(simulations, "columns", [])
            or MEASUREMENT not in simulations.columns
            or "time" not in simulations.columns
        ):
            return None
        covered = set(_design_point_keys(rows))
        rows = simulations[[key in covered for key in _design_point_keys(simulations)]]
        if rows.empty:
            return []

    import numpy as np

    from . import residual_tests as rt

    out: list[dict[str, Any]] = []
    # Group by the full design point INCLUDING time: rows sharing (observable, experiment, time) are
    # replicates of one true value, so their spread is pure noise. A single-measurement design point
    # (n_g = 1) contributes nothing — which is why a time series without repeats yields no points.
    # dropna only for the MEASUREMENT table. The simulation records store only FINITE floats, so
    # blasi's steady-state `time = inf` arrives as None and every one of its 15 design points would
    # be dropped — hence dropna=False there. On the measurement table a blank
    # observable/condition/time cell is genuinely missing data, and pooling every such row into ONE
    # NaN-keyed group makes its BETWEEN-point spread read as within-point noise: that silently moves
    # the pooled floor `sigma_replicate_noise_estimate` hands the 1.5 inflation gate (a two-row toy
    # case moved it 0.0129 -> 22.36), so those rows stay dropped, exactly as before this split. (An
    # experiment-less row is NOT such a row: its `experimentId` is "" by construction, see
    # :mod:`petab_mcp.problem`, so it groups like any other.) THIS
    # is the rule for what a design point IS; `_design_point_keys` below normalises a missing time only
    # to MATCH the two frames, and never makes a NaN-keyed row into a design point.
    group_cols = [c for c in (OBSERVABLE_ID, EXPERIMENT_ID, "time") if c in rows.columns]
    for _key, series in rows.groupby(group_cols, sort=False, dropna=simulations is None):
        oid = (
            series[OBSERVABLE_ID].iloc[0]
            if OBSERVABLE_ID in series.columns and len(series)
            else None
        )
        gain = gains.get(str(oid), 1.0)
        v = _transform_values(series[MEASUREMENT].dropna().astype(float) / gain, coordinate)
        v = np.asarray(v, dtype=float)
        v = v[np.isfinite(v)]
        pure = rt.replicate_pure_error(v.tolist())
        if pure is None:
            continue
        out.append(
            {
                "observable_id": None if oid is None else str(oid),
                "experiment_id": _first_value(series, EXPERIMENT_ID),
                "time": _first_value(series, "time"),
                "n": pure["n"],
                "df": pure["df"],
                "ssq": pure["ssq"],
                "ybar": pure["mean"],
                "s": pure["sd"],
                "prediction": _constant_prediction(series, gain, coordinate),
            }
        )
    return out


def _design_point_keys(frame) -> list[tuple]:
    """One hashable ``(observable, experiment, time)`` key per row of ``frame`` — used ONLY to match
    rows of the MEASUREMENT table against rows of the SIMULATION records, never to decide what counts
    as a design point (the ``groupby`` above does that, and it still DROPS NaN keys on the measurement
    table; see its comment).

    For matching, a non-finite or missing time collapses to ``None`` on both sides, because the
    simulation records store only finite floats (blasi's steady-state ``time = inf`` arrives as None)
    and because a NaN would never compare equal to itself anyway. A missing column contributes
    ``None``, so a frame without an experiment column still matches one whose experiment is blank. A
    measurement row with a blank time therefore contributes a key here and is still excluded from the
    pooled floor — scoping wider than the grouping is the fail-open direction: it can only admit
    simulation rows that the grouping then declines to make a design point out of.
    """

    def _label(value):
        # "" is how an experiment-less row spells its experiment (see petab_mcp.problem); it must
        # meet a record that carries None for the same thing.
        return None if value is None or value != value or value == "" else str(value)

    def _time(value):
        try:
            t = float(value)
        except (TypeError, ValueError):
            return None
        return t if math.isfinite(t) else None

    n = len(frame)
    columns = getattr(frame, "columns", [])
    obs = frame[OBSERVABLE_ID] if OBSERVABLE_ID in columns else [None] * n
    cond = frame[EXPERIMENT_ID] if EXPERIMENT_ID in columns else [None] * n
    time = frame["time"] if "time" in columns else [None] * n
    return [(_label(o), _label(c), _time(t)) for o, c, t in zip(obs, cond, time)]


def _first_value(series, column: str):
    """The first value of ``column`` in a design point's rows (they share it by construction), or
    ``None`` when the column is absent."""
    if column not in getattr(series, "columns", []) or not len(series):
        return None
    value = series[column].iloc[0]
    return None if value != value else value  # NaN -> None


def _constant_prediction(series, gain: float, coordinate: str):
    """The model's prediction for one design point, on the noise coordinate, or ``None``.

    Amplitude-normalised by the SAME gain as the measurements (a per-gel factor multiplies signal
    and noise alike, so it has to cancel from both sides), and returned only when the simulated
    value is constant across the point's replicates — replicates of one design point share one
    prediction by construction, so a spread means the grouping is not a design point and the honest
    answer is "unknown".
    """
    if "simulation" not in getattr(series, "columns", []):
        return None
    import numpy as np

    v = _transform_values(series["simulation"].dropna().astype(float) / gain, coordinate)
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    if not len(v) or not bool(np.allclose(v, v[0], rtol=1e-9, atol=0.0)):
        return None
    return float(v[0])


#: Symbolic noiseFormulas by the observable table's TEXT (see :func:`_symbolic_noise_formulas`).
#: Small and bounded: one entry per distinct table, and only the most recent few are kept.
_NOISE_FORMULA_CACHE: dict[tuple, dict] = {}
_NOISE_FORMULA_CACHE_MAX = 8


def _symbolic_noise_formulas(odf) -> dict:
    """``{observable_id: sympy_expr}`` for the whole observable table, memoised on its CONTENT.

    ``petab.get_symbolic_noise_formulas`` re-parses EVERY noiseFormula in the table through the antlr
    PEtab-math grammar, and :func:`_noise_expr_and_observable` needs it once per OBSERVABLE — so a
    family covering an n-observable table paid n*n parses on every gain resolution, which is why
    :func:`_gain_normalised_gains` dominated both the replicate lack-of-fit arm and the pooled noise
    floor. Measured per sigma on this machine, gain resolution / whole
    :func:`replicate_pure_error_points` call: blasi (16 observables, 256 parses) 35.8 -> 5.8 ms of
    40.8 -> 11.2 ms; schwen/std (22 observables) 54.3 -> 10.9 ms of 88.6 -> 42.4 ms; armistead 1.4 ->
    0.75 ms of 4.7 -> 3.0 ms. Both consumers run once per sigma per fit, so a two-sigma problem paid
    it four times.

    Keyed on the ids and formula TEXT, never on the frame's identity: the agent re-authors the
    observable table mid-session, and an ``id()``-keyed cache would be free to serve a stale
    expression for a rewritten formula. Building the key costs one tuple of strings; a miss costs what
    the call always cost. sympy expressions are immutable and every caller ``.subs()`` into a new one,
    so sharing them is safe.

    One ``.get()``, and the result returned from a local: another session's thread may clear the
    cache between a membership test and a read, so that pair can raise ``KeyError`` even on a hit.
    """
    from petab.v2.calculate import get_symbolic_noise_formulas

    try:
        key = (tuple(str(i) for i in odf.index), tuple(str(v) for v in odf[NOISE_FORMULA]))
    except Exception:  # noqa: BLE001 - unkeyable table: just don't cache it
        return get_symbolic_noise_formulas(odf)
    formulas = _NOISE_FORMULA_CACHE.get(key)
    if formulas is None:
        formulas = get_symbolic_noise_formulas(odf)
        if len(_NOISE_FORMULA_CACHE) >= _NOISE_FORMULA_CACHE_MAX:
            _NOISE_FORMULA_CACHE.clear()
        _NOISE_FORMULA_CACHE[key] = formulas
    return formulas


def _noise_expr_and_observable(problem, pid: str, rows):
    """``(sympy_expr, observable_symbol, observable_id)`` for the noise of the rows ``pid`` covers,
    with the measurement-table ``noiseParameters`` overrides already substituted, or ``None``.

    Overrides are resolved the way petab does it — the observable's declared noise placeholders are
    filled in order, a token naming a PARAMETER becomes that parameter's symbol and a numeric one
    the number — so the fitted sigma survives as a symbol we can substitute a value for. (Getting
    this wrong is what made an earlier attempt evaluate the formula without reference to the sigma
    at all.) Requires the covered rows to share one observable and one override list, which is the
    normal case; ``None`` otherwise so callers fail open rather than average incomparable formulas.
    """
    odf = getattr(problem, "observable_df", None)
    if odf is None or NOISE_FORMULA not in odf.columns or OBSERVABLE_ID not in rows.columns:
        return None
    observable_ids = rows[OBSERVABLE_ID].dropna().unique()
    if len(observable_ids) != 1:
        return None
    observable_id = str(observable_ids[0])
    override_cells = (
        set(rows[NOISE_PARAMETERS].dropna().astype(str))
        if NOISE_PARAMETERS in rows.columns
        else set()
    )
    # Differing override lists across rows (fiedler's per-gel scale factors) mean the covered rows
    # do not share one sd expression; the caller must fail open rather than pick one arbitrarily.
    if len(override_cells) > 1:
        return None
    try:
        import sympy as sp

        expr = _symbolic_noise_formulas(odf).get(observable_id)
        if expr is None:
            return None
        values = split_overrides(next(iter(override_cells)) if override_cells else None)
        overrides = {
            sp.Symbol(name, real=True): (
                sp.Symbol(str(value), real=True) if isinstance(value, str) else value
            )
            for name, value in zip(placeholders(odf, observable_id, "noise"), values)
        }
        expr = expr.subs(overrides)
        # The formula may name the observable id OR the model quantity the observable reports
        # (armistead: observable `Cer_obs`, noiseFormula `noiseParameter1_Cer_obs*Cer`). Resolve
        # the level symbol through observableFormula rather than guessing, so the derivative below
        # is taken with respect to the symbol actually present.
        candidates = [sp.Symbol(observable_id, real=True)]
        if OBSERVABLE_FORMULA in odf.columns:
            try:
                candidates += [
                    sp.Symbol(name, real=True)
                    for name in sorted(formula_symbols(odf.loc[observable_id, OBSERVABLE_FORMULA]))
                ]
            except KeyError:
                pass
        free = expr.free_symbols
        level = next((c for c in candidates if c in free), candidates[0])
        return expr, level, observable_id
    except Exception:  # noqa: BLE001 - advisory scale only
        return None


def _sympify_real(formula: str):
    """``sympify(formula)`` with every free SYMBOL declared real, or ``None`` if it will not parse.

    petab builds its formulas with ``real=True`` assumptions and ``Symbol(x) != Symbol(x, real=True)``,
    so an assumption-free parse silently fails to cancel against the noise expression — which is why
    a gain the observable formula names DIRECTLY was never detected.

    Binding names via ``locals`` must skip anything sympy resolves as a CALLABLE:
    :func:`formula_symbols` is a bare identifier regex, so ``scale * exp(-k*t)`` yields ``exp`` too,
    and binding ``exp -> Symbol('exp')`` makes the parse raise ``'Symbol' object is not callable``.
    Swallowed by the caller's ``except``, that turned every observable whose formula uses ``exp`` /
    ``log`` / ``sqrt`` / ``piecewise`` into "no amplitude", losing gain normalisation entirely and
    judging its sigma in the un-normalised coordinate — the same wrong-coordinate verdict the
    real=True change was made to fix. No shipped example has a call in an observableFormula today,
    so nothing in the suite covers it; an agent-authored model will.
    """
    try:
        import sympy as sp

        # Skip by CALL SYNTAX, not by sympy's namespace. `callable(getattr(sp, name, None))`
        # catches `exp`/`log`/`sqrt` but misses `log10`, which sympy has no attribute for and
        # PEtab uses freely — binding it as a Symbol raises exactly the same way. An identifier
        # immediately followed by `(` is a call whatever sympy knows about it; sympify then
        # builds an undefined Function for it, which is fine here since only the multiplicative
        # structure matters.
        called = set(re.findall(r"(?<![\w.])([A-Za-z_]\w*)\s*\(", formula))
        names = {
            n
            for n in formula_symbols(formula)
            if n not in called and not callable(getattr(sp, n, None))
        }
        return sp.sympify(formula, locals={n: sp.Symbol(n, real=True) for n in names})
    except Exception:  # noqa: BLE001 - advisory scale only; caller fails open
        return None


def _observable_amplitude(problem, observable_id: str, noise_expr):
    """The multiplicative AMPLITUDE of ``observable_id`` — a factor appearing in BOTH the observable
    formula and the noise formula (fiedler's per-gel ``s_pErk_20140430_gel1``) — as a sympy
    expression, or ``None`` when the observable has no such dual-role factor (the usual case).

    Such a factor is a nuisance gain of the ASSAY, not of the noise: the gel multiplies signal and
    measurement error alike, so both the sd (``a*sigma``) and the data carry it. It therefore cancels
    from the only quantity the plausibility gate uses — the dimensionless sd/spread ratio — and the
    honest way to compute that ratio is in the coordinate where it has already cancelled. Dividing it
    out is what lets a sigma SHARED across gels be summarised at all: without it each gel implies its
    own effective sd, they disagree, and :func:`_per_observable_agreement` fails open (fiedler's
    ``noise_sigma_effective`` was ``{}``).

    Requires exactly ONE such symbol, dividing BOTH formulas multiplicatively; anything else returns
    ``None`` so callers fail open instead of normalising by a factor that isn't a pure gain.
    """
    odf = getattr(problem, "observable_df", None)
    if odf is None or OBSERVABLE_FORMULA not in odf.columns:
        return None
    try:
        import sympy as sp

        scaling = observable_scaling_ids(problem)
        candidates = sorted((s for s in noise_expr.free_symbols if str(s) in scaling), key=str)
        if not candidates:
            return None
        # EVERY qualifying candidate, not just a lone one. A noise formula may name more than one
        # scaling symbol and only some of them be gains: isensee's flattened microscopy readout is
        # `noiseParameter1_<obs> * s_pRII_global * rho`, whose override resolves noiseParameter1 to
        # a PER-EXPERIMENT `s_pRII_JI09_<run>`. That per-experiment factor is a true assay gain --
        # it multiplies the whole observable -- while `s_pRII_global` sits INSIDE the formula
        # (`b_pRII_global + s_pRII_global * (...)`) and is not a multiplicative factor at all.
        # Requiring exactly one candidate bailed on that observable, so the gain was never divided
        # out, the three experiments implied three different effective sds,
        # `_per_observable_agreement` failed open, and `effective_sigma` came back None. That in
        # turn made `_reference_sigma_flag` unpairable and dropped the judge onto the data-driven
        # band -- which isensee's own REFERENCE sigma exceeds (0.165 against a 0.75*data_std cap of
        # 0.092), so every candidate was flagged `inflated` on a threshold the truth fails.
        # The per-candidate tests below are exactly the ones this function already applied to a
        # lone candidate, so nothing that was not a pure dual-role gain becomes one.
        # real=True on every symbol, matching how petab builds its formulas. A bare `sympify`
        # yields assumption-free symbols, and `Symbol(x) != Symbol(x, real=True)`, so
        # `simplify(obs_expr / candidate)` would not cancel and `_is_pure_gain` would reject a
        # gain the formula names DIRECTLY — isensee's Western readout (`s_pRII_Western * (...)`
        # against `s_pRII_Western * rho_pRII_Western`) is exactly that case.
        obs_expr = _sympify_real(str(odf.loc[observable_id, OBSERVABLE_FORMULA]))
        if obs_expr is None:
            return None
        obs_expr = _resolve_observable_parameters(problem, observable_id, obs_expr)
        gains = [c for c in candidates if _is_pure_gain(c, noise_expr, obs_expr)]
        if not gains:
            return None
        amplitude = gains[0]
        for extra in gains[1:]:
            amplitude = amplitude * extra
        amplitude = sp.simplify(amplitude)
        # The PRODUCT must still divide both sides exactly — each factor does individually, but
        # verify the combination rather than assuming it composes.
        for side in (noise_expr, obs_expr):
            if set(amplitude.free_symbols) & set(sp.simplify(side / amplitude).free_symbols):
                return None
        return amplitude
    except Exception:  # noqa: BLE001 - advisory scale only
        return None


def _resolve_observable_parameters(problem, observable_id: str, obs_expr):
    """``obs_expr`` with the measurement table's ``observableParameters`` overrides substituted.

    Resolved ONCE, before any candidate is tested, because the substitution depends on the
    observable rather than on the candidate. A gain reaches the observable formula through an
    observable placeholder while the NOISE side already carries the substituted value, so without
    this the same quantity has two different names and no candidate can match.

    Unchanged when there is no measurement table, no override column, or the covered rows disagree
    about the override — the caller then tests against the raw formula, which is the pre-existing
    behaviour for a gain the formula names directly (``materialize`` FLATTENS an example whose
    overrides vary within an observable, writing the value into the formula and dropping the
    placeholder; isensee's Western readout is exactly that).
    """
    try:
        import sympy as sp

        mdf = getattr(problem, "measurement_df", None)
        if mdf is None or OBSERVABLE_PARAMETERS not in mdf.columns:
            return obs_expr
        cells = set(
            mdf.loc[mdf[OBSERVABLE_ID] == observable_id, OBSERVABLE_PARAMETERS].dropna().astype(str)
        )
        if len(cells) != 1:
            return obs_expr
        # Symbols are matched BY NAME: petab builds its formulas with `real=True` assumptions while
        # a plain `sympify` does not, and `Symbol(x) != Symbol(x, real=True)`, so identity matching
        # silently no-ops the substitution and reports "no amplitude" for every observable with one.
        by_name = {str(s): s for s in obs_expr.free_symbols}
        names = placeholders(problem.observable_df, observable_id, "observable")
        return obs_expr.subs(
            {
                by_name[key]: (sp.Symbol(str(v), real=True) if isinstance(v, str) else v)
                for key, v in zip(names, split_overrides(next(iter(cells))))
                if key in by_name
            }
        )
    except Exception:  # noqa: BLE001 - advisory scale only
        return obs_expr


def _is_pure_gain(candidate, noise_expr, obs_expr) -> bool:
    """Whether ``candidate`` is an assay gain: a factor dividing BOTH the noise sd and the
    observable formula exactly.

    "Exactly" is the load-bearing word — a pure gain ``a`` satisfies ``sd/a`` free of ``a``, which
    ``sqrt(a**2 + s**2)`` would not, and likewise for the observable. A scaling symbol that appears
    only additively inside the formula (isensee's ``b_pRII_global + s_pRII_global * (...)``) fails
    the observable half and is correctly declined.
    """
    try:
        import sympy as sp

        name = str(candidate)
        if name in {str(s) for s in sp.simplify(noise_expr / candidate).free_symbols}:
            return False
        if name not in {str(s) for s in obs_expr.free_symbols}:
            return False
        quotient = sp.simplify(obs_expr / candidate)
        if name in {str(s) for s in quotient.free_symbols}:
            return False
        # A gain MULTIPLIES the observable; it is not the whole observable. When the quotient is a
        # bare number the candidate IS the level symbol -- `observableFormula = x` against
        # `noiseFormula = noiseParameter1_o*x` is proportional noise, where `expr / y` already
        # removes it and there is no separate assay gain to normalise the DATA by. Accepting it
        # returns a SPECIES as the amplitude, which `_amplitude_value` cannot reduce to a number,
        # so `sigma_data_std` goes None and the sigma becomes unassessable. Before the real=True
        # fix above this was masked by the symbol-assumption mismatch rather than excluded.
        return bool(quotient.free_symbols)
    except Exception:  # noqa: BLE001 - advisory scale only
        return False


def _amplitude_value(problem, amplitude, fitted_values=None) -> float | None:
    """``amplitude`` as a number, preferring THIS fit's value over the table's nominal (the agent's
    own gel gains are what its measurements were scaled by). ``None`` if it won't reduce."""
    try:
        import sympy as sp

        subs = {}
        pdf = getattr(problem, "parameter_df", None)
        if pdf is not None and NOMINAL_VALUE in pdf.columns:
            for name in pdf.index:
                try:
                    subs[sp.Symbol(str(name), real=True)] = float(pdf.loc[name, NOMINAL_VALUE])
                except (TypeError, ValueError):
                    continue
        for name, value in (fitted_values or {}).items():
            try:
                subs[sp.Symbol(str(name), real=True)] = float(value)
            except (TypeError, ValueError):
                continue
        value = sp.simplify(amplitude.subs(subs))
        if value.free_symbols:
            return None
        result = float(value)
        return result if math.isfinite(result) and result != 0 else None
    except Exception:  # noqa: BLE001
        return None


def _per_observable_agreement(problem, pid, rows, fn):
    """Apply ``fn(sub_rows)`` to each covered observable's rows and return the common result, or
    ``None`` if they disagree or any is ``None``.

    A sigma shared across many observables (blasi: one ``sigma`` for 16 acetylation states) is the
    normal case, and each observable declares its OWN noise placeholder, so the expression must be
    resolved per observable. Requiring agreement keeps the reported verdict
    meaningful: if one observable's noise is proportional and another's additive, no single scalar
    describes the sigma and we fail open.
    """
    if OBSERVABLE_ID not in rows.columns:
        return fn(rows)
    results = []
    for _oid, sub in rows.groupby(OBSERVABLE_ID, sort=False):
        result = fn(sub)
        if result is None:
            return None
        results.append(result)
    if not results:
        return None
    first = results[0]
    for other in results[1:]:
        if isinstance(first, float) and isinstance(other, float):
            if not math.isclose(first, other, rel_tol=1e-9):
                return None
        elif other != first:
            return None
    return first


def sigma_noise_form(problem, pid: str, rows=None) -> str | None:
    """How the noise depends on the observable, decided from the SYMBOLIC formula, not from data:

    * ``"additive"`` — the sd does not involve the observable, so it IS an sd on the observable's
      transformation scale (``d(expr)/d(y) == 0``).
    * ``"proportional"`` — the sd is ``cv * y``, i.e. ``expr / y`` is free of ``y``; the parameter
      is a dimensionless coefficient of variation.
    * ``None`` — anything else (a mixed ``sqrt(a**2 + (b*y)**2)``, an unresolvable formula), where
      no single scalar summarises the noise and the caller must fail open.

    Reading the expression is the whole point: an earlier attempt inferred this from the DISPERSION
    of evaluated values, which made the verdict depend on where the measurement magnitudes happened
    to sit — the same model and sigma scored collapsed / ok / inflated on different data ranges.
    """
    if rows is None:
        rows = _measurement_rows_for_sigma(problem, pid)
    if rows is None or rows.empty:
        return None

    def _form(sub_rows):
        resolved = _noise_expr_and_observable(problem, pid, sub_rows)
        if resolved is None:
            return None
        expr, y, _obs = resolved
        try:
            import sympy as sp

            if sp.simplify(sp.diff(expr, y)) == 0:
                return "additive"
            if y not in sp.simplify(expr / y).free_symbols:
                return "proportional"
            return None
        except Exception:  # noqa: BLE001
            return None

    return _per_observable_agreement(problem, pid, rows, _form)


def sigma_scale_coupled(problem, pid: str) -> bool:
    """True when ``pid`` shares its noiseFormula with an ESTIMATED co-parameter, so the sigma's own
    numeric value is not comparable ACROSS MODELS -- only the combination is.

    Deliberately NOT restricted to a product, and does not inspect the formula's FORM: any estimated
    co-parameter (``gain * sigma``, ``sigma + offset``) leaves the raw sigma under-determined on its
    own, and every such case is one where a cross-model raw-sigma ratio is the wrong statistic. A
    FIXED co-parameter is different and is not flagged: it is a declared constant of the problem,
    identical on both sides, so it cancels from the ratio.

    fiedler is the live case: ``noiseFormula = noiseParameter1 * noiseParameter2`` resolves to
    ``s_pMek_<gel> * sigma_pMek``, and the gels' ``s_pMek_*`` are estimated over 77.7 … 999.99. Only
    the product is identified, so a model whose internal pMek units differ fits a proportionally
    different sigma for the SAME noise. Note what IS comparable: the product is the noise sd in
    measurement units (404.2 * 5.8e-4 = 0.235 against a pMek data std of 1.35 -- plausible), so the
    reference basis is recoverable in principle; see
    :func:`petab_harness.runner.scoring._reference_sigma_flag` for why it is declined today and what
    recording it would take. :func:`sigma_effective_value` already documents the half of this that
    was known -- "two models that split the same fit differently between gain and sigma report
    different values here" -- which is harmless for the data-driven ratio (the gain cancels from
    sd/spread) but not for a comparison against another model's fitted sigma.

    Unflagged, this made the judge's reference-sigma gate fire on 19 of 20 fiedler runs in the Aug
    8-10 sweep, silently failing 17 that had BEATEN the BIC bar (best dBIC -28.8, while the one
    scored pass was +0.50). The candidates' own data-driven numbers said their noise was fine
    (fraction_of_data_std ~0.31, fitted_over_data_noise ~0.56); only the cross-model raw-sigma ratio
    called them 179x inflated. Of the 21 examples here, fiedler is the ONLY one with this shape.
    """
    rows = _measurement_rows_for_sigma(problem, pid)
    if rows is None or rows.empty or NOISE_PARAMETERS not in rows.columns:
        return False
    for cell in rows[NOISE_PARAMETERS].dropna().astype(str).unique():
        parts = [p.strip() for p in cell.split(";")]
        if len(parts) < 2 or pid not in parts:
            continue
        for other in parts:
            # A co-factor that is itself ESTIMATED is what breaks comparability. A FIXED co-factor
            # is a declared constant of the problem: identical on both sides, so the ratio survives.
            if other != pid and other and is_parameter_estimated(problem, other):
                return True
    return False


def sigma_effective_value(
    problem, pid: str, fitted: float, fitted_values: "dict[str, float] | None" = None
) -> float | None:
    """The sd implied by ``pid = fitted``, expressed on the scale :func:`sigma_data_std` measures.

    Evaluated from the noiseFormula, so a COMPOSITE sd (fiedler's
    ``noiseParameter1*noiseParameter2`` — a per-gel scale times the sigma, where neither factor is
    an sd alone) comes out right. For ``proportional`` noise the value returned is the coefficient
    of variation, which is the sd in log space, matching the coordinate
    :func:`sigma_noise_coordinate` reports for that form. ``None`` when the form is not one of the
    two summarisable cases, or the expression will not reduce to a number.

    A per-gel AMPLITUDE is divided out (see :func:`_observable_amplitude`), so what comes back is
    the sd in the amplitude-normalised coordinate — which is what makes it a SINGLE number when one
    sigma is shared by gels of differing gain, and what makes it comparable to
    :func:`sigma_data_std`, measured in the same coordinate. Note this is a deliberate choice of
    coordinate, not "the sd in measurement units": the gain cancels from the sd/spread RATIO the
    gate uses, so the ratio is unaffected, but two models that split the same fit differently
    between gain and sigma report different values here. For fiedler the amplitude is an exact
    divisor of the formula, so the normalised value coincides with the raw sigma.
    """
    rows = _measurement_rows_for_sigma(problem, pid)
    if rows is None or rows.empty:
        return None
    form = sigma_noise_form(problem, pid, rows)
    if form is None:
        return None

    def _value(sub_rows):
        resolved = _noise_expr_and_observable(problem, pid, sub_rows)
        if resolved is None:
            return None
        expr, y, _obs = resolved
        try:
            import sympy as sp

            # Substitute every parameter value we know: the fitted sigma, its co-parameters from this
            # fit, then the table's nominals for anything unfitted.
            subs = {}
            pdf = getattr(problem, "parameter_df", None)
            if pdf is not None and NOMINAL_VALUE in pdf.columns:
                for name in pdf.index:
                    try:
                        subs[sp.Symbol(str(name), real=True)] = float(pdf.loc[name, NOMINAL_VALUE])
                    except (TypeError, ValueError):
                        continue
            for name, value in {**(fitted_values or {}), pid: fitted}.items():
                try:
                    subs[sp.Symbol(str(name), real=True)] = float(value)
                except (TypeError, ValueError):
                    continue
            # For proportional noise divide out the observable FIRST, so the result is the cv and no
            # value of y is needed at all; additive noise is already independent of y.
            target = sp.simplify(expr / y) if form == "proportional" else expr
            # Report the sd in the AMPLITUDE-NORMALISED coordinate sigma_data_std measures in, so a
            # per-gel gain cancels (see _observable_amplitude). Divide only if the amplitude SURVIVES
            # into `target`: for proportional noise, dividing by `y` above may already have removed it
            # (when `y` is the observable, which carries the gain) or may not (when `y` resolved to the
            # underlying species, which does not), and keying off the noise FORM instead of the actual
            # expression got that wrong in both directions — it left the gain in for a proportional
            # formula naming the species, so a correct fit read `inflated` by exactly the gain.
            amplitude = _observable_amplitude(problem, _obs, expr)
            # SUBSET, not membership and not intersection.
            #   * membership (`amplitude in target.free_symbols`) is wrong because a multi-gain
            #     amplitude is a sympy `Mul`, never a member of a set of Symbols — it silently
            #     skipped the division for the very case the multi-candidate branch serves, so the
            #     gain left the data side (sigma_data_std) but not the sigma side and a correct fit
            #     read `inflated` by exactly the gain;
            #   * intersection is wrong the other way: with amplitude `a*b` and a `target` holding
            #     only `b` (for proportional noise `y` may resolve to one of the gain symbols and
            #     divide it out already), a partial overlap still divides and leaves a spurious
            #     `1/a`, which then reduces to a NUMBER under the substitutions below — a silently
            #     wrong effective sigma feeding the plausibility gate, rather than failing open.
            # Requiring every factor to be present divides only when the whole gain is there, and
            # reduces to the old behaviour for a bare Symbol.
            if amplitude is not None and amplitude.free_symbols <= target.free_symbols:
                target = sp.simplify(target / amplitude)
            value = sp.simplify(target.subs(subs))
            if value.free_symbols:  # an unresolved symbol -> we do not guess, we fail open
                return None
            result = float(value)
            return result if math.isfinite(result) and result > 0 else None
        except Exception:  # noqa: BLE001 - advisory scale only
            return None

    return _per_observable_agreement(problem, pid, rows, _value)


def sigma_upper_bound_cap(
    problem,
    pid: str,
    max_fraction: float = NOISE_SIGMA_MAX_FRACTION_OF_DATA_STD,
    fitted_values=None,
    data_std: float | None = None,
) -> float | None:
    """Data-driven upper-bound cap for noise-sigma parameter ``pid``: ``max_fraction`` times
    :func:`sigma_data_std`. ``None`` if that std can't be computed — callers must not
    tighten/reject on a ``None`` cap.

    A sigma free to explain a large fraction of the data's own variability isn't measurement
    noise anymore, it's covering for the model: an optimizer can make almost ANY mechanism
    "pass" a chi2 goodness-of-fit check by simply inflating sigma until residuals fit inside
    it. See ``NOISE_SIGMA_MAX_FRACTION_OF_DATA_STD``.
    """
    # `data_std` lets a caller that already computed the std reuse it — the amplitude resolution
    # behind it costs real sympy work (see _observable_amplitude), and noise_sigma_report would
    # otherwise pay for it twice per sigma. The cap FORMULA stays here, so it cannot drift.
    std = sigma_data_std(problem, pid, fitted_values) if data_std is None else data_std
    return max_fraction * std if std is not None else None


def is_parameter_estimated(problem, pid: str) -> bool:
    """Whether ``pid``'s current ``estimate`` flag is truthy (robust to ``NaN``/missing —
    treated as not-estimated, same idiom as ``set_model``'s prior-estimated snapshot)."""
    df = problem.parameter_df
    if pid not in df.index or ESTIMATE not in df.columns:
        return False
    try:
        return as_bool(df.loc[pid, ESTIMATE])
    except (TypeError, ValueError):
        return False


# -- initial-condition report ----------------------------------------------
def initial_conditions_report(problem, added: set[str] | None = None) -> list[dict[str, Any]]:
    """Per-species account of how each t=0 initial condition is resolved *after* reconcile.

    Antimony can only declare a species' t=0 *value* — it has no way to say "estimate this
    initial condition". Whether that declared value actually governs the simulated t=0 state is
    decided by the reconcile/parameter/condition tables, not the model text, and the outcome is
    otherwise invisible: a declared t=0 is silently OVERRIDDEN when its ``<species>_init``
    parameter is estimated (the fit then sets t=0, the declared value is only the start) or when
    the species is driven per-condition by a condition target. Surfacing this after every
    model op is what lets the caller see immediately whether a t=0 "fix" took — instead of
    re-authoring the model under a series of different IC naming schemes that all fit identically
    because none of them ever changed the estimate status (see issue #57). ICs are ESTIMATED by
    default; an IC of exactly 1 (a unit normalization) is held FIXED, and an IC of exactly 0 (an
    empty pool, including an unwritten initial) is not parametrised at all. Decided by value in
    reconcile, not switchable from the model text.

    For each species returns ``{species, ic_parameter, declared_t0, estimated, set_by_condition,
    t0_source, declared_t0_used, newly_added, note}``:

    * ``declared_t0`` — the t=0 value declared in the Antimony (the IC parameter's value);
    * ``estimated`` — whether the IC parameter is a FREE (fit) parameter or FIXED;
    * ``set_by_condition`` — whether the species (or its IC parameter) is a condition target,
      i.e. set per condition, which overrides the Antimony t=0;
    * ``t0_source`` — ``"fixed"`` | ``"estimated"`` | ``"condition_table"``;
    * ``declared_t0_used`` — the bottom line: does the declared t=0 govern the simulated t=0 state
      (``True`` only for a plain FIXED IC), or is it overridden;
    * ``newly_added`` — whether this op introduced the IC parameter (from the reconcile diff).
    """
    sm = problem.model.sbml_model
    cond_targets = condition_target_symbols(problem)
    added = added or set()
    out: list[dict[str, Any]] = []
    for i in range(sm.getNumSpecies()):
        sp = sm.getSpecies(i)
        sid = sp.getId()
        ic = _ic_symbol_for(sm, sid)
        declared: float | None = None
        if ic is not None and sm.getParameter(ic) is not None:
            declared = float(sm.getParameter(ic).getValue())
        elif sp.isSetInitialConcentration():
            declared = float(sp.getInitialConcentration())
        elif sp.isSetInitialAmount():
            declared = float(sp.getInitialAmount())
        elif ic is None:
            # No IC parameter AND no declared initial: an unset species is an empty pool in SBML,
            # and `ensure_parametric_initial_conditions` deliberately mints nothing for it. Report
            # the value it actually simulates from rather than None, which would otherwise fall
            # through the FIXED branch below and be described as "shipped as a fixed parameter".
            declared = 0.0
        estimated = ic is not None and is_parameter_estimated(problem, ic)
        set_by_condition = sid in cond_targets or (ic is not None and ic in cond_targets)
        if set_by_condition:
            t0_source, declared_t0_used, note = (
                "condition_table",
                False,
                "set per condition by the condition table — the Antimony t=0 is NOT used.",
            )
        elif estimated:
            t0_source, declared_t0_used, note = (
                "estimated",
                False,
                "ESTIMATED free parameter (the default for initial conditions) — the fit sets "
                "t=0; the declared value is only the starting guess.",
            )
        else:
            # FIXED. Under the always-on default, an IC is only fixed at the two leak-free value
            # conventions (0 = empty pool, 1 = unit normalization) or because it was shipped as a
            # fixed parameter row; say which, so the caller sees why it is not estimated.
            if ic is None and declared in (None, 0.0):
                why = (
                    "an empty pool (t=0 == 0), carried as a literal — no `<species>_init` "
                    "parameter exists for it, so it costs nothing in k"
                )
            elif ic is None:
                # No minted `<species>_init` does NOT imply a zero initial. An assignment-rule
                # target reaches here with `ic` None (ensure_parametric_initial_conditions drops
                # the initial assignment and continues) while the species still declares a
                # non-zero concentration. Claiming "t=0 == 0" then contradicted `declared_t0` in
                # the very same record — a self-contradicting answer to the one question this
                # report exists to settle.
                why = (
                    f"carried as a literal (t=0 == {declared:g}) — no `<species>_init` "
                    "parameter exists for it, so it costs nothing in k"
                )
            elif declared == 0.0:
                why = "an empty pool (t=0 == 0)"
            elif declared == 1.0:
                why = "a unit normalization (t=0 == 1)"
            else:
                why = "shipped as a fixed parameter"
            t0_source, declared_t0_used, note = (
                "fixed",
                True,
                f"FIXED at the declared t=0 value ({why}) — this value governs the simulated "
                "t=0 state.",
            )
        out.append(
            {
                "species": sid,
                "ic_parameter": ic,
                "declared_t0": declared,
                "estimated": bool(estimated),
                "set_by_condition": bool(set_by_condition),
                "t0_source": t0_source,
                "declared_t0_used": bool(declared_t0_used),
                "newly_added": bool(ic in added) if ic else False,
                "note": note,
            }
        )
    return out


# -- pruning ---------------------------------------------------------------
def prune_orphans(problem) -> dict[str, list[str]]:
    """Remove species/parameters no reaction, rule, IC, or fixed table references."""
    sm = problem.model.sbml_model
    keep = table_referenced_symbols(problem)
    # a species is "used" if it participates in a reaction or is referenced by any
    # model math (kinetic law, rule, initial assignment) — not only reactions.
    # A rule TARGET is never an orphan, for species exactly as for parameters below: deleting it
    # leaves the rule pointing at a deleted id (invalid SBML) and the whole edit is rejected.
    # `_model_math_names` walks rule MATH only, so a forcing species driven by a rule that nothing
    # reads -- the `input_species` shape -- would otherwise be removed out from under its own rule.
    rule_targets = {sm.getRule(i).getVariable() for i in range(sm.getNumRules())}
    used_species = _reaction_species(sm) | (_model_math_names(sm) & species_ids(sm)) | rule_targets

    removed_species: list[str] = []
    for sid in list(species_ids(sm)):
        if sid in used_species or sid in keep:
            continue
        # orphan species: drop it, its IC assignment, and IC parameter
        ic = _ic_symbol_for(sm, sid)
        _drop_initial_assignment(sm, sid)
        if ic and sm.getParameter(ic) is not None:
            sm.removeParameter(ic)
        sm.removeSpecies(sid)
        removed_species.append(sid)

    # A rule's TARGET is assigned by that rule, so it is never an orphan even when nothing reads
    # it: removing it would leave the rule pointing at a deleted id, which is invalid SBML
    # ("the value of an <assignmentRule>'s 'variable' must be the identifier of an existing
    # <compartment>, <species>, <parameter> …") and the whole edit is then rejected. Published
    # models really do carry such write-only quantities — an inert control compound's gated input
    # (isensee's `fourABnOH`), or a pass-through alias of a condition target (raimundez's
    # `FM`/`HM`) — and those rules are also what makes their condition targets count as USED.
    # `_model_math_names` only walks rule MATH (the right-hand side), so add the targets here.
    keep_names = keep | _model_math_names(sm) | _reaction_species(sm) | rule_targets
    # keep IC params of remaining species
    for i in range(sm.getNumSpecies()):
        ic = _ic_symbol_for(sm, sm.getSpecies(i).getId())
        if ic:
            keep_names.add(ic)

    removed_params: list[str] = []
    removed_initial_assignments: list[str] = []
    for pid in list(parameter_ids(sm)):
        if pid in keep_names:
            continue
        # An initialAssignment's SYMBOL (its target) appears in none of the keep sets:
        # `_model_math_names` walks IA *math* (the right-hand side), so `IPR_total = A*B` keeps A
        # and B but not IPR_total itself. Removing the parameter while leaving its IA behind
        # produced a dangling IA -- invalid SBML -- and the whole edit was then rejected for a
        # corruption the server introduced (11 rejected calls in the Aug 8-10 sweep, on
        # `IPR_total`, `k_bind_S6`, `delta`, `pEGFR_tot`). Drop the IA WITH its symbol rather than
        # keeping a parameter nothing reads: an unreferenced parameter is a genuine orphan, and its
        # IA is only meaningful while the parameter exists. Reported so the caller can say so.
        if _drop_initial_assignment(sm, pid):
            removed_initial_assignments.append(pid)
        sm.removeParameter(pid)
        removed_params.append(pid)

    return {
        "species": removed_species,
        "parameters": removed_params,
        "initial_assignments": removed_initial_assignments,
    }


# -- parameter-table sync --------------------------------------------------
def _ic_parameter_ids(sm) -> set[str]:
    ic: set[str] = set()
    for i in range(sm.getNumInitialAssignments()):
        ia = sm.getInitialAssignment(i)
        # symbol is a species; math names are the IC parameters
        if sm.getSpecies(ia.getSymbol()) is not None:
            ic |= _ast_names(ia.getMath())
    return ic


def _direct_ic_parameter_ids(sm) -> set[str]:
    """IC parameters that are the SOLE symbol of a species initialAssignment (`<sid> = <param>` —
    the shape `_set_parametric_ic` and an author's own `A = A_init` produce): a directly settable
    initial condition. Excludes participants of a COMPOUND expression (`Rec = ini_R1*ini_R2fold`),
    which the {0,1} value convention must NOT fix individually — a participant valued 1 there is a
    fold/normalisation seed the author means to estimate, not a fixed unit-normalisation IC."""
    ic: set[str] = set()
    for i in range(sm.getNumInitialAssignments()):
        ia = sm.getInitialAssignment(i)
        if sm.getSpecies(ia.getSymbol()) is not None:
            names = _ast_names(ia.getMath())
            if len(names) == 1:
                ic |= names
    return ic


def sync_parameter_table(problem, shipped=None) -> dict[str, list[str]]:
    """Add rows for new model parameters; remove rows for parameters gone from the model
    (keeping observable/noise parameters). Returns the add/remove diff.

    ``shipped`` is the parameter table the PROBLEM shipped with
    (:attr:`session.PetabSession.shipped_parameter_df`). A parameter being added that appears
    there gets its SHIPPED row back — bounds, scale and estimate — instead of the defaults derived
    below. That matters because a row can leave the table and come back: the content-free seed
    declares no mechanism, so every kinetic and initial-condition row a zero-shot problem ships is
    pruned at v0 and re-added by the caller's first authoring. Minting defaults there would quietly
    replace the shipped search space, so the candidate and the reference would no longer be fit
    over the same box.

    Initial conditions are ESTIMATED by default -- a declared t=0 is only a guess and must never
    be silently frozen (issue #90). Two leak-free VALUE conventions stay FIXED: an IC of exactly
    ``0`` (an empty pool -- nothing to estimate) and an IC of exactly ``1`` (a unit normalization,
    e.g. a conserved-total set to 1). A caller that needs some other IC fixed ships it as a fixed
    parameter row (reconcile never re-touches existing rows)."""
    sm = problem.model.sbml_model
    valid = set(problem.model.get_valid_parameters_for_parameter_table())
    # A parameter that is a CONDITION TARGET is set per condition (a PEtab condition change),
    # not a parameter-table row — PEtab forbids it in the parameter table. Drop such targets
    # from the sync so a condition table with parametric inputs ships verbatim
    # (see condition_target_symbols); without this, reconcile force-added them and petab lint then
    # rejected the problem ("… not allowed to occur in the parameters table").
    valid -= condition_target_symbols(problem)
    table_ids = set(problem.parameter_df.index)
    ic_params = _ic_parameter_ids(sm)
    exp_params = _exponent_symbols(sm)  # cooperativity/Hill exponents -> lin scale, tight bounds
    keep_table = table_referenced_symbols(problem)  # observable/noise params etc.
    # ...but NOT a symbol PEtab forbids outright. An assignment-/rate-rule TARGET is computed by
    # the model, so petab's own get_valid_parameters_for_parameter_table excludes it (its
    # `disallowed_set` is exactly the rule variables) and lint rejects the problem if it appears:
    # "… is not allowed to occur in the parameters table". `keep_table` was overriding that
    # exclusion whenever an observable formula happened to reference the same symbol — the normal
    # shape for a rule-computed readout — so the row survived the sync and the whole edit was
    # refused. Measured on one sweep: this was the single largest refusal cause, 85 rejections on
    # `prevalence` alone (an assignment-rule target read by an observable), plus IR1in/IR2in/
    # Deaths/Cumulative_cases in the same shape. The agent's model was legal; the sync produced a
    # table petab forbids, so the fix belongs here and not in the model text.
    keep_table -= _rule_target_symbols(sm)
    # Same failure, second shape: a symbol that USED to be a parameter and is now a SPECIES. PEtab
    # forbids species in the parameter table exactly as it forbids rule targets, but `keep_table`
    # overrode that too whenever an observable formula referenced the symbol -- which is the normal
    # shape when the agent promotes a lumped parameter into an explicit state. The stale row then
    # survived the sync and lint refused the problem with "Extraneous parameter(s) in parameter
    # table". 20 rejected calls in the Aug 8-10 sweep, every one of them the same fujita pair
    # {'pEGFR_Akt', 'pAkt_S6'} (18 of them inside a single run, which looped on it for 18 calls).
    # Critically the agent had NO WAY OUT: no tool in the surface can delete a parameter-table row,
    # so the state was unreachable from the agent side and the retry could only fail again.
    keep_table -= species_ids(sm)

    added, removed = [], []

    shipped_ids = set(getattr(shipped, "index", []))
    for pid in sorted(valid - table_ids):
        if pid in shipped_ids:
            # The problem defined this parameter; reintroducing it restores that definition
            # rather than guessing a new one. Column-by-column so the target table gains any
            # column the shipped row carries that it does not have yet.
            for col, cell in shipped.loc[pid].items():
                problem.parameter_df.loc[pid, col] = cell
            added.append(pid)
            continue
        value = problem.model.get_parameter_value(pid)
        if pid in ic_params:
            # Initial conditions are ESTIMATED by default (issue #90): a declared t=0 is only a
            # guess, and freezing it hands the optimizer a value it can never correct (crauste
            # Naive: ~460 guess frozen where the truth is ~8090). A seeded/guessed IC is a
            # strictly-positive concentration, so it is estimated on LOG10 with the usual wide
            # bracketing bounds — exactly like a positive kinetic parameter — resolving seeds that
            # span many orders (a receptor pool ~1e7 beside a kinase ~1e-2).
            #   TWO leak-free VALUE conventions stay FIXED instead:
            #     v0 == 0.0  -> an empty pool: nothing to estimate, and it cannot live on log10.
            #     v0 == 1.0  -> a unit normalization: a conserved-total / normalization convention
            #                   set to 1 (e.g. sneyd IPR_R, blasi x_0ac). Fixing it BY VALUE names
            #                   no species, so — unlike shipping it as a fixed param — it leaks no
            #                   hidden structure to a zero-shot agent.
            # Any OTHER IC that must stay fixed is shipped as a fixed parameter row (which this sync
            # never re-touches). A nonzero non-positive nominal can't live on log10 -> linear.
            try:
                v0 = float(value) if value is not None else 0.0
            except (TypeError, ValueError):
                v0 = 0.0
            lb, ub = _default_bounds(value)
            if v0 == 0.0 or v0 == 1.0:
                scale, estimate = LIN, False  # empty pool / unit normalization: FIXED
            elif v0 > 0.0:
                scale, estimate = LOG10, True  # positive seed/guess: estimate on log10
            else:
                scale, estimate = LIN, True  # nonzero but non-positive: cannot live on log10
        elif pid in exp_params:
            # An exponent (Hill coefficient): estimate on a LINEAR scale with tight bounds so it
            # cannot run off to a near-step-function on the wide default log10 range.
            lb, ub = _exponent_bounds(value)
            scale, estimate = LIN, True
        else:
            # A generic rate constant: estimate on log10 when the nominal is positive (the usual
            # case for a kinetic parameter). A non-positive nominal cannot live on a log scale,
            # so fall back to linear, avoiding an invalid log10 row (a non-positive bound) that
            # the lint would then reject.
            lb, ub = _default_bounds(value)
            try:
                positive = value is not None and float(value) > 0
            except (TypeError, ValueError):
                positive = False
            scale, estimate = (LOG10 if positive else LIN), True
        problem.parameter_df.loc[pid, PARAMETER_SCALE] = scale
        problem.parameter_df.loc[pid, LOWER_BOUND] = lb
        problem.parameter_df.loc[pid, UPPER_BOUND] = ub
        problem.parameter_df.loc[pid, NOMINAL_VALUE] = value
        problem.parameter_df.loc[pid, ESTIMATE] = estimate
        added.append(pid)

    # Enforce the leak-free {0,1} IC-fix convention on IC rows that ALREADY EXIST — the add-loop
    # above only reaches params in `valid - table_ids`, so an IC parameter that was in the table
    # before this sync never gets the value-based treatment. That is not hypothetical: a model-less
    # problem is seeded with a content-free model whose species come from the observables, and
    # reconcile parametrises their ICs into ESTIMATED `<sid>_init` rows (nominal 0.1). When the
    # agent then authors the real mechanism using the SAME state names (e.g. blasi's 16 acetylation
    # states x_0ac..x_4ac), those seed rows are already present, so the {0,1}->FIXED convention was
    # SILENTLY SKIPPED and every steady-state IC stayed estimated — 16 structurally non-identifiable
    # free parameters the agent has no tool to remove (a parameters-arg edit sets bounds/scale only,
    # never estimate). Re-derive the FIXED convention here for a non-shipped IC row whose current
    # model value is exactly 0 (empty pool) or 1 (unit normalization); a shipped row is left as the
    # problem defined it, and an IC with any other value stays estimated (its fitted nominal, which
    # we must not clobber, is never exactly 0 or 1). Restricted to DIRECT ICs (`<sid> = <param>`):
    # a participant of a COMPOUND parametric IC (`Rec = ini_R1*ini_R2fold`) valued 1 is a fold /
    # normalisation seed to estimate, not a unit-normalisation IC to fix.
    for pid in sorted((_direct_ic_parameter_ids(sm) & table_ids) - shipped_ids):
        if pid not in problem.parameter_df.index:
            continue
        try:
            v0 = float(problem.model.get_parameter_value(pid))
        except (TypeError, ValueError):
            continue
        if v0 == 0.0 or v0 == 1.0:
            cur = problem.parameter_df.loc[pid, ESTIMATE]
            try:
                already_fixed = not as_bool(cur)
            except (TypeError, ValueError):
                already_fixed = False
            if not already_fixed:
                # Re-derive bounds to bracket the new nominal (exactly as the add-loop does via
                # _default_bounds) — otherwise the stale positive-log10 seed bounds are left, so an
                # empty-pool IC ends fixed at 0.0 with a lowerBound of 1e-4 ABOVE it.
                lb, ub = _default_bounds(v0)
                problem.parameter_df.loc[pid, PARAMETER_SCALE] = LIN
                problem.parameter_df.loc[pid, LOWER_BOUND] = lb
                problem.parameter_df.loc[pid, UPPER_BOUND] = ub
                problem.parameter_df.loc[pid, NOMINAL_VALUE] = v0
                problem.parameter_df.loc[pid, ESTIMATE] = False

    for pid in sorted(table_ids - valid - keep_table):
        problem.parameter_df = problem.parameter_df.drop(index=pid)
        removed.append(pid)

    return {"added": added, "removed": removed}


#: The species minted when a model would otherwise have none. Named to be obviously not the
#: caller's: it exists only to keep the "there is always at least one species" invariant true.
PLACEHOLDER_SPECIES = "petab_mcp_placeholder"


def ensure_one_species(sm) -> bool:
    """Guarantee the model has at least one species, minting :data:`PLACEHOLDER_SPECIES` if not.

    AMICI's SBML import SEGFAULTS on a zero-species model, taking the server process with it
    where no caller can catch or degrade it. That single state is why `has_model()` is an
    asserted invariant and why the guards it replaced could be deleted from ``fit``,
    ``ensure_simulation``, ``_restore_or_simulate`` and ``identifiability``.

    Minting a placeholder keeps the invariant without distorting what the model MEANS, which the
    alternative — declaring observable-formula identifiers as species — does not: on a blinded
    problem whose formulas name only ``readout_*`` measurement mappings, the mappings themselves
    look like states, get wired into reactions, and can then never be assigned. Every readout is
    a constant instead, and this placeholder carries the invariant.

    Self-clearing: it is referenced by nothing, so the next :func:`prune_orphans` removes it the
    moment the caller's model has a real species. Returns whether one was minted.
    """
    if sm.getNumSpecies() > 0:
        return False
    comp = sm.getCompartment(0)
    if comp is None:
        comp = sm.createCompartment()
        comp.setId("petab_mcp_placeholder_compartment")
        comp.setConstant(True)
        comp.setSize(1.0)
        comp.setSpatialDimensions(3)
    s = sm.createSpecies()
    s.setId(PLACEHOLDER_SPECIES)
    s.setCompartment(comp.getId())
    s.setInitialConcentration(0.0)
    s.setConstant(False)
    s.setBoundaryCondition(False)
    s.setHasOnlySubstanceUnits(False)
    return True


def reconcile(problem, shipped=None) -> dict[str, Any]:
    """Full reconciliation after a structural edit: prune, parametric ICs, table sync.

    Initial conditions are estimated by default; an IC of exactly 1 is held FIXED and an IC of
    exactly 0 gets no parameter at all (see :func:`ensure_parametric_initial_conditions`). ``shipped`` is the problem's original parameter table, so a
    row the model reintroduces comes back as the problem defined it — see
    :func:`sync_parameter_table`.

    Runs unchanged on the content-free seed: it needs no special pass, because a shipped row it
    prunes is restored the moment the caller's model declares that parameter.
    """
    sm = problem.model.sbml_model
    pruned = prune_orphans(problem)
    # AFTER the prune, or it would be pruned as the orphan it deliberately is.
    ensure_one_species(sm)
    # Species whose IC is set per-condition (a condition target) are driven by the condition
    # table, not estimated — skip parametrising them (see ensure_parametric_initial_conditions).
    # `protect_ids` is the SHIPPED table's own ids: if the problem author shipped a `<sid>_init`
    # row, that row is part of the problem definition and the zero/unset skip must not delete it
    # out from under them (`session.protected_fixed_parameter_ids` deliberately excludes IC
    # parameters, so nothing else would catch it).
    ensure_parametric_initial_conditions(
        sm,
        skip_species=condition_target_symbols(problem),
        protect_ids=set(shipped.index) if shipped is not None else None,
    )
    synced = sync_parameter_table(problem, shipped=shipped)
    return {"pruned": pruned, "parameters": synced}
