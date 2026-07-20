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
from petab.v1.C import (
    CONDITION_ID,
    CONDITION_NAME,
    ESTIMATE,
    LIN,
    LOG10,
    LOWER_BOUND,
    MEASUREMENT,
    NOISE_FORMULA,
    NOISE_PARAMETERS,
    NOMINAL_VALUE,
    OBSERVABLE_FORMULA,
    OBSERVABLE_ID,
    OBSERVABLE_PARAMETERS,
    PARAMETER_SCALE,
    UPPER_BOUND,
)

from .constants import (
    EXPONENT_LOWER_BOUND,
    EXPONENT_UPPER_BOUND,
    NOISE_SIGMA_MAX_FRACTION_OF_DATA_STD,
)
from .errors import ParameterError
from .parameters import _default_bounds

# Leading (?<![\w.]) so a digit-glued run is never read as an identifier: the `e5` in `1e5`
# and the `e` in `2.5e-3` (scientific-notation exponents) stay part of their numeric literal.
_IDENT_RE = re.compile(r"(?<![\w.])[A-Za-z_]\w*")


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


def rate_law_symbols(rate_law: str) -> set[str]:
    """Names referenced by a rate-law / expression string."""
    math = libsbml.parseL3Formula(rate_law)
    if math is None:
        raise ValueError(f"Could not parse expression: {rate_law!r}")
    return _ast_names(math)


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


def species_ids(sm) -> set[str]:
    return {sm.getSpecies(i).getId() for i in range(sm.getNumSpecies())}


def parameter_ids(sm) -> set[str]:
    return {sm.getParameter(i).getId() for i in range(sm.getNumParameters())}


def default_compartment(sm) -> str:
    if sm.getNumCompartments() == 0:
        c = sm.createCompartment()
        c.setId("default_compartment")
        c.setConstant(True)
        c.setSize(1.0)
    return sm.getCompartment(0).getId()


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


def add_species_with_parametric_ic(sm, sid: str, initial_value: float = 0.0) -> None:
    comp = default_compartment(sm)
    sp = sm.createSpecies()
    sp.setId(sid)
    sp.setCompartment(comp)
    sp.setHasOnlySubstanceUnits(False)
    sp.setBoundaryCondition(False)
    sp.setConstant(False)
    sp.setInitialConcentration(float(initial_value))
    _set_parametric_ic(sm, sid, float(initial_value))


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


def ensure_parametric_initial_conditions(sm) -> list[str]:
    """Give every species a parametric IC; returns the IC parameter ids created/kept."""
    ic_params: list[str] = []
    for i in range(sm.getNumSpecies()):
        sp = sm.getSpecies(i)
        sid = sp.getId()
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
        value = 0.0
        if sp.isSetInitialConcentration():
            value = sp.getInitialConcentration()
        elif sp.isSetInitialAmount():
            value = sp.getInitialAmount()
        ic_params.append(_set_parametric_ic(sm, sid, value))
    return ic_params


def add_parameter(sm, pid: str, value: float = 0.1) -> None:
    if sm.getParameter(pid) is None:
        p = sm.createParameter()
        p.setId(pid)
        p.setConstant(True)
        p.setValue(float(value))


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


def input_species(sm) -> set[str]:
    """Species that are model INPUTS / forcing terms rather than free dynamic states: held
    constant, on the boundary, or driven by an assignment- or rate-rule (a forcing function).

    Their trajectory is imposed, not fitted — like a condition-table column — so they are
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
    names = _observable_and_measurement_symbols(problem)
    cdf = problem.condition_df
    if cdf is not None:
        for col in cdf.columns:
            if col != CONDITION_NAME:
                names.add(str(col))
            for val in cdf[col]:
                names |= formula_symbols(val)
    return names


def condition_column_symbols(problem) -> set[str]:
    """Parameters/species that are CONDITION-TABLE COLUMN HEADERS (excluding conditionId/Name).

    A model parameter that appears as a condition-table column is set PER CONDITION — a PEtab
    condition override — so it must NOT be written into the parameter table: PEtab rejects it there
    ("… is not allowed to occur in the parameters table"). Excluding these from the parameter-table
    sync lets a benchmark condition table with parametric input columns (EGF doses, insulin steps,
    rate multipliers, genotype switches) ship VERBATIM instead of being rewritten onto species.
    Mirrors petab's own ``get_valid_parameters_for_parameter_table`` (which blacklists the condition
    columns). Cell *values* that reference an estimated parameter (a shared dose amplitude) are a
    different thing — those stay, via :func:`table_referenced_symbols`."""
    cdf = problem.condition_df
    if cdf is None:
        return set()
    return {str(c) for c in cdf.columns} - {CONDITION_NAME, CONDITION_ID}


def observation_model_symbols(problem) -> set[str]:
    """Symbols belonging to the FIXED *observation model* specifically — observable-scaling
    factors and noise sigmas resolved through the observable/noise formulas and their
    measurement-level ``observableParameters``/``noiseParameters`` overrides.

    Deliberately EXCLUDES condition-table symbols (those are legitimate model DRIVING INPUTS —
    e.g. a dose species fed through a condition column — not observation-model constants).
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


def parameter_roles(problem) -> dict[str, str]:
    """Best-effort role for every parameter-table id — ``{parameter_id: role}``.

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
    for raw in problem.parameter_df.index:
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
    ``noiseFormula`` rather than through the ``noiseParameter1_<obsId>`` placeholder
    convention) via the observable(s) whose formula names it. ``None`` if no match/no data."""
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
        obs_ids = [oid for oid, formula in odf[NOISE_FORMULA].items()
                   if pid in formula_symbols(formula)]
        if obs_ids:
            return mdf[mdf[OBSERVABLE_ID].isin(obs_ids)]
    return None


def sigma_data_std(problem, pid: str) -> float | None:
    """Raw measurement std (across every time point / condition) of the observable(s)
    noise-sigma parameter ``pid`` models the noise for — the basis for its bound cap (see
    :func:`sigma_upper_bound_cap`) and for reporting a fitted sigma against its reference (see
    ``fitting.noise_sigma_report``). ``None`` if ``pid`` isn't a resolvable sigma, or there's
    too little data to compute a std.
    """
    rows = _measurement_rows_for_sigma(problem, pid)
    if rows is None or rows.empty:
        return None
    values = rows[MEASUREMENT].dropna().astype(float)
    if len(values) < 2:
        return None
    std = float(values.std(ddof=0))
    return std if std > 0 else None


def sigma_upper_bound_cap(
    problem, pid: str, max_fraction: float = NOISE_SIGMA_MAX_FRACTION_OF_DATA_STD,
) -> float | None:
    """Data-driven upper-bound cap for noise-sigma parameter ``pid``: ``max_fraction`` times
    :func:`sigma_data_std`. ``None`` if that std can't be computed — callers must not
    tighten/reject on a ``None`` cap.

    A sigma free to explain a large fraction of the data's own variability isn't measurement
    noise anymore, it's covering for the model: an optimizer can make almost ANY mechanism
    "pass" a chi2 goodness-of-fit check by simply inflating sigma until residuals fit inside
    it. See ``NOISE_SIGMA_MAX_FRACTION_OF_DATA_STD``.
    """
    std = sigma_data_std(problem, pid)
    return max_fraction * std if std is not None else None


def is_parameter_estimated(problem, pid: str) -> bool:
    """Whether ``pid``'s current ``estimate`` flag is truthy (robust to ``NaN``/missing —
    treated as not-estimated, same idiom as ``set_model``'s prior-estimated snapshot)."""
    df = problem.parameter_df
    if pid not in df.index or ESTIMATE not in df.columns:
        return False
    try:
        return float(df.loc[pid, ESTIMATE]) != 0
    except (TypeError, ValueError):
        return False


# -- initial-condition report ----------------------------------------------
def initial_conditions_report(
    problem, added: set[str] | None = None
) -> list[dict[str, Any]]:
    """Per-species account of how each t=0 initial condition is resolved *after* reconcile.

    Antimony can only declare a species' t=0 *value* — it has no way to say "estimate this
    initial condition". Whether that declared value actually governs the simulated t=0 state is
    decided by the reconcile/parameter/condition tables, not the model text, and the outcome is
    otherwise invisible: a declared t=0 is silently OVERRIDDEN when its ``<species>_init``
    parameter is estimated (the fit then sets t=0, the declared value is only the start) or when
    the species is driven per-condition by a condition-table column. Surfacing this after every
    model op is what lets the caller see immediately whether a t=0 "fix" took — instead of
    re-authoring the model under a series of different IC naming schemes that all fit identically
    because none of them ever changed the estimate status (see issue #57). Whether ICs are
    estimated at all is fixed by the scenario (the ``estimate_initial_conditions`` server
    setting), not switchable from the model text.

    For each species returns ``{species, ic_parameter, declared_t0, estimated, set_by_condition,
    t0_source, declared_t0_used, newly_added, note}``:

    * ``declared_t0`` — the t=0 value declared in the Antimony (the IC parameter's value);
    * ``estimated`` — whether the IC parameter is a FREE (fit) parameter or FIXED;
    * ``set_by_condition`` — whether the species (or its IC parameter) is a condition-table column,
      i.e. set per condition, which overrides the Antimony t=0;
    * ``t0_source`` — ``"fixed"`` | ``"estimated"`` | ``"condition_table"``;
    * ``declared_t0_used`` — the bottom line: does the declared t=0 govern the simulated t=0 state
      (``True`` only for a plain FIXED IC), or is it overridden;
    * ``newly_added`` — whether this op introduced the IC parameter (from the reconcile diff).
    """
    sm = problem.model.sbml_model
    cond_cols = condition_column_symbols(problem)
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
        estimated = ic is not None and is_parameter_estimated(problem, ic)
        set_by_condition = sid in cond_cols or (ic is not None and ic in cond_cols)
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
                "ESTIMATED free parameter — the fit sets t=0; the declared value is only the "
                "starting guess.",
            )
        else:
            t0_source, declared_t0_used, note = (
                "fixed",
                True,
                "FIXED at the declared t=0 value — this value governs the simulated t=0 state.",
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
    used_species = _reaction_species(sm) | (_model_math_names(sm) & species_ids(sm))

    removed_species: list[str] = []
    for sid in list(species_ids(sm)):
        if sid in used_species or sid in keep:
            continue
        # orphan species: drop it, its IC assignment, and IC parameter
        ic = _ic_symbol_for(sm, sid)
        for i in range(sm.getNumInitialAssignments()):
            if sm.getInitialAssignment(i).getSymbol() == sid:
                sm.removeInitialAssignment(i)
                break
        if ic and sm.getParameter(ic) is not None:
            sm.removeParameter(ic)
        sm.removeSpecies(sid)
        removed_species.append(sid)

    keep_names = keep | _model_math_names(sm) | _reaction_species(sm)
    # keep IC params of remaining species
    for i in range(sm.getNumSpecies()):
        ic = _ic_symbol_for(sm, sm.getSpecies(i).getId())
        if ic:
            keep_names.add(ic)

    removed_params: list[str] = []
    for pid in list(parameter_ids(sm)):
        if pid in keep_names:
            continue
        sm.removeParameter(pid)
        removed_params.append(pid)

    return {"species": removed_species, "parameters": removed_params}


# -- parameter-table sync --------------------------------------------------
def _ic_parameter_ids(sm) -> set[str]:
    ic: set[str] = set()
    for i in range(sm.getNumInitialAssignments()):
        ia = sm.getInitialAssignment(i)
        # symbol is a species; math names are the IC parameters
        if sm.getSpecies(ia.getSymbol()) is not None:
            ic |= _ast_names(ia.getMath())
    return ic


def sync_parameter_table(problem, estimate_ics: bool = False) -> dict[str, list[str]]:
    """Add rows for new model parameters; remove rows for parameters gone from the model
    (keeping observable/noise parameters). Returns the add/remove diff.

    ``estimate_ics`` (opt-in, per scenario) makes initial-condition parameters ESTIMATED rather
    than fixed at the observed t=0 value -- EXCEPT species that start empty (t=0 == 0), which stay
    fixed at 0 (a scenario can thus estimate only the seeded pools without shipping named ICs)."""
    sm = problem.model.sbml_model
    valid = set(problem.model.get_valid_parameters_for_parameter_table())
    # A parameter that is a CONDITION-TABLE COLUMN is set per condition (a PEtab condition
    # override), not a parameter-table row — PEtab forbids it in the parameter table. Drop such
    # columns from the sync so a condition table with parametric input columns ships verbatim
    # (see condition_column_symbols); without this, reconcile force-added them and petab lint then
    # rejected the problem ("… not allowed to occur in the parameters table").
    valid -= condition_column_symbols(problem)
    table_ids = set(problem.parameter_df.index)
    ic_params = _ic_parameter_ids(sm)
    exp_params = _exponent_symbols(sm)  # cooperativity/Hill exponents -> lin scale, tight bounds
    keep_table = table_referenced_symbols(problem)  # observable/noise params etc.

    added, removed = [], []

    for pid in sorted(valid - table_ids):
        value = problem.model.get_parameter_value(pid)
        if pid in ic_params:
            # Initial conditions: FIXED at the observed t=0 value by default. When the scenario
            # opts in (estimate_ics), ESTIMATE them instead — treating the noisy t=0 measurement as
            # a parameter to fit and counting it honestly in k, not as ground truth. Opting in
            # requires the scenario's BIC reference to estimate its ICs too, so the comparison stays
            # fair; left off, published problems keep their standard ICs.
            # EXCEPTION: a species that STARTS EMPTY (t=0 value == 0) stays FIXED at 0 even under
            # estimate_ics. "Estimating" an empty pool is meaningless, and fixing it lets a scenario
            # estimate ONLY the genuinely-seeded species without having to ship named <sid>_init
            # seeds — the candidate's zero-start species are fixed at 0 by this same rule, so the
            # BIC comparison stays at parity (a scenario can seed a few pools and leave the rest 0).
            # A seeded IC is a strictly-positive concentration, so it is estimated on LOG10 with the
            # usual wide bracketing bounds — exactly like a positive kinetic parameter. This resolves
            # seeds spanning many orders of magnitude (a receptor pool ~1e7 beside a kinase ~1e-2),
            # which a linear IC scale could not.
            try:
                v0 = float(value) if value is not None else 0.0
            except (TypeError, ValueError):
                v0 = 0.0
            lb, ub = _default_bounds(value)
            if bool(estimate_ics) and v0 > 0.0:
                scale, estimate = LOG10, 1
            elif bool(estimate_ics) and v0 != 0.0:
                scale, estimate = LIN, 1          # nonzero but non-positive: cannot live on log10
            else:
                scale, estimate = LIN, 0          # empty pool, or estimate_ics off: fixed
        elif pid in exp_params:
            # An exponent (Hill coefficient): estimate on a LINEAR scale with tight bounds so it
            # cannot run off to a near-step-function on the wide default log10 range.
            lb, ub = _exponent_bounds(value)
            scale, estimate = LIN, 1
        else:
            # A generic rate constant: estimate on log10 when the nominal is positive (the usual
            # case for a kinetic parameter). A non-positive nominal cannot live on a log scale,
            # so fall back to linear — mirroring the guard in parameters.sync_parameters and
            # avoiding an invalid log10 row (non-positive bound) that the lint would then reject.
            lb, ub = _default_bounds(value)
            try:
                positive = value is not None and float(value) > 0
            except (TypeError, ValueError):
                positive = False
            scale, estimate = (LOG10 if positive else LIN), 1
        problem.parameter_df.loc[pid, PARAMETER_SCALE] = scale
        problem.parameter_df.loc[pid, LOWER_BOUND] = lb
        problem.parameter_df.loc[pid, UPPER_BOUND] = ub
        problem.parameter_df.loc[pid, NOMINAL_VALUE] = value
        problem.parameter_df.loc[pid, ESTIMATE] = estimate
        added.append(pid)

    for pid in sorted(table_ids - valid - keep_table):
        problem.parameter_df = problem.parameter_df.drop(index=pid)
        removed.append(pid)

    return {"added": added, "removed": removed}


def reconcile(problem, estimate_ics: bool = False) -> dict[str, Any]:
    """Full reconciliation after a structural edit: prune, parametric ICs, table sync.

    ``estimate_ics`` (opt-in) makes initial-condition parameters estimated (see
    :func:`sync_parameter_table`)."""
    sm = problem.model.sbml_model
    pruned = prune_orphans(problem)
    ensure_parametric_initial_conditions(sm)
    synced = sync_parameter_table(problem, estimate_ics=estimate_ics)
    return {"pruned": pruned, "parameters": synced}


# -- manual parameter-table sync (the `sync_parameters` tool) ---------------
def parameter_sync_diff(problem) -> dict[str, Any]:
    """What a parameter-table sync would change, WITHOUT mutating anything.

    Keyed the same way as :func:`sync_parameter_table` (petab's
    ``get_valid_parameters_for_parameter_table`` — every model parameter except rule targets):

    * ``model_only`` — model parameters missing from the table (a sync would ADD these);
    * ``table_only`` — orphaned table rows: not backed by a model parameter AND not referenced
      by the fixed observable/noise/condition tables (a sync would REMOVE these). Rows the
      fixed tables reference (e.g. ``sigma_*``) are kept and never listed here.
    """
    valid = set(problem.model.get_valid_parameters_for_parameter_table())
    # A condition-table column is set per condition, not a parameter-table row (PEtab forbids it
    # there), so `sync_parameter_table` excludes these from its adds — mirror that here, or the
    # report claims a sync would add e.g. `dose` while `add` correctly leaves it out.
    valid -= condition_column_symbols(problem)
    table_ids = set(problem.parameter_df.index)
    keep_table = table_referenced_symbols(problem)
    orphans = table_ids - valid - keep_table
    return {
        "in_sync": valid.issubset(table_ids) and not orphans,
        "model_only": sorted(valid - table_ids),
        "table_only": sorted(orphans),
    }


def sync_parameters(problem, policy: str = "report", estimate_ics: bool = False) -> dict[str, Any]:
    """Reconcile the parameter table with the model ('report' or 'add').

    ``'report'`` returns :func:`parameter_sync_diff` and changes nothing. ``'add'`` runs the
    SAME :func:`sync_parameter_table` the commit pipeline runs — adding rows for model
    parameters missing from the table (with the initial-condition / Hill-exponent / scale
    handling) AND removing orphaned rows — so the manual tool and the automatic commit path
    reconcile identically. Returns ``{policy, added, removed, diff}``.
    """
    if policy == "report":
        return {"policy": policy, "added": [], "removed": [], "diff": parameter_sync_diff(problem)}
    if policy == "add":
        synced = sync_parameter_table(problem, estimate_ics=estimate_ics)
        return {"policy": policy, "added": synced["added"], "removed": synced["removed"],
                "diff": parameter_sync_diff(problem)}
    raise ParameterError(f"Unknown sync policy {policy!r}; use 'report' or 'add'.")
