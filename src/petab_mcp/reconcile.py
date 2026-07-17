"""Keep the SBML model and the PEtab parameter table consistent after a model edit.

Enforces the invariants:

* **Parametric initial conditions** — every species ``S`` gets an initial-assignment to
  a parameter ``S_init``.
* **Parameter-table sync** — every model parameter (rate constants + IC parameters) has a
  row in the parameter table; parameters no longer in the model are removed, *except*
  those used by the fixed observable/noise formulas (e.g. ``sigma_*``).
* **Orphan pruning** — species/parameters left unreferenced by any reaction, rule,
  initial assignment, or fixed table are dropped.
"""

from __future__ import annotations

import re
from typing import Any

import libsbml
from petab.v1.C import (
    ESTIMATE,
    LIN,
    LOG10,
    LOWER_BOUND,
    NOMINAL_VALUE,
    PARAMETER_SCALE,
    UPPER_BOUND,
)

from .parameters import _default_bounds

_IDENT_RE = re.compile(r"[A-Za-z_]\w*")


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
    lb, ub = 1.0, 8.0
    if nominal is not None:
        n = float(nominal)
        lb, ub = min(lb, n), max(ub, n)
    return lb, ub


def formula_symbols(formula: Any) -> set[str]:
    """Best-effort identifier extraction from an observable/noise/condition formula."""
    if formula is None:
        return set()
    text = str(formula)
    # drop numeric-looking tokens and common function names
    ident = set(_IDENT_RE.findall(text))
    return {t for t in ident if not t.isdigit()}


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


def table_referenced_symbols(problem) -> set[str]:
    """Symbols the *fixed* tables (observables, conditions) depend on."""
    names: set[str] = set()
    odf = problem.observable_df
    if odf is not None:
        for col in ("observableFormula", "noiseFormula"):
            if col in odf.columns:
                for val in odf[col]:
                    names |= formula_symbols(val)
    cdf = problem.condition_df
    if cdf is not None:
        for col in cdf.columns:
            if col != "conditionName":
                names.add(str(col))
            for val in cdf[col]:
                names |= formula_symbols(val)
    # measurement-level observable/noise parameter overrides may reference parameters
    mdf = problem.measurement_df
    if mdf is not None:
        for col in ("observableParameters", "noiseParameters"):
            if col in mdf.columns:
                for val in mdf[col]:
                    names |= formula_symbols(val)
    return names


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


def _ic_bounds(value: float) -> tuple[float, float]:
    """Bounds for an ESTIMATED initial-condition parameter: a concentration, so non-negative,
    on a linear scale, generously bracketing the observed t=0 value."""
    try:
        v = max(0.0, float(value))
    except (TypeError, ValueError):
        v = 0.0
    return 0.0, max(1e2, v * 10.0)


def sync_parameter_table(problem, estimate_ics: bool = False) -> dict[str, list[str]]:
    """Add rows for new model parameters; remove rows for parameters gone from the model
    (keeping observable/noise parameters). Returns the add/remove diff.

    ``estimate_ics`` (opt-in, per scenario) makes initial-condition parameters ESTIMATED rather
    than fixed at the observed t=0 value."""
    sm = problem.model.sbml_model
    valid = set(problem.model.get_valid_parameters_for_parameter_table())
    table_ids = set(problem.parameter_df.index)
    ic_params = _ic_parameter_ids(sm)
    exp_params = _exponent_symbols(sm)  # cooperativity/Hill exponents -> lin scale, tight bounds
    keep_table = table_referenced_symbols(problem)  # observable/noise params etc.

    added, removed = [], []

    for pid in sorted(valid - table_ids):
        value = problem.model.get_parameter_value(pid)
        if pid in ic_params:
            # Initial conditions: FIXED at the observed t=0 value by default. When the scenario
            # opts in (estimate_ics), ESTIMATE them instead (non-negative, linear) — treating the
            # noisy t=0 measurement as a parameter to fit and count honestly in k, not ground
            # truth. Opting in requires the scenario's BIC reference to estimate its ICs too, so
            # the comparison stays fair; left off, published problems keep their standard ICs.
            lb, ub = _ic_bounds(value) if estimate_ics else _default_bounds(value)
            scale, estimate = LIN, (1 if estimate_ics else 0)
        elif pid in exp_params:
            # An exponent (Hill coefficient): estimate on a LINEAR scale with tight bounds so it
            # cannot run off to a near-step-function on the wide default log10 range.
            lb, ub = _exponent_bounds(value)
            scale, estimate = LIN, 1
        else:
            lb, ub = _default_bounds(value)
            scale, estimate = LOG10, 1
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
