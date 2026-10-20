"""Parameter-table inspection, editing, and model reconciliation.

The parameter table is a DataFrame indexed by ``parameterId`` with ``lowerBound``,
``upperBound``, ``nominalValue``, ``estimate`` (a bool) and the server's ``parameterScale``
extension column (lin/log/log10; see :mod:`petab_mcp.problem`). Edits operate directly on
``problem.parameter_df``.
"""

from __future__ import annotations

import hashlib
from typing import Any

from petab.v2.C import (
    ESTIMATE,
    LOWER_BOUND,
    NOMINAL_VALUE,
    PARAMETER_ID,
    UPPER_BOUND,
)

from .constants import DEFAULT_PARAMETER_LOWER_BOUND, DEFAULT_PARAMETER_UPPER_BOUND
from .errors import ParameterError
from .formatting import _clean
from .problem import PARAMETER_SCALE, SCALES, as_bool

_SCALES = set(SCALES)
_NUMERIC_FIELDS = {LOWER_BOUND, UPPER_BOUND, NOMINAL_VALUE}
_EDITABLE_FIELDS = (
    NOMINAL_VALUE,
    LOWER_BOUND,
    UPPER_BOUND,
    ESTIMATE,
    PARAMETER_SCALE,
)


def sha16(text: str) -> str:
    """A short, stable content hash (first 16 hex chars of SHA-256) of ``text``.

    The one truncated-SHA256 idiom shared by every identity key in the package: the SBML
    (compile) hash, the problem fingerprint, and the fit fingerprint. Centralised so the
    digest length/algorithm is defined once.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _default_bounds(nominal) -> tuple[float, float]:
    """Bounds for an auto-added parameter that always bracket its nominal value.

    A ZERO nominal is bracketed by [0, ub], not by dipping below: the zero case is in practice
    always an initial CONCENTRATION (reconcile minting `<sid>_init` for a species that starts
    empty), and the old `nominal - 1.0` handed every such row a lower bound of -1.0. Those rows
    ship fixed, so the fit never sampled the negative range — but the -1.0 was visible in every
    parameter read and draft TSV (which deliberately omit the estimate column), where it read as
    a live negative-concentration box and sent one segfault investigation down a dead end. Zero
    is IN [0, ub], so the bracket contract holds. A genuinely negative nominal (a log-space or
    difference parameter) keeps the old dip, which is correct there.
    """
    lb, ub = DEFAULT_PARAMETER_LOWER_BOUND, DEFAULT_PARAMETER_UPPER_BOUND
    if nominal is None:
        return lb, ub
    nominal = float(nominal)
    if nominal > 0:
        return min(lb, nominal), max(ub, nominal)
    if nominal == 0:
        return 0.0, ub
    return min(lb, nominal - 1.0), max(ub, nominal + 1.0)


def _param_record(df, pid: str) -> dict[str, Any]:
    row = df.loc[pid]
    rec: dict[str, Any] = {PARAMETER_ID: pid}
    for col in df.columns:
        rec[col] = _clean(row[col])
    if ESTIMATE in rec and rec[ESTIMATE] is not None:
        rec[ESTIMATE] = as_bool(rec[ESTIMATE])
    return rec


def parameter_df_to_records(df) -> list[dict[str, Any]]:
    """Return a parameter-table DataFrame as a list of JSON-friendly records."""
    return [_param_record(df, pid) for pid in df.index]


def parameters_to_records(problem, shipped=None) -> list[dict[str, Any]]:
    """Return the problem's parameter table as JSON-friendly records, each annotated with its
    ``role``, its ``status`` ("estimated" / "fixed") and ``in_model``.

    The ``role`` (see :func:`reconcile.parameter_roles`) tells a mechanistic KINETIC constant
    apart from an observation-model factor (a scaling/offset or noise sigma), an initial
    condition, a Hill exponent, or a structural CONSTANT; ``status`` restates the raw
    ``estimate`` flag in words. Both make a parameter's declared role legible at a glance —
    so a scaling factor isn't repurposed as a rate constant, and a known constant is used as the
    fixed parameter the problem ships rather than spent as an estimated (k-counting) one. The
    estimated/fixed status is set by the shipped table and reconcile, not toggled by hand.

    ``shipped`` is the parameter table the PROBLEM shipped with
    (:attr:`session.PetabSession.shipped_parameter_df`). Any row there that the current model
    does not declare is appended with ``in_model: false``, carrying the values the problem gave
    it. Those rows are NOT in the live parameter table — PEtab has no place for a parameter no
    model uses, so reconcile prunes them — but they are still parameters the problem HANDED the
    caller, and a model-less session starts with every one of them pruned (the content-free seed
    declares no mechanism). Without this they would be invisible until the caller happened to
    guess the name: it could not see the constants it is required to build on, nor the ones it
    is required to keep declared. Re-declaring such a parameter in the model restores exactly
    the row shown here (see :func:`reconcile.sync_parameter_table`).
    """
    from .reconcile import parameter_roles  # lazy: reconcile imports from this module

    current = set(problem.parameter_df.index)
    absent = [pid for pid in getattr(shipped, "index", []) if pid not in current]
    roles = parameter_roles(problem, extra_ids=absent)
    records = [{**r, "in_model": True} for r in parameter_df_to_records(problem.parameter_df)]
    if absent:
        records += [{**r, "in_model": False} for r in parameter_df_to_records(shipped.loc[absent])]
    for rec in records:
        pid = str(rec.get(PARAMETER_ID))
        rec["role"] = roles.get(pid)
        est = rec.get(ESTIMATE)
        rec["status"] = None if est is None else ("estimated" if est else "fixed")
    return records


def parameter_df_to_tsv(df) -> str:
    """Serialize the parameter table to PEtab v2 TSV text (index = ``parameterId``, ``estimate``
    spelled ``true``/``false``).

    This is the on-disk form of a version's parameter table (a real DataFrame, not JSON
    records) and the canonical serialization hashed by :func:`problem_fingerprint`.
    """
    if ESTIMATE in df.columns:
        df = df.assign(**{ESTIMATE: ["true" if as_bool(v) else "false" for v in df[ESTIMATE]]})
    return df.to_csv(sep="\t")


def problem_fingerprint(sbml_str: str, parameter_df) -> str:
    """Short hash of the full problem identity: model SBML + the parameter-table DataFrame.

    Distinct from the module hash the AMICI compile cache is keyed on: this also
    changes when a parameter's bounds / scale / estimate / value change, so it correctly
    identifies a *fit* — whose result depends on the parameter table, not just the model
    structure. Keying fit artifacts (the pyPESTO result cache) and provenance on this,
    rather than the model alone, keeps two versions that differ only in bounds distinct.
    """
    payload = sbml_str + "\n" + parameter_df_to_tsv(parameter_df)
    return sha16(payload)


def estimated_mask(parameter_df) -> Any:
    """Boolean mask over ``parameter_df.index``: True where ``estimate`` is (NaN/missing → False).

    The single expression every "which parameters are estimated" site shares, so the coercion
    rule (:func:`problem.as_bool`) lives in one place and cannot drift. Assumes ``ESTIMATE`` is a
    column; raises ``ValueError`` on a cell that is not a boolean — callers wrap it with their own
    fallback (``estimated_ids`` treats that as "all free", the re-author snapshot as "none free").
    """
    return parameter_df[ESTIMATE].fillna(False).map(as_bool).astype(bool)


def estimated_ids(parameter_df) -> list[str]:
    """Sorted ids of the ESTIMATED (``estimate`` != 0) parameters.

    The free-parameter SET that, together with the reaction network, defines a model's
    *structure* for :func:`structure_fingerprint`. Robust to a missing/NaN/non-numeric
    ``estimate`` column (the same idiom ``set_model`` uses for its prior-estimated snapshot).
    """
    if parameter_df is None or ESTIMATE not in getattr(parameter_df, "columns", []):
        return []
    try:
        mask = estimated_mask(parameter_df)
    except (TypeError, ValueError):  # non-numeric estimate column -> treat every id as free
        return sorted(str(p) for p in parameter_df.index)
    return sorted(str(p) for p in parameter_df.index[mask])


def structure_fingerprint(sbml_model, parameter_df) -> str:
    """Short hash of the model STRUCTURE: every equation that shapes the simulated trajectory or a
    readout, plus the SET of estimated parameters — deliberately IGNORING nominal (starting)
    values, bounds, and scales.

    The equations are the species (with the flags that change how reactions move them), each
    reaction's reactant/product stoichiometry + rate-law FORM, every rule (assignment, rate,
    algebraic), initial assignment and event, with function calls expanded. Rules carry most of a
    model built from statements — each rate is an assignment rule the rate law multiplies, and each
    ``readout_*`` mapping is one too — so a fingerprint of the reactions alone called a changed rate
    or readout a duplicate and the change was never fitted.

    This is the identity used to recognise a *re-authored structure* across ``set_model`` calls:
    two authorings that describe the same equations with the same estimated-parameter set share a
    fingerprint even if their starting guesses or bounds differ. That "ignore nominal values"
    property is essential — the automatic fit overwrites an estimated parameter's ``nominalValue``
    with the fitted optimum, so a re-author's *authored* starting values never match the committed
    version's *fitted* ones; a value-sensitive hash (``problem_fingerprint``) would therefore never
    recognise the duplicate. It is coarser than ``problem_fingerprint`` (the exact fit) — it answers
    "have we visited this structure before?", which is what de-duplicating re-authors, warning on
    revisited worse families, and stopping identical-re-author loops need (see
    :mod:`petab_mcp.dedup`). Component ids that nothing references (a reaction's, an event's) are
    excluded for the same reason: the same model authored under other names is the same structure.
    """
    import libsbml

    sm = sbml_model
    functions = sm.getListOfFunctionDefinitions()

    def form(math) -> str:
        # Canonical infix form, whitespace stripped so trivial spacing differences don't change
        # the structural identity. Calls are expanded: a version is stored as an Antimony render,
        # which inlines every function, so the restored model calls none the authored one did.
        if math is None:
            return ""
        if functions.size():
            math = math.deepCopy()
            libsbml.SBMLTransforms.replaceFD(math, functions)
        return (libsbml.formulaToL3String(math) or "").replace(" ", "")

    species = sorted(
        (s.getId(), s.getBoundaryCondition(), s.getHasOnlySubstanceUnits())
        for s in sm.getListOfSpecies()
    )
    reactions: list[tuple] = []
    for i in range(sm.getNumReactions()):
        rxn = sm.getReaction(i)
        reactants = sorted(
            (rxn.getReactant(j).getSpecies(), float(rxn.getReactant(j).getStoichiometry()))
            for j in range(rxn.getNumReactants())
        )
        products = sorted(
            (rxn.getProduct(j).getSpecies(), float(rxn.getProduct(j).getStoichiometry()))
            for j in range(rxn.getNumProducts())
        )
        kl = rxn.getKineticLaw()
        law = form(kl.getMath()) if kl is not None else ""
        # Reaction id is deliberately excluded — an identical network authored with different
        # reaction ids (J1 vs r1) is the same structure. Sorting makes it order-independent.
        reactions.append((tuple(reactants), tuple(products), law))
    reactions.sort()
    events = []
    for ev in sm.getListOfEvents():
        trigger, delay, priority = ev.getTrigger(), ev.getDelay(), ev.getPriority()
        events.append(
            (
                form(trigger.getMath()) if trigger is not None else "",
                trigger.getInitialValue() if trigger is not None else None,
                trigger.getPersistent() if trigger is not None else None,
                form(delay.getMath()) if delay is not None else "",
                form(priority.getMath()) if priority is not None else "",
                ev.getUseValuesFromTriggerTime(),
                tuple(
                    sorted(
                        (a.getVariable(), form(a.getMath())) for a in ev.getListOfEventAssignments()
                    )
                ),
            )
        )
    payload = repr(
        {
            "species": species,
            "reactions": reactions,
            "rules": sorted(
                (r.getElementName(), r.getVariable(), form(r.getMath()))
                for r in sm.getListOfRules()
            ),
            "initial_assignments": sorted(
                (ia.getSymbol(), form(ia.getMath())) for ia in sm.getListOfInitialAssignments()
            ),
            "events": sorted(events, key=repr),
            "estimated": estimated_ids(parameter_df),
        }
    )
    return sha16(payload)


def _coerce_estimate(value: Any) -> bool:
    try:
        return as_bool(value)
    except ValueError:
        raise ParameterError(f"Cannot interpret estimate value {value!r} as true/false.") from None


def update_parameters(problem, updates: list[dict[str, Any]]) -> dict[str, Any]:
    """Apply a batch of parameter-table edits, transactionally.

    Each update is a dict with an ``id`` (or ``parameterId``) plus any editable
    column (``nominalValue``, ``lowerBound``, ``upperBound``, ``estimate``,
    ``parameterScale``). Unknown ids, non-editable columns, bad bounds, or an invalid scale
    raise ``ParameterError`` and leave the table unchanged.
    """
    if not isinstance(updates, list):
        raise ParameterError("`updates` must be a list of {id, ...field} objects.")
    df = problem.parameter_df

    resolved: list[tuple[str, dict[str, Any]]] = []
    for upd in updates:
        pid = upd.get("id", upd.get(PARAMETER_ID))
        if pid is None:
            raise ParameterError("Each update must include an 'id'.")
        if pid not in df.index:
            from .reconcile import not_a_parameter

            why = not_a_parameter(problem, pid) or f"Known ids: {sorted(df.index)}"
            raise ParameterError(f"Unknown parameter id {pid!r}. {why}")
        fields: dict[str, Any] = {}
        for field, value in upd.items():
            if field in ("id", PARAMETER_ID):
                continue
            if field not in _EDITABLE_FIELDS:
                raise ParameterError(
                    f"Field {field!r} is not editable. Editable: {_EDITABLE_FIELDS}"
                )
            fields[field] = value
        resolved.append((pid, fields))

    touched = [pid for pid, _ in resolved]
    # Snapshot the whole (small) table so rollback also undoes any column an edit created.
    snapshot = df.copy(deep=True)

    try:
        for pid, fields in resolved:
            for col, value in fields.items():
                if col in _NUMERIC_FIELDS:
                    value = float(value)
                elif col == ESTIMATE:
                    value = _coerce_estimate(value)
                elif col == PARAMETER_SCALE:
                    value = str(value)
                    if value not in _SCALES:
                        raise ParameterError(
                            f"Invalid parameterScale {value!r}; use one of {sorted(_SCALES)}."
                        )
                df.loc[pid, col] = value
            lb = df.loc[pid, LOWER_BOUND] if LOWER_BOUND in df.columns else None
            ub = df.loc[pid, UPPER_BOUND] if UPPER_BOUND in df.columns else None
            # `>=`, not `>`: a ZERO-WIDTH box (lb == ub) is a real attempt to "pin" a value, and
            # it used to slip through here and reach pyPESTO, which raises `lb<ub not fulfilled`
            # — a message the compile/fit failure path then reads as an ILL-POSED ODE, sending
            # the agent off to rewrite a model that was fine.
            if lb is not None and ub is not None and float(lb) >= float(ub):
                raise ParameterError(
                    f"Parameter {pid!r}: lower bound {lb} is not below upper bound {ub}. A "
                    "zero-width box does not FIX a parameter — the row stays estimate=1 and the "
                    "fit then fails outright. Give a real interval that BRACKETS the value."
                )
    except ParameterError:
        problem.parameter_df = snapshot
        raise
    except Exception as exc:
        problem.parameter_df = snapshot
        raise ParameterError(f"Invalid parameter update: {exc}") from exc

    return {
        "updated": touched,
        "parameters": [_param_record(problem.parameter_df, pid) for pid in touched],
    }
