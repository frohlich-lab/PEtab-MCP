"""Score a model against HELD-OUT measurements — the predictive check, in one of two modes.

NOT :mod:`petab_mcp.validation`, which lints a PEtab problem for structural errors.

* **fixed-θ** (:func:`score_problem`) — the same problem under conditions the training set does not
  cover, so the fitted parameters carry over and the model is EVALUATED at them (fujita's EGF
  pulses and ramps; lucarelli's other cell types, which bring their own rows of the training
  table's input columns). Nothing is fitted, and no inner solve either.
* **transfer** (:func:`refit_problem`) — the same MECHANISM on a different system, whose parameters
  are not the training ones (oliveira's Bahia). The fitted point means nothing there, so both the
  candidate and the reference structure are REFIT on the held-out data and compared by nll and by
  BIC over ``k`` estimated parameters and ``n`` held-out rows. A held-out set that brings its own
  parameter table IS a transfer set (:class:`HeldOut`).

One implementation serves both sides: the candidate through the server tool, the reference through
``petab_harness.benchmark``, so no solver setting or chi2 convention can drift between the two
numbers that get subtracted. Report-only — the judge records it and never gates on it, and the
tool is absent from the panel's ``ALLOWED_TOOLS``. Rationale in ``docs/RATIONALE.md``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from petab.v1.C import (
    ESTIMATE,
    LOWER_BOUND,
    NOMINAL_VALUE,
    OBSERVABLE_ID,
    PARAMETER_SCALE,
    UPPER_BOUND,
    OBSERVABLE_PARAMETERS,
    PREEQUILIBRATION_CONDITION_ID,
    SIMULATION_CONDITION_ID,
)

#: ``status`` values a score can come back with — named rather than folded into ``nll: None``,
#: because each is a distinct thing to see in a sweep:
#:
#: * ``ok`` — integrated on the held-out conditions and scored.
#: * ``no_data`` — no held-out measurements.
#: * ``table_mismatch`` — the table names an observable or condition the problem does not have.
#: * ``flattening_required`` — see :func:`_unsupported_reason`.
#: * ``non_finite`` — could not be integrated there, or the likelihood diverged. A result about
#:   the model, not an infrastructure failure.
STATUS_OK = "ok"
STATUS_NO_DATA = "no_data"
STATUS_TABLE_MISMATCH = "table_mismatch"
STATUS_FLATTENING_REQUIRED = "flattening_required"
STATUS_NON_FINITE = "non_finite"
#: * ``input_missing`` — transfer only: the held-out system changes a FIXED input the candidate has
#:   no fixed parameter to receive. A finding about the model, not a table error.
STATUS_INPUT_MISSING = "input_missing"

#: ``mode`` values a score carries — which of the two comparisons it is (module docstring);
#: :mod:`petab_harness.runner.scoring` reads it to refuse subtracting across kinds.
MODE_FIXED = "fixed"
MODE_TRANSFER = "transfer"


@dataclass(frozen=True)
class HeldOut:
    """A held-out set as read from disk: the measurements, plus the condition and parameter tables
    it brings with it, if any.

    ``parameter_df`` is what makes it a TRANSFER set — the statement that the parameter values are
    not the training ones, which is exactly when evaluating the fitted point is meaningless. No
    separate mode flag, because a flag and a table can disagree and the table is the fact.
    ``condition_df`` is the set's own condition table: for a transfer fit it REPLACES the training
    one; for a fixed-θ set it holds the rows the set ADDS — new values of the inputs the training
    table already has (:func:`_with_held_out_conditions`) — and ``None`` means every held-out
    condition is already in the training table (fujita).
    """
    measurement_df: Any
    condition_df: Any = None
    parameter_df: Any = None

    @property
    def mode(self) -> str:
        return MODE_TRANSFER if self.parameter_df is not None else MODE_FIXED


#: File names next to the measurements. Written by ``petab_harness.benchmark.materialize``, read
#: back by :func:`read_held_out`; the two agree through these names and nowhere else.
HELD_OUT_CONDITIONS = "conditions.tsv"
HELD_OUT_PARAMETERS = "parameters.tsv"


def _unsupported_reason(problem, view) -> str | None:
    """Why this pair of problems cannot be scored against each other, or ``None``.

    The one structural limit is FLATTENING: a training and a held-out table that both need
    ``petab.flatten_timepoint_specific_output_overrides`` generally flatten to different observable
    sets, so the compiled module does not describe both and the two likelihoods are expressed over
    different vocabularies. Refused loudly rather than rebuilt, because the comparability problem
    survives the rebuild.

    Unreachable via the shipped paths, because the harness refuses such an example when it
    assembles the held-out set (which is where the UNFLATTENED problem is still available), and
    kept as a cheap guard for any other caller.
    """
    from .fitting import needs_flattening

    for label, prob in (("the problem's own", problem), ("the held-out", view)):
        if needs_flattening(prob):
            return (f"{label} measurement table needs petab flattening, so the two sides would be "
                    "scored over different observable sets (see validation_data._unsupported_reason)")
    return None


def _table_mismatch(problem, measurement_df) -> str | None:
    """Names in the held-out table that the problem does not define, or ``None``.

    Every observable and condition a held-out set references must already exist in the shipped
    tables; one that does not means the table was built against a different problem, which would
    otherwise surface deep inside AMICI's parameter mapping.
    """
    known_obs = {str(i) for i in getattr(problem.observable_df, "index", ())}
    known_cond = {str(i) for i in getattr(problem.condition_df, "index", ())}
    problems = []
    # observableId and simulationConditionId are REQUIRED columns of a PEtab measurement table, so
    # they are read directly; preequilibrationConditionId is optional and guarded.
    unknown = sorted({str(v) for v in measurement_df[OBSERVABLE_ID]} - known_obs)
    if unknown:
        problems.append(f"observable(s) {unknown} are not in the observable table")
    seen = {str(v) for v in measurement_df[SIMULATION_CONDITION_ID].dropna()}
    unknown = sorted(seen - known_cond)
    if unknown:
        problems.append(f"{SIMULATION_CONDITION_ID} {unknown} are not in the condition table")
    if PREEQUILIBRATION_CONDITION_ID in measurement_df.columns:
        seen = {str(v) for v in measurement_df[PREEQUILIBRATION_CONDITION_ID].dropna()}
        unknown = sorted(seen - known_cond)
        if unknown:
            problems.append(
                f"{PREEQUILIBRATION_CONDITION_ID} {unknown} are not in the condition table")
    return "; ".join(problems) or None


def _with_held_out_conditions(view, condition_df) -> str | None:
    """Append a fixed-θ set's own condition rows to ``view``'s table, or say why they cannot be.

    A held-out row is a new VALUE of the inputs the training table already has (another cell
    type's Smad abundances, a knockdown level), which a model that takes those inputs predicts
    without a refit. A new COLUMN is a new input, i.e. a different model, and an id the training
    table already defines would redefine a condition the model was fitted under; both are
    refused. ``conditionName`` is free text and ignored. Shared by the server's scorer and the
    harness's setup check (``petab_harness.benchmark._curated_held_out``), so the two cannot
    disagree about what a fixed-θ set may bring.
    """
    import pandas as pd
    from petab.v1.C import CONDITION_NAME

    if condition_df is None:
        return None
    if view.condition_df is None:
        return ("the held-out set brings a condition table but the problem has none, so there is "
                "no column set to line the two up by")
    mine = view.condition_df.drop(columns=[CONDITION_NAME], errors="ignore")
    theirs = condition_df.drop(columns=[CONDITION_NAME], errors="ignore")
    if set(mine.columns) != set(theirs.columns):
        return (f"the held-out condition table has columns {sorted(map(str, theirs.columns))} "
                f"against the problem's {sorted(map(str, mine.columns))}; a condition column is a "
                "model input, so the two tables describe different models")
    redefined = sorted(set(theirs.index.astype(str)) & set(mine.index.astype(str)))
    if redefined:
        return (f"held-out condition(s) {redefined} are already in the problem's condition table; "
                "a fixed-θ set adds conditions, it does not redefine the ones the model was "
                "fitted under")
    view.condition_df = pd.concat([mine, theirs[list(mine.columns)]])
    return None


#: Bounds pyPESTO clips an analytically-solved scaling to (it honours the parameter table's
#: bounds, verified: a too-tight box returns the bound itself). 0 below, because a gel's scale
#: factor is non-negative — a model whose prediction ANTI-correlates with the blot gets s = 0, i.e.
#: "predicts nothing usable", not a sign flip that would flatter it — and effectively unbounded above.
HELD_OUT_SCALING_UPPER_BOUND = 1e12


def held_out_scalings(problem, measurement_df) -> "set[str]":
    """The ``observableParameters`` overrides in the held-out table that ``problem`` does NOT
    define — the per-blot scale factors of a held-out Western blot. Empty for fujita, whose
    held-out pulse/ramp rows re-use the shipped scalings (measured on the calibration's scale).

    A token counts when it is not numeric, not a parameter-table row and not a model entity;
    anything else is an ordinary PEtab override and left alone.

    Whether the observable is actually a scale of the model output is pyPESTO's to enforce:
    :func:`pypesto.hierarchical.petab.validate_hierarchical_petab_problem` refuses a formula that
    is not of the expected form for a ``scaling``, and refuses one in a noise formula outright.
    """
    import pandas as pd
    from petab.v1.C import OBSERVABLE_PARAMETERS, PARAMETER_SEPARATOR

    out: set[str] = set()
    if measurement_df is None or OBSERVABLE_PARAMETERS not in measurement_df.columns:
        return out
    known = {str(i) for i in getattr(problem.parameter_df, "index", ())}
    model = getattr(problem, "model", None)
    for cell in measurement_df[OBSERVABLE_PARAMETERS]:
        if pd.isna(cell):
            continue
        for tok in str(cell).split(PARAMETER_SEPARATOR):
            tok = tok.strip()
            if not tok or tok in known or _is_number(tok):
                continue
            if model is not None and model.has_entity_with_id(tok):
                continue
            out.add(tok)
    return out


def _is_number(token: str) -> bool:
    try:
        float(token)
    except ValueError:
        return False
    return True


def _without_extraneous(view):
    """``view``'s parameter table minus the rows petab's lint would call EXTRANEOUS.

    Swapping a held-out table in retires the overrides the training rows carried (raimundez's 79
    per-gel scalings), and petab refuses a parameter table with rows nothing references — the
    same lint that protects every other swap, so it is satisfied rather than disabled. Only rows
    the held-out problem cannot use are dropped; the kinetics stay, and a caller's vector is
    matched to what remains BY NAME (:func:`_reduced_from_named`).
    """
    if getattr(view, "model", None) is None:
        return view.parameter_df
    from petab.v1.parameters import get_valid_parameters_for_parameter_table

    valid = get_valid_parameters_for_parameter_table(
        view.model, view.condition_df, view.observable_df, view.measurement_df)
    return view.parameter_df.loc[[pid for pid in view.parameter_df.index if pid in valid]]


def _solve_held_out_scalings(view, scalings, *, build_dir, model_name, point) -> dict | None:
    """The held-out scalings at their analytical optimum for ``view``'s model at ``point``, or
    ``None`` when the objective is not finite there.

    pyPESTO's hierarchical importer does the work: the scalings are appended to the parameter
    table as estimated ``parameterType: scaling`` rows, everything the model already has keeps its
    value, and the inner solver returns the least-squares scale per row group — the same machinery
    :func:`petab_mcp.fitting._build_hierarchical_problem` uses for a fit. ``point`` maps a pyPESTO
    problem to the reduced vector its objective takes, so the outer parameters are evaluated at
    exactly the point the caller asked for.
    """
    import copy

    import numpy as np
    import pandas as pd
    import pypesto.petab
    from petab.v1.C import (ESTIMATE, LIN, LOWER_BOUND, NOMINAL_VALUE, PARAMETER_ID,
                            PARAMETER_SCALE, UPPER_BOUND)
    from pypesto.C import FVAL, INNER_PARAMETERS, PARAMETER_TYPE, InnerParameterType

    from .fitting import _apply_amici_solver_defaults

    hier = copy.copy(view)
    pdf = _without_extraneous(view).copy()
    pdf[PARAMETER_TYPE] = None
    rows = pd.DataFrame(
        {PARAMETER_SCALE: LIN, LOWER_BOUND: 0.0, UPPER_BOUND: HELD_OUT_SCALING_UPPER_BOUND,
         NOMINAL_VALUE: 1.0, ESTIMATE: 1, PARAMETER_TYPE: InnerParameterType.SCALING},
        index=pd.Index(sorted(scalings), name=PARAMETER_ID))
    hier.parameter_df = pd.concat([pdf, rows])
    pp = pypesto.petab.PetabImporter(hier, output_folder=str(build_dir), model_name=model_name,
                                     hierarchical=True).create_problem()
    _apply_amici_solver_defaults(pp)
    ret = pp.objective(np.asarray(point(pp, hier.parameter_df), dtype=float), sensi_orders=(0,),
                       return_dict=True)
    if not np.isfinite(float(ret[FVAL])):
        return None
    ids = pp.objective.calculator.get_interpretable_inner_par_ids()
    return {str(i): float(v) for i, v in zip(ids, ret[INNER_PARAMETERS])}


def _with_solved_scalings(cells, solved: dict):
    """``observableParameters`` cells with each solved scaling id replaced by its NUMBER, so the
    joint objective sees a plain numeric override and no parameter-table row is needed for it."""
    import pandas as pd
    from petab.v1.C import PARAMETER_SEPARATOR

    def one(cell):
        if pd.isna(cell):
            return cell
        toks = [t.strip() for t in str(cell).split(PARAMETER_SEPARATOR)]
        return PARAMETER_SEPARATOR.join(repr(solved[t]) if t in solved else t for t in toks)
    return cells.map(one)


def _reduced_from_named(pp, x_full, x_names) -> "list[float]":
    """``pp``'s REDUCED vector from a caller's FULL vector, matched by NAME.

    By name rather than by list equality because ``pp.x_names`` is not always the caller's list:
    a held-out view may carry FEWER parameter rows than the problem the caller fitted (petab
    refuses a parameter table with rows nothing references, and swapping a curated held-out table
    in retires the overrides the training rows carried), so ``pp.x_names`` can be a strict subset.
    Requiring the two lists to be equal would refuse such a call outright, while matching by name
    serves it and still refuses the thing worth refusing: a name ``pp`` needs that the caller did
    not supply, which is what a silently re-ordered vector looks like from here.

    Arity is checked FIRST and reported as its own failure. The name matching below would
    otherwise truncate a short vector to the names it happens to cover — and the caller would be
    told its NAMES were wrong when what it actually passed was a vector already reduced to free
    coordinates. Two different mistakes; only this one is about length.
    """
    if len(x_full) != len(x_names):
        raise ValueError(
            f"score_problem: x_full has {len(x_full)} value(s) for {len(x_names)} name(s); it "
            "must be the FULL vector (fixed parameters included), in x_names order")
    values = {str(n): float(v) for n, v in zip(x_names, x_full)}
    missing = [str(n) for n in pp.x_names if str(n) not in values]
    if missing:
        raise ValueError(
            "score_problem: the caller's x_names do not match the held-out problem's (it needs "
            f"{missing[:5]}{'...' if len(missing) > 5 else ''}); the vector would be evaluated "
            "permuted")
    full = [values[str(n)] for n in pp.x_names]
    free = getattr(pp, "x_free_indices", None)
    return [full[i] for i in free] if free is not None else full


def score_problem(problem, measurement_df, *, build_dir: str | Path, model_name: str,
                  x_full: "list[float] | None" = None,
                  x_names: "list[str] | None" = None,
                  condition_df=None) -> dict[str, Any]:
    """Evaluate ``problem``'s model at ONE fixed parameter point against ``measurement_df``.

    ``problem`` supplies everything but the data, and its ``nominalValue``\\ s are the point —
    which is where every fit writes its optimum back, unless ``x_full`` overrides. Nothing is
    optimised.

    ``build_dir``/``model_name`` name the pyPESTO output target — pass the pair the TRAINING
    problem was built with, so an identical model reuses its compiled module.

    ``x_full``/``x_names`` (both or neither) give the point explicitly, on each parameter's
    estimation scale in ``x_names`` order — for a caller holding a fitted vector that is NOT in the
    table, e.g. a reference multistart's winning start. Matched to this problem's own names rather
    than assumed aligned, and every name the problem needs must be supplied
    (:func:`_reduced_from_named`): a silent re-ordering or truncation would report a plausible,
    wrong likelihood.

    ``condition_df`` is the held-out set's own condition table — the rows it ADDS to the problem's
    (:class:`HeldOut`, :func:`_with_held_out_conditions`); ``None`` when every held-out condition
    is a training one.

    Returns ``{status, nll, n, chi2, n_conditions, observables}`` or ``{status, reason}``. ``chi2``
    is the sum of squares of AMICI's weighted residual vector (via
    :meth:`engine.AmiciEngine._res_from_rdatas`), ``None`` where AMICI did not populate it.
    """
    import copy

    import numpy as np
    import pypesto.petab
    from pypesto.C import FVAL
    from pypesto.C import RDATAS as PP_RDATAS

    from .engine import AmiciEngine
    from .fitting import _apply_amici_solver_defaults, problem_x_from_nominal

    if (x_full is None) != (x_names is None):
        raise ValueError("score_problem: pass both x_full and x_names, or neither")
    # Arity UP FRONT, before anything is built: it is a pure statement about the caller's two
    # arguments and needs no problem, and a curated set's scaling solve compiles a model before
    # the point is ever reduced. `_reduced_from_named` re-checks it for callers that reach it
    # directly.
    if x_full is not None and len(x_full) != len(x_names):
        raise ValueError(
            f"score_problem: x_full has {len(x_full)} value(s) for {len(x_names)} name(s); it "
            "must be the FULL vector (fixed parameters included), in x_names order")
    if measurement_df is None or len(measurement_df) == 0:
        return {"status": STATUS_NO_DATA, "reason": "the held-out measurement table is empty"}

    # Shallow copy, fresh tables: the model is shared (it is what we are scoring, and deep-copying
    # an SBML document per call is pure cost), but every dataframe pyPESTO reads is this call's own,
    # so nothing downstream can mutate the caller's session state.
    view = copy.copy(problem)
    view.measurement_df = measurement_df.copy()
    view.parameter_df = problem.parameter_df.copy()
    view.observable_df = problem.observable_df.copy()
    if problem.condition_df is not None:
        view.condition_df = problem.condition_df.copy()
    mismatch = _with_held_out_conditions(view, condition_df) or _table_mismatch(view, measurement_df)
    if mismatch:
        return {"status": STATUS_TABLE_MISMATCH, "reason": mismatch}
    # A curated held-out blot names its own scale factor, which no training measurement can
    # inform; those tokens, and only those, are solved below (see the module docstring).
    scalings = held_out_scalings(problem, measurement_df)
    # Dropped, not carried: petab's lint rejects a visualization table whose datasetIds are absent
    # from the measurement table, which is what swapping in held-out data does. Nothing here plots.
    view.visualization_df = None

    unsupported = _unsupported_reason(problem, view)
    if unsupported:
        return {"status": STATUS_FLATTENING_REQUIRED, "reason": unsupported}

    def _point(pp, parameter_df) -> "list[float]":
        """The REDUCED vector ``pp``'s objective takes, at the caller's point or the table's
        nominals — a closure because the scaling solve below needs the SAME point as the final
        evaluation, one pyPESTO problem apart."""
        if x_full is None:
            return problem_x_from_nominal(pp, parameter_df)
        return _reduced_from_named(pp, x_full, x_names)

    solved = None
    if scalings:
        # The blots' own scale factors, solved for THIS model at THIS point, then written into the
        # table as numbers — so the joint evaluation below is the same code path as for an example
        # that has none, and `nll`/`chi2` come off that one path either way.
        solved = _solve_held_out_scalings(view, scalings, build_dir=build_dir,
                                          model_name=model_name, point=_point)
        if solved is None:
            return {"status": STATUS_NON_FINITE,
                    "reason": "the objective is not finite on the held-out conditions while "
                              "solving the held-out scalings (the model could not be integrated "
                              "there, or its likelihood diverged)"}
        view.measurement_df[OBSERVABLE_PARAMETERS] = _with_solved_scalings(
            view.measurement_df[OBSERVABLE_PARAMETERS], solved)
        # Swapping the held-out table in retires the overrides the TRAINING rows carried, and petab
        # refuses a parameter table with rows nothing references. Only on this path: an example
        # without curated scalings re-uses the shipped ones, so nothing is retired.
        view.parameter_df = _without_extraneous(view)

    # AFTER the scalings branch, deliberately: it rewrites the measurement table and drops the
    # parameter rows the held-out data retires, and petab's lint — which `create_problem` runs —
    # rejects a table with rows nothing references. Building the importer first would fail on the
    # very rows `_without_extraneous` exists to remove.
    importer = pypesto.petab.PetabImporter(view, output_folder=str(build_dir),
                                          model_name=model_name)
    pp = importer.create_problem()
    _apply_amici_solver_defaults(pp)

    if x_full is None:
        # `problem_x_from_nominal` owns the reduction convention and the scale question. Read off
        # the VIEW's table, not the caller's problem: the two are the same today, but the view is
        # what `pp` was built from, so any path that adds or drops a parameter row on it (a curated
        # held-out set retires the training rows' overrides) would otherwise take the point from a
        # table that no longer describes this problem — and report a plausible, wrong likelihood.
        reduced = problem_x_from_nominal(pp, view.parameter_df)
    else:
        reduced = _reduced_from_named(pp, x_full, x_names)

    ret = pp.objective(np.asarray(reduced, dtype=float), sensi_orders=(0,), return_dict=True)
    nll = float(ret[FVAL])
    if not np.isfinite(nll):
        return {"status": STATUS_NON_FINITE,
                "reason": "the objective is not finite on the held-out conditions (the model "
                          "could not be integrated there, or its likelihood diverged)"}
    res = AmiciEngine._res_from_rdatas(ret[PP_RDATAS])
    chi2 = float(sum(float(v) ** 2 for v in res)) if res else None

    out = {
        "status": STATUS_OK,
        "mode": MODE_FIXED,
        "nll": nll,
        "chi2": chi2,
        **_data_shape(measurement_df),
    }
    if solved is not None:
        # The solved per-blot factors, recorded so a held-out score can be read back against the
        # scale it was computed on. Absent for an example with no curated scalings.
        out["held_out_scalings"] = solved
    return out


def _data_shape(measurement_df) -> dict[str, Any]:
    """``{n, n_conditions, observables}`` of a held-out table — the same three fields on both
    modes' payloads, so a consumer reads them from one place."""
    return {
        "n": int(len(measurement_df)),
        "n_conditions": int(measurement_df[SIMULATION_CONDITION_ID].nunique())
        if SIMULATION_CONDITION_ID in measurement_df.columns else None,
        "observables": sorted({str(v) for v in measurement_df[OBSERVABLE_ID]})
        if OBSERVABLE_ID in measurement_df.columns else [],
    }


def _reject_untransferable_table(held_out_df, published_df, view, *,
                                 system_differs: bool = False) -> None:
    """Validate a transfer source against the FULL published training parameter table at setup.

    Refuse a new estimated mechanism parameter the published problem cannot use — new
    observation-model parameters and unused fixed rows are permitted — and a table identical to the
    published one in every value, bound, scale and estimate flag, unless ``system_differs`` says
    the set's condition table already makes it another system (new values of the published inputs
    under the published parameters: lucarelli's cell types). This check cannot run against the
    zero-shot shipped subset, which omits kinetics, or the candidate, whose rates may have different
    names. Candidate refits retain their own estimated values instead.
    """
    from . import reconcile as _reconcile

    manageable = {str(sym) for sym in _reconcile.observation_model_symbols(view)}
    unmanageable = sorted(
        str(pid) for pid in held_out_df.index
        if pid not in published_df.index and str(pid) not in manageable
        and int(float(held_out_df.loc[pid, ESTIMATE])) == 1)
    if unmanageable:
        raise ValueError(
            f"the held-out parameter table estimates parameter(s) {', '.join(unmanageable)} "
            "that the training problem does not have, and that are not observation-model "
            "parameters of the held-out "
            "tables; the published model has no counterpart for them, so the refit cannot manage "
            "them")
    if system_differs or list(held_out_df.index) != list(published_df.index):
        return
    for pid in held_out_df.index:
        if int(float(held_out_df.loc[pid, ESTIMATE])) != int(float(published_df.loc[pid, ESTIMATE])):
            return
        if PARAMETER_SCALE in held_out_df.columns and (str(held_out_df.loc[pid, PARAMETER_SCALE])
                                                       != str(published_df.loc[pid, PARAMETER_SCALE])):
            return
        for col in (NOMINAL_VALUE, LOWER_BOUND, UPPER_BOUND):
            if col not in held_out_df.columns:
                continue
            a, b = float(held_out_df.loc[pid, col]), float(published_df.loc[pid, col])
            if math.isnan(a) and math.isnan(b):
                continue
            if not math.isclose(a, b, rel_tol=1e-12):
                return
    raise ValueError(
        "the held-out parameter table is identical to the training one, so nothing would transfer; "
        "a held-out set ships a parameter table only when it describes a DIFFERENT system — drop "
        "the table to score at the fitted point instead")


def transfer_view(problem, held_out: HeldOut, shipped_parameter_df):
    """The candidate's problem moved onto the held-out system as a new view, or the status saying
    why it cannot be: ``(view, None)`` or ``(None, {status, reason})``.

    The view shares the model and owns every table it changes, so the session's problem is never
    touched. Three tables move:

    * the held-out MEASUREMENTS replace the training ones;
    * the held-out CONDITION table replaces the training one, and must have its column set — a
      column is a model input, so a new one is a different model;
    * the FIXED INPUTS that CHANGE between the systems are carried into the candidate's parameter
      table by name — a row fixed in both the held-out and the shipped training table
      (``shipped_parameter_df``) with a different value there, e.g. oliveira's ``population``.

    Every other held-out row is left alone, and the held-out table's ESTIMATED values are never
    used as the candidate's start. A required input the candidate has no fixed row to receive is
    :data:`STATUS_INPUT_MISSING` — a result about the model, not a plumbing failure.
    """
    import copy

    view = copy.copy(problem)
    view.measurement_df = held_out.measurement_df.copy()
    view.parameter_df = problem.parameter_df.copy()
    view.visualization_df = None
    if held_out.condition_df is not None:
        if problem.condition_df is None:
            return None, {"status": STATUS_TABLE_MISMATCH,
                          "reason": "the held-out set brings a condition table but the problem has "
                                    "none, so there is no column set to line the two models up by"}
        mine = list(problem.condition_df.columns)
        theirs = list(held_out.condition_df.columns)
        if sorted(mine) != sorted(theirs):
            return None, {"status": STATUS_TABLE_MISMATCH,
                          "reason": f"the held-out condition table's columns {theirs} differ from "
                                    f"the problem's {mine}; a condition column is a model input, so "
                                    "the two tables describe different models"}
        view.condition_df = held_out.condition_df.copy()
    mismatch = _table_mismatch(view, view.measurement_df)
    if mismatch:
        return None, {"status": STATUS_TABLE_MISMATCH, "reason": mismatch}

    if held_out.parameter_df is not None:
        pdf, shipped = view.parameter_df, shipped_parameter_df
        missing = []
        for pid, row in held_out.parameter_df.iterrows():
            if int(float(row[ESTIMATE])) != 0 or pid not in shipped.index:
                continue
            if int(float(shipped.loc[pid, ESTIMATE])) != 0:
                continue
            value = float(row[NOMINAL_VALUE])
            if math.isclose(value, float(shipped.loc[pid, NOMINAL_VALUE]), rel_tol=1e-12):
                continue
            if pid not in pdf.index or int(float(pdf.loc[pid, ESTIMATE])) != 0:
                missing.append(str(pid))
                continue
            pdf.loc[pid, NOMINAL_VALUE] = value
        if missing:
            return None, {"status": STATUS_INPUT_MISSING,
                          "reason": f"the held-out system changes the fixed input(s) {', '.join(missing)}, "
                                    "which the model does not carry as fixed parameters of that "
                                    "name, so it cannot be moved to the held-out system"}
    return view, None


def refit_problem(problem, *, build_dir: str | Path, model_name: str,
                  n_starts: int) -> dict[str, Any]:
    """REFIT ``problem`` on its own measurements and report the optimum — the transfer score.

    ``problem`` is already the held-out problem, and its ``nominalValue``\\ s are the START point
    (the candidate's training optimum via :func:`transfer_view`; the collection's published values
    for the reference). Every ``estimate = 1`` row is refit and ``k`` counts exactly those, so it is
    the same ``k`` the training score uses. Nothing is written back.

    One implementation, both sides, on the server's own fitting path — Fides on AMICI
    sensitivities, start-0 seeded at the nominal point, whose objective is also taken as a
    candidate optimum (a start whose gradient fails at iteration 0 would otherwise lose a good
    point). The per-start cap and the worker count are NOT arguments: both sides run on the
    server's own budget (``PETAB_MCP_FIT_PER_START_TIMEOUT``, ``PETAB_MCP_FIT_N_WORKERS``), the same
    one the fit that produced the candidate ran on, so neither side can be given a wider search
    than the other by its caller.

    Returns ``{status, mode, nll, k, n, chi2, n_conditions, observables, n_starts, n_converged,
    fitted}`` or ``{status, reason}`` for :data:`STATUS_NO_DATA`,
    :data:`STATUS_TABLE_MISMATCH`, :data:`STATUS_FLATTENING_REQUIRED` or
    :data:`STATUS_NON_FINITE`.
    """
    import copy
    import logging

    import numpy as np
    import pypesto.optimize
    import pypesto.petab
    from pypesto.C import FVAL
    from pypesto.C import RDATAS as PP_RDATAS

    from .engine import AmiciEngine, _ensure_swig_on_path
    from .fitting import (FIT_SEED, _apply_amici_solver_defaults, _fides_maxtime_options,
                          _fit_engine, _fitted_values, problem_x_full_from_nominal)

    measurement_df = problem.measurement_df
    if measurement_df is None or len(measurement_df) == 0:
        return {"status": STATUS_NO_DATA, "reason": "the held-out measurement table is empty"}
    mismatch = _table_mismatch(problem, measurement_df)
    if mismatch:
        return {"status": STATUS_TABLE_MISMATCH, "reason": mismatch}
    view = copy.copy(problem)
    view.parameter_df = problem.parameter_df.copy()
    view.visualization_df = None
    unsupported = _unsupported_reason(view, view)
    if unsupported:
        return {"status": STATUS_FLATTENING_REQUIRED, "reason": unsupported}

    _ensure_swig_on_path()
    np.random.seed(FIT_SEED)
    importer = pypesto.petab.PetabImporter(view, output_folder=str(build_dir),
                                          model_name=model_name)
    pp = importer.create_problem()
    _apply_amici_solver_defaults(pp)
    x_start = np.asarray(problem_x_full_from_nominal(pp, view.parameter_df), dtype=float)
    pp.set_x_guesses(x_start[None, :])
    optimizer = pypesto.optimize.FidesOptimizer(
        verbose=logging.ERROR, options=_fides_maxtime_options())
    result = pypesto.optimize.minimize(
        pp, optimizer=optimizer, n_starts=int(n_starts),
        engine=_fit_engine(int(n_starts)), progress_bar=False)

    finite = [r for r in result.optimize_result.list
              if r.get("x") is not None and r.get("fval") is not None
              and math.isfinite(float(r["fval"]))]
    fvals = [float(r["fval"]) for r in finite]
    best = min(finite, key=lambda r: float(r["fval"])) if finite else None
    nll = float(best["fval"]) if best is not None else math.inf
    x_best = None if best is None else np.asarray(best["x"], dtype=float)
    x_red_start = pp.get_reduced_vector(x_start)
    fval_start = float(pp.objective(x_red_start))
    if math.isfinite(fval_start) and fval_start < nll:
        nll, x_best = fval_start, x_start
    if not math.isfinite(nll):
        return {"status": STATUS_NON_FINITE,
                "reason": "no multistart produced a finite optimum on the held-out system and the "
                          "start point has none either (the model could not be integrated there, "
                          "or its likelihood diverged)"}
    ret = pp.objective(pp.get_reduced_vector(x_best), sensi_orders=(0,), return_dict=True)
    res = AmiciEngine._res_from_rdatas(ret[PP_RDATAS])
    k = int(sum(int(float(v)) == 1 for v in view.parameter_df[ESTIMATE]))
    return {
        "status": STATUS_OK,
        "mode": MODE_TRANSFER,
        "nll": float(ret[FVAL]),
        "k": k,
        "chi2": float(sum(float(v) ** 2 for v in res)) if res else None,
        **_data_shape(measurement_df),
        "n_starts": int(n_starts),
        "n_converged": int(sum(1 for v in fvals if v <= nll + 1e-3)),
        "fitted": _fitted_values(pp.x_names, x_best, getattr(pp, "x_free_indices", None),
                                 view.parameter_df),
    }


def score_session(session, engine, held_out: HeldOut,
                  n_starts: int | None = None) -> dict[str, Any]:
    """The session's CURRENT model against ``held_out``, in the set's own mode.

    Fixed-θ: :func:`score_problem` at the parameter table's ``nominalValue``\\ s, which is where
    every fit writes its optimum back — so this scores exactly the model the judge is about to score
    on the training data. Transfer: :func:`transfer_view` then :func:`refit_problem`, with the
    session's own default multistart when ``n_starts`` is ``None``.

    Held under ``session.lock`` so the model and the table are one snapshot. The build target is
    ``session.sbml_hash()`` — the model STRUCTURE only, not ``problem_hash``, which also covers the
    parameter table. That is the right key because it is the key the AMICI compile cache itself
    uses: a module is a function of the SBML plus the free/fixed SPLIT, and a fixed-θ score leaves
    the split alone (it evaluates at the training table's nominals). So this reuses whatever module
    was compiled for the model and pays a
    build only for a model nobody built (a restored version re-renders its SBML in a different
    order) — bounded, once per run, after the agent has finished.
    """
    with session.lock:
        kwargs = engine.pypesto_importer_kwargs(session.sbml_hash())
        build = {"build_dir": kwargs["output_folder"], "model_name": kwargs["model_name"]}
        if held_out.mode == MODE_FIXED:
            return score_problem(session.problem, held_out.measurement_df,
                                 condition_df=held_out.condition_df, **build)
        view, failed = transfer_view(session.problem, held_out, session.shipped_parameter_df)
        if failed:
            return {**failed, "mode": MODE_TRANSFER}
        starts = session.auto_fit_starts if n_starts is None else int(n_starts)
        return refit_problem(view, n_starts=starts, **build)


def read_held_out(measurements_path: str | Path) -> HeldOut:
    """Load a held-out set from its measurement table and the optional tables beside it.

    ``measurements_path`` is the held-out measurement TSV; a ``conditions.tsv`` /
    ``parameters.tsv`` in the same directory are the set's own condition and parameter
    tables (:data:`HELD_OUT_CONDITIONS`, :data:`HELD_OUT_PARAMETERS`). Read with PEtab's
    own readers, so the server and the harness parse the files identically.
    """
    import petab.v1 as petab

    path = Path(measurements_path)
    cond = path.with_name(HELD_OUT_CONDITIONS)
    pars = path.with_name(HELD_OUT_PARAMETERS)
    return HeldOut(
        measurement_df=read_measurements(path),
        condition_df=petab.get_condition_df(str(cond)) if cond.is_file() else None,
        parameter_df=petab.get_parameter_df(str(pars)) if pars.is_file() else None,
    )


def read_measurements(path: str | Path):
    """Load a held-out PEtab measurement table from ``path``.

    A thin, single-place reader so the server and the harness parse the file identically (PEtab's
    own TSV conventions, not ``pd.read_csv`` defaults spelled out twice).
    """
    import petab.v1 as petab

    return petab.get_measurement_df(str(path))
