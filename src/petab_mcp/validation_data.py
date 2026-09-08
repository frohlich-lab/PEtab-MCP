"""Score a model against HELD-OUT measurements — the predictive check.

NOT :mod:`petab_mcp.validation`, which lints a PEtab problem for structural errors.

The model in front of us was fitted to one set of measurements; :func:`score_problem` asks how well
it accounts for a set it never saw. The fitted parameters carry over, so the model is EVALUATED at
them (fujita's EGF pulses and ramps) — nothing is fitted here, and no inner solve either, which is
what makes the number comparable to the training nll rather than merely adjacent to it.

One implementation serves both sides: the candidate through the server tool, the reference through
``petab_harness.benchmark``, so no solver setting or chi2 convention can drift between the two
numbers that get subtracted. Report-only — the judge records it and never gates on it, and the
tool is absent from the panel's ``ALLOWED_TOOLS``. Rationale in ``docs/RATIONALE.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from petab.v1.C import (
    OBSERVABLE_ID,
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

#: ``mode`` a score carries — one kind so far; :mod:`petab_harness.runner.scoring` reads it to
#: refuse subtracting two scores of different kinds.
MODE_FIXED = "fixed"


@dataclass(frozen=True)
class HeldOut:
    """A held-out set as read from disk: the measurements, plus the condition table beside them if
    there is one.

    ``condition_df`` is carried but not yet used by :func:`score_problem`: every held-out condition
    must already be in the training table (fujita).
    """
    measurement_df: Any
    condition_df: Any = None

    @property
    def mode(self) -> str:
        return MODE_FIXED


#: File names next to the measurements. Written by ``petab_harness.benchmark.materialize``, read
#: back by :func:`read_held_out`; the two agree through these names and nowhere else.
HELD_OUT_CONDITIONS = "conditions.tsv"


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


def score_problem(problem, measurement_df, *, build_dir: str | Path,
                  model_name: str) -> dict[str, Any]:
    """Evaluate ``problem``'s model at ONE fixed parameter point against ``measurement_df``.

    ``problem`` supplies everything but the data, and its ``nominalValue``\\ s are the point —
    which is where every fit writes its optimum back. Nothing is optimised.

    ``build_dir``/``model_name`` name the pyPESTO output target — pass the pair the TRAINING
    problem was built with, so an identical model reuses its compiled module.

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
    mismatch = _table_mismatch(view, measurement_df)
    if mismatch:
        return {"status": STATUS_TABLE_MISMATCH, "reason": mismatch}
    # Dropped, not carried: petab's lint rejects a visualization table whose datasetIds are absent
    # from the measurement table, which is what swapping in held-out data does. Nothing here plots.
    view.visualization_df = None

    unsupported = _unsupported_reason(problem, view)
    if unsupported:
        return {"status": STATUS_FLATTENING_REQUIRED, "reason": unsupported}

    importer = pypesto.petab.PetabImporter(view, output_folder=str(build_dir),
                                          model_name=model_name)
    pp = importer.create_problem()
    _apply_amici_solver_defaults(pp)

    # `problem_x_from_nominal` owns the reduction convention and the scale question.
    reduced = problem_x_from_nominal(pp, problem.parameter_df)

    ret = pp.objective(np.asarray(reduced, dtype=float), sensi_orders=(0,), return_dict=True)
    nll = float(ret[FVAL])
    if not np.isfinite(nll):
        return {"status": STATUS_NON_FINITE,
                "reason": "the objective is not finite on the held-out conditions (the model "
                          "could not be integrated there, or its likelihood diverged)"}
    res = AmiciEngine._res_from_rdatas(ret[PP_RDATAS])
    chi2 = float(sum(float(v) ** 2 for v in res)) if res else None

    return {
        "status": STATUS_OK,
        "mode": MODE_FIXED,
        "nll": nll,
        "chi2": chi2,
        **_data_shape(measurement_df),
    }


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


def score_session(session, engine, held_out: HeldOut) -> dict[str, Any]:
    """The session's CURRENT model against ``held_out``.

    :func:`score_problem` at the parameter table's ``nominalValue``\\ s, which is where every fit
    writes its optimum back — so this scores exactly the model the judge is about to score on the
    training data.

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
        return score_problem(session.problem, held_out.measurement_df,
                             build_dir=kwargs["output_folder"], model_name=kwargs["model_name"])


def read_held_out(measurements_path: str | Path) -> HeldOut:
    """Load a held-out set from its measurement table and the optional tables beside it.

    ``measurements_path`` is the held-out measurement TSV;
    a ``conditions.tsv`` in the same directory is the set's own condition table
    (:data:`HELD_OUT_CONDITIONS`). Read with PEtab's own readers, so the server and the harness
    parse the files identically.
    """
    import petab.v1 as petab

    path = Path(measurements_path)
    cond = path.with_name(HELD_OUT_CONDITIONS)
    return HeldOut(
        measurement_df=read_measurements(path),
        condition_df=petab.get_condition_df(str(cond)) if cond.is_file() else None,
    )


def read_measurements(path: str | Path):
    """Load a held-out PEtab measurement table from ``path``.

    A thin, single-place reader so the server and the harness parse the file identically (PEtab's
    own TSV conventions, not ``pd.read_csv`` defaults spelled out twice).
    """
    import petab.v1 as petab

    return petab.get_measurement_df(str(path))
