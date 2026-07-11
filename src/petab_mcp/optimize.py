"""Parameter estimation with pyPESTO (PEtab v1).

pyPESTO consumes a ``petab.v1.Problem`` directly: it builds an AMICI-backed objective
and runs multi-start optimization. The best start's parameter vector comes back on the
optimization scale (per ``parameterScale``), so we unscale before writing it into the
parameter table's ``nominalValue``. Requires the ``[optimize]`` extra (pyPESTO), which
also pulls in AMICI.
"""

from __future__ import annotations

from typing import Any

from petab.v1.C import NOMINAL_VALUE, PARAMETER_SCALE

from .errors import MissingDependencyError


def optimize(session, engine, n_starts: int = 10, seed: int = 0) -> dict[str, Any]:
    """Estimate the free parameters; writes the best fit back into the table."""
    if getattr(session, "batch_active", False):
        from .errors import BatchError

        raise BatchError("Cannot optimize while a batch is open; commit_batch or abort_batch first.")
    try:
        import numpy as np
        import pypesto.optimize
        import pypesto.petab
        from petab.v1.parameters import unscale
    except Exception as exc:  # pyPESTO (or a dep) missing
        raise MissingDependencyError("Optimization", "pypesto", "optimize") from exc

    from .engine import _ensure_swig_on_path

    _ensure_swig_on_path()
    np.random.seed(seed)  # best-effort determinism for start-point sampling

    # objective at the current nominal values, for improvement tracking
    objective_before = None
    try:
        objective_before = float(engine.evaluate_objective()["neg_log_likelihood"])
    except Exception:  # engine unavailable / model not simulatable yet
        objective_before = None

    h = session.sbml_hash()
    importer = pypesto.petab.PetabImporter(
        session.problem,
        output_folder=str(engine.build_root / f"pypesto_{h}"),
        model_name=f"petab_mcp_pp_{h}",
    )
    problem = importer.create_problem()
    result = pypesto.optimize.minimize(problem, n_starts=n_starts, progress_bar=False)

    best = result.optimize_result.list[0]
    x = [float(v) for v in best["x"]]

    df = session.problem.parameter_df
    fitted: dict[str, float] = {}
    nllh = float(best["fval"])
    with session.lock:
        for name, value in zip(problem.x_names, x):
            if name in df.index:
                linear = float(unscale(value, df.loc[name, PARAMETER_SCALE]))
                df.loc[name, NOMINAL_VALUE] = linear
                fitted[name] = linear
        training = _record_training(session, n_starts, nllh, fitted, objective_before)

    return {
        "backend": "pypesto",
        "n_starts": n_starts,
        "neg_log_likelihood": nllh,
        "fitted_parameters": fitted,
        "version": training.get("version"),
        "improvement": training,
    }


def _record_training(session, n_starts, nllh, fitted, objective_before) -> dict[str, Any]:
    """Snapshot a training version and log a training event with improvement metrics.

    Tracks the fit's own improvement (``self_improvement``) and how this trained model
    compares to the best previously trained version (``improvement_vs_best_prior`` /
    ``is_new_best``), so versions can be ranked by objective.
    """
    provenance = getattr(session, "provenance", None)
    if provenance is None:
        return {"version": None}
    from .parameters import parameters_to_records

    # best objective among prior training runs (lower nllh = better fit)
    best_prior_nllh = None
    best_prior_version = None
    for ev in provenance.recent_events(0):  # full history
        if ev.get("type") == "training" and ev.get("neg_log_likelihood") is not None:
            v = float(ev["neg_log_likelihood"])
            if best_prior_nllh is None or v < best_prior_nllh:
                best_prior_nllh, best_prior_version = v, ev.get("version")

    improvement = {
        "neg_log_likelihood": nllh,
        "objective_before": objective_before,
        "self_improvement": (objective_before - nllh) if objective_before is not None else None,
        "best_prior_version": best_prior_version,
        "best_prior_neg_log_likelihood": best_prior_nllh,
        "improvement_vs_best_prior": (best_prior_nllh - nllh) if best_prior_nllh is not None else None,
        "is_new_best": best_prior_nllh is None or nllh < best_prior_nllh,
    }

    version = provenance.snapshot_version(
        model_ant=session.model.to_antimony(),
        parameters=parameters_to_records(session.problem),
        label="optimize",
        kind="training",
        meta={"n_starts": n_starts, **improvement},
    )
    provenance.append_event(
        {
            "type": "training",
            "method": "pypesto",
            "n_starts": n_starts,
            "fitted_parameters": fitted,
            "version": version,
            **improvement,
        }
    )
    return {"version": version, **improvement}
