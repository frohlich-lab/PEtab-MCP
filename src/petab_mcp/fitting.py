"""Fit + simulate orchestration run automatically on every model commit.

The model-building loop is: *commit → fit → simulate → store a simulation dataframe →
statistical analysis reads that stored dataframe*. This module owns the middle two
steps. A commit (see :mod:`operations` and :meth:`session.commit_batch`) calls
:func:`fit_and_simulate` *before* it snapshots a version, so the single version it writes
carries the fitted parameters, the fit metrics, and the per-point simulation records.

Fitting uses pyPESTO (an AMICI-backed objective); a single :meth:`AmiciEngine.residuals`
call then materialises the simulation dataframe (measured vs. simulated, with raw and
noise-weighted residuals). AMICI and pyPESTO are hard dependencies and fitting is assumed
to succeed: a fit/simulation failure propagates (the caller reverts the commit), rather
than being swallowed into a degraded "committed but not fitted" state.
"""

from __future__ import annotations

import logging
import math
import time
from typing import Any

from petab.v1.C import NOMINAL_VALUE, PARAMETER_SCALE


def _fmt(x: Any) -> str:
    """Compact float formatting for log lines (``n/a`` for missing)."""
    return f"{x:.6g}" if isinstance(x, (int, float)) and not isinstance(x, bool) else "n/a"


def _run_fit(session, engine, n_starts: int, seed: int = 0) -> dict[str, Any]:
    """Estimate the free parameters with pyPESTO and write the best fit back.

    Writes the best start's (unscaled) values into ``nominalValue``. Returns the fit's
    ``neg_log_likelihood`` (pyPESTO ``fval``), the fitted values, and the objective at the
    pre-fit nominal values (for improvement tracking). No batch guard, no snapshot, no
    simulation — callers own those.
    """
    import numpy as np
    import pypesto.optimize
    import pypesto.petab
    from petab.v1.parameters import unscale

    from .engine import _ensure_swig_on_path

    _ensure_swig_on_path()
    np.random.seed(seed)  # best-effort determinism for start-point sampling

    # objective at the current nominal values, for improvement tracking
    objective_before = float(engine.evaluate_objective()["neg_log_likelihood"])

    h = session.sbml_hash()
    importer = pypesto.petab.PetabImporter(
        session.problem,
        output_folder=str(engine.build_root / f"pypesto_{h}"),
        model_name=f"petab_mcp_pp_{h}",
    )
    problem = importer.create_problem()

    log = session.logger
    n_free = len(getattr(problem, "x_free_indices", None) or problem.x_names)
    log.info(
        "fit: start n_starts=%d seed=%d n_free=%d sbml=%s objective_before=%s",
        n_starts, seed, n_free, h, _fmt(objective_before),
    )
    t0 = time.perf_counter()
    result = pypesto.optimize.minimize(problem, n_starts=n_starts, progress_bar=False)
    duration = time.perf_counter() - t0

    # pyPESTO's own multistart report -> pypesto.log (its native logging is otherwise
    # sparse for a successful minimize; warnings/errors it emits are captured there too).
    try:
        logging.getLogger("pypesto").info("multistart summary:\n%s", result.summary())
    except Exception:  # summary() API drift across pyPESTO versions must not fail a fit
        pass

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

    # Convergence summary: how many multistarts landed near the best objective.
    fvals = [
        float(s["fval"]) for s in result.optimize_result.list
        if s.get("fval") is not None and math.isfinite(float(s["fval"]))
    ]
    n_converged = sum(1 for v in fvals if v <= nllh + 1e-3)
    log.info(
        "fit: done best_nllh=%s self_improvement=%s time=%.2fs starts=%d converged~%d "
        "fval[min/max]=%s/%s fitted=%s",
        _fmt(nllh), _fmt(objective_before - nllh), duration, len(fvals), n_converged,
        _fmt(min(fvals) if fvals else None), _fmt(max(fvals) if fvals else None),
        {k: round(v, 6) for k, v in fitted.items()},
    )
    return {"fitted": fitted, "neg_log_likelihood": nllh, "objective_before": objective_before}


def fit_and_simulate(
    session, engine, *, do_fit: bool, n_starts: int, seed: int = 0
) -> dict[str, Any]:
    """Fit (optional) and simulate, storing the result on ``session.simulation``.

    ``do_fit`` runs the optimiser first (writing fitted ``nominalValue``s); either way a
    single :meth:`AmiciEngine.residuals` call builds the per-point records. The returned
    payload (also stored on ``session.simulation``) is the *simulation dataframe* the stats
    tools read. Fitting is assumed to succeed: a fit/simulation error propagates so the
    caller can revert the commit — it is not swallowed into a degraded result.

    ``status`` is ``"fit"`` (a fit ran) or ``"simulate_only"`` (``do_fit`` was false — an
    intentional simulate-only pass, e.g. ``n_starts=0`` or a rollback restore).
    """
    with session.lock:
        fit = _run_fit(session, engine, n_starts, seed) if do_fit else None
        sim = engine.residuals()
        records = sim["records"]

        payload = {
            "status": "fit" if do_fit else "simulate_only",
            "fitted_parameters": fit["fitted"] if fit else None,
            "neg_log_likelihood": sim["neg_log_likelihood"],
            "chi2": sim["chi2"],
            "n_data_points": sim["n_data_points"],
            "records": records,
            "objective_before": fit["objective_before"] if fit else None,
            "version": None,
        }
        session.simulation = payload
        session.logger.info(
            "fit_and_simulate: status=%s nllh=%s chi2=%s n_points=%s",
            payload["status"], _fmt(payload["neg_log_likelihood"]),
            _fmt(payload["chi2"]), payload["n_data_points"],
        )
        return payload


def ensure_simulation(session, engine) -> dict[str, Any] | None:
    """Populate ``session.simulation`` with a simulate-only pass if it is empty.

    Used at cold start (no commit yet) and after a failed fit, so the stats tools always
    have a dataframe to read without re-simulating on every call.
    """
    sim = getattr(session, "simulation", None)
    if sim is None or not sim.get("records"):
        fit_and_simulate(session, engine, do_fit=False, n_starts=0)
    return session.simulation


def compute_improvement(
    session, nllh: float | None, objective_before: float | None = None
) -> dict[str, Any]:
    """Improvement metrics for a fit, ranking it against all prior fitted versions.

    Scans the provenance log for the best (lowest) ``neg_log_likelihood`` recorded by any
    prior event — every committed version now carries one — so versions can be ranked by
    objective. Must be called *before* the caller appends the current event, so a fit never
    compares against itself.
    """
    provenance = getattr(session, "provenance", None)
    best_prior_nllh = None
    best_prior_version = None
    if provenance is not None:
        for ev in provenance.recent_events(0):  # full history
            v = ev.get("neg_log_likelihood")
            if v is not None:
                v = float(v)
                if best_prior_nllh is None or v < best_prior_nllh:
                    best_prior_nllh, best_prior_version = v, ev.get("version")

    self_improvement = None
    if objective_before is not None and nllh is not None:
        self_improvement = objective_before - nllh
    improvement_vs_best_prior = None
    if best_prior_nllh is not None and nllh is not None:
        improvement_vs_best_prior = best_prior_nllh - nllh
    is_new_best = best_prior_nllh is None or (nllh is not None and nllh < best_prior_nllh)

    return {
        "neg_log_likelihood": nllh,
        "objective_before": objective_before,
        "self_improvement": self_improvement,
        "best_prior_version": best_prior_version,
        "best_prior_neg_log_likelihood": best_prior_nllh,
        "improvement_vs_best_prior": improvement_vs_best_prior,
        "is_new_best": is_new_best,
    }


def build_sim_payload_for_version(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    """The clean simulation blob persisted alongside a version (``simulations.json``).

    Returns ``None`` when there is nothing worth storing (no per-point records), so old
    version dirs and failed/skipped fits simply carry no simulation file.
    """
    if not payload or not payload.get("records"):
        return None
    return {
        "status": payload.get("status"),
        "neg_log_likelihood": payload.get("neg_log_likelihood"),
        "chi2": payload.get("chi2"),
        "n_data_points": payload.get("n_data_points"),
        "fitted_parameters": payload.get("fitted_parameters"),
        "records": payload.get("records"),
    }
