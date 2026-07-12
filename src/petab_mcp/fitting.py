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

import hashlib
import json
import logging
import math
import time
from typing import Any

from petab.v1.C import ESTIMATE, NOMINAL_VALUE, PARAMETER_SCALE

from .parameters import parameter_df_to_tsv


def _fmt(x: Any) -> str:
    """Compact float formatting for log lines (``n/a`` for missing)."""
    return f"{x:.6g}" if isinstance(x, (int, float)) and not isinstance(x, bool) else "n/a"


def _flt(v: Any) -> float | None:
    """Coerce to a finite float, or ``None`` (missing / NaN / Inf)."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if (math.isnan(f) or math.isinf(f)) else f


def _int(v: Any) -> int | None:
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def fit_fingerprint(
    sbml_str: str, parameter_df, n_starts: int, seed: int = 0, optimizer: str = "fides"
) -> str:
    """Identity of an *estimation problem* — the key under which a fit is cached/reused.

    A multistart fit's result is determined by the model, the estimation setup (bounds,
    scale, which parameters are estimated, and the fixed parameters' values), the number of
    starts, the seed, and the optimizer — but NOT by an estimated parameter's ``nominalValue``
    (that is a starting guess the multistart optimizer does not use). Those starting values
    are therefore blanked out so the key is stable across the fit (which overwrites them with
    the optimum), letting a later commit of the same estimation problem reuse the result.
    """
    df = parameter_df
    if ESTIMATE in df.columns and NOMINAL_VALUE in df.columns:
        df = df.copy()
        try:
            estimated = df[ESTIMATE].fillna(0).astype(float) != 0
            df.loc[estimated, NOMINAL_VALUE] = float("nan")
        except (TypeError, ValueError):  # non-numeric estimate column -> hash it as-is
            pass
    payload = "\n".join(
        [sbml_str, parameter_df_to_tsv(df), f"n_starts={n_starts};seed={seed};optimizer={optimizer}"]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _fit_cache_path(session, fitkey: str):
    return session.provenance.root / "cache" / f"fit_{fitkey}.json"


def _load_fit_cache(session, fitkey: str) -> dict[str, Any] | None:
    """Return a cached fit for this estimation problem, or ``None``. Never raises."""
    try:
        path = _fit_cache_path(session, fitkey)
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 - a corrupt/unreadable cache entry just forces a re-fit
        session.logger.warning("could not read fit cache %s", fitkey, exc_info=True)
    return None


def _save_fit_cache(session, fitkey: str, entry: dict[str, Any]) -> None:
    """Persist a fit under its estimation-problem key. Best-effort; never raises."""
    try:
        path = _fit_cache_path(session, fitkey)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"fitkey": fitkey, **entry}, indent=2), encoding="utf-8")
    except Exception:  # noqa: BLE001 - failing to cache must never fail a fit
        session.logger.warning("could not write fit cache %s", fitkey, exc_info=True)


def _apply_fitted(session, fitted: dict[str, float]) -> None:
    """Write cached fitted (linear-scale) values into ``nominalValue`` — the fit's effect."""
    df = session.problem.parameter_df
    with session.lock:
        for name, value in (fitted or {}).items():
            if name in df.index:
                try:
                    df.loc[name, NOMINAL_VALUE] = float(value)
                except (TypeError, ValueError):
                    pass


def _write_pypesto_result(session, result, problem_hash: str) -> str | None:
    """Write the pyPESTO ``Result`` to a per-fit HDF5 file; return its path (or ``None``).

    The file is cached under ``<provenance>/cache/`` keyed by the *problem* hash — model
    SBML **and** parameter table, so fits that differ only in bounds/scale don't collide —
    and is copied into a version directory when that fit is snapshotted. Best-effort: the
    objective (an AMICI model) is not serialised (``problem=False``), and any storage
    failure is logged and swallowed so it never fails a fit; the JSON traces remain the
    primary, always-present artifact.
    """
    try:
        import pypesto.store as pypesto_store

        cache_dir = session.provenance.root / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        path = cache_dir / f"pypesto_result_{problem_hash}.hdf5"
        try:
            pypesto_store.write_result(result, str(path), overwrite=True, problem=False)
        except TypeError:  # older signature without the `problem` kwarg
            pypesto_store.write_result(result, str(path), overwrite=True)
        return str(path)
    except Exception:  # noqa: BLE001 - HDF5/pyPESTO storage is a best-effort interop artifact
        session.logger.warning("could not write pyPESTO result file", exc_info=True)
        return None


def _optimization_payload(result) -> dict[str, Any]:
    """Per-start results + per-iteration fval/time traces from a pyPESTO result.

    Returns ``{"n_starts", "starts", "trace"}`` for :mod:`data_access` to expose as the
    ``optimization_starts`` / ``optimization_trace`` SQL tables. Best-effort and defensive:
    pyPESTO's result/history API varies across versions and a start can lack a recorded
    trace, so anything unreadable is omitted rather than failing the fit.

    ``start`` is the rank in the (objective-sorted) result list — ``0`` is the best.
    """
    def _get(obj, key):
        try:
            return obj[key]
        except Exception:  # noqa: BLE001 - OptimizerResult is dict-like but tolerant access
            return getattr(obj, key, None)

    try:
        olist = list(result.optimize_result.list)
    except Exception:  # noqa: BLE001 - shape drift across pyPESTO versions
        olist = []

    starts: list[dict[str, Any]] = []
    trace: list[dict[str, Any]] = []
    for rank, r in enumerate(olist):
        rid = _get(r, "id")
        msg = _get(r, "message")
        starts.append({
            "start": rank,
            "id": str(rid) if rid is not None else None,
            "fval": _flt(_get(r, "fval")),
            "time": _flt(_get(r, "time")),
            "n_fval": _int(_get(r, "n_fval")),
            "n_grad": _int(_get(r, "n_grad")),
            "n_hess": _int(_get(r, "n_hess")),
            "exitflag": _int(_get(r, "exitflag")),
            "message": str(msg) if msg is not None else None,
        })
        history = _get(r, "history")
        if history is None:
            continue
        try:
            fvals = list(history.get_fval_trace())
        except Exception:  # noqa: BLE001 - trace not recorded / API drift
            continue
        try:
            times = list(history.get_time_trace())
        except Exception:  # noqa: BLE001
            times = []
        for j, v in enumerate(fvals):
            trace.append({
                "start": rank,
                "iteration": j,
                "fval": _flt(v),
                "time": _flt(times[j]) if j < len(times) else None,
            })
    return {"n_starts": len(starts), "starts": starts, "trace": trace}


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

    log = session.logger
    # objective at the current nominal values, for improvement tracking
    objective_before = float(engine.evaluate_objective()["neg_log_likelihood"])

    # Reuse a previous fit of this exact estimation problem instead of re-optimizing: the
    # cache is keyed on the problem identity (model + bounds/scale/estimate + fixed values +
    # n_starts + seed), so a hit is a genuinely equivalent fit. We still simulate below, so
    # only the (expensive) multistart optimization is skipped.
    fitkey = fit_fingerprint(session.model_sbml(), session.problem.parameter_df, n_starts, seed)
    if getattr(session, "fit_cache", True):
        cached = _load_fit_cache(session, fitkey)
        if cached is not None:
            _apply_fitted(session, cached.get("fitted") or {})
            session.last_fit_cache_hit = True
            opt = {"n_starts": cached.get("n_starts"), "starts": [], "trace": [],
                   **(cached.get("optimization") or {}),
                   "result_file": cached.get("result_file")}
            log.info("fit: cache hit fitkey=%s nllh=%s (optimizer skipped)",
                     fitkey, _fmt(cached.get("neg_log_likelihood")))
            return {"fitted": cached.get("fitted") or {},
                    "neg_log_likelihood": cached.get("neg_log_likelihood"),
                    "objective_before": objective_before, "optimization": opt}
    session.last_fit_cache_hit = False

    h = session.sbml_hash()
    importer = pypesto.petab.PetabImporter(
        session.problem,
        output_folder=str(engine.build_root / f"pypesto_{h}"),
        model_name=f"petab_mcp_pp_{h}",
    )
    problem = importer.create_problem()

    free = getattr(problem, "x_free_indices", None)
    n_free = len(free) if free is not None else len(problem.x_names)
    log.info(
        "fit: start optimizer=fides n_starts=%d seed=%d n_free=%d sbml=%s objective_before=%s",
        n_starts, seed, n_free, h, _fmt(objective_before),
    )
    # Fides: a Newton trust-region optimizer that exploits AMICI's sensitivities; more
    # robust than the default L-BFGS-B on stiff ODE likelihoods. Its per-iteration logging
    # is routed to pypesto.log (see logging_setup.configure_pypesto).
    optimizer = pypesto.optimize.FidesOptimizer(verbose=logging.INFO)
    # Record the per-iteration optimization history in memory (no trace files) so the
    # traces can be exposed as SQL tables. Older pyPESTO without HistoryOptions just runs
    # without a trace.
    minimize_kwargs: dict[str, Any] = {}
    try:
        minimize_kwargs["history_options"] = pypesto.HistoryOptions(trace_record=True)
    except Exception:  # noqa: BLE001 - HistoryOptions absent/renamed in this pyPESTO
        pass
    t0 = time.perf_counter()
    result = pypesto.optimize.minimize(
        problem, optimizer=optimizer, n_starts=n_starts, progress_bar=False, **minimize_kwargs
    )
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
    optimization = _optimization_payload(result)
    optimization["result_file"] = _write_pypesto_result(session, result, session.problem_hash())
    # Cache this fit under the estimation-problem key so an identical future commit can reuse
    # it instead of re-optimizing (see `fit_fingerprint` / the cache check above).
    if getattr(session, "fit_cache", True):
        _save_fit_cache(session, fitkey, {
            "n_starts": n_starts,
            "fitted": fitted,
            "neg_log_likelihood": nllh,
            "optimization": {k: optimization.get(k) for k in ("n_starts", "starts", "trace")},
            "result_file": optimization.get("result_file"),
        })
    return {"fitted": fitted, "neg_log_likelihood": nllh, "objective_before": objective_before,
            "optimization": optimization}


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
    intentional simulate-only pass, e.g. ``n_starts=0`` or a rollback restore). When a fit
    runs, its pyPESTO per-start results and per-iteration traces are stored on
    ``session.optimization`` (``None`` on a simulate-only pass).
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
        # Optimization traces/results are a live diagnostic of the *last* fit (not persisted
        # per version): set them when a fit ran, clear them on a simulate-only pass.
        session.optimization = fit["optimization"] if fit else None
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


def fit_result_block(
    fitsim: dict[str, Any] | None, improvement: dict[str, Any] | None = None
) -> dict[str, Any]:
    """The fit fields every commit path returns (single op, batch, rollback, reset).

    Keeps the tool-result shape identical across commit kinds so an agent reads the same
    keys everywhere. ``fitsim`` is a :func:`fit_and_simulate` payload (or the restored
    simulation for rollback/reset); ``improvement`` is the :func:`compute_improvement` dict.
    """
    fitsim = fitsim or {}
    return {
        "fit_status": fitsim.get("status"),
        "neg_log_likelihood": fitsim.get("neg_log_likelihood"),
        "chi2": fitsim.get("chi2"),
        "fitted_parameters": fitsim.get("fitted_parameters"),
        "improvement": improvement or None,
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


def build_optimization_payload_for_version(
    payload: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """The optimization blob persisted alongside a version (``optimization.json`` + hdf5).

    ``payload`` is ``session.optimization`` (a :func:`_optimization_payload` dict, possibly
    with a ``result_file``). Returns ``None`` when there is no fit to store (simulate-only
    passes, legacy versions), so those version dirs carry no optimization files. The
    ``result_file`` — the pyPESTO HDF5 written for this model — is carried through so the
    snapshot can copy it into the version directory.
    """
    if not payload or not payload.get("starts"):
        return None
    return {
        "n_starts": payload.get("n_starts"),
        "starts": payload.get("starts"),
        "trace": payload.get("trace"),
        "result_file": payload.get("result_file"),
    }
