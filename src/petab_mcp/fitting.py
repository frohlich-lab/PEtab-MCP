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
        xvec = _get(r, "x")
        try:
            xlist = [float(v) for v in xvec] if xvec is not None else None
        except Exception:  # noqa: BLE001 - x shape drift; spread just becomes unavailable
            xlist = None
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
            # Full scaled parameter vector at this start's optimum — the raw material for the
            # multistart-spread identifiability signal (get_identifiability). Order matches the
            # payload's x_names (set by the caller from problem.x_names).
            "x": xlist,
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
    # pyPESTO builds its OWN AMICI model (a second compile of the same structure, under a
    # distinct module name). Charge it to the compile bucket like the engine's build, but —
    # mirroring the engine's `_models` guard — only the FIRST time this structure is built:
    # a re-fit of the same SBML (e.g. a parameter-bound edit) reuses the compiled module, so
    # counting it again would over-report compiles. Runs under `session.lock` (the caller
    # holds it), so both the accumulator writes and the `_pypesto_hashes` update are safe.
    _first_pypesto_build = h not in engine._pypesto_hashes
    _compile_t0 = time.perf_counter()
    importer = pypesto.petab.PetabImporter(
        session.problem,
        output_folder=str(engine.build_root / f"pypesto_{h}"),
        model_name=f"petab_mcp_pp_{h}",
    )
    problem = importer.create_problem()
    if _first_pypesto_build:
        session.compile_seconds += time.perf_counter() - _compile_t0
        session.n_compiles += 1
        engine._pypesto_hashes.add(h)

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
    # The multistart optimization is the recurring "fit" cost (a fit-cache hit returns above
    # without reaching here, so it correctly adds nothing). Under `session.lock`.
    session.fit_seconds += duration
    session.n_fits += 1

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
    optimization["x_names"] = list(problem.x_names)  # column order for each start's "x"
    optimization["result_file"] = _write_pypesto_result(session, result, session.problem_hash())
    # Cache this fit under the estimation-problem key so an identical future commit can reuse
    # it instead of re-optimizing (see `fit_fingerprint` / the cache check above).
    if getattr(session, "fit_cache", True):
        _save_fit_cache(session, fitkey, {
            "n_starts": n_starts,
            "fitted": fitted,
            "neg_log_likelihood": nllh,
            "optimization": {k: optimization.get(k)
                             for k in ("n_starts", "starts", "trace", "x_names")},
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
            # fitted_parameters are always LINEAR (unscaled) values. parameter_scales gives each
            # one's ESTIMATION scale (lin/log10), co-located so a reader never mistakes a linear
            # value for its log10 coordinate — e.g. a Hill coefficient fitted to 15.3 (linear) is
            # 15.3, NOT 10**15.3. (An analyst once made exactly that error.)
            "fitted_parameters": fit["fitted"] if fit else None,
            "parameter_scales": (
                {pid: str(session.problem.parameter_df.loc[pid, PARAMETER_SCALE])
                 for pid in fit["fitted"]}
                if fit else None),
            # The estimated-parameter count k, so the goodness-of-fit metrics (bic/reduced_chi2)
            # are available even on a simulate-only pass where `fitted_parameters` is None.
            "n_estimated": int(session.problem.n_estimated),
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


def goodness_of_fit(
    nll: float | None, chi2: float | None, k: int | None, n: int | None
) -> dict[str, Any]:
    """Complexity-penalised fit scores, so a commit is judged on more than raw nll.

    The agent otherwise sees only the raw nll/chi2 and cannot tell "already at the noise
    floor" from "keep going". These derived scores make that explicit:

    * ``bic = k*ln(n) + 2*nll`` and ``aic = 2*k + 2*nll`` (lower is better) charge for every
      free parameter, so an edit that adds a parameter for a marginal nll drop scores WORSE.
    * ``reduced_chi2 = chi2 / (n - k)`` is ≈ 1 when the model fits to the noise floor.
    * ``chi2_gof_pvalue`` is the upper-tail chi2 p-value on ``chi2`` with ``n - k`` degrees
      of freedom: the fit is adequate (not rejected) when this is not tiny (e.g. ≳ 0.05).

    ``k`` is the number of estimated parameters, ``n`` the number of data points. ``bic`` and
    ``aic`` are ``None`` only when ``nll``/``k``/``n`` are missing; the chi2-based fields
    (``chi2_dof``, ``reduced_chi2``, ``chi2_gof_pvalue``) are additionally ``None`` when the
    degrees of freedom ``n - k`` are non-positive (an under-determined fit) or ``chi2`` is
    missing.
    """
    import math

    out: dict[str, Any] = {
        "bic": None, "aic": None, "chi2_dof": None,
        "reduced_chi2": None, "chi2_gof_pvalue": None,
    }
    if nll is None or k is None or n is None or n <= 0:
        return out
    out["bic"] = k * math.log(n) + 2.0 * nll
    out["aic"] = 2.0 * k + 2.0 * nll
    dof = n - k
    if dof > 0:
        out["chi2_dof"] = dof
        if chi2 is not None:
            out["reduced_chi2"] = chi2 / dof
            try:
                from scipy.stats import chi2 as _chi2_dist

                out["chi2_gof_pvalue"] = float(_chi2_dist.sf(chi2, dof))
            except Exception:  # pragma: no cover - scipy always present via pypesto
                pass
    return out


def fit_result_block(
    fitsim: dict[str, Any] | None, improvement: dict[str, Any] | None = None
) -> dict[str, Any]:
    """The fit fields every commit path returns (single op, batch, rollback, reset).

    Keeps the tool-result shape identical across commit kinds so an agent reads the same
    keys everywhere. ``fitsim`` is a :func:`fit_and_simulate` payload (or the restored
    simulation for rollback/reset); ``improvement`` is the :func:`compute_improvement` dict.
    Beyond the raw nll/chi2 it also carries the complexity-penalised
    :func:`goodness_of_fit` scores (bic/aic/reduced_chi2/gof p-value) so the agent can see
    when the fit is already at the noise floor and stop adding parameters.
    """
    fitsim = fitsim or {}
    fitted = fitsim.get("fitted_parameters")
    nll = fitsim.get("neg_log_likelihood")
    chi2 = fitsim.get("chi2")
    n = fitsim.get("n_data_points")
    # k = number of estimated parameters. Prefer len(fitted_parameters) after a fit; fall back
    # to the payload's n_estimated on simulate-only / rollback / reset paths (fitted is None
    # there), so the goodness-of-fit metrics are present on every commit, not just fitted ones.
    k = len(fitted) if fitted is not None else fitsim.get("n_estimated")
    block = {
        "fit_status": fitsim.get("status"),
        "neg_log_likelihood": nll,
        "chi2": chi2,
        "n_data_points": n,
        # Surface the estimated-parameter count k next to bic so the parameter penalty in
        # bic = k*ln(n) + 2*nll is legible — adding a free parameter must earn its keep.
        "n_estimated": k,
        # fitted_parameters: LINEAR (unscaled) values; parameter_scales: each one's estimation
        # scale (lin/log10). Kept side by side so a linear value is never misread as a log10 one.
        "fitted_parameters": fitted,
        "parameter_scales": fitsim.get("parameter_scales"),
        "improvement": improvement or None,
    }
    block.update(goodness_of_fit(nll, chi2, k, n))
    return block


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


# --------------------------------------------------------------------------- identifiability
# Which estimated parameters are superfluous? The lead over-parameterises because raw nll/bic
# don't say WHICH parameters are unconstrained or trading off. get_identifiability answers that
# from two cheap, complementary signals at the fitted optimum:
#   * the Fisher Information Matrix (FIM ≈ Hessian of the nllh, from AMICI sensitivities) ->
#     per-parameter standard error, marginal + partial correlations, and — via its
#     eigendecomposition — the non-identifiable parameter COMBINATIONS (flat directions);
#   * the multistart spread: how much each parameter wanders across near-optimal starts (a
#     model-free practical-identifiability check that needs no Hessian).
# The FIM is a LOCAL linearization and can understate non-identifiability for strongly nonlinear
# models, so the two are meant to be read together. All output is bounded (top-N pairs/directions).


def _identifiability_stats(
    params_meta: list[dict[str, Any]],
    fim,
    spread: dict[str, Any],
    *,
    top_n: int = 8,
    corr_threshold: float = 0.9,
    flat_rel: float = 1e-3,
    se_unconstrained: float = 1.0,
    spread_unstable: float = 0.5,
) -> dict[str, Any]:
    """Pure statistics core (no pyPESTO/AMICI). ``params_meta`` is one dict per estimated
    parameter — ``{id, scale, estimate, estimate_scaled, at_bound}`` — in the SAME order as the
    rows/cols of ``fim`` (an ``m×m`` numpy array or ``None``). ``spread`` maps parameter id ->
    ``{std,min,max,n}`` from the multistart. Returns the compact, bounded diagnosis dict."""
    import numpy as np

    names = [p["id"] for p in params_meta]
    m = len(names)
    per = {p["id"]: {
        "id": p["id"], "scale": p.get("scale"), "estimate": p.get("estimate"),
        "at_bound": bool(p.get("at_bound")), "se_scaled": None, "fold_uncertainty": None,
        "multistart_std": None, "multistart_range": None,
    } for p in params_meta}

    # multistart spread (always available if the payload carried per-start x)
    for pid, s in (spread or {}).items():
        if pid in per and isinstance(s, dict):
            per[pid]["multistart_std"] = s.get("std")
            if s.get("min") is not None and s.get("max") is not None:
                per[pid]["multistart_range"] = round(float(s["max"]) - float(s["min"]), 4)

    correlations: list[dict[str, Any]] = []
    partials: list[dict[str, Any]] = []
    flat_directions: list[dict[str, Any]] = []
    fim_info = None

    if fim is not None and m >= 1:
        H = np.asarray(fim, dtype=float)
        cov = np.linalg.pinv(H)
        var = np.clip(np.diag(cov), 0.0, None)
        se = np.sqrt(var)
        for i, pid in enumerate(names):
            per[pid]["se_scaled"] = round(float(se[i]), 4)
            sc = (params_meta[i].get("scale") or "").lower()
            if np.isfinite(se[i]):
                # a legible multiplicative uncertainty for log-scaled params
                if sc in ("log10", "log"):
                    base = 10.0 if sc == "log10" else math.e
                    per[pid]["fold_uncertainty"] = round(float(base ** se[i]), 3)
        # marginal correlations from the covariance
        dvar = np.sqrt(np.outer(var, var))
        with np.errstate(divide="ignore", invalid="ignore"):
            marg = np.where(dvar > 0, cov / dvar, 0.0)
        # partial correlations from the FIM (precision matrix) itself
        dh = np.sqrt(np.outer(np.abs(np.diag(H)), np.abs(np.diag(H))))
        with np.errstate(divide="ignore", invalid="ignore"):
            pcorr = np.where(dh > 0, -H / dh, 0.0)
        for i in range(m):
            for j in range(i + 1, m):
                if abs(marg[i, j]) >= corr_threshold:
                    correlations.append({"a": names[i], "b": names[j], "r": round(float(marg[i, j]), 3)})
                if abs(pcorr[i, j]) >= corr_threshold:
                    partials.append({"a": names[i], "b": names[j], "pr": round(float(pcorr[i, j]), 3)})
        correlations.sort(key=lambda d: -abs(d["r"]))
        partials.sort(key=lambda d: -abs(d["pr"]))
        correlations = correlations[:top_n]
        partials = partials[:top_n]
        # flat directions: eigenvectors with near-zero eigenvalue = non-identifiable combinations
        evals, evecs = np.linalg.eigh(H)
        emax = float(evals.max()) if evals.size else 0.0
        for idx in range(len(evals)):
            ev = float(evals[idx])
            rel = ev / emax if emax > 0 else 0.0
            if emax > 0 and rel < flat_rel:
                vec = evecs[:, idx]
                order = np.argsort(-np.abs(vec))
                participants = [{"id": names[k], "weight": round(float(vec[k]), 3)}
                                for k in order[:4] if abs(vec[k]) > 0.15]
                flat_directions.append({"eigenvalue": round(ev, 6), "rel": round(rel, 6),
                                        "participants": participants})
        rank = int(np.linalg.matrix_rank(H))
        cond = float(emax / max(float(evals[evals > 0].min()) if (evals > 0).any() else emax, 1e-300))
        fim_info = {"rank": rank, "n_estimated": m, "condition": round(cond, 1),
                    "rank_deficient": rank < m}

    # simplification candidates: rank estimated params by identifiability red flags
    corr_partner: dict[str, list] = {}
    for c in correlations:
        corr_partner.setdefault(c["a"], []).append((c["b"], c["r"]))
        corr_partner.setdefault(c["b"], []).append((c["a"], c["r"]))
    flat_members = {p["id"] for d in flat_directions for p in d["participants"]}
    candidates = []
    for pid in names:
        p = per[pid]
        reasons = []
        if p["at_bound"]:
            reasons.append("at_bound")
        if p["se_scaled"] is not None and p["se_scaled"] >= se_unconstrained:
            reasons.append(f"poorly_constrained(se={p['se_scaled']})")
        if pid in corr_partner:
            b, r = max(corr_partner[pid], key=lambda t: abs(t[1]))
            reasons.append(f"collinear_with:{b}(r={r})")
        if pid in flat_members:
            reasons.append("flat_direction")
        if p["multistart_std"] is not None and p["multistart_std"] >= spread_unstable:
            reasons.append(f"multistart_unstable(std={round(p['multistart_std'],3)})")
        if reasons:
            candidates.append({"id": pid, "n_flags": len(reasons), "reasons": reasons})
    candidates.sort(key=lambda d: -d["n_flags"])

    method = ("fim+multistart" if fim is not None and spread else
              "fim" if fim is not None else
              "multistart" if spread else "none")
    return {
        "available": method != "none",
        "method": method,
        "n_estimated": m,
        "fim": fim_info,
        "parameters": [per[n] for n in names],
        "correlations": correlations,
        "partial_correlations": partials,
        "flat_directions": flat_directions[:top_n],
        "candidates": candidates[:12],
        "notes": ("FIM is a LOCAL linearization of uncertainty — read se/correlations together "
                  "with multistart_std (a model-free spread) and flat_directions (the actual "
                  "non-identifiable parameter combinations). High |r|≈1, large fold_uncertainty, "
                  "at_bound, or a flat direction all mark a parameter as a simplification "
                  "candidate: fix it, remove its term, or merge a collinear pair into one "
                  "parameter (e.g. a ratio/product)."),
    }


def _compute_fim(session, engine):
    """FIM (≈ Hessian of the nllh) at the current fitted point, in free-scaled parameter space.

    Returns ``(fim_ndarray, free_names)`` or ``None`` (best-effort — a missing FIM just drops
    the correlation half of the diagnosis, leaving the multistart spread)."""
    try:
        import numpy as np
        import pypesto.petab
        from petab.v1.parameters import scale

        from .engine import _ensure_swig_on_path
        _ensure_swig_on_path()
        h = session.sbml_hash()
        importer = pypesto.petab.PetabImporter(
            session.problem,
            output_folder=str(engine.build_root / f"pypesto_{h}"),
            model_name=f"petab_mcp_pp_{h}",
        )
        problem = importer.create_problem()
        df = session.problem.parameter_df
        x_full = []
        for name in problem.x_names:
            if name in df.index:
                x_full.append(float(scale(float(df.loc[name, NOMINAL_VALUE]),
                                          df.loc[name, PARAMETER_SCALE])))
            else:
                x_full.append(0.0)
        free_idx = list(problem.x_free_indices)
        free_names = [problem.x_names[i] for i in free_idx]
        x_free = np.asarray(problem.get_reduced_vector(np.asarray(x_full, dtype=float)), dtype=float)
        mfree = len(free_names)

        def _finite_square(a):
            a = np.asarray(a, dtype=float)
            return a.shape == (mfree, mfree) and np.isfinite(a).all()

        # Preferred: the objective's Hessian (AMICI returns the FIM approximation for a
        # Gaussian LSQ). Fallback: build the FIM from residual sensitivities sres^T sres.
        try:
            ret = problem.objective(x_free, sensi_orders=(0, 1, 2), return_dict=True)
            for key in ("hess", "HESS"):
                if key in ret and _finite_square(ret[key]):
                    return np.asarray(ret[key], dtype=float), free_names
        except Exception:  # noqa: BLE001 - order-2 unsupported for this objective; try sres
            pass
        try:
            ret = problem.objective(x_free, sensi_orders=(0, 1), return_dict=True)
            for key in ("sres", "SRES"):
                if key in ret and ret[key] is not None:
                    sres = np.asarray(ret[key], dtype=float)
                    fim = sres.T @ sres
                    if _finite_square(fim):
                        return fim, free_names
        except Exception:  # noqa: BLE001
            pass
        return None
    except Exception:  # noqa: BLE001 - importer/model build failure; degrade to spread-only
        return None


def _multistart_spread(session, free_names: list[str]) -> dict[str, Any]:
    """Per-parameter spread across near-optimal multistarts (scaled space), from the stored
    optimization payload's per-start ``x``. Empty when x/x_names were not recorded."""
    import numpy as np

    opt = getattr(session, "optimization", None) or {}
    starts = opt.get("starts") or []
    xnames = opt.get("x_names") or []
    if not starts or not xnames:
        return {}
    col = {n: i for i, n in enumerate(xnames)}
    fvals = [float(s["fval"]) for s in starts
             if s.get("fval") is not None and math.isfinite(float(s["fval"]))]
    if not fvals:
        return {}
    best = min(fvals)
    # near-optimal band; fall back to the best few starts if the band is too tight
    band = [s for s in starts if s.get("fval") is not None and s.get("x") is not None
            and math.isfinite(float(s["fval"])) and float(s["fval"]) <= best + 1.0]
    if len(band) < 3:
        band = sorted([s for s in starts if s.get("x") is not None and s.get("fval") is not None],
                      key=lambda s: float(s["fval"]))[:5]
    out: dict[str, Any] = {}
    for n in free_names:
        j = col.get(n)
        if j is None:
            continue
        vals = [float(s["x"][j]) for s in band if s.get("x") is not None and j < len(s["x"])]
        if len(vals) >= 2:
            arr = np.asarray(vals)
            out[n] = {"std": float(arr.std()), "min": float(arr.min()),
                      "max": float(arr.max()), "n": len(vals)}
    return out


def identifiability(session, engine=None, *, top_n: int = 8, corr_threshold: float = 0.9) -> dict[str, Any]:
    """Parameter-identifiability diagnosis of the current fit — which estimated parameters are
    unconstrained or trade off (the basis for model simplification). Combines the FIM (SE +
    marginal/partial correlations + flat directions) with the multistart spread. Bounded output.
    """
    from petab.v1.C import LOWER_BOUND, UPPER_BOUND
    from petab.v1.parameters import scale

    df = session.problem.parameter_df
    engine = engine or getattr(session, "engine", None)

    # FIM first — its free-parameter order is the canonical order for everything else.
    fim = free_names = None
    if engine is not None:
        got = _compute_fim(session, engine)
        if got is not None:
            fim, free_names = got
    if free_names is None:  # no FIM: use the estimated params in table order
        free_names = [str(i) for i in df.index
                      if int(df.loc[i, ESTIMATE]) == 1] if ESTIMATE in df.columns else []

    def _scaled(name, col):
        try:
            return float(scale(float(df.loc[name, col]), df.loc[name, PARAMETER_SCALE]))
        except Exception:  # noqa: BLE001 - missing bound/value
            return None

    params_meta = []
    for name in free_names:
        est_s = _scaled(name, NOMINAL_VALUE) if name in df.index else None
        lb = _scaled(name, LOWER_BOUND) if (name in df.index and LOWER_BOUND in df.columns) else None
        ub = _scaled(name, UPPER_BOUND) if (name in df.index and UPPER_BOUND in df.columns) else None
        at_bound = False
        if est_s is not None and lb is not None and ub is not None and ub > lb:
            tol = 1e-3 * (ub - lb)
            at_bound = (est_s - lb) <= tol or (ub - est_s) <= tol
        params_meta.append({
            "id": name,
            "scale": (df.loc[name, PARAMETER_SCALE] if name in df.index else None),
            "estimate": (float(df.loc[name, NOMINAL_VALUE]) if name in df.index else None),
            "estimate_scaled": est_s,
            "at_bound": at_bound,
        })

    spread = _multistart_spread(session, free_names)
    return _identifiability_stats(params_meta, fim, spread,
                                  top_n=top_n, corr_threshold=corr_threshold)
