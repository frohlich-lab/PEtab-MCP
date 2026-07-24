"""Fit + simulate orchestration run by the explicit ``fit`` tool.

The model-building loop is: *edit → fit → simulate → store a simulation dataframe →
statistical analysis reads that stored dataframe*. This module owns the middle two
steps. The ``fit`` tool (see :meth:`session.PetabSession.fit`) calls :func:`fit_and_simulate`
*before* it snapshots a version, so the single version it writes carries the fitted parameters,
the fit metrics, and the per-point simulation records. Edits (see :mod:`operations`) are
fit-free and do not touch this module (issue #81).

Fitting uses pyPESTO (an AMICI-backed objective); a single :meth:`AmiciEngine.residuals`
call then materialises the simulation dataframe (measured vs. simulated, with raw and
noise-weighted residuals). AMICI and pyPESTO are hard dependencies and fitting is assumed
to succeed: a fit/simulation failure propagates (the caller reverts the commit), rather
than being swallowed into a degraded "committed but not fitted" state.
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
from typing import Any

from petab.v1.C import (
    ESTIMATE,
    LIN,
    LOG,
    LOG10,
    LOWER_BOUND,
    NOMINAL_VALUE,
    PARAMETER_SCALE,
    UPPER_BOUND,
)

from . import reconcile as _reconcile
from .constants import (
    NEW_BEST_MIN_DELTA_BIC,
    NOISE_SIGMA_COLLAPSE_FRACTION,
    PARAMETER_AT_BOUND_RELATIVE_TOL,
)
from .formatting import finite_float as _flt
from .parameters import estimated_mask, parameter_df_to_tsv, sha16

# pyPESTO's parameter-table column that tags an inner (hierarchical) parameter.
_PARAMETER_TYPE_COL = "parameterType"

def _env_float(name: str, default: float) -> float:
    """A finite, strictly-positive float from ``name``, or ``default`` when the env var is unset,
    non-numeric, or not a finite positive number — a bad override can't NaN/Inf/<=0 the cap or
    crash import."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        val = float(raw)
    except (TypeError, ValueError):
        return default
    return val if math.isfinite(val) and val > 0 else default


def _env_int(name: str, default: int) -> int:
    """A positive int (>=1) from ``name``, or ``default`` for an unset/non-numeric/<1 override."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        val = int(raw)
    except (TypeError, ValueError):
        return default
    return val if val >= 1 else default


# Per-start wall-clock cap for a single Fides optimization start (seconds; env
# PETAB_MCP_FIT_PER_START_TIMEOUT, default 60). Fides defaults MAXTIME to inf, so one stiff start
# can run unbounded and out-idle the MCP transport, which the harness then mistakes for a hang.
# Bounding each start makes a full multistart fit's worst-case duration finite and predictable,
# which the harness sizes its sse_read_timeout + stall watchdog to (see examples/_harness/
# fit_budget.py). The harness exports its own resolved values onto this server's environment, so
# both sides use the SAME cap + worker count. Kept generous — Fides returns best-so-far for a
# capped start, so too tight a cap would silently degrade fit quality.
FIT_PER_START_TIMEOUT_S = _env_float("PETAB_MCP_FIT_PER_START_TIMEOUT", 60.0)


def _default_fit_workers() -> int:
    """Default multistart worker count when ``PETAB_MCP_FIT_N_WORKERS`` is unset.

    Prefer the CPUs a batch scheduler actually granted (``SLURM_CPUS_PER_TASK``) over
    ``os.cpu_count()``: the latter reports the whole machine and ignores a cgroup/cpuset limit,
    so under Slurm — which pins the process to N cores but leaves ``os.cpu_count()`` reporting
    the node's full count — a ``box minus one core`` default would massively oversubscribe the
    allocation (e.g. ~255 workers pinned into a 4-core grant). Off a scheduler, fall back to
    ``box minus one core`` as before.
    """
    slurm = _env_int("SLURM_CPUS_PER_TASK", 0)
    if slurm >= 1:
        return slurm
    return max(1, (os.cpu_count() or 1) - 1)


# Max pyPESTO worker processes for the multistart (env PETAB_MCP_FIT_N_WORKERS; default =
# SLURM_CPUS_PER_TASK when set, else box minus one core — see _default_fit_workers). The batch
# runner dispatches examples one at a time, so a fit has the whole machine; running the starts in
# parallel cuts wall-clock to ~ceil(n_starts / workers) waves. Set to 1 to force sequential
# fitting. examples/_harness/fit_budget.py keeps the SAME default (its _default_workers mirrors
# _default_fit_workers) and, when the harness drives a run, also exports its resolved value as
# PETAB_MCP_FIT_N_WORKERS onto this server — so the harness's budget matches the real parallelism
# whether the value comes from the shared default or the export.
FIT_N_WORKERS = _env_int("PETAB_MCP_FIT_N_WORKERS", _default_fit_workers())


def _fides_maxtime_options() -> dict[str, Any]:
    """``{fides.Options.MAXTIME: FIT_PER_START_TIMEOUT_S}`` when this ``fides`` exposes ``MAXTIME``,
    else ``{}`` (older/absent fides just runs uncapped)."""
    try:
        import fides
        return {fides.Options.MAXTIME: FIT_PER_START_TIMEOUT_S}
    except Exception:  # noqa: BLE001 - no fides / no MAXTIME on this version: run uncapped
        return {}


def _fit_engine(n_starts: int):
    """A pyPESTO engine that runs the multistart across up to ``FIT_N_WORKERS`` worker processes
    (never more than there are starts), or ``None`` (pyPESTO's sequential ``SingleCoreEngine``) for
    a single effective worker or when a parallel engine can't be constructed."""
    workers = max(1, min(FIT_N_WORKERS, int(n_starts)))
    if workers < 2:
        return None
    try:
        from pypesto.engine import MultiProcessEngine
        return MultiProcessEngine(n_procs=workers)
    except Exception:  # noqa: BLE001 - parallel engine unavailable: fall back to sequential
        return None


def _fmt(x: Any) -> str:
    """Compact float formatting for log lines (``n/a`` for missing)."""
    return f"{x:.6g}" if isinstance(x, (int, float)) and not isinstance(x, bool) else "n/a"


def _int(v: Any) -> int | None:
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _fitted_values(x_names, x_values, free_indices, parameter_df) -> dict[str, float]:
    """Map a pyPESTO optimizer result (a FULL parameter vector) to
    ``{estimated parameter id: linear value}``.

    pyPESTO's ``x_names`` and a result's ``x`` span *every* parameter, fixed ones included;
    only ``free_indices`` are actually estimated. Restricting to those (and unscaling each to
    linear) is what keeps ``fitted_parameters`` — and therefore the parameter count ``k`` that
    drives bic/aic/reduced_chi2 — the ESTIMATED parameters only, matching ``problem.n_estimated``.
    Recording the fixed ones too would overcount ``k`` and corrupt every complexity-penalised
    score. ``free_indices`` of ``None`` falls back to every index (an all-free problem).
    """
    from petab.v1.parameters import unscale

    idx = list(free_indices) if free_indices is not None else list(range(len(x_names)))
    fitted: dict[str, float] = {}
    for i in idx:
        name = x_names[i]
        if name in parameter_df.index:
            fitted[name] = float(unscale(float(x_values[i]), parameter_df.loc[name, PARAMETER_SCALE]))
    return fitted


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
            estimated = estimated_mask(df)
            df.loc[estimated, NOMINAL_VALUE] = float("nan")
        except (TypeError, ValueError):  # non-numeric estimate column -> hash it as-is
            pass
    payload = "\n".join(
        [sbml_str, parameter_df_to_tsv(df), f"n_starts={n_starts};seed={seed};optimizer={optimizer}"]
    )
    return sha16(payload)


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


def _build_hierarchical_problem(session, importer_kwargs: dict[str, Any]):
    """Best-effort hierarchical pyPESTO problem for ``session``'s current problem.

    Returns ``(pypesto_problem, inner_scales)`` where ``inner_scales`` maps each inner
    parameter id to the estimation scale it was tagged on (so the fit can un-scale the
    analytically-solved value back to linear), or ``(None, {})`` when the problem has no
    cleanly-separable inner parameters (see :func:`reconcile.classify_inner_parameters`) or
    pyPESTO rejects the hierarchical setup.

    The whole win is optimization efficiency/robustness: the inner parameters (noise sigmas,
    observable-scaling factors) are solved analytically per objective evaluation instead of by
    the outer multistart, shrinking the outer search (e.g. a benchmark's 13 estimated params →
    5 outer). It does NOT change what the parameters converge to — a sigma's analytic optimum is
    the same RMS the joint optimizer finds — so the fit's nllh/identity is unchanged.

    Tagging is applied to a COPY of the parameter table handed to pyPESTO; the session's own
    table is left untouched (the agent keeps seeing ordinary parameters, ``n_estimated``/``k``
    stays correct because inner parameters are still ``estimate=1`` there, and the fit-cache key
    is unaffected). On ANY failure this returns ``(None, {})`` so the caller falls back to the
    ordinary joint fit — hierarchical never risks a fit that would otherwise succeed.
    """
    inner = _reconcile.classify_inner_parameters(session.problem)
    if not inner:
        return None, {}
    try:
        import copy

        import pypesto.petab
        from pypesto.hierarchical.petab import correct_parameter_df_bounds

        df = session.problem.parameter_df.copy(deep=True)
        df[_PARAMETER_TYPE_COL] = None
        inner_scales: dict[str, str] = {}
        for pid, typ in inner.items():
            df.loc[pid, _PARAMETER_TYPE_COL] = typ
            # pyPESTO requires sigma inner parameters on a LINEAR scale (its validator rejects
            # log/log10 for sigma). Scaling/offset keep their scale.
            if typ == _reconcile.INNER_TYPE_SIGMA:
                df.loc[pid, PARAMETER_SCALE] = LIN
            inner_scales[pid] = str(df.loc[pid, PARAMETER_SCALE])
        # Sigma → [0, inf], scaling/offset → [-inf, inf]; required by the inner solver.
        df = correct_parameter_df_bounds(df)

        hier_problem = copy.copy(session.problem)  # shallow: shares model/measurement/etc.
        hier_problem.parameter_df = df
        importer = pypesto.petab.PetabImporter(hier_problem, hierarchical=True, **importer_kwargs)
        return importer.create_problem(), inner_scales
    except Exception:  # noqa: BLE001 - hierarchical is best-effort; fall back to a joint fit
        session.logger.warning(
            "hierarchical optimization setup failed (%s inner params: %s); "
            "falling back to a joint fit", len(inner), sorted(inner), exc_info=True
        )
        return None, {}


def _inner_fitted_values(pypesto_problem, best, inner_scales: dict[str, str]) -> dict[str, float]:
    """Analytically-solved inner-parameter values from a hierarchical fit result, as
    ``{parameter_id: linear value}`` — merged into ``fitted`` alongside the outer parameters so
    the parameter count ``k`` and the value writeback both cover every estimated parameter.

    pyPESTO returns ``best["inner_parameters"]`` positionally; the calculator's
    ``get_interpretable_inner_par_ids()`` gives the matching id order, and each value is on the
    parameter's own estimation scale (so it un-scales to linear exactly like an outer one).
    Returns ``{}`` if the result carries no inner parameters or the ids can't be recovered.
    """
    values = best.get("inner_parameters")
    if values is None:
        return {}
    try:
        from petab.v1.parameters import unscale

        calc = pypesto_problem.objective.calculator
        ids = calc.get_interpretable_inner_par_ids()
        if len(ids) != len(values):
            return {}
        return {pid: float(unscale(float(v), inner_scales.get(pid, LIN)))
                for pid, v in zip(ids, values)}
    except Exception:  # noqa: BLE001 - inner-value recovery is best-effort reporting
        return {}


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


def estimation_problem(session, engine):
    """The pyPESTO problem for the session's CURRENT estimation problem, cached on the engine.

    ONE pyPESTO problem now serves both jobs that need a likelihood: the multistart fit, and
    every plain evaluation of the objective (:meth:`AmiciEngine.evaluate_objective` /
    :meth:`AmiciEngine.residuals`). That unification is the fix for a class of bug, not just an
    optimisation — while the fit and the evaluation each built their own AMICI model and their
    own solver, the two could disagree about the very same likelihood at the very same
    parameters, and did: a crauste fit converged at nll 266.62 while re-simulating its optimum
    failed to integrate past t=22 and reported NaN. Sharing the objective makes "the nll of
    this fit" a single number computed one way.

    It also halves the compiling: the fit used to build ``petab_mcp_pp_<hash>`` while the engine
    built ``petab_mcp_<hash>`` from the same SBML — two full C++ builds of one structure, ~80s
    each on a cluster node.

    Cached on ``engine`` under :meth:`session.problem_hash` — the model AND the parameter table,
    since pyPESTO bakes bounds/scales/fixed values into the problem at construction, so a
    bounds edit (or the nominal-value write-back after a fit) must rebuild it. Rebuilding is
    cheap relative to a compile: the AMICI module is keyed on the SBML hash alone and is reused.

    Returns ``(problem, inner_scales)`` — see :func:`_build_hierarchical_problem` for the
    latter (``{}`` for an ordinary joint problem).
    """
    import pypesto.petab

    key = session.problem_hash()
    cached = getattr(engine, "_estimation_problem_cache", None)
    if cached is not None and cached[0] == key:
        return cached[1], cached[2]

    h = session.sbml_hash()
    # Charge the (one-time) model build to the compile bucket only the FIRST time a structure
    # is seen, mirroring the engine's own `_models` guard: rebuilding the pyPESTO problem for a
    # changed parameter table reuses the compiled module and must not count as a compile.
    first_build = h not in engine._pypesto_hashes
    t0 = time.perf_counter()
    importer_kwargs = engine.pypesto_importer_kwargs(h)
    # Prefer a hierarchical objective (inner-solve the noise sigmas / observable-scaling
    # factors, shrinking the outer search) when the problem has cleanly-separable inner
    # parameters; fall back to the ordinary joint objective otherwise.
    problem, inner_scales = _build_hierarchical_problem(session, importer_kwargs)
    if problem is None:
        importer = pypesto.petab.PetabImporter(session.problem, **importer_kwargs)
        problem = importer.create_problem()
        inner_scales = {}
    if first_build:
        session.compile_seconds += time.perf_counter() - t0
        session.n_compiles += 1
        engine._pypesto_hashes.add(h)

    engine._estimation_problem_cache = (key, problem, inner_scales)
    return problem, inner_scales


def problem_x_from_nominal(problem, parameter_df) -> list[float]:
    """The pyPESTO parameter vector for the current ``nominalValue``s.

    Returns the FREE parameters only, in ``x_free_indices`` order — the convention a
    problem-bound objective expects: :meth:`pypesto.Problem.__init__` rewrites its objective to
    take the reduced vector and re-insert ``x_fixed_vals`` itself. (An optimizer *result*'s
    ``x``, by contrast, is full-dimensional — see :func:`_fitted_values`. The two differ, which
    is easy to get wrong: passing the full vector raises a bare numpy shape mismatch.)

    Values are expressed on each parameter's ESTIMATION scale, whereas the session's table
    holds linear ones. ``problem.x_scales`` is the authority on the scale — not the session's
    table — because a hierarchical problem is built from a modified copy (sigma parameters are
    forced to a linear scale there).

    Fixed parameters are pyPESTO's baked-in values, not the table's; that stays correct because
    :func:`estimation_problem` caches on ``session.problem_hash()``, which changes whenever any
    parameter's value/bounds/scale/estimate does, forcing a rebuild.
    """
    from petab.v1.parameters import scale as petab_scale

    full: list[float] = []
    for name, sc in zip(problem.x_names, problem.x_scales):
        val = _flt(parameter_df.loc[name, NOMINAL_VALUE]) if name in parameter_df.index else None
        if val is None:
            # No usable nominal value (a parameter pyPESTO knows about that the table doesn't,
            # or a NaN cell): leave pyPESTO's own value in place rather than silently zeroing.
            fixed = dict(zip(problem.x_fixed_indices, problem.x_fixed_vals))
            full.append(float(fixed.get(len(full), 0.0)))
            continue
        full.append(float(petab_scale(val, sc)))
    free = getattr(problem, "x_free_indices", None)
    if free is None:
        return full
    return [full[i] for i in free]


def _run_fit(session, engine, n_starts: int, seed: int = 0) -> dict[str, Any]:
    """Estimate the free parameters with pyPESTO and write the best fit back.

    Writes the best start's (unscaled) values into ``nominalValue``. Returns the fit's
    ``neg_log_likelihood`` (pyPESTO ``fval``), the fitted values, and the objective at the
    pre-fit nominal values (for improvement tracking). No snapshot, no simulation — callers
    own those.
    """
    import numpy as np
    import pypesto.optimize
    import pypesto.petab

    from .engine import _ensure_swig_on_path

    _ensure_swig_on_path()
    np.random.seed(seed)  # best-effort determinism for start-point sampling

    log = session.logger
    # objective at the current nominal values, for improvement tracking
    # May be None: `evaluate_objective` reports a non-finite likelihood as a MISSING score, and
    # an unstable model at its pre-fit nominal values legitimately has one. That is not a reason
    # to refuse to fit it — the multistart samples its own start points and can still find a
    # good optimum — so carry the missing value through (compute_improvement and _fmt both
    # handle None) instead of `float(None)`, which would abort the fit and, via
    # `_compile_fit_failure_message`, tell the agent its model is ill-posed for estimation.
    objective_before = _flt(engine.evaluate_objective()["neg_log_likelihood"])

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

    # The SAME problem the engine evaluates the objective against (see estimation_problem):
    # one AMICI model, one solver, one notion of this model's likelihood.
    problem, inner_scales = estimation_problem(session, engine)
    hierarchical = bool(inner_scales)
    if hierarchical:
        log.info("fit: hierarchical objective (inner params: %s)", sorted(inner_scales))

    free = getattr(problem, "x_free_indices", None)
    n_free = len(free) if free is not None else len(problem.x_names)
    # Parallelise the multistart across worker processes when more than one is warranted; None runs
    # pyPESTO's sequential engine. Sized so the harness's fit budget (ceil(n_starts/workers) waves)
    # matches the real wall-clock — see examples/_harness/fit_budget.py.
    pp_engine = _fit_engine(n_starts)
    workers = getattr(pp_engine, "n_procs", 1) or 1
    log.info(
        "fit: start optimizer=fides n_starts=%d workers=%d seed=%d n_free=%d sbml=%s "
        "objective_before=%s", n_starts, workers, seed, n_free, session.sbml_hash(),
        _fmt(objective_before),
    )
    # Fides: a Newton trust-region optimizer that exploits AMICI's sensitivities; more
    # robust than the default L-BFGS-B on stiff ODE likelihoods. Its per-iteration logging
    # is routed to pypesto.log (see logging_setup.configure_pypesto). MAXTIME caps each start's
    # wall time (FIT_PER_START_TIMEOUT_S) so a stiff start can't run unbounded and hang the
    # transport; Fides returns that start's best-so-far when the cap is hit.
    optimizer = pypesto.optimize.FidesOptimizer(
        verbose=logging.INFO, options=_fides_maxtime_options())
    # Record the per-iteration optimization history in memory (no trace files) so the
    # traces can be exposed as SQL tables. Older pyPESTO without HistoryOptions just runs
    # without a trace.
    minimize_kwargs: dict[str, Any] = {}
    try:
        minimize_kwargs["history_options"] = pypesto.HistoryOptions(trace_record=True)
    except Exception:  # noqa: BLE001 - HistoryOptions absent/renamed in this pyPESTO
        pass
    t0 = time.perf_counter()
    try:
        result = pypesto.optimize.minimize(
            problem, optimizer=optimizer, n_starts=n_starts, engine=pp_engine,
            progress_bar=False, **minimize_kwargs
        )
    except Exception as exc:  # noqa: BLE001 - retry sequentially before giving up
        # A parallel multistart can construct its engine fine yet still fail at runtime
        # (pickling/spawn issues, a worker dying). That is a parallel-only failure — the same fit
        # runs fine on the sequential engine — so fall back to it rather than failing the whole
        # commit. Sequential is slower, so this can exceed the parallel-sized fit budget on a big
        # n_starts; that is an acceptable rare-path cost vs. a hard failure (the run's resume logic
        # backstops a timeout). A sequential fit that itself fails re-raises normally.
        if pp_engine is None:
            raise
        log.warning("fit: parallel multistart failed (%s: %s); retrying sequentially",
                    type(exc).__name__, exc)
        result = pypesto.optimize.minimize(
            problem, optimizer=optimizer, n_starts=n_starts, engine=None,
            progress_bar=False, **minimize_kwargs
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
    df = session.problem.parameter_df
    nllh = float(best["fval"])
    # Record and write back only the ESTIMATED parameters. pyPESTO's `x`/`x_names` include the
    # fixed parameters too, so filtering to `x_free_indices` (mirroring `_compute_fim`) keeps
    # `fitted` the free parameters only — otherwise k = len(fitted) overcounts and every
    # complexity-penalised score (bic/aic/reduced_chi2) is wrong and disagrees with n_estimated.
    fitted = _fitted_values(problem.x_names, best["x"], getattr(problem, "x_free_indices", None), df)
    if hierarchical:
        # Merge the analytically-solved inner parameters (not in the outer vector above) so
        # `fitted` covers EVERY estimated parameter — keeping k = len(fitted) consistent with
        # session.n_estimated, and writing the inner values back to the table like any other fit.
        fitted.update(_inner_fitted_values(problem, best, inner_scales))
    with session.lock:
        for name, linear in fitted.items():
            if name in df.index:
                df.loc[name, NOMINAL_VALUE] = linear

    # Convergence summary: how many multistarts landed near the best objective.
    fvals = [
        float(s["fval"]) for s in result.optimize_result.list
        if s.get("fval") is not None and math.isfinite(float(s["fval"]))
    ]
    n_converged = sum(1 for v in fvals if v <= nllh + 1e-3)
    log.info(
        "fit: done best_nllh=%s self_improvement=%s time=%.2fs starts=%d converged~%d "
        "fval[min/max]=%s/%s fitted=%s",
        _fmt(nllh), _fmt(objective_before - nllh if objective_before is not None else None),
        duration, len(fvals), n_converged,
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


#: Relative gap between the optimizer's objective and the re-simulated one above which the
#: two are reported as disagreeing. Both evaluate the same likelihood at the same parameters,
#: so anything past a loose numerical tolerance means the two evaluations differ in substance
#: (solver settings, or a parameter that did not round-trip), not in rounding.
OBJECTIVE_AGREEMENT_RTOL = 1e-3


def _reconcile_objective(session, fit: dict[str, Any] | None,
                         sim: dict[str, Any]) -> tuple[float | None, str | None]:
    """Pick the authoritative neg-log-likelihood for a fit, returning ``(nllh, note)``.

    Two numbers claim to be "the" nll of a fitted model: the objective pyPESTO minimised
    (``fit["neg_log_likelihood"]``, its best start's ``fval``) and the one obtained by
    re-simulating at the written-back parameters (``sim["neg_log_likelihood"]``). They are
    the same quantity at the same point and normally agree — but not always: the
    re-simulation goes through :meth:`AmiciEngine.residuals`, whose ``simulate_petab`` call
    uses AMICI's default solver settings rather than the ones pyPESTO's objective was
    configured with. At a stiff optimum (parameters pinned at their bounds) the default
    solver can fail to integrate the full time course where the optimizer's succeeded,
    yielding a NaN llh for a fit that in fact converged.

    Taking the re-simulated value unconditionally — as this did — meant a converged fit
    could be recorded as NaN, which then propagated into a NaN BIC and, via
    :meth:`session.record_fit_progress`, into a best-BIC checkpoint that no later fit could
    ever beat (NaN comparisons are always False). So the optimizer's value wins whenever it
    is finite: it is what was actually optimised, and it is defined over the complete
    dataset. The re-simulated value is kept in the payload either way, and a disagreement
    is surfaced as a note rather than silently resolved.

    A simulate-only pass (no fit) has only the simulated value, which is then authoritative.
    """
    nll_sim = _flt(sim.get("neg_log_likelihood"))
    if fit is None:
        return nll_sim, None
    nll_opt = _flt(fit.get("neg_log_likelihood"))
    n_missing = sim.get("n_missing_simulations") or 0
    if nll_opt is None:
        # The optimizer itself has no usable objective (every start non-finite). Nothing to
        # prefer; the fit genuinely failed to produce a score.
        return nll_sim, ("optimizer returned no finite objective"
                         if nll_sim is None else None)
    if nll_sim is None:
        note = (f"re-simulating the fitted point did not reach {n_missing} of "
                f"{sim.get('n_expected_points')} measured points, so its likelihood is "
                f"undefined; reporting the optimizer's objective ({_fmt(nll_opt)}) instead. "
                "The fit converged — this is a solver/integration gap at the optimum, not a "
                "failed fit.")
        session.logger.warning("fit: %s", note)
        return nll_opt, note
    if abs(nll_opt - nll_sim) > OBJECTIVE_AGREEMENT_RTOL * max(1.0, abs(nll_opt)):
        note = (f"optimizer objective {_fmt(nll_opt)} disagrees with the re-simulated "
                f"{_fmt(nll_sim)} at the same parameters; reporting the optimizer's.")
        session.logger.warning("fit: %s", note)
        return nll_opt, note
    return nll_opt, None


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
        nllh, nll_note = _reconcile_objective(session, fit, sim)

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
            # The OPTIMIZER's objective at its best start when that is available (see
            # _reconcile_objective) — the quantity actually minimised, and the one every
            # downstream score (bic, is_new_best, the judge) must agree with.
            "neg_log_likelihood": nllh,
            # What re-simulating the fitted point produced, kept alongside so a
            # disagreement between the two stays visible rather than silently resolved.
            "neg_log_likelihood_simulated": sim["neg_log_likelihood"],
            "objective_note": nll_note,
            "chi2": sim["chi2"],
            "n_data_points": sim["n_data_points"],
            "n_missing_simulations": sim.get("n_missing_simulations"),
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
    # A model-less problem (zero-shot, before any set_model) has nothing to simulate — and
    # handing its empty model to AMICI segfaults the server (see AmiciEngine._get_model). Skip
    # simulation entirely and leave session.simulation empty; fit_summary reads `... or {}` and
    # reports None metrics, which the judge scores as a clean "no valid fit" rather than crashing.
    if not session.has_model():
        session.simulation = None
        return None
    sim = getattr(session, "simulation", None)
    if sim is None or not sim.get("records"):
        fit_and_simulate(session, engine, do_fit=False, n_starts=0)
    return session.simulation


def _scaled(value: float, scale: str) -> float:
    """``value`` in its estimation-scale coordinate, so "distance to bound" is measured the
    way the optimizer actually sees it (decades for log10, not raw linear units)."""
    if scale == LOG10:
        return math.log10(value)
    if scale == LOG:
        return math.log(value)
    return value


def parameters_at_bound(
    fitted: dict[str, float] | None, parameter_df: Any,
    tol: float = PARAMETER_AT_BOUND_RELATIVE_TOL,
) -> list[dict[str, Any]]:
    """Fitted parameters sitting within ``tol`` (fraction of the scaled bound span) of their
    lower or upper bound — the standard fit report's early-warning for "the optimum is
    unbounded", not just "the search space happened to be tight". See
    ``PARAMETER_AT_BOUND_RELATIVE_TOL`` for why this matters more than it looks: an exponent
    (or any parameter) that re-pins to whatever bound you set, widening after widening, is a
    sign the model is missing a mechanism — not that the box needs to move again.

    Returns ``[]`` when there is nothing to flag (no fit yet, or every value sits mid-range).
    Each flagged entry is ``{parameter_id, side, value, bound_value}`` with ``side`` "lower" or
    "upper".
    """
    if not fitted or parameter_df is None:
        return []
    flags = []
    for pid, value in fitted.items():
        if pid not in parameter_df.index:
            continue
        row = parameter_df.loc[pid]
        scale = str(row[PARAMETER_SCALE])
        try:
            lb, ub = float(row[LOWER_BOUND]), float(row[UPPER_BOUND])
            v, l, u = _scaled(float(value), scale), _scaled(lb, scale), _scaled(ub, scale)
        except (TypeError, ValueError):
            continue
        span = u - l
        if span <= 0:
            continue
        frac = (v - l) / span
        if frac <= tol:
            flags.append({"parameter_id": pid, "side": "lower", "value": value, "bound_value": lb})
        elif frac >= 1 - tol:
            flags.append({"parameter_id": pid, "side": "upper", "value": value, "bound_value": ub})
    return flags


def noise_sigma_report(problem: Any, fitted: dict[str, float] | None) -> dict[str, dict[str, Any]]:
    """Every ESTIMATED noise-sigma parameter's fitted value against its data-driven reference —
    surfaced proactively on EVERY commit, not just once it's pinned at its bound (see
    :func:`parameters_at_bound`), so the agent sees where the reference comes from before it
    ever gets there. Keyed by parameter id; each entry is ``{fitted, upper_bound_cap, data_std,
    fraction_of_data_std}`` (the last two ``None`` if the observable's std can't be computed).
    This is purely informational/for the commit-time guard in ``operations`` — it never rewrites
    the parameter table's actual bounds.

    ``{}`` is normal — most parameters aren't noise sigmas at all.
    """
    if not fitted or problem is None:
        return {}
    from . import reconcile as R

    out: dict[str, dict[str, Any]] = {}
    for pid in sorted(R.noise_sigma_ids(problem) & set(fitted)):
        value = fitted[pid]
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        data_std = R.sigma_data_std(problem, pid)
        out[pid] = {
            "fitted": value,
            "upper_bound_cap": R.sigma_upper_bound_cap(problem, pid),
            "data_std": data_std,
            "fraction_of_data_std": (value / data_std) if data_std else None,
        }
    return out


def noise_sigma_inflated(report: "dict[str, Any] | None") -> bool:
    """True if any ESTIMATED noise sigma in a ``noise_sigma_report`` is fitted ABOVE its
    data-driven cap — i.e. the noise is absorbing model misfit rather than measuring it. The single
    source of truth for "this sigma is inflated", used by the session's plateau-adequacy gate AND
    the judge's success gate so they can't drift apart. Best-effort: an empty/unreadable report or
    a non-numeric entry is treated as NOT inflated."""
    if not isinstance(report, dict):
        return False
    for entry in report.values():
        if not isinstance(entry, dict):
            continue
        fitted, cap = entry.get("fitted"), entry.get("upper_bound_cap")
        try:
            if fitted is not None and cap is not None and float(fitted) > float(cap):
                return True
        except (TypeError, ValueError):
            continue
    return False


def noise_sigma_collapsed(report: "dict[str, Any] | None") -> bool:
    """True if any ESTIMATED noise sigma is fitted implausibly SMALL — below
    ``constants.NOISE_SIGMA_COLLAPSE_FRACTION`` of its observable's own data std, i.e. the model is
    explaining nearly all the data's variability (overfitting the noise for a suspiciously good BIC).
    The other tail of :func:`noise_sigma_inflated`, and likewise the single source of truth shared by
    the session's adequacy gate and the judge's success gate. Best-effort: an empty/unreadable report
    or a non-numeric ``fraction_of_data_std`` is treated as NOT collapsed."""
    if not isinstance(report, dict):
        return False
    for entry in report.values():
        if not isinstance(entry, dict):
            continue
        frac = entry.get("fraction_of_data_std")
        if isinstance(frac, (int, float)) and frac < NOISE_SIGMA_COLLAPSE_FRACTION:
            return True
    return False


def fit_summary(session, engine) -> dict[str, Any]:
    """The current fit metrics, no per-point records — a cheap score read (backs
    ``get_fit_summary``).

    Ensures a simulation dataframe exists, then returns ``{status, neg_log_likelihood, chi2,
    n_data_points, fitted_parameters, parameter_scales, version}`` plus the complexity-
    penalised :func:`goodness_of_fit` scores, ``parameters_at_bound`` (see
    :func:`parameters_at_bound`), and ``noise_sigma_report`` (see :func:`noise_sigma_report`).
    ``k`` uses ``n_estimated`` (well-defined even on a simulate-only pass). Pairs fitted values
    with their estimation scale even when a restored/rollback payload lacks the map, so a linear
    value is never misread as a log10 one.
    """
    sim = ensure_simulation(session, engine) or {}
    out = {k: sim.get(k) for k in (
        "status", "neg_log_likelihood", "chi2", "n_data_points",
        "n_missing_simulations", "objective_note",
        "fitted_parameters", "parameter_scales", "version",
    )}
    pdf = session.problem.parameter_df
    if out.get("fitted_parameters") and not out.get("parameter_scales"):
        out["parameter_scales"] = {
            pid: str(pdf.loc[pid, PARAMETER_SCALE])
            for pid in out["fitted_parameters"] if pid in pdf.index}
    out.update(goodness_of_fit(
        sim.get("neg_log_likelihood"), sim.get("chi2"),
        int(session.problem.n_estimated), sim.get("n_data_points"),
    ))
    out["parameters_at_bound"] = parameters_at_bound(out.get("fitted_parameters"), pdf)
    out["noise_sigma_report"] = noise_sigma_report(session.problem, out.get("fitted_parameters"))
    # These metrics are the LAST fit's. Edits are fit-free, so the working model can have moved on
    # since then; flag it so a caller never mistakes a stale score for the edited model's score.
    # `version` above is the version these metrics belong to; call `fit` to score the working model.
    out["has_unfit_edits"] = session.has_unfit_edits()
    return out


def compute_improvement(
    session, nllh: float | None, objective_before: float | None = None,
    fitsim: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Improvement metrics for a fit, ranking it against the best model so far **by BIC**.

    ``is_new_best`` tracks BIC — the same complexity-penalised score the judge gates on and the
    early-stop checkpoint stores — so the flag, the non-improving-streak reset, and the
    best-model checkpoint all move on one coherent notion of "better". A fit is a new best only
    when its BIC beats the reference by at least ``NEW_BEST_MIN_DELTA_BIC`` (the conventional
    "positive evidence" bar), so a raw nll drop bought by an extra parameter that actually RAISES
    BIC does NOT count, and neither does a 1e-6 noise-floor twitch.

    The reference is the session's best-BIC CHECKPOINT (``best_bic`` / ``best_version`` /
    ``best_nll``), set on the previous new best — not the raw running minimum. Comparing against
    the checkpoint is what lets a slow, steady BIC creep of < ``NEW_BEST_MIN_DELTA_BIC`` per op
    still register a new best once it CUMULATIVELY clears the bar, instead of starving the streak
    while the fit genuinely improves. Must be called BEFORE :meth:`session.record_fit_progress`
    updates the checkpoint, so a fit never compares against itself. The first fitted version (no
    checkpoint yet) is always a new best.
    """
    fitsim = fitsim or {}
    bic = goodness_of_fit(
        nllh, fitsim.get("chi2"), fitsim.get("n_estimated"), fitsim.get("n_data_points"),
    ).get("bic")

    # A non-finite checkpoint is no checkpoint: treat it as absent so a session that somehow
    # acquired one (a legacy provenance seed, a version written before goodness_of_fit
    # rejected non-finite scores) recovers on its next fit instead of staying locked out.
    best_prior_bic = _flt(getattr(session, "best_bic", None))
    best_prior_version = getattr(session, "best_version", None)
    best_prior_nllh = getattr(session, "best_nll", None)

    self_improvement = None
    if objective_before is not None and nllh is not None:
        self_improvement = objective_before - nllh
    improvement_vs_best_prior = None  # BIC delta: > 0 means this fit is better than the checkpoint
    if best_prior_bic is not None and bic is not None:
        improvement_vs_best_prior = best_prior_bic - bic
    # ``<=`` so an improvement EXACTLY at the threshold counts as a new best — i.e. ΔBIC >= the
    # gate, matching the docstring's "at least NEW_BEST_MIN_DELTA_BIC" (a strict ``<`` would exclude
    # the boundary and leave the streak un-reset there).
    is_new_best = best_prior_bic is None or (
        bic is not None and bic <= best_prior_bic - NEW_BEST_MIN_DELTA_BIC
    )

    return {
        "neg_log_likelihood": nllh,
        "objective_before": objective_before,
        "self_improvement": self_improvement,
        "bic": bic,
        "best_prior_version": best_prior_version,
        "best_prior_bic": best_prior_bic,
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
    * ``reduced_chi2 = chi2 / (n - k)`` is ≈ 1 when the model fits to the noise floor, and
      ``chi2_gof_pvalue`` is the upper-tail chi2 p-value on ``chi2`` with ``n - k`` degrees of
      freedom. Both are only meaningful with a KNOWN noise level: when the observable noise
      sigmas are ESTIMATED, chi2 pins near ``n`` by construction, so reduced_chi2 ≈ 1 and the
      p-value are uninformative — rank models by ``bic``/``aic``, not by these.

    ``k`` is the number of estimated parameters, ``n`` the number of data points. ``bic`` and
    ``aic`` are ``None`` only when ``nll``/``k``/``n`` are missing; the chi2-based fields
    (``chi2_dof``, ``reduced_chi2``, ``chi2_gof_pvalue``) are additionally ``None`` when the
    degrees of freedom ``n - k`` are non-positive (an under-determined fit) or ``chi2`` is
    missing.
    """
    out: dict[str, Any] = {
        "bic": None, "aic": None, "chi2_dof": None,
        "reduced_chi2": None, "chi2_gof_pvalue": None,
    }
    # Non-finite inputs are MISSING scores, not scores. A NaN nll would otherwise produce
    # bic = k*ln(n) + 2*NaN = NaN, which is a float and passes every `is not None` guard
    # downstream — and because NaN compares False against everything, a NaN adopted as the
    # best-BIC checkpoint locks out every subsequent fit for the rest of the run (the
    # `is_new_best` test can never fire again). Coercing to None here is the single upstream
    # chokepoint that keeps a failed evaluation from masquerading as a real score.
    nll = _flt(nll)
    chi2 = _flt(chi2)
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
    fitsim: dict[str, Any] | None, improvement: dict[str, Any] | None = None,
    parameter_df: Any = None, problem: Any = None,
) -> dict[str, Any]:
    """The fit fields every fit-bearing result returns (``fit``, rollback, reset).

    Keeps the tool-result shape identical across those paths so an agent reads the same
    keys everywhere. ``fitsim`` is a :func:`fit_and_simulate` payload (or the restored
    simulation for rollback/reset); ``improvement`` is the :func:`compute_improvement` dict.
    Beyond the raw nll/chi2 it also carries the complexity-penalised
    :func:`goodness_of_fit` scores (bic/aic/reduced_chi2/gof p-value) so the agent can see
    when the fit is already at the noise floor and stop adding parameters,
    ``parameters_at_bound`` (see :func:`parameters_at_bound`; empty when ``parameter_df`` is
    not passed) so a parameter pinned to its bound is flagged in the SAME report instead of
    requiring a separate ``get_parameters()`` comparison, and ``noise_sigma_report`` (see
    :func:`noise_sigma_report`; empty when ``problem`` is not passed).
    """
    fitsim = fitsim or {}
    fitted = fitsim.get("fitted_parameters")
    nll = fitsim.get("neg_log_likelihood")
    chi2 = fitsim.get("chi2")
    n = fitsim.get("n_data_points")
    # k = number of estimated parameters (n_estimated) — the honest BIC complexity count. It
    # matches compute_improvement / record_fit_progress / fit_summary and includes any inner
    # (analytically-profiled) params that len(fitted_parameters) omits, so the block's bic agrees
    # with improvement.bic within the same result. Fall back to len(fitted_parameters) only when
    # the payload carries no n_estimated (legacy simulate-only payloads).
    k = fitsim.get("n_estimated")
    if k is None and fitted is not None:
        k = len(fitted)
    block = {
        "fit_status": fitsim.get("status"),
        "neg_log_likelihood": nll,
        "chi2": chi2,
        "n_data_points": n,
        # Non-zero means the trajectory did not reach every measured point, so chi2 is None
        # and the likelihood came from the optimizer rather than the re-simulation.
        "n_missing_simulations": fitsim.get("n_missing_simulations"),
        "objective_note": fitsim.get("objective_note"),
        # Surface the estimated-parameter count k next to bic so the parameter penalty in
        # bic = k*ln(n) + 2*nll is legible — adding a free parameter must earn its keep.
        "n_estimated": k,
        # fitted_parameters: LINEAR (unscaled) values; parameter_scales: each one's estimation
        # scale (lin/log10). Kept side by side so a linear value is never misread as a log10 one.
        "fitted_parameters": fitted,
        "parameter_scales": fitsim.get("parameter_scales"),
        "improvement": improvement or None,
        "parameters_at_bound": parameters_at_bound(fitted, parameter_df),
        "noise_sigma_report": noise_sigma_report(problem, fitted),
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
        # Carried so a restored version keeps the reason its likelihood was/wasn't usable,
        # rather than looking like a clean fit after a rollback.
        "n_missing_simulations": payload.get("n_missing_simulations"),
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
