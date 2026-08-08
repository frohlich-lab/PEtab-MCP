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

import copy
import json
import logging
import math
import os
import time
from dataclasses import dataclass
from typing import Any

from petab.v1.C import (
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
    NOISE_SIGMA_MIN_POINTS,
    NEW_BEST_MIN_DELTA_BIC,
    NOISE_SIGMA_COLLAPSE_FRACTION,
    PARAMETER_AT_BOUND_RELATIVE_TOL,
)
from .formatting import finite_float as _flt
from .parameters import parameter_df_to_tsv, sha16, structure_fingerprint

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
# PETAB_MCP_FIT_PER_START_TIMEOUT, default 180). Fides defaults MAXTIME to inf, so one stiff start
# can run unbounded and out-idle the MCP transport, which the harness then mistakes for a hang.
# Bounding each start makes a full multistart fit's worst-case duration finite and predictable,
# which the harness sizes its sse_read_timeout + stall watchdog to (see
# petab_harness/fit_budget.py). The harness exports its own resolved values onto this server's environment, so
# both sides use the SAME cap + worker count. Kept generous — Fides returns best-so-far for a
# capped start, so too tight a cap would silently degrade fit quality.
FIT_PER_START_TIMEOUT_S = _env_float("PETAB_MCP_FIT_PER_START_TIMEOUT", 180.0)


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
# fitting. petab_harness/fit_budget.py keeps the SAME default (its _default_workers mirrors
# _default_fit_workers) and, when the harness drives a run, also exports its resolved value as
# PETAB_MCP_FIT_N_WORKERS onto this server — so the harness's budget matches the real parallelism
# whether the value comes from the shared default or the export.
FIT_N_WORKERS = _env_int("PETAB_MCP_FIT_N_WORKERS", _default_fit_workers())

# AMICI absolute integration tolerance for every candidate fit (env PETAB_MCP_FIT_AMICI_ATOL).
# AMICI's stock 1e-16 is tight enough that on a stiff model the forward-sensitivity solve cannot
# make progress: the objective VALUE evaluates, the GRADIENT comes back non-finite, and Fides
# reads +INF at iteration 0 — a start that reports as failed when really the solver could not
# differentiate there. petab_harness/fit_budget.py keeps the SAME default in FIT_AMICI_ATOL and
# applies it to the REFERENCE fit; both sides must use one tolerance or the reference and the
# candidate are optimized against different objectives and ΔBIC stops being a comparison. See
# that constant for the lucarelli measurement that set the value.
FIT_AMICI_ATOL = _env_float("PETAB_MCP_FIT_AMICI_ATOL", 1e-10)


def _apply_amici_solver_defaults(problem) -> None:
    """Set :data:`FIT_AMICI_ATOL` on ``problem``'s AMICI solver, in place. Best-effort: a problem
    whose objective exposes no AMICI solver is left alone rather than raising, since a missing
    tolerance degrades a fit while a raise here would break every fit path at once. Mirrors
    ``petab_harness.fit_budget.apply_amici_solver_defaults``."""
    solver = getattr(getattr(problem, "objective", None), "amici_solver", None)
    setter = (getattr(solver, "set_absolute_tolerance", None)
              or getattr(solver, "setAbsoluteTolerance", None))
    if setter is not None:
        setter(FIT_AMICI_ATOL)

# The multistart's RNG seed and optimizer. Named constants rather than parameters: the fit's own
# log line hard-codes "optimizer=fides", so a signature knob would be configurable in name only.
FIT_SEED = 0
FIT_OPTIMIZER = "fides"

# -- warm starts (see `warm_start_guesses`) ---------------------------------
# Fraction of the multistart re-entered at the PREVIOUS fit's optima — its best one (which the
# write-back left in the nominal values) and then its other converged basins, each remapped onto
# this problem's parameter vector. For a NESTED edit (a new parameter added at a neutral value)
# the best one reproduces the parent's likelihood exactly, so a fit that starts there cannot
# converge WORSE than its parent unless the added structure genuinely cannot help. That is the
# whole point: an "adding this term made the fit worse" result is otherwise indistinguishable
# from an under-converged multistart that never found the parent's basin (see :mod:`nesting`),
# and the panel has read the latter as the former. Re-entering the OTHER basins matters for the
# same reason one notch down: they are the local optima the parent's own multistart paid to find,
# and an additive edit does not destroy them.
# The other half stays FRESH random: an added parameter can create new local optima that none of
# the parent's basins contain, and only an unbiased sample finds those.
WARM_START_FRACTION = 0.5
# Spread of the FALLBACK warm starts, as a fraction of each parameter's box width in ESTIMATION
# scale. Used only to fill the warm budget when the previous fit cannot supply enough DISTINCT
# optima — the run's first fit, or a multistart that converged everything to one point — so the
# warm half explores that point's neighbourhood instead of re-running one optimization N times.
WARM_START_JITTER = 0.05


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
            # multistart-spread identifiability signal (:mod:`identifiability`). Order matches the
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
    # is seen: rebuilding the pyPESTO problem for a changed parameter table reuses the compiled
    # module and must not count as a compile.
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
    # After the branch, so the hierarchical and joint objectives are integrated identically.
    _apply_amici_solver_defaults(problem)
    if first_build:
        session.compile_seconds += time.perf_counter() - t0
        session.n_compiles += 1
        engine._pypesto_hashes.add(h)

    engine._estimation_problem_cache = (key, problem, inner_scales)
    return problem, inner_scales


@dataclass(frozen=True)
class FitRecord:
    """What was fitted, frozen the moment the optimizer finished — the ONLY thing a diagnosis of
    that fit may read (see :mod:`identifiability`).

    Every number a diagnosis reports is a local expansion around the optimum of ONE estimation
    problem, so the diagnosis has to name that problem. Naming it by HASH and resolving the hash
    from live session state is what let a diagnosis pay an AMICI C++ build: the de-dup no-op
    re-authors the working SBML into a hash nothing is built for, and the diagnosis then priced —
    and compiled — a structure that was never fitted. Holding the built ``pypesto_problem`` makes
    that unreachable rather than merely guarded: there is no hash to resolve, no importer to
    construct, so no diagnosis path can reach codegen.

    ``problem`` is a DEEP copy of the PEtab problem: the parameters-only edit path reconciles the
    live SBML in place, so a shallow copy's model could still be moved under a frozen record.
    ``x_free``/``lb``/``ub`` are the FREE coordinates on the estimation scale — pyPESTO's own,
    which is the authority (a hierarchical problem re-scales its sigmas), and the space the FIM is
    expressed in. The fitted point is carried here rather than re-read from
    ``parameter_df[nominalValue]`` so an agent edit to the table cannot silently move the point a
    diagnosis expands around.

    The two fingerprints are COMPARISONS only — never keys to rebuild anything with — and they
    answer different questions, which is why both are here. ``structure_hash``
    (:func:`parameters.structure_fingerprint`) is the reaction network plus the estimated-parameter
    set, so it says whether the geometry frozen here still describes the model in front of the
    lead; a diagnosis is STALE exactly when that moved. ``problem_hash`` also covers the parameter
    table, so it additionally moves when the working table alone does — which happens on the de-dup
    no-op, where the same structure is re-authored with a new starting guess and no fit runs. That
    is not a stale diagnosis: the optimum, the structure and HEAD's fit are all still this record's.

    No AMICI output folder can be named from anything here: the SBML hash is deliberately absent,
    since it is the one argument :meth:`engine.AmiciEngine.pypesto_importer_kwargs` takes and a
    record holding it would put a build target back within a diagnosis's reach.

    The optimizer's own stored Hessian is deliberately NOT carried: Fides reports the Hessian of
    whatever update scheme it ran with, so taking it would change the statistic from problem to
    problem. The FIM is still evaluated on this frozen objective — one sensitivity evaluation, and
    nothing to build.
    """

    pypesto_problem: Any
    problem: Any
    structure_hash: str
    problem_hash: str
    x_names: tuple[str, ...]
    x_free: tuple[float, ...]
    scales: tuple[str, ...]
    lb: tuple[float, ...]
    ub: tuple[float, ...]
    optimization: dict[str, Any] | None
    hierarchical: bool


def freeze_fit(session, problem, best, *, hierarchical: bool,
               optimization: dict[str, Any] | None) -> FitRecord:
    """Freeze the fit that just finished into a :class:`FitRecord`.

    Called from :func:`_run_fit` after the write-back, which is the only moment where the built
    problem, the optimizer's best point and the table it was written into all agree.
    """
    import numpy as np

    free = getattr(problem, "x_free_indices", None)
    idx = list(free) if free is not None else list(range(len(problem.x_names)))
    # A result's `x` is FULL-dimensional (fixed parameters included) — the same convention
    # `_fitted_values` reduces — while `problem.lb`/`ub` are already the free ones. Reduce whichever
    # arrives full so all three tables are in ONE order (idx), which is the FIM's order too.
    x_full = np.asarray(best["x"], dtype=float)
    lb = np.asarray(problem.lb, dtype=float)
    ub = np.asarray(problem.ub, dtype=float)
    if lb.size != len(idx):
        lb, ub = lb[idx], ub[idx]
    return FitRecord(
        pypesto_problem=problem,
        problem=copy.deepcopy(session.problem),
        structure_hash=structure_fingerprint(
            session.problem.model.sbml_model, session.problem.parameter_df),
        problem_hash=session.problem_hash(),
        x_names=tuple(str(problem.x_names[i]) for i in idx),
        x_free=tuple(float(x_full[i]) for i in idx),
        scales=tuple(str(problem.x_scales[i]) for i in idx),
        # pyPESTO's lb/ub are already the REDUCED (free) bounds, in the same order as `idx`.
        lb=tuple(float(v) for v in lb),
        ub=tuple(float(v) for v in ub),
        optimization=optimization,
        hierarchical=hierarchical,
    )


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
    full = problem_x_full_from_nominal(problem, parameter_df)
    free = getattr(problem, "x_free_indices", None)
    if free is None:
        return full
    return [full[i] for i in free]


def problem_x_full_from_nominal(problem, parameter_df) -> list[float]:
    """The FULL (``dim_full``) scaled parameter vector for the current ``nominalValue``s — fixed
    parameters included, in ``problem.x_names`` order.

    The shape pyPESTO's ``x_guesses`` wants (see :func:`warm_start_guesses`);
    :func:`problem_x_from_nominal` reduces it to the free coordinates an objective takes.
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
    return full


def previous_optima(problem, optimization: dict[str, Any] | None, x0) -> list[list[float]]:
    """The PREVIOUS fit's per-start optima, remapped onto this problem's parameter vector and
    ranked best-first — the raw material for the warm starts.

    ``optimization`` is a :func:`_optimization_payload` dict (``session.optimization``), whose
    ``starts`` are already sorted by objective (rank 0 = best) and carry the full scaled vector
    ``x`` in ``x_names`` order. Each is remapped rather than used as-is: the previous model has a
    DIFFERENT parameter vector — that is the point of the edit — so a shared parameter is carried
    across (through LINEAR, since the two problems can scale it differently: a hierarchical
    problem forces its sigmas to a linear scale) and anything the previous fit did not have keeps
    its value from ``x0``, the current nominal point. For an additive edit that means every
    previous local optimum is re-entered with the new parameter at its neutral value.

    Returns ``[]`` when there is no usable previous fit: no payload (the run's first fit), or one
    written before ``x_names``/``x_scales`` were recorded — remapping without the scales would
    silently mix log10 and linear coordinates, which is worse than not warm-starting at all.
    Starts whose objective or vector is non-finite are dropped: a failed start is not a basin.
    """
    from petab.v1.parameters import scale as petab_scale
    from petab.v1.parameters import unscale

    starts = (optimization or {}).get("starts") or []
    prev_names = (optimization or {}).get("x_names")
    prev_scales = (optimization or {}).get("x_scales")
    if not starts or not prev_names or not prev_scales:
        return []
    # Where each shared parameter sits in the previous vector, and the two scales to convert
    # between. Built once: the same map serves every start.
    cur_index = {name: i for i, name in enumerate(problem.x_names)}
    shared = [
        (cur_index[name], j, str(prev_scales[j]), str(problem.x_scales[cur_index[name]]))
        for j, name in enumerate(prev_names)
        if name in cur_index and j < len(prev_scales)
    ]
    if not shared:
        return []

    out: list[list[float]] = []
    for start in starts:
        xs = start.get("x")
        if not xs or _flt(start.get("fval")) is None:
            continue
        point = list(x0)
        try:
            for cur_i, prev_j, prev_sc, cur_sc in shared:
                value = _flt(xs[prev_j])
                if value is None:
                    raise ValueError("non-finite coordinate")
                point[cur_i] = float(petab_scale(unscale(value, prev_sc), cur_sc))
        except (IndexError, KeyError, TypeError, ValueError):
            continue  # a malformed/short vector is one unusable start, not a broken fit
        out.append(point)
    return out


def warm_start_guesses(problem, parameter_df, n_starts: int, optimization=None):
    """``WARM_START_FRACTION`` of the multistart's start points, seeded from the PREVIOUS fit.
    Returns a ``(n_warm, dim_full)`` array — pyPESTO fills the first ``n_warm`` starts from it
    and samples the remaining ``n_starts - n_warm`` freshly.

    Start 0 is the current nominal point: the previous fit's BEST optimum (``_run_fit`` writes it
    back) with any newly-introduced parameter at its neutral value. For a nested edit that point
    reproduces the parent's likelihood exactly, so the fit cannot end up worse than its parent
    unless the added structure truly does not help — which is what makes "adding this term made
    the fit worse" a readable result instead of a coin flip between a real answer and a search
    that never found the parent's basin (see :mod:`nesting`).

    Starts 1..n_warm-1 are the previous multistart's OTHER optima, best-first (see
    :func:`previous_optima`), deduplicated against each other. Those are real, converged basins
    of the parent likelihood — under an additive edit they stay basins — so re-entering them is a
    far better use of the warm half than perturbing one point at random. Only when the previous
    fit cannot supply enough distinct optima (the run's first fit, or a multistart that converged
    everything to one point) is the remainder filled by jittering start 0 (Gaussian,
    ``WARM_START_JITTER`` of each box's width in estimation scale), so the warm/fresh split still
    holds. Every start is clipped into the box.

    Deterministic: jitter is drawn from a private ``default_rng(FIT_SEED)``, so it neither
    depends on nor perturbs the global stream pyPESTO samples its fresh starts from.

    Returns an empty ``(0, dim_full)`` array when there is nothing to warm-start from
    (``n_starts < 2``, no free parameters, or a free parameter with no usable nominal value),
    which pyPESTO treats as "sample every start".
    """
    import numpy as np

    dim_full = len(problem.x_names)
    free_idx = getattr(problem, "x_free_indices", None)
    # `None` means an all-free problem; an EMPTY list means an all-FIXED one. Collapsing the two
    # (a plain `or` falls back on both) would warm-start a problem with nothing to estimate.
    free = list(range(dim_full)) if free_idx is None else list(free_idx)
    n_warm = int(n_starts * WARM_START_FRACTION)
    if n_warm < 1 or not free:
        return np.zeros((0, dim_full))
    # Every FREE parameter needs a usable nominal value. Where one is missing,
    # `problem_x_full_from_nominal` substitutes pyPESTO's own (0.0 for a free coordinate) — a
    # meaningless start point, so fall back to a wholly fresh sample instead.
    if any(n not in parameter_df.index or _flt(parameter_df.loc[n, NOMINAL_VALUE]) is None
           for n in (problem.x_names[i] for i in free)):
        return np.zeros((0, dim_full))

    x0 = np.asarray(problem_x_full_from_nominal(problem, parameter_df), dtype=float)
    # Only the FREE coordinates have to be finite: pyPESTO re-inserts its own `x_fixed_vals` for
    # the fixed ones, so a fixed parameter that scales to -inf (nominalValue 0 on log10) never
    # reaches the optimizer and must not disable warm starts for the parameters that do.
    if not np.all(np.isfinite(x0[free])):
        return np.zeros((0, dim_full))
    # Bounds are free-space vectors (pyPESTO drops the fixed coordinates), so clipping (and the
    # jitter fallback) happen on the free sub-vector and are written back into the full one.
    lb = np.asarray(problem.lb, dtype=float)
    ub = np.asarray(problem.ub, dtype=float)

    def _clipped(point):
        row = np.asarray(point, dtype=float)
        row[free] = np.clip(row[free], lb, ub)
        return row

    # Start 0 (the previous best + neutral new parameters) then the previous fit's other optima,
    # best-first. Deduplicated on the FREE coordinates: a multistart that converged many starts to
    # the same optimum would otherwise spend the warm half re-running one optimization. Compared
    # after clipping, so two points that differ only outside the box count as one.
    seen: set[tuple] = set()
    guesses: list = []
    for point in [x0, *previous_optima(problem, optimization, x0)]:
        row = _clipped(point)
        if not np.all(np.isfinite(row[free])):
            continue
        key = tuple(np.round(row[free], 9))
        if key in seen:
            continue
        seen.add(key)
        guesses.append(row)
        if len(guesses) == n_warm:
            break

    if len(guesses) < n_warm:
        # Not enough distinct previous optima — scatter around start 0 to fill the warm budget.
        rng = np.random.default_rng(FIT_SEED)
        # An unbounded coordinate has no box width to scale by, and `inf * normal()` would put
        # inf/NaN start points in front of the optimizer (clipping to an infinite bound leaves
        # them there). Give those coordinates zero spread: their warm starts stay at the nominal
        # value, which is at least feasible.
        width = ub - lb
        spread = WARM_START_JITTER * np.where(np.isfinite(width), width, 0.0)
        base = guesses[0][free]
        while len(guesses) < n_warm:
            row = np.array(guesses[0])
            row[free] = np.clip(base + rng.normal(0.0, 1.0, size=len(free)) * spread, lb, ub)
            guesses.append(row)
    return np.asarray(guesses)


def _run_fit(session, engine, n_starts: int) -> dict[str, Any]:
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
    np.random.seed(FIT_SEED)  # best-effort determinism for start-point sampling

    log = session.logger
    # objective at the current nominal values, for improvement tracking
    # May be None: `evaluate_objective` reports a non-finite likelihood as a MISSING score, and
    # an unstable model at its pre-fit nominal values legitimately has one. That is not a reason
    # to refuse to fit it — the multistart samples its own start points and can still find a
    # good optimum — so carry the missing value through (compute_improvement and _fmt both
    # handle None) instead of `float(None)`, which would abort the fit and, via
    # `_compile_fit_failure_message`, tell the agent its model is ill-posed for estimation.
    objective_before = _flt(engine.evaluate_objective()["neg_log_likelihood"])

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
    # matches the real wall-clock — see petab_harness/fit_budget.py.
    pp_engine = _fit_engine(n_starts)
    workers = getattr(pp_engine, "n_procs", 1) or 1
    # WARM STARTS: seed half the multistart from the PREVIOUS fit — its best optimum (which the
    # write-back left in the nominal values) plus its other converged optima, remapped onto this
    # problem — so a nested edit's fit begins at points that already reproduce its parent's
    # likelihood, and a worse result means the structure, not the search (see
    # `warm_start_guesses` and :mod:`nesting`). `session.optimization` still holds that previous
    # fit here; `fit_and_simulate` replaces it only after this returns, and a rollback restores
    # the version's own. Skipped when the nominal point has no finite objective: an unstable
    # model there has nothing worth starting from, and the fresh sample is the only way in.
    # `problem` is cached on the engine, so ALWAYS assign — an empty array clears guesses left by
    # a previous fit rather than silently reusing them.
    guesses = (warm_start_guesses(problem, session.problem.parameter_df, n_starts,
                                  optimization=session.optimization)
               if objective_before is not None else np.zeros((0, len(problem.x_names))))
    problem.x_guesses_full = guesses
    log.info(
        "fit: start optimizer=fides n_starts=%d warm_starts=%d workers=%d seed=%d n_free=%d "
        "sbml=%s objective_before=%s", n_starts, len(guesses), workers, FIT_SEED, n_free,
        session.sbml_hash(), _fmt(objective_before),
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
    # ...and the scale each of those coordinates is expressed on, so the NEXT fit can remap these
    # optima onto its own (differently-parameterised, possibly differently-scaled) vector and
    # re-enter them as warm starts. See `previous_optima`.
    optimization["x_scales"] = [str(s) for s in problem.x_scales]
    optimization["result_file"] = _write_pypesto_result(session, result, session.problem_hash())
    return {"fitted": fitted, "neg_log_likelihood": nllh, "objective_before": objective_before,
            "optimization": optimization,
            # Frozen HERE, where the built problem, the optimizer's point and the written-back
            # table all agree — every later diagnosis of this fit reads only this (see FitRecord).
            "record": freeze_fit(session, problem, best, hierarchical=hierarchical,
                                 optimization=optimization)}


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


def fit_and_simulate(session, engine, *, do_fit: bool, n_starts: int) -> dict[str, Any]:
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
        fit = _run_fit(session, engine, n_starts) if do_fit else None
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
        # The frozen record of the fit, for every later diagnosis of it (see FitRecord). A
        # simulate-only pass does NOT clear it: it produced no optimum of its own, and the record
        # a diagnosis reads is labelled STALE by comparison rather than thrown away — otherwise a
        # restore would leave the parameter diagnosis unavailable for the rest of the run.
        if fit is not None:
            session.fit_record = fit["record"]
        session.logger.info(
            "fit_and_simulate: status=%s nllh=%s chi2=%s n_points=%s",
            payload["status"], _fmt(payload["neg_log_likelihood"]),
            _fmt(payload["chi2"]), payload["n_data_points"],
        )
        return payload


def ensure_simulation(session, engine) -> dict[str, Any] | None:
    """Populate ``session.simulation`` with a simulate-only pass if it is empty, once a mechanism
    has been authored — but NEVER by compiling one.

    Every READ that wants per-point records comes through here (``get_fit_summary``,
    ``get_fit_diagnosis``'s residual half, ``query_data``, the ``simulations`` resource), so this is
    where "no diagnosis may trigger AMICI codegen" is enforced for that half; the parameter half
    reads a frozen :class:`FitRecord` and has nothing to resolve. Simulating a structure whose module
    already exists costs an integration (hundredths of a second) and is a legitimate on-demand read.
    Simulating one nothing has built costs a full C++ build — 16 s on the toy, and it is a build for a
    structure NOBODY FITTED on the path that reaches here without a simulation: the de-dup no-op
    re-authors the working SBML, so the read would compile a model the run never scored, to describe
    a fit that happened on another one.

    Returns ``None`` (no records) in both refusing cases — an unauthored model, whose content-free
    seed has no dynamics and would simulate to constants that read as a finite likelihood, and an
    unbuilt structure. Callers already handle it: ``data_access._frames`` builds an empty-but-typed
    table and :func:`data_access.fit_diagnosis` says which of the two it was.
    """
    if not session.model_authored():
        return session.simulation
    sim = session.simulation
    if sim is None or not sim.get("records"):
        if not engine.module_built(session.sbml_hash()):
            session.logger.info(
                "no simulation for this read: the working structure has no compiled module, and a "
                "read never pays a build (sbml_hash=%s)", session.sbml_hash())
            return session.simulation
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


def _entry_inflated(entry: "dict[str, Any]") -> bool:
    """Whether ONE noise_sigma_report entry is inflated: fitted above its data-driven cap (the
    noise absorbing model misfit). Shared by the per-entry ``status`` and :func:`noise_sigma_inflated`
    so the agent-facing status can't drift from the gate that fails the run.

    The data-std cap is the ONLY arm, and NO ratio against a noise floor is one: #140 added such an
    arm (sigma above 1.5x the pure-error replicate floor); it is gone. Dividing two point
    estimates discards the denominator's uncertainty, so no single threshold survives the suite —
    blasi needs <=1.004 for any power, schwen >7.83 not to condemn its own reference. The sharp test
    keeps each point's df instead (per-design-point lack-of-fit, in the residual-structure verdict).
    Evidence: BENCHMARK_NOTES, "Pure-error replicate noise floor".
    """
    # Judge the EFFECTIVE sd (noiseFormula evaluated at the fitted value) against the cap, which is
    # a fraction of the same-scale data std; `fitted` alone is dimensionally wrong for
    # proportional/CV and log-space sigmas. Falls back to `fitted` for entries built before
    # effective_sigma existed (older persisted reports).
    # Only fall back to `fitted` for an OLD persisted entry that predates effective_sigma. A
    # FRESH entry whose key is present but None means the formula could not be evaluated, and
    # judging the raw parameter against a scaled cap is precisely the dimensionally-invalid
    # comparison this gate exists to avoid — such an entry is "unknown", not "inflated".
    if "effective_sigma" in entry:
        eff = entry.get("effective_sigma")
    else:
        eff = entry.get("fitted")
    cap = entry.get("upper_bound_cap")
    try:
        return eff is not None and cap is not None and float(eff) > float(cap)
    except (TypeError, ValueError):
        return False


def _entry_collapsed(entry: "dict[str, Any]") -> bool:
    """Whether ONE noise_sigma_report entry is collapsed: fraction of its observable's data std
    below ``NOISE_SIGMA_COLLAPSE_FRACTION`` (overfitting the noise). Shared by the per-entry
    ``status`` and :func:`noise_sigma_collapsed`."""
    frac = entry.get("fraction_of_data_std")
    return isinstance(frac, (int, float)) and frac < NOISE_SIGMA_COLLAPSE_FRACTION


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
    # ESTIMATED sigmas only. A FIXED sigma is a declared constant of the problem — the agent
    # cannot inflate or collapse it — so judging its plausibility can only produce false positives,
    # and it also made `noise_sigma_estimated` (bool of this report) true for problems whose sigmas
    # are all fixed. oliveira is the case: sd_cumulative_cases ships fixed at 1.0 against cumulative
    # counts whose marginal std is 390 (dominated by the trend the model explains), so it read
    # "collapsed" at 0.0026 for a noise level the benchmark declared, not one any fit chose.
    for pid in sorted(R.plausibility_sigma_ids(problem) & set(fitted)):
        # Fail OPEN when there is no parameter table to consult (a bare observable+measurement
        # problem): unknown estimate status must not silently drop the sigma from the report.
        if getattr(problem, "parameter_df", None) is not None \
                and not R.is_parameter_estimated(problem, pid):
            continue
        value = fitted[pid]
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        # Pass the fit: a per-gel amplitude is normalised out with THIS fit's gains
        # (see reconcile._observable_amplitude), so both sides of the ratio below use
        # the same coordinate.
        data_std = R.sigma_data_std(problem, pid, fitted)
        # Compare the EFFECTIVE sd (the noiseFormula evaluated at this value) against a data std
        # taken on the SAME estimation scale — not the bare parameter against the raw linear std.
        # A proportional/CV parameter and a log-space sd are both dimensionally incomparable to the
        # raw std, which used to flag correct models by orders of magnitude (see sigma_data_std /
        # sigma_effective_value).
        effective = R.sigma_effective_value(problem, pid, value, fitted_values=fitted)
        # A MODEL-FREE noise yardstick, independent of both the model and any ground truth. Prefer
        # the UNBIASED replicate (pure-error) estimate — the pooled scatter within repeated design
        # points — wherever the data supplies enough of them; fall back to the biased-upward
        # successive-difference estimate otherwise. Both are REPORTED, neither GATES (see
        # _entry_inflated); `method` records which yardstick a quoted ratio is against.
        data_noise = R.sigma_replicate_noise_estimate(problem, pid, fitted)
        data_noise_method = "replicate"
        if data_noise is None:
            data_noise = R.sigma_difference_noise_estimate(problem, pid, fitted)
            data_noise_method = "difference"
        # How many measurements this sigma is estimated FROM. A sigma is an ML estimate like any
        # other, and its sampling spread is set by this count: pooled over 100 points a fitted sigma
        # is pinned within a few percent, over 7 it can land a factor of 3 off the truth by chance
        # alone. Any reader comparing a fitted sigma against a reference needs the count to know
        # what a discrepancy is worth (see petab_harness.runner.scoring._reference_sigma_flag), and
        # below NOISE_SIGMA_MIN_POINTS the LOCAL verdict is withheld as degenerate too (see
        # upper_bound_cap).
        #
        # Best-effort like every other field here: this is the ONE call in the loop that reaches
        # into the measurement table directly, so a problem without one (or with a shape it cannot
        # read) must cost the count, not the report. A missing count is a supported state — the
        # judge's band falls back to its fixed tolerance — while a raise here fails the commit.
        #
        # ``too_few`` is tracked separately rather than derived from ``n_points``, because the two
        # consumers have to read a missing count differently.
        # ``_measurement_rows_for_sigma`` returns None both for "no rows back this sigma" and for
        # "no measurement table at all", and the REPORTED count stays None for either — 0 would
        # assert a group size we do not have. The local verdict, though, is degenerate whenever no
        # rows were found, so it is withheld there; an EXCEPTION leaves it exactly as it was, since
        # an unreadable count is no evidence about the group's size.
        try:
            rows = R._measurement_rows_for_sigma(problem, pid)
            n_points = None if rows is None else int(len(rows))
            too_few = n_points is None or n_points < NOISE_SIGMA_MIN_POINTS
        except Exception:  # noqa: BLE001 - advisory metadata; never fail a fit for it
            n_points = None
            too_few = False
        entry = {
            "fitted": value,
            "effective_sigma": effective,
            "n_points": n_points,
            # Both verdict inputs are withheld when the group is too small: _entry_inflated keys
            # on upper_bound_cap and _entry_collapsed on fraction_of_data_std, so nulling only one
            # would leave the other still flagging on a degenerate comparison. data_std and the
            # Rice estimate below are still REPORTED — the numbers are informative, the verdict
            # is not. The real bound cap (reconcile.sigma_upper_bound_cap, used when bounds are
            # actually set) is untouched by this; only the gate's view of it is.
            "upper_bound_cap": None if too_few else R.sigma_upper_bound_cap(
                problem, pid, fitted_values=fitted, data_std=data_std),
            "data_std": data_std,
            "fraction_of_data_std": None if too_few else (
                (effective / data_std) if (data_std and effective is not None) else None),
            "data_noise_estimate": data_noise,
            "data_noise_method": data_noise_method,
            "fitted_over_data_noise": (effective / data_noise)
                                      if (data_noise and effective is not None) else None,
        }
        # Per-entry plausibility from the SAME predicates the top-level noise_sigma_inflated/
        # collapsed use (single source of truth, so the agent-facing status can't drift from the
        # gate that fails the run). "unknown" when the observable's std can't be computed (data_std
        # None -> nothing to assess against): report that honestly rather than a reassuring "ok".
        if too_few:
            entry["status"] = "unknown"
        elif data_std is None or effective is None:
            # Either half missing means there is no dimensionally valid comparison to make: no std
            # to judge against, or a noiseFormula that would not reduce to a number. Both are
            # "unknown", NOT the reassuring "ok" the predicates below would fall through to — they
            # return False on a None, which reads as "assessed and fine" (see _entry_inflated,
            # which already documents this entry as unknown rather than inflated).
            entry["status"] = "unknown"
        elif _entry_inflated(entry):
            entry["status"] = "inflated"     # noise absorbing model misfit
        elif _entry_collapsed(entry):
            entry["status"] = "collapsed"    # overfitting the noise for a good-looking score
        else:
            entry["status"] = "ok"
        out[pid] = entry
    return out


def noise_sigma_inflated(report: "dict[str, Any] | None") -> bool:
    """True if any ESTIMATED noise sigma in a ``noise_sigma_report`` is fitted ABOVE its
    data-driven cap — i.e. the noise is absorbing model misfit rather than measuring it. The single
    source of truth for "this sigma is inflated", used by the session's plateau-adequacy gate AND
    the judge's success gate so they can't drift apart. Best-effort: an empty/unreadable report or
    a non-numeric entry is treated as NOT inflated."""
    if not isinstance(report, dict):
        return False
    return any(_entry_inflated(e) for e in report.values() if isinstance(e, dict))


def noise_sigma_collapsed(report: "dict[str, Any] | None") -> bool:
    """True if any ESTIMATED noise sigma is fitted implausibly SMALL — below
    ``constants.NOISE_SIGMA_COLLAPSE_FRACTION`` of its observable's own data std, i.e. the model is
    explaining nearly all the data's variability (overfitting the noise for a suspiciously good BIC).
    The other tail of :func:`noise_sigma_inflated`, and likewise the single source of truth shared by
    the session's adequacy gate and the judge's success gate. Best-effort: an empty/unreadable report
    or a non-numeric ``fraction_of_data_std`` is treated as NOT collapsed."""
    if not isinstance(report, dict):
        return False
    return any(_entry_collapsed(e) for e in report.values() if isinstance(e, dict))


# Agent-facing advice when the fit's noise model is implausible. Surfaced proactively on every
# get_fit_summary (not just at a plateau stop) and echoed by the harness finish gate, so the agent
# learns WHY a good-looking BIC won't be accepted and fixes the mechanism instead of the noise.
NOISE_SIGMA_INFLATED_ADVICE = (
    "an ESTIMATED noise sigma is inflated above its data-driven cap, absorbing the model misfit — "
    "the chi2 goodness-of-fit is not genuine adequacy here. Do NOT widen or lean on the noise; fix "
    "the mechanism so the residuals shrink at a plausible sigma (this result's residual_structure "
    "verdict still flags the structure). A fit resting on an inflated sigma will "
    "NOT be accepted as finished."
)
NOISE_SIGMA_COLLAPSED_ADVICE = (
    "an ESTIMATED noise sigma has collapsed far below the data's own variability — the fit is "
    "overfitting the noise for a good-looking score, not genuine adequacy. Prefer a simpler / more "
    "parsimonious mechanism at a plausible sigma rather than one that drives the residuals to ~0. "
    "A fit resting on a collapsed sigma will NOT be accepted as finished."
)


def noise_sigma_advice_for(inflated: bool, collapsed: bool) -> str | None:
    """Advice from already-computed inflated/collapsed flags (so a caller that needs both the flags
    AND the advice — e.g. :func:`fit_summary` — computes the two predicates ONCE). Inflation is
    reported first when both tails are present, since it is the more common misfit-masking case."""
    if inflated:
        return NOISE_SIGMA_INFLATED_ADVICE
    if collapsed:
        return NOISE_SIGMA_COLLAPSED_ADVICE
    return None


def noise_sigma_advice(report: "dict[str, Any] | None") -> str | None:
    """Concise agent-facing advice for an implausible noise model, or ``None`` when the fit's
    estimated sigmas are plausible."""
    return noise_sigma_advice_for(noise_sigma_inflated(report), noise_sigma_collapsed(report))


def fit_summary(session, engine) -> dict[str, Any]:
    """The current fit metrics, no per-point records — a cheap score read (backs
    ``get_fit_summary``).

    Ensures a simulation dataframe exists, then returns ``{status, neg_log_likelihood, chi2,
    n_data_points, fitted_parameters, parameter_scales, version}`` plus the complexity-
    penalised :func:`goodness_of_fit` scores, ``parameters_at_bound`` (see
    :func:`parameters_at_bound`), ``noise_sigma_report`` (see :func:`noise_sigma_report`), and
    ``residual_structure`` (see :func:`data_access.residual_structure_rollup`).
    ``k`` uses ``n_estimated`` (well-defined even on a simulate-only pass). Pairs fitted values
    with their estimation scale even when a restored/rollback payload lacks the map, so a linear
    value is never misread as a log10 one.

    Both adequacy verdicts ride on this read: it is the score read for every caller that did not
    run the fit — the analysts, the harness finish gate, the final scoring.
    """
    sim = ensure_simulation(session, engine) or {}
    out = {k: sim.get(k) for k in (
        "status", "neg_log_likelihood", "chi2", "n_data_points",
        "n_missing_simulations", "objective_note",
        "fitted_parameters", "parameter_scales", "version",
    )}
    pdf = session.problem.parameter_df
    out["parameter_scales"] = _parameter_scales_for(
        out.get("fitted_parameters"), out.get("parameter_scales"), pdf)
    out.update(goodness_of_fit(
        sim.get("neg_log_likelihood"), sim.get("chi2"),
        int(session.problem.n_estimated), sim.get("n_data_points"),
    ))
    out["parameters_at_bound"] = parameters_at_bound(out.get("fitted_parameters"), pdf)
    _report = noise_sigma_report(session.problem, out.get("fitted_parameters"))
    out["noise_sigma_report"] = _report
    # Proactive plausibility verdict on EVERY read (not just at a plateau stop): a good BIC bought
    # with an inflated/collapsed estimated sigma is not real adequacy and will not be accepted as
    # finished. Surfaced here so the agent can act on it, and read by the harness finish gate.
    _inflated = noise_sigma_inflated(_report)
    _collapsed = noise_sigma_collapsed(_report)
    out["noise_sigma_inflated"] = _inflated
    out["noise_sigma_collapsed"] = _collapsed
    _advice = noise_sigma_advice(_report)  # report-based: quotes the replicate ratio when reliable
    if _advice is not None:
        out["noise_sigma_advice"] = _advice
    # Cheap: the rollup reads the STORED records, no re-simulation. Lazy — data_access imports us.
    from .data_access import residual_structure_rollup

    out["residual_structure"] = residual_structure_rollup(session, sims=None)
    # The PARAMETERISATION counterpart, on the same reasoning: the cheapest ties BIC would pay for
    # (see :func:`identifiability.merge_hints`). They ride on the fit RESULT, which only the caller
    # that RAN the fit holds — so a caller reading the score afterwards (the analysts, the harness
    # finish gate, the final scoring) had no way to see them. Re-serving the stored hints costs
    # nothing: `stored_merge_hints` returns them only while the working problem still IS the one that
    # priced them, and None otherwise. ABSENT when there are none, like `noise_sigma_advice`.
    from .identifiability import stored_merge_hints

    _hints = stored_merge_hints(session)
    if _hints:
        out["merge_hints"] = _hints
    # The run's OUTSTANDING invalid "does adding structure help?" experiment, if any (see
    # :mod:`nesting`): a commit that gained parameters yet fit WORSE than its parent, which for a
    # nested model is impossible. Sticky across the rollback that usually follows, so the harness
    # finish gate can surface it at the ARM step and a finish cannot rest on that comparison.
    # ABSENT when there is none — an omitted key must read as "nothing outstanding".
    _invalid = session.invalid_comparison
    if _invalid:
        out["invalid_comparison"] = _invalid
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

    The reference is the session's best-BIC :class:`~petab_mcp.session.Checkpoint`, set on the
    previous new best — not the raw running minimum. Comparing against
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
    _best = session.best
    best_prior_bic = _flt(_best.bic) if _best else None
    best_prior_version = _best.version if _best else None
    best_prior_nllh = _best.nll if _best else None

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


def _parameter_scales_for(fitted, stored, parameter_df) -> "dict[str, str] | None":
    """Each fitted parameter's ESTIMATION scale, preferring the payload's own map.

    A payload restored from a version carries no map; the live table is exact there, since
    ``apply_version`` has just made it that version's.

    ``None`` rather than an empty map when nothing resolves: a caller testing ``is None`` would
    read ``{}`` as a valid answer and treat every linear value as a known scale, which is the
    misreading this pairing exists to prevent.
    """
    if not fitted or stored:
        return stored
    if parameter_df is None:
        return None
    derived = {pid: str(parameter_df.loc[pid, PARAMETER_SCALE])
               for pid in fitted if pid in parameter_df.index}
    return derived or None


def fit_result_block(
    fitsim: dict[str, Any] | None, improvement: dict[str, Any] | None = None,
    parameter_df: Any = None, problem: Any = None,
) -> dict[str, Any]:
    """The fit fields every fit-bearing result returns (a fit, and a restore).

    Keeps the tool-result shape identical across those paths so an agent reads the same
    keys everywhere. ``fitsim`` is a :func:`fit_and_simulate` payload (or the restored
    simulation for a restore); ``improvement`` is the :func:`compute_improvement` dict.
    Beyond the raw nll/chi2 it also carries the complexity-penalised
    :func:`goodness_of_fit` scores (bic/aic/reduced_chi2/gof p-value) so the agent can see
    when the fit is already at the noise floor and stop adding parameters,
    ``parameters_at_bound`` (see :func:`parameters_at_bound`; empty when ``parameter_df`` is
    not passed) so a parameter pinned to its bound is flagged in the SAME report instead of
    requiring a separate ``get_parameters()`` comparison, the noise-model verdict
    (``noise_sigma_report`` plus ``noise_sigma_inflated`` / ``noise_sigma_collapsed`` and, only
    when implausible, ``noise_sigma_advice`` — the same flat keys :func:`fit_summary` reports, so
    a score read off ``fit`` names things exactly as one read off ``get_fit_summary``), and the
    compact
    ``residual_structure`` verdict the caller stashed on ``fitsim`` (see
    :func:`data_access.residual_structure_rollup`) so the agent learns whether its model's FORM is
    still incomplete without spending a separate ``get_fit_diagnosis`` call — and, on the same
    reasoning for the model's PARAMETERISATION, the ``merge_hints`` the caller stashed there (see
    :func:`identifiability.merge_hints`), present only when a tie is worth taking.
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
        # The multistart count the fit actually ran with (``None`` for a restored version, which
        # did not fit). Reported because `n_starts` OVERRIDES the server default rather than adding
        # to it, so this is how a caller confirms it fitted harder and not weaker.
        "n_starts": fitsim.get("n_starts"),
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
        "parameter_scales": _parameter_scales_for(fitted, fitsim.get("parameter_scales"),
                                                  parameter_df),
        "improvement": improvement or None,
        "parameters_at_bound": parameters_at_bound(fitted, parameter_df),
        # Prefer the report PetabSession.fit already computed and stashed, so the event, the
        # adequacy gate and this block cannot disagree; computed here for the paths that don't
        # stash one (a restore).
        **noise_sigma_block(problem, fitted, report=fitsim.get("noise_sigma_report")),
        # The COMPACT residual-structure verdict only — which observables the sign/runs tests or the
        # replicate lack-of-fit test flag, a one-line summary, and the bounded replicate verdict
        # (`replicate_lack_of_fit`: the counts plus at most REPLICATE_LOF_MAX_REPORTED_POINTS named
        # points). Deliberately not the per-observable table or worst_points: those would ride on
        # EVERY fit result and pile up in the agent's re-sent history, so they stay behind
        # get_fit_diagnosis (issue #113).
        "residual_structure": fitsim.get("residual_structure"),
    }
    # The cheapest parameter ties BIC would pay for (see :func:`identifiability.merge_hints`), ABSENT
    # when there are none — like `noise_sigma_advice` and `nesting_advisory`: this rides on every fit
    # and is re-sent in the agent's history each turn (PR #23), so it is a few lines only when there
    # is something to act on. Only a real fit carries them: a restore and a de-dup no-op ran no
    # optimizer, so they have no optimum to price a tie at, and the finish gate names any tie still
    # outstanding at the ARM step (petab_harness.prompts.FINISH_ARMED_MERGE_TMPL) instead.
    hints = fitsim.get("merge_hints")
    if hints:
        block["merge_hints"] = hints
    block.update(goodness_of_fit(nll, chi2, k, n))
    return block


def noise_sigma_block(
    problem: Any, fitted: dict[str, float] | None, report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The noise-model verdict as it appears on any fit-bearing result.

    The raw ``noise_sigma_report`` plus the ``noise_sigma_inflated`` / ``noise_sigma_collapsed``
    verdict and — ONLY when implausible — ``noise_sigma_advice``, mirroring :func:`fit_summary`'s
    flat keys so a score read off ``fit`` names things exactly as one read off ``get_fit_summary``.

    Shared by :func:`fit_result_block` and the ``fit`` no-op deliberately: because the advice key
    is ABSENT when the noise model is plausible, a result that omits these keys entirely reads to
    an agent as "plausible". Any fit-bearing shape must therefore carry the whole verdict or none
    of it, and the lead has no ``get_fit_summary`` to fall back on. ``report`` short-circuits the
    computation for a caller that already has one.
    """
    if report is None:
        report = noise_sigma_report(problem, fitted)
    inflated = noise_sigma_inflated(report)
    collapsed = noise_sigma_collapsed(report)
    block = {
        "noise_sigma_report": report,
        "noise_sigma_inflated": inflated,
        "noise_sigma_collapsed": collapsed,
    }
    advice = noise_sigma_advice(report)  # report-based: quotes the replicate ratio when reliable
    if advice is not None:
        block["noise_sigma_advice"] = advice
    return block


def build_sim_payload_for_version(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    """The clean simulation blob persisted alongside a version (``simulations.json``).

    Returns ``None`` when there is nothing worth storing (no per-point records), so old
    version dirs and failed/skipped fits simply carry no simulation file.
    """
    from .identifiability import STORED_DIAGNOSIS_KEY

    if not payload or not payload.get("records"):
        return None
    out = {
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
    # The PARAMETER-GEOMETRY verdict of this fit (see identifiability.fit_diagnosis_block). Persisted
    # because it cannot be recomputed later without the built problem the optimizer used: a process
    # that RESUMED the run has no fit record, and re-deriving one there names an AMICI output folder
    # from live state — a full C++ build. The residual verdict beside it is recomputed instead, from
    # the records right here. Omitted when absent, so a key present always means a verdict.
    block = payload.get(STORED_DIAGNOSIS_KEY)
    if block:
        out[STORED_DIAGNOSIS_KEY] = block
    # The NOISE-model report of this fit: which sigmas were estimated, what they came out at, and
    # each one's data-driven cap and fraction of its observable's std. Persisted for the same reason
    # as the diagnosis above — a later reader cannot reproduce it. Recomputing needs the fitted
    # values AND the per-observable data spread of the fit that produced them, and the judge, which
    # is the reader that matters, would otherwise have to RESTORE each version to see it: N restores
    # that mutate the working state and write N entries into the run's own provenance, to answer a
    # question the fit already answered. Small (a handful of numbers per sigma) and omitted when
    # absent, so a key present always means a report.
    report = payload.get("noise_sigma_report")
    if report:
        out["noise_sigma_report"] = report
    return out


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
        # Without these a rollback restores the per-start optima with no way to read them: the
        # `x` vectors are positional. Rollback-then-re-author is exactly the path where reusing
        # them as warm starts matters most (see `previous_optima`).
        "x_names": payload.get("x_names"),
        "x_scales": payload.get("x_scales"),
    }
