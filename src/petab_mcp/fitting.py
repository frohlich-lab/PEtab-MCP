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
import logging
import math
import os
import statistics
import time
from dataclasses import dataclass
from typing import Any

from petab.v2.C import (
    ESTIMATE,
    EXPERIMENT_ID,
    LOWER_BOUND,
    NOISE_DISTRIBUTION,
    NOISE_PARAMETERS,
    NOMINAL_VALUE,
    NORMAL,
    OBSERVABLE_FORMULA,
    OBSERVABLE_ID,
    OBSERVABLE_PARAMETERS,
    UPPER_BOUND,
)

from . import criteria
from . import literals as _literals
from . import reconcile as _reconcile
from .constants import (
    NOISE_SIGMA_MIN_POINTS,
    NEW_BEST_MIN_DELTA_SCORE,
    NOISE_SIGMA_COLLAPSE_FRACTION,
    PARAMETER_AT_BOUND_RELATIVE_TOL,
)
from .engine import build_lock
from .errors import ModelBuildBusyError, ModelUpdateError, PetabMcpError, disk_full_error
from .formatting import finite_float as _flt
from .parameters import structure_fingerprint
from .problem import (
    LOG,
    LOG10,
    PARAMETER_SCALE,
    V1_PREEQUILIBRATION_CONDITION_ID,
    V1_SIMULATION_CONDITION_ID,
    scale,
    split_overrides,
    unscale,
)

# pyPESTO's parameter-table column that tags an inner (hierarchical) parameter.
_PARAMETER_TYPE_COL = "parameterType"


def _env_float(name: str, default: float | None) -> float | None:
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
# can run unbounded and out-idle the MCP transport, which a client then mistakes for a hang.
# Bounding each start makes a full multistart fit's worst-case duration finite and predictable, so
# a client can size its read timeout and stall watchdog to it; one that exports its own values onto
# this server's environment has both sides use the SAME cap + worker count. Kept generous — Fides
# returns best-so-far for a capped start, so too tight a cap would silently degrade fit quality.
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
# SLURM_CPUS_PER_TASK when set, else box minus one core — see _default_fit_workers). A fit is
# assumed to have the whole machine; running the starts in parallel cuts wall-clock to
# ~ceil(n_starts / workers) waves. Set to 1 to force sequential fitting. A client that budgets a
# fit's wall-clock in those waves should export its own value as PETAB_MCP_FIT_N_WORKERS onto this
# server, so its budget matches the real parallelism.
FIT_N_WORKERS = _env_int("PETAB_MCP_FIT_N_WORKERS", _default_fit_workers())

# Expected seconds of starts one dask task runs back to back (env PETAB_MCP_FIT_TASK_SECONDS,
# default 270 = 4.5 min): long against a task's scheduling cost, short enough that concurrent fits
# still take turns every few minutes. A start is expected to take the previous fit's median, or the
# per-start cap FIT_PER_START_TIMEOUT_S without one (see _starts_per_task).
FIT_TASK_SECONDS = _env_float("PETAB_MCP_FIT_TASK_SECONDS", 270.0)

# A fixed number of starts per dask task (env PETAB_MCP_FIT_STARTS_PER_TASK); takes precedence over
# FIT_TASK_SECONDS when set.
FIT_STARTS_PER_TASK = _env_int("PETAB_MCP_FIT_STARTS_PER_TASK", 0)

# AMICI absolute integration tolerance for every candidate fit (env PETAB_MCP_FIT_AMICI_ATOL).
# AMICI's stock 1e-16 is tight enough that on a stiff model the forward-sensitivity solve cannot
# make progress: the objective VALUE evaluates, the GRADIENT comes back non-finite, and Fides
# reads +INF at iteration 0 — a start that reports as failed when really the solver could not
# differentiate there. A REFERENCE fit made outside this server must use the same tolerance, or the
# reference and the candidate are optimized against different objectives and ΔBIC stops being a
# comparison.
FIT_AMICI_ATOL = _env_float("PETAB_MCP_FIT_AMICI_ATOL", 1e-10)

# AMICI's relative integration tolerance and its steady-state relative and absolute tolerances for
# every candidate fit and held-out score (env PETAB_MCP_FIT_AMICI_RTOL,
# PETAB_MCP_FIT_AMICI_RTOL_STEADY_STATE, PETAB_MCP_FIT_AMICI_ATOL_STEADY_STATE). Unset, AMICI's own
# default holds. A problem whose pre-equilibration does not converge at the defaults states its own
# tolerances, and whoever starts the server exports them here, so the server integrates it as the
# problem's own scoring does: at the defaults such a problem can score ~0.2 nats from its converged
# value, and differently on PEtab v1 and v2; a steady-state rtol of 1e-9 brought both to 1.4e-5.
FIT_AMICI_RTOL = _env_float("PETAB_MCP_FIT_AMICI_RTOL", None)
FIT_AMICI_RTOL_STEADY_STATE = _env_float("PETAB_MCP_FIT_AMICI_RTOL_STEADY_STATE", None)
FIT_AMICI_ATOL_STEADY_STATE = _env_float("PETAB_MCP_FIT_AMICI_ATOL_STEADY_STATE", None)


def _apply_amici_solver_defaults(problem) -> None:
    """Set :data:`FIT_AMICI_ATOL`, and each of the tolerances above that is set, on ``problem``'s
    AMICI solver, in place. Best-effort: a problem whose objective exposes no AMICI solver is left
    alone rather than raising, since a missing tolerance degrades a fit while a raise here would
    break every fit path at once.

    Set on the server's solver, before a fit's starts are pickled: AMICI pickles a solver with its
    settings, so a dask or SLURM worker integrates with them without reading the environment."""
    solver = getattr(getattr(problem, "objective", None), "amici_solver", None)
    for setters, value in (
        (("set_absolute_tolerance", "setAbsoluteTolerance"), FIT_AMICI_ATOL),
        (("set_relative_tolerance", "setRelativeTolerance"), FIT_AMICI_RTOL),
        (
            ("set_relative_tolerance_steady_state", "setRelativeToleranceSteadyState"),
            FIT_AMICI_RTOL_STEADY_STATE,
        ),
        (
            ("set_absolute_tolerance_steady_state", "setAbsoluteToleranceSteadyState"),
            FIT_AMICI_ATOL_STEADY_STATE,
        ),
    ):
        setter = next((getattr(solver, s) for s in setters if hasattr(solver, s)), None)
        if setter is not None and value is not None:
            setter(value)


# The warm-start jitter's seed and the multistart's optimizer. Named constants rather than
# parameters: the fit's own log line hard-codes "optimizer=fides", so a signature knob would be
# configurable in name only.
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
# and an agent has read the latter as the former. Re-entering the OTHER basins matters for the
# same reason one notch down: they are the local optima the parent's own multistart paid to find,
# and an additive edit does not destroy them.
# The other half stays FRESH random: an added parameter can create new local optima that none of
# the parent's basins contain, and only an unbiased sample finds those.
WARM_START_FRACTION = 0.5
# Spread of the FALLBACK warm starts, as a fraction of each parameter's box width in ESTIMATION
# scale. Used only to fill the warm budget when the previous fit cannot supply enough DISTINCT
# optima — the session's first fit, or a multistart that converged everything to one point — so the
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


def _starts_per_task(n_starts: int, previous: dict[str, Any] | None) -> int:
    """How many starts one dask task runs, from 1 to ``n_starts``: ``FIT_STARTS_PER_TASK`` when set,
    else as many as fill ``FIT_TASK_SECONDS`` at the median start time of ``previous``, the
    session's last fit (``session.optimization``), or at ``FIT_PER_START_TIMEOUT_S`` without one."""
    times = [s["time"] for s in (previous or {}).get("starts") or [] if s.get("time")]
    expected = statistics.median(times) if times else FIT_PER_START_TIMEOUT_S
    return max(1, int(min(n_starts, FIT_STARTS_PER_TASK or FIT_TASK_SECONDS // expected)))


def _fit_engine(
    n_starts: int,
    shared=None,
    previous: dict[str, Any] | None = None,
    rerun: frozenset[str] = frozenset(),
):
    """The pyPESTO engine a multistart runs on: ``shared``, the server's
    :class:`~petab_mcp.dask_engine.DaskEngine`, when it has one, batched by :func:`_starts_per_task`;
    otherwise up to ``FIT_N_WORKERS`` local worker processes (never more than there are starts), or
    ``None`` (pyPESTO's sequential ``SingleCoreEngine``) for a single effective worker. Both pools
    stop a start past its cap, rerunning the starts in ``rerun`` once, and end a fit whose starts
    stop finishing (:mod:`petab_mcp.multistart`)."""
    if shared is not None:
        return shared.batched(_starts_per_task(n_starts, previous), FIT_PER_START_TIMEOUT_S, rerun)
    workers = max(1, min(FIT_N_WORKERS, int(n_starts)))
    if workers < 2:
        return None
    from .multistart import ProcessEngine

    return ProcessEngine(workers, FIT_PER_START_TIMEOUT_S, rerun)


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
    idx = list(free_indices) if free_indices is not None else list(range(len(x_names)))
    fitted: dict[str, float] = {}
    for i in idx:
        name = x_names[i]
        if name in parameter_df.index:
            fitted[name] = float(
                unscale(float(x_values[i]), parameter_df.loc[name, PARAMETER_SCALE])
            )
    return fitted


def table_fitted_values(parameter_df) -> dict[str, float]:
    """``{estimated parameter id: nominalValue}``: a fit writes every estimated parameter's value
    back into the table, so this is that fit's ``fitted_parameters``. A version keeps them by name
    only in its ``parameters.tsv``, and a restore reads them back through here."""
    if ESTIMATE not in parameter_df.columns:
        return {}
    return {
        str(pid): float(parameter_df.loc[pid, NOMINAL_VALUE])
        for pid, est in parameter_df[ESTIMATE].items()
        if est
    }


def pypesto_problem(problem, importer_kwargs: dict[str, Any], *, validate: bool = True):
    """The pyPESTO problem for ``problem`` (a :class:`problem.WorkingProblem`), in estimation
    coordinates: pyPESTO's v2 importer takes each row's ``parameterScale`` as its
    ``parameter_scales`` and hands them to AMICI's simulator, so every pyPESTO problem the server
    builds — the fit's, a held-out score's, a transfer refit's — is optimised on the same
    coordinates. Inner (hierarchically solved) parameters are left out: they are solved on a linear
    scale, and the importer refuses another one for them.

    ``validate`` runs libpetab's lint on the way in: the session's own problem was linted on every
    edit that produced it, a held-out view was not. Passes ``hierarchical`` and the output target
    through ``importer_kwargs``.

    The one place a module is compiled — ``create_problem`` builds into a target folder that is not
    there yet — so it runs under :func:`engine.build_lock`.
    """
    import pypesto.petab
    from petab import v2

    pdf = problem.parameter_df
    scales = pdf[PARAMETER_SCALE]
    if importer_kwargs.get("hierarchical") and _PARAMETER_TYPE_COL in pdf.columns:
        scales = scales[pdf[_PARAMETER_TYPE_COL].isna()]
    petab_problem = problem.to_petab()
    # AMICI simulates every experiment in the table, and one without measurements comes back as an
    # rdata with no outputs, which every reader of the rdatas takes for a failed simulation. It adds
    # nothing to the likelihood, so AMICI is not shown it: a fixed-θ held-out view keeps all the
    # training experiments, and a v2 problem may ship unmeasured ones.
    measured = set(problem.measurement_df[EXPERIMENT_ID])
    petab_problem.experiment_tables = [
        v2.ExperimentTable([e for e in petab_problem.experiments if e.id in measured])
    ]
    importer = pypesto.petab.PetabImporter(
        petab_problem, validate_petab=validate, parameter_scales=dict(scales), **importer_kwargs
    )
    with build_lock(importer_kwargs["output_folder"], importer_kwargs["model_name"]):
        return importer.create_problem()


def _build_hierarchical_problem(session, importer_kwargs: dict[str, Any], petab_problem=None):
    """Best-effort hierarchical pyPESTO problem for ``session``'s current problem.

    Returns ``(pypesto_problem, inner)`` where ``inner`` maps each inner parameter id to its
    pyPESTO type (see :func:`reconcile.classify_inner_parameters`), or ``(None, {})`` when the
    problem has no cleanly-separable inner parameters, has an observable whose noise is not normal,
    or pyPESTO rejects the hierarchical setup.

    The whole win is optimization efficiency/robustness: the inner parameters (noise sigmas,
    observable-scaling factors) are solved analytically per objective evaluation instead of by
    the outer multistart, shrinking the outer search (e.g. 13 estimated params → 5 outer). It
    does NOT change what the parameters converge to — a sigma's analytic optimum is
    the same RMS the joint optimizer finds — so the fit's nllh/identity is unchanged. pyPESTO
    solves inner parameters in linear space whatever their ``parameterScale``.

    Tagging is applied to a COPY of the parameter table handed to pyPESTO; the session's own
    table is left untouched (the agent keeps seeing ordinary parameters, ``n_estimated``/``k``
    stays correct because inner parameters are still estimated there, and the fit-cache key
    is unaffected). On ANY failure this returns ``(None, {})`` so the caller falls back to the
    ordinary joint fit — hierarchical never risks a fit that would otherwise succeed.
    """
    # `petab_problem` is the AMICI-facing problem (possibly a flattened copy); classification
    # and the parameter table come from the SESSION's problem, which shares the parameter table.
    base = petab_problem if petab_problem is not None else session.problem
    inner = _reconcile.classify_inner_parameters(session.problem)
    if not inner:
        return None, {}
    # pyPESTO refuses a non-normal noise distribution only on an observable that HAS inner
    # parameters, and scores every other observable as normal on the linear scale whatever its
    # distribution (`calculate_quantitative_result`): a log-normal observable beside an
    # inner-solved sigma comes out with a wrong likelihood, silently (5.8 nll on a two-observable
    # toy). Such a problem is fitted jointly.
    if (session.problem.observable_df[NOISE_DISTRIBUTION] != NORMAL).any():
        return None, {}
    try:
        from pypesto.hierarchical.petab import correct_parameter_df_bounds

        df = session.problem.parameter_df.copy(deep=True)
        df[_PARAMETER_TYPE_COL] = None
        for pid, typ in inner.items():
            df.loc[pid, _PARAMETER_TYPE_COL] = typ
        # A sigma's inner solve needs [0, inf]; scaling/offset keep their box.
        df = correct_parameter_df_bounds(df)

        view = copy.copy(base)  # shallow: shares the model and the design tables
        view.parameter_df = df
        return (
            pypesto_problem(view, {**importer_kwargs, "hierarchical": True}, validate=False),
            inner,
        )
    except ModelBuildBusyError:
        # Not a hierarchical failure: falling back would fit jointly once the other build lands.
        raise
    except Exception:  # noqa: BLE001 - hierarchical is best-effort; fall back to a joint fit
        session.logger.warning(
            "hierarchical optimization setup failed (%s inner params: %s); "
            "falling back to a joint fit",
            len(inner),
            sorted(inner),
            exc_info=True,
        )
        return None, {}


def _inner_fitted_values(pypesto_problem, best, inner: dict[str, str]) -> dict[str, float]:
    """Analytically-solved inner-parameter values from a hierarchical fit result, as
    ``{parameter_id: linear value}`` — merged into ``fitted`` alongside the outer parameters so
    the parameter count ``k`` and the value writeback both cover every estimated parameter.

    pyPESTO returns ``best["inner_parameters"]`` positionally; the calculator's
    ``get_interpretable_inner_par_ids()`` gives the matching id order, and each value is linear.
    Raises :class:`ModelUpdateError` unless every id in ``inner`` comes back: the table would keep
    its pre-fit values beside a score taken at the inner optimum, and a pass with no fit scores the
    table's sigmas and scalings as they stand (see :func:`estimation_problem`). Failing the fit
    reverts the edit, so no version records a score its table does not hold.
    """
    try:
        values = best.get("inner_parameters")
        ids = pypesto_problem.objective.calculator.get_interpretable_inner_par_ids()
        found = {pid: float(v) for pid, v in zip(ids, values)} if len(ids) == len(values) else {}
    except Exception:  # noqa: BLE001 - no values, or pyPESTO's API moved: reported below
        found = {}
    missing = sorted(set(inner) - set(found))
    if missing:
        raise ModelUpdateError(
            "the fit ran, but the server could not read the fitted values of "
            f"{', '.join(missing)} back from pyPESTO's result, so it cannot write the optimum "
            "into the parameter table. This is a server-side failure, not a problem with the "
            "model: do not re-author it."
        )
    return found


def _write_pypesto_result(session, result, problem_hash: str) -> str | None:
    """Write the pyPESTO ``Result`` to a per-fit HDF5 file; return its path (or ``None``).

    The file is cached under ``<provenance>/cache/`` keyed by the *problem* hash — model
    SBML **and** parameter table, so fits that differ only in bounds/scale don't collide —
    and is copied into a version directory when that fit is snapshotted. Best-effort: the
    objective (an AMICI model) is not serialised (``problem=False``), and any storage
    failure is logged and swallowed so it never fails a fit; the JSON traces remain the
    primary, always-present artifact.

    A full disk is the exception, and is looked for BEFORE the write: h5py segfaults closing a file
    the disk refused, which no ``except`` catches, and took the server down in 15 runs of one sweep
    (PEtab-MCP#83). It raises :class:`~petab_mcp.errors.InfrastructureError` instead.
    """
    if (disk_full := disk_full_error(None, session.provenance.root)) is not None:
        raise disk_full
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
    exitflag_counts: dict[str, int] = {}
    for rank, r in enumerate(olist):
        rid = _get(r, "id")
        xvec = _get(r, "x")
        # Aggregate the enumerated per-start outcome instead of storing one prose row per start:
        # the converged-vs-truncated question is a HISTOGRAM question ("what fraction of starts
        # hit FTOL vs the MAXTIME boundary"), and n_starts copies of `message` text answered it
        # only via fragile grepping while dominating the stored payload. str() of the enum keeps
        # the label stable across fides versions without importing its ExitFlag type here.
        flag = _get(r, "exitflag")
        if flag is not None:
            key = str(flag)
            exitflag_counts[key] = exitflag_counts.get(key, 0) + 1
        try:
            xlist = [float(v) for v in xvec] if xvec is not None else None
        except Exception:  # noqa: BLE001 - x shape drift; spread just becomes unavailable
            xlist = None
        starts.append(
            {
                "start": rank,
                "id": str(rid) if rid is not None else None,
                "fval": _flt(_get(r, "fval")),
                "time": _flt(_get(r, "time")),
                "n_fval": _int(_get(r, "n_fval")),
                "n_grad": _int(_get(r, "n_grad")),
                "n_hess": _int(_get(r, "n_hess")),
                # Full scaled parameter vector at this start's optimum — the raw material for the
                # multistart-spread identifiability signal (:mod:`identifiability`). Order matches the
                # payload's x_names (set by the caller from problem.x_names).
                "x": xlist,
            }
        )
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
            trace.append(
                {
                    "start": rank,
                    "iteration": j,
                    "fval": _flt(v),
                    "time": _flt(times[j]) if j < len(times) else None,
                }
            )
    return {
        "n_starts": len(starts),
        "starts": starts,
        "trace": trace,
        # Sorted for stable JSON; most common outcome first so the headline reads at a glance.
        "exitflag_counts": dict(sorted(exitflag_counts.items(), key=lambda kv: -kv[1])),
    }


def needs_flattening(problem) -> bool:
    """Whether AMICI needs a FLATTENED copy of ``problem`` — i.e. whether any observable/noise
    parameter override differs WITHIN one (observable, experiment) pair.

    That is the actual AMICI limitation (its v2 importer refuses such a problem outright):
    per-experiment overrides are handled natively, so only TIMEPOINT-specific overrides — two
    rows of the same observable in the same experiment carrying different overrides — force
    ``flatten_timepoint_specific_output_overrides``. Grouping by observable alone would flatten
    problems whose overrides AMICI takes as shipped, multiplying their observables (measured:
    20 -> 101, 3 -> 123, 4 -> 22); of the problems this was measured on, only one genuinely
    trips this predicate.

    Flattening is an AMICI implementation detail, so it happens HERE, on a copy, at the compile
    boundary (:func:`amici_facing_problem`) — the problem the session holds, the agent reads and
    the diagnostics consume stays unflattened.
    """
    mdf = getattr(problem, "measurement_df", None)
    if mdf is None:
        return False
    keys = [OBSERVABLE_ID, EXPERIMENT_ID]
    if V1_SIMULATION_CONDITION_ID in mdf.columns:
        # A v1 table — a published problem can be checked before anything upgrades it —
        # simulates once per (simulation, pre-equilibration) condition pair.
        keys = [
            OBSERVABLE_ID,
            *(
                c
                for c in (V1_SIMULATION_CONDITION_ID, V1_PREEQUILIBRATION_CONDITION_ID)
                if c in mdf.columns
            ),
        ]
    # Whether a NUMERIC override counts differs by column, and the difference is AMICI's
    # internals, not a style choice:
    #
    # * noiseParameters: a single numeric sigma never enters the parameter mapping at all --
    #   AMICI writes it straight into the ExpData as that datapoint's standard deviation, so
    #   per-timepoint numbers are already exact, and AMICI's own check lets a group of them
    #   through. MEASURED under v1, not assumed: on a problem with 4 groups of 4 distinct numeric
    #   sigmas each, the nominal llh was bit-identical flattened and unflattened. Every other
    #   numeric-sigma problem would be flattened for nothing, which is not free: it multiplies the
    #   observable count.
    #   Only a group whose overrides are ALL single numbers is exempt; one symbolic override
    #   among them is a mapping AMICI refuses to vary.
    # * observableParameters: these DO become mapped parameters, one value per experiment, so a
    #   number that varies between timepoints of one experiment has nowhere to live. No problem
    #   this was measured on does that, so it is untested rather than measured -- which is
    #   exactly why it errs toward flattening: an unnecessary flatten costs observables, a missed
    #   one cannot be imported at all.
    for col in (OBSERVABLE_PARAMETERS, NOISE_PARAMETERS):
        if col not in mdf.columns:
            continue
        sub = mdf[[*keys, col]].dropna(subset=[col])
        cells = sub[col].astype(str)
        for _, group in cells.groupby([sub[k] for k in keys], sort=False, dropna=False):
            distinct = set(group)
            if len(distinct) < 2:
                continue
            if col == NOISE_PARAMETERS and all(_single_number(c) for c in distinct):
                continue
            return True
    return False


def _single_number(cell) -> bool:
    overrides = split_overrides(cell)
    return len(overrides) == 1 and isinstance(overrides[0], float)


def amici_facing_problem(session):
    """The problem AMICI is allowed to see: ``session.problem`` itself, or — only when
    :func:`needs_flattening` says AMICI cannot take it as-is — a view with flattened observable
    and measurement tables.

    Cached on the session under ``problem_hash`` (flattening a large problem re-derives every
    table). AMICI's ``flatten_timepoint_specific_output_overrides`` mutates the problem it is
    given, which is why it runs on a deep copy: the session's problem is the single source the
    agent-facing reads, the reconcile and the residual machinery all consume, and none of them
    may ever see flattened observable ids. Simulation frames coming BACK from a flattened build
    are mapped to the original ids by ``unflatten_simulation_df`` -- but only AFTER the
    simulation frame has been paired with the measurements IN THIS VOCABULARY. Unflattening
    first and merging against the natural measurements mispairs every override-carrying row
    (see ``AmiciEngine._simulation_records``, which measures it).

    Returns ``(problem, flattened)``.
    """
    if not needs_flattening(session.problem):
        return session.problem, False
    key = session.problem_hash()
    cached = getattr(session, "_amici_problem_cache", None)
    if cached is not None and cached[0] == key:
        return cached[1], True
    from amici.importers.petab import flatten_timepoint_specific_output_overrides

    flat_petab = copy.deepcopy(session.problem.to_petab())
    flatten_timepoint_specific_output_overrides(flat_petab)
    flat = session.problem.replace(
        observable_df=flat_petab.observable_df, measurement_df=flat_petab.measurement_df
    )
    session._amici_problem_cache = (key, flat)
    session.logger.info(
        "flattening timepoint-specific output overrides for the AMICI build only "
        "(%d -> %d observables); the problem itself stays unflattened",
        len(session.problem.observable_df),
        len(flat.observable_df),
    )
    return flat, True


def estimation_problem(session, engine, *, solve_inner: bool = True):
    """The pyPESTO problem for the session's CURRENT estimation problem, cached on the engine.

    ONE AMICI model and solver set-up now serve both jobs that need a likelihood: the multistart
    fit, and every plain evaluation of the objective (:meth:`AmiciEngine.evaluate_objective` /
    :meth:`AmiciEngine.residuals`). That unification is the fix for a class of bug, not just an
    optimisation — while the fit and the evaluation each built their own AMICI model and their
    own solver, the two could disagree about the very same likelihood at the very same
    parameters, and did: a fit converged at nll 266.62 while re-simulating its optimum
    failed to integrate past t=22 and reported NaN. Sharing the objective makes "the nll of
    this fit" a single number computed one way.

    It also halves the compiling: the fit used to build ``petab_mcp_pp_<hash>`` while the engine
    built ``petab_mcp_<hash>`` from the same SBML — two full C++ builds of one structure, ~80s
    each on a cluster node.

    Cached on ``engine`` under :meth:`session.problem_hash` and ``solve_inner`` — the model AND
    the parameter table, since pyPESTO bakes bounds/scales/fixed values into the problem at
    construction, so a bounds edit (or the nominal-value write-back after a fit) must rebuild it.
    Rebuilding is cheap relative to a compile: the AMICI module is keyed on
    :meth:`session.module_hash`, which a bounds or value edit does not move, and is reused.

    ``solve_inner`` (the fit's objective) prefers the hierarchical objective, which solves the
    noise sigmas and observable scalings at every evaluation, and still falls back to the plain
    objective when the problem has no inner parameters or the hierarchical build fails.
    ``solve_inner=False`` evaluates at the table's own values: a pass with no fit scores the
    table's sigmas and scalings as they stand. Solving them instead scored a shipped toy, delivered
    at ``sigma_B`` = 0.3, 10.7 deviance units better than its own table (#8).

    Returns ``(problem, inner)`` — see :func:`_build_hierarchical_problem` for the latter
    (``{}`` for an ordinary joint problem).
    """
    key = (session.problem_hash(), solve_inner)
    cached = getattr(engine, "_estimation_problem_cache", None)
    if cached is not None and cached[0] == key:
        return cached[1], cached[2]

    h = session.module_hash()
    # Charge the (one-time) model build to the compile bucket only the FIRST time a structure
    # is seen: rebuilding the pyPESTO problem for a changed parameter table reuses the compiled
    # module and must not count as a compile.
    first_build = h not in engine._pypesto_hashes
    t0 = time.perf_counter()
    importer_kwargs = engine.pypesto_importer_kwargs(h)
    # Prefer a hierarchical objective (inner-solve the noise sigmas / observable-scaling
    # factors, shrinking the outer search) when the problem has cleanly-separable inner
    # parameters; fall back to the ordinary joint objective otherwise.
    amici_problem, _ = amici_facing_problem(session)
    problem, inner = (
        _build_hierarchical_problem(session, importer_kwargs, petab_problem=amici_problem)
        if solve_inner
        else (None, {})
    )
    if problem is None:
        # Not linted again: every edit that produced this problem was (see validation).
        problem = pypesto_problem(amici_problem, importer_kwargs, validate=False)
        inner = {}
    # After the branch, so the hierarchical and joint objectives are integrated identically.
    _apply_amici_solver_defaults(problem)
    if first_build:
        session.compile_seconds += time.perf_counter() - t0
        session.n_compiles += 1
        engine._pypesto_hashes.add(h)

    engine._estimation_problem_cache = (key, problem, inner)
    return problem, inner


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
    ``x_free``/``lb``/``ub`` are the FREE coordinates on the estimation scale — pyPESTO's own
    (see :func:`pypesto_problem`), and the space the FIM is expressed in. The fitted point is
    carried here rather than re-read from ``parameter_df[nominalValue]`` so an agent edit to the
    table cannot silently move the point a diagnosis expands around.

    The two fingerprints are COMPARISONS only — never keys to rebuild anything with — and they
    answer different questions, which is why both are here. ``structure_hash``
    (:func:`parameters.structure_fingerprint`) is the model's equations plus the estimated-parameter
    set, so it says whether the geometry frozen here still describes the model in front of the
    agent; a diagnosis is STALE exactly when that moved. ``problem_hash`` also covers the parameter
    table, so it additionally moves when the working table alone does — which happens on the de-dup
    no-op, where the same structure is re-authored with a new starting guess and no fit runs. That
    is not a stale diagnosis: the optimum, the structure and HEAD's fit are all still this record's.

    No AMICI output folder can be named from anything here: the module hash is deliberately absent,
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


def freeze_fit(
    session, problem, best, *, hierarchical: bool, optimization: dict[str, Any] | None
) -> FitRecord:
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
            session.problem.model.sbml_model, session.problem.parameter_df
        ),
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
    full: list[float] = []
    for name, sc in zip(problem.x_names, problem.x_scales):
        val = _flt(parameter_df.loc[name, NOMINAL_VALUE]) if name in parameter_df.index else None
        if val is None:
            # No usable nominal value (a parameter pyPESTO knows about that the table doesn't,
            # or a NaN cell): leave pyPESTO's own value in place rather than silently zeroing.
            fixed = dict(zip(problem.x_fixed_indices, problem.x_fixed_vals))
            full.append(float(fixed.get(len(full), 0.0)))
            continue
        full.append(float(scale(val, sc)))
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

    Returns ``[]`` when there is no usable previous fit: no payload (the first fit), or one
    written before ``x_names``/``x_scales`` were recorded — remapping without the scales would
    silently mix log10 and linear coordinates, which is worse than not warm-starting at all.
    Starts whose objective or vector is non-finite are dropped: a failed start is not a basin.
    """
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
                point[cur_i] = float(scale(unscale(value, prev_sc), cur_sc))
        except (IndexError, KeyError, TypeError, ValueError):
            continue  # a malformed/short vector is one unusable start, not a broken fit
        out.append(point)
    return out


def warm_start_guesses(problem, parameter_df, n_starts: int, optimization=None):
    """``WARM_START_FRACTION`` of the multistart's start points, seeded from the PREVIOUS fit.
    Returns a ``(n_warm, dim_full)`` array — pyPESTO fills the first ``n_warm`` starts from it
    and samples the remaining ``n_starts - n_warm`` freshly.

    The warm starts are seeded from the PREVIOUS fit's optima — and any parameter the previous
    fit did not have (one this edit just introduced) is NOT left at the reconcile-minted default.
    A minted mid-range default enters the model FULLY ACTIVE, which is the opposite of a nested
    comparison: measured on one problem, additive edits warm-"started" at objective_before +1220
    and +2740 against parents at -158, so every such comparison reported a false negative and
    agents responded by buying 300-400 random starts per fit.

    A new parameter's NEUTRAL value depends on its role, which cannot be inferred: a gate
    neutralises at its LOWER bound (mechanism off), a transfer/pass-through rate at its UPPER
    bound (no delay), and unit scale (1.0) is the common declared convention. So the warm rows
    cover all three, in that order of likelihood: row 0 puts every new parameter at UNIT scale
    (the declared convention, and what the reconcile usually mints), row 1 at its lower bound,
    row 2 at its upper bound. Where one edit adds SEVERAL new parameters, further rows vary one
    parameter at a time against unit for the others -- enumerated, not sampled, because random
    draws over 3^k combinations miss the case that matters more often than not.

    Order is load-bearing: the candidates are capped at half the warm budget and truncated from
    the FRONT, so on a small budget it is the unit row that survives. The guarantee that one warm
    start reproduces the parent exactly -- which is what keeps a nested edit from fitting worse
    than its parent (see :mod:`nesting`) -- therefore holds once the budget affords the relevant
    candidate: all three uniform rows need n_starts >= 12 at WARM_START_FRACTION 0.5. Below that
    only unit is tried, which is the single best guess but not a proof.

    The neutral candidates come first (see the paragraph above); the previous multistart's OTHER
    optima follow, best-first (see :func:`previous_optima`), deduplicated against each other. Those are real, converged basins
    of the parent likelihood — under an additive edit they stay basins — so re-entering them is a
    far better use of the warm half than perturbing one point at random. Only when the previous
    fit cannot supply enough distinct optima (the first fit, or a multistart that converged
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
    if any(
        n not in parameter_df.index or _flt(parameter_df.loc[n, NOMINAL_VALUE]) is None
        for n in (problem.x_names[i] for i in free)
    ):
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

    # The neutral candidates, then x0 itself, then the previous fit's other optima, best-first.
    # Deduplicated on the FREE coordinates: a multistart that converged many starts to
    # the same optimum would otherwise spend the warm half re-running one optimization. Compared
    # after clipping, so two points that differ only outside the box count as one.
    # NEUTRAL CANDIDATES for parameters this edit introduced, each built from x0 -- the PARENT'S
    # OWN OPTIMUM -- so one of them reproduces the parent exactly. They must be derived from x0
    # and not written onto the other optima's rows: rows 1..n are DIFFERENT basins, so pairing
    # the upper-bound candidate with the parent's 2nd-best point reproduces that point, not the
    # parent, and the nested guarantee is lost for every parameter whose neutral value is not the
    # lower bound (the RNA delay chain is exactly that case).
    #
    # Which value is neutral cannot be inferred from the symbol: a GATE neutralises at its lower
    # bound (mechanism off), a TRANSFER/pass-through rate at its upper bound (no delay), and unit
    # scale is the common declared convention. So all three are offered and the optimizer decides.
    # An infinite bound is not a usable start (clipping cannot repair -inf, which IS the bound),
    # so that candidate falls back to unit scale.
    prev_names = set((optimization or {}).get("x_names") or [])
    new_free = [i for i in free if str(problem.x_names[i]) not in prev_names] if prev_names else []
    candidates: list = []
    if new_free:
        pos = {i: free.index(i) for i in new_free}
        unit_of = {
            i: float(np.clip(float(scale(1.0, str(problem.x_scales[i]))), lb[pos[i]], ub[pos[i]]))
            for i in new_free
        }

        def _row(pick_of):
            row = np.array(x0, dtype=float)
            for i in new_free:
                j = pos[i]
                v = {"lb": lb[j], "ub": ub[j], "unit": unit_of[i]}[pick_of[i]]
                row[i] = v if np.isfinite(v) else unit_of[i]
            return row

        # UNIT first: it is the declared convention and the value the reconcile usually mints,
        # so it is the likeliest neutralisation and must survive a small warm budget.
        picks = [{i: p for i in new_free} for p in ("unit", "lb", "ub")]
        # One edit can add parameters with DIFFERENT neutral directions (a delay chain's
        # pass-through rate neutralises at ub while a gate added in the same batch neutralises at
        # lb), and no uniform row neutralises both. ENUMERATED, not sampled: 5 iid draws over 3^k
        # combinations cover the k=3 case only ~17% of the time and can duplicate the uniform
        # rows. Varying ONE parameter against unit for the others is 2k rows and covers every
        # case where a single parameter's neutral direction differs from the rest -- which is the
        # realistic shape, since an edit adds one mechanism whose gate differs from the defaults.
        if len(new_free) > 1:
            for i in new_free:
                for side in ("lb", "ub"):
                    picks.append({j: (side if j == i else "unit") for j in new_free})
        # Never spend more than half the warm budget on candidates: they all sit on x0, so past
        # a point they crowd out the parent's OTHER basins for no gain. At n_warm < 2 that leaves
        # one candidate (unit), which is the single best guess.
        candidates = [_row(pk) for pk in picks[: max(1, n_warm // 2)]]

    seen: set[tuple] = set()
    guesses: list = []
    # Candidates FIRST: they are the nested-comparison guarantee, and a previous optimum that
    # displaces one is a worse use of the warm budget than the point that reproduces the parent.
    # With no new parameters they all collapse to x0 and dedup leaves the original behaviour.
    for point in [*candidates, x0, *previous_optima(problem, optimization, x0)]:
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
        # Scatter around the nominal point -- the PARENT'S optimum -- not guesses[0], which is
        # now a candidate row sitting at a BOUND; perturbing that and clipping back would pile
        # roughly half the filler starts exactly onto the bound. CLIPPED, because x0 itself can
        # sit outside the box (a `parameters` edit in the same call can narrow a bound away from
        # the minted nominal value), and an out-of-box base would put every filler on a bound for
        # the same reason.
        anchor = _clipped(np.array(x0, dtype=float))
        base = anchor[free]
        while len(guesses) < n_warm:
            row = np.array(anchor, dtype=float)
            row[free] = np.clip(base + rng.normal(0.0, 1.0, size=len(free)) * spread, lb, ub)
            guesses.append(row)
    return np.asarray(guesses)


def parent_starts(problem, guesses, optimization) -> frozenset[str]:
    """The ids pyPESTO gives the warm starts seeded at the parent's optimum: the rows of
    :func:`warm_start_guesses` that hold every parameter of ``optimization``, the previous fit, at
    its fitted value (the neutral candidates and the nominal point). One of them keeps a nested
    edit's fit within reach of its parent's, so a fit reruns one stopped past its cap
    (:class:`~petab_mcp.multistart.CappedEngine`). Empty without a previous fit."""
    import numpy as np

    previous = set((optimization or {}).get("x_names") or [])
    shared = [i for i, name in enumerate(problem.x_names) if name in previous]
    if not shared or not len(guesses):
        return frozenset()
    at = guesses[0][shared]
    return frozenset(str(j) for j, row in enumerate(guesses) if np.array_equal(row[shared], at))


def _run_fit(session, engine, n_starts: int) -> dict[str, Any]:
    """Estimate the free parameters with pyPESTO and write the best fit back.

    Writes the best start's (unscaled) values into ``nominalValue``. Returns the fit's
    ``neg_log_likelihood`` (pyPESTO ``fval``), the fitted values, and the objective at the
    pre-fit nominal values (for improvement tracking). No snapshot, no simulation — callers
    own those.
    """
    import numpy as np
    import pypesto.optimize

    from .engine import _ensure_swig_on_path

    _ensure_swig_on_path()

    log = session.logger
    # objective at the current nominal values, for improvement tracking
    # May be None: `evaluate_objective` reports a non-finite likelihood as a MISSING score, and
    # an unstable model at its pre-fit nominal values legitimately has one. That is not a reason
    # to refuse to fit it — the multistart samples its own start points and can still find a
    # good optimum — so carry the missing value through (compute_improvement and _fmt both
    # handle None) instead of `float(None)`, which would abort the fit and, via
    # `_compile_fit_failure_message`, tell the agent its model is ill-posed for estimation.
    before = engine.evaluate_objective()
    objective_before = _flt(before["neg_log_likelihood"])

    # The SAME problem the engine evaluates the objective against (see estimation_problem):
    # one AMICI model, one solver, one notion of this model's likelihood.
    problem, inner = estimation_problem(session, engine)
    hierarchical = bool(inner)
    if hierarchical:
        log.info("fit: hierarchical objective (inner params: %s)", sorted(inner))

    free = getattr(problem, "x_free_indices", None)
    n_free = len(free) if free is not None else len(problem.x_names)
    # WARM STARTS: seed half the multistart from the PREVIOUS fit — its best optimum (which the
    # write-back left in the nominal values) plus its other converged optima, remapped onto this
    # problem — so a nested edit's fit begins at points that already reproduce its parent's
    # likelihood, and a worse result means the structure, not the search (see
    # `warm_start_guesses` and :mod:`nesting`). `session.optimization` still holds that previous
    # fit here; `fit_and_simulate` replaces it only after this returns, and a rollback restores
    # the version's own.
    #
    # This used to be SKIPPED when the nominal point had no finite objective ("an unstable model
    # there has nothing worth starting from"). That was true when the only warm start was the
    # nominal point itself, and false now: a non-finite objective there is the SIGNATURE of a
    # newly-minted parameter entering fully active (a mid-range default making the ODE
    # non-integrable, or driving a log observable to -inf), and the neutral candidates are
    # exactly the points that switch that mechanism off and recover the parent's integrable
    # model. Skipping them threw away the rescue in the one case it was built for -- an added
    # delay chain, abandoned as "numerically invalid", took this path.
    #
    # `problem` is cached on the engine, so ALWAYS assign — an empty array clears guesses left by
    # a previous fit rather than silently reusing them.
    # ...but ONLY when there is something to rescue the point WITH. With no newly-added
    # parameter there are no candidates, so a non-finite objective at the nominal point means
    # exactly what it used to: the warm half would be spent jittering around a region that just
    # failed, where the fresh sample is the only way in.
    _has_new = bool(set(problem.x_names) - set((session.optimization or {}).get("x_names") or []))
    guesses = (
        warm_start_guesses(
            problem, session.problem.parameter_df, n_starts, optimization=session.optimization
        )
        if (objective_before is not None or (_has_new and session.optimization))
        else np.zeros((0, len(problem.x_names)))
    )
    problem.x_guesses_full = guesses
    # Parallelise the multistart across worker processes when more than one is warranted; None runs
    # pyPESTO's sequential engine. Sized so a client's fit budget (ceil(n_starts/workers) waves)
    # matches the real wall-clock — see FIT_N_WORKERS. A server with a cluster runs every start
    # there instead.
    pp_engine = _fit_engine(
        n_starts,
        engine.fit_engine,
        session.optimization,
        parent_starts(problem, guesses, session.optimization),
    )
    workers = getattr(pp_engine, "n_procs", 1) or 1
    if engine.fit_engine is not None:
        workers = f"dask({pp_engine.starts_per_task} starts per task)"
    log.info(
        "fit: start optimizer=fides n_starts=%d warm_starts=%d workers=%s n_free=%d "
        "sbml=%s objective_before=%s",
        n_starts,
        len(guesses),
        workers,
        n_free,
        session.sbml_hash(),
        _fmt(objective_before),
    )
    # Fides: a Newton trust-region optimizer that exploits AMICI's sensitivities; more
    # robust than the default L-BFGS-B on stiff ODE likelihoods. MAXTIME caps each start's
    # wall time (FIT_PER_START_TIMEOUT_S) so a stiff start can't run unbounded and hang the
    # transport; Fides returns that start's best-so-far when the cap is hit.
    #
    # WARNING, not INFO. At INFO fides emits one line PER ITERATION PER START: a single
    # session wrote a 1.85 GB server.log, and ten concurrent sessions put ~18 GB on a SHARED
    # filesystem. That filesystem once hit its quota and five concurrent jobs died -- another
    # job was the proximate cause, but this was the server's own contribution to the pressure.
    #
    # This level is the only lever that reaches the volume. fides names each optimizer's logger
    # `fides_<n>` (fides/logging.py) with its OWN StreamHandler, so `logging_setup`'s handler on
    # the logger named "fides" never catches them; and the fits run in pyPESTO WORKER PROCESSES,
    # whose stderr is inherited -- which is how the output reached server.log rather than
    # pypesto.log in the first place. A parent-side handler cannot fix either half.
    #
    # KNOWN COST, and the reason this is safe anyway: WARNING drops the CONVERGENCE stopping
    # reasons, not just the iteration trace. In fides/minimize.py the FTOL/XTOL/GTOL messages
    # are `logger.info` (623/632/641/649) while only MAXITER/MAXTIME/DELTA_TOO_SMALL are
    # `logger.warning` (673/685/693), so grepping the text log after this change would see
    # truncations and no convergences -- reporting ~0% converged and inverting the very
    # converged-vs-truncated reading that tells a structural gap from an under-converged fit.
    #
    # Do not read that from the log. Every start's outcome is recorded STRUCTURALLY: pyPESTO
    # returns each start's `exitflag` (an enumerated ExitFlag.FTOL / EXCEEDED_BOUNDARY / ...),
    # and `collect_optimization` below aggregates them into `exitflag_counts` — the
    # converged-vs-truncated histogram the analysis actually needs, reported in the fit block
    # and stored with the version. That is the durable source and it is strictly better than
    # parsing prose, which is why the level can go all the way to ERROR: the WARNING-level
    # truncation messages are the same information as the EXCEEDED_BOUNDARY flags, line by
    # line, n_starts times per fit. `get_fitting_log` is likewise unaffected: it reads the
    # in-memory HistoryOptions(trace_record=True) trace, not this stream.
    optimizer = pypesto.optimize.FidesOptimizer(
        verbose=logging.ERROR, options=_fides_maxtime_options()
    )
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
            problem,
            optimizer=optimizer,
            n_starts=n_starts,
            engine=pp_engine,
            progress_bar=False,
            **minimize_kwargs,
        )
        # The starts the engine stopped past their cap and scored as failed (multistart).
        killed = getattr(pp_engine, "killed", [])
    except Exception as exc:  # noqa: BLE001 - retry sequentially before giving up
        # A parallel multistart can construct its engine fine yet still fail at runtime
        # (pickling/spawn issues, a worker dying). That is a parallel-only failure — the same fit
        # runs fine on the sequential engine — so fall back to it rather than failing the whole
        # commit. Sequential is slower, so this can exceed the parallel-sized fit budget on a big
        # n_starts; that is an acceptable rare-path cost vs. a hard failure (resuming the session
        # backstops a timeout). A sequential fit that itself fails re-raises normally. A failure the
        # server raised is not retried: a cluster's would run every start on the server node, and a
        # wedged or cancelled fit must end (see :mod:`petab_mcp.multistart`).
        if pp_engine is None or isinstance(exc, PetabMcpError):
            raise
        log.warning(
            "fit: parallel multistart failed (%s: %s); retrying sequentially",
            type(exc).__name__,
            exc,
        )
        result = pypesto.optimize.minimize(
            problem,
            optimizer=optimizer,
            n_starts=n_starts,
            engine=None,
            progress_bar=False,
            **minimize_kwargs,
        )
        killed = []
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
    # An all-starts-failed multistart leaves `x` as None. Indexing it below raises a bare
    # TypeError, which _compile_fit_failure_message can only read as an ill-posed ODE — a
    # confident, WRONG diagnosis that sends the agent to rewrite a structurally fine model.
    # Caught here at the source rather than by string-matching the TypeError downstream, where
    # any future None-indexing in the compile/fit path would inherit the same specific text.
    if best.get("x") is None:
        # Branch the diagnosis on what actually failed, using the objective already evaluated at
        # the nominal point above. The old message led unconditionally with the log-transform
        # hypothesis; followed literally, it sent an agent to re-initialise its pools twice while
        # the real failure was AMICI FORWARD_FAILURE (integration), and the correct delay mechanism
        # was abandoned as "numerically invalid".
        # Two causes produce this, and they need OPPOSITE fixes. `objective_before` cannot
        # arbitrate: engine._objective_evaluate stores
        # finite_float(fval), so it is None for a failed SIMULATION *and* for a finite simulation
        # whose objective is inf (log10(0) at a measured point) -- branching on it would state the
        # wrong cause confidently in exactly the case the original text was written for. So name
        # both, and name the log line that distinguishes them, rather than guessing.
        #
        # Except where the nominal evaluation already shows a third cause: a measured point the
        # model leaves UNDEFINED while the rest simulate, a readout that is 0/0 there (a ratio
        # `A/(A + B)` with both starting at 0, measured at t=0). The plain objective drops such a
        # point and stays finite; the hierarchical one, which solves the noise from every residual,
        # is non-finite at every start. Named, with the readout.
        undefined = _undefined_points(session, before.get("simulations") or [])
        if undefined:
            raise ModelUpdateError(
                "the fit produced NO usable optimum: every multistart failed, because the model "
                f"leaves measured points UNDEFINED: {undefined}. A readout there is not a number "
                "(NaN or infinite): a division by an amount that is 0 at that time (0/0 where "
                "every species in a ratio starts at 0) or a log of 0. "
                "Define each readout so it holds at every measured time, t=0 included."
            )
        raise ModelUpdateError(
            "the fit produced NO usable optimum: every multistart failed, so the best result "
            "carries no parameter vector. The objective was non-finite at EVERY start — a "
            "property of the problem, not the optimizer. There are two causes and they need "
            "OPPOSITE fixes, so check which one this is BEFORE editing — the fitting log tells "
            "them apart (have it read if you cannot read it yourself): (a) the SIMULATION "
            "fails — 'FORWARD_FAILURE' / 'AMICI simulation failed' appears in the log; the ODE "
            "is non-integrable or extremely stiff, usually from the last structural change (a "
            "new pool with no sink, a rate at an extreme default, or writing into a symbol the "
            "observables use as a normalisation constant), and re-initialising species will NOT "
            "help. (b) the simulation succeeds but the OBSERVABLE is undefined — a "
            "log-transformed observable reaching 0 or negative at a measured time (a species "
            "started at exactly 0 and measured at t=0 gives log10(0) = -inf); there, fixing the "
            "initial value or the bound that admits it IS the fix."
        )
    df = session.problem.parameter_df
    nllh = float(best["fval"])
    # Record and write back only the ESTIMATED parameters. pyPESTO's `x`/`x_names` include the
    # fixed parameters too, so filtering to `x_free_indices` (mirroring `_compute_fim`) keeps
    # `fitted` the free parameters only — otherwise k = len(fitted) overcounts and every
    # complexity-penalised score (bic/aic/reduced_chi2) is wrong and disagrees with n_estimated.
    fitted = _fitted_values(
        problem.x_names, best["x"], getattr(problem, "x_free_indices", None), df
    )
    if hierarchical:
        # Merge the analytically-solved inner parameters (not in the outer vector above) so
        # `fitted` covers EVERY estimated parameter — keeping k = len(fitted) consistent with
        # session.n_estimated, and writing the inner values back to the table like any other fit.
        fitted.update(_inner_fitted_values(problem, best, inner))
    with session.lock:
        for name, linear in fitted.items():
            if name in df.index:
                df.loc[name, NOMINAL_VALUE] = linear

    # Convergence summary: how many multistarts landed near the best objective.
    fvals = [
        float(s["fval"])
        for s in result.optimize_result.list
        if s.get("fval") is not None and math.isfinite(float(s["fval"]))
    ]
    n_converged = sum(1 for v in fvals if v <= nllh + 1e-3)
    log.info(
        "fit: done best_nllh=%s self_improvement=%s time=%.2fs starts=%d converged~%d "
        "fval[min/max]=%s/%s",
        _fmt(nllh),
        _fmt(objective_before - nllh if objective_before is not None else None),
        duration,
        len(fvals),
        n_converged,
        _fmt(min(fvals) if fvals else None),
        _fmt(max(fvals) if fvals else None),
    )
    optimization = _optimization_payload(result)
    optimization["x_names"] = list(problem.x_names)  # column order for each start's "x"
    # ...and the scale each of those coordinates is expressed on, so the NEXT fit can remap these
    # optima onto its own (differently-parameterised, possibly differently-scaled) vector and
    # re-enter them as warm starts. See `previous_optima`.
    optimization["x_scales"] = [str(s) for s in problem.x_scales]
    optimization["n_starts_killed"] = len(killed)
    optimization["result_file"] = _write_pypesto_result(session, result, session.problem_hash())
    return {
        "fitted": fitted,
        "neg_log_likelihood": nllh,
        "objective_before": objective_before,
        "optimization": optimization,
        # Frozen HERE, where the built problem, the optimizer's point and the written-back
        # table all agree — every later diagnosis of this fit reads only this (see FitRecord).
        "record": freeze_fit(
            session, problem, best, hierarchical=hierarchical, optimization=optimization
        ),
    }


#: Relative gap between the optimizer's objective and the re-simulated one above which the
#: two are reported as disagreeing. Both evaluate the same likelihood at the same parameters,
#: so anything past a loose numerical tolerance means the two evaluations differ in substance
#: (solver settings, or a parameter that did not round-trip), not in rounding.
OBJECTIVE_AGREEMENT_RTOL = 1e-3


def _undefined_points(session, records: list[dict[str, Any]]) -> str:
    """The measured points a simulation that otherwise ran leaves without a value, by observable,
    with the readout each reads; ``""`` when there are none, or when nothing simulated at all (an
    integration failure, a different cause)."""
    missing = [r for r in records if r.get("measurement") is not None and r.get("simulation") is None]
    if not missing or len(missing) == len(records):
        return ""
    import libsbml

    from .reconcile import formula_identifiers

    obs = session.problem.observable_df
    sm = getattr(session.problem.model, "sbml_model", None)
    rules = {}
    if sm is not None:
        for i in range(sm.getNumRules()):
            rule = sm.getRule(i)
            if rule.isAssignment():
                rules[rule.getVariable()] = libsbml.formulaToL3String(rule.getMath())
    parts = []
    for obs_id in sorted({str(r["observable_id"]) for r in missing}):
        times = sorted({r["time"] for r in missing if str(r["observable_id"]) == obs_id})
        shown = ", ".join(f"t={t:g}" for t in times[:4]) + (", ..." if len(times) > 4 else "")
        reads = ""
        if obs is not None and obs_id in obs.index:
            for name in sorted(formula_identifiers(obs.loc[obs_id, OBSERVABLE_FORMULA])):
                if name in rules:
                    reads += f"; it reads `{name} := {rules[name]}`"
        parts.append(f"`{obs_id}` at {shown}{reads}")
    return "; ".join(parts)


def _reconcile_objective(
    session, fit: dict[str, Any] | None, sim: dict[str, Any]
) -> tuple[float | None, str | None]:
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
        return nll_sim, ("optimizer returned no finite objective" if nll_sim is None else None)
    if nll_sim is None:
        note = (
            f"re-simulating the fitted point did not reach {n_missing} of "
            f"{sim.get('n_expected_points')} measured points, so its likelihood is "
            f"undefined; reporting the optimizer's objective ({_fmt(nll_opt)}) instead. "
            "The fit converged — this is a solver/integration gap at the optimum, not a "
            "failed fit."
        )
        session.logger.warning("fit: %s", note)
        return nll_opt, note
    if _disagree(nll_opt, nll_sim):
        note = (
            f"optimizer objective {_fmt(nll_opt)} disagrees with the re-simulated "
            f"{_fmt(nll_sim)} at the same parameters; reporting the optimizer's."
        )
        session.logger.warning("fit: %s", note)
        return nll_opt, note
    return nll_opt, None


def _disagree(expected: float, nll: float) -> bool:
    return abs(expected - nll) > OBJECTIVE_AGREEMENT_RTOL * max(1.0, abs(expected))


def _committed_score_note(session, nll_sim: float | None) -> str | None:
    """A note, and a warning in the log, when a pass with no fit scores the state of the last
    version applied at the table's own values and does not reproduce the score that version was
    committed with; ``None`` otherwise. A fit writes the sigmas and scalings it solved into the
    table, so the two agree unless the table does not hold that point. The payload keeps the
    table's score."""
    committed = getattr(session, "version_score", None)
    if committed is None or committed[0] != session.problem_hash():
        return None
    nll_committed = _flt(committed[1])
    if nll_committed is None or (nll_sim is not None and not _disagree(nll_committed, nll_sim)):
        return None
    note = (
        f"this version was committed with nll {_fmt(nll_committed)}, but scoring it at its table's "
        f"own values (sigmas and scalings as they stand) gives {_fmt(nll_sim)}: the table does "
        "not hold the point that score was taken at."
    )
    session.logger.warning("simulate-only: %s", note)
    return note


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
        # The effective parameter count at the optimum the fit just froze (see
        # identifiability.effective_parameter_count). It reaches a client through the stored
        # version (build_sim_payload_for_version) and no agent-facing result — docs/RATIONALE.md.
        # Computed here rather than beside the diagnosis in the session because compute_improvement
        # ranks THIS payload, and under TIC it needs the count to rank at all. None on a
        # simulate-only pass: no optimum of its own. `k` is complexity_k, so the literals reach
        # the count as the inner-solved parameters do, one unit each: TIC prices them too.
        n_estimated = int(session.problem.n_estimated)
        constants = _literals.hardcoded_constants(session.model.sbml_model)
        k_eff = None
        if fit is not None:
            from .identifiability import effective_parameter_count

            k_eff = effective_parameter_count(
                fit["record"].pypesto_problem,
                fit["record"].x_free,
                k=complexity_k({"n_estimated": n_estimated, "hardcoded_constants": constants}),
            )
        # A pass with no fit scores the table's sigmas and scalings as they stand.
        sim = engine.residuals(solve_inner=do_fit)
        records = sim["records"]
        nllh, nll_note = _reconcile_objective(session, fit, sim)
        if fit is None:
            nll_note = _committed_score_note(session, nllh)

        payload = {
            "status": "fit" if do_fit else "simulate_only",
            # fitted_parameters are always LINEAR (unscaled) values. parameter_scales gives each
            # one's ESTIMATION scale (lin/log10), co-located so a reader never mistakes a linear
            # value for its log10 coordinate — e.g. a Hill coefficient fitted to 15.3 (linear) is
            # 15.3, NOT 10**15.3. (An agent once made exactly that error.)
            "fitted_parameters": fit["fitted"] if fit else None,
            "parameter_scales": (
                {
                    pid: str(session.problem.parameter_df.loc[pid, PARAMETER_SCALE])
                    for pid in fit["fitted"]
                }
                if fit
                else None
            ),
            # The estimated-parameter count, so the goodness-of-fit metrics (bic/reduced_chi2)
            # are available even on a simulate-only pass where `fitted_parameters` is None —
            # plus the literals `complexity_k` prices on top of it (see :mod:`literals`).
            "n_estimated": n_estimated,
            "hardcoded_constants": constants,
            # ...and the count the data actually see, which is what TIC charges for. Otherwise
            # not agent-facing: fit_result_block and goodness_of_fit deliberately do not copy it.
            "k_eff": k_eff,
            # The OPTIMIZER's objective at its best start when that is available (see
            # _reconcile_objective) — the quantity actually minimised, and the one every
            # downstream score (bic, is_new_best, a client's own scoring) must agree with.
            "neg_log_likelihood": nllh,
            # What re-simulating the fitted point produced, kept alongside so a
            # disagreement between the two stays visible rather than silently resolved.
            "neg_log_likelihood_simulated": sim["neg_log_likelihood"],
            "objective_note": nll_note,
            "chi2": sim["chi2"],
            "n_data_points": sim["n_data_points"],
            "n_missing_simulations": sim.get("n_missing_simulations"),
            # Per-start exit histogram, set HERE rather than bridged from `session.optimization`
            # afterwards: the `optimization` dict never reaches this payload (it is assigned to
            # the session a few lines below), so a consumer reading `fitsim["optimization"]` gets
            # None on every fit -- which is exactly how this value silently stayed None after the
            # per-start `exitflag`/`message` columns were deleted and Fides dropped to ERROR on
            # the strength of it. Putting it on the payload at the point it is known removes the
            # seam instead of adding a second one.
            "exitflag_counts": (fit["optimization"] or {}).get("exitflag_counts") if fit else None,
            # ...and, the same way, how many starts were stopped past their cap (see
            # :mod:`multistart`): scored as failed, so the fit searched less than `n_starts` names.
            "n_starts_killed": (fit["optimization"] or {}).get("n_starts_killed") if fit else None,
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
        # restore would leave the parameter diagnosis unavailable for the rest of the session.
        if fit is not None:
            session.fit_record = fit["record"]
        session.logger.info(
            "fit_and_simulate: status=%s nllh=%s chi2=%s n_points=%s",
            payload["status"],
            _fmt(payload["neg_log_likelihood"]),
            _fmt(payload["chi2"]),
            payload["n_data_points"],
        )
        return payload


def score_at_nominal(session, engine) -> dict[str, Any]:
    """Score a SHIPPED model at its table's ``nominalValue``s: compile, simulate, and report the
    estimated parameters' nominal values as the fitted point (``{id: linear value}`` + scales),
    labelled ``scored_at_nominal``. No optimiser runs. Used to score a problem that arrives
    finished; see ``docs/RATIONALE.md``."""
    with session.lock:
        payload = fit_and_simulate(session, engine, do_fit=False, n_starts=0)
        pdf = session.problem.parameter_df
        payload["fitted_parameters"] = table_fitted_values(pdf)
        payload["parameter_scales"] = {
            pid: str(pdf.loc[pid, PARAMETER_SCALE]) for pid in payload["fitted_parameters"]
        }
        payload["status"] = "scored_at_nominal"
        session.simulation = payload
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
    re-authors the working SBML, so the read would compile a model no fit ever scored, to describe
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
        if not engine.module_built(session.module_hash()):
            session.logger.info(
                "no simulation for this read: the working structure has no compiled module, and a "
                "read never pays a build (module_hash=%s)",
                session.module_hash(),
            )
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
    fitted: dict[str, float] | None,
    parameter_df: Any,
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
            v, lo, up = _scaled(float(value), scale), _scaled(lb, scale), _scaled(ub, scale)
        except (TypeError, ValueError):
            continue
        span = up - lo
        if span <= 0:
            continue
        frac = (v - lo) / span
        if frac <= tol:
            flags.append({"parameter_id": pid, "side": "lower", "value": value, "bound_value": lb})
        elif frac >= 1 - tol:
            flags.append({"parameter_id": pid, "side": "upper", "value": value, "bound_value": ub})
    return flags


def _entry_inflated(entry: "dict[str, Any]") -> bool:
    """Whether ONE noise_sigma_report entry is inflated: fitted above its data-driven cap (the
    noise absorbing model misfit). Shared by the per-entry ``status`` and :func:`noise_sigma_inflated`
    so the agent-facing status can't drift from the adequacy gate built on it.

    The data-std cap is the ONLY arm, and NO ratio against a noise floor is one: #140 added such an
    arm (sigma above 1.5x the pure-error replicate floor); it is gone. Dividing two point
    estimates discards the denominator's uncertainty, so no single threshold survives the problems
    it was measured on — one needs <=1.004 for any power, another >7.83 not to condemn its own
    reference. The sharp test keeps the floor's df instead (per-design-point lack-of-fit against
    that same pooled floor, in the residual-structure verdict).
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
    # are all fixed. The case: a sigma shipped fixed at 1.0 against cumulative counts whose
    # marginal std is 390 (dominated by the trend the model explains), so it read "collapsed" at
    # 0.0026 for a noise level the problem declared, not one any fit chose.
    for pid in sorted(R.plausibility_sigma_ids(problem) & set(fitted)):
        # Fail OPEN when there is no parameter table to consult (a bare observable+measurement
        # problem, whose table is empty): unknown estimate status must not silently drop the
        # sigma from the report.
        pdf = getattr(problem, "parameter_df", None)
        if pdf is not None and len(pdf) and not R.is_parameter_estimated(problem, pid):
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
        # what a discrepancy is worth, and below NOISE_SIGMA_MIN_POINTS the LOCAL verdict is
        # withheld as degenerate too (see upper_bound_cap).
        #
        # Best-effort like every other field here: this is the ONE call in the loop that reaches
        # into the measurement table directly, so a problem without one (or with a shape it cannot
        # read) must cost the count, not the report. A missing count is a supported state — a
        # reader falls back to a fixed tolerance — while a raise here fails the commit.
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
            "upper_bound_cap": None
            if too_few
            else R.sigma_upper_bound_cap(problem, pid, fitted_values=fitted, data_std=data_std),
            "data_std": data_std,
            "fraction_of_data_std": None
            if too_few
            else ((effective / data_std) if (data_std and effective is not None) else None),
            "data_noise_estimate": data_noise,
            "data_noise_method": data_noise_method,
            "fitted_over_data_noise": (effective / data_noise)
            if (data_noise and effective is not None)
            else None,
        }
        # Per-entry plausibility from the SAME predicates the top-level noise_sigma_inflated/
        # collapsed use (single source of truth, so the agent-facing status can't drift from the
        # adequacy gate). "unknown" when the observable's std can't be computed (data_std
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
            entry["status"] = "inflated"  # noise absorbing model misfit
        elif _entry_collapsed(entry):
            entry["status"] = "collapsed"  # overfitting the noise for a good-looking score
        else:
            entry["status"] = "ok"
        # Whether this sigma's NUMBER can be compared to another model's fitted sigma at all. The
        # data-driven verdicts above are unaffected (the gain cancels from sd/spread), but a
        # comparison against a reference divides one model's sigma by another's, and that ratio is
        # meaningless when an estimated gain shares the formula -- see reconcile.sigma_scale_coupled.
        try:
            entry["scale_coupled"] = R.sigma_scale_coupled(problem, pid)
        except Exception:  # noqa: BLE001 - a report field must never break a fit
            entry["scale_coupled"] = False
        out[pid] = entry
    return out


def noise_sigma_inflated(report: "dict[str, Any] | None") -> bool:
    """True if any ESTIMATED noise sigma in a ``noise_sigma_report`` is fitted ABOVE its
    data-driven cap — i.e. the noise is absorbing model misfit rather than measuring it. The single
    source of truth for "this sigma is inflated", used by the session's plateau-adequacy gate AND
    a client's success gate so they can't drift apart. Best-effort: an empty/unreadable report or
    a non-numeric entry is treated as NOT inflated."""
    if not isinstance(report, dict):
        return False
    return any(_entry_inflated(e) for e in report.values() if isinstance(e, dict))


def noise_sigma_collapsed(report: "dict[str, Any] | None") -> bool:
    """True if any ESTIMATED noise sigma is fitted implausibly SMALL — below
    ``constants.NOISE_SIGMA_COLLAPSE_FRACTION`` of its observable's own data std, i.e. the model is
    explaining nearly all the data's variability (overfitting the noise for a suspiciously good BIC).
    The other tail of :func:`noise_sigma_inflated`, and likewise the single source of truth shared by
    the session's adequacy gate and a client's success gate. Best-effort: an empty/unreadable report
    or a non-numeric ``fraction_of_data_std`` is treated as NOT collapsed."""
    if not isinstance(report, dict):
        return False
    return any(_entry_collapsed(e) for e in report.values() if isinstance(e, dict))


# Agent-facing advice when the fit's noise model is implausible. Surfaced proactively on every
# get_fit_summary (not just at a plateau stop) and echoed by a client's finish gate, so the agent
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


def slim_fit_payload(session, out: dict, version: int | None = None) -> dict:
    """Public entry point: slim ``out`` in place and return it, for callers that assemble a fit
    payload themselves (``session._fit_result`` and friends build it via
    :func:`fit_result_block`, which has no session and so cannot do this itself). The
    make_changes result is the payload the agent sees MOST — slimming only get_fit_summary left
    the dominant instance untouched, and shipped two different shapes for the same quantity.

    ``version`` MUST be passed by those callers. The diff needs to know which version this fit
    is, and `fit_result_block` does not carry one: every session caller puts `version` as a
    SIBLING of the block, so reading it off the block found None and `_slim_fitted_parameters`
    returned on its first guard. That made this a silent no-op on all three of make_changes,
    the duplicate-commit result and rollback -- i.e. on the dominant instance it was written for,
    leaving the 84-118-entry dict riding on every one of them. Only `get_fit_summary`, which
    happens to build `version` into its own out dict, was ever slimmed.

    Past what it shares with `get_fit_summary`, a fit result names a scale and an at-bound pin
    only where it is new since the parent (:func:`_since_parent`)."""
    own = {**out, "version": version} if version is not None and "version" not in out else out
    scales = own.get("parameter_scales")
    parent = _slim_fitted_parameters(session, own)
    if parent is not None:
        own["parameter_scales"] = scales
        _since_parent(own, parent)
    return own if own is out else {k: v for k, v in own.items() if k != "version"}


def _sig(value: Any, digits: int = 4) -> Any:
    """``value`` to ``digits`` significant digits when it is a finite number, else unchanged."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return value
    return float(f"{f:.{digits}g}") if math.isfinite(f) else value


def _since_parent(out: dict, parent_df: Any) -> None:
    """Cut a slimmed fit result's ``parameter_scales`` and ``parameters_at_bound`` to what is new
    since the parent version, in place.

    A named parameter's scale is listed where it is new or re-scaled; an earlier result named every
    other one. A pin is listed unless the parent sat at the same side and bound, so a re-pin after a
    widening still shows; ``n_parameters_at_bound`` counts every pin when any is left out.
    Best-effort, like the slimming before it."""
    try:
        prior = parent_df[PARAMETER_SCALE].astype(str).to_dict()
        held = {
            (f["parameter_id"], f["side"]): f["bound_value"]
            for f in parameters_at_bound(table_fitted_values(parent_df), parent_df)
        }
    except Exception:  # noqa: BLE001 - never degrade the score payload over a slimming aid
        return
    scales = out.get("parameter_scales") or {}
    out["parameter_scales"] = {k: v for k, v in scales.items() if prior.get(k) != v}
    pins = out.get("parameters_at_bound") or []
    new = [
        {**f, "value": _sig(f["value"])}
        for f in pins
        if not math.isclose(held.get((f["parameter_id"], f["side"]), math.nan), f["bound_value"])
    ]
    out["parameters_at_bound"] = new
    if len(new) < len(pins):
        out["n_parameters_at_bound"] = len(pins)


def _slim_fitted_parameters(session, out: dict) -> Any:
    """Replace the full fitted-values dict with a CHANGED-vs-previous-commit view, in place.

    The full dict is 84-118 entries on the large problems and rides on every make_changes /
    get_fit_summary payload, each re-sent in the caller's history on every later turn — on one
    such problem an agent's history grew ~15.5k tokens/turn and three of nine sessions died on the
    context window before their fit budget. Most entries barely move between commits; the
    caller acts on the ones that DID. Full values stay one call away (`get_parameters` serves
    nominalValue = the last fit) and the ARCHIVE keeps the complete values per version
    (parameters.tsv), so only the push is slimmed — the pull is intact.

    Semantics: `fitted_parameters_changed` = entries NEW or moved >1% (relative, absolute floor
    for zeros) vs the PREVIOUS version's stored fit, to 4 significant digits;
    `n_parameters_unchanged` counts the rest.
    The FIRST fit keeps the full dict under the original key — nothing to diff against, and the
    first read is where the whole vector genuinely informs. Best-effort: any failure keeps the
    full dict, because a slimming aid must never degrade the score payload.

    Returns the parent's parameter table when it slimmed, else ``None``.
    """
    fitted = out.get("fitted_parameters")
    version = out.get("version")
    if not isinstance(fitted, dict) or not isinstance(version, int) or version <= 0:
        return None
    try:
        # The PARENT, not version-1: after a restore/rollback the tip of the abandoned branch is
        # not what this fit descends from, and diffing against it reports parameters identical to
        # the real parent as "moved" and vice versa. provenance records `parent` per version.
        meta = session.provenance.get_meta(version) or {}
        base_v = meta.get("parent")
        if not isinstance(base_v, int) or base_v < 0:
            base_v = version - 1
        parent = session.provenance.get_parameter_df(base_v)
        before = table_fitted_values(parent)
        if not before:
            return None

        def _num(x):
            try:
                f = float(x)
            except (TypeError, ValueError):
                return None
            return f if math.isfinite(f) else None

        changed = {}
        for pid, val in fitted.items():
            new_v, old_v = _num(val), _num(before.get(pid))
            # NON-FINITE is always reported: a parameter that diverged to NaN/inf is the single
            # most diagnostic entry in the vector, and `abs(nan - x) > tol` is False, so a plain
            # comparison would file it as "unchanged" and drop it from the payload entirely.
            if new_v is None or old_v is None:  # non-finite, unparseable, or NEW
                changed[pid] = val
                continue
            if abs(new_v - old_v) / max(abs(old_v), 1e-12) > 0.01:
                changed[pid] = val
        # REMOVALS are a change too: a structural re-author that drops a parameter would
        # otherwise be invisible, since the loop above only walks the current vector.
        removed = sorted(set(before) - set(fitted))
        scales = out.get("parameter_scales") or {}
        out["fitted_parameters_changed"] = {pid: _sig(val) for pid, val in changed.items()}
        if removed:
            out["parameters_removed"] = removed
        out["n_parameters_unchanged"] = len(fitted) - len(changed)
        out["fitted_parameters_note"] = (
            f"only parameters NEW, REMOVED or moved >1% vs version {base_v} (this fit's parent) "
            f"are listed; {len(fitted) - len(changed)} unchanged "
            "(read any value via get_parameters)."
        )
        out["fitted_parameters"] = None
        # Scales for everything the payload still NAMES (changed + removed). Nulling them
        # wholesale when nothing changed left a read with no scale information at all, right
        # after _parameter_scales_for reconstructed it precisely so a linear value is never
        # misread as a log10 one.
        named = set(changed) | set(removed)
        out["parameter_scales"] = {k: v for k, v in scales.items() if k in named}
    except Exception:  # noqa: BLE001 - never degrade the score payload over a slimming aid
        return None
    return parent


def fit_summary(session, engine) -> dict[str, Any]:
    """The current fit metrics, no per-point records — a cheap score read (backs
    ``get_fit_summary``).

    Ensures a simulation dataframe exists, then returns ``{status, neg_log_likelihood, chi2,
    n_data_points, fitted_parameters, parameter_scales, version}`` plus the complexity-
    penalised :func:`goodness_of_fit` scores, ``parameters_at_bound`` (see
    :func:`parameters_at_bound`), ``noise_sigma_report`` (see :func:`noise_sigma_report`), and
    ``residual_structure`` (see :func:`data_access.residual_structure_rollup`).
    ``k`` is :func:`complexity_k` — ``n_estimated`` plus the model's ``hardcoded_constants``
    (well-defined even on a simulate-only pass); both addends ride on the result. Pairs fitted
    values with their estimation scale even when a restored/rollback payload lacks the map, so a
    linear value is never misread as a log10 one.

    Both adequacy verdicts ride on this read: it is the score read for every caller that did not
    run the fit — another agent, a client's finish gate, a final scoring.
    """
    sim = ensure_simulation(session, engine) or {}
    out = {
        k: sim.get(k)
        for k in (
            "status",
            "neg_log_likelihood",
            "chi2",
            "n_data_points",
            "n_missing_simulations",
            "objective_note",
            "fitted_parameters",
            "parameter_scales",
            "version",
        )
    }
    pdf = session.problem.parameter_df
    out["parameter_scales"] = _parameter_scales_for(
        out.get("fitted_parameters"), out.get("parameter_scales"), pdf
    )
    # Live counts, not the payload's: `ensure_simulation` guarantees the simulation matches the
    # working model, and a restored legacy payload may predate either field.
    out["n_estimated"] = int(session.problem.n_estimated)
    out["hardcoded_constants"] = _literals.hardcoded_constants(session.model.sbml_model)
    out["k"] = complexity_k(out)
    out.update(
        goodness_of_fit(
            sim.get("neg_log_likelihood"),
            sim.get("chi2"),
            out["k"],
            sim.get("n_data_points"),
            criterion=session.criterion,
            k_eff=sim.get("k_eff"),
        )
    )
    out["parameters_at_bound"] = parameters_at_bound(out.get("fitted_parameters"), pdf)
    _report = noise_sigma_report(session.problem, out.get("fitted_parameters"))
    out["noise_sigma_report"] = _report
    # Proactive plausibility verdict on EVERY read (not just at a plateau stop): a good BIC bought
    # with an inflated/collapsed estimated sigma is not real adequacy and will not be accepted as
    # finished. Surfaced here so the agent can act on it, and read by a client's finish gate.
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
    # that RAN the fit holds — so a caller reading the score afterwards (another agent, a client's
    # finish gate, a final scoring) had no way to see them. Re-serving the stored hints costs
    # nothing: `stored_merge_hints` returns them only while the working problem still IS the one that
    # priced them, and None otherwise. ABSENT when there are none, like `noise_sigma_advice`.
    from .identifiability import stored_merge_hints

    _hints = stored_merge_hints(session)
    if _hints:
        out["merge_hints"] = _hints
    # The session's OUTSTANDING invalid "does adding structure help?" experiment, if any (see
    # :mod:`nesting`): a commit that gained parameters yet fit WORSE than its parent, which for a
    # nested model is impossible. Sticky across the rollback that usually follows, so a client's
    # finish gate can surface it and a finish cannot rest on that comparison.
    # ABSENT when there is none — an omitted key must read as "nothing outstanding".
    _invalid = session.invalid_comparison
    if _invalid:
        out["invalid_comparison"] = _invalid
    # LAST, after every field derived FROM the fitted values has been computed. Slimming earlier
    # silently disabled the noise-sigma verdict: noise_sigma_report(problem, None) returns {},
    # which zeroes noise_sigma_inflated/collapsed — a finish gate's gamed-noise refusal and a
    # scorer's sigma-implausibility check both read exactly those keys, so a session that bought a
    # good BIC with an inflated sigma would have scored success. A payload diet must be the last
    # step, never an input to one.
    _slim_fitted_parameters(session, out)
    return out


def compute_improvement(
    session,
    nllh: float | None,
    objective_before: float | None = None,
    fitsim: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Improvement metrics for a fit, ranking it against the best model so far **by the session's
    complexity criterion** (``bic`` unless configured otherwise — see :mod:`criteria`).

    ``is_new_best`` tracks ``criterion_value`` — the same complexity-penalised score the
    early-stop checkpoint stores — so the flag, the non-improving-streak reset, and the
    best-model checkpoint all move on one coherent notion of "better". A fit is a new best only
    when that score beats the reference by at least ``NEW_BEST_MIN_DELTA_SCORE`` (the conventional
    "positive evidence" bar), so a raw nll drop bought by an extra parameter that actually RAISES
    the score does NOT count, and neither does a 1e-6 noise-floor twitch.

    The reference is the session's best-score :class:`~petab_mcp.session.Checkpoint`, set on the
    previous new best — not the raw running minimum. Comparing against
    the checkpoint is what lets a slow, steady creep of < ``NEW_BEST_MIN_DELTA_SCORE`` per op
    still register a new best once it CUMULATIVELY clears the bar, instead of starving the streak
    while the fit genuinely improves. Must be called BEFORE :meth:`session.record_fit_progress`
    updates the checkpoint, so a fit never compares against itself. The first scored fit (no
    checkpoint yet) is always a new best. :meth:`session.PetabSession.fit` withdraws the flag
    before stamping it where the checkpoint declines the fit, naming why in
    ``new_best_declined``, so ``is_new_best`` marks exactly the versions that became the best.

    The returned block carries the ranked pair ``criterion``/``criterion_value`` beside the
    canonical ``bic``, and is stamped into the fit event and the version meta — which is what
    lets a resumed session and the de-dup registry recover the score. Only those three: this block
    rides on every fit result and is re-sent in the agent's history each turn, and the full
    criterion table is already on the fit block (and in the version meta) one level up. It also
    states its own gate: past the first fit, ``is_new_best`` is ``improvement_vs_best_prior``
    (over ``best_prior_criterion_value``) reaching ``new_best_min_improvement``.
    """
    fitsim = fitsim or {}
    scores = goodness_of_fit(
        nllh,
        fitsim.get("chi2"),
        complexity_k(fitsim),
        fitsim.get("n_data_points"),
        criterion=session.criterion,
        k_eff=fitsim.get("k_eff"),
    )
    score = scores.get("criterion_value")

    # A non-finite checkpoint is no checkpoint: treat it as absent so a session that somehow
    # acquired one (a legacy provenance seed, a version written before goodness_of_fit
    # rejected non-finite scores) recovers on its next fit instead of staying locked out.
    _best = session.best
    best_prior_score = _flt(_best.score) if _best else None
    best_prior_version = _best.version if _best else None
    best_prior_nllh = _best.nll if _best else None

    self_improvement = None
    if objective_before is not None and nllh is not None:
        self_improvement = objective_before - nllh
    # Criterion delta: > 0 means this fit is better than the checkpoint.
    improvement_vs_best_prior = None
    if best_prior_score is not None and score is not None:
        improvement_vs_best_prior = best_prior_score - score
    is_new_best = clears_new_best_bar(score, best_prior_score)

    return {
        "neg_log_likelihood": nllh,
        "objective_before": objective_before,
        "self_improvement": self_improvement,
        "bic": scores.get("bic"),
        "criterion": scores.get("criterion"),
        "criterion_value": score,
        "best_prior_version": best_prior_version,
        "best_prior_criterion_value": best_prior_score,
        "best_prior_neg_log_likelihood": best_prior_nllh,
        "improvement_vs_best_prior": improvement_vs_best_prior,
        # The gate itself, so an `is_new_best: false` says what the fit needed: an improvement
        # over `best_prior_criterion_value` of at least this.
        "new_best_min_improvement": NEW_BEST_MIN_DELTA_SCORE,
        "is_new_best": is_new_best,
    }


def clears_new_best_bar(score: float | None, best_score: float | None) -> bool:
    """The ``is_new_best`` gate: there is a ``score``, and no checkpoint yet or it beats the
    checkpoint by at least ``NEW_BEST_MIN_DELTA_SCORE``. One function because a durable resume
    replays it over the fit events (:meth:`session.PetabSession._seed_progress_from_provenance`),
    and a resume that gated differently would rebuild a checkpoint the live run never had.
    """
    # ``<=`` so an improvement EXACTLY at the threshold counts as a new best — i.e. Δscore >= the
    # gate (a strict ``<`` would exclude the boundary and leave the streak un-reset there).
    return score is not None and (
        best_score is None or score <= best_score - NEW_BEST_MIN_DELTA_SCORE
    )


def complexity_k(fitsim: dict[str, Any] | None) -> int | None:
    """The ``k`` every penalised score charges: a payload's ``n_estimated`` plus its
    ``hardcoded_constants`` (see :mod:`literals`), so a value frozen into a rate law costs
    exactly what estimating it costs. ``None`` when the payload carries no ``n_estimated``; a
    payload without ``hardcoded_constants`` predates the count and is charged none.
    """
    fitsim = fitsim or {}
    k = fitsim.get("n_estimated")
    if k is None:
        return None
    return int(k) + len(fitsim.get("hardcoded_constants") or ())


def goodness_of_fit(
    nll: float | None,
    chi2: float | None,
    k: int | None,
    n: int | None,
    criterion: "criteria.Criterion | None" = None,
    *,
    k_eff: float | None = None,
) -> dict[str, Any]:
    """Complexity-penalised fit scores, so a commit is judged on more than raw nll.

    The agent otherwise sees only the raw nll/chi2 and cannot tell "already at the noise
    floor" from "keep going". These derived scores make that explicit:

    * ``bic = k*ln(n) + 2*nll``, ``aic = 2*k + 2*nll``, plus ``aicc`` (AIC's small-sample
      correction), ``caic`` (BIC plus one unit per parameter), ``hqc`` (Hannan-Quinn) and ``tic``
      (Takeuchi: ``2*k_eff + 2*nll``, charging the EFFECTIVE count ``k_eff`` the fit reports
      rather than ``k``) — see :mod:`criteria`. Lower is better; each charges for every free
      parameter, so an edit that adds a parameter for a marginal nll drop scores WORSE. All of
      them ride on every result at their textbook weight, whichever one the session ranks by, so a
      fit can be re-read under a criterion it was not run with. ``k_eff`` itself is an INPUT
      here and not an output: it is for a client's own scoring, and is on no result the agent reads.
    * ``criterion`` names the one ACTUALLY ranked by — the session's
      (``PetabSession.criterion``, which every session caller passes; plain BIC when a bare score
      is read outside a session) — and ``criterion_value`` is that score. Every gate in the loop
      — ``is_new_best``, the best-score checkpoint, the plateau stop, the de-dup registry — reads
      ``criterion_value`` and nothing else, so they cannot disagree.
    * ``reduced_chi2 = chi2 / (n - k)`` is ≈ 1 when the model fits to the noise floor, and
      ``chi2_gof_pvalue`` is the upper-tail chi2 p-value on ``chi2`` with ``n - k`` degrees of
      freedom. Both are only meaningful with a KNOWN noise level: when the observable noise
      sigmas are ESTIMATED, chi2 pins near ``n`` by construction, so reduced_chi2 ≈ 1 and the
      p-value are uninformative — rank models by ``criterion_value``, not by these.

    ``k`` is the complexity count — :func:`complexity_k`: the estimated parameters plus the
    model's hardcoded constants — ``n`` the number of data points, ``k_eff`` the fit's effective
    count (``None`` when the fit did not produce one). The criterion scores are ``None`` when
    ``nll``/``k``/``n`` are missing, and each is additionally ``None`` where that criterion is
    undefined (``aicc`` at ``n <= k+1``, ``hqc`` below ``n = 3``, ``tic`` without a ``k_eff``).
    The chi2-based fields (``chi2_dof``, ``reduced_chi2``, ``chi2_gof_pvalue``) are ``None`` when
    the degrees of freedom ``n - k`` are non-positive (an under-determined fit) or ``chi2`` is
    missing.
    """
    crit = criterion or criteria.Criterion()
    k_eff = _flt(k_eff)
    out: dict[str, Any] = {
        **{name: None for name in criteria.PENALTIES},
        "criterion": crit.name,
        "criterion_value": None,
        "chi2_dof": None,
        "reduced_chi2": None,
        "chi2_gof_pvalue": None,
    }
    # Non-finite inputs are MISSING scores, not scores. A NaN nll would otherwise produce
    # a score of k*ln(n) + 2*NaN = NaN, which is a float and passes every `is not None` guard
    # downstream — and because NaN compares False against everything, a NaN adopted as the
    # best-score checkpoint locks out every subsequent fit for the rest of the session (the
    # `is_new_best` test can never fire again). Coercing to None here is the single upstream
    # chokepoint that keeps a failed evaluation from masquerading as a real score.
    nll = _flt(nll)
    chi2 = _flt(chi2)
    if nll is None or k is None or n is None or n <= 0:
        return out
    out.update(crit.values(nll, k, n, k_eff))
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
    derived = {
        pid: str(parameter_df.loc[pid, PARAMETER_SCALE])
        for pid in fitted
        if pid in parameter_df.index
    }
    return derived or None


def fit_result_block(
    fitsim: dict[str, Any] | None,
    improvement: dict[str, Any] | None = None,
    parameter_df: Any = None,
    problem: Any = None,
    criterion: "criteria.Criterion | None" = None,
) -> dict[str, Any]:
    """The fit fields every fit-bearing result returns (a fit, and a restore).

    Keeps the tool-result shape identical across those paths so an agent reads the same
    keys everywhere. ``fitsim`` is a :func:`fit_and_simulate` payload (or the restored
    simulation for a restore); ``improvement`` is the :func:`compute_improvement` dict;
    ``criterion`` is the session's, which every session caller passes.
    Beyond the raw nll/chi2 it also carries the complexity-penalised
    :func:`goodness_of_fit` scores (every named criterion, the ranked `criterion`/`criterion_value`
    pair, reduced_chi2/gof p-value) so the agent can see
    when the fit is already at the noise floor and stop adding parameters,
    ``parameters_at_bound`` (see :func:`parameters_at_bound`; empty when ``parameter_df`` is
    not passed) so a parameter pinned to its bound is flagged in the SAME report instead of
    requiring a separate ``get_parameters()`` comparison, the noise-model verdict (see
    :func:`noise_sigma_block`), and the compact
    ``residual_structure`` verdict the caller stashed on ``fitsim`` (see
    :func:`data_access.residual_structure_on_fit`) so the agent learns whether its model's FORM is
    still incomplete without spending a separate ``get_fit_diagnosis`` call — and, on the same
    reasoning for the model's PARAMETERISATION, the ``merge_hints`` the caller stashed there (see
    :func:`identifiability.merge_hints`), present only when a tie is worth taking.
    ``get_fit_summary`` carries the noise report and the rollup in full.
    """
    from .data_access import residual_structure_on_fit

    fitsim = fitsim or {}
    fitted = fitsim.get("fitted_parameters")
    nll = fitsim.get("neg_log_likelihood")
    chi2 = fitsim.get("chi2")
    n = fitsim.get("n_data_points")
    # k = complexity_k: n_estimated (which includes any inner, analytically-profiled params that
    # len(fitted_parameters) omits) plus the hardcoded constants — matching compute_improvement /
    # record_fit_progress / fit_summary, so the block's bic agrees with improvement.bic within the
    # same result. Fall back to len(fitted_parameters) only when the payload carries no
    # n_estimated (legacy simulate-only payloads).
    k = complexity_k(fitsim)
    if k is None and fitted is not None:
        k = len(fitted)
    constants = fitsim.get("hardcoded_constants") or []
    killed = fitsim.get("n_starts_killed")
    block = {
        "fit_status": fitsim.get("status"),
        # The multistart count the fit actually ran with (``None`` for a restored version, which
        # did not fit). Reported because `n_starts` OVERRIDES the server default rather than adding
        # to it, so this is how a caller confirms it fitted harder and not weaker.
        "n_starts": fitsim.get("n_starts"),
        # How many of them were stopped past their cap and scored as failed, ABSENT when none was:
        # this block is re-sent in the caller's history every turn.
        **({"n_starts_killed": killed} if killed else {}),
        # Per-start outcome histogram (e.g. {"ExitFlag.FTOL": 81, "ExitFlag.EXCEEDED_BOUNDARY":
        # 19}): how many starts CONVERGED on their own criterion vs hit the per-start time cap.
        # A fit dominated by boundary exits is optimiser-budget-limited, not structure-limited --
        # the distinction that decides whether "fit harder" or "change the model" is the right
        # next move, previously only recoverable by grepping optimizer stdout that is no longer
        # emitted (fides runs at ERROR level; see the optimizer construction).
        "exitflag_counts": fitsim.get("exitflag_counts"),
        "neg_log_likelihood": nll,
        "chi2": chi2,
        "n_data_points": n,
        # Non-zero means the trajectory did not reach every measured point, so chi2 is None
        # and the likelihood came from the optimizer rather than the re-simulation.
        "n_missing_simulations": fitsim.get("n_missing_simulations"),
        "objective_note": fitsim.get("objective_note"),
        # Surface k next to the scores so the parameter penalty in bic = k*ln(n) + 2*nll (or
        # whichever criterion the session ranks by) is legible — adding a free parameter must earn
        # its keep — and its two addends, so a k that exceeds n_estimated is explained by the
        # literals listed rather than read as a bug. The list rides only when non-empty: it is
        # re-sent in the caller's history every later turn.
        "n_estimated": fitsim.get("n_estimated"),
        "k": k,
        **({"hardcoded_constants": constants} if constants else {}),
        # fitted_parameters: LINEAR (unscaled) values; parameter_scales: each one's estimation
        # scale (lin/log10). Kept side by side so a linear value is never misread as a log10 one.
        "fitted_parameters": fitted,
        "parameter_scales": _parameter_scales_for(
            fitted, fitsim.get("parameter_scales"), parameter_df
        ),
        # The ranked pair is stripped from the AGENT's copy: it is already at the top level of
        # this block, and this dict is re-sent in the agent's history every turn (PR #23). The
        # RAW `improvement` keeps it — the fit event is spread from that one, and the durable
        # resume seeds its checkpoint off exactly those two keys.
        "improvement": (
            {k: v for k, v in improvement.items() if k not in ("criterion", "criterion_value")}
            if improvement
            else None
        ),
        "parameters_at_bound": parameters_at_bound(fitted, parameter_df),
        # Prefer the report PetabSession.fit already computed and stashed, so the event, the
        # adequacy gate and this block cannot disagree; computed here for the paths that don't
        # stash one (a restore).
        **noise_sigma_block(problem, fitted, report=fitsim.get("noise_sigma_report")),
        # The COMPACT residual-structure verdict only — the first few observables the sign/runs
        # tests or the replicate lack-of-fit test flag, a one-line summary, and the bounded
        # replicate verdict where the design has replicates (see
        # data_access.residual_structure_on_fit). Deliberately not the per-observable table or
        # worst_points: those would ride on EVERY fit result and pile up in the agent's re-sent
        # history, so they stay behind get_fit_diagnosis (issue #113).
        "residual_structure": residual_structure_on_fit(fitsim.get("residual_structure")),
    }
    # The cheapest parameter ties BIC would pay for (see :func:`identifiability.merge_hints`), ABSENT
    # when there are none — like `noise_sigma_advice` and `nesting_advisory`: this rides on every fit
    # and is re-sent in the agent's history each turn (PR #23), so it is a few lines only when there
    # is something to act on. Only a real fit carries them: a restore and a de-dup no-op ran no
    # optimizer, so they have no optimum to price a tie at, and a finish gate reading the score
    # (fit_summary's `merge_hints`) names any tie still outstanding instead.
    hints = fitsim.get("merge_hints")
    if hints:
        block["merge_hints"] = hints
    block.update(goodness_of_fit(nll, chi2, k, n, criterion=criterion, k_eff=fitsim.get("k_eff")))
    return block


def noise_sigma_block(
    problem: Any,
    fitted: dict[str, float] | None,
    report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The noise-model verdict as it appears on any fit-bearing result.

    The ``noise_sigma_inflated`` / ``noise_sigma_collapsed`` verdict, named as in
    :func:`fit_summary`, and — ONLY when one is set — ``noise_sigma_advice`` and the
    ``noise_sigma_report`` entries of the sigmas it flags. ``get_fit_summary`` keeps every entry.

    Shared by every fit-bearing result deliberately: because the advice key is ABSENT when the
    noise model is plausible, a result that omits these keys entirely reads to an agent as
    "plausible". Any fit-bearing shape must therefore carry the whole verdict or none of it, and
    the agent may have no ``get_fit_summary`` to fall back on. ``report`` short-circuits the
    computation for a caller that already has one.
    """
    if report is None:
        report = noise_sigma_report(problem, fitted)
    inflated = noise_sigma_inflated(report)
    collapsed = noise_sigma_collapsed(report)
    block = {"noise_sigma_inflated": inflated, "noise_sigma_collapsed": collapsed}
    if inflated or collapsed:
        block["noise_sigma_advice"] = noise_sigma_advice_for(inflated, collapsed)
        block["noise_sigma_report"] = {
            pid: e for pid, e in report.items() if _entry_inflated(e) or _entry_collapsed(e)
        }
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
        # The effective parameter count TIC charges for. Persisted like the diagnosis below: it
        # is a statistic of the built objective at the optimum, which a later process (a resume, a
        # restore) cannot reproduce without a build. This blob is where a client's scoring reads it.
        "k_eff": payload.get("k_eff"),
        "records": payload.get("records"),
    }
    # The PARAMETER-GEOMETRY verdict of this fit (see identifiability.fit_diagnosis_block). Persisted
    # because it cannot be recomputed later without the built problem the optimizer used: a process
    # that RESUMED the session has no fit record, and re-deriving one there names an AMICI output
    # folder from live state — a full C++ build. The residual verdict beside it is recomputed
    # instead, from the records right here. Omitted when absent, so a key present always means a
    # verdict.
    block = payload.get(STORED_DIAGNOSIS_KEY)
    if block:
        out[STORED_DIAGNOSIS_KEY] = block
    # The NOISE-model report of this fit: which sigmas were estimated, what they came out at, and
    # each one's data-driven cap and fraction of its observable's std. Persisted for the same reason
    # as the diagnosis above — a later reader cannot reproduce it. Recomputing needs the fitted
    # values AND the per-observable data spread of the fit that produced them, and a scorer reading
    # the stored versions would otherwise have to RESTORE each version to see it: N restores that
    # mutate the working state and write N entries into the session's own provenance, to answer a
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
        # Persisted per version because it is the ONLY surviving record of how each start ended:
        # the per-start `exitflag`/`message` columns were removed in favour of this histogram and
        # the optimizer log runs at ERROR, so without it a restored version can no longer answer
        # "was this fit optimiser-limited or structure-limited?".
        "exitflag_counts": payload.get("exitflag_counts"),
        "n_starts_killed": payload.get("n_starts_killed"),
        "result_file": payload.get("result_file"),
        # Without these a rollback restores the per-start optima with no way to read them: the
        # `x` vectors are positional. Rollback-then-re-author is exactly the path where reusing
        # them as warm starts matters most (see `previous_optima`).
        "x_names": payload.get("x_names"),
        "x_scales": payload.get("x_scales"),
    }
