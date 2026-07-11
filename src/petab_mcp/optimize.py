"""Parameter estimation.

pyPESTO is the intended primary backend. However, the current pyPESTO release
cannot consume a PEtab v2 problem (its importer runs the v1 linter), so when the
pyPESTO path fails we fall back to a small multi-start optimizer built directly on
AMICI's objective (``engine.evaluate_objective``) via SciPy. When a future pyPESTO
gains v2 support, the primary path activates with no API change here.

Either way, the best fit is written back into the working parameter table's
``nominal_value`` and returned. Requires the ``[optimize]`` and/or ``[amici]``
extras (the fallback needs AMICI + SciPy, both transitive deps of the sim stack).
"""

from __future__ import annotations

import math
from typing import Any


class _PypestoV2Unsupported(RuntimeError):
    """Raised when pyPESTO cannot build the v2 problem, triggering the fallback."""


def optimize(session, engine, n_starts: int = 5, seed: int = 0) -> dict[str, Any]:
    """Estimate the free parameters, writing the best fit back into the table."""
    note = None
    try:
        return _optimize_pypesto(session, n_starts, seed)
    except _PypestoV2Unsupported as exc:
        note = str(exc)
    except ImportError:
        note = "pyPESTO is not installed; used the AMICI+SciPy fallback optimizer."

    result = _optimize_scipy(session, engine, n_starts, seed)
    result["backend"] = "amici+scipy"
    result["note"] = note or (
        "pyPESTO does not yet support PEtab v2; used the AMICI+SciPy fallback optimizer."
    )
    return result


def _optimize_pypesto(session, n_starts: int, seed: int) -> dict[str, Any]:
    import pypesto
    import pypesto.optimize
    import pypesto.petab

    try:
        importer = pypesto.petab.PetabImporter(session.problem)
        problem = importer.create_problem()
    except Exception as exc:  # v1-linter crash on v2 problems, etc.
        raise _PypestoV2Unsupported(
            f"pyPESTO could not build the PEtab v2 problem "
            f"({type(exc).__name__}); used the AMICI+SciPy fallback optimizer."
        ) from exc

    result = pypesto.optimize.minimize(problem, n_starts=n_starts, progress_bar=False)
    best = result.optimize_result.list[0]
    # NOTE: pyPESTO returns x on the optimization scale (e.g. log10 for log-scaled
    # parameters). This path is currently unreachable (pyPESTO cannot build a v2
    # problem); when it becomes reachable, unscale via the importer before writing
    # back into the linear `nominal_value`.
    x = best["x"]
    fitted = {name: float(x[i]) for i, name in enumerate(problem.x_names)}
    _write_back(session, fitted)
    return {
        "backend": "pypesto",
        "n_starts": n_starts,
        "neg_log_likelihood": float(best["fval"]),
        "fitted_parameters": fitted,
    }


def _optimize_scipy(session, engine, n_starts: int, seed: int) -> dict[str, Any]:
    import numpy as np
    from scipy.optimize import minimize

    free = [p for p in session.problem.parameters if p.estimate]
    if not free:
        return {
            "n_starts": 0,
            "fitted_parameters": {},
            "message": "No estimated parameters to optimize.",
        }

    ids = [p.id for p in free]
    lb = np.array([float(p.lb) for p in free], dtype=float)
    ub = np.array([float(p.ub) for p in free], dtype=float)
    bounds = list(zip(lb, ub))
    original = {p.id: p.nominal_value for p in free}  # to restore on total failure

    def objective(x: "np.ndarray") -> float:
        for p, xi in zip(free, x):
            p.nominal_value = float(xi)
        val = float(engine.evaluate_objective()["neg_log_likelihood"])
        return val if math.isfinite(val) else 1e12  # keep the optimizer well-behaved

    # First start = current nominal values (clipped into bounds).
    x0 = np.array(
        [
            float(p.nominal_value)
            if p.nominal_value is not None
            else math.sqrt(max(l, 1e-12) * u)
            for p, l, u in zip(free, lb, ub)
        ],
        dtype=float,
    )
    x0 = np.clip(x0, lb, ub)

    rng = np.random.default_rng(seed)
    starts = [x0]
    for _ in range(max(0, n_starts - 1)):
        sample = np.empty_like(lb)
        for j in range(len(lb)):
            if lb[j] > 0:  # log-uniform when strictly positive
                sample[j] = math.exp(rng.uniform(math.log(lb[j]), math.log(ub[j])))
            else:
                sample[j] = rng.uniform(lb[j], ub[j])
        starts.append(sample)

    best_x, best_f = None, math.inf
    try:
        for start in starts:
            res = minimize(objective, start, method="L-BFGS-B", bounds=bounds)
            if math.isfinite(res.fun) and res.fun < best_f:
                best_f, best_x = float(res.fun), np.asarray(res.x, dtype=float)
    except Exception:
        # `objective` mutates nominal_value in place; restore on any failure so a
        # crashed optimization never leaves the table at arbitrary probe values.
        _write_back(session, original)
        raise

    if best_x is None:  # no start produced a finite objective
        _write_back(session, original)  # restore what we perturbed
        return {
            "n_starts": len(starts),
            "fitted_parameters": {},
            "message": "Optimization failed: no start produced a finite objective.",
        }

    fitted = {pid: float(v) for pid, v in zip(ids, best_x)}
    _write_back(session, fitted)
    return {
        "n_starts": len(starts),
        "neg_log_likelihood": best_f,
        "fitted_parameters": fitted,
    }


def _write_back(session, values: dict[str, float | None]) -> None:
    pmap = {p.id: p for p in session.problem.parameters}
    for pid, value in values.items():
        if pid in pmap:
            pmap[pid].nominal_value = None if value is None else float(value)
