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
    with session.lock:
        for name, value in zip(problem.x_names, x):
            if name in df.index:
                linear = float(unscale(value, df.loc[name, PARAMETER_SCALE]))
                df.loc[name, NOMINAL_VALUE] = linear
                fitted[name] = linear

    return {
        "backend": "pypesto",
        "n_starts": n_starts,
        "neg_log_likelihood": float(best["fval"]),
        "fitted_parameters": fitted,
    }
