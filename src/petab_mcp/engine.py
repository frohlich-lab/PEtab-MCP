"""AMICI simulation/objective engine for the working PEtab v1 problem.

Uses AMICI's legacy PEtab v1 API (``amici.importers.petab.v1.import_petab_problem`` +
``amici.sim.sundials.petab.v1.simulate_petab``), which accepts in-memory
``petab.v1.Problem`` objects. Compiled models are cached by SBML hash: a model edit
changes the hash and triggers a recompile, while parameter edits are cheap — the
current ``nominalValue``s are passed as ``problem_parameters`` (on linear scale).

AMICI is an optional dependency; importing this module is fine without it, but the
simulate/objective calls raise a clear ``MissingDependencyError`` when it is absent.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from petab.v1.C import (
    MEASUREMENT,
    OBSERVABLE_ID,
    SIMULATION,
    SIMULATION_CONDITION_ID,
    TIME,
)

from .errors import MissingDependencyError


def _ensure_swig_on_path() -> None:
    """AMICI compiles generated C++ via CMake+SWIG; make sure ``swig`` is found."""
    if shutil.which("swig"):
        return
    try:
        import swig  # the pip `swig` package bundles the binary under data/bin

        cand = Path(swig.__file__).resolve().parent / "data" / "bin"
        if (cand / "swig").exists():
            os.environ["PATH"] = str(cand) + os.pathsep + os.environ.get("PATH", "")
    except Exception:  # best-effort; AMICI will error clearly if still missing
        pass


def _require_amici():
    try:
        import amici  # noqa: F401
        from amici.importers.petab.v1 import import_petab_problem  # noqa: F401
        from amici.sim.sundials.petab.v1 import simulate_petab  # noqa: F401
    except Exception as exc:
        raise MissingDependencyError("Simulation", "amici", "amici") from exc


class AmiciEngine:
    """Compile-cached AMICI backend bound to a :class:`PetabSession`."""

    def __init__(self, session, build_root: str | Path | None = None) -> None:
        self.session = session
        self.build_root = (
            Path(build_root)
            if build_root
            else Path(tempfile.gettempdir()) / "petab_mcp_amici"
        )
        self.build_root.mkdir(parents=True, exist_ok=True)
        self._models: dict[str, Any] = {}

    # -- compilation cache -------------------------------------------------
    def _get_model(self):
        _require_amici()
        _ensure_swig_on_path()
        h = self.session.sbml_hash()
        if h not in self._models:
            from amici.importers.petab.v1 import import_petab_problem

            # compile_=None reuses an already-compiled build for this SBML hash
            # (recompiling a module already imported in-process would fail). The
            # hash-keyed cache is always valid: identical SBML + fixed tables map
            # to an identical compiled model, so there is nothing to invalidate.
            self._models[h] = import_petab_problem(
                self.session.problem,
                output_dir=str(self.build_root / f"m_{h}"),
                model_name=f"petab_mcp_{h}",
                compile_=None,
            )
        return self._models[h]

    def _problem_parameters(self) -> dict[str, float]:
        import math

        from petab.v1.C import NOMINAL_VALUE

        df = self.session.problem.parameter_df
        params: dict[str, float] = {}
        for pid in df.index:
            val = df.loc[pid, NOMINAL_VALUE]
            try:
                fval = float(val)
            except (TypeError, ValueError):
                continue
            if not (math.isnan(fval) or math.isinf(fval)):
                params[pid] = fval
        return params

    def _simulate(self):
        from amici.sim.sundials.petab.v1 import simulate_petab

        model = self._get_model()
        params = self._problem_parameters()
        result = simulate_petab(
            self.session.problem,
            amici_model=model,
            problem_parameters=params,
            scaled_parameters=False,
        )
        return model, result, params

    # -- public API --------------------------------------------------------
    def simulate(self) -> dict[str, Any]:
        """Simulate all conditions with the current model + parameter values."""
        with self.session.lock:
            model, result, params = self._simulate()
            records = self._simulation_records(model, result)
        return {"parameters": params, "simulations": records}

    def evaluate_objective(self) -> dict[str, Any]:
        """Score the current model + parameters against the fixed measurements."""
        import numpy as np
        from amici.sim.sundials.petab.v1 import LLH, RES

        with self.session.lock:
            model, result, params = self._simulate()
            records = self._simulation_records(model, result)

        llh = float(result[LLH])
        res = result.get(RES)
        chi2 = (
            float(np.sum(np.asarray(res, dtype=float) ** 2))
            if res is not None and len(np.asarray(res)) > 0
            else None
        )
        return {
            "neg_log_likelihood": -llh,
            "log_likelihood": llh,
            "chi2": chi2,
            "n_data_points": len(records),
            "simulations": records,
        }

    # -- helpers -----------------------------------------------------------
    def _simulation_records(self, model, result) -> list[dict[str, Any]]:
        import math

        from amici.sim.sundials.petab.v1 import RDATAS, rdatas_to_simulation_df

        meas = self.session.problem.measurement_df
        simdf = rdatas_to_simulation_df(result[RDATAS], model, meas)
        keys = [OBSERVABLE_ID, SIMULATION_CONDITION_ID, TIME]
        merged = meas.merge(simdf[keys + [SIMULATION]], on=keys, how="left")

        def num(v):
            try:
                f = float(v)
                return None if math.isnan(f) or math.isinf(f) else f
            except (TypeError, ValueError):
                return None

        records: list[dict[str, Any]] = []
        for _, row in merged.iterrows():
            m = num(row.get(MEASUREMENT))
            s = num(row.get(SIMULATION))
            records.append(
                {
                    "observable_id": row.get(OBSERVABLE_ID),
                    "condition_id": row.get(SIMULATION_CONDITION_ID),
                    "time": num(row.get(TIME)),
                    "measurement": m,
                    "simulation": s,
                    "residual": (m - s) if (m is not None and s is not None) else None,
                }
            )
        return records

    def invalidate(self) -> None:
        """No-op: the compile cache is keyed by SBML hash and never goes stale.

        Kept for API compatibility. Clearing it would force a recompile of a
        module already imported in-process, which AMICI rejects.
        """
        return None
