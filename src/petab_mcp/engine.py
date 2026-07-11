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
    LIN,
    MEASUREMENT,
    OBSERVABLE_ID,
    OBSERVABLE_TRANSFORMATION,
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
        from amici.sim.sundials.petab.v1 import LLH, RES

        with self.session.lock:
            model, result, params = self._simulate()
            records = self._simulation_records(model, result)

        llh = float(result[LLH])
        chi2 = self._chi2(records, result.get(RES))
        return {
            "neg_log_likelihood": -llh,
            "log_likelihood": llh,
            "chi2": chi2,
            "n_data_points": len(records),
            "simulations": records,
        }

    def residuals(self) -> dict[str, Any]:
        """Per-point residuals (raw + noise-weighted) with an objective summary.

        This is the data source for fit inspection: every measurement point is
        returned with its ``residual`` (natural units), noise-weighted
        ``residual_weighted`` (the chi2 term), and evaluated ``noise``. See
        :mod:`petab_mcp.fit` for the cross-section aggregation built on top.
        """
        from amici.sim.sundials.petab.v1 import LLH, RES

        with self.session.lock:
            model, result, params = self._simulate()
            records = self._simulation_records(model, result)

        llh = float(result[LLH])
        return {
            "neg_log_likelihood": -llh,
            "chi2": self._chi2(records, result.get(RES)),
            "n_data_points": len(records),
            "records": records,
        }

    # -- helpers -----------------------------------------------------------
    @staticmethod
    def _chi2(records: list[dict[str, Any]], res=None) -> float | None:
        """Chi2 = sum of squared noise-weighted residuals.

        Computed from the petab-canonical weighted residuals in ``records`` (which
        equal ``petab.v1.calculate.calculate_chi2``); falls back to AMICI's ``RES``
        vector only when weighted residuals are unavailable. This keeps chi2 defined
        across AMICI versions where the ``RES`` field is not populated.
        """
        weighted = [
            r["residual_weighted"] for r in records if r.get("residual_weighted") is not None
        ]
        if weighted:
            return float(sum(w * w for w in weighted))
        if res is not None:
            import numpy as np

            arr = np.asarray(res, dtype=float)
            return float(np.sum(arr**2)) if arr.size else None
        return None

    def _noise_context(self):
        """Symbolic noise formulas + tables needed to evaluate weighted residuals.

        Returns ``(noise_formulas, observable_df, parameter_df)`` or ``None`` if the
        noise model cannot be prepared (in which case weighted residuals are skipped).
        """
        try:
            from petab.v1.calculate import get_symbolic_noise_formulas

            problem = self.session.problem
            return (
                get_symbolic_noise_formulas(problem.observable_df),
                problem.observable_df,
                problem.parameter_df,
            )
        except Exception:  # pragma: no cover - defensive; noise model unusable
            return None

    def _weighted_residual(self, ctx, row, m, s):
        """petab-consistent (noise, weighted residual) for one point, or (None, None).

        The weighted residual is ``(scale(m) - scale(s)) / sigma`` on the observable's
        transformation scale — the quantity that enters chi2 — matching
        ``petab.v1.calculate.calculate_residuals``.
        """
        if ctx is None or m is None or s is None:
            return None, None
        noise_formulas, observable_df, parameter_df = ctx
        try:
            from petab.v1 import scale as petab_scale
            from petab.v1.calculate import evaluate_noise_formula

            obs_id = row.get(OBSERVABLE_ID)
            trafo = LIN
            if obs_id in observable_df.index:
                trafo = observable_df.loc[obs_id].get(OBSERVABLE_TRANSFORMATION, LIN) or LIN
            noise = float(evaluate_noise_formula(row, noise_formulas, parameter_df, s))
            if noise == 0.0:
                return noise, None
            weighted = (petab_scale(m, trafo) - petab_scale(s, trafo)) / noise
            return noise, float(weighted)
        except Exception:  # noise formula references model states / unresolved params
            return None, None

    def _simulation_records(self, model, result) -> list[dict[str, Any]]:
        import math

        from amici.sim.sundials.petab.v1 import RDATAS, rdatas_to_simulation_df

        meas = self.session.problem.measurement_df
        simdf = rdatas_to_simulation_df(result[RDATAS], model, meas)
        keys = [OBSERVABLE_ID, SIMULATION_CONDITION_ID, TIME]
        merged = meas.merge(simdf[keys + [SIMULATION]], on=keys, how="left")
        ctx = self._noise_context()

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
            residual = (m - s) if (m is not None and s is not None) else None
            noise, weighted = self._weighted_residual(ctx, row, m, s)
            records.append(
                {
                    "observable_id": row.get(OBSERVABLE_ID),
                    "condition_id": row.get(SIMULATION_CONDITION_ID),
                    "time": num(row.get(TIME)),
                    "measurement": m,
                    "simulation": s,
                    "residual": residual,
                    "residual_weighted": weighted,
                    "noise": noise,
                }
            )
        return records

    def invalidate(self) -> None:
        """No-op: the compile cache is keyed by SBML hash and never goes stale.

        Kept for API compatibility. Clearing it would force a recompile of a
        module already imported in-process, which AMICI rejects.
        """
        return None
