"""AMICI simulation/objective engine for the working PEtab v2 problem.

Uses AMICI 1.0's PEtab v2 API (``amici.importers.petab.PetabImporter`` +
``amici.sim.sundials.petab.PetabSimulator``). Compiled models are cached by SBML
hash: a model edit changes the hash and triggers a recompile, while parameter
edits are cheap — the current ``nominal_value``s are passed to ``simulate()`` as
``problem_parameters`` (mutating the table alone does NOT propagate).

AMICI is an optional dependency; importing this module is fine without it, but the
simulate/objective calls raise a clear ``MissingDependencyError`` when it is absent.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

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
        from amici.importers.petab import PetabImporter  # noqa: F401
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
        self._importers: dict[str, Any] = {}
        self._simulators: dict[str, Any] = {}

    # -- compilation cache -------------------------------------------------
    def _get_simulator(self):
        _require_amici()
        _ensure_swig_on_path()
        h = self.session.sbml_hash()
        if h not in self._simulators:
            from amici.importers.petab import PetabImporter

            importer = PetabImporter(
                self.session.problem,
                output_dir=str(self.build_root / f"m_{h}"),
                module_name=f"petab_mcp_{h}",
                compile_=True,
                verbose=False,
            )
            self._importers[h] = importer
            self._simulators[h] = importer.create_simulator()
        return self._simulators[h]

    def _problem_parameters(self) -> dict[str, float]:
        return {
            p.id: float(p.nominal_value)
            for p in self.session.problem.parameters
            if p.nominal_value is not None
        }

    # -- public API --------------------------------------------------------
    def simulate(self) -> dict[str, Any]:
        """Simulate all experiments with the current model + parameter values."""
        with self.session.lock:
            sim = self._get_simulator()
            params = self._problem_parameters()
            result = sim.simulate(params)
            records = self._simulation_records(sim, result)
        return {"parameters": params, "simulations": records}

    def evaluate_objective(self) -> dict[str, Any]:
        """Score the current model + parameters against the fixed measurements."""
        import numpy as np

        with self.session.lock:
            sim = self._get_simulator()
            params = self._problem_parameters()
            result = sim.simulate(params)
            records = self._simulation_records(sim, result)

        llh = float(result.llh)
        res = getattr(result, "res", None)
        if callable(res):  # PetabSimulationResult.res is a method
            res = res()
        chi2 = float(np.sum(np.asarray(res, dtype=float) ** 2)) if res is not None else None
        return {
            "neg_log_likelihood": -llh,
            "log_likelihood": llh,
            "chi2": chi2,
            "n_data_points": len(records),
            "simulations": records,
        }

    # -- helpers -----------------------------------------------------------
    def _simulation_records(self, sim, result) -> list[dict[str, Any]]:
        import math

        from amici.importers.petab import rdatas_to_simulation_df

        simdf = rdatas_to_simulation_df(result.rdatas, sim.model, self.session.problem)
        meas = self.session.problem.measurement_df
        keys = ["observableId", "experimentId", "time"]
        merged = meas.merge(
            simdf[keys + ["simulation"]], on=keys, how="left"
        )

        def num(v):
            try:
                f = float(v)
                return None if math.isnan(f) or math.isinf(f) else f
            except (TypeError, ValueError):
                return None

        records: list[dict[str, Any]] = []
        for _, row in merged.iterrows():
            m = num(row.get("measurement"))
            s = num(row.get("simulation"))
            records.append(
                {
                    "observable_id": row.get("observableId"),
                    "experiment_id": row.get("experimentId"),
                    "time": num(row.get("time")),
                    "measurement": m,
                    "simulation": s,
                    "residual": (m - s) if (m is not None and s is not None) else None,
                }
            )
        return records

    def invalidate(self) -> None:
        """Drop cached simulators/importers (they self-rebuild on next use)."""
        self._importers.clear()
        self._simulators.clear()
