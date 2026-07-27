"""AMICI simulation/objective engine for the working PEtab v1 problem.

Uses AMICI's legacy PEtab v1 API (``amici.importers.petab.v1.import_petab_problem`` +
``amici.sim.sundials.petab.v1.simulate_petab``), which accepts in-memory
``petab.v1.Problem`` objects. Compiled models are cached by SBML hash: a model edit
changes the hash and triggers a recompile, while parameter edits are cheap — the
current ``nominalValue``s are passed as ``problem_parameters`` (on linear scale).

AMICI is a hard dependency (the server fits + simulates on every commit), so the imports
here are unconditional; ``swig`` is still located on ``PATH`` for the C++ build.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import time
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

from .formatting import finite_float


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


def _objective_amici_model(objective):
    """The AMICI model behind a pyPESTO objective.

    A plain ``AmiciObjective`` exposes ``amici_model`` directly, but pyPESTO builds an
    ``AggregatedObjective`` (which wraps several per-condition sub-objectives and has NO
    ``amici_model`` of its own) for some PEtab problems — intermittently, depending on the model,
    which was crashing evaluation with ``'AggregatedObjective' object has no attribute
    'amici_model'``. Fall through to the aggregate's sub-objectives and return the first
    ``amici_model`` found (they all share one compiled model)."""
    model = getattr(objective, "amici_model", None)
    if model is not None:
        return model
    # Explicit None check (not truthiness): an aggregate that legitimately exposes an EMPTY
    # `_objectives` must not fall through to `objectives` — an empty list is still the right,
    # authoritative container.
    subs = getattr(objective, "_objectives", None)
    if subs is None:
        subs = getattr(objective, "objectives", None)
    for sub in subs or []:
        model = getattr(sub, "amici_model", None)
        if model is not None:
            return model
    raise AttributeError(
        f"{type(objective).__name__} exposes no amici_model (nor do any sub-objectives)")


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
        # SBML hashes pyPESTO has already built a problem for this run. Mirrors ``_models``
        # for the engine's own build: it lets the fitter charge the (one-time) pyPESTO model
        # build to ``compile_seconds`` only the FIRST time a structure is seen, so a re-fit of
        # the same structure (a parameter-only edit) doesn't over-count compiles.
        self._pypesto_hashes: set[str] = set()

    def pypesto_importer_kwargs(self, sbml_hash: str | None = None) -> dict[str, str]:
        """Output kwargs for a pyPESTO ``PetabImporter``, keyed on the model's SBML hash so
        pyPESTO's own compiled module is cached/reused per structure (mirroring ``_models``).

        Shared by the fit (:mod:`fitting`) and the FIM (:mod:`identifiability`) so the
        output-folder / model-name convention lives in one place. Pass ``sbml_hash`` when the
        caller already computed it (avoids re-hashing); otherwise it is read from the session.
        """
        h = sbml_hash or self.session.sbml_hash()
        return {
            "output_folder": str(self.build_root / f"pypesto_{h}"),
            "model_name": f"petab_mcp_pp_{h}",
        }

    # -- compilation cache -------------------------------------------------
    def _get_model(self):
        _ensure_swig_on_path()
        # AMICI's SBML import SEGFAULTS (SIGSEGV, killing the whole server process) on a
        # model-less problem — an empty model whose observables reference undefined output
        # parameters, i.e. a zero-shot problem before any set_model. Refuse to hand such a model
        # to AMICI: raise a clean, actionable error instead of a hard crash. (blasi went 0/5 in a
        # sweep because a get_fit_summary/simulate on its model-less state crashed the server; the
        # simulate paths also short-circuit on has_model() — see fitting.ensure_simulation — so
        # this is the defense-in-depth backstop for any caller.)
        if not self.session.has_model():
            from .errors import ModelUpdateError

            raise ModelUpdateError(
                "No model to simulate or fit: the problem has no species/reactions yet. Build a "
                "model with make_changes before requesting a simulation, fit, or fit summary."
            )
        h = self.session.sbml_hash()
        if h not in self._models:
            # Time the model BUILD (compile/import). This runs once per unique SBML hash —
            # every later call for the same hash is a dict hit and adds nothing — and is
            # always under `session.lock` (callers simulate/evaluate under it), so the
            # accumulator write is safe. See `PetabSession.compile_seconds`.
            _compile_t0 = time.perf_counter()
            from amici.importers.petab.v1 import import_petab_problem

            build_dir = self.build_root / f"m_{h}"

            # Dump the exact model about to be handed to AMICI *before* the import call. AMICI's
            # SBML import + C++ codegen can hard-crash the process (a segfault-class failure in
            # libsbml/codegen on a pathological model) that no Python exception can catch and that
            # leaves no traceback — the MCP client just sees the connection drop (httpx.ReadError).
            # When that happens, this INFO line is the last thing in server.log and pinpoints the
            # offending model. Only logged on a fresh build (once per SBML hash), so it is not
            # noisy. (A blasi sweep run died exactly this way with no record of the culprit model.)
            # Wrapped whole in a broad guard: this is best-effort diagnostics and must NEVER block
            # the build (a missing sbml_model or a throwing libsbml accessor would otherwise do so).
            log = getattr(self.session, "logger", None)
            if log is not None:
                try:
                    sm = self.session.problem.model.sbml_model
                    ant = self.session.problem.model.to_antimony()
                    log.info(
                        "AMICI import starting (sbml_hash=%s, species=%d, reactions=%d); if the "
                        "server process dies during this import with no traceback, THIS is the "
                        "model that crashed it:\n%s",
                        h, sm.getNumSpecies(), sm.getNumReactions(), ant,
                    )
                except Exception:  # noqa: BLE001 - diagnostic only, never block the build
                    log.info("AMICI import starting (sbml_hash=%s); model dump unavailable", h)

            # compile_=None reuses an already-compiled build for this SBML hash
            # (recompiling a module already imported in-process would fail). A
            # COMPLETE hash-keyed build is always valid: identical SBML + fixed
            # tables map to an identical compiled model. Two failure modes poison
            # a hash for every future run (the dir lives outside work/, so
            # cleanup() never clears it):
            #   1. a build interrupted mid-compile (a killed run, a crash) leaves
            #      `build_dir` non-empty but not importable; compile_=None then
            #      refuses with AMICI's "Cannot compile to <dir>: not empty";
            #   2. a COMPLETE build that no longer imports (an AMICI/Python ABI
            #      change, a corrupt/half-written extension), which raises an
            #      ImportError at load time.
            # Recover from either by wiping the build and forcing a fresh compile
            # once (compile_=True — the recovery is a rebuild, not a reuse).
            try:
                self._models[h] = import_petab_problem(
                    self.session.problem,
                    output_dir=str(build_dir),
                    model_name=f"petab_mcp_{h}",
                    compile_=None,
                )
            except (ValueError, ImportError) as exc:
                # For a ValueError, match AMICI's specific "output dir not empty"
                # failure so an unrelated ValueError still surfaces; an ImportError
                # (a complete-but-unloadable build) always warrants a rebuild.
                if isinstance(exc, ValueError):
                    msg = str(exc)
                    if "Cannot compile to" not in msg or "not empty" not in msg:
                        raise
                shutil.rmtree(build_dir, ignore_errors=True)
                build_dir.mkdir(parents=True, exist_ok=True)
                self._models[h] = import_petab_problem(
                    self.session.problem,
                    output_dir=str(build_dir),
                    model_name=f"petab_mcp_{h}",
                    compile_=True,
                )
            self.session.compile_seconds += time.perf_counter() - _compile_t0
            self.session.n_compiles += 1
        return self._models[h]

    # -- public API --------------------------------------------------------
    @staticmethod
    def _res_from_rdatas(rdatas) -> list[float] | None:
        """AMICI's complete residual vector across all conditions, or ``None``.

        The chi2 fallback source (see :meth:`_chi2`): when a point has a residual but no
        petab-weighted one — its noise formula could not be evaluated — the record-based sum
        would undercount, and AMICI's own ``res`` still covers every point. ``simulate_petab``
        used to hand this over as ``result[RES]``; taking the objective's route instead, it has
        to be gathered off the per-condition ``ReturnData``. Returns ``None`` if any condition
        lacks one (an AMICI build or reporting mode that does not populate ``res``), so the
        caller falls through to an honest missing score rather than a partial sum. Only the sum
        of squares is used, so ordering across conditions does not matter.
        """
        out: list[float] = []
        for rd in rdatas:
            r = getattr(rd, "res", None)
            if r is None or len(r) == 0:
                return None
            out.extend(float(v) for v in r)
        return out or None

    def _objective_evaluate(self):
        """``(nllh, records, res)`` from the pyPESTO objective at the current nominal values.

        The single evaluation path for "what is this model's likelihood". It deliberately does
        NOT build a second AMICI model with its own default-configured solver (as a direct
        ``simulate_petab`` call would): evaluating the likelihood there while the fit evaluated
        it through pyPESTO's objective is what let the two disagree — a converged fit could be
        recorded as NaN because the *other* path failed to integrate its optimum. Here the
        number the fit minimised and the number reported for it are produced by the same
        objective, so they cannot drift.

        Per-point records still come from AMICI's ``ReturnData``: pyPESTO hands them back when
        the objective is called with ``return_dict=True`` (it switches AMICI to full reporting),
        so the simulation dataframe is unchanged.
        """
        from amici.sim.sundials.petab.v1 import RDATAS
        from pypesto.C import FVAL
        from pypesto.C import RDATAS as PP_RDATAS

        from . import fitting

        problem, _ = fitting.estimation_problem(self.session, self)
        x = fitting.problem_x_from_nominal(problem, self.session.problem.parameter_df)
        ret = problem.objective(x, sensi_orders=(0,), return_dict=True)
        rdatas = ret[PP_RDATAS]
        # pyPESTO's fval IS the negative log-likelihood it minimises.
        nllh = finite_float(ret[FVAL])
        records = self._simulation_records(_objective_amici_model(problem.objective), {RDATAS: rdatas})
        return nllh, records, self._res_from_rdatas(rdatas)

    def evaluate_objective(self) -> dict[str, Any]:
        """Score the current model + parameters against the fixed measurements."""
        with self.session.lock:
            nllh, records, res = self._objective_evaluate()

        chi2 = self._chi2(records, res)
        return {
            "neg_log_likelihood": nllh,
            "log_likelihood": -nllh if nllh is not None else None,
            "chi2": chi2,
            "n_data_points": len(records),
            "simulations": records,
        }

    def residuals(self) -> dict[str, Any]:
        """Per-point residuals (raw + noise-weighted) with an objective summary.

        This is the data source for fit inspection: every measurement point is
        returned with its ``residual`` (natural units), noise-weighted
        ``residual_weighted`` (the chi2 term), and evaluated ``noise``. These records
        back the ``simulations`` table exposed for SQL analysis via
        :mod:`petab_mcp.data_access`.
        """
        with self.session.lock:
            nllh, records, res = self._objective_evaluate()

        n_expected, n_missing = self.simulation_gap(records)
        return {
            "neg_log_likelihood": nllh,
            "chi2": self._chi2(records, res),
            "n_data_points": len(records),
            # How many measured points the trajectory never reached. Non-zero is why an
            # llh comes back NaN, so it travels with the payload instead of being
            # discoverable only by querying per-point records for nulls.
            "n_expected_points": n_expected,
            "n_missing_simulations": n_missing,
            "records": records,
        }

    # -- helpers -----------------------------------------------------------
    @staticmethod
    def _expected_points(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """The points that contribute to the objective: every measured point.

        A measurement contributes whether or not the simulation reached it — the
        neg-log-likelihood sums over all of them — so this, not the set that happens to
        have a residual, is the denominator chi2 must cover.
        """
        return [r for r in records if r.get("measurement") is not None]

    @classmethod
    def simulation_gap(cls, records: list[dict[str, Any]]) -> tuple[int, int]:
        """``(n_expected, n_missing)`` — measured points, and how many the simulation
        never reached (integration failed/stopped early, so ``simulation`` came back
        NaN and :func:`finite_float` turned it into ``None``).

        ``n_missing > 0`` means the trajectory is incomplete: AMICI's llh is NaN for
        that evaluation, and any chi2 computed from the surviving points would describe
        a different (smaller) dataset than ``n_data_points``.
        """
        expected = cls._expected_points(records)
        missing = sum(1 for r in expected if r.get("residual") is None)
        return len(expected), missing

    @classmethod
    def _chi2(cls, records: list[dict[str, Any]], res=None) -> float | None:
        """Chi2 = sum of squared noise-weighted residuals, or ``None`` if incomplete.

        Prefers the petab-canonical weighted residuals in ``records`` (which equal
        ``petab.v1.calculate.calculate_chi2``), but only when they are available for
        *every* point that contributes to the objective — a point can be missing its
        weighted residual either because its noise formula could not be evaluated, or
        because the simulation never reached it. Summing only the available terms would
        undercount and disagree with the reported neg-log-likelihood: a model whose
        trajectory dies a third of the way through the time course would otherwise report
        a plausible finite chi2 over the points it did reach, scored against the FULL
        ``n_data_points`` (wrong dof, wrong reduced_chi2, wrong GoF p-value) while its
        nll went NaN. So an incomplete set falls back to AMICI's complete ``RES`` vector,
        and failing that returns ``None`` — an honestly missing score that
        :func:`fitting.goodness_of_fit` already handles — rather than a partial sum that
        is indistinguishable from a real one.
        """
        expected = cls._expected_points(records)
        contributing = [r for r in expected if r.get("residual") is not None]
        weighted = [r.get("residual_weighted") for r in contributing]
        complete = len(contributing) == len(expected)
        if complete and contributing and all(w is not None for w in weighted):
            return float(sum(w * w for w in weighted))
        if res is not None:
            import numpy as np

            arr = np.asarray(res, dtype=float)
            # A partial simulation leaves NaN in RES too; only a finite total is usable.
            if arr.size and np.all(np.isfinite(arr)):
                return float(np.sum(arr**2))
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
        from amici.sim.sundials.petab.v1 import RDATAS, rdatas_to_simulation_df

        meas = self.session.problem.measurement_df
        simdf = rdatas_to_simulation_df(result[RDATAS], model, meas)
        keys = [OBSERVABLE_ID, SIMULATION_CONDITION_ID, TIME]
        # ``rdatas_to_simulation_df`` (a thin wrapper over amici's
        # ``rdatas_to_measurement_df``) carries exactly one simulation row per
        # measurement row, but replicate measurements share the same
        # (observableId, simulationConditionId, time) key. Merging on ``keys`` alone
        # would cross-join each size-r replicate group into r*r rows, inflating
        # ``n_data_points`` (and chi2, which counts each weighted residual r*r times).
        # Tag each row with its within-group replicate index so the merge stays
        # one-to-one with the measurements: ``len(merged) == len(meas)``. The simulated
        # value is identical across a replicate group, so pairing by index is exact
        # regardless of the (possibly reordered) row order in ``simdf``.
        rep = "_petab_mcp_replicate"
        while rep in meas.columns or rep in simdf.columns:  # avoid clobbering real columns
            rep += "_"
        meas = meas.assign(**{rep: meas.groupby(keys, sort=False).cumcount()})
        simdf = simdf.assign(**{rep: simdf.groupby(keys, sort=False).cumcount()})
        # ``validate="1:1"`` asserts (keys + rep) is unique on both sides — a fail-fast
        # guard if amici's per-measurement simulation ordering ever changes.
        merged = meas.merge(
            simdf[keys + [rep, SIMULATION]], on=keys + [rep], how="left", validate="1:1"
        )
        ctx = self._noise_context()

        records: list[dict[str, Any]] = []
        for _, row in merged.iterrows():
            m = finite_float(row.get(MEASUREMENT))
            s = finite_float(row.get(SIMULATION))
            residual = (m - s) if (m is not None and s is not None) else None
            noise, weighted = self._weighted_residual(ctx, row, m, s)
            records.append(
                {
                    "observable_id": row.get(OBSERVABLE_ID),
                    "condition_id": row.get(SIMULATION_CONDITION_ID),
                    "time": finite_float(row.get(TIME)),
                    "measurement": m,
                    "simulation": s,
                    "residual": residual,
                    "residual_weighted": weighted,
                    "noise": noise,
                }
            )
        return records
