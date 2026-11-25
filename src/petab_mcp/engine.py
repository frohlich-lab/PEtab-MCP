"""AMICI simulation/objective engine for the working PEtab v2 problem.

Every likelihood is evaluated through the pyPESTO objective :func:`fitting.estimation_problem`
builds on AMICI's PEtab v2 import (``amici.importers.petab.PetabImporter`` +
``amici.sim.sundials.petab.PetabSimulator``). Compiled models are cached by
:meth:`session.PetabSession.module_hash`, in a folder per AMICI (:func:`amici_identity`): a model
edit changes it and triggers a recompile, while parameter edits are cheap — the current
``nominalValue``s are passed on each parameter's estimation scale.

AMICI is a hard dependency (the server fits + simulates on every commit), so the imports
here are unconditional; ``swig`` is still located on ``PATH`` for the C++ build.
"""

from __future__ import annotations

import importlib.metadata
import json
import logging
import os
import shutil
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from functools import cache
from pathlib import Path
from typing import Any

from petab.v2.C import (
    EXPERIMENT_ID,
    MEASUREMENT,
    NOISE_PARAMETERS,
    OBSERVABLE_ID,
    SIMULATION,
    TIME,
)

from . import filelock
from .formatting import finite_float, measurement_time
from .problem import observable_transformation, scale


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


@cache
def amici_identity() -> str:
    """The installed AMICI: its version (what ``amici.__version__`` reports), followed by the first
    12 characters of the git commit it was installed from when the install records one.

    A module another AMICI generated still imports, and may run without complaint, so
    :class:`AmiciEngine` keeps the modules of each identity in a folder of their own. The commit is
    needed because the pins here move between AMICI commits that share a version string
    (PEtab-MCP#81, #82). It is read from the install's PEP 610 record (``direct_url.json``), which
    pip and uv write for a ``git+`` requirement such as this package's pin. Deliberately not
    ``amici.__commit__``: that reads ``FETCH_HEAD`` from a ``.git`` next to a source tree, and is
    ``"unknown"`` for an installed copy. An install that records no commit (a release wheel, an
    editable checkout) is identified by its version alone, so rebuilding AMICI from a checkout that
    moved needs a fresh build root.
    """
    dist = importlib.metadata.distribution("amici")
    vcs = json.loads(dist.read_text("direct_url.json") or "{}").get("vcs_info", {})
    return f"{dist.version}-{vcs['commit_id'][:12]}" if "commit_id" in vcs else dist.version


def _build_finished(output_folder: str | Path, model_name: str) -> bool:
    """Did a build of the module ``model_name`` under ``output_folder`` finish: is its compiled
    extension there, with the mark its holder writes once the module imported (``.built``)? An
    unreadable folder raises ``OSError``."""
    folder = Path(output_folder)
    return (folder / ".built").exists() and any((folder / model_name).glob(f"_{model_name}*.so"))


def _looks_finished(output_folder: str | Path, model_name: str) -> bool:
    """:func:`_build_finished`, with an unreadable folder counted as not built."""
    try:
        return _build_finished(output_folder, model_name)
    except OSError:
        return False


@contextmanager
def build_lock(output_folder: str | Path, model_name: str) -> Iterator[None]:
    """Hold ``<output_folder>.lock`` while pyPESTO may build the module ``model_name`` there,
    waiting while another session or process, on this node or another, holds it.

    The build root is shared by every session and process (on a fit pool, every run's server, each
    building on its own node; the pool's workers only import) and keyed by AMICI and module hash, so
    two of them reaching one structure name one folder — and AMICI deletes the files it finds there
    before it generates code, so a second build corrupts the first. The second waits for the first
    (:func:`petab_mcp.filelock.lock`, which holds across the nodes of a GPFS share; ``flock`` does
    not), then finds the module finished and imports it. Nothing bounds the wait here: it lasts the
    holder's build, which its client bounds, and a holder that dies releases the lock. A module
    whose build FINISHED takes no lock: importing it is safe beside anything, and that is what every
    re-fit of a structure does. Finished means its holder marked it (``.built``) once its own import
    succeeded — the extension alone is no signal, since the install step writes it before the build
    is done.

    AMICI compiles only into a missing folder and imports whatever folder is there, so a build that
    stopped would fail every later build of its structure. Under the lock, an unfinished folder is
    one nobody is writing: it is removed before the build, so a waiter whose holder died builds the
    module itself, and a build that raises removes its own. A finished one is never removed, since
    other processes import it without the lock, so the look under the lock lets a read error through
    rather than count the folder unfinished.
    """
    folder = Path(output_folder)
    if _looks_finished(folder, model_name):
        yield
        return
    folder.parent.mkdir(parents=True, exist_ok=True)
    with filelock.lock(folder.with_name(f"{folder.name}.lock")):
        if _build_finished(folder, model_name):  # the build we waited for, or one before the lock
            yield
            return
        if folder.exists():
            logging.getLogger("petab_mcp").info("removing %s, an unfinished build", folder)
            shutil.rmtree(folder)
        try:
            yield
        except BaseException:
            if folder.exists():
                shutil.rmtree(folder, ignore_errors=True)
            raise
        (folder / ".built").touch()


def resolve_build_root(build_root: str | Path | None) -> Path:
    """``build_root``, or where a server given none builds its models: under the temporary
    directory as it is when called, which the test suite sets per xdist worker."""
    return Path(build_root) if build_root else Path(tempfile.gettempdir()) / "petab_mcp_amici"


class AmiciEngine:
    """Compile-cached AMICI backend bound to a :class:`PetabSession`."""

    def __init__(self, session, build_root: str | Path | None = None, fit_engine=None) -> None:
        self.session = session
        self.build_root = resolve_build_root(build_root)
        # Where this AMICI's modules are built and imported; see amici_identity.
        self.module_root = self.build_root / f"amici-{amici_identity()}"
        self.module_root.mkdir(parents=True, exist_ok=True)
        # The server's DaskEngine, which every fit and held-out refit of the session runs its starts
        # on; None runs them in local processes (see fitting._fit_engine).
        self.fit_engine = fit_engine
        # Module hashes pyPESTO has already built a problem for this session: it lets the fitter
        # charge the (one-time) model build to ``compile_seconds`` only the FIRST time a structure
        # is seen, so a re-fit of the same structure (a parameter-only edit) doesn't over-count
        # compiles.
        self._pypesto_hashes: set[str] = set()
        # Module hashes already written to the log by :meth:`_log_pending_import`.
        self._dumped_hashes: set[str] = set()

    def pypesto_importer_kwargs(self, module_hash: str) -> dict[str, str]:
        """Output kwargs for a pyPESTO ``PetabImporter``, keyed on the problem's module hash so
        pyPESTO's own compiled module is cached/reused per structure.

        Its ONE caller — :func:`fitting.estimation_problem` — constructs the importer on the very
        next line, which makes this the one choke point every AMICI build passes through, and so
        also where the pre-import model dump is emitted (:meth:`_log_pending_import`).

        ``module_hash`` is REQUIRED. It used to default to re-reading ``session.module_hash()``, and
        that read is how a DIAGNOSIS came to name an output folder for a structure nothing was
        built for — a full C++ build to price a model that was never fitted. Nothing may resolve a
        build target from live state inside the choke point; the fit passes the hash it fitted, and
        a diagnosis holds its built problem instead (see :class:`fitting.FitRecord`).
        """
        self._log_pending_import(module_hash)
        folder, model_name = self._module_target(module_hash)
        return {"output_folder": str(folder), "model_name": model_name}

    def _module_target(self, module_hash: str) -> tuple[Path, str]:
        """Where pyPESTO's module for ``module_hash`` lives — the naming convention itself, so
        :meth:`pypesto_importer_kwargs` (which asks for a build there) and :meth:`module_built`
        (which asks whether one happened) cannot disagree about the location."""
        return self.module_root / f"pypesto_{module_hash}", f"petab_mcp_pp_{module_hash}"

    def module_built(self, module_hash: str) -> bool:
        """Is the compiled AMICI module for this structure ALREADY there — i.e. can a problem be
        constructed for it without codegen and a C++ build?

        This is what lets a READ (a simulate-only pass for a diagnosis) tell "import a module that
        exists" — hundredths of a second — from "compile a structure nobody has built", which is
        tens of seconds and which no read may pay (see :func:`fitting.ensure_simulation`).

        Two sources, because a session outlives its process: ``_pypesto_hashes`` for what this one
        built, and the build root for what an earlier one did. What is looked for on disk is a
        FINISHED build (see :func:`build_lock`), not the output folder or its ``.so`` — a crashed or
        half-written build leaves those behind, and the next build removes them and recompiles.

        The disk half is not a resume story, and deliberately not treated as one: a restored model's
        SBML comes back with its parameter declarations in a different ORDER, so it hashes to a
        module nobody built even though the same structure sits compiled next to it. That is why the
        parameter diagnosis of a resumed session is re-served from what the fit stored rather than
        recomputed (see :func:`identifiability.fit_diagnosis_block`) — this only keeps a READ from
        paying for the mismatch.
        """
        if module_hash in self._pypesto_hashes:
            return True
        # an unreadable build root reads as nothing built, rather than risk a build
        return _looks_finished(*self._module_target(module_hash))

    def _log_pending_import(self, module_hash: str) -> None:
        """Dump the exact model about to be handed to AMICI, BEFORE the import call.

        AMICI's SBML import + C++ codegen can hard-crash the process (a segfault-class failure in
        libsbml/codegen on a pathological model) that no Python exception can catch and that
        leaves no traceback — the MCP client just sees the connection drop (``httpx.ReadError``).
        When that happens this INFO line is the last thing in ``petab_mcp.log`` and pinpoints the
        offending model; a session once died exactly this way with no record of the culprit.

        Emitted once per module hash, so it costs one write per STRUCTURE rather than per fit: a
        re-fit or a bounds-only edit rebuilds the pyPESTO problem but reuses the compiled module,
        and only a build can crash. Recording the hash before logging (not after) means a crash
        *inside* the logging call still can't put this on a retry loop.

        Wrapped whole in a broad guard: this is best-effort diagnostics and must NEVER block a
        build — a throwing libsbml accessor or an un-round-trippable model would otherwise do so.
        """
        log = self.session.logger
        if module_hash in self._dumped_hashes:
            return
        self._dumped_hashes.add(module_hash)
        try:
            sm = self.session.problem.model.sbml_model
            ant = self.session.problem.model.to_antimony()
            log.info(
                "AMICI import starting (module_hash=%s, species=%d, reactions=%d); if the server "
                "process dies during this import with no traceback, THIS is the model that "
                "crashed it:\n%s",
                module_hash,
                sm.getNumSpecies(),
                sm.getNumReactions(),
                ant,
            )
        except Exception:  # noqa: BLE001 - diagnostic only, never block the build
            log.info("AMICI import starting (module_hash=%s); model dump unavailable", module_hash)

    # NOTE: there is no compile cache here any more. This class used to build its OWN AMICI
    # model (`_get_model`) alongside the one pyPESTO builds for the fit — two full C++ builds of
    # one structure. Unifying every likelihood evaluation onto `fitting.estimation_problem` (see
    # its docstring) left that second build unused, so it is gone: the AMICI model now comes off
    # the pyPESTO objective, and pyPESTO caches its own compiled module under
    # `pypesto_importer_kwargs` — which is also where the pre-import dump `_get_model` used to
    # emit now lives.

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

    def _objective_evaluate(self, solve_inner: bool = True, gradient: bool = False):
        """``(nllh, records, res)`` from the pyPESTO objective at the current nominal values.
        ``solve_inner`` (the fit's objective) solves the noise sigmas and observable scalings at
        every evaluation instead of reading them; ``solve_inner=False`` evaluates at the table's
        own values. See :func:`fitting.estimation_problem`. ``gradient`` evaluates it with its
        gradient, as the fit's optimizer does.

        The single evaluation path for "what is this model's likelihood". It deliberately does
        NOT build a second AMICI model with its own default-configured solver (as a direct
        ``simulate_petab`` call would): evaluating the likelihood there while the fit evaluated
        it through pyPESTO's objective is what let the two disagree — a converged fit could be
        recorded as NaN because the *other* path failed to integrate its optimum.

        The same objective still gives two values at one point when one evaluation has the
        gradient and the other has not. With sensitivities, AMICI integrates a pre- or
        post-equilibration until they have converged as well as the states (its
        ``sensi_steady_state_check``), so a slow mode leaves the steady state reached without
        them short of the one the optimizer scored: a pre-equilibrated toy re-scored 6% worse
        that way (#265). Hence ``gradient`` on the re-scoring of a fit.

        Per-point records still come from AMICI's ``ReturnData``: pyPESTO hands them back when
        the objective is called with ``return_dict=True`` (it switches AMICI to full reporting),
        so the simulation dataframe is unchanged. Returns ``(nllh, records, res, states,
        failed)``: :meth:`_state_trajectories` off the same ``ReturnData``, and the experiments
        whose simulation failed, with AMICI's status (:func:`fitting.amici_failures`).
        """
        from pypesto.C import FVAL, RDATAS

        from . import fitting

        problem, _ = fitting.estimation_problem(self.session, self, solve_inner=solve_inner)
        x = fitting.problem_x_from_nominal(problem, self.session.problem.parameter_df)
        ret = problem.objective(x, sensi_orders=(0, 1) if gradient else (0,), return_dict=True)
        rdatas = ret[RDATAS]
        # pyPESTO's fval IS the negative log-likelihood it minimises.
        nllh = finite_float(ret[FVAL])
        records = self._simulation_records(problem.objective, rdatas)
        if any(r.get("measurement") is not None and r.get("simulation") is None for r in records):
            # A measured point with no simulated value has no likelihood, so neither has the
            # model: a MISSING score, as for a failed integration. Not the fval: pyPESTO's
            # hierarchical (relative-data) objective drops a NaN simulation from its sums and
            # returns a finite, LOWER value at sensitivity orders 0 and 1 (ICB-DCM/pyPESTO#1794;
            # AMICI's own llh is NaN) -- a 0/0 readout at t=0 scored finite and lower than any
            # real fit.
            nllh = None
        states = self._state_trajectories(problem.objective.amici_model, rdatas)
        failed = fitting.amici_failures(rdatas)
        return nllh, records, self._res_from_rdatas(rdatas), states, failed

    def _state_trajectories(self, model, rdatas) -> dict[str, Any] | None:
        """The model's species and assignment rules at every time AMICI simulated, off the
        ``rdatas`` of one evaluation: ``{"ids", "points", "values"}``, one ``[experiment_id, time]``
        and one row of values in ``ids`` order per point, which :mod:`data_access` serves as
        ``state_trajectories``. ``None`` where an experiment returned no output, as the records then
        hold no simulation either.

        Those times are the measurement times, each once. AMICI's full reporting holds every
        state (``x``) and expression (``w``) there, so this needs no simulation of its own, where
        a time grid would integrate every experiment again for a read. The states are AMICI's
        state vector, rate-rule targets included. Of the expressions only the SBML model's
        assignment-rule targets are kept, not the ``flux_*`` AMICI adds per reaction.
        """
        import numpy as np

        if self._unusable_rdatas(rdatas):
            return None
        sbml = self.session.model.sbml_model
        rules = {r.getVariable() for r in sbml.getListOfRules() if r.isAssignment()}
        expressions = list(model.get_expression_ids())
        kept = [i for i, e in enumerate(expressions) if e in rules]
        experiments = set(self.session.problem.measurement_df[EXPERIMENT_ID])
        points, values = [], []
        for rd in rdatas:
            # AMICI names an experiment-less simulation with an id of its own: "" here.
            eid = rd.id if rd.id in experiments else ""
            x = np.asarray(rd.x)
            w = np.asarray(rd.w)[:, kept] if kept else np.empty((len(x), 0))
            ts = list(rd.ts)
            for j, t in enumerate(ts):
                if j and t == ts[j - 1]:
                    continue  # a replicate's repeat of the time before it
                points.append([eid, finite_float(t)])
                values.append([finite_float(v) for v in (*x[j], *w[j])])
        ids = [*model.get_state_ids(), *(expressions[i] for i in kept)]
        return {"ids": ids, "points": points, "values": values}

    def evaluate_objective(self) -> dict[str, Any]:
        """Score the current model + parameters against the fixed measurements."""
        with self.session.lock:
            nllh, records, res, _, _ = self._objective_evaluate()

        chi2 = self._chi2(records, res)
        return {
            "neg_log_likelihood": nllh,
            "log_likelihood": -nllh if nllh is not None else None,
            "chi2": chi2,
            "n_data_points": len(records),
            "simulations": records,
        }

    def residuals(self, solve_inner: bool = True, gradient: bool = False) -> dict[str, Any]:
        """Per-point residuals (raw + noise-weighted) with an objective summary; ``solve_inner``
        and ``gradient`` as in :meth:`_objective_evaluate`.

        This is the data source for fit inspection: every measurement point is
        returned with its ``residual`` (natural units), noise-weighted
        ``residual_weighted`` (signed (m − s)/σ; its square is the chi2 term), and evaluated
        ``noise``. These records back the ``simulations`` table exposed for SQL analysis via
        :mod:`petab_mcp.data_access`.
        """
        with self.session.lock:
            nllh, records, res, states, failed = self._objective_evaluate(solve_inner, gradient)

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
            "state_trajectories": states,
            # Which experiments did not simulate, and AMICI's status for each: the fit's
            # re-scoring names them where it leaves the fit unscored.
            "failed_experiments": failed,
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
        ``petab.v2.calculate.calculate_chi2``), but only when they are available for
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

    @staticmethod
    def _noise_context(problem):
        """Symbolic noise formulas + tables needed to evaluate ``problem``'s weighted residuals.

        Returns ``(noise_formulas, observable_df, parameter_df)`` or ``None`` if the
        noise model cannot be prepared (in which case weighted residuals are skipped).
        """
        try:
            from petab.v2.calculate import get_symbolic_noise_formulas

            return (
                get_symbolic_noise_formulas(problem.observable_df),
                problem.observable_df,
                problem.parameter_df,
            )
        except Exception:  # pragma: no cover - defensive; noise model unusable
            return None

    @staticmethod
    def _weighted_residual(ctx, row, m, s, sigmay=None):
        """petab-consistent (noise, weighted residual) for one point, or (None, None).

        The weighted residual is ``(scale(m) - scale(s)) / sigma`` in the coordinate the
        observable's noise distribution is additive in — the quantity that enters chi2 —
        matching ``petab.v2.calculate.calculate_residuals``. ``sigma`` is the noise formula as
        petab evaluates it from the parameter table and the measurement's overrides, or, where
        the formula names a model entity petab cannot evaluate (``sd*readout_X``, #289),
        ``sigmay``: the noise AMICI evaluated for the point, which its likelihood used.
        """
        if ctx is None or m is None or s is None:
            return None, None
        noise_formulas, observable_df, parameter_df = ctx
        try:
            from petab.v2.calculate import evaluate_noise_formula

            obs_id = row.get(OBSERVABLE_ID)
            trafo = observable_transformation(observable_df, obs_id)
            observable = observable_df.loc[obs_id].to_dict()
            try:
                noise = float(
                    evaluate_noise_formula(row, noise_formulas, parameter_df, s, observable)
                )
            except Exception:  # the formula names a model entity
                noise = sigmay
            if noise is None:
                return None, None
            if noise == 0.0:
                return noise, None
            weighted = (scale(m, trafo) - scale(s, trafo)) / noise
            return noise, float(weighted)
        except Exception:  # an observable petab cannot read
            return None, None

    @staticmethod
    def _sigmay(model, rdatas, experiments) -> dict[tuple, float | None]:
        """AMICI's ``sigmay`` by ``(experiment_id, observable_id, time, replicate)``, the replicate
        counted in measurement order as ``ExpData`` lays replicates out: the ``r``-th measurement
        at a time sits ``r`` rows after the time's first. ``experiments`` are the measurement
        table's ids; AMICI's id for an experiment-less simulation is ``""`` here."""
        import numpy as np

        observables = list(model.get_observable_ids())
        out: dict[tuple, float | None] = {}
        for rd in rdatas:
            eid = rd.id if rd.id in experiments else ""
            sigmay = np.asarray(rd.sigmay)
            first: dict[float, int] = {}
            for j, t in enumerate(float(t) for t in rd.ts):
                r = j - first.setdefault(t, j)
                for i, obs in enumerate(observables):
                    out[(eid, obs, t, r)] = finite_float(sigmay[j, i])
        return out

    @staticmethod
    def _unusable_rdatas(rdatas) -> int:
        """How many conditions came back with no usable simulation output.

        AMICI's failfast path returns a ``ReturnData`` whose output arrays are ``None`` rather than
        raising. ``rdatas_to_simulation_df`` then iterates them and dies with
        ``'NoneType' object is not iterable`` — a bare ``TypeError`` that
        ``_compile_fit_failure_message`` reads as an ILL-POSED ODE, telling the agent to rewrite a
        model whose only problem was that one integration failed at its nominal values. That is the
        precise outcome ``fitting._run_fit`` documents it must avoid (see its ``objective_before``
        comment): a failed pre-fit evaluation is a MISSING score, not a structural verdict.
        Checked per condition here rather than by string-matching the TypeError downstream.
        """
        bad = 0
        for rd in rdatas:
            ts = getattr(rd, "ts", None)
            y = getattr(rd, "y", None)
            if ts is None or y is None or len(ts) == 0:
                bad += 1
        return bad

    @staticmethod
    def _noise_parameters(row) -> str | None:
        """The measurement row's ``noiseParameters`` cell as text, ``None`` when blank."""
        value = row.get(NOISE_PARAMETERS)
        return None if value is None or value != value or value == "" else str(value)

    @staticmethod
    def _records_without_simulation(meas) -> list[dict[str, Any]]:
        """One record per measurement with ``simulation`` unset — the honest shape for an
        evaluation that produced no trajectory. ``simulation_gap`` counts every one of these as
        missing, ``_chi2`` declines to score them and ``goodness_of_fit`` reports the gap, so the
        whole downstream chain already handles this; it only ever needed the records to exist."""
        return [
            {
                "observable_id": row.get(OBSERVABLE_ID),
                "experiment_id": row.get(EXPERIMENT_ID),
                "time": measurement_time(row.get(TIME)),
                "measurement": finite_float(row.get(MEASUREMENT)),
                "simulation": None,
                "residual": None,
                "residual_weighted": None,
                "noise": None,
                "noise_parameters": AmiciEngine._noise_parameters(row),
            }
            for _, row in meas.iterrows()
        ]

    def _simulation_records(self, objective, rdatas) -> list[dict[str, Any]]:
        """The session's records: :meth:`_records_from_rdatas` on the session's problem."""
        from .fitting import amici_facing_problem

        amici_problem, flattened = amici_facing_problem(self.session)
        return self._records_from_rdatas(
            self.session.problem, objective, rdatas, amici_problem if flattened else None
        )

    @classmethod
    def _records_from_rdatas(
        cls, problem, objective, rdatas, flattened=None
    ) -> list[dict[str, Any]]:
        """One record per measurement of ``problem``, off the ``rdatas`` its ``objective``
        returned: the rows of ``simulations.json``, and of a held-out score's
        (:mod:`petab_mcp.validation_data`). ``flattened`` is the flattened copy AMICI was built
        from when it was (:func:`fitting.amici_facing_problem`); a held-out view never is."""
        from amici.importers.petab import unflatten_simulation_df

        # The AMICI model was built from the AMICI-FACING problem, which may be a flattened
        # copy (timepoint-specific overrides; see fitting.needs_flattening). The rdatas name
        # observables in THAT problem's vocabulary, so the simulation frame is paired with its
        # measurement table and mapped back to the session's (unflattened) observable ids with
        # AMICI's own inverse before anything downstream reads it — every consumer (residual
        # structure, query_data's `simulations`, the sigma checks) keys on the unflattened ids.
        n_bad = cls._unusable_rdatas(rdatas)
        if n_bad:
            # Partial success is treated the same as total failure on purpose: when any experiment
            # fails to integrate, AMICI's likelihood for the evaluation is already NaN (see
            # `simulation_gap`), so keeping the surviving experiments' simulations would describe a
            # smaller dataset than `n_data_points` while looking like a complete one.
            logging.getLogger("petab_mcp").info(
                "simulation: %d/%d experiments returned no usable trajectory; reporting a missing "
                "score instead of a partial one",
                n_bad,
                len(rdatas),
            )
            return cls._records_without_simulation(problem.measurement_df)
        meas = (problem if flattened is None else flattened).measurement_df
        # One row per measurement of the problem AMICI imported, grouped by experiment; an
        # experiment-less measurement comes back with a NaN id, which is "" here (see
        # petab_mcp.problem).
        simdf = objective.rdatas_to_simulation_df(rdatas)
        simdf[EXPERIMENT_ID] = [v if isinstance(v, str) else "" for v in simdf[EXPERIMENT_ID]]
        keys = [OBSERVABLE_ID, EXPERIMENT_ID, TIME]
        # PAIR IN THE AMICI VOCABULARY, then relabel — not the other way round. Unflattening the
        # simulation frame first and merging against the natural measurements looked equivalent
        # and is not: flattening is triggered PRECISELY when two rows share
        # (observable, experiment, time) with different overrides, so in the natural vocabulary
        # those rows are not replicates and their simulated values differ — the positional
        # `cumcount` pairing below would match them by row order. Measured under v1 on a problem
        # with per-gel overrides: with the gel rows in a different but perfectly legal TSV order,
        # 72/72 rows mispaired (one measurement drew a simulation 26x off), silently corrupting
        # every residual, chi2 and sigma check. In the FLATTENED vocabulary the override lives in
        # the observable id, so (observable, experiment, time) is unique per design point and the
        # merge cannot mispair.
        # The simulation frame carries exactly one row per measurement row, but replicate
        # measurements share the same (observableId, experimentId, time) key. Merging on ``keys``
        # alone would cross-join each size-r replicate group into r*r rows, inflating
        # ``n_data_points`` (and chi2, which counts each weighted residual r*r times).
        # Tag each row with its within-group replicate index so the merge stays
        # one-to-one with the measurements: ``len(merged) == len(meas)``. Pairing WITHIN a
        # group is by position, which is exact only because a group reaching this point is a
        # true replicate group -- same observable, experiment, time AND overrides -- whose rows
        # therefore share one simulated value. That is what the flattened vocabulary above buys;
        # in the natural one the claim is false for override-carrying rows.
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
        # Row by row with `merged`, looked up in AMICI's vocabulary like the simulation.
        sigmay = cls._sigmay(objective.amici_model, rdatas, set(meas[EXPERIMENT_ID]))
        sigmas = [
            sigmay.get((e, o, float(t), r))
            for e, o, t, r in zip(
                merged[EXPERIMENT_ID], merged[OBSERVABLE_ID], merged[TIME], merged[rep]
            )
        ]
        if flattened is not None:
            # Back to the session's vocabulary for everything downstream (records, residual
            # structure, query_data). AMICI's inverse does exactly this and nothing else -- it
            # replaces the observableId column -- so `merged` (a measurement frame carrying a
            # `simulation` column) is precisely the shape it documents.
            #
            # The noise substitution below reads the NATURAL observable_df's formula together
            # with this row's own noiseParameters, which flattening leaves untouched, so the
            # pair stays consistent.
            merged = unflatten_simulation_df(merged, problem.to_petab())
        ctx = cls._noise_context(problem)

        records: list[dict[str, Any]] = []
        for (_, row), sigma in zip(merged.iterrows(), sigmas):
            m = finite_float(row.get(MEASUREMENT))
            s = finite_float(row.get(SIMULATION))
            residual = (m - s) if (m is not None and s is not None) else None
            noise, weighted = cls._weighted_residual(ctx, row, m, s, sigma)
            records.append(
                {
                    "observable_id": row.get(OBSERVABLE_ID),
                    "experiment_id": row.get(EXPERIMENT_ID),
                    "time": measurement_time(row.get(TIME)),
                    "measurement": m,
                    "simulation": s,
                    "residual": residual,
                    "residual_weighted": weighted,
                    "noise": noise,
                    "noise_parameters": cls._noise_parameters(row),
                }
            )
        return records
