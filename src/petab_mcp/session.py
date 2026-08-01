"""Session state: a PEtab v1 problem with a fixed design, an Antimony-canonical model,
and persisted provenance/versioning.

The experimental design (measurements, conditions, observables) is fixed. The model
(edited only through the constrained operations in :mod:`operations`) and the parameter
table are mutable, and every committed change is snapshotted as a version and logged.
"""

from __future__ import annotations

import math
import threading
from pathlib import Path
from typing import Any

from petab.v1 import Problem
from petab.v1.models.sbml_model import SbmlModel

from . import logging_setup
from .constants import (
    ADEQUATE_GOF_PVALUE,
    OVERFIT_GOF_PVALUE,
    DEFAULT_AUTO_FIT_STARTS,
    NON_IMPROVING_STREAK_CAP,
    PLATEAU_MIN_BIC_GAIN,
    PLATEAU_WINDOW,
)
from .data_access import residual_flag, residual_structure_rollup
from .errors import ModelUpdateError, PetabMcpError, ValidationError
from .fitting import noise_sigma_collapsed, noise_sigma_inflated
from .parameters import problem_fingerprint, sha16
from .provenance import ProvenanceStore
from .references import EutilsReferenceResolver

# Appended to an early-stop ``stop_reason`` only when the best model is NOT adequate (see
# progress_report): a stall with an inadequate best is the signature of STRUCTURAL inadequacy —
# the model's form, not the optimizer or the bounds. The nudge stays model-agnostic (issue #82):
# it names generic structural MOVE CATEGORIES, never a specific mechanism, so it can't teach to a
# withheld-model benchmark.
_STRUCTURAL_PLATEAU_ADVICE = (
    "The best model is still not adequate and the fit has plateaued: this is the signature of "
    "STRUCTURAL inadequacy — the model's FORM cannot fit the data — not an optimizer or bounds "
    "problem, so re-parameterizing or widening bounds will not help. Change the model's structure "
    "rather than re-fitting the same one: consider a generic structural move (categories, not a "
    "specific answer) — add a latent / unobserved state; add a feedback / regulation reaction; or "
    "raise the interaction order (linear -> nonlinear / higher-order) — and re-author it with "
    "make_changes."
)

# Appended instead of the structural nudge when a stall happened without ever scoring a model.
# Not a modelling problem: the likelihood could not be evaluated (typically the simulation does
# not reach every measured timepoint), so no fit produced a comparable BIC.
_UNSCORED_STALL_ADVICE = (
    "No fit in this run produced a usable score, so there is no best model to compare against: "
    "this is an EVALUATION failure, not evidence about the model's form. The likelihood is "
    "undefined when the simulation does not reach every measured timepoint — check "
    "get_fit_summary's n_missing_simulations and the per-point records for unsimulated points, "
    "and address why the trajectory stops (stiffness / a parameter pinned at a bound / an "
    "unbounded rate) before drawing any conclusion about the structure."
)

# Appended to the structural nudge when the best model's chi2 GoF only looks adequate because an
# ESTIMATED noise sigma inflated to absorb the misfit (see fitting.noise_sigma_inflated / constants.
# NOISE_SIGMA_MAX_FRACTION_OF_DATA_STD). The chi2 GoF test is then powerless — it rescales to
# whatever sigma makes chi2/dof ~= 1 — so an underfit model reads as "adequate" and the run would
# stop early on it. The fix is mechanistic, not noise-side.
_SIGMA_INFLATED_ADVICE = (
    " NOTE: the best model's fit only looks statistically adequate because its ESTIMATED noise "
    "sigma is inflated above its data-driven cap, absorbing the misfit — the chi2 goodness-of-fit "
    "is not real adequacy here. Do NOT widen or lean on the noise; fix the mechanism so the "
    "residuals shrink at a plausible sigma (the residual sign/runs tests still flag the structure)."
)

# The other tail: the best model's ESTIMATED noise sigma collapsed implausibly small, so the fit
# is explaining nearly all the data's variability — overfitting the noise for a good-looking score.
_SIGMA_COLLAPSED_ADVICE = (
    " NOTE: the best model's ESTIMATED noise sigma has collapsed far below the data's own "
    "variability — the fit is absorbing nearly all the data into the mean, the fingerprint of "
    "overfitting the noise, so its low score is not genuine adequacy. Prefer a simpler / more "
    "parsimonious mechanism at a plausible sigma rather than a more flexible one that drives the "
    "residuals to ~0."
)


def _noise_sigma_assessable(report: "dict[str, Any] | None") -> bool:
    """Whether ANY entry in a noise_sigma_report was actually assessable — i.e. carries a usable
    ``fraction_of_data_std``. An entry whose observable std can't be computed is reported with
    ``status: "unknown"`` and makes both plausibility predicates fail open, so callers must not
    read "not inflated and not collapsed" as evidence of plausibility."""
    if not isinstance(report, dict):
        return False
    return any(isinstance(e, dict) and isinstance(e.get("fraction_of_data_std"), (int, float))
               for e in report.values())


class PetabSession:
    """Holds the working problem, provenance store, and reference resolver."""

    def __init__(
        self,
        yaml_path: str | Path,
        provenance_dir: str | Path | None = None,
        reference_mode: str = "best_effort",
        resolver: Any | None = None,
    ) -> None:
        self.yaml_path = Path(yaml_path).resolve()
        if not self.yaml_path.is_file():
            raise FileNotFoundError(f"PEtab YAML not found: {self.yaml_path}")
        self.problem: Problem = Problem.from_yaml(str(self.yaml_path))
        # The parameter table exactly as the problem SHIPPED it, captured before anything can
        # touch it — the reference for "which constants did the problem fix?" (see
        # `protected_fixed_parameter_ids`). It used to be read back from version 0, which was the
        # same thing only for as long as v0 was a verbatim snapshot of the load; the content-free
        # seed reconciles v0, and a durable resume replaces the working table wholesale. Reading
        # the YAML's own table is true on every process start, resume included, because
        # `Problem.from_yaml` above always re-reads the shipped `parameters.tsv`.
        self.shipped_parameter_df = self.problem.parameter_df.copy(deep=True)

        # Antimony is the canonical model form: prefer a model.ant next to the YAML.
        ant_file = self.yaml_path.parent / "model.ant"
        if ant_file.is_file():
            self._set_model_from_antimony(ant_file.read_text(encoding="utf-8"))
        elif self.problem.model is None:
            # No Antimony source and no SBML in the YAML: start from an empty model so the
            # working object always exists. `_seed_initial_model` below replaces it with a
            # content-free but VALID one; the empty skeleton survives only if seeding fails.
            self._set_model_from_antimony("model model\nend")

        self.lock = threading.RLock()
        root = Path(provenance_dir) if provenance_dir else self.yaml_path.parent / ".petab_mcp"
        self.provenance = ProvenanceStore(root)
        self.resolver = resolver or EutilsReferenceResolver(mode=reference_mode)
        self.logger = logging_setup.configure(self.provenance.root)
        # pyPESTO's own optimizer logs go to a separate <root>/pypesto.log
        logging_setup.configure_pypesto(self.provenance.root)

        # Ids of parameters that ship FIXED and are structural/mechanistic constants — the ones
        # the agent must neither free nor remove. Derived once from the initial (v0) problem and
        # cached (the shipped design never changes); see `protected_fixed_parameter_ids`.
        self._protected_fixed_ids: set[str] | None = None

        # Uncommitted working-model edits since the last fit. Edits (set_model, reaction/rate-law
        # edits, parameter bounds/scale) mutate the working model WITHOUT fitting or snapshotting a
        # version; each records an entry here. The explicit `fit` tool consumes this list: it fits
        # the working model, snapshots one version, and clears the list. So `pending_edits` non-empty
        # is exactly "the working model has been edited but not yet scored" (see `has_unfit_edits`).
        self.pending_edits: list[dict[str, Any]] = []

        # Fit + simulate on the explicit `fit` tool. The engine is wired by the server after
        # (it needs the session); until then, and in engine-free unit tests, the auto-fit
        # hook no-ops. `simulation` holds the current per-point simulation dataframe.
        self.engine: Any | None = None
        self.simulation: dict[str, Any] | None = None
        # pyPESTO traces/results of the last fit (live diagnostic; not persisted per version).
        self.optimization: dict[str, Any] | None = None
        # Reuse a cached fit when an identical estimation problem is committed again
        # (keyed on the problem identity + n_starts/seed); `last_fit_cache_hit` records
        # whether the most recent fit was served from that cache.
        self.fit_cache: bool = True
        self.last_fit_cache_hit: bool | None = None
        self.auto_fit: bool = True
        self.auto_fit_starts: int = DEFAULT_AUTO_FIT_STARTS

        # Cumulative server-side timing for this run, split so the two very different costs
        # are legible: `compile_seconds` is AMICI model BUILD time (the one-time C++ compile
        # per unique structure — ONE build now, made by `fitting.estimation_problem` and shared
        # by the fit and every objective evaluation), `fit_seconds`
        # is pyPESTO OPTIMIZATION time (the recurring multistart `minimize`). Both accumulate
        # only when work actually happens (a compile/fit cache hit adds nothing), and every
        # write is under `self.lock` (fits/sims already hold it). Read via `get_run_timing`.
        self.compile_seconds: float = 0.0
        self.fit_seconds: float = 0.0
        self.n_compiles: int = 0
        self.n_fits: int = 0

        # Early-stop chain state (see constants.py). Every genuine fitted commit is recorded as
        # one "op" so the harness can stop a run that has stopped improving and score the BEST
        # model rather than the last-committed one. ``is_new_best`` tracks BIC (see
        # fitting.compute_improvement), so the checkpoint below IS the version at which the last
        # (gated) new best fired — one coherent BIC notion drives the flag, the streak, and the
        # checkpoint:
        #   * ``non_improving_streak`` — consecutive fitted ops since the last new best; the hard
        #     stall cap fires on it.
        #   * ``best_version`` / ``best_bic`` / ``best_nll`` / ``best_gof_pvalue`` — the best-BIC
        #     fitted model so far (the checkpoint the judge restores + scores; also the reference
        #     compute_improvement gates the next fit against).
        #   * ``op_history`` — per-op (version, bic, running best_bic, gof p, is_new_best), used to
        #     surface "best-BIC improvement over the last N ops" and to detect a plateau.
        # All writes happen under ``self.lock`` on the commit path (fits already hold it); reads go
        # through ``progress_report`` which takes the lock for a consistent snapshot.
        self.non_improving_streak: int = 0
        self.best_version: int | None = None
        self.best_bic: float | None = None
        self.best_nll: float | None = None
        self.best_gof_pvalue: float | None = None
        # The best model's noise-model status, for the adequacy gate. When the noise sigma is
        # ESTIMATED the chi2 GoF is uninformative (a free sigma is fit to the residual std, so
        # chi2/dof ~= 1 by construction), so adequacy omits GoF and judges the sigma's plausibility
        # instead: best_sigma_inflated (sigma absorbing misfit past its data-driven cap) means NOT
        # adequate. Fixed-noise problems keep the GoF. (Issue: blasi/sneyd underfits stopped early
        # as "adequate" on a meaningless GoF.)
        self.best_sigma_estimated: bool = False
        self.best_sigma_inflated: bool = False
        # ...and collapsed (fitted implausibly small = overfitting the noise). Estimated-sigma
        # adequacy requires NEITHER inflated NOR collapsed, matching the judge's success gate.
        self.best_sigma_collapsed: bool = False
        # The best model's residual-structure flag (sign/runs tests, which survive estimated sigmas
        # unlike chi2 GoF). A structurally-mis-specified best model is NOT adequate regardless of the
        # noise model, so a run whose residuals stay structured keeps searching instead of
        # plateau-stopping early (sneyd: underfits stopped as "adequate" while flagging residuals).
        self.best_residual_flagged: bool = False
        # Whether the best model's estimated sigmas could actually be ASSESSED (at least one entry
        # has a usable data_std, i.e. status is not "unknown"). Without this, an unassessable sigma
        # makes both predicates fail open to False and the model reads "adequate" on no evidence.
        self.best_sigma_assessable: bool = False
        self.op_history: list[dict[str, Any]] = []

        # De-duplication memory for identical fits (see :mod:`dedup` and issue #59): how many times
        # each *identical* estimation problem has been handed to `fit` this run, keyed by structure
        # fingerprint + estimation setup (see `dedup.reauthor_key`) so that only truly identical
        # problems — not the same structure with changed bounds/scale — count toward the loop guard.
        # A within-run counter (a fresh run legitimately starts over), not persisted.
        self._fit_attempts: dict[str, int] = {}

        if self.provenance.current_version < 0:
            # Make v0 a VALID model before snapshotting it — the empty skeleton is the one state
            # nothing downstream can survive (see `_seed_initial_model`).
            self._seed_initial_model()
            # Route through _snapshot_version (not the store directly) so the initial version also
            # carries a structure_hash in its meta — otherwise v0 would be invisible to the de-dup
            # registry and a re-author of the initial model wouldn't be recognised as a revisit.
            self._snapshot_version(label="initial", kind="initial", meta={}, sim=None)
            self.provenance.append_event({"type": "initial", "version": 0})
        else:
            # durable resume: restore the working state from the latest version
            self.apply_version(self.provenance.current_version)
            # ...and seed the best-BIC checkpoint from the prior events, so the first commit after
            # a resume is judged against the run's real best rather than treated as a free new best.
            self._seed_progress_from_provenance()

    def _seed_initial_model(self) -> None:
        """Replace a model-less problem's empty skeleton with a CONTENT-FREE but VALID model.

        The empty skeleton is not a neutral "no model yet" placeholder: it is a broken PEtab
        problem whose observables reference entities nothing defines, and AMICI's SBML import
        SEGFAULTS on it — killing the server process, which no caller can catch or degrade. Every
        path that touches AMICI therefore had to be guarded against reaching it, and v0 could not
        be restored at all.

        Seeding removes the state instead of guarding it. :func:`reconcile.seed_model_antimony`
        declares one species per identifier the FIXED tables already name and nothing else — no
        reactions, no rules — so v0 becomes importable, simulatable and restorable while carrying
        no mechanism (see that function for why this leaks nothing). ``model_authored`` stays
        False until a ``make_changes`` call adds actual dynamics, so the zero-shot contract is
        unchanged.

        The seed is RECONCILED exactly like any other model — no special pass. It declares no
        mechanism, so the kinetic and initial-condition rows a zero-shot problem ships are pruned
        at v0; that is safe because :func:`reconcile.sync_parameter_table` RESTORES a shipped row
        the moment the caller's model reintroduces that parameter, bounds and priors intact.

        It then LINTS, under the same rules every edit is held to, and a failure RAISES. It
        deliberately does not degrade to the empty skeleton: that state is the one nothing
        downstream survives (AMICI's SBML import segfaults on it, killing the process), and
        leaving it reachable is what forced a ``has_model()`` check into every path that touches
        AMICI. Refusing to construct instead means a problem whose fixed tables cannot produce a
        valid model fails at startup, naming the lint errors, rather than starting and dying
        later with no traceback — and it is what makes "there is always a model" a real
        invariant rather than an expectation.
        """
        if self.has_model():
            return
        from . import reconcile as R
        from .validation import validate_problem

        antimony = R.seed_model_antimony(self.problem)
        if antimony is None:
            raise ValidationError(
                "this problem ships without a model, and its fixed tables name nothing that "
                "could be declared as one: no observable formula references an entity, no "
                "condition column names a target, and the parameter table is empty. There is "
                "nothing to simulate and nothing an initial model could be built from."
            )
        try:
            self._set_model_from_antimony(antimony)
            R.reconcile(self.problem, shipped=self.shipped_parameter_df)
        except Exception as exc:
            raise ValidationError(
                f"could not build a valid initial model for this model-less problem "
                f"({type(exc).__name__}: {exc}). The seed is generated from the fixed "
                "observable/condition/parameter tables, so this means those tables name "
                "something that cannot be declared as a model entity."
            ) from exc
        # The invariant is ENFORCED here, not assumed. The seed can lint perfectly and still
        # declare no SPECIES — a problem whose observable formulas are all constants names no
        # entity, so only the parameter rows get declared — which lands right back in the state
        # this method exists to remove. Nothing downstream would notice until AMICI segfaulted
        # on it, so check the predicate itself rather than trusting the construction.
        if not self.has_model():
            raise ValidationError(
                "this problem ships without a model and none can be built from its fixed tables: "
                "no observable formula references a model entity and no condition column names a "
                "target, so there are no species to simulate. Nothing here can be fit."
            )
        report = validate_problem(self.problem)
        if not report["valid"]:
            raise ValidationError(
                "the initial model generated for this model-less problem does not validate, so "
                "the problem cannot be worked on: its fixed tables and the model they imply are "
                "inconsistent.",
                issues=[i["message"] for i in report["issues"]
                        if i["level"] in ("error", "critical")],
            )
        self.logger.info(
            "seeded a content-free initial model: %d species, no reactions",
            self.model.sbml_model.getNumSpecies())

    def _seed_progress_from_provenance(self) -> None:
        """Seed ``best_bic`` / ``best_version`` / ``best_nll`` from prior events (durable resume).

        Scans the log for the lowest recorded ``bic`` so ``compute_improvement`` and the streak
        pick up where the last process left off. The streak and ``op_history`` stay reset — they
        are within-run heuristics whose exact sequence isn't reconstructed. Best-effort: events
        written before ``bic`` was recorded simply don't seed. Called under construction, so no
        lock is needed.
        """
        try:
            for ev in self.provenance.recent_events(0):
                bic = ev.get("bic")
                if bic is None:
                    continue
                bic = float(bic)
                # Skip non-finite scores from older events, so a resume can't seed the
                # checkpoint with a NaN that no later fit could beat (see record_fit_progress).
                if not math.isfinite(bic):
                    continue
                if self.best_bic is None or bic < self.best_bic:
                    self.best_bic = bic
                    self.best_version = ev.get("version")
                    nll = ev.get("neg_log_likelihood")
                    self.best_nll = float(nll) if nll is not None else None
                    # Seed the checkpoint's adequacy too (persisted in the fit event). Without it
                    # best_gof_pvalue stays None after a resume, so the best model reads as
                    # never-adequate and the adequate-plateau stop can't fire (an older event
                    # written before this field simply leaves it None, as before).
                    gofp = ev.get("chi2_gof_pvalue")
                    self.best_gof_pvalue = float(gofp) if gofp is not None else None
                    # Persisted alongside GoF so a resumed best keeps its adequacy verdict; older
                    # events without the fields default to False (fixed / not inflated), so the
                    # adequacy gate falls back to the GoF branch exactly as before.
                    self.best_sigma_estimated = bool(ev.get("noise_sigma_estimated", False))
                    self.best_sigma_inflated = bool(ev.get("noise_sigma_inflated", False))
                    self.best_sigma_collapsed = bool(ev.get("noise_sigma_collapsed", False))
                    self.best_residual_flagged = bool(ev.get("residual_flagged", False))
                    self.best_sigma_assessable = bool(ev.get("noise_sigma_assessable", False))
        except Exception:  # noqa: BLE001 - seeding is best-effort; a resume must never fail on it
            self.logger.warning("could not seed best-BIC checkpoint from provenance", exc_info=True)

    # -- model access ------------------------------------------------------
    @property
    def model(self):
        if self.problem.model is None:
            raise ValueError("The PEtab problem has no model.")
        return self.problem.model

    def has_model(self) -> bool:
        """Whether there is a model at all (any species or reactions).

        THIS IS NOT THE ZERO-SHOT FLAG — use :meth:`model_authored` for that. A model-less
        problem is SEEDED at construction with a content-free but valid model (see
        :meth:`_seed_initial_model`), so this is normally True from the first moment, including
        for a zero-shot problem that has authored nothing.

        In fact this is now an INVARIANT rather than a question: it is True for the whole life of
        every session. A problem that ships a model has one; a problem that does not is seeded
        with one, and :meth:`_seed_initial_model` REFUSES TO CONSTRUCT the session if it cannot
        be (including the case where the seed lints but declares no species). ``make_changes``
        then refuses to commit a model without dynamics, so nothing can take it away again.

        That is what removed the ``has_model()`` checks that used to sit in ``fit``,
        ``ensure_simulation``, ``_restore_or_simulate`` and ``identifiability``. They existed for
        one state — the empty skeleton — which AMICI's SBML import segfaults on, taking the
        server process with it where no caller can catch or degrade. Making the state
        unreachable is what let every one of them go; the two call sites left are the seeding
        path itself, which establishes the invariant rather than defending against it.
        """
        model = self.problem.model
        if model is None:
            return False
        sm = model.sbml_model
        return sm.getNumReactions() > 0 or sm.getNumSpecies() > 0

    def model_authored(self) -> bool:
        """Whether a MECHANISM has been authored — the zero-shot flag, reported as
        ``model_authored`` by ``describe_problem``.

        True once the model has dynamics: at least one reaction, or a rate/assignment rule. The
        content-free seed (:meth:`_seed_initial_model`) has neither by construction, so it reads
        False exactly as the old empty skeleton did, and ``make_changes`` refuses to commit a
        model that would leave it False (nothing to simulate is nothing to score). So this flips
        exactly once, on the first successful authoring call, which is what the harness preflight,
        the driver's no-model guard and the judge all key on.
        """
        from . import reconcile as R

        model = self.problem.model
        return model is not None and R.has_dynamics(model.sbml_model)

    def model_sbml(self) -> str:
        return self.model.to_sbml_str()

    def sbml_hash(self) -> str:
        return sha16(self.model_sbml())

    def problem_hash(self) -> str:
        """Hash of the full problem identity: model SBML + the parameter table.

        Unlike ``sbml_hash`` (structure only — the right key for the AMICI compile cache),
        this changes when a parameter's bounds/scale/estimate/value change, so it uniquely
        identifies a *fit* and is the correct key for fit artifacts (the pyPESTO result
        cache). Keying those on ``sbml_hash`` alone would collide across versions that
        differ only in the parameter table.
        """
        return problem_fingerprint(self.model_sbml(), self.problem.parameter_df)

    def _set_model_from_antimony(self, antimony: str) -> None:
        rel = getattr(self.problem.model, "rel_path", None) if self.problem.model else None
        model = SbmlModel.from_antimony(antimony, model_id=getattr(self.problem.model, "model_id", None))
        model.rel_path = rel or "model.xml"
        self.problem.model = model

    def protected_fixed_parameter_ids(self) -> set[str]:
        """Ids of parameters that ship FIXED (``estimate=0``) and are mechanistic/structural
        constants — a conservation total, a known rate, a measured concentration. These must not
        be freed *or* removed: validation rejects a commit that turns one estimated or drops it
        from the model (see ``validation._fixed_constant_role_issues``).

        Derived from the SHIPPED parameter table (``shipped_parameter_df``, captured at load)
        and cached — the shipped design never changes.
        EXCLUDED (each has its own handling, so neither is "protected"):

        * initial-condition parameters — removed legitimately when their species is removed;
          detected from the v0 model, with an ``*_init`` name fallback for the parameters a
          zero-shot v0 (an empty skeleton, or a content-free seed) cannot classify;
        * observation-model factors (noise sigmas, observable scaling/offset) — pinned by the
          fixed observable/measurement tables and given their own estimation paths.
        """
        if self._protected_fixed_ids is not None:
            return self._protected_fixed_ids
        self._protected_fixed_ids = self._compute_protected_fixed_ids()
        return self._protected_fixed_ids

    def _compute_protected_fixed_ids(self) -> set[str]:
        from petab.v1.C import ESTIMATE

        from . import reconcile as R

        ref_df = self.shipped_parameter_df
        if ref_df is None or ESTIMATE not in getattr(ref_df, "columns", []):
            return set()

        def _is_fixed(value: Any) -> bool:
            try:
                return float(value) == 0
            except (TypeError, ValueError):
                return False

        fixed = {str(pid) for pid in ref_df.index if _is_fixed(ref_df.loc[pid, ESTIMATE])}
        if not fixed:
            return set()

        ic: set[str] = {p for p in fixed if p.endswith("_init")}
        try:  # initial-condition parameters declared by the v0 model (catches author-named ICs)
            snap = self.provenance.get_version(0)
            v0_model = SbmlModel.from_antimony(snap["model_antimony"], model_id=None)
            ic |= {str(x) for x in R._ic_parameter_ids(v0_model.sbml_model)}
        except Exception:  # best-effort — the *_init fallback above still applies
            pass

        obs = {str(x) for x in (R.noise_sigma_ids(self.problem) | R.observable_scaling_ids(self.problem))}
        return fixed - ic - obs

    def describe(self) -> dict[str, Any]:
        """A summary of the problem — model size, tables, free parameters, current fit metrics.

        Backs the ``describe_problem`` tool. ``neg_log_likelihood``/``chi2`` come from the
        stored simulation and are ``None`` until the first commit/simulation.
        """
        p = self.problem
        sm = self.model.sbml_model
        return {
            "model_id": self.model.model_id,
            # Has a MECHANISM been authored? False for a problem that ships without one, whose
            # initial version is a content-free seed (species only, no dynamics) rather than the
            # broken empty skeleton it used to be. Replaces `model_initialized`, which since the
            # seed is True from construction and no longer answers the question anyone asked.
            "model_authored": self.model_authored(),
            "n_species": int(sm.getNumSpecies()),
            "n_reactions": int(sm.getNumReactions()),
            "n_model_parameters": int(sm.getNumParameters()),
            "n_table_parameters": int(len(p.parameter_df)),
            "n_estimated": int(p.n_estimated),
            "free_parameters": [str(x) for x in p.x_free_ids],
            "observables": [str(x) for x in p.get_observable_ids()],
            "n_conditions": int(len(p.condition_df)),
            "n_measurements": int(p.n_measurements),
            "n_reactions_detail": [sm.getReaction(i).getId() for i in range(sm.getNumReactions())],
            "current_version": self.provenance.current_version,
            # Working-model edits (set_model, reaction/rate-law edits, bounds/scale) do NOT fit or
            # version — the explicit `fit` tool does. `has_unfit_edits` is True when the working
            # model has been edited since the last fit, so the metrics below (from the last fitted
            # version) are stale for the working model until `fit` is called.
            "has_unfit_edits": self.has_unfit_edits(),
            "n_pending_edits": len(self.pending_edits),
            # The multistart count a fit uses when `n_starts` is omitted. Reported because
            # `n_starts` OVERRIDES it rather than adding to it: without knowing the number, a
            # caller cannot tell whether the value it passes fits harder or (silently) weaker,
            # and guessing low degrades every fit it makes.
            "fit_n_starts_default": int(self.auto_fit_starts),
            "fixed_tables": ["measurements", "conditions", "observables"],
            "neg_log_likelihood": (self.simulation or {}).get("neg_log_likelihood"),
            "chi2": (self.simulation or {}).get("chi2"),
        }

    # -- versioning --------------------------------------------------------
    def apply_version(self, version: int) -> None:
        """Set the working model + parameter table to a stored version (fixed tables kept).

        The parameter table is restored directly as the stored DataFrame (no JSON records
        round-trip). Also restores the version's stored simulation dataframe (``None`` for
        legacy versions that predate it). Stays engine-free: this runs during durable resume
        in ``__init__`` before the engine is wired.
        """
        with self.lock:
            snap = self.provenance.get_version(version)
            self._set_model_from_antimony(snap["model_antimony"])
            self.problem.parameter_df = self.provenance.get_parameter_df(version)
            self.simulation = snap.get("simulation")
            # Restore this version's fit artifacts too: the optimization traces/results are
            # persisted per version (``None`` for legacy/unfitted versions), so a rollback or
            # durable resume brings them back rather than dropping them.
            self.optimization = snap.get("optimization")
            # The working model now equals a stored version, so there are no uncommitted edits.
            self.pending_edits = []

    def _snapshot_version(
        self, *, label: str, kind: str, meta: dict[str, Any],
        sim: dict[str, Any] | None, model_ant: str | None = None,
    ) -> int:
        """Snapshot the current model + parameter table as a new version.

        The single place every commit path (single op, batch, rollback, reset) shares for
        writing a version: it always attaches the per-version simulation and optimization
        payloads, so those callers only supply the ``label``/``kind``/``meta`` and the
        simulation blob ``sim``. ``model_ant`` defaults to the current model's Antimony;
        pass it when the caller already serialized it, to avoid re-serializing.
        """
        from . import fitting
        from .parameters import structure_fingerprint

        # Stamp the structure fingerprint + BIC into every version's meta so the de-dup registry
        # (see :func:`dedup.build_registry`) can recognise a re-authored structure and rank
        # structure families by fit. Computed once here so every commit path records it
        # uniformly; best-effort, since a fingerprint/score failure must never fail a snapshot.
        meta = dict(meta or {})
        if "structure_hash" not in meta:
            try:
                meta["structure_hash"] = structure_fingerprint(
                    self.model.sbml_model, self.problem.parameter_df)
            except Exception:  # noqa: BLE001 - advisory metadata only
                pass
        if "bic" not in meta:
            nll = (sim or {}).get("neg_log_likelihood")
            fitted = (sim or {}).get("fitted_parameters")
            # k = n_estimated (the honest BIC complexity count), so a rolled-back / reset version's
            # bic matches the fitted version it restored: the restored payload is stamped with
            # n_estimated (see apply_version) and self.problem.n_estimated is authoritative for the
            # state being snapshotted. len(fitted_parameters) can differ (it omits analytically-
            # profiled inner params), so it is only a last-resort fallback.
            k = (sim or {}).get("n_estimated")
            if k is None:
                try:
                    k = int(self.problem.n_estimated)
                except Exception:  # noqa: BLE001 - bic is advisory metadata; never fail a snapshot
                    k = len(fitted) if fitted is not None else None
            meta["bic"] = fitting.goodness_of_fit(
                nll, (sim or {}).get("chi2"), k, (sim or {}).get("n_data_points")).get("bic")

        return self.provenance.snapshot_version(
            model_ant=self.model.to_antimony() if model_ant is None else model_ant,
            parameters=self.problem.parameter_df,
            label=label,
            kind=kind,
            meta=meta,
            simulation=fitting.build_sim_payload_for_version(sim),
            optimization=fitting.build_optimization_payload_for_version(self.optimization),
        )

    @staticmethod
    def _coerce_n_starts(n_starts: int | None) -> int | None:
        """Validate ``n_starts`` — ``None`` (server default) or an integer >= 1 — returning the
        coerced int. Raises :class:`ValidationError` otherwise.

        0 used to mean "simulate without optimising". It is refused now: every change is scored,
        and a version whose parameters were never optimised is not a score — it records whatever
        the nominal values happened to be, then competes on BIC against versions that were fit.
        The simulate-only pass itself is unaffected; it is still how a restore re-simulates a
        stored fit (``fit_and_simulate(do_fit=False)``), which is not a caller-reachable path.
        """
        if n_starts is None:
            return None
        try:
            n_starts = int(n_starts)
        except (TypeError, ValueError):
            raise ValidationError(f"n_starts must be an integer >= 1, got {n_starts!r}.")
        if n_starts < 1:
            raise ValidationError(f"n_starts must be an integer >= 1, got {n_starts}.")
        return n_starts

    # -- fit + simulate ----------------------------------------------------
    def _auto_fit_and_improvement(
        self, n_starts: int | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Fit + simulate for a ``fit`` call, returning (payload, improvement).

        ``n_starts`` is a multistart count >= 1; ``None`` uses the server default. No-ops (status
        ``"skipped"``) when no engine is wired or ``auto_fit`` is off. A fit/simulation error
        propagates to the caller.
        """
        from . import fitting

        n_starts = self._coerce_n_starts(n_starts)

        engine = self.engine
        if engine is None or not self.auto_fit:
            return {"status": "skipped", "neg_log_likelihood": None, "chi2": None,
                    "fitted_parameters": None, "records": []}, {}
        starts = self.auto_fit_starts if n_starts is None else n_starts
        # A fit needs at least one free parameter. Catch the zero-estimated case here with a clear,
        # actionable message rather than letting pyPESTO fail cryptically ("'NoneType' object is
        # not iterable") deep in the optimizer.
        if int(getattr(self.problem, "n_estimated", 0) or 0) == 0:
            raise ModelUpdateError(
                "the model has NO estimated (free) parameters, so there is nothing to "
                "fit. Give it at least one free parameter — a new symbol you use in a "
                "rate law becomes an estimated parameter automatically, so declare one "
                "(or keep using the shipped free parameters) — then re-author with "
                "make_changes."
            )
        fitsim = fitting.fit_and_simulate(self, engine, do_fit=True, n_starts=starts)
        # The multistart count this fit ACTUALLY used, so a caller never has to infer whether its
        # `n_starts` raised or lowered the server default (`n_starts` overrides it).
        fitsim["n_starts"] = starts
        nllh = fitsim.get("neg_log_likelihood")
        improvement = (
            fitting.compute_improvement(self, nllh, fitsim.get("objective_before"), fitsim=fitsim)
            if nllh is not None
            else {}
        )
        return fitsim, improvement

    def _restore_or_simulate(self) -> dict[str, Any] | None:
        """Ensure ``self.simulation`` is populated after restoring a version (no refit)."""
        from . import fitting

        if (self.simulation is None or not self.simulation.get("records")) and self.engine is not None:
            # A simulate-only pass regenerates the dataframe but must NOT drop the optimization
            # artifacts `apply_version` just restored — `fit_and_simulate` clears them on a
            # do_fit=False pass — so save and restore them around it.
            saved_optimization = self.optimization
            fitting.fit_and_simulate(self, self.engine, do_fit=False, n_starts=0)
            self.optimization = saved_optimization
        # A payload restored from a stored version predates the n_estimated field; stamp it
        # (from the just-applied version's parameter table) so fit_result_block can still
        # derive k and report the goodness-of-fit metrics on rollback/reset.
        sim = self.simulation
        if sim is not None and sim.get("n_estimated") is None:
            sim["n_estimated"] = int(self.problem.n_estimated)
        return sim

    # -- early-stop chain --------------------------------------------------
    def record_fit_progress(
        self, version: int, fitsim: dict[str, Any], improvement: dict[str, Any],
    ) -> None:
        """Record one genuine fitted commit as an "op" for the early-stop chain.

        Called from the commit paths (single op + batch) *after* a fit ran. Restorations
        (rollback/reset) and simulate-only passes are NOT ops — the caller only invokes this when
        a real fit produced this version — so they never perturb the streak or the checkpoint.
        Resets the non-improving streak and adopts this version as the best-BIC checkpoint when
        ``is_new_best`` fires, else increments the streak. Since ``is_new_best`` tracks BIC (see
        :func:`fitting.compute_improvement`), a new best is exactly a new best-BIC checkpoint — the
        flag, the streak, and the checkpoint stay in lock step on one score. EXCEPTION: a new best
        whose lower BIC was bought by an implausible ESTIMATED noise sigma (inflated/collapsed) does
        NOT reset the streak, and is NOT adopted at all when it would OVERWRITE a currently-plausible
        best (the judge would fail the gamed model). So a plausible would-pass checkpoint survives a
        later gamed fit, and an all-gamed run trips the streak cap instead of running to the
        recursion limit. A gamed fit is still recorded as best over an absent/already-gamed best.

        Takes ``self.lock`` itself (a reentrant ``RLock``, so this is safe whether or not the
        caller already holds it) so the write path is self-contained and symmetric with the locked
        read in :meth:`progress_report`.
        """
        from . import fitting

        nll = fitsim.get("neg_log_likelihood")
        chi2 = fitsim.get("chi2")
        n = fitsim.get("n_data_points")
        # k = estimated-parameter count. Use n_estimated (the estimate=1 count) so this BIC matches
        # the judge's — it counts k the same way (fitted_parameters can include inner/fixed params).
        k = fitsim.get("n_estimated")
        gof = fitting.goodness_of_fit(nll, chi2, k, n)
        bic = gof.get("bic")
        gof_p = gof.get("chi2_gof_pvalue")
        is_new_best = bool(improvement.get("is_new_best"))
        # Noise-model plausibility of THIS fit (single source of truth shared with the judge's
        # success gate). Reuse the flags the commit path stashed on fitsim; fall back to computing
        # them for direct callers (e.g. tests) that pass a raw fitsim without them.
        report = fitsim.get("noise_sigma_report")
        sigma_estimated = fitsim.get("noise_sigma_estimated", bool(report))
        sig_inflated = (fitsim["noise_sigma_inflated"] if "noise_sigma_inflated" in fitsim
                        else noise_sigma_inflated(report))
        sig_collapsed = (fitsim["noise_sigma_collapsed"] if "noise_sigma_collapsed" in fitsim
                         else noise_sigma_collapsed(report))
        sig_assessable = (fitsim["noise_sigma_assessable"] if "noise_sigma_assessable" in fitsim
                          else _noise_sigma_assessable(report))
        sigma_implausible = bool(sigma_estimated and (sig_inflated or sig_collapsed))

        with self.lock:
            # `bic is not None` is NOT enough: a NaN is a float, so a fit whose likelihood
            # came back non-finite would be adopted as the checkpoint and then never beaten
            # (every later `bic <= NaN - delta` is False). goodness_of_fit already maps
            # non-finite scores to None; this is the belt-and-braces guard at the write site,
            # so the checkpoint invariant — "best_bic is a real, beatable number" — holds
            # locally no matter how the score was produced.
            usable = bic is not None and math.isfinite(bic)
            # Don't let a sigma-GAMED fit OVERWRITE a currently-plausible best checkpoint: the judge
            # would fail the gamed model, so a plausible (would-pass) checkpoint must survive a
            # later, lower-BIC gamed fit — else the extra search runway could demote a passing best
            # to a gamed one and flip a would-pass to a fail. A gamed fit IS still adopted when
            # there is no plausible best to protect (best is None, or is itself already gamed), so an
            # all-gamed run still records its lowest-BIC gamed model (the adequacy gate + judge then
            # handle it), keeping the sigma-adequacy gate live rather than dead.
            best_is_plausible = (self.best_version is not None
                                 and not (self.best_sigma_inflated or self.best_sigma_collapsed))
            if is_new_best and usable and not (sigma_implausible and best_is_plausible):
                self.best_bic = bic
                self.best_version = version
                self.best_nll = nll
                self.best_gof_pvalue = gof_p
                self.best_sigma_estimated = sigma_estimated
                self.best_sigma_inflated = sig_inflated
                self.best_sigma_collapsed = sig_collapsed
                self.best_residual_flagged = bool(fitsim.get("residual_flagged", False))
                self.best_sigma_assessable = sig_assessable
                # A sigma-gamed "improvement" is not genuine progress even when adopted (it can only
                # be adopted here over an absent/already-gamed best): it does NOT reset the
                # non-improving streak, so an all-gamed run trips the streak cap instead of burning
                # to the recursion limit.
                if sigma_implausible:
                    self.non_improving_streak += 1
                else:
                    self.non_improving_streak = 0
            else:
                # Not progress -> advance the streak: a non-improving fit, a non-finite score, OR a
                # gamed fit blocked from overwriting a plausible best. None is adopted; none resets
                # the streak, so a stalled/gaming run still trips the cap instead of resetting it.
                self.non_improving_streak += 1

            self.op_history.append({
                "version": version,
                "bic": bic,
                "best_bic": self.best_bic,   # running best AFTER this op — for the plateau window
                "gof_pvalue": gof_p,
                "is_new_best": is_new_best,
            })

    def progress_report(self) -> dict[str, Any]:
        """A snapshot of the early-stop chain state — the harness polls this to stop a stalled
        run, and it is surfaced in each commit result so the agent sees progress every turn.

        ``should_stop`` (with a human-readable ``stop_reason``) is the deterministic early-stop
        signal: the hard non-improving-streak cap, or a plateau (best model adequate AND the best
        BIC gained less than ``PLATEAU_MIN_BIC_GAIN`` over the last ``PLATEAU_WINDOW`` ops). When a
        stop fires and the best model is **still not adequate**, the stall is the signature of
        STRUCTURAL inadequacy, so ``stop_reason`` also carries the issue-#82 nudge to change the
        model's FORM rather than re-parameterize. An adequate stop is a success — it gets no such
        nudge (the model fits; there is nothing structural to fix).
        """
        with self.lock:
            hist = self.op_history
            n_ops = len(hist)
            streak = self.non_improving_streak
            # best-BIC improvement over the last PLATEAU_WINDOW ops (positive => BIC still falling).
            bic_gain_over_window = None
            if n_ops > PLATEAU_WINDOW:
                prev = hist[-PLATEAU_WINDOW - 1]["best_bic"]
                cur = hist[-1]["best_bic"]
                if prev is not None and cur is not None:
                    bic_gain_over_window = prev - cur

            # Adequacy of the best model. The right test depends on the noise model:
            #  * ESTIMATED sigma — the chi2 GoF is uninformative (a free sigma is fit to the
            #    residual std, so chi2/dof ~= 1 by construction for good AND bad models alike), so
            #    OMIT it and judge on the sigma's plausibility: an INFLATED sigma (absorbing misfit)
            #    or a COLLAPSED one (overfitting the noise) means NOT adequate; otherwise adequate.
            #    Requiring neither mirrors the judge's sigma check, so a model the judge would reject
            #    for its sigma is never plateau-stopped here as "adequate". (Overall adequacy is
            #    STRICTER than the judge, not lock-step: it ALSO requires structure-free residuals
            #    below — which the judge does NOT check — so don't drop that to "restore lock-step".)
            #    This is what caught blasi/sneyd, which used to stop early on a meaningless GoF.
            #  * FIXED sigma — chi2 GoF against the known noise IS a real adequacy test; keep it,
            #    but TWO-SIDED. p below ADEQUATE_GOF_PVALUE means the residuals are too big for the
            #    known noise (underfit); p above OVERFIT_GOF_PVALUE means they are far too SMALL —
            #    the model is fitting the noise, which is not adequacy either. Testing only the
            #    lower tail scored rahman runs at chi2/n = 0.016 / 0.051 (p = 0.9999999) as
            #    "adequate". This tail only exists for a fixed sigma: an ESTIMATED sigma absorbs
            #    residual shrinkage into itself, pinning chi2/n at ~1 from below.
            gof_adequate = (
                self.best_gof_pvalue is not None
                and ADEQUATE_GOF_PVALUE <= self.best_gof_pvalue <= OVERFIT_GOF_PVALUE
            )
            if self.best_sigma_estimated and self.best_sigma_assessable:
                best_adequate = not (self.best_sigma_inflated or self.best_sigma_collapsed)
            else:
                # Either no estimated sigma at all, or one we could NOT assess (every entry status
                # "unknown" — e.g. the observable's std is uncomputable, or its noise formula is
                # composite). In the unassessable case both sigma predicates fail open to False, so
                # `not (False or False)` would declare the model adequate on NO evidence at all,
                # skipping the chi2 branch too and leaving the run with zero adequacy test. Either
                # way, fall back to the (weak, but real) GoF.
                best_adequate = gof_adequate
            # ...AND the residuals must not still show significant STRUCTURE. The sign/runs tests
            # survive estimated sigmas (unlike chi2 GoF), so a best model that fits the magnitude but
            # is systematically biased/trended has an incomplete FORM — not adequate, so the run keeps
            # searching rather than plateau-stopping (sneyd: underfits that stopped as "adequate" with
            # a plausible sigma while flagging structured residuals). A genuinely-better-than-truth
            # model can still flag residuals (irreducible data structure); it simply forgoes the early
            # plateau stop and terminates at the streak cap instead. The extra runway can't turn a
            # PASSING run into a failing one: record_fit_progress won't let a later lower-BIC gamed
            # fit OVERWRITE an already-plausible best (see the gate there), so once a run has a
            # plausible would-pass checkpoint it survives to the judge. (It does NOT make every best
            # plausible: an all-gamed run whose best was never plausible still keeps its gamed best,
            # which the judge then fails on the sigma — as it would have with or without this gate.)
            if self.best_residual_flagged:
                best_adequate = False
            stop_reason = None
            if streak >= NON_IMPROVING_STREAK_CAP:
                stop_reason = (
                    f"non-improving-streak cap reached ({streak} consecutive fitted ops with no "
                    f"new best, cap {NON_IMPROVING_STREAK_CAP})."
                )
            elif (best_adequate and bic_gain_over_window is not None
                  and bic_gain_over_window < PLATEAU_MIN_BIC_GAIN):
                basis = ("estimated noise sigma within its plausible range" if self.best_sigma_estimated
                         else f"chi2 GoF p={self.best_gof_pvalue:.3g}")
                stop_reason = (
                    f"plateau: best model is adequate ({basis}) and "
                    f"best BIC improved by only {bic_gain_over_window:.3g} over the last "
                    f"{PLATEAU_WINDOW} ops (< {PLATEAU_MIN_BIC_GAIN})."
                )
            # A stall with NO checkpoint at all is a different diagnosis: nothing has ever
            # produced a usable score, so the fits are failing to evaluate rather than the
            # model's form failing to fit. Telling the agent to change the STRUCTURE there
            # sends it re-authoring models to escape what is actually an evaluation problem.
            if stop_reason is not None and self.best_version is None:
                stop_reason = f"{stop_reason} {_UNSCORED_STALL_ADVICE}"
            # The structural-inadequacy nudge fires only when a stop coincides with a best model
            # that is NOT adequate — appending "the model's FORM cannot fit" to an adequate stop
            # would contradict it. (An adequate plateau is a success stop.)
            elif stop_reason is not None and not best_adequate:
                stop_reason = f"{stop_reason} {_STRUCTURAL_PLATEAU_ADVICE}"
                # If the inadequacy is a masking noise model, add the matching noise-side warning so
                # the agent fixes the mechanism instead of leaning on the sigma.
                if self.best_sigma_inflated:
                    stop_reason = f"{stop_reason}{_SIGMA_INFLATED_ADVICE}"
                elif self.best_sigma_collapsed:
                    stop_reason = f"{stop_reason}{_SIGMA_COLLAPSED_ADVICE}"
            return {
                "n_fitted_ops": n_ops,
                "non_improving_streak": streak,
                "streak_cap": NON_IMPROVING_STREAK_CAP,
                "best_version": self.best_version,
                "best_bic": self.best_bic,
                "best_gof_pvalue": self.best_gof_pvalue,
                "bic_improvement_over_last_n": bic_gain_over_window,
                "plateau_window": PLATEAU_WINDOW,
                "should_stop": stop_reason is not None,
                "stop_reason": stop_reason,
            }

    # -- working-model edits + fit -----------------------------------------
    def has_unfit_edits(self) -> bool:
        """Whether the working model carries edits made since the last fit (see `pending_edits`)."""
        return bool(self.pending_edits)

    def record_edit(self, operation: str, args: dict[str, Any], justification: str) -> None:
        """Log a working-model edit as pending an eventual fit (see `pending_edits`).

        Called from :mod:`operations` after an edit is applied + reconciled + validated. Callers
        already hold ``self.lock``.
        """
        self.pending_edits.append(
            {"operation": operation, "args": args, "justification": justification})

    def _fit_dedup(self, starts: int) -> tuple[dict[str, Any], str]:
        """Decide what an explicit `fit` should do with the current working model (see :mod:`dedup`).

        The de-dup gates the FIT — the expensive step. Returns ``(decision, fit_key)`` where
        ``decision`` is a :func:`dedup.decide` result and ``fit_key`` identifies this estimation
        problem (structure + setup) for the identical-fit loop guard. This method is READ-ONLY: it
        does not bump the fit-attempt counter — the caller counts an attempt only once it actually
        proceeds to a real fit (so a no-op skip or a rejected fit never inflates the guard). Callers
        hold ``self.lock``.

        ``starts`` is the multistart count this fit would run with. A skip is downgraded when it
        is HIGHER than the count HEAD was fit with: the model is unchanged, but a bigger multistart
        is a genuinely different (more thorough) search that can find a better optimum, so
        returning HEAD's weaker stored metrics would silently ignore a deliberate "fit harder".
        """
        from . import dedup as D
        from .parameters import structure_fingerprint

        structure_hash = structure_fingerprint(self.model.sbml_model, self.problem.parameter_df)
        head_version = self.provenance.current_version
        try:
            head_meta = self.provenance.get_meta(head_version)  # meta.json only — no heavy payload
        except Exception:  # noqa: BLE001 - unreadable HEAD just disables the skip/warn logic
            head_meta = {}
        head_structure_hash = head_meta.get("structure_hash")
        # A "skip" is only safe against a HEAD carrying a REAL optimized fit: a simulate-only or
        # unfitted (v0) HEAD also records a neg_log_likelihood, so gate on fit_status == "fit"
        # (stamped by every fit/rollback/reset that restores a real fit) — else a real fit after a
        # simulate-only one would be wrongly skipped, returning un-optimized metrics.
        head_is_fit = head_meta.get("fit_status") == "fit"
        try:
            head_df = self.provenance.get_parameter_df(head_version)
            same_setup = (
                D.estimation_setup_signature(self.problem.parameter_df)
                == D.estimation_setup_signature(head_df)
            )
        except Exception:  # noqa: BLE001 - can't read HEAD's table: treat setup as changed
            same_setup = False
        fit_key = D.reauthor_key(structure_hash, self.problem.parameter_df)
        prior_attempts = self._fit_attempts.get(fit_key, 0)  # READ only; incremented on proceed
        decision = D.decide(
            structure_hash=structure_hash,
            head_structure_hash=head_structure_hash,
            same_setup=same_setup,
            head_version=head_version,
            registry=D.build_registry(self),
            prior_attempts=prior_attempts,
        )
        # A "skip" means byte-identical to HEAD. Only honour it when HEAD is a real fit — otherwise
        # (the initial/unfitted model, or a simulate-only HEAD) this is the FIRST real fit of the
        # structure and must run.
        if decision["action"] == "skip" and not head_is_fit:
            decision = {"action": "proceed", "warnings": decision.get("warnings", [])}
        # ... nor when this fit would search HARDER than HEAD's did. The structure hash and the
        # setup signature both ignore the multistart count, so without this an explicit
        # `fit(n_starts=10*default)` on an unchanged model would return HEAD's weaker stored
        # metrics as if it had run. A HEAD from before n_starts was recorded (or a rollback
        # snapshot) has no `n_starts` at all, which is unknown rather than smaller, so the
        # isinstance check below keeps the skip.
        if decision["action"] == "skip":
            head_starts = head_meta.get("n_starts")
            if isinstance(head_starts, int) and starts > head_starts:
                decision = {
                    "action": "proceed",
                    "warnings": decision.get("warnings", []) + [
                        f"The model is unchanged from v{head_version}, but this fit runs "
                        f"{starts} multistarts against its {head_starts}, so it is being re-fit "
                        "rather than skipped. If you did not mean to search harder, omit n_starts."
                    ],
                }
        return decision, fit_key

    def fit(self, n_starts: int | None = None, justification: str | None = None) -> dict[str, Any]:
        """Fit the current working model, snapshot a version, and score it.

        The single fitting path: the ``fit`` tool calls this directly, and an editor called with
        ``fit=true`` (issue #114) routes through it too, so the de-duplication, the snapshot and the
        metrics are identical either way.

        Edits (``set_model``, reaction/rate-law edits, bounds/scale) mutate the working model
        without fitting; ``fit`` scores the assembled model: it fits the free parameters, simulates,
        snapshots ONE version carrying the fitted parameters + metrics + simulation dataframe,
        records the early-stop-chain op, and clears the pending edits. ``n_starts`` overrides the
        server default (and must be >= 1); ``justification`` is optional.

        The result also carries the two adequacy verdicts this method computes anyway for the
        plateau-adequacy gate: ``residual_structure`` (the compact sign/runs verdict on whether the
        model's FORM is still incomplete) and the noise-model verdict
        (``noise_sigma_inflated``/``noise_sigma_collapsed`` + advice). Returning them saves the
        caller a separate ``get_fit_summary``/``get_fit_diagnosis`` round-trip; ``get_fit_diagnosis``
        remains the way to get the per-observable table and worst points behind the verdict, and
        ``get_fit_summary`` the way for a caller that did NOT run the fit to read the same score.

        Before the (expensive) fit it de-duplicates: if the working model is byte-identical to the
        current fitted version the refit is skipped and the stored metrics are returned
        (``committed: false, no_change: true``) — unless this fit runs MORE multistarts than that
        version did, which is a more thorough search and does run; a structure revisited from
        history warns that it is already committed and scored; and the same structure fitted too
        many times is refused, to break an unproductive loop.
        """
        from . import fitting
        from .operations import _compile_fit_failure_message

        with self.lock:
            # Validate n_starts BEFORE the de-dup so a rejected value never counts toward the
            # identical-fit loop guard (which is only bumped once we commit to a real fit below).
            n_starts = self._coerce_n_starts(n_starts)

            # The count this fit would actually run with — the de-dup needs it to tell a genuine
            # "fit harder" from a redundant refit (see `_fit_dedup`). Mirrors the same expression
            # in `_auto_fit_and_improvement`.
            starts = self.auto_fit_starts if n_starts is None else n_starts
            decision, fit_key = self._fit_dedup(starts)
            if decision["action"] == "terminate":
                raise ModelUpdateError(decision["message"])
            if decision["action"] == "skip":
                return self._fit_noop_result(decision)
            warnings = decision.get("warnings", [])
            # We are committing to a real fit of this structure — count it toward the loop guard now
            # (a no-op skip / terminate / rejected n_starts above never reaches here, so only genuine
            # fit attempts are counted).
            self.note_fit_attempt(fit_key)

            # The fit is the commit. `_auto_fit_and_improvement` raises a clean ModelUpdateError when
            # the model has no estimated parameters; wrap any OTHER AMICI/pyPESTO failure (a model
            # that lints clean but cannot be compiled — a reserved-name codegen clash, an unsupported
            # construct) as a ModelUpdateError so the tool returns committed=false with an actionable
            # message rather than crashing. The working model is NOT reverted: edits are fit-free, so
            # a failed fit leaves the edited model in place for the agent to repair and re-fit.
            try:
                fitsim, improvement = self._auto_fit_and_improvement(n_starts)
            except PetabMcpError:
                raise
            except Exception as exc:  # noqa: BLE001 - surface as a clean, actionable rejection
                raise ModelUpdateError(_compile_fit_failure_message(exc)) from exc

            staged = list(self.pending_edits)
            nllh = fitsim.get("neg_log_likelihood")
            # Persist the goodness-of-fit p-value in the fit meta + event so a durable resume can
            # restore the best-BIC checkpoint's adequacy (see _seed_progress_from_provenance);
            # the events otherwise carry bic/nll but not the p-value.
            gof_pvalue = fitting.goodness_of_fit(
                nllh, fitsim.get("chi2"), fitsim.get("n_estimated"),
                fitsim.get("n_data_points")).get("chi2_gof_pvalue")
            # Residual-structure verdict of THIS fit (sign/runs tests survive estimated sigmas),
            # computed ONCE here and stashed on fitsim like the noise flags below. Three consumers:
            # the compact rollup rides on the fit RESULT so the agent doesn't spend a
            # get_fit_diagnosis call to learn its model's form is still incomplete (issue #113);
            # the bare bool goes into the version meta + event, and feeds the adequacy gate in
            # record_fit_progress/progress_report so a structurally-mis-specified best model is NOT
            # treated as adequate (see issue: sneyd underfits plateau-stopped as "adequate" while
            # flagging structured residuals). Must run BEFORE _snapshot_version so the version meta
            # can carry the flag; safe here since _auto_fit_and_improvement already set
            # self.simulation, which is what the diagnosis reads.
            fitsim["residual_structure"] = residual_structure_rollup(self)
            fitsim["residual_flagged"] = residual_flag(fitsim["residual_structure"])
            version = self._snapshot_version(
                label="fit",
                kind="fit",
                meta={"staged_edits": staged, "n_edits": len(staged),
                      # Recorded so a later fit can tell whether it would search HARDER than this
                      # one did (see `_fit_dedup`) — the structure/setup hashes both ignore it.
                      "n_starts": fitsim.get("n_starts"),
                      "fit_status": fitsim.get("status"), "neg_log_likelihood": nllh,
                      "chi2": fitsim.get("chi2"), "chi2_gof_pvalue": gof_pvalue,
                      "residual_flagged": fitsim["residual_flagged"],
                      "fitted_parameters": fitsim.get("fitted_parameters"), **improvement},
                sim=fitsim,
            )
            fitsim["version"] = version
            # Noise-sigma report + flags computed ONCE here and stashed on fitsim, so this event,
            # the checkpoint update in record_fit_progress (below) and the returned fit block all
            # read the SAME verdict instead of recomputing (or disagreeing).
            #
            # The report must be COMPUTED here: fit_and_simulate's payload never carried a
            # `noise_sigma_report` key, so the `fitsim.get(...)` this used to do read None on every
            # real fit — which silently pinned noise_sigma_estimated/inflated/collapsed to False in
            # the event, in the best-BIC checkpoint and in the plateau-adequacy gate, disabling the
            # implausible-sigma checks entirely. The unit tests inject the key into a synthetic
            # fitsim and call record_fit_progress directly, so they never caught it.
            _nsr = fitting.noise_sigma_report(self.problem, fitsim.get("fitted_parameters"))
            fitsim["noise_sigma_report"] = _nsr
            fitsim["noise_sigma_estimated"] = bool(_nsr)
            fitsim["noise_sigma_inflated"] = noise_sigma_inflated(_nsr)
            fitsim["noise_sigma_collapsed"] = noise_sigma_collapsed(_nsr)
            fitsim["noise_sigma_assessable"] = _noise_sigma_assessable(_nsr)
            # NOTE: fitsim["residual_flagged"] is already set above from the SAME rollup, as the
            # TRI-STATE bool|None (None = the sign/runs tests could not run at all). Do not
            # recompute it here: that both duplicated the residual pass and coerced None -> False,
            # so the version meta recorded null while the fit event recorded false for one fit, and
            # a resumed run restored "residuals verified clean" for a model never tested.
            event = self.provenance.append_event(
                {"type": "fit", "version": version, "justification": justification,
                 "staged_edits": staged, "n_edits": len(staged),
                 "fit_status": fitsim.get("status"), "neg_log_likelihood": nllh,
                 "chi2": fitsim.get("chi2"), "chi2_gof_pvalue": gof_pvalue,
                 "noise_sigma_estimated": fitsim["noise_sigma_estimated"],
                 "noise_sigma_inflated": fitsim["noise_sigma_inflated"],
                 "noise_sigma_collapsed": fitsim["noise_sigma_collapsed"],
                 "residual_flagged": fitsim["residual_flagged"],
                 "noise_sigma_assessable": fitsim["noise_sigma_assessable"],
                 "fitted_parameters": fitsim.get("fitted_parameters"), **improvement}
            )
            # Record this as an early-stop-chain op only when a real fit ran (simulate-only passes
            # don't move the streak/checkpoint).
            if fitsim.get("status") == "fit":
                self.record_fit_progress(version, fitsim, improvement)
            self.pending_edits = []

        return {"committed": True, "version": version, "warnings": warnings,
                "event_timestamp": event["timestamp"],
                "progress": self.progress_report(),
                **fitting.fit_result_block(
                    fitsim, improvement, self.problem.parameter_df, problem=self.problem)}

    def _fit_noop_result(self, decision: dict[str, Any]) -> dict[str, Any]:
        """Short-circuit result for a ``fit`` skipped as byte-identical to the current version.

        No recompile, no fit, no new version — the stored metrics stand. The pending edits (which
        netted to no change) are cleared and the no-op is logged so the avoided churn stays
        auditable in the provenance trace.
        """
        dup_v = decision["duplicate_of"]
        try:
            meta = self.provenance.get_meta(dup_v)  # meta.json only — the stored fit scalars
        except Exception:  # noqa: BLE001 - unreadable version: report the no-op without metrics
            meta = {}
        self.provenance.append_event({
            "type": "fit_noop", "duplicate_of_version": dup_v,
            "n_edits": len(self.pending_edits),
        })
        self.logger.info("fit no-op: identical to v%s — skipped recompile+fit+snapshot", dup_v)
        self.pending_edits = []
        from . import fitting

        structure = residual_structure_rollup(self)
        flag = residual_flag(structure)
        # Mirror the scalar fit keys a real fit returns (fit_status/n_starts/nll/chi2/bic), so a
        # caller that reads them off the result doesn't hit a missing key on a no-op; `no_change:
        # true` and the thinner shape (no per-point/per-parameter block) still distinguish it. Full
        # per-point detail is unchanged from the duplicate version — read it from that version.
        return {
            "committed": False, "no_change": True, "version": dup_v,
            "duplicate_of_version": dup_v, "message": decision.get("message"),
            "warnings": decision.get("warnings", []),
            # The multistart count the metrics being returned came from — i.e. the DUPLICATE
            # version's, since this call ran no fit of its own. `None` for a version snapshotted
            # before n_starts was recorded.
            "n_starts": meta.get("n_starts"),
            "fit_status": meta.get("fit_status"),
            "neg_log_likelihood": meta.get("neg_log_likelihood"),
            "chi2": meta.get("chi2"), "bic": meta.get("bic"),
            # BOTH adequacy verdicts, on the same terms a real fit reports them. The noise verdict
            # cannot be omitted here: its advice key is absent when the noise model is plausible,
            # so a result carrying none of these keys reads as "plausible" — and the lead has no
            # get_fit_summary to check against. Computed from the CURRENT state, which is the
            # duplicate version's by construction.
            **fitting.noise_sigma_block(
                self.problem, (self.simulation or {}).get("fitted_parameters")),
            # The structure verdict too: every doc tells the agent to read it off the fit result,
            # so a no-op must not be the one shape that lacks it. Recomputed from the CURRENT
            # simulation, which is exactly the duplicate version's — that identity is what
            # "no_change" means — rather than read from the meta, which stores only the flag.
            "residual_structure": structure,
            # Falls back to the stored flag if the tests can't run now; None (never False) when
            # neither is available: "unknown", not a clean bill of health.
            "residual_flagged": (meta.get("residual_flagged") if flag is None else flag),
        }

    def rollback_to(self, version: int, justification: str | None = None) -> dict[str, Any]:
        """Revert to a prior version, appending a new (auditable) version + event.

        Restores the target version's stored simulation dataframe (no refit — this is a
        restoration, not a new model); simulate-only fallback if that version lacks one.

        EVERY version is restorable, v0 included. That is a property of the initial model being
        VALID (see :meth:`_seed_initial_model`): v0 is either the model the problem shipped or a
        content-free seed, never the broken empty skeleton, so going back to it lands on
        something that simulates like any other version. Restoring v0 on a zero-shot problem is
        the legitimate "throw my mechanism away and start over" move, and it leaves
        ``model_authored`` False again, exactly as at construction.
        """
        from . import fitting

        with self.lock:
            self.apply_version(version)
            sim = self._restore_or_simulate()
            nllh = sim.get("neg_log_likelihood") if sim else None
            # Verdict of the RESTORED state, so the fit block reads the same everywhere and the
            # agent restoring a prior best sees its structure verdict without a follow-up call.
            structure = residual_structure_rollup(self)
            if sim is not None:
                sim["residual_structure"] = structure
            new_version = self._snapshot_version(
                label=f"rollback->v{version}",
                kind="rollback",
                # Carry the restored fit's status so the de-dup can tell a rollback of a REAL fit
                # ("fit") from one of a simulate-only/unfitted version — else re-fitting a restored
                # model would be wrongly skipped or wrongly re-run.
                meta={"rolled_back_to": version, "justification": justification,
                      "fit_status": (sim or {}).get("status"), "neg_log_likelihood": nllh,
                      "residual_flagged": residual_flag(structure)},
                sim=sim,
            )
            self.provenance.append_event(
                {
                    "type": "rollback",
                    "rolled_back_to": version,
                    "version": new_version,
                    "justification": justification,
                    "neg_log_likelihood": nllh,
                }
            )
        return {"rolled_back_to": version, "version": new_version,
                **fitting.fit_result_block(
                    sim, parameter_df=self.problem.parameter_df, problem=self.problem)}

    def restore_version(self, version: int, justification: str | None = None) -> dict[str, Any]:
        """Bring a prior version's model + parameters back as the working state (no refit).

        The cheap, correct way to "go back" to an earlier structure — it restores the target
        version's stored fit instead of recompiling and refitting, and is what ``set_model``'s
        de-dup guidance points at instead of re-authoring an already-visited model (a re-author
        recompiles, refits, and appends a byte-identical version; this does none of that). It is
        exactly :meth:`rollback_to` under a name that reads correctly for that workflow — it
        appends the same auditable rollback version and event.
        """
        return self.rollback_to(version, justification)

    def note_fit_attempt(self, key: str) -> int:
        """Record a ``fit`` of the estimation problem ``key`` (a structure+setup identity — see
        :func:`dedup.reauthor_key`), returning the count of PRIOR identical fits this run (0 the
        first time). Backs the identical-fit loop guard (see :mod:`dedup`). Callers already hold
        ``self.lock``."""
        prior = self._fit_attempts.get(key, 0)
        self._fit_attempts[key] = prior + 1
        return prior

    # -- lifecycle ---------------------------------------------------------
    def reset(self) -> dict[str, Any]:
        """Revert model + parameters to the initial version (v0).

        Safe on a zero-shot problem because v0 is the content-free SEED rather than the empty
        skeleton (see :meth:`_seed_initial_model`): resetting lands on a model that simulates,
        with ``model_authored`` back to False.
        """
        from . import fitting

        with self.lock:
            self.apply_version(0)
            sim = self._restore_or_simulate()
            nllh = sim.get("neg_log_likelihood") if sim else None
            structure = residual_structure_rollup(self)
            if sim is not None:
                sim["residual_structure"] = structure
            new_version = self._snapshot_version(
                label="reset",
                kind="reset",
                # Carry the restored sim's status (v0 is unfitted, so "simulate_only"/None) so the
                # de-dup treats a reset HEAD as not-a-real-fit and a following fit runs.
                meta={"reset_to": 0, "fit_status": (sim or {}).get("status"),
                      "neg_log_likelihood": nllh,
                      "residual_flagged": residual_flag(structure)},
                sim=sim,
            )
            self.provenance.append_event(
                {"type": "reset", "version": new_version, "neg_log_likelihood": nllh}
            )
        return {"reset": True, "version": new_version,
                **fitting.fit_result_block(
                    sim, parameter_df=self.problem.parameter_df, problem=self.problem)}
