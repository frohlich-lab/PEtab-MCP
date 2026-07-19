"""Session state: a PEtab v1 problem with a fixed design, an Antimony-canonical model,
and persisted provenance/versioning.

The experimental design (measurements, conditions, observables) is fixed. The model
(edited only through the constrained operations in :mod:`operations`) and the parameter
table are mutable, and every committed change is snapshotted as a version and logged.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

from petab.v1 import Problem
from petab.v1.models.sbml_model import SbmlModel

from . import logging_setup
from .constants import DEFAULT_AUTO_FIT_STARTS
from .errors import BatchError, ModelUpdateError, ValidationError
from .parameters import problem_fingerprint, sha16
from .provenance import ProvenanceStore
from .references import EutilsReferenceResolver


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

        # Antimony is the canonical model form: prefer a model.ant next to the YAML.
        ant_file = self.yaml_path.parent / "model.ant"
        if ant_file.is_file():
            self._set_model_from_antimony(ant_file.read_text(encoding="utf-8"))
        elif self.problem.model is None:
            # No Antimony source and no SBML in the YAML: start from an empty model so
            # the working object always exists. The model is "uninitialized" (see
            # `has_model`) until the agent authors one via the `set_model` operation.
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

        # batch (staging) state — chain several operations into one version
        self.batch_active: bool = False
        self.batch_label: str | None = None
        self.batch_start_version: int = -1
        self.staged: list[dict[str, Any]] = []

        # Fit + simulate on commit. The engine is wired by the server after construction
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
        # per unique structure — engine `_get_model` and pyPESTO's own build), `fit_seconds`
        # is pyPESTO OPTIMIZATION time (the recurring multistart `minimize`). Both accumulate
        # only when work actually happens (a compile/fit cache hit adds nothing), and every
        # write is under `self.lock` (fits/sims already hold it). Read via `get_run_timing`.
        self.compile_seconds: float = 0.0
        self.fit_seconds: float = 0.0
        self.n_compiles: int = 0
        self.n_fits: int = 0

        if self.provenance.current_version < 0:
            self.provenance.snapshot_version(
                model_ant=self.model.to_antimony(),
                parameters=self.problem.parameter_df,
                label="initial",
                kind="initial",
            )
            self.provenance.append_event({"type": "initial", "version": 0})
        else:
            # durable resume: restore the working state from the latest version
            self.apply_version(self.provenance.current_version)

    # -- model access ------------------------------------------------------
    @property
    def model(self):
        if self.problem.model is None:
            raise ValueError("The PEtab problem has no model.")
        return self.problem.model

    def has_model(self) -> bool:
        """Whether a model has been *initialized* (has any species or reactions).

        A freshly loaded problem with no Antimony/SBML source carries an empty model
        skeleton; ``set_model`` authors the real model into it (and can re-author it later).
        Structural edits (``add_reaction`` etc.) require this to be True.
        """
        model = self.problem.model
        if model is None:
            return False
        sm = model.sbml_model
        return sm.getNumReactions() > 0 or sm.getNumSpecies() > 0

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

        Derived from the initial (v0) problem and cached — the shipped design never changes.
        EXCLUDED (each has its own handling, so neither is "protected"):

        * initial-condition parameters — removed legitimately when their species is removed;
          detected from the v0 model, with an ``*_init`` name fallback for a model-less v0 whose
          empty model can't classify them;
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

        try:
            ref_df = self.provenance.get_parameter_df(0)
        except Exception:  # no v0 yet (shouldn't happen post-init) — nothing to protect
            return set()
        if ESTIMATE not in getattr(ref_df, "columns", []):
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
            "model_initialized": self.has_model(),
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
            "batch_active": self.batch_active,
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

        return self.provenance.snapshot_version(
            model_ant=self.model.to_antimony() if model_ant is None else model_ant,
            parameters=self.problem.parameter_df,
            label=label,
            kind=kind,
            meta=meta,
            simulation=fitting.build_sim_payload_for_version(sim),
            optimization=fitting.build_optimization_payload_for_version(self.optimization),
        )

    # -- auto fit + simulate on commit -------------------------------------
    def _auto_fit_and_improvement(
        self, n_starts: int | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Fit + simulate for a commit, returning (payload, improvement).

        ``n_starts`` of 0 means simulate-only (no optimisation) for this commit; ``None``
        uses the server default. No-ops (status ``"skipped"``) when no engine is wired or
        ``auto_fit`` is off. A fit/simulation error propagates — the caller reverts.
        """
        from . import fitting

        if n_starts is not None:
            try:
                n_starts = int(n_starts)
            except (TypeError, ValueError):
                raise ValidationError(f"n_starts must be an integer >= 0, got {n_starts!r}.")
            if n_starts < 0:
                raise ValidationError(f"n_starts must be an integer >= 0, got {n_starts}.")

        engine = self.engine
        if engine is None or not self.auto_fit:
            return {"status": "skipped", "neg_log_likelihood": None, "chi2": None,
                    "fitted_parameters": None, "records": []}, {}
        starts = self.auto_fit_starts if n_starts is None else n_starts
        # A fit needs at least one free parameter. Catch the zero-estimated case here — this
        # covers BOTH a single-op commit and commit_batch — with a clear, actionable message
        # rather than letting pyPESTO fail cryptically ("'NoneType' object is not iterable")
        # deep in the optimizer. A simulate-only pass (starts == 0) optimises nothing, so it
        # is exempt.
        if starts != 0 and int(getattr(self.problem, "n_estimated", 0) or 0) == 0:
            raise ModelUpdateError(
                "the model has NO estimated (free) parameters, so there is nothing to "
                "fit. Give it at least one free parameter — a new symbol you use in a "
                "rate law becomes an estimated parameter automatically, so declare one "
                "(or keep using the shipped free parameters) — then re-author with "
                "set_model."
            )
        fitsim = fitting.fit_and_simulate(self, engine, do_fit=(starts != 0), n_starts=starts or 1)
        nllh = fitsim.get("neg_log_likelihood")
        improvement = (
            fitting.compute_improvement(self, nllh, fitsim.get("objective_before"))
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

    # -- batching ----------------------------------------------------------
    def _require_no_open_batch(self, action: str) -> None:
        if self.batch_active:
            raise BatchError(
                f"Cannot {action} while a batch is open; commit_batch or abort_batch first."
            )

    def begin_batch(self, label: str | None = None) -> dict[str, Any]:
        with self.lock:
            if self.batch_active:
                raise BatchError("A batch is already open.")
            self.batch_active = True
            self.batch_label = label
            self.batch_start_version = self.provenance.current_version
            self.staged = []
            self.provenance.append_event({"type": "batch_begin", "label": label})
        return {"batch_active": True, "label": label, "start_version": self.batch_start_version}

    def commit_batch(self, label: str | None = None, n_starts: int | None = None) -> dict[str, Any]:
        from . import fitting

        with self.lock:
            if not self.batch_active:
                raise BatchError("No batch is open.")
            staged = list(self.staged)
            # Fit + simulate the staged result once, before the single snapshot, so the
            # batch version carries fitted params + metrics + the simulation dataframe.
            # (batch_active is still True here; we call the orchestrator directly so no
            # batch guard trips.)
            fitsim, improvement = self._auto_fit_and_improvement(n_starts)
            nllh = fitsim.get("neg_log_likelihood")
            version = self._snapshot_version(
                label=label or self.batch_label or "batch",
                kind="batch",
                meta={"staged_operations": staged, "n_staged": len(staged),
                      "fit_status": fitsim.get("status"), "neg_log_likelihood": nllh,
                      "chi2": fitsim.get("chi2"),
                      "fitted_parameters": fitsim.get("fitted_parameters"), **improvement},
                sim=fitsim,
            )
            fitsim["version"] = version
            self.provenance.append_event(
                {"type": "batch_commit", "version": version, "label": label or self.batch_label,
                 "n_staged": len(staged), "fit_status": fitsim.get("status"),
                 "neg_log_likelihood": nllh, "chi2": fitsim.get("chi2"),
                 "fitted_parameters": fitsim.get("fitted_parameters"), **improvement}
            )
            self._clear_batch()
        return {"committed": True, "version": version, "n_staged": len(staged),
                **fitting.fit_result_block(
                    fitsim, improvement, self.problem.parameter_df, problem=self.problem)}

    def abort_batch(self) -> dict[str, Any]:
        with self.lock:
            if not self.batch_active:
                raise BatchError("No batch is open.")
            start = self.batch_start_version
            n = len(self.staged)
            self.apply_version(start)  # discard staged edits
            self.provenance.append_event({"type": "batch_abort", "restored_version": start,
                                          "discarded": n})
            self._clear_batch()
        return {"aborted": True, "restored_version": start, "discarded": n}

    def _clear_batch(self) -> None:
        self.batch_active = False
        self.batch_label = None
        self.batch_start_version = -1
        self.staged = []

    def rollback_to(self, version: int, justification: str | None = None) -> dict[str, Any]:
        """Revert to a prior version, appending a new (auditable) version + event.

        Restores the target version's stored simulation dataframe (no refit — this is a
        restoration, not a new model); simulate-only fallback if that version lacks one.
        """
        from . import fitting

        with self.lock:
            self._require_no_open_batch("rollback")
            self.apply_version(version)
            sim = self._restore_or_simulate()
            nllh = sim.get("neg_log_likelihood") if sim else None
            new_version = self._snapshot_version(
                label=f"rollback->v{version}",
                kind="rollback",
                meta={"rolled_back_to": version, "justification": justification,
                      "neg_log_likelihood": nllh},
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

    # -- lifecycle ---------------------------------------------------------
    def reset(self) -> dict[str, Any]:
        """Revert model + parameters to the initial version (v0)."""
        from . import fitting

        with self.lock:
            self._require_no_open_batch("reset")
            self.apply_version(0)
            sim = self._restore_or_simulate()
            nllh = sim.get("neg_log_likelihood") if sim else None
            new_version = self._snapshot_version(
                label="reset",
                kind="reset",
                meta={"reset_to": 0, "neg_log_likelihood": nllh},
                sim=sim,
            )
            self.provenance.append_event(
                {"type": "reset", "version": new_version, "neg_log_likelihood": nllh}
            )
        return {"reset": True, "version": new_version,
                **fitting.fit_result_block(
                    sim, parameter_df=self.problem.parameter_df, problem=self.problem)}
