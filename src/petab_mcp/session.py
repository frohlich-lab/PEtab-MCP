"""Session state: a PEtab v1 problem with a fixed design, an Antimony-canonical model,
and persisted provenance/versioning.

The experimental design (measurements, conditions, observables) is fixed. The model
(edited only through the constrained operations in :mod:`operations`) and the parameter
table are mutable, and every committed change is snapshotted as a version and logged.
"""

from __future__ import annotations

import hashlib
import threading
from pathlib import Path
from typing import Any

import pandas as pd
from petab.v1 import Problem
from petab.v1.C import ESTIMATE, PARAMETER_ID
from petab.v1.models.sbml_model import SbmlModel

from . import logging_setup
from .errors import BatchError
from .parameters import parameters_to_records
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

        self.lock = threading.RLock()
        root = Path(provenance_dir) if provenance_dir else self.yaml_path.parent / ".petab_mcp"
        self.provenance = ProvenanceStore(root)
        self.resolver = resolver or EutilsReferenceResolver(mode=reference_mode)
        self.logger = logging_setup.configure(self.provenance.root)

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
        self.auto_fit: bool = True
        self.auto_fit_starts: int = 5

        if self.provenance.current_version < 0:
            self.provenance.snapshot_version(
                model_ant=self.model.to_antimony(),
                parameters=parameters_to_records(self.problem),
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

    def model_sbml(self) -> str:
        return self.model.to_sbml_str()

    def sbml_hash(self) -> str:
        return hashlib.sha256(self.model_sbml().encode("utf-8")).hexdigest()[:16]

    def _set_model_from_antimony(self, antimony: str) -> None:
        rel = getattr(self.problem.model, "rel_path", None) if self.problem.model else None
        model = SbmlModel.from_antimony(antimony, model_id=getattr(self.problem.model, "model_id", None))
        model.rel_path = rel or "model.xml"
        self.problem.model = model

    # -- versioning --------------------------------------------------------
    def _param_df_from_records(self, records: list[dict[str, Any]]) -> pd.DataFrame:
        df = pd.DataFrame(records)
        if PARAMETER_ID in df.columns:
            df = df.set_index(PARAMETER_ID)
        if ESTIMATE in df.columns:
            df[ESTIMATE] = df[ESTIMATE].apply(lambda v: int(bool(v)))
        return df

    def apply_version(self, version: int) -> None:
        """Set the working model + parameter table to a stored version (fixed tables kept).

        Also restores the version's stored simulation dataframe (``None`` for legacy
        versions that predate it). Stays engine-free: this runs during durable resume in
        ``__init__`` before the engine is wired.
        """
        with self.lock:
            snap = self.provenance.get_version(version)
            self._set_model_from_antimony(snap["model_antimony"])
            self.problem.parameter_df = self._param_df_from_records(snap["parameters"])
            self.simulation = snap.get("simulation")

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

        engine = self.engine
        if engine is None or not self.auto_fit:
            return {"status": "skipped", "neg_log_likelihood": None, "chi2": None,
                    "fitted_parameters": None, "records": []}, {}
        starts = self.auto_fit_starts if n_starts is None else int(n_starts)
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
            fitting.fit_and_simulate(self, self.engine, do_fit=False, n_starts=0)
        return self.simulation

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
            version = self.provenance.snapshot_version(
                model_ant=self.model.to_antimony(),
                parameters=parameters_to_records(self.problem),
                label=label or self.batch_label or "batch",
                kind="batch",
                meta={"staged_operations": staged, "n_staged": len(staged),
                      "fit_status": fitsim.get("status"), "neg_log_likelihood": nllh,
                      "chi2": fitsim.get("chi2"),
                      "fitted_parameters": fitsim.get("fitted_parameters"), **improvement},
                simulation=fitting.build_sim_payload_for_version(fitsim),
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
                "fit_status": fitsim.get("status"), "neg_log_likelihood": nllh}

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
            new_version = self.provenance.snapshot_version(
                model_ant=self.model.to_antimony(),
                parameters=parameters_to_records(self.problem),
                label=f"rollback->v{version}",
                kind="rollback",
                meta={"rolled_back_to": version, "justification": justification,
                      "neg_log_likelihood": nllh},
                simulation=fitting.build_sim_payload_for_version(sim),
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
        return {"rolled_back_to": version, "version": new_version}

    # -- lifecycle ---------------------------------------------------------
    def reset(self) -> dict[str, Any]:
        """Revert model + parameters to the initial version (v0)."""
        from . import fitting

        with self.lock:
            self._require_no_open_batch("reset")
            self.apply_version(0)
            sim = self._restore_or_simulate()
            nllh = sim.get("neg_log_likelihood") if sim else None
            new_version = self.provenance.snapshot_version(
                model_ant=self.model.to_antimony(),
                parameters=parameters_to_records(self.problem),
                label="reset",
                kind="reset",
                meta={"reset_to": 0, "neg_log_likelihood": nllh},
                simulation=fitting.build_sim_payload_for_version(sim),
            )
            self.provenance.append_event(
                {"type": "reset", "version": new_version, "neg_log_likelihood": nllh}
            )
        return {"reset": True, "version": new_version}
