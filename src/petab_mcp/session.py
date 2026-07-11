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
        """Set the working model + parameter table to a stored version (fixed tables kept)."""
        with self.lock:
            snap = self.provenance.get_version(version)
            self._set_model_from_antimony(snap["model_antimony"])
            self.problem.parameter_df = self._param_df_from_records(snap["parameters"])

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

    def commit_batch(self, label: str | None = None) -> dict[str, Any]:
        with self.lock:
            if not self.batch_active:
                raise BatchError("No batch is open.")
            staged = list(self.staged)
            version = self.provenance.snapshot_version(
                model_ant=self.model.to_antimony(),
                parameters=parameters_to_records(self.problem),
                label=label or self.batch_label or "batch",
                kind="batch",
                meta={"staged_operations": staged, "n_staged": len(staged)},
            )
            self.provenance.append_event(
                {"type": "batch_commit", "version": version, "label": label or self.batch_label,
                 "n_staged": len(staged)}
            )
            self._clear_batch()
        return {"committed": True, "version": version, "n_staged": len(staged)}

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
        """Revert to a prior version, appending a new (auditable) version + event."""
        with self.lock:
            self._require_no_open_batch("rollback")
            self.apply_version(version)
            new_version = self.provenance.snapshot_version(
                model_ant=self.model.to_antimony(),
                parameters=parameters_to_records(self.problem),
                label=f"rollback->v{version}",
                kind="rollback",
                meta={"rolled_back_to": version, "justification": justification},
            )
            self.provenance.append_event(
                {
                    "type": "rollback",
                    "rolled_back_to": version,
                    "version": new_version,
                    "justification": justification,
                }
            )
        return {"rolled_back_to": version, "version": new_version}

    # -- lifecycle ---------------------------------------------------------
    def reset(self) -> dict[str, Any]:
        """Revert model + parameters to the initial version (v0)."""
        with self.lock:
            self._require_no_open_batch("reset")
            self.apply_version(0)
            new_version = self.provenance.snapshot_version(
                model_ant=self.model.to_antimony(),
                parameters=parameters_to_records(self.problem),
                label="reset",
                kind="reset",
                meta={"reset_to": 0},
            )
            self.provenance.append_event({"type": "reset", "version": new_version})
        return {"reset": True, "version": new_version}
