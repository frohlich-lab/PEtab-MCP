"""Session state: a single PEtab v2 problem with a fixed design and a mutable model.

``PetabSession`` owns the working ``petab.v2.Problem``. The experimental design —
measurements, experiments, conditions, observables — is never mutated by any tool;
only the model and the parameter table change. ``reset()`` reloads the problem from
its YAML, discarding model/parameter edits.
"""

from __future__ import annotations

import hashlib
import threading
from pathlib import Path

from petab.v2 import Problem


class PetabSession:
    """Holds the working PEtab problem and derived state (locks, model hash)."""

    def __init__(self, yaml_path: str | Path) -> None:
        self.yaml_path = Path(yaml_path).resolve()
        if not self.yaml_path.is_file():
            raise FileNotFoundError(f"PEtab YAML not found: {self.yaml_path}")
        self.problem: Problem = Problem.from_yaml(str(self.yaml_path))
        # A re-entrant lock guards mutation + evaluation; local MCP usage is
        # effectively single-user, but tool handlers may still interleave.
        self.lock = threading.RLock()

    # -- model access ------------------------------------------------------
    @property
    def model(self):
        """The working SBML model wrapper (``petab.v2`` ``SbmlModel``)."""
        if self.problem.model is None:
            raise ValueError("The PEtab problem has no model.")
        return self.problem.model

    def model_sbml(self) -> str:
        """Serialize the current model to an SBML string."""
        return self.model.to_sbml_str()

    def sbml_hash(self) -> str:
        """Stable short hash of the current SBML, used to key the compile cache."""
        return hashlib.sha256(self.model_sbml().encode("utf-8")).hexdigest()[:16]

    # -- lifecycle ---------------------------------------------------------
    def reset(self) -> None:
        """Reload the problem from disk, discarding model/parameter edits."""
        with self.lock:
            self.problem = Problem.from_yaml(str(self.yaml_path))
