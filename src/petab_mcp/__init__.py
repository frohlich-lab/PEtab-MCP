"""petab_mcp — a local MCP server for interacting with a PEtab (v1) problem.

The server wraps a single PEtab v1 problem and exposes it over the Model Context
Protocol. The experimental design stays fixed — measurements, conditions and
observables are read-only — while an agent edits the **model** (expressed in Antimony
via constrained operations). Edits are fit-free; the dedicated ``fit`` tool fits the free
parameters and simulates against the fixed data with AMICI/pyPESTO, storing a simulation
dataframe on the new version for statistical analysis.
"""

from __future__ import annotations

__version__ = "0.1.0"

from .errors import (
    MissingDependencyError,
    ModelUpdateError,
    ParameterError,
    PetabMcpError,
    UnknownEntityError,
    ValidationError,
)
from .session import PetabSession

__all__ = [
    "__version__",
    "PetabSession",
    "PetabMcpError",
    "MissingDependencyError",
    "ModelUpdateError",
    "ValidationError",
    "ParameterError",
    "UnknownEntityError",
]
