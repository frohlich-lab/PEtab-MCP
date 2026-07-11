"""petab_mcp — a local MCP server for interacting with a PEtab (v2) problem.

The server wraps a single PEtab v2 problem and exposes it over the Model Context
Protocol. The experimental design stays fixed — measurements, experiments,
conditions and observables are read-only — while an agent may edit the **model**
(SBML, via Antimony or SBML text) and the **parameter table**, then simulate and
score the result against the fixed data with AMICI.
"""

from __future__ import annotations

__version__ = "0.1.0"

from .errors import (
    FixedTableError,
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
    "FixedTableError",
    "ParameterError",
    "UnknownEntityError",
]
