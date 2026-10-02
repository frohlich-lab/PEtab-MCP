"""petab_mcp — a local MCP server for interacting with a PEtab (v2) problem.

The server wraps a single PEtab v2 problem and exposes it over the Model Context
Protocol. The experimental design stays fixed — measurements, conditions, experiments and
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
from . import antimony_guard

# Process-wide, package-level: libantimony keeps global state and is not thread-safe, while
# FastMCP dispatches each tool call on its own worker thread (20/20 draft-arm runs of the
# 2026-08-15 sneyd A/B died on signal 11 without this). Installed here rather than in session.py
# so the altitude matches the scope, and because importing ANY submodule runs this file.
antimony_guard.install()

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
