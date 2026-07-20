"""Typed errors for petab_mcp.

These are caught at the MCP tool boundary and turned into clear, actionable
error messages for the calling agent. Keeping them typed (rather than raising
bare ``ValueError``) lets the server distinguish "the user asked for something
impossible" (a 4xx-style problem) from an internal bug.
"""

from __future__ import annotations


class PetabMcpError(Exception):
    """Base class for all petab_mcp errors surfaced to the client."""


class MissingDependencyError(PetabMcpError):
    """A tool needs an optional dependency that is not installed.

    Parameters
    ----------
    feature:
        Human-readable name of what the user tried to do (e.g. ``"simulation"``).
    package:
        The importable module that was missing (e.g. ``"amici"``).
    extra:
        The pip extra that provides it (e.g. ``"amici"``), used to build the hint.
    """

    def __init__(self, feature: str, package: str, extra: str) -> None:
        self.feature = feature
        self.package = package
        self.extra = extra
        super().__init__(
            f"{feature} requires the optional dependency '{package}', which is not "
            f"installed. Install it with: pip install 'petab-mcp[{extra}]'"
        )


class ModelUpdateError(PetabMcpError):
    """An attempt to update the model failed (parse or conversion error)."""


class ValidationError(PetabMcpError):
    """A candidate problem failed PEtab linting or the fixed-table guardrail."""

    def __init__(self, message: str, issues: list[str] | None = None) -> None:
        self.issues = issues or []
        super().__init__(message)


class ParameterError(PetabMcpError):
    """An invalid parameter-table update (unknown id, bad bounds/scale, ...)."""


class UnknownEntityError(PetabMcpError):
    """A referenced id (parameter, observable, experiment, ...) does not exist."""


class BatchError(PetabMcpError):
    """An invalid batch/staging action (e.g. committing with no batch open)."""
