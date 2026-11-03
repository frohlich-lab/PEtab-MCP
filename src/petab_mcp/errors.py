"""Typed errors for petab_mcp.

These are caught at the MCP tool boundary and turned into clear, actionable
error messages for the calling agent. Keeping them typed (rather than raising
bare ``ValueError``) lets the server distinguish "the user asked for something
impossible" (a 4xx-style problem) from an internal bug.

:class:`InfrastructureError` is the third kind, the host failing the call, and the one that is not
turned into a message for the agent: it ends the tool call as an MCP error a client can recognise.
"""

from __future__ import annotations

import errno
import os
import tempfile
from collections.abc import Iterator
from pathlib import Path


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


class ProtocolError(PetabMcpError):
    """The problem's ``inputs.tsv`` would not be read as written, or an input names what no
    experiment sets (:mod:`petab_mcp.protocols`). Raised as the session opens."""


class ParameterError(PetabMcpError):
    """An invalid parameter-table update (unknown id, bad bounds/scale, ...)."""


class UnknownEntityError(PetabMcpError):
    """A referenced id (parameter, observable, experiment, ...) does not exist."""


class EntityMapError(PetabMcpError):
    """The entity map (``--entity-map``) would not be read as written (:mod:`petab_mcp.seeding`).
    Raised as the server opens, before any tool runs."""


class MechanismError(PetabMcpError):
    """A row's mechanism cannot be built.

    ``cause`` is the machine-readable rejection class a refused call carries as ``error_cause``.
    ``library_gap`` is a gap in the library — the policy exists, but no entry expresses it for the
    statement class — and the signal that the library needs one, so its raisers name it. Every
    other refusal, a policy named nowhere among them, is a request to fix, ``malformed_mechanism``,
    and that is the default, so a new refusal never inflates the gaps.
    """

    def __init__(self, *args, cause: str = "malformed_mechanism"):
        super().__init__(*args)
        self.cause = cause


class FitBackendError(PetabMcpError):
    """The cluster a fit's starts run on failed them (scheduler gone, a worker killed, a start
    cancelled or unable to run on a worker)."""


class FitCancelledError(PetabMcpError):
    """The client cancelled the call whose fit this was (MCP ``notifications/cancelled``), and its
    starts were stopped (see :mod:`petab_mcp.multistart`)."""


#: The ``error_cause`` of a ``make_changes`` refused at once because another change on the session
#: is still being fitted, and the word its ``error`` starts with after ``make_changes: ``, which a
#: client matches (gemot's ``circuit_breaker``). Nothing was applied, and that fit's result becomes
#: the next version.
FIT_IN_PROGRESS = "fit_in_progress"


#: What every :class:`InfrastructureError` reads as, before its cause: the text of the MCP error a
#: tool call ends with is ``infrastructure_error: <cause>: <detail>``.
INFRASTRUCTURE_ERROR = "infrastructure_error"

#: The cause a full disk or an exceeded quota (ENOSPC, EDQUOT) is reported under.
DISK_FULL = "disk_full"

#: The cause a fit whose starts stopped finishing is reported under (see
#: :mod:`petab_mcp.multistart`): its workers were stopped and nothing was scored.
FIT_WEDGED = "fit_wedged"

_DISK_FULL_ERRNOS = frozenset({errno.ENOSPC, errno.EDQUOT})

#: How a full disk reads where no ``OSError`` carries its errno: a linker's or ``cmake --install``'s
#: output, h5py's message. glibc's wording, which a cluster's build prints, and this platform's own
#: (macOS spells it "Disc quota exceeded").
_DISK_FULL_TEXT = frozenset(
    {"No space left on device", "Disk quota exceeded"} | {os.strerror(e) for e in _DISK_FULL_ERRNOS}
)

#: What :func:`disk_full_error` writes to tell whether a folder's disk takes a write. A MiB rather
#: than the one block a full disk refuses, so a disk with only a few left fails here, not in h5py.
_PROBE_BYTES = 1 << 20


class InfrastructureError(PetabMcpError):
    """The host failed the call, so nothing the agent can change fixes it: ``cause`` names how
    (:data:`DISK_FULL`, :data:`FIT_WEDGED`).

    Not a rejection. The tool call ends as an MCP error reading ``infrastructure_error: <cause>:
    <detail>``, which a client takes as the end of the run; the edit it carried is not counted
    toward the identical-fit guard when it ends the fit. Raised for a full disk anywhere in edit,
    build, fit or store (:func:`disk_full_error`), and for a fit whose starts stopped finishing.
    """

    def __init__(self, cause: str, detail: str) -> None:
        self.cause = cause
        super().__init__(f"{INFRASTRUCTURE_ERROR}: {cause}: {detail}")


def _chain(exc: BaseException | None) -> Iterator[BaseException]:
    seen: set[int] = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        yield exc
        exc = exc.__cause__ or exc.__context__


def _disk_full_line(exc: BaseException) -> str | None:
    """The line naming a full disk in a failed command's output on ``exc``, or in ``exc``'s own text
    if it is an ``OSError`` (h5py's carries no errno). Not in any other message: a refusal may quote
    the agent's own text, as a SQL error quotes the table it names."""
    output = getattr(exc, "output", None) or getattr(exc, "stderr", None) or ""
    if isinstance(output, bytes):
        output = output.decode("utf-8", "replace")
    text = f"{exc}\n{output}" if isinstance(exc, OSError) else str(output)
    for line in text.splitlines():
        if any(sig in line for sig in _DISK_FULL_TEXT):
            return line.strip()
    return None


def _probe(folder: Path) -> InfrastructureError | None:
    try:
        with tempfile.TemporaryFile(dir=folder, prefix=".disk_probe_") as probe:
            probe.write(b"\xff" * _PROBE_BYTES)
            probe.flush()
            os.fsync(probe.fileno())
    except OSError as exc:
        if exc.errno in _DISK_FULL_ERRNOS:
            return InfrastructureError(DISK_FULL, f"{exc} (writing in {folder})")
    return None


def disk_full_error(
    exc: BaseException | None, *folders: str | Path | None
) -> InfrastructureError | None:
    """``exc`` as a :data:`DISK_FULL` :class:`InfrastructureError` if a full disk caused it, else
    ``None``.

    Read off ``exc``'s cause/context chain first: an :class:`InfrastructureError` already raised, an
    ``OSError`` with ENOSPC or EDQUOT, then the text a full disk leaves in an ``OSError``'s message
    or a failed build's output. Then a file is written and synced in each of ``folders``, for the
    failures that name no disk at all: a build whose ``cmake --install`` failed exits 0, and only
    the import after it fails (``Module file not found``, no ``get_model``). ``exc`` may be ``None``
    to only probe.
    """
    chain = list(_chain(exc))
    for e in chain:
        if isinstance(e, InfrastructureError):
            return e
    for e in reversed(chain):  # the root cause first: the wrappers' text repeats it
        if isinstance(e, OSError) and e.errno in _DISK_FULL_ERRNOS:
            return InfrastructureError(DISK_FULL, str(e))
    for e in reversed(chain):
        if (line := _disk_full_line(e)) is not None:
            return InfrastructureError(DISK_FULL, line)
    for folder in folders:
        if folder is not None and (found := _probe(Path(folder))) is not None:
            return found
    return None
