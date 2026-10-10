"""Model-building sessions that outlive the server process, and which connection works on which.

A session is a directory under the registry root, named by its id::

    <root>/<id>/session.json   its SessionSpec: the problem, and how it is scored
    <root>/<id>/LOCK           flock-ed while the session is live in some process
    <root>/<id>/...            its provenance store (versions/, HEAD, logs)

The id is DURABLE — a caller's slug or a generated one, naming a directory that outlives the
process — so resume is just :meth:`SessionRegistry.get` of an id not live here: it takes the
``LOCK`` and rebuilds the workspace from ``session.json``, and the workspace resumes from its
store's HEAD as it is built.

ONE LIVE OBJECT PER ID: every connection resuming an id gets the same workspace, serialised by
its own lock. It holds the ``LOCK`` until the process exits; there is no close, since a compiled
AMICI module cannot be unloaded anyway — recycle the server instead. A second PROCESS asking for
a live id fails with :class:`~petab_mcp.errors.SessionLockedError` rather than sharing the store:
two writers would race on HEAD and the version numbers, and nothing could merge their histories.
That refusal is all the coordination there is; nothing queues or waits (the lock is
:mod:`petab_mcp.filelock`).

The server is not imported. It hands over an opener, ``open_fn(spec, store_root, created)``, that
builds a workspace, and routes each call by its connection's id — so this module needs neither
FastMCP nor AMICI.
"""

from __future__ import annotations

import json
import os
import re
import secrets
import threading
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

from . import filelock
from .errors import NoSessionBoundError, PetabMcpError, SessionLockedError, UnknownEntityError

SPEC_FILE = "session.json"
LOCK_FILE = "LOCK"

#: A session id. Starting alphanumeric and having no ``/``, it names exactly one directory
#: under the root — never ``..``, a dotfile or a path.
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")


@dataclass(frozen=True)
class SessionSpec:
    """What ``session.json`` records; the server's opener takes these as keyword arguments of the
    same names. Paths are absolute, and ``criterion``/``auto_fit_starts`` are resolved before
    create, so a resumed session ranks exactly as it started even if ``PETAB_MCP_CRITERION`` has
    changed since."""

    yaml_path: Path
    criterion: str
    auto_fit_starts: int
    validation_measurements: Path | None
    score_shipped_model: bool

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, default=str)

    @classmethod
    def from_json(cls, text: str) -> SessionSpec:
        d = json.loads(text)
        held_out = d["validation_measurements"]
        return cls(
            yaml_path=Path(d["yaml_path"]),
            criterion=d["criterion"],
            auto_fit_starts=d["auto_fit_starts"],
            validation_measurements=None if held_out is None else Path(held_out),
            score_shipped_model=d["score_shipped_model"],
        )


class SessionRegistry[W]:
    """The sessions under ``root`` live in this process, and the session each connection is
    bound to. ``W`` is whatever ``open_fn(spec, store_root, created)`` builds."""

    def __init__(self, root: Path, open_fn: Callable[[SessionSpec, Path, bool], W]) -> None:
        self.root = Path(root)
        self._open_fn = open_fn
        # Held by create, get's load path and bind only. A worker forked while it is held never
        # touches the registry, so the fork hazard in logging_setup._RateLimitFilter cannot bite.
        self._lock = threading.Lock()
        self._live: dict[str, tuple[W, int]] = {}  # id -> (workspace, its LOCK descriptor)
        self._bindings: dict[str, str] = {}  # connection id -> session id

    def create(self, spec: SessionSpec, session_id: str | None = None) -> tuple[str, W]:
        """Start a session under ``session_id`` (generated when ``None``; refused when taken or
        not a slug) and return ``(id, workspace)``."""
        with self._lock:
            if session_id is None:
                session_id = secrets.token_hex(4)
                while (self.root / session_id).exists():
                    session_id = secrets.token_hex(4)
            try:
                self._store(session_id).mkdir(parents=True)
            except FileExistsError:
                raise PetabMcpError(
                    f"session {session_id!r} already exists: continue it with resume_session, "
                    "or open a new session under another id"
                ) from None
            return session_id, self._open(session_id, spec)

    def get(self, session_id: str) -> W:
        """The live workspace of ``session_id``, resumed from its store if not live yet."""
        live = self._live.get(session_id)
        if live is not None:
            return live[0]
        with self._lock:
            if session_id in self._live:  # another thread resumed it while we waited
                return self._live[session_id][0]
            if not self._store(session_id).is_dir():
                raise UnknownEntityError(f"no session {session_id!r} under {self.root}")
            return self._open(session_id, None)

    def bind(self, connection: str, session_id: str) -> str | None:
        """Route ``connection`` to the live session ``session_id``; return its previous one."""
        with self._lock:
            if session_id not in self._live:
                raise UnknownEntityError(f"session {session_id!r} is not open in this server")
            previous = self._bindings.get(connection)
            self._bindings[connection] = session_id
            return previous

    def bound(self, connection: str) -> tuple[str, W]:
        """``(id, workspace)`` of the session ``connection`` is bound to.

        Takes no lock, so routing a call never waits behind a slow open: each dict read is atomic
        under the GIL, and no entry of either dict is ever removed.
        """
        session_id = self._bindings.get(connection)
        if session_id is None:
            raise NoSessionBoundError()
        return session_id, self._live[session_id][0]

    def _store(self, session_id: str) -> Path:
        if not _ID.fullmatch(session_id):
            raise PetabMcpError(
                f"invalid session id {session_id!r}: 1-64 of [A-Za-z0-9._-], starting with a "
                "letter or digit"
            )
        return self.root / session_id

    def _open(self, session_id: str, spec: SessionSpec | None) -> W:
        """Take the store's ``LOCK`` and build its workspace: a new session (``spec`` given)
        writes ``session.json`` first, a resume reads it back. A failure releases the ``LOCK``,
        so the session is free for a retry rather than held by this process for good."""
        store = self.root / session_id
        try:
            fd = filelock.try_lock(store / LOCK_FILE)
        except filelock.Held as held:
            raise SessionLockedError(session_id, held.holder) from None
        created = spec is not None
        try:
            if created:
                tmp = store / f"{SPEC_FILE}.tmp"
                tmp.write_text(spec.to_json())
                os.replace(tmp, store / SPEC_FILE)  # a crash leaves all of session.json or none
            else:
                spec = SessionSpec.from_json((store / SPEC_FILE).read_text())
            workspace = self._open_fn(spec, store, created)
        except BaseException:
            os.close(fd)
            raise
        self._live[session_id] = (workspace, fd)
        return workspace
