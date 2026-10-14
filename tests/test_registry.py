"""The session registry (:mod:`petab_mcp.registry`) and its store lock (:mod:`petab_mcp.filelock`),
through a fake opener — no server, no AMICI. A second registry over the same root, or a lock the
test takes itself, stands in for another server process: on a local disk flock refuses a second
open of a held file even within one process.
"""

from __future__ import annotations

import errno
import os
import threading
from pathlib import Path

import pytest

from petab_mcp import filelock
from petab_mcp import registry as R
from petab_mcp.errors import (
    NoSessionBoundError,
    PetabMcpError,
    SessionLockedError,
    UnknownEntityError,
)

_SPEC = R.SessionSpec(
    yaml_path=Path("/data/problem.yaml"),
    criterion="aicc",
    auto_fit_starts=7,
    validation_measurements=Path("/data/held_out.tsv"),
    score_shipped_model=True,
)


def _registry(root: Path, fail: Exception | None = None):
    """A registry whose opener records each ``(spec, store_root, created)`` and returns a fresh
    object as the workspace — or raises ``fail``."""
    calls = []

    def open_fn(spec, store, created):
        calls.append((spec, store, created))
        if fail is not None:
            raise fail
        return object()

    return R.SessionRegistry(root, open_fn), calls


def test_create_lays_out_the_session_and_opens_it_once(tmp_path):
    """create() writes <root>/<id>/{session.json, LOCK} and builds the workspace exactly once: a
    later get() of the live id hands back that object, never a second one."""
    reg, calls = _registry(tmp_path)
    sid, ws = reg.create(_SPEC)
    assert R.SessionSpec.from_json((tmp_path / sid / "session.json").read_text()) == _SPEC
    assert (tmp_path / sid / "LOCK").read_text().split()[0] == str(os.getpid())
    assert calls == [(_SPEC, tmp_path / sid, True)]
    assert reg.get(sid) is ws and len(calls) == 1


def test_taken_or_malformed_id_is_refused(tmp_path):
    """A taken id points to resume_session instead of opening a second session on its store, and
    an id that is not a slug never reaches the filesystem (no ``..`` out of the root)."""
    reg, calls = _registry(tmp_path / "sessions")
    reg.create(_SPEC, "run-1")
    with pytest.raises(PetabMcpError, match="resume_session"):
        reg.create(_SPEC, "run-1")
    for bad in ["", ".hidden", "..", "../escape", "a/b", "x" * 65]:
        with pytest.raises(PetabMcpError, match="invalid session id"):
            reg.create(_SPEC, bad)
    assert len(calls) == 1
    assert [p.name for p in tmp_path.iterdir()] == ["sessions"]
    assert [p.name for p in (tmp_path / "sessions").iterdir()] == ["run-1"]


def test_held_lock_refuses_get_until_released(tmp_path):
    """While another holder has the LOCK, get() raises SessionLockedError naming the holder's pid
    and builds nothing; once released, the same get() resumes — created=False, with the spec read
    back from session.json."""
    (tmp_path / "s1").mkdir()
    (tmp_path / "s1" / "session.json").write_text(_SPEC.to_json())
    fd = filelock.try_lock(tmp_path / "s1" / "LOCK")
    reg, calls = _registry(tmp_path)
    with pytest.raises(SessionLockedError) as exc:
        reg.get("s1")
    assert exc.value.holder.split()[0] == str(os.getpid())
    assert exc.value.holder in str(exc.value) and calls == []
    os.close(fd)
    reg.get("s1")
    assert calls == [(_SPEC, tmp_path / "s1", False)]


def test_second_registry_cannot_get_a_live_id(tmp_path):
    """An id live in one server is refused to a second over the same root, whose opener never
    runs: two live workspaces never share a store."""
    first, _ = _registry(tmp_path)
    first.create(_SPEC, "s1")
    second, calls = _registry(tmp_path)
    with pytest.raises(SessionLockedError):
        second.get("s1")
    assert calls == []


def test_failing_opener_releases_the_lock(tmp_path):
    """An opener that raises, on create or on resume, propagates and leaves the LOCK free for a
    retry — not held by this process until it exits."""
    reg, _ = _registry(tmp_path, fail=RuntimeError("boom"))
    with pytest.raises(RuntimeError, match="boom"):
        reg.create(_SPEC, "s1")
    os.close(filelock.try_lock(tmp_path / "s1" / "LOCK"))
    with pytest.raises(RuntimeError, match="boom"):
        reg.get("s1")
    os.close(filelock.try_lock(tmp_path / "s1" / "LOCK"))


def test_unknown_id(tmp_path):
    """get() of an id with no directory is an UnknownEntityError and creates nothing."""
    reg, calls = _registry(tmp_path)
    with pytest.raises(UnknownEntityError):
        reg.get("nope")
    with pytest.raises(PetabMcpError, match="invalid session id"):
        reg.get("../nope")
    assert calls == [] and list(tmp_path.iterdir()) == []


def test_bind_rebind_and_unbound(tmp_path):
    """bound() follows a connection's latest bind(), which returns what it replaced; an unbound
    connection is told the two tools that bind one."""
    reg, _ = _registry(tmp_path)
    s1, w1 = reg.create(_SPEC, "s1")
    s2, w2 = reg.create(_SPEC, "s2")
    with pytest.raises(NoSessionBoundError, match="open_session.*resume_session"):
        reg.bound("c")
    assert reg.bind("c", s1) is None and reg.bound("c") == (s1, w1)
    assert reg.bind("c", s2) == s1 and reg.bound("c") == (s2, w2)
    assert reg.bind("d", s2) is None and reg.bound("d") == (s2, w2)
    with pytest.raises(UnknownEntityError):
        reg.bind("c", "s3")


def test_bound_does_not_wait_for_a_slow_open(tmp_path):
    """Routing takes no lock, so a bound connection's call is routed while another thread is still
    inside create() building a slow session (an AMICI compile, in the server)."""
    entered, release = threading.Event(), threading.Event()

    def open_fn(spec, store, created):
        if store.name == "slow":
            entered.set()
            release.wait()
        return store.name

    reg = R.SessionRegistry(tmp_path, open_fn)
    reg.create(_SPEC, "fast")
    reg.bind("c", "fast")
    creating = threading.Thread(target=reg.create, args=(_SPEC, "slow"))
    creating.start()
    try:
        assert entered.wait(5)
        routed = []
        router = threading.Thread(target=lambda: routed.append(reg.bound("c")))
        router.start()
        router.join(5)
        assert routed == [("fast", "fast")]
    finally:
        release.set()
        creating.join()


def test_filesystem_without_flock_fails_loudly(tmp_path, monkeypatch):
    """A mount that refuses flock is an error naming the file — never an unlocked open."""

    def no_flock(fd, operation):
        raise OSError(errno.ENOLCK, os.strerror(errno.ENOLCK))

    monkeypatch.setattr(filelock.fcntl, "flock", no_flock)
    with pytest.raises(PetabMcpError, match="local disk") as exc:
        filelock.try_lock(tmp_path / "LOCK")
    assert str(tmp_path / "LOCK") in str(exc.value)
