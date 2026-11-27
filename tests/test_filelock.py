"""The lock :mod:`petab_mcp.filelock` takes."""

from __future__ import annotations

import errno
import os
import subprocess
import sys
import threading
import time

import pytest

from petab_mcp import filelock
from petab_mcp.errors import PetabMcpError

_PROBE = """
import fcntl, os, sys
fd = os.open(sys.argv[1], os.O_RDWR)
try:
    fcntl.lockf(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
except OSError:
    print("refused")
else:
    print("taken")
"""


def _probe(path):
    """What a server on another node asks of ``path``: can it take the file's POSIX lock now?"""
    cmd = [sys.executable, "-c", _PROBE, str(path)]
    return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()


def _taker(path):
    """A thread that takes the lock on ``path``, then releases it; its event is set once it held."""
    held = threading.Event()

    def _take():
        with filelock.lock(path):
            held.set()

    thread = threading.Thread(target=_take, daemon=True)
    thread.start()
    return thread, held


def test_a_second_thread_waits_for_the_first(tmp_path):
    """POSIX locks are per process, so a second thread's ``lockf`` succeeds at once: two sessions of
    one server building one structure would race. The second waits until the first releases."""
    path = tmp_path / "LOCK"
    with filelock.lock(path):
        thread, held = _taker(path)
        assert not held.wait(0.5)
    assert held.wait(60)
    thread.join(60)


def test_other_processes_see_the_lock_while_any_thread_holds_it(tmp_path):
    """PEtab-MCP#189, Check 4: ``flock`` is node-local on a GPFS share, so a server on another node
    took the build lock while one held it, and deleted that build as unfinished. What another node
    sees is the file's POSIX lock: held while a thread holds the lock, still held while a second
    thread waits for it (closing any descriptor of the file would drop it), free once both are done.
    """
    path = tmp_path / "LOCK"
    with filelock.lock(path):
        assert _probe(path) == "refused"
        thread, held = _taker(path)
        assert not held.wait(0.5)
        assert _probe(path) == "refused"
    assert held.wait(60)
    thread.join(60)
    assert _probe(path) == "taken"


def test_a_forked_child_waits_for_its_parent(tmp_path):
    """A child forked while its parent holds the lock does not hold it (POSIX locks are not
    inherited), and must not inherit the held mutex either, or its own lock would wait forever."""
    path = tmp_path / "LOCK"
    with filelock.lock(path):
        pid = os.fork()
        if pid == 0:
            code = 1
            try:
                with filelock.lock(path):
                    code = 0
            finally:
                os._exit(code)
        time.sleep(0.5)
        assert os.waitpid(pid, os.WNOHANG) == (0, 0), "took the lock its parent holds"
    deadline = time.monotonic() + 60
    while (status := os.waitpid(pid, os.WNOHANG)) == (0, 0) and time.monotonic() < deadline:
        time.sleep(0.01)
    if status == (0, 0):
        os.kill(pid, 9)
        os.waitpid(pid, 0)
    assert status != (0, 0), "still waiting after its parent released the lock"
    assert os.waitstatus_to_exitcode(status[1]) == 0


def test_filesystem_without_posix_locks_fails_loudly(tmp_path, monkeypatch):
    """A mount that refuses POSIX locks is an error naming the file — never an unlocked open — and
    leaves the lock free for the next try."""

    def no_lockf(fd, operation):
        raise OSError(errno.ENOLCK, os.strerror(errno.ENOLCK))

    monkeypatch.setattr(filelock.fcntl, "lockf", no_lockf)
    with pytest.raises(PetabMcpError, match="POSIX") as exc, filelock.lock(tmp_path / "LOCK"):
        pass
    assert str(tmp_path / "LOCK") in str(exc.value)
    monkeypatch.undo()
    thread, held = _taker(tmp_path / "LOCK")
    assert held.wait(60)
    thread.join(60)
