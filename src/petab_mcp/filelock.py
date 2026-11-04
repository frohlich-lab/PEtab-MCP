"""Exclusive file locks: a second thread, process or node on shared on-disk state waits, not races.

:func:`lock` holds a file for a block, waiting while another holds it. Its one caller is the build
lock (:func:`~petab_mcp.engine.build_lock`), whose build root a fit pool's servers share across
nodes. It takes two locks, in this order:

* A mutex per path, in this process, for its threads and sessions.
* POSIX ``lockf`` on the file, for every other process, on this node or another. The OS releases it
  when the holder dies, SIGKILL included, so there is no stale lock to clean up, and nothing
  deletes a lock file (that races). Not a pid file either (stale after a crash, meaningless on
  another host). A filesystem that refuses it raises; nothing goes on unlocked.

Not ``flock``: on NEMO's GPFS share it is node-local, and a waiter on a second node took it while
the first held it, then deleted the build in progress as unfinished; ``lockf`` there made it wait
until the holder released it or was killed (PEtab-MCP#189, Check 4). And not ``flock`` beside
``lockf``: the two conflict on macOS even on one descriptor, and Linux emulates ``flock`` on NFS as
a POSIX lock owned by the open file (``flock(2)``), so either way the process waits for itself.

``lockf`` alone is per process: a second thread's ``lockf`` succeeds at once, and closing ANY
descriptor of the file drops the lock. The mutex is what makes that safe: a thread opens the file
only once it holds the mutex, and closes it before releasing it, so while one thread holds the lock
no other thread has the file open, and nothing it does on any path, an error's included, can close
a descriptor of it. A forked child holds none of the parent's locks: POSIX locks are not inherited,
and the child's mutexes start afresh.
"""

from __future__ import annotations

import fcntl
import os
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from .errors import PetabMcpError

#: One mutex per lock file (by real path) this process has locked.
_mutexes: dict[str, threading.Lock] = {}
os.register_at_fork(after_in_child=_mutexes.clear)


@contextmanager
def lock(path: str | Path) -> Iterator[None]:
    """Hold ``path`` (created if missing) for the block, waiting while another holds it."""
    with _mutexes.setdefault(os.path.realpath(path), threading.Lock()):
        fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o666)
        try:
            try:
                fcntl.lockf(fd, fcntl.LOCK_EX)
            except OSError as exc:
                raise PetabMcpError(
                    f"cannot lock {path}: {exc.strerror}. Its filesystem may not support POSIX "
                    "(fcntl) locks, which the build root needs (some cluster mounts do not)."
                ) from exc
            yield
        finally:
            os.close(fd)
