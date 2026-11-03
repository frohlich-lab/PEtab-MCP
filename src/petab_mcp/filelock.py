"""Exclusive file locks: a second process on shared on-disk state waits, not races.

:func:`lock` takes ``flock(LOCK_EX)``, waiting while another holds it, and returns the descriptor.
Its one caller is the build lock (:func:`~petab_mcp.engine.build_lock`).

Why ``flock``:

* The OS releases it when the holder dies, SIGKILL included, so there is no stale lock to clean
  up, and nothing deletes a lock file (that races).
* On a local disk two opens of the file conflict even within one process.
* Not a pid file (stale after a crash, meaningless on another host), and not POSIX ``lockf``:
  that is per process — a second open in the same process succeeds, and closing ANY descriptor of
  the file drops the lock. Linux emulates ``flock`` that way on NFS, so there a process must never
  open a lock file it holds twice.
* A filesystem that refuses ``flock`` (some cluster mounts) raises; nothing goes on unlocked.

A child forked while a lock is held inherits the descriptor and with it the lock, which then lasts
until that child exits too. The fit workers are started by a fork server
(:data:`petab_mcp.multistart.START_METHOD`), not forked from the server, so they hold none.
"""

from __future__ import annotations

import fcntl
import os
from pathlib import Path

from .errors import PetabMcpError


def lock(path: str | Path) -> int:
    """Lock ``path`` (created if missing), waiting while another holds it, and return its
    descriptor. The caller owns the descriptor: closing it releases the lock.
    """
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o666)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
        except OSError as exc:
            raise PetabMcpError(
                f"cannot lock {path}: {exc.strerror}. Its filesystem may not support flock "
                "(some cluster mounts do not); put it on a local disk."
            ) from exc
    except BaseException:
        os.close(fd)
        raise
    return fd
