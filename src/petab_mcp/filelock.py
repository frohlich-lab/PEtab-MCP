"""Non-blocking exclusive file locks: a second process on shared on-disk state fails, not races.

:func:`try_lock` takes ``flock(LOCK_EX | LOCK_NB)`` and returns the descriptor, or raises
:class:`Held` with the line the holder wrote into the file (``pid host time``), which each caller
turns into its own error. Nothing waits or retries.

Why ``flock``:

* The OS releases it when the holder dies, SIGKILL included, so there is no stale lock to clean
  up. The holder line only feeds the next contender's message; nothing deletes a lock file (that
  races).
* On a local disk two opens of the file conflict even within one process.
* Not a pid file (stale after a crash, meaningless on another host), and not POSIX ``lockf``:
  that is per process — a second open in the same process succeeds, and closing ANY descriptor of
  the file drops the lock. Linux emulates ``flock`` that way on NFS, so there a process must never
  open a lock file it holds twice.
* A filesystem that refuses ``flock`` (some cluster mounts) raises; nothing goes on unlocked.

A child forked while a lock is held inherits the descriptor and with it the lock. The fit workers
are started by a fork server (:data:`petab_mcp.multistart.START_METHOD`), which forks them from
itself, not from the server, so they hold no session's lock.
"""

from __future__ import annotations

import fcntl
import os
import socket
from datetime import datetime, timezone
from pathlib import Path

from .errors import PetabMcpError


class Held(Exception):
    """The lock is taken: ``holder`` is the line its holder wrote (``""`` if not yet written)."""

    def __init__(self, holder: str) -> None:
        super().__init__(holder)
        self.holder = holder


def try_lock(path: str | Path) -> int:
    """Lock ``path`` (created if missing) and return its descriptor, or raise :class:`Held`.

    The caller owns the descriptor: closing it releases the lock.
    """
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o666)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise Held(os.pread(fd, 1024, 0).decode(errors="replace").strip()) from None
        except OSError as exc:
            raise PetabMcpError(
                f"cannot lock {path}: {exc.strerror}. Its filesystem may not support flock "
                "(some cluster mounts do not); put it on a local disk."
            ) from exc
        os.ftruncate(fd, 0)
        stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        os.write(fd, f"{os.getpid()} {socket.gethostname()} {stamp}\n".encode())
    except BaseException:
        os.close(fd)
        raise
    return fd
