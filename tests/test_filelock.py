"""The lock :mod:`petab_mcp.filelock` takes."""

from __future__ import annotations

import errno
import os

import pytest

from petab_mcp import filelock
from petab_mcp.errors import PetabMcpError


def test_filesystem_without_flock_fails_loudly(tmp_path, monkeypatch):
    """A mount that refuses flock is an error naming the file — never an unlocked open."""

    def no_flock(fd, operation):
        raise OSError(errno.ENOLCK, os.strerror(errno.ENOLCK))

    monkeypatch.setattr(filelock.fcntl, "flock", no_flock)
    with pytest.raises(PetabMcpError, match="local disk") as exc:
        filelock.try_lock(tmp_path / "LOCK")
    assert str(tmp_path / "LOCK") in str(exc.value)
