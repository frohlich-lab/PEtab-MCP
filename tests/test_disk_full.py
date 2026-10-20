"""A full disk ends the call as an infrastructure error, never as advice to the agent (#83).

In one sweep the lab filesystem's quota filled for a few hours: 241 ``make_changes`` calls came
back "could not be compiled/fit … ill-posed" (the build's install step failed, so only the import
after it did), 40 as raw ``[Errno 122]`` tool errors that the identical-fit guard then counted, and
15 servers segfaulted in h5py writing the fit result. Agents rewrote models that were fine. Nothing
here compiles: pyPESTO's builder is replaced, and a full disk is the probe's write failing.
"""

from __future__ import annotations

import asyncio
import errno
import os
import subprocess
import tempfile
from pathlib import Path

import pypesto.petab
import pypesto.store
import pytest
from fastmcp.exceptions import ToolError

from petab_mcp import fitting
from petab_mcp.dedup import MAX_IDENTICAL_REAUTHORS
from petab_mcp.engine import AmiciEngine
from petab_mcp.errors import InfrastructureError, ModelUpdateError, disk_full_error
from petab_mcp.references import StubReferenceResolver
from petab_mcp.server import create_server
from petab_mcp.session import PetabSession

TOY = Path(__file__).parent / "data" / "toy_v2"
_PREFIX = "infrastructure_error: disk_full: "

_ANTIMONY = """model conversion_reaction
  compartment compartment_ = 1;
  species A in compartment_;
  species B in compartment_;
  A = A_init;
  B = B_init;
  A_init = 1.0;
  B_init = 0.0;
  k1 = 0.5;
  k2 = 0.1;
  k3 = 0.1;
  r1: A => B; k1 * A;
  r2: B => A; k2 * B;
  r3: A => ; k3 * A;
end
"""


def _quota() -> OSError:
    return OSError(errno.EDQUOT, os.strerror(errno.EDQUOT))


@pytest.fixture
def disk_full(monkeypatch):
    """Every probe write is refused, as on a filesystem whose quota is used up."""

    def _refused(*a, **k):
        raise _quota()

    monkeypatch.setattr(tempfile, "TemporaryFile", _refused)


@pytest.fixture
def install_failed(monkeypatch):
    """pyPESTO's builder fails the way it did once ``cmake --install`` hit the quota: the build
    exits 0, and the import after it finds no module."""

    def _create_problem(self, *a, **k):
        raise ModuleNotFoundError(f"Module file not found: {self.output_folder}/x/x.py")

    monkeypatch.setattr(pypesto.petab.PetabImporter, "create_problem", _create_problem)


def _open(tmp_path):
    session = PetabSession(
        str(TOY / "problem.yaml"),
        provenance_dir=str(tmp_path / "prov"),
        resolver=StubReferenceResolver(),
    )
    session.engine = AmiciEngine(session, build_root=str(tmp_path / "build"))
    return session


# -- what counts as a full disk -----------------------------------------------------------------
def test_a_full_disk_is_found_however_it_is_wrapped():
    """The errno on an OSError anywhere in the chain, or the text a linker leaves in a build's
    output; the cause the client matches is in the message's prefix."""
    try:
        try:
            raise _quota()
        except OSError as inner:
            raise RuntimeError("store failed") from inner
    except RuntimeError as outer:
        wrapped = outer
    link = subprocess.CalledProcessError(
        1, ["cmake"], output=b"[ 90%] Linking\nld: final link failed: Disk quota exceeded\n"
    )
    for exc in (OSError(errno.ENOSPC, os.strerror(errno.ENOSPC)), wrapped, link):
        found = disk_full_error(exc)
        assert found is not None and found.cause == "disk_full"
        assert str(found).startswith(_PREFIX)
    assert str(disk_full_error(link)) == _PREFIX + "ld: final link failed: Disk quota exceeded"


def test_a_missing_file_is_not_a_full_disk(tmp_path):
    """A pruned scratch dir or a missing results file was once read as the host's fault and told
    the agent not to touch a model that WAS the problem; only ENOSPC/EDQUOT count, and a disk that
    takes the probe's write is not full."""
    assert disk_full_error(OSError(errno.ENOENT, "No such file or directory")) is None
    assert disk_full_error(ModuleNotFoundError("Module file not found: x.py"), tmp_path) is None


# -- the fit ------------------------------------------------------------------------------------
def test_a_failed_install_on_a_full_disk_is_not_blamed_on_the_model(
    tmp_path, install_failed, disk_full
):
    """The 241 calls: the import after the build fails with nothing about the disk, so the build
    directory is written to before the model is called ill-posed. Nothing was scored, so the retries
    do not trip the identical-fit guard either."""
    session = _open(tmp_path)
    for _ in range(MAX_IDENTICAL_REAUTHORS + 1):
        with pytest.raises(InfrastructureError, match="quota exceeded"):
            session.fit()


def test_a_failed_import_on_a_disk_with_room_is_still_the_models(tmp_path, install_failed):
    session = _open(tmp_path)
    with pytest.raises(ModelUpdateError, match="ill-posed"):
        session.fit()


def test_the_result_file_is_not_written_to_a_full_disk(tmp_path, disk_full, monkeypatch):
    """h5py segfaults closing a file the disk refused, which no except catches: 15 servers died that
    way. The disk is looked at before h5py is reached."""
    writes = []
    monkeypatch.setattr(pypesto.store, "write_result", lambda *a, **k: writes.append(a))
    with pytest.raises(InfrastructureError):
        fitting._write_pypesto_result(_open(tmp_path), object(), "h")
    assert writes == []


# -- the tool boundary --------------------------------------------------------------------------
def _server(tmp_path):
    return create_server(
        str(TOY / "problem.yaml"),
        provenance_dir=str(tmp_path / "prov"),
        resolver=StubReferenceResolver(),
    )


def _call(mcp, name, args):
    return asyncio.run(mcp.call_tool(name, args)).structured_content


def test_make_changes_on_a_full_disk_ends_with_the_infrastructure_error(tmp_path, monkeypatch):
    """Not a rejection and not "the fit crashed unexpectedly": the call ends as the error a client
    stops the run on, the edit is undone, and the agent's error log does not carry it."""

    def _store_failed(self, *a, **k):
        raise _quota()

    monkeypatch.setattr(PetabSession, "_auto_fit_and_improvement", _store_failed)
    mcp = _server(tmp_path)
    with pytest.raises(ToolError) as ended:
        _call(mcp, "make_changes", {"antimony": _ANTIMONY, "justification": "add a loss of A"})
    assert str(ended.value).startswith(_PREFIX), ended.value
    params = _call(mcp, "get_parameters", {})
    assert "k3" not in {r["parameterId"] for r in params.get("result", params)}
    assert "Disk quota" not in str(_call(mcp, "get_errors", {}))


def test_a_version_on_disk_before_the_disk_filled_is_kept(tmp_path, monkeypatch):
    """HEAD decides whether the change was scored, as for any failure after the snapshot: undoing
    the edit then would leave a working model that is not the version HEAD names."""

    def _snapshot_then_full(self, *a, **k):
        self._snapshot_version(meta={"fit_status": "fit"}, sim=None)
        raise _quota()

    monkeypatch.setattr(PetabSession, "fit", _snapshot_then_full)
    mcp = _server(tmp_path)
    with pytest.raises(ToolError, match=f"^{_PREFIX}"):
        _call(mcp, "make_changes", {"antimony": _ANTIMONY, "justification": "add a loss of A"})
    params = _call(mcp, "get_parameters", {})
    assert "k3" in {r["parameterId"] for r in params.get("result", params)}


def test_a_full_disk_under_any_tool_reads_the_same(tmp_path, monkeypatch):
    def _refused(self, *a, **k):
        raise OSError(errno.ENOSPC, os.strerror(errno.ENOSPC))

    monkeypatch.setattr(PetabSession, "describe", _refused)
    with pytest.raises(ToolError, match=f"^{_PREFIX}"):
        _call(_server(tmp_path), "describe_problem", {})
