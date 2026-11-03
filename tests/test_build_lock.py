"""A module build waits for another instead of racing it: :func:`petab_mcp.engine.build_lock`.

Every session and process shares one build root keyed by AMICI and module hash, so these tests run
another server's build in a process of its own and drive the real build paths — the fit's and the
held-out scorer's — with pyPESTO's builder replaced, so nothing compiles.
"""

from __future__ import annotations

import errno
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path

import pypesto.petab
import pytest

from petab_mcp import filelock
from petab_mcp import fitting
from petab_mcp import validation_data as V
from petab_mcp.engine import AmiciEngine, build_lock
from petab_mcp.references import StubReferenceResolver
from petab_mcp.session import PetabSession

TOY = Path(__file__).parent / "data" / "toy_v2"


class _Built(Exception):
    """The stand-in builder was reached."""


@pytest.fixture
def builds(monkeypatch):
    """pyPESTO's builder and ``shutil.rmtree``, each replaced by a recorder; the builder then stops."""
    calls = []

    def _create_problem(self, *a, **k):
        calls.append(("build", self.output_folder))
        raise _Built

    monkeypatch.setattr(pypesto.petab.PetabImporter, "create_problem", _create_problem)
    monkeypatch.setattr(shutil, "rmtree", lambda *a, **k: calls.append(("rmtree", a)))
    return calls


@pytest.fixture
def locks(monkeypatch):
    """``filelock.lock``, recording each path it is asked for before it waits for that lock."""
    paths = []
    lock = filelock.lock

    def _lock(path):
        paths.append(Path(path))
        return lock(path)

    monkeypatch.setattr(filelock, "lock", _lock)
    return paths


def _builder(monkeypatch, folder):
    """pyPESTO's builder, replaced by a recorder of what it finds in ``folder`` (``None`` for no
    folder); it then stops."""
    found = []

    def _create_problem(self, *a, **k):
        found.append(sorted(p.name for p in folder.iterdir()) if folder.exists() else None)
        raise _Built

    monkeypatch.setattr(pypesto.petab.PetabImporter, "create_problem", _create_problem)
    return found


def _open(tmp_path):
    session = PetabSession(
        str(TOY / "problem.yaml"),
        provenance_dir=str(tmp_path / "prov"),
        resolver=StubReferenceResolver(),
    )
    return session, AmiciEngine(session, build_root=str(tmp_path / "build"))


def _install(folder, name):
    """What AMICI's install step writes into ``folder`` for the module ``name``: its extension."""
    (folder / name).mkdir(parents=True)
    (folder / name / f"_{name}.so").write_bytes(b"\x00")


def _build_lands(engine, h):
    """What a finished build of module ``h`` leaves behind: its compiled extension, and the mark its
    builder writes once that imported."""
    folder = engine.module_root / f"pypesto_{h}"
    _install(folder, f"petab_mcp_pp_{h}")
    (folder / ".built").touch()


_HOLDER = """
import sys, time
from pathlib import Path
from petab_mcp.engine import build_lock
folder, name, release = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
with build_lock(folder, name):
    (folder / name).mkdir(parents=True)
    (folder / name / f"_{name}.so").touch()
    print("building", flush=True)
    while not release.exists():
        time.sleep(0.01)
"""


@contextmanager
def _another_server_building(folder, name, release):
    """Another server's build of module ``name`` in ``folder``, in a process of its own: under the
    build lock it writes the extension, says so, and holds until ``release`` exists. Killed on exit.
    """
    cmd = [sys.executable, "-c", _HOLDER, str(folder), name, str(release)]
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True) as holder:
        try:
            assert holder.stdout.readline() == "building\n"
            yield holder
        finally:
            holder.kill()


def _waits_for_the_lock(waiter, locks):
    """``waiter`` (a future) asks for the build lock, and still waits for it half a second later."""
    deadline = time.monotonic() + 60
    while not (locks or waiter.done()) and time.monotonic() < deadline:
        time.sleep(0.01)
    assert locks, "never asked for the build lock"
    with pytest.raises(TimeoutError):
        waiter.result(timeout=0.5)


def test_a_build_another_server_holds_is_waited_for_then_imported(tmp_path, locks, monkeypatch):
    """PEtab-MCP#218, check 1: two servers on one build root build the same structure at once. The
    second waits for the first, the extension already there notwithstanding (the install step
    writes it before the build is done), then finds the module finished and imports it, removing
    nothing."""
    session, engine = _open(tmp_path)
    folder, name = engine._module_target(session.module_hash())
    found = _builder(monkeypatch, folder)
    release = tmp_path / "release"
    with ThreadPoolExecutor(1) as pool, _another_server_building(folder, name, release) as holder:
        waiter = pool.submit(fitting.estimation_problem, session, engine)
        _waits_for_the_lock(waiter, locks)
        release.touch()
        with pytest.raises(_Built):
            waiter.result(timeout=60)
        assert holder.wait(60) == 0
    assert found == [[".built", name]]


def test_a_waiter_whose_holder_is_killed_builds_the_module_itself(tmp_path, locks, monkeypatch):
    """PEtab-MCP#218, check 2: the server holding the build is killed mid-build, which releases its
    lock and leaves its folder unfinished. The waiter removes that folder and builds the module."""
    session, engine = _open(tmp_path)
    folder, name = engine._module_target(session.module_hash())
    found = _builder(monkeypatch, folder)
    release = tmp_path / "never"
    with ThreadPoolExecutor(1) as pool, _another_server_building(folder, name, release) as holder:
        waiter = pool.submit(fitting.estimation_problem, session, engine)
        _waits_for_the_lock(waiter, locks)
        holder.kill()
        with pytest.raises(_Built):
            waiter.result(timeout=60)
    assert found == [None]


def test_a_built_module_takes_no_lock(tmp_path, builds, locks):
    """Importing a built module is safe beside anything, so it takes no lock: the build path reaches
    pyPESTO, which imports, without asking for one."""
    session, engine = _open(tmp_path)
    h = session.module_hash()
    _build_lands(engine, h)
    with pytest.raises(_Built):
        fitting.estimation_problem(session, engine)
    assert locks == []
    assert builds == [("build", str(engine.module_root / f"pypesto_{h}"))]


def test_a_failed_build_is_removed_so_the_next_one_compiles(tmp_path, monkeypatch):
    """PEtab-MCP#84: AMICI compiles only into a missing folder and imports one that is there, so the
    folder of a build whose install step hit the disk quota failed every later fit of its structure
    (``has no attribute 'get_model'``). A build that raises removes its folder, so the next one
    starts without it and nothing reads the structure as built."""
    session, engine = _open(tmp_path)
    h = session.module_hash()
    folder = engine.module_root / f"pypesto_{h}"
    found = []

    def _quota_hit(self, *a, **k):
        found.append(folder.exists())
        _install(folder, f"petab_mcp_pp_{h}")
        raise OSError(errno.EDQUOT, "Disk quota exceeded")

    monkeypatch.setattr(pypesto.petab.PetabImporter, "create_problem", _quota_hit)
    for _ in range(2):
        with pytest.raises(OSError, match="Disk quota exceeded"):
            fitting.estimation_problem(session, engine)
        assert not folder.exists()
    assert found == [False, False]
    assert not engine.module_built(h)


def test_a_build_that_stopped_is_removed_before_the_next(tmp_path):
    """A build killed outright raises nothing in its process, so its folder stays, extension and
    all, without the mark. The next holder removes it before building, and marks what it built."""
    folder, name = tmp_path / "pypesto_h", "petab_mcp_pp_h"
    _install(folder, name)
    (folder / "wrapfunctions.cpp").write_text("// codegen ran")
    with build_lock(folder, name):
        assert not folder.exists()
        _install(folder, name)
    assert (folder / ".built").exists()


def test_a_finished_build_is_never_removed(tmp_path, monkeypatch):
    """Other processes import a finished module without the lock, so no failure removes it: not one
    that follows its lock-free import, and not one that follows a build it waited for."""
    folder, name = tmp_path / "pypesto_h", "petab_mcp_pp_h"
    lock = filelock.lock

    def _lands_first(path):
        _install(folder, name)
        (folder / ".built").touch()
        return lock(path)

    monkeypatch.setattr(filelock, "lock", _lands_first)
    with pytest.raises(_Built), build_lock(folder, name):
        assert (folder / ".built").exists()
        raise _Built
    monkeypatch.setattr(filelock, "lock", lock)
    with pytest.raises(_Built), build_lock(folder, name):
        raise _Built
    assert (folder / ".built").exists() and (folder / name / f"_{name}.so").exists()


def test_an_unreadable_folder_is_not_taken_for_an_unfinished_one(tmp_path, monkeypatch):
    """Under the lock the look decides a removal, so a read error there (too many open files)
    fails the build instead of reading a finished module, which others import, as unfinished."""
    folder, name = tmp_path / "pypesto_h", "petab_mcp_pp_h"
    _install(folder, name)
    (folder / ".built").touch()

    def _emfile(self, *a, **k):
        raise OSError(errno.EMFILE, "Too many open files")

    monkeypatch.setattr(Path, "glob", _emfile)
    with pytest.raises(OSError, match="Too many open files"), build_lock(folder, name):
        pass
    monkeypatch.undo()
    assert (folder / ".built").exists() and (folder / name / f"_{name}.so").exists()


def test_a_held_out_build_locks_its_own_sibling_folder(tmp_path, builds, locks):
    """The held-out scorer builds a SIBLING module (``<dir>_<tag>``) for a design that may differ
    from the training one, and takes the lock next to that folder, not the training module's."""
    session, engine = _open(tmp_path)
    kwargs = engine.pypesto_importer_kwargs(session.module_hash())
    build = {"build_dir": kwargs["output_folder"], "model_name": kwargs["model_name"]}
    with pytest.raises(_Built):
        V.refit_problem(session.problem, n_starts=1, **build)
    ((_, sibling),) = builds
    assert Path(sibling).name.startswith(f"{Path(kwargs['output_folder']).name}_")
    assert locks == [Path(f"{sibling}.lock")]


def _engine_as(session, tmp_path, monkeypatch, identity):
    """An engine over ``_open``'s build root, under a faked AMICI identity."""
    monkeypatch.setattr("petab_mcp.engine.amici_identity", lambda: identity)
    return AmiciEngine(session, build_root=str(tmp_path / "build"))


def test_a_module_another_amici_built_is_never_imported(tmp_path, builds, monkeypatch):
    """PEtab-MCP#103: a build root outlives an AMICI pin bump, and pins move between commits that
    share a version string. The module the old AMICI finished, on disk or in its process, reads as
    unbuilt to the new one, whose fit builds its own beside it and leaves the old one alone."""
    session, _ = _open(tmp_path)
    h = session.module_hash()
    old = _engine_as(session, tmp_path, monkeypatch, "1.1.0-dc6ce2ae320d")
    new = _engine_as(session, tmp_path, monkeypatch, "1.1.0-a4dbb08da4d3")
    assert old._module_target(h)[0] != new._module_target(h)[0]
    _build_lands(old, h)
    old._pypesto_hashes.add(h)
    assert old.module_built(h) and not new.module_built(h)
    with pytest.raises(_Built):
        fitting.estimation_problem(session, new)
    assert builds == [("build", str(new._module_target(h)[0]))]


def test_the_folder_a_build_marks_is_the_one_module_built_reads(tmp_path, monkeypatch):
    """The fit builds where :meth:`pypesto_importer_kwargs` points and :func:`build_lock` marks it,
    and :meth:`module_built` (a read's guard against paying a build) must look there, under the same
    AMICI, and nowhere another AMICI looks."""
    session, _ = _open(tmp_path)
    h = session.module_hash()
    engine = _engine_as(session, tmp_path, monkeypatch, "1.1.0-a4dbb08da4d3")
    kwargs = engine.pypesto_importer_kwargs(h)
    with build_lock(kwargs["output_folder"], kwargs["model_name"]):
        _install(Path(kwargs["output_folder"]), kwargs["model_name"])
    assert engine.module_built(h)
    assert not _engine_as(session, tmp_path, monkeypatch, "1.1.0-dc6ce2ae320d").module_built(h)
