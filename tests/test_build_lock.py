"""A module build fails fast instead of racing another: :func:`petab_mcp.engine.build_lock`.

Every session and process shares one build root keyed by module hash, so these tests hold a module's
lock file the way another build would and drive the real build paths — the fit's, the hierarchical
one's and the held-out scorer's — with pyPESTO's builder replaced, so nothing compiles.
"""

from __future__ import annotations

import errno
import os
import shutil
from pathlib import Path

import pypesto.petab
import pytest

from petab_mcp import filelock
from petab_mcp import fitting
from petab_mcp import validation_data as V
from petab_mcp.engine import AmiciEngine, build_lock
from petab_mcp.errors import ModelBuildBusyError
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


def _open(tmp_path, problem_dir=TOY):
    session = PetabSession(
        str(problem_dir / "problem.yaml"),
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
    folder = engine.build_root / f"pypesto_{h}"
    _install(folder, f"petab_mcp_pp_{h}")
    (folder / ".built").touch()


def test_a_build_another_holds_fails_fast_and_touches_nothing(tmp_path, builds):
    """While another session or process holds a module's lock, the fit's build of that module raises
    ModelBuildBusyError carrying the holder's line — before pyPESTO's builder, which would clear the
    folder the holder is writing, or any rmtree runs. Once the holder lets go, the build goes ahead."""
    session, engine = _open(tmp_path)
    folder = engine.build_root / f"pypesto_{session.module_hash()}"
    held = filelock.try_lock(f"{folder}.lock")
    with pytest.raises(ModelBuildBusyError) as busy:
        fitting.estimation_problem(session, engine)
    os.close(held)
    assert busy.value.holder.split()[0] == str(os.getpid()) and busy.value.holder in str(busy.value)
    assert builds == []
    with pytest.raises(_Built):
        fitting.estimation_problem(session, engine)
    assert builds == [("build", str(folder))]


def test_a_built_module_takes_no_lock(tmp_path, builds):
    """Importing a built module is safe beside anything, so it takes no lock: the build path reaches
    pyPESTO, which imports, even while the lock file is held."""
    session, engine = _open(tmp_path)
    h = session.module_hash()
    _build_lands(engine, h)
    held = filelock.try_lock(engine.build_root / f"pypesto_{h}.lock")
    with pytest.raises(_Built):
        fitting.estimation_problem(session, engine)
    os.close(held)
    assert builds == [("build", str(engine.build_root / f"pypesto_{h}"))]


def test_an_extension_without_the_finished_mark_still_meets_the_lock(tmp_path):
    """The install step writes the extension before its build is done, so finding it is no proof the
    build finished: a caller still meets the holder's lock, and the error names the AMICI module. A
    build that returns marks the module, and the next import takes no lock."""
    folder, name = tmp_path / "pypesto_h", "petab_mcp_pp_h"
    _install(folder, name)
    held = filelock.try_lock(f"{folder}.lock")
    with pytest.raises(ModelBuildBusyError, match=name), build_lock(folder, name):
        pass
    os.close(held)
    with build_lock(folder, name):
        _install(folder, name)
    held = filelock.try_lock(f"{folder}.lock")
    with build_lock(folder, name):
        pass
    os.close(held)


def test_a_failed_build_is_removed_so_the_next_one_compiles(tmp_path, monkeypatch):
    """PEtab-MCP#84: AMICI compiles only into a missing folder and imports one that is there, so the
    folder of a build whose install step hit the disk quota failed every later fit of its structure
    (``has no attribute 'get_model'``). A build that raises removes its folder, so the next one
    starts without it and nothing reads the structure as built."""
    session, engine = _open(tmp_path)
    h = session.module_hash()
    folder = engine.build_root / f"pypesto_{h}"
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
    that follows its lock-free import, and not one that follows a build which landed between the
    look for it and taking the lock."""
    folder, name = tmp_path / "pypesto_h", "petab_mcp_pp_h"
    try_lock = filelock.try_lock

    def _lands_first(path):
        _install(folder, name)
        (folder / ".built").touch()
        return try_lock(path)

    monkeypatch.setattr(filelock, "try_lock", _lands_first)
    with pytest.raises(_Built), build_lock(folder, name):
        assert (folder / ".built").exists()
        raise _Built
    monkeypatch.setattr(filelock, "try_lock", try_lock)
    with pytest.raises(_Built), build_lock(folder, name):
        raise _Built
    assert (folder / ".built").exists() and (folder / name / f"_{name}.so").exists()


def test_a_busy_build_is_not_counted_as_a_fit_of_its_structure(tmp_path, builds):
    """The error says a retry is safe, so retrying must not trip the identical-fit loop guard, which
    refuses a structure fitted MAX_IDENTICAL_REAUTHORS times: a busy build fitted nothing."""
    from petab_mcp.dedup import MAX_IDENTICAL_REAUTHORS

    session, engine = _open(tmp_path)
    session.engine = engine
    held = filelock.try_lock(engine.build_root / f"pypesto_{session.module_hash()}.lock")
    for _ in range(MAX_IDENTICAL_REAUTHORS + 1):
        with pytest.raises(ModelBuildBusyError):
            session.fit()
    os.close(held)


def test_a_busy_hierarchical_build_is_not_retried_as_a_joint_one(tmp_path, builds, monkeypatch):
    """The hierarchical setup falls back to a joint objective when it fails, and contention is not a
    failure of it: falling back would import the module as soon as the other build landed and fit the
    model JOINTLY — a different fit of the same model, chosen by timing."""
    work = tmp_path / "toy"
    shutil.copytree(TOY, work)
    params = work / "parameters.tsv"
    params.write_text(params.read_text().replace("0.1\tfalse", "0.1\ttrue"))  # sigma_B: inner
    session, engine = _open(tmp_path, work)
    h = session.module_hash()

    def _lands_while_refused(path):
        _build_lands(engine, h)
        raise filelock.Held("4242 elsewhere 2026-10-10T00:00:00+00:00")

    monkeypatch.setattr(filelock, "try_lock", _lands_while_refused)
    with pytest.raises(ModelBuildBusyError):
        fitting.estimation_problem(session, engine)
    assert builds == []


def test_a_held_out_build_locks_its_own_sibling_folder(tmp_path, builds):
    """The held-out scorer builds a SIBLING module (``<dir>_<tag>``) for a design that may differ
    from the training one; it takes the lock next to that folder, so the training module's build
    does not stop it and a build of the sibling does."""
    session, engine = _open(tmp_path)
    kwargs = engine.pypesto_importer_kwargs(session.module_hash())
    build = {"build_dir": kwargs["output_folder"], "model_name": kwargs["model_name"]}
    training = filelock.try_lock(f"{kwargs['output_folder']}.lock")
    with pytest.raises(_Built):
        V.refit_problem(session.problem, n_starts=1, **build)
    os.close(training)
    ((_, sibling),) = builds
    assert Path(sibling).name.startswith(f"{Path(kwargs['output_folder']).name}_")
    held = filelock.try_lock(f"{sibling}.lock")
    with pytest.raises(ModelBuildBusyError):
        V.refit_problem(session.problem, n_starts=1, **build)
    os.close(held)
    assert len(builds) == 1
