"""Shared test fixtures and AMICI/SWIG discovery for the test session."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

DATA_DIR = Path(__file__).parent / "data" / "toy_v2"
YAML_PATH = DATA_DIR / "problem.yaml"


def pytest_configure(config):  # noqa: D401 - pytest hook
    # The server package already owns this (AMICI compiles a C++ model per structure and needs
    # the SWIG the wheel was built against). It used to be copy-pasted here as a third identical
    # implementation; call the shipped one instead.
    from petab_mcp.engine import _ensure_swig_on_path

    _ensure_swig_on_path()


def confirmed(call):
    """``call()``, a ``make_changes``, sent again where it was refused as an implicit base off the
    best (PEtab-MCP#330), as an agent that means HEAD does. Whether a fit leaves HEAD the best
    depends on its random starts, so a test that commits in sequence cannot know."""
    out = call()
    return call() if out.get("error_cause") == "implicit_base" else out


@pytest.fixture
def yaml_path() -> str:
    return str(YAML_PATH)


def rows_with_bindings(pairs, library) -> list[dict]:
    """``pairs`` of a `statements` row (its line in the notation, or its JSON) and its binding,
    given apart as `make_changes` takes them and joined as it joins them
    (:func:`petab_mcp.operations.bound_rows`): the rows a build reads."""
    from petab_mcp.operations import bound_rows

    def prefix(row) -> str:
        return row.split(":", 1)[0].strip() if isinstance(row, str) else row["prefix"]

    return bound_rows([row for row, _ in pairs], {prefix(r): b for r, b in pairs}, library)


# The shipped toy model (tests/data/toy_v2/model.ant). Structural edits in tests are made by
# re-authoring this base (via ``set_model``) with extra reaction lines spliced in.
_TOY_BASE_MODEL = """model conversion_reaction
  compartment compartment_ = 1;
  species A in compartment_;
  species B in compartment_;
  A = A_init;
  B = B_init;
  A_init = 1.0;
  B_init = 0.0;
  k1 = 0.5;
  k2 = 0.1;
  r1: A => B; k1 * A;
  r2: B => A; k2 * B;
{extra}end
"""


@pytest.fixture
def reauthor_toy():
    """Return a helper that re-authors the toy model with ``extra_reactions`` (Antimony lines)
    spliced in via ``set_model`` and then ``fit``s it, for tests that just need a committed,
    versioned structural change.

    Since issue #81 an edit is fit-free (it snapshots no version); ``fit`` is what commits the
    version. This helper does both so a caller gets a committed version in one call. Engine-free
    (the plain ``session`` fixture wires no engine), ``fit`` still snapshots a ``kind="fit"``
    version recording the edited model — the fit itself is skipped. Returns the ``fit`` result."""
    from petab_mcp import operations as O

    def _reauthor(session, extra_reactions=(), *, justification="test", **kwargs):
        body = "".join(f"  {line}\n" for line in extra_reactions)
        antimony = _TOY_BASE_MODEL.format(extra=body)
        O.set_model(session, antimony, justification=justification, **kwargs)
        return session.fit()

    return _reauthor


@pytest.fixture
def broken_model():
    """An ``SbmlModel`` missing entities the fixed observable/condition tables need, for tests
    that assert validation rejects it. Built here rather than in the server package: the
    production build path is ``operations.set_model`` -> ``SbmlModel.from_antimony``, so a
    text->model helper is a test fixture, not shipped API."""
    from petab.v2.models.sbml_model import SbmlModel

    return SbmlModel.from_antimony(
        "model conversion_reaction\n species A = 1;\n k1 = 0.5;\n r1: A => ; k1 * A;\nend",
        model_id="conversion_reaction",
    )


@pytest.fixture
def session(yaml_path, tmp_path):
    from petab_mcp.references import StubReferenceResolver
    from petab_mcp.session import PetabSession

    return PetabSession(
        yaml_path,
        provenance_dir=str(tmp_path / "prov"),
        resolver=StubReferenceResolver(),
    )


@pytest.fixture(scope="session", autouse=True)
def _private_tempdir_per_worker(tmp_path_factory):
    """Give each xdist worker its own ``tempfile.gettempdir()``.

    `amici_build_root` below is already worker-private, but only tests that ASK for it get that
    isolation. Anything constructing a server without an explicit `build_root` --
    `create_server(...)` in test_make_changes/test_server, the commonest shape -- falls back to
    `AmiciEngine`'s default, `tempfile.gettempdir()/petab_mcp_amici`. That is ONE directory shared
    by every worker AND by every concurrent test session on the machine, keyed by the hash of the
    same toy model nearly all of them use. Workers then compile into and clear the same
    `pypesto_<hash>` path, and one of them loses:

        fit rejected: [Errno 2] No such file or directory:
        '/tmp/.../petab_mcp_amici/pypesto_10ba47402a196890'

    reported as an ill-posed model, which is a wrong and expensive diagnosis. Raising `-n` from 4
    to 8 made it show up; it was always a race, just a narrower one. `tempfile.gettempdir()`
    returns `tempfile.tempdir` when set, so pointing that at the worker's own tree fixes every
    caller of the default at once rather than one test helper at a time.
    """
    import tempfile

    previous = tempfile.tempdir
    tempfile.tempdir = str(tmp_path_factory.mktemp("tmpdir"))
    try:
        yield
    finally:
        tempfile.tempdir = previous


@pytest.fixture(scope="session")
def amici_build_root(tmp_path_factory):
    """One AMICI build root shared by the whole test session.

    The engine caches compiled models by SBML hash, and nearly every test here uses the same
    toy model — but a function-scoped ``tmp_path`` gave each test a private build root, so the
    cache could never hit and every test paid a fresh ~60-90s C++ compile (fit tests twice
    over, since pyPESTO builds its own module). Sharing the root lets a structure compile once
    per session. Nothing leaks between tests: the root holds only compiled artifacts, keyed by
    the hash of the model that produced them, and a genuinely different model still gets its
    own build directory.
    """
    return tmp_path_factory.mktemp("amici_build_root")


@pytest.fixture
def engine(session, amici_build_root):
    from petab_mcp.engine import AmiciEngine

    return AmiciEngine(session, build_root=str(amici_build_root))


@pytest.fixture
def fit_session(session, amici_build_root):
    """A session with the AMICI engine wired and auto-fit on (small n_starts).

    Use for tests that exercise the ``fit`` path; the engine is reachable as
    ``fit_session.engine``.
    """
    from petab_mcp.engine import AmiciEngine

    session.engine = AmiciEngine(session, build_root=str(amici_build_root))
    session.auto_fit = True
    session.auto_fit_starts = 2
    return session


def _write_empty_sbml(path: Path, model_id: str = "conversion_reaction") -> None:
    import libsbml

    doc = libsbml.SBMLDocument(3, 2)
    model = doc.createModel()
    model.setId(model_id)
    model.setMetaId(model_id)
    libsbml.writeSBMLToFile(doc, str(path))


def with_empty_sbml(yaml_path) -> None:
    """Point a copy of the toy's YAML at an EMPTY SBML model, the shape a model-less problem ships
    in (the toy itself ships Antimony)."""
    yaml_path = Path(yaml_path)
    _write_empty_sbml(yaml_path.parent / "conversion_reaction.xml")
    yaml_path.write_text(
        yaml_path.read_text().replace(
            "location: model.ant\n    language: antimony",
            "location: conversion_reaction.xml\n    language: sbml",
        )
    )


@pytest.fixture
def empty_yaml(tmp_path) -> str:
    """A PEtab problem that ships WITHOUT a model (empty SBML, no model.ant)."""
    work = tmp_path / "empty"
    work.mkdir()
    for name in ("problem.yaml", "measurements.tsv", "observables.tsv", "parameters.tsv"):
        shutil.copy(DATA_DIR / name, work / name)
    with_empty_sbml(work / "problem.yaml")
    return str(work / "problem.yaml")


@pytest.fixture
def add_condition():
    """Return a helper giving a copied toy problem one condition that sets ``target`` to
    ``value``, applied by an experiment every measurement is taken in — the v2 shape of a
    one-column v1 condition table."""

    def _add(yaml_path, target: str, value: float) -> None:
        work = Path(yaml_path).parent
        (work / "conditions.tsv").write_text(
            f"conditionId\ttargetId\ttargetValue\nc0\t{target}\t{value}\n", encoding="utf-8"
        )
        (work / "experiments.tsv").write_text(
            "experimentId\ttime\tconditionId\ne0\t0\tc0\n", encoding="utf-8"
        )
        meas = work / "measurements.tsv"
        meas.write_text(meas.read_text().replace("obs_B\t\t", "obs_B\te0\t"), encoding="utf-8")
        yaml = Path(yaml_path)
        yaml.write_text(
            yaml.read_text()
            + "condition_files:\n- conditions.tsv\nexperiment_files:\n- experiments.tsv\n",
            encoding="utf-8",
        )

    return _add


@pytest.fixture
def empty_session(empty_yaml, tmp_path):
    """A session whose problem ships model-less — ``has_model()`` is False until ``set_model``."""
    from petab_mcp.references import StubReferenceResolver
    from petab_mcp.session import PetabSession

    return PetabSession(
        empty_yaml,
        provenance_dir=str(tmp_path / "prov"),
        resolver=StubReferenceResolver(),
    )


@pytest.fixture
def dosed_session(empty_yaml, tmp_path):
    """A model-less session whose one experiment pre-equilibrates with the condition target
    ``dose`` at 0 and sets it to 1 at t = 0, ``obs_B`` (``B``) measured after."""
    from petab_mcp.references import StubReferenceResolver
    from petab_mcp.session import PetabSession

    work = Path(empty_yaml).parent
    (work / "conditions.tsv").write_text(
        "conditionId\ttargetId\ttargetValue\npre\tdose\t0\ntreat\tdose\t1\n", encoding="utf-8"
    )
    (work / "experiments.tsv").write_text(
        "experimentId\ttime\tconditionId\ne0\t-inf\tpre\ne0\t0\ttreat\n", encoding="utf-8"
    )
    meas = work / "measurements.tsv"
    meas.write_text(meas.read_text().replace("obs_B\t\t", "obs_B\te0\t"), encoding="utf-8")
    yaml = Path(empty_yaml)
    yaml.write_text(
        yaml.read_text()
        + "condition_files:\n- conditions.tsv\nexperiment_files:\n- experiments.tsv\n",
        encoding="utf-8",
    )
    return PetabSession(
        empty_yaml, provenance_dir=str(tmp_path / "prov"), resolver=StubReferenceResolver()
    )


#: The toy's model without its back reaction, the second version of :func:`viz_store`.
_TOY_ONE_WAY = (DATA_DIR / "model.ant").read_text().replace("  r2: B => A; k2 * B;\n", "")


def _viz_records(sim_scale: float) -> list[dict]:
    """``obs_B`` in ``e_step`` (a dose step at t = 4) and ``e_pre`` (pre-equilibrated), each point's
    simulation ``sim_scale`` times its measurement."""
    out = []
    for experiment in ("e_step", "e_pre"):
        for t in (0.0, 2.0, 4.0, 6.0, 8.0):
            m = 0.1 * t + (0.5 if experiment == "e_pre" else 0.0)
            s = sim_scale * m
            out.append({"observable_id": "obs_B", "experiment_id": experiment, "time": t,
                        "measurement": m, "simulation": s, "residual": m - s,
                        "residual_weighted": (m - s) / 0.1, "noise": 0.1,
                        "noise_parameters": None})
    return out


def _viz_optimization(fvals: list[float]) -> dict:
    return {
        "n_starts": len(fvals),
        "starts": [{"start": i, "id": str(i), "fval": f, "time": 0.1 * (i + 1), "n_fval": 5,
                    "n_grad": 5, "n_hess": 5, "x": [-0.3 + 0.01 * i, -1.0 - 0.1 * i, 0.1, 1.0, 0.0]}
                   for i, f in enumerate(fvals)],
        "trace": [{"start": i, "iteration": it, "fval": f + 2.0 / (it + 1), "time": 0.01 * it}
                  for i, f in enumerate(fvals) for it in range(4)],
        "x_names": ["k1", "k2", "sigma_B", "A_init", "B_init"],
        "x_scales": ["log10", "log10", "lin", "lin", "lin"],
        "exitflag_counts": {"ExitFlag.FTOL": len(fvals)},
        "n_starts_killed": 0,
    }


def _viz_hdf5(path: Path, fvals: list[float]) -> None:
    """A pypesto_result.hdf5 in the layout pyPESTO's storage writes: FTOL, GTOL, then a start the
    server stopped past its cap (exit flag -1, as Fides' MAXITER)."""
    import h5py

    ended = [(1, "Converged according to fval difference"), (3, "Converged according to gradient"),
             (-1, "stopped by the server: still running past its 180s cap")]
    with h5py.File(path, "w") as fh:
        for i, (f, (flag, message)) in enumerate(zip(fvals, ended)):
            group = fh.create_group(f"optimization/results/{i}")
            group.attrs.update({"id": str(i), "fval": f, "exitflag": flag, "message": message})
            group["x0"] = [-0.5, -1.5, 0.1, 1.0, 0.0]


@pytest.fixture
def viz_store(tmp_path) -> tuple[Path, Path]:
    """A store as the server writes it, two fitted versions of the toy (the second without its
    back reaction), for the viewer (``petab_mcp.viz``); its problem has a two-period experiment
    and a pre-equilibrated one. v1's ``optimization.json`` is gzipped and it has no pyPESTO
    result file, as in a run gemot archived; v0 has one. Returns the store's root and the problem's
    YAML."""
    import gzip
    import json

    problem = tmp_path / "problem"
    shutil.copytree(DATA_DIR, problem)
    (problem / "conditions.tsv").write_text(
        "conditionId\ttargetId\ttargetValue\nlow\tA\t1\nhigh\tA\t3\n", encoding="utf-8")
    (problem / "experiments.tsv").write_text(
        "experimentId\ttime\tconditionId\ne_step\t0\tlow\ne_step\t4\thigh\n"
        "e_pre\t-inf\tlow\ne_pre\t0\thigh\n", encoding="utf-8")
    (problem / "measurements.tsv").write_text(
        "observableId\texperimentId\ttime\tmeasurement\n"
        + "".join(f"obs_B\t{r['experiment_id']}\t{r['time']}\t{r['measurement']}\n"
                  for r in _viz_records(1.0)), encoding="utf-8")
    yaml = problem / "problem.yaml"
    yaml.write_text(yaml.read_text() + "condition_files:\n- conditions.tsv\n"
                    "experiment_files:\n- experiments.tsv\n", encoding="utf-8")

    root = tmp_path / "prov"
    params = (problem / "parameters.tsv").read_text()
    versions = [
        (0, -1, (problem / "model.ant").read_text(), 12.0, [5.0, 5.2, 9.0], 0.9, True),
        (1, 0, _TOY_ONE_WAY, 10.0, [4.0, 4.1, 4.5, 30.0], 0.95, True),
    ]
    for v, parent, model, score, fvals, scale, best in versions:
        vdir = root / "versions" / f"v{v}"
        vdir.mkdir(parents=True)
        (vdir / "model.ant").write_text(model)
        (vdir / "parameters.tsv").write_text(params)
        (vdir / "meta.json").write_text(json.dumps({
            "version": v, "label": "fit", "kind": "fit", "parent": parent, "n_starts": len(fvals),
            "criterion": "bic", "criterion_value": score, "neg_log_likelihood": min(fvals),
            "k": 2, "is_new_best": best, "wall_seconds": 3}))
        (vdir / "simulations.json").write_text(json.dumps(
            {"status": "fit", "records": _viz_records(scale)}))
        blob = json.dumps(_viz_optimization(fvals))
        if v == 0:
            _viz_hdf5(vdir / "pypesto_result.hdf5", fvals)
        if v == 1:
            with gzip.open(vdir / "optimization.json.gz", "wt") as fh:
                fh.write(blob)
        else:
            (vdir / "optimization.json").write_text(blob)
    (root / "HEAD").write_text("1")
    (root / "FORMAT").write_text("petab-v2")
    (root / "provenance.jsonl").write_text("".join(
        json.dumps({"type": "fit", "version": v, "justification": why}) + "\n"
        for v, why in ((0, "the toy as shipped"), (1, "is the back reaction needed?"))))
    return root, yaml
