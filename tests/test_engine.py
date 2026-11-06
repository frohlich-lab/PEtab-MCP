from __future__ import annotations

import shutil

import pandas as pd
import pytest

from pathlib import Path


def test_evaluate_objective(engine):
    obj = engine.evaluate_objective()
    assert isinstance(obj["neg_log_likelihood"], float)
    assert obj["log_likelihood"] == pytest.approx(-obj["neg_log_likelihood"])
    assert obj["n_data_points"] == 6
    assert len(obj["simulations"]) == 6
    rec = obj["simulations"][0]
    assert {
        "observable_id",
        "experiment_id",
        "time",
        "measurement",
        "simulation",
        "residual",
    } <= set(rec)


def test_replicate_measurements_not_double_counted(engine, session):
    """Replicate measurements must not inflate n_data_points / chi2.

    ``_simulation_records`` used to merge the measurement and simulation tables on
    (observableId, condition, time) alone. For replicate rows that share
    a key, that cross-joins each size-r group into r*r rows, so a table with r=2
    replicates reported ``n_data_points == sum(r**2) == 24`` (and a ~2x-inflated chi2)
    instead of the true 12. The merge is now one-to-one with the measurements.
    """
    import numpy as np

    base = session.problem.measurement_df
    # Two replicates per point (distinct measured values, identical key): 6 groups of
    # size 2 => 12 true points, which the buggy merge would have blown up to 24.
    dup = base.copy()
    dup["measurement"] = dup["measurement"] + 0.01
    rep = pd.concat([base, dup], ignore_index=True)
    session.problem = session.problem.replace(measurement_df=rep)

    obj = engine.evaluate_objective()
    # one record per measurement row (Sum r), not the buggy cross-join (Sum r**2)
    assert obj["n_data_points"] == len(rep) == 2 * len(base)
    assert len(obj["simulations"]) == len(rep)

    # every (obs, experiment, time) group has exactly its 2 replicates, sharing one simulation
    groups: dict[tuple, list[float]] = {}
    for r in obj["simulations"]:
        groups.setdefault((r["observable_id"], r["experiment_id"], r["time"]), []).append(
            r["simulation"]
        )
    assert {len(v) for v in groups.values()} == {2}
    for sims in groups.values():
        assert sims[0] == pytest.approx(sims[1])  # replicates simulate identically

    # chi2 counts each weighted residual once, not r*r times: it equals the sum over
    # the records and stays consistent with AMICI's likelihood. Derive the noise term
    # from the per-record noise (not a hard-coded sigma) so the identity holds even if
    # the toy problem's noise model or size changes.
    weighted = np.array([r["residual_weighted"] for r in obj["simulations"]], dtype=float)
    noise = np.array([r["noise"] for r in obj["simulations"]], dtype=float)
    assert obj["chi2"] == pytest.approx(float(np.sum(weighted**2)), rel=1e-9)
    nllh = 0.5 * (obj["chi2"] + float(np.sum(np.log(2 * np.pi * noise**2))))
    assert nllh == pytest.approx(obj["neg_log_likelihood"], rel=1e-6)


def test_records_carry_weighted_residuals(engine):
    rec = engine.evaluate_objective()["simulations"][0]
    assert {"residual", "residual_weighted", "noise", "noise_parameters"} <= set(rec)
    # obs_B noise is sigma_B (nominal 0.1); weighted residual = residual / sigma
    assert rec["noise"] == pytest.approx(0.1)
    assert rec["residual_weighted"] == pytest.approx(rec["residual"] / rec["noise"])


def test_residuals_summary_matches_chi2(engine):
    import math

    import numpy as np

    data = engine.residuals()
    assert data["n_data_points"] == 6
    assert {"neg_log_likelihood", "chi2", "records"} <= set(data)

    # chi2 is the sum of squared noise-weighted residuals (petab-canonical)
    weighted = np.array([r["residual_weighted"] for r in data["records"]], dtype=float)
    noise = np.array([r["noise"] for r in data["records"]], dtype=float)
    assert float(np.sum(weighted**2)) == pytest.approx(data["chi2"], rel=1e-9)

    # the Gaussian log-likelihood identity ties chi2 + noise back to AMICI's objective,
    # cross-validating that our weighted residuals reproduce the fitted likelihood
    nllh = 0.5 * (data["chi2"] + float(np.sum(np.log(2 * math.pi * noise**2))))
    assert nllh == pytest.approx(data["neg_log_likelihood"], rel=1e-6)


def test_parameter_change_no_recompile(engine, session):
    from petab_mcp import parameters as P

    obj0 = engine.evaluate_objective()
    h = session.sbml_hash()
    P.update_parameters(session.problem, [{"id": "k1", "nominalValue": 1.0}])
    obj1 = engine.evaluate_objective()
    assert session.sbml_hash() == h  # model unchanged
    assert obj1["neg_log_likelihood"] != obj0["neg_log_likelihood"]


def test_simulate_after_operation_recompiles(engine, session, reauthor_toy):
    before = engine.evaluate_objective()["neg_log_likelihood"]
    reauthor_toy(session, ["J: B => D; k3 * B;"], justification="B to D", references=["PMID:1"])
    after = engine.evaluate_objective()  # new species/param -> recompiles
    assert after["neg_log_likelihood"] != before
    assert after["n_data_points"] == 6


def test_fit_and_simulate_improves_fit(fit_session):
    from petab_mcp import fitting as F

    engine = fit_session.engine
    before = engine.evaluate_objective()["neg_log_likelihood"]
    result = F.fit_and_simulate(fit_session, engine, do_fit=True, n_starts=3)
    assert result["status"] == "fit"
    assert result["neg_log_likelihood"] <= before + 1e-6
    # fitted_parameters holds ONLY the estimated params (k1, k2) — never the fixed
    # sigma_B/A_init/B_init — so k = len(fitted) equals n_estimated and the scores stay honest.
    assert set(result["fitted_parameters"]) == {"k1", "k2"}
    assert len(result["records"]) == 6
    assert fit_session.simulation is result  # stored on the session


def test_fit_records_optimization_trace(fit_session):
    from petab_mcp import fitting as F

    F.fit_and_simulate(fit_session, fit_session.engine, do_fit=True, n_starts=3)
    opt = fit_session.optimization
    assert opt is not None
    # one result row per start; start 0 is the best (lowest fval)
    assert opt["n_starts"] == 3 and len(opt["starts"]) == 3
    fvals = [s["fval"] for s in opt["starts"]]
    assert fvals[0] == min(v for v in fvals if v is not None)
    # a per-iteration trace was recorded for at least the best start
    assert any(t["start"] == 0 for t in opt["trace"])
    assert all({"start", "iteration", "fval"} <= set(t) for t in opt["trace"])

    # a simulate-only pass clears the (non-persisted) trace
    F.fit_and_simulate(fit_session, fit_session.engine, do_fit=False, n_starts=0)
    assert fit_session.optimization is None


def test_a_fit_leaves_usable_warm_starts_for_the_next_one(fit_session):
    """`previous_optima` needs BOTH x_names and x_scales, or it returns [] and the next fit loses
    its warm starts."""
    from petab_mcp import fitting as F

    F.fit_and_simulate(fit_session, fit_session.engine, do_fit=True, n_starts=2)
    opt = fit_session.optimization
    assert opt and opt["starts"]
    assert opt.get("x_names") and opt.get("x_scales")
    assert len(opt["x_names"]) == len(opt["x_scales"])

    x0 = [0.0] * len(opt["x_names"])
    assert F.previous_optima(F.estimation_problem(fit_session, fit_session.engine)[0], opt, x0)


def test_fit_tracks_improvement_vs_versions(fit_session):
    from petab_mcp import operations as O

    # edits are fit-free; `fit` scores them and tracks improvement across the fitted versions
    O.change_parameters(
        fit_session,
        [{"parameter_id": "k1", "lower_bound": 1e-3, "upper_bound": 1e3}],
        justification="widen k1",
    )
    first = fit_session.fit()
    assert first["fit_status"] == "fit"
    imp1 = first["improvement"]
    assert fit_session.provenance.get_meta(first["version"])["objective_before"] is not None
    assert imp1["self_improvement"] is not None
    assert imp1["is_new_best"] is True  # nothing to beat yet
    assert imp1["improvement_vs_best_prior"] is None

    O.change_parameters(
        fit_session,
        [{"parameter_id": "k2", "lower_bound": 1e-3, "upper_bound": 1e3}],
        justification="widen k2",
    )
    second = fit_session.fit()
    imp2 = second["improvement"]
    assert second["progress"]["best_version"] is not None
    assert imp2["improvement_vs_best_prior"] is not None


def test_a_problem_that_cannot_be_seeded_refuses_to_load(tmp_path):
    """The empty model is UNREACHABLE, not guarded. AMICI's SBML import segfaults on it — a
    process kill no `except` can catch — so every path that touches AMICI used to carry a
    `has_model()` check. Those are gone: a model-less problem is seeded with a valid model, and
    if it cannot be, the session REFUSES TO CONSTRUCT rather than starting in the one state
    nothing downstream survives. A problem whose observables name nothing declarable is the
    case: there is no model the fixed tables could imply."""
    import shutil

    from conftest import with_empty_sbml

    from petab_mcp.errors import ValidationError
    from petab_mcp.references import StubReferenceResolver
    from petab_mcp.session import PetabSession

    data = Path(__file__).parent / "data" / "toy_v2"
    work = tmp_path / "unseedable"
    work.mkdir()
    for name in ("problem.yaml", "measurements.tsv", "parameters.tsv"):
        shutil.copy(data / name, work / name)
    # An observable whose formula is a bare NUMBER references no entity, so nothing can be
    # declared from it; the parameter table's rows all resolve through the noise formula.
    (work / "observables.tsv").write_text(
        "observableId\tobservableFormula\tnoiseFormula\nobs_B\t1.0\tsigma_B\n", encoding="utf-8"
    )
    with_empty_sbml(work / "problem.yaml")

    with pytest.raises(ValidationError):
        PetabSession(
            str(work / "problem.yaml"),
            provenance_dir=str(tmp_path / "prov"),
            resolver=StubReferenceResolver(),
        )


def test_fit_objective_and_reported_nll_are_the_same_number(fit_session):
    """The fit's objective and the reported nll must be the same likelihood.

    They are the same likelihood at the same parameters, but they used to be computed by two
    different AMICI models with two different solvers — the fit through pyPESTO's objective,
    the report by re-simulating via simulate_petab. That let them disagree: a fit converged at nll
    266.62 while the re-simulation of its own optimum failed to integrate past t=22 and reported
    NaN, which then poisoned the best-BIC checkpoint and cost the session its score. Both now go
    through one shared objective (fitting.estimation_problem).

    Sharing that objective makes the FORMULA and the parameters identical, but each side still
    runs its own adaptive integration, so the agreement is bounded by the ODE solver's tolerance,
    not by floating-point rounding — a rounding-level bound here is a flake waiting for a solver
    or platform that lands the optimum slightly differently. The bound is therefore
    OBJECTIVE_AGREEMENT_RTOL, the same threshold `_reconcile_objective` uses in production to
    decide these two numbers "differ in substance"; the regression above is far coarser than it
    (a NaN or an unreached point fails at any tolerance).
    """
    from petab_mcp import fitting

    res = fit_session.fit(justification="test")
    reported = res["neg_log_likelihood"]
    assert reported is not None

    rtol = fitting.OBJECTIVE_AGREEMENT_RTOL
    # Re-evaluating the committed model must reproduce the number that was reported for it.
    again = fit_session.engine.evaluate_objective()["neg_log_likelihood"]
    assert again == pytest.approx(reported, rel=rtol)

    # ...and that number is the optimizer's own objective at the fitted point.
    problem, _ = fitting.estimation_problem(fit_session, fit_session.engine)
    x = fitting.problem_x_from_nominal(problem, fit_session.problem.parameter_df)
    assert float(problem.objective(x)) == pytest.approx(reported, rel=rtol)


def test_the_experiments_are_simulated_on_the_amici_threads_the_server_is_given(
    tmp_path, amici_build_root, monkeypatch
):
    """``PETAB_MCP_FIT_AMICI_N_THREADS`` reaches AMICI's call, whose threads share out the
    experiments; unset, it is one thread, as before the knob. Each experiment is simulated on its
    own, so four threads score the problem as one does, gradient included."""
    import amici.sim.sundials

    from petab_mcp import fitting
    from petab_mcp.engine import AmiciEngine
    from petab_mcp.references import StubReferenceResolver
    from petab_mcp.session import PetabSession

    work = tmp_path / "problem"
    shutil.copytree(Path(__file__).parent / "data" / "toy_v2", work)
    (work / "conditions.tsv").write_text(
        "conditionId\ttargetId\ttargetValue\n" + "".join(f"c{i}\tA\t{i + 1}\n" for i in range(4))
    )
    (work / "experiments.tsv").write_text(
        "experimentId\ttime\tconditionId\n" + "".join(f"e{i}\t0\tc{i}\n" for i in range(4))
    )
    header, *rows = (work / "measurements.tsv").read_text().splitlines()
    measured = [r.replace("obs_B\t\t", f"obs_B\te{i}\t") for i in range(4) for r in rows]
    (work / "measurements.tsv").write_text("\n".join([header, *measured]) + "\n")
    yaml = work / "problem.yaml"
    yaml.write_text(
        yaml.read_text()
        + "condition_files:\n- conditions.tsv\nexperiment_files:\n- experiments.tsv\n"
    )

    calls = []
    run = amici.sim.sundials.run_simulations

    def counted(model, solver, edatas, *args, num_threads=1, **kwargs):
        calls.append((len(edatas), num_threads))
        return run(model, solver, edatas, *args, num_threads=num_threads, **kwargs)

    monkeypatch.setattr(amici.sim.sundials, "run_simulations", counted)

    def score(name):
        session = PetabSession(
            str(yaml), provenance_dir=str(tmp_path / name), resolver=StubReferenceResolver()
        )
        session.engine = AmiciEngine(session, build_root=str(amici_build_root))
        problem, _ = fitting.estimation_problem(session, session.engine)
        x = fitting.problem_x_from_nominal(problem, session.problem.parameter_df)
        calls.clear()
        fval, grad = problem.objective(x, sensi_orders=(0, 1))
        return fval, grad, list(calls)

    one = score("default")
    monkeypatch.setattr(fitting, "FIT_AMICI_N_THREADS", 4)
    four = score("four")

    assert one[2] == [(4, 1)] and four[2] == [(4, 4)]
    assert four[0] == one[0] and (four[1] == one[1]).all()


def test_one_amici_build_per_structure(fit_session):
    """The fit and the objective evaluation must share ONE compiled model.

    They used to build two (`petab_mcp_<hash>` and `petab_mcp_pp_<hash>`) from the same SBML —
    at ~80s a build on a cluster node, that was ~20 minutes of a 73-minute session spent
    compiling the same structure twice.
    """
    fit_session.fit(justification="test")
    n_after_first = fit_session.n_compiles
    # A re-fit of the SAME structure must not compile again...
    fit_session.engine.evaluate_objective()
    fit_session.fit(justification="test again")
    assert fit_session.n_compiles == n_after_first
    # ...and the objective's model is the one the engine reports on.
    problem, _ = __import__("petab_mcp.fitting", fromlist=["x"]).estimation_problem(
        fit_session, fit_session.engine
    )
    assert problem.objective.amici_model is not None
    # ...and the real build is one `build_lock` takes for finished; were the extension elsewhere,
    # every later build would remove the module and compile again, while n_compiles stayed put.
    from petab_mcp.engine import _build_finished

    assert _build_finished(*fit_session.engine._module_target(fit_session.module_hash()))


# -- pre-import model dump ----------------------------------------------------


def _log_text(session) -> str:
    """The human-readable log this session writes (``<store>/petab_mcp.log``)."""
    for handler in session.logger.handlers:
        handler.flush()
    return (Path(session.provenance.root) / "petab_mcp.log").read_text(encoding="utf-8")


def test_the_model_is_dumped_before_the_amici_import(engine, session):
    """AMICI's SBML import + C++ codegen can kill the process outright — a segfault-class
    failure that raises nothing and leaves no traceback, so the client just sees the connection
    drop. The only forensic trail is a dump written BEFORE the import call; a session once died
    exactly this way with no record of which model did it.

    `pypesto_importer_kwargs` is the choke point: the ONE build site (the fit) constructs its
    importer on the next line, so dumping here covers every AMICI build without a second hook.
    """
    engine.pypesto_importer_kwargs(session.sbml_hash())

    text = _log_text(session)
    assert "AMICI import starting" in text
    assert session.sbml_hash() in text
    assert "k1*A" in text  # the actual model, not just a hash


def test_the_pre_import_dump_is_written_once_per_structure(engine, session, reauthor_toy):
    """Once per STRUCTURE, not once per fit. Only a build can crash, and a re-fit or a
    bounds-only edit rebuilds the pyPESTO problem while reusing the compiled module — dumping
    per call would bury the log under re-prints of an already-recorded model."""
    engine.pypesto_importer_kwargs(session.sbml_hash())
    engine.pypesto_importer_kwargs(session.sbml_hash())
    assert _log_text(session).count("AMICI import starting") == 1

    reauthor_toy(session, ["J: B => D; k3 * B;"], justification="B to D")
    engine.pypesto_importer_kwargs(session.sbml_hash())  # new structure -> a new build -> a dump
    text = _log_text(session)
    assert text.count("AMICI import starting") == 2
    assert "k3*B" in text


def test_a_failing_dump_never_blocks_the_build(engine, session, monkeypatch):
    """Diagnostics must not become the outage. A model that will not round-trip to Antimony
    still has to be handed to AMICI — and is still worth recording by hash."""

    def _boom():
        raise RuntimeError("antimony round-trip failed")

    monkeypatch.setattr(session.problem.model, "to_antimony", _boom)

    kwargs = engine.pypesto_importer_kwargs(session.sbml_hash())
    assert kwargs["model_name"] == f"petab_mcp_pp_{session.sbml_hash()}"
    assert "model dump unavailable" in _log_text(session)


def test_module_built_asks_for_a_finished_build_not_the_folder(session, tmp_path):
    """The predicate a READ rests on: `ensure_simulation` will simulate a structure whose module
    exists and refuse one whose does not, because the second costs a C++ build (16 s on the toy) and
    no read may pay one. A HALF-WRITTEN build must therefore read as absent — the output folder, its
    generated C++ and even its extension survive a crashed or interrupted build, and the next build
    removes them and recompiles from scratch."""
    from petab_mcp.engine import AmiciEngine

    # a PRIVATE build root: the shared session-scoped one already holds the toy's module
    engine = AmiciEngine(session, build_root=str(tmp_path / "build"))
    h = session.sbml_hash()
    assert engine.module_built(h) is False  # cold build root

    folder = engine.module_root / f"pypesto_{h}"
    package = folder / f"petab_mcp_pp_{h}"
    package.mkdir(parents=True)
    (folder / "wrapfunctions.cpp").write_text("// codegen ran")
    (package / "__init__.py").write_text("")
    assert engine.module_built(h) is False  # generated, not built

    (package / f"_petab_mcp_pp_{h}.cpython-312-darwin.so").write_bytes(b"\x00")
    assert engine.module_built(h) is False  # installed, but its build never marked it finished

    (folder / ".built").touch()
    assert engine.module_built(h) is True

    # ...and what THIS session built is known without going to disk at all
    other = "0123456789abcdef"
    assert engine.module_built(other) is False
    engine._pypesto_hashes.add(other)
    assert engine.module_built(other) is True


class _AmiciDist:
    """An installed AMICI as ``importlib.metadata`` reports it, with its PEP 610 record."""

    version = "1.1.0"

    def __init__(self, direct_url):
        self.direct_url = direct_url

    def read_text(self, name):
        return self.direct_url if name == "direct_url.json" else None


@pytest.mark.parametrize(
    ("direct_url", "identity"),
    [
        (
            '{"url": "https://github.com/AMICI-dev/AMICI.git", "subdirectory": "python/sdist", '
            '"vcs_info": {"vcs": "git", "commit_id": "a4dbb08da4d3c782ba3a2bab47c92586467ea5bc"}}',
            "1.1.0-a4dbb08da4d3",
        ),
        ('{"url": "file:///src/AMICI/python/sdist", "dir_info": {"editable": true}}', "1.1.0"),
        (None, "1.1.0"),
    ],
)
def test_amici_identity_names_the_commit_a_git_pin_installed(monkeypatch, direct_url, identity):
    """PEtab-MCP#103: the pins move between AMICI commits that share a version string, so a git
    install is told apart by the commit its install recorded; anything else by its version."""
    import importlib.metadata

    from petab_mcp import engine as E

    monkeypatch.setattr(importlib.metadata, "distribution", lambda name: _AmiciDist(direct_url))
    E.amici_identity.cache_clear()
    try:
        assert E.amici_identity() == identity
    finally:
        E.amici_identity.cache_clear()


# -- a failed integration is a MISSING score, never an "ill-posed model" verdict -------------
class _DeadRData:
    """A ReturnData as AMICI's failfast path leaves it: output arrays are None, no exception.

    Feeding this to `rdatas_to_simulation_df` raised a bare `TypeError: 'NoneType' object is not
    iterable`, which `_compile_fit_failure_message` then reported to the agent as an ill-posed ODE
    -- sending it to rewrite a structurally fine model. 32 rejected `make_changes` calls in one
    batch of sessions, every one of them AFTER a successful compile (~94 min of wasted builds).
    """

    ts = None
    y = None
    res = None
    status = -1


def test_failed_integration_yields_records_with_no_simulation(engine, session):
    meas = session.problem.measurement_df
    recs = engine._simulation_records(None, [_DeadRData()])

    # It returns, rather than raising the TypeError that used to be misdiagnosed downstream.
    assert len(recs) == len(meas)
    assert all(r["simulation"] is None for r in recs)
    assert all(r["residual"] is None for r in recs)
    # Measurements are still carried, so the gap is reportable rather than invisible.
    assert all(r["measurement"] is not None for r in recs)
    n_expected, n_missing = type(engine).simulation_gap(recs)
    assert n_expected == len(meas) and n_missing == len(meas)


def test_partial_integration_failure_is_not_scored_as_a_smaller_dataset(engine, session):
    """One good experiment + one dead one must NOT yield a confident score over the survivors:
    AMICI's likelihood is already NaN for such an evaluation, so a partial chi2 would describe a
    smaller dataset than `n_data_points` while looking complete."""
    good = engine._objective_evaluate()  # a real, working evaluation first
    assert good[1] and any(r["simulation"] is not None for r in good[1])

    class _Live:
        ts = [0.0]
        y = [[1.0]]
        res = [0.0]
        status = 0

    recs = engine._simulation_records(None, [_Live(), _DeadRData()])
    assert all(r["simulation"] is None for r in recs)
    assert engine._chi2(recs, None) is None  # declines to score, rather than partial-summing


def test_unusable_rdatas_counts_only_the_dead_conditions(engine):
    class _Live:
        ts = [0.0, 1.0]
        y = [[1.0], [2.0]]

    E = type(engine)
    assert E._unusable_rdatas([_Live(), _Live()]) == 0
    assert E._unusable_rdatas([_Live(), _DeadRData()]) == 1
    assert E._unusable_rdatas([_DeadRData(), _DeadRData()]) == 2
    # An empty time vector is just as unusable as a None one.
    assert E._unusable_rdatas([type("E0", (), {"ts": [], "y": []})()]) == 1


def test_an_unmeasured_experiment_leaves_the_fit_statistics_intact(tmp_path, amici_build_root):
    """AMICI returns no outputs for an experiment without measurements, which every reader of the
    rdatas would count a failed simulation — blanking chi2 and every residual. A fixed-θ held-out
    view keeps all the training experiments, and a v2 problem may ship unmeasured ones."""
    import shutil

    from conftest import DATA_DIR

    from petab_mcp.engine import AmiciEngine
    from petab_mcp.references import StubReferenceResolver
    from petab_mcp.session import PetabSession

    work = tmp_path / "toy"
    shutil.copytree(DATA_DIR, work)
    (work / "conditions.tsv").write_text(
        "conditionId\ttargetId\ttargetValue\nc0\tA\t1.0\nc1\tA\t2.0\n"
    )
    (work / "experiments.tsv").write_text("experimentId\ttime\tconditionId\ne0\t0\tc0\ne1\t0\tc1\n")
    meas = work / "measurements.tsv"
    meas.write_text(meas.read_text().replace("obs_B\t\t", "obs_B\te0\t"))
    yaml = work / "problem.yaml"
    yaml.write_text(
        yaml.read_text()
        + "condition_files:\n- conditions.tsv\nexperiment_files:\n- experiments.tsv\n"
    )
    session = PetabSession(
        str(yaml), provenance_dir=str(tmp_path / "prov"), resolver=StubReferenceResolver()
    )
    out = AmiciEngine(session, build_root=str(amici_build_root)).evaluate_objective()
    assert out["chi2"] is not None and out["n_data_points"] == 6
    assert all(r["simulation"] is not None for r in out["simulations"])
