"""The TRANSFER kind of held-out check: a second problem on the same mechanism, on which both the
candidate and the reference are REFIT (``petab_mcp.validation_data``).

Tested apart from each other, exactly as the fixed-θ kind is in ``test_validation_data.py``:
which tables move onto the candidate's view and which do not, how a bundle's units are read, and
what the registered tool accepts. The AMICI-backed tests drive the whole server-side path
(``score_session``) on the toy problem, because "the refit runs and returns the shape the caller
reads" is not something a mocked build can establish. Assembling a transfer set, and what the
caller does with the payload, are the caller's to test.
"""

from __future__ import annotations

import math
import shutil
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from petab_mcp import validation_data as V
from petab_mcp.problem import WorkingProblem

TOY = Path(__file__).parent / "data" / "toy_v2"


def _params(rows):
    df = pd.DataFrame(
        rows,
        columns=[
            "parameterId",
            "parameterScale",
            "lowerBound",
            "upperBound",
            "nominalValue",
            "estimate",
        ],
    )
    return df.set_index("parameterId")


def _conditions(ids, target="dose"):
    """A condition table in which each of ``ids`` sets ``target``."""
    return pd.DataFrame({"conditionId": list(ids), "targetId": target, "targetValue": 1.0})


def _experiments(ids, conditions=None):
    """One single-period experiment per id, applying ``conditions`` (none by default)."""
    conditions = [None] * len(ids) if conditions is None else list(conditions)
    return pd.DataFrame({"experimentId": list(ids), "time": 0.0, "conditionId": conditions})


def _rows(experiment, times=(0.0, 1.0), observable="obs_B"):
    return pd.DataFrame(
        {
            "observableId": [observable] * len(times),
            "experimentId": [experiment] * len(times),
            "time": list(times),
            "measurement": [1.0] * len(times),
        }
    )


def _held_out(experiment="e1", target="dose", **kwargs):
    """A transfer set measured in its own experiment ``experiment``, whose condition ``c1`` sets
    ``target``."""
    return V.HeldOut(
        measurement_df=kwargs.pop("measurement_df", _rows(experiment)),
        condition_df=_conditions(["c1"], target),
        experiment_df=_experiments(["e1"], ["c1"]),
        parameter_df=kwargs.pop("parameter_df", _HELD_OUT_PARAMS),
    )


def test_a_held_out_set_is_a_transfer_set_iff_it_brings_its_own_parameter_table(tmp_path):
    """The mode is READ OFF the files, not declared beside them: a parameter table is the statement
    that the parameters are not shared, and a flag next to it would be a second thing able to
    disagree with the first."""
    meas = tmp_path / "measurements.tsv"
    _rows("e1").to_csv(meas, sep="\t", index=False)
    assert V.read_held_out(meas).mode == V.MODE_FIXED

    _conditions(["c1"]).to_csv(tmp_path / V.HELD_OUT_CONDITIONS, sep="\t", index=False)
    held = V.read_held_out(meas)
    assert held.mode == V.MODE_FIXED and list(held.condition_df["conditionId"]) == ["c1"]

    _params([("A_init", "lin", 0.0, 100.0, 2.0, 0)]).reset_index().to_csv(
        tmp_path / V.HELD_OUT_PARAMETERS, sep="\t", index=False
    )
    held = V.read_held_out(meas)
    assert held.mode == V.MODE_TRANSFER
    assert held.parameter_df.loc["A_init", "nominalValue"] == 2.0


def _candidate():
    """A candidate problem with the toy problem's tables, A_init fixed at 1.0, measured in one
    experiment whose condition sets the input ``dose``."""
    return WorkingProblem(
        measurement_df=_rows("e0"),
        observable_df=pd.DataFrame(
            {"observableFormula": ["B"], "noiseFormula": ["sigma_B"]},
            index=pd.Index(["obs_B"], name="observableId"),
        ),
        condition_df=_conditions(["c0"]),
        experiment_df=_experiments(["e0"], ["c0"]),
        parameter_df=_params(
            [
                ("k1", "log10", 1e-4, 100.0, 0.5, 1),
                ("k2", "log10", 1e-4, 100.0, 0.1, 1),
                ("sigma_B", "lin", 1e-4, 100.0, 0.1, 0),
                ("A_init", "lin", 0.0, 100.0, 1.0, 0),
            ]
        ),
    )


_SHIPPED = _params(
    [
        ("k1", "log10", 1e-4, 100.0, 0.5, 1),
        ("k2", "log10", 1e-4, 100.0, 0.1, 1),
        ("sigma_B", "lin", 1e-4, 100.0, 0.1, 0),
        ("A_init", "lin", 0.0, 100.0, 1.0, 0),
    ]
)


_HELD_OUT_PARAMS = _params(
    [
        ("k1", "log10", 1e-4, 100.0, 7.0, 1),  # estimated: the published held-out optimum
        ("sigma_B", "lin", 1e-4, 100.0, 0.1, 0),  # fixed, unchanged
        ("A_init", "lin", 0.0, 100.0, 2.0, 0),  # fixed, CHANGED: the input that must move
        ("sigma_extra", "lin", 1e-4, 100.0, 3.0, 0),  # fixed, never in the training problem
    ]
)


def test_the_view_moves_only_the_changed_fixed_inputs_and_never_the_start_point():
    """Three kinds of held-out row and three different fates. `A_init` (fixed, changed) is written
    into the candidate; `k1` (estimated) is NOT — the held-out table's value is the published optimum
    for that system and would hand the candidate the answer, so its refit starts from its own
    training value; `sigma_extra` (a fixed quantity the training problem never named) binds to
    nothing. And the caller's problem is untouched throughout."""
    problem = _candidate()
    view, failed = V.transfer_view(problem, _held_out(), _SHIPPED)
    assert failed is None
    assert view.parameter_df.loc["A_init", "nominalValue"] == 2.0
    assert view.parameter_df.loc["k1", "nominalValue"] == 0.5, (
        "the start point is the candidate's own"
    )
    assert "sigma_extra" not in view.parameter_df.index
    assert list(view.condition_df["conditionId"]) == ["c1"]
    assert list(view.experiment_df["experimentId"]) == ["e1"]
    assert list(view.measurement_df["experimentId"]) == ["e1", "e1"]
    # nothing leaked back
    assert problem.parameter_df.loc["A_init", "nominalValue"] == 1.0
    assert list(problem.condition_df["conditionId"]) == ["c0"]
    assert list(problem.measurement_df["experimentId"]) == ["e0", "e0"]


def test_a_candidate_that_cannot_receive_a_changed_input_is_a_finding_not_a_crash():
    """A fixed input (here `A_init`; a population size, say) can change between the training and
    the held-out system. A model that hard-coded it (no row), renamed it, or freed it has no fixed
    parameter to receive the new value, so it cannot be moved to the held-out system at all —
    reported as its own status, because a refit at the TRAINING value would be a plausible, wrong
    number."""
    no_row = _candidate()
    no_row.parameter_df = no_row.parameter_df.drop(index="A_init")
    view, failed = V.transfer_view(no_row, _held_out(), _SHIPPED)
    assert view is None and failed["status"] == V.STATUS_INPUT_MISSING
    assert "A_init" in failed["reason"]

    freed = _candidate()
    freed.parameter_df.loc["A_init", "estimate"] = True
    view, failed = V.transfer_view(freed, _held_out(), _SHIPPED)
    assert view is None and failed["status"] == V.STATUS_INPUT_MISSING


@pytest.mark.parametrize("rate", ["k1", "candidate_rate"])
def test_a_transfer_keeps_the_candidates_own_kinetic_parameters(rate):
    """The shipped subset omits kinetics; a candidate may also rename them. Neither permits
    copying the reference's fitted rates into the candidate or rejecting its transfer refit."""
    problem = _candidate()
    problem.parameter_df = problem.parameter_df.rename(index={"k1": rate})
    view, failed = V.transfer_view(problem, _held_out(), _SHIPPED.drop(index=["k1", "k2"]))
    assert failed is None
    pd.testing.assert_frame_equal(
        view.parameter_df.drop(index="A_init"), problem.parameter_df.drop(index="A_init")
    )
    assert view.parameter_df.loc[rate, "nominalValue"] == 0.5
    assert view.parameter_df.loc["A_init", "nominalValue"] == 2.0


def test_a_held_out_system_may_bring_its_own_estimated_noise_parameters():
    """A new noise sigma or observable scaling IS manageable: it enters through the observation
    model — the held-out measurement table's `noiseParameters` override — not the ODE."""
    rows = [
        ("k1", "log10", 1e-4, 100.0, 7.0, 1),
        ("sigma_B", "lin", 1e-4, 100.0, 0.1, 0),
        ("A_init", "lin", 0.0, 100.0, 2.0, 0),
        ("sigma_new", "lin", 1e-4, 100.0, 0.3, 1),
    ]
    held = _held_out(
        measurement_df=_rows("e1").assign(noiseParameters="sigma_new"), parameter_df=_params(rows)
    )
    view, failed = V.transfer_view(_candidate(), held, _SHIPPED)
    assert failed is None, failed


def test_a_held_out_condition_table_with_other_targets_is_another_model():
    """A condition target is an input the model declares and the compiled module maps; a held-out
    table setting another one describes a model the candidate is not, and reusing the module would
    be wrong."""
    view, failed = V.transfer_view(_candidate(), _held_out(target="temperature"), _SHIPPED)
    assert view is None and failed["status"] == V.STATUS_TABLE_MISMATCH
    assert "temperature" in failed["reason"]


def test_held_out_experiments_are_checked_against_the_held_out_table_not_the_training_one():
    """The whole point of the design swap: `e1` is not a training experiment and must still pass,
    while a measurement row naming an experiment in NEITHER table must not."""
    view, failed = V.transfer_view(_candidate(), _held_out(), _SHIPPED)
    assert failed is None and view is not None

    view, failed = V.transfer_view(_candidate(), _held_out("e_nowhere"), _SHIPPED)
    assert failed["status"] == V.STATUS_TABLE_MISMATCH and "e_nowhere" in failed["reason"]


def _never_build(monkeypatch):
    import petab_mcp.fitting as fitting

    def _boom(*a, **k):  # pragma: no cover - must not be reached
        raise AssertionError("nothing must be built for a declined table")

    monkeypatch.setattr(fitting, "pypesto_problem", _boom)


def test_refit_declines_a_bad_table_before_any_build(monkeypatch):
    """Same contract as `score_problem`: the guards run before the build, so an empty or foreign
    table costs nothing and names its own problem."""
    _never_build(monkeypatch)
    problem = _candidate()
    empty = problem.replace(measurement_df=problem.measurement_df.iloc[0:0])
    out = V.refit_problem(empty, build_dir="/tmp/x", model_name="m", n_starts=1)
    assert out["status"] == V.STATUS_NO_DATA

    foreign = problem.replace(measurement_df=_rows("e_elsewhere"))
    out = V.refit_problem(foreign, build_dir="/tmp/x", model_name="m", n_starts=1)
    assert out["status"] == V.STATUS_TABLE_MISMATCH and "e_elsewhere" in out["reason"]


def test_score_session_refits_the_toy_on_a_transfer_set_and_leaves_the_session_alone(
    session, engine
):
    """The one AMICI-backed test of a single system: the toy model, its A_init doubled, is REFIT on
    measurements it never saw, on the toy's own experiment-less design. Pins the payload SHAPE the
    caller reads (`mode`, `k`, `n`, a finite `nll`), that `k` is the count of estimated rows (k1,
    k2 — not sigma_B or A_init), and
    that the session's own table and model are exactly as they were: a transfer refit that wrote
    its optimum back would change what the caller scores on the training data."""
    meas = session.problem.measurement_df
    held_meas = meas.assign(measurement=meas["measurement"] * 2.0)
    held = V.HeldOut(
        measurement_df=held_meas,
        parameter_df=_params(
            [("A_init", "lin", 0.0, 100.0, 2.0, 0), ("k1", "log10", 1e-4, 100.0, 0.9, 1)]
        ),
    )
    before = session.problem.parameter_df.copy(deep=True)
    sbml_before = session.model_sbml()
    session.shipped_parameter_df = session.shipped_parameter_df.drop(index=["k1", "k2"])

    out = V.score_session(session, engine, held, n_starts=2)

    assert out["status"] == V.STATUS_OK, out
    assert out["mode"] == V.MODE_TRANSFER
    assert out["k"] == 2 and out["n"] == len(held_meas) and out["n_experiments"] == 1
    assert out["observables"] == ["obs_B"] and out["n_starts"] == 2
    assert math.isfinite(out["nll"]) and out["chi2"] is not None
    # the doubled data at A_init = 2 is the SAME curve shape, so the refit from the training
    # optimum must land near the training likelihood — a refit at the training A_init (the
    # input NOT moved) could not
    view, _ = V.transfer_view(session.problem, held, session.shipped_parameter_df)
    assert view.parameter_df.loc["A_init", "nominalValue"] == 2.0
    pd.testing.assert_frame_equal(session.problem.parameter_df, before)
    assert session.model_sbml() == sbml_before
    assert session.problem.condition_df is None and session.problem.experiment_df is None


def test_a_transfer_refit_runs_at_its_sessions_per_start_cap(session, tmp_path, monkeypatch):
    """A held-out refit runs at the per-start cap its session's own fits ran on, which a server
    hosting several problems sets per session (#187); a caller without a session, such as the
    reference bar's, gets the default every session starts with."""
    from petab_mcp import fitting

    caps = []
    monkeypatch.setattr(
        V, "_refit_one", lambda view, *, cap, **_: caps.append(cap) or {"status": V.STATUS_OK}
    )
    build = {"output_folder": str(tmp_path / "build"), "model_name": "m"}
    engine = SimpleNamespace(pypesto_importer_kwargs=lambda h: build, fit_engine=None)
    held = V.HeldOut(
        measurement_df=session.problem.measurement_df,
        parameter_df=_params([("A_init", "lin", 0.0, 100.0, 2.0, 0)]),
    )
    session.fit_per_start_timeout = 1200.0

    V.score_session(session, engine, held, n_starts=2)
    V.refit_problem(session.problem, build_dir=tmp_path / "b", model_name="m", n_starts=1)

    assert caps == [1200.0, fitting.FIT_PER_START_TIMEOUT_S]


def test_the_server_tool_takes_n_starts_and_refits_a_transfer_bundle(tmp_path):
    """The registered tool is what whoever runs the session reaches over MCP: it must accept
    `n_starts` (or that call fails and the caller records `unavailable`) and must read the bundle
    as a transfer set. Driven in-process through `create_server` like the doc-consistency tests."""
    import asyncio

    from petab_mcp.server import create_server

    prob = tmp_path / "problem"
    shutil.copytree(TOY, prob)
    val = tmp_path / "validation"
    val.mkdir()
    shutil.copy(TOY / "measurements.tsv", val / "measurements.tsv")
    _params([("A_init", "lin", 0.0, 100.0, 2.0, 0)]).reset_index().to_csv(
        val / V.HELD_OUT_PARAMETERS, sep="\t", index=False
    )

    mcp = create_server(
        str(prob / "problem.yaml"),
        provenance_dir=str(tmp_path / "prov"),
        build_root=str(tmp_path / "build"),
        auto_fit=False,
        validation_measurements=str(val / "measurements.tsv"),
    )
    tools = {t.name: t for t in asyncio.run(mcp.list_tools())}
    assert "n_starts" in tools["evaluate_validation_data"].parameters["properties"]


def test_read_held_out_reads_the_sets_units(tmp_path):
    """`units.tsv` beside the bundle: {experimentId: unitId}, the file whoever assembles the set
    writes and the server reads, so the two agree on the split through the file and nowhere else. A
    v1 table keyed by conditionId is refused like a v1 set."""
    _rows("e1").to_csv(tmp_path / "measurements.tsv", sep="\t", index=False)
    _params([("A_init", "lin", 0.0, 100.0, 2.0, 0)]).reset_index().to_csv(
        tmp_path / V.HELD_OUT_PARAMETERS, sep="\t", index=False
    )
    pd.DataFrame({"experimentId": ["e1", "e2"], "unitId": ["u1", "u2"]}).to_csv(
        tmp_path / V.HELD_OUT_UNITS, sep="\t", index=False
    )

    held = V.read_held_out(tmp_path / "measurements.tsv")

    assert held.mode == V.MODE_TRANSFER and held.units == {"e1": "u1", "e2": "u2"}
    pd.DataFrame({"conditionId": ["c1"], "unitId": ["u1"]}).to_csv(
        tmp_path / V.HELD_OUT_UNITS, sep="\t", index=False
    )
    with pytest.raises(ValueError, match="keyed by experimentId"):
        V.read_held_out(tmp_path / "measurements.tsv")


def test_a_row_whose_experiment_is_in_no_unit_is_declined_before_any_build(monkeypatch):
    _never_build(monkeypatch)
    problem = _candidate().replace(
        measurement_df=pd.concat([_rows("e0"), _rows("e1")]),
        experiment_df=_experiments(["e0", "e1"], ["c0", "c0"]),
    )

    out = V.refit_problem(
        problem, build_dir="/tmp/x", model_name="m", n_starts=1, units={"e0": "u0"}
    )

    assert out["status"] == V.STATUS_TABLE_MISMATCH and "e1" in out["reason"]


def test_score_session_refits_each_unit_on_its_own_and_sums_the_units(session, engine):
    """AMICI-backed: the toy's measurements in two experiments of the set's own, two units. Each
    unit is refit alone from the same start, the payload sums nll and n over them and carries each
    unit's block, and `k` is one refit's count — not the sum, which would price the same rows
    twice."""
    meas = session.problem.measurement_df
    held_meas = pd.concat(
        [
            meas.assign(experimentId="e1", measurement=meas.measurement * 2),
            meas.assign(experimentId="e2", measurement=meas.measurement * 3),
        ],
        ignore_index=True,
    )
    held = V.HeldOut(
        measurement_df=held_meas,
        experiment_df=_experiments(["e1", "e2"]),
        parameter_df=_params([("A_init", "lin", 0.0, 100.0, 2.0, 0)]),
        units={"e1": "u1", "e2": "u2"},
    )

    out = V.score_session(session, engine, held, n_starts=2)

    assert out["status"] == V.STATUS_OK, out
    assert out["mode"] == V.MODE_TRANSFER and set(out["units"]) == {"u1", "u2"}
    assert out["n"] == len(held_meas) == sum(u["n"] for u in out["units"].values())
    assert math.isclose(out["nll"], sum(u["nll"] for u in out["units"].values()))
    assert out["units"]["u1"]["k"] == 2 and out["k"] == 4, "k sums the units' own counts"
    assert all(u["n_experiments"] == 1 and "fitted" in u for u in out["units"].values())
    assert session.problem.experiment_df is None


@pytest.mark.parametrize("retry_steps", [V.RETRY_SOLVER["max_steps"], 1])
def test_a_refit_optimum_that_does_not_simulate_again_is_non_finite(
    session, engine, monkeypatch, retry_steps
):
    """AMICI-backed (#288): the solver is left one step once the multistart is done, so the unit's
    optimum, finite to the optimizer, fails when simulated again for its records. It is simulated
    again at `RETRY_SOLVER` (#323), which scores it; where that fails too it is no score:
    `non_finite`, naming the unit, the experiment and AMICI's status, never `ok` with `nll: null`.
    `tolerances` records both attempts either way."""
    import pypesto.optimize

    minimize = pypesto.optimize.minimize

    def _then_one_step(problem, **kwargs):
        result = minimize(problem, **kwargs)
        problem.objective.amici_solver.set_max_steps(1)
        return result

    monkeypatch.setattr(pypesto.optimize, "minimize", _then_one_step)
    monkeypatch.setattr(V, "RETRY_SOLVER", {**V.RETRY_SOLVER, "max_steps": retry_steps})
    meas = session.problem.measurement_df
    held = V.HeldOut(
        measurement_df=meas.assign(experimentId="e1", measurement=meas.measurement * 2),
        experiment_df=_experiments(["e1"]),
        parameter_df=_params([("A_init", "lin", 0.0, 100.0, 2.0, 0)]),
        units={"e1": "u1"},
    )

    out = V.score_session(session, engine, held, n_starts=2)

    if retry_steps > 1:
        assert out["status"] == V.STATUS_OK and math.isfinite(out["nll"]), out
        tolerances = out["units"]["u1"]["tolerances"]
    else:
        assert out["status"] == V.STATUS_NON_FINITE and out["unit"] == "u1", out
        assert "nll" not in out
        assert out["reason"].startswith("unit `u1`: the refit's optimum (nll "), out["reason"]
        assert "fails in `e1` (AMICI_TOO_MUCH_WORK at t=" in out["reason"], out["reason"]
        tolerances = out["tolerances"]
    assert len(tolerances) == 2 and tolerances[-1] == V.RETRY_SOLVER


def test_a_transfer_score_stores_each_units_records_at_its_own_optimum(session, engine):
    """AMICI-backed, on a committed version: the per-point records of a transfer score (#119) are
    each unit's, at THAT unit's refitted optimum, in the head version's directory. With sigma_B
    estimated, the noise is the refitted sigma, not the 0.1 the refit started from: the records
    read it off the parameter table, which holds the start until the optimum is written in.
    Checked against the toy's closed form at each unit's `fitted` values, A_init moved to 2."""
    import json

    import numpy as np

    session.problem.parameter_df.loc["sigma_B", "estimate"] = True
    session.provenance.snapshot_version(session.model.to_antimony(), session.problem.parameter_df)
    meas = session.problem.measurement_df
    held = V.HeldOut(
        measurement_df=pd.concat(
            [
                meas.assign(experimentId="e1", measurement=meas.measurement * 2),
                meas.assign(experimentId="e2", measurement=meas.measurement * 3),
            ],
            ignore_index=True,
        ),
        experiment_df=_experiments(["e1", "e2"]),
        parameter_df=_params([("A_init", "lin", 0.0, 100.0, 2.0, 0)]),
        units={"e1": "u1", "e2": "u2"},
    )

    out = V.score_session(session, engine, held, n_starts=2)

    assert out["status"] == V.STATUS_OK, out
    assert "records" not in out and all("records" not in u for u in out["units"].values())
    stored = json.loads(Path(out["records_file"]).read_text())
    assert Path(out["records_file"]).parent == session.provenance.versions_dir / "v0"
    recs = stored["records"]
    assert len(recs) == out["n"] and [r["experiment_id"] for r in recs] == ["e1"] * 6 + ["e2"] * 6
    sigmas = set()
    for r in recs:
        fitted = out["units"][held.units[r["experiment_id"]]]["fitted"]
        k1, k2, sigma = fitted["k1"], fitted["k2"], fitted["sigma_B"]
        sim = 2.0 * k1 / (k1 + k2) * (1 - np.exp(-(k1 + k2) * r["time"]))
        assert r["simulation"] == pytest.approx(sim, rel=1e-4, abs=1e-8)
        assert r["noise"] == pytest.approx(sigma)
        assert r["residual_weighted"] == pytest.approx((r["measurement"] - sim) / sigma, abs=1e-3)
        sigmas.add(round(sigma, 6))
    assert len(sigmas) == 2 and 0.1 not in sigmas, sigmas
    assert sum(r["residual_weighted"] ** 2 for r in recs) == pytest.approx(out["chi2"])


def test_a_single_named_unit_is_still_carried_under_units(monkeypatch):
    """`units` names the systems; naming one is a declaration, not the absence of one, so its
    block is carried the same way — a consumer reading `units` must not find it missing."""
    seen = []

    def _fake_one(view, *, target, n_starts, fit_engine, cap):
        seen.append(view)
        return {
            "status": "ok",
            "mode": "transfer",
            "nll": 1.0,
            "k": 1,
            "n": 2,
            "chi2": 2.0,
            "n_experiments": 1,
            "observables": ["obs_B"],
            "n_starts": n_starts,
            "n_converged": 1,
            "fitted": {},
            "records": [],
        }

    monkeypatch.setattr(V, "_refit_one", _fake_one)

    out = V.refit_problem(
        _candidate(), build_dir="/tmp/x", model_name="m", n_starts=1, units={"e0": "only"}
    )

    assert list(out["units"]) == ["only"] and out["nll"] == 1.0 and out["k"] == 1
    assert len(seen) == 1


def _fake_refit(monkeypatch):
    """Replace one system's refit: record the view it was handed, report a fixed optimum."""
    seen = []

    def _fake_one(view, *, target, n_starts, fit_engine, cap):
        seen.append(view)
        return {
            "status": "ok",
            "mode": "transfer",
            "nll": 1.0,
            "k": 1,
            "n": len(view.measurement_df),
            "chi2": 2.0,
            "n_experiments": 1,
            "observables": ["obs_B"],
            "n_starts": n_starts,
            "n_converged": 1,
            "fitted": {},
            "records": [],
        }

    monkeypatch.setattr(V, "_refit_one", _fake_one)
    return seen


def test_a_v1_transfer_set_is_refused_before_any_build(monkeypatch):
    """The transfer side of the v1 refusal in ``test_validation_data.py``: the view is not built,
    and the status says how to convert the set."""
    _never_build(monkeypatch)
    held = V.HeldOut(
        measurement_df=_rows("c1").rename(columns={"experimentId": "simulationConditionId"}),
        parameter_df=_HELD_OUT_PARAMS,
    )

    view, failed = V.transfer_view(_candidate(), held, _SHIPPED)

    assert view is None and failed["status"] == V.STATUS_V2_REQUIRED
    assert "petab1to2" in failed["reason"]


def _write_v2_parameters(path: Path, rows, *, scale: bool) -> None:
    """A v2 parameter table: ``true``/``false`` estimates, and libpetab v1's ``parameterScale``
    column only if ``scale``."""
    table = _params(rows).reset_index()
    table["estimate"] = table["estimate"].map({1: "true", 0: "false"})
    if not scale:
        table = table.drop(columns="parameterScale")
    table.to_csv(path, sep="\t", index=False)


def test_a_v2_parameter_row_without_a_scale_takes_the_training_rows(tmp_path):
    """v2 has no ``parameterScale``; a v2 table without the column read every row as ``lin``, so a
    refit on it estimated a log10 rate linearly. A row without a scale takes the training
    row's; one the training table lacks is refused, and without a training table the cell is left
    blank rather than guessed. A scale the table states is kept."""
    meas = tmp_path / "measurements.tsv"
    _rows("e1").to_csv(meas, sep="\t", index=False)
    rows = [("k1", "", 1e-4, 100.0, 7.0, 1), ("A_init", "", 0.0, 100.0, 2.0, 0)]
    _write_v2_parameters(tmp_path / V.HELD_OUT_PARAMETERS, rows, scale=False)

    held = V.read_held_out(meas, training_parameter_df=_SHIPPED)
    assert held.parameter_df["parameterScale"].to_dict() == {"k1": "log10", "A_init": "lin"}
    assert held.parameter_df["estimate"].tolist() == [True, False]
    assert V.read_held_out(meas).parameter_df["parameterScale"].isna().all()

    _write_v2_parameters(
        tmp_path / V.HELD_OUT_PARAMETERS, [("k1", "lin", 1e-4, 100.0, 7.0, 1)], scale=True
    )
    held = V.read_held_out(meas, training_parameter_df=_SHIPPED)
    assert held.parameter_df.loc["k1", "parameterScale"] == "lin"

    rows.append(("sigma_extra", "", 1e-4, 100.0, 3.0, 0))
    _write_v2_parameters(tmp_path / V.HELD_OUT_PARAMETERS, rows, scale=False)
    with pytest.raises(ValueError, match="sigma_extra"):
        V.read_held_out(meas, training_parameter_df=_SHIPPED)


def test_a_native_v2_transfer_set_is_read_from_disk_and_refit_per_unit(tmp_path, monkeypatch):
    """A transfer set as PEtab v2 writes it -- measurements, experiments, long-format
    conditions, a parameter table without ``parameterScale`` and units, all by experimentId --
    read through ``read_held_out`` and refit one unit at a time on
    the set's own design, with the changed fixed input carried into the candidate."""
    seen = _fake_refit(monkeypatch)
    meas = tmp_path / "measurements.tsv"
    pd.concat([_rows("e1"), _rows("e2")]).to_csv(meas, sep="\t", index=False)
    _conditions(["c1", "c2"]).to_csv(tmp_path / V.HELD_OUT_CONDITIONS, sep="\t", index=False)
    _experiments(["e1", "e2"], ["c1", "c2"]).to_csv(
        tmp_path / V.HELD_OUT_EXPERIMENTS, sep="\t", index=False
    )
    _write_v2_parameters(
        tmp_path / V.HELD_OUT_PARAMETERS, [("A_init", "", 0.0, 100.0, 2.0, 0)], scale=False
    )
    pd.DataFrame({"experimentId": ["e1", "e2"], "unitId": ["u1", "u2"]}).to_csv(
        tmp_path / V.HELD_OUT_UNITS, sep="\t", index=False
    )

    held = V.read_held_out(meas, training_parameter_df=_SHIPPED)

    assert held.mode == V.MODE_TRANSFER and held.units == {"e1": "u1", "e2": "u2"}
    assert held.condition_df[["conditionId", "targetId"]].values.tolist() == [
        ["c1", "dose"],
        ["c2", "dose"],
    ]
    assert held.parameter_df.loc["A_init", "parameterScale"] == "lin"
    view, failed = V.transfer_view(_candidate(), held, _SHIPPED)
    assert failed is None and view.parameter_df.loc["A_init", "nominalValue"] == 2.0
    out = V.refit_problem(view, build_dir="/tmp/x", model_name="m", n_starts=1, units=held.units)
    assert out["status"] == V.STATUS_OK and set(out["units"]) == {"u1", "u2"}
    assert [sorted(set(v.experiment_df["experimentId"])) for v in seen] == [["e1"], ["e2"]]


def test_read_units_refuses_an_experiment_in_two_units_or_a_blank(tmp_path):
    (tmp_path / "u.tsv").write_text("experimentId\tunitId\ne1\tu1\ne1\tu2\n")
    with pytest.raises(ValueError, match="e1"):
        V.read_units(tmp_path / "u.tsv")
    (tmp_path / "u.tsv").write_text("experimentId\tunitId\ne1\t\n")
    with pytest.raises(ValueError, match="unitId"):
        V.read_units(tmp_path / "u.tsv")


def test_a_refit_sees_only_the_observables_its_rows_measure():
    """The per-refit view keeps the observables the rows measure and nothing else — the sigma of
    an unmeasured observable is not a parameter of that refit (pruned by petab's lint where a model
    is present)."""
    problem = _candidate()
    problem = problem.replace(
        observable_df=pd.concat(
            [
                problem.observable_df,
                pd.DataFrame(
                    {"observableFormula": ["A"], "noiseFormula": ["1"]},
                    index=pd.Index(["obs_A"], name="observableId"),
                ).reindex(columns=problem.observable_df.columns),
            ]
        )
    )

    uview = V._unit_view(problem, _rows("e0"))

    assert list(uview.observable_df.index) == ["obs_B"]
    assert list(problem.observable_df.index) == ["obs_B", "obs_A"], "the caller's view is untouched"
