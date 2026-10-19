"""The fixed-θ HELD-OUT check: :func:`petab_mcp.validation_data.score_problem` and its guards.

Tested at the boundary rather than through an end-to-end session: which tables the scorer copies,
what it declines before paying for a build, and how it refuses a parameter vector it cannot line
up. The objective evaluation past those guards needs an AMICI build; its correctness is pinned by
re-deriving a reference structure's own held-out score through the same functions. Assembling a
held-out set, and what the caller does with the score, are the caller's to test.

Held-out sets in v1 form (`validation_data.read_held_out` + `as_v2`): held-out files written for
PEtab v1 predate the v2 server, so they are keyed onto experiments the way the problem's own
upgrade keys its training measurements.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from petab_mcp import validation_data as V
from petab_mcp.problem import WorkingProblem

DATA = Path(__file__).parent / "data"


def _write(path: Path, rows: list[list]) -> None:
    path.write_text("".join("\t".join(map(str, r)) + "\n" for r in rows))


def test_a_v1_held_out_set_lands_in_the_experiments_the_upgrade_makes(tmp_path):
    """A pair whose conditions change something (the set's own or the training problem's) becomes
    the experiment libpetab's ``petab1to2`` names for it; a pair that changes nothing simulates the
    model's defaults, keyed by ``""`` however many such pairs there are."""
    m = tmp_path / "measurements.tsv"
    _write(
        m,
        [
            [
                "observableId",
                "preequilibrationConditionId",
                "simulationConditionId",
                "time",
                "measurement",
            ],
            ["obs_B", "", "c_new", 1.0, 0.3],
            ["obs_B", "", "c_dose", 2.0, 0.4],
            ["obs_B", "c_dose", "c_new", 2.5, 0.45],
            ["obs_B", "", "basal", 3.0, 0.5],
            ["obs_B", "", "other_basal", 4.0, 0.6],
        ],
    )
    _write(tmp_path / "conditions.tsv", [["conditionId", "A_init"], ["c_new", 3.0]])
    _write(tmp_path / "units.tsv", [["conditionId", "unitId"], ["c_new", "u1"], ["c_dose", "u2"]])
    held_out = V.read_held_out(m)
    assert held_out.condition_df[["conditionId", "targetId"]].values.tolist() == [
        ["c_new", "A_init"]
    ]

    training = WorkingProblem.from_yaml(DATA / "toy_v2" / "problem.yaml")
    training.condition_df = pd.DataFrame(
        {"conditionId": ["c_dose"], "targetId": ["A_init"], "targetValue": [2.0]}
    )
    upgraded = V.as_v2(held_out, training)
    new, dose, pre = "experiment____c_new", "experiment____c_dose", "experiment__c_dose___c_new"
    assert upgraded.measurement_df["experimentId"].tolist() == [new, dose, pre, "", ""]
    edf = upgraded.experiment_df
    assert edf[edf["experimentId"] == pre]["conditionId"].tolist() == ["c_dose", "c_new"]
    assert sorted(edf["experimentId"].unique()) == sorted([new, dose, pre])
    assert upgraded.units == {new: "u1", dose: "u2", pre: "u1"}
    assert V.as_v2(upgraded, training) is upgraded  # already v2


def test_a_v1_held_out_row_on_a_training_design_lands_in_the_training_experiment(tmp_path):
    """The training problem is upgraded by ``petab1to2``; a held-out row on one of its condition
    pairs must be keyed to the very experiment that made, or the scorer would add a second one."""
    import shutil

    work = tmp_path / "toy"
    shutil.copytree(DATA / "toy_v1", work)
    _write(work / "conditions.tsv", [["conditionId", "A"], ["c0", 1.0]])
    held = pd.read_csv(work / "measurements.tsv", sep="\t")
    held["preequilibrationConditionId"] = ["c0"] * 3 + [""] * 3
    held.to_csv(work / "measurements.tsv", sep="\t", index=False)
    training = WorkingProblem.from_yaml(work / "problem.yaml")
    trained = training.measurement_df["experimentId"].tolist()
    assert trained == ["experiment__c0___c0"] * 3 + ["experiment____c0"] * 3

    upgraded = V.as_v2(V.HeldOut(held), training)

    assert upgraded.measurement_df["experimentId"].tolist() == trained
    assert V._with_held_out_design(training.replace(), None, upgraded.experiment_df) is None


def test_a_curated_scaling_is_solved_for_a_held_out_experiment(tmp_path):
    """The fixed-θ scorer end to end: a new experiment whose blot names its own scale. The view
    keeps the training experiment with no rows left in it, and writes the solved scale into its
    measurement table before the final evaluation. Checked against the toy's closed form."""
    import shutil

    import numpy as np

    work = tmp_path / "toy"
    shutil.copytree(DATA / "toy_v2", work)
    _write(
        work / "observables.tsv",
        [
            ["observableId", "observableFormula", "noiseFormula", "observablePlaceholders"],
            ["obs_B", "observableParameter1_obs_B * B", "sigma_B", "observableParameter1_obs_B"],
        ],
    )
    _write(work / "conditions.tsv", [["conditionId", "targetId", "targetValue"], ["c0", "A", 1.0]])
    _write(work / "experiments.tsv", [["experimentId", "time", "conditionId"], ["e0", 0, "c0"]])
    m = pd.read_csv(work / "measurements.tsv", sep="\t", keep_default_na=False)
    m.assign(experimentId="e0", observableParameters="1.0").to_csv(
        work / "measurements.tsv", sep="\t", index=False
    )
    yaml = work / "problem.yaml"
    yaml.write_text(
        yaml.read_text()
        + "condition_files:\n- conditions.tsv\nexperiment_files:\n- experiments.tsv\n"
    )
    training = WorkingProblem.from_yaml(yaml)

    held = training.measurement_df.assign(
        experimentId="e_ho",
        observableParameters="s_blot",
        measurement=2 * training.measurement_df["measurement"],
    )
    out = V.score_problem(
        training,
        held,
        build_dir=str(tmp_path / "build"),
        model_name="toy",
        condition_df=pd.DataFrame(
            {"conditionId": ["c_ho"], "targetId": ["A"], "targetValue": [1.0]}
        ),
        experiment_df=pd.DataFrame(
            {"experimentId": ["e_ho"], "time": [0.0], "conditionId": ["c_ho"]}
        ),
    )
    assert out["status"] == "ok" and out["chi2"] is not None, out
    k1, k2 = 0.5, 0.1  # the nominal point; A(0) = 1, B(0) = 0
    b = k1 / (k1 + k2) * (1 - np.exp(-(k1 + k2) * held["time"].to_numpy(float)))
    y = held["measurement"].to_numpy(float)
    assert out["held_out_scalings"]["s_blot"] == pytest.approx(float(y @ b / (b @ b)), rel=1e-4)


def _problem(measurements, *, observables=("obs_A",), experiments=("e_train", "e_test")):
    """A minimal problem with just the tables the guards read: each experiment applies its own
    condition, which sets the model input ``dose``."""
    conditions = [f"c_{e.removeprefix('e_')}" for e in experiments]
    return WorkingProblem(
        measurement_df=measurements,
        observable_df=pd.DataFrame(
            {
                "observableFormula": ["x"] * len(observables),
                "noiseFormula": ["1"] * len(observables),
            },
            index=pd.Index(list(observables), name="observableId"),
        ),
        condition_df=pd.DataFrame(
            {"conditionId": conditions, "targetId": "dose", "targetValue": 0.0}
        ),
        experiment_df=pd.DataFrame(
            {"experimentId": list(experiments), "time": 0.0, "conditionId": conditions}
        ),
        parameter_df=pd.DataFrame(
            {
                "parameterScale": ["lin"],
                "lowerBound": [0.0],
                "upperBound": [1.0],
                "nominalValue": [0.5],
                "estimate": [True],
            },
            index=pd.Index(["k"], name="parameterId"),
        ),
    )


def _rows(experiment, times=(0.0, 1.0), observable="obs_A"):
    return pd.DataFrame(
        {
            "observableId": [observable] * len(times),
            "experimentId": [experiment] * len(times),
            "time": list(times),
            "measurement": [1.0] * len(times),
        }
    )


def _new_design(target="dose", value=0.5, condition="c_knockdown", experiment="e_knockdown"):
    """A fixed-θ set's own condition and the experiment that applies it."""
    return (
        pd.DataFrame({"conditionId": [condition], "targetId": [target], "targetValue": [value]}),
        pd.DataFrame({"experimentId": [experiment], "time": [0.0], "conditionId": [condition]}),
    )


class _Stop(RuntimeError):
    pass


def _intercept_build(monkeypatch, build=None):
    """Replace the scorer's pyPESTO build: record ``(view, importer kwargs)`` per call and hand
    back ``build(view, kwargs)``, or stop the scorer right there without one."""
    import petab_mcp.fitting as fitting

    seen = []

    def _build(view, kwargs, **_):
        seen.append((view, kwargs))
        if build is None:
            raise _Stop
        return build(view, kwargs)

    monkeypatch.setattr(fitting, "pypesto_problem", _build)
    monkeypatch.setattr(fitting, "_apply_amici_solver_defaults", lambda pp: None)
    return seen


def _never_build(monkeypatch):
    import petab_mcp.fitting as fitting

    def _boom(*a, **k):  # pragma: no cover - must not be reached
        raise AssertionError("nothing must be built for a declined table")

    monkeypatch.setattr(fitting, "pypesto_problem", _boom)


def test_scoring_never_mutates_the_problem_it_was_handed(monkeypatch):
    """``score_session`` runs against the LIVE session problem, so a table it swapped in place would
    change what the agent reads and what the caller scores on the training data — the held-out
    check would corrupt the very session it is reporting on. The scorer builds a view with its own
    dataframe objects; this pins that, without paying for an AMICI build, by intercepting the
    build and inspecting the view it was handed.
    """
    seen = _intercept_build(monkeypatch)
    problem = _problem(_rows("e_train"))
    original_measurements = problem.measurement_df

    with pytest.raises(_Stop):
        V.score_problem(problem, _rows("e_test"), build_dir="/tmp/x", model_name="m")

    view = seen[0][0]
    assert view is not problem
    assert list(view.measurement_df["experimentId"]) == ["e_test", "e_test"]
    assert view.experiment_df is not problem.experiment_df
    # ...and the caller's own problem is exactly as it was
    assert problem.measurement_df is original_measurements
    assert list(problem.measurement_df["experimentId"]) == ["e_train", "e_train"]


def test_the_held_out_view_is_fixed_at_the_point_and_built_beside_the_training_module(
    monkeypatch,
):
    """The view handed to the build fixes every parameter at the table's point. AMICI compiles a
    NON-estimated parameter as a model constant, so that split is not the training problem's, and
    the view must not land on the training build target: a module compiled for one split is wrong
    for another."""
    seen = _intercept_build(monkeypatch)
    problem = _problem(_rows("e_train"))

    with pytest.raises(_Stop):
        V.score_problem(
            problem,
            _rows("e_train", times=(2.0, 3.0)),
            build_dir="/tmp/train_build",
            model_name="train_module",
        )

    view, kwargs = seen[0]
    assert not view.parameter_df["estimate"].any()
    assert view.parameter_df["nominalValue"].to_dict() == {"k": 0.5}
    assert problem.parameter_df["estimate"].all(), "the caller's table keeps its split"
    assert kwargs["output_folder"] != "/tmp/train_build"


def test_an_empty_or_foreign_held_out_table_is_declined_before_any_build(monkeypatch):
    """The guards run BEFORE the build, so a bad table costs nothing and names its own problem
    rather than surfacing from inside AMICI's parameter mapping."""
    _never_build(monkeypatch)
    problem = _problem(_rows("e_train"))

    empty = V.score_problem(problem, _rows("e_test").iloc[0:0], build_dir="/tmp/x", model_name="m")
    assert empty["status"] == V.STATUS_NO_DATA

    foreign = V.score_problem(problem, _rows("e_elsewhere"), build_dir="/tmp/x", model_name="m")
    assert foreign["status"] == V.STATUS_TABLE_MISMATCH
    assert "e_elsewhere" in foreign["reason"]

    unknown_obs = V.score_problem(
        problem, _rows("e_test", observable="obs_Z"), build_dir="/tmp/x", model_name="m"
    )
    assert unknown_obs["status"] == V.STATUS_TABLE_MISMATCH
    assert "obs_Z" in unknown_obs["reason"]


def test_the_server_only_registers_the_tool_when_given_held_out_data(
    monkeypatch, yaml_path, tmp_path
):
    """No held-out file, no tool — so a problem without a validation set cannot accidentally be
    scored against a stale or unrelated table, and the caller's "unavailable" status is reached by
    the tool genuinely not existing.

    Asserted through the registrar actually being CALLED, not by grepping ``create_server``'s
    source: a reformat must not fail this, and an equivalent rewrite must not pass it while
    dropping the gate."""
    import inspect

    from petab_mcp import server as server_mod
    from petab_mcp.server import create_server

    calls = []
    monkeypatch.setattr(
        server_mod,
        "_register_validation_data_tools",
        lambda mcp, route: calls.append(route().held_out),
    )
    assert "validation_measurements" in inspect.signature(create_server).parameters

    create_server(
        yaml_path, build_root=str(tmp_path / "b1"), provenance_dir=str(tmp_path / "prov1")
    )
    assert calls == []

    meas = tmp_path / "held.tsv"
    _rows("e_test").to_csv(meas, sep="\t", index=False)
    create_server(
        yaml_path,
        build_root=str(tmp_path / "b2"),
        provenance_dir=str(tmp_path / "prov2"),
        validation_measurements=str(meas),
    )
    assert len(calls) == 1
    assert isinstance(calls[0], V.HeldOut)
    assert len(calls[0].measurement_df) == 2


def test_an_explicit_point_must_come_with_matching_names():
    """A caller-supplied vector is in ITS problem's x_names order. The two problems are built from
    the same parameter table so they agree today, but a silent re-ordering would evaluate a permuted
    vector and report a plausible, wrong likelihood — so the names are checked, not assumed."""
    problem = _problem(_rows("e_train"))
    with pytest.raises(ValueError, match="both x_full and x_names, or neither"):
        V.score_problem(problem, _rows("e_test"), build_dir="/tmp/x", model_name="m", x_full=[0.5])


# Each of these exists because its ABSENCE was a silent-wrong-number risk, so each needs a test
# that fails if it is relaxed back. (The repo's own lesson: 15 defects survived a green suite
# because the plumbing around the statistic went untested — this IS the plumbing.)
def test_a_short_x_full_is_refused_rather_than_evaluated_misaligned(monkeypatch):
    """The name check compares `x_names` against the problem's; it says NOTHING about how many
    values arrived with them. A shorter vector whose free indices all happen to be in range would
    otherwise be silently indexed and reported as a plausible held-out likelihood for a parameter
    point nobody chose."""
    _never_build(monkeypatch)
    problem = _problem(_rows("e_train"))

    with pytest.raises(ValueError, match="must be the FULL vector"):
        V.score_problem(
            problem,
            _rows("e_test"),
            build_dir="/tmp/x",
            model_name="m",
            x_full=[0.5],  # one value for three parameters
            x_names=["k", "sigma", "scale"],
        )  # ...but the NAMES match, so only length catches it


def test_a_view_with_fewer_parameter_rows_is_served_not_refused():
    """The caller fitted the TRAINING problem; the held-out view can carry fewer parameter rows
    than that (petab refuses rows nothing references, and swapping a curated held-out table in
    retires the overrides the training rows carried). `pp.x_names` is then a strict subset of the
    caller's, which list equality would refuse outright — so it is matched by name."""

    class _PP:
        x_names = ["k", "sigma"]  # the view dropped the training scalings
        x_free_indices = [0, 1]

    reduced = V._reduced_from_named(_PP(), [1.0, 2.0, 3.0], ["k", "sigma", "s_blot_retired"])
    assert reduced == [1.0, 2.0]  # in pp.x_names order, the extra name ignored


def test_a_name_the_view_needs_and_the_caller_omitted_is_refused():
    """The other direction is still a hard error: a vector that does not cover what the problem
    needs would be evaluated permuted and report a plausible, wrong likelihood."""

    class _PP:
        x_names = ["k", "sigma", "scale"]
        x_free_indices = [0, 1, 2]

    with pytest.raises(ValueError, match="do not match the held-out problem's"):
        V._reduced_from_named(_PP(), [1.0, 2.0, 3.0], ["k", "sigma", "unrelated"])


def test_arity_is_reported_as_arity_not_as_a_name_mismatch():
    """A vector already reduced to free coordinates is a LENGTH mistake. Zipping it against the
    full name list would truncate silently and then blame the names."""

    class _PP:
        x_names = ["k", "sigma", "scale"]
        x_free_indices = [0]

    with pytest.raises(ValueError, match="must be the FULL vector"):
        V._reduced_from_named(_PP(), [0.5], ["k", "sigma", "scale"])


class _Objective:
    def __call__(self, x, **kwargs):
        return {"fval": 1.0, "rdatas": []}


class _PP:
    x_names = ["k"]
    x_free_indices = [0]
    x_scales = ["lin"]
    objective = _Objective()


def test_score_problem_adds_a_fixed_sets_design_before_building(monkeypatch):
    """The scorer's view carries the set's own condition and experiment, so a row on a new
    experiment is a table the build can take, not a mismatch. The training ones, which no held-out
    row uses, are dropped (libpetab would warn of an unused condition), and the caller's problem
    keeps its tables."""
    seen = _intercept_build(monkeypatch, lambda view, kwargs: _PP())
    problem = _problem(_rows("e_train"))
    conditions, experiments = _new_design()

    out = V.score_problem(
        problem,
        _rows("e_knockdown"),
        build_dir="/tmp/x",
        model_name="m",
        condition_df=conditions,
        experiment_df=experiments,
    )

    assert out["status"] == V.STATUS_OK and out["n_experiments"] == 1
    view = seen[0][0]
    assert list(view.condition_df["conditionId"]) == ["c_knockdown"]
    assert list(view.experiment_df["experimentId"]) == ["e_knockdown"]
    assert list(problem.condition_df["conditionId"]) == ["c_train", "c_test"]
    assert list(problem.experiment_df["experimentId"]) == ["e_train", "e_test"]


def test_a_fixed_sets_design_is_checked_before_any_build(monkeypatch):
    """Same contract as for a foreign measurement table: the guards run before the build."""
    _never_build(monkeypatch)
    problem = _problem(_rows("e_train"))

    conditions, experiments = _new_design(target="temperature")
    out = V.score_problem(
        problem,
        _rows("e_knockdown"),
        build_dir="/tmp/x",
        model_name="m",
        condition_df=conditions,
        experiment_df=experiments,
    )
    assert out["status"] == V.STATUS_TABLE_MISMATCH
    assert "different models" in out["reason"]

    redefined, _ = _new_design(condition="c_train")
    out = V.score_problem(
        problem, _rows("e_test"), build_dir="/tmp/x", model_name="m", condition_df=redefined
    )
    assert out["status"] == V.STATUS_TABLE_MISMATCH
    assert "does not redefine" in out["reason"]

    moved = pd.DataFrame({"experimentId": ["e_train"], "time": [5.0], "conditionId": ["c_train"]})
    out = V.score_problem(
        problem, _rows("e_train"), build_dir="/tmp/x", model_name="m", experiment_df=moved
    )
    assert out["status"] == V.STATUS_TABLE_MISMATCH
    assert "different periods" in out["reason"]


def test_score_session_hands_a_fixed_sets_design_to_the_scorer(monkeypatch):
    """`score_session` is the server's one entry point; a fixed set's own design reaches the
    scorer through it, or the tool reports `table_mismatch` for every curated set that brings any."""
    import contextlib
    import types

    seen = {}

    def _score(problem, measurement_df, **kwargs):
        seen.update(kwargs)
        return {"status": "ok"}

    monkeypatch.setattr(V, "score_problem", _score)
    session = types.SimpleNamespace(
        lock=contextlib.nullcontext(), problem=object(), module_hash=lambda: "h"
    )
    engine = types.SimpleNamespace(
        pypesto_importer_kwargs=lambda h: {"output_folder": "/tmp/x", "model_name": "m"}
    )
    conditions, experiments = _new_design()

    V.score_session(
        session,
        engine,
        V.HeldOut(_rows("e_knockdown"), condition_df=conditions, experiment_df=experiments),
    )

    assert seen["condition_df"] is conditions and seen["experiment_df"] is experiments


def test_read_held_out_gives_a_fixed_set_its_condition_rows(tmp_path):
    """On disk: measurements.tsv + conditions.tsv and NO parameters.tsv is a fixed-θ set with rows
    of its own — what a caller writes for such a set and what the server reads back, a v1
    condition table upgraded to v2's changes."""
    _rows("e_knockdown").rename(columns={"experimentId": "simulationConditionId"}).to_csv(
        tmp_path / "measurements.tsv", sep="\t", index=False
    )
    pd.DataFrame({"conditionId": ["c_knockdown"], "dose": [0.5]}).to_csv(
        tmp_path / V.HELD_OUT_CONDITIONS, sep="\t", index=False
    )

    held = V.read_held_out(tmp_path / "measurements.tsv")

    assert held.mode == V.MODE_FIXED
    assert held.condition_df.values.tolist() == [["c_knockdown", "dose", 0.5]]


def test_score_session_scores_the_toy_at_its_fitted_point_on_an_experiment_of_the_sets_own(
    session, engine
):
    """The one AMICI-backed check of the fixed-θ side through the session: the toy's measurements
    moved onto an experiment `e1` the training problem does not have, brought by the set. Scored
    at the training point, not refit — `mode: fixed`, no `k` — and the session's own tables are
    left as they were."""
    import math

    training = session.problem.measurement_df
    held = V.HeldOut(
        measurement_df=training.assign(experimentId="e1"),
        experiment_df=pd.DataFrame({"experimentId": ["e1"], "time": [0.0], "conditionId": [None]}),
    )

    out = V.score_session(session, engine, held)

    assert out["status"] == V.STATUS_OK, out
    assert out["mode"] == V.MODE_FIXED and "k" not in out
    assert out["n"] == len(training) and out["n_experiments"] == 1
    assert math.isfinite(out["nll"])
    assert session.problem.experiment_df is None
    assert session.problem.measurement_df is training


def _scaled_problem(formula="observableParameter1_obs_A * x"):
    """A problem whose one observable carries a scaling placeholder, like every blot readout."""
    problem = _problem(_rows("e_train").assign(observableParameters="s_train"))
    return problem.replace(
        observable_df=pd.DataFrame(
            {
                "observableFormula": [formula],
                "noiseFormula": ["1"],
                "observablePlaceholders": ["observableParameter1_obs_A"],
            },
            index=pd.Index(["obs_A"], name="observableId"),
        ),
        parameter_df=pd.DataFrame(
            {
                "parameterScale": ["lin", "lin"],
                "lowerBound": [0.0, 0.0],
                "upperBound": [1.0, 10.0],
                "nominalValue": [0.5, 1.0],
                "estimate": [True, True],
            },
            index=pd.Index(["k", "s_train"], name="parameterId"),
        ),
    )


def test_held_out_scalings_are_the_overrides_the_problem_does_not_define():
    """A held-out blot's scale factor is whatever override the parameter table lacks; a shipped
    scaling (``s_train`` here) or a number is an ordinary PEtab override and not touched."""
    held = pd.concat(
        [
            _rows("e_test").assign(observableParameters="s_blot"),
            _rows("e_test", times=(2.0,)).assign(observableParameters="s_train"),
            _rows("e_test", times=(3.0,)).assign(observableParameters="2.5"),
        ]
    )
    assert V.held_out_scalings(_scaled_problem(), held) == {"s_blot"}


def test_held_out_scalings_are_solved_hierarchically_then_written_in_as_numbers(monkeypatch):
    """The two-pass shape, pinned without an AMICI build: pass one hands pyPESTO a HIERARCHICAL
    problem whose only estimated rows the model does not already have are the held-out scalings,
    tagged `parameterType: scaling` (the model's own parameters keep their values); pass two is
    the ordinary joint objective with the solved values substituted as plain numbers, so the
    reported nll/chi2 come off the same code path as for a set without held-out scalings."""

    class _Calc:
        def get_interpretable_inner_par_ids(self):
            return ["s_blot"]

    class _HierObjective:
        calculator = _Calc()

        def __call__(self, x, **kwargs):
            return {"fval": 1.0, "inner_parameters": [3.5], "rdatas": []}

    class _HierPP(_PP):
        objective = _HierObjective()

    seen = _intercept_build(
        monkeypatch, lambda view, kwargs: _HierPP() if kwargs.get("hierarchical") else _PP()
    )

    held = _rows("e_test").assign(observableParameters="s_blot")
    out = V.score_problem(_scaled_problem(), held, build_dir="/tmp/x", model_name="m")

    assert [bool(k.get("hierarchical")) for _, k in seen] == [True, False], (
        "hierarchical solve first, joint evaluation second"
    )
    hier_view, joint_view = seen[0][0], seen[1][0]
    row = hier_view.parameter_df.loc["s_blot"]
    assert row["parameterType"] == "scaling" and row["estimate"]
    assert float(row["lowerBound"]) == 0.0, "a gel's scale factor is non-negative"
    assert list(joint_view.measurement_df["observableParameters"]) == ["3.5", "3.5"]
    assert out["status"] == V.STATUS_OK
    assert out["held_out_scalings"] == {"s_blot": 3.5}


def test_a_callers_vector_is_matched_by_name_so_a_smaller_held_out_problem_still_scores(
    monkeypatch,
):
    """The held-out view drops the training-only overrides and its hierarchical twin drops the inner
    scalings, so the problem pyPESTO builds has FEWER names than the reference fit's vector. Matching
    by name — not position — is what lets `_reference_validation` hand over its full vector."""
    assert V._reduced_from_named(_PP(), [0.5, 2.0], ["s_train", "k"]) == [2.0]
    with pytest.raises(ValueError, match="do not match"):
        V._reduced_from_named(_PP(), [0.5], ["s_train"])
    # ...and a caller whose vector is shorter than its own names is caught before any build
    _never_build(monkeypatch)
    with pytest.raises(ValueError, match="must be the FULL vector"):
        V.score_problem(
            _scaled_problem(),
            _rows("e_test"),
            build_dir="/tmp/x",
            model_name="m",
            x_full=[0.5],
            x_names=["k", "s_train"],
        )


def test_the_extraneous_prune_is_petabs_lint_not_ours():
    """Why `_without_extraneous` exists at all. Swapping a held-out table in retires the overrides
    the TRAINING rows referenced, and petab's own parameter-table lint — which the build runs,
    hierarchical or not — refuses the rows left behind. The hierarchical solver cannot help: the
    lint fires before there is an objective to solve."""
    from petab.v2.lint import CheckAllParametersPresentInParameterTable
    from petab.v2.models.sbml_model import SbmlModel

    lint = CheckAllParametersPresentInParameterTable()
    view = _scaled_problem().replace(
        measurement_df=_rows("e_test").assign(observableParameters="s_blot")
    )
    view.model = SbmlModel.from_antimony(
        "model m\n  species x = 1;\n  k = 0.5;\n  dose = 0;\n  r: x => ; k * x;\nend",
        model_id="m",
    )
    view.parameter_df = pd.concat(
        [
            view.parameter_df,
            pd.DataFrame(
                {
                    "parameterScale": ["lin"],
                    "lowerBound": [0.0],
                    "upperBound": [1e12],
                    "nominalValue": [1.0],
                    "estimate": [True],
                },
                index=pd.Index(["s_blot"], name="parameterId"),
            ),
        ]
    )

    issue = lint.run(view.to_petab())
    assert issue is not None and "Extraneous" in issue.message and "s_train" in issue.message

    kept = V._without_extraneous(view)
    assert "s_train" not in kept.index, "the retired training override is dropped"
    assert {"k", "s_blot"} <= set(kept.index), "the kinetics and the held-out scaling stay"
    view.parameter_df = kept
    assert lint.run(view.to_petab()) is None


def test_a_held_out_set_using_part_of_the_observation_model_scores(tmp_path, monkeypatch):
    """A held-out set may reference the scale and noise of ONE of several measurement days. The
    view kept every training day's rows and the training-only conditions, and libpetab refused the
    rows nothing referenced: ``ValueError: Invalid PEtab v2 problem``. Here the training rows split
    over two days, one experiment each, and the held-out ones use day 2's; the view must lint
    clean and score at day 2's values, at the table's point and at a caller's ``x_full``. Checked
    against the toy's closed form."""
    import math
    import shutil

    import numpy as np

    from petab_mcp import fitting

    work = tmp_path / "toy"
    shutil.copytree(DATA / "toy_v2", work)
    _write(
        work / "observables.tsv",
        [
            ["observableId", "observableFormula", "noiseFormula"]
            + ["observablePlaceholders", "noisePlaceholders"],
            ["obs_B", "observableParameter1_obs_B * B", "noiseParameter1_obs_B"]
            + ["observableParameter1_obs_B", "noiseParameter1_obs_B"],
        ],
    )
    _write(work / "conditions.tsv", [["conditionId", "targetId", "targetValue"], ["c0", "A", 1.0]])
    _write(
        work / "experiments.tsv",
        [["experimentId", "time", "conditionId"], ["e_d1", 0, "c0"], ["e_d2", 0, "c0"]],
    )
    m = pd.read_csv(work / "measurements.tsv", sep="\t", keep_default_na=False)
    day = pd.Series(["d1"] * 3 + ["d2"] * (len(m) - 3))
    m.assign(
        experimentId="e_" + day, observableParameters="s_" + day, noiseParameters="sd_" + day
    ).to_csv(work / "measurements.tsv", sep="\t", index=False)
    _write(
        work / "parameters.tsv",
        [
            "parameterId parameterScale lowerBound upperBound nominalValue estimate".split(),
            ["k1", "log10", 1e-4, 100.0, 0.5, "true"],
            ["k2", "log10", 1e-4, 100.0, 0.1, "true"],
            ["A_init", "lin", 0.0, 100.0, 1.0, "false"],
            ["B_init", "lin", 0.0, 100.0, 0.0, "false"],
            ["s_d1", "lin", 0.01, 100.0, 1.0, "true"],
            ["s_d2", "lin", 0.01, 100.0, 2.0, "true"],
            ["sd_d1", "lin", 0.01, 100.0, 0.1, "true"],
            ["sd_d2", "lin", 0.01, 100.0, 0.3, "true"],
        ],
    )
    yaml = work / "problem.yaml"
    yaml.write_text(
        yaml.read_text()
        + "condition_files:\n- conditions.tsv\nexperiment_files:\n- experiments.tsv\n"
    )
    training = WorkingProblem.from_yaml(yaml)

    held = pd.DataFrame(
        {
            "observableId": "obs_B",
            "experimentId": "e_ho",
            "time": [1.0, 2.0, 4.0],
            "measurement": [0.5, 0.9, 1.2],
            "observableParameters": "s_d2",
            "noiseParameters": "sd_d2",
        }
    )
    built = []
    real = fitting.pypesto_problem
    monkeypatch.setattr(
        fitting, "pypesto_problem", lambda p, kw, **k: built.append(p) or real(p, kw, **k)
    )

    def score(**point):
        return V.score_problem(
            training,
            held,
            build_dir=str(tmp_path / "build"),
            model_name="toy",
            condition_df=pd.DataFrame(
                {"conditionId": ["c_ho"], "targetId": ["A"], "targetValue": [1.0]}
            ),
            experiment_df=pd.DataFrame(
                {"experimentId": ["e_ho"], "time": [0.0], "conditionId": ["c_ho"]}
            ),
            **point,
        )

    def nll(k1, k2, s, sd):
        b = k1 / (k1 + k2) * (1 - np.exp(-(k1 + k2) * held["time"].to_numpy(float)))
        r = (held["measurement"].to_numpy(float) - s * b) / sd
        return float(np.sum(0.5 * np.log(2 * math.pi * sd**2) + 0.5 * r**2))

    out = score()
    assert out["status"] == "ok", out
    assert [i.message for i in built[-1].to_petab().validate()] == []
    # AMICI integrates at its default rtol, ~1e-6 off the closed form here (platform-dependent);
    # day 1's values would miss by ~11.
    assert out["nll"] == pytest.approx(nll(0.5, 0.1, 2.0, 0.3), rel=1e-4)

    # The caller's vector names day 1 too; the view needs none of it.
    names = ["k1", "k2", "A_init", "B_init", "s_d1", "s_d2", "sd_d1", "sd_d2"]
    x = [math.log10(0.4), math.log10(0.2), 1.0, 0.0, 9.0, 1.5, 9.0, 0.25]
    out = score(x_full=x, x_names=names)
    assert out["status"] == "ok", out
    assert out["nll"] == pytest.approx(nll(0.4, 0.2, 1.5, 0.25), rel=1e-4)
