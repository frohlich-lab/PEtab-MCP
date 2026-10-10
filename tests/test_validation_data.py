"""The fixed-θ HELD-OUT check: :func:`petab_mcp.validation_data.score_problem` and its guards.

Tested at the boundary rather than through an end-to-end run: which tables the scorer copies, what
it declines before paying for a build, and how it refuses a parameter vector it cannot line up.
The objective evaluation past those guards needs an AMICI build; its correctness is pinned by the
reference's own held-out score, re-derived by the benchmark's ``setup.py --refit``. The harness's
materialisation of a held-out set, and its judge's projection of the score, are tested in gemot.

Held-out sets in v1 form (`validation_data.read_held_out` + `as_v2`): the benchmark's held-out
files predate the v2 server, so they are keyed onto experiments the way the problem's own upgrade
keys its training measurements.
"""

from __future__ import annotations

import copy
from pathlib import Path

import pandas as pd
import pytest

from petab_mcp import validation_data as V
from petab_mcp.problem import WorkingProblem, v1_experiment_id

DATA = Path(__file__).parent / "data"


def _write(path: Path, rows: list[list]) -> None:
    path.write_text("".join("\t".join(map(str, r)) + "\n" for r in rows))


def test_a_v1_held_out_set_lands_in_the_experiments_the_upgrade_makes(tmp_path):
    """A condition that changes something (the set's own or the training problem's) becomes an
    experiment; one that changes nothing simulates the model's defaults, keyed by ``""``."""
    m = tmp_path / "measurements.tsv"
    _write(
        m,
        [
            ["observableId", "simulationConditionId", "time", "measurement"],
            ["obs_B", "c_new", 1.0, 0.3],
            ["obs_B", "c_dose", 2.0, 0.4],
            ["obs_B", "basal", 3.0, 0.5],
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
    new, dose = v1_experiment_id("", "c_new"), v1_experiment_id("", "c_dose")
    assert upgraded.measurement_df["experimentId"].tolist() == [new, dose, ""]
    assert sorted(upgraded.experiment_df["experimentId"]) == sorted([new, dose])
    assert upgraded.units == {new: "u1", dose: "u2"}
    assert V.as_v2(upgraded, training) is upgraded  # already v2


def test_a_curated_scaling_is_solved_for_a_held_out_experiment(tmp_path):
    """The fixed-θ scorer end to end: a new experiment whose blot names its own scale. The view
    keeps the training experiment with no rows left in it, and writes the solved scale into its
    measurement table before the final evaluation. Checked against the toy's closed form."""
    import shutil

    import numpy as np
    import pytest

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


def _problem(measurements, *, observables=("obs_A",), conditions=("c_train", "c_test")):
    """A minimal petab problem stand-in with just the tables the guards read."""
    import petab.v1 as petab

    return petab.Problem(
        model=None,
        measurement_df=measurements,
        observable_df=pd.DataFrame(
            {
                "observableFormula": ["x"] * len(observables),
                "noiseFormula": ["1"] * len(observables),
            },
            index=pd.Index(list(observables), name="observableId"),
        ),
        condition_df=pd.DataFrame(
            {"dose": [0.0] * len(conditions)}, index=pd.Index(list(conditions), name="conditionId")
        ),
        parameter_df=pd.DataFrame(
            {
                "parameterScale": ["lin"],
                "lowerBound": [0.0],
                "upperBound": [1.0],
                "nominalValue": [0.5],
                "estimate": [1],
            },
            index=pd.Index(["k"], name="parameterId"),
        ),
    )


def _rows(condition, times=(0.0, 1.0), observable="obs_A"):
    return pd.DataFrame(
        {
            "observableId": [observable] * len(times),
            "simulationConditionId": [condition] * len(times),
            "time": list(times),
            "measurement": [1.0] * len(times),
        }
    )


def test_scoring_never_mutates_the_problem_it_was_handed(monkeypatch):
    """``score_session`` runs against the LIVE session problem, so a table it swapped in place would
    change what the agent reads and what the judge scores on the training data — the held-out check
    would corrupt the very run it is reporting on. The scorer builds a shallow view with its own
    dataframe objects; this pins that, without paying for an AMICI build, by intercepting the
    importer and inspecting the view it was handed.

    Also pins that the visualization table is DROPPED: petab's lint rejects one whose datasetIds are
    absent from the measurement table, which is exactly what swapping in held-out data does, and
    ``PetabImporter`` raises a bare "Invalid PEtab problem." for it (observed on the collection's
    fujita).
    """
    import pypesto.petab

    from petab_mcp import validation_data

    problem = _problem(_rows("c_train"))
    problem.visualization_df = pd.DataFrame({"plotId": ["p1"], "datasetId": ["c_train_obs_A"]})
    original_measurements = problem.measurement_df
    held_out = _rows("c_test")

    captured = {}

    class _Stop(RuntimeError):
        pass

    def _importer(petab_problem, **kwargs):
        captured["view"] = petab_problem
        raise _Stop

    monkeypatch.setattr(pypesto.petab, "PetabImporter", _importer)
    with pytest.raises(_Stop):
        validation_data.score_problem(problem, held_out, build_dir="/tmp/x", model_name="m")

    view = captured["view"]
    assert view is not problem
    assert list(view.measurement_df["simulationConditionId"]) == ["c_test", "c_test"]
    assert view.visualization_df is None, "the training visualization table would fail petab lint"
    # ...and the caller's own problem is exactly as it was
    assert problem.measurement_df is original_measurements
    assert list(problem.measurement_df["simulationConditionId"]) == ["c_train", "c_train"]
    assert problem.visualization_df is not None


def test_the_held_out_view_keeps_the_training_estimate_split(monkeypatch):
    """The view handed to the importer must mark exactly the parameters the training table marks.

    AMICI compiles a NON-estimated parameter as a model constant, so a compiled module is a
    function of the SBML plus the free/fixed split — not of the SBML alone. ``score_session``
    therefore hands over the build target the training problem was compiled with and reuses that
    module, which is correct only while the view's split matches. This pins the copy that keeps it
    matching; a view that freed or fixed a parameter would silently evaluate against constants
    baked in for a different split.
    """
    import pypesto.petab

    from petab_mcp import validation_data

    problem = _problem(_rows("c_train"))
    captured = {}

    class _Stop(RuntimeError):
        pass

    def _importer(petab_problem, **kwargs):
        captured["view"] = petab_problem
        captured["kwargs"] = kwargs
        raise _Stop

    monkeypatch.setattr(pypesto.petab, "PetabImporter", _importer)
    with pytest.raises(_Stop):
        validation_data.score_problem(
            problem, _rows("c_test"), build_dir="/tmp/train_build", model_name="train_module"
        )

    view = captured["view"]
    assert view.parameter_df["estimate"].to_dict() == problem.parameter_df["estimate"].to_dict()
    # ...and the target is passed through untouched, so score_session naming the TRAINING pair is
    # what makes the reuse land on the training module.
    assert captured["kwargs"]["output_folder"] == "/tmp/train_build"
    assert captured["kwargs"]["model_name"] == "train_module"


def test_an_empty_or_foreign_held_out_table_is_declined_before_any_build(monkeypatch):
    """The guards run BEFORE the importer, so a bad table costs nothing and names its own problem
    rather than surfacing from inside AMICI's parameter mapping."""
    import pypesto.petab

    from petab_mcp import validation_data

    def _boom(*a, **k):  # pragma: no cover - must not be reached
        raise AssertionError("the importer must not be reached for a declined table")

    monkeypatch.setattr(pypesto.petab, "PetabImporter", _boom)
    problem = _problem(_rows("c_train"))

    empty = validation_data.score_problem(
        problem, _rows("c_test").iloc[0:0], build_dir="/tmp/x", model_name="m"
    )
    assert empty["status"] == validation_data.STATUS_NO_DATA

    foreign = validation_data.score_problem(
        problem, _rows("c_elsewhere"), build_dir="/tmp/x", model_name="m"
    )
    assert foreign["status"] == validation_data.STATUS_TABLE_MISMATCH
    assert "c_elsewhere" in foreign["reason"]

    unknown_obs = validation_data.score_problem(
        problem, _rows("c_test", observable="obs_Z"), build_dir="/tmp/x", model_name="m"
    )
    assert unknown_obs["status"] == validation_data.STATUS_TABLE_MISMATCH
    assert "obs_Z" in unknown_obs["reason"]


def test_the_server_only_registers_the_tool_when_given_held_out_data(
    monkeypatch, yaml_path, tmp_path
):
    """No held-out file, no tool — so an example without a validation set cannot accidentally be
    scored against a stale or unrelated table, and the judge's "unavailable" status is reached by
    the tool genuinely not existing.

    Asserted through the registrar actually being CALLED, not by grepping ``create_server``'s
    source: a reformat must not fail this, and an equivalent rewrite must not pass it while
    dropping the gate."""
    import inspect

    from petab_mcp import server as server_mod
    from petab_mcp import validation_data
    from petab_mcp.server import create_server

    calls = []
    monkeypatch.setattr(
        server_mod,
        "_register_validation_data_tools",
        lambda mcp, ctx, held_out: calls.append(held_out),
    )
    assert "validation_measurements" in inspect.signature(create_server).parameters

    create_server(
        yaml_path, build_root=str(tmp_path / "b1"), provenance_dir=str(tmp_path / "prov1")
    )
    assert calls == []

    meas = tmp_path / "held.tsv"
    _rows("c_test").to_csv(meas, sep="\t", index=False)
    create_server(
        yaml_path,
        build_root=str(tmp_path / "b2"),
        provenance_dir=str(tmp_path / "prov2"),
        validation_measurements=str(meas),
    )
    assert len(calls) == 1
    assert isinstance(calls[0], validation_data.HeldOut)
    assert len(calls[0].measurement_df) == 2


def test_an_explicit_point_must_come_with_matching_names(monkeypatch):
    """A caller-supplied vector is in ITS problem's x_names order. The two problems are built from
    the same parameter table so they agree today, but a silent re-ordering would evaluate a permuted
    vector and report a plausible, wrong likelihood — so the names are checked, not assumed."""
    from petab_mcp import validation_data

    problem = _problem(_rows("c_train"))
    with pytest.raises(ValueError, match="both x_full and x_names, or neither"):
        validation_data.score_problem(
            problem, _rows("c_test"), build_dir="/tmp/x", model_name="m", x_full=[0.5]
        )


# Each of these exists because its ABSENCE was a silent-wrong-number risk, so each needs a test
# that fails if it is relaxed back. (The repo's own lesson: 15 defects survived a green suite
# because the plumbing around the statistic went untested — this IS the plumbing.)
def test_a_short_x_full_is_refused_rather_than_evaluated_misaligned(monkeypatch):
    """The name check compares `x_names` against the problem's; it says NOTHING about how many
    values arrived with them. A shorter vector whose free indices all happen to be in range would
    otherwise be silently indexed and reported as a plausible held-out likelihood for a parameter
    point nobody chose."""
    import pypesto.petab

    from petab_mcp import validation_data

    class _FakePP:
        x_names = ["k", "sigma", "scale"]
        x_free_indices = [0]

        def objective(self, *a, **k):  # pragma: no cover - must not be reached
            raise AssertionError("the objective must not be evaluated on a bad vector")

    class _FakeImporter:
        def __init__(self, *a, **k):
            pass

        def create_problem(self):
            return _FakePP()

    monkeypatch.setattr(pypesto.petab, "PetabImporter", _FakeImporter)
    problem = _problem(_rows("c_train"))

    with pytest.raises(ValueError, match="must be the FULL vector"):
        validation_data.score_problem(
            problem,
            _rows("c_test"),
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
    from petab_mcp.validation_data import _reduced_from_named

    class _PP:
        x_names = ["k", "sigma"]  # the view dropped the training scalings
        x_free_indices = [0, 1]

    reduced = _reduced_from_named(_PP(), [1.0, 2.0, 3.0], ["k", "sigma", "s_blot_retired"])
    assert reduced == [1.0, 2.0]  # in pp.x_names order, the extra name ignored


def test_a_name_the_view_needs_and_the_caller_omitted_is_refused():
    """The other direction is still a hard error: a vector that does not cover what the problem
    needs would be evaluated permuted and report a plausible, wrong likelihood."""
    from petab_mcp.validation_data import _reduced_from_named

    class _PP:
        x_names = ["k", "sigma", "scale"]
        x_free_indices = [0, 1, 2]

    with pytest.raises(ValueError, match="do not match the held-out problem's"):
        _reduced_from_named(_PP(), [1.0, 2.0, 3.0], ["k", "sigma", "unrelated"])


def test_arity_is_reported_as_arity_not_as_a_name_mismatch():
    """A vector already reduced to free coordinates is a LENGTH mistake. Zipping it against the
    full name list would truncate silently and then blame the names."""
    from petab_mcp.validation_data import _reduced_from_named

    class _PP:
        x_names = ["k", "sigma", "scale"]
        x_free_indices = [0]

    with pytest.raises(ValueError, match="must be the FULL vector"):
        _reduced_from_named(_PP(), [0.5], ["k", "sigma", "scale"])


def test_score_problem_appends_a_fixed_sets_condition_rows_before_building(monkeypatch):
    """The scorer's view carries the training conditions AND the set's own, so a row on a new
    condition is a table the importer can build, not a mismatch; the caller's problem keeps its
    table."""
    import pypesto.petab

    from petab_mcp import validation_data

    seen = []

    class _Objective:
        def __call__(self, x, **kwargs):
            return {"fval": 1.0, "rdatas": []}

    class _PP:
        x_names = ["k"]
        x_free_indices = [0]
        x_scales = ["lin"]
        objective = _Objective()

    class _Importer:
        def __init__(self, petab_problem, **kwargs):
            seen.append(petab_problem)

        def create_problem(self):
            return _PP()

    monkeypatch.setattr(pypesto.petab, "PetabImporter", _Importer)
    import petab_mcp.fitting as fitting

    monkeypatch.setattr(fitting, "_apply_amici_solver_defaults", lambda pp: None)
    problem = _problem(_rows("c_train"))
    new = pd.DataFrame({"dose": [0.5]}, index=pd.Index(["c_knockdown"], name="conditionId"))

    out = validation_data.score_problem(
        problem, _rows("c_knockdown"), build_dir="/tmp/x", model_name="m", condition_df=new
    )

    assert out["status"] == validation_data.STATUS_OK and out["n_conditions"] == 1
    assert list(seen[0].condition_df.index) == ["c_train", "c_test", "c_knockdown"]
    assert list(problem.condition_df.index) == ["c_train", "c_test"]


def test_a_fixed_sets_condition_table_is_checked_before_any_build(monkeypatch):
    """Same contract as for a foreign measurement table: the guards run before the importer."""
    import pypesto.petab

    from petab_mcp import validation_data

    def _boom(*a, **k):  # pragma: no cover - must not be reached
        raise AssertionError("the importer must not be reached for a declined table")

    monkeypatch.setattr(pypesto.petab, "PetabImporter", _boom)
    problem = _problem(_rows("c_train"))

    other_column = pd.DataFrame(
        {"temperature": [37.0]}, index=pd.Index(["c_new"], name="conditionId")
    )
    out = validation_data.score_problem(
        problem, _rows("c_new"), build_dir="/tmp/x", model_name="m", condition_df=other_column
    )
    assert out["status"] == validation_data.STATUS_TABLE_MISMATCH
    assert "different models" in out["reason"]

    redefined = pd.DataFrame({"dose": [0.5]}, index=pd.Index(["c_train"], name="conditionId"))
    out = validation_data.score_problem(
        problem, _rows("c_test"), build_dir="/tmp/x", model_name="m", condition_df=redefined
    )
    assert out["status"] == validation_data.STATUS_TABLE_MISMATCH
    assert "does not redefine" in out["reason"]


def test_score_session_hands_a_fixed_sets_condition_rows_to_the_scorer(monkeypatch):
    """`score_session` is the server's one entry point; a fixed set's own rows reach the scorer
    through it, or the tool reports `table_mismatch` for every curated set that brings any."""
    import contextlib
    import types

    from petab_mcp import validation_data

    seen = {}

    def _score(problem, measurement_df, **kwargs):
        seen.update(kwargs)
        return {"status": "ok"}

    monkeypatch.setattr(validation_data, "score_problem", _score)
    session = types.SimpleNamespace(
        lock=contextlib.nullcontext(), problem=object(), sbml_hash=lambda: "h"
    )
    engine = types.SimpleNamespace(
        pypesto_importer_kwargs=lambda h: {"output_folder": "/tmp/x", "model_name": "m"}
    )
    rows = pd.DataFrame({"dose": [0.5]}, index=pd.Index(["c_knockdown"], name="conditionId"))

    validation_data.score_session(
        session, engine, validation_data.HeldOut(_rows("c_knockdown"), condition_df=rows)
    )

    assert seen["condition_df"] is rows


def test_read_held_out_gives_a_fixed_set_its_condition_rows(tmp_path):
    """On disk: measurements.tsv + conditions.tsv and NO parameters.tsv is a fixed-θ set with rows
    of its own — what `materialize` writes for such a set and what the server reads back."""
    from petab_mcp import validation_data

    _rows("c_knockdown").to_csv(tmp_path / "measurements.tsv", sep="\t", index=False)
    pd.DataFrame({"conditionId": ["c_knockdown"], "dose": [0.5]}).to_csv(
        tmp_path / validation_data.HELD_OUT_CONDITIONS, sep="\t", index=False
    )

    held = validation_data.read_held_out(tmp_path / "measurements.tsv")

    assert held.mode == validation_data.MODE_FIXED
    assert list(held.condition_df.index) == ["c_knockdown"]


def test_score_session_scores_the_toy_at_its_fitted_point_on_a_condition_of_the_sets_own(
    session, engine
):
    """The one AMICI-backed check of the fixed-θ side: the toy's measurements moved onto a condition
    `c1` the training table does not have, brought by the set. Scored at the training point, not
    refit — `mode: fixed`, no `k` — and the session's own table is left as it was."""
    import math
    from pathlib import Path

    from petab_mcp import validation_data

    toy = Path(__file__).parent / "data" / "toy_v1"
    held_meas = pd.read_csv(toy / "measurements.tsv", sep="\t").assign(simulationConditionId="c1")
    held = validation_data.HeldOut(
        measurement_df=held_meas,
        condition_df=pd.DataFrame(index=pd.Index(["c1"], name="conditionId")),
    )

    out = validation_data.score_session(session, engine, held)

    assert out["status"] == validation_data.STATUS_OK, out
    assert out["mode"] == validation_data.MODE_FIXED and "k" not in out
    assert out["n"] == len(held_meas) and out["n_conditions"] == 1
    assert math.isfinite(out["nll"])
    assert list(session.problem.condition_df.index) == ["c0"]


def _scaled_problem(formula="observableParameter1_obs_A * x"):
    """A problem whose one observable carries a scaling placeholder, like every blot readout."""
    import petab.v1 as petab

    return petab.Problem(
        model=None,
        measurement_df=_rows("c_train").assign(observableParameters="s_train"),
        observable_df=pd.DataFrame(
            {"observableFormula": [formula], "noiseFormula": ["noiseParameter1_obs_A"]},
            index=pd.Index(["obs_A"], name="observableId"),
        ),
        condition_df=pd.DataFrame(
            {"dose": [0.0, 0.0]}, index=pd.Index(["c_train", "c_test"], name="conditionId")
        ),
        parameter_df=pd.DataFrame(
            {
                "parameterScale": ["lin", "lin"],
                "lowerBound": [0.0, 0.0],
                "upperBound": [1.0, 10.0],
                "nominalValue": [0.5, 1.0],
                "estimate": [1, 1],
            },
            index=pd.Index(["k", "s_train"], name="parameterId"),
        ),
    )


def test_held_out_scalings_are_the_overrides_the_problem_does_not_define():
    """A held-out blot's scale factor is whatever override the parameter table lacks; a shipped
    scaling (fujita's rows) or a number is an ordinary PEtab override and not touched."""
    from petab_mcp import validation_data

    held = pd.concat(
        [
            _rows("c_test").assign(observableParameters="s_blot"),
            _rows("c_test", times=(2.0,)).assign(observableParameters="s_train"),
            _rows("c_test", times=(3.0,)).assign(observableParameters="2.5"),
        ]
    )
    assert validation_data.held_out_scalings(_scaled_problem(), held) == {"s_blot"}


def test_held_out_scalings_are_solved_hierarchically_then_written_in_as_numbers(monkeypatch):
    """The two-pass shape, pinned without an AMICI build: pass one hands pyPESTO a HIERARCHICAL
    problem whose only estimated rows are the held-out scalings tagged `parameterType: scaling`
    (the model's own parameters keep their values, the training-only overrides are gone); pass two
    is the ordinary joint objective with the solved values substituted as plain numbers, so the
    reported nll/chi2 come off the same code path as for an example without held-out scalings."""
    import pypesto.petab

    from petab_mcp import validation_data

    seen = []

    class _Calc:
        def get_interpretable_inner_par_ids(self):
            return ["s_blot"]

    class _Objective:
        calculator = _Calc()

        def __call__(self, x, **kwargs):
            return {"fval": 1.0, "inner_parameters": [3.5], "rdatas": []}

    class _PP:
        x_names = ["k"]
        x_free_indices = [0]
        x_scales = ["lin"]
        objective = _Objective()

    class _Importer:
        def __init__(self, petab_problem, **kwargs):
            seen.append((petab_problem, kwargs.get("hierarchical", False)))

        def create_problem(self):
            return _PP()

    monkeypatch.setattr(pypesto.petab, "PetabImporter", _Importer)
    monkeypatch.setattr(
        validation_data, "_apply_amici_solver_defaults", lambda pp: None, raising=False
    )
    import petab_mcp.fitting as fitting

    monkeypatch.setattr(fitting, "_apply_amici_solver_defaults", lambda pp: None)

    held = _rows("c_test").assign(observableParameters="s_blot")
    out = validation_data.score_problem(_scaled_problem(), held, build_dir="/tmp/x", model_name="m")

    assert [h for _, h in seen] == [True, False], (
        "hierarchical solve first, joint evaluation second"
    )
    hier_view, joint_view = seen[0][0], seen[1][0]
    row = hier_view.parameter_df.loc["s_blot"]
    assert row["parameterType"] == "scaling" and int(row["estimate"]) == 1
    assert float(row["lowerBound"]) == 0.0, "a gel's scale factor is non-negative"
    assert list(joint_view.measurement_df["observableParameters"]) == ["3.5", "3.5"]
    assert out["status"] == validation_data.STATUS_OK
    assert out["held_out_scalings"] == {"s_blot": 3.5}


def test_a_callers_vector_is_matched_by_name_so_a_smaller_held_out_problem_still_scores(
    monkeypatch,
):
    """The held-out view drops the training-only overrides and its hierarchical twin drops the inner
    scalings, so the problem pyPESTO builds has FEWER names than the reference fit's vector. Matching
    by name — not position — is what lets `_reference_validation` hand over its full vector."""
    import pypesto.petab

    from petab_mcp import validation_data

    class _PP:
        x_names = ["k"]
        x_free_indices = [0]
        x_scales = ["lin"]

        class objective:
            @staticmethod
            def __call__(x, **kwargs):
                raise AssertionError("not reached")

    assert validation_data._reduced_from_named(_PP(), [0.5, 2.0], ["s_train", "k"]) == [2.0]
    with pytest.raises(ValueError, match="do not match"):
        validation_data._reduced_from_named(_PP(), [0.5], ["s_train"])
    # ...and a caller whose vector is shorter than its own names is caught before any build
    monkeypatch.setattr(pypesto.petab, "PetabImporter", lambda *a, **k: None)
    with pytest.raises(ValueError, match="must be the FULL vector"):
        validation_data.score_problem(
            _scaled_problem(),
            _rows("c_test"),
            build_dir="/tmp/x",
            model_name="m",
            x_full=[0.5],
            x_names=["k", "s_train"],
        )


def _sbml_scaled_problem():
    """`_scaled_problem` with a REAL (tiny) SBML model, which the lint below needs: petab derives
    the set of legitimate parameter ids from the model, so with `model=None` it cannot tell an
    extraneous row from a kinetic one — and `_without_extraneous` short-circuits for the same
    reason. Returns `(problem, doc)`; the document is returned to keep libsbml's owner alive."""
    import libsbml
    import petab.v1 as petab
    from petab.v1.models.sbml_model import SbmlModel

    doc = libsbml.SBMLDocument(3, 2)
    sm = doc.createModel()
    sm.setId("m")
    comp = sm.createCompartment()
    comp.setId("cell"), comp.setConstant(True), comp.setSize(1.0), comp.setSpatialDimensions(3)
    sp_ = sm.createSpecies()
    sp_.setId("x"), sp_.setCompartment("cell"), sp_.setInitialConcentration(1.0)
    sp_.setHasOnlySubstanceUnits(False), sp_.setBoundaryCondition(False), sp_.setConstant(False)
    par = sm.createParameter()  # the kinetic the parameter table declares
    par.setId("k"), par.setValue(0.5), par.setConstant(True)
    base = _scaled_problem()
    return petab.Problem(
        model=SbmlModel(sbml_model=sm),
        observable_df=base.observable_df,
        condition_df=base.condition_df,
        measurement_df=base.measurement_df,
        parameter_df=base.parameter_df,
    ), doc


def test_the_extraneous_prune_is_petabs_lint_not_ours():
    """Why `_without_extraneous` exists at all. Swapping a held-out table in retires the overrides
    the TRAINING rows referenced, and petab's own parameter-table lint — which
    `PetabImporter.create_problem` runs, hierarchical or not — refuses the rows left behind. The
    hierarchical solver cannot help: the lint fires before there is an objective to solve."""
    from petab.v1.lint import check_parameter_df

    problem, _doc = _sbml_scaled_problem()
    held = _rows("c_test").assign(observableParameters="s_blot")
    view = copy.copy(problem)
    view.measurement_df = held
    view.parameter_df = pd.concat(
        [
            problem.parameter_df,
            pd.DataFrame(
                {
                    "parameterScale": ["lin"],
                    "lowerBound": [0.0],
                    "upperBound": [1e12],
                    "nominalValue": [1.0],
                    "estimate": [1],
                },
                index=pd.Index(["s_blot"], name="parameterId"),
            ),
        ]
    )

    with pytest.raises((AssertionError, ValueError), match="[Ee]xtraneous"):
        check_parameter_df(
            view.parameter_df,
            view.model,
            view.observable_df,
            view.measurement_df,
            view.condition_df,
        )

    from petab_mcp import validation_data

    kept = validation_data._without_extraneous(view)
    assert "s_train" not in kept.index, "the retired training override is dropped"
    assert {"k", "s_blot"} <= set(kept.index), "the kinetics and the held-out scaling stay"
    check_parameter_df(kept, view.model, view.observable_df, view.measurement_df, view.condition_df)
