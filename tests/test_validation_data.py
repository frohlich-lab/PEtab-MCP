"""Held-out sets in v1 form (`validation_data.read_held_out` + `as_v2`): the benchmark's held-out
files predate the v2 server, so they are keyed onto experiments the way the problem's own upgrade
keys its training measurements."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

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
