"""The working problem (`petab_mcp.problem`): its table conventions and the v1 upgrade.

The upgrade is held to the standard the scores downstream need: an upgraded problem must be the
SAME estimation problem, so every likelihood recorded against the v1 problem stays valid.
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import pytest

from petab_mcp import problem as P

DATA = Path(__file__).parent / "data"


def test_the_v1_toy_upgrades_to_exactly_the_v2_toy():
    """Every table, and the module fingerprint — so the two even share a compiled module."""
    old = P.WorkingProblem.from_yaml(DATA / "toy_v1" / "problem.yaml")
    new = P.WorkingProblem.from_yaml(DATA / "toy_v2" / "problem.yaml")
    for name in ("parameter_df", "observable_df", "measurement_df", "condition_df"):
        a, b = getattr(old, name), getattr(new, name)
        assert (a is None and b is None) or a.equals(b), name
    assert old.experiment_df is None and new.experiment_df is None
    assert old.module_fingerprint() == new.module_fingerprint()


def test_the_upgrade_keeps_each_parameter_scale():
    """libpetab's own upgrade drops `parameterScale`; the server estimates on it."""
    pdf = P.WorkingProblem.from_yaml(DATA / "toy_v1" / "problem.yaml").parameter_df
    assert dict(pdf[P.PARAMETER_SCALE]) == {
        "k1": "log10",
        "k2": "log10",
        "sigma_B": "lin",
        "A_init": "lin",
        "B_init": "lin",
    }


@pytest.mark.parametrize("transformation", ["lin", "log", "log10"])
def test_a_v1_observable_keeps_its_likelihood(transformation):
    """v2 has no log10 distribution. ``petab1to2`` maps log10-normal onto log-normal with the same
    noise formula, which moves every likelihood by n·ln(ln 10) and rescales what sigma means; the
    server multiplies the noise formula by log(10) instead, so the densities are IDENTICAL."""
    from petab.v1.calculate import calculate_llh as llh_v1
    from petab.v2.calculate import calculate_llh as llh_v2

    observables = pd.DataFrame(
        {
            "observableId": ["o"],
            "observableFormula": ["x"],
            "noiseFormula": ["noiseParameter1_o"],
            "observableTransformation": [transformation],
        }
    ).set_index("observableId")
    measurements = pd.DataFrame(
        {
            "observableId": ["o"] * 3,
            "simulationConditionId": ["c"] * 3,
            "time": [0.0, 1.0, 2.0],
            "measurement": [1.0, 2.0, 4.0],
            "noiseParameters": ["sd"] * 3,
        }
    )
    parameters = pd.DataFrame(
        {"parameterId": ["sd"], "nominalValue": [0.3], "estimate": [0]}
    ).set_index("parameterId")
    simulation = measurements.rename(columns={"measurement": "simulation"}).assign(
        simulation=[1.1, 1.8, 4.4]
    )
    v1 = llh_v1(measurements, simulation, observables, parameters)

    _, measurements2 = P.v1_measurements(measurements, set())
    observables2 = P._v1_observables(observables)
    simulation2 = measurements2.rename(columns={"measurement": "simulation"}).assign(
        simulation=[1.1, 1.8, 4.4]
    )
    v2 = llh_v2(measurements2, simulation2, observables2, parameters)
    assert v2 == pytest.approx(v1, rel=1e-12)


def test_a_v1_pre_equilibration_with_an_empty_condition_is_kept():
    """A pre-equilibration whose condition sets nothing still equilibrates the model first. The
    pair becomes an experiment with a -inf period; ``petab1to2`` drops the experiment entirely."""
    measurements = pd.DataFrame(
        {
            "observableId": ["o", "o"],
            "preequilibrationConditionId": ["basal", None],
            "simulationConditionId": ["basal", "basal"],
            "time": [1.0, 1.0],
            "measurement": [1.0, 2.0],
        }
    )
    experiments, upgraded = P.v1_measurements(measurements, changing=set())
    assert list(upgraded["experimentId"]) == ["experiment__basal___basal", ""]
    assert experiments["time"].tolist() == [-math.inf, 0.0]
    assert experiments["conditionId"].isna().all()  # neither period changes anything


def test_a_v1_experiment_id_column_is_kept_under_another_name():
    """v1 had no experimentId column, so one there is free text (brannmark's block names)."""
    measurements = pd.DataFrame(
        {
            "observableId": ["o"],
            "simulationConditionId": ["c"],
            "time": [1.0],
            "measurement": [1.0],
            "experimentId": ["dose_response"],
        }
    )
    _, upgraded = P.v1_measurements(measurements, changing={"c"})
    assert upgraded["experimentId"].tolist() == [P.v1_experiment_id("", "c")]
    assert upgraded["v1_experimentId"].tolist() == ["dose_response"]


def test_parameter_table_conventions_hold_on_every_assignment():
    """``estimate`` a bool (v2's strings and v1's 0/1 alike), numeric columns float, the scale
    always present, priors and a reader's stray ``index`` column gone."""
    wp = P.WorkingProblem()
    wp.parameter_df = pd.DataFrame(
        {
            "index": [0, 1],
            "parameterId": ["a", "b"],
            "lowerBound": [0, 1],
            "upperBound": [10, 100],
            "nominalValue": [1, 2],
            "estimate": ["true", 0],
            "priorDistribution": ["normal", ""],
            "priorParameters": ["0;1", ""],
        }
    )
    df = wp.parameter_df
    assert df.index.name == "parameterId" and "index" not in df.columns
    assert df["estimate"].tolist() == [True, False] and df["estimate"].dtype == bool
    assert df["lowerBound"].dtype == float  # a fitted float can be written into it
    assert df[P.PARAMETER_SCALE].tolist() == ["lin", "lin"]
    assert "priorDistribution" not in df.columns and "priorParameters" not in df.columns
    df.loc["b", "estimate"] = True  # writes land in the stored frame
    assert wp.x_free_ids == ["a", "b"]


def test_an_experiment_less_measurement_is_keyed_by_the_empty_string():
    """NaN would drop the row from every groupby and never equal itself in a SQL join."""
    wp = P.WorkingProblem.from_yaml(DATA / "toy_v2" / "problem.yaml")
    assert set(wp.measurement_df["experimentId"]) == {""}
    assert {m.experiment_id for m in wp.to_petab().measurements} == {None}


def test_to_petab_validates_and_carries_the_scale_as_an_extra_field():
    wp = P.WorkingProblem.from_yaml(DATA / "toy_v2" / "problem.yaml")
    petab_problem = wp.to_petab()
    assert not [i for i in petab_problem.validate() if i.level >= 30]
    k1 = petab_problem["k1"]
    assert k1.model_extra == {P.PARAMETER_SCALE: "log10"}  # and no `index` extra


def test_the_module_fingerprint_moves_with_what_is_compiled_only():
    wp = P.WorkingProblem.from_yaml(DATA / "toy_v2" / "problem.yaml")
    base = wp.module_fingerprint()
    wp.parameter_df.loc["k1", "nominalValue"] = 7.0  # a value: set per simulation
    wp.measurement_df = wp.measurement_df.assign(measurement=0.5)
    assert wp.module_fingerprint() == base
    wp.parameter_df.loc["sigma_B", "estimate"] = True  # a constant would have no sensitivity
    assert wp.module_fingerprint() != base


def test_experiment_inputs_are_the_targets_in_effect_while_simulated():
    """Pre-equilibration changes persist unless the simulated period resets them (v2)."""
    wp = P.WorkingProblem()
    wp.condition_df = pd.DataFrame(
        {
            "conditionId": ["pre", "pre", "dose"],
            "targetId": ["inhibitor", "dose", "dose"],
            "targetValue": [1.0, 0.0, 5.0],
        }
    )
    wp.experiment_df = pd.DataFrame(
        {
            "experimentId": ["e", "e", "f"],
            "time": [-math.inf, 0.0, 0.0],
            "conditionId": ["pre", "dose", "dose"],
        }
    )
    inputs = P.experiment_inputs(wp)
    assert inputs.loc["e", "dose"] == 5.0 and inputs.loc["e", "inhibitor"] == 1.0
    assert inputs.loc["f", "dose"] == 5.0 and math.isnan(inputs.loc["f", "inhibitor"])


def test_two_v1_conditions_that_change_nothing_stay_two_series():
    """``""`` is one simulation: two condition-free v1 series would merge in every per-experiment
    diagnostic (runs tests, replicate groups, a unit per experiment)."""
    measurements = pd.DataFrame(
        {
            "observableId": ["o", "o", "o"],
            "simulationConditionId": ["line_a", "line_b", "line_a"],
            "time": [1.0, 1.0, 2.0],
            "measurement": [1.0, 2.0, 3.0],
        }
    )
    experiments, upgraded = P.v1_measurements(measurements, changing=set())
    a, b = P.v1_experiment_id("", "line_a"), P.v1_experiment_id("", "line_b")
    assert upgraded["experimentId"].tolist() == [a, b, a]
    assert sorted(experiments["experimentId"]) == [a, b]
    _, alone = P.v1_measurements(
        measurements[measurements["simulationConditionId"] == "line_a"], set()
    )
    assert set(alone["experimentId"]) == {""}


def test_a_write_into_a_table_reaches_the_libpetab_problem():
    """Cached tables are keyed on content: a held-out view writes its solved scalings into its
    measurement table in place."""
    wp = P.WorkingProblem.from_yaml(DATA / "toy_v2" / "problem.yaml")
    assert wp.to_petab().measurements[1].measurement == pytest.approx(0.3)
    wp.measurement_df["measurement"] = 0.5
    assert {m.measurement for m in wp.to_petab().measurements} == {0.5}


def test_views_over_different_rows_of_one_design_share_a_module():
    import copy

    wp = P.WorkingProblem.from_yaml(DATA / "toy_v2" / "problem.yaml")
    view = copy.copy(wp)
    view.measurement_df = wp.measurement_df.iloc[3:]
    assert view.module_fingerprint() == wp.module_fingerprint()
