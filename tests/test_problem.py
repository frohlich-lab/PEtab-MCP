"""The working problem (`petab_mcp.problem`): its table conventions and the v1 upgrade.

The upgrade is held to the standard the scores downstream need: an upgraded problem must be the
SAME estimation problem, so every likelihood recorded against the v1 problem stays valid.
"""

from __future__ import annotations

import math
import shutil
from pathlib import Path

import pandas as pd
import pytest

from petab_mcp import problem as P

DATA = Path(__file__).parent / "data"


def test_the_v1_toy_upgrades_to_exactly_the_v2_toy():
    """Every table up to column order, and so the module fingerprint once both hold one model
    (toy_v1 ships SBML, toy_v2 the Antimony it was made from)."""
    old = P.WorkingProblem.from_yaml(DATA / "toy_v1" / "problem.yaml")
    new = P.WorkingProblem.from_yaml(DATA / "toy_v2" / "problem.yaml")
    for name in ("parameter_df", "observable_df", "measurement_df", "condition_df"):
        a, b = getattr(old, name), getattr(new, name)
        assert (a is None and b is None) or a.sort_index(axis=1).equals(b.sort_index(axis=1)), name
    assert old.experiment_df is None and new.experiment_df is None
    new.model = old.model
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
def test_a_v1_observable_keeps_its_likelihood(tmp_path, transformation):
    """v2 has no log10 distribution: ``petab1to2`` maps log10-normal onto log-normal and rescales
    the noise by ln(10) (PEtab-dev/libpetab-python#535), so the densities are IDENTICAL,
    placeholder overrides included -- provided nothing rescales a second time, which the upgrade
    did once #535 landed beside its own ``log(10) *``."""
    from petab import v1
    from petab.v1.calculate import calculate_llh as llh_v1
    from petab.v2.calculate import calculate_llh as llh_v2

    shutil.copytree(DATA / "toy_v1", tmp_path, dirs_exist_ok=True)
    pd.DataFrame(
        {
            "observableId": ["obs_B"],
            "observableFormula": ["B"],
            "noiseFormula": ["noiseParameter1_obs_B"],
            "observableTransformation": [transformation],
        }
    ).to_csv(tmp_path / "observables.tsv", sep="\t", index=False)
    mdf = pd.read_csv(tmp_path / "measurements.tsv", sep="\t")
    # strictly positive, for the log scales
    mdf.assign(measurement=mdf["measurement"] + 0.1, noiseParameters="sigma_B").to_csv(
        tmp_path / "measurements.tsv", sep="\t", index=False
    )
    old = v1.Problem.from_yaml(tmp_path / "problem.yaml")
    new = P.WorkingProblem.from_yaml(tmp_path / "problem.yaml")

    def simulated(df):
        return df.rename(columns={"measurement": "simulation"}).assign(
            simulation=lambda d: 1.1 * d["simulation"]
        )

    expected = llh_v1(
        old.measurement_df, simulated(old.measurement_df), old.observable_df, old.parameter_df
    )
    llh = llh_v2(
        new.measurement_df, simulated(new.measurement_df), new.observable_df, new.parameter_df
    )
    assert llh == pytest.approx(expected, rel=1e-12)


def test_a_v1_problem_that_does_not_lint_still_upgrades(tmp_path):
    """Every benchmark problem fails v1 lint as shipped: its observables name readouts the agent's
    model has yet to define. petab1to2 refused all of them while it linted, so the server could
    start on none (PEtab-dev/libpetab-python#534 made the lint optional)."""
    from petab import v1

    shutil.copytree(DATA / "toy_v1", tmp_path, dirs_exist_ok=True)
    odf = pd.read_csv(tmp_path / "observables.tsv", sep="\t")
    odf.loc[0, "observableFormula"] = "readout_B"
    odf.to_csv(tmp_path / "observables.tsv", sep="\t", index=False)
    assert v1.lint_problem(v1.Problem.from_yaml(tmp_path / "problem.yaml"))  # it does not lint
    new = P.WorkingProblem.from_yaml(tmp_path / "problem.yaml")
    assert new.observable_df.iloc[0]["observableFormula"] == "readout_B"


def _v1_toy_with_stimulus(path: Path, stimulus: str) -> Path:
    """toy_v1 at ``path`` with ``k1`` scaled by ``1 + stim`` and ``stim := stimulus``: v1's way to
    write a protocol, since v1 had no experiments."""
    from petab.v2.models.sbml_model import SbmlModel

    shutil.copytree(DATA / "toy_v1", path, dirs_exist_ok=True)
    ant = (DATA / "toy_v1" / "model.ant").read_text()
    ant = ant.replace("k1 * A;", f"k1 * A * (1 + stim);\n  stim := {stimulus};\n  t_on = 5;")
    sbml = SbmlModel.from_antimony(ant, model_id="conversion_reaction").to_sbml_str()
    (path / "conversion_reaction.xml").write_text(sbml)
    return path / "problem.yaml"


def test_a_time_switched_assignment_becomes_an_experiment_period(tmp_path):
    """v1 wrote a protocol into the model as an assignment that switches with time;
    ``assignments_to_experiments`` moves it into the experiment table, as v2 writes protocols, and
    the problem says it did (held-out sets keyed by v1 conditions cannot follow it)."""
    new = P.WorkingProblem.from_yaml(_v1_toy_with_stimulus(tmp_path, "piecewise(0, time < 5, 2)"))

    assert new.converted_assignments == ("stim",)
    assert new.model.sbml_model.getAssignmentRuleByVariable("stim") is None
    edf, cdf = new.experiment_df, new.condition_df
    assert (
        set(new.measurement_df["experimentId"])
        == set(edf["experimentId"])
        == {"default_experiment"}
    )
    switch = cdf.loc[cdf["conditionId"] == edf.loc[edf["time"] == 5, "conditionId"].item()]
    assert switch[["targetId", "targetValue"]].values.tolist() == [["stim", 2.0]]
    # The experiment still starts at 0. libpetab dropped that period, since the model already
    # holds stim = 0, and a v2 experiment starts at its first period: AMICI simulated from 5, and
    # the toy's measurements at 0-4 scored 35 nats worse than under the rule.
    assert (
        sorted(edf["time"]) == [0.0, 5.0] and edf.loc[edf["time"] == 0, "conditionId"].isna().all()
    )
    assert "modelId" not in new.measurement_df.columns  # written like petab1to2 writes tables
    # the scales petab1to2 drops still carry over: the conversion adds no parameter-table row
    assert new.parameter_df.loc[["k1", "k2"], P.PARAMETER_SCALE].tolist() == ["log10", "log10"]
    assert new.replace().converted_assignments == ("stim",)


@pytest.mark.parametrize(
    "case",
    ["estimated switching time", "does not lint"],
)
def test_an_assignment_that_cannot_move_keeps_its_rule(tmp_path, case):
    """The rule is a valid v2 model, only not an experiment, so the problem is served as
    ``petab1to2`` made it. ``assignments_to_experiments`` raises NotImplementedError for a
    switching time the fit estimates (Oliveira's), and it lints what it returns, which a problem
    whose model the agent has yet to write fails -- calling it there would refuse the problem."""
    stimulus = "piecewise(0, time < 5, 2)"
    if case == "estimated switching time":
        stimulus = "piecewise(0, time < t_on, 2)"
    yaml = _v1_toy_with_stimulus(tmp_path, stimulus)
    if case == "estimated switching time":
        with open(tmp_path / "parameters.tsv", "a") as f:
            f.write("t_on\tlin\t0\t10\t5\t1\n")
    else:
        odf = pd.read_csv(tmp_path / "observables.tsv", sep="\t")
        odf.loc[0, "observableFormula"] = "readout_B"
        odf.to_csv(tmp_path / "observables.tsv", sep="\t", index=False)

    new = P.WorkingProblem.from_yaml(yaml)

    assert new.converted_assignments == ()
    assert new.model.sbml_model.getAssignmentRuleByVariable("stim") is not None
    assert new.experiment_df is None and set(new.measurement_df["experimentId"]) == {""}


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
    wp = wp.replace(measurement_df=wp.measurement_df.assign(measurement=0.5))
    assert wp.module_fingerprint() == base
    wp.parameter_df.loc["sigma_B", "estimate"] = True  # a constant would have no sensitivity
    assert wp.module_fingerprint() != base


def test_experiment_inputs_are_the_targets_in_effect_while_simulated():
    """Pre-equilibration changes persist unless the simulated period resets them (v2); a target a
    later period changes has no single value, one it sets again to the same value keeps it."""
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
            "experimentId": ["e", "e", "f", "g", "g", "g"],
            "time": [-math.inf, 0.0, 0.0, -math.inf, 0.0, 10.0],
            "conditionId": ["pre", "dose", "dose", "pre", "dose", "pre"],
        }
    )
    inputs = P.experiment_inputs(wp)
    assert inputs.loc["e", "dose"] == 5.0 and inputs.loc["e", "inhibitor"] == 1.0
    assert inputs.loc["f", "dose"] == 5.0 and math.isnan(inputs.loc["f", "inhibitor"])
    assert math.isnan(inputs.loc["g", "dose"]) and inputs.loc["g", "inhibitor"] == 1.0


def test_the_design_tables_are_read_only_and_a_view_is_a_new_problem():
    wp = P.WorkingProblem.from_yaml(DATA / "toy_v2" / "problem.yaml")
    with pytest.raises(AttributeError):
        wp.measurement_df = wp.measurement_df.iloc[3:]
    view = wp.replace(measurement_df=wp.measurement_df.iloc[3:])
    view.parameter_df.loc["k1", "nominalValue"] = 7.0
    assert len(view.measurement_df) == 3 and len(wp.measurement_df) == 6
    assert wp.parameter_df.loc["k1", "nominalValue"] == 0.5  # not shared
    assert view.model is wp.model and view.observable_df.equals(wp.observable_df)


def test_views_over_different_rows_of_one_design_share_a_module():
    wp = P.WorkingProblem.from_yaml(DATA / "toy_v2" / "problem.yaml")
    view = wp.replace(measurement_df=wp.measurement_df.iloc[3:])
    assert view.module_fingerprint() == wp.module_fingerprint()
