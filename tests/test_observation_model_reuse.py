"""Pin the three tiers of the observation-model-reuse gate, and the rule-target prune rules.

These encode decisions that a plain refactor can silently reverse in either direction:

* tightening back to a blanket rejection re-breaks the published Bachmann mechanism (its total
  abundances are BOTH initial conditions and the relative observables' normalisers), which would
  reject any agent that rediscovers it;
* loosening to a blanket warning re-opens the Brannmark cheat a sweep actually caught — two runs
  wired the fixed observable-scaling factors straight into their rate laws instead of introducing
  the ~14 real kinetic parameters.

The discriminator is the symbol's ROLE, not where it appears, so the fixtures below are the two
shapes side by side: a scaling used AS the rate constant (error) versus the same scaling used as a
normaliser ALONGSIDE a dedicated rate constant (warning).
"""

from __future__ import annotations

import pandas as pd
import pytest
from petab.v2.C import (
    CONDITION_ID,
    ESTIMATE,
    EXPERIMENT_ID,
    LOWER_BOUND,
    MEASUREMENT,
    NOISE_FORMULA,
    NOMINAL_VALUE,
    OBSERVABLE_FORMULA,
    OBSERVABLE_ID,
    PARAMETER_ID,
    TARGET_ID,
    TARGET_VALUE,
    TIME,
    UPPER_BOUND,
)
from petab.v2.models.sbml_model import SbmlModel

from petab_mcp import reconcile, validation
from petab_mcp.problem import PARAMETER_SCALE, WorkingProblem


def _problem(
    antimony: str,
    *,
    observable_formula: str,
    noise_formula: str,
    extra_params: list[str] = (),
    condition_cols: dict | None = None,
):
    """A minimal PEtab problem whose observation model names `scale_obs` / `sd_obs`.

    ``condition_cols`` maps a condition target to its value in condition ``c0`` (which experiment
    ``e0`` applies), the shape a v1 condition column had."""
    observable_df = pd.DataFrame(
        [
            {
                OBSERVABLE_ID: "obs_A",
                OBSERVABLE_FORMULA: observable_formula,
                NOISE_FORMULA: noise_formula,
            }
        ]
    ).set_index(OBSERVABLE_ID)
    changes = [
        {CONDITION_ID: "c0", TARGET_ID: target, TARGET_VALUE: values[0]}
        for target, values in (condition_cols or {}).items()
    ]
    condition_df = pd.DataFrame(changes) if changes else None
    experiment_df = (
        pd.DataFrame([{EXPERIMENT_ID: "e0", TIME: 0.0, CONDITION_ID: "c0"}]) if changes else None
    )
    measurement_df = pd.DataFrame(
        [
            {
                OBSERVABLE_ID: "obs_A",
                EXPERIMENT_ID: "e0" if changes else "",
                TIME: 1.0,
                MEASUREMENT: 1.0,
            }
        ]
    )
    rows = [
        {
            PARAMETER_ID: p,
            PARAMETER_SCALE: "log10",
            LOWER_BOUND: 1e-3,
            UPPER_BOUND: 1e3,
            NOMINAL_VALUE: 1.0,
            ESTIMATE: True,
        }
        for p in ["scale_obs", "sd_obs", *extra_params]
    ]
    parameter_df = pd.DataFrame(rows).set_index(PARAMETER_ID)
    return WorkingProblem(
        model=SbmlModel.from_antimony(antimony),
        condition_df=condition_df,
        experiment_df=experiment_df,
        observable_df=observable_df,
        measurement_df=measurement_df,
        parameter_df=parameter_df,
    )


def _levels(problem) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {"error": set(), "warning": set()}
    for issue in validation._observation_model_reuse_issues(problem):
        # the offending id is the first quoted token of the message
        out[issue["level"]].add(issue["message"].split("'")[1])
    return out


CHEAT = """
model cheat
  compartment cell = 1;
  species A in cell, B in cell;
  r1: A => B; cell * A * scale_obs;
  A = 10; B = 0; scale_obs = 1; sd_obs = 1;
end
"""

NORMALISER = """
model normaliser
  compartment cell = 1;
  species A in cell, B in cell;
  r1: A => B; cell * A * k_fwd / scale_obs;
  A = 10; B = 0; scale_obs = 1; sd_obs = 1; k_fwd = 1;
end
"""

SIGMA_IN_RATE_LAW = """
model sigma_reuse
  compartment cell = 1;
  species A in cell, B in cell;
  r1: A => B; cell * A * k_fwd / sd_obs;
  A = 10; B = 0; scale_obs = 1; sd_obs = 1; k_fwd = 1;
end
"""

CHEAT_WITH_INPUT = """
model cheat_with_input
  compartment cell = 1;
  species A in cell, B in cell;
  r1: A => B; cell * A * scale_obs * dose_level;
  A = 10; B = 0; scale_obs = 1; sd_obs = 1; dose_level = 1;
end
"""


def test_scaling_as_rate_constant_is_an_error():
    """The Brannmark shape: the observable scaling IS the reaction's rate constant."""
    problem = _problem(CHEAT, observable_formula="A * scale_obs", noise_formula="sd_obs")
    assert "scale_obs" in _levels(problem)["error"]


def test_scaling_as_normaliser_is_only_a_warning():
    """The bachmann shape: the same symbol normalising alongside a dedicated rate constant."""
    problem = _problem(
        NORMALISER,
        observable_formula="A * scale_obs",
        noise_formula="sd_obs",
        extra_params=["k_fwd"],
    )
    levels = _levels(problem)
    assert levels["error"] == set()
    assert "scale_obs" in levels["warning"]


def test_noise_sigma_in_model_math_is_always_an_error():
    """A sigma has no legitimate mechanistic role, even beside a real rate constant."""
    problem = _problem(
        SIGMA_IN_RATE_LAW,
        observable_formula="A * scale_obs",
        noise_formula="sd_obs",
        extra_params=["k_fwd"],
    )
    assert "sd_obs" in _levels(problem)["error"]


def test_condition_target_does_not_count_as_the_reactions_own_parameter():
    """A dose target is an experimental INPUT, so multiplying one in must not launder the cheat."""
    problem = _problem(
        CHEAT_WITH_INPUT,
        observable_formula="A * scale_obs",
        noise_formula="sd_obs",
        condition_cols={"dose_level": [1.0]},
    )
    assert "scale_obs" in _levels(problem)["error"]


RULE_TARGET_PARAM = """
model rule_target_param
  compartment cell = 1;
  species A in cell, B in cell;
  r1: A => B; cell * A * k_fwd;
  drive := k_fwd * 2;
  A = 10; B = 0; k_fwd = 1;
end
"""

RULE_TARGET_SPECIES = """
model rule_target_species
  compartment cell = 1;
  species A in cell, B in cell, Drug in cell;
  r1: A => B; cell * A * k_fwd;
  Drug := k_fwd * 2;
  A = 10; B = 0; k_fwd = 1;
end
"""


@pytest.mark.parametrize(
    "antimony,target",
    [(RULE_TARGET_PARAM, "drive"), (RULE_TARGET_SPECIES, "Drug")],
    ids=["parameter", "species"],
)
def test_prune_orphans_keeps_a_rule_target_nothing_reads(antimony, target):
    """A rule TARGET is assigned by that rule, so it is never an orphan.

    Removing it leaves the rule pointing at a deleted id — invalid SBML that rejects the whole
    edit. `_model_math_names` walks rule right-hand sides only, so both the parameter and the
    species branch of `prune_orphans` need the targets folded in explicitly.
    """
    problem = _problem(
        antimony, observable_formula="A * scale_obs", noise_formula="sd_obs", extra_params=["k_fwd"]
    )
    reconcile.prune_orphans(problem)
    sm = problem.model.sbml_model
    assert target in reconcile.parameter_ids(sm) | reconcile.species_ids(sm)
    rule_targets = {sm.getRule(i).getVariable() for i in range(sm.getNumRules())}
    assert target in rule_targets, "the rule itself must survive alongside its target"


# --- the plausibility gate abstains where the spread ratio is degenerate ----------------------
def test_sigma_gate_abstains_below_the_minimum_sample_size():
    """A sigma judged against fewer than NOISE_SIGMA_MIN_POINTS measurements gets no verdict.

    `sigma_data_std` is a POPULATION std (``ddof=0``), the same functional form as the Gaussian
    MLE for sigma, so once the model reproduces a small group's mean and resolves nothing within
    it the two are equal BY CONSTRUCTION and `fraction_of_data_std` is 1 — a property of the
    estimator, not of the fit. bachmann is the real case: four absolute readouts with n=3 all land
    on 0.999994, agreeing to six significant figures across four different observables, while the
    model-free Rice estimator puts them at 0.67-1.09.

    Both verdict inputs must be withheld, not just one: `_entry_inflated` keys on
    `upper_bound_cap` and `_entry_collapsed` on `fraction_of_data_std`.
    """
    from petab_mcp import fitting as F
    from petab_mcp.constants import NOISE_SIGMA_MIN_POINTS

    antimony = """
    model tiny
      compartment cell = 1;
      species A in cell, B in cell;
      r1: A => B; cell * A * k_fwd;
      A = 10; B = 0; k_fwd = 1;
    end
    """
    problem = _problem(
        antimony, observable_formula="A", noise_formula="sd_obs", extra_params=["k_fwd"]
    )
    below = NOISE_SIGMA_MIN_POINTS - 1
    problem = problem.replace(
        measurement_df=pd.DataFrame(
            [
                {
                    OBSERVABLE_ID: "obs_A",
                    EXPERIMENT_ID: "",
                    TIME: float(t),
                    MEASUREMENT: 1.0 + 0.1 * t,
                }
                for t in range(below)
            ]
        )
    )
    entry = F.noise_sigma_report(problem, {"sd_obs": 1.0}).get("sd_obs")
    assert entry is not None, "the sigma must still be REPORTED, only left unjudged"
    assert entry["n_points"] == below
    assert entry["status"] == "unknown"
    assert entry["upper_bound_cap"] is None and entry["fraction_of_data_std"] is None
    assert not F.noise_sigma_inflated({"sd_obs": entry})
    assert not F.noise_sigma_collapsed({"sd_obs": entry})

    # ...and with enough points the gate engages again.
    problem = problem.replace(
        measurement_df=pd.DataFrame(
            [
                {
                    OBSERVABLE_ID: "obs_A",
                    EXPERIMENT_ID: "",
                    TIME: float(t),
                    MEASUREMENT: 1.0 + 0.1 * t,
                }
                for t in range(NOISE_SIGMA_MIN_POINTS + 3)
            ]
        )
    )
    entry = F.noise_sigma_report(problem, {"sd_obs": 1.0})["sd_obs"]
    assert entry["status"] != "unknown"
    assert entry["fraction_of_data_std"] is not None
