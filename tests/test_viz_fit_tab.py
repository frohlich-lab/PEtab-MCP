"""How the Fit tab tells experiments apart (``petab_mcp.viz.fit_tab``): what each period changes,
and only the inputs that set an experiment apart."""

from __future__ import annotations

import numpy as np
import pandas as pd

from petab_mcp.problem import WorkingProblem
from petab_mcp.viz import fit_tab as F


def test_a_period_says_what_its_condition_changed(viz_store):
    problem = WorkingProblem.from_yaml(viz_store[1])
    table, preeq = F.periods(problem, "e_step", 8.0)
    assert preeq is None
    assert list(table["start"]) == [0.0, 4.0]
    assert list(table["end"]) == [4.0, 8.0 + 1e-9]
    assert list(table["changed"]) == ["A = 1", "A: 1 → 3"]

    table, preeq = F.periods(problem, "e_pre", 8.0)
    assert preeq == "low"
    assert list(table["changed"]) == ["A: 1 → 3"]


def test_a_title_names_only_what_sets_an_experiment_apart(monkeypatch):
    """Brannmark's experiments all name a second dose and its time, and only the two-step one sets
    them: naming every varying target put both in every title."""
    wide = pd.DataFrame(
        {"dose_1": [0.0, 0.1, 1.2], "time_2": [0.0, 0.0, 4.0], "dose_2": [0.0, 0.0, 8.8]},
        index=["d0", "d01", "two_steps"],
    )
    monkeypatch.setattr(F, "experiment_inputs", lambda problem: wide)
    titles = F.distinguishing_inputs(None)
    assert titles["d01"] == "dose_1=0.1"
    assert titles["two_steps"] == "dose_1=1.2, time_2=4, dose_2=8.8"


def test_each_period_is_a_band_that_says_what_it_sets_and_changes(viz_store):
    """The periods shade each plot, pre-equilibration first; a band's hover names its condition,
    what it changes and sets, and the formula of an input the experiment sets."""
    problem = WorkingProblem.from_yaml(viz_store[1])
    table, preeq = F.periods(problem, "e_step", 8.0)
    bands = F.period_bands(table, preeq, ["U(t) = A*time"])
    assert list(bands["left"]) == [0.0, 4.0] and bands["alpha"].nunique() == 2
    assert "period 2 of 2" in bands["info"][1] and "changes: A: 1 → 3" in bands["info"][1]
    assert "U(t) = A*time" in bands["info"][0]
    table, preeq = F.periods(problem, "e_pre", 8.0)
    bands = F.period_bands(table, preeq, [])
    assert bands["info"][0].startswith("<b>pre-equilibration</b>, condition low")
    assert bands["right"][0] == bands["left"][1]


def test_a_chi2_of_pure_noise_is_a_standard_normal_z_whatever_its_n():
    """The residuals view sets each observable's and experiment's χ² against a standard normal,
    whatever its number of measurements."""
    rng = np.random.default_rng(0)
    n = rng.integers(2, 80, 4000)
    chi2 = np.array([(rng.standard_normal(k) ** 2).sum() for k in n])
    z = F.chi2_z(chi2, n)
    assert abs(z.mean()) < 0.05 and abs(z.std() - 1) < 0.05
