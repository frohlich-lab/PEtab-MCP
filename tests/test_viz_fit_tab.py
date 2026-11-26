"""How the Fit tab tells experiments apart (``petab_mcp.viz.fit_tab``): what each period changes,
and only the inputs that set an experiment apart."""

from __future__ import annotations

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
