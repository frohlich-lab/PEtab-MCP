"""How the Fit tab tells experiments apart (``petab_mcp.viz.fit_tab``): what each period changes,
and only the inputs that set an experiment apart."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

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


def test_what_an_experiment_sets_is_plotted_under_its_data(viz_store):
    """The periods strip above the data named conditions; what a reader needs is the inputs over
    time, under the data on its time axis: A steps from 1 to 3 at t = 4, and pre-equilibrated at
    1 before a period of 3."""
    problem = WorkingProblem.from_yaml(viz_store[1])
    step = F.protocol(problem, (), "e_step", 8.0)
    ts, vs, stepped = step.curves["A"]
    assert stepped and list(ts) == [0.0, 4.0, 8.0] and list(vs) == [1.0, 3.0, 3.0]
    assert step.starts == [4.0] and step.constant == {}
    pre = F.protocol(problem, (), "e_pre", 8.0)
    assert pre.preeq == {"A": 1.0} and list(pre.curves["A"][1]) == [3.0, 3.0]


def test_an_input_that_ramps_within_a_period_is_drawn_from_its_formula():
    """Fujita's EGF is no condition target but inputs.tsv's ramp over three of them; the targets
    are shown as the curve they make, not listed."""
    from types import SimpleNamespace

    from petab_mcp.protocols import Input

    problem = SimpleNamespace(
        experiment_df=pd.DataFrame({"experimentId": ["e"], "time": [0.0], "conditionId": ["c"]}),
        condition_df=pd.DataFrame(
            {
                "conditionId": ["c"] * 4,
                "targetId": ["EGF_0", "EGF_rate", "EGF_end", "init_S"],
                "targetValue": [1.0, 0.5, 10.0, "S_tot"],
            }
        ),
    )
    egf = Input(
        "EGF",
        "piecewise(EGF_rate*time + EGF_0, time <= EGF_end, 0)",
        ("EGF_0", "EGF_rate", "EGF_end"),
    )
    proto = F.protocol(problem, (egf,), "e", 20.0)
    ts, vs, stepped = proto.curves["EGF(t)"]
    assert not stepped and vs[0] == 1.0 and vs[-1] == 0.0
    assert np.interp(4.0, ts, vs) == pytest.approx(3.0, abs=0.01)
    assert proto.constant == {"init_S": "S_tot"}


def test_the_settings_line_puts_what_sets_an_experiment_apart_first():
    proto = F.Protocol(constant={"init_S2": "S2tot", "EGF_0": "10"})
    assert F.settings_line(proto, "EGF_0=10") == "Set throughout: **EGF_0 = 10**, init_S2 = S2tot"
