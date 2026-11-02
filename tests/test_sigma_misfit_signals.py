"""#195: where an estimated noise sigma absorbed a misfit — the per-sigma heterogeneity test, the
ranked fitted/data-noise ratio, and the caveat that no structure flag is weak evidence while sigmas,
scales or offsets are estimated. The statistic is pinned on its own, and the assembly end to end
through a real fit and ``get_fit_diagnosis``."""

from __future__ import annotations

import asyncio
import math
import shutil
from pathlib import Path

import numpy as np
import pytest

from petab_mcp import data_access
from petab_mcp import residual_tests as rt

DATA_DIR = Path(__file__).parent / "data" / "toy_v2"


# -- the statistic ---------------------------------------------------------------------------------
def _normal(n, scale=1.0, seed=0):
    return (np.random.default_rng(seed).normal(size=n) * scale).tolist()


def test_one_inflated_series_is_found_and_ranked_first():
    series = {f"e{i}": _normal(10, seed=i) for i in range(4)}
    series["e9"] = _normal(10, scale=4.0, seed=9)
    out = rt.sigma_heterogeneity(series)
    assert out["p"] < 1e-3 and out["n_series"] == 5
    assert out["worst"][0]["series"] == "e9" and out["worst"][0]["n"] == 10


def test_series_sharing_one_spread_are_not_ranked_as_heterogeneous():
    out = rt.sigma_heterogeneity({f"e{i}": _normal(10, seed=i) for i in range(5)})
    assert out["p"] > 0.05


def test_a_series_missed_by_a_constant_is_heterogeneous():
    """Centred on zero, not on each series' mean: a constant miss inflates the shared sigma as much
    as a shape miss, and a mean-centred test cannot see it."""
    series = {f"e{i}": _normal(10, seed=i) for i in range(4)}
    series["e9"] = [r + 3.0 for r in _normal(10, seed=9)]
    assert rt.sigma_heterogeneity(series)["p"] < 1e-3


def test_a_series_its_own_scale_and_offset_fit_exactly_does_not_decide_p():
    """Bartlett's log-variance reads a zero-variance series as p = 0."""
    series = {f"e{i}": _normal(10, seed=i) for i in range(4)}
    series["exact"] = [0.0, 0.0]
    assert rt.sigma_heterogeneity(series)["p"] > 0.05


def test_a_series_shorter_than_min_power_n_marks_the_test_low_power():
    series = {f"e{i}": _normal(10, seed=i) for i in range(3)}
    assert rt.sigma_heterogeneity(series)["low_power"] is False
    series["short"] = _normal(rt.MIN_POWER_N - 1, seed=9)
    assert rt.sigma_heterogeneity(series)["low_power"] is True


def test_too_few_series_is_no_verdict():
    assert rt.sigma_heterogeneity({"e1": _normal(10), "e2": [0.3]}) is None
    assert rt.sigma_heterogeneity({"e1": [None, math.nan, 1.0], "e2": _normal(5)}) is None


# -- the caveat ------------------------------------------------------------------------------------
def test_the_caveat_names_what_is_estimated(session):
    assert data_access.structure_tests_caveat(session.problem) is None  # the toy's sigma is fixed
    session.problem.parameter_df.loc["sigma_B", "estimate"] = True
    assert "Estimated noise sigmas absorb" in data_access.structure_tests_caveat(session.problem)
    odf = session.problem.observable_df.copy()
    odf.loc["obs_B", "observableFormula"] = "scale_B * B"
    pdf = session.problem.parameter_df
    pdf.loc["scale_B"] = pdf.loc["sigma_B"]
    session.problem = session.problem.replace(observable_df=odf, parameter_df=pdf)
    caveat = data_access.structure_tests_caveat(session.problem)
    assert "noise sigmas and observable scales/offsets absorb" in caveat


# -- end to end: a real fit, read through get_fit_diagnosis ----------------------------------------
_TIMES = (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0)
_A0 = {"e1": 1.0, "e2": 1.25, "e3": 1.5, "e4": 1.75, "e5": 2.0, "e6": 1.0}


def _problem(tmp_path, overshoot: float) -> str:
    """The toy A <=> B model's problem with six experiments (A(0) per ``_A0``) sharing ONE
    estimated ``sigma_B``, and an ``obs_A`` read at four times in e1 under its own estimated
    ``sigma_A``. The data are the toy model at its true rates, plus a transient B overshoot of
    ``overshoot * A(0)`` in e6 that a monotone A <=> B cannot follow. e6's ``obs_B`` is read twice,
    the second time under ``sigma_C``: the same design points as ``sigma_B``'s, told apart only by
    the ``noiseParameters`` override."""
    work = tmp_path / "problem"
    work.mkdir()
    shutil.copy(DATA_DIR / "model.ant", work / "model.ant")
    (work / "problem.yaml").write_text(
        (DATA_DIR / "problem.yaml").read_text()
        + "condition_files:\n- conditions.tsv\nexperiment_files:\n- experiments.tsv\n",
        encoding="utf-8",
    )
    (work / "observables.tsv").write_text(
        "observableId\tobservableFormula\tnoiseFormula\tobservablePlaceholders\tnoisePlaceholders\n"
        "obs_B\tB\tnoiseParameter1_obs_B\t\tnoiseParameter1_obs_B\nobs_A\tA\tsigma_A\t\t\n",
        encoding="utf-8",
    )
    (work / "parameters.tsv").write_text(
        (DATA_DIR / "parameters.tsv")
        .read_text()
        .replace(
            "sigma_B\tlin\t0.0001\t100.0\t0.1\tfalse", "sigma_B\tlin\t0.0001\t100.0\t0.1\ttrue"
        )
        + "sigma_A\tlin\t0.0001\t100.0\t0.1\ttrue\n"
        + "sigma_C\tlin\t0.0001\t100.0\t0.1\ttrue\n",
        encoding="utf-8",
    )
    (work / "conditions.tsv").write_text(
        "conditionId\ttargetId\ttargetValue\n"
        + "".join(f"c{e}\tA\t{a0}\n" for e, a0 in _A0.items()),
        encoding="utf-8",
    )
    (work / "experiments.tsv").write_text(
        "experimentId\ttime\tconditionId\n" + "".join(f"{e}\t0\tc{e}\n" for e in _A0),
        encoding="utf-8",
    )
    rng = np.random.default_rng(4)
    k = 0.5 + 0.1
    rows = ["observableId\texperimentId\ttime\tmeasurement\tnoiseParameters"]

    def b_of(e, a0, t):
        b = a0 * 0.5 / k * (1 - math.exp(-k * t))
        return b + overshoot * a0 * t * math.exp(-t) if e == "e6" else b

    for e, a0 in _A0.items():
        for t in _TIMES:
            b = b_of(e, a0, t)
            rows.append(f"obs_B\t{e}\t{t}\t{b + 0.02 * rng.normal()}\tsigma_B")
            if e == "e1" and t in (1.0, 2.0, 4.0, 8.0):
                rows.append(f"obs_A\t{e}\t{t}\t{a0 - b + 0.02 * rng.normal()}\t")
    rows += [
        f"obs_B\te6\t{t}\t{b_of('e6', _A0['e6'], t) + 0.02 * rng.normal()}\tsigma_C" for t in _TIMES
    ]
    (work / "measurements.tsv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    return str(work / "problem.yaml")


def _fitted_diagnosis(tmp_path, overshoot: float) -> dict:
    from petab_mcp.references import StubReferenceResolver
    from petab_mcp.server import create_server

    mcp = create_server(
        _problem(tmp_path, overshoot),
        provenance_dir=str(tmp_path / "prov"),
        resolver=StubReferenceResolver(),
        auto_fit=True,
    )

    def call(name, args):
        return asyncio.run(mcp.call_tool(name, args)).structured_content

    fit = call(
        "make_changes",
        {
            "parameters": [{"parameter_id": "k1", "lower_bound": 1e-3, "upper_bound": 1e3}],
            "justification": "fit",
            "n_starts": 4,
        },
    )
    assert fit["committed"] is True and fit["fit_status"] == "fit"
    return call("get_fit_diagnosis", {})


@pytest.mark.parametrize("overshoot", [0.0, 0.3])
def test_the_diagnosis_points_at_the_experiment_a_shared_sigma_absorbed(tmp_path, overshoot):
    diag = _fitted_diagnosis(tmp_path, overshoot)
    (het,) = [h for h in diag["sigma_heterogeneity"] if h["parameter_id"] == "sigma_B"]
    assert het["n_series"] == 6 and len(het["worst"]) == 3 and het["low_power"] is False
    # sigma_C's rows share e6's design points but are not sigma_B's
    assert all(w["n"] == len(_TIMES) for w in het["worst"])
    if overshoot:
        assert het["p"] < 0.01
        assert het["worst"][0]["series"] == "e6" and het["worst"][0]["rms"] > 1.5
    else:
        assert het["p"] > 0.05
    # the ratio is ranked, and sigma_A's four points are a low-power hint where the report says
    # `unknown`, not a dropped entry
    ratios = {r["parameter_id"]: r for r in diag["noise_sigma"]["fitted_over_data_noise"]}
    assert ratios["sigma_A"]["low_power"] is True
    assert diag["noise_sigma"]["report"]["sigma_A"]["status"] == "unknown"
    assert ratios["sigma_B"]["low_power"] is False and ratios["sigma_B"]["n_points"] == 60
    assert "noise sigmas absorb" in diag["structure_tests_caveat"]
