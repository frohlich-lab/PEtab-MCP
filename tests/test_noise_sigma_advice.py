"""The agent-facing noise-sigma advice (fitting.noise_sigma_advice): a concise, actionable string
when the fit's ESTIMATED noise model is implausible, else None. Surfaced proactively in
get_fit_summary/get_fit_diagnosis and echoed by the harness finish gate, so the agent learns WHY a
good-looking BIC won't be accepted and fixes the mechanism instead of the noise. The inflated/
collapsed predicates themselves are covered in test_result_fields; this pins the advice wiring.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("petab", reason="petab_mcp dependency")

from petab_mcp.constants import NOISE_SIGMA_COLLAPSE_FRACTION  # noqa: E402
from petab_mcp.fitting import (  # noqa: E402
    NOISE_SIGMA_COLLAPSED_ADVICE,
    NOISE_SIGMA_INFLATED_ADVICE,
    _entry_collapsed,
    _entry_inflated,
    noise_sigma_advice,
    noise_sigma_advice_for,
    noise_sigma_report,
)


def _entry(fitted, cap, frac):
    return {"fitted": fitted, "upper_bound_cap": cap, "data_std": 1.0, "fraction_of_data_std": frac}


def test_none_when_plausible_or_empty():
    assert noise_sigma_advice(None) is None
    assert noise_sigma_advice({}) is None
    # below its cap, above the collapse floor -> plausible -> no advice.
    assert noise_sigma_advice({"s": _entry(0.08, 0.1, 0.4)}) is None


def test_inflated_advice():
    a = noise_sigma_advice({"s": _entry(0.5, 0.1, 2.5)})   # fitted above cap
    assert a == NOISE_SIGMA_INFLATED_ADVICE
    assert "inflated" in a and "will NOT be accepted" in a


def test_collapsed_advice():
    frac = NOISE_SIGMA_COLLAPSE_FRACTION / 2
    a = noise_sigma_advice({"s": _entry(frac, 0.5, frac)})  # fraction below floor
    assert a == NOISE_SIGMA_COLLAPSED_ADVICE
    assert "collapsed" in a and "will NOT be accepted" in a


def test_inflation_reported_first_when_both_present():
    # One inflated + one collapsed sigma (different observables): advice leads with inflation, the
    # more common misfit-masking case (matches noise_sigma_advice's documented ordering).
    both = {"hi": _entry(0.5, 0.1, 2.5),
            "lo": _entry(NOISE_SIGMA_COLLAPSE_FRACTION / 2, 0.5, NOISE_SIGMA_COLLAPSE_FRACTION / 2)}
    assert noise_sigma_advice(both) == NOISE_SIGMA_INFLATED_ADVICE


def test_shared_per_entry_predicates():
    """_entry_inflated/_entry_collapsed are the single source the top-level flags AND the report
    status both derive from, so they can't drift. Unassessable (None cap/frac) is neither."""
    assert _entry_inflated(_entry(0.5, 0.1, 2.5)) and not _entry_collapsed(_entry(0.5, 0.1, 2.5))
    lo = _entry(NOISE_SIGMA_COLLAPSE_FRACTION / 2, 0.5, NOISE_SIGMA_COLLAPSE_FRACTION / 2)
    assert _entry_collapsed(lo) and not _entry_inflated(lo)
    assert not _entry_inflated(_entry(0.08, 0.1, 0.4)) and not _entry_collapsed(_entry(0.08, 0.1, 0.4))
    assert not _entry_inflated({"fitted": 0.5, "upper_bound_cap": None})   # cap unassessable
    assert not _entry_collapsed({"fraction_of_data_std": None})            # frac unassessable


def test_advice_for_flags_derives_without_rescanning():
    assert noise_sigma_advice_for(True, False) == NOISE_SIGMA_INFLATED_ADVICE
    assert noise_sigma_advice_for(False, True) == NOISE_SIGMA_COLLAPSED_ADVICE
    assert noise_sigma_advice_for(True, True) == NOISE_SIGMA_INFLATED_ADVICE   # inflation first
    assert noise_sigma_advice_for(False, False) is None


def test_report_status_per_entry_including_unknown(monkeypatch):
    """noise_sigma_report tags each entry ok/inflated/collapsed/unknown. 'unknown' when the
    observable's std can't be computed (data_std None) — reported honestly, not as a reassuring
    'ok'. Reconcile's data-driven helpers are stubbed so this stays a pure unit test."""
    import petab_mcp.reconcile as R
    data_std = {"inf": 0.2, "col": 1.0, "fine": 0.2, "unk": None}
    cap = {"inf": 0.1, "col": 0.5, "fine": 0.1, "unk": None}
    monkeypatch.setattr(R, "is_parameter_estimated", lambda p, pid: True)
    monkeypatch.setattr(R, "plausibility_sigma_ids", lambda problem: set(data_std))
    monkeypatch.setattr(R, "sigma_data_std", lambda p, pid: data_std[pid])
    monkeypatch.setattr(R, "sigma_upper_bound_cap", lambda p, pid: cap[pid])
    # bare (non-proportional) noise model: the effective sd IS the fitted value
    monkeypatch.setattr(R, "sigma_effective_value",
                        lambda p, pid, fitted, fitted_values=None: fitted)
    monkeypatch.setattr(R, "sigma_difference_noise_estimate", lambda p, pid: None)
    monkeypatch.setattr(R, "is_parameter_estimated", lambda p, pid: True)
    monkeypatch.setattr(R, "plausibility_sigma_ids", lambda problem: set(data_std))
    rep = noise_sigma_report(object(), {"inf": 0.5, "col": NOISE_SIGMA_COLLAPSE_FRACTION / 10,
                                   "fine": 0.05, "unk": 0.3})
    assert rep["inf"]["status"] == "inflated"    # 0.5 > cap 0.1
    assert rep["col"]["status"] == "collapsed"   # well below the collapse floor
    assert rep["fine"]["status"] == "ok"         # below cap, above floor
    assert rep["unk"]["status"] == "unknown"     # data_std None -> unassessable, not 'ok'


# --- dimensional validity of the plausibility scale (issue #118) ---------------------------
def _sigma_scale_case(transformation, noise_formula, measurements, fitted):
    """Build a tiny PEtab problem and report the fitted sigma's fraction_of_data_std."""
    import pandas as pd
    import petab.v1 as petab
    from petab_mcp.fitting import noise_sigma_report
    obs = pd.DataFrame({"observableId": ["o"], "observableFormula": ["x"],
                        "observableTransformation": [transformation],
                        "noiseFormula": [noise_formula]}).set_index("observableId")
    meas = pd.DataFrame({"observableId": ["o"] * len(measurements),
                         "simulationConditionId": ["c"] * len(measurements),
                         "time": list(range(len(measurements))),
                         "measurement": measurements,
                         "noiseParameters": ["sd_o"] * len(measurements)})
    p = petab.Problem(observable_df=obs, measurement_df=meas)
    rep = noise_sigma_report(p, {"sd_o": fitted})
    return rep["sd_o"]["fraction_of_data_std"]


def test_log10_observable_sigma_is_judged_on_the_log_scale():
    """A log10 observable's sigma is a LOG-space sd, so it must be compared against the std of the
    TRANSFORMED data. Judging it against the raw linear std (the pre-#118 bug) made a normal ~20%
    relative error read as ~1e-7 of the data std -> a spurious 'collapsed'."""
    # Values spanning 4 decades: linear std is huge, log10 std is ~1.3.
    meas = [1e2, 1e3, 1e4, 1e5, 1e6]
    frac = _sigma_scale_case("log10", "noiseParameter1_o", meas, fitted=0.1)
    assert 0.01 < frac < 1.0, frac          # sane, same-scale ratio
    assert frac > 1e-3                      # NOT the ~1e-7 the linear-std bug produced


def test_proportional_noise_parameter_is_scaled_by_the_observable_magnitude():
    """With noiseFormula = noiseParameter1_o*x the parameter is a dimensionless CV, so its
    effective sd is cv * |typical value|. Comparing the bare CV against a data std (the pre-#118
    bug) flagged a correct 10% CV on small-magnitude data as wildly 'inflated'."""
    # Signal spans a decade (a real time course); the 10% CV is small against THAT spread.
    meas = [3e-4, 9e-4, 2.7e-3, 8e-3, 2.4e-2, 7e-2]   # small magnitudes, wide range
    frac = _sigma_scale_case("lin", "noiseParameter1_o*x", meas, fitted=0.10)  # a 10% CV
    # Assert against the ACTUAL gate, not a loose sanity bound: an assertion many times wider than
    # NOISE_SIGMA_MAX_FRACTION_OF_DATA_STD would stay green while the case it documents is flagged.
    from petab_mcp.constants import NOISE_SIGMA_MAX_FRACTION_OF_DATA_STD as CAP
    assert frac <= CAP, f"a correct 10% CV must not read as inflated: frac={frac} cap={CAP}"


# --- the truth must pass its own gate (the check that would have caught #118) ---------------
# Empty: the truth now passes its own gate on EVERY example. oliveira cleared once the gate
# stopped judging FIXED sigmas (its two ship estimate=0, so no fit can move them), and perelson
# cleared once this test began reading the truth from the reference FIT rather than from a
# parameters.tsv nominalValue that is a linear placeholder (1e5) for a log10-space sd.
_TRUTH_KNOWN_FAILURES: set[str] = set()


_EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"
# Repo-root-relative, NOT cwd-relative: a bare glob("examples/...") silently yields [] when pytest
# runs from anywhere but the repo root, and an empty parametrize is a GREEN skip — the check then
# never runs while looking like it passed.
_BENCHMARK_YAMLS = sorted(str(p) for p in _EXAMPLES_DIR.glob("*/problem/problem.yaml"))
# Only the 3 synthetic examples ship their problem/ in git; the benchmark ones are materialized by
# examples/setup.py and gitignored, so on a bare checkout (including CI) this parametrizes over the
# synthetic trio alone. Named here so the shortfall is visible rather than inferred from a count.
_SYNTHETIC = {"enzyme_kinetics", "predator_prey", "repressilator"}


def test_benchmark_examples_are_materialized_or_explicitly_absent():
    """Guard the guard: make the coverage of the ground-truth check below OBSERVABLE.

    If the benchmark problems are materialized, the parametrized test covers them; if not, this
    states so loudly instead of leaving a green run that silently checked only the synthetic trio.
    """
    found = {Path(p).parents[1].name for p in _BENCHMARK_YAMLS}
    assert found, (
        f"no examples/*/problem/problem.yaml under {_EXAMPLES_DIR} — the ground-truth gate check "
        "cannot run at all")
    if not (found - _SYNTHETIC):
        pytest.skip(
            "only the synthetic examples are materialized, so the ground-truth gate check below "
            "does NOT cover the benchmark problems (fiedler/oliveira/perelson/...). Run "
            "`python examples/setup.py` first to cover them.")


@pytest.mark.parametrize("yaml_path", _BENCHMARK_YAMLS)
def test_ground_truth_passes_its_own_plausibility_gate(yaml_path):
    """The data-generating parameter vector must not be flagged inflated/collapsed. A plausibility
    test that rejects the TRUTH is measuring the wrong thing — before #118 this failed on 7 of 16
    examples because the gate compared a log-space / CV parameter to a raw linear std."""
    petab = pytest.importorskip("petab.v1")
    from petab_mcp import fitting as F
    from petab_mcp import reconcile as R
    example = Path(yaml_path).parents[1].name
    problem = petab.Problem.from_yaml(yaml_path)
    ids = [i for i in R.plausibility_sigma_ids(problem) if i in problem.parameter_df.index]
    # Prefer the REFERENCE FIT's own sigmas as the truth. For an ESTIMATED sigma the table's
    # nominalValue is a starting point, not an answer, and perelson's is a linear placeholder (1e5)
    # for what is a log10-space sd — feeding that in tests the gate against a value that is not the
    # truth. The reference fit puts it at 0.1228. Fall back to nominalValue where a reference sigma
    # is absent (fixed sigmas, or a reference predating the field). Ids can differ between the
    # pristine collection problem and the materialized one, so a lone sigma is paired positionally.
    reference_path = Path(yaml_path).parent / "reference.json"
    reference_sigmas = {}
    if reference_path.exists():
        reference_sigmas = (json.loads(reference_path.read_text()) or {}).get("noise_sigma") or {}
    if len(reference_sigmas) == 1 and len(ids) == 1:
        reference_sigmas = {ids[0]: next(iter(reference_sigmas.values()))}
    truth = {i: float(reference_sigmas.get(i, problem.parameter_df.loc[i, "nominalValue"]))
             for i in ids}
    report = F.noise_sigma_report(problem, truth)
    flagged = F.noise_sigma_inflated(report) or F.noise_sigma_collapsed(report)
    if example in _TRUTH_KNOWN_FAILURES and flagged:
        # Conditional, NOT an imperative pytest.xfail(): an unconditional call raises before the
        # assert and can never XPASS, so a real fix would leave the suite green and the stale entry
        # would sit here forever with whatever diagnosis it was written with.
        pytest.xfail(f"{example}: known ground-truth/gate limitation (see #118)")
    assert not flagged, f"{example}: ground truth flagged -> " + str(
        {k: v.get("fraction_of_data_std") for k, v in report.items()})


# --- model-free difference-based noise estimate (advisory only) ------------------------------
def test_difference_noise_estimate_recovers_a_known_noise_level():
    """Rice-style successive-difference estimate on a SMOOTH mean plus known noise recovers the
    noise scale without any model or replicates — the property that makes it usable on a real
    problem, where neither a ground truth nor a reference fit exists."""
    import numpy as np
    import pandas as pd
    petab = pytest.importorskip("petab.v1")
    from petab_mcp.reconcile import sigma_difference_noise_estimate

    rng = np.random.default_rng(0)
    t = np.linspace(0.0, 10.0, 200)
    true_sigma = 0.05
    y = np.sin(t) * 2.0 + rng.normal(0.0, true_sigma, size=t.size)   # smooth mean + noise
    obs = pd.DataFrame({"observableId": ["o"], "observableFormula": ["x"],
                        "observableTransformation": ["lin"],
                        "noiseFormula": ["noiseParameter1_o"]}).set_index("observableId")
    meas = pd.DataFrame({"observableId": "o", "simulationConditionId": "c",
                         "time": t, "measurement": y, "noiseParameters": "sd_o"})
    est = sigma_difference_noise_estimate(
        petab.Problem(observable_df=obs, measurement_df=meas), "sd_o")
    # Densely sampled smooth mean -> the upward bias is small; allow a generous band.
    assert est is not None and 0.5 * true_sigma < est < 2.5 * true_sigma, est


def test_difference_noise_estimate_is_advisory_only(monkeypatch):
    """The estimate is REPORTED but must never gate: the inflated/collapsed predicates ignore it,
    so a wild estimate cannot flip a plausible sigma (it is biased high and pathological on
    cumulative series, so it must not be load-bearing)."""
    import petab_mcp.reconcile as R
    monkeypatch.setattr(R, "noise_sigma_ids", lambda problem: {"s"})
    monkeypatch.setattr(R, "sigma_data_std", lambda p, pid: 1.0)
    monkeypatch.setattr(R, "sigma_upper_bound_cap", lambda p, pid: 0.75)
    monkeypatch.setattr(R, "sigma_effective_value",
                        lambda p, pid, fitted, fitted_values=None: fitted)
    monkeypatch.setattr(R, "sigma_difference_noise_estimate", lambda p, pid: 1e6)  # absurd
    monkeypatch.setattr(R, "is_parameter_estimated", lambda p, pid: True)
    monkeypatch.setattr(R, "plausibility_sigma_ids", lambda problem: {"s"})
    rep = noise_sigma_report(object(), {"s": 0.4})               # plausible: 0.4 of a 1.0 std
    assert rep["s"]["data_noise_estimate"] == 1e6                 # reported ...
    assert rep["s"]["status"] == "ok"                             # ... but not gating
    from petab_mcp.fitting import noise_sigma_collapsed, noise_sigma_inflated
    assert not noise_sigma_inflated(rep) and not noise_sigma_collapsed(rep)


# --- the noise FORM is read from the symbolic expression, never from the data -----------------
def _form_case(noise_formula, measurements, transformation="lin", observable_formula="x"):
    import pandas as pd
    petab = pytest.importorskip("petab.v1")
    from petab_mcp import reconcile as R
    obs = pd.DataFrame({"observableId": ["o"], "observableFormula": [observable_formula],
                        "observableTransformation": [transformation],
                        "noiseFormula": [noise_formula]}).set_index("observableId")
    meas = pd.DataFrame({"observableId": "o", "simulationConditionId": "c",
                         "time": range(len(measurements)), "measurement": measurements,
                         "noiseParameters": "sd_o"})
    pdf = pd.DataFrame({"parameterScale": ["lin"], "lowerBound": [1e-9], "upperBound": [1e3],
                        "nominalValue": [0.1], "estimate": [1]},
                       index=pd.Index(["sd_o"], name="parameterId"))
    p = petab.Problem(observable_df=obs, measurement_df=meas, parameter_df=pdf)
    return p, R


def test_noise_form_read_from_the_expression_not_the_data():
    """additive vs proportional is decided by differentiating the formula. Deciding it from the
    DISPERSION of evaluated values (an earlier attempt) made the same model+sigma score differently
    on different measurement ranges."""
    tight, wide = [1.0, 1.1, 0.9, 1.05], [1e-3, 1e-1, 1e1, 1e3]
    for meas in (tight, wide):
        p, R = _form_case("noiseParameter1_o", meas)
        assert R.sigma_noise_form(p, "sd_o") == "additive", meas
        p, R = _form_case("noiseParameter1_o*x", meas)
        assert R.sigma_noise_form(p, "sd_o") == "proportional", meas


def test_mixed_noise_form_fails_open():
    """sqrt(additive**2 + (cv*y)**2) has no single scalar summary, so the form (and hence the
    coordinate and the effective sd) must be None rather than an arbitrary pick."""
    p, R = _form_case("sqrt(noiseParameter1_o**2 + (0.1*x)**2)", [1e-3, 1e-1, 1e1, 1e3])
    assert R.sigma_noise_form(p, "sd_o") is None
    assert R.sigma_noise_coordinate(p, "sd_o") is None
    assert R.sigma_effective_value(p, "sd_o", 0.1) is None


def test_effective_sigma_tracks_the_fitted_value():
    """The decisive regression guard: an earlier implementation dropped the sigma from its own
    formula, so effective_sigma was a constant function of the data and NO sigma could be judged."""
    p, R = _form_case("noiseParameter1_o*x", [1e-3, 1e-2, 1e-1, 1.0])
    assert R.sigma_effective_value(p, "sd_o", 0.1) == pytest.approx(0.1)
    assert R.sigma_effective_value(p, "sd_o", 5.0) == pytest.approx(5.0)


def test_composite_formula_multiplies_both_factors():
    """A sd built as scale*sigma (fiedler's noiseParameter1*noiseParameter2): neither factor is an
    sd alone, so the effective value must be their product."""
    import pandas as pd
    petab = pytest.importorskip("petab.v1")
    from petab_mcp import reconcile as R
    obs = pd.DataFrame({"observableId": ["o"], "observableFormula": ["x"],
                        "observableTransformation": ["lin"],
                        "noiseFormula": ["noiseParameter1_o*noiseParameter2_o"]}).set_index("observableId")
    meas = pd.DataFrame({"observableId": "o", "simulationConditionId": "c", "time": range(4),
                         "measurement": [1.0, 2.0, 3.0, 4.0], "noiseParameters": "scale;sd_o"})
    pdf = pd.DataFrame({"parameterScale": ["lin"] * 2, "lowerBound": [1e-9] * 2,
                        "upperBound": [1e3] * 2, "nominalValue": [4.0, 0.25], "estimate": [1, 1]},
                       index=pd.Index(["scale", "sd_o"], name="parameterId"))
    p = petab.Problem(observable_df=obs, measurement_df=meas, parameter_df=pdf)
    assert R.sigma_effective_value(p, "sd_o", 0.25) == pytest.approx(1.0)   # 4.0 * 0.25
