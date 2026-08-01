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


def _ratio_entry(ratio, method, fitted=0.5, cap=10.0):
    # cap high (fitted < cap) so the data-std cap arm never fires -> isolates the ratio arm.
    return {"effective_sigma": fitted, "upper_bound_cap": cap, "data_std": 1.0,
            "fraction_of_data_std": 0.05, "fitted_over_data_noise": ratio,
            "data_noise_method": method}


def test_replicate_ratio_gate_flags_inflation_below_the_loose_cap():
    from petab_mcp.constants import NOISE_SIGMA_INFLATION_RATIO
    # A REPLICATE (pure-error) floor is trustworthy, so a fitted sigma well above it is inflated even
    # while below the far-looser data-std cap — the case the cap alone misses (blasi's gaming runs).
    assert _entry_inflated(_ratio_entry(NOISE_SIGMA_INFLATION_RATIO + 0.3, "replicate")) is True
    # advice IS produced for a replicate inflation (its quantified content is pinned separately, in
    # test_replicate_inflation_advice_is_quantified_and_reliable).
    assert noise_sigma_advice({"s": _ratio_entry(2.4, "replicate")}) is not None


def test_replicate_ratio_gate_passes_a_reference_quality_fit():
    # ratio ~1 (fitted sigma ~= the replicate noise floor) is a good fit -> not flagged.
    assert _entry_inflated(_ratio_entry(0.98, "replicate")) is False
    assert noise_sigma_advice({"s": _ratio_entry(0.98, "replicate")}) is None


def test_difference_floor_ratio_is_not_trusted():
    # The biased difference-based floor collides with legitimate references, so a high ratio THERE is
    # not flagged (only the conservative cap can) — schwen/oliveira must never trip on it.
    assert _entry_inflated(_ratio_entry(7.8, "difference")) is False
    assert noise_sigma_advice({"s": _ratio_entry(7.8, "difference")}) is None


def test_data_std_cap_still_flags_independent_of_method():
    # The pre-existing cap arm is untouched: fitted above its data-std cap is inflated whatever the
    # noise-estimate method (or absence of a ratio).
    assert _entry_inflated(_ratio_entry(1.0, "difference", fitted=20.0, cap=10.0)) is True
    assert _entry_inflated(_entry(0.5, 0.1, 2.5)) is True


def test_replicate_inflation_advice_is_quantified_and_reliable():
    from petab_mcp.fitting import NOISE_SIGMA_INFLATED_ADVICE
    a = noise_sigma_advice({"s": _ratio_entry(2.4, "replicate")})
    # the stronger, quantified message: names the ratio + that the floor is a measured pure-error one
    assert a is not None and "2.4x" in a and "PURE-ERROR" in a and "will NOT be accepted" in a
    assert a != NOISE_SIGMA_INFLATED_ADVICE   # NOT the generic cap-based string


def test_cap_only_inflation_uses_the_generic_advice():
    from petab_mcp.fitting import NOISE_SIGMA_INFLATED_ADVICE
    # inflated via the data-std cap on a DIFFERENCE floor -> no reliable ratio to quote -> generic.
    assert noise_sigma_advice({"s": _ratio_entry(1.0, "difference", fitted=20.0, cap=10.0)}) \
        == NOISE_SIGMA_INFLATED_ADVICE


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
    monkeypatch.setattr(R, "sigma_data_std", lambda p, pid, fitted_values=None: data_std[pid])
    monkeypatch.setattr(R, "sigma_upper_bound_cap", lambda p, pid, max_fraction=None, fitted_values=None, data_std=None: cap[pid])
    # bare (non-proportional) noise model: the effective sd IS the fitted value
    monkeypatch.setattr(R, "sigma_effective_value",
                        lambda p, pid, fitted, fitted_values=None: fitted)
    monkeypatch.setattr(R, "sigma_difference_noise_estimate", lambda p, pid, fitted_values=None: None)
    monkeypatch.setattr(R, "sigma_replicate_noise_estimate", lambda p, pid, fitted_values=None: None)
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
# schwen's published reference UNDER-FITS its Insulin replicates ~7.8x — its ELISA calibration
# rails at bounds (km_nExpID*=1e8, scaleElisa/fragments=1.0; see BENCHMARK_NOTES "schwen"), so the
# TRUE model legitimately trips the new replicate-ratio inflation gate. That is a reference-quality
# flag on a genuinely mediocre benchmark, not a gate error.
_TRUTH_KNOWN_FAILURES: set[str] = {"schwen"}


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
    monkeypatch.setattr(R, "sigma_data_std", lambda p, pid, fitted_values=None: 1.0)
    monkeypatch.setattr(R, "sigma_upper_bound_cap", lambda p, pid, max_fraction=None, fitted_values=None, data_std=None: 0.75)
    monkeypatch.setattr(R, "sigma_effective_value",
                        lambda p, pid, fitted, fitted_values=None: fitted)
    monkeypatch.setattr(R, "sigma_difference_noise_estimate", lambda p, pid, fitted_values=None: 1e6)  # absurd
    monkeypatch.setattr(R, "sigma_replicate_noise_estimate", lambda p, pid, fitted_values=None: None)  # no replicates -> fall back to difference
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


# --- a per-gel AMPLITUDE in the noise formula (fiedler) --------------------------------------
def _two_gel_problem(noise_formula="noiseParameter1_{o}*noiseParameter2_{o}",
                     gains=(1.0, 10.0), sigma=0.1, base=(1.0, 2.0, 3.0, 4.0)):
    """A fiedler-shaped problem: ONE sigma shared by two observables ("gels") whose measurements
    carry different multiplicative gains, each gain naming itself in BOTH the observable and the
    noise overrides. The underlying signal is identical, so everything that differs between the two
    observables is the gain — which is exactly what must NOT be read as data spread."""
    import pandas as pd
    import petab.v1 as petab
    obs_rows, meas_rows = [], []
    for i, gain in enumerate(gains, start=1):
        oid = f"o{i}"
        obs_rows.append({"observableId": oid, "observableFormula": f"x * observableParameter1_{oid}",
                         "observableTransformation": "lin",
                         "noiseFormula": noise_formula.format(o=oid)})
        for t, value in enumerate(base):
            meas_rows.append({"observableId": oid, "simulationConditionId": "c", "time": t,
                              "measurement": value * gain,
                              "observableParameters": f"s{i}", "noiseParameters": f"s{i};sigma"})
    par = pd.DataFrame({
        "parameterId": [f"s{i}" for i in range(1, len(gains) + 1)] + ["sigma"],
        "parameterScale": ["log10"] * (len(gains) + 1),
        "lowerBound": [1e-5] * (len(gains) + 1), "upperBound": [1e5] * (len(gains) + 1),
        "nominalValue": list(gains) + [sigma], "estimate": [1] * (len(gains) + 1),
    }).set_index("parameterId")
    return petab.Problem(observable_df=pd.DataFrame(obs_rows).set_index("observableId"),
                         measurement_df=pd.DataFrame(meas_rows), parameter_df=par)


def test_a_shared_sigma_with_per_gel_gains_is_summarisable():
    """Before this fix each gel implied its OWN effective sd (gain*sigma), the per-observable values
    disagreed, and the whole entry failed open — fiedler's `noise_sigma_effective` was `{}` and its
    sigmas were reported 'ok' on no evidence at all."""
    from petab_mcp import reconcile as R
    # NEITHER gain is 1.0, deliberately: with a leading gain of 1.0 the expected value coincides
    # with gel 1's UN-normalised sd, so "return the first observable's result" would also pass and
    # the assertion would not distinguish the fix from dropping the agreement requirement.
    p = _two_gel_problem(gains=(2.0, 10.0), sigma=0.1)
    fitted = {"s1": 2.0, "s2": 10.0, "sigma": 0.1}
    # The gain divides out, so the sd in the normalised coordinate IS the raw sigma — one number
    # for both gels, where the un-normalised formula gives 0.2 and 1.0.
    assert R.sigma_effective_value(p, "sigma", 0.1, fitted_values=fitted) == pytest.approx(0.1)


def test_the_gain_is_divided_out_of_the_data_std_too():
    """Both sides of the ratio must live in the same coordinate. Pooling the RAW measurements counts
    the between-gel gain difference as spread (on fiedler: 0.096 -> 0.688, ~7x), which is not noise
    the sigma has to cover, so the sigma reads far more plausible than it is."""
    from petab_mcp import reconcile as R
    p = _two_gel_problem(gains=(1.0, 10.0))
    fitted = {"s1": 1.0, "s2": 10.0, "sigma": 0.1}
    normalised = R.sigma_data_std(p, "sigma", fitted)
    # Both gels normalise back onto the SAME underlying series, so the pooled std equals that
    # series' own std — the gain contributes nothing.
    import numpy as np
    assert normalised == pytest.approx(float(np.std([1.0, 2.0, 3.0, 4.0], ddof=0)))
    # Sanity: the un-normalised pooling really is much larger, i.e. this test would fail loudly.
    raw = float(np.std([1.0, 2.0, 3.0, 4.0] + [10.0, 20.0, 30.0, 40.0], ddof=0))
    assert raw > 3 * normalised


def test_a_gain_that_is_not_a_pure_multiplier_fails_open():
    """Normalising is only valid for a factor that scales signal and noise ALIKE. A noise formula
    where the gel factor enters non-multiplicatively (sqrt(s^2 + sigma^2)) has no such coordinate,
    so the amplitude must be refused rather than divided out anyway."""
    from petab_mcp import reconcile as R
    p = _two_gel_problem(noise_formula="sqrt(noiseParameter1_{o}^2 + noiseParameter2_{o}^2)")
    rows = R._measurement_rows_for_sigma(p, "sigma")
    sub = rows[rows["observableId"] == "o1"]
    expr = R._noise_expr_and_observable(p, "sigma", sub)[0]
    assert R._observable_amplitude(p, "o1", expr) is None


def test_an_unevaluable_sigma_is_unknown_not_ok():
    """A fresh entry whose noiseFormula would not reduce to a number has no dimensionally valid
    comparison to make. The inflated/collapsed predicates both return False on a None, so falling
    through to 'ok' would report 'assessed and fine' about something never assessed."""
    from petab_mcp import reconcile as R
    from petab_mcp.fitting import _entry_collapsed, _entry_inflated, noise_sigma_report
    entry = {"fitted": 3.0, "effective_sigma": None, "upper_bound_cap": 0.5,
             "data_std": 1.0, "fraction_of_data_std": None}
    assert not _entry_inflated(entry) and not _entry_collapsed(entry)  # both silent...
    # ...so the STATUS is what has to carry it. Assert on the real report: an unevaluable formula
    # (effective None) with a perfectly good data_std must not read "ok".
    p = _two_gel_problem(gains=(2.0, 10.0))
    monkeypatch = pytest.MonkeyPatch()
    try:
        monkeypatch.setattr(R, "sigma_effective_value", lambda *a, **kw: None)
        report = noise_sigma_report(p, {"s1": 2.0, "s2": 10.0, "sigma": 0.1})
        assert report["sigma"]["data_std"] is not None      # the OTHER half is assessable
        assert report["sigma"]["status"] == "unknown", report["sigma"]
    finally:
        monkeypatch.undo()


def test_partial_gain_resolution_does_not_pool_two_coordinates():
    """If only SOME observables' gains resolve, normalising just those pools normalised values with
    un-normalised ones — a std over two different coordinates. That is worse than not normalising at
    all, because nothing about the resulting number looks wrong. Fail open instead."""
    from petab_mcp import reconcile as R
    p = _two_gel_problem(gains=(1.0, 10.0))
    # o2's gain becomes unresolvable: drop its nominal so neither the table nor the fit supplies it.
    p.parameter_df.loc["s2", "nominalValue"] = float("nan")
    assert R.sigma_data_std(p, "sigma", {"s1": 1.0, "sigma": 0.1}) is None
    # ...and with it supplied, the same problem DOES resolve — so the None above is the guard
    # firing, not the fixture being broken.
    assert R.sigma_data_std(p, "sigma", {"s1": 1.0, "s2": 10.0, "sigma": 0.1}) is not None


def test_the_report_wires_the_cap_to_the_fraction_it_reports():
    """Pin the ARGUMENT WIRING of the cap, which no stubbed test can see: `sigma_upper_bound_cap`
    takes `max_fraction` third and `data_std` fifth, so passing the std positionally would compute
    `data_std * data_std` instead of `0.75 * data_std` and silently stop the inflation gate firing.
    Asserting the identity through the real path is what catches that."""
    from petab_mcp.constants import NOISE_SIGMA_MAX_FRACTION_OF_DATA_STD as CAP
    from petab_mcp.fitting import noise_sigma_report
    p = _two_gel_problem(gains=(2.0, 10.0))
    entry = noise_sigma_report(p, {"s1": 2.0, "s2": 10.0, "sigma": 0.1})["sigma"]
    assert entry["upper_bound_cap"] == pytest.approx(CAP * entry["data_std"])
    assert entry["fraction_of_data_std"] == pytest.approx(
        entry["effective_sigma"] / entry["data_std"])


def _gain_and_ungained_problem():
    """One sigma shared by a GAINED observable and an ungained one — blasi's shape (a single sigma
    over many observables), where only some carry a gel scale."""
    import pandas as pd
    import petab.v1 as petab
    obs = pd.DataFrame([
        {"observableId": "o1", "observableFormula": "x * observableParameter1_o1",
         "observableTransformation": "lin",
         "noiseFormula": "noiseParameter1_o1*noiseParameter2_o1"},
        {"observableId": "o2", "observableFormula": "z", "observableTransformation": "lin",
         "noiseFormula": "noiseParameter1_o2"},
    ]).set_index("observableId")
    rows = []
    for t, v in enumerate([1.0, 2.0, 3.0, 4.0]):
        rows.append({"observableId": "o1", "simulationConditionId": "c", "time": t,
                     "measurement": v * 2, "observableParameters": "s1",
                     "noiseParameters": "s1;sigma"})
        rows.append({"observableId": "o2", "simulationConditionId": "c", "time": t,
                     "measurement": v, "observableParameters": None,
                     "noiseParameters": "sigma"})
    par = pd.DataFrame({"parameterId": ["s1", "sigma"], "parameterScale": ["log10"] * 2,
                        "lowerBound": [1e-5] * 2, "upperBound": [1e5] * 2,
                        "nominalValue": [2.0, 0.1], "estimate": [1, 1]}).set_index("parameterId")
    return petab.Problem(observable_df=obs, measurement_df=pd.DataFrame(rows), parameter_df=par)


def test_an_ungained_observable_does_not_take_the_gate_dark():
    """"This observable has no amplitude" (already on the common coordinate, gain 1) must not be
    conflated with "this amplitude will not reduce to a number". Conflating them made a sigma shared
    by a gained and an ungained observable return None, so the entry read 'unknown', the
    inflated/collapsed predicates could never fire, and the bound-widening advisory short-circuited
    on a None cap — the whole gate off, for a problem shape nothing is wrong with."""
    from petab_mcp import reconcile as R
    p = _gain_and_ungained_problem()
    assert R.sigma_data_std(p, "sigma", {"s1": 2.0, "sigma": 0.1}) is not None
    assert R.sigma_upper_bound_cap(p, "sigma", fitted_values={"s1": 2.0, "sigma": 0.1}) is not None


def test_a_proportional_gain_is_normalised_out_of_both_sides():
    """The two sides of the ratio must be normalised under the same rule. Keying the division off
    the noise FORM (additive only) while the data side divided for every form left the gain in the
    numerator for a proportional formula naming the gain, so an unchanged noise level read as
    `inflated` by exactly the gain."""
    import pandas as pd
    import petab.v1 as petab
    from petab_mcp import reconcile as R

    def build(s1):
        obs = pd.DataFrame([{
            "observableId": "o1", "observableFormula": "x * observableParameter1_o1",
            "observableTransformation": "lin",
            "noiseFormula": "noiseParameter1_o1*noiseParameter2_o1*o1"}]).set_index("observableId")
        rows = [{"observableId": "o1", "simulationConditionId": "c", "time": t,
                 "measurement": v * s1, "observableParameters": "s1",
                 "noiseParameters": "s1;cv"} for t, v in enumerate([1.0, 2.0, 3.0, 4.0])]
        par = pd.DataFrame({"parameterId": ["s1", "cv"], "parameterScale": ["log10"] * 2,
                            "lowerBound": [1e-5] * 2, "upperBound": [1e5] * 2,
                            "nominalValue": [s1, 0.1], "estimate": [1, 1]}).set_index("parameterId")
        return petab.Problem(observable_df=obs, measurement_df=pd.DataFrame(rows), parameter_df=par)

    fractions = []
    for s1 in (1.0, 10.0):  # identical underlying data and noise level, different gel gain
        p = build(s1)
        fit = {"s1": s1, "cv": 0.1}
        fractions.append(R.sigma_effective_value(p, "cv", 0.1, fitted_values=fit)
                         / R.sigma_data_std(p, "cv", fit))
    assert fractions[0] == pytest.approx(fractions[1]), fractions


def test_the_difference_estimate_shares_the_sigma_coordinate():
    """`fitted_over_data_noise` divides the (normalised) effective sigma by this estimate, so the
    estimate must difference the NORMALISED measurements too. Differencing raw values while the
    numerator is normalised put the ratio out by the gel gain — advising a correct model that its
    noise is orders of magnitude too small."""
    from petab_mcp import reconcile as R
    p = _two_gel_problem(gains=(2.0, 10.0), sigma=0.1)
    fit = {"s1": 2.0, "s2": 10.0, "sigma": 0.1}
    estimate = R.sigma_difference_noise_estimate(p, "sigma", fit)
    # The two gels normalise onto the SAME underlying series, so the estimate is that series' own
    # difference-based noise — it must not scale with the gains.
    doubled = R.sigma_difference_noise_estimate(
        _two_gel_problem(gains=(4.0, 20.0), sigma=0.1), "sigma",
        {"s1": 4.0, "s2": 20.0, "sigma": 0.1})
    assert estimate == pytest.approx(doubled), (estimate, doubled)
