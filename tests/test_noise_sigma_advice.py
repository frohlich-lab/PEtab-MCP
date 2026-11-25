"""The agent-facing noise-sigma advice (fitting.noise_sigma_advice): a concise, actionable string
when the fit's ESTIMATED noise model is implausible, else None. Surfaced proactively in
get_fit_summary/get_fit_diagnosis and echoed by a client's finish gate, so the agent learns WHY a
good-looking BIC won't be accepted and fixes the mechanism instead of the noise. The inflated/
collapsed predicates themselves are covered in test_result_fields; this pins the advice wiring.
"""

from __future__ import annotations

import math

import pytest

from petab_mcp import residual_tests as rt  # noqa: E402

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


def _upgraded(observable_df=None, measurement_df=None, parameter_df=None):
    """The v1-shaped tables these cases are written in, upgraded the way a shipped v1 problem is
    (:func:`petab_mcp.problem._from_v1`): libpetab's observable conversion -- which itself puts the
    exact ``log(10)`` factor on a log10 observable's noise formula (PEtab-dev/libpetab-python#535),
    and warns that it did -- and one simulation condition that changes nothing, i.e. no
    experiment. Adding the factor here as well, as this did until libpetab took it on, scaled every
    log10 fixture by ln(10) twice (pinned by the test below)."""
    import warnings

    from petab.v2.petab1to2 import v1v2_observable_df

    from petab_mcp.problem import WorkingProblem

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "PEtab v2 does not support log10", UserWarning)
        odf = v1v2_observable_df(observable_df).set_index("observableId")
    return WorkingProblem(
        observable_df=odf,
        measurement_df=measurement_df.drop(columns="simulationConditionId").assign(experimentId=""),
        parameter_df=parameter_df,
    )


def test_the_fixtures_upgrade_a_log10_observable_as_the_server_does(tmp_path):
    """`_upgraded` stands in for `_from_v1`, so a fixture's effective sigma is the one a served
    problem has. It once kept its own log(10) after libpetab began adding it, and every log10 case
    here was judged on a sigma ln(10) times too large while its broad assertions stayed green."""
    import shutil
    from pathlib import Path

    import pandas as pd

    from petab_mcp.problem import WorkingProblem

    shutil.copytree(Path(__file__).parent / "data" / "toy_v1", tmp_path, dirs_exist_ok=True)
    obs = pd.DataFrame(
        {
            "observableId": ["obs_B"],
            "observableFormula": ["B"],
            "noiseFormula": ["sigma_B"],
            "observableTransformation": ["log10"],
        }
    )
    obs.to_csv(tmp_path / "observables.tsv", sep="\t", index=False)
    served = WorkingProblem.from_yaml(tmp_path / "problem.yaml").observable_df
    meas = pd.read_csv(tmp_path / "measurements.tsv", sep="\t")
    fixture = _upgraded(observable_df=obs.set_index("observableId"), measurement_df=meas)
    assert fixture.observable_df.loc["obs_B", "noiseFormula"] == served.loc["obs_B", "noiseFormula"]


def _entry(fitted, cap, frac):
    return {"fitted": fitted, "upper_bound_cap": cap, "data_std": 1.0, "fraction_of_data_std": frac}


def test_none_when_plausible_or_empty():
    assert noise_sigma_advice(None) is None
    assert noise_sigma_advice({}) is None
    # below its cap, above the collapse floor -> plausible -> no advice.
    assert noise_sigma_advice({"s": _entry(0.08, 0.1, 0.4)}) is None


def test_inflated_advice():
    a = noise_sigma_advice({"s": _entry(0.5, 0.1, 2.5)})  # fitted above cap
    assert a == NOISE_SIGMA_INFLATED_ADVICE
    assert "inflated" in a and "will NOT be accepted" in a


def _ratio_entry(ratio, method, fitted=0.5, cap=10.0):
    # cap high (fitted < cap) so the data-std cap arm never fires -> the ratio is the ONLY signal.
    return {
        "effective_sigma": fitted,
        "upper_bound_cap": cap,
        "data_std": 1.0,
        "fraction_of_data_std": 0.05,
        "fitted_over_data_noise": ratio,
        "data_noise_method": method,
    }


@pytest.mark.parametrize("ratio", [0.98, 1.476, 2.4, 7.83, 100.0])
def test_the_pure_error_ratio_never_gates_however_large(ratio):
    """No ratio against a noise FLOOR is an inflation verdict — not even against the unbiased
    replicate (pure-error) one (the arm #140 added is gone; see fitting._entry_inflated).

    The FORM is what is wrong, so this spans the measured range instead of bracketing a threshold:
    0.98 is a steady-state problem's TRUE model, 1.476 a decisively wrong fit of it (ΔBIC +194) that
    the shipped 1.5 was ALSO silent on, 2.4 a noise-gaming fit of it, 7.83 another problem's own
    REFERENCE. No constant separates those; the per-design-point lack-of-fit test does (see below).
    """
    for method in ("replicate", "difference"):
        assert _entry_inflated(_ratio_entry(ratio, method)) is False, method
        assert noise_sigma_advice({"s": _ratio_entry(ratio, method)}) is None, method


# --- where the deleted arm's coverage went (#140 -> here) -------------------------------------
# The two fits of that steady-state problem #140 was written for. Each is (label, n_structural,
# sigma/pure-error ratio, [(n_i, z_i) per replicated design point]) with
# z_i = (ȳ_i - f_i)/(s_i/√n_i). Both sat at 0.28/0.38 of the data-std cap, so the deleted ratio arm
# was their ONLY guard. n_i/s_i are model-free (the problem's measurement table); z_i is
# reconstructed from each fit's recorded mean_weighted_residual × its fitted sigma (exact: one
# design point per observable, one shared sigma). Literals because the archived fits are not in this
# repository. #144's own tests already cover the ΔBIC +194 fit and the truth from their real
# predictions — these two are this change's obligation, not that one's.
_ARCHIVED_GAMING_FITS = [
    (
        "0a19fa0c k=6 ΔBIC +437",
        5,
        2.414,
        [
            (18, -48.847),
            (18, -36.338),
            (18, 14.389),
            (12, -9.512),
            (18, 9.518),
            (18, -17.379),
            (18, -10.129),
            (18, -12.232),
            (12, 0.015),
            (18, -0.713),
            (18, -0.206),
            (18, 40.836),
            (12, 4.916),
            (18, 0.848),
            (18, -8.477),
        ],
    ),
    (
        "52a4101b k=23 ΔBIC +386",
        22,
        1.812,
        [
            (12, -6.113),
            (18, 6.602),
            (18, 75.441),
            (18, -5.277),
            (18, 6.554),
            (18, 2.219),
            (12, -2.333),
            (18, 6.467),
            (18, 12.322),
            (18, -9.410),
            (18, -5.639),
            (18, -11.513),
            (18, -0.827),
            (12, -4.037),
            (18, 1.184),
        ],
    ),
]


def _lof_points(points):
    """The measured deviations as a lack-of-fit family: s = 1 and ȳ_i - f_i = z_i/√n_i reproduce each
    z_i exactly, so the statistic is pinned on real numbers without the archived models."""
    return [
        {"label": f"point{i}", "n": n, "ybar": z / math.sqrt(n), "s": 1.0, "prediction": 0.0}
        for i, (n, z) in enumerate(points)
    ]


def test_the_replicate_lof_arm_condemns_the_fit_the_ratio_arm_was_written_for():
    """ΔBIC +437 at k=6: condemned at alpha 0.01, where the deleted ratio arm had it at 2.41 against
    a 1.5 threshold. This is the coverage transfer.

    The literals are z against each point's OWN scatter. Against the scatter pooled over the shared
    sigma each z moves by s_i/s_g, which on that problem is at least 0.14 (its tightest point, 0.0361
    against 0.2579) — and the fit is still condemned with every z shrunk by that much."""
    label, n_structural, _ratio, points = _ARCHIVED_GAMING_FITS[0]
    out = rt.replicate_lack_of_fit(_lof_points(points))
    assert out["available"] is True and out["status"] == "flagged", label
    assert out["max_abs_z"] == pytest.approx(48.85, abs=0.01)
    pooled_df = 12 * 17 + 3 * 11  # twelve points of 18 replicates, three of 12
    assert out["min_p_holm"] == pytest.approx(
        15 * rt.student_t_two_sided_p(48.847, pooled_df), rel=1e-2
    )
    shrunk = [(n, z * 0.036101 / 0.257908) for n, z in points]
    worst_case = rt.replicate_lack_of_fit(_lof_points(shrunk))
    assert worst_case["status"] == "flagged" and len(worst_case["flagged"]) == 3, worst_case


def test_the_other_one_is_the_parsimony_gate_s_and_the_arm_says_so():
    """ΔBIC +386 at k=23 does NOT transfer to this arm, and must not be assumed to: 22 structural
    parameters against 15 usable design points trips the vacuity guard, so the verdict is NO
    EVIDENCE (available False), not a pass.

    That is the correct handoff rather than a hole. The problem's data are 15 distinct design
    points, so a model with 22 structural parameters has a Jacobian of rank <= 15 and a
    rank-DEFICIENT FIM by construction — which is the parsimony refusal of a client's finish gate
    (#135), aimed at exactly this and landed only AFTER this fit. Both this guard and that gate turn
    on the same condition (n_structural >= n_points), which is what makes the pair complete. What
    the deleted ratio arm contributed here was a second-best proxy ("your noise is 1.8x the floor")
    for a diagnosis that is really "you have 22 parameters for 15 points".

    Asserted on the MERGE, which is where the vacuity guard lives now: the problem has one noise
    family, so its pooled point count is that family's, but the count the parameters are
    subtracted from is a whole-problem one (data_access._merge_lof_verdicts)."""
    from petab_mcp import data_access

    label, n_structural, _ratio, points = _ARCHIVED_GAMING_FITS[1]
    arm = rt.replicate_lack_of_fit(_lof_points(points))
    out = data_access._merge_lof_verdicts([arm], {}, n_structural=n_structural)
    assert out["available"] is False and out["status"] == "vacuous", label
    assert out["n_flagged"] == 0 and out["covered_observables"] == []  # no evidence, nothing passed
    # ... and it is ONLY the parameter count: the same points at a parsimonious k are condemned.
    assert data_access._merge_lof_verdicts([arm], {}, n_structural=6)["status"] == "flagged"


def test_data_std_cap_still_flags_independent_of_method():
    # The pre-existing cap arm is untouched: fitted above its data-std cap is inflated whatever the
    # noise-estimate method (or absence of a ratio).
    assert _entry_inflated(_ratio_entry(1.0, "difference", fitted=20.0, cap=10.0)) is True
    assert _entry_inflated(_ratio_entry(1.0, "replicate", fitted=20.0, cap=10.0)) is True
    assert _entry_inflated(_entry(0.5, 0.1, 2.5)) is True


def test_cap_only_inflation_uses_the_generic_advice():
    from petab_mcp.fitting import NOISE_SIGMA_INFLATED_ADVICE

    # The cap arm is the only inflation arm, so it is also the only advice: one string, whatever
    # noise floor the report happens to carry.
    for method in ("difference", "replicate"):
        assert (
            noise_sigma_advice({"s": _ratio_entry(2.4, method, fitted=20.0, cap=10.0)})
            == NOISE_SIGMA_INFLATED_ADVICE
        ), method


def test_collapsed_advice():
    frac = NOISE_SIGMA_COLLAPSE_FRACTION / 2
    a = noise_sigma_advice({"s": _entry(frac, 0.5, frac)})  # fraction below floor
    assert a == NOISE_SIGMA_COLLAPSED_ADVICE
    assert "collapsed" in a and "will NOT be accepted" in a


def test_inflation_reported_first_when_both_present():
    # One inflated + one collapsed sigma (different observables): advice leads with inflation, the
    # more common misfit-masking case (matches noise_sigma_advice's documented ordering).
    both = {
        "hi": _entry(0.5, 0.1, 2.5),
        "lo": _entry(NOISE_SIGMA_COLLAPSE_FRACTION / 2, 0.5, NOISE_SIGMA_COLLAPSE_FRACTION / 2),
    }
    assert noise_sigma_advice(both) == NOISE_SIGMA_INFLATED_ADVICE


def test_shared_per_entry_predicates():
    """_entry_inflated/_entry_collapsed are the single source the top-level flags AND the report
    status both derive from, so they can't drift. Unassessable (None cap/frac) is neither."""
    assert _entry_inflated(_entry(0.5, 0.1, 2.5)) and not _entry_collapsed(_entry(0.5, 0.1, 2.5))
    lo = _entry(NOISE_SIGMA_COLLAPSE_FRACTION / 2, 0.5, NOISE_SIGMA_COLLAPSE_FRACTION / 2)
    assert _entry_collapsed(lo) and not _entry_inflated(lo)
    assert not _entry_inflated(_entry(0.08, 0.1, 0.4)) and not _entry_collapsed(
        _entry(0.08, 0.1, 0.4)
    )
    assert not _entry_inflated({"fitted": 0.5, "upper_bound_cap": None})  # cap unassessable
    assert not _entry_collapsed({"fraction_of_data_std": None})  # frac unassessable


def test_advice_for_flags_derives_without_rescanning():
    assert noise_sigma_advice_for(True, False) == NOISE_SIGMA_INFLATED_ADVICE
    assert noise_sigma_advice_for(False, True) == NOISE_SIGMA_COLLAPSED_ADVICE
    assert noise_sigma_advice_for(True, True) == NOISE_SIGMA_INFLATED_ADVICE  # inflation first
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
    monkeypatch.setattr(
        R,
        "sigma_upper_bound_cap",
        lambda p, pid, max_fraction=None, fitted_values=None, data_std=None: cap[pid],
    )
    # bare (non-proportional) noise model: the effective sd IS the fitted value
    monkeypatch.setattr(
        R, "sigma_effective_value", lambda p, pid, fitted, fitted_values=None: fitted
    )
    monkeypatch.setattr(
        R, "sigma_difference_noise_estimate", lambda p, pid, fitted_values=None: None
    )
    monkeypatch.setattr(
        R, "sigma_replicate_noise_estimate", lambda p, pid, fitted_values=None: None
    )
    monkeypatch.setattr(R, "is_parameter_estimated", lambda p, pid: True)
    monkeypatch.setattr(R, "plausibility_sigma_ids", lambda problem: set(data_std))
    rep = noise_sigma_report(
        object(), {"inf": 0.5, "col": NOISE_SIGMA_COLLAPSE_FRACTION / 10, "fine": 0.05, "unk": 0.3}
    )
    assert rep["inf"]["status"] == "inflated"  # 0.5 > cap 0.1
    assert rep["col"]["status"] == "collapsed"  # well below the collapse floor
    assert rep["fine"]["status"] == "ok"  # below cap, above floor
    assert rep["unk"]["status"] == "unknown"  # data_std None -> unassessable, not 'ok'


# --- dimensional validity of the plausibility scale (issue #118) ---------------------------
def _sigma_scale_case(transformation, noise_formula, measurements, fitted):
    """Build a tiny PEtab problem and report the fitted sigma's fraction_of_data_std."""
    import pandas as pd
    from petab_mcp.fitting import noise_sigma_report

    obs = pd.DataFrame(
        {
            "observableId": ["o"],
            "observableFormula": ["x"],
            "observableTransformation": [transformation],
            "noiseFormula": [noise_formula],
        }
    ).set_index("observableId")
    meas = pd.DataFrame(
        {
            "observableId": ["o"] * len(measurements),
            "simulationConditionId": ["c"] * len(measurements),
            "time": list(range(len(measurements))),
            "measurement": measurements,
            "noiseParameters": ["sd_o"] * len(measurements),
        }
    )
    p = _upgraded(observable_df=obs, measurement_df=meas)
    rep = noise_sigma_report(p, {"sd_o": fitted})
    return rep["sd_o"]["fraction_of_data_std"]


def test_log10_observable_sigma_is_judged_on_the_log_scale():
    """A log10 observable's sigma is a LOG-space sd, so it must be compared against the std of the
    TRANSFORMED data. Judging it against the raw linear std (the pre-#118 bug) made a normal ~20%
    relative error read as ~1e-7 of the data std -> a spurious 'collapsed'."""
    # Values spanning 4 decades: linear std is huge, log10 std is ~1.3.
    import statistics

    meas = [1e2, 1e3, 1e4, 1e5, 1e6]
    frac = _sigma_scale_case("log10", "noiseParameter1_o", meas, fitted=0.1)
    # Exactly the log10-space sigma over the log10 data's (population) std, whatever scale the two
    # are carried in. The range check this replaces passed 0.163 too: the ratio with ln(10) on the
    # noise twice, which every log10 fixture had once libpetab added the factor itself.
    assert frac == pytest.approx(0.1 / statistics.pstdev([math.log10(m) for m in meas]))
    assert frac > 1e-3  # NOT the ~1e-7 the linear-std bug produced


def test_a_flagged_sigma_states_its_limit_in_its_own_units():
    """#329: lucarelli's sd_Bmp4 read fitted 0.164 beside effective_sigma 0.377 and cap 0.316, a
    log10 sd beside ln ones, and agents mis-sized the change. The entry now states the value of
    the parameter itself at which the verdict turns."""
    import pandas as pd

    from petab_mcp import reconcile as R

    obs = pd.DataFrame(
        {"observableFormula": ["x"], "observableTransformation": ["log10"]},
        index=pd.Index(["o"], name="observableId"),
    ).assign(noiseFormula="noiseParameter1_o")
    meas = [1e2, 1e3, 1e4, 1e5, 1e6]
    meas = pd.DataFrame(
        {
            "observableId": "o",
            "simulationConditionId": "c",
            "time": range(len(meas)),
            "measurement": meas,
            "noiseParameters": "sd_o",
        }
    )
    p = _upgraded(observable_df=obs, measurement_df=meas)
    for fitted, key, limit in (
        (5.0, "fitted_cap", "upper_bound_cap"),
        (1e-3, "fitted_floor", None),
    ):
        entry = noise_sigma_report(p, {"sd_o": fitted})["sd_o"]
        at = entry[key]
        assert (at < fitted) is (key == "fitted_cap")
        bound = entry[limit] if limit else NOISE_SIGMA_COLLAPSE_FRACTION * entry["data_std"]
        assert R.sigma_effective_value(p, "sd_o", at) == pytest.approx(bound)
    assert "fitted_cap" not in noise_sigma_report(p, {"sd_o": 0.1})["sd_o"]


def test_proportional_noise_parameter_is_scaled_by_the_observable_magnitude():
    """With noiseFormula = noiseParameter1_o*x the parameter is a dimensionless CV, so its
    effective sd is cv * |typical value|. Comparing the bare CV against a data std (the pre-#118
    bug) flagged a correct 10% CV on small-magnitude data as wildly 'inflated'."""
    # Signal spans a decade (a real time course); the 10% CV is small against THAT spread.
    meas = [3e-4, 9e-4, 2.7e-3, 8e-3, 2.4e-2, 7e-2]  # small magnitudes, wide range
    frac = _sigma_scale_case("lin", "noiseParameter1_o*x", meas, fitted=0.10)  # a 10% CV
    # Assert against the ACTUAL gate, not a loose sanity bound: an assertion many times wider than
    # NOISE_SIGMA_MAX_FRACTION_OF_DATA_STD would stay green while the case it documents is flagged.
    from petab_mcp.constants import NOISE_SIGMA_MAX_FRACTION_OF_DATA_STD as CAP

    assert frac <= CAP, f"a correct 10% CV must not read as inflated: frac={frac} cap={CAP}"


# --- model-free difference-based noise estimate (advisory only) ------------------------------
def test_difference_noise_estimate_recovers_a_known_noise_level():
    """Rice-style successive-difference estimate on a SMOOTH mean plus known noise recovers the
    noise scale without any model or replicates — the property that makes it usable on a real
    problem, where neither a ground truth nor a reference fit exists."""
    import numpy as np
    import pandas as pd

    from petab_mcp.reconcile import sigma_difference_noise_estimate

    rng = np.random.default_rng(0)
    t = np.linspace(0.0, 10.0, 200)
    true_sigma = 0.05
    y = np.sin(t) * 2.0 + rng.normal(0.0, true_sigma, size=t.size)  # smooth mean + noise
    obs = pd.DataFrame(
        {
            "observableId": ["o"],
            "observableFormula": ["x"],
            "observableTransformation": ["lin"],
            "noiseFormula": ["noiseParameter1_o"],
        }
    ).set_index("observableId")
    meas = pd.DataFrame(
        {
            "observableId": "o",
            "simulationConditionId": "c",
            "time": t,
            "measurement": y,
            "noiseParameters": "sd_o",
        }
    )
    est = sigma_difference_noise_estimate(_upgraded(observable_df=obs, measurement_df=meas), "sd_o")
    # Densely sampled smooth mean -> the upward bias is small; allow a generous band.
    assert est is not None and 0.5 * true_sigma < est < 2.5 * true_sigma, est


def test_difference_noise_estimate_is_advisory_only(monkeypatch):
    """The estimate is REPORTED but must never gate: the inflated/collapsed predicates ignore it,
    so a wild estimate cannot flip a plausible sigma (it is biased high and pathological on
    cumulative series, so it must not be load-bearing)."""
    import petab_mcp.reconcile as R

    monkeypatch.setattr(R, "noise_sigma_ids", lambda problem: {"s"})
    monkeypatch.setattr(R, "sigma_data_std", lambda p, pid, fitted_values=None: 1.0)
    monkeypatch.setattr(
        R,
        "sigma_upper_bound_cap",
        lambda p, pid, max_fraction=None, fitted_values=None, data_std=None: 0.75,
    )
    monkeypatch.setattr(
        R, "sigma_effective_value", lambda p, pid, fitted, fitted_values=None: fitted
    )
    monkeypatch.setattr(
        R, "sigma_difference_noise_estimate", lambda p, pid, fitted_values=None: 1e6
    )  # absurd
    monkeypatch.setattr(
        R, "sigma_replicate_noise_estimate", lambda p, pid, fitted_values=None: None
    )  # no replicates -> fall back to difference
    monkeypatch.setattr(R, "is_parameter_estimated", lambda p, pid: True)
    monkeypatch.setattr(R, "plausibility_sigma_ids", lambda problem: {"s"})
    rep = noise_sigma_report(object(), {"s": 0.4})  # plausible: 0.4 of a 1.0 std
    assert rep["s"]["data_noise_estimate"] == 1e6  # reported ...
    assert rep["s"]["status"] == "ok"  # ... but not gating
    from petab_mcp.fitting import noise_sigma_collapsed, noise_sigma_inflated

    assert not noise_sigma_inflated(rep) and not noise_sigma_collapsed(rep)


# --- the noise FORM is read from the symbolic expression, never from the data -----------------
def _form_case(noise_formula, measurements, transformation="lin", observable_formula="x"):
    import pandas as pd

    from petab_mcp import reconcile as R

    obs = pd.DataFrame(
        {
            "observableId": ["o"],
            "observableFormula": [observable_formula],
            "observableTransformation": [transformation],
            "noiseFormula": [noise_formula],
        }
    ).set_index("observableId")
    meas = pd.DataFrame(
        {
            "observableId": "o",
            "simulationConditionId": "c",
            "time": range(len(measurements)),
            "measurement": measurements,
            "noiseParameters": "sd_o",
        }
    )
    pdf = pd.DataFrame(
        {
            "parameterScale": ["lin"],
            "lowerBound": [1e-9],
            "upperBound": [1e3],
            "nominalValue": [0.1],
            "estimate": [1],
        },
        index=pd.Index(["sd_o"], name="parameterId"),
    )
    p = _upgraded(observable_df=obs, measurement_df=meas, parameter_df=pdf)
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
    p, R = _form_case("sqrt(noiseParameter1_o^2 + (0.1*x)^2)", [1e-3, 1e-1, 1e1, 1e3])
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
    """A sd built as scale*sigma (noiseParameter1*noiseParameter2): neither factor is an sd alone,
    so the effective value must be their product."""
    import pandas as pd

    from petab_mcp import reconcile as R

    obs = pd.DataFrame(
        {
            "observableId": ["o"],
            "observableFormula": ["x"],
            "observableTransformation": ["lin"],
            "noiseFormula": ["noiseParameter1_o*noiseParameter2_o"],
        }
    ).set_index("observableId")
    meas = pd.DataFrame(
        {
            "observableId": "o",
            "simulationConditionId": "c",
            "time": range(4),
            "measurement": [1.0, 2.0, 3.0, 4.0],
            "noiseParameters": "scale;sd_o",
        }
    )
    pdf = pd.DataFrame(
        {
            "parameterScale": ["lin"] * 2,
            "lowerBound": [1e-9] * 2,
            "upperBound": [1e3] * 2,
            "nominalValue": [4.0, 0.25],
            "estimate": [1, 1],
        },
        index=pd.Index(["scale", "sd_o"], name="parameterId"),
    )
    p = _upgraded(observable_df=obs, measurement_df=meas, parameter_df=pdf)
    assert R.sigma_effective_value(p, "sd_o", 0.25) == pytest.approx(1.0)  # 4.0 * 0.25


# --- a per-gel AMPLITUDE in the noise formula -----------------------------------------------
def _two_gel_problem(
    noise_formula="noiseParameter1_{o}*noiseParameter2_{o}",
    gains=(1.0, 10.0),
    sigma=0.1,
    base=(1.0, 2.0, 3.0, 4.0),
):
    """A per-gel problem: ONE sigma shared by two observables ("gels") whose measurements
    carry different multiplicative gains, each gain naming itself in BOTH the observable and the
    noise overrides. The underlying signal is identical, so everything that differs between the two
    observables is the gain — which is exactly what must NOT be read as data spread."""
    import pandas as pd

    obs_rows, meas_rows = [], []
    for i, gain in enumerate(gains, start=1):
        oid = f"o{i}"
        obs_rows.append(
            {
                "observableId": oid,
                "observableFormula": f"x * observableParameter1_{oid}",
                "observableTransformation": "lin",
                "noiseFormula": noise_formula.format(o=oid),
            }
        )
        for t, value in enumerate(base):
            meas_rows.append(
                {
                    "observableId": oid,
                    "simulationConditionId": "c",
                    "time": t,
                    "measurement": value * gain,
                    "observableParameters": f"s{i}",
                    "noiseParameters": f"s{i};sigma",
                }
            )
    par = pd.DataFrame(
        {
            "parameterId": [f"s{i}" for i in range(1, len(gains) + 1)] + ["sigma"],
            "parameterScale": ["log10"] * (len(gains) + 1),
            "lowerBound": [1e-5] * (len(gains) + 1),
            "upperBound": [1e5] * (len(gains) + 1),
            "nominalValue": list(gains) + [sigma],
            "estimate": [1] * (len(gains) + 1),
        }
    ).set_index("parameterId")
    return _upgraded(
        observable_df=pd.DataFrame(obs_rows).set_index("observableId"),
        measurement_df=pd.DataFrame(meas_rows),
        parameter_df=par,
    )


def test_a_shared_sigma_with_per_gel_gains_is_summarisable():
    """Before this fix each gel implied its OWN effective sd (gain*sigma), the per-observable values
    disagreed, and the whole entry failed open — one per-gel problem's `noise_sigma_effective` was
    `{}` and its sigmas were reported 'ok' on no evidence at all."""
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
    the between-gel gain difference as spread (on one per-gel problem: 0.096 -> 0.688, ~7x), which
    is not noise the sigma has to cover, so the sigma reads far more plausible than it is."""
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

    entry = {
        "fitted": 3.0,
        "effective_sigma": None,
        "upper_bound_cap": 0.5,
        "data_std": 1.0,
        "fraction_of_data_std": None,
    }
    assert not _entry_inflated(entry) and not _entry_collapsed(entry)  # both silent...
    # ...so the STATUS is what has to carry it. Assert on the real report: an unevaluable formula
    # (effective None) with a perfectly good data_std must not read "ok".
    p = _two_gel_problem(gains=(2.0, 10.0))
    monkeypatch = pytest.MonkeyPatch()
    try:
        monkeypatch.setattr(R, "sigma_effective_value", lambda *a, **kw: None)
        report = noise_sigma_report(p, {"s1": 2.0, "s2": 10.0, "sigma": 0.1})
        assert report["sigma"]["data_std"] is not None  # the OTHER half is assessable
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
        entry["effective_sigma"] / entry["data_std"]
    )


def _gain_and_ungained_problem():
    """One sigma shared by a GAINED observable and an ungained one — a common shape (a single sigma
    over many observables), where only some carry a gel scale."""
    import pandas as pd

    obs = pd.DataFrame(
        [
            {
                "observableId": "o1",
                "observableFormula": "x * observableParameter1_o1",
                "observableTransformation": "lin",
                "noiseFormula": "noiseParameter1_o1*noiseParameter2_o1",
            },
            {
                "observableId": "o2",
                "observableFormula": "z",
                "observableTransformation": "lin",
                "noiseFormula": "noiseParameter1_o2",
            },
        ]
    ).set_index("observableId")
    rows = []
    for t, v in enumerate([1.0, 2.0, 3.0, 4.0]):
        rows.append(
            {
                "observableId": "o1",
                "simulationConditionId": "c",
                "time": t,
                "measurement": v * 2,
                "observableParameters": "s1",
                "noiseParameters": "s1;sigma",
            }
        )
        rows.append(
            {
                "observableId": "o2",
                "simulationConditionId": "c",
                "time": t,
                "measurement": v,
                "observableParameters": None,
                "noiseParameters": "sigma",
            }
        )
    par = pd.DataFrame(
        {
            "parameterId": ["s1", "sigma"],
            "parameterScale": ["log10"] * 2,
            "lowerBound": [1e-5] * 2,
            "upperBound": [1e5] * 2,
            "nominalValue": [2.0, 0.1],
            "estimate": [1, 1],
        }
    ).set_index("parameterId")
    return _upgraded(observable_df=obs, measurement_df=pd.DataFrame(rows), parameter_df=par)


def test_an_ungained_observable_does_not_take_the_gate_dark():
    """ "This observable has no amplitude" (already on the common coordinate, gain 1) must not be
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
    from petab_mcp import reconcile as R

    def build(s1):
        obs = pd.DataFrame(
            [
                {
                    "observableId": "o1",
                    "observableFormula": "x * observableParameter1_o1",
                    "observableTransformation": "lin",
                    "noiseFormula": "noiseParameter1_o1*noiseParameter2_o1*o1",
                }
            ]
        ).set_index("observableId")
        rows = [
            {
                "observableId": "o1",
                "simulationConditionId": "c",
                "time": t,
                "measurement": v * s1,
                "observableParameters": "s1",
                "noiseParameters": "s1;cv",
            }
            for t, v in enumerate([1.0, 2.0, 3.0, 4.0])
        ]
        par = pd.DataFrame(
            {
                "parameterId": ["s1", "cv"],
                "parameterScale": ["log10"] * 2,
                "lowerBound": [1e-5] * 2,
                "upperBound": [1e5] * 2,
                "nominalValue": [s1, 0.1],
                "estimate": [1, 1],
            }
        ).set_index("parameterId")
        return _upgraded(observable_df=obs, measurement_df=pd.DataFrame(rows), parameter_df=par)

    fractions = []
    for s1 in (1.0, 10.0):  # identical underlying data and noise level, different gel gain
        p = build(s1)
        fit = {"s1": s1, "cv": 0.1}
        fractions.append(
            R.sigma_effective_value(p, "cv", 0.1, fitted_values=fit)
            / R.sigma_data_std(p, "cv", fit)
        )
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
        _two_gel_problem(gains=(4.0, 20.0), sigma=0.1),
        "sigma",
        {"s1": 4.0, "s2": 20.0, "sigma": 0.1},
    )
    assert estimate == pytest.approx(doubled), (estimate, doubled)


# --- amplitude resolution: the two ways the multi-gain branch went wrong -------------------
def test_a_partial_amplitude_overlap_does_not_divide():
    """The gain must divide `target` only when EVERY factor is present. Membership was wrong (a
    multi-gain amplitude is a Mul, never in a set of Symbols, so the division was skipped and the
    sigma read inflated by the gain); intersection was wrong the other way — with amplitude `a*b`
    and a target holding only `b`, dividing leaves a spurious `1/a` that then reduces to a NUMBER
    under substitution, returning a silently wrong effective sigma instead of failing open."""
    import sympy as sp

    a, b, sigma = sp.symbols("a b sigma", real=True)
    assert not ((a * b).free_symbols <= (b * sigma).free_symbols)  # partial -> do not divide
    assert (a * b).free_symbols <= (a * b * sigma).free_symbols  # complete -> divide
    assert {a} <= (a * sigma).free_symbols  # bare Symbol still works


def test_observable_formulas_with_function_calls_still_parse():
    """`formula_symbols` is a bare identifier regex, so it reports `exp`/`log10` as symbols.
    Binding those through `sympify(locals=...)` raises "'Symbol' object is not callable", which
    the caller swallows into "no amplitude" — silently losing gain normalisation for every
    observable whose formula uses a function, and judging its sigma in the un-normalised
    coordinate. `log10` is the case a sympy-namespace check misses: sympy has no such attribute."""
    from petab_mcp.reconcile import _sympify_real

    for formula in ("scale * exp(-k*t) + offset", "log10(y)*s", "piecewise(a, t > 1, b)*gain"):
        assert _sympify_real(formula) is not None, formula
    # and the plain case the real=True binding exists for still resolves
    expr = _sympify_real("s_A * (2*A_2 + x)")
    assert expr is not None and all(sym.is_real for sym in expr.free_symbols)


# --- a scale the noise formula names, behind an offset (isensee) ----------------------------
def _offset_scale_problem(scale_estimate=1):
    """isensee's Calpha readout at its smallest: a background before the scale (``b + s*x``) and a
    noise formula that names the scale (``s*rho``), so ``s`` is no amplitude to divide out."""
    import pandas as pd

    obs = pd.DataFrame(
        {
            "observableId": ["o"],
            "observableFormula": ["b + s*x"],
            "observableTransformation": ["lin"],
            "noiseFormula": ["s*rho"],
        }
    ).set_index("observableId")
    meas = pd.DataFrame(
        {
            "observableId": "o",
            "simulationConditionId": "c",
            "time": range(8),
            "measurement": [1100.0, 700.0, 480.0, 330.0, 240.0, 180.0, 150.0, 125.0],
        }
    )
    pdf = pd.DataFrame(
        {
            "parameterScale": ["log10"] * 3,
            "lowerBound": [1e-5] * 3,
            "upperBound": [1e5] * 3,
            "nominalValue": [100.0, 1000.0, 0.03],
            "estimate": [1, scale_estimate, 1],
        },
        index=pd.Index(["b", "s", "rho"], name="parameterId"),
    )
    return _upgraded(observable_df=obs, measurement_df=meas, parameter_df=pdf)


def test_a_split_between_scale_and_sigma_does_not_move_the_verdict():
    """Two models whose x differs 100x fit s 100x smaller and rho 100x larger: the same noise,
    s*rho = 30 in measurement units. The scale was taken for the observable, so the noise read as
    proportional and rho alone was judged against std(ln y): isensee r1 read `collapsed` on every
    new best from v5 on, at the noise of fits that beat the reference (#237)."""
    from petab_mcp import reconcile as R

    p = _offset_scale_problem()
    one, other = (
        noise_sigma_report(p, {"b": 100.0, "s": 1000.0 / c, "rho": 0.03 * c})["rho"]
        for c in (1.0, 100.0)
    )
    assert one["effective_sigma"] == pytest.approx(30.0)
    for key in ("effective_sigma", "data_std", "fraction_of_data_std"):
        assert other[key] == pytest.approx(one[key]), key
    assert one["status"] == other["status"] == "ok"
    # Its raw number is still the model's own, so a judge must not compare it across models.
    assert one["scale_coupled"] and other["scale_coupled"]
    assert not R.sigma_scale_coupled(_offset_scale_problem(scale_estimate=0), "rho")
