"""The replicate lack-of-fit arm end to end: the PEtab-shaped plumbing, the assembler, and the two
MEASURED verdicts it exists to separate.

The steady-state fixture below is not synthetic. ``POINTS`` are the exact per-design-point
replicate statistics of a published problem's tables (252 measurements, 15 steady-state design
points, 12-18 replicates each) on the coordinate the noise is additive on (natural log), and the two
prediction vectors are the analytic steady states of (a) the true model at its ML rates and (b) the
k=7 model one session FINISHED EARLY on, +194 delta_BIC from the truth. Every existing gate passed
that fit: its chi2 GoF p-value was 0.366 (structurally blind — with one shared sigma at its ML
value chi2 == n identically, so the p is a function of (n, k) alone and is 252 for BOTH models), its
fitted/pure-error sigma ratio was 1.4759 against a 1.5 threshold, and the sign/runs tests were
degenerate (one condition at t = inf, so every per-observable series has length 1: n = 1, p = 1.0,
runs test undefined). This arm is the axis that does separate them — 5.3x on max|z| — and these two
tests pin BOTH sides of it, since an arm that only flagged would deadlock the problem.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import Path

import pytest

from petab_mcp import data_access
from petab_mcp import residual_tests as rt

# (observable_id, n_i, ybar_i, s_i) — mean and sample sd of ln(measurement) per design point.
POINTS = (
    ("obs_01", 18, -0.311863269728, 0.036101003463),
    ("obs_02", 18, -5.744799819267, 0.372613753422),
    ("obs_03", 18, -2.611454859236, 0.112692916352),
    ("obs_04", 18, -4.827318442081, 0.152629120103),
    ("obs_05", 18, -3.391665176881, 0.089759344772),
    ("obs_06", 18, -3.365558081462, 0.090215958257),
    ("obs_07", 12, -3.051510292724, 0.190013889404),
    ("obs_08", 18, -6.021776754099, 0.396673194850),
    ("obs_09", 18, -5.123188389731, 0.189632683880),
    ("obs_10", 18, -5.064471621570, 0.217571224812),
    ("obs_11", 18, -6.664247257430, 0.249307902193),
    ("obs_12", 18, -3.459907704182, 0.158533682324),
    ("obs_13", 12, -4.872181822718, 0.319077430247),
    ("obs_14", 18, -5.695979797840, 0.306448993080),
    ("obs_15", 12, -5.486528063759, 0.588239737943),
)

# ln f_i of the TRUE model at its ML rates (analytic steady state of the 16-state lattice; the fit
# it comes from reproduces the reference sigma to 4e-6 relative).
TRUTH_LOG_PREDICTIONS = {
    "obs_01": -0.304128311163,
    "obs_02": -5.749325368864,
    "obs_03": -2.701577549184,
    "obs_04": -4.799570332862,
    "obs_05": -3.326784579852,
    "obs_06": -3.398962034961,
    "obs_07": -3.037730675204,
    "obs_08": -6.054943509061,
    "obs_09": -5.173557290812,
    "obs_10": -5.049248235691,
    "obs_11": -6.614709872525,
    "obs_12": -3.459066210193,
    "obs_13": -4.871116165067,
    "obs_14": -5.696037699444,
    "obs_15": -5.446641319354,
}

# ln f_i of the early-finish fit's 6-structural-parameter model (from its own committed
# simulations, v5).
EARLY_LOG_PREDICTIONS = {
    "obs_01": -0.420760661015,
    "obs_02": -5.598210361989,
    "obs_03": -2.251076300578,
    "obs_04": -5.043182772857,
    "obs_05": -3.145348828195,
    "obs_06": -3.077413116003,
    "obs_07": -3.121376070903,
    "obs_08": -5.687968786988,
    "obs_09": -5.351771717759,
    "obs_10": -5.371534277646,
    "obs_11": -6.818728908224,
    "obs_12": -2.788212473906,
    "obs_13": -4.667378149055,
    "obs_14": -5.857326402023,
    "obs_15": -5.602262443320,
}


def _measured_points(predictions):
    return [
        {"label": obs, "n": n, "ybar": ybar, "s": s, "prediction": predictions[obs]}
        for obs, n, ybar, s in POINTS
    ]


def test_steady_state_true_model_is_not_flagged():
    """The truth's worst design point sits 3.393 SE out (Holm p 0.0519) — INSIDE alpha 0.01, with
    ~5x margin. This is the side that matters most: at alpha 0.05 the truth lands at 0.0519, a coin
    flip that would deadlock the problem about half the time.

    ...and at its 9 structural parameters over 15 points the pass is a POWERED one (df 6), which is
    the whole-problem judgement, so it is asserted where it is made — on the merge."""
    out = rt.replicate_lack_of_fit(_measured_points(TRUTH_LOG_PREDICTIONS))
    assert out["available"] is True
    assert out["status"] == "pass" and out["flagged"] == []
    assert out["n_points"] == 15
    assert out["max_abs_z"] == pytest.approx(3.393, abs=5e-3)
    assert out["min_p_holm"] == pytest.approx(0.0519, rel=1e-2)
    assert out["min_p_holm"] > rt.REPLICATE_LOF_ALPHA * 5  # margin, not a near miss
    merged = data_access._merge_lof_verdicts([out], {}, n_structural=9)
    assert merged["lof_df"] == 6 and "weak" not in merged, merged


def test_steady_state_early_finish_is_flagged():
    """The fit every other gate passed: worst point 17.976 SE out at obs_12, Holm p 2.55e-11
    — nine orders of magnitude past alpha, and the flagged points NAME the states whose structure is
    missing."""
    out = rt.replicate_lack_of_fit(_measured_points(EARLY_LOG_PREDICTIONS))
    assert out["available"] is True and out["status"] == "flagged"
    assert out["max_abs_z"] == pytest.approx(17.976, abs=5e-3)
    assert out["min_p_holm"] == pytest.approx(2.55e-11, rel=1e-2)
    assert out["points"][0]["label"] == "obs_12"
    assert out["points"][0]["z"] == pytest.approx(-17.976, abs=5e-3)
    # the reported points are the worst FIVE, in the measured order
    assert [p["label"] for p in out["points"]] == [
        "obs_12",
        "obs_03",
        "obs_06",
        "obs_01",
        "obs_05",
    ]
    assert set(out["flagged"]) >= {
        "obs_12",
        "obs_03",
        "obs_06",
        "obs_01",
        "obs_05",
    }


def test_separation_is_on_this_axis_and_not_the_sigma_ratio():
    """5.3x on max|z|, against 1.50x on the fitted/pure-error sigma axis the existing gate uses (the
    early-finish fit sat at 1.4759 against a 1.5 threshold — it missed by 0.0024). Pinning the ratio
    here is what keeps a future retuning from quietly collapsing the separation."""
    truth = rt.replicate_lack_of_fit(_measured_points(TRUTH_LOG_PREDICTIONS))
    early = rt.replicate_lack_of_fit(_measured_points(EARLY_LOG_PREDICTIONS))
    assert early["max_abs_z"] / truth["max_abs_z"] == pytest.approx(5.30, abs=0.05)


def test_z_is_exactly_invariant_to_the_sigma_convention():
    """The early-finish fit's fitted-over-pure-error ratio moves between the ML (1.4759) and
    RSS/(n-k) (1.4968) conventions — straddling the 1.5 threshold. z_i never touches the model
    sigma, so it cannot."""
    early = _measured_points(EARLY_LOG_PREDICTIONS)
    assert rt.replicate_lack_of_fit(early) == rt.replicate_lack_of_fit(early)
    arm = rt.replicate_lack_of_fit(early)
    assert (
        data_access._merge_lof_verdicts([arm], {}, n_structural=7)["max_abs_z"]
        == data_access._merge_lof_verdicts([arm], {}, n_structural=6)["max_abs_z"]
    )


def test_holm_does_not_use_a_lack_of_fit_f_statistic():
    """Not an F on m - p degrees of freedom. That rule MANUFACTURES a rejection of the problem's own
    true model (sum z_i^2 = 27.12 over m - p = 15 - 8 = 7 gives p = 0.0016) because the shared-sigma
    fit's projection is OBLIQUE to this statistic's weighting and removes almost no df (exact
    E[T|H0] = 13.60, not 7). The per-point Holm rule needs no df bookkeeping, and it passes the
    truth."""
    truth = _measured_points(TRUTH_LOG_PREDICTIONS)
    total = sum(((p["ybar"] - p["prediction"]) / (p["s"] / math.sqrt(p["n"]))) ** 2 for p in truth)
    assert total == pytest.approx(27.12, abs=0.05)  # the statistic an F test would divide
    assert rt.replicate_lack_of_fit(truth)["status"] == "pass"


# -- the assembler: reconcile's per-point plumbing + the rollup that carries the verdict -----------
def _rec(obs, exp, t, m, s):
    """A simulation record as engine._simulation_records builds it (one row per MEASUREMENT, and
    finite floats only — which is why a steady-state ``time = inf`` arrives as None)."""
    return {
        "observable_id": obs,
        "experiment_id": exp,
        "time": t,
        "measurement": m,
        "simulation": s,
        "residual": m - s,
        "residual_weighted": (m - s) / 0.5,
        "noise": 0.5,
    }


def _replicated_records(session, deviation: float, n_points: int = 3):
    """Turn the toy problem's measurements into ``n_points`` REPLICATED design points (4 replicates
    each), with the model off by ``deviation`` at the last one. The toy problem estimates 2
    structural parameters, so n_points >= 5 is what leaves the arm enough lack-of-fit degrees of
    freedom (REPLICATE_LOF_MIN_LOF_DF) to report a pass as POWERED rather than weak — the default 3
    is the k=14-over-15-points shape in miniature, df 1. Rewrites BOTH the measurement table (which
    reconcile's pure-error grouping reads) and the simulation records, so the two agree the way a
    real commit's do."""
    import pandas as pd

    recs = []
    for i, t in enumerate(float(x) for x in range(n_points)):
        for rep in range(4):
            off = deviation if i == n_points - 1 else 0.0
            recs.append(_rec("obs_B", "", t, 1.0 + 0.01 * rep + off, 1.0 + 0.015))
    session.problem = session.problem.replace(
        measurement_df=pd.DataFrame(
            [
                {
                    "observableId": r["observable_id"],
                    "experimentId": r["experiment_id"],
                    "time": r["time"],
                    "measurement": r["measurement"],
                }
                for r in recs
            ]
        )
    )
    session.simulation = {"records": recs}
    return recs


def test_rollup_flags_the_observable_from_the_replicate_arm(session):
    """The arm sets the flag through the EXISTING consumer: `flagged_observables` is what the
    session's plateau-adequacy gate reads (PR #111), so a lack-of-fit finding blocks an early
    plateau stop with no new gate and no new instruction to the agent."""
    _replicated_records(session, deviation=0.5)
    out = data_access.residual_structure_rollup(session)
    lof = out["replicate_lack_of_fit"]
    assert lof["available"] is True and lof["status"] == "flagged"
    assert "obs_B" in out["flagged_observables"]
    assert data_access.residual_flag(out) is True
    # the advice reaches the agent through the summary, and names the point and its deviation
    assert "Replicate lack-of-fit" in out["summary"] and "obs_B" in out["summary"]
    # ...but never leaks the reference
    assert "delta_bic" not in out["summary"] and "true model" not in out["summary"].lower()
    # and the per-observable table agrees with the rolled-up verdict rather than contradicting it —
    # the arm flags a DESIGN POINT, so without that marking the row would read structure_flag: false
    # while its observable sits in flagged_observables
    diag = data_access.fit_diagnosis(session)
    assert [o["structure_flag"] for o in diag["per_observable"]] == [True]


def test_rollup_replicate_arm_passes_a_model_that_fits(session):
    """An unflagged POWERED arm is a genuine pass — and must clear `low_power`, which otherwise says
    "weak evidence" on the strength of the (degenerate) sign tests alone.

    5 design points against the toy problem's 2 structural parameters leaves df = 3, the point at
    which a pass is worth reporting as one; at 3 points it would be a WEAK pass and correctly credit
    no coverage."""
    _replicated_records(session, deviation=0.0, n_points=5)
    out = data_access.residual_structure_rollup(session)
    assert out["replicate_lack_of_fit"]["status"] == "pass"
    assert out["replicate_lack_of_fit"]["lof_df"] == 3
    assert "weak" not in out["replicate_lack_of_fit"]  # nothing to qualify, so no key
    assert out["flagged_observables"] == []
    assert out["low_power"] is False
    assert data_access.residual_flag(out) is False


def test_rollup_reports_no_evidence_rather_than_a_pass_without_replicates(session):
    """The toy problem measures each design point once. An empty `flagged_observables` reads as
    adequacy to a consumer, so the arm has to say outright that it tested NOTHING."""
    from tests.test_data_access import SIM_RECORDS

    session.simulation = {"records": SIM_RECORDS}
    out = data_access.residual_structure_rollup(session)
    lof = out["replicate_lack_of_fit"]
    assert lof["available"] is False and lof["status"] == "no_replicated_points"
    assert lof["n_points"] == 0 and lof["max_abs_z"] is None
    # ...and the sign/runs arm's low_power verdict is untouched where the replicate arm abstains
    assert out["low_power"] is True


def test_rollup_verdict_shape_is_the_same_on_every_path(session):
    """One key name, one shape: the fit result and the degraded "nothing to test" path carry the same
    CORE keys, since the agent reads this off a fit result.

    The optional ones are omitted when they have nothing to say, and that is ONE policy for the whole
    verdict (`data_access._LOF_OMIT_WHEN_EMPTY_KEYS`) rather than a judgement per key: `advice` never
    rides on a fit at all, `untested_status` only when a sibling family really went untested, and
    `lof_df`/`weak` only where the arm ran with a structural count to compare against. 13 of 16
    problems measured cannot run this arm, so a null on each of those is a per-fit tax forever
    (PR #23). The STATISTIC keeps every key on every path — that shape is the assembler's contract,
    and rebuilding it from a partial key list is how `weak` was dropped in the first place."""
    core = {
        "available",
        "status",
        "n_points",
        "alpha",
        "max_abs_z",
        "min_p_holm",
        "points",
        "n_flagged",
    }
    session.simulation = None
    assert set(data_access.residual_structure_rollup(session)["replicate_lack_of_fit"]) == core

    _replicated_records(session, deviation=0.5)
    rollup = data_access.residual_structure_rollup(session)
    # a FLAG on 3 points against the toy problem's 2 structural parameters: the df is reported, and
    # `weak` is absent because a flag is conclusive however few directions remain
    assert set(rollup["replicate_lack_of_fit"]) == core | {"lof_df"}
    # ...and the verdict is carried ONCE, on the fit result: the diagnosis returns the detail behind
    # it, never a second copy
    assert "residual_structure" not in data_access.fit_diagnosis(session)
    # the advice is carried ONCE (in the summary), not duplicated into the payload
    assert "advice" not in rollup["replicate_lack_of_fit"]

    # ...a WEAK pass is the one path that adds both
    _replicated_records(session, deviation=0.0)
    weak = data_access.residual_structure_rollup(session)["replicate_lack_of_fit"]
    assert set(weak) == core | {"lof_df", "weak"}

    # ...and an untested sibling adds exactly that key
    thin = rt.replicate_lack_of_fit([_point(f"thin{i}", 30.0, n=2) for i in range(4)])
    thick = [_point(f"obs_B[c0, t={t}]", 0.3) for t in (0.0, 1.0, 2.0)]
    merged = data_access._merge_lof_verdicts(
        [thin, rt.replicate_lack_of_fit(thick)], {p["label"]: "obs_B" for p in thick}
    )
    assert set(data_access._lof_reported(merged)) == core | {"untested_status"}


def test_the_statistics_documented_shape_is_its_actual_shape():
    """The "fixed shape" key list in `replicate_lack_of_fit`'s docstring IS the assembler's contract,
    and the last regression came from rebuilding the verdict off a list like it that had gone stale —
    so an added key that is not in the list is the same defect one step earlier. Both paths, since the
    skipped one has to carry every key too."""
    import inspect

    doc = inspect.getdoc(rt.replicate_lack_of_fit)
    listed = set(re.findall(r"[a-z_]+", re.search(r"``\{(available.*?)\}``", doc, re.S).group(1)))
    for verdict in (rt.replicate_lack_of_fit(_flat_points(15)), rt.replicate_lof_unavailable()):
        assert set(verdict) == listed, set(verdict) ^ listed
    # ...and the df keys are NOT among them: the statistic sees one family, and neither operand of
    # the df is a per-family quantity (rt.lof_power). A second owner computing `weak` off this
    # family's points is what made the arm and the payload disagree on 22 of 48 measured cells.
    assert not listed & {"lof_df", "weak", "n_structural", "min_lof_df"}, listed


def test_rollup_replicate_arm_never_raises(session, monkeypatch):
    """Fail OPEN: this runs on every commit and feeds a gate, so a plumbing failure must degrade to
    NO EVIDENCE rather than take down the fit."""
    from petab_mcp import reconcile as R

    _replicated_records(session, deviation=0.5)
    monkeypatch.setattr(
        R,
        "replicate_pure_error_points",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    out = data_access.residual_structure_rollup(session)
    assert out["replicate_lack_of_fit"]["available"] is False
    assert out["flagged_observables"] == []


def _point(label, z, *, n=4, s=0.1, observable="obs_B"):
    """A design point whose deviation is exactly ``z`` standard errors, already labelled the way
    ``data_access._labelled_points`` labels one."""
    return {
        "label": label,
        "n": n,
        "ybar": z * s / math.sqrt(n),
        "s": s,
        "prediction": 0.0,
        "observable_id": observable,
    }


def test_rollup_runs_the_replicate_arm_without_any_weighted_residual(session):
    """z_i is sigma-INDEPENDENT by construction, so the arm must not be gated behind the
    sigma-DERIVED `residual_weighted` column — which is NaN for the WHOLE table in exactly the cases
    this arm exists for: `engine._weighted_residual` returns (None, None) when the noise formula
    references model states or unresolved parameters, and (noise, None) when the evaluated sd is
    0.0, a COLLAPSED sigma. Gated, the verdict degraded to `n_points: 0` and residual_flag None, so
    the plateau gate was not blocked and the agent was told nothing had been tested."""
    recs = _replicated_records(session, deviation=0.5)
    for r in recs:  # a collapsed sigma: noise 0 -> no weighted residual at all
        r["noise"], r["residual_weighted"] = 0.0, None
    out = data_access.residual_structure_rollup(session)
    lof = out["replicate_lack_of_fit"]
    assert lof["available"] is True and lof["status"] == "flagged" and lof["n_points"] == 3
    assert out["flagged_observables"] == ["obs_B"]
    assert data_access.residual_flag(out) is True
    assert "Replicate lack-of-fit" in out["summary"]
    # ...and the head attributes the finding only to the test that RAN: quoting the sign/runs
    # threshold for tests that produced no p at all is the same over-claim in the other direction.
    assert out["summary"].startswith(
        "systematic residual structure in obs_B (replicate lack-of-fit)"
    )
    assert "sign-test bias" not in out["summary"]
    # get_fit_diagnosis's own no-weighted-residuals early return has no rows and no verdict to
    # contradict this one — the replicate arm reaches the agent through the fit result
    assert data_access.fit_diagnosis(session)["per_observable"] == []


def test_rollup_does_not_claim_sign_randomness_when_the_sign_tests_could_not_run(session):
    """The other half of the no-weighted-residual case: a PASSING replicate arm must not be summarised
    as "residuals look sign-random", which asserts a test that never happened.

    5 design points, not the default 3, so this is a POWERED pass against the toy problem's 2
    structural parameters (df 3) — at 3 points the pass would be a weak one, and this test is about
    what the summary says of a real pass."""
    recs = _replicated_records(session, deviation=0.0, n_points=5)
    for r in recs:
        r["noise"], r["residual_weighted"] = 0.0, None
    out = data_access.residual_structure_rollup(session)
    assert out["replicate_lack_of_fit"]["status"] == "pass"
    assert "weak" not in out["replicate_lack_of_fit"]
    assert "sign-random" not in out["summary"]
    assert "the sign and runs tests could not run" in out["summary"]
    # ...while the replicate arm's own pass IS reported, so this is not read as "nothing tested"
    assert "replicate lack-of-fit: none over 5" in out["summary"]
    assert data_access.residual_flag(out) is False


def test_low_power_survives_for_an_observable_the_replicate_arm_never_covered(session):
    """A powered arm is a genuine pass only WHERE IT RAN. Clearing `low_power` globally asserted
    sign-randomness for observables with no replicates at all, and that over-claim reaches a
    client's scoring and its finish gate.

    `n_structural=0` here (a model with nothing estimated but its sigmas) so that 3 points make a
    POWERED arm — the coverage question this test asks is about WHICH observables, not about how many
    directions (that is test_an_underpowered_arm_credits_no_coverage_even_when_it_flagged)."""
    points = [_point(f"obs_B[c0, t={t}]", 0.3) for t in (0.0, 1.0, 2.0)]
    lof = data_access._merge_lof_verdicts(
        [rt.replicate_lack_of_fit(points)], {p["label"]: "obs_B" for p in points}, n_structural=0
    )
    assert lof["status"] == "pass"
    underpowered = {"structure_low_power": True, "structure_flag": False}
    out = data_access._rollup_structure(
        [("obs_A", underpowered), ("obs_B", underpowered)], replicate_lof=lof
    )
    assert out["low_power"] is True  # obs_A has no replicates; nothing tested it
    # ...and the prose NAMES it, because the same summary reports a genuine pass for the observable
    # the arm covered: a blanket "the tests are UNDERPOWERED at this sample size" beside "this arm IS
    # powered, so that is a pass" is a contradiction the agent cannot resolve (measured: every fit
    # of one problem).
    assert "UNDERPOWERED for obs_A" in out["summary"]
    assert "obs_B" not in out["summary"].split("replicate lack-of-fit")[0]
    assert "this arm IS powered" in out["summary"]
    # ...and where the arm DID cover every underpowered observable it still clears the flag
    covered_only = data_access._rollup_structure([("obs_B", underpowered)], replicate_lof=lof)
    assert covered_only["low_power"] is False
    assert "UNDERPOWERED" not in covered_only["summary"]


def test_merge_does_not_hide_a_family_the_arm_could_not_test():
    """Two measured shapes side by side: 126 replicated points with one df each (untestable)
    beside 3 thick ones that pass. Keeping only the AVAILABLE arms dropped the other's status
    entirely and emitted "this arm IS powered, so that is a pass" with 126 of 129 points never
    looked at."""
    thin = rt.replicate_lack_of_fit([_point(f"thin{i}", 30.0, n=2) for i in range(126)])
    thick_points = [_point(f"obs_B[c0, t={t}]", 0.3) for t in (0.0, 1.0, 2.0)]
    thick = rt.replicate_lack_of_fit(thick_points)
    assert thin["status"] == "replicates_too_thin" and thick["status"] == "pass"
    merged = data_access._merge_lof_verdicts(
        [thin, thick], {p["label"]: "obs_B" for p in thick_points}
    )
    assert merged["status"] == "pass" and merged["n_points"] == 3
    assert merged["untested_status"] == "replicates_too_thin"
    assert "replicates_too_thin" in data_access._replicate_lof_line(merged)


def test_merge_keeps_a_flag_whose_point_has_no_observable_id():
    """`replicate_pure_error_points` reports observable_id None when the rows carry no
    observableId column. Mapping two flagged points to {None, 'obs_A'} made `sorted` raise
    TypeError, which _replicate_lof's broad except swallowed — turning a genuine lack-of-fit FLAG
    into "no evidence"."""
    points = [_point("p_none", 12.0), _point("p_obs", 12.0), _point("p_ok", 0.1)]
    verdict = rt.replicate_lack_of_fit(points)
    merged = data_access._merge_lof_verdicts(
        [verdict], {"p_none": None, "p_obs": "obs_A", "p_ok": "obs_A"}
    )
    assert merged["status"] == "flagged" and merged["n_flagged"] == 2
    # the unattributable point falls back to its own LABEL, which is a real string the agent can act
    # on (and not the literal "None")
    assert merged["flagged_observables"] == ["obs_A", "p_none"]


def test_the_summary_speaks_when_the_arm_could_not_run(session):
    """Silence is for the two unremarkable, PERMANENT cases — no repeated design points at all, and
    no noise parameter to scope the arm by (13 of 16 problems measured, between them). The catch-all
    `unavailable` is a real failure ON a problem with replicates (the sigma's observables mix lin and
    log, a gain will not resolve, or no prediction is constant across a point's replicates) and was
    silent too, so the agent could not tell it apart from a pass."""
    for silent in ("no_replicated_points", "no_noise_parameter"):
        assert data_access._replicate_lof_line({"available": False, "status": silent}) == ""
    recs = _replicated_records(session, deviation=0.0)
    for i, r in enumerate(recs):  # break the per-point constancy of the prediction
        r["simulation"] += 0.001 * i
    out = data_access.residual_structure_rollup(session)
    assert out["replicate_lack_of_fit"]["status"] == "unavailable"
    assert "replicate lack-of-fit: NO EVIDENCE" in out["summary"]


def test_a_problem_whose_noise_is_fixed_data_says_nothing_every_fit(session):
    """Four measured problems: the noiseFormula is a `noiseParameter1_*` placeholder and the
    measurement table's noiseParameters cells hold NUMBERS, so no noise parameter survives the
    parameter-table intersection and the arm cannot be scoped at all. Measured, those four also
    have ZERO replicated design points. Reporting the catch-all there put
    170 chars into the summary on EVERY fit, forever, for a condition no re-author can change — the
    exact per-fit tax PR #23 cut. Its own status, and silent."""
    import pandas as pd

    odf = session.problem.observable_df.copy()
    odf.loc["obs_B", "noiseFormula"] = "noiseParameter1_obs_B"
    session.problem = session.problem.replace(
        observable_df=odf,
        measurement_df=pd.DataFrame(
            [
                {
                    "observableId": "obs_B",
                    "experimentId": "",
                    "time": t,
                    "measurement": 1.0,
                    "noiseParameters": 0.023108,
                }
                for t in (0.0, 1.0, 2.0)
            ]
        ),
    )
    session.simulation = {"records": [_rec("obs_B", "", t, 1.0, 1.0) for t in (0.0, 1.0, 2.0)]}
    assert data_access._noise_parameter_ids(session.problem) == []
    out = data_access.residual_structure_rollup(session)
    assert out["replicate_lack_of_fit"]["status"] == "no_noise_parameter"
    assert "replicate lack-of-fit" not in out["summary"]


def test_a_benign_family_cannot_silence_an_unresolvable_one():
    """One sigma resolving to zero replicated points (benign, silent) beside one that will not
    resolve at all (a real failure on a problem that DOES have replicates): ranking the catch-all
    BELOW `no_replicated_points` let the benign sibling win the merged status, so the line went
    silent again and finding 10's fix was defeated wherever a problem has more than one sigma."""
    verdicts = [
        rt.replicate_lof_unavailable("no_replicated_points"),
        rt.replicate_lof_unavailable("unavailable"),
    ]
    for order in (verdicts, list(reversed(verdicts))):
        merged = data_access._merge_lof_verdicts(order, {})
        assert merged["status"] == "unavailable"
        assert "NO EVIDENCE" in data_access._replicate_lof_line(merged)


def test_a_pass_is_not_hedged_against_a_family_with_nothing_in_it():
    """`untested_status` exists so a partial pass cannot read as a whole one — but a
    `no_replicated_points` family has no replicated design points in it, so "a further family could
    NOT be tested" hedges against an empty set. Silent statuses are skipped; a real one survives."""
    thick = [_point(f"obs_B[c0, t={t}]", 0.3) for t in (0.0, 1.0, 2.0)]
    passing = rt.replicate_lack_of_fit(thick)
    labels = {p["label"]: "obs_B" for p in thick}
    for silent in ("no_replicated_points", "no_noise_parameter"):
        merged = data_access._merge_lof_verdicts(
            [passing, rt.replicate_lof_unavailable(silent)], labels
        )
        assert merged.get("untested_status") is None  # nothing to say, so not even a null key
        assert "could NOT be tested" not in data_access._replicate_lof_line(merged)
    real = data_access._merge_lof_verdicts(
        [passing, rt.replicate_lof_unavailable("unavailable")], labels
    )
    assert real["untested_status"] == "unavailable"
    assert "could NOT be tested (unavailable)" in data_access._replicate_lof_line(real)


def test_the_line_reports_a_flag_from_the_payload_that_rides_on_a_fit():
    """`advice` is stripped from what rides on a fit (`_LOF_ROLLUP_ONLY_KEYS`), so dispatching the
    line on its presence turned a FLAGGED verdict into "that is a pass" for any caller summarising
    the projection rather than the merge. Dispatch on the STATUS."""
    points = [_point("far", 25.0), _point("n0", 0.1), _point("n1", 0.1)]
    merged = data_access._merge_lof_verdicts(
        [rt.replicate_lack_of_fit(points)], {p["label"]: "obs_B" for p in points}
    )
    assert merged["status"] == "flagged"
    line = data_access._replicate_lof_line(data_access._lof_reported(merged))
    assert "pass" not in line and "1 of 3 replicated design points" in line


def _split_sigma_records(session, deviation: float):
    """Two noise parameters governing different ROWS of ONE observable, through per-row
    `noiseParameters` overrides (a per-gel shape): sigma_lo owns t in {0,1,2}, sigma_hi owns
    t in {3,4,5}, four replicates each. The model is off by ``deviation`` at t=5 — one of sigma_hi's
    points, and none of sigma_lo's."""
    import pandas as pd

    recs, sigmas = [], []
    for i, t in enumerate((0.0, 1.0, 2.0, 3.0, 4.0, 5.0)):
        for rep in range(4):
            off = deviation if t == 5.0 else 0.0
            recs.append(_rec("obs_B", "", t, 1.0 + 0.01 * rep + off, 1.0 + 0.015))
            sigmas.append("sigma_lo" if i < 3 else "sigma_hi")
    odf = session.problem.observable_df.copy()
    odf.loc["obs_B", "noiseFormula"] = "noiseParameter1_obs_B"
    session.problem = session.problem.replace(
        observable_df=odf,
        measurement_df=pd.DataFrame(
            [
                {
                    "observableId": r["observable_id"],
                    "experimentId": r["experiment_id"],
                    "time": r["time"],
                    "measurement": r["measurement"],
                    "noiseParameters": sigma,
                }
                for r, sigma in zip(recs, sigmas)
            ]
        ),
    )
    pdf = session.problem.parameter_df
    for pid in ("sigma_lo", "sigma_hi"):
        pdf.loc[pid] = pdf.loc["sigma_B"]
    session.simulation = {"records": recs}
    return recs


def test_each_noise_parameter_tests_only_the_rows_it_governs(session):
    """The sigma's row scoping is per ROW (a `noiseParameters` override), but the simulation path
    collapsed it to an OBSERVABLE-ID filter, so both sigmas tested all six of obs_B's design points:
    every count doubled, the reported points came back duplicated, and each arm Holm-corrected over
    — and applied its own gain and coordinate to — rows the other sigma governs."""
    _split_sigma_records(session, deviation=0.5)
    lof = data_access.residual_structure_rollup(session)["replicate_lack_of_fit"]
    assert lof["available"] is True and lof["status"] == "flagged"
    assert lof["n_points"] == 6  # the six REAL design points, not 6 per sigma
    assert lof["n_flagged"] == 1  # one deviating point, flagged once
    labels = [p["label"] for p in lof["points"]]
    assert labels == ["obs_B[t=5.0]"] and len(labels) == len(set(labels))


def test_labels_are_unique_across_noise_families(session):
    """A label is the ONLY thing the agent can act on, so it has to name one design point. The bare
    observable id is used where that observable has a single replicated point — decided WITHIN a
    family, two distinct points of one observable owned by different sigmas were each the only point
    of their own family, so the payload printed the same name twice with nothing to tell them apart.
    The count is taken across every family now."""
    import pandas as pd

    odf = session.problem.observable_df.copy()
    odf.loc["obs_B", "noiseFormula"] = "noiseParameter1_obs_B"
    odf.loc["obs_A"] = {"observableFormula": "A", "noiseFormula": "noiseParameter1_obs_A"}
    rows, recs = [], []
    #  sigma_lo owns obs_B at t=0 and obs_A at t in {0,1}; sigma_hi owns obs_B at t=3 and obs_A at
    #  t in {2,3} — one obs_B point per family, three usable points per family.
    for sigma, obs_b_t, obs_a_ts in (("sigma_lo", 0.0, (0.0, 1.0)), ("sigma_hi", 3.0, (2.0, 3.0))):
        for obs, t in [("obs_B", obs_b_t)] + [("obs_A", t) for t in obs_a_ts]:
            for rep in range(4):
                rows.append(
                    {
                        "observableId": obs,
                        "experimentId": "",
                        "time": t,
                        "measurement": 1.0 + 0.01 * rep,
                        "noiseParameters": sigma,
                    }
                )
                recs.append(_rec(obs, "", t, 1.0 + 0.01 * rep, 1.0))
    session.problem = session.problem.replace(observable_df=odf, measurement_df=pd.DataFrame(rows))
    pdf = session.problem.parameter_df
    for pid in ("sigma_lo", "sigma_hi"):
        pdf.loc[pid] = pdf.loc["sigma_B"]
    session.simulation = {"records": recs}
    lof = data_access.residual_structure_rollup(session)["replicate_lack_of_fit"]
    assert lof["available"] is True and lof["n_points"] == 6
    # obs_B's two points are qualified by time even though each is alone in its own family
    assert (
        data_access._labelled_points(
            [{"observable_id": "obs_B", "experiment_id": "", "time": 0.0}], Counter({"obs_B": 2})
        )[0]["label"]
        == "obs_B[t=0.0]"
    )


def test_design_points_match_across_a_steady_state_time(session):
    """A steady-state family sits at ``time = inf``, which the simulation records store as None
    (they keep only finite floats). Scoping the records to the sigma's design points therefore has
    to normalise the two sides to the same key, or the 15 points of the steady-state problem above
    would match nothing and the arm would go dark on the very problem it was built for."""
    import pandas as pd
    from petab_mcp import reconcile as R

    recs = _replicated_records(session, deviation=0.5)
    for r in recs:  # the records' side of a steady-state point
        r["time"] = None
    mdf = session.problem.measurement_df
    session.problem = session.problem.replace(
        measurement_df=mdf.assign(time=float("inf"))
    )  # the table's side
    frame = pd.DataFrame(recs).rename(
        columns={"observable_id": "observableId", "experiment_id": "experimentId"}
    )
    points = R.replicate_pure_error_points(session.problem, "sigma_B", simulations=frame)
    assert len(points) == 1 and points[0]["n"] == 12  # one steady-state design point, 12 replicates
    assert points[0]["prediction"] is not None
    assert data_access.residual_structure_rollup(session)["replicate_lack_of_fit"]["n_points"] == 1


def test_structural_count_includes_a_dual_role_scaling_noise_symbol(session):
    """Per-gel scale factors (eight on one measured problem) can sit in BOTH noiseParameters and
    observableParameters. `plausibility_sigma_ids` keeps them out of the noise families (scored as an
    sd they read 900x inflated), so the STRUCTURAL count must not exclude them as well — they move
    the prediction, which is exactly the flexibility the vacuous guard weighs. Excluded from both
    sets, eight real degrees of freedom went uncounted."""
    import pandas as pd
    from petab_mcp import reconcile as R

    odf = session.problem.observable_df.copy()
    odf.loc["obs_B", "noiseFormula"] = "noiseParameter1_obs_B"
    odf.loc["obs_B", "observableFormula"] = "observableParameter1_obs_B * B"
    session.problem = session.problem.replace(
        observable_df=odf,
        measurement_df=pd.DataFrame(
            [
                {
                    "observableId": "obs_B",
                    "experimentId": "",
                    "time": t,
                    "measurement": 0.5,
                    "noiseParameters": "s_gel",
                    "observableParameters": "s_gel",
                }
                for t in (0.0, 1.0)
            ]
        ),
    )
    pdf = session.problem.parameter_df
    pdf.loc["s_gel"] = pdf.loc["k1"]  # estimated, like the real gel factors
    assert "s_gel" in R.noise_sigma_ids(session.problem)  # a noise symbol...
    assert "s_gel" not in R.plausibility_sigma_ids(session.problem)  # ...but not a judgeable sd
    # k1, k2 and s_gel are estimated; only the pure sigma placeholder is not structural flexibility
    assert data_access._n_estimated_structural(session.problem) == 3


def test_the_pooled_noise_floor_ignores_a_design_point_with_a_missing_key(session):
    """`dropna=False` is needed only for the SIMULATION records, whose `time` stores finite floats
    only (a steady-state t = inf arrives as None). On the MEASUREMENT table a blank
    observable/time cell is missing data: pooling every such row into ONE NaN-keyed group counts
    its between-point spread as within-point noise, silently moving the floor the 1.5 inflation
    gate reads. (A blank experiment is not missing: it is the experiment-less design.)"""
    import pandas as pd
    from petab_mcp import reconcile as R

    _replicated_records(session, deviation=0.0)
    floor = R.sigma_replicate_noise_estimate(session.problem, "sigma_B")
    session.problem = session.problem.replace(
        measurement_df=pd.concat(
            [
                session.problem.measurement_df,
                pd.DataFrame(
                    [
                        {
                            "observableId": "obs_B",
                            "experimentId": "",
                            "time": float("nan"),
                            "measurement": v,
                        }
                        for v in (0.0, 100.0)
                    ]
                ),
            ],
            ignore_index=True,
        )
    )
    assert R.sigma_replicate_noise_estimate(session.problem, "sigma_B") == pytest.approx(floor)


def test_reconcile_exposes_per_point_pure_error_and_pools_to_the_same_floor(session):
    """`sigma_replicate_noise_estimate`'s pooled number IS the pooling of these per-point sums of
    squares — the refactor must not move the existing noise floor."""
    from petab_mcp import reconcile as R

    recs = _replicated_records(session, deviation=0.0)
    points = R.replicate_pure_error_points(session.problem, "sigma_B")
    assert [p["n"] for p in points] == [4, 4, 4]
    assert all(p["df"] == 3 for p in points)
    assert all(p["prediction"] is None for p in points)  # no simulations passed -> unknown
    pooled = math.sqrt(sum(p["ssq"] for p in points) / sum(p["df"] for p in points))
    assert R.sigma_replicate_noise_estimate(session.problem, "sigma_B") == pytest.approx(pooled)

    # with the simulation records, each point also carries the prediction it is judged against
    import pandas as pd

    frame = pd.DataFrame(recs).rename(
        columns={"observable_id": "observableId", "experiment_id": "experimentId"}
    )
    with_pred = R.replicate_pure_error_points(session.problem, "sigma_B", simulations=frame)
    assert [round(p["prediction"], 6) for p in with_pred] == [1.015, 1.015, 1.015]
    assert [p["ybar"] for p in with_pred] == pytest.approx([p["ybar"] for p in points])


# -- the df guard (a k=14 fit over 15 points) -------------------------------
def _flat_points(m: int, z: float = 0.1):
    """``m`` replicated design points whose predictions all sit ``z`` SE from their own mean."""
    return [
        {"label": f"p{i}", "n": 18, "ybar": 0.0, "s": 1.0, "prediction": -z / 18**0.5}
        for i in range(m)
    ]


def _merged_flat(m: int, n_structural, *, min_lof_df=None, families: int = 1, z: float = 0.1):
    """``families`` families of ``m`` flat replicated points each, merged the way ``_replicate_lof``
    merges them: ONE whole-model structural count against the POOLED points."""
    arms = [
        rt.replicate_lack_of_fit([{**p, "label": f"f{f}{p['label']}"} for p in _flat_points(m, z)])
        for f in range(families)
    ]
    return data_access._merge_lof_verdicts(
        arms, {}, n_structural=n_structural, min_lof_df=min_lof_df
    )


def test_a_pass_needs_lack_of_fit_degrees_of_freedom():
    """14 structural parameters over 15 points is not a pass, it is saturation.

    Refusing only at EXACT saturation let df = 1 through, and that pass is what a client's finish
    gate showed the agent as "this arm IS powered" on a k=14 fit of the steady-state problem — a
    model whose 14 free parameters had placed every prediction within 0.107 SE of its replicate
    mean by construction.
    """
    v = _merged_flat(15, 14)  # df = 1: evidence, but one direction
    assert v["status"] == "pass" and v["weak"] is True and v["lof_df"] == 1, v
    assert _merged_flat(15, 13)["weak"] is True  # df = 2
    v = _merged_flat(15, 12)  # df = 3: a pass worth the name
    assert v["status"] == "pass" and "weak" not in v, v
    v = _merged_flat(15, 15)  # df = 0: nothing was tested at all
    assert v["available"] is False and v["status"] == "vacuous", v
    assert v["n_points"] == 15  # ...and it says how much was not tested


def test_a_flag_stands_at_any_lack_of_fit_df():
    """One residual direction is weak evidence for a PASS but conclusive for a FLAG: the point
    deviates by more than its own replicate noise allows AFTER the fit had its way."""
    pts = _flat_points(15)
    pts[0]["prediction"] = -8.0 / 18**0.5  # ~8 SE out
    arm = rt.replicate_lack_of_fit(pts)
    v = data_access._merge_lof_verdicts([arm], {}, n_structural=14)  # df = 1
    assert v["available"] is True and v["status"] == "flagged", v
    assert v["flagged_observables"] == ["p0"] and "weak" not in v, v
    # ...but at df <= 0 the arm is VACUOUS and the flag goes with it: a fit that flexible could have
    # placed that point anywhere, so NO EVIDENCE is the honest reading and the fail-open one. This is
    # the one place the flag does not survive, and it is why `vacuous` is decided on the POOLED count
    # rather than per family — a family whose own points are outnumbered is not a saturated problem.
    v = data_access._merge_lof_verdicts([arm], {}, n_structural=15)
    assert v["available"] is False and v["status"] == "vacuous" and v["n_flagged"] == 0, v


def test_the_df_guard_leaves_the_cases_that_motivated_the_arm_testable():
    """The steady-state problem's truth (8-9 structural over 15 points, df 6-7) and the
    under-parameterised fit that motivated the arm (6-7, df 8-9) must both keep a real, un-weak
    test."""
    for n_structural in (9, 8, 7, 6):
        v = _merged_flat(15, n_structural)
        assert v["available"] is True and "weak" not in v, (n_structural, v)


def test_an_uncountable_structural_count_is_not_a_powered_pass():
    """`_n_estimated_structural` returns None when the parameter table cannot answer, and an
    UNCOUNTED direction is not a spare one. The df keys are then absent — but absent `weak` used to
    read as "not weak" all the way out: the line said "this arm IS powered", the arm credited full
    coverage and cleared `low_power`, and a client's record of it read `lof_weak: False`. That is
    the maximal claim on the least evidence, in a diagnostic whose whole job is to fail open."""
    merged = _merged_flat(15, None)
    assert merged["status"] == "pass"
    assert "lof_df" not in merged and "weak" not in merged, merged
    line = data_access._replicate_lof_line(merged)
    assert "IS powered" not in line and "UNKNOWN" in line, line
    # ...and it certifies nothing: no coverage credit, so `low_power` survives for the observable
    points = [_point(f"obs_B[c0, t={t}]", 0.3) for t in (0.0, 1.0, 2.0)]
    labels = {p["label"]: "obs_B" for p in points}
    unknown = data_access._merge_lof_verdicts([rt.replicate_lack_of_fit(points)], labels)
    assert unknown["covered_observables"] == [], unknown
    underpowered = {"structure_low_power": True, "structure_flag": False}
    assert (
        data_access._rollup_structure([("obs_B", underpowered)], replicate_lof=unknown)["low_power"]
        is True
    )


def test_the_weak_marking_reaches_the_summary_through_the_whole_assembly(session):
    """The ASSEMBLY path, end to end, not the statistic: the earlier version of this test called
    `_merge_lof_verdicts` and `_replicate_lof_line` by hand, which is not where the keys were lost.

    `_lof_payload` and `_lof_reported` project the verdict twice and `_structure_summary` rebuilds the
    prose, and dropping `weak` anywhere along there left a client's finish gate saying "this arm IS
    powered" on a df = 1 pass, in three sessions on the steady-state problem — with the statistic
    already correct. So this goes through `residual_structure_rollup` on a real session: 3
    replicated design points against the toy problem's 2 structural parameters is df 1, the
    k=14-over-15-points shape in miniature."""
    _replicated_records(session, deviation=0.0, n_points=3)
    out = data_access.residual_structure_rollup(session)
    lof = out["replicate_lack_of_fit"]
    assert lof["status"] == "pass" and lof["weak"] is True and lof["lof_df"] == 1, lof
    assert "WEAK evidence of form" in out["summary"], out["summary"]
    assert "IS powered" not in out["summary"], out["summary"]
    # ...and a weak pass certifies nothing, so it credits no coverage and low_power survives
    assert out["low_power"] is True
    assert data_access.residual_flag(out) is False  # advisory: it is still not a FLAG
    # ...and no second projection of it on the diagnosis path, where it used to be repeated
    assert "residual_structure" not in data_access.fit_diagnosis(session)


def test_lack_of_fit_df_subtracts_the_shared_parameters_once_across_noise_families():
    """Points ADD across noise families; the structural parameters do NOT — they are one whole-model
    count that moves every family's predictions, so it is subtracted ONCE.

    Judged per family the df was `family points - n_structural`, and summing those computed
    `total points - F * n_structural`, which is not a df either: two families of 15 replicated points
    at n_structural=14 reported df 2 and a WEAK pass, where the honest count is 30 - 14 = 16 and the
    pass is strongly powered. That is the steady-state shape with a second sigma, and the summary
    would have told the agent its pass was weak when it was not."""
    merged = _merged_flat(5, 3, families=2)
    assert merged["n_points"] == 10
    assert merged["lof_df"] == 7, merged  # 10 - 3, not (5 - 3) + (5 - 3)
    assert "weak" not in merged, merged  # 7 >= REPLICATE_LOF_MIN_LOF_DF

    # the steady-state shape, at one, two and three sigmas: F * 15 - 14, never F * (15 - 14)
    assert [_merged_flat(15, 14, families=f)["lof_df"] for f in (1, 2, 3)] == [1, 16, 31]
    twice = _merged_flat(15, 14, families=2)
    assert "weak" not in twice and "IS powered" in data_access._replicate_lof_line(twice)


def test_a_pooled_family_that_is_powered_is_not_refused_as_vacuous():
    """The other half of the same arithmetic, and the one that reached the agent as a FALSE
    sentence: saturation is a property of the PROBLEM, not of one noise family.

    Judged per family, two families of 12 points against 20 shared parameters were each vacuous, so
    the merged verdict was NO EVIDENCE — "this model has as many estimated structural parameters as
    there are replicated design points" — of 24 points against 20, which leaves 4 directions and a
    POWERED pass. One measured problem sits in this regime: three usable families of 4 points
    against 10 structural parameters, pooled df 2, a thin pass rather than nothing at all."""
    v = _merged_flat(12, 20, families=2)
    assert v["available"] is True and v["status"] == "pass", v
    assert v["lof_df"] == 4 and "weak" not in v, v

    three_small = _merged_flat(4, 10, families=3)
    assert three_small["available"] is True and three_small["lof_df"] == 2, three_small
    assert three_small["weak"] is True, three_small
    assert "NO EVIDENCE" not in data_access._replicate_lof_line(three_small)
    # ...and a problem that really IS saturated still reports nothing: 3 x 4 points against 12
    assert _merged_flat(4, 12, families=3)["status"] == "vacuous"


def test_the_merge_honours_a_caller_supplied_threshold():
    """One owner for the rule AND for the number: `rt.lof_power` holds the only comparison and the
    only default. The threshold used to live in the statistic and again in the merge, where the copy
    read the module constant — so a caller's `min_lof_df=` was honoured by one and discarded by the
    other, and a weak pass came back out labelled "this arm IS powered"."""
    merged = data_access._merge_lof_verdicts(
        [rt.replicate_lack_of_fit(_flat_points(15))], {}, n_structural=11, min_lof_df=8
    )  # df 4 < 8
    assert merged["weak"] is True and merged["lof_df"] == 4, merged
    assert "IS powered" not in data_access._replicate_lof_line(merged)
    # ...and the default is not a second copy of the number: at the same df the module threshold of 3
    # calls it powered
    assert "weak" not in _merged_flat(15, 11), _merged_flat(15, 11)


def test_an_underpowered_arm_credits_no_coverage_even_when_it_flagged():
    """Coverage follows the DF band, not the verdict. `weak` is forced False as soon as anything
    flagged, so keying the credit on it let a df = 1 arm that flagged obs_A certify obs_B as
    sign-random — on the strength of the same arm the summary calls too flexible to mean anything.
    That over-claim reaches a client's scoring and its finish gate."""
    pts = [_point("obs_A", 8.0, n=18, s=1.0, observable="obs_A")]
    pts += [_point(f"obs_B[c0, t={i}]", 0.1, n=18, s=1.0) for i in range(14)]
    labels = {p["label"]: p["observable_id"] for p in pts}
    arm = rt.replicate_lack_of_fit(pts)  # 15 points
    merged = data_access._merge_lof_verdicts([arm], labels, n_structural=14)  # df 1
    assert merged["status"] == "flagged" and merged["lof_df"] == 1 and "weak" not in merged
    assert merged["flagged_observables"] == ["obs_A"]
    assert merged["covered_observables"] == [], merged  # obs_B is NOT certified
    underpowered = {"structure_low_power": True, "structure_flag": False}
    out = data_access._rollup_structure([("obs_B", underpowered)], replicate_lof=merged)
    assert out["low_power"] is True, out
    # ...while the same arm with directions to spare does credit it
    powered = data_access._merge_lof_verdicts([arm], labels, n_structural=5)
    assert powered["covered_observables"] == ["obs_A", "obs_B"]
    assert (
        data_access._rollup_structure([("obs_B", underpowered)], replicate_lof=powered)["low_power"]
        is False
    )


def test_no_unavailable_status_in_the_priority_table_is_unreachable():
    """`too_few_lof_df` was added to the table by the abandoned design where the guard REFUSED below
    min_lof_df. Nothing emits it and `_replicate_lof_line` has no branch for it, so if anything ever
    did the line would fall through to the catch-all and give the agent a reason that is not the
    reason. Every entry has to be a status some code path actually produces.

    Counting the EMITTER call, not the bare string: counting occurrences of the name made the guard
    satisfiable by mentioning the dead status in a comment, which is how a dead entry gets kept."""
    src = "".join(p.read_text() for p in (Path(data_access.__file__).parent).glob("*.py"))
    emitter = r'replicate_lof_unavailable\(\s*"%s"'
    orphans = [
        s for s in data_access._LOF_STATUS_PRIORITY if not re.search(emitter % re.escape(s), src)
    ]
    assert not orphans, orphans
