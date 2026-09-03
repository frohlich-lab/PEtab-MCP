"""Model-free regime diagnostics (`petab_mcp.regimes`).

Every case is SYNTHETIC and built from a known generating process, so the expected verdict is known
independently of any benchmark example — the point of the tool is that it reports a property of the
data, so its tests should not need a model either.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from petab.v1.C import (MEASUREMENT, OBSERVABLE_ID, SIMULATION_CONDITION_ID, TIME)

from petab_mcp.regimes import data_regimes


class _P:
    """The two tables `data_regimes` reads. Nothing else is touched."""

    def __init__(self, measurement_df, condition_df=None, observable_df=None):
        self.measurement_df = measurement_df
        self.condition_df = condition_df
        self.observable_df = observable_df


class _S:
    def __init__(self, problem):
        self.problem = problem


def _meas(rows) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=[OBSERVABLE_ID, SIMULATION_CONDITION_ID, TIME, MEASUREMENT])


def _series(oid, cid, t, y, reps=2, noise=0.01, seed=0):
    """One trace, with `reps` replicates per point so the noise floor is the unbiased estimate."""
    rng = np.random.default_rng(seed)
    return [(oid, cid, float(ti), float(yi + rng.normal(0, noise)))
            for ti, yi in zip(t, y) for _ in range(reps)]


# -- shape: the pulse that constant production + first-order decay cannot make -----------------
def test_a_rise_and_fall_is_reported_as_pulsatile():
    t = np.linspace(0, 10, 12)
    y = 5.0 * t * np.exp(-t)                      # interior peak, both legs far above noise
    r = data_regimes(_S(_P(_meas(_series("obs", "c0", t, y)))))
    shape = r["observables"]["obs"]["shape"]
    assert shape["testable"] is True
    # The ESTABLISHED half of the contract every test shares. `shape` is the only test that
    # routinely fires, so if it does not fill `requirement` the agent's "act on a non-null
    # requirement" rule has nothing to act on — which is how it shipped.
    assert shape["requirement"] is not None
    assert shape["support_for_extension"] == "established"
    assert "rise AND fall" in shape["requirement"]
    assert shape["n_peaks"] == 1
    assert 0.0 < shape["extremum_time_fraction"] < 0.5      # peaks early, as t*exp(-t) does
    assert shape["min_leg_sigmas"] > 3.0


def test_a_monotone_decay_is_not_called_pulsatile():
    t = np.linspace(0, 10, 12)
    y = 3.0 * np.exp(-0.4 * t)
    r = data_regimes(_S(_P(_meas(_series("obs", "c0", t, y)))))
    shape = r["observables"]["obs"]["shape"]
    assert shape["n_excursions"] == 0
    # Absence of evidence stays absence: no requirement, no support, and the text must NOT read as
    # a licence to keep the model simple — that phrasing is what the A/B measured as harmful.
    assert shape["requirement"] is None
    assert shape["support_for_extension"] == "none"
    assert shape["verdict"].startswith("monotone")
    assert "NOT evidence" in shape["verdict"]


def test_pure_noise_rarely_reads_as_an_excursion():
    """The null that matters, asserted as a RATE.

    Legs are measured to the SURROUNDING minima, so on a flat noisy trace they approach the sample
    RANGE — which grows with series length and would read as a huge excursion against a fixed bar.
    A threshold detector has a false-positive rate by construction, so asserting zero on a handful
    of arbitrary seeds tests luck, not the threshold. Measured 3.0% over 200 seeds at the calibrated
    `_EXCURSION_K`; this asserts the rate has not drifted into double digits, which is what a
    regression in the scaling (or dropping the replicate correction) would do.
    """
    t = np.linspace(0, 10, 12)
    y = np.full_like(t, 2.0)
    fired = sum(
        bool(data_regimes(_S(_P(_meas(_series("obs", "c0", t, y, noise=0.01, seed=s)))))
             ["observables"]["obs"]["shape"]["n_excursions"])
        for s in range(60)
    )
    assert fired <= 6, f"{fired}/60 pure-noise traces read as excursions — threshold has drifted"


# -- timescales -------------------------------------------------------------------------------
def test_two_well_separated_timescales_are_detected():
    t = np.linspace(0, 30, 40)
    y = 5.0 * np.exp(-2.0 * t) + 5.0 * np.exp(-0.08 * t)     # 25x apart
    r = data_regimes(_S(_P(_meas(_series("obs", "c0", t, y, noise=0.005)))))
    ts = r["observables"]["obs"]["timescales"]
    assert ts["testable"] is True
    assert ts["requirement"] == "at least two distinguishable timescales"
    assert ts["support_for_extension"] == "established"
    # POSITIVE favours the extension. The old field was named `_two_vs_one` and computed
    # `one - two`, so its sign contradicted its name in the payload the agent reads.
    assert ts["delta_bic_favouring_two"] > 6.0


def test_a_single_exponential_establishes_no_requirement_and_no_ceiling():
    """One exponential must not produce a verdict the agent can read as 'one is enough'.

    This is the shape of finding that made the tool harmful: an absence of evidence stated as a
    fact ("one timescale is sufficient") is read as a ceiling, and the model is built without a
    process the data simply could not resolve. The test pins BOTH halves — no requirement, and no
    sentence claiming one timescale suffices.
    """
    t = np.linspace(0, 30, 40)
    y = 5.0 * np.exp(-0.3 * t)
    ts = data_regimes(_S(_P(_meas(_series("obs", "c0", t, y, noise=0.005)))))["observables"]["obs"]["timescales"]
    assert ts["requirement"] is None
    # "none", not "none or suggestive": data generated by ONE exponential must not come back as
    # "WEAK SUPPORT for a second relaxing process". Accepting `suggestive` here stopped pinning the
    # false-POSITIVE direction, which is the direction that pushes structure in on noise. Stable:
    # the margin is -1.0 to -7.4 across seeds, nowhere near the > 0 that opens the suggestive branch.
    assert ts["support_for_extension"] == "none"
    v = ts["verdict"]
    # No AFFIRMATIVE ceiling: the words may appear, but only inside the disclaimer denying them.
    assert "timescale is sufficient" not in v
    assert "NO REQUIREMENT" in v and "NOT evidence that one is enough" in v


# -- dose-response ----------------------------------------------------------------------------
def _dose_problem(hill, seed=0):
    doses = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]
    t = np.linspace(0, 5, 6)
    rows, cond = [], {}
    for i, d in enumerate(doses):
        amp = 10.0 * d ** hill / (0.3 ** hill + d ** hill)
        rows += _series("obs", f"c{i}", t, amp * t * np.exp(-t) + 0.1, noise=0.002, seed=seed + i)
        cond[f"c{i}"] = {"dose": d}
    cdf = pd.DataFrame.from_dict(cond, orient="index")
    cdf.index.name = SIMULATION_CONDITION_ID
    return _P(_meas(rows), condition_df=cdf)


def test_a_cooperative_dose_response_is_flagged_as_steeper_than_one_binding_step():
    r = data_regimes(_S(_dose_problem(hill=3.0)))
    dr = r["observables"]["obs"]["dose_response"]
    assert dr["testable"] is True and dr["input"] == "dose"
    assert dr["max_log_log_slope"] > 1.5
    assert "steeper than one binding step" in dr["verdict"]


def test_a_single_step_dose_response_is_not_flagged():
    r = data_regimes(_S(_dose_problem(hill=1.0)))
    dr = r["observables"]["obs"]["dose_response"]
    assert dr["max_log_log_slope"] < 1.5
    assert dr["requirement"] is None
    # Neither direction may be claimed. No affirmative ceiling ("consistent with a single ...", the
    # phrasing the A/B measured as harmful), AND no false positive: a Hill-1 dose-response must not
    # come back as WEAK SUPPORT for cooperativity. Stable at "none" across seeds (slope ~0.71).
    assert "consistent with a single" not in dr["verdict"]
    assert dr["support_for_extension"] == "none"
    assert "steeper than one binding step" not in dr["verdict"]


# -- conservation: the one verdict that REMOVES a parameter, so false positives are expensive ---
def test_a_real_conserved_total_is_found():
    t = np.linspace(0, 10, 12)
    a = 4.0 * np.exp(-0.5 * t)
    rows = _series("A", "c0", t, a, noise=0.005, seed=1) + \
        _series("B", "c0", t, 4.0 - a, noise=0.005, seed=2)   # A + B == 4 exactly
    c = data_regimes(_S(_P(_meas(rows))))["conservation"]
    assert c["testable"] is True
    assert c["n_conditions_holding"] == c["n_conditions_tested"] == 1
    assert "a conserved total is available" in c["verdict"]
    assert c["per_condition"][0]["sum_range_over_largest_component"] < 0.34


def test_two_flat_observables_are_not_a_conserved_total():
    """The sum of two constants is constant. Without requiring the COMPONENTS to move, that reads as
    a conservation law and licenses removing a parameter that was never there."""
    t = np.linspace(0, 10, 12)
    rows = _series("A", "c0", t, np.full_like(t, 2.0), noise=0.005, seed=1) + \
        _series("B", "c0", t, np.full_like(t, 3.0), noise=0.005, seed=2)
    c = data_regimes(_S(_P(_meas(rows))))["conservation"]
    assert "conserved total is available" not in c["verdict"]
    assert c["per_condition"][0]["n_moving"] == 0


def test_unrelated_observables_are_not_a_conserved_total():
    """Two independent decays on different scales: both move, the sum is not markedly steadier than
    its largest component. This is the fujita shape (three unrelated phospho-readouts)."""
    t = np.linspace(0, 10, 12)
    rows = _series("A", "c0", t, 5.0 * np.exp(-0.5 * t), noise=0.005, seed=1) + \
        _series("B", "c0", t, 3.0 * np.exp(-0.2 * t), noise=0.005, seed=2)
    c = data_regimes(_S(_P(_meas(rows))))["conservation"]
    assert "conserved total is available" not in c["verdict"]


# -- honesty about what the design cannot answer ----------------------------------------------
def test_an_undersized_design_reports_unknown_rather_than_a_default():
    """`testable: false` with a reason, never a reassuring 'no' — a design that cannot answer a
    question must not look like one that answered it negatively."""
    rows = _series("obs", "c0", np.array([0.0, 1.0, 2.0]), np.array([1.0, 2.0, 1.5]))
    r = data_regimes(_S(_P(_meas(rows))))["observables"]["obs"]
    assert r["shape"]["testable"] is False and "time points" in r["shape"]["reason"]
    assert r["timescales"]["testable"] is False
    assert r["dose_response"]["testable"] is False


def test_no_measurements_degrades_rather_than_raising():
    empty = _meas([])
    assert data_regimes(_S(_P(empty)))["observables"] == {}


def test_the_noise_method_is_named():
    """A verdict measured against a biased-upward fallback must say so — the caller cannot judge a
    '3 sigma' claim without knowing which sigma."""
    t = np.linspace(0, 10, 12)
    y = 5.0 * t * np.exp(-t)
    with_reps = data_regimes(_S(_P(_meas(_series("obs", "c0", t, y, reps=3)))))
    assert with_reps["observables"]["obs"]["noise_method"] == "replicate"
    no_reps = data_regimes(_S(_P(_meas(_series("obs", "c0", t, y, reps=1)))))
    assert no_reps["observables"]["obs"]["noise_method"] == "difference"
