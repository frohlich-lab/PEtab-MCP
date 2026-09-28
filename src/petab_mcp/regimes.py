"""Model-free regime diagnostics: what the MEASUREMENTS demand, before any model is written.

Every model imposes a factorisation on the data — states acting independently, a single relaxation
timescale, a linear input, no conserved total. A misfit that survives re-fitting is usually one of
those assumptions being false, and the fastest way to find out WHICH is to ask the data directly.

This module answers that from ``measurements.tsv`` and ``conditions.tsv`` ALONE. It never reads the
model, the parameter table or any simulation, which has two consequences worth stating:

* it can be run BEFORE the first model exists, so a structural requirement is known up front rather
  than discovered by ten fits that cannot meet it; and
* nothing it reports can leak an answer, because everything it reports is a property of the shipped
  data. The same code produces the same kind of verdict on a problem nobody has looked at.

Why a tool and not an instruction: in the Aug 8-10 sweep, 49% of the pivotal element-instances in
passing runs were harness-computed telemetry the agent RELAYED (verified — the quoted `(dnll, dscore)`
pairs satisfy the harness's own identity to <0.06, which an agent cannot fake), while no provenance
class distinguished passing runs from failing ones. Numbers the harness computes get used;
procedures the prompt describes are a lottery. So compute the numbers.

THE ANALYSIS COORDINATE. Every verdict is computed where the noise is ADDITIVE, not on the raw
linear scale: an observable declared ``log``/``log10`` is transformed first, and its noise floor and
every sigma-threshold below live in that coordinate. Judging a log-scale observable against one
absolute linear sigma is dimensionally invalid — it hides low-amplitude structure and inflates
high-amplitude wiggle — which is the same trap ``reconcile.sigma_data_std`` documents having fallen
into once already.

Each verdict carries the evidence that produced it and an explicit ``testable`` flag: a design that
cannot answer a question gets "unknown", never a reassuring default. Noise is estimated from
replicates where the design has them and from successive-difference scatter otherwise, with the
method named in the payload.
"""

from __future__ import annotations

import math
from typing import Any

from petab.v1.C import (
    MEASUREMENT,
    OBSERVABLE_ID,
    OBSERVABLE_TRANSFORMATION,
    SIMULATION_CONDITION_ID,
    TIME,
)

# A feature must clear this many noise sigmas to be called real. 3 is the conventional bar and it is
# what the "beyond ~3 sigma" advice downstream is calibrated to; below 2 we actively say "no".
_STRONG_SIGMA = 3.0
_WEAK_SIGMA = 2.0

# Multiplier on the one-sided noise-maximum null for calling an interior excursion real. Calibrated
# empirically, not chosen by taste — see the table in `_shape`.
_EXCURSION_K = 2.25

# A second exponential has to earn its two extra parameters on the usual BIC evidence scale.
# Deliberately BIC and not the run's model-selection criterion (:mod:`criteria`): this is a fixed
# TEST on one observable's shape, run against its own threshold, not a score models are ranked by
# — a diagnostic whose verdict moved when the search objective was reconfigured would be reporting
# the configuration rather than the data.
_TIMESCALE_MIN_DELTA_BIC = 6.0

_MIN_POINTS_SHAPE = 5
# The two-exponential form has FIVE free parameters. Six points let it very nearly interpolate, and
# the BIC penalty difference (2*ln(6) = 3.6) cannot offset the collapse in residual sum of squares:
# measured on pure single-exponential data, 6 points reported "two timescales" in 15% of seeds. The
# floor is set so the richer model is at least ~2.5x over-determined.
_MIN_POINTS_TIMESCALE = 13
_MIN_POINTS_CONSERVATION = 5
_MIN_DOSES = 3

# Payload budget. Every result the lead requests is re-sent in its history on EVERY later turn, so an
# unbounded per-observable report is a standing tax: measured at 61 kB (~15k tokens) on isensee's 123
# observables. Reported observables are capped and the omission is stated, and `observables=` scopes
# the call to the ones actually being reasoned about.
_MAX_OBSERVABLES = 12


def _finite(x) -> "float | None":
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _coordinate(sub, transformation: "str | None") -> tuple[Any, str]:
    """(values in the coordinate where noise is additive, its name).

    Falls back to linear — and says so — when a declared log transform cannot be applied because the
    data contain non-positive values, rather than silently producing -inf or a ~-300 pseudo-value.
    """
    import numpy as np

    y = sub[MEASUREMENT].to_numpy(dtype=float)
    t = (transformation or "lin").strip().lower()
    if t in ("log", "log10") and np.all(y[np.isfinite(y)] > 0):
        return (np.log(y) if t == "log" else np.log10(y)), t
    return y, "lin"


def _noise_floor(frame) -> tuple["float | None", str]:
    """(sigma, method) in the analysis coordinate: pooled replicate scatter, else successive diffs.

    Replicates are the unbiased estimate — repeated measurements at one design point differ only by
    noise. The successive-difference fallback is biased UPWARD (it also contains real curvature), so
    a feature judged against it is judged conservatively, which is the right direction for a tool
    whose verdicts license structural changes.
    """
    import numpy as np

    within, dof = 0.0, 0
    all_y = frame["_y"].to_numpy(dtype=float)  # hoisted: was rebuilt once PER GROUP
    for _, idx in frame.groupby([SIMULATION_CONDITION_ID, TIME], sort=False).indices.items():
        v = all_y[idx]
        v = v[np.isfinite(v)]
        if len(v) > 1:
            within += float(((v - v.mean()) ** 2).sum())
            dof += len(v) - 1
    if dof >= 2:
        return math.sqrt(within / dof), "replicate"

    diffs = []
    for _, g in frame.groupby(SIMULATION_CONDITION_ID, sort=False):
        v = g.sort_values(TIME)["_y"].to_numpy(dtype=float)
        v = v[np.isfinite(v)]
        if len(v) > 2:
            diffs.extend(np.diff(v) / math.sqrt(2.0))
    if len(diffs) >= 3:
        return float(np.std(diffs, ddof=1)), "difference"
    return None, "none"


def _series(frame) -> list[tuple[str, Any, Any, int]]:
    """Per-condition (id, times, mean values, replicates-per-point), steady-state rows dropped."""
    import numpy as np

    out = []
    for cid, g in frame.groupby(SIMULATION_CONDITION_ID, sort=False):
        g = g[np.isfinite(g[TIME])]
        if g.empty:
            continue
        grouped = g.groupby(TIME, sort=True)["_y"]
        m, counts = grouped.mean(), grouped.count()
        t, y = m.index.to_numpy(dtype=float), m.to_numpy(dtype=float)
        keep = np.isfinite(t) & np.isfinite(y)
        if keep.sum() >= 2:
            out.append((str(cid), t[keep], y[keep], int(counts.min())))
    return out


def _extremum_legs(y, idx) -> tuple[float, float]:
    """Rise and fall of the excursion at ``idx``, measured to the SURROUNDING minima.

    Measuring from the first and last samples instead silently discards every trace that starts or
    ends near its peak — an oscillator, or any readout already rising when observation began. On the
    shipped predator_prey data that turned an 11-sigma excursion into "flat/indeterminate".
    """
    import numpy as np

    left = float(y[idx] - np.min(y[: idx + 1]))
    right = float(y[idx] - np.min(y[idx:]))
    return left, right


def _shape(series, sigma) -> dict[str, Any]:
    """Does the response show an interior EXCURSION — a peak or a trough — or only relax?

    The distinction is structural, not cosmetic: constant production balanced by first-order decay
    cannot produce an interior extremum under a sustained input, whatever its rate constants. Such an
    excursion therefore REQUIRES a depletable pool, a second slower state, or feedback. A trough
    falsifies the same form as a peak, so both are detected.

    ONE condition clearing the bar is decisive — the topology is falsified by a single counterexample
    — so there is no quorum over conditions. The 3-sigma test on both legs is the noise guard; a
    quorum on top of it would only discard real evidence, which matters because in a dose design it
    is normal for the low doses to be flat and only the top one or two to peak.
    """
    import numpy as np

    if sigma is None or sigma <= 0:
        return {"testable": False, "reason": "no noise estimate"}
    usable = [s for s in series if len(s[1]) >= _MIN_POINTS_SHAPE]
    if not usable:
        return {"testable": False, "reason": f"fewer than {_MIN_POINTS_SHAPE} time points"}

    peaks, troughs, monotone, frac, strength = 0, 0, 0, [], []
    for _, t, y, reps in usable:
        hit = False
        # `y` is a replicate MEAN, so its noise is sigma/sqrt(reps); judging its excursions against
        # the per-MEASUREMENT sigma would understate their significance by that factor. Same
        # correction as `_conservation` makes for the same reason.
        sig_m = sigma / math.sqrt(max(reps, 1))
        # Legs are measured to the SURROUNDING MINIMA, so under pure noise a leg is a one-sided
        # maximum deviation and BOTH legs together are close to the sample RANGE — which grows with
        # the series length. A fixed 3-sigma bar therefore calls noise an excursion more and more
        # often the longer the series. The bar scales as _EXCURSION_K * sqrt(2 ln n) * sigma;
        # measured false-positive rate on pure noise, 3000 draws per cell:
        #
        # Calibrated END TO END through this module (200 pure-noise problems, 12 points, 2
        # replicates, sigma estimated from the replicates exactly as a caller's would be) against
        # whether fujita's real pulses survive:
        #
        #        k    false positives on pure noise    fujita excursions
        #      2.00              6.5%                          4
        #      2.25              3.0%                          2   <- chosen
        #      2.50              1.5%                          0   (loses the flagship case)
        #
        # Note the rate is ~15x the idealised one (0.2% at k=2.25 with a KNOWN sigma) because sigma
        # is itself estimated here, and an underestimate inflates every feature. Quote 3%, not 0.2%.
        # Floored at the standing 3-sigma bar so a very short series is never judged more loosely.
        bar = sig_m * max(_STRONG_SIGMA, _EXCURSION_K * math.sqrt(2.0 * math.log(max(len(y), 2))))
        for kind, i in (("peak", int(np.argmax(y))), ("trough", int(np.argmin(y)))):
            if not (0 < i < len(y) - 1):
                continue
            a, b = _extremum_legs(y if kind == "peak" else -y, i)
            if a > bar and b > bar:
                peaks, troughs = peaks + (kind == "peak"), troughs + (kind == "trough")
                frac.append(float(t[i] / t[-1]) if t[-1] > 0 else 0.0)
                strength.append(float(min(a, b) / sig_m))
                hit = True
        if hit:
            continue
        # Genuinely monotone, not merely "the ends differ": a trace that dips and returns is neither
        # an excursion (legs too small) nor monotone, and calling it monotone would be a false claim.
        d = np.diff(y)
        # `sig_m`, not `sigma`: `y` is a replicate mean, so a DIFFERENCE of two of them has sd
        # sigma*sqrt(2/reps). Testing against the per-measurement sigma made the bar stricter the
        # MORE replicates a design had, degrading a clean monotone decay to "flat/indeterminate".
        big = np.abs(d) > _WEAK_SIGMA * sig_m * math.sqrt(2.0)
        if big.any() and (np.all(d[big] > 0) or np.all(d[big] < 0)):
            monotone += 1
    n_exc = peaks + troughs
    # `requirement` / `support_for_extension`, in the same shape the other tests use. This one is
    # the reason the fields exist at all: an interior excursion is the module's only SINGLE-
    # COUNTEREXAMPLE finding (no rate constants make constant production plus first-order decay
    # rise and fall), so it is the one test that routinely establishes anything -- and it was the
    # one test that never filled the field the tool description, the lead prompt and the server
    # docstring all tell the agent to act on. The agent was reading None for the only real finding
    # it gets, while the two tests that can essentially never fire were the only ones able to
    # report. Absence stays absence: monotone/flat set `requirement: None`.
    _EXCURSION = (
        "the response must rise AND fall — constant production with first-order decay "
        "cannot produce this at any rate constants, so a depletable pool, a second "
        "state, or feedback is REQUIRED"
    )
    return {
        "testable": True,
        "n_conditions": len(usable),
        "n_excursions": n_exc,
        "n_peaks": peaks,
        "n_troughs": troughs,
        "n_monotone": monotone,
        "extremum_time_fraction": round(float(np.median(frac)), 3) if frac else None,
        "min_leg_sigmas": round(float(np.median(strength)), 1) if strength else None,
        "requirement": _EXCURSION if n_exc else None,
        "support_for_extension": "established" if n_exc else "none",
        "verdict": (
            _EXCURSION
            if n_exc
            else "monotone over every condition tested — NO REQUIREMENT either way, and NOT "
            "evidence that one state suffices"
            if monotone
            else "flat/indeterminate: this design does not resolve the shape — NO REQUIREMENT "
            "either way, and NOT evidence that the response is simple"
        ),
    }


def _timescales(series, sigma) -> dict[str, Any]:
    """One relaxation timescale, or more? Compared by BIC so the extra pair of parameters must be
    earned, on a series long enough that the 5-parameter form cannot simply interpolate.

    LIKE `_dose_response`, THIS NEVER FIRED ON ANY PROBLEM WE HAVE, and for the same reason: the
    bar sits above what these designs can clear. Measured over 184 calls in 20 traced runs -- fujita
    is `testable: false` 188/188 (8 points per condition, under `_MIN_POINTS_TIMESCALE`), and sneyd
    is `testable` but lands at margin -4.2 in ALL 90 calls: it has exactly 13 points, the bare
    minimum, where a 5-parameter two-exponential is BIC-penalised out of existence. Not near the
    threshold -- the wrong side of it, deterministically, on a problem whose reference model has
    k=15. So `support: none` here is a statement about the gate, not about the system.
    Read the note on `z` in `_dose_response` for why a null verdict is not a neutral one."""
    import numpy as np

    if sigma is None or sigma <= 0:
        return {"testable": False, "reason": "no noise estimate"}
    # Every condition long enough, not just the longest. `max(..., key=len)` is first-wins on ties,
    # and a dose design shares one time grid across doses -- so it always analysed whichever
    # condition came first in the table, conventionally the vehicle control, which by construction
    # has the least dynamics. `_shape` already scans them all for the same reason.
    usable = [s for s in series if len(s[1]) >= _MIN_POINTS_TIMESCALE]
    cand = max(usable, key=lambda s: float(np.ptp(s[2])), default=None)
    if cand is None or len(cand[1]) < _MIN_POINTS_TIMESCALE:
        return {
            "testable": False,
            "reason": f"fewer than {_MIN_POINTS_TIMESCALE} time points (a two-exponential form "
            f"has 5 free parameters and would interpolate a shorter series)",
        }
    _, t, y, _reps = cand
    i = int(np.argmax(y))
    if len(t) - i < _MIN_POINTS_TIMESCALE:
        i = 0
    t, y = t[i:] - t[i], y[i:]
    if len(t) < _MIN_POINTS_TIMESCALE or not np.ptp(y):
        return {"testable": False, "reason": "no resolvable decay"}

    try:
        from scipy.optimize import curve_fit
    except Exception:  # pragma: no cover - scipy always present with pypesto
        return {"testable": False, "reason": "scipy unavailable"}

    def _bic(fn, p0, k):
        try:
            # The optimizer explores large rate constants en route, so `exp` overflows harmlessly.
            with np.errstate(over="ignore", invalid="ignore"):
                popt, _ = curve_fit(fn, t, y, p0=p0, maxfev=20000)
        except Exception:  # noqa: BLE001 - a failed fit is simply no evidence for that form
            return None
        rss = float(((y - fn(t, *popt)) ** 2).sum())
        n = len(t)
        return n * math.log(max(rss, 1e-300) / n) + k * math.log(n)

    span = float(t[-1]) or 1.0
    a0, c0 = float(y[0] - y[-1]), float(y[-1])
    b1 = _bic(lambda x, a, k, c: a * np.exp(-k * x) + c, [a0, 1.0 / span, c0], 3)
    b2 = _bic(
        lambda x, a, k1, b, k2, c: a * np.exp(-k1 * x) + b * np.exp(-k2 * x) + c,
        [a0 / 2, 3.0 / span, a0 / 2, 0.3 / span, c0],
        5,
    )
    if b1 is None:
        return {"testable": False, "reason": "single-exponential fit failed"}
    if b2 is None:
        # An optimizer failure is not a negative result. Falling through with `b2 is None` reported
        # `testable: True, support: none` -- byte-identical to a design that genuinely cannot
        # distinguish a second timescale -- so "the comparison did not run" and "the comparison ran
        # and found nothing" were indistinguishable downstream. Same treatment as the
        # single-exponential failure above: a question the analysis could not answer is untestable.
        return {
            "testable": False,
            "reason": "two-exponential fit failed, so the comparison with "
            "the single-exponential form could not be made",
        }
    # POSITIVE, favouring the EXTENSION: > 0 means the two-timescale form has the lower BIC.
    # Named for what it measures — `delta_bic_two_vs_one` was computed as `b1 - b2` and so read
    # backwards, shipping a number that contradicted the sentence beside it.
    # Threshold on the RAW difference and round only for display: rounding first moves the
    # boundary, so a true margin of 6.04 would round to 6.0 and stop being "established".
    raw_margin = b1 - b2  # both forms fitted: the comparison is real
    margin = round(raw_margin, 1)
    established = raw_margin > _TIMESCALE_MIN_DELTA_BIC
    if established:
        verdict = "at least two distinguishable timescales"
        support = "established"
    elif raw_margin > 0:
        verdict = (
            f"a second timescale is weakly favoured (BIC lower by {margin}, under the "
            f"{_TIMESCALE_MIN_DELTA_BIC} needed to call it) — WEAK SUPPORT for a second "
            "relaxing process, not a finding"
        )
        support = "suggestive"
    else:
        verdict = (
            "this design does not distinguish a second timescale — NO REQUIREMENT either "
            "way, and NOT evidence that one is enough"
        )
        support = "none"
    return {
        "testable": True,
        "n_points": int(len(t)),
        # Only ever set when the extension is ESTABLISHED. Absence is absence of evidence.
        "requirement": "at least two distinguishable timescales" if established else None,
        "support_for_extension": support,
        "delta_bic_favouring_two": margin,
        "verdict": verdict,
    }


def _dose_response(frame, conditions, sigma, coordinate) -> dict[str, Any]:
    """Steepest local slope of response against a numeric input, in log-log.

    A slope reliably above 1 cannot come from a single mass-action binding step — it needs
    cooperativity, a cascade, or saturation elsewhere. The estimate is a MAXIMUM over adjacent dose
    pairs and over candidate input columns, so its band is Bonferroni-widened by the number of
    comparisons actually made; without that the winner crosses the bar on noise alone.
    """
    import numpy as np

    if sigma is None or conditions is None or getattr(conditions, "empty", True):
        return {"testable": False, "reason": "no condition table or noise estimate"}

    # AMPLITUDE per condition = max over time of the replicate MEAN. Two traps avoided: the max over
    # raw replicates is biased upward by an amount that grows with the group size (which varies
    # across conditions, faking a dose trend), and a median over the whole time course measures the
    # baseline rather than the response whenever the trace is a pulse.
    grouped = frame.groupby([SIMULATION_CONDITION_ID, TIME])["_y"]
    peak = grouped.mean().groupby(level=0).max()
    # `peak` is a MEAN of replicates, so its noise is sigma/sqrt(reps). Propagating the raw
    # per-measurement sigma widened the slope band by ~sqrt(reps) and turned a genuinely cooperative
    # dose-response into "consistent with a single binding step" — a false negative in the direction
    # that costs the run the insight. `_shape` and `_conservation` both make this correction.
    _reps = int(grouped.count().min() or 1)
    sigma = sigma / math.sqrt(max(_reps, 1))
    best, n_comparisons = None, 0
    for col in conditions.columns:
        vals = {}
        for cid in peak.index:
            if cid not in conditions.index:
                continue
            d = _finite(conditions.at[cid, col])
            if d is not None and d > 0:
                vals[cid] = d
        if len(set(vals.values())) < _MIN_DOSES:
            continue
        pts = sorted((d, float(peak[c])) for c, d in vals.items())
        x = np.log10([p[0] for p in pts])
        y = np.array([p[1] for p in pts], dtype=float)
        if not np.all(np.isfinite(y)):
            continue
        # In a log coordinate the response IS already log; in a linear one it must be logged, which
        # requires positivity. A non-positive response is not an error and must not crash the tool:
        # skip the column and say so. (alkan ships 384 non-positive measurements.)
        if coordinate in ("log", "log10"):
            ly = y / math.log(10.0) if coordinate == "log" else y
            ly_sigma = sigma / math.log(10.0) if coordinate == "log" else sigma
        else:
            if np.any(y <= 0):
                continue
            ly, ly_sigma = np.log10(y), None
        slopes, unc = [], []
        for a in range(len(x) - 1):
            dx = x[a + 1] - x[a]
            if dx <= 0:
                continue
            slopes.append((ly[a + 1] - ly[a]) / dx)
            if ly_sigma is not None:
                s2 = 2.0 * ly_sigma**2
            else:  # propagate the LINEAR sigma through the log at each level
                s2 = sum((sigma / v) ** 2 for v in (y[a], y[a + 1])) / (math.log(10) ** 2)
            unc.append(math.sqrt(s2) / dx)
        if not slopes:
            continue
        n_comparisons += len(slopes)
        j = int(np.argmax(slopes))
        # The rounded pair is what the PAYLOAD shows; the raw pair is what the verdict is decided
        # on. Deciding on the rounded ones let an uncertainty below 0.005 round to exactly 0.0 and
        # collapse the Bonferroni band to nothing, degenerating `established` into the bare
        # point-estimate test the widening exists to prevent. Bounded (the band is worth <=0.02
        # slope units there, and every real problem sits at 0.31-1.35, far above it) but wrong in
        # the same way as thresholding a rounded BIC margin: a threshold never reads a display value.
        cand = {
            "input": str(col),
            "n_doses": len(pts),
            "max_log_log_slope": round(float(slopes[j]), 2),
            "slope_uncertainty": round(float(unc[j]), 2),
            "_raw_slope": float(slopes[j]),
            "_raw_unc": float(unc[j]),
        }
        if best is None or cand["max_log_log_slope"] > best["max_log_log_slope"]:
            best = cand
    if best is None:
        return {
            "testable": False,
            "reason": f"no numeric input with >={_MIN_DOSES} positive levels and a positive "
            f"response",
        }
    # Bonferroni over every (column, adjacent-pair) slope the maximum was taken over.
    #
    # THIS `z` IS WHY THE ARM MEASURED HARMFUL. Read before touching the verdict wording below:
    # the wording was already rewritten once and it did NOT fix the arm (sneyd 1/20 -> 4/10, still
    # short of 15/20 with the tool OFF). The binding constraint is this line.
    #
    # At the sizes these designs give, `established` is unreachable. sneyd: slope 1.85 +- 0.89 with
    # z=3.97 over 7 comparisons, so calling it needs slope > 1 + 3.97*0.89 = 4.53. fujita pEGFR:
    # 1.19 +- 1.35, needs > 6.12. No biochemical dose-response has a log-log slope of 4.5-6, so the
    # probe returns `requirement: null` on every problem we have -- while the CENTRAL estimate of
    # 1.85 is most of a second binding step and is really there.
    #
    # A null here is not neutral, which is the part that cost a 115-run A/B to learn. Traced over
    # 40 runs: the discriminating feature on sneyd is a power in a rate law (present in 80% of best
    # models with the tool off vs 35% with it on, and worth 26-33 bic in BOTH arms).
    #
    # NOTE WHAT THAT EXPONENT IS, because the obvious reading is wrong: the reference model has NO
    # exponent anywhere. It is the six-state Sneyd-Falcke gating network with SATURATING rate laws
    # (`(k4*L5 + l6)*Ca/(L5 + Ca)`), k=15. The agent's exponent is a SURROGATE for the states it has
    # not built -- a way to sharpen a too-broad bell-shaped Ca response at k~9 without adding a
    # species, in the agent's own words. So this test does not suppress "a correct cooperativity
    # prior"; it suppresses the cheapest available REPAIR for a model that is still too small. That
    # repair is what buys the bic, which is why the harm survives however the verdict is worded.
    #
    # WHERE the exponent comes from, and why the timing is everything (sneyd, 20 runs per arm):
    #
    #                                      tool OFF   tool ON
    #   at fit 1, from the pre-fit panel        12         1
    #   later, forced by residual structure      7        10
    #   never                                    1         9
    #
    # Two routes, and this test decides which one the run gets. `get_data_regimes` is consulted
    # BEFORE the first model in 10/10 traced runs -- that is its whole selling point, valid with no
    # model -- so its only chance to speak lands exactly where the pre-fit route would fire, and
    # what it says there is "this design does not establish a slope above one". The up-front route
    # collapses 12 -> 1 and the run is left re-deriving the same structure from residuals fit by
    # fit, 9/20 times never getting there. The arms' bic curves are identical over fits 1-3 and
    # separate from fit 5 onward, which is exactly that delay.
    #
    # It is NOT literature-driven, so do not look for the fix in the biocurator: only 2 of 19
    # introductions cite curated biology or a PMID, and BOTH arms cite PMIDs in 20/20 first models.
    # The attribution is the systems-scientist mechanistic lens up front, residual diagnostics late.
    #
    # The damage is therefore confined to problems that HAVE the structure this test is too weak to
    # establish. fujita's slopes are 0.58-1.19 — shallow, so the null is roughly right there and the
    # arms come out level (median best bic +141.1 on vs +143.9 off). sneyd's 1.85 is the case that
    # costs. Do not validate a fix on fujita: it cannot tell the arms apart.
    #
    # So: do not re-litigate the phrasing. Either make this test able to fire at realistic effect
    # sizes and design sizes, or keep the arm off (petab_harness.ablations.data_regimes).
    # TWO bars, and the reported slope is a MAXIMUM over every column and adjacent dose pair, so
    # neither may be the bare point estimate. `selection` is the expected maximum of that many
    # standard normals -- the widening the selection itself forces, before any confidence margin --
    # and `z` adds the margin on top for the ESTABLISHED bar.
    selection = math.sqrt(2.0 * math.log(max(n_comparisons, 2)))
    z = _WEAK_SIGMA + selection
    slope, unc_j = best["max_log_log_slope"], best["slope_uncertainty"]
    raw_slope, raw_unc = best.pop("_raw_slope"), best.pop("_raw_unc")
    established = raw_slope - z * raw_unc > 1.0
    # SUGGESTIVE clears the selection bias but not the confidence margin. It used to be the raw
    # `slope > 1.0`, which is the un-widened comparison this function's own docstring says "the
    # winner crosses on noise alone" -- so fujita pEGFR, 1.19 +- 1.35 with the uncertainty LARGER
    # than the distance to one, was returning "WEAK SUPPORT ... a reason to TRY it" on nothing.
    # Pushing structure in on noise is the one failure this tool must not add to the one it has.
    suggestive = raw_slope - selection * raw_unc > 1.0
    _STEEP = "steeper than one binding step (cooperativity / cascade / saturation elsewhere)"
    if established:
        verdict, support = _STEEP, "established"
    elif suggestive:
        # The estimate survives the selection bias but not the confidence margin. Reporting that as
        # "consistent with a single binding step" states the one reading the data leans AGAINST, and
        # that phrasing is what had to go: a model built to a single binding step would start from a
        # shape the data disputes.
        # round(z): the raw float renders as `+-3.9704896987915534x0.89` in text the lead re-sends
        # in its history every later turn, and `band_z` beside it is already rounded to 2 dp.
        verdict = (
            f"central slope {slope} exceeds one by more than the selection bias alone "
            f"(+-{round(selection, 2)}x{unc_j}), but not by the full band (+-{round(z, 2)}x"
            f"{unc_j}), so a single binding step is not excluded — WEAK SUPPORT for "
            "cooperativity / a cascade / saturation, not a finding"
        )
        support = "suggestive"
    else:
        verdict = (
            "this design does not establish a slope above one — NO REQUIREMENT either way, "
            "and NOT evidence that a single binding step suffices"
        )
        support = "none"
    best.update(
        {
            "testable": True,
            "n_comparisons": n_comparisons,
            "band_z": round(z, 2),
            "requirement": _STEEP if established else None,
            "support_for_extension": support,
            "verdict": verdict,
        }
    )
    return best


def _conservation(by_obs) -> dict[str, Any]:
    """Is some SUM of co-measured observables constant while its parts move?

    A conserved total is the one finding here that makes a model CHEAPER — imposing it removes a
    parameter — so a false positive is expensive and the test is deliberately strict:

    * WITHIN a condition. A real law holds per experiment; different conditions load different
      totals, and pooling them measures the between-condition step instead of the law.
    * LIKE FOR LIKE. The sum's peak-to-trough is compared against the components' peak-to-trough.
      Comparing a standard deviation against a range passes on the unit mismatch alone -- that is
      exactly how the three-gene repressilator, which has no conservation law at all, read as
      conserved because its species are ~120 degrees out of phase.
    * The components must MOVE, or the sum of two constants is trivially constant.
    """
    import numpy as np

    # A conserved total is a statement about a SUM, which only means anything if the terms share
    # units. Every observable here carries its own analysis coordinate, so an all-log problem
    # (blasi, schwen, and 100 of bachmann's 101) would be testing whether sum(log y) = log(prod y)
    # is constant -- a different law entirely -- and a mixed problem (lucarelli) would be adding
    # log values to linear ones outright. Restrict to the linear-coordinate observables and say so.
    linear = sorted(o for o in by_obs if by_obs[o].get("coordinate", "lin") == "lin")
    skipped = len(by_obs) - len(linear)
    obs = linear
    if len(obs) < 2:
        return {
            "testable": False,
            "reason": (
                "fewer than two observables on a linear scale"
                + (
                    f" ({skipped} are log-transformed, where a SUM is not the quantity a "
                    f"conservation law constrains)"
                    if skipped
                    else ""
                )
            ),
        }

    per_cond, tested = [], 0
    conds = {c for o in obs for c, _, _, _ in by_obs[o]["series"]}
    for cid in sorted(conds):
        # The observables co-measured in THIS condition, at their common time points. Requiring all
        # observables at one time is hopeless on a real design (bachmann has 101, isensee 123).
        curves = {}
        for o in obs:
            for c, t, y, reps in by_obs[o]["series"]:
                if c == cid:
                    curves[o] = (t, y, reps)
        if len(curves) < 2:
            continue
        common = set.intersection(*(set(t.tolist()) for t, _, _ in curves.values()))
        if len(common) < _MIN_POINTS_CONSERVATION:
            continue
        ts = np.array(sorted(common))
        stack, sig2, ranges, reps_by_obs = [], 0.0, {}, {}
        ok = True
        for o, (t, y, reps) in curves.items():
            s = by_obs[o]["sigma"]
            if s is None:
                ok = False
                break
            v = np.array([y[int(np.where(t == x)[0][0])] for x in ts], dtype=float)
            stack.append(v)
            ranges[o] = float(np.ptp(v))
            reps_by_obs[o] = reps
            # `y` is a replicate MEAN, so its noise is sigma/sqrt(reps); using the per-measurement
            # sigma here would make the constancy test ~sqrt(reps) too permissive.
            sig2 += (s**2) / max(reps, 1)
        if not ok or not stack:
            continue
        tested += 1
        tot = np.sum(stack, axis=0)
        sig = math.sqrt(sig2)
        npts = len(ts)
        noise_range = 2.0 * math.sqrt(2.0 * math.log(max(npts, 2))) * sig
        # "Does THIS component move" is judged against ITS OWN noise, not the pooled sigma of the
        # sum, and against one expected noise range rather than two. Judging it against the sum's
        # sigma made the bar scale with the number of observables and their combined noise, so it
        # was unreachable on every shipped example -- repressilator's three species swing ~14x their
        # own noise floor and still scored `n_moving: 0`. That made `holds` (which needs >=2 moving)
        # impossible in production, i.e. the one verdict advertised as making a model CHEAPER could
        # never fire. The SUM's constancy is still judged against the pooled sigma below, which is
        # the right yardstick for a sum.
        # 1.5x the expected noise range of ITS OWN noise. Judging against the sum's pooled sigma
        # (and a 2x bar) made this unreachable on every shipped example; dropping the margin
        # altogether put a genuinely FLAT pair right on the boundary, since the observed range of n
        # noise samples lands at its expectation about half the time. 1.5x clears both: flat pairs
        # stay still, and repressilator's ~14-sigma swings still count as moving.
        moving = [
            o
            for o in ranges
            if ranges[o]
            > 1.5
            * 2.0
            * math.sqrt(2.0 * math.log(max(npts, 2)))
            * ((by_obs[o]["sigma"] or float("inf")) / math.sqrt(max(reps_by_obs.get(o, 1), 1)))
        ]
        largest = max(ranges.values()) if ranges else 0.0
        sum_range = float(np.ptp(tot))
        per_cond.append(
            {
                "condition": cid,
                "n_points": npts,
                "n_moving": len(moving),
                "sum_range_over_largest_component": round(sum_range / largest, 3)
                if largest
                else None,
                # 1.5x the EXPECTED noise range, not 1.0x: the observed range of a genuinely constant
                # sum lands at the expectation about half the time, so a bar set exactly there is a coin
                # flip on real conservation. The like-for-like ratio below is the discriminating test.
                "holds": bool(
                    sum_range < 1.5 * noise_range
                    and len(moving) >= 2
                    and largest
                    and sum_range < largest / _STRONG_SIGMA
                ),
            }
        )
    if not tested:
        return {
            "testable": False,
            "reason": (
                f"no condition has >={_MIN_POINTS_CONSERVATION} time points where at "
                f"least two observables are measured together"
            ),
        }
    holds = [c for c in per_cond if c["holds"]]
    established = len(holds) == tested
    _CONSERVED = (
        "the summed observables are constant while the components vary in every "
        "condition tested — a conserved total is available, and imposing it REMOVES a "
        "parameter"
    )
    return {
        "testable": True,
        "observables": obs,
        "n_conditions_tested": tested,
        "log_observables_excluded": skipped,
        "n_conditions_holding": len(holds),
        "per_condition": per_cond[:6],
        # Same contract as the other three tests -- see the note in `_shape`. Note this one can
        # only ever REMOVE a parameter, so a null here costs nothing structural; it is filled for
        # consistency, so "every entry carries a requirement" is true of the payload as a whole.
        "requirement": _CONSERVED if established else None,
        "support_for_extension": (
            "established" if established else "suggestive" if holds else "none"
        ),
        "verdict": (
            _CONSERVED
            if established
            else f"a conserved total holds in only {len(holds)} of {tested} conditions — not a "
            "law, and WEAK SUPPORT at most"
            if holds
            else "no conserved total over these observables — NO REQUIREMENT either way"
        ),
    }


def data_regimes(session, observables: "str | None" = None) -> dict[str, Any]:
    """Model-free structural requirements read off the measurements. See the module docstring."""
    problem = session.problem
    mdf = getattr(problem, "measurement_df", None)
    empty = mdf is None or getattr(mdf, "empty", True)
    cdf = getattr(problem, "condition_df", None)
    odf = getattr(problem, "observable_df", None)

    out: dict[str, Any] = {}
    by_obs: dict[str, dict[str, Any]] = {}
    omitted = 0
    if not empty:
        wanted = {s.strip() for s in observables.split(",")} if observables else None
        ids = sorted(mdf[OBSERVABLE_ID].unique())
        if wanted:
            ids = [i for i in ids if str(i) in wanted]
        omitted = max(0, len(ids) - _MAX_OBSERVABLES)
        for oid in ids[:_MAX_OBSERVABLES]:
            sub = mdf[mdf[OBSERVABLE_ID] == oid].copy()
            trans = None
            if odf is not None and OBSERVABLE_TRANSFORMATION in getattr(odf, "columns", []):
                if oid in odf.index:
                    trans = str(odf.at[oid, OBSERVABLE_TRANSFORMATION])
            sub["_y"], coord = _coordinate(sub, trans)
            sigma, method = _noise_floor(sub)
            series = _series(sub)
            by_obs[str(oid)] = {"sigma": sigma, "series": series, "coordinate": coord}
            out[str(oid)] = {
                "n_points": int(len(sub)),
                "n_conditions": int(sub[SIMULATION_CONDITION_ID].nunique()),
                "analysis_coordinate": coord,
                "declared_transformation": trans,
                "noise_floor": None if sigma is None else round(sigma, 6),
                "noise_method": method,
                "shape": _shape(series, sigma),
                "timescales": _timescales(series, sigma),
                "dose_response": _dose_response(sub, cdf, sigma, coord),
            }
    payload = {
        "observables": out,
        "conservation": (
            _conservation(by_obs)
            if by_obs
            else {
                "testable": False,
                "reason": (
                    "no measurement table"
                    if empty
                    else "the `observables` filter matched no observable id"
                ),
            }
        ),
        "how_to_read": (
            "Two kinds of entry, and they carry different weight. A non-null `requirement` is "
            "something ANY model must reproduce, measured in the coordinate where that "
            "observable's noise is additive (`analysis_coordinate`) against its own noise floor: "
            "an interior excursion rules out constant production with first-order decay, two "
            "distinguishable timescales rule out a single relaxing pool, a log-log slope above "
            "one rules out a single mass-action binding step. Everything else is EVIDENCE, graded "
            "by `support_for_extension` — `suggestive` means the estimate leans toward the richer "
            "structure without establishing it, and is a reason to TRY that structure. "
            "THESE FINDINGS CAN FORCE STRUCTURE IN; THEY NEVER RULE IT OUT. `testable: false` "
            "means the design cannot answer, and a null `requirement` means this design did not "
            "settle it — neither is permission to simplify. No evidence for a second timescale is "
            "not evidence that one suffices, and a slope this design cannot push above one may "
            "still be above one. Build what the mechanism needs. The one entry that genuinely "
            "constrains is `conservation`: imposing a conserved total removes a parameter."
        ),
    }
    if omitted:
        payload["omitted_observables"] = omitted
        payload["note"] = (
            f"{omitted} further observable(s) not reported (cap {_MAX_OBSERVABLES}), and the "
            f"conservation verdict was computed on the {_MAX_OBSERVABLES} shown -- an alphabetical "
            f"slice, not a co-measured set -- so treat it as provisional here. Pass "
            f'`observables="id1,id2"` to scope this call to the ones you are reasoning about.'
        )
    if empty:
        payload["note"] = "no measurement table"
    return payload
