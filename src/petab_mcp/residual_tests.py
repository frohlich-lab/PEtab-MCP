"""Residual-structure tests: the sign test, the Wald–Wolfowitz runs test, the replicate
lack-of-fit test, and a per-sigma variance-heterogeneity test.

These answer "do the residuals still show systematic structure?" — the half of the
fit-adequacy question the magnitude-based scores (``reduced_chi2``, the χ² GoF p-value,
``mean_weighted_residual``) cannot answer when the observable noise sigmas are ESTIMATED.
With a free σ the optimizer inflates the noise to absorb misfit, so every weighted residual
shrinks uniformly toward zero and the magnitude scores pin near their null regardless of how
wrong the model's FORM is (issue #54). Both tests here look only at the SIGN of each residual,
so a uniform rescaling of the residuals leaves them unchanged — they detect a wrong form that
the magnitude scores are blind to, using only the fit's own residuals (no reference model, so
no answer leak).

* **Sign test** — of the non-zero residuals for an observable, how many are positive? Under a
  correct model each sign is ~equally likely, so a lopsided count is a systematic over- or
  under-shoot (bias). Exact two-sided binomial p-value.

* **Wald–Wolfowitz runs test** — order a series' residuals by time and count the *runs*
  (maximal same-sign stretches). Too FEW runs means long same-sign stretches: the model drifts
  systematically across the time course (positive serial correlation — the signature of a
  missing dynamic term). Computed from the EXACT small-sample distribution of the run count,
  because the usual normal approximation is invalid at the handful of points per series a typical
  problem has.

* **Replicate lack-of-fit** — where a design point was measured SEVERAL times, its replicates
  differ by noise alone, so replicate scatter is a model-free yardstick for how far the prediction
  is allowed to sit from the data. It is pooled over every replicated point whose noise the same
  sigma governs (``s_g``, :func:`pooled_replicate_scatter`):
  ``z_i = (ybar_i - f_i) / (s_g / sqrt(n_i))``. This is the axis the two sign-based tests cannot
  reach on a steady-state problem, where a design point IS an observable and every series has
  length 1 (see :func:`replicate_lack_of_fit`).

* **Variance heterogeneity** — whether the series one estimated noise sigma covers share one
  spread of weighted residuals (:func:`sigma_heterogeneity`), naming the series it was inflated
  for. It feeds no gate: it ranks where to look.

Everything here is pure numeric code over lists of numbers; :mod:`petab_mcp.data_access`
feeds it the per-point residuals (and :mod:`petab_mcp.reconcile` the per-point replicate
statistics, on the coordinate where the noise is additive) and surfaces the results through
the ``fit`` result's ``residual_structure`` verdict (the per-observable detail behind it goes
out through ``get_fit_diagnosis``).

THE FIRST THREE FEED THE SESSION'S PLATEAU-ADEQUACY GATE, through the one ``flagged_observables``
list ``session.py`` reads via ``data_access.residual_flag`` — the sign/runs tests have fed it since
PR #111 (underfits that plateau-stopped as "adequate" with a plausible sigma while flagging
structured residuals), and the replicate test joins the same list rather than adding a gate.
So loosening any alpha here loosens a STOP CONDITION; none of this is merely advisory. What differs
is POWER, not consequence: at these sample sizes the sign/runs tests are weak, so a non-significant
result there is reported as ``low_power`` rather than as a pass, while the replicate test has real
power wherever design points repeat AND enough LACK-OF-FIT degrees of freedom remain after the fit
(:func:`lof_power`: below ``REPLICATE_LOF_MIN_LOF_DF`` of them a PASS is reported ``weak``, and with
none left — or with no replicated design points at all — the arm reports NO EVIDENCE). That df is a
WHOLE-PROBLEM count — replicated points add across noise families while the structural parameters
are shared — so :func:`replicate_lack_of_fit` here judges NO df at all: it tests one family's points
and ``data_access._merge_lof_verdicts`` applies :func:`lof_power` once, to the pooled total. It is
the FLAG that gates, on all three tests; ``weak`` and ``low_power`` are advisory qualifications of a
non-flag, read by the prose and by no stop condition.
"""

from __future__ import annotations

import math
from typing import Any

from .constants import (
    REPLICATE_LOF_ALPHA,
    REPLICATE_LOF_MAX_REPORTED_POINTS,
    REPLICATE_LOF_MIN_DF,
    REPLICATE_LOF_MIN_LOF_DF,
    REPLICATE_LOF_MIN_POINTS,
)

#: Below this many usable points a series/observable has too little power for the p-value to
#: mean much; results are still returned but flagged ``low_power`` so a caller does not read a
#: non-significant result as "the fit is fine".
MIN_POWER_N = 8


def _signs(residuals: list[float]) -> list[int]:
    """+1 / -1 per residual, dropping zeros and non-finite values.

    Exactly-zero residuals carry no sign information (the sign test's null is about the sign of
    a non-zero deviation) and would bias the runs count, so both tests operate on this list.
    """
    out: list[int] = []
    for r in residuals:
        if r is None:
            continue
        try:
            f = float(r)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(f) or f == 0.0:
            continue
        out.append(1 if f > 0 else -1)
    return out


def sign_test(residuals: list[float]) -> dict[str, Any]:
    """Two-sided exact binomial sign test on the residual signs.

    ``p`` is the probability, under an unbiased model (each sign equally likely), of a split at
    least as lopsided as observed. Small ``p`` ⇒ systematic bias for this observable. ``None``
    when there are no usable (non-zero, finite) residuals.
    """
    signs = _signs(residuals)
    n = len(signs)
    n_pos = sum(1 for s in signs if s > 0)
    n_neg = n - n_pos
    if n == 0:
        return {"n": 0, "n_pos": 0, "n_neg": 0, "p": None, "low_power": True}
    # Exact two-sided binomial tail at p=0.5. scipy.stats.binomtest is the reference; compute it
    # directly so this module carries no scipy dependency of its own (and matches exactly at the
    # boundary, where binom_test's doubling convention differs across versions).
    k = min(n_pos, n_neg)
    tail = 2.0 * sum(math.comb(n, i) for i in range(0, k + 1)) / (2.0**n)
    p = min(1.0, tail)
    return {"n": n, "n_pos": n_pos, "n_neg": n_neg, "p": p, "low_power": n < MIN_POWER_N}


def _runs_pmf(n1: int, n2: int) -> dict[int, float]:
    """Exact distribution of the number of runs R for ``n1`` +'s and ``n2`` -'s.

    Standard combinatorics: among the ``C(n1+n2, n1)`` equally-likely arrangements, the count
    with exactly ``r`` runs is ``2·C(n1-1,k-1)·C(n2-1,k-1)`` for ``r = 2k`` and
    ``C(n1-1,k)·C(n2-1,k-1) + C(n1-1,k-1)·C(n2-1,k)`` for ``r = 2k+1``. Exact (not the normal
    approximation) because these series have only a handful of points, where the approximation
    is poor.
    """
    total = math.comb(n1 + n2, n1)
    pmf: dict[int, float] = {}
    for r in range(2, n1 + n2 + 1):
        if r % 2 == 0:
            k = r // 2
            ways = 2 * math.comb(n1 - 1, k - 1) * math.comb(n2 - 1, k - 1)
        else:
            k = (r - 1) // 2
            ways = math.comb(n1 - 1, k) * math.comb(n2 - 1, k - 1) + math.comb(
                n1 - 1, k - 1
            ) * math.comb(n2 - 1, k)
        if ways:
            pmf[r] = ways / total
    return pmf


def _count_runs(signs: list[int]) -> int:
    """Number of maximal same-sign stretches in ``signs``."""
    if not signs:
        return 0
    runs = 1
    for a, b in zip(signs, signs[1:]):
        if a != b:
            runs += 1
    return runs


def runs_test(residuals: list[float]) -> dict[str, Any]:
    """Wald–Wolfowitz runs test on the time-ordered residual signs (EXACT small-sample p).

    ``residuals`` must already be ordered by time. Returns the observed run count ``R``, its
    expected value under randomness, an exact two-sided ``p`` (probability of a run count at
    least as extreme in either direction), and ``direction``: ``"few_runs"`` (R below expected —
    serial correlation / a systematic time trend) or ``"many_runs"`` (R above — over-alternation),
    or ``None`` when R equals expectation. ``p`` is ``None`` when either sign is absent (a runs
    test is undefined with only one sign — that is the sign test's job).
    """
    signs = _signs(residuals)
    n = len(signs)
    n_pos = sum(1 for s in signs if s > 0)
    n_neg = n - n_pos
    base = {"n": n, "n_pos": n_pos, "n_neg": n_neg, "runs": _count_runs(signs)}
    if n_pos == 0 or n_neg == 0:
        # All one sign: no runs structure to test (the bias is the sign test's finding).
        return {**base, "expected_runs": None, "p": None, "direction": None, "low_power": True}
    pmf = _runs_pmf(n_pos, n_neg)
    expected = 2.0 * n_pos * n_neg / n + 1.0
    r_obs = base["runs"]
    p_obs = pmf.get(r_obs, 0.0)
    # Two-sided by the total-probability convention: sum the mass of every run count no more
    # likely than the observed one. Robust for an asymmetric discrete distribution where a
    # doubled one-sided tail can exceed 1.
    p = min(1.0, sum(pr for pr in pmf.values() if pr <= p_obs + 1e-12))
    direction = None
    if r_obs < expected:
        direction = "few_runs"
    elif r_obs > expected:
        direction = "many_runs"
    return {
        **base,
        "expected_runs": round(expected, 3),
        "p": p,
        "direction": direction,
        "low_power": n < MIN_POWER_N,
    }


def fisher_combine(pvalues: list[float]) -> float | None:
    """Combine independent p-values by Fisher's method (``-2·Σ ln p`` ~ χ²_{2m}).

    Used to roll a per-(condition) runs-test p-value up to one number per observable when an
    observable is measured under several conditions (each an independent residual series). A
    single series passes straight through. ``None`` when the list is empty.
    """
    ps = [p for p in pvalues if p is not None]
    if not ps:
        return None
    if len(ps) == 1:
        return ps[0]
    # Clamp away from 0 so a single exact-0 discrete p-value doesn't send the statistic to inf.
    stat = -2.0 * sum(math.log(max(p, 1e-300)) for p in ps)
    dof = 2 * len(ps)
    try:
        from scipy.stats import chi2 as _chi2

        return float(_chi2.sf(stat, dof))
    except Exception:  # pragma: no cover - scipy always present via pypesto
        return None


# -- replicate lack-of-fit -----------------------------------------------------------------


def student_t_two_sided_p(t: float, df: float) -> float | None:
    """``P(|T| >= |t|)`` for ``T ~ Student-t(df)``.

    Student-t and not a normal tail because the sd in the denominator is estimated from the
    replicates themselves, on the noise sigma's pooled ``df = sum(n_i - 1)``: 6 on the smallest
    family the replicate arm tests (3 points of 3 replicates), where a normal tail would over-state
    the significance of every point; on a large family the two agree. ``None`` on a
    non-finite ``t`` or ``df < 1`` (fail open — a diagnostic must never manufacture a verdict from
    bad input).

    scipy, not a hand-rolled incomplete beta: the binomial and runs tails above are hand-rolled to
    keep this module scipy-free, but that premise does not hold — importing :mod:`petab_mcp` at all
    already puts ``scipy.stats`` in ``sys.modules`` (via pypesto), so no import was ever saved on any
    path and 85 lines of Numerical-Recipes continued fraction bought nothing. (NOT because
    :func:`fisher_combine` imports ``scipy.stats.chi2`` on the same per-fit path: it returns before
    that import whenever it has a single p-value, the steady-state case, and on a collapsed sigma
    the sign/runs tests do not run at all.) Degrades to ``None`` if the import ever fails, like
    fisher_combine.
    """
    try:
        t = float(t)
        df = float(df)
    except (TypeError, ValueError):
        return None
    if not (math.isfinite(t) and math.isfinite(df)) or df < 1:
        return None
    if t == 0.0:
        return 1.0
    try:
        from scipy.stats import t as _t

        return float(2.0 * _t.sf(abs(t), df))
    except Exception:  # pragma: no cover - scipy always present via pypesto
        return None


def holm_adjusted(pvalues: list[float]) -> list[float]:
    """Holm–Bonferroni step-down adjusted p-values, in the INPUT order.

    ``p_adj`` for the k-th smallest of ``m`` p-values is ``(m-k)·p``, then made monotone
    non-decreasing along the sorted order and capped at 1. Holm rather than plain Bonferroni
    because the question is "is ANY replicated point off?" across a family of up to ~150 points,
    where Bonferroni on the minimum alone throws away the rest of the family; and rather than a
    false-discovery rule because the consumer is a GATE — one flagged point blocks a plateau stop,
    so the error that matters is the family-wise one.
    """
    ps = [float(p) for p in pvalues]
    m = len(ps)
    out = [1.0] * m
    running = 0.0
    for rank, i in enumerate(sorted(range(m), key=lambda j: ps[j])):
        running = max(running, min(1.0, (m - rank) * ps[i]))
        out[i] = running
    return out


def replicate_pure_error(values: list[float]) -> dict[str, Any] | None:
    """Pure error of ONE design point's replicates: ``{n, mean, sd}``, ``sd`` the sample sd
    (``ddof = 1``). ``None`` below two finite values — a point measured once has no scatter of its
    own.
    """
    vals: list[float] = []
    for v in values:
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if math.isfinite(f):
            vals.append(f)
    n = len(vals)
    if n < 2:
        return None
    mean = math.fsum(vals) / n
    sd = math.sqrt(math.fsum((v - mean) ** 2 for v in vals) / (n - 1))
    return {"n": n, "mean": mean, "sd": sd}


def pooled_replicate_scatter(points: list[dict[str, Any]]) -> tuple[float, int] | None:
    """``(s, df)``: the replicate scatter pooled over ``points``, each ``{n, s}`` (one design point's
    replicate count and sample sd)::

        s^2 = sum (n_i - 1) * s_i^2 / sum (n_i - 1)        df = sum (n_i - 1)

    THE one pooling: the noise report's floor
    (:func:`petab_mcp.reconcile.sigma_replicate_noise_estimate`) and the lack-of-fit yardstick
    (:func:`replicate_lack_of_fit`) both call it on the points of one noise sigma, so the two cannot
    disagree. A point with ``n < 2`` or a non-finite or negative ``s`` adds nothing; ``None`` when
    nothing is left to pool.
    """
    df = 0
    ssq: list[float] = []
    for p in points or []:
        try:
            n, s = int(p["n"]), float(p["s"])
        except (KeyError, TypeError, ValueError):
            continue
        if n < 2 or not math.isfinite(s) or s < 0:
            continue
        df += n - 1
        ssq.append((n - 1) * s * s)
    return (math.sqrt(math.fsum(ssq) / df), df) if df else None


def lof_power(
    n_points: int, n_structural: int | None, min_lof_df: int | None = None
) -> tuple[int | None, str]:
    """``(lof_df, band)``: how much a replicate lack-of-fit PASS over ``n_points`` replicated design
    points is worth against ``n_structural`` estimated structural parameters.

    THE one owner of the lack-of-fit-df rule — one function, one threshold comparison, one place the
    default ``REPLICATE_LOF_MIN_LOF_DF`` is read. The rule lived in two code paths before (the
    per-family statistic and the merge) computing it from two different point counts, so the arm
    and the payload the agent was shown could disagree about the same pass; and the merge's copy
    read the module constant, discarding a caller's ``min_lof_df=`` override.

    ``lof_df = n_points - n_structural`` is what a pass is worth: a fit with nearly as many free
    structural parameters as replicated points can place its predictions on the replicate means in
    all but a few directions WHATEVER its form, so small |z| there is mostly flexibility. The band:

    * ``"unknown"`` — no structural count (``n_structural`` None), so the df cannot be computed. NOT
      the same as powered: the caller must claim no power it cannot show (``data_access``' prose says
      the pass could not be weighed, and credits no coverage for it).
    * ``"vacuous"`` — ``lof_df <= 0``: at saturation nothing was tested at all, so the arm reports NO
      EVIDENCE (``available: False``) rather than a pass.
    * ``"weak"`` — ``0 < lof_df < min_lof_df``: a PASS is real but thin evidence, reported ``weak``.
      A FLAG stands unchanged at this band and at any df — a point beyond what replicate scatter
      allows, after the fit had its way, is real evidence however few directions remain.
      Do NOT make these UNAVAILABLE instead: that converts legitimate flags on small
      families into no evidence, a worse error than the one it fixed.
    * ``"powered"`` — a pass worth reporting as one (and the only band that credits coverage).

    ``n_points`` is the POOLED count over every noise family that could be tested, and
    ``n_structural`` a WHOLE-MODEL count subtracted ONCE: the design points ADD across families
    (each has its own set of them) while every estimated parameter is free to move every family's
    predictions, and no per-family split of them exists. Applying this per family instead both
    understated the df (two families of 15 points at 14 structural reported 2, not 30 - 14 = 16) and
    called one problem's three 4-point families vacuous one by one, telling the agent the model "has
    as many estimated structural parameters as there are replicated design points" of 12 points
    against 10 — a sentence that was simply false.
    """
    min_lof_df = REPLICATE_LOF_MIN_LOF_DF if min_lof_df is None else min_lof_df
    if n_structural is None:
        return None, "unknown"
    lof_df = n_points - n_structural
    if lof_df <= 0:
        return lof_df, "vacuous"
    return lof_df, "weak" if lof_df < min_lof_df else "powered"


def _sig3(value: float | None) -> float | None:
    """``value`` to 3 significant digits (the reported precision; the FLAG is always decided on the
    unrounded number)."""
    return None if value is None else float(f"{float(value):.3g}")


def replicate_lack_of_fit(
    points: list[dict[str, Any]],
    *,
    alpha: float = REPLICATE_LOF_ALPHA,
    min_df: int = REPLICATE_LOF_MIN_DF,
    min_points: int = REPLICATE_LOF_MIN_POINTS,
    max_reported: int = REPLICATE_LOF_MAX_REPORTED_POINTS,
) -> dict[str, Any]:
    """Per-design-point lack-of-fit against the replicate scatter of the point's noise sigma.

    ``points`` are the replicated design points of ONE noise parameter, each a mapping
    ``{label, n, ybar, s, prediction}`` on the coordinate where that parameter's noise is additive
    (:func:`petab_mcp.reconcile.replicate_pure_error_points` builds them). Their scatter is pooled
    into ``s_g`` on ``df_g`` (:func:`pooled_replicate_scatter`, the noise report's
    ``data_noise_estimate`` for that parameter), and per point::

        se_i = s_g / sqrt(n_i)        z_i = (ybar_i - f_i) / se_i

    and ``p_i`` is the two-sided Student-t tail on ``df_g``, Holm-adjusted across the family. A
    point whose adjusted p falls below ``alpha`` is FLAGGED: the model cannot reach that point no
    matter how its parameters move, which is missing STRUCTURE, not noise.

    Why this arm exists and why it is scoped this way is in ``constants.REPLICATE_LOF_ALPHA``. The
    guards, each with the measured case that forces it:

    * ``min_df`` — a point is tested from n_i - 1 >= 2; thinner points still count towards
      ``s_g`` (see ``constants.REPLICATE_LOF_MIN_DF``).
    * ``min_points`` — fewer than 3 usable points and there is no family to correct over.
    * anything non-finite or missing is DROPPED, and a family with nothing usable (or no scatter
      to pool) reports NO EVIDENCE (``available: False``) — never a pass.

    NO lack-of-fit-df judgement happens here, on purpose. How much a pass is worth is
    ``pooled points - n_structural`` (:func:`lof_power`), and neither operand is knowable from ONE
    family: the replicated points add across noise families while the structural count is a
    whole-model property of the parameter table. So the arm reports what it tested and
    ``data_access._merge_lof_verdicts`` applies :func:`lof_power` ONCE, to the pooled total — which
    is also where ``vacuous`` (df <= 0, nothing tested at all) and ``weak`` (a thin pass) are
    decided. Judging it here as well was two owners of one rule computing it from two different
    point counts: they disagreed on 22 of 48 measured (families x points x parameters) cells, and
    the per-family version called one problem's three 4-point families vacuous one at a time — NO
    EVIDENCE, with a summary line claiming as many parameters as replicated points, for 12 pooled
    points against 10 parameters (pooled df 2, a thin pass but a real test).

    The verdict is a fixed shape whatever happens — EVERY key on EVERY path, including the skipped
    ones, because the assembler reads them off each arm and a missing key would be a KeyError:
    ``{available, status, n_points, alpha, flagged, tested, max_abs_z, min_p_holm, points}``. Keep
    this list complete when adding a key: an assembler that REBUILDS the verdict from a list like
    it and silently dropped ``lof_df``/``weak``, which left the summary claiming "this arm IS
    powered" on a df = 1 pass, in three sessions on one problem.

    Fixed shape HERE, not on the wire: what rides on a fit result is a projection of the MERGED
    verdict (``data_access._lof_payload``), and that one omits every key with nothing to say — this
    verdict's shape is the assembler's contract, not the agent's payload.

    ``points`` are the ``max_reported`` worst FLAGGED points —
    only flagged ones, because the advice built from them tells the agent that is where the form is
    wrong, and a point the model already reproduces is not. ``tested`` names every point that
    entered the test, which is how the assembler knows which observables a passing arm may be
    credited for (:func:`petab_mcp.data_access._rollup_structure`); it is stripped before the
    verdict rides on a fit result, where only the bounded ``points`` list may go.
    """
    pooled = pooled_replicate_scatter(points)
    s_g, df_g = pooled if pooled and pooled[0] > 0 else (None, 0)
    usable: list[dict[str, Any]] = []
    # Counted so that "nothing to test" can name WHICH kind of nothing it was — each unavailable
    # status has to mean exactly one thing (see replicate_lof_unavailable).
    n_replicated = 0  # design points measured more than once at all
    n_thin = 0  # ...of those, the ones below min_df (e.g. a problem whose every n_i is 2)
    for p in points or []:
        try:
            n = int(p["n"])
        except (KeyError, TypeError, ValueError):
            continue
        if n < 2:
            continue
        n_replicated += 1
        if n - 1 < min_df:
            n_thin += 1
            continue
        try:
            ybar = float(p["ybar"])
            s = float(p["s"])
            f = float(p["prediction"])
        except (KeyError, TypeError, ValueError):
            continue
        finite = math.isfinite(ybar) and math.isfinite(f) and math.isfinite(s)
        if s_g is None or not finite or s < 0:
            continue
        z = (ybar - f) / (s_g / math.sqrt(n))
        pv = student_t_two_sided_p(z, df_g)
        if pv is None:
            continue
        usable.append({"label": p.get("label"), "n": n, "z": z, "p": pv})

    if not usable:
        if n_replicated == 0:
            return replicate_lof_unavailable("no_replicated_points", alpha)
        if n_thin == n_replicated:
            return replicate_lof_unavailable("replicates_too_thin", alpha)
        # Replicated, thick enough, and still not testable: no prediction, or a non-finite input.
        return replicate_lof_unavailable("unavailable", alpha)
    if len(usable) < min_points:
        return replicate_lof_unavailable("too_few_usable_points", alpha, n_points=len(usable))

    for point, p_holm in zip(usable, holm_adjusted([u["p"] for u in usable])):
        point["p_holm"] = p_holm
    flagged = [u["label"] for u in usable if u["p_holm"] < alpha]
    worst = sorted(usable, key=lambda u: abs(u["z"]), reverse=True)
    # Reported points are the worst FLAGGED ones. Taking the worst of the whole family instead made
    # the advice name PASSING points as "where the form is wrong" whenever fewer than max_reported
    # points were flagged — with one flag among four, three points at z = 0 were listed as
    # mechanisms to add.
    named = [u for u in worst if u["p_holm"] < alpha][:max_reported]
    return {
        "available": True,
        "status": "flagged" if flagged else "pass",
        # The pooled count the df is computed from is the SUM of these over the arms that ran, so
        # this is the only df input the arm owns (see lof_power).
        "n_points": len(usable),
        "alpha": alpha,
        "flagged": sorted(str(label) for label in flagged),
        "tested": [u["label"] for u in usable],
        # z to 3 decimals (not 3 significant digits): a worst point at 17.976 and one at 18.0 read
        # the same to 3 s.f., and this number is the one a regression pins.
        "max_abs_z": round(abs(worst[0]["z"]), 3),
        "min_p_holm": _sig3(min(u["p_holm"] for u in usable)),
        "points": [
            {"label": u["label"], "n": u["n"], "z": round(u["z"], 3), "p_holm": _sig3(u["p_holm"])}
            for u in named
        ],
    }


def replicate_lof_unavailable(
    status: str = "unavailable", alpha: float = REPLICATE_LOF_ALPHA, n_points: int = 0
) -> dict[str, Any]:
    """The degraded verdict: the same keys, and honest that the arm produced NO EVIDENCE — which a
    consumer must be able to tell apart from a pass (an empty ``flagged`` alone reads as adequacy).
    An arm that could not run tested NOTHING, so ``tested`` is empty and no observable may be
    credited with a pass on its account.

    ``vacuous`` is one of these and is raised by the ASSEMBLER, from the pooled df (:func:`lof_power`),
    not from inside the statistic — see :func:`replicate_lack_of_fit`.
    """
    # Every key of the available path, so the verdict has ONE shape whatever happened: the assembler
    # reads them off each arm and a missing key would be a KeyError.
    return {
        "available": False,
        "status": status,
        "n_points": n_points,
        "alpha": alpha,
        "flagged": [],
        "tested": [],
        "max_abs_z": None,
        "min_p_holm": None,
        "points": [],
    }


# -- variance heterogeneity across the series one noise sigma covers ----------------------------

#: Fewest finite weighted residuals a series needs to enter :func:`sigma_heterogeneity`.
HETEROGENEITY_MIN_SERIES_N = 2


def sigma_heterogeneity(series: dict[str, list[float]]) -> dict[str, Any] | None:
    """Whether one noise sigma fits every series it covers: Levene's test about ZERO, i.e. a one-way
    ANOVA of ``|r|`` across the series, ``r`` the weighted residuals.

    ``series`` maps a label to one series' weighted residuals. Under a correct model with one shared
    sigma every ``r`` has the same spread about 0 whichever series it is in; a sigma inflated to
    absorb one badly fitted series leaves that series' ``|r|`` larger than the rest. Returns
    ``{p, n_series, low_power, worst}``, ``worst`` every tested series as ``{series, n, rms}``
    (``rms`` of its ``r``, against about 1 over the whole sigma at its ML value), largest first;
    ``None`` below two series of :data:`HETEROGENEITY_MIN_SERIES_N` points. ``low_power`` when a
    series has fewer than :data:`MIN_POWER_N` points: a 1.5-sigma miss in it is found 0.11-0.30 of
    the time at alpha 0.05 and about half at 8, whether 2 or 6 series share the sigma.

    Centred on zero, not on each series' mean: the ML sigma is the rms about the prediction, so a
    series missed by a constant inflates it as much as one missed in shape, and only this centring
    sees that (power 0.50 against 0.05 for the mean-centred tests, one series of six biased by 1.5
    sigma). Not Bartlett: it reads a series its own scale and offset fit exactly as p = 0, and under
    t3 noise it false-alarms 45% at alpha 0.05 where this test holds 4.4%. The p ranks and gates
    nothing: bachmann's passing fit still shows p = 0.006 on pEpoR (#195).
    """
    import warnings

    from scipy.stats import f_oneway

    groups: dict[str, list[float]] = {}
    for label, values in series.items():
        finite = []
        for v in values:
            try:
                f = float(v)
            except (TypeError, ValueError):
                continue
            if math.isfinite(f):
                finite.append(f)
        if len(finite) >= HETEROGENEITY_MIN_SERIES_N:
            groups[label] = finite
    if len(groups) < 2:
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # every |r| equal: scipy warns and returns nan
        p = float(f_oneway(*([abs(f) for f in v] for v in groups.values())).pvalue)
    rms = {label: math.sqrt(math.fsum(f * f for f in v) / len(v)) for label, v in groups.items()}
    return {
        "p": _sig3(p) if math.isfinite(p) else None,
        "n_series": len(groups),
        "low_power": min(map(len, groups.values())) < MIN_POWER_N,
        "worst": [
            {"series": label, "n": len(groups[label]), "rms": round(rms[label], 3)}
            for label in sorted(groups, key=rms.__getitem__, reverse=True)
        ],
    }
