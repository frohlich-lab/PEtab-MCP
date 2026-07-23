"""Sign-based residual-structure tests: the sign test and the Wald–Wolfowitz runs test.

These answer "do the residuals still show systematic structure?" — the half of the
fit-adequacy question the magnitude-based scores (``reduced_chi2``, the χ² GoF p-value,
``mean_weighted_residual``) cannot answer when the observable noise sigmas are ESTIMATED.
With a free σ the optimizer inflates the noise to absorb misfit, so every weighted residual
shrinks uniformly toward zero and the magnitude scores pin near their null regardless of how
wrong the model's FORM is (issue #54). Both tests here look only at the SIGN of each residual,
so a uniform rescaling of the residuals leaves them unchanged — they detect a wrong form that
the magnitude scores are blind to, using only the run's own residuals (no reference model, so
no answer leak).

* **Sign test** — of the non-zero residuals for an observable, how many are positive? Under a
  correct model each sign is ~equally likely, so a lopsided count is a systematic over- or
  under-shoot (bias). Exact two-sided binomial p-value.

* **Wald–Wolfowitz runs test** — order a series' residuals by time and count the *runs*
  (maximal same-sign stretches). Too FEW runs means long same-sign stretches: the model drifts
  systematically across the time course (positive serial correlation — the signature of a
  missing dynamic term). Computed from the EXACT small-sample distribution of the run count,
  because the usual normal approximation is invalid at the handful of points per series these
  benchmarks have.

Everything here is pure numeric code over a list of residuals; :mod:`petab_mcp.data_access`
feeds it the per-point residuals and surfaces the results through ``get_fit_diagnosis``. The
results are ADVISORY diagnostics for the agent, deliberately not wired into any automated
gate: at these sample sizes the tests have low power, so a non-significant result is weak
evidence and is flagged as such rather than treated as a pass.
"""

from __future__ import annotations

import math
from typing import Any

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
    tail = 2.0 * sum(math.comb(n, i) for i in range(0, k + 1)) / (2.0 ** n)
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
            ways = (math.comb(n1 - 1, k) * math.comb(n2 - 1, k - 1)
                    + math.comb(n1 - 1, k - 1) * math.comb(n2 - 1, k))
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
        return {**base, "expected_runs": None, "p": None, "direction": None,
                "low_power": True}
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
    return {**base, "expected_runs": round(expected, 3), "p": p, "direction": direction,
            "low_power": n < MIN_POWER_N}


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
