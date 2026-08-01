"""Central tuning defaults for the estimation / model-building workflow.

Gathers the cross-cutting numeric "knobs" that were otherwise scattered as module-local
literals — and, for the multistart count, duplicated between :mod:`session` and
:mod:`server`. Keeping them here means a maintainer adjusts the estimation defaults in one
place.

Deliberately NOT centralised: single-module implementation details whose meaning is local —
the SBML lint categories (:mod:`validation`), the AMICI reserved-name hints
(:mod:`operations`), the SQL table schema (:mod:`data_access`), the E-utilities URLs
(:mod:`references`), and the fitter's env-var-backed runtime knobs
(``FIT_PER_START_TIMEOUT_S`` / ``FIT_N_WORKERS`` in :mod:`fitting`). Those stay in the module
that owns them.
"""

from __future__ import annotations

# -- fit / multistart ------------------------------------------------------
# Default pyPESTO multistart count per commit. The session default and the ``create_server`` /
# CLI default read this one value so they cannot drift; overridable per commit via ``n_starts``.
DEFAULT_AUTO_FIT_STARTS = 5

# -- early-stop chain (effect-size gate + non-improving-streak cap + best-BIC checkpoint) ---
# The model-building loop otherwise wanders past the noise floor until the LangGraph recursion
# limit or a provider error fires — a whole-model op is ~4 super-steps, so by the time a 500-step
# limit trips, 100+ ops (observed streaks of 40-46 non-improving versions) are already burned. The
# recursion limit is the wrong lever; these three coupled knobs make the loop stop when the score
# stops moving, and score the BEST model rather than whatever happened to be committed last.
#
# (a) Minimum BIC improvement for a fit to count as a NEW BEST — the gate on ``is_new_best``
# (see :func:`fitting.compute_improvement`). ``is_new_best`` tracks BIC (the same score the judge
# and the checkpoint use), so the flag, the non-improving-streak reset, and the best-model
# checkpoint all move on one coherent notion of "better". Without a threshold the flag fired on
# ANY score twitch (down to ~1e-6, or an nll drop bought by an extra parameter that actually
# RAISES BIC), so noise-floor wander looked like progress and kept resetting the streak. ΔBIC ≥ 2
# is the conventional "positive evidence" bar. The comparison is against the current best-BIC
# CHECKPOINT (not the raw running minimum), so a steady sub-threshold BIC creep still registers a
# new best once it CUMULATIVELY clears the bar, rather than starving the streak while it improves.
NEW_BEST_MIN_DELTA_BIC = 2.0

# (b) Hard cap on consecutive fitted commits with NO new best (post the gate above): once the
# streak reaches this, the run stops. This is the load-bearing early-stop lever — the worst runs
# never reach statistical adequacy, so an adequacy-gated stop alone would let them run to the
# recursion limit. Sized well below the ~40-46-op streaks observed in the trace review.
NON_IMPROVING_STREAK_CAP = 15

# (b, softer) Plateau early-stop: once the best model is statistically adequate AND the best BIC
# has improved by less than ``PLATEAU_MIN_BIC_GAIN`` over the last ``PLATEAU_WINDOW`` fitted ops,
# stop — there is nothing left to gain. Complements the hard cap (which catches never-adequate
# runs); this one lets an already-good run finish promptly instead of polishing the noise floor.
PLATEAU_MIN_BIC_GAIN = 1.0
PLATEAU_WINDOW = 5

# The chi2 goodness-of-fit p-value at/above which the best model is considered statistically
# adequate for the plateau gate (its residuals are consistent with the known measurement noise).
ADEQUATE_GOF_PVALUE = 0.05

# The OTHER tail of the same chi2 GoF test. A p-value close to 1 means chi2 is far BELOW its
# degrees of freedom — the residuals are much smaller than the KNOWN noise, i.e. the model is
# fitting the noise rather than the signal. Testing only `p >= 0.05` is one-sided and scores such
# an overfit as "adequate" (observed: rahman runs at chi2/n = 0.016 and 0.051 with p = 0.9999999).
# Only meaningful for a FIXED sigma: when sigma is ESTIMATED the MLE absorbs residual shrinkage
# into sigma itself, pinning chi2/n at ~1 from below (empirical floor 0.98 over 40 runs), so this
# tail is unreachable there and the estimated-sigma branch judges the sigma's plausibility instead.
OVERFIT_GOF_PVALUE = 0.995

# -- default parameter-table bounds ----------------------------------------
# Bounds for an auto-added model parameter missing from the table (linear values; always widened
# to bracket the nominal). Kept fairly TIGHT (five decades) so a multistart over an auto-created
# parameter can actually converge — a very wide range (e.g. 1e-5..1e5) makes a several-parameter
# fit multimodal and prone to sticking in a poor local optimum. Wide enough for the kinetic scales
# these models use; only widened further when a nominal value falls outside.
DEFAULT_PARAMETER_LOWER_BOUND = 1e-3
DEFAULT_PARAMETER_UPPER_BOUND = 1e2

# -- cooperativity / Hill exponent bounds ----------------------------------
# Tight LINEAR bounds for a cooperativity / Hill exponent (typical range ~1-8), always widened to
# bracket the nominal. Estimating an exponent on the wide default log10 range lets it run to a
# near-step-function (a repressilator run fitted a Hill coefficient of 15), which makes the ODE
# stiff and the fit plateau; tight linear bounds keep it biophysically sane.
EXPONENT_LOWER_BOUND = 1.0
EXPONENT_UPPER_BOUND = 8.0

# -- at-bound reporting -----------------------------------------------------
# A fitted parameter within this fraction of its lower/upper bound (measured on its own
# estimation scale, so a log10 parameter is judged by decades, not raw linear distance) is
# flagged in the standard fit report. A parameter repeatedly re-pinning to its bound after a
# bounds widening usually means the OPTIMUM is unbounded (e.g. a Hill exponent wanting to run
# to a near-step-function to fake a mechanism the model is missing) — a modeling problem, not
# a search-space one — so this is meant to catch that early rather than relying on the agent
# to eyeball `get_parameters()` against the fitted value after every commit.
PARAMETER_AT_BOUND_RELATIVE_TOL = 0.01

# -- noise-sigma reporting ---------------------------------------------------
# An ESTIMATED noise-sigma parameter above this fraction of its observable's own raw
# measurement std (across every time point / condition) is flagged in the standard fit report.
# A sigma free to float lets the optimizer make almost ANY mechanism look "statistically
# adequate" (chi2_gof_pvalue >= 0.05) by simply inflating the assumed noise until residuals fit
# inside it — a benchmark sweep did exactly this (sigma fit to 20-60x its nominal scale) while
# the underlying mechanism was missing whole reaction pathways. A sigma that has to explain a
# large fraction of the data's own variability isn't measurement noise anymore; it's covering
# for the model.
#
# CALIBRATION (#118): both thresholds are measured in the coordinate where the noise is ADDITIVE
# (see reconcile.sigma_noise_coordinate) — the only scale on which this ratio means anything. They
# are deliberately WIDE, because the benchmark's own ground-truth sigmas span ~0.0004-0.59 of their
# data std: a tight band rejects the true model. The band's job is to catch the pathological tails
# (a sigma driven to ~0 to overfit the noise, or inflated many-fold to absorb misfit), NOT to
# adjudicate a merely unusual-looking noise level.
NOISE_SIGMA_MAX_FRACTION_OF_DATA_STD = 0.75

# The other tail: a fitted noise sigma BELOW this fraction of its observable's own data std is
# implausibly small — the model is explaining ~all of the data's variability, the fingerprint of
# overfitting the noise. Shared by the plateau adequacy gate (session) and the judge's success gate
# so both agree on "collapsed".
NOISE_SIGMA_COLLAPSE_FRACTION = 0.01

# -- replicate (pure-error) inflation gate -----------------------------------
# Where measurements REPEAT at the same (observable, condition, time), their scatter is pure noise —
# an UNBIASED, model-free floor (reconcile.sigma_replicate_noise_estimate). Unlike the successive-
# difference estimate (biased, and it collides with legitimate references — see BENCHMARK_NOTES),
# this floor is trustworthy, so where it exists the gate flags a fitted sigma exceeding it by more
# than this factor: the noise is absorbing model misfit, not measuring it. Calibrated on the pure-
# error ratio of each TRUE model: good references sit at ~0.8-1.0 (blasi 0.98, armistead 0.82-0.95),
# while blasi's noise-gaming runs sit at 1.8-2.4 — so 1.5 separates them with margin on both sides.
# (schwen ~7.8 / oliveira flag their OWN reference-quality problems, not the gate's — see BENCHMARK_NOTES.)
NOISE_SIGMA_INFLATION_RATIO = 1.5
# Minimum pooled replicate degrees of freedom to TRUST the pure-error floor for the ratio gate above.
# Below this the estimate is too noisy; the gate falls back to the (conservative) data-std cap only.
NOISE_SIGMA_REPLICATE_MIN_DOF = 6

# -- replicate lack-of-fit (the residual arm of the structure verdict) -------
# The SAME repeated design points, used the other way round: instead of pooling their scatter into
# one floor to judge sigma against, ask per point whether the PREDICTION sits further from the data
# than that point's OWN scatter allows —
#     z_i = (ybar_i - f_i) / (s_i / sqrt(n_i))   on the noise-additive coordinate.
# This is the axis blasi's early finish hid behind. The chi2 GoF p-value cannot see it (with one
# shared sigma at its ML value chi2 == n identically, so the p-value is a function of (n, k) alone:
# 252 for the true model AND for a 6-parameter run 194 BIC worse), the sign/runs tests cannot see it
# (a single steady-state condition makes every per-observable series length 1 — n=1, p=1.0,
# undefined runs test), and the inflation ratio above barely separates them (1.4759 vs 1.5). z_i
# separates by 5.3x on max|z| (truth 3.393 vs run 17.976) and 41x on sum z_i^2, and because it uses
# only s_i it is EXACTLY invariant to the ML vs RSS/(n-k) sigma convention — which the ratio is not.
#
# Per-point two-sided Student-t p on df_i = n_i - 1, HOLM-adjusted across the replicated points of
# ONE noise parameter, flagged below this alpha. 0.01, NOT 0.05: blasi's TRUE model lands at Holm
# p = 0.0519, so a 5% rule is a coin flip that would deadlock the example about half the time (its
# MC-calibrated 5% critical value for max|z| is 3.97 against the truth's 3.393 — 15% headroom). At
# 0.01 the truth keeps 2.2x headroom (MC crit 5.26) while the run is flagged at 2.55e-11, and that
# margin also absorbs the naive Holm null being ~2x anti-conservative versus the MC one — so no
# Monte-Carlo calibration is needed at runtime.
#
# NOT a variance-ratio F test: the shared-sigma fit weights by diag(n_i) while sum z_i^2 weights by
# diag(n_i / s_i^2), so the projection is OBLIQUE and removes almost no df (exact E[T|H0] = 13.60 at
# m = 15 with rank J = 8, not m - p = 7). Dividing by m - p inflates F by up to 2.14x and
# MANUFACTURES a rejection of blasi's true model at p = 0.0016. Per-point Holm needs no df bookkeeping
# at all, which is why it is the rule here.
#
# READ THE MARGINS ABOVE AS A FLOOR, NOT AS CALIBRATION. Every one of them assumes the n_i replicates
# of a point are independent draws, and on blasi they are NOT: rows 1-12 and 13-18 of each design
# point are two experimental blocks (the three n_i = 12 observables stop exactly at row 12; Welch per
# observable, Fisher chi2 = 54.67 on 24 df, p = 3.5e-4; 12 of 14 shifts share a sign, with the
# abundant 0ac moving down while every minor state moves up — one compositional degree of freedom,
# not noise). Within-block variance is 0.044704 against a naive within-point 0.0665166, so s_i/sqrt(n_i)
# understates se(ybar_i) by a median 2.19x and by 10.3x at observable_0ac; block-aware, the truth's
# excess-variance ratio falls 3.330 -> 0.339 and the failing run's 98.5 -> 13.4. That bias runs the SAFE
# way here — it inflates every z_i, so the truth looks worse than it is and still clears 0.01 — but it
# means these p-values are not exact, and a future tightening toward 0.05 has less headroom than the
# numbers above suggest. Holm survives it for a different reason: it controls FWER under ARBITRARY
# dependence, which matters because the z_i are strongly correlated (blasi's states sum to 1; top
# eigenvalue 6.2 of 15) — a Fisher combine or a sum-of-squares aggregate would NOT be valid here.
# Nothing corrects for the blocks on purpose: block membership exists only in measurement ROW ORDER,
# which PEtab gives no meaning, so no shipped code may read it. Subtracting within-point block means
# would move the pooled floor 0.257908 -> 0.248610 and put the failing run at 1.5311, tripping the
# inflation ratio above — worth knowing, not worth depending on row order for.
REPLICATE_LOF_ALPHA = 0.01

# A design point's own scatter is only a usable yardstick from this many degrees of freedom
# (df_i = n_i - 1) up. 2 EXCLUDES schwen, whose 126 replicated points ALL have n_i = 2: those are
# technical replicates of a bound-railed insulin-ELISA calibration whose reference sigma is 7.8x
# their pure error, so a pure-error test there rejects schwen's OWN true model (pooled lack-of-fit
# F = 156.2 on (98, 126), p = 1e-100). One df also cannot tell a real deviation from a lucky pair,
# and Levene/Bartlett are degenerate there (see BENCHMARK_NOTES).
REPLICATE_LOF_MIN_DF = 2

# Below this many usable replicated points the Holm family is too small to conclude anything, so the
# arm reports NO EVIDENCE rather than a pass. 3 is the smallest count at which a single flagged point
# survives a Holm correction at alpha 0.01 without needing an absurd |z|.
REPLICATE_LOF_MIN_POINTS = 3

# At most this many of the worst FLAGGED points (by |z|) ride on a fit result — flagged only, since
# the advice built from them tells the agent that is where the form is wrong, and a point the model
# already reproduces is not one of those. The verdict is re-sent in the agent's history on EVERY
# subsequent turn (PR #23 cut ~40% of the per-call prompt by trimming exactly this kind of payload),
# so the arm emits a one-line verdict plus a handful of NAMED points — never the full table (blasi
# would be 15 rows per fit, forever).
REPLICATE_LOF_MAX_REPORTED_POINTS = 5

# -- reference-sigma comparison (the judge's preferred noise check, #118) ----
# When reference.json records the TRUE model's own fitted noise sigmas, the judge compares a
# candidate's sigma to those instead of to a fraction of the marginal data std — the dimensionally
# sound comparison, and the one that doesn't conflate variance the model EXPLAINS with variance it
# can't (which is why fiedler/oliveira looked "collapsed" at 0.0004-0.0026 of the marginal std).
# A candidate sigma is implausible when it falls outside
# [sigma_ref / FACTOR, sigma_ref * FACTOR]. Deliberately generous: a good model that is not
# literally the truth still has somewhat different residuals, and a sharp test (e.g. the exact
# chi2 point-null n*s^2/s_ref^2 ~ chi2_{n-k}) rejects legitimate near-misses once n is large.
NOISE_SIGMA_REFERENCE_FACTOR = 3.0

# -- repeated-widening backstop ---------------------------------------------
# A parameter pinned at a bound with a flat / collinear FIM direction is UNIDENTIFIABLE, not
# under-explored: widening its box just lets the optimum slide to the new edge and re-pin for a
# negligible fit gain, chasing the search space toward infinity (armistead widened one rate 4x
# this way before finally fixing it). ``change_parameters`` REJECTS a THIRD such widening
# of a still-pinned parameter when the previous widening bought less than this much BIC. The
# threshold is in BIC because a bounds-only re-fit keeps k and n fixed, so an NLL improvement
# converts as delta_bic = 2 * delta_nll; a BIC gain below ~2 is "not worth more than a bare
# mention". The skill guidance is stricter still (widen at most once) — this is the hard backstop.
WIDENING_NEGLIGIBLE_BIC_GAIN = 2.0

# -- structural-inadequacy nudge (issue #82) --------------------------------
# When the misfit is STRUCTURAL — the model's structure itself cannot fit the data, not the
# optimizer or the bounds — the fix is to change the model's FORM (add a latent/unobserved state,
# add a feedback/regulation reaction, raise the interaction order), NOT to re-parameterize or widen
# bounds within the same structure. The failure mode this addresses (seen across the crauste
# deep-dive) is an agent that correctly DIAGNOSES structural inadequacy and then re-fits the same
# structure anyway. The
# early-stop plateau/streak signal already detects "the score has stopped moving", so rather than
# adding a separate detector the structural-change nudge is folded into the ``stop_reason`` those
# stops emit (see ``session._STRUCTURAL_PLATEAU_ADVICE`` / ``progress_report``), model-agnostically
# (categories of structural move, never a specific mechanism).

# -- reference validation --------------------------------------------------
# Network timeout (seconds) for resolving a reference against NCBI E-utilities.
REFERENCE_TIMEOUT_SECONDS = 15.0
