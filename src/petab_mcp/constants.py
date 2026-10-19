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

# -- early-stop chain (effect-size gate + non-improving-streak cap + best-score checkpoint) ---
# The model-building loop otherwise wanders past the noise floor until the client's step limit or
# a provider error fires — a whole-model op is ~4 agent steps, so by the time a 500-step limit
# trips, 100+ ops (observed streaks of 40-46 non-improving versions) are already burned. The step
# limit is the wrong lever; these three coupled knobs make the loop stop when the score
# stops moving, and score the BEST model rather than whatever happened to be committed last.
#
# The "score" throughout is the session's complexity criterion — BIC by default, one of the others
# when configured (:mod:`criteria`). Both thresholds below are in deviance units (2·Δnll), the
# scale every named penalty shares, so they carry over unchanged when the penalty is swapped.
# They are literals rather than derived from the criterion; docs/RATIONALE.md says what that costs.
#
# (a) Minimum score improvement for a fit to count as a NEW BEST — the gate on ``is_new_best``
# (see :func:`fitting.compute_improvement`). ``is_new_best`` tracks the same score the checkpoint
# stores, so the flag, the non-improving-streak reset, and the best-model
# checkpoint all move on one coherent notion of "better". Without a threshold the flag fired on
# ANY score twitch (down to ~1e-6, or an nll drop bought by an extra parameter that actually
# RAISES the score), so noise-floor wander looked like progress and kept resetting the streak.
# Δ ≥ 2 is the conventional "positive evidence" bar. The comparison is against the current
# best-score CHECKPOINT (not the raw running minimum), so a steady sub-threshold creep still
# registers a new best once it CUMULATIVELY clears the bar, rather than starving the streak while
# it improves.
NEW_BEST_MIN_DELTA_SCORE = 2.0

# (b) Hard cap on consecutive fitted commits with NO new best (post the gate above): once the
# streak reaches this, the session stops. This is the load-bearing early-stop lever — the worst
# sessions never reach statistical adequacy, so an adequacy-gated stop alone would let them go on
# to the client's step limit. Sized well below the ~40-46-op streaks observed in the trace review.
NON_IMPROVING_STREAK_CAP = 15

# (b, softer) Plateau early-stop: once the best model is statistically adequate AND the best score
# has improved by less than ``PLATEAU_MIN_SCORE_GAIN`` over the last ``PLATEAU_WINDOW`` fitted ops,
# stop — there is nothing left to gain. Complements the hard cap (which catches never-adequate
# sessions); this one lets an already-good session finish promptly instead of polishing the noise
# floor.
PLATEAU_MIN_SCORE_GAIN = 1.0
PLATEAU_WINDOW = 5

# (c) There is deliberately NO hard cap on total fitted ops -- see the note in
# `session.progress_report`. One was tried at 20 and removed: it bounded real waste (73 of 333 clean
# sessions measured exceeded 20 fits despite the streak cap, using 43% of all fits) but paid for it
# with ~1.8% of passes, charged unevenly against the agents that explore longest. Bound cost in the
# client, not in the scoring path.

# The chi2 goodness-of-fit p-value at/above which the best model is considered statistically
# adequate for the plateau gate (its residuals are consistent with the known measurement noise).
ADEQUATE_GOF_PVALUE = 0.05

# The OTHER tail of the same chi2 GoF test. A p-value close to 1 means chi2 is far BELOW its
# degrees of freedom — the residuals are much smaller than the KNOWN noise, i.e. the model is
# fitting the noise rather than the signal. Testing only `p >= 0.05` is one-sided and scores such
# an overfit as "adequate" (observed: fits at chi2/n = 0.016 and 0.051 with p = 0.9999999).
# Only meaningful for a FIXED sigma: when sigma is ESTIMATED the MLE absorbs residual shrinkage
# into sigma itself, pinning chi2/n at ~1 from below (empirical floor 0.98 over 40 fits), so this
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
# near-step-function (one fit landed a Hill coefficient of 15), which makes the ODE
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
# inside it — agents did exactly this (sigma fit to 20-60x its nominal scale) while
# the underlying mechanism was missing whole reaction pathways. A sigma that has to explain a
# large fraction of the data's own variability isn't measurement noise anymore; it's covering
# for the model.
#
# CALIBRATION (#118): both thresholds are measured in the coordinate where the noise is ADDITIVE
# (see reconcile.sigma_noise_coordinate) — the only scale on which this ratio means anything. They
# are deliberately WIDE, because published problems' own ground-truth sigmas span ~0.0004-0.59 of
# their data std: a tight band rejects the true model. The band's job is to catch the pathological
# tails (a sigma driven to ~0 to overfit the noise, or inflated many-fold to absorb misfit), NOT to
# adjudicate a merely unusual-looking noise level.
NOISE_SIGMA_MAX_FRACTION_OF_DATA_STD = 0.75

# The other tail: a fitted noise sigma BELOW this fraction of its observable's own data std is
# implausibly small — the model is explaining ~all of the data's variability, the fingerprint of
# overfitting the noise. Shared by the plateau adequacy gate (session) and a client's success gate
# so both agree on "collapsed".
NOISE_SIGMA_COLLAPSE_FRACTION = 0.01

#: Fewest measurements an observable group needs before its sigma is judged against the data
#: SPREAD at all. Below this the ratio carries no information: `sigma_data_std` is a POPULATION
#: std (``ddof=0``), which is the same functional form as the Gaussian MLE for sigma, so as soon as
#: the model reproduces the group's mean and resolves nothing within it the two are equal BY
#: CONSTRUCTION and the fraction is 1. One problem is the case — four of its absolute readouts have
#: n=3 and every one lands on fraction 0.999994, four different observables agreeing to six
#: significant figures because it is an identity, not a measurement. The model-free Rice estimator
#: puts those same four at 0.67-1.09, i.e. unremarkable. Below this many points the gate reports
#: `status: "unknown"` rather than a verdict it cannot support.
NOISE_SIGMA_MIN_POINTS = 5

# -- replicate (pure-error) noise floor (REPORTED, never gating) --------------
# Where measurements REPEAT at the same (observable, condition, time), their scatter is pure noise —
# an UNBIASED, model-free floor (reconcile.sigma_replicate_noise_estimate), reported per sigma as
# data_noise_estimate / fitted_over_data_noise. Deliberately NO threshold on that ratio: #140 had one
# (1.5x = inflated), since deleted — see fitting._entry_inflated.
#
# Minimum pooled replicate degrees of freedom to report the floor at all. Below this the estimate is
# too noisy to show, and the report falls back to the (biased-upward) difference estimate.
NOISE_SIGMA_REPLICATE_MIN_DOF = 6

# -- replicate lack-of-fit (the residual arm of the structure verdict) -------
# The SAME repeated design points, used the other way round: instead of pooling their scatter into
# one floor to judge sigma against, ask per point whether the PREDICTION sits further from the data
# than that point's OWN scatter allows —
#     z_i = (ybar_i - f_i) / (s_i / sqrt(n_i))   on the noise-additive coordinate.
# This is the axis a premature finish hid behind on a problem whose 16 states sum to 1 (the worked
# case throughout this block). The chi2 GoF p-value cannot see it (with one shared sigma at its ML
# value chi2 == n identically, so the p-value is a function of (n, k) alone: 252 for the true model
# AND for a 6-parameter fit 194 BIC worse), the sign/runs tests cannot see it (a single
# steady-state condition makes every per-observable series length 1 — n=1, p=1.0, undefined runs
# test), and the inflation ratio above barely separates them (1.4759 vs 1.5). z_i separates by
# 5.3x on max|z| (truth 3.393 vs that fit 17.976) and 41x on sum z_i^2, and because it uses only
# s_i it is EXACTLY invariant to the ML vs RSS/(n-k) sigma convention — which the ratio is not.
#
# Per-point two-sided Student-t p on df_i = n_i - 1, HOLM-adjusted across the replicated points of
# ONE noise parameter, flagged below this alpha. 0.01, NOT 0.05: that problem's TRUE model lands at
# Holm p = 0.0519, so a 5% rule is a coin flip that would deadlock the problem about half the time
# (its MC-calibrated 5% critical value for max|z| is 3.97 against the truth's 3.393 — 15% headroom).
# At 0.01 the truth keeps 2.2x headroom (MC crit 5.26) while the failing fit is flagged at
# 2.55e-11, and that margin also absorbs the naive Holm null being ~2x anti-conservative versus the
# MC one — so no Monte-Carlo calibration is needed at runtime.
#
# NOT a variance-ratio F test: the shared-sigma fit weights by diag(n_i) while sum z_i^2 weights by
# diag(n_i / s_i^2), so the projection is OBLIQUE and removes almost no df (exact E[T|H0] = 13.60 at
# m = 15 with rank J = 8, not m - p = 7). Dividing by m - p inflates F by up to 2.14x and
# MANUFACTURES a rejection of that problem's true model at p = 0.0016. Per-point Holm needs no df
# bookkeeping at all, which is why it is the rule here.
#
# READ THE MARGINS ABOVE AS A FLOOR, NOT AS CALIBRATION. Every one of them assumes the n_i replicates
# of a point are independent draws, and on that problem they are NOT: rows 1-12 and 13-18 of each
# design point are two experimental blocks (the three n_i = 12 observables stop exactly at row 12;
# Welch per observable, Fisher chi2 = 54.67 on 24 df, p = 3.5e-4; 12 of 14 shifts share a sign, with
# the abundant state moving down while every minor state moves up — one compositional degree of
# freedom, not noise). Within-block variance is 0.044704 against a naive within-point 0.0665166, so
# s_i/sqrt(n_i) understates se(ybar_i) by a median 2.19x and by 10.3x at the abundant state;
# block-aware, the truth's excess-variance ratio falls 3.330 -> 0.339 and the failing fit's
# 98.5 -> 13.4. That bias runs the SAFE way here — it inflates every z_i, so the truth looks worse
# than it is and still clears 0.01 — but it means these p-values are not exact, and a future
# tightening toward 0.05 has less headroom than the numbers above suggest. Holm survives it for a
# different reason: it controls FWER under ARBITRARY dependence, which matters because the z_i are
# strongly correlated (the states sum to 1; top eigenvalue 6.2 of 15) — a Fisher combine or a
# sum-of-squares aggregate would NOT be valid here. Nothing corrects for the blocks on purpose:
# block membership exists only in measurement ROW ORDER, which PEtab gives no meaning, so no shipped
# code may read it. Subtracting within-point block means would move the pooled floor
# 0.257908 -> 0.248610 and put the failing fit at 1.5311, tripping the inflation ratio above — worth
# knowing, not worth depending on row order for.
REPLICATE_LOF_ALPHA = 0.01

# A design point's own scatter is only a usable yardstick from this many degrees of freedom
# (df_i = n_i - 1) up. 2 EXCLUDES a problem whose 126 replicated points ALL have n_i = 2: those are
# technical replicates of a bound-railed assay calibration whose reference sigma is 7.8x their pure
# error, so a pure-error test there rejects that problem's OWN true model (pooled lack-of-fit
# F = 156.2 on (98, 126), p = 1e-100). One df also cannot tell a real deviation from a lucky pair,
# and Levene/Bartlett are degenerate there.
REPLICATE_LOF_MIN_DF = 2

# Below this many usable replicated points the Holm family is too small to conclude anything, so the
# arm reports NO EVIDENCE either way. 3 is the smallest count at which a single flagged point
# survives a Holm correction at alpha 0.01 without needing an absurd |z|.
#
# This is the floor for a FAMILY to be testable at all, not the threshold for a pass —
# REPLICATE_LOF_MIN_LOF_DF below is that, and it subtracts the estimated structural parameters from
# the POOLED points of every family. So a 3-point family can FLAG, and a problem whose 3-point
# families add up can still pass un-weak; a lone one against any real model cannot. The two numbers
# being equal is a coincidence of 3 being small, not a shared meaning.
REPLICATE_LOF_MIN_POINTS = 3

# Lack-of-fit degrees of freedom (POOLED usable points MINUS the whole-model structural parameter
# count, subtracted once — residual_tests.lof_power) below which a PASS is too weak to report as one.
# A FLAG is unaffected and stands at any df: a point that deviates by more than its own replicate
# noise allows AFTER fitting is real evidence however few df remain.
#
# The asymmetry is the whole point. Refusing only at exact saturation (df <= 0) lets df = 1 report a
# pass, and a summary calling that "powered" endorses a model on the strength of its own flexibility.
# Note what df = 1 does and does NOT mean, taking as the worked case a problem whose 16 states sum
# to 1: the data occupy 15 dimensions and 14 structural
# parameters trace at most a 14-dimensional manifold in them. The fit can zero 14 of the deviations
# but NOT the 15th, so max|z| = 0.107 in the remaining direction is genuine evidence — one degree of
# freedom of it. Weak, not absent. What was wrong was calling it powered, not testing at all.
#
# 3 is where a pass starts carrying enough directions to mean something, and it leaves the cases that
# matter untouched: that problem's TRUE model has 8-9 structural parameters over those 15 points
# (df 6-7) and the under-parameterised fit that motivated this arm had 6-7 (df 8-9).
#
# It also raises the bar for a POWERED pass above REPLICATE_LOF_MIN_POINTS above: a family needs
# min_points to be testable at all AND the problem needs min_lof_df spare directions for its pass to
# stand un-weak.
#
# ADVISORY, and only that. The df BAND reaches the summary prose the agent reads and the coverage
# credit that clears `low_power` — that last one keys on the band, since `weak` is False on a flag
# by construction. NO stop condition consumes any of it: a weak pass clears the plateau stop and the
# finish gate exactly as a powered one does. That is deliberate (the objective and how a fit is
# scored stay as published); the reader is an agent reading the prose, so here the wording IS the
# mechanism.
REPLICATE_LOF_MIN_LOF_DF = 3

# At most this many of the worst FLAGGED points (by |z|) ride on a fit result — flagged only, since
# the advice built from them tells the agent that is where the form is wrong, and a point the model
# already reproduces is not one of those. The verdict is re-sent in the agent's history on EVERY
# subsequent turn (PR #23 cut ~40% of the per-call prompt by trimming exactly this kind of payload),
# so the arm emits a one-line verdict plus a handful of NAMED points — never the full table (a
# 15-point problem would be 15 rows per fit, forever).
REPLICATE_LOF_MAX_REPORTED_POINTS = 5

# -- reference-sigma comparison (the preferred noise check, #118) -----------
# When the TRUE model's own fitted noise sigmas are recorded, a client compares a candidate's sigma
# to those instead of to a fraction of the marginal data std — the dimensionally sound comparison,
# and the one that doesn't conflate variance the model EXPLAINS with variance it can't (which is
# why two true models looked "collapsed" at 0.0004-0.0026 of the marginal std).
#
# A fitted sigma is an ML estimate, so how far from the reference it may legitimately land is set by
# how many points it was pooled over: (fitted/reference)^2 ~ F(d_candidate, d_reference). A sigma is
# implausible only OUTSIDE the central 1-alpha interval of that F, Bonferroni-split across the
# sigmas tested. alpha matches the replicate lack-of-fit arm's, for the same reason: this decides a
# fit's verdict, so the tail has to be one a legitimate fit essentially never lands in.
#
# This WIDENS a fixed tolerance factor of 3, which was the ~99% interval for a sigma with about 7
# residual df applied to every sigma regardless of its own — far too tight on a small group
# (one dose-response sigma is pooled over 7 points; at its real df a ratio of 0.25 is a ~1-2%
# event, and the fixed band called it certain gaming and discarded a fit at dBIC -21.65).
#
# It does NOT address the other half of that miscalibration: over 100 points a ratio of 2.9 cannot
# happen by chance, and the union with the fixed factor below still passes it, exactly as before.
# Tightening there needs a null that admits an honest near-miss — a candidate that is not literally
# the truth has somewhat larger residuals for reasons that are not sampling noise — which this F is
# not. Left deliberately.
# The floor on tolerance, and what the band falls back to where df cannot support anything sharper.
# It is NOT a ceiling: a small group's sampling band is wider than this, and the union of the two
# bands uses that width. Keeping it means the df-aware band only ever RELAXES this
# gate — a sigma pooled over hundreds of points has a sampling band tighter than +-20%, and applying
# that would fail honest near-misses, whose residuals are legitimately somewhat larger than the
# truth's for reasons that are not sampling noise.
NOISE_SIGMA_REFERENCE_FACTOR = 3.0
NOISE_SIGMA_BAND_ALPHA = 0.01
# Below this many residual degrees of freedom the SAMPLING band is meaningless — with one or two,
# nothing short of an absurd ratio is distinguishable from noise, so a band computed there is
# measuring its own threshold rather than the model. The fixed factor above is then applied on its
# own: the gate stays on (an absurd ratio is still absurd), it just gains no df-awareness.
NOISE_SIGMA_BAND_MIN_DF = 3.0

# -- repeated-widening backstop ---------------------------------------------
# A parameter pinned at a bound with a flat / collinear FIM direction is UNIDENTIFIABLE, not
# under-explored: widening its box just lets the optimum slide to the new edge and re-pin for a
# negligible fit gain, chasing the search space toward infinity (one agent widened one rate 4x
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
# bounds within the same structure. The failure mode this addresses (seen across one problem's
# deep-dive) is an agent that correctly DIAGNOSES structural inadequacy and then re-fits the same
# structure anyway. The
# early-stop plateau/streak signal already detects "the score has stopped moving", so rather than
# adding a separate detector the structural-change nudge is folded into the ``stop_reason`` those
# stops emit (see ``session._STRUCTURAL_PLATEAU_ADVICE`` / ``progress_report``), model-agnostically
# (categories of structural move, never a specific mechanism).

# -- reference validation --------------------------------------------------
# Network timeout (seconds) for resolving a reference against NCBI E-utilities.
REFERENCE_TIMEOUT_SECONDS = 15.0
