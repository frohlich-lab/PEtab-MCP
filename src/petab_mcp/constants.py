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

# -- estimated initial-condition bounds ------------------------------------
# Bounds for an ESTIMATED initial-condition parameter: a concentration, so non-negative, on a
# linear scale, generously bracketing the observed t=0 value (upper = the larger of the floor and
# a multiple of that value).
IC_LOWER_BOUND = 0.0
IC_UPPER_BOUND_FLOOR = 1e2
IC_UPPER_BOUND_FACTOR = 10.0

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
NOISE_SIGMA_MAX_FRACTION_OF_DATA_STD = 0.5

# -- reference validation --------------------------------------------------
# Network timeout (seconds) for resolving a reference against NCBI E-utilities.
REFERENCE_TIMEOUT_SECONDS = 15.0
