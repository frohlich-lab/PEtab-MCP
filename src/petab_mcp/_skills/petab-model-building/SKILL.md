---
name: petab-model-building
description: >-
  Read before building or fitting a PEtab parameter-estimation problem. The PEtab semantics
  that are easy to get wrong: which symbols belong to the observation model and must stay out
  of the mechanism, how the condition table drives the model, what the parameter table decides
  (and what model text cannot), why a fitted value is linear on a log10 scale, and why
  goodness-of-fit is meaningless when the noise sigmas are estimated.
---

# PEtab semantics that trip up model building

A PEtab problem is four tables plus a model. The model says what the system *does*; the tables
say what was measured, under what conditions, and which quantities the data determine. Most
mistakes come from confusing those roles.

## The observation model is not the mechanism

Symbols resolved through the **observable** or **noise** formulas — or through the measurement
table's `observableParameters` / `noiseParameters` columns — are scaling factors and noise
sigmas. They describe how the readout relates to the state, not how the system evolves.

**Never put one in a reaction, rule, or initial assignment.** Doing so lets a model reach an
adequate-looking fit by repurposing part of the fixed observation model instead of introducing
the kinetic parameter the mechanism actually needs — a model that is severely
under-mechanistic while looking fine on its own metrics. Introduce your own parameter for a
rate, always.

## The condition table drives the model

Each condition column sets a model quantity for that condition — a dose, an input level, a
genotype switch. Two consequences:

- **Names must match.** The column targets a model entity by name; a mismatch means the
  perturbation silently never happens.
- **The model must actually use it.** If nothing references the quantity, every condition
  simulates identically and the experimental design is thrown away.

A condition-set initial value is an experimental *input* to reproduce, not a value to fit.

## The parameter table decides what is estimated — model text cannot

`estimate` is a column of the parameter table, not something the model language can express.
So:

- The rows with `estimate=0` are the complete set of constants you were given. A quantity you
  believe you know, but which is not in that set, must still be **estimated** — the data are
  meant to determine it.
- You cannot fix a value by writing a literal (that hides it from the table entirely) or by
  giving it an ultra-narrow box (the row still says `estimate=1`, so it is still a free
  parameter, however tight the bounds).
- Whether an **initial condition** is estimated is likewise a table property. Renaming it in the
  model — `A_init`, `A_basal`, `p0_A` — changes nothing.

## A fitted value is linear, whatever the scale

`parameterScale` is commonly `log10`. Bounds are given on that scale, but `nominalValue` and
the fitted value are **linear** numbers, never their log10 coordinates. Reading a fitted `1e-3`
as "10⁻³ on a log scale" is a factor-of-1000 error, and it is the usual reason a parameter is
wrongly declared pathological.

A parameter that must stay positive (a rate) suits `log10`. A **fraction, efficiency, or
bounded multiplier** does not: on a wide log10 box it drifts to physically meaningless values —
an inhibition fraction above 1 that flips a rate negative. Put those on a linear scale with
bounds that keep them physical.

Exponents (Hill / cooperativity coefficients) belong on a **linear** scale with tight bounds
(≈1–8). On a wide log10 scale an exponent fits to a huge value — a near-step function that
makes the ODE stiff and stalls the fit.

## Estimated noise sigmas make goodness-of-fit meaningless

If a `sigma_*` parameter has `estimate=1`, the fit inflates it to absorb the residuals. Then
χ² pins near the number of data points by construction: `reduced_chi2 ≈ 1` and its p-value look
fine no matter how wrong the mechanism is — they measure the fitted noise level, not the
mechanism. Widening or freeing such a sigma "improves" them for the same empty reason.

What still carries information is the residual **structure**: a systematic per-observable bias,
or a trend of the residuals with time. Sign-based tests (sign test, runs test) survive because
they ignore magnitude. Read that structure on the **raw** residual, not the noise-weighted one —
dividing by an inflated sigma shrinks every weighted residual toward zero, so the weighted view
looks adequate by construction.
