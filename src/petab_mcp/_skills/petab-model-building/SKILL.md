---
name: petab-model-building
description: >-
  Read before building or fitting a PEtab parameter-estimation problem. The PEtab semantics
  that are easy to get wrong: which symbols belong to the observation model and must stay out
  of the mechanism, how the condition table drives the model, what the parameter table decides
  (and what model text cannot), why a fitted value is linear on a log10 scale, why steady-state
  data cannot pin initial conditions, and why goodness-of-fit is meaningless when sigmas are fit.
---

# PEtab semantics that trip up model building

A PEtab problem is four tables plus a model. The model says what the system *does*; the tables
say what was measured, under what conditions, and which quantities the data determine. Most
mistakes come from confusing those roles.

## The observation model is not the mechanism

Symbols resolved through the **observable** or **noise** formulas — or through the measurement
table's `observableParameters` / `noiseParameters` columns — are scaling factors and noise
sigmas. They describe how the readout relates to the state, not how the system evolves.

**Never put one in a reaction, rule, or initial assignment.** That lets a model fake an adequate
fit by repurposing the fixed observation model instead of the kinetic parameter the mechanism
needs — under-mechanistic but fine on its own metrics. Always introduce your own parameter for a
rate.

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

## What may NOT be a parameter-table row

You do not write that table — it is synced from the model. Two kinds of symbol are forbidden in
it, and a problem carrying one is rejected outright:

- **A quantity the model computes**: the target of an assignment or rate rule (`prevalence :=
  I/N;`). It is not a free parameter, so PEtab excludes it.
- **A condition-table column**: an input supplied per condition, not fitted.

The sync handles both — don't add a row for a rule-computed quantity, and don't expect one to
appear.

Observable/noise symbols are the opposite case: a `sigma_*` or scaling factor **belongs** in the
table (only `observableParameter1_*` placeholders don't) — keep them out of the mechanism, not the
table.

If a rejection names **Missing** or **Extraneous** parameters, the model and the tables disagree
about a NAME — usually one you renamed on one side only.

## Initial assignments refer to things that already exist

`A = A_init;` becomes an SBML `<initialAssignment>`. Its target must be a **declared** species,
compartment or parameter — and it may not also be the variable of an **assignment rule**: give a
quantity an initial value *or* compute it with a rule, never both.

## Steady-state data can't identify ICs or the rate scale

With all measurement times `inf` (steady state), only the equilibrium is fit — it ignores where
you started and is unchanged by scaling every rate. So initial conditions and the absolute rate
scale are unidentifiable: give each species its physical start (`x_0 = 1`, rest `0`; a 0/1 value
is held fixed) instead of estimating it, and keep the one fixed rate as the scale.

## A fitted value is linear, whatever the scale

`parameterScale` is commonly `log10`. Bounds are given on that scale, but `nominalValue` and
the fitted value are **linear** numbers, never their log10 coordinates. Reading a fitted `1e-3`
as "10⁻³ on a log scale" is a factor-of-1000 error, and it is the usual reason a parameter is
wrongly declared pathological.

A rate (must stay positive) suits `log10`. A **fraction, efficiency, or bounded multiplier** does
not — on a wide log10 box it drifts past its physical range (an inhibition fraction above 1 flips
a rate negative); put those on a linear scale with physical bounds. Exponents (Hill /
cooperativity) also belong on a **linear** scale with tight bounds (≈1–8): on log10 they fit to a
huge value — a near-step function that makes the ODE stiff and stalls the fit.

## Estimated noise sigmas make goodness-of-fit meaningless

If a `sigma_*` parameter has `estimate=1`, the fit inflates it to absorb the residuals. Then
χ² pins near the number of data points by construction: `reduced_chi2 ≈ 1` and its p-value look
fine no matter how wrong the mechanism is — they measure the fitted noise level, not the
mechanism. Widening or freeing such a sigma "improves" them for the same empty reason.

What still carries information is the residual **structure** — a per-observable bias or a trend
with time. Sign/runs tests survive because they ignore magnitude. Read it on the **raw** residual:
dividing by an inflated sigma shrinks every weighted residual toward zero, so the weighted view
looks adequate by construction.
