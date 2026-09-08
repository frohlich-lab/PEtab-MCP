---
name: petab-model-building
description: >-
  Read before building or fitting a PEtab parameter-estimation problem. The semantics that are
  easy to get wrong: which symbols belong to the observation model rather than the mechanism, how
  the condition table drives the model, what the parameter table decides and model text cannot,
  how an initial value fixes its own row, why bounds and fitted values are linear whatever the
  scale, and why goodness-of-fit is meaningless when sigmas are fit.
---

# PEtab semantics that trip up model building

A PEtab problem is four tables plus a model: the model says what the system *does*, the tables
say what was measured and what the data can determine. Most mistakes confuse those roles.

## The observation model is not the mechanism

Symbols resolved through the **observable**/**noise** formulas — or the measurement table's
`observableParameters`/`noiseParameters` columns — are scaling factors and noise sigmas: they
describe how the readout relates to the state, not how the system evolves.

**Never let one stand in for a rate constant.** A rate law that uses an observation symbol and
declares no parameter of its own is rejected — that is how a model fakes a fit by repurposing the
observation model instead of the kinetic parameter it needs. A **noise sigma** is
rejected anywhere in model math: reusing one only buys a looser noise model. Alongside a dedicated
rate constant, or in an initial assignment, such a symbol only warns — one quantity is often both,
a total abundance that sets an initial condition *and* divides a relative readout.

## The condition table drives the model

Each condition column sets a model quantity for that condition — a dose, an input, a genotype
switch. Two consequences:

- **Names must match.** The column targets a model entity by name; a mismatch means the
  perturbation silently never happens.
- **The model must actually use it.** If nothing references it, every condition simulates
  identically and the design is thrown away. Exception: a column holding the SAME value in every
  condition encodes no design, so it need only exist.

A condition-set initial value is an experimental *input*, not a value to fit.

## The parameter table decides what is estimated — model text cannot

`estimate` is a column of the parameter table, not something the model language can express:

- The rows with `estimate=0` are the complete set of constants you were given. A quantity you
  believe you know but which is not in that set must still be **estimated**.
- To hold a value fixed, BUILD ON a constant the problem ships with `estimate=0` and keep
  declaring it. Writing the number as a literal does not make it fixed — it makes it unfittable,
  which is not the same thing; and an ultra-narrow box leaves the row `estimate=1` either way.
- Whether an **initial condition** is estimated is likewise a table property you cannot set
  here — it is DERIVED from the initial value you write in the model. Renaming it in the model —
  `A_init`, `A_basal`, `p0_A` — changes nothing.

## What may NOT be a parameter-table row

You do not write that table — the sync does, and it excludes both forbidden symbols:

- **A quantity the model computes**: an assignment- or rate-rule target (`prevalence := I/N;`).
- **A condition-table column**: an input supplied per condition, not fitted.

Observable/noise symbols are the opposite case: a `sigma_*` or scaling factor **belongs** in the
table (only `observableParameter1_*` placeholders don't) — out of the mechanism.

**Missing**/**Extraneous** parameters in a rejection mean the model and tables disagree about a
NAME — usually one you renamed on one side only.

## Initial assignments refer to things that already exist

`A = A_init;` becomes an SBML `<initialAssignment>`. Its target must be a **declared** species,
compartment or parameter, and may not also be an **assignment rule** variable: give a quantity an
initial value *or* compute it with a rule, never both.

## Steady-state data can't identify ICs or the rate scale

With all times `inf` only the equilibrium is fit: it ignores where you started and is unchanged
by scaling every rate, so ICs and the absolute rate scale are unidentifiable. Give each species
its physical start (`x_0 = 1`, rest `0`, which fixes those rows) rather than estimating it, and
keep the one fixed rate as the scale.

## A fitted value is linear, whatever the scale

`parameterScale` (commonly `log10`) changes only the coordinate the OPTIMIZER searches in.
Everything you read or write — bounds, `nominalValue`, the fitted value — is **LINEAR**. A rate
ranging over 1e-4..1e2 has bounds `1e-4` and `1e2`; `-4`/`2` is rejected, since a log10 row needs
a positive lower bound.

**Use a fitted value exactly as reported — never exponentiate it.** A fitted `1e-3` on a `log10`
row IS 0.001. Treating it as the log10 coordinate and computing `10^0.001 ≈ 1.0` is off by a
factor of 1000, and is the usual reason a healthy parameter gets called pathological.

A rate (must stay positive) suits `log10`. A **fraction, efficiency, or bounded multiplier** does
not — on a wide log10 box it drifts past its physical range (an inhibition fraction above 1 flips
a rate negative); put those on a linear scale with physical bounds. Exponents (Hill) likewise:
**linear**, tight bounds (≈1–8) — on log10 they fit a near-step function that stiffens the ODE.

## Estimated noise sigmas make goodness-of-fit meaningless

If a `sigma_*` parameter has `estimate=1`, the fit inflates it to absorb the residuals, so χ²
pins near the data-point count by construction: `reduced_chi2 ≈ 1` and its p-value look fine
however wrong the mechanism is. Widening or freeing such a sigma "improves" them for the same
empty reason.

What still carries information is the residual **structure** — a per-observable bias or a trend
with time; sign/runs tests survive because they ignore magnitude. Read it on the **raw** residual:
dividing by an inflated sigma shrinks every weighted residual toward zero.
