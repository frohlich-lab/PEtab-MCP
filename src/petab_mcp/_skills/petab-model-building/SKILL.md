---
name: petab-model-building
description: >-
  Use when interacting with the petab-mcp server to work a single PEtab v1
  parameter-estimation problem: read the fixed experimental design, author or re-author the
  whole model in Antimony with set_model (the single model editor) and adjust parameter search
  spaces (change_parameter_scale, change_parameter_bounds) — these edits are FIT-FREE — then call
  `fit` once to score the assembled change and snapshot a version, and inspect the fit by querying
  the data with SQL (query_data over the PEtab tables + the simulation dataframe/residuals),
  provenance, and versions. Explains the disciplined, auditable tool workflow, the rules each tool
  enforces, how to read tool results, and how to recover from rejected operations. Read this
  before calling any petab-mcp tool.
---

# Working a PEtab problem with petab-mcp

`petab-mcp` wraps **one** PEtab v1 parameter-estimation problem and turns model
building into a disciplined, auditable loop. The tools are deliberately narrow:
the experimental design is read-only, the model changes only through `set_model` (the single
model editor — author or replace the whole model in Antimony), and the parameter search space
through `change_parameter_scale` / `change_parameter_bounds`. These edits are **fit-free** — they
mutate the working model and are logged, but scoring is a separate, explicit step: you assemble a
coherent change through edits, then call `fit` **once** to score it and snapshot a version. Work
*with* those constraints — do not look for an "edit the measurements" tool or an incremental
single-reaction editor; there aren't any, by design. To write the Antimony that `set_model` takes,
see the companion `antimony-authoring` skill.

For the exhaustive per-tool signature/return reference, read `tools.md` in this
skill (also available from the server as `get_skill("petab-model-building",
file="tools.md")`).

## The invariants (read first)

- **The design is fixed.** `measurements`, `conditions`, and `observables` are
  read-only. No tool mutates them; an edit that breaks a fixed observable or
  condition is *rejected*, not applied.
- **The model is authored in Antimony with `set_model`.** If the problem ships *without* a
  model (`describe_problem` reports `model_initialized: false`), author the whole model from
  Antimony with `set_model`. `set_model` builds the first model and re-authors an existing one
  wholesale. Antimony is the canonical source of truth; SBML is
  derived.
- **Every model operation must cite a justification.** The justification must be a
  non-empty string. References are **optional** for all operations; any you provide are
  validated against PubMed.
- **Consistency is automatic.** After each operation the server *reconciles*
  SBML↔PEtab: new rate-law symbols become estimated model parameters (log10 scale by default,
  with bounds bracketing the nominal value), new species get a **parametric initial condition**
  (`S = S_init`, with `S_init` added to the parameter table on a linear scale, *fixed*), and
  orphaned entities are pruned. The parameter table therefore always matches the model.
  - **Initial conditions are FIXED by default** (whether they are estimated is a fixed
    per-scenario setting, not something you switch in-loop). Every edit's **`initial_conditions`**
    block reports each species' t=0 status (`fixed` / `estimated` / `condition_table`, and whether
    the declared t=0 is used) — read it rather than re-authoring to try to change an IC. *How* ICs
    reconcile is covered in the `antimony-authoring` skill.
  - **Exponents are special.** A new symbol used as an **exponent** — a Hill / cooperativity
    coefficient, e.g. `n_hill` in `K^n_hill/(K^n_hill + x^n_hill)` — is created on a **linear**
    scale with **tight bounds** (≈1–8), not the default log10. This is deliberate: on a wide
    log10 scale an exponent fits to a huge value (a near-step-function) that makes the ODE stiff
    and the fit plateau. Keep it that way — if you set an exponent's scale/bounds by hand, use
    `lin` with a tight range; never put a Hill coefficient on log10 with wide bounds.
  - **Noise sigmas have a reference, not a free run.** Every fit's `noise_sigma_report`
    (also in `get_fit_summary`) lists each ESTIMATED noise-sigma parameter's fitted value
    against a data-driven reference — a fraction of its observable's own raw measurement std
    (across every time point/condition), see `NOISE_SIGMA_MAX_FRACTION_OF_DATA_STD`. A sigma
    free to explain a large share of the data's own variability isn't measurement noise
    anymore, it's covering for the mechanism: a benchmark run once let its sigmas float to
    20-60x their nominal scale, which alone made a model missing whole reaction pathways pass
    its own goodness-of-fit check. The shipped bounds themselves are left alone, but
    `change_parameter_bounds` REJECTS an attempt to WIDEN a sigma's `upperBound` past this
    reference — if a fit pins a sigma at its current bound, that's the SAME signal as any other
    at-bound parameter below: fix the mechanism, don't try to widen the box.
- **Observation-model parameters are off-limits inside the mechanism.** An observable-scaling
  factor or noise sigma — anything resolved through the measurement table's
  `observableParameters`/`noiseParameters` columns or an observable/noise formula — must NEVER
  also appear inside a reaction, rule, or initial assignment. Reusing one there is a REJECTED
  edit (an SBML-vs-PEtab structural check, not a lint warning): it lets a model dodge
  introducing a real kinetic parameter by repurposing one that's actually part of the fixed
  observation model. Two independent runs on the same benchmark did exactly this — wired the
  four fixed observable-scaling factors directly into their reaction rate laws instead of the
  ~14 real kinetic parameters the scenario expected — and both looked "adequate" until the fit
  was compared against the true mechanism. Introduce your OWN parameter for a reaction rate,
  always.
- **Parameter roles are enforced — respect what each parameter IS.** `get_parameters()` gives
  every row a `role` and a `status`; keep to them.
  - A `noise_sigma` or `observable_scaling` factor belongs to the fixed observation model, never
    the mechanism (the previous invariant — reusing one in a rate law is a *rejected* edit).
  - **You don't set a parameter's estimated/fixed status — there is no tool to toggle it.** A
    symbol you introduce in the Antimony comes in as an *estimated* parameter (new rate-law
    symbols are added estimated); a parameter the problem ships fixed stays fixed only while your
    model keeps declaring it. So the way to control what is estimated is the *model*, not a flag.
  - **Don't spend an estimated parameter on a value you already know.** A parameter you introduce
    comes in `estimate=1` and counts toward *k* (the estimated-parameter count), inflating BIC
    however tight its bounds — a narrow box via `change_parameter_bounds` does not make it "fixed",
    it just makes an expensive parameter impersonate a constant (and a literal in the rate law is
    not the answer either — it hides the value from the parameter table). The known fixed constants
    a problem cares about already ship *fixed* in its parameter table; build the mechanism on those
    (keep them declared) instead of introducing your own. A run that pinned three known constants
    with ultra-narrow bounds fit a 6-parameter model that lost on BIC to the 3-parameter winners
    that didn't estimate those constants at all.
  - **A shipped-fixed mechanistic constant can't be removed or freed.** A shipped-fixed
    parameter is a known constant — a conservation total (e.g. a `*_total` held at 1), a measured
    concentration, a fixed rate. Dropping it from the model — re-authoring without it, or removing
    the reaction that uses it, so reconcile prunes it — is a *rejected* edit
    (`check: "fixed_constant_removed"`); so is having it come back estimated after a
    drop-and-reintroduce (`check: "fixed_constant_made_estimated"`): an estimated conservation
    total drifts off its value (a run let totals fixed at 1 wander to 9 / 0.001 / 0.4) and adds to
    *k*. Keep such constants declared and build the mechanism on them, as shipped. (Initial-
    condition parameters are pruned normally with their species, and observation-model factors are
    exempt — they have their own estimation paths.)
  - **The model must actually USE every fixed constant and every condition-table quantity.** A
    fixed constant you declare but never reference in a rate law/rule is pruned as an orphan and
    then trips the removal rule above — so *use* each one, don't just declare it. Likewise, every
    condition-table column sets a model quantity per condition (a dose, an input level, a genotype
    switch); if the model never uses that quantity, the conditions don't perturb it — every
    condition simulates identically and the experimental design is ignored. Declaring a
    condition-table target the model never references is a *rejected* edit
    (`check: "unused_condition_target"`): wire each condition quantity into the mechanism (an
    input/driver in a rate law, or the initial state of a species that participates in the
    dynamics).
- **Everything is versioned — a version is a fit.** Every operation (edits, `fit`,
  rollbacks, resets) appends a provenance event, so nothing is lost. A **version** is
  specifically a fit snapshot: an edit mutates the working model and records an `edit`
  event but does *not* create a version, and only `fit` (or a rollback/reset) snapshots
  one. You can diff, roll back, or reset.
- **Fitting is a separate, explicit step.** Edits never fit — they mutate the working
  model, reconcile, and lint, but carry no score. A dedicated **`fit`** tool is the only
  thing that fits: it runs a pyPESTO multistart over the current working model's free
  parameters, simulates, and snapshots a single version carrying the fitted parameters,
  the fit metrics (neg_log_likelihood, chi2, bic, improvement vs prior versions), and a
  simulation dataframe. Parameter *values* are set only by this fit. `fit` de-duplicates:
  re-fitting a model byte-identical to the current version reuses the stored fit (skipping
  the optimizer) — transparently, with identical results. The discipline is: **assemble a
  coherent structural change through edits, then call `fit` once to score it** — don't fit
  after every micro-edit, and don't stack more edits on a stale score.

## Step 1 — orient yourself (read-only)

Always start by understanding the problem before changing it:

- `describe_problem()` — model size, free parameters, table counts, current
  version, and whether the working model has unfit edits (`has_unfit_edits`,
  `n_pending_edits`).
- `get_model("antimony")` — the canonical model text. Use `get_model("sbml")` for
  SBML, or `get_model_structure()` for a structured species/parameters/reactions view.
- `list_observables()`, `list_conditions()`, `get_measurements(condition_id?, summarize=True,
  observable_id?)` — the fixed data you are fitting. These constrain every edit.
- `get_parameters()` — the parameter table (id, bounds, nominalValue, estimate,
  scale, priors), each row annotated with its **`role`** (`kinetic` / `noise_sigma` /
  `observable_scaling` / `initial_condition` / `exponent` / `constant`) and its **`status`**
  (`estimated` / `fixed`). Read these before you wire a symbol into a rate law or decide what to
  estimate — they keep parameter roles straight (see the role invariant below).
- `validate_problem()` — lint the current problem and see any outstanding issues.

## Step 2 — edit the model (whole-model authoring, justified, fit-free)

`set_model` applies the (re-)authored model to a *copy* of the problem, reconciles SBML↔PEtab,
lints the candidate, and **only applies if the result is valid**. On a lint failure the model and
parameter table are reverted and nothing changes. An edit is **fit-free**: it does *not* fit, does
*not* simulate, and does *not* create a version — its result carries no fit metrics, only
`applied: true`, the reconcile/IC report, and `pending_fit: true` with `n_pending_edits`. Score the
assembled change by calling `fit` (Step 4) once you have made all the edits that belong together.

### `set_model(antimony, justification, references?)` — author the whole model

`set_model` **builds** the first model when the
problem ships model-less (`describe_problem` → `model_initialized: false`) and
**re-authors / replaces** an existing model wholesale. Author the
entire model as one Antimony string declaring the compartments, species (with their initial
conditions), parameters, and reactions:

```
model my_model
  compartment c = 1;
  species S1 in c; species S2 in c;
  S1 = S1_init; S2 = S2_init;
  S1_init = 1.0; S2_init = 0.0;
  k_f = 0.1; k_r = 0.05;
  J1: S1 -> S2; k_f*S1 - k_r*S2;   // reversible: net rate = forward - reverse
end
```

It reconciles and lints — **fit-free** (call `fit`, Step 4, to score it) — and reverts on a lint
failure so a bad (re-)author changes nothing. Every structural change — adding or removing a
reaction, changing a rate law — is made this way, by re-authoring the whole model in one coherent
step. Because the whole model is replaced, the result **warns** if the re-author no longer declares
a previously-*estimated* parameter (its fitted value is dropped); fixed noise / initial-condition
parameters and anything the observable, noise, or condition tables reference are preserved.

**Writing reactions inside the model** (see the `antimony-authoring` skill for the full guide):

- Use `->` between reactants and products; a rate law after `;` is required. Stoichiometry is a
  leading number: `2 S1 -> S2`. New symbols in a rate law that are not species become
  **parameters** (reconciled into the parameter table with default bounds — tune them in Step 3);
  new species get a parametric initial condition.
- **Prefer reversible reactions.** A reaction carries one kinetic law and the dynamics come
  entirely from it (the arrow adds no flux — reactions are created `reversible=false`), so model
  a reversible step X⇌Y as a **single** reaction with a net `forward - reverse` rate law
  (`J: X -> Y; k_f*X - k_r*Y`) rather than two opposing one-way reactions. Use a forward-only law
  (`k_f*X`) only for a genuinely irreversible step — synthesis, degradation, an effectively
  one-way catalytic step.

To go **back** to a prior model, use `restore_version(n)` — never re-author it. A re-author is
just another edit that then has to be re-fit; `fit` de-duplicates and, for a structure already in
the history, warns you to use `restore_version(n)` instead (see Step 4). `restore_version(n)`
reuses that version's stored fit without recompiling or refitting.

### Writing a good justification + references

- **justification**: a concise mechanistic reason for the edit (what the reaction
  represents and why it belongs). Non-empty is required; empty strings are rejected.
- **references**: an *optional* list of citation strings. Accepted forms include PubMed IDs
  (`"PMID:12345678"` or just `"12345678"`) and DOIs (`"10.1016/j.cell.2020.01.001"`).
  In `best_effort` mode (the default) references are verified when the network
  allows and recorded as unverified otherwise, but they must always be *present*.
  In `required` mode an unresolvable reference rejects the operation; `off` skips
  network validation entirely.

## Step 3 — parameters

Parameter **values** (`nominalValue`) are set only by `fit` — there is no manual
value-editing tool. What you *can* change, through justified, fit-free operations, is a
parameter's search space:

- `change_parameter_scale(parameter_id, scale, justification, references?)` —
  `scale` is `"lin"`, `"log"`, or `"log10"`. Switching to `log`/`log10` requires the
  parameter's `lowerBound` and `nominalValue` to be **> 0**, or the operation is rejected.
- `change_parameter_bounds(parameter_id, lower_bound, upper_bound, justification,
  references?)`.

Both are **fit-free** edits like the model edits — reach for them when a fit lands on a bound
or a scale is inappropriate, then call `fit` afterward to re-score. A bounds edit no longer
refits by itself. Read the current table with `get_parameters()`; use
`sync_parameters("report"|"add")` to reconcile the table with the model (operations already
reconcile, so this is rarely needed).

**Fractional modifiers belong on a LINEAR scale with STRICT bounds.** A parameter that is a
fraction / efficiency / bounded multiplier — an inhibition or activation factor in a term like
`(1 − alpha)` or `(1 + alpha)`, a split ratio, a saturating gain — is auto-added on `log10` (the
default for any positive parameter), which is the wrong space for it: on a wide log10 box it can
drift to a physically meaningless value (e.g. an inhibition fraction `> 1` that flips a reaction
rate NEGATIVE) that fits the data slightly better but is nonsense. As soon as you introduce such a
parameter, `change_parameter_scale` it to `"lin"` and `change_parameter_bounds` it to the range that
keeps it physical (typically `[0, 1]`, or whatever keeps the rate factor it enters positive) — a
justified, fit-free pair of edits. This is a modelling choice you make up front from the term's
meaning, not a reaction to a bad fit.

**A parameter that keeps re-pinning to whatever bound you set is not a bounds problem.** Every
fit's `parameters_at_bound` (also in `get_fit_summary`) flags any fitted parameter within
~1% of its lower/upper bound — check it after every fit, not just when a fit looks off.
**Widen at most once.** If the SAME parameter shows up in `parameters_at_bound` again at the new
edge, moving it further won't help: a parameter pinned at a bound with a flat or collinear
**FIM direction is *unidentifiable*, not under-explored** — the data don't constrain it, so the
optimizer slides it to whatever edge you offer. Confirm the diagnosis with `get_identifiability`
(a `flat_direction` or `collinear_with` flag on that parameter, or a large `fold_uncertainty`),
then **do NOT widen or flip its scale again** — repeated widening just chases the value toward
infinity (100 → 1e6 → log scale) for a negligible or negative gain while it re-pins. Judge that
gain in **BIC, not raw NLL**: a bounds-only re-fit leaves the parameter count `k` and data count
`n` unchanged, so `improvement_vs_best_prior` (an NLL delta) converts as `ΔBIC = 2·ΔNLL`, and a
BIC gain below ~2 is negligible. Resolve it **structurally**, in priority order — the fix
removes the redundant degree of freedom from the *model*; it never hard-codes a value (there is
no value/`estimate` tool, and a literal in a rate law is just a magic number — parameter values
come only from the fit):

1. **Remove it** — re-author with `set_model` dropping the reaction/term it belongs to, when
   that parameter is what makes its mechanism redundant.
2. **Use its limiting form** — a rate the fit pushes toward infinity usually means an
   *irreversible* reaction (drop the reverse term) or an *instantaneous* / quasi-steady-state
   step (replace the kinetics with its algebraic limit, which has no free rate); a rate pushed
   toward zero means the reaction isn't needed. Taking the limit deletes the parameter — you are
   not choosing a number for it.

**If neither is possible, leave it be.** An unidentifiable parameter you can't cleanly remove or
take a limit of is not a failure to fix — just stop touching it: do NOT widen again, flip its
scale, or hard-code a value. Leaving it pinned costs you nothing beyond one unconstrained
parameter; the alternatives (a runaway search space or a magic number) are worse. (When
`get_identifiability` shows it is `collinear_with` a partner rather than flat on its own, one
extra option is to merge the pair into the single combination the data *can* constrain — their
product or ratio, e.g. `k_cat*E_tot` → one `Vmax` — but that is optional, not required.)

A single re-pin can also mean the *model* is missing a mechanism the data need (often an exponent
trying to fake a near-step-function) — use `query_data`'s bias/time-trend diagnostics (Step 5) to
see what the current structure can't reproduce, then re-author with the missing reaction rather
than stretch a bound. The rule is the same in every case: never answer a re-pin with another
bounds widening — make a **structural** edit by re-authoring with `set_model`, or leave the
parameter be. The harness enforces this as a backstop: `change_parameter_bounds` **rejects a third
widening** of a still-pinned parameter whose previous widening bought a negligible BIC gain.

If the parameter pinned at its bound is a **noise sigma**, don't even try widening it —
`change_parameter_bounds` will reject a widen past its data-driven cap outright (see the
invariant above). A sigma at that cap means the current mechanism's residuals are already as
large as the data's own variability; the fix is exclusively a model-structure one.

## Step 4 — fit the assembled change (the only fitting step)

Edits are fit-free, so `fit` is the one tool that scores. Make all the edits that belong
to a coherent structural change, then call `fit` **once**:

`fit(n_starts?, justification?)` fits the current working model's free parameters with a
pyPESTO multistart, simulates, and snapshots a **single** version carrying the fitted
parameters, the full fit block (`neg_log_likelihood`, `chi2`, `bic`, `aic`, …), and the
simulation dataframe. `n_starts` overrides the server's default multistart count for this
fit (`0` = simulate without optimising). On success it returns `{"committed": true,
"version": N, "warnings", "progress", …fit block…}`.

`fit` de-duplicates redundant work:

- If the working model is **byte-identical to the current version**, `fit` skips the refit
  and returns the stored metrics: `{"committed": false, "no_change": true, "version",
  "duplicate_of_version", "message", "warnings", "neg_log_likelihood", "bic"}`.
- A structure **already elsewhere in the history** is fit but the result **warns** to use
  `restore_version(n)` instead (which reuses the stored fit without recompiling); the *same*
  structure fit too many times is **refused** to break an unproductive loop.
- A model that **can't compile or has no free parameters** returns `{"committed": false,
  "error": …}` and **leaves the working model in place** to repair — a fit failure does *not*
  revert your edits, so fix the model (so it simulates / has free parameters) and fit again.

The intended discipline is one fit per coherent structural change: fit after you have
assembled a change, *before* deciding the next step — not after every micro-edit, and never
stacking further edits on a stale score. Never answer a bad fit with a reflexive "widen a bound
and refit" loop: assemble the change, score it once, read the fit, then decide.

## Step 5 — read the fit

The score comes from `fit` (Step 4), the one tool that fits and simulates. The `fit` result
and the new version carry the fit metrics — `neg_log_likelihood` and `chi2` (**lower is
better**), the complexity-penalised `bic`, plus improvement metrics (`objective_before`,
`self_improvement`, `improvement_vs_best_prior`, `is_new_best`), so competing versions can be
ranked. `is_new_best` and `improvement_vs_best_prior` track **BIC** (not raw nll) against your
best model so far, and `is_new_best` is gated on a **minimum effect size** (ΔBIC ≥ 2, the
"positive evidence" bar): a fit only counts as a new best when it meaningfully lowers BIC — so
a 1e-6 twitch, or an nll drop bought by an extra parameter that actually *raises* BIC, is not
mistaken for progress. Pass `n_starts` on `fit` to fit harder, or `n_starts=0` to simulate
without fitting.

The `fit` result also carries a `progress` block — `non_improving_streak` (fitted ops since
the last real new best), the lowest-BIC `best_version` seen so far, and
`bic_improvement_over_last_n` (best-BIC gained over the last few ops). When the score stops
moving — the streak reaches its cap, or the best model is already adequate and the BIC has
flattened — the run is stopped and the **best-BIC version is scored**, not whatever happened
to be committed last. So don't keep editing a fit that has reached the noise floor; finish.

**When the fit plateaus, the verdict is structural — change the model's FORM, don't re-fit the
same structure.** When the run stops making progress, the `progress` block's `should_stop` fires
and `stop_reason` carries this nudge. A plateau — the streak reaching its cap, or the BIC flat over
the last few ops — is the signature of the classic failure mode: the model's *structure* cannot
reproduce the data — a missing latent process, an absent feedback reaction, or a rate law of the
wrong form — so an **optimizer/bounds** problem's cure (re-fit, widen bounds, flip a scale,
re-author a near-duplicate) will **not** help. This is the same lesson as the single-parameter
re-pin in Step 3, scaled up to the whole model: diagnosing the misfit as structural and then
re-parameterizing anyway is not progress. If the plateaued best model is **still not adequate**,
the fix is exclusively structural.

When you see it — or reach the same conclusion yourself (several distinct parameters pinned at
their bounds at once, multistarts converging cleanly to the same wall, the residual *shape*
unchanged across fits) — the next model **must change form**, not re-fit the same one.
Re-parameterizing, widening bounds, or re-authoring a near-duplicate is **disallowed as the
response to a structural verdict**. Confirm first with `get_identifiability` and the residual bias
/ time-trend queries (Step 5's `query_data`) to see what the current structure *can't* reproduce,
then reach for a generic structural move — these are **categories to consider, not a prescribed
answer for any particular problem**:

- **add a latent / unobserved state** — a species or driver the observation model doesn't measure
  but the dynamics need;
- **raise the interaction order** — linear → nonlinear, add a higher-order / saturating /
  cooperative term where a first-order rate law is too rigid;
- **add feedback / regulation** — a product or downstream species acting back on an earlier rate.

Pick the category the residual diagnostics point at, re-author the whole model with that
structural change via `set_model`, and re-read the fit. (If the diagnostics say a *single*
parameter is simply unidentifiable and no structural move removes it, that's Step 3's "leave it
pinned" case — not this one; this nudge is about a *plateau* / a *run of re-fits*, the signature of
a missing mechanism.)

**Rank models by `bic` (or `aic`), not by the χ² p-value.** `get_fit_summary` also reports
`reduced_chi2` and `chi2_gof_pvalue`, but these test goodness-of-fit against a *known* noise
level. When a problem's observable noise sigmas are **estimated** (they appear in the parameter
table with `estimate=1`), the fit inflates them to absorb the residuals, so `chi2` pins near the
number of data points by construction — `reduced_chi2 ≈ 1` and its p-value become uninformative
(often stuck at the noise floor no matter how wrong the mechanism is). Do **not** treat
`chi2_gof_pvalue ≥ 0.05` as the success criterion, restructure to raise it, or widen a sigma to
lower `reduced_chi2`. Judge whether a mechanism is missing from the residual **structure** — a
systematic per-observable bias or a residual time-trend (Step 5's SQL / `get_fit_diagnosis`),
read on the *raw* `residual` when a sigma is estimated — and rank the candidates that pass by
`bic`.

Inspect the result with:

- `get_fit_summary()` — just the metrics (`neg_log_likelihood`, `chi2`, …), no records —
  the cheapest way to check the score.
- `query_data(sql, max_rows?)` — a DuckDB query over a disposable copy of the data (any SQL
  is allowed and cannot affect the session). This is how you diagnose *where* the model fits
  poorly: there is no fixed menu of residual statistics —
  you write the breakdown you need. The tables are the PEtab design (`measurements`,
  `conditions`, `observables`), the `parameters` table, the `simulations` dataframe
  from the last fit, and the last fit's pyPESTO multistart (`optimization_starts`,
  `optimization_trace` — see below) — one row per measurement point with `measurement`, `simulation`,
  `residual` (natural units), `residual_weighted` (the noise-normalized χ² term), and
  `noise` (σ). Call `list_data_tables()` for the exact columns. Filter to
  `residual_weighted IS NOT NULL` for residual stats — a row can have a NULL residual (an
  unsimulated point, or a noise model that could not be evaluated), which would otherwise
  inflate a `count(*)`. Examples:
  - per-observable χ² (which observable fits worst):
    `SELECT observable_id, count(*) AS n, sum(residual_weighted*residual_weighted) AS chi2
     FROM simulations WHERE residual_weighted IS NOT NULL
     GROUP BY observable_id ORDER BY chi2 DESC`
  - the worst-fit points:
    `SELECT * FROM simulations WHERE residual_weighted IS NOT NULL
     ORDER BY abs(residual_weighted) DESC LIMIT 10`
  - a systematic-bias check (mean weighted residual per observable — far from 0 ⇒ a
    consistent over/under-shoot):
    `SELECT observable_id, avg(residual_weighted) AS bias, count(*) AS n
     FROM simulations WHERE residual_weighted IS NOT NULL GROUP BY observable_id`
  - a time-trend check per observable/condition (residuals drifting with time ⇒ wrong
    dynamics/asymptote):
    `SELECT observable_id, condition_id, corr(time, residual_weighted) AS trend, count(*) AS n
     FROM simulations WHERE residual_weighted IS NOT NULL
     GROUP BY observable_id, condition_id HAVING count(*) >= 4`

  **If a noise sigma is estimated, re-run your bias/trend checks on raw `residual`, not just
  `residual_weighted`.** A weighted residual is `residual / noise` — once a fit inflates its
  sigma, the weighted view looks adequate BY CONSTRUCTION regardless of whether the underlying
  fit is actually good, because dividing by a bigger sigma shrinks every weighted residual
  toward zero. The raw residual (natural units, independent of whatever sigma the optimizer
  picked) still shows the true bias/trend a bad mechanism leaves behind. Check
  `noise_sigma_report` (Step 3) after every fit; if a sigma is approaching its reference,
  treat `residual_weighted` looking clean as suspect and confirm against raw `residual` instead
  of taking chi2_gof_pvalue at face value.
  - a convergence check on the last fit's multistart (did the starts agree on the optimum?):
    `SELECT count(*) AS n_starts, min(fval) AS best, max(fval) AS worst,
            sum(CASE WHEN fval <= (SELECT min(fval) FROM optimization_starts) + 1e-3
                     THEN 1 ELSE 0 END) AS n_converged
     FROM optimization_starts`
  - the objective descent of the best start (`optimization_trace`, one row per iteration):
    `SELECT iteration, fval FROM optimization_trace WHERE start = 0 ORDER BY iteration`

  The `simulations`, `optimization_starts`, and `optimization_trace` tables are persisted
  with each fitted version and restored on `rollback_to`, so they always reflect the
  current version's fit.

A typical loop: make the edits for one coherent structural change → `fit` once to score it →
`query_data` to see where it still fits poorly → make the next justified edit(s) → `fit`
again → compare `is_new_best` / `improvement_vs_best_prior` across versions (or
`diff_versions`, which reports the neg-log-likelihood delta).

## Step 6 — provenance, versions, and recovery

- `list_versions()` — every snapshot with its kind (`initial`, `fit`, `rollback`,
  `reset`), label, timestamp, and fit metrics — rank versions by fit. A version is always a
  fit snapshot; edits do not appear here (they are `edit` events in the provenance log).
- `get_version(v)` / `diff_versions(from, to)` — inspect a version, or see the Antimony
  diff + parameter add/remove/change + the neg-log-likelihood delta between two.
- `get_provenance(limit)` — recent events, each with a `type` (`initial`, `edit`, `fit`,
  `fit_noop`, `rollback`, `reset`) — every edit, fit, no-op fit, rollback, and reset.
- `get_errors(limit)` — recent **rejected** operations and failed tool calls. When an edit
  returns `applied: false` or a `fit` returns `committed: false`, look here for the full
  context.
- `rollback_to(version, justification?)` — restore a prior version (appends a new,
  auditable version rather than erasing history).
- `restore_version(version, justification?)` — go **back** to an earlier model: the same
  as `rollback_to`, aptly named. Use it (not a `set_model` re-author) to return to a prior
  model — it reuses that version's stored fit instead of recompiling and refitting.
- `current_version()` — the HEAD version plus `has_unfit_edits` / `n_pending_edits` (whether
  the working model has edits not yet scored by a `fit`).
- `reset()` — revert model + parameters to the initial version (v0).

## Reading tool results

- **Model/parameter edits** (`set_model`, `change_parameter_scale`,
  `change_parameter_bounds`) return
  `{"applied": true, "operation": ..., "version": null, "reconcile": {...},
  "initial_conditions": [...], "references": ..., "justification": ..., "validation": ...,
  "warnings": [...], "parameter_diff": {...}, "pending_fit": true, "n_pending_edits": ...,
  "message": ...}` on success. An edit carries **no** fit metrics (no `neg_log_likelihood`,
  no `bic`, no `committed` key) and `version` is `null` — it has not been scored yet; call
  `fit` to score the assembled change. On a rejected edit they return `{"applied": false,
  "error": "...", "issues": [...]}` — the model is unchanged. Fix the reported issue (e.g. add
  a reference, correct the rate law, respect the fixed observables) and retry.
- **`fit`** returns `{"committed": true, "version": N, "warnings", "progress", …fit block…}`
  on a fit that snapshots a version; `{"committed": false, "no_change": true, "version",
  "duplicate_of_version", "message", "warnings", "neg_log_likelihood", "bic"}` when the working
  model is identical to the current version (the stored fit is reused); or `{"committed": false,
  "error": "..."}` when the model can't compile or has no free parameters — a fit failure
  **leaves your edits in place** so you can repair the model and fit again.
- **Other tools** return `{"ok": false, "error": "..."}` on a known failure.
- A rejected edit or failed `fit` is **not silent**: it is logged and surfaced via
  `get_errors()`.
- An edit is **atomic**: if reconcile/lint fails, the whole edit is reverted (nothing
  changes) and the error is surfaced. On a successful `fit`, the fit block's `fit_status` is
  `"fit"` (fitted) or `"simulate_only"` (you passed `n_starts=0`).

## A minimal end-to-end recipe

```
describe_problem()                         # orient
get_model("antimony"); get_parameters()    # see model + parameters
list_observables(); get_measurements()     # see what you're fitting

set_model("model m\n  compartment c = 1;\n  species S in c; species P in c;\n"
          "  S = S_init; P = P_init; S_init = 1.0; P_init = 0.0; k = 0.1;\n"
          "  v1: S -> P; k*S\nend",     # ILLUSTRATIVE placeholders — use YOUR model's
          justification="… mechanistic reason …")     # species/params, not S/P/k
                                        # -> applied (fit-free); new symbols added,
                                        #    pending_fit: true — no score yet
# … any further edits that belong to the same coherent change …
fit(justification="score the assembled change")  # -> committed + version N; result carries
                                                  #    neg_log_likelihood / bic / is_new_best
query_data("SELECT observable_id, sum(residual_weighted*residual_weighted) AS chi2 "
           "FROM simulations WHERE residual_weighted IS NOT NULL "
           "GROUP BY observable_id ORDER BY chi2 DESC")  # where is it worst?
list_versions(); diff_versions(0, 1)       # audit what changed
```

Assemble a coherent change through a justified `set_model` re-author (plus any parameter-space
edits), `fit` once to score it, read the fit before the next step, and let the version history
tell the story.
