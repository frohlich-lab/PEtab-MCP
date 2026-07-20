---
name: petab-model-building
description: >-
  Use when interacting with the petab-mcp server to work a single PEtab v1
  parameter-estimation problem: read the fixed experimental design, edit the model
  through the constrained justified operations (add_reaction, change_rate_law,
  remove_reaction) — each commit automatically fits and simulates — then inspect the
  fit by querying the data with SQL (query_data over the PEtab tables + the simulation
  dataframe/residuals), provenance, and versions. Explains the
  disciplined, auditable tool workflow, the rules each tool enforces, how to read tool
  results, and how to recover from rejected operations. Read this before calling any
  petab-mcp tool.
---

# Working a PEtab problem with petab-mcp

`petab-mcp` wraps **one** PEtab v1 parameter-estimation problem and turns model
building into a disciplined, auditable loop. The tools are deliberately narrow:
the experimental design is read-only, the model changes only through `set_model`
(author or replace the whole model in Antimony) or the constrained justified edits
(`add_reaction` / `change_rate_law` / `remove_reaction`), and every committed change is
versioned and logged. Work *with* those constraints — do not look for an "edit the
measurements" tool; there isn't one, by design. To write the Antimony that `set_model`
and the edits take, see the companion `antimony-authoring` skill.

For the exhaustive per-tool signature/return reference, read `tools.md` in this
skill (also available from the server as `get_skill("petab-model-building",
file="tools.md")`).

## The invariants (read first)

- **The design is fixed.** `measurements`, `conditions`, and `observables` are
  read-only. No tool mutates them; an edit that breaks a fixed observable or
  condition is *rejected*, not applied.
- **The model is authored in Antimony with `set_model`.** If the problem ships *without* a
  model (`describe_problem` reports `model_initialized: false`), author the whole model from
  Antimony with `set_model`. `set_model` is the single whole-model operation — it also
  re-authors an existing model wholesale, the clean way to make a coupled/symmetric or
  multi-reaction structural change in one coherent step; for smaller, local changes refine
  with `add_reaction`, `change_rate_law`, `remove_reaction`. Antimony is the canonical source
  of truth; SBML is derived.
- **Every model operation must cite a justification.** The justification must be a
  non-empty string. References are **optional** for all operations; any you provide are
  validated against PubMed.
- **Consistency is automatic.** After each operation the server *reconciles*
  SBML↔PEtab: new rate-law symbols become estimated model parameters (log10 scale by default,
  with bounds bracketing the nominal value), new species get a **parametric initial condition**
  (`S = S_init`, with `S_init` added to the parameter table on a linear scale, *fixed*), and
  orphaned entities are pruned. The parameter table therefore always matches the model.
  - **Initial conditions are FIXED by default** (whether they are estimated is a fixed
    per-scenario setting, not something you switch in-loop). Every commit's **`initial_conditions`**
    block reports each species' t=0 status (`fixed` / `estimated` / `condition_table`, and whether
    the declared t=0 is used) — read it rather than re-authoring to try to change an IC. *How* ICs
    reconcile is covered in the `antimony-authoring` skill.
  - **Exponents are special.** A new symbol used as an **exponent** — a Hill / cooperativity
    coefficient, e.g. `n_hill` in `K^n_hill/(K^n_hill + x^n_hill)` — is created on a **linear**
    scale with **tight bounds** (≈1–8), not the default log10. This is deliberate: on a wide
    log10 scale an exponent fits to a huge value (a near-step-function) that makes the ODE stiff
    and the fit plateau. Keep it that way — if you set an exponent's scale/bounds by hand, use
    `lin` with a tight range; never put a Hill coefficient on log10 with wide bounds.
  - **Noise sigmas have a reference, not a free run.** Every commit's `noise_sigma_report`
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
- **Everything is versioned.** Each committed operation snapshots a new version
  and appends a provenance event. Nothing is lost; you can diff, roll back, or
  reset.
- **Fitting is automatic.** Every commit fits the free parameters and simulates
  before snapshotting, so each version carries the fitted parameters, the fit metrics
  (neg_log_likelihood, chi2, improvement vs prior versions), and a simulation dataframe.
  You never call a fit or simulate tool by hand; parameter *values* are set only by this
  fit. Fits are cached by estimation-problem identity, so re-committing a model+bounds you
  already fit reuses the stored fit (skipping the optimizer) — transparently, with identical
  results.

## Step 1 — orient yourself (read-only)

Always start by understanding the problem before changing it:

- `describe_problem()` — model size, free parameters, table counts, current
  version, and whether a batch is open.
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

## Step 2 — edit the model (constrained + justified)

Each of these applies the edit to a *copy* of the model, reconciles SBML↔PEtab,
lints the candidate, and **only commits if the result is valid**. On failure the
model and parameter table are reverted and nothing is versioned.

### `set_model(antimony, justification, references?)` — author the whole model

`set_model` is the single whole-model operation: it **builds** the first model when the
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

It reconciles, lints, and fits exactly like the incremental edits, and reverts on failure so
a bad (re-)author changes nothing. Reach for it for the initial build and for any change that
is **structural and coupled** — the same rate-law form across several reactions, a symmetric
set that must stay identical, or adding/removing several reactions at once — where a sequence
of incremental edits would leave the model inconsistent between steps. Because the whole model
is replaced, the result **warns** if the re-author no longer declares a previously-*estimated*
parameter (its fitted value is dropped); fixed noise / initial-condition parameters and
anything the observable, noise, or condition tables reference are preserved. Prefer the
incremental edits below for a genuinely local change (they keep the history fine-grained);
prefer `set_model` for a whole-structure move.

`set_model` remembers the structures it has already seen (issue #59): a genuinely new
structure always commits, but re-authoring the **current** model is a no-op (`no_change:
true` — no fit, no new version), re-authoring an **earlier** model commits with a warning to
use `restore_version(n)` instead, and an identical structure re-authored repeatedly is
**refused** to break an unproductive loop. To go **back** to a prior model, use
`restore_version(n)` — never re-author it (a re-author recompiles and refits to the same
result and appends a duplicate version).

### `add_reaction(reaction, justification, references)`

`reaction` is a single Antimony reaction line (species/parameter names below are
**illustrative placeholders** — use the actual entities from *your* model, not these):

```
J1: S1 + S2 -> S3; k_f*S1*S2
```

- An `id:` prefix is optional — omit it and the server assigns `J1`, `J2`, …
- Use `->` between reactants and products. Stoichiometry is written as a
  leading number: `2 S1 -> S2`.
- A **rate law after `;` is required** for a new reaction.
- **Prefer reversible reactions.** A reaction carries one kinetic law and the dynamics come
  entirely from it (the arrow adds no flux — reactions are created `reversible=false`), so model
  a reversible step X⇌Y as a **single** reaction with a net `forward - reverse` rate law
  (`J: X -> Y; k_f*X - k_r*Y`) rather than two opposing one-way reactions. Use a forward-only law
  (`k_f*X`) only for a genuinely irreversible step — synthesis, degradation, an effectively
  one-way catalytic step. (See the `antimony-authoring` skill.)
- New species mentioned in the reaction are created automatically (with a
  parametric initial condition). New symbols in the rate law that are not species
  become **parameters** — after adding, call `get_parameters()` and set sensible
  `nominalValue`/bounds/`estimate` for them (see Step 3).

### `change_rate_law(reaction_id, rate_law, justification, references)`

Replace one reaction's kinetic law. New symbols become parameters and are
reconciled just like `add_reaction`.

### `remove_reaction(reaction_id, justification, references)`

Delete a reaction and prune any species/parameters left orphaned by the removal.

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

Parameter **values** (`nominalValue`) are set only by the automatic fit on commit —
there is no manual value-editing tool. What you *can* change, through versioned, justified
operations, is a parameter's search space:

- `change_parameter_scale(parameter_id, scale, justification, references?)` —
  `scale` is `"lin"`, `"log"`, or `"log10"`. Switching to `log`/`log10` requires the
  parameter's `lowerBound` and `nominalValue` to be **> 0**, or the operation is rejected.
- `change_parameter_bounds(parameter_id, lower_bound, upper_bound, justification,
  references?)`.

Both are commits, so they re-fit and re-simulate like any model edit — reach for them when
a fit lands on a bound or a scale is inappropriate. Read the current table with
`get_parameters()`; use `sync_parameters("report"|"add")` to reconcile the table with the
model (operations already reconcile, so this is rarely needed).

**A parameter that keeps re-pinning to whatever bound you set is not a bounds problem.** Every
commit's `parameters_at_bound` (also in `get_fit_summary`) flags any fitted parameter within
~1% of its lower/upper bound — check it after every commit, not just when a fit looks off.
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

1. **Remove it** — drop the reaction/term it belongs to (`remove_reaction` / `change_rate_law`)
   when that parameter is what makes its mechanism redundant.
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
see what the current structure can't reproduce, then add the missing reaction rather than stretch
a bound. The rule is the same in every case: never answer a re-pin with another bounds widening —
make a **structural** edit (`add_reaction` / `change_rate_law` / `set_model`) or leave the
parameter be. The harness enforces this as a backstop: `change_parameter_bounds` **rejects a third
widening** of a still-pinned parameter whose previous widening bought a negligible BIC gain.

If the parameter pinned at its bound is a **noise sigma**, don't even try widening it —
`change_parameter_bounds` will reject a widen past its data-driven cap outright (see the
invariant above). A sigma at that cap means the current mechanism's residuals are already as
large as the data's own variability; the fix is exclusively a model-structure one.

## Step 4 — batching (group edits into one version)

To land several related edits as a **single** version:

1. `begin_batch(label)` — start staging. Subsequent operations are applied and
   logged but not individually versioned, and **not fitted** — the fit is deferred.
2. Run the operations (`add_reaction`, `change_rate_law`, …).
3. `commit_batch(label)` — fit + simulate the staged result **once**, then snapshot
   everything as one version, **or** `abort_batch()` — discard the staged edits and
   restore the pre-batch version.

While a batch is open, `rollback_to` and `reset` are blocked — commit or abort first.
Batching is also how you avoid re-fitting after every intermediate edit: a mechanism
spanning several reactions is fitted just once, on `commit_batch`.

## Step 5 — read the fit (fitting is automatic)

You never call a fit or simulate tool: **every commit** (a model edit, a parameter-scale/
bounds change, or `commit_batch`) fits the free parameters and simulates before
snapshotting. The commit result and the new version carry the fit metrics —
`neg_log_likelihood` and `chi2` (**lower is better**), the complexity-penalised `bic`, plus
improvement metrics (`objective_before`, `self_improvement`, `improvement_vs_best_prior`,
`is_new_best`), so competing versions can be ranked. `is_new_best` and `improvement_vs_best_prior`
track **BIC** (not raw nll) against your best model so far, and `is_new_best` is gated on a
**minimum effect size** (ΔBIC ≥ 2, the "positive evidence" bar): a fit only counts as a new best
when it meaningfully lowers BIC — so a 1e-6 twitch, or an nll drop bought by an extra parameter
that actually *raises* BIC, is not mistaken for progress. Pass `n_starts` on a commit to fit
harder, or `n_starts=0` to simulate without fitting.

The commit result also carries a `progress` block — `non_improving_streak` (fitted ops since
the last real new best), the lowest-BIC `best_version` seen so far, and
`bic_improvement_over_last_n` (best-BIC gained over the last few ops). When the score stops
moving — the streak reaches its cap, or the best model is already adequate and the BIC has
flattened — the run is stopped and the **best-BIC version is scored**, not whatever happened
to be committed last. So don't keep editing a fit that has reached the noise floor; finish.

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
  from the last commit, and the last fit's pyPESTO multistart (`optimization_starts`,
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
  `noise_sigma_report` (Step 3) after every commit; if a sigma is approaching its reference,
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

A typical loop: edit the model (it auto-fits) → `query_data` to see where it still fits
poorly → make the next justified edit → compare `is_new_best` / `improvement_vs_best_prior`
across versions (or `diff_versions`, which reports the neg-log-likelihood delta).

## Step 6 — provenance, versions, and recovery

- `list_versions()` — every snapshot with its kind (`initial`, `alteration`,
  `batch`, `rollback`, `reset`), label, timestamp, and fit metrics — rank versions by fit.
- `get_version(v)` / `diff_versions(from, to)` — inspect a version, or see the Antimony
  diff + parameter add/remove/change + the neg-log-likelihood delta between two.
- `get_provenance(limit)` — recent events (alterations, batches, rollbacks, resets).
- `get_errors(limit)` — recent **rejected** operations and failed tool calls. When
  an operation returns `committed: false`, look here for the full context.
- `rollback_to(version, justification?)` — restore a prior version (appends a new,
  auditable version rather than erasing history).
- `restore_version(version, justification?)` — go **back** to an earlier model: the same
  as `rollback_to`, aptly named. Use it (not a `set_model` re-author) to return to a prior
  model — it reuses that version's stored fit instead of recompiling and refitting.
- `current_version()` — the HEAD version and batch state.
- `reset()` — revert model + parameters to the initial version (v0).

## Reading tool results

- **Model/parameter operations** return `{"committed": true, "version": N,
  "fit_status": ..., "neg_log_likelihood": ..., "improvement": {...}, ...}` on success —
  the fit metrics come back with the commit. On a rejected edit they return
  `{"committed": false, "error": "...", "issues": [...]}` — the model is unchanged. Fix the
  reported issue (e.g. add a reference, correct the rate law, respect the fixed
  observables) and retry.
- **Other tools** return `{"ok": false, "error": "..."}` on a known failure.
- A rejected operation is **not silent**: it is logged and surfaced via
  `get_errors()`.
- A commit is **atomic**: if the fit or simulation fails, the whole edit is rejected
  (nothing changes) and the error is surfaced — fix the model so it simulates and retry.
  On success, `fit_status` is `"fit"` (fitted) or `"simulate_only"` (you passed
  `n_starts=0`).

## A minimal end-to-end recipe

```
describe_problem()                         # orient
get_model("antimony"); get_parameters()    # see model + parameters
list_observables(); get_measurements()     # see what you're fitting

add_reaction("v1: S -> P; k*S",            # ILLUSTRATIVE placeholders — use YOUR model's
             justification="… mechanistic reason …",   # species/params, not S/P/k
             references=["PMID:XXXXXXXX"]) # -> committed + auto-fitted; new symbols added
                                           #    result carries neg_log_likelihood / is_new_best
query_data("SELECT observable_id, sum(residual_weighted*residual_weighted) AS chi2 "
           "FROM simulations WHERE residual_weighted IS NOT NULL "
           "GROUP BY observable_id ORDER BY chi2 DESC")  # where is it worst?
list_versions(); diff_versions(0, 1)       # audit what changed
```

Keep edits small and justified, read the fit after each change, and let the version
history tell the story.
