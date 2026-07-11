---
name: petab-model-building
description: >-
  Use when interacting with the petab-mcp server to work a single PEtab v1
  parameter-estimation problem: read the fixed experimental design, edit the model
  through the constrained justified operations (add_reaction, change_rate_law,
  remove_reaction) — each commit automatically fits and simulates — then inspect the
  fit (inspect_fit / get_simulations), provenance, and versions. Explains the
  disciplined, auditable tool workflow, the rules each tool enforces, how to read tool
  results, and how to recover from rejected operations. Read this before calling any
  petab-mcp tool.
---

# Working a PEtab problem with petab-mcp

`petab-mcp` wraps **one** PEtab v1 parameter-estimation problem and turns model
building into a disciplined, auditable loop. The tools are deliberately narrow:
the experimental design is read-only, the model can only change through three
justified operations, and every committed change is versioned and logged. Work
*with* those constraints — do not look for a "replace the model" or "edit the
measurements" tool; there isn't one, by design.

For the exhaustive per-tool signature/return reference, read `tools.md` in this
skill (also available from the server as `get_skill("petab-model-building",
file="tools.md")`).

## The invariants (read first)

- **The design is fixed.** `measurements`, `conditions`, and `observables` are
  read-only. No tool mutates them; an edit that breaks a fixed observable or
  condition is *rejected*, not applied.
- **The model is authored in Antimony** and edited *only* through
  `add_reaction`, `change_rate_law`, `remove_reaction`. Antimony is the canonical
  source of truth; SBML is derived.
- **Every model operation must cite a justification and ≥1 reference.** The
  justification must be a non-empty string; references are validated against
  PubMed. Parameter-scale/bounds operations require a justification but references
  are optional.
- **Consistency is automatic.** After each operation the server *reconciles*
  SBML↔PEtab: new rate-law symbols become model parameters, new species get a
  **parametric initial condition** (`S = S_init`, with `S_init` added to the
  parameter table on a linear scale), and orphaned entities are pruned. The
  parameter table therefore always matches the model.
- **Everything is versioned.** Each committed operation snapshots a new version
  and appends a provenance event. Nothing is lost; you can diff, roll back, or
  reset.
- **Fitting is automatic.** Every commit fits the free parameters and simulates
  before snapshotting, so each version carries the fitted parameters, the fit metrics
  (neg_log_likelihood, chi2, improvement vs prior versions), and a simulation dataframe.
  You never call a fit or simulate tool by hand; parameter *values* are set only by this
  fit.

## Step 1 — orient yourself (read-only)

Always start by understanding the problem before changing it:

- `describe_problem()` — model size, free parameters, table counts, current
  version, and whether a batch is open.
- `get_model("antimony")` — the canonical model text. Use `get_model("sbml")` for
  SBML, or `get_model_structure()` for a structured species/parameters/reactions view.
- `list_observables()`, `list_conditions()`, `get_measurements(condition_id?,
  observable_id?)` — the fixed data you are fitting. These constrain every edit.
- `get_parameters()` — the parameter table (id, bounds, nominalValue, estimate,
  scale, priors).
- `validate_problem()` — lint the current problem and see any outstanding issues.

## Step 2 — edit the model (constrained + justified)

Each of these applies the edit to a *copy* of the model, reconciles SBML↔PEtab,
lints the candidate, and **only commits if the result is valid**. On failure the
model and parameter table are reverted and nothing is versioned.

### `add_reaction(reaction, justification, references)`

`reaction` is a single Antimony reaction line:

```
J1: A + B -> C; k_f*A*B
```

- An `id:` prefix is optional — omit it and the server assigns `J1`, `J2`, …
- Use `->` (or `=>`) between reactants and products. Stoichiometry is written as a
  leading number: `2 A -> B`.
- A **rate law after `;` is required** for a new reaction.
- **Reactions are irreversible** (one kinetic law). Model a reversible step
  A⇌B either as a single net rate law `J: A -> B; k_f*A - k_r*B`, or as two
  separate reactions.
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
- **references**: a list of ≥1 citation strings. Accepted forms include PubMed IDs
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
`neg_log_likelihood` and `chi2` (**lower is better**) plus improvement metrics
(`objective_before`, `self_improvement`, `improvement_vs_best_prior`, `is_new_best`), so
competing versions can be ranked by objective. Pass `n_starts` on a commit to fit harder,
or `n_starts=0` to simulate without fitting.

Inspect the result with:

- `get_simulations()` — the stored simulation dataframe: fit metrics + per-point records
  (measurement, simulation, residual, noise-weighted residual, noise).
- `inspect_fit(top_n)` — residual statistics overall and across each cross section
  (observables, experiments/conditions, timepoints), the `top_n` worst-fit points, and
  systematic-bias flags — so you see *where* the model fits poorly. It reads the stored
  dataframe (no re-simulation).

A typical loop: edit the model (it auto-fits) → `inspect_fit` to see where it still fits
poorly → make the next justified edit → compare `is_new_best` / `improvement_vs_best_prior`
across versions.

## Step 6 — provenance, versions, and recovery

- `list_versions()` — every snapshot with its kind (`initial`, `alteration`,
  `batch`, `training`, `rollback`, `reset`), label, and timestamp.
- `get_version(v)` / `diff_versions(from, to)` — inspect a version, or see the
  Antimony diff + parameter add/remove/change between two.
- `get_provenance(limit)` — recent events (alterations, training, rollbacks).
- `get_errors(limit)` — recent **rejected** operations and failed tool calls. When
  an operation returns `committed: false`, look here for the full context.
- `rollback_to(version, justification?)` — restore a prior version (appends a new,
  auditable version rather than erasing history).
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

add_reaction("J: B -> C; k3*B",
             justification="B is enzymatically converted to C (…)",
             references=["PMID:12345678"]) # -> committed + auto-fitted; k3 + C_init added
                                           #    result carries neg_log_likelihood / is_new_best
inspect_fit()                              # where does it still fit poorly?
get_simulations()                          # the simulated-vs-measured dataframe
list_versions(); diff_versions(0, 1)       # audit what changed
```

Keep edits small and justified, read the fit after each change, and let the version
history tell the story.
