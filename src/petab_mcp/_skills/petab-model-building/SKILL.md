---
name: petab-model-building
description: >-
  Use when interacting with the petab-mcp server to work a single PEtab v1
  parameter-estimation problem: read the fixed experimental design, edit the model
  through the constrained justified operations (add_reaction, change_rate_law,
  remove_reaction), tune parameters, score/optimize against the fixed data, and
  inspect provenance and versions. Explains the disciplined, auditable tool workflow,
  the rules each tool enforces, how to read tool results, and how to recover from
  rejected operations. Read this before calling any petab-mcp tool.
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

## Step 1 — orient yourself (read-only)

Always start by understanding the problem before changing it:

- `describe_problem()` — model size, free parameters, table counts, current
  version, whether a batch is open, and `amici_available` (whether simulation/
  optimization tools will work).
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

Two distinct paths, for two purposes:

- **Versioned, justified structural intent** — use the operations. These snapshot
  a version and require a justification:
  - `change_parameter_scale(parameter_id, scale, justification, references?)` —
    `scale` is `"lin"`, `"log"`, or `"log10"`. Switching to `log`/`log10` requires
    the parameter's `lowerBound` and `nominalValue` to be **> 0**, or the operation
    is rejected.
  - `change_parameter_bounds(parameter_id, lower_bound, upper_bound,
    justification, references?)`.
- **Quick, non-versioned tuning** — `update_parameters(updates)` where each update
  is `{"id": ..., "nominalValue"?, "lowerBound"?, "upperBound"?, "estimate"?,
  "parameterScale"?}`. This is the fast path for setting nominal values, toggling
  `estimate`, or adjusting bounds during exploration. It is transactional (a bad
  update leaves the table unchanged) but does **not** create a version.
- `sync_parameters(policy)` — reconcile the table with the model. `"report"`
  (default) lists model parameters missing from the table; `"add"` inserts them
  with default bounds and `estimate=1`. Operations already reconcile, so you rarely
  need this, but it's useful after low-level tuning.

Rule of thumb: use the **operations** when the change reflects a modelling
decision worth auditing; use `update_parameters` for throwaway tuning before an
`optimize` run.

## Step 4 — batching (group edits into one version)

To land several related edits as a **single** version:

1. `begin_batch(label)` — start staging. Subsequent operations are applied and
   logged but not individually versioned.
2. Run the operations (`add_reaction`, `change_rate_law`, …).
3. `commit_batch(label)` — snapshot everything as one version, **or**
   `abort_batch()` — discard the staged edits and restore the pre-batch version.

While a batch is open, `optimize`, `rollback_to`, and `reset` are blocked — commit
or abort first. Use a batch when a mechanism spans multiple reactions and you want
the version history to read as one coherent change.

## Step 5 — score and fit

These need the simulation stack (`amici`, and `pypesto` for `optimize`). Check
`describe_problem()["amici_available"]` first; if false, install the extras
(`pip install 'petab-mcp[amici,optimize]'`).

- `simulate()` — simulate every experiment at the current parameters.
- `evaluate_objective()` — score current model/parameters vs the fixed data
  (returns a negative log-likelihood; **lower is better**).
- `optimize(n_starts, seed)` — multi-start estimation with pyPESTO. It writes the
  best fit back into the table and records a **training version** with improvement
  metrics: `objective_before`, `self_improvement`, `improvement_vs_best_prior`,
  and `is_new_best`. Because each `optimize` is versioned and scored, competing
  model versions can be ranked by objective.

A typical fit loop: edit the model → `update_parameters` to set reasonable starts/
bounds → `optimize` → compare `is_new_best` / `improvement_vs_best_prior`.

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

- **Model/parameter operations** return `{"committed": true, "version": N, ...}`
  on success. On a rejected edit they return `{"committed": false, "error": "...",
  "issues": [...]}` — the model is unchanged. Fix the reported issue (e.g. add a
  reference, correct the rate law, respect the fixed observables) and retry.
- **Other tools** return `{"ok": false, "error": "..."}` on a known failure.
- A rejected operation is **not silent**: it is logged and surfaced via
  `get_errors()`.
- If a tool reports a missing optional dependency, it needs `amici`/`pypesto` —
  install the extras and retry; other tools keep working without them.

## A minimal end-to-end recipe

```
describe_problem()                         # orient; check amici_available
get_model("antimony"); get_parameters()    # see model + parameters
list_observables(); get_measurements()     # see what you're fitting

add_reaction("J: B -> C; k3*B",
             justification="B is enzymatically converted to C (…)",
             references=["PMID:12345678"]) # -> committed, new version; k3 + C_init added
get_parameters()                           # set nominal/bounds for k3, C_init
update_parameters([{"id":"k3","nominalValue":0.1,"lowerBound":1e-4,"upperBound":10}])

evaluate_objective()                       # baseline score
optimize(n_starts=20, seed=0)              # fit; check is_new_best
list_versions(); diff_versions(0, 2)       # audit what changed
```

Keep edits small and justified, score after each meaningful change, and let the
version history tell the story.
