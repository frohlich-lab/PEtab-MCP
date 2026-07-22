# petab-mcp tool & resource catalog

The complete surface of the `petab-mcp` server. See `SKILL.md` for the workflow
and the rules behind these tools.

## Read-only — the fixed design

| Tool | Signature | Returns / notes |
| --- | --- | --- |
| `describe_problem` | `()` | Summary: `model_id`, `model_initialized` (false ⇒ author the model with `set_model` first), species/reaction/parameter counts, `free_parameters`, observables, condition/measurement counts, `current_version`, `has_unfit_edits`, `n_pending_edits`, `fixed_tables`. |
| `list_observables` | `()` | The fixed observables (formulas, noise models). |
| `list_conditions` | `()` | The fixed conditions and the model changes they apply. |
| `get_measurements` | `(condition_id?, observable_id?, summarize=True)` | Compact summary of the measurements by default (per observable×condition: count, time span, value min/max/mean, a few preview points); pass `summarize=False` for the full measurement rows. |
| `get_model` | `(format="antimony")` | Model as text. `format` is `"antimony"` (canonical) or `"sbml"`. |
| `get_model_structure` | `()` | Structured species / parameters / reactions view. |
| `validate_problem` | `()` | Lint the current problem; reports `valid` + `issues` (includes the role checks: an observation-model factor reused in the mechanism, a shipped-fixed constant removed or made estimated, and a condition-table target the model never uses). |

## Model operations (justification required, references optional)

Each applies to a model copy, reconciles SBML↔PEtab, lints, then commits the edit to the
working model — or reverts on a lint failure. Edits are **fit-free**: they do NOT fit, do NOT
simulate, and do NOT create a version (call `fit` to score the assembled change). Returns
`{applied: true, operation, version: null, reconcile, initial_conditions, references,
justification, validation, warnings, parameter_diff, pending_fit: true, n_pending_edits,
message}` on success — **no** fit metrics (no `neg_log_likelihood`, no `bic`, no `committed`
key) — or `{applied: false, error, issues?}` on rejection. `warnings` carries advisory notes
(e.g. an added unobserved species). `initial_conditions` is a per-species list —
`{species, ic_parameter, declared_t0, estimated, set_by_condition, t0_source
(fixed|estimated|condition_table), declared_t0_used, newly_added, note}`. **Initial conditions are
ESTIMATED by default.** You do NOT set an IC's estimated/fixed status (there is no tool for that —
see below); reconcile derives it from the value you declare in the model: a species declared at
exactly **0** (empty pool) or exactly **1** (a unit-normalisation / conserved-total convention) is
FIXED; a species whose initial is set by a **condition-table column** is driven per-condition
(`t0_source: condition_table`); any other positive value is ESTIMATED (`t0_source: estimated`,
counted in *k*). A parameter the problem ships FIXED and you reference as an IC stays fixed. Read
this block to see how each species' IC was classified rather than re-authoring.
(`parameters_at_bound` and `noise_sigma_report` are fit artifacts and come back on the `fit`
result, not on an edit — see `get_fit_summary` below.) A model that reuses a fixed
observable-scaling factor or noise sigma (anything resolved through the measurement table's
`observableParameters`/`noiseParameters` or an observable/noise formula) inside a reaction,
rule, or initial assignment is a hard **rejection** (`check: "observation_model_reuse"` in
`issues`) — introduce a dedicated parameter instead. A parameter the problem ships **fixed**
(`estimate=0`) that is a mechanistic/structural constant — a conservation total, a known rate —
must not be **removed** (`check: "fixed_constant_removed"`) or turned **estimated**
(`check: "fixed_constant_made_estimated"`): dropping it changes the problem, and estimating it
lets its value drift and adds to *k*, inflating BIC. There is no tool to toggle a parameter
between fixed and estimated, so keep such constants declared and build on them as the problem
ships them; observation-model factors and initial conditions are exempt (pruned normally with
their species / their own estimation paths). Finally, a **condition-table target the model never
uses** is a hard **rejection** (`check: "unused_condition_target"`): a condition column sets a
model quantity per condition, so if the model never references it the conditions don't perturb the
model — use each condition quantity in the mechanism.

| Tool | Signature | Notes |
| --- | --- | --- |
| `set_model` | `(antimony, justification, references?)` | Author the **whole** model from an Antimony string — the single model editor. **Builds** the first model when none is initialized (`describe_problem` → `model_initialized: false`) and **re-authors / replaces** an existing one. Every structural change — adding/removing a reaction, changing a rate law — is made this way, in one coherent step. New species → parametric IC; new rate-law symbols → parameters (reconciled into the table). Reconciles + lints, **fit-free**, reverts on a lint failure. Warns if a previously-**estimated** parameter is dropped (fixed noise/IC params + table-referenced symbols are preserved). To go back to a prior model use `restore_version`, not a `set_model` re-author (`fit` de-duplicates a structure already in the history — see `fit` below). |

## Parameter operations (justification required, references optional)

**Fit-free** edits — same pipeline / result shape as model operations (they reconcile + lint
but do NOT fit; call `fit` to re-score). Parameter **values** (`nominalValue`) are set only by
the fit; these change the search space.

| Tool | Signature | Notes |
| --- | --- | --- |
| `change_parameter_scale` | `(parameter_id, scale, justification, references?)` | An edit — does not fit; call `fit` to re-score. `scale` ∈ `{"lin","log","log10"}`. `log`/`log10` require `lowerBound` and `nominalValue` > 0. |
| `change_parameter_bounds` | `(parameter_id, lower_bound, upper_bound, justification, references?)` | An edit — does not fit; call `fit` to re-score. Set new bounds. REJECTED for an ESTIMATED noise-sigma parameter if `upper_bound` exceeds its data-driven reference (a fraction of the observable's raw measurement std). Also REJECTED as a **third widening** of a still-pinned parameter whose last widening bought a negligible BIC gain — a parameter re-pinning at a bound with a flat/collinear FIM direction (see `get_identifiability`) is unidentifiable, not under-explored: remove its reaction or use its limiting (irreversible/instantaneous) form instead of widening again; if neither is possible, leave it pinned (don't hard-code a value — there is no value/estimate tool, and a literal is a magic number). |

## Parameters (read-only + reconcile)

| Tool | Signature | Notes |
| --- | --- | --- |
| `get_parameters` | `()` | The parameter table: id, bounds, `nominalValue` (fit-only), `estimate`, scale, priors, plus `role` (`kinetic`/`noise_sigma`/`observable_scaling`/`initial_condition`/`exponent`/`constant`) and `status` (`estimated`/`fixed`). |
| `sync_parameters` | `(policy="report")` | Reconcile table with model. `"report"` lists model params missing from the table and orphaned table rows (no change); `"add"` runs the full reconcile — adds missing params (with IC/exponent/scale handling) and removes orphaned rows — the same reconcile every edit runs. |

## Fit + simulation

`fit` is the **only** tool that fits and simulates — edits are fit-free. Make the edits that
belong to one coherent structural change, then call `fit` once to score them and snapshot a
version. Read the score with `get_fit_summary`, and diagnose the fit by querying the data with
SQL:

| Tool | Signature | Notes |
| --- | --- | --- |
| `fit` | `(n_starts?, justification?)` | Fit the current working model's free parameters (pyPESTO multistart), simulate, and snapshot a **single** version carrying the fitted parameters, the full fit block (`fit_status`, `neg_log_likelihood`, `chi2`, `n_data_points`, `n_estimated`, `fitted_parameters`, `parameter_scales`, `improvement`, `parameters_at_bound`, `noise_sigma_report`, `bic`, `aic`, `reduced_chi2`, `chi2_gof_pvalue`), and the simulation dataframe. `n_starts` overrides the server default multistart count (`0` = simulate without optimising). Returns `{committed: true, version: N, warnings, progress, …fit block…}`. **De-dup:** if the working model is byte-identical to the current version, skips the refit and returns the stored metrics — `{committed: false, no_change: true, version, duplicate_of_version, message, warnings, neg_log_likelihood, bic}`; a structure already elsewhere in the history is fit but **warns** to use `restore_version(n)`; the same structure fit too many times is **refused**. A model that can't compile or has no free parameters returns `{committed: false, error}` and **leaves the working model in place** to repair — a fit failure does NOT revert your edits. |
| `get_fit_summary` | `()` | Just the metrics of the last fit — `status`, `neg_log_likelihood`, `chi2`, `n_data_points`, `fitted_parameters`, `version` — **no records**. The cheapest score check. `parameters_at_bound` flags any fitted parameter within ~1% of its lower/upper bound (its own estimation scale) — `[]` when none. Widen at most once; the SAME parameter re-pinning after a widening means it is unidentifiable (a flat/collinear FIM direction — confirm with `get_identifiability`), not under-explored, so don't widen or flip its scale again — remove its reaction or use its limiting (irreversible/instantaneous) form; if neither is possible, leave it pinned rather than hard-code a value. |
| `list_data_tables` | `()` | The SQL tables `query_data` exposes: name, description, columns, row count. |
| `query_data` | `(sql, max_rows=50)` | Read-only DuckDB query over the tables below. Returns `{columns, rows, row_count, truncated}` (capped at `max_rows`; raise it explicitly for a full dump). |
| `get_fitting_log` | `(tail_lines=200)` | The tail of the pyPESTO/Fides optimizer log (`pypesto.log`): multistart spread, convergence, parameters pinned at bounds, ill-conditioning — the numerical view of the fit. `0` returns the whole file. |
| `get_fit_diagnosis` | `(worst_n=8)` | Compact, size-bounded residual diagnosis (fixed aggregations, not open SQL): one row per observable with weighted `chi2`, `mean_weighted_residual` (bias), and worst \|residual\|, plus the `worst_n` largest-residual points. The cheap way to locate a poor fit without pulling per-point data. |
| `get_identifiability` | `(top_n=8)` | Parameter-identifiability diagnosis (FIM at the optimum + multistart spread): a ranked `candidates` list (`at_bound` / `poorly_constrained` / `collinear` / `flat_direction` / `multistart_unstable`) — the basis for SIMPLIFYING an over-parameterised model. Needs a fitted model. |
| `get_run_timing` | `()` | Cumulative server-side timing: `{compile_seconds, fit_seconds, n_compiles, n_fits}` — the AMICI compile vs pyPESTO optimization split (cache hits add nothing). |

### `query_data` tables

| Table | Columns (key ones) | Notes |
| --- | --- | --- |
| `measurements` | `observableId`, `simulationConditionId`, `time`, `measurement` | The fixed measurement data. |
| `conditions` | `conditionId`, … | The fixed conditions. |
| `observables` | `observableId`, `observableFormula`, `noiseFormula` | The fixed observables. |
| `parameters` | `parameterId`, `lowerBound`, `upperBound`, `nominalValue`, `estimate`, `parameterScale`, priors | Parameter table (`nominalValue` fit-only). |
| `simulations` | `observable_id`, `condition_id`, `time`, `measurement`, `simulation`, `residual`, `residual_weighted`, `noise` | Per-point dataframe from the last fit. `residual` = `measurement - simulation` (natural units); `residual_weighted` = `(m - s)/σ` (the χ² term); `noise` = σ. Generated on demand at cold start; empty until the first fit. |
| `optimization_starts` | `start`, `id`, `fval`, `time`, `n_fval`, `n_grad`, `n_hess`, `exitflag`, `message` | pyPESTO multistart results from the last fit — one row per start (`start`=0 is best). Empty on a simulate-only pass / before the first fit. |
| `optimization_trace` | `start`, `iteration`, `fval`, `time` | pyPESTO per-iteration trace from the last fit — the objective over each start's run. |

The `simulations` / `optimization_*` tables are persisted per fitted version (and the pyPESTO
result is stored as `pypesto_result.hdf5`), so `rollback_to` restores a version's fit
artifacts, not only its model.

Residual statistics, per-observable/-condition/-time breakdowns, worst-fit points, and bias
or time-trend diagnostics are all just SQL. Filter to `residual_weighted IS NOT NULL` for
residual stats — a row can have a NULL residual (unsimulated point / un-evaluable noise
model), which would inflate a `count(*)`. E.g.
`SELECT observable_id, count(*) AS n, sum(residual_weighted*residual_weighted) AS chi2 FROM simulations WHERE residual_weighted IS NOT NULL GROUP BY observable_id ORDER BY chi2 DESC`
or `SELECT * FROM simulations WHERE residual_weighted IS NOT NULL ORDER BY abs(residual_weighted) DESC LIMIT 10`.

`query_data` runs against a disposable in-memory **copy** of the tables, so any SQL is
allowed and cannot affect the session; host file/network access (`read_csv`, `ATTACH`,
`COPY`, `INSTALL`/`LOAD`, …) is disabled.

## Provenance, versions & errors

| Tool | Signature | Notes |
| --- | --- | --- |
| `get_provenance` | `(limit=50)` | Recent events, each with a `type` ∈ `{initial, edit, fit, fit_noop, rollback, reset}` — every edit, fit, no-op fit, rollback, and reset. |
| `get_errors` | `(limit=50)` | Recent rejected operations / failed tool calls (an edit `applied: false` or a `fit` `committed: false`). |
| `list_versions` | `()` | All versions with kind (`initial`/`fit`/`rollback`/`reset`) / label / timestamp **and fit metrics** — rank versions by fit. A version is always a fit snapshot; edits are not versions (they are `edit` events in the provenance log). |
| `get_version` | `(version)` | A version's Antimony model + parameter table. |
| `diff_versions` | `(from_version, to_version)` | Antimony unified diff + parameter add/remove/change + a `fit` block (each version's nll/chi2 and the neg-log-likelihood delta). |
| `rollback_to` | `(version, justification?)` | Restore a prior version as a new version. |
| `restore_version` | `(version, justification?)` | Go **back** to an earlier model: restores its model + stored fit as the working state (no recompile/refit), appending a new version. Prefer this over re-authoring an earlier model with `set_model` (that refits to the same result and, once detected as a repeat, is warned then refused). Same effect as `rollback_to`, aptly named for going back. |
| `current_version` | `()` | HEAD version + `has_unfit_edits` / `n_pending_edits` (whether the working model has edits not yet scored by a `fit`). |
| `get_progress` | `()` | Early-stop-chain progress (a cheap counter read, no fit): `{n_fitted_ops, non_improving_streak, streak_cap, best_version, best_bic, best_gof_pvalue, bic_improvement_over_last_n, plateau_window, should_stop, stop_reason}`. `best_version` is the lowest-BIC model committed so far (the checkpoint to score). When `should_stop` fires, a plateaued `stop_reason` carries the structural-inadequacy nudge — if the score has stopped moving and the best model is still not adequate, the misfit is the model's FORM: the next model must change the structure (add a latent state, add a feedback reaction, raise interaction order), not re-fit/widen the same one. See Step 5 of the skill. |
| `reset` | `()` | Revert model + parameters to the initial version (v0). |

## Skills (this documentation, over MCP)

| Tool | Signature | Notes |
| --- | --- | --- |
| `list_skills` | `()` | The skills the server exposes: `name`, `description`, `files`. |
| `get_skill` | `(name, file?)` | A skill's `SKILL.md`, or a named supporting file (e.g. `file="tools.md"`). |

## Resources

Read-only content, mirrored as MCP resources:

- `petab://model/antimony`, `petab://model/sbml`
- `petab://observables`, `petab://conditions`, `petab://measurements`,
  `petab://parameters`
- `petab://simulations` (stored simulation dataframe),
  `petab://data/schema` (the `query_data` table schema),
  `petab://fit/log` (the pyPESTO optimizer log)
- `petab://provenance`, `petab://versions`, `petab://errors`
- `petab://skills` (index of skills), `petab://skills/{name}` (a skill's `SKILL.md`)
