# petab-mcp tool & resource catalog

The complete surface of the `petab-mcp` server. See `SKILL.md` for the workflow
and the rules behind these tools.

## Read-only — the fixed design

| Tool | Signature | Returns / notes |
| --- | --- | --- |
| `describe_problem` | `()` | Summary: `model_id`, `model_initialized` (false ⇒ author the model with `set_model` first), species/reaction/parameter counts, `free_parameters`, observables, condition/measurement counts, `current_version`, `batch_active`, `fixed_tables`. |
| `list_observables` | `()` | The fixed observables (formulas, noise models). |
| `list_conditions` | `()` | The fixed conditions and the model changes they apply. |
| `get_measurements` | `(condition_id?, observable_id?, summarize=True)` | Compact summary of the measurements by default (per observable×condition: count, time span, value min/max/mean, a few preview points); pass `summarize=False` for the full measurement rows. |
| `get_model` | `(format="antimony")` | Model as text. `format` is `"antimony"` (canonical) or `"sbml"`. |
| `get_model_structure` | `()` | Structured species / parameters / reactions view. |
| `validate_problem` | `()` | Lint the current problem; reports `valid` + `issues`. |

## Model operations (justification required, references optional)

Each applies to a model copy, reconciles SBML↔PEtab, lints, then (on a non-batch commit)
**fits + simulates** and commits — or reverts. Returns `{committed, version, staged,
reconcile, references, justification, validation, warnings, event_timestamp, fit_status,
neg_log_likelihood, chi2, fitted_parameters, improvement}` on success, or
`{committed: false, error, issues?}` on rejection. `warnings` carries advisory notes (e.g.
an added unobserved species).

| Tool | Signature | Notes |
| --- | --- | --- |
| `set_model` | `(antimony, justification, references?, n_starts?)` | Author the **whole** model from an Antimony string — the single whole-model op. **Builds** the first model when none is initialized (`describe_problem` → `model_initialized: false`) and **re-authors / replaces** an existing one (never refused). Use for the initial build and for a coupled/symmetric or multi-reaction **structural** change in one coherent step; reconciles + fits, reverts on failure. Warns if a previously-**estimated** parameter is dropped (fixed noise/IC params + table-referenced symbols are preserved). |
| `add_reaction` | `(reaction, justification, references?, n_starts?)` | `reaction` is one Antimony line, e.g. `"J: S1 + S2 -> S3; k_f*S1*S2"` (placeholder names). Rate law after `;` is required. New species → parametric IC; new rate-law symbols → parameters. Requires an initialized model. |
| `change_rate_law` | `(reaction_id, rate_law, justification, references?, n_starts?)` | Replace a reaction's kinetic law; new symbols reconciled. |
| `remove_reaction` | `(reaction_id, justification, references?, n_starts?)` | Delete a reaction; prune orphaned species/parameters. |

`n_starts` overrides the server default multistart count for the fit on this commit
(`0` = simulate without fitting).

## Parameter operations (justification required, references optional)

Versioned commits — same pipeline / result shape as model operations (they also fit +
simulate). Parameter **values** (`nominalValue`) are set only by the fit; these change the
search space.

| Tool | Signature | Notes |
| --- | --- | --- |
| `change_parameter_scale` | `(parameter_id, scale, justification, references?, n_starts?)` | `scale` ∈ `{"lin","log","log10"}`. `log`/`log10` require `lowerBound` and `nominalValue` > 0. |
| `change_parameter_bounds` | `(parameter_id, lower_bound, upper_bound, justification, references?, n_starts?)` | Set new bounds. |

## Parameters (read-only + reconcile)

| Tool | Signature | Notes |
| --- | --- | --- |
| `get_parameters` | `()` | The parameter table: id, bounds, `nominalValue` (fit-only), `estimate`, scale, priors. |
| `sync_parameters` | `(policy="report")` | Reconcile table with model. `"report"` lists model params missing from the table and orphaned table rows (no change); `"add"` runs the full reconcile — adds missing params (with IC/exponent/scale handling) and removes orphaned rows — the same sync the commit pipeline runs. |

## Batching

| Tool | Signature | Notes |
| --- | --- | --- |
| `begin_batch` | `(label="")` | Start staging: operations apply + log but defer versioning. |
| `commit_batch` | `(label="")` | Snapshot all staged operations as one version. |
| `abort_batch` | `()` | Discard staged edits; restore the pre-batch version. |

While a batch is open, `rollback_to` / `reset` are blocked, and staged edits are not
fitted; `commit_batch` fits + simulates the staged result once.

## Fit + simulation (automatic on commit; read-only tools)

Fitting and simulation are not tools you call — they run on every commit. Read the score
with `get_fit_summary`, and diagnose the fit by querying the data with SQL:

| Tool | Signature | Notes |
| --- | --- | --- |
| `get_fit_summary` | `()` | Just the metrics — `status`, `neg_log_likelihood`, `chi2`, `n_data_points`, `fitted_parameters`, `version` — **no records**. The cheapest score check. |
| `list_data_tables` | `()` | The SQL tables `query_data` exposes: name, description, columns, row count. |
| `query_data` | `(sql, max_rows=1000)` | Read-only DuckDB query over the tables below. Returns `{columns, rows, row_count, truncated}`. |
| `get_fitting_log` | `(tail_lines=200)` | The tail of the pyPESTO/Fides optimizer log (`pypesto.log`): multistart spread, convergence, parameters pinned at bounds, ill-conditioning — the numerical view of the fit. `0` returns the whole file. |

### `query_data` tables

| Table | Columns (key ones) | Notes |
| --- | --- | --- |
| `measurements` | `observableId`, `simulationConditionId`, `time`, `measurement` | The fixed measurement data. |
| `conditions` | `conditionId`, … | The fixed conditions. |
| `observables` | `observableId`, `observableFormula`, `noiseFormula` | The fixed observables. |
| `parameters` | `parameterId`, `lowerBound`, `upperBound`, `nominalValue`, `estimate`, `parameterScale`, priors | Parameter table (`nominalValue` fit-only). |
| `simulations` | `observable_id`, `condition_id`, `time`, `measurement`, `simulation`, `residual`, `residual_weighted`, `noise` | Per-point dataframe from the last commit. `residual` = `measurement - simulation` (natural units); `residual_weighted` = `(m - s)/σ` (the χ² term); `noise` = σ. Generated on demand at cold start; empty until the first commit. |
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
| `get_provenance` | `(limit=50)` | Recent events (alterations, batches, rollbacks, resets). |
| `get_errors` | `(limit=50)` | Recent rejected operations / failed tool calls. |
| `list_versions` | `()` | All versions with kind/label/timestamp **and fit metrics** — rank versions by fit. |
| `get_version` | `(version)` | A version's Antimony model + parameter table. |
| `diff_versions` | `(from_version, to_version)` | Antimony unified diff + parameter add/remove/change + a `fit` block (each version's nll/chi2 and the neg-log-likelihood delta). |
| `rollback_to` | `(version, justification?)` | Restore a prior version as a new version. |
| `current_version` | `()` | HEAD version + batch state. |
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
