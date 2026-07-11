# petab-mcp tool & resource catalog

The complete surface of the `petab-mcp` server. See `SKILL.md` for the workflow
and the rules behind these tools.

## Read-only — the fixed design

| Tool | Signature | Returns / notes |
| --- | --- | --- |
| `describe_problem` | `()` | Summary: `model_id`, species/reaction/parameter counts, `free_parameters`, observables, condition/measurement counts, `current_version`, `batch_active`, `fixed_tables`. |
| `list_observables` | `()` | The fixed observables (formulas, noise models). |
| `list_conditions` | `()` | The fixed conditions and the model changes they apply. |
| `get_measurements` | `(condition_id?, observable_id?)` | The fixed measurement rows, optionally filtered. |
| `get_model` | `(format="antimony")` | Model as text. `format` is `"antimony"` (canonical) or `"sbml"`. |
| `get_model_structure` | `()` | Structured species / parameters / reactions view. |
| `validate_problem` | `()` | Lint the current problem; reports `valid` + `issues`. |

## Model operations (justification + references required)

Each applies to a model copy, reconciles SBML↔PEtab, lints, then (on a non-batch commit)
**fits + simulates** and commits — or reverts. Returns `{committed, version, staged,
reconcile, references, justification, validation, event_timestamp, fit_status,
neg_log_likelihood, chi2, fitted_parameters, improvement}` on success, or
`{committed: false, error, issues?}` on rejection.

| Tool | Signature | Notes |
| --- | --- | --- |
| `add_reaction` | `(reaction, justification, references, n_starts?)` | `reaction` is one Antimony line, e.g. `"J: A + B -> C; k_f*A*B"`. Rate law after `;` is required. New species → parametric IC; new rate-law symbols → parameters. |
| `change_rate_law` | `(reaction_id, rate_law, justification, references, n_starts?)` | Replace a reaction's kinetic law; new symbols reconciled. |
| `remove_reaction` | `(reaction_id, justification, references, n_starts?)` | Delete a reaction; prune orphaned species/parameters. |

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
| `sync_parameters` | `(policy="report")` | Reconcile table with model. `"report"` lists missing model params; `"add"` inserts them with default bounds + `estimate=1`. |

## Batching

| Tool | Signature | Notes |
| --- | --- | --- |
| `begin_batch` | `(label="")` | Start staging: operations apply + log but defer versioning. |
| `commit_batch` | `(label="")` | Snapshot all staged operations as one version. |
| `abort_batch` | `()` | Discard staged edits; restore the pre-batch version. |

While a batch is open, `rollback_to` / `reset` are blocked, and staged edits are not
fitted; `commit_batch` fits + simulates the staged result once.

## Fit + simulation (automatic on commit; read-only tools)

Fitting and simulation are not tools you call — they run on every commit. Inspect the
result with:

| Tool | Signature | Notes |
| --- | --- | --- |
| `get_simulations` | `()` | The stored simulation dataframe: fit metrics (`neg_log_likelihood`, `chi2`, `status`, `fitted_parameters`) + per-point `records` (measurement, simulation, residual, `residual_weighted`, noise). Generated on demand if none exists yet. |
| `inspect_fit` | `(top_n=10)` | Residual stats overall + across observables / experiments / timepoints, `top_n` worst-fit points, and bias flags. Reads the stored dataframe (no re-simulation). |

## Provenance, versions & errors

| Tool | Signature | Notes |
| --- | --- | --- |
| `get_provenance` | `(limit=50)` | Recent events (alterations, training, rollbacks). |
| `get_errors` | `(limit=50)` | Recent rejected operations / failed tool calls. |
| `list_versions` | `()` | All versions with kind/label/timestamp. |
| `get_version` | `(version)` | A version's Antimony model + parameter table. |
| `diff_versions` | `(from_version, to_version)` | Antimony unified diff + parameter add/remove/change. |
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
- `petab://fit` (current `inspect_fit` report), `petab://simulations` (stored simulation
  dataframe)
- `petab://provenance`, `petab://versions`, `petab://errors`
- `petab://skills` (index of skills), `petab://skills/{name}` (a skill's `SKILL.md`)
