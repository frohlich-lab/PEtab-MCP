# petab-mcp tool & resource catalog

The complete surface of the `petab-mcp` server. See `SKILL.md` for the workflow
and the rules behind these tools.

## Read-only — the fixed design

| Tool | Signature | Returns / notes |
| --- | --- | --- |
| `describe_problem` | `()` | Summary: `model_id`, species/reaction/parameter counts, `free_parameters`, observables, condition/measurement counts, `current_version`, `batch_active`, `fixed_tables`, `amici_available`. |
| `list_observables` | `()` | The fixed observables (formulas, noise models). |
| `list_conditions` | `()` | The fixed conditions and the model changes they apply. |
| `get_measurements` | `(condition_id?, observable_id?)` | The fixed measurement rows, optionally filtered. |
| `get_model` | `(format="antimony")` | Model as text. `format` is `"antimony"` (canonical) or `"sbml"`. |
| `get_model_structure` | `()` | Structured species / parameters / reactions view. |
| `validate_problem` | `()` | Lint the current problem; reports `valid` + `issues`. |

## Model operations (justification + references required)

Each applies to a model copy, reconciles SBML↔PEtab, lints, then commits or
reverts. Returns `{committed, version, staged, reconcile, references,
justification, validation, event_timestamp}` on success, or
`{committed: false, error, issues?}` on rejection.

| Tool | Signature | Notes |
| --- | --- | --- |
| `add_reaction` | `(reaction, justification, references)` | `reaction` is one Antimony line, e.g. `"J: A + B -> C; k_f*A*B"`. Rate law after `;` is required. New species → parametric IC; new rate-law symbols → parameters. |
| `change_rate_law` | `(reaction_id, rate_law, justification, references)` | Replace a reaction's kinetic law; new symbols reconciled. |
| `remove_reaction` | `(reaction_id, justification, references)` | Delete a reaction; prune orphaned species/parameters. |

## Parameter operations (justification required, references optional)

Versioned. Same commit pipeline / result shape as model operations.

| Tool | Signature | Notes |
| --- | --- | --- |
| `change_parameter_scale` | `(parameter_id, scale, justification, references?)` | `scale` ∈ `{"lin","log","log10"}`. `log`/`log10` require `lowerBound` and `nominalValue` > 0. |
| `change_parameter_bounds` | `(parameter_id, lower_bound, upper_bound, justification, references?)` | Set new bounds. |

## Parameters (quick, non-versioned)

| Tool | Signature | Notes |
| --- | --- | --- |
| `get_parameters` | `()` | The parameter table: id, bounds, `nominalValue`, `estimate`, scale, priors. |
| `update_parameters` | `(updates)` | Each update: `{id, nominalValue?, lowerBound?, upperBound?, estimate?, parameterScale?}`. Transactional; **not** versioned. |
| `sync_parameters` | `(policy="report")` | Reconcile table with model. `"report"` lists missing model params; `"add"` inserts them with default bounds + `estimate=1`. |

## Batching

| Tool | Signature | Notes |
| --- | --- | --- |
| `begin_batch` | `(label="")` | Start staging: operations apply + log but defer versioning. |
| `commit_batch` | `(label="")` | Snapshot all staged operations as one version. |
| `abort_batch` | `()` | Discard staged edits; restore the pre-batch version. |

While a batch is open, `optimize` / `rollback_to` / `reset` are blocked.

## Evaluation (needs `amici`; `optimize` also needs `pypesto`)

| Tool | Signature | Notes |
| --- | --- | --- |
| `simulate` | `()` | Simulate all experiments at current parameters. |
| `evaluate_objective` | `()` | Negative log-likelihood vs the fixed data (**lower is better**). |
| `optimize` | `(n_starts=5, seed=0)` | Multi-start fit; writes best fit back; records a training version with `objective_before`, `self_improvement`, `improvement_vs_best_prior`, `is_new_best`. |

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
- `petab://provenance`, `petab://versions`, `petab://errors`
- `petab://skills` (index of skills), `petab://skills/{name}` (a skill's `SKILL.md`)
