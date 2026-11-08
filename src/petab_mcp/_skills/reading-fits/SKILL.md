---
name: reading-fits
description: >-
  Reference for reading a fit: what each field of a fit result and of its diagnosis means, when
  a score or test says nothing, and DuckDB idioms for summarising the fit's tables one row per
  group.
---

# Reading a fit

## Scores

- `neg_log_likelihood`, `chi2`, `n_data_points`.
- `bic`, `aic`, `aicc`, `caic`, `hqc`, `tic` are `2·nll + penalty`, lower is better, each at its
  textbook weight. `criterion` names the one models are ranked by, `criterion_value` is its value.
- `k` = `n_estimated` (estimated parameter rows) + `hardcoded_constants` (the numeric literals in
  the model's math, each distinct value charged once).
- `chi2_dof`, `reduced_chi2`, `chi2_gof_pvalue` mean something only with a KNOWN noise level.
  With estimated sigmas chi2 sits near `n_data_points` by construction.
- Fitted values are LINEAR whatever the estimation scale; `parameter_scales` gives each scale.
- A change's result lists only what is new since the parent version in
  `fitted_parameters_changed`, `parameter_scales`, `parameters_at_bound` and `merge_hints`, only
  flagged sigmas in `noise_sigma_report`, and three flagged observables in `residual_structure`,
  with `n_parameters_at_bound`, `n_merge_hints` and `n_flagged_observables` the full counts. A fit
  summary has every pin, tie, sigma and flagged observable.
- `is_new_best`: this version became the best model, `progress.best_version`. A score that
  cleared the new-best gate without that carries `new_best_declined`, saying why.

## Flags

- `parameters_at_bound`: fitted values within ~1% of a bound, on their estimation scale. Widen
  at most once. A parameter that re-pins after a widening is unidentifiable and its optimum
  unbounded, often an exponent faking a mechanism the model lacks: remove its reaction or take its
  limiting (irreversible or instantaneous) form, and if neither is possible leave it pinned.
- `noise_sigma_report`: each estimated noise sigma against a data-driven reference (a fraction of
  its observable's measurement std), with a `status` of `ok`, `inflated`, `collapsed`, or
  `unknown` (no std to compare with). `data_noise_estimate` is a model-free noise level from
  successive measurement differences, biased HIGH, so a sigma far above it (`fitted_over_data_noise`)
  is strong evidence of inflation and one below it only weak evidence of collapse.
  `noise_sigma_inflated` / `noise_sigma_collapsed` with `noise_sigma_advice` flag either case: an
  inflated sigma absorbs misfit, a collapsed one overfits the noise, and either way the score is
  not genuine adequacy. The fix is the model's form, not the noise.
- `residual_structure`: the verdict over the residual tests, with a replicate lack-of-fit arm
  where the design has replicates. A flag marks where the fit is off, not proof of a better form,
  and stops nothing. Its `sigma_heterogeneity` names the sigmas whose `p` (below) survives Holm.
- `merge_hints`: the cheapest few priced ties (below).

## Diagnosis: residuals

- Per observable: `chi2`, `mean_weighted_residual` (bias: far from 0 is a systematic over- or
  undershoot), the worst |residual|, `sign_test` (bias) and `runs_test` (time trend, serial
  correlation), each with `p` and `low_power`. They read only residual SIGNS, so no sigma rescales
  them, but an estimated scale or offset takes up a series' bias and few points leave them little
  power: a `low_power` non-significant result is weak evidence, not a pass.
- `sigma_heterogeneity`: per estimated sigma, `p` that the series it covers share one spread of
  weighted residuals, and the `worst` series by `rms` (about 1 over the whole sigma). A series far
  above 1 is what the sigma was inflated for: look at that experiment first. A pointer, not a
  verdict, `low_power` as above. `noise_sigma.fitted_over_data_noise` ranks the same question
  per sigma.
- On a problem with many observables the table keeps `max_observables` rows by chi2, up to 3 more
  by |bias| and up to 6 more carrying a `structure_flag`; `omitted` says what was dropped.
  Aggregating over it aggregates a worst-first subsample.
- A model that is not built yet reports a `status` and no rows.

## Diagnosis: parameters

`identifiability` describes `version`'s fit: `available: false` before one, `stale: true` once a
different structure is in the working state.

- Per parameter: `se_scaled`, `fold_uncertainty`, `at_bound`, and `multistart_std` (how far it
  wanders across near-optimal starts, a model-free check); `correlations` and
  `partial_correlations`; `flat_directions`, the parameter combinations the data cannot pin;
  `candidates` ranks the parameters to cut. The FIM is local: read it with the multistart spread.
- `merge_candidates` prices giving a pair one shared value: `dnll`, `dscore` (negative when the
  merge lowers the score), `move_scaled` (how far the tie moves a parameter; cheap but many decades
  rarely survives a refit). Only same-role, same-scale pairs with neither member at a bound are
  priced, so an absent pair may be one that was skipped, not a dear one.

## SQL: one row per group

In a `GROUP BY` every selected expression is an aggregate or a grouping key: a window function
(`... OVER (...)`) is per row and cannot appear, and aggregates do not nest. `ORDER BY` goes
INSIDE the aggregate's parentheses (`first(measurement ORDER BY time)`), `FILTER` after them
(`count(*) FILTER (WHERE time > 0)`); neither hangs on a scalar function like `log10`.

| what | aggregate |
|---|---|
| value at earliest / latest time | `arg_min(measurement, time)` / `arg_max(measurement, time)` |
| peak and when | `max(measurement)` / `arg_max(time, measurement)` |
| trend | `regr_slope(measurement, time)`, `corr(time, measurement)` |
| spread | `stddev_samp`, `var_samp` (there is no `var`) |

```sql
SELECT observableId, experimentId, count(*) AS n,
       arg_min(measurement, time) AS y_at_t0, max(measurement) AS y_peak,
       arg_max(time, measurement) AS t_peak, regr_slope(measurement, time) AS slope
FROM measurements GROUP BY observableId, experimentId

SELECT observable_id, count(*) AS n, sum(residual_weighted * residual_weighted) AS chi2
FROM simulations WHERE residual_weighted IS NOT NULL
GROUP BY observable_id ORDER BY chi2 DESC
```

DuckDB is not SQLite or pandas: check that a function exists before leaning on it.
`optimization_starts` holds one row per start (`start` 0 is the best), `optimization_trace` the
objective per start and iteration.
