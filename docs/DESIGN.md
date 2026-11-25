# Design notes

Principles to hold onto when extending this package.

- **No free-form editing.** The model is only ever changed through `make_changes`, and every
  change must carry a justification. Don't add a second way to write to the model.
- **A change and its score never come apart.** `make_changes` always fits and snapshots a
  version — there is no edit without a fit, and no fit without a version.
- **SBML and PEtab stay consistent automatically.** Reconciliation (new symbols, orphan
  pruning, parametric initial conditions) is the server's job, not the agent's — but a few
  cases (an IC of exactly 0, a condition-table override, an assignment rule, a compound
  expression) are deliberately left untouched because minting there would be wrong, not just
  unnecessary. Read [`src/petab_mcp/reconcile.py`](../src/petab_mcp/reconcile.py) before
  "fixing" one of these.
- **Every operation is versioned and logged.** Provenance under `.petab_mcp/` is append-only
  and durable; extend what a version records rather than bypassing it.
- **Iteration is cheap; results are not cached.** Structural edits recompile, parameter
  edits don't — but a model an earlier version already fitted (its structure, its bounds, as many
  starts) is refused a wasted refit, never served from a fit-result cache (there isn't one). A
  refit is a `make_changes` without a model route, and a new version.
- **Fit inspection is SQL, not a growing menu.** A new way to look at a fit should be a
  `query_data` query over the DuckDB tables, not a new bespoke tool.
- **Comparisons must be valid before they're reported.** A commit that isn't nested in what
  it's compared against is flagged, not silently ranked by score.
- **One score, named, on one scale.** Models are ranked by a complexity-penalised score of the
  form `2·nll + penalty` — `bic` by default; `caic`, `aic`, `aicc`, `hqc` or `tic` (Takeuchi,
  charging the effective parameter count `k_eff = tr(J·H⁻¹)` the server computes) via
  `--criterion`. Every fit reports every score plus `criterion`/`criterion_value`, and every
  gate — new best, plateau stop, de-dup, merge price, the harness judge — reads that one
  value, so they cannot disagree; canonical BIC is recorded on every run regardless, and
  `result.json` carries the whole table. Named penalties and nothing between them: the fixed
  effect-size thresholds carry over when the penalty is swapped, not when it is scaled. See
  [`src/petab_mcp/criteria.py`](../src/petab_mcp/criteria.py) before adding one.
- **The estimation scale is the server's.** PEtab v2 dropped `parameterScale`, so the server keeps
  the column as an extension of the parameter table and hands it to pyPESTO's v2 importer as its
  `parameter_scales`, which puts AMICI's simulator on it too — every pyPESTO problem is built
  through `fitting.pypesto_problem`, never handed to pyPESTO linear. A problem that states a
  prior is refused at load: the objective is the likelihood the criterion scores.
- **A statement says everything its rate reads.** On the `statements` route a row's statement is
  the claim and its binding the maths behind it, the two given apart under the row's prefix
  (`statements=["r1: RAF phosphorylates MEK"], bindings={"r1": "two_step kf=k_on"}`), so a rate (a
  written `rate`, a `scheme`) reads only the agents the statement names: its roles, the partners they are written bound or not
  bound to, and the agents it states regulate its rate without taking part, `inhibited_by`,
  `activated_by` or `modulated_by` ("RAF phosphorylates MEK, inhibited by sorafenib"). A scheme
  reads one by its kind, in a rate or as a molecule its rules bind (sorafenib sequestering RAF),
  on sites named after each other, so the complex is the one the partner notation writes
  (`RAF{!sorafenib}`) and the claim stays on the statement. Each is
  recorded as the kind the reaction's rate shows for every value of its names within their
  bounds: inhibited where it never rises with it, activated where it never falls, modulated where
  neither can be shown. That rate is the row's net rate as the model builds it, read off the
  reactions its rules make in the whole model's expansion: the row's own agents, the states other
  rows give them and its bindings, its intermediates steady and its products absent, since where a
  step sits decides which way its own rate moves the reaction. A direction counts where sympy
  proves it and the rate, read at points across its names' ranges, moves no other way; it is
  signed again whenever a row, a binding or a bound changes. A regulator that rate does not change
  with, acting only on a step it cannot see (a product's degradation, a complex coming apart), is
  refused, pointing to that step's own row: a regulator is a claim about its statement's own
  reaction, so each claim keeps one meaning and a causal graph never puts it on another reaction.
  Products stay absent in that rate, so a regulator that acts only with the product present
  (feedback through it, autoregulation, autocatalysis) does not move it either: its refusal names
  the rows that state it, the product stated as a regulator too, or that step on a row of its own
  so.
  Any other claim the maths does not bear out is recorded as what it does show, beside what was
  stated, never refused: the edit reports it and asks the agent to reconsider, and `get_model` and
  the English mark it ("activated by sorafenib (from its rate)")
  ([`statements`](../src/petab_mcp/statements.py), [`regulation`](../src/petab_mcp/regulation.py)).
  INDRA has no statement about a
  reaction, so a causal graph gets the nearest one it has, wider than the claim
  (`statements.as_indra`: "inhibited by sorafenib" on RAF's step is `Inhibition(sorafenib, RAF)`).
  A condition input the entity map
  does not map to an agent is no agent, so a row's binding may read it (`kf="k*dose"`) and the
  condition table says what it is, unless a seed of an agent at its id alone makes it that agent.
  Which inputs are agents is the problem author's call, made in the entity map, or the modeller's,
  made by such a seed; one mapped either way is read only as the agent it is.
- **A constraint is a claim the model must make.** A session may start with causal constraints
  (`--constraints`), statements rows without bindings that each end `, directly` (one row's claim
  entails it) or `, indirectly` (the network the rows expand to carries it, with its sign), with
  `not ..., directly` and `X does not affect Y` forbidding what they name; `make_changes` refuses a
  model that breaks one, a parameter edit included, since a bound can turn a regulator. Each claim
  is a row's own, its regulators as their rates record them, so a constraint rests on the maths,
  never on what the agent wrote; `activates` is about the forms the constraints declare active,
  never inferred from the agent's rows. An indirect constraint is traced, as the MSB 2023 paper
  traces a signal: every molecule gets a tag site, and BioNetGen expands the rows' model with the
  subject's molecules tagged and each rule handing a driver's tag to what it changes or makes
  ([`tracing`](../src/petab_mcp/tracing.py)). Which way a model moves an object is not decidable from
  its structure once it has feedback, so an indirect constraint holds wherever the trace moves it as
  claimed, whatever else the trace finds: that sign is the fit's to say
  ([`constraints`](../src/petab_mcp/constraints.py)). With `--constraint-updates any`,
  `set_constraints` replaces them for whoever drives the session, logged with whether it loosens or
  tightens them; a version a new set breaks keeps the verdicts it was committed under, but is
  flagged, cannot be restored and no longer ranks as the best.
- **A hardcoded constant is not free.** `k` is the estimated parameters plus every numeric
  literal in the model's math bar a few structural ones (each distinct value once — see
  [`src/petab_mcp/literals.py`](../src/petab_mcp/literals.py)), so freezing a fitted value costs
  what estimating it costs and there is no threshold to round past. A genuinely known constant
  ships as a fixed row: the problem author's call, not the agent's.
