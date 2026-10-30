# Rationale Log

**Adding an entry is the exception, not the routine.** Most PRs end with nothing written here, and
that is the correct outcome. An entry earns its place only when a reader would otherwise UNDO the
choice — the obvious simplification is wrong, the tried alternative failed, the number was
measured. Design detail that the code, its docstrings or the commit message already carries is not
a rationale; it is noise that buries the entries that matter.

**Does it belong to one module that still exists? Then it is not a log entry.** Put it in that
module's docstring, next to the code a reader would undo. The log is for what spans modules, what
is deliberately ABSENT (a criterion not in the table, a field no result carries, a price table
there is no code for), and what outlives the code it explains — an entry about a function deleted
last week is doing its job.

**Budget: at most one new entry per PR, two or three sentences.** A PR that seems to need two
entries needs one, about the choice a reader is likeliest to undo. Name the failure, the measured
cost or the PR number, and stop; do not narrate the investigation, restate what the code does, or
enumerate every case you checked. Prefer editing or deleting a stale entry over appending beside
it. `#167`, `#179` and `#245` predate this budget and still carry two entries each; they were left
split because separate code sites cite a different one of them by name, so they are a legacy
exception and not a precedent.

The numbers on the entries below are pull requests of
[frohlich-lab/petab_mcp](https://github.com/frohlich-lab/petab_mcp), where this package was
developed before it moved here.

PEtab-MCP#172 A client that times out on a running fit waits for it with `wait_for_fit`, which blocks on the server, and not with a cheap status call: each poll is a full model call re-sending the agent's context (250k+ tokens on perica_egf, where one run polled `list_versions` 3-5 times in a row), so what a poll costs is set by how often it is made, not by its answer. The wait is an async tool that holds no worker thread, because sync tools share anyio's 40 threads with the fits that are already running.

PEtab-MCP#166 The species the statements route places at t = 0 are seeds (`make_changes`' `seeds`, `set_seed`/`remove_seed`, `seed <name>:` in a record), no longer starts: the same call takes `n_starts` for the multistart, and perica_egf's agent traces read "starts named twice" beside "every multistart failed". `<name>_init` and the reduction's `start_fraction` keep their names: they are ids in stored models and parameter tables.

PEtab-MCP#134 Indirect causal constraints are traced on the rows' model by BioNetGen with tag sites (`tracing`), deliberately NOT by INDRA's `PysbModelChecker`, and no longer on chains of the rows' claims read by hand. The checker's Kappa influence map reads rule patterns, never a rate law, so it cannot see a regulator a mechanism reads only in its rate: on #134's prototype it refused 9 of 27 true claims of the reference formulations (bachmann's SOCS3 feedback, isensee's IBMX and H89). The claims chain (#136's first version) needed a hand-written special case for each mechanic it met (a complex coming apart, sequestration at any step, a row taken twice); tracing leaves those to BioNetGen's matching and scored 27/0/3/7 (true held/refused, false held/refused) on #134's 37 scored claims against the chain's 26/1/4/6. No structural check settles a sign under feedback (a route of the opposite sign existed for most true indirect claims), so a constraint holds wherever the trace moves it as claimed, and the other sign is not reported: both are true where both routes exist.

PEtab-MCP#135 A row's statement and its binding are separate arguments of `make_changes` (`statements`, and `bindings` by prefix), and a row still written the one-line way, `prefix: <statement> | <policy> ...`, or as JSON carrying `policy`, is refused with the split it should have been (`notation.InlineBinding`, `operations.carried_binding`), not also read: two notations for one row would let the claim and its maths travel as one again, which the contract between them (#127/#128) keeps apart. Versions stored before keep the old shape on disk and are read through `provenance.statements_record`.

PEtab-MCP#128 A row's recorded regulator kinds are signed again at every commit on that version's own parameter table, condition values and measurement times (`regulation.resign`, from `operations._commit_change`), not once when the row is built: `k*(1 + a*C)` reads `activated by C` only while `a`'s lower bound is not negative, so a bound edit with no row in it can change what a version records (review of #128). A condition input ranges over the model's own value as well as its conditions', since that value holds wherever no period sets it, and a bounded parameter keeps both ends. Two kinds of row are not signed up, after a check of every recorded kind against the steady state with products present (the second review of #127/#128): one whose J can go negative or pass a pole on those ranges, where J's sign no longer says which way the reaction runs (sweep multi108: J(M2=0.5) = -2, J(1.5) = +2), so it records `modulated by` for every regulator; and a `modulated by` that also acts on a step taking the product away, which J, its product absent, does not see (sweep case33/gexpr5: J rises with M while the product's steady state falls). A stated direction keeps J's there, and the warning names the step.

PEtab-MCP#127 A statement's regulator is signed on the statement's own reaction rate, its products absent, read off the row as the model builds it (`regulation.net_rate`), never off the mechanism alone: a mechanism signed once on placeholder agents with every name positive missed a pinned negative constant, two names for one agent and states other rows give, and crashed on `ANY` (review of #128). One acting only on a step that rate cannot see, a product's degradation or a complex coming apart, is refused, pointing to that step's own row: signed on the products' steady state instead it would give one claim two meanings, and `as_indra` would put the regulator on the wrong reaction, an agent that speeds mRNA decay projected as an inhibitor of the mRNA's synthesis (sweep-55max c1). Products stay held at 0 in that rate (decided 2026-10-24), so a claim keeps that one meaning, a regulator of the reaction run from its reactants; in the 570-entry regulator corpus none of the agent-written refusals depends on it (5 of 61 do, all reviewer repros), and a regulator that needs the product present is told to state the product as a regulator too. A regulator whose pattern matches only some of the row's own forms (`ERK{pT202}` matching the enzyme bound to its product) is refused, not read at its steady state: that amount is set by the very rates the regulator changes, so no direction of J in it is a claim about something held apart from the reaction; one matching every form of the enzyme reads its conserved total, which is held (review of #127). Forms that differ only in sites a row never reads are lumped before J is solved, and the lumping is checked at a point where each form has an amount of its own rather than trusted: a rate reading such a site after all would make the lumped J a different function, and the check then leaves the forms apart (the second review of #127/#128, EGFR with four sites other rows set). The check reads the rules as written, on an expansion of its own where `reduction` changed them: a reduced rule's catalyst is a rate factor and a split monomer is blocks, neither the statement's agent.

PEtab-MCP#115 There is deliberately no run-time check that a statements model's reduced form (`reduction`) simulates as its unreduced expansion: one was built (`verify_reduction`) and removed, because comparing the two through AMICI reported false mismatches wherever the steady-state solver stopped two equal systems at different points (a stiff pre-equilibration came back at 786,256 against 998,002 while reporting success), and let real errors through wherever that noise was large; pre-equilibration convergence is PEtab-MCP#131. Exactness is held offline instead, by building each case with `problem_sbml.to_antimony(..., reduce=False)` and integrating both (`tests/test_reduction.py`, `tests/test_reduction_corpus.py`).

PEtab-MCP#122 A row's binding may still read a condition input the entity map does not map to an agent (`kf="k*dose"`), although the statement does not name it (`mechanisms.bind`). Refusing it leaves a dose only one way in, as the declared seed of an agent the statement names, and a seed is evaluated once, before pre-equilibration: perelson's `rit`, 0 while pre-equilibrating and 1 after, would hold its pre-equilibration 0 for the whole experiment. A dose that should read as an agent is mapped to one in the entity map, which makes it a held species its periods set.

PEtab-MCP#99 The replicate lack-of-fit yardstick is the replicate scatter pooled over one noise sigma's design points (`residual_tests.replicate_lack_of_fit`), neither the point's own scatter nor the fitted sigma the likelihood uses. At n_i = 3 a point's own scatter rests on three values, so the tightest triplicates ranked first by chance: on perica_egf all five flags sat 10-40x below their observable's typical scatter, and the agent spent three model revisions on them. A fitted sigma absorbs misfit and shrinks every z exactly when the misfit is largest: against it none of those five points survives Holm, including the two worst-fitted ERK readouts that the pooled scatter still flags.

PEtab-MCP#35 What is present at t = 0 may be DECLARED with the statements (`make_changes`' `seeds`, `problem_sbml._declare`), not inferred, when the trace's default does not fit. The default seeds every unmade monomer at an estimated `<monomer>_init` and every made one empty; on models whose species start at a fixed unit pool, at a shipped expression or at a condition-table column, that cost an estimated parameter the data cannot inform, and the usual workarounds (a zero-rate synthesis row so a monomer counts as made; renaming a species after a condition column) all left a dead parameter or a dead rule. A seed is a modelling claim, so it is sent with the rows rather than put in the entity map: the map is supplied with the problem, and a seed there would both prescribe part of the model's structure and be out of the modeller's reach. A declared monomer gets no default seed at all, so its seeds are either wholly the default or wholly declared and never a mix. A number reaches reconcile as `<name>_init` and is treated like any other starting value, so a unit pool is fixed by the same {0,1} convention; 0 alone is written as a literal with no parameter, because reconcile leaves a condition-set species' assignment in place and a zero-valued `_init` behind it was estimated as a plain constant.

PEtab-MCP#30 An experiment's input is the problem's: a species its condition table targets, or one the problem's `inputs.tsv` holds to a protocol that changes within a period (e.g. a stimulus ramped within a period, `protocols`). It is never asserted as a statement or built by a mechanism, and not declared in the entity map either: a protocol held in the entity map as well as in `inputs.tsv` drifts, since only one copy is checked. A `Stimulus` statement with a schedule mechanism was built and removed: the schedule reads only the clock, sets a value rather than a rate, carries numbers that belong to experiments rather than the system, and is known by design, so it sat outside every pressure the library is judged by. A model built from INDRA's statements alone does not commit without the input either, since the problem's dose columns would lint as unused; the INDRA warm start that would build such a model comes later, with its wiring and a pinned corpus (PEtab-MCP#43).

#300 A fixed-θ held-out set may bring ROWS of the training condition table but never a new target, and the rows are appended at score time (`validation_data._with_held_out_design`) rather than shipped in the agent's `problem/conditions.tsv`. The shipped-table route is the obvious simplification and worked on main before #300; it tells the agent which experiments are held out and moves `Scenario.version`, which is what kept raimundez's EGFR knockdown (Fig 6E) filed as a transfer set in #290's first draft although the paper predicted it at the fitted point.

#89 `validation.stoichiometry_conflict_issues` and its `set_model` caller are gone: antimony's 3.2
floor makes the id-uniqueness collision they translated — `N A -> N B` failing conversion outright
— unreachable. The STATIC `frozen_parametric_stoichiometry_issues` stays, because the footgun is
NOT fixed; on 3.2.0 the collision merely went quiet, emitting a frozen `stoichiometry` plus an
assignment rule reading an undeclared symbol, which libSBML passes with warnings and no error. A
floor rather than the rejected cap, since an older antimony would restore an exception nothing here
translates any more.

#245 `identifiability._compute_fim` and `effective_parameter_count` take the FIM and the scores
from the residual rows of one call, not from the objective's half-curvature `hess` — a deliberate
change to agent-facing statistics, not a refactor. With sigmas FIXED the two agree to 3.6e-17
relative, so no shipped example moves; an ESTIMATED sigma's own diagonal doubles, so `se_scaled`
falls by sqrt(2) and a sigma-sigma merge tie doubles (predator_prey 1.026 -> 2.093 nll). Building
the rows toggles AMICI's `add_sigma_residuals` by hand on the engine's ONE cached objective, so it
happens under `session.lock` and is restored in a `finally`; the remaining hazard is a
`get_fit_diagnosis` racing a simulate on another thread, which doubles that simulation's
residuals.

#245 The three eigenvalue floors this matrix feeds — `pinv` in `_identifiability_stats`,
truncation in `effective_parameter_count`, the CLAMP in `merge_candidates` — are deliberately NOT
unified: truncating in `merge_candidates` would price the one FREE merge as the report's most
expensive. Unifying the FIM is not a licence to unify its inverses.

#243 A `make_changes` result carries no flat `warnings` list — the payload diet cut it at the tool
boundary (17% of payload bytes, duplicated on 62.8% of 915 payloads) while `bundle_result` went on
computing it. Do not reintroduce it; filter `messages` by level. The operations-layer `warnings` is
a DIFFERENT field and stays.

#167 No scalar weight on the complexity penalty in `petab_mcp.criteria` — named penalties and
nothing between them. One was built and removed: every effect-size threshold in the loop
(`NEW_BEST_MIN_DELTA_SCORE`, `PLATEAU_MIN_SCORE_GAIN`, `dedup.WORSE_SCORE_MARGIN`, the benchmark
bar) is a fixed count of deviance units calibrated on BIC, which survives SWAPPING the penalty but
not SCALING it. Deriving those thresholds from the criterion rather than writing them as literals
in `constants.py` is the precondition for reintroducing the knob.

#167 Criteria left out of the table on purpose. ICOMP is the one measure that charges for
parameter INTERDEPENDENCE, this project's actual failure mode, but it is zero for any number of
orthogonal parameters and infinite on a rank-deficient FIM, so no fixed effect-size threshold means
anything under it — it belongs beside `k_eff` as a diagnostic. Anything integrating over parameter
space is excluded by construction (a fit here is one optimum, not a posterior): marginal
likelihood, DIC, WAIC, PSIS-LOO, stochastic complexity; FPE / Mallows' Cp / GCV would rank nothing
differently, and EBIC needs a countable model space that a freely authored network is not.

#179 `identifiability.effective_parameter_count` does not take the objective's `hess`, because
AMICI's `fFIM` is `sresᵀ·sres` alone: `rdata.cpp` derives the exact expected Hessian and then drops
the `(∂σ∂σᵀ)(2r² − σ²)/σ⁴` term, which is sound for an optimizer and wrong for a statistic that
inverts the matrix — an ESTIMATED sigma keeps half its Fisher information (toy σσ 31.8 against 63.6
by finite differences), inflating `k_eff` from 3.3 to 4.0 at k=3. The ρ²-weighted sigma rows restore
exactly that term's expected value, so this matrix is AMICI's FIM plus the piece it documents
leaving out, and identical to it wherever sigma is fixed or inner-solved. Do not simplify it back to
`ret[HESS]`.

#179 `k_eff` is deliberately absent from every result the agent reads — `goodness_of_fit`,
`fit_result_block`, `get_fit_summary`, `list_versions` meta, every tool docstring. A number on
every fit result is a number the agent starts optimising; adding it to the score table for symmetry
with `bic`/`aic` is the change to refuse.
