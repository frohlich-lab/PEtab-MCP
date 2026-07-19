---
name: antimony-authoring
description: >-
  Use when writing the Antimony model string for petab-mcp's set_model (or the
  add_reaction / change_rate_law edits) — the syntax essentials, the SBML↔PEtab reconcile
  contract the server applies afterwards, the naming rules that avoid AMICI code-generation
  clashes, and the recurring parse/lint/reconcile failure modes with their fixes. Read this
  before authoring or editing a model so a set_model call commits instead of being rejected.
  A living reference: extend the "Known failure modes" list as new ones surface.
---

# Authoring Antimony for petab-mcp

The model is authored in **Antimony** — a compact text language for SBML. You pass a whole
Antimony string to `set_model` (build or re-author), and single Antimony fragments to
`add_reaction` / `change_rate_law`. This skill is about writing Antimony the server will
**accept** on the first try: the goal is to spend the model-building loop on *mechanism*, not
on fixing parse errors. Most model-definition failures come from a handful of avoidable
mistakes catalogued below.

## Minimal template

```
model my_model
  compartment cell = 1;            // at least one compartment
  species A in cell;               // every species lives in a compartment
  species B in cell;
  A = 1.0;                         // initial amount/concentration (a number or a *_init param)
  B = 0.0;
  kf = 0.1;                        // parameters are plain assignments
  kr = 0.05;
  r1: A => B; kf * A;              // reaction: id: reactants => products; RATE LAW;
  r2: B => A; kr * B;
end
```

- Wrap everything in `model <id> … end`. The `<id>` names the model (it wins over any
  placeholder id from a model-less load).
- `=>` is an **irreversible** reaction; `->` is **reversible** (AMICI/PEtab treat these
  differently — use `=>` and write each direction explicitly unless you truly want one
  reversible reaction). A reaction **must** have a rate law after the `;`.
- `^` is exponentiation (`p^2`), `*` `/` `+` `-` as usual. Rate laws are ordinary algebra over
  species and parameters.
- An empty/degradation reaction has no product: `deg: A => ; kdeg * A;`. A synthesis has no
  reactant: `syn: => A; ksyn;`.

## What the server does after you author (the reconcile contract)

`set_model` and the edits don't just parse the Antimony — they **reconcile** it with the PEtab
parameter table, then **lint** and **fit**. Author with this in mind:

- **Every species gets a parametric initial condition, FIXED at its declared t=0.** The server
  rewrites `A = 1.0` into `A = A_init` with `A_init` added to the parameter table (linear scale,
  `estimate = 0` — *fixed at 1.0*). You may write `A = A_init; A_init = 1.0;` yourself, but you
  don't have to. **Antimony cannot say "estimate this initial condition"** — the declared number
  is only the t=0 *value*; whether it is fitted is a parameter-table property, not model text. So
  writing the IC under a different name (`A_init`, `A_basal`, `p0_A`, a literal, an assignment
  rule) never changes whether it's estimated — they all reconcile to the same fixed `A_init` and
  fit identically. Whether initial conditions are estimated at all is set by the problem/scenario,
  not switchable from the model, so **don't re-author trying to make one estimable.** Every commit
  returns an **`initial_conditions`** block reporting, per species, whether its t=0 is `fixed`
  (declared value used), `estimated` (the fit sets it), or `condition_table` (set per condition) —
  read it to see the mode instead of guessing.
- **New rate-law symbols become estimated parameters.** Any name in a rate law that isn't a
  species or an existing table parameter is created as a new parameter (log10 scale,
  `estimate = 1`). This is how you introduce a new kinetic constant — just use it in the rate
  law (e.g. adding `Ki` in `Vmax*S/(Km+S+S^2/Ki)` creates `Ki` as an estimated parameter).
- **Orphans are pruned.** A species or parameter no longer used by any reaction is dropped —
  including, on a whole-model `set_model`, its fitted value. `set_model` warns when it drops a
  previously-*estimated* parameter.
- **Fixed/known parameters are preserved** as long as something references them: the
  observation noise `sigma_*` (referenced by an observable's noiseFormula) and any parameter a
  condition column sets stay in the table even if no rate law mentions them.

## Parametrise kinetic constants — no magic numbers

Every **kinetic or physical quantity** in a rate law must be a NAMED parameter, never a bare
numeric literal: rates, thresholds, amplitudes, basal levels, and — especially —
**exponents** (Hill / cooperativity coefficients). A hard-coded number is *fixed*: the fit
cannot touch it, so if your guess is wrong the model can never correct it.

- **Bad:** `prod: => x1; 10 * K^2 / (K^2 + x3^2);` — the `10` (max rate) and both `2`s (the
  Hill exponent) are hard-coded and un-estimable.
- **Good:** `prod: => x1; alpha * K^n_hill / (K^n_hill + x3^n_hill);` — `alpha`, `K`, `n_hill`
  are named, so each is created as an estimated parameter (and because `n_hill` sits in an
  exponent, the server auto-assigns it a *linear* scale with tight bounds — see the
  petab-model-building skill).
- **Exception — structural constants stay literal.** Stoichiometric coefficients (`2 A => B`)
  and the `1` in a Hill / Michaelis–Menten denominator (`.../(1 + ...)`) are part of the
  functional *form*, not tunable quantities; leave those as numbers.

A hard-coded exponent is the costly one: `x^4` silently pins the cooperativity at 4, and no
amount of fitting can move it (a repressilator run hard-coded `^4` and could not fit the
data). Name it and let it be estimated.

## Naming rules (avoid these — they cause hard-to-read failures)

- **Do NOT use single-letter parameter names that AMICI reserves for code generation** — in
  particular `h` (its event/Heaviside vector) and `e` (Euler's number); `t` is time. Use
  descriptive multi-letter names (`handling`, `attack`, `efficiency`), *not* `h`, `e`, `a`.
  This is a silent code-generation clash, not a clean parse error, so it is worth avoiding up
  front. (Single-letter *species* like `A`, `B`, `X`, `Y`, `S`, `P` are fine.)
- **Match the experimental design's names.** If a condition column sets a species' initial
  value (e.g. a dose column `S`), your species **must** be named `S` for the doses to apply;
  a mismatch surfaces as a petab-lint "missing entity" error. Likewise use the observable
  formulas' species names.
- **Never put a `sigma_*` (observation noise) or `*_init` (initial-condition) parameter into a
  rate law.** They belong to the noise / experimental model, not the kinetics; folding one
  into a rate law is a modelling error even if it parses.
- Declare (or deliberately introduce, per the reconcile contract) every symbol a rate law
  uses. An accidental typo becomes a new estimated parameter rather than an error — read the
  result's `parameter_diff.added` to catch that.

## The author → lint → fix loop

`set_model` is atomic: it parses, reconciles, lints, and fits, and **reverts on any failure**
so a bad author changes nothing. On rejection the result has `committed: false` with `error`
(and often `issues`). Read it and fix:

- *"Failed to parse antimony model: …"* — a syntax error (missing `;`, a reaction with no rate
  law, an unbalanced expression, a species not in a compartment). Fix the fragment named in
  the message.
- *petab-lint "… missing entity"* / *"observable/condition references X"* — the model doesn't
  declare a species/parameter the fixed design needs. Add it (usually a species name you
  renamed or dropped).
- *an SBML-consistency `issue` (`check: sbml:…`)* — libsbml flagged a structural problem in the
  model (a dangling reference, invalid MathML, an overdetermined system). Read the message and
  fix the offending construct. (Unit-consistency is *not* checked — these models are unit-less
  by design.)
- *"the model was accepted by validation but could not be compiled/fit: …"* — the model is
  valid SBML but AMICI cannot generate code for it; the classic cause is a reserved-name clash
  (single-letter `h`/`e`). Rename to multi-letter parameters and retry.
- A fit that commits but is statistically inadequate (`reduced_chi2 ≫ 1`, tiny
  `chi2_gof_pvalue`) is **not** an authoring failure — that is a *mechanism* problem for the
  model-building loop, not this skill.

Prefer to fix and re-`set_model` the whole coherent model rather than patching with many small
edits when the structure itself is what changed.

## Known failure modes (living list — append as new ones are seen)

- Single-letter kinetic parameter (`h`, `e`, `a`) → AMICI code-generation clash. Fix:
  multi-letter names.
- Substrate/species name not matching a condition dose column → doses silently not applied /
  lint "missing entity". Fix: name the species exactly as the condition column.
- Reaction line without a rate law (`J: A => B;`) → parse error. Fix: add `; <rate>`.
- Reversible `->` used where two irreversible `=>` were intended → wrong dynamics / extra
  implied reverse flux. Fix: write each direction as its own `=>` reaction.
- Dropping a needed kinetic parameter on a whole-model `set_model` re-author → its fitted value
  is lost (see the `warnings`). Fix: re-declare it in the Antimony if it was unintended.
- Hard-coded numeric constant in a rate law — a rate, threshold, or (worst) an exponent like
  `^4` → the quantity is fixed and un-estimable, so a wrong guess can never be corrected by the
  fit. Fix: replace the literal with a named parameter (structural constants — stoichiometry,
  the `1` in a Hill/MM denominator — excepted).
- Re-authoring a model repeatedly to make an initial condition estimable — trying `A_init`,
  `A_basal`, `p0_A`, a literal, then an assignment rule → every scheme reconciles to the same
  *fixed* `A_init` and the fit is identical each time (the NLL never moves). Antimony has no
  estimate-status syntax, and whether ICs are estimated is a fixed per-scenario setting. Fix: stop
  re-authoring; read the commit's `initial_conditions` report to see each IC's actual status.
