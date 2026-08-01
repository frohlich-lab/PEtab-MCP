---
name: antimony-authoring
description: >-
  Read before writing an Antimony model string. The syntax and naming mistakes that make an
  Antimony model fail to parse, fail to compile under AMICI, or silently fit wrong — expecting
  the arrow to supply a reverse flux, hard-coded exponents, typos that become parameters, and
  the syntax that actually gets edits rejected: the rate law separator, declaration keywords
  Antimony does not have, `<->`, `2*X` stoichiometry, and names already taken by a built-in
  function or reserved by AMICI. Also why a parameter can read back as `name = ;`.
---

# Writing Antimony that works

Antimony is a text language for SBML. These are the mistakes that actually bite; everything
else about the syntax is guessable.

```
model my_model
  compartment cell = 1;         // at least one, and every species lives in one
  species A in cell; species B in cell;
  A = 1.0; B = 0.0;             // initial values
  kf = 0.1; kr = 0.05;          // parameters are plain assignments
  r1: A -> B; kf*A - kr*B;      // ONE reversible reaction: net = forward - reverse
end
```

## The arrow carries no flux — the rate law is the whole ODE

A reaction contributes **exactly** its rate law and nothing else. The arrow says which species
the rate is added to and subtracted from; it supplies no flux of its own.

`->` and `=>` differ only in the SBML `reversible` flag they set (`->` sets it true, `=>`
false). That flag is metadata — it changes no dynamics, and both arrows fit identically — but
a structure view will report it, so do not read `reversible: true` as "the reverse is handled".

So a reverse flux exists only if you write one. `r1: A -> B; kf*A` is one-way — declaring the
reaction "reversible" does not add a `kr*B` term. Write the reverse explicitly, either as a net
rate law or as a second reaction:

```
r1: A -> B; kf*A - kr*B          // one reaction, net rate — compact
r1: A -> B; kf*A                 // or two reactions — identical dynamics
r2: B -> A; kr*B
```

Both give `dA/dt = -kf*A + kr*B`. Prefer the single net rate law because it keeps the reverse
term visible next to the forward one and costs one reaction instead of two — not because the
split form behaves differently. It doesn't.

Use a forward-only law only for a genuinely one-way step: synthesis, degradation, an
effectively irreversible catalytic step.

## Every kinetic quantity must be a NAMED parameter

A bare number in a rate law is frozen — the fit cannot touch it, so a wrong guess can never be
corrected.

- Wrong: `prod: -> x1; 10 * K^2/(K^2 + x3^2)` — the max rate and both exponents are hard-coded.
- Right: `prod: -> x1; alpha * K^n/(K^n + x3^n)` — `alpha`, `K`, `n` are named and estimable.

**Exponents are the costly case.** `x^4` pins cooperativity at 4 forever. Name it.

Only *structural* constants stay literal: stoichiometric coefficients (`2 A -> B`) and the `1`
in a Hill/Michaelis–Menten denominator. Stoichiometry must be a literal — a parameter there is
frozen to a constant and dropped.

## Naming

- **Never** use `h` or `e` as parameter names — AMICI reserves them for code generation
  (`h` is the event/Heaviside vector, `e` is Euler's number) and the clash surfaces as a
  confusing compile failure, not a parse error. `t` is time. Use multi-letter names.
  Single-letter *species* (`A`, `S`, `P`) are fine.
- Match the names the problem already uses. A species driven by a condition-table column, or
  read by an observable formula, must be spelled exactly as that table spells it.
- **A typo becomes a new parameter, not an error.** `kf*A` mistyped as `kF*A` silently creates
  `kF`. Check what was added after an edit.

## Other syntax worth knowing

- A reaction **must** have a rate law: `J: A -> B;` alone is a parse error.
- Degradation has no product: `deg: A -> ; kdeg*A`. Synthesis has no reactant: `syn: -> A; ksyn`.
- `^` is exponentiation. Rate laws are ordinary algebra over species and parameters.

## A parameter can read back as `name = ;`

A parameter with no SBML value serializes to Antimony as `name = ;` (empty right-hand side) — how
an estimated parameter whose value lives in the PEtab parameter table round-trips. Not corruption
(it re-parses), not zero; the value is in the table, not the model text.

## Parse errors that actually happen

Antimony names the unexpected token, not the rule
you broke, so recognise them by shape.

- **The rate law follows a `;`, never a `:`.** The colon only ends the reaction *name*.
  Wrong: `R1: S -> E : foi;` — Right: `R1: S -> E; foi;`
- **`param` is not a keyword** (nor `parameter`). A parameter is a plain assignment.
  Wrong: `param S_on = 1.0;` — Right: `S_on = 1.0;`
  (`species`, `compartment`, `var` and `const` ARE real keywords — do not strip those.)
- **There is no `<->` arrow.** Only `->` and `=>`. A reversible step is a net rate law
  (see above). Wrong: `v: E + S <-> ES; ...` — Right: `v: E + S -> ES; k1*E*S - k_1*ES`
- **Stoichiometry is a space, not a `*`.** Wrong: `R: P -> 2*P;` — Right: `R: P -> 2 P;`
- **Reactants and products are species lists, not expressions.** Wrong:
  `v: (RAF_total - pRaf) -> pRaf;` — put the algebra in the rate law:
  `v: Raf -> pRaf; kact*(RAF_total - pRaf)`
- **Compartment placement is its own statement.** Wrong: `X = 3.64 in cell;` — Right:
  `species X in cell; X = 3.64;`

## Names that are already taken

Beyond the AMICI-reserved `h`/`e`/`t` above, Antimony refuses a name that is a **built-in
function**. Verified rejected: `gamma`, `log`, `exp`, `sin`, `abs`, `delay`, `piecewise`. The
error reads `syntax error, unexpected name of an existing function`, which does not sound like a
name clash at all. `gamma` is the one that bites in practice — it is a natural name for a decay
or recovery rate.

Prefix or qualify instead: `gamma_deg`, `k_gamma`, `decay_rate`.

Do not over-correct: `alpha`, `beta` and `kappa` are ordinary identifiers and are fine.
