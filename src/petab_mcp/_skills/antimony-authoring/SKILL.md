---
name: antimony-authoring
description: >-
  Read before writing an Antimony model string. The syntax and naming mistakes that make a model
  fail to parse, fail to compile under AMICI, or silently fit wrong — expecting the arrow to
  supply a reverse flux, typos that become parameters, `:=` vs `=`, stoichiometry, and names
  reserved by AMICI or taken by a built-in.
---

# Writing Antimony that works

Antimony is a text language for SBML. These are the mistakes that bite; the rest is guessable.

```
model my_model
  compartment cell = 1;         // at least one; every species lives in one
  species A in cell; species B in cell;
  A = 1.0; B = 0.0;             // initial values
  kf = 0.1; kr = 0.05;          // parameters are plain assignments
  r1: A -> B; kf*A - kr*B;      // ONE reversible reaction: net = forward - reverse
end
```

## The arrow carries no flux — the rate law is the whole ODE

A reaction contributes **exactly** its rate law and nothing else. The arrow says which species
the rate is added to and subtracted from; it supplies no flux of its own.

`->` and `=>` differ only in the SBML `reversible` flag, which is metadata: it changes no
dynamics. To know whether a reverse flux exists, read the RATE LAW, not the flag: it exists only
if you wrote one — `r1: A -> B; kf*A` is one-way. Write it as a net rate law
(`r1: A -> B; kf*A - kr*B`, preferred: the reverse term stays next to the forward one) or as a
second reaction (`r2: B -> A; kr*B`). Both give `dA/dt = -kf*A + kr*B`. Use a forward-only law
only for a genuinely one-way step: synthesis, degradation, an irreversible catalytic step.

## Every kinetic quantity must be a NAMED parameter

A bare number in a rate law is frozen — the fit cannot touch it, so a wrong guess can never be
corrected.
Wrong: `prod: -> x1; 10 * K^2/(K^2 + x3^2)` — the max rate and both exponents are hard-coded.
Right: `prod: -> x1; alpha * K^n/(K^n + x3^n)` — `n` estimated, so cooperativity is a finding.
Exponents are the trap here: a literal `x^4` asserts cooperativity of 4 and no fit can revise it.

## Naming

- **AMICI reserves seven lowercase names for generated code:** `x` states, `p` parameters,
  `k` constants, `w` expressions, `h` Heaviside, `t` time, `e` Euler's. Never use one for
  a parameter or a species. The clash is INVISIBLE until AMICI compiles — the model parses and
  validates, then the build fails and the whole edit is thrown away, costing a full edit-and-fit
  cycle, not a parse error. `k` and `p` bite most, being the textbook symbols for a rate constant
  and a production rate: write `k_mat`, `p_vir`. Uppercase is unaffected — single-letter species
  `A`, `S`, `P`, `V`, `T`, `I` are fine.
- **Time is the built-in `time`.** `t` is AMICI-reserved; any other name (`tau`) becomes a
  parameter the fit ESTIMATES, so the gate compiles and commits with its time dependence gone.
- **A reaction id must not be an Antimony keyword** (`at`, `in`, `is`, `end`, `var`, `const`,
  `event`, `species`, `model`): `at: -> X; k*C;` fails with `unexpected ->`, never naming the id.
- **Never assign a symbol to itself.** `k = k;` is a definition loop and refuses to load. A shipped
  constant needs no assignment — reference it by name.
- Match the names the problem already uses: a species a condition targets, or one read by an
  observable formula, must be spelled exactly as that table spells it.
- **A typo becomes a new parameter, not an error.** `kf*A` mistyped as `kF*A` silently creates
  `kF`. After an edit, read the parameter list back and look for a name you did not intend.

## Other syntax worth knowing

- A reaction **must** have a rate law: `J: A -> B;` alone is a parse error.
- Degradation has no product (`deg: A -> ; kdeg*A`), synthesis no reactant (`syn: -> A; ksyn`).

## A parameter can read back as `name = ;`

A parameter with no SBML value serializes as `name = ;`: not corruption, not zero — the value
lives in the PEtab table. Write a plausible number yourself; an empty value lands a NaN nominal.

## Parse errors that actually happen

Antimony names the unexpected token, not the rule you broke — recognise them by shape.

- **The rate law follows a `;`, never a `:`.** The colon only ends the reaction *name*.
  Wrong: `R1: S -> E : foi;` — Right: `R1: S -> E; foi;`
- **`param` is not a keyword** (nor `parameter`). A parameter is a plain assignment.
  Wrong: `param S_on = 1.0;` — Right: `S_on = 1.0;`
  (`species`, `compartment`, `var`, `const` ARE real keywords — keep those.) A comma list needs
  one: `T0 = 11000, K0 = 3.9e-7;` is `unexpected ','`; write `const T0 = 11000, K0 = 3.9e-7;` or
  one assignment per `;`.
- **`:=` is a RULE (all times); `=` is an initial value.** A derived quantity (a total, a
  normalised signal) is `y := expr;`; with `=` it is a constant and its observable goes flat.
- **There is no `<->` arrow.** Only `->` and `=>`. A reversible step is a net rate law
  (see above). Wrong: `v: E + S <-> ES; ...` — Right: `v: E + S -> ES; k1*E*S - k_1*ES`
- **Stoichiometry is a NUMERIC LITERAL after a space — never `*`, never a parameter.**
  Wrong: `R: P -> 2*P;` — Right: `R: P -> 2 P;`
  `I -> I + NN*V` fails to parse; "fixing" `*` to a space PARSES but freezes the coefficient at
  NN's nominal and drops NN from the model. A burst size/yield/gain goes in the rate law:
  Wrong: `r_burst: I -> I + NN V; delta*I;` — Right: `r_burst: -> V; NN*delta*I;` (production
  only — `I` was on both sides, so never consumed; its loss stays its own reaction).
- **Reactants and products are species lists, not expressions.** Wrong:
  `v: (RAF_total - pRaf) -> pRaf;` — algebra goes in the rate law:
  `v: Raf -> pRaf; kact*(RAF_total - pRaf)`.
- **Compartment placement is its own statement.** Wrong: `X = 3.64 in cell;` — Right:
  `species X in cell; X = 3.64;`

## Names that are already taken

Beyond the AMICI-reserved names above, Antimony refuses a **built-in function** name. Verified
rejected: `gamma`, `log`, `exp`, `sin`, `abs`, `delay`, `piecewise`. The error (`unexpected name of an
existing function`) does not sound like a name clash. `gamma` bites most — a natural name for a
decay rate; prefix it (`gamma_deg`). `alpha`, `beta`, `kappa` are fine.
