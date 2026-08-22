---
name: antimony-authoring
description: >-
  Read before writing an Antimony model string. The syntax and naming mistakes that make a model
  fail to parse, fail to compile under AMICI, or silently fit wrong — expecting the arrow to supply
  a reverse flux, hard-coded exponents, typos that become parameters, and the syntax that actually
  gets edits rejected: the rate-law separator, declaration keywords Antimony lacks, `<->`,
  stoichiometry, and names taken by a built-in function or reserved by AMICI.
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
corrected. Wrong: `prod: -> x1; 10 * K^2/(K^2 + x3^2)` — the max rate and both exponents are
hard-coded. Right: `prod: -> x1; alpha * K^n/(K^n + x3^n)`. **Exponents are the costly case**:
`x^4` pins cooperativity at 4 forever.

Only *structural* constants stay literal: stoichiometric coefficients (`2 A -> B`) and the `1`
in a Hill/Michaelis–Menten denominator (see below: a parameter must never be one).

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
- Match the names the problem already uses: a species driven by a condition column, or read by
  an observable formula, must be spelled exactly as that table spells it.
- **A typo becomes a new parameter, not an error.** `kf*A` mistyped as `kF*A` silently creates
  `kF`. After an edit, read the parameter list back and look for a name you did not intend.

## Other syntax worth knowing

- A reaction **must** have a rate law: `J: A -> B;` alone is a parse error.
- Degradation has no product (`deg: A -> ; kdeg*A`), synthesis no reactant (`syn: -> A; ksyn`).

## A parameter can read back as `name = ;`

A parameter with no SBML value serializes as `name = ;` — how an estimated parameter whose value
lives in the PEtab table round-trips. Not corruption, not zero: the value is in the table.
Write a plausible number instead — an empty value lands a NaN nominal on the row.

## Parse errors that actually happen

Antimony names the unexpected token, not the rule you broke — recognise them by shape.

- **The rate law follows a `;`, never a `:`.** The colon only ends the reaction *name*.
  Wrong: `R1: S -> E : foi;` — Right: `R1: S -> E; foi;`
- **`param` is not a keyword** (nor `parameter`). A parameter is a plain assignment.
  Wrong: `param S_on = 1.0;` — Right: `S_on = 1.0;`
  (`species`, `compartment`, `var`, `const` ARE real keywords — do not strip those.) A comma list
  needs one: `T0 = 11000, K0 = 3.9e-7;` is `unexpected ','`; write `const T0 = 11000, K0 = 3.9e-7;`
  or one assignment per `;`.
- **There is no `<->` arrow.** Only `->` and `=>`. A reversible step is a net rate law
  (see above). Wrong: `v: E + S <-> ES; ...` — Right: `v: E + S -> ES; k1*E*S - k_1*ES`
- **Stoichiometry is a NUMERIC LITERAL separated by a space — never a `*`, never a parameter.**
  Wrong: `R: P -> 2*P;` — Right: `R: P -> 2 P;`
  A parameter coefficient is the trap: `I -> I + NN*V` fails to parse, and "fixing" the `*` to a
  space gives `I -> I + NN V`, which PARSES but freezes the coefficient at NN's nominal and drops
  NN from the model. A burst size, yield or gain belongs in the rate law:
  Wrong: `r_burst: I -> I + NN V; delta*I;` — Right: `r_burst: -> V; NN*delta*I;` (production
  only; the loss of `I` stays its own reaction, and `I` on both sides never consumed it anyway).
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
