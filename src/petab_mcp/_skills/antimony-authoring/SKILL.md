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
Wrong: `r1: A -> B; 0.3 * A * D^2` — the rate constant and the exponent are hard-coded.
Right: `r1: A -> B; k1 * A * D^m` — `k1` and `m` estimated, so the data decide both.
Exponents are the easy ones to miss: a literal `D^2` asserts that power and no fit can revise it.

## Naming

- **AMICI reserves seven lowercase names for generated code:** `x` states, `p` parameters,
  `k` constants, `w` expressions, `h` Heaviside, `t` time, `e` Euler's. Never use one for
  a parameter or a species. The clash is INVISIBLE until AMICI compiles — the model parses and
  validates, then the build fails and the whole edit is thrown away, costing a full edit-and-fit
  cycle, not a parse error. `k` and `p` bite most, being the textbook symbols for a rate constant
  and a production rate: write `k_on`, `p_syn`. Uppercase is unaffected — single-letter species
  `A`, `B`, `C`, `D` are fine.
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

## Parse errors that actually happen

Antimony names the unexpected token, not the rule you broke — recognise them by shape.

- **The rate law follows a `;`, never a `:`.** The colon only ends the reaction *name*.
  Wrong: `r1: A -> B : k1*A;` — Right: `r1: A -> B; k1*A;`
- **`param` is not a keyword** (nor `parameter`). A parameter is a plain assignment.
  Wrong: `param k1 = 0.1;` — Right: `k1 = 0.1;`
  (`species`, `compartment`, `var`, `const` ARE real keywords — keep those.) A comma list needs
  one: `k1 = 0.1, k2 = 0.2;` is `unexpected ','`; write `const k1 = 0.1, k2 = 0.2;` or one
  assignment per `;`.
- **`:=` is a RULE (all times); `=` is an initial value.** A derived quantity (a total, a
  normalised signal) is `y := expr;`; with `=` it is a constant and its observable goes flat.
- **There is no `<->` arrow.** Only `->` and `=>`. A reversible step is a net rate law
  (see above). Wrong: `v: A + B <-> C; ...` — Right: `v: A + B -> C; kf*A*B - kr*C`
- **Stoichiometry is a NUMERIC LITERAL after a space — never `*`, never a parameter.**
  Wrong: `r: A -> 2*B;` — Right: `r: A -> 2 B;`
  `A -> A + Y*B` fails to parse; "fixing" `*` to a space PARSES but freezes the coefficient at
  Y's nominal and drops Y from the model. A yield or gain goes in the rate law:
  Wrong: `r: A -> A + Y B; kp*A;` — Right: `r: -> B; Y*kp*A;` (production only — `A` was on
  both sides, so never consumed; its loss stays its own reaction).
- **Reactants and products are species lists, not expressions.** Wrong:
  `v: (A_tot - pA) -> pA;` — algebra goes in the rate law:
  `v: A -> pA; kact*(A_tot - pA)`.
- **Compartment placement is its own statement.** Wrong: `X = 3.64 in cell;` — Right:
  `species X in cell; X = 3.64;`

## Names that are already taken

Beyond the AMICI-reserved names above, Antimony refuses a **built-in function** name. Verified
rejected: `gamma`, `log`, `exp`, `sin`, `abs`, `delay`, `piecewise`. The error (`unexpected name of an
existing function`) does not sound like a name clash. `gamma` bites most — a natural name for a
decay rate; prefix it (`gamma_deg`). `alpha`, `beta`, `kappa` are fine.
