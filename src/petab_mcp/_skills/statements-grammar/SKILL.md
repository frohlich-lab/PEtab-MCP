---
name: statements-grammar
route: statements
description: >-
  Read before your first `statements` call. By example: a row and its binding, a condition input
  read in a binding or seeded at its id, several regulators of one kind, a `rate` beside a
  `scheme` (where `amount()` is required or refused, and what multiplies the rate), a site the
  writer and the reader name alike, and `seeds` beside `set_amount`.
---

# Writing rows and bindings

A row is the claim, under a prefix; its binding, under the same prefix, is the maths: the policy
and its parameter bindings.

    statements=["r1: A phosphorylates B at S10"], bindings={"r1": "two_step kf=k_on"}

## A condition input is read in a binding, or is the agent seeded at its id

Read it as a parameter in the binding of each row it drives:

    statements=["r1: B is phosphorylated at S10"], bindings={"r1": 'basal k="k0*dose"'}

Or seed an agent at the input's id alone, which makes that agent the input; rows name the agent:

    statements=["r1: A phosphorylates B at S10"], bindings={"r1": "two_step"},
    seeds=[{"agent": "A", "amount": "dose"}]

Every period sets `A`. It is held while the rows only bind it or catalyse with it, and started when
a row makes, converts or degrades it. A binding reads parameters only: beside this seed,
`k="k0*dose"` is refused.

The binding and the seed at the id follow every period's value. A seed's amount that reads the
input (`2*dose`) takes the first period's value alone, the pre-equilibration one where there is
one. An agent `dose` with no seed at its id (`dose phosphorylates B`) is a molecule of its own,
which the condition table never sets.

## Several regulators of one kind are one clause

    "r1: E phosphorylates S at T20, inhibited by C and D"

In JSON each kind is a list, `"inhibited_by": [{"name": "C"}, {"name": "D"}]`; a rate reads them
by position, `inhibited_by_0` and `inhibited_by_1`. Never split the row into one per regulator.

## `rate` and `scheme`: two languages

| | `rate` | `scheme` |
|---|---|---|
| reaction | an existing entry's (`based_on`) | the rules you write |
| an amount | bare: `sub`, `enz`, `inhibited_by` | `amount(sub)`, `amount(inhibited_by)` |
| `amount()` | refused | required |
| `piecewise` | allowed | refused |

The same inhibited Michaelis-Menten step for `r1: E phosphorylates S at T20, inhibited by C`, each
way:

    {"statement_type": "Phosphorylation", "policy": "mm", "based_on": "indra_default",
     "parameters": ["kcat", "Km", "Ki"], "rate": "kcat/(Km + sub)/(1 + inhibited_by/Ki)"}
    {"statement_type": "Phosphorylation", "policy": "mm_s", "parameters": ["kcat", "Km", "Ki"],
     "scheme": "enz() + sub(mod='off') >> enz() + sub(mod='on') @ kcat/(Km + amount(sub))/(1 + amount(inhibited_by)/Ki)"}

Both rates are MULTIPLIED by the reaction's reactants, the enzyme included: each gives
`kcat·E·S/((Km + S)(1 + C/Ki))`. A rate written `kcat*enz/(Km + sub)` counts the enzyme twice; a
constant `k` alone is mass action.

## A site is the one its statement names

`at S10` puts the phosphate on site `S10`, and a statement naming no residue (`B{p}`) on site
`phospho`. A row reading `B{pS10}` never sees the second: give the writing and the reading rows
the same residue and position, or neither. A seed does not join two sites.

## `seeds` and `set_amount`

`seeds` is what is present at t = 0, and how much: `seeds=[{"agent": "A{pY10}", "amount": 0.5}]`,
or as an edit `{"op": "set_seed", "seed": {"agent": "A{pY10}", "amount": 0.5}}`. `set_amount`
seeds nothing: `{"op": "set_amount", "name": "pA", "agent": "A{pY10}"}` names an amount a readout
formula reads as `pA`.
