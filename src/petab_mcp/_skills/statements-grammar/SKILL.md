---
name: statements-grammar
route: statements
description: >-
  Read before your first `statements` call. By example: a row and its binding, each verb and tag,
  a condition input read in a binding or seeded at its id, regulators stated and read, the
  mechanism library's rules, a `rate` beside a `scheme` (where `amount()` is required or refused,
  and what multiplies the rate), a site the writer and the reader name alike, and each edit.
---

# Writing rows and bindings

A row is the claim, under a prefix; its binding, under the same prefix, is the maths: the policy
and its parameter bindings.

    statements=["r1: A phosphorylates B at S10"], bindings={"r1": "two_step kf=k_on"}

## A row's verb is its statement type

    E phosphorylates B at S10       Phosphorylation; every modification alike (acetylates...)
    B is phosphorylated at S10      the same, enzyme-less
    E autophosphorylates at Y10     Autophosphorylation
    A binds B                       Complex
    E increases B, B is produced    IncreaseAmount
    E decreases B, B is degraded    DecreaseAmount
    A + B -> C by E                 Conversion; `by E` optional, `nothing` for no product
    A becomes B                     PopulationTransition
    A moves from cytoplasm to nucleus    Translocation
    E activates B, E inhibits B     Activation, Inhibition
    E is a GEF for B, E is a GAP for B   Gef, Gap

No other verb reads: `E converts A to B` is `A -> B by E`.

## An agent's tags, in braces: `A{pS10, kinase:on}`

    pS10, -pS10      phosphorylated at S10, not phosphorylated there; acK5, ubK48 alike
    kinase:on        an activity, or :off
    @nucleus         a location
    !B, ~B           bound to B, not bound to B
    mut:V600E        a mutation

A site is one residue letter and its position (`pRY` is refused). A bound partner is `!B`, never
a bare tag: `A{B}` is refused.

An enzyme without a tag catalyses in every state, `E` and `E{p}` alike. To make the active form
the enzyme, name its state: `E{p} phosphorylates B`.

## A condition input is read in a binding, or is the agent seeded at its id

Read it as a parameter in the binding of each row it drives:

    statements=["r1: B is phosphorylated at S10"], bindings={"r1": 'basal k="k0*dose"'}

Or seed an agent at the input's id alone, which makes that agent the input; rows name the agent:

    statements=["r1: A phosphorylates B at S10"], bindings={"r1": "two_step"},
    seeds=[{"agent": "A", "amount": "dose"}]

Every period sets `A`. It is held while rows only catalyse with it or bind it reversibly, and started
when a row makes, converts, degrades, modifies or binds it one way. A binding reads parameters only:
beside this seed, `k="k0*dose"` is refused.

The binding and the seed at the id follow every period's value. A seed's amount that reads the
input (`2*dose`) takes the first period's value alone, the pre-equilibration one where there is
one. An agent `dose` with no seed at its id (`dose phosphorylates B`) is a molecule of its own,
which the condition table never sets.

## Regulators: stated in the row, read by its rate

    "r1: E phosphorylates S at T20, inhibited by C and D"

A row's mechanism reads only the regulators the row states, and must read every one it states:
`inhibited by C` in the row and `inhibited_by` in the rate or scheme, or neither. No library entry
reads one. Several of one kind are one clause, in JSON a list, `"inhibited_by": [{"name": "C"},
{"name": "D"}]`, which a rate reads by position, `inhibited_by_0` and `inhibited_by_1`. Never
split the row into one per regulator.

## The mechanism library

A policy is one statement type's, and `list_mechanisms` lists each type's with the parameters
each binds: `basal` is a modification's (`B is phosphorylated`), and a DecreaseAmount has no
`mass_action`.

- `based_on` names an entry of the mechanism's own `statement_type`.
- A `rate` re-rates exactly one irreversible reaction: `indra_two_step` builds three, so a rate
  based on it is refused; write a `scheme`.
- A binding binds only its mechanism's own parameters: `mm Km=K1` for the `mm` below, never
  `mm K=K1`.
- A definition you change takes a new name, and none takes a built-in's (`basal`).

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

## Edits

    {"op": "add_row", "row": "r2: B is degraded", "binding": "release"}
    {"op": "replace_row", "row": "r2: E decreases B"}
    {"op": "rebind", "prefix": "r2", "binding": "indra_default"}
    {"op": "remove_row", "prefix": "r2"}
    {"op": "set_seed", "seed": {"agent": "A{pY10}", "amount": 0.5}}
    {"op": "remove_seed", "name": "A_pY10"}

`add_row` takes a new prefix; `replace_row` (its binding kept unless it gives one), `rebind` and
`remove_row` an existing one. A `row` or `binding` is one line or its JSON.

A seed is what is present at t = 0, and how much; beside `statements`, `seeds=[{"agent":
"A{pY10}", "amount": 0.5}]`. Its fields nest under `seed`: `{"op": "set_seed", "agent": "A",
"amount": 1}` is refused. It is named after its agent's form (`A_pY10`) unless it gives `name`.
`set_amount` seeds nothing: `{"op": "set_amount", "name": "pA", "agent": "A{pY10}"}` names an
amount a readout formula reads as `pA`.
