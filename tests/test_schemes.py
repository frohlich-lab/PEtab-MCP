"""Schemes: a mechanism as PySB rules over a statement's roles (:mod:`petab_mcp.schemes`).

The notation both reads the library and writes into it, so the two must invert each other: an
entry's description, compiled as a written mechanism, builds the entry's own rules. Checked for
every entry on its placeholder statement, and for a spread of entries on statements of real
agents, where sites are named after the partners and the statement's own residue.

What it does NOT promise: a row agent's own written states (``AKT1{pS473}``) apply to every
occurrence of it in a scheme, where INDRA's entries apply them to some rules and not others (its
enzyme-substrate release ignores them).
"""

from __future__ import annotations

import re

import pytest
from indra.statements import (
    Activation,
    ActivityCondition,
    Agent,
    BoundCondition,
    Complex,
    Conversion,
    Dephosphorylation,
    IncreaseAmount,
    Inhibition,
    ModCondition,
    Phosphorylation,
)

from petab_mcp import mechanisms as M
from petab_mcp import schemes as S
from petab_mcp.errors import MechanismError


def _canon(model, emission):
    """The rules an emission built, comparable across models: each side's complexes as sets of
    molecules with their site conditions, an explicit `None` complex dropped (PySB builds the same
    reaction either way)."""

    def side(rp):
        return sorted(
            tuple(
                sorted(
                    (mp.monomer.name, tuple(sorted((k, str(v)) for k, v in mp.site_conditions.items())))
                    for mp in cp.monomer_patterns
                )
            )
            for cp in rp.complex_patterns
            if cp is not None
        )

    out = []
    for name in emission.rules:
        rule = model.rules[name]
        rates = [rule.rate_forward] + ([rule.rate_reverse] if rule.is_reversible else [])
        out.append(
            (
                side(rule.rule_expression.reactant_pattern),
                side(rule.rule_expression.product_pattern),
                rule.is_reversible,
                rule.delete_molecules,
                tuple(type(r).__name__ for r in rates),
            )
        )
    return sorted(out)


def _round_trip(make, stmt_type, policy):
    text = S.describe(stmt_type, policy, M.REGISTRY)
    library = dict(M.REGISTRY)
    library[(stmt_type, "written")] = M.written(
        {"statement_type": stmt_type, "policy": "written", "scheme": text,
         "parameters": S.rate_parameters(text)},
        library,
    )
    original, (e1,) = M.assemble([(make(), policy, "r")], {}, M.REGISTRY)
    compiled, (e2,) = M.assemble([(make(), "written", "r")], {}, library)
    return text, _canon(original, e1), _canon(compiled, e2)


_DESCRIBABLE = []
for _key in sorted(M.REGISTRY):
    try:
        S.describe(*_key, M.REGISTRY)
    except MechanismError:
        continue
    _DESCRIBABLE.append(_key)


def test_the_library_is_describable_but_for_abstract_types():
    undescribed = {k for k in M.REGISTRY if k not in _DESCRIBABLE}
    assert {t for t, _ in undescribed} <= S.ABSTRACT | {"ActiveForm"}  # ActiveForm builds no rule
    assert len(_DESCRIBABLE) > 150


@pytest.mark.parametrize("key", _DESCRIBABLE, ids=["/".join(k) for k in _DESCRIBABLE])
def test_every_entry_round_trips_on_its_placeholder(key):
    text, original, compiled = _round_trip(
        lambda: S.placeholder(key[0], enzyme=key[1] != M.BASAL), *key
    )
    assert compiled == original, text


_PHOSPHO = lambda: Phosphorylation(Agent("EGFR"), Agent("AKT1"), "S", "473")  # noqa: E731
_DEPHOSPHO = lambda: Dephosphorylation(Agent("PTEN"), Agent("AKT1"), "S", "473")  # noqa: E731
_REAL = [
    (_PHOSPHO, p)
    for p in ("indra_two_step", "indra_default", "indra_michaelis_menten", "processive",
              "indra_interactions_only")
] + [
    (_DEPHOSPHO, p) for p in ("indra_two_step", "dissociative", "indra_default")
] + [
    # `basal` refuses a statement that names an enzyme.
    (lambda: Phosphorylation(None, Agent("AKT1"), "S", "473"), "basal"),
    (lambda: Dephosphorylation(None, Agent("AKT1"), "S", "473"), "basal"),
] + [
    (lambda: Complex([Agent("EGF"), Agent("EGFR")]), p)
    for p in ("indra_default", "stated", "assembly", "nondepleting")
] + [
    (lambda: Activation(Agent("RAF1"), Agent("MAP2K1")), p)
    for p in ("indra_default", "indra_michaelis_menten", "indra_interactions_only")
] + [
    (lambda: IncreaseAmount(Agent("TF"), Agent("GENE")), p)
    for p in ("indra_default", "mass_action", "indra_hill", "self_renewal")
] + [
    (lambda: Conversion(Agent("E"), [Agent("A")], [Agent("B")]), p)
    for p in ("mass_action", "release", "indra_default")
]


@pytest.mark.parametrize("make, policy", _REAL, ids=[f"{m().__class__.__name__}/{p}" for m, p in _REAL])
def test_entries_round_trip_on_real_agents(make, policy):
    """Sites named after the partners (`akt1`, `egfr`) and the statement's residue (`S473`)."""
    text, original, compiled = _round_trip(make, type(make()).__name__, policy)
    assert compiled == original, text


def test_two_step_reads_as_pysb():
    assert S.describe("Phosphorylation", "indra_two_step", M.REGISTRY).splitlines() == [
        "enz(sub=None) + sub(mod='off', enz=None) >> enz(sub=1) % sub(mod='off', enz=1) @ kf",
        "enz(sub=1) % sub(mod='off', enz=1) >> enz(sub=None) + sub(mod='on', enz=None) @ kc",
        "enz(sub=1) % sub(enz=1) >> enz(sub=None) + sub(enz=None) @ kr",
    ]


def test_a_scheme_writes_a_shape_no_entry_has():
    """An ordered two-site phosphorylation: the enzyme binds, sets S473, then T308 while still
    bound, then lets go. No library entry modifies two sites in one binding."""
    scheme = """
        enz(sub=None) + sub(enz=None, mod='off') | enz(sub=1) % sub(enz=1, mod='off') @ kf, kr
        enz(sub=1) % sub(enz=1, mod='off', T308='u') >> enz(sub=1) % sub(enz=1, mod='on', T308='u') @ kc1
        enz(sub=1) % sub(enz=1, mod='on', T308='u') >> enz(sub=None) + sub(enz=None, mod='on', T308='p') @ kc2
    """
    library = dict(M.REGISTRY)
    library[("Phosphorylation", "ordered")] = M.written(
        {"statement_type": "Phosphorylation", "policy": "ordered", "scheme": scheme,
         "parameters": ["kf", "kr", "kc1", "kc2"]},
        library,
    )
    model, (emission,) = M.assemble([(_PHOSPHO(), "ordered", "r1")], {}, library)
    assert len(emission.rules) == 3
    assert set(emission.parameters) == {"r1_kf", "r1_kr", "r1_kc1", "r1_kc2"}
    assert model.monomers["AKT1"].site_states["T308"] == ["u", "p"]
    assert emission.sites_added["AKT1"]["T308"] == ["u", "p"]
    assert model.rules[emission.rules[0]].is_reversible


def test_a_rate_may_read_amounts_and_time():
    scheme = "enz() + sub(mod='off') >> enz() + sub(mod='on') @ kc/(Km + amount(sub(mod='off'))) * exp(-time/tau)"
    library = dict(M.REGISTRY)
    library[("Phosphorylation", "decaying_mm")] = M.written(
        {"statement_type": "Phosphorylation", "policy": "decaying_mm", "scheme": scheme,
         "parameters": ["kc", "Km", "tau"]},
        library,
    )
    model, (emission,) = M.assemble([(_PHOSPHO(), "decaying_mm", "r1")], {}, library)
    assert set(emission.parameters) == {"r1_kc", "r1_Km", "r1_tau"}
    expr = model.rules[emission.rules[0]].rate_forward.expr
    assert "time" in str(expr) and "amount" in str(expr)


def _inhibited(scheme: str, stmt):
    library = dict(M.REGISTRY)
    library[("Phosphorylation", "inhibited")] = M.written(
        {"statement_type": "Phosphorylation", "policy": "inhibited", "scheme": scheme,
         "parameters": ["kf", "Ki"]},
        library,
    )
    return M.assemble([(stmt, "inhibited", "r1")], {}, library)


@pytest.mark.parametrize(
    "scheme",
    [
        "enz() + sub(mod='off') >> enz() + sub(mod='on') @ kf/(1 + amount(PTEN)/Ki)",
        "enz() + sub(mod='off') >> enz() + sub(mod='on') @ kf/(1 + amount(PTEN{pS380})/Ki)",
        # Mass action reads a reactant, and this one goes back unchanged.
        "enz() + sub(mod='off') + PTEN() >> enz() + sub(mod='on') + PTEN() @ kf/Ki",
    ],
)
def test_a_scheme_reads_no_agent_its_statement_does_not_name(scheme):
    """PEtab-MCP#122: `EGFR phosphorylates AKT1` read PTEN in its rate, so the statement said
    nothing of what slowed it. Each form is refused naming PTEN and the statement that states it."""
    with pytest.raises(MechanismError) as refused:
        _inhibited(scheme, _PHOSPHO())
    assert re.search(
        r"row 'r1': mechanism 'inhibited' reads PTEN.*`EGFR phosphorylates AKT1 at S473, "
        r"inhibited by PTEN(\{pS380\})?`.*read as `amount\(inhibited_by\)`",
        str(refused.value),
    )


def test_a_scheme_reads_a_regulator_its_statement_states():
    """`amount(inhibited_by)` reads the agent the statement says slows the reaction, as it states
    it."""
    p_pten = Agent("PTEN", mods=[ModCondition("phosphorylation", "S", "380")])
    library = dict(M.REGISTRY)
    library[("Phosphorylation", "inhibited")] = M.written(
        {"statement_type": "Phosphorylation", "policy": "inhibited", "parameters": ["kf", "Ki"],
         "scheme": "enz() + sub(mod='off') >> enz() + sub(mod='on') @ "
                   "kf/(1 + amount(inhibited_by)/Ki)"},
        library,
    )
    model, _ = M.assemble(
        [(_PHOSPHO(), "inhibited", "r1", None, {"inhibited_by": [p_pten]})], {}, library
    )
    (amount,) = (o for o in model.observables if o.name.endswith("inhibited_by"))
    assert str(amount.reaction_pattern) == "PTEN(S380=('p', WILD))"


def test_a_scheme_reads_a_partner_its_statement_names():
    """The same rate, with EGFR stated as not bound to PTEN: `amount(PTEN)` reads PTEN as the
    statement writes it, phosphorylated."""
    p_pten = Agent("PTEN", mods=[ModCondition("phosphorylation", "S", "380")])
    stmt = Phosphorylation(
        Agent("EGFR", bound_conditions=[BoundCondition(p_pten, False)]), Agent("AKT1"), "S", "473"
    )
    model, (emission,) = _inhibited(
        "enz() + sub(mod='off') >> enz() + sub(mod='on') @ kf/(1 + amount(PTEN)/Ki)", stmt
    )
    (amount,) = (o for o in model.observables if "amount" in o.name)
    assert str(amount.reaction_pattern) == "PTEN(S380='p')"


#: A drug that sequesters the enzyme, which acts only free of it.
_SEQUESTERED = (
    "enz(inhibited_by=None) + inhibited_by(enz=None) | enz(inhibited_by=1) % inhibited_by(enz=1) "
    "@ kon, koff\n"
    "enz(inhibited_by=None) + sub(mod='off') >> enz(inhibited_by=None) + sub(mod='on') @ kf"
)


def _bound(scheme: str, stated: dict):
    library = dict(M.REGISTRY)
    library[("Phosphorylation", "bound")] = M.written(
        {"statement_type": "Phosphorylation", "policy": "bound", "scheme": scheme,
         "parameters": S.rate_parameters(scheme)},
        library,
    )
    return M.assemble([(_PHOSPHO(), "bound", "r1", None, stated)], {}, library), library


def test_a_scheme_binds_a_regulator_its_statement_states():
    """PEtab-MCP#140: `E + I <-> EI` was writable only with the drug a partner (`EGFR{~drug}`),
    which dropped the claim that it inhibits. Bound as `inhibited_by`, it is the agent the
    statement states, in the states it writes, and its sites are named as the partner notation
    names them, so the complex is the one `EGFR{!drug}` reads."""
    drug = Agent("drug", mods=[ModCondition("phosphorylation", "S", "10")])
    (model, (emission,)), library = _bound(_SEQUESTERED, {"inhibited_by": [drug]})
    assert [str(model.rules[r].rule_expression) for r in emission.rules] == [
        "EGFR(drug=None) + drug(S10='p', egfr=None) | EGFR(drug=1) % drug(S10='p', egfr=1)",
        "EGFR(drug=None) + AKT1(S473='u') >> EGFR(drug=None) + AKT1(S473='p')",
    ]
    assert emission.regulator_molecules == ("inhibited_by",)
    assert S.describe("Phosphorylation", "bound", library) == _SEQUESTERED


@pytest.mark.parametrize(
    "scheme, match",
    [
        # The issue's own sketch: EGFR's site `inh` is none `EGFR{!drug}` reads.
        (_SEQUESTERED.replace("enz(inhibited_by=", "enz(inh="),
         r"bond 1 joins `enz\(inh=1\)` and `inhibited_by\(enz=1\)`: a regulator binds on sites "
         r"named after each other, `enz\(inhibited_by=1\) % inhibited_by\(enz=1\)`"),
        (_SEQUESTERED.replace("inhibited_by(enz=None)", "inhibited_by(enz=None, S10='p')"),
         r"`inhibited_by\(S10='p'\)` writes a state on a regulator: .* a state it needs is "
         r"written on the statement's agent \(`inhibited by X\{pS10\}`\)"),
        (_SEQUESTERED.replace("@ kf", "@ kf*amount(inhibited_by(mod='on'))"),
         r"`inhibited_by\(mod='on'\)` writes a state on a regulator"),
    ],
)
def test_a_regulators_molecule_has_binding_sites_named_after_its_partners_only(scheme, match):
    """A state on the molecule would read a form of the regulator the statement does not state,
    and a bond on sites named otherwise forms a complex other rows cannot read as `E{!X}`."""
    with pytest.raises(MechanismError, match=match):
        _bound(scheme, {"inhibited_by": [Agent("drug")]})


def test_a_regulator_bound_by_its_own_name_or_unstated_is_refused_naming_its_kind():
    with pytest.raises(MechanismError, match=r"reads drug as an agent of its own.* the molecule "
                       r"`inhibited_by` in a scheme's rules, which bind it on sites named after "
                       r"each other \(`enz\(inhibited_by=1\) % inhibited_by\(enz=1\)`\)"):
        _bound(_SEQUESTERED.replace("inhibited_by", "drug"), {"inhibited_by": [Agent("drug")]})
    with pytest.raises(MechanismError, match=r"reads \['inhibited_by'\], which its statement does "
                       r"not state"):
        _bound(_SEQUESTERED, {})


@pytest.mark.parametrize(
    "scheme, match",
    [
        ("enz() + sub() >> enz() + sub()", "rate after `@`"),
        ("enz() + sub() | enz() % sub() @ kf", "takes 2 rate"),
        ("enz() + sub(mod='half') >> enz() + sub(mod='on') @ kf", "'off' or 'on'"),
        ("enz() + sub(mod=1.5) >> enz() + sub() @ kf", "site value"),
        ("enz() + sub() >> enz() + sub() @ hill(kf)", "no function"),
        ("enz() + sub() + enz() @ kf", "reactants >> products"),
        ("enz() + sub() >> enz() + sub() @ kf | kf", "is not allowed"),
        ("enz() + sub() >> enz() + sub() @ kf*exp()", "takes one argument"),
        # A bond across `+` reached the agent as PySB's own "Dangling bond(s) [1] in TGFbR(b=1)".
        ("enz(b=None) + sub(e=None) >> enz(b=1) + sub(e=1) @ kf",
         r"bond 1 joins `enz\(b=1\)` and `sub\(e=1\)`, which `\+` writes as two complexes: join "
         r"them with `%`, `enz\(b=1\) % sub\(e=1\)`"),
        ("enz(b=None) + sub(e=None) >> enz(b=None) % sub(e=1) @ kf",
         r"bond 1 is on 1 site\(s\) of its complex \(sub\(e=1\)\), not two"),
        ("enz() + sub() >> enz() + sub() @ kf*amount(sub(e=1))",
         r"rate 'kf\*amount\(sub\(e=1\)\)': bond 1 is on 1 site"),
    ],
)
def test_a_scheme_that_does_not_read_is_refused(scheme, match):
    with pytest.raises(MechanismError, match=match):
        library = dict(M.REGISTRY)
        library[("Phosphorylation", "bad")] = M.written(
            {"statement_type": "Phosphorylation", "policy": "bad", "scheme": scheme,
             "parameters": ["kf"]},
            library,
        )
        M.assemble([(_PHOSPHO(), "bad", "r1")], {}, library)


@pytest.mark.parametrize(
    "scheme, match",
    [
        # AKT1's S473 takes 'u' and 'p': a rule needing 'x' never fires.
        ("enz() + sub(S473='x') >> enz() + sub(S473='p') @ kf",
         r"state AKT1's site 'S473' does not take; AKT1's sites are S473 \['u', 'p'\]"),
        # Nothing gives EGFR a kinase activity, so nothing would ever make it active.
        ("enz(act='active') + sub(mod='off') >> enz(act='active') + sub(mod='on') @ kf",
         "site 'kinase', which EGFR does not have; EGFR has no sites"),
        # A site added with one state starts every AKT1 phosphorylated at T308.
        ("enz() + sub(mod='off', T308='p') >> enz() + sub(mod='on', T308='p') @ kf",
         "adds site 'T308' to AKT1 with the one state 'p'"),
        # Slips of AKT1's own S473, each a new two-state site the rules would move alone.
        ("enz() + sub(mod='off', s473='u') >> enz() + sub(mod='on', s473='p') @ kf",
         r"adds site 's473' to AKT1: did you mean 'S473'\? AKT1's sites are S473"),
        ("enz() + sub(mod='off', pS473='u') >> enz() + sub(mod='on', pS473='p') @ kf",
         "did you mean 'S473'"),
    ],
)
def test_a_scheme_names_only_states_its_molecules_take(scheme, match):
    """A rule naming a state its molecule never takes never fires, and nothing said so."""
    library = dict(M.REGISTRY)
    library[("Phosphorylation", "bad")] = M.written(
        {"statement_type": "Phosphorylation", "policy": "bad", "scheme": scheme, "parameters": ["kf"]}, library
    )
    with pytest.raises(MechanismError, match=match):
        M.assemble([(_PHOSPHO(), "bad", "r1")], {}, library)


def _assemble_written(scheme, entities=None):
    library = dict(M.REGISTRY)
    library[("Phosphorylation", "w")] = M.written(
        {"statement_type": "Phosphorylation", "policy": "w", "scheme": scheme, "parameters": ["kf"]}, library
    )
    return M.assemble([(_PHOSPHO(), "w", "r1")], entities or {}, library)


def test_numbered_sites_a_scheme_adds_are_not_slips_of_each_other():
    """`dom1` and `dom2` differ by one character but are two sites, not a slip of one name."""
    model, _ = _assemble_written(
        "enz() + sub(mod='off', dom1='a', dom2='a') >> enz() + sub(mod='on', dom1='b', dom2='b') @ kf"
    )
    assert model.monomers["AKT1"].site_states["dom2"] == ["a", "b"]


def test_a_state_on_a_binding_site_is_refused_naming_it_one():
    entities = {"AKT1": Agent("AKT1", bound_conditions=[BoundCondition(Agent("PDK1"))])}
    with pytest.raises(MechanismError, match=r"site 'pdk1' does not take; .*pdk1 \(binding\)"):
        _assemble_written("enz() + sub(pdk1='x') >> enz() + sub(pdk1='x') @ kf", entities)


@pytest.mark.parametrize(
    "agent, scheme, match",
    [
        # `act` on a phosphorylation's enzyme is its kinase activity; the agents had written it
        # for the one activity EGFR has, and the message named both sites but not what to write.
        (Agent("EGFR", activity=ActivityCondition("activity", True)),
         "enz(act='active') + sub(mod='off') >> enz(act='active') + sub(mod='on') @ kf",
         r"site 'kinase', which EGFR does not have; EGFR's sites are activity \['inactive', "
         r"'active'\]. Write `enz\(activity='active'\)` instead"),
        (Agent("EGFR", mods=[ModCondition("phosphorylation", "Y", "1068")]),
         "enz(mod='on') + sub(mod='off') >> enz(mod='on') + sub(mod='on') @ kf",
         r"site 'S473', which EGFR does not have; .* Write `enz\(Y1068='p'\)` instead"),
    ],
)
def test_a_site_the_molecule_lacks_names_the_one_to_write(agent, scheme, match):
    with pytest.raises(MechanismError, match=match):
        _assemble_written(scheme, {"EGFR": agent})


def test_mod_outside_a_modification_names_the_agents_own_site():
    """`subj(mod='on')` on an Inhibition, meaning the phosphorylated subject, was refused as "`mod`
    is a modification statement's own site", with nothing on how to say it."""
    irs1 = Agent("IRS1", mods=[ModCondition("phosphorylation", "Y", "612")])
    library = dict(M.REGISTRY)
    library[("Inhibition", "w")] = M.written(
        {"statement_type": "Inhibition", "policy": "w", "parameters": ["kf"],
         "scheme": "subj(mod='on') + obj(act='active') >> subj(mod='on') + obj(act='inactive') @ kf"},
        library,
    )
    with pytest.raises(
        MechanismError,
        match=r"Inhibition is none. Write the site itself, as `subj\(Y612='p'\)`, or the "
        r"modification on the row's agent, .*; IRS1's sites are Y612 \['u', 'p'\]",
    ):
        M.assemble([(Inhibition(irs1, Agent("PI3K")), "w", "r1")], {}, library)


def _scheme(rate, parameters=("kf",), problem_ids=None):
    """A phosphorylation at ``rate``, written as a scheme declaring ``parameters``."""
    return M.written(
        {"statement_type": "Phosphorylation", "policy": "s", "parameters": list(parameters),
         "scheme": f"enz() + sub(mod='off') >> enz() + sub(mod='on') @ {rate}"},
        M.REGISTRY,
        problem_ids,
    )


def test_a_scheme_rate_names_only_the_parameters_it_declares():
    """A scheme minted every name it met, so a typo or a problem's own id became a new constant
    fitted in its place."""
    with pytest.raises(MechanismError, match=r"names \['kff'\].*Did you mean: kff -> 'kf'"):
        _scheme("kff")
    with pytest.raises(MechanismError, match=r"`parameters`"):
        _scheme("kf*kr")
    with pytest.raises(MechanismError, match=r"declares \['kr'\].*never names"):
        _scheme("kf", ("kf", "kr"))


def test_a_scheme_rate_reads_a_role_only_as_its_amount():
    with pytest.raises(MechanismError, match=r"`sub` is a role: read its amount, `amount\(sub\)`"):
        _scheme("kf*sub")


def test_a_scheme_rate_does_not_read_a_starting_amount():
    """A scheme minted `@ kf*EGF_init` as a constant `r1_EGF_init`, fitted apart from EGF's start."""
    with pytest.raises(MechanismError, match=r"\['EGF_init'\].*`amount\(<pattern>\)`"):
        _scheme("kf*EGF_init")


_IDS = {"d": M.SHIPPED, "EGF_0": M.CONDITION, "pERK": M.SPECIES}


@pytest.mark.parametrize(
    "name, match",
    [
        ("d", r"`d` is a parameter the problem ships\. A problem parameter or condition input "
              r"enters through a row's binding"),
        ("EGF_0", r"`EGF_0` is a condition input\. .*`K=EGF_0` "
                  r"\(`bindings=\{\"r1\": \"<policy> K=EGF_0\"\}`\)"),
        ("pERK", r"`pERK`: a species of the entity map, read as the amount of the agent it maps "
                 r"to, written by that agent's own name and sites, not by this id"),
    ],
)
def test_a_problem_id_in_a_scheme_rate_says_how_it_enters(name, match):
    """A shipped FIXED `d`, written in a scheme's rate, became a new estimated `r_d`."""
    with pytest.raises(MechanismError, match=match):
        _scheme(f"kf*{name}", problem_ids=_IDS)


def test_a_problem_id_gets_only_its_own_remedy():
    """A shipped start, `A_init`, was offered the binding, which then failed where the seed makes
    `A_init` again; a shipped `kd` beside a declared `k` was offered "did you mean 'k'"."""
    ids = {"A_init": M.SHIPPED, "kd": M.SHIPPED}
    with pytest.raises(MechanismError) as refused:
        _scheme("kf*A_init", problem_ids=ids)
    assert "starts at" in str(refused.value) and "binding" not in str(refused.value)
    with pytest.raises(MechanismError) as refused:
        _scheme("k*kd", ("k",), ids)
    assert "binding" in str(refused.value) and "Did you mean" not in str(refused.value)


def test_a_declared_parameter_may_not_take_a_problem_id():
    with pytest.raises(MechanismError, match=r"declares \['d'\].*name yours apart.*row's binding"):
        _scheme("kf*d", ("kf", "d"), _IDS)


def test_a_scheme_and_a_rate_together_are_refused():
    with pytest.raises(MechanismError, match="scheme and a rate"):
        M.written(
            {"statement_type": "Phosphorylation", "policy": "both", "rate": "k",
             "scheme": "enz() + sub() >> enz() + sub() @ k"},
            M.REGISTRY,
        )


def test_a_builtin_name_is_refused_for_a_scheme():
    with pytest.raises(MechanismError, match="built-in"):
        M.written(
            {"statement_type": "Phosphorylation", "policy": "indra_two_step",
             "scheme": "enz() + sub() >> enz() + sub() @ k"},
            M.REGISTRY,
        )
