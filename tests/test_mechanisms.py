"""The mechanism registry and the assembler that applies it."""

from __future__ import annotations

import json

import pysb
import pytest
from indra.statements import (
    Activation,
    ActiveForm,
    Agent,
    Dephosphorylation,
    ModCondition,
    Phosphorylation,
)

from petab_mcp import mechanisms as M
from petab_mcp.errors import MechanismError
from petab_mcp.statements import PopulationTransition, ungrounded

_PHOSPHO = Phosphorylation(Agent("MEK", db_refs={"HGNC": "6840"}), Agent("ERK"), "T", "185")
_TRANSITION = PopulationTransition(Agent("naive"), Agent("effector"))


def test_emission_lists_every_parameter_the_model_holds():
    """The PEtab parameter table is built from emissions, not by scanning the model, so a
    parameter a policy mints without listing it would never be estimated."""
    model, emissions = M.assemble(
        [(_PHOSPHO, M.INDRA_DEFAULT, "r1"), (_TRANSITION, "mass_action", "r2")]
    )
    emitted = {p for e in emissions for p in e.parameters}
    assert emitted == {p.name for p in model.parameters}
    assert {r for e in emissions for r in e.rules} == {r.name for r in model.rules}


def test_ungrounded_agent_assembles():
    """Grounding is optional: a population with no ``db_refs`` is reported, never rejected."""
    assert ungrounded(_TRANSITION) == ["naive", "effector"]
    model, _ = M.assemble([(_TRANSITION, "mass_action", "r1")])
    assert {m.name for m in model.monomers} == {"naive", "effector"}


def test_modified_substrate_gets_its_site():
    """INDRA's monomer pass, not the policy, declares the phospho site; without it the rule's
    pattern names a site the monomer does not have."""
    model, _ = M.assemble([(_PHOSPHO, M.INDRA_DEFAULT, "r1")])
    assert model.monomers["ERK"].site_states == {"T185": ["u", "p"]}


@pytest.mark.parametrize("policy", [M.INDRA_DEFAULT, "indra_two_step"])
def test_unregistered_pair_is_a_mechanism_error(policy):
    """A library gap must stay distinguishable from a bug, so it is not a bare ``KeyError``. It is
    a policy another statement type has, or one INDRA implements, asked of a type that lacks it."""
    with pytest.raises(MechanismError, match="no mechanism") as exc:
        M.assemble([(_TRANSITION, policy, "r1")])
    assert exc.value.cause == "library_gap"


def test_a_misspelt_policy_is_malformed_not_a_library_gap():
    """`indra_defualt` was refused as "no mechanism" and counted as a library gap, so typos
    inflated the library's to-do list. It is malformed, and says what it is closest to: the type's
    own policies first, then the others, saying the type lacks them."""
    with pytest.raises(MechanismError, match="did you mean 'indra_default'") as exc:
        M.assemble([(_PHOSPHO, "indra_defualt", "r1")])
    assert exc.value.cause == "malformed_mechanism"
    with pytest.raises(MechanismError, match="'mass_action', which Phosphorylation lacks") as exc:
        M.assemble([(_PHOSPHO, "mass_actoin", "r1")])
    assert exc.value.cause == "malformed_mechanism"
    assert "write one with `mechanisms`" in str(exc.value)


def test_a_malformed_request_is_not_a_library_gap():
    """Every refusal counted as a library gap, so the library's to-do list was mostly typos."""
    with pytest.raises(MechanismError, match="built-in") as exc:
        M.written(
            {"statement_type": "PopulationTransition", "policy": "mass_action", "rate": "k"},
            M.REGISTRY,
        )
    assert exc.value.cause == "malformed_mechanism"


def test_basal_refuses_a_statement_that_names_an_enzyme():
    """`PTEN dephosphorylates AKT1 | basal` built a rule without PTEN."""
    stmt = Dephosphorylation(Agent("PTEN"), Agent("AKT1"), "S", "473")
    with pytest.raises(MechanismError, match="names PTEN, which it would leave out") as exc:
        M.assemble([(stmt, M.BASAL, "r1")])
    assert exc.value.cause == "malformed_mechanism"


def test_basal_addition_refuses_an_enzyme_too_and_builds_without_one():
    with pytest.raises(MechanismError, match="names MEK, which it would leave out"):
        M.assemble([(_PHOSPHO, M.BASAL, "r1")])
    _, (emission,) = M.assemble([(Phosphorylation(None, Agent("ERK"), "T", "185"), M.BASAL, "r1")])
    assert emission.rules


def _akt1(s473: bool) -> Agent:
    """AKT1 written at S473, phosphorylated or not, and at pT308."""
    s473 = ModCondition("phosphorylation", "S", "473", s473)
    return Agent("AKT1", mods=[s473, ModCondition("phosphorylation", "T", "308")])


@pytest.mark.parametrize(
    "stmt, policy, change",
    [
        (Dephosphorylation(None, _akt1(True), "S", "473"), M.BASAL, ["p", "u"]),
        (Dephosphorylation(Agent("PTEN"), _akt1(True), "S", "473"), M.INDRA_DEFAULT, ["p", "u"]),
        (Phosphorylation(None, _akt1(False), "S", "473"), M.BASAL, ["u", "p"]),
        (Phosphorylation(Agent("MTOR"), _akt1(False), "S", "473"), M.INDRA_DEFAULT, ["u", "p"]),
    ],
)
def test_a_substrate_written_in_the_state_it_leaves_still_changes(stmt, policy, change):
    """`AKT1{pS473} is dephosphorylated at S473` built `pS473 >> pS473`: the substrate's written
    site overrode the state the rule sets, under `basal` and INDRA's policies alike. The modified
    site is the rule's, every other written site stays, and the row's statement is untouched."""
    model, _ = M.assemble([(stmt, policy, "r1")])
    (rule,) = model.rules
    sides = [
        M._plain(mp.site_conditions)
        for side in (rule.reactant_pattern, rule.product_pattern)
        for cp in side.complex_patterns
        for mp in cp.monomer_patterns
        if mp.monomer.name == "AKT1"
    ]
    assert [s["S473"] for s in sides] == change
    assert [s["T308"] for s in sides] == ["p", "p"]
    assert len(stmt.sub.mods) == 2


def test_rows_differing_only_in_the_written_modified_site_are_twins():
    """Both build the same INDRA rule, so the second built nothing and was refused as a policy
    that builds no rule, not as the repeat it is."""
    written = Agent("AKT1", mods=[ModCondition("phosphorylation", "S", "473")])
    rows = [
        (Dephosphorylation(Agent("PTEN"), written, "S", "473"), M.INDRA_DEFAULT, "r1"),
        (Dephosphorylation(Agent("PTEN"), Agent("AKT1"), "S", "473"), M.INDRA_DEFAULT, "r2"),
    ]
    with pytest.raises(MechanismError, match="asserts the statement of row 'r1' again"):
        M.assemble(rows)


def test_a_policy_indra_cannot_assemble_is_a_library_gap():
    with pytest.raises(MechanismError, match="INDRA has no default assembly") as exc:
        M._indra("default")(_TRANSITION, pysb.Model(_export=False), "r1")
    assert exc.value.cause == "library_gap"


def test_reregistering_an_entry_raises():
    """A silent override builds whichever policy imported last. A programming error, not a
    refusal: no agent call reaches it."""
    with pytest.raises(RuntimeError, match="already registered"):
        M.mechanism(PopulationTransition, "mass_action")(lambda *a: None)


def test_every_indra_statement_has_its_default_entry():
    """Without INDRA's own assembly as an entry, every INDRA statement but one phosphorylation
    form was a library gap, and a seed of INDRA statements could not be built at all. Its
    parameters carry the row's prefix: INDRA's own `kf_ea_activation_1` names no row, and
    widening one bound meant looking each name up."""
    stmt = Activation(Agent("EGFR"), Agent("AKT"))
    model, (emission,) = M.assemble([(stmt, M.INDRA_DEFAULT, "r1")])
    assert emission.rules and emission.parameters == ("r1_kf",)
    assert {p.name for p in model.parameters} == {"r1_kf"}



def test_a_row_that_builds_no_rule_is_refused():
    """INDRA builds a rule once and none for an ActiveForm, so a repeated statement built nothing
    beside an orphan `r2_kf`, and an ActiveForm row nothing at all, both without a word."""
    activation = Activation(Agent("EGFR"), Agent("AKT"))
    twin = Activation(Agent("EGFR"), Agent("AKT"))
    with pytest.raises(MechanismError, match="row 'r2' asserts the statement of row 'r1' again"):
        M.assemble([(activation, M.INDRA_DEFAULT, "r1"), (twin, M.INDRA_DEFAULT, "r2")])
    with pytest.raises(MechanismError, match="row 'r1': ActiveForm under 'indra_default' builds no rule"):
        M.assemble([(ActiveForm(Agent("AKT"), "kinase", True), M.INDRA_DEFAULT, "r1")])

# A mechanism the agent writes; ``{extra}`` splices more fields into the definition.
_BASE_WRITTEN = (
    '{{"statement_type": "Phosphorylation", "policy": "saturating", "rate": "kcat / (Km + enz)", '
    '"parameters": ["kcat", "Km"]{extra}}}'
)


_BASE_WRITTEN_DEF = json.loads(_BASE_WRITTEN.format(extra=""))


def _written(extra=""):
    library = dict(M.REGISTRY)
    definition = json.loads(_BASE_WRITTEN.format(extra=extra))
    library[("Phosphorylation", definition["policy"])] = M.written(definition, library)
    return library


def test_a_policy_written_for_another_type_is_a_library_gap():
    """What the agent wrote is a policy like the library's own: asked of another type, a gap."""
    with pytest.raises(MechanismError, match="no mechanism") as exc:
        M.assemble([(_TRANSITION, "saturating", "r1")], library=_written())
    assert exc.value.cause == "library_gap"


def test_a_written_mechanism_keeps_the_reaction_and_takes_the_rate():
    """The agent writes a mechanism when none fits; only the rate is its, so what is consumed
    and made — and with it non-negativity — stays the base entry's."""
    model, (emission,) = M.assemble([(_PHOSPHO, "saturating", "r1")], library=_written())
    assert emission.parameters == ("r1_Km", "r1_kcat")
    (rule,) = model.rules
    assert rule.rate_forward.name == "r1_rate"
    assert {s.name for s in rule.rate_forward.expr.free_symbols} == {"r1_kcat", "r1_Km", "r1_enz"}


def test_a_written_rate_names_only_what_the_mechanism_allows():
    """sympy evaluates the text it parses, so a name outside the allowed set — a typo, or
    anything reaching Python — is refused before parsing."""
    for rate in ("kcat / (Km + substrate)", "__import__('os')", "kcat.real"):
        with pytest.raises(MechanismError, match="may name only"):
            _written(f', "rate": "{rate}"')  # a repeated JSON key: the later one wins


def test_a_refused_rate_name_comes_with_the_names_it_could_have_meant():
    with pytest.raises(MechanismError, match=r"Did you mean: kcatt -> 'kcat'"):
        _written(', "rate": "kcatt / (Km + enz)"')


def test_a_written_rate_does_not_read_a_starting_amount():
    """`<name>_init` is a species' start; a rate reads amounts through its roles and modifiers."""
    with pytest.raises(MechanismError, match=r"\['ERK_init'\].*amount a species starts at"):
        _written(', "rate": "kcat * ERK_init / (Km + enz)"')


def test_a_problem_id_in_a_written_rate_says_how_it_enters():
    """A shipped FIXED `d` was refused with only the names a rate may read, so the agent declared
    `d` and got a new estimated one."""
    ids = {"d": M.SHIPPED}
    with pytest.raises(MechanismError, match=r"`d` is a parameter the problem ships.*row's binding"):
        M.written(_BASE_WRITTEN_DEF | {"rate": "kcat * d / (Km + enz)"}, M.REGISTRY, ids)
    with pytest.raises(MechanismError, match=r"declares \['d'\].*name yours apart"):
        M.written(_BASE_WRITTEN_DEF | {"parameters": ["kcat", "Km", "d"]}, M.REGISTRY, ids)


@pytest.mark.parametrize("rate", ["kcat,Km", "exp", "2kcat", "kcat Km", "exp()"])
def test_a_written_rate_that_is_no_expression_is_refused(rate):
    """Each passed the token check and crashed make_changes in sympify, at assembly."""
    with pytest.raises(MechanismError, match="does not read as one expression"):
        _written(f', "rate": "{rate}"')


@pytest.mark.parametrize(
    "extra, match",
    [
        ("k-1", "named by ASCII letters"),
        ("lambda", "named by ASCII letters"),
        ("base_kf", "names taken already"),
        ("rate", "names taken already"),
        ("sub", "names taken already"),
        ("time", "names taken already"),
        ("exp", "names taken already"),
    ],
)
def test_a_declared_parameter_must_be_a_new_name(extra, match):
    """`k-1` and `base_kf` crashed PySB; `sub`, `time`, `rate` and `exp` were made parameters
    that the rate's own names then shadowed."""
    with pytest.raises(MechanismError, match=match):
        M.written(_BASE_WRITTEN_DEF | {"parameters": ["kcat", "Km", extra]}, M.REGISTRY)


def test_a_declared_parameter_the_rate_never_names_is_refused():
    with pytest.raises(MechanismError, match=r"declares \['Ki'\].*never names"):
        M.written(_BASE_WRITTEN_DEF | {"parameters": ["kcat", "Km", "Ki"]}, M.REGISTRY)


def test_a_rate_reading_a_role_the_row_leaves_empty_is_refused():
    """A rate reading `enz` of an enzyme-less row, or `members` of a complex of two, became a free
    symbol, and BioNetGen aborted naming neither the row nor the role."""
    from indra.statements import Complex

    library = dict(M.REGISTRY)
    for stmt_type, base, rate in (("Phosphorylation", "basal", "k*enz"),
                                  ("Complex", "assembly", "k*members")):
        library[(stmt_type, "w")] = M.written(
            {"statement_type": stmt_type, "policy": "w", "based_on": base, "rate": rate,
             "parameters": ["k"]},
            library,
        )
    for stmt, role in ((Phosphorylation(None, Agent("ERK"), "T", "185"), "enz"),
                       (Complex([Agent("A"), Agent("B")]), "members")):
        with pytest.raises(MechanismError, match=rf"row 'r1': .* reads \['{role}'\], which its "):
            M.assemble([(stmt, "w", "r1")], library=library)


def test_a_mechanism_for_no_statement_type_is_refused():
    with pytest.raises(MechanismError, match="no statement type"):
        M.written(
            {"statement_type": "Phosphorylatoin", "policy": "w", "scheme": "sub() >> sub() @ k",
             "parameters": ["k"]},
            M.REGISTRY,
        )


def test_a_written_mechanism_cannot_shadow_a_built_in():
    """Rows recorded against a built-in entry would silently change meaning."""
    with pytest.raises(MechanismError, match="built-in"):
        _written(', "policy": "indra_default"')


# -- a row's parameter binding ---------------------------------------------------
def _sbml_parameters(rows, entities=None):
    from petab.v1.models.sbml_model import SbmlModel

    from petab_mcp import problem_sbml

    antimony, emissions = problem_sbml.to_antimony(rows, entities or {})
    sbml = SbmlModel.from_antimony(antimony).sbml_model
    return {p.getId() for p in sbml.getListOfParameters()}, emissions


def test_rows_bound_to_one_id_share_one_parameter():
    """Rows that run at one shared rate: unbound, each row minted its own and the model
    estimated one constant per row for one."""
    a, b = PopulationTransition(Agent("A"), Agent("B")), PopulationTransition(Agent("B"), Agent("C"))
    params, emissions = _sbml_parameters(
        [(a, "mass_action", "ab", {"k": "k_shared"}), (b, "mass_action", "bc", {"k": "k_shared"})]
    )
    assert "k_shared" in params and not {"ab_k", "bc_k"} & params
    assert [e.parameters for e in emissions] == [("k_shared",), ("k_shared",)]
    assert emissions[0].bound == {"k": "k_shared"}


_FORWARD = PopulationTransition(Agent("A"), Agent("B"))
_BACKWARD = PopulationTransition(Agent("B"), Agent("A"))


def test_a_row_may_share_a_later_rows_parameter():
    """Bound as each row was built, a row sharing `back_k` before the row minting it renamed its
    own to `back_k`, and that row's own then clashed with it: PySB's duplicate name, raw."""
    params, emissions = _sbml_parameters(
        [(_FORWARD, "mass_action", "fwd", {"k": "back_k"}), (_BACKWARD, "mass_action", "back")]
    )
    assert "back_k" in params and "fwd_k" not in params
    assert [e.parameters for e in emissions] == [("back_k",), ("back_k",)]


def test_a_row_shares_what_the_row_it_names_reads():
    """In either order: sharing a pinned row's rate reads the pin, not the constant it replaced,
    which nothing else reads, and a row renamed to a shared id is found by its own name too."""
    for rows in (
        [(_BACKWARD, "mass_action", "back", {"k": 0.5}), (_FORWARD, "mass_action", "fwd", {"k": "back_k"})],
        [(_FORWARD, "mass_action", "fwd", {"k": "back_k"}), (_BACKWARD, "mass_action", "back", {"k": 0.5})],
    ):
        model, _ = M.assemble(rows)
        assert {rule.rate_forward.name for rule in model.rules} == {"back_k_pinned"}
    model, _ = M.assemble(
        [(_FORWARD, "mass_action", "fwd", {"k": "k_on"}), (_BACKWARD, "mass_action", "back", {"k": "fwd_k"})]
    )
    assert {rule.rate_forward.name for rule in model.rules} == {"k_on"}


def test_a_row_sharing_a_scaled_constant_shares_the_constant():
    """A binding may scale the constant it replaces (`k="fwd_k*(1 - drug/K)"`); the constant
    stays, and a row naming it shares the constant, not the drug's scaling."""
    model, _ = M.assemble(
        [
            (_FORWARD, "mass_action", "fwd", {"k": "fwd_k*(1 - drug/K)"}),
            (_BACKWARD, "mass_action", "back", {"k": "fwd_k"}),
        ]
    )
    forward, backward = model.rules
    assert forward.rate_forward.name == "fwd_k_bound"
    assert backward.rate_forward is model.parameters["fwd_k"]


def test_a_row_bound_to_a_number_estimates_nothing():
    """A reference may fix a rate to 1; a minted constant there is a k."""
    params, emissions = _sbml_parameters([(_TRANSITION, "mass_action", "r1", {"k": 1.0})])
    assert "r1_k" not in params and emissions[0].parameters == ()


def test_a_binding_names_only_what_the_row_makes():
    """A misspelt kind would bind nothing and leave the rate unshared without a word."""
    with pytest.raises(MechanismError, match=r"binds \['kcat'\], but its mechanism makes only"):
        M.assemble([(_TRANSITION, "mass_action", "r1", {"kcat": "k"})])


def test_a_binding_reads_as_an_expression_over_parameters_or_is_refused():
    """Checked token by token only, `kcat Km` and `2kcat` reached sympy and `kcat,Km` PySB, which
    raised out of make_changes unexplained; a molecule's name became a parameter clashing with it,
    and `time` one BioNetGen refused."""
    for text in ("kcat Km", "2kcat", "kcat,Km", "kcat/(Km+"):
        with pytest.raises(MechanismError, match="which is no expression: a binding is a shared"):
            M.assemble([(_TRANSITION, "mass_action", "r1", {"k": text})])
    for text, what in (("2*naive", "a molecule"), ("naive", "a molecule"), ("exp", "a function"),
                       ("k*time", "time")):
        with pytest.raises(MechanismError, match=f"which names {what}, not a parameter"):
            M.assemble([(_TRANSITION, "mass_action", "r1", {"k": text})])


def test_a_written_mechanism_binds_like_any_entry():
    """A written rate over a constant the reference shares binds like any entry's."""
    library = dict(M.REGISTRY)
    library[("PopulationTransition", "sat")] = M.written(
        {
            "statement_type": "PopulationTransition",
            "policy": "sat",
            "based_on": "mass_action",
            "rate": "vmax / (K + source)",
            "parameters": ["vmax", "K"],
        },
        library,
    )
    model, (emission,) = M.assemble([(_TRANSITION, "sat", "r1", {"K": "K_half"})], library=library)
    assert set(emission.parameters) == {"r1_vmax", "K_half"}
    assert "K_half" in str(model.expressions["r1_rate"].expr)


# -- written mechanisms: modifiers and list roles ----------------------------------
def _library_with(definition):
    library = dict(M.REGISTRY)
    key = (definition["statement_type"], definition["policy"])
    library[key] = M.written(definition, library)
    return library


_DIVISIVE = {
    "statement_type": "PopulationTransition",
    "policy": "inhibited",
    "based_on": "mass_action",
    "rate": "k * Ki / (Ki + inh)",
    "parameters": ["k", "Ki"],
    "modifiers": ["inh"],
}


def test_a_written_rate_reads_a_modifier_the_row_names():
    """A drug that slows a reaction without taking part in it: the rate reads the amount of an
    agent no role holds, which the row names."""
    library = _library_with(_DIVISIVE)
    model, _ = M.assemble(
        [(_TRANSITION, "inhibited", "r1", None, {"inh": Agent("drug")})], library=library
    )
    assert "drug" in model.monomers.keys()
    assert "r1_inh" in str(model.expressions["r1_rate"].expr)
    assert model.observables["r1_inh"].reaction_pattern.complex_patterns[0].monomer_patterns[
        0
    ].monomer.name == "drug"


def test_a_row_must_fill_every_modifier_slot():
    """An unfilled slot would leave a name in the rate that means nothing."""
    with pytest.raises(MechanismError, match=r"reads modifiers \['inh'\]"):
        M.assemble([(_TRANSITION, "inhibited", "r1")], library=_library_with(_DIVISIVE))


def test_modifiers_need_an_entry_that_reads_them():
    """A modifier given to an entry whose rate cannot read it would be dropped without a word."""
    with pytest.raises(MechanismError, match="names modifiers"):
        M.assemble([(_TRANSITION, "mass_action", "r1", None, {"inh": Agent("drug")})])


def test_a_written_rate_reads_a_list_role():
    """Conversion's substrate is the list `obj_from`; a rate could not read it, so substrate
    inhibition had no route but a private entry."""
    from indra.statements import Conversion

    library = _library_with(
        {
            "statement_type": "Conversion",
            "policy": "haldane",
            "based_on": M.INDRA_DEFAULT,
            "rate": "vmax / (Km + obj_from + obj_from_0^2 / Ki)",
            "parameters": ["vmax", "Km", "Ki"],
        }
    )
    model, _ = M.assemble(
        [(Conversion(None, [Agent("S")], [Agent("P")]), "haldane", "r1")], library=library
    )
    rate = str(model.expressions["r1_rate"].expr)
    assert "r1_obj_from" in rate and "r1_obj_from_0" in rate


def test_a_row_binds_a_rate_to_an_expression_over_shared_ids():
    """Off-rates written as KD*kf of the reference's own parameters; minted per row instead,
    they kept k but lost the reference's bounds."""
    params, emissions = _sbml_parameters(
        [(_TRANSITION, "mass_action", "r1", {"k": "KD * kf_on"})]
    )
    assert {"KD", "kf_on"} <= params and "r1_k" not in params
    assert set(emissions[0].parameters) == {"KD", "kf_on"}
