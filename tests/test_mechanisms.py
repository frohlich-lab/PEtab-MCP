"""The mechanism registry and the assembler that applies it."""

from __future__ import annotations

import pytest
from indra.statements import Activation, ActiveForm, Agent, Dephosphorylation, Phosphorylation

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


def test_reregistering_an_entry_raises():
    """A silent override builds whichever policy imported last. A programming error, not a
    refusal: no agent call reaches it."""
    with pytest.raises(RuntimeError, match="already registered"):
        M.mechanism(PopulationTransition, "mass_action")(lambda *a: None)


def test_every_indra_statement_has_its_default_entry():
    """Without INDRA's own assembly as an entry, every INDRA statement but one phosphorylation
    form was a library gap, and a seed drawn from the corpus could not be built at all. Its
    parameters carry the row's prefix: INDRA's own `kf_ea_activation_1` names no row, and
    widening one bound on Fujita meant looking each name up."""
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


def _written(extra=""):
    import json

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


def test_a_written_mechanism_cannot_shadow_a_built_in():
    """Rows recorded against a built-in entry would silently change meaning."""
    with pytest.raises(MechanismError, match="built-in"):
        _written(', "policy": "indra_default"')
