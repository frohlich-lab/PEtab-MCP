"""The mechanism registry and the assembler that applies it."""

from __future__ import annotations

import pytest
from indra.statements import Activation, Agent, Phosphorylation

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


def test_unregistered_pair_is_a_mechanism_error():
    """A library gap must stay distinguishable from a bug, so it is not a bare ``KeyError``."""
    with pytest.raises(MechanismError, match="no mechanism"):
        M.assemble([(_TRANSITION, "no_such_mechanism", "r1")])


def test_reregistering_an_entry_raises():
    """A silent override builds whichever policy imported last."""
    with pytest.raises(MechanismError, match="already registered"):
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


# -- a row's parameter binding ---------------------------------------------------
def _sbml_parameters(rows, entities=None):
    from petab.v1.models.sbml_model import SbmlModel

    from petab_mcp import problem_sbml

    antimony, emissions = problem_sbml.to_antimony(rows, entities or {})
    sbml = SbmlModel.from_antimony(antimony).sbml_model
    return {p.getId() for p in sbml.getListOfParameters()}, emissions


def test_rows_bound_to_one_id_share_one_parameter():
    """Blasi's 25 acetylations run at one `a_basal`: unbound, each row minted its own and the
    model estimated 25 constants for one."""
    a, b = PopulationTransition(Agent("A"), Agent("B")), PopulationTransition(Agent("B"), Agent("C"))
    params, emissions = _sbml_parameters(
        [(a, "mass_action", "ab", {"k": "k_shared"}), (b, "mass_action", "bc", {"k": "k_shared"})]
    )
    assert "k_shared" in params and not {"ab_k", "bc_k"} & params
    assert [e.parameters for e in emissions] == [("k_shared",), ("k_shared",)]
    assert emissions[0].bound == {"k": "k_shared"}


def test_a_row_bound_to_a_number_estimates_nothing():
    """Perelson's infection runs at a rate fixed to 1; a minted constant there is a k."""
    params, emissions = _sbml_parameters([(_TRANSITION, "mass_action", "r1", {"k": 1.0})])
    assert "r1_k" not in params and emissions[0].parameters == ()


def test_a_binding_names_only_what_the_row_makes():
    """A misspelt kind would bind nothing and leave the rate unshared without a word."""
    with pytest.raises(MechanismError, match=r"binds \['kcat'\], but its mechanism makes only"):
        M.assemble([(_TRANSITION, "mass_action", "r1", {"kcat": "k"})])


def test_a_written_mechanism_binds_like_any_entry():
    """Sneyd's gating steps are written rates over constants the reference shares."""
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
