"""The mechanism vocabulary's INDRA subclasses: serialisation, deduplication, grounding."""

from __future__ import annotations

import json

from indra.preassembler import Preassembler
from indra.statements import Agent, stmts_from_json

from petab_mcp import statements as S

# One agent as INDRA serialises it; ``{extra}`` splices extra fields into the agent's JSON.
_BASE_AGENT = '{{"name": "{name}"{extra}}}'


def _agent(name: str, extra: str = "") -> Agent:
    return Agent._from_json(json.loads(_BASE_AGENT.format(name=name, extra=extra)))


def test_vocabulary_round_trips_through_indra_json():
    """INDRA resolves ``type`` in its own module and DROPS what it cannot find with only a
    warning, so an unregistered subclass would vanish from a restored version, not fail."""
    stmt = S.PopulationTransition(_agent("a", ', "db_refs": {"HGNC": "6840"}'), _agent("b"))
    (back,) = stmts_from_json([stmt.to_json()])
    assert type(back) is S.PopulationTransition
    assert [a.name for a in back.agent_list()] == ["a", "b"]
    assert back.agent_list()[0].db_refs == {"HGNC": "6840"}


def test_same_pair_deduplicates_and_reversed_pair_does_not():
    """``Preassembler`` deduplicates on ``matches_key``; a direction-blind key would merge
    naive->effector with effector->naive."""
    same = [S.PopulationTransition(_agent("naive"), _agent("effector")) for _ in range(2)]
    assert same[0].matches_key() == same[1].matches_key()
    assert len(Preassembler(None, same).combine_duplicates()) == 1
    reverse = S.PopulationTransition(_agent("effector"), _agent("naive"))
    assert reverse.matches_key() != same[0].matches_key()


# -- a statement as written ----------------------------------------------------------
# A phosphorylation as a row writes it; ``{sub}`` is its substrate, ``{extra}`` splices fields.
_BASE_PHOSPHO = '{{"type": "Phosphorylation", "enz": {{"name": "MEK"}}, "sub": {sub}{extra}}}'


def _problems(sub: str = '{"name": "ERK"}', extra: str = "") -> list[str]:
    return S.problems(json.loads(_BASE_PHOSPHO.format(sub=sub, extra=extra)))


def test_what_indra_writes_reads_as_written():
    """INDRA's own JSON (ids, belief, evidence) is what a stored row and a warm start hold."""
    assert _problems() == []
    stmt = S.PopulationTransition(_agent("a"), _agent("b"))
    assert S.problems(stmt.to_json()) == []


def test_a_missing_field_is_named_with_what_it_takes():
    """A mod without `mod_type` crashed the assembly raw ('NoneType' object has no attribute
    'residue'); a phosphorylation without its substrate reached INDRA's reader as None."""
    (problem,) = _problems(sub='{"name": "ERK", "mods": [{"residue": "T"}]}')
    assert problem.startswith("`sub.mods[0]` has no `mod_type`, one of ['acetylation', ")
    (problem,) = S.problems({"type": "Phosphorylation", "enz": {"name": "MEK"}})
    assert problem == 'Phosphorylation needs `sub`, an agent: {"name": "A"}'


def test_a_regulated_activity_defaults_as_its_constructor_does():
    """INDRA's `Activation` defaults `obj_activity` to `activity`, but its JSON reader passes the
    missing field on as None, and the assembly failed raw on it."""
    statement = {"type": "Activation", "subj": {"name": "A"}, "obj": {"name": "B"}}
    assert S.problems(statement) == []
    assert S.with_defaults(statement)["obj_activity"] == "activity"
    assert S.with_defaults({**statement, "obj_activity": "kinase"})["obj_activity"] == "kinase"


def test_a_misspelt_name_is_answered_with_the_names_it_may_be():
    """A misspelt role built a model without it (`Model has no rules`), a misspelt agent field an
    agent without its modification, committed; a misspelt type got only "not statements INDRA can
    read" and a hint about list roles."""
    assert _problems(extra=', "enzyme": {"name": "MEK"}') == [
        "Phosphorylation has no field 'enzyme' (did you mean 'enz'?), its fields are "
        "['enz', 'sub', 'residue', 'position']"
    ]
    (problem,) = _problems(sub='{"name": "ERK", "mod": [{"mod_type": "phosphorylation"}]}')
    assert problem.startswith("`sub` has no field 'mod' (did you mean 'mods'?), an agent has [")
    (problem,) = _problems(sub='{"name": "ERK", "mods": [{"mod_type": "phosphorilation"}]}')
    assert "`mod_type` 'phosphorilation' (did you mean 'phosphorylation'?)" in problem
    (problem,) = S.problems({"type": "Phosphorilation"})
    assert problem.startswith("`type` 'Phosphorilation' is no statement type (did you mean 'Phosph")
    (problem,) = S.problems({"type": "Conversion", "obj_from": {"name": "A"}})
    assert problem.startswith("`obj_from` is a list of agents")
    activation = {"type": "Activation", "subj": {"name": "A"}, "obj": {"name": "B"}}
    assert S.problems({**activation, "obj_activity": "kinas"}) == [
        "`obj_activity` 'kinas' (did you mean 'kinase'?) is none of INDRA's activities "
        "['activity', 'catalytic', 'gap', 'gef', 'gtpbound', 'kinase', 'phosphatase', "
        "'transcription']"
    ]


def test_ungrounded_ignores_reader_text():
    """``TEXT`` is the string a reader saw, not an identity: counting it as a grounding would
    let every literature-extracted agent pass as grounded."""
    stmt = S.PopulationTransition(
        _agent("akt", ', "db_refs": {"TEXT": "Akt", "UP": "P31749"}'),
        _agent("x", ', "db_refs": {"TEXT": "something"}'),
    )
    assert S.ungrounded(stmt) == ["x"]


def test_a_statement_states_its_regulators_and_says_them_in_english():
    """PEtab-MCP#122: "RAF phosphorylates MEK" read as a reaction nothing slows while its rate
    read sorafenib. The regulator is the statement's, so its English says it; INDRA reads the
    rest."""
    stmt = {
        "type": "Phosphorylation", "enz": {"name": "RAF"}, "sub": {"name": "MEK"},
        "residue": "S", "position": "217",
        "inhibited_by": [{"name": "sorafenib"}],
        "activated_by": [{"name": "ERK", "mods": [{"mod_type": "phosphorylation",
                                                    "residue": "T", "position": "202"}]}],
    }
    assert S.problems(stmt) == []
    assert S.english(stmt) == (
        "RAF phosphorylates MEK on S217, inhibited by sorafenib, activated by ERK phosphorylated "
        "on T202."
    )
    assert set(S.regulators(stmt)) == {"inhibited_by", "activated_by"}
    assert "inhibited_by" not in S.indra_part(stmt)
    transition = {"type": "PopulationTransition", "source": {"name": "A"},
                  "target": {"name": "B"}, "modulated_by": [{"name": "Ca"}]}
    assert S.english(transition) == "A becomes B, modulated by Ca."
    assert S.problems({**transition, "modulated_by": {"name": "Ca"}}) == [
        '`modulated_by` is a list of agents: [{"name": "A"}, ...]'
    ]


def _projected(row: str) -> list[str]:
    from petab_mcp import notation

    return [str(stmt) for stmt in S.as_indra(notation.parse(row)["statement"])[1:]]


def test_a_regulator_is_projected_onto_what_indra_can_read():
    """A causal graph assembled from INDRA statements drops the regulator fields, which INDRA has
    no reading for: each signed one becomes the INDRA statement nearest its claim."""
    # A catalysed reaction: its catalyst's activity there.
    assert _projected("r: RAF{pS338} phosphorylates MEK at S217, inhibited by sorafenib") == [
        "Inhibition(sorafenib(), RAF(mods: (phosphorylation, S, 338)), kinase)"
    ]
    assert _projected("r: A -> B by E{kinase:on}, activated by X") == [
        "Activation(X(), E(kinase), kinase)"
    ]
    # Without a catalyst: what it makes, signed by the claim and what the reaction does to it.
    assert _projected("r: SKI is produced, inhibited by X") == ["DecreaseAmount(X(), SKI())"]
    assert _projected("r: SKI is degraded, inhibited by X") == ["IncreaseAmount(X(), SKI())"]
    assert _projected("r: A becomes B, activated by X") == ["IncreaseAmount(X(), B())"]
    assert _projected("r: MEK is dephosphorylated at S217, activated by X") == [
        "IncreaseAmount(X(), MEK(mods: (phosphorylation, S, 217, False)))"
    ]
    # A self-modification changes its own enzyme: the form it makes.
    assert _projected("r: EGFR autophosphorylates at Y1068, activated by X") == [
        "IncreaseAmount(X(), EGFR(mods: (phosphorylation, Y, 1068)))"
    ]
    # No direction, no INDRA statement.
    assert _projected("r: A becomes B, modulated by Ca") == []


def test_a_projection_keeps_the_claim_it_comes_from_in_a_signed_graph():
    """INDRA's signed graph keeps the projected edge, and its evidence names the reaction it
    regulates. The scorer gives the row's evidence source the priors INDRA's belief engine has
    only for its own readers and databases."""
    from indra.assemblers.indranet import IndraNetAssembler
    from indra.belief import SimpleScorer

    from petab_mcp import notation

    row = "r: RAF phosphorylates MEK at S217, inhibited by sorafenib # [paper] text"
    stmts = S.as_indra(notation.parse(row)["statement"])
    scorer = SimpleScorer({"rand": {"paper": 0.1}, "syst": {"paper": 0.05}})
    graph = IndraNetAssembler(stmts).make_model(graph_type="signed", belief_scorer=scorer)
    assert ("sorafenib", "RAF", 1) in graph.edges(keys=True)  # key 1: a negative edge
    (evidence,) = stmts[1].evidence
    assert (evidence.source_api, evidence.text) == ("paper", "text")
    assert evidence.annotations["regulates"] == (
        "RAF phosphorylates MEK on S217, inhibited by sorafenib."
    )


def test_a_modulator_says_the_direction_its_rate_proves_as_derived():
    """A regulator stated `modulated_by` whose rate proves a direction reads as that kind, marked
    as the rate's, in English and in the INDRA projection; one it cannot sign stays unsigned."""
    from petab_mcp import notation

    statement = notation.parse(
        "r: RAF phosphorylates MEK at S217, modulated by sorafenib and Ca"
    )["statement"]
    derived = {"modulated_by_0": "inhibited_by"}
    assert S.english(statement, derived) == (
        "RAF phosphorylates MEK on S217, inhibited by sorafenib (from its rate), modulated by Ca."
    )
    projected = S.as_indra(statement, derived)[1:]
    assert [str(p) for p in projected] == ["Inhibition(sorafenib(), RAF(), kinase)"]
    assert projected[0].evidence[0].annotations["from_rate"] is True
    assert S.as_indra(statement)[1:] == []


def test_a_claim_its_rate_reverses_or_cannot_sign_reads_as_the_rate_shows():
    """A reversed claim reads and projects as the kind its rate shows; a downgraded one reads as
    unsigned and projects to nothing."""
    from petab_mcp import notation

    statement = notation.parse(
        "r: RAF phosphorylates MEK at S217, inhibited by sorafenib and Ca"
    )["statement"]
    derived = {"inhibited_by_0": "activated_by", "inhibited_by_1": "modulated_by"}
    assert S.english(statement, derived) == (
        "RAF phosphorylates MEK on S217, activated by sorafenib (from its rate), modulated by Ca "
        "(from its rate)."
    )
    assert [str(p) for p in S.as_indra(statement, derived)[1:]] == [
        "Activation(sorafenib(), RAF(), kinase)"
    ]
