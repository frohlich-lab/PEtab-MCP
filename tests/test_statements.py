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
_TYPES = {"Phosphorylation", "Activation", "Conversion", "PopulationTransition"}
# A phosphorylation as a row writes it; ``{sub}`` is its substrate, ``{extra}`` splices fields.
_BASE_PHOSPHO = '{{"type": "Phosphorylation", "enz": {{"name": "MEK"}}, "sub": {sub}{extra}}}'


def _problems(sub: str = '{"name": "ERK"}', extra: str = "") -> list[str]:
    return S.problems(json.loads(_BASE_PHOSPHO.format(sub=sub, extra=extra)), _TYPES)


def test_what_indra_writes_reads_as_written():
    """INDRA's own JSON (ids, belief, evidence) is what a stored row and a warm start hold."""
    assert _problems() == []
    stmt = S.PopulationTransition(_agent("a"), _agent("b"))
    assert S.problems(stmt.to_json(), _TYPES) == []


def test_a_missing_field_is_named_with_what_it_takes():
    """A mod without `mod_type` crashed the assembly raw ('NoneType' object has no attribute
    'residue'); a phosphorylation without its substrate reached INDRA's reader as None."""
    (problem,) = _problems(sub='{"name": "ERK", "mods": [{"residue": "T"}]}')
    assert problem.startswith("`sub.mods[0]` has no `mod_type`, one of ['acetylation', ")
    (problem,) = S.problems({"type": "Phosphorylation", "enz": {"name": "MEK"}}, _TYPES)
    assert problem == 'Phosphorylation needs `sub`, an agent: {"name": "A"}'


def test_a_regulated_activity_defaults_as_its_constructor_does():
    """INDRA's `Activation` defaults `obj_activity` to `activity`, but its JSON reader passes the
    missing field on as None, and the assembly failed raw on it."""
    statement = {"type": "Activation", "subj": {"name": "A"}, "obj": {"name": "B"}}
    assert S.problems(statement, _TYPES) == []
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
    assert S.problems({"type": "Phosphorilation"}, _TYPES) == [
        "`type` 'Phosphorilation' is no statement type (did you mean 'Phosphorylation'?)"
    ]
    (problem,) = S.problems({"type": "Conversion", "obj_from": {"name": "A"}}, _TYPES)
    assert problem.startswith("`obj_from` is a list of agents")


def test_ungrounded_ignores_reader_text():
    """``TEXT`` is the string a reader saw, not an identity: counting it as a grounding would
    let every literature-extracted agent pass as grounded."""
    stmt = S.PopulationTransition(
        _agent("akt", ', "db_refs": {"TEXT": "Akt", "UP": "P31749"}'),
        _agent("x", ', "db_refs": {"TEXT": "something"}'),
    )
    assert S.ungrounded(stmt) == ["x"]
