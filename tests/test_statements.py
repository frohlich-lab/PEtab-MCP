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


def test_ungrounded_ignores_reader_text():
    """``TEXT`` is the string a reader saw, not an identity: counting it as a grounding would
    let every literature-extracted agent pass as grounded."""
    stmt = S.PopulationTransition(
        _agent("akt", ', "db_refs": {"TEXT": "Akt", "UP": "P31749"}'),
        _agent("x", ', "db_refs": {"TEXT": "something"}'),
    )
    assert S.ungrounded(stmt) == ["x"]
