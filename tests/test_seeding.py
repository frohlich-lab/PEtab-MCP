"""The entity map: what it declares, and what it refuses rather than misread."""

from __future__ import annotations

import re

import pytest

from petab_mcp import seeding as Sd
from petab_mcp.errors import EntityMapError

# The problem's species, as the entity-map file supplies them; ``{extra}`` splices more entries.
_BASE_ENTITY_MAP = (
    '{{"MEK": {{"name": "MAP2K1", "db_refs": {{"HGNC": "6840"}}}}, '
    '"pERK": {{"name": "MAPK1", "db_refs": {{"HGNC": "6871"}}, '
    '"mods": [{{"mod_type": "phosphorylation", "residue": "T", "position": "185"}}]}}{extra}}}'
)


def _map_file(tmp_path, text):
    (tmp_path / "entities.json").write_text(text)
    return str(tmp_path / "entities.json")


def test_inputs_are_declared_in_the_entity_map(tmp_path):
    """An input is a fact about the experiment, declared beside the readouts; an entry without one
    is no input."""
    extra = ', "EGF": {"name": "EGF", "input": "piecewise(EGF_0, time < EGF_end, 0)"}'
    path = _map_file(tmp_path, _BASE_ENTITY_MAP.format(extra=extra))
    assert Sd.load_inputs(path) == {"EGF": "piecewise(EGF_0, time < EGF_end, 0)"}


@pytest.mark.parametrize(
    ("text", "refusal"),
    [
        ('[{"name": "MAPK1"}]', "is a JSON list"),
        (
            _BASE_ENTITY_MAP.format(extra=', "S6": {"db_refs": {"HGNC": "10429"}}'),
            "entry 'S6' needs",
        ),
        (_BASE_ENTITY_MAP.format(extra=', "EGF": {"name": "EGF", "input": 1}'), "input 1 is no"),
        (
            _BASE_ENTITY_MAP.format(extra=', "EGF": {"name": "EGF", "inputs": "EGF_0"}'),
            "entry 'EGF': 'inputs' is no key read here; did you mean 'input'?",
        ),
    ],
)
def test_a_map_that_would_be_misread_is_refused(tmp_path, text, refusal):
    """Each of these loaded without a word: a list raised AttributeError, a nameless entry became
    ``None``, a number was taken as a protocol, and a misspelled ``input`` was dropped, so EGF
    would have been built as a molecule instead of held to its dose."""
    path = _map_file(tmp_path, text)
    for load in (Sd.load_entity_map, Sd.load_inputs):
        with pytest.raises(EntityMapError, match=re.escape(refusal)):
            load(path)


def test_a_key_read_elsewhere_is_kept(tmp_path):
    """PEtabAgentBench's maps carry ``ptms`` for their own use; only a near-miss of a key read here
    is a typo."""
    extra = ', "S6": {"name": "RPS6", "ptms": [["S", "235"]]}'
    assert Sd.load_entity_map(_map_file(tmp_path, _BASE_ENTITY_MAP.format(extra=extra)))["S6"]
