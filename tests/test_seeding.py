"""The entity map: what it declares, and what it refuses."""

from __future__ import annotations

import re
import shutil
from pathlib import Path

import pytest

from petab_mcp import seeding as Sd
from petab_mcp.errors import EntityMapError
from petab_mcp.references import StubReferenceResolver
from petab_mcp.server import create_server

# The problem's species, as the entity-map file supplies them; ``{extra}`` splices more entries.
_BASE_ENTITY_MAP = (
    '{{"MEK": {{"name": "MAP2K1", "db_refs": {{"HGNC": "6840"}}}}, '
    '"pERK": {{"name": "MAPK1", "db_refs": {{"HGNC": "6871"}}, '
    '"mods": [{{"mod_type": "phosphorylation", "residue": "T", "position": "185"}}]}}{extra}}}'
)


def _map_file(tmp_path, text):
    (tmp_path / "entities.json").write_text(text)
    return str(tmp_path / "entities.json")


@pytest.mark.parametrize(
    ("text", "refusal"),
    [
        ('[{"name": "MAPK1"}]', "is a JSON list"),
        (
            _BASE_ENTITY_MAP.format(extra=', "S6": {"db_refs": {"HGNC": "10429"}}'),
            "entry 'S6' needs",
        ),
        (
            _BASE_ENTITY_MAP.format(extra=', "EGF": {"name": "EGF", "input": "EGF_0"}'),
            "entry 'EGF' declares an input; declare it in the inputs.tsv",
        ),
        (
            _BASE_ENTITY_MAP.format(extra=', "S6": {"name": "RPS6", "db_ref": {}}'),
            "entry 'S6': 'db_ref' is no key read here; did you mean 'db_refs'?",
        ),
    ],
)
def test_a_map_that_would_be_misread_is_refused(tmp_path, text, refusal):
    """Each of these loaded without a word: a list raised AttributeError, a nameless entry became
    ``None``, and a misspelled key was dropped. A map's ``input``, which inputs.tsv replaced,
    would be dropped too, and EGF built as a molecule instead of held to its dose."""
    with pytest.raises(EntityMapError, match=re.escape(refusal)):
        Sd.load_entity_map(_map_file(tmp_path, text))


def test_a_key_read_elsewhere_is_kept(tmp_path):
    """PEtabAgentBench's maps carry ``ptms`` for their own use; only a near-miss of a key read here
    is a typo."""
    extra = ', "S6": {"name": "RPS6", "ptms": [["S", "235"]]}'
    assert Sd.load_entity_map(_map_file(tmp_path, _BASE_ENTITY_MAP.format(extra=extra)))["S6"]


def test_an_input_names_only_condition_columns_and_time(tmp_path, add_condition):
    """Three PEtabAgentBench#6 maps put fitted constants in an input (Boehm's Epo degradation,
    Fiedler's stimulus time constants, Oliveira's transmission steps), handing the agent part of
    the reference's model where it can neither see nor change it."""
    shutil.copytree(Path(__file__).parent / "data" / "toy_v2", tmp_path / "problem")
    yaml_path = tmp_path / "problem" / "problem.yaml"
    add_condition(yaml_path, "dose", 1.0)

    def open_with(name, protocol):
        entities = f'{{"A": {{"name": "A"}}, "S": {{"name": "S", "input": "{protocol}"}}}}'
        return create_server(
            str(yaml_path),
            provenance_dir=str(tmp_path / name),
            resolver=StubReferenceResolver(),
            auto_fit=False,
            entity_map=_map_file(tmp_path, entities),
        )

    open_with("columns", "dose*exp(-time)")
    with pytest.raises(EntityMapError, match=re.escape("'S' names k1 (estimated)")):
        open_with("estimated", "dose*exp(-k1*time)")
