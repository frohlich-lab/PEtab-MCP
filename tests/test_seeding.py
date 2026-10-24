"""The entity map: what it declares, and what it refuses."""

from __future__ import annotations

import asyncio
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
    """A map may carry keys other tools read (``ptms``); only a near-miss of a key read here is a
    typo."""
    extra = ', "S6": {"name": "RPS6", "ptms": [["S", "235"]]}'
    assert Sd.load_entity_map(_map_file(tmp_path, _BASE_ENTITY_MAP.format(extra=extra)))["S6"]


_CATALYSIS = {
    "statement": {"type": "Phosphorylation", "enz": {"name": "A"}, "sub": {"name": "B"}},
    "prefix": "r1",
}
_CONVERSION = {
    "statement": {"type": "PopulationTransition", "source": {"name": "A"}, "target": {"name": "B"}},
    "prefix": "r1",
}


@pytest.mark.parametrize(
    ("row", "policy", "held"), [(_CATALYSIS, "indra_default", True), (_CONVERSION, "mass_action", False)]
)
def test_a_species_the_conditions_set_is_held_while_the_rows_only_read_it(
    tmp_path, add_condition, row, policy, held
):
    """Drugs and hormones the conditions set and the rows only read: built as molecules, the
    doses were used up by what they drive and started at an amount every condition overwrites.
    Metabolites or a substrate the conditions start and the rows convert: held for being
    condition targets, they never changed."""
    shutil.copytree(Path(__file__).parent / "data" / "toy_v2", tmp_path / "problem")
    yaml_path = tmp_path / "problem" / "problem.yaml"
    add_condition(yaml_path, "A", 2.0)
    mcp = create_server(
        str(yaml_path),
        provenance_dir=str(tmp_path / "prov"),
        resolver=StubReferenceResolver(),
        auto_fit=False,
        entity_map=_map_file(tmp_path, '{"A": {"name": "A"}, "B": {"name": "B"}}'),
    )

    def call(name, args):
        return asyncio.run(mcp.call_tool(name, args)).structured_content

    assert call("make_changes", {"justification": "A acts on B", "statements": [row],
                                 "bindings": {"r1": policy}})["applied"]
    assert ("$A" in call("get_model", {})["result"]) is held
    ics = {r["species"]: r for r in call("get_model_structure", {})["initial_conditions"]}
    assert ics["A"]["t0_source"] == "condition_table"
    assert (ics["A"]["ic_parameter"] is None) is held
