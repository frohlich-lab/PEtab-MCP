"""The entity map: what it declares."""

from __future__ import annotations

from petab_mcp import seeding as Sd

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
