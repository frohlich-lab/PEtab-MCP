"""The entity map: what links this PEtab problem to INDRA.

The entity map is SUPPLIED, not derived: no corpus knows that this problem's ``pAkt`` means
UniProt P31749 phosphorylated at S473. It is read from a path on the command line, so
benchmark-specific data stays out of this package. Each species maps to an INDRA ``Agent`` in
INDRA's own JSON, so the state rides in INDRA's own ``mods``::

    {"pAkt": {"name": "AKT1", "db_refs": {"UP": "P31749"},
              "mods": [{"mod_type": "phosphorylation", "residue": "S", "position": "473"}]}}

The same map names each species' PySB pattern (:func:`problem_sbml.to_antimony`), which is what
binds a model built from statements to the fixed observable table. An entry may also declare its
species an experimental INPUT, with the protocol that sets it over the condition table's columns
(:func:`load_inputs`)::

    {"EGF": {"name": "EGF", "db_refs": {"HGNC": "3229"},
             "input": "piecewise(EGF_0 + EGF_rate*time, time < EGF_end, 0)"}}

An input is a fact about the experiment, like a readout, so it is declared beside the readouts
rather than asserted as a statement or built by a mechanism (see RATIONALE.md).

The module is named for the INDRA warm start, which seeds a model from a pinned statement corpus
filtered to this map. That comes later, with its wiring and a pinned corpus (PEtab-MCP#43); until
then the map is all this module reads.
"""

from __future__ import annotations

import difflib
import json
from pathlib import Path

from indra.statements import Agent

from .errors import EntityMapError

#: The keys an entry is read by: INDRA's ``Agent`` JSON, and ``input``. Any other key is left to
#: the map's other readers (PEtabAgentBench's ``ptms``), except one close to these, which is a typo
#: that would otherwise be dropped without a word.
ENTRY_KEYS = (
    "name",
    "db_refs",
    "mods",
    "mutations",
    "bound_conditions",
    "activity",
    "location",
    "input",
)


def _entries(path: str) -> dict[str, dict]:
    """The entity map at ``path``, refused at the first entry that would not be read as written."""
    entries = json.loads(Path(path).read_text())
    if not isinstance(entries, dict):
        raise EntityMapError(
            f"entity map {path} is a JSON {type(entries).__name__}; write an object from each "
            'species id to its INDRA agent: {"pERK": {"name": "MAPK1", ...}}'
        )
    for species, entry in entries.items():
        where = f"entity map {path}, entry {species!r}"
        if not isinstance(entry, dict):
            raise EntityMapError(
                f'{where} is {entry!r}; write the INDRA agent it is: {{"name": "MAPK1", ...}}'
            )
        if not entry.get("name") or not isinstance(entry["name"], str):
            raise EntityMapError(
                f'{where} needs the name of the INDRA agent it is: "name": "MAPK1"'
            )
        if not isinstance(entry.get("input", ""), str):
            raise EntityMapError(
                f"{where}: input {entry['input']!r} is no formula; write the protocol as a string "
                "over the condition table's columns and time: "
                '"input": "dose*piecewise(0, time < t_on, 1)"'
            )
        for key in sorted(entry.keys() - set(ENTRY_KEYS)):
            if near := difflib.get_close_matches(key, ENTRY_KEYS, n=1):
                raise EntityMapError(
                    f"{where}: {key!r} is no key read here; did you mean {near[0]!r}?"
                )
    return entries


def load_entity_map(path: str) -> dict[str, Agent]:
    """Species id to the INDRA ``Agent`` it is, from the JSON file at ``path``."""
    return {k: Agent._from_json(v) for k, v in _entries(path).items()}


def load_inputs(path: str) -> dict[str, str]:
    """Species id to the protocol formula an experiment sets it by, for each entry of the entity
    map at ``path`` that declares one."""
    return {k: v["input"] for k, v in _entries(path).items() if "input" in v}
