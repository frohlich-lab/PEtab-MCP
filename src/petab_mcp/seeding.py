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

import json
from pathlib import Path

from indra.statements import Agent


def load_entity_map(path: str) -> dict[str, Agent]:
    """Species id to the INDRA ``Agent`` it is, from the JSON file at ``path``."""
    return {k: Agent._from_json(v) for k, v in json.loads(Path(path).read_text()).items()}


def load_inputs(path: str) -> dict[str, str]:
    """Species id to the protocol formula an experiment sets it by, for each entry of the entity
    map at ``path`` that declares one."""
    entries = json.loads(Path(path).read_text())
    return {k: v["input"] for k, v in entries.items() if "input" in v}
