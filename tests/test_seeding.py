"""The entity map, what it declares and refuses, and the INDRA warm start it filters a corpus by."""

from __future__ import annotations

import asyncio
import gzip
import json
import re
import shutil
from itertools import combinations_with_replacement
from pathlib import Path

import pytest
from indra.statements import (
    Activation,
    Agent,
    Autophosphorylation,
    Dephosphorylation,
    Phosphorylation,
    Translocation,
)

from petab_mcp import seeding as Sd
from petab_mcp.errors import EntityMapError, SeedError
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
    "policy": "indra_default",
    "prefix": "r1",
}
_CONVERSION = {
    "statement": {"type": "PopulationTransition", "source": {"name": "A"}, "target": {"name": "B"}},
    "policy": "mass_action",
    "prefix": "r1",
}


@pytest.mark.parametrize(("row", "held"), [(_CATALYSIS, True), (_CONVERSION, False)])
def test_a_species_the_conditions_set_is_held_while_the_rows_only_read_it(
    tmp_path, add_condition, row, held
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

    assert call("make_changes", {"justification": "A acts on B", "statements": [row]})["applied"]
    assert ("$A" in call("get_model", {})["result"]) is held
    ics = {r["species"]: r for r in call("get_model_structure", {})["initial_conditions"]}
    assert ics["A"]["t0_source"] == "condition_table"
    assert (ics["A"]["ic_parameter"] is None) is held


# -- the INDRA warm start ------------------------------------------------------------------------
def _hgnc(name, hgnc, *site, bound=None):
    mods = (
        [{"mod_type": "phosphorylation", "residue": site[0], "position": site[1]}] if site else []
    )
    bound_conditions = [{"agent": bound.to_json(), "is_bound": True}] if bound else []
    return Agent._from_json(
        {
            "name": name,
            "db_refs": {"HGNC": hgnc},
            "mods": mods,
            "bound_conditions": bound_conditions,
        }
    )


AKT1, PDPK1, MTOR, PP2A = (
    _hgnc(n, h)
    for n, h in [("AKT1", "391"), ("PDPK1", "8816"), ("MTOR", "3942"), ("PPP2CA", "9299")]
)
PAKT1_S473, PAKT1_T308 = _hgnc("AKT1", "391", "S", "473"), _hgnc("AKT1", "391", "T", "308")


def test_a_site_any_readout_of_the_substrate_measures_seeds():
    """Checked a readout at a time, pAKT1-S473 and pAKT1-T308 each refused the other's site, and
    neither PDPK1's T308 nor MTOR's S473 phosphorylation seeded (PEtab-MCP#43, bug 1). A site no
    readout measures is still refused."""
    measured = [Phosphorylation(PDPK1, AKT1, "T", "308"), Phosphorylation(MTOR, AKT1, "S", "473")]
    seeded = Sd.seed([*measured, Phosphorylation(PP2A, AKT1, "T", "450")], [PAKT1_S473, PAKT1_T308])
    assert sorted(s.get_hash() for s in seeded) == sorted(s.get_hash() for s in measured)


def test_an_open_site_removal_is_read_as_the_measured_site_and_ranked_with_it():
    """INDRA names an open site `phospho`, which no observable reads: an open-site
    Dephosphorylation seeded a rule that never lowered the measured pAKT1, and ranked as touching
    nothing measured, behind a better-believed Activation of the same pair (PEtab-MCP#43, bug 2)."""
    removal, activation = Dephosphorylation(PP2A, AKT1), Activation(PP2A, AKT1)
    removal.belief, activation.belief = 0.5, 0.9
    [chosen] = Sd.seed([activation, removal], [PAKT1_S473])
    assert isinstance(chosen, Dephosphorylation)
    assert (chosen.residue, chosen.position) == ("S", "473")


def _corpus_file(tmp_path, records, **pinned):
    path = tmp_path / "corpus.json.gz"
    with gzip.open(path, "wt") as fh:
        json.dump({**pinned, "statements": records}, fh)
    return str(path)


def test_an_unreadable_record_leaves_every_count_on_its_own_statement(tmp_path):
    """INDRA's reader drops a record it cannot read, and pairing its statements with the records
    by position raised `zip() argument 2 is longer than argument 1` (PEtab-MCP#43, bug 3)."""
    records = [
        {"type": "NotAStatement", "id": "x"},
        Phosphorylation(PDPK1, AKT1).to_json() | {"n_evidence": 7},
        Dephosphorylation(PP2A, AKT1).to_json() | {"n_evidence": 3},
    ]
    stmts, pinned = Sd.load_corpus(_corpus_file(tmp_path, records, date="2026-10-20"))
    assert [(type(s).__name__, s.n_evidence) for s in stmts] == [
        ("Phosphorylation", 7),
        ("Dephosphorylation", 3),
    ]
    assert pinned == {"date": "2026-10-20"}


def test_one_statement_per_agent_set_caps_fujitas_seed_at_twelve():
    """Fujita's "717 -> 12" was this ceiling, not the ranking: four entities make 14 agent sets,
    12 of them touching a readout, so a corpus with every reading of every pair seeds 12 and the
    cap of 25 never binds (PEtab-MCP#43)."""
    ents = {
        n: _hgnc(n, h)
        for n, h in [("EGF", "3229"), ("EGFR", "3236"), ("AKT1", "391"), ("RPS6", "10429")]
    }
    stmts = [
        kind(x, y)
        for a, b in combinations_with_replacement(ents.values(), 2)
        for kind in (Activation, Phosphorylation)
        for x, y in ((a, b), (b, a))
    ]
    stmts += [Translocation(a, None, "nucleus") for a in ents.values()]
    assert len(Sd.seed(stmts, [ents["EGFR"], ents["AKT1"], ents["RPS6"]])) == 12


def test_a_corpus_agent_takes_the_name_the_map_grounds_it_by():
    """PySB keys a molecule on its name: a corpus `AKT1` beside a map's `Akt` built two
    molecules, and the readout read the one no statement touched."""
    entities = {"pAkt": _hgnc("Akt", "391", "S", "473"), "PDK1": PDPK1}
    [kept] = Sd.filter_corpus([Phosphorylation(PDPK1, AKT1), Phosphorylation(MTOR, AKT1)], entities)
    assert [a.name for a in kept.agent_list()] == ["PDPK1", "Akt"]


_MEK, _ERK = _hgnc("MAP2K1", "6840"), _hgnc("MAPK1", "6871")
# The toy observes `B`, here phospho-T185 MAPK1; `A` is MAP2K1.
_WARM_MAP = json.dumps({"A": _MEK.to_json(), "B": _hgnc("MAPK1", "6871", "T", "185").to_json()})


def _warm_server(tmp_path, yaml_path, corpus):
    return create_server(
        yaml_path,
        provenance_dir=str(tmp_path / "prov"),
        resolver=StubReferenceResolver(),
        auto_fit=False,
        entity_map=_map_file(tmp_path, _WARM_MAP),
        corpus=corpus,
    )


def test_a_fresh_session_starts_from_the_warm_start_and_a_resumed_one_is_left_alone(
    tmp_path, empty_yaml
):
    """Nothing called the warm start, so no seed was ever committed (PEtab-MCP#43). It is the
    session's first version, built by the statements route; a row whose rule never fires (MAPK1
    phosphorylating itself while bound to MAP2K1, which nothing binds) is left out and named, and
    a resumed session is not seeded again over the agent's work."""
    autophosphorylation = Autophosphorylation(_hgnc("MAPK1", "6871", bound=_MEK))
    autophosphorylation.belief = 0.5
    corpus = _corpus_file(
        tmp_path,
        [Phosphorylation(_MEK, _ERK).to_json(), autophosphorylation.to_json()],
        source="test",
        date="2026-10-20",
    )
    mcp = _warm_server(tmp_path, empty_yaml, corpus)

    def call(name, args):
        return asyncio.run(mcp.call_tool(name, args)).structured_content

    rows = call("get_model", {"format": "statements"})["result"]
    assert "seed1: MAP2K1 phosphorylates MAPK1 at T185 | default" in rows
    assert "seed2" not in rows
    assert len(call("list_versions", {})["result"]) == 1
    [why] = {
        e["justification"] for e in call("get_provenance", {})["result"] if "justification" in e
    }
    assert why.startswith("INDRA warm start: 1 statement of the pinned corpus corpus.json.gz")
    assert why.endswith("Left out, as their rules never fire or chain: seed2.")
    resumed = _warm_server(tmp_path, empty_yaml, corpus)
    assert (
        len(asyncio.run(resumed.call_tool("list_versions", {})).structured_content["result"]) == 1
    )


def test_a_corpus_that_produces_no_readout_is_refused_as_the_server_opens(tmp_path, empty_yaml):
    """An Activation leaves phospho-T185 MAPK1 unmade, so the observable would read nothing: the
    server refuses to open rather than start from a model it cannot score."""
    corpus = _corpus_file(tmp_path, [Activation(_MEK, _ERK).to_json()])
    with pytest.raises(SeedError, match=r"readouts \['B'\] read no species"):
        _warm_server(tmp_path, empty_yaml, corpus)
