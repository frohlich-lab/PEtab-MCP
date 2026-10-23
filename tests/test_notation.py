"""The one-line statements notation: every row it writes reads back as the same statement."""

from __future__ import annotations

import pytest
from indra.statements import stmts_from_json

from petab_mcp import mechanisms as M
from petab_mcp import notation as N

_EGFR = {"name": "EGFR", "db_refs": {"HGNC": "3236"}}
_GROUND = {"EGFR": {"HGNC": "3236"}}


def _mod(mod_type, residue=None, position=None, is_modified=True):
    return {"mod_type": mod_type, "residue": residue, "position": position,
            "is_modified": is_modified}


def _canonical(row):
    """A row as INDRA reads it, ids and evidence dropped: what the model is built from."""
    (st,) = stmts_from_json([row["statement"]])
    js = st.to_json()
    for key in ("id", "evidence", "belief", "matches_hash", "supports", "supported_by"):
        js.pop(key, None)
    return {**row, "statement": js}


ROWS = [
    {"statement": {"type": "Phosphorylation",
                   "enz": {**_EGFR, "mods": [_mod("phosphorylation", "Y", "1068")]},
                   "sub": {"name": "AKT1"}, "residue": "S", "position": "473"},
     "policy": "indra_two_step", "prefix": "r23",
     "parameters": {"kf": "k_on", "kr": 2.5, "kc": "KD*kf"}},
    {"statement": {"type": "DecreaseAmount", "subj": None,
                   "obj": {**_EGFR, "mods": [_mod("phosphorylation", "Y", "1068", False)],
                           "bound_conditions": [{"agent": {"name": "EGF"}, "is_bound": False}]}},
     "policy": "indra_default", "prefix": "r10", "parameters": {"kf": "EGFR_turnover"}},
    {"statement": {"type": "Dephosphorylation", "enz": None,
                   "sub": {"name": "AKT1", "bound_conditions": [
                       {"agent": {"name": "RPS6"}, "is_bound": False}]},
                   "residue": "S", "position": "473"},
     "policy": "basal", "prefix": "r7"},
    {"statement": {"type": "Autophosphorylation",
                   "enz": {**_EGFR, "bound_conditions": [{"agent": {"name": "EGF"},
                                                          "is_bound": True}]},
                   "residue": "Y", "position": "1068"},
     "policy": "indra_default", "prefix": "r9"},
    {"statement": {"type": "Complex", "members": [{"name": "SMAD2"}, {"name": "SMAD3"},
                                                  {"name": "SMAD3"}]},
     "policy": "assembly", "prefix": "tri"},
    {"statement": {"type": "Conversion", "subj": {"name": "EPO"},
                   "obj_from": [{"name": "STAT5A"}, {"name": "STAT5A"}],
                   "obj_to": [{"name": "D", "location": "plasma membrane"}]},
     "policy": "mass_action", "prefix": "dim"},
    {"statement": {"type": "Conversion", "subj": None, "obj_from": [{"name": "A"}],
                   "obj_to": []},
     "policy": "release", "prefix": "rel"},
    {"statement": {"type": "PopulationTransition", "source": {"name": "naive"},
                   "target": {"name": "effector"}},
     "policy": "mass_action", "prefix": "tr"},
    {"statement": {"type": "Translocation", "agent": {"name": "STAT5"},
                   "from_location": "cytoplasm", "to_location": "nucleus"},
     "policy": "indra_default", "prefix": "imp"},
    {"statement": {"type": "Acetylation", "enz": None,
                   "sub": {"name": "H4", "mods": [_mod("acetylation", "K", "5"),
                                                  _mod("acetylation", "K", "8", False)]},
                   "residue": "K", "position": "12"},
     "policy": "basal", "prefix": "ac", "parameters": {"k": "a_basal"}},
    {"statement": {"type": "Modification", "enz": {"name": "CDK"},
                   "sub": {"name": "Cell", "mods": [_mod("modification", None, "G2", False)]},
                   "residue": None, "position": "S"},
     "policy": "indra_default", "prefix": "ph"},
    {"statement": {"type": "Activation", "subj": {"name": "RAS", "activity": {
        "activity_type": "gtpbound", "is_active": True}}, "obj": {"name": "RAF"},
                   "obj_activity": "kinase"},
     "policy": "indra_default", "prefix": "act"},
    {"statement": {"type": "IncreaseAmount", "subj": {"name": "Prey"},
                   "obj": {"name": "Predator"},
                   "inhibited_by": [{"name": "drug", "mutations": [
                       {"position": "600", "residue_from": "V", "residue_to": "E"}]}],
                   "modulated_by": [{"name": "season"}, {"name": "rain"}]},
     "policy": "self_renewal", "prefix": "birth", "parameters": {"k": 1.0}},
]


@pytest.mark.parametrize("row", ROWS, ids=[r["prefix"] for r in ROWS])
def test_a_row_reads_back_as_the_statement_it_was(row):
    """The notation is a second way to write the SAME rows: parse(render(row)) builds the model
    row builds, groundings taken from the entity map by name."""
    text = N.render(row, M.REGISTRY)
    assert text is not None, row
    back = N.parse(text, _GROUND, M.REGISTRY)
    assert _canonical(back) == _canonical(row), text


def test_a_policy_keeps_its_prefix_where_dropping_it_is_ambiguous():
    """`indra_default` is written `default`; a library holding an entry named `default` for the
    type would read it back as that entry, so the prefix stays."""
    row = ROWS[1]
    assert "| default " in N.render(row, M.REGISTRY)
    library = {**M.REGISTRY, ("DecreaseAmount", "default"): None}
    assert "| indra_default " in N.render(row, library)


def test_a_bare_indra_policy_is_indras_where_the_type_lacks_it():
    """`A becomes B | two_step` kept `two_step`, a name no entry has, so asking INDRA's policy of
    a type without it was refused as a misspelling, not counted as the library gap it is."""
    assert N.parse("conv: A becomes B | two_step", library=M.REGISTRY)["policy"] == "indra_two_step"


def test_what_the_notation_cannot_write_stays_json():
    """Evidence beyond one source and text, or a statement type it has no verb for, renders as
    None: the caller keeps JSON."""
    with_evidence = {**ROWS[2], "statement": {**ROWS[2]["statement"],
                                              "evidence": [{"text": "x"}, {"text": "y"}]}}
    assert N.render(with_evidence, M.REGISTRY) is None
    gef = {"statement": {"type": "Gef", "gef": {"name": "SOS1"}, "ras": {"name": "HRAS"}},
           "policy": "indra_default", "prefix": "g"}
    assert N.render(gef, M.REGISTRY) is None


@pytest.mark.parametrize(
    "text, where",
    [
        ("r1 EGFR phosphorylates AKT1 | default", "prefix"),
        ("r1: EGFR phosphorylates AKT1", "'|'"),
        ("r1: EGFR frobnicates AKT1 | default", "unknown verb"),
        ("r1: EGFR{qY1} phosphorylates AKT1 | default", "unknown modification"),
        ("r1: EGFR{kinase:maybe} phosphorylates AKT1 | default", ":on or :off"),
    ],
)
def test_a_malformed_row_says_where(text, where):
    with pytest.raises(N.NotationError, match=where):
        N.parse(text)


def test_a_statements_evidence_rides_as_a_comment():
    """A statement may carry a source and a justification; written as JSON only, such a row
    could not use the notation at all."""
    row = {**ROWS[2], "statement": {**ROWS[2]["statement"], "evidence": [
        {"source_api": "curated", "text": "Doe et al. 2020 (10.1000/xyz123) | v4 # x"}]}}
    text = N.render(row, M.REGISTRY)
    assert text.endswith("# [curated] Doe et al. 2020 (10.1000/xyz123) | v4 # x")
    back = N.parse(text, _GROUND, M.REGISTRY)
    assert back["statement"]["evidence"] == row["statement"]["evidence"]
    assert _canonical(back) == _canonical(row)
