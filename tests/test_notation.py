"""The one-line statements notation: every row it writes reads back as the same statement, and
every binding as the same binding."""

from __future__ import annotations

import json

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
    ({"statement": {"type": "Phosphorylation",
                    "enz": {**_EGFR, "mods": [_mod("phosphorylation", "Y", "1068")]},
                    "sub": {"name": "AKT1"}, "residue": "S", "position": "473"}, "prefix": "r23"},
     {"policy": "indra_two_step", "parameters": {"kf": "k_on", "kr": 2.5, "kc": "KD*kf"}}),
    ({"statement": {"type": "DecreaseAmount", "subj": None,
                    "obj": {**_EGFR, "mods": [_mod("phosphorylation", "Y", "1068", False)],
                            "bound_conditions": [{"agent": {"name": "EGF"}, "is_bound": False}]}}, "prefix": "r10"},
     {"policy": "indra_default", "parameters": {"kf": "EGFR_turnover"}}),
    ({"statement": {"type": "Dephosphorylation", "enz": None,
                    "sub": {"name": "AKT1", "bound_conditions": [
                        {"agent": {"name": "RPS6"}, "is_bound": False}]},
                    "residue": "S", "position": "473"}, "prefix": "r7"},
     {"policy": "basal"}),
    ({"statement": {"type": "Autophosphorylation",
                    "enz": {**_EGFR, "bound_conditions": [{"agent": {"name": "EGF"},
                                                           "is_bound": True}]},
                    "residue": "Y", "position": "1068"}, "prefix": "r9"},
     {"policy": "indra_default"}),
    ({"statement": {"type": "Complex", "members": [{"name": "SMAD2"}, {"name": "SMAD3"},
                                                   {"name": "SMAD3"}]}, "prefix": "tri"},
     {"policy": "assembly"}),
    ({"statement": {"type": "Conversion", "subj": {"name": "EPO"},
                    "obj_from": [{"name": "STAT5A"}, {"name": "STAT5A"}],
                    "obj_to": [{"name": "D", "location": "plasma membrane"}]}, "prefix": "dim"},
     {"policy": "mass_action"}),
    ({"statement": {"type": "Conversion", "subj": None, "obj_from": [{"name": "A"}],
                    "obj_to": []}, "prefix": "rel"},
     {"policy": "release"}),
    ({"statement": {"type": "PopulationTransition", "source": {"name": "naive"},
                    "target": {"name": "effector"}}, "prefix": "tr"},
     {"policy": "mass_action"}),
    ({"statement": {"type": "Translocation", "agent": {"name": "STAT5"},
                    "from_location": "cytoplasm", "to_location": "nucleus"}, "prefix": "imp"},
     {"policy": "indra_default"}),
    ({"statement": {"type": "Acetylation", "enz": None,
                    "sub": {"name": "H4", "mods": [_mod("acetylation", "K", "5"),
                                                   _mod("acetylation", "K", "8", False)]},
                    "residue": "K", "position": "12"}, "prefix": "ac"},
     {"policy": "basal", "parameters": {"k": "a_basal"}}),
    ({"statement": {"type": "Modification", "enz": {"name": "CDK"},
                    "sub": {"name": "Cell", "mods": [_mod("modification", None, "G2", False)]},
                    "residue": None, "position": "S"}, "prefix": "ph"},
     {"policy": "indra_default"}),
    ({"statement": {"type": "Activation", "subj": {"name": "RAS", "activity": {
         "activity_type": "gtpbound", "is_active": True}}, "obj": {"name": "RAF"},
                    "obj_activity": "kinase"}, "prefix": "act"},
     {"policy": "indra_default"}),
    ({"statement": {"type": "IncreaseAmount", "subj": {"name": "Prey"},
                    "obj": {"name": "Predator"},
                    "inhibited_by": [{"name": "drug", "mutations": [
                        {"position": "600", "residue_from": "V", "residue_to": "E"}]}],
                    "modulated_by": [{"name": "season"}, {"name": "rain"}]}, "prefix": "birth"},
     {"policy": "self_renewal", "parameters": {"k": 1.0}}),
    ({"statement": {"type": "Gef", "gef": {"name": "SOS1"}, "ras": {"name": "HRAS"},
                    "activated_by": [{"name": "X"}]}, "prefix": "gef"},
     {"policy": "indra_default"}),
    ({"statement": {"type": "Gap", "gap": {"name": "RASA1"}, "ras": {"name": "HRAS", "activity": {
         "activity_type": "gtpbound", "is_active": True}}}, "prefix": "gap"},
     {"policy": "indra_default"}),
]


@pytest.mark.parametrize("row, binding", ROWS, ids=[r["prefix"] for r, _ in ROWS])
def test_a_row_and_its_binding_read_back_as_they_were(row, binding):
    """The notation is a second way to write the SAME rows: parse(render(row)) builds the model
    row builds, groundings taken from the entity map by name; and its binding, apart, reads back
    as the same binding."""
    text = N.render(row)
    assert text is not None, row
    back = N.parse(text, _GROUND)
    assert _canonical(back) == _canonical(row), text
    kind = row["statement"]["type"]
    said = N.render_binding(binding, kind, M.REGISTRY)
    assert said is not None and "|" not in text + said, binding
    assert N.parse_binding(said, kind, M.REGISTRY) == binding, said


def test_a_policy_keeps_its_prefix_where_dropping_it_is_ambiguous():
    """`indra_default` is written `default`; a library holding an entry named `default` for the
    type would read it back as that entry, so the prefix stays."""
    _, binding = ROWS[1]
    assert N.render_binding(binding, "DecreaseAmount", M.REGISTRY).startswith("default ")
    library = {**M.REGISTRY, ("DecreaseAmount", "default"): None}
    assert N.render_binding(binding, "DecreaseAmount", library).startswith("indra_default ")


def test_a_bare_indra_policy_is_indras_where_the_type_lacks_it():
    """`two_step` on `A becomes B` kept `two_step`, a name no entry has, so asking INDRA's policy
    of a type without it was refused as a misspelling, not counted as the library gap it is."""
    binding = N.parse_binding("two_step", "PopulationTransition", M.REGISTRY)
    assert binding["policy"] == "indra_two_step"


def test_what_the_notation_cannot_write_stays_json():
    """Evidence beyond one source and text, or a statement type it has no verb for, renders as
    None: the caller keeps JSON."""
    row = ROWS[2][0]
    with_evidence = {**row, "statement": {**row["statement"],
                                          "evidence": [{"text": "x"}, {"text": "y"}]}}
    assert N.render(with_evidence) is None
    form = {"statement": {"type": "ActiveForm", "agent": {"name": "RAF"}, "activity": "kinase",
                          "is_active": True},
            "prefix": "f"}
    assert N.render(form) is None
    assert N.render_binding({"policy": "basal", "parameters": {"k": True}}, "Phosphorylation") is None
    assert N.render_binding({}, "Phosphorylation") is None  # no policy: nothing to write


def test_a_binding_reads_back_whatever_its_names_hold():
    """The policy was written bare and a value quoted on spaces and `|` alone, so a written
    mechanism named with a space, or a value holding `#` or a quote, rendered as a binding that
    read back otherwise or not at all."""
    binding = {"policy": "my rate", "parameters": {"kf": "k#1", "kr": 'a"b', "kc": "2.5"}}
    said = N.render_binding(binding, "Phosphorylation")
    assert N.parse_binding(said, "Phosphorylation") == binding, said


def test_a_prefix_the_notation_cannot_read_back_stays_json():
    """`r.1` rendered as a row, which `parse` refused for its prefix."""
    row = {**ROWS[2][0], "prefix": "r.1"}
    assert N.render(row) is None


@pytest.mark.parametrize(
    "text, where",
    [
        ("r1 EGFR phosphorylates AKT1", "prefix"),
        ("r1: EGFR frobnicates AKT1", "unknown verb"),
        ("r1: EGFR{qY1} phosphorylates AKT1", "unknown modification"),
        ("r1: EGFR{kinase:maybe} phosphorylates AKT1", ":on or :off"),
    ],
)
def test_a_malformed_row_says_where(text, where):
    with pytest.raises(N.NotationError, match=where):
        N.parse(text)


def test_a_row_with_its_binding_inline_is_refused_with_the_two_apart():
    """The row is the claim and its binding the maths, given apart: a row still written the way
    the route took before, `<statement> | <binding>`, is refused, one notation, with the row and
    the binding it splits into, each of which reads as written."""
    with pytest.raises(N.InlineBinding) as refused:
        N.parse('b_p: A{!L} phosphorylates B at S10 | two_step kf=k_on kr="k*2" # [src] Doe 2000')
    said = str(refused.value)
    row, binding = "b_p: A{!L} phosphorylates B at S10 # [src] Doe 2000", 'two_step kf=k_on kr="k*2"'
    assert f"`{row}` with `bindings={json.dumps({'b_p': binding})}`" in said
    # Sent in an edit, `bindings` is refused: the advice is that edit with its own `binding`.
    edit = json.dumps({"op": "replace_row", "row": row, "binding": binding})
    assert f"`{edit}`" in refused.value.advice("replace_row")
    assert N.parse(row)["statement"]["evidence"] == [{"source_api": "src", "text": "Doe 2000"}]
    assert N.parse_binding(binding, "Phosphorylation", M.REGISTRY) == {
        "policy": "indra_two_step", "parameters": {"kf": "k_on", "kr": "k*2"}}


@pytest.mark.parametrize(
    "text, where",
    [
        ("", "names its policy"),
        ("r1: default kf=k", "its key in `bindings`"),
        ("default kf=k # [src] text", "carries no evidence"),
        ("inhibited inh:=C", "names no regulator"),
        ("default kf", "expected '='"),
    ],
)
def test_a_malformed_binding_says_where(text, where):
    with pytest.raises(N.NotationError, match=where):
        N.parse_binding(text, "Phosphorylation", M.REGISTRY)


def test_a_statements_evidence_rides_as_a_comment():
    """A statement may carry a source and a justification; written as JSON only, such a row
    could not use the notation at all. Its text may hold a `|`: the comment is read first."""
    row = {**ROWS[2][0], "statement": {**ROWS[2][0]["statement"], "evidence": [
        {"source_api": "curated", "text": "Doe et al. 2020 (10.1000/xyz123) | v4 # x"}]}}
    text = N.render(row)
    assert text.endswith("# [curated] Doe et al. 2020 (10.1000/xyz123) | v4 # x")
    back = N.parse(text, _GROUND)
    assert back["statement"]["evidence"] == row["statement"]["evidence"]
    assert _canonical(back) == _canonical(row)


def test_a_derived_kind_is_marked_and_reads_back_as_stated():
    """get_model marks a modulated regulator with the direction its rate proves; the marked row,
    sent back, states what it stated."""
    row = N.parse("r: A becomes B, modulated by C and D")
    text = N.render(row, {"modulated_by_1": "activated_by"})
    assert text == "r: A becomes B, modulated by C and D (activates, from its rate)"
    assert N.parse(text) == row


def test_a_reversed_or_downgraded_kind_is_marked_and_reads_back_as_stated():
    row = N.parse("r: A becomes B, inhibited by C and D")
    text = N.render(row, {"inhibited_by_0": "activated_by", "inhibited_by_1": "modulated_by"})
    assert text == (
        "r: A becomes B, inhibited by C (activates, from its rate) and D (modulates, from its "
        "rate)"
    )
    assert N.parse(text) == row
