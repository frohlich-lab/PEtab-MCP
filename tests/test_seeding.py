"""Seeding from the pinned corpus: entity filtering, then the readout-ranked cap."""

from __future__ import annotations

import gzip
import json

from indra.statements import (
    Activation,
    Agent,
    Complex,
    Inhibition,
    Phosphorylation,
    Translocation,
    stmts_to_json,
)

from petab_mcp import seeding as Sd

# The problem's species, as the entity-map file supplies them; ``{extra}`` splices more entries.
_BASE_ENTITY_MAP = (
    '{{"MEK": {{"name": "MAP2K1", "db_refs": {{"HGNC": "6840"}}}}, '
    '"pERK": {{"name": "MAPK1", "db_refs": {{"HGNC": "6871"}}, '
    '"mods": [{{"mod_type": "phosphorylation", "residue": "T", "position": "185"}}]}}{extra}}}'
)

_MEK = Agent("MAP2K1", db_refs={"HGNC": "6840", "TEXT": "MEK"})
_ERK = Agent("MAPK1", db_refs={"HGNC": "6871"})
_RAF = Agent("RAF1", db_refs={"HGNC": "9829"})


def _corpus(tmp_path, stmts, extra=""):
    """Write ``stmts`` as the corpus file and the entity map beside it; filter one by the other."""
    (tmp_path / "corpus.json").write_text(json.dumps(stmts_to_json(stmts)))
    (tmp_path / "entities.json").write_text(_BASE_ENTITY_MAP.format(extra=extra))
    entities = Sd.load_entity_map(str(tmp_path / "entities.json"))
    return Sd.filter_corpus(str(tmp_path / "corpus.json"), entities)


def test_filter_needs_every_agent_mapped(tmp_path):
    """One unmapped agent would put a species the problem does not have into the seed."""
    kept = _corpus(tmp_path, [Phosphorylation(_MEK, _ERK), Phosphorylation(_RAF, _MEK)])
    assert [str(s) for s in kept] == [str(Phosphorylation(_MEK, _ERK))]
    kept = _corpus(
        tmp_path,
        [Phosphorylation(_RAF, _MEK)],
        extra=', "RAF": {"name": "RAF1", "db_refs": {"HGNC": "9829"}}',
    )
    assert len(kept) == 1


def test_filter_ignores_reader_text(tmp_path):
    """A shared ``TEXT`` string is not a shared identity."""
    stranger = Agent("MEK_like", db_refs={"TEXT": "MEK"})
    assert _corpus(tmp_path, [Phosphorylation(stranger, _ERK)]) == []


_AKT = Agent("AKT", db_refs={"FPLX": "AKT"})
_S6 = Agent("RPS6", db_refs={"HGNC": "10429"})
_P_ERK = Agent._from_json(
    {
        "name": "MAPK1",
        "db_refs": {"HGNC": "6871"},
        "mods": [{"mod_type": "phosphorylation", "residue": "T", "position": "185"}],
    }
)


def _ranked(*beliefs_and_stmts):
    for belief, stmt in beliefs_and_stmts:
        stmt.belief = belief
    return [stmt for _, stmt in beliefs_and_stmts]


def test_readouts_take_turns():
    """By belief alone the most-studied gene takes the whole cap: Fujita's measured S6 got no
    statement while AKT and EGFR filled all 25."""
    stmts = _ranked(
        (1.0, Activation(_MEK, _ERK)),
        (1.0, Activation(_RAF, _ERK)),
        (0.5, Phosphorylation(_AKT, _S6)),
    )
    seeded = Sd.seed(stmts, [_P_ERK, _S6], budget=2)
    assert stmts[2] in seeded


def test_unusable_statements_never_seed():
    """What filled the first Fujita seed: no cause, a site the data does not measure, and an
    n-ary complex that BioNetGen expands without end."""
    stmts = _ranked(
        (1.0, Phosphorylation(None, _ERK)),
        (1.0, Phosphorylation(_MEK, _ERK, "Y", "187")),
        (1.0, Complex([_MEK, _ERK, _RAF])),
        (0.1, Phosphorylation(_MEK, _ERK, "T", "185")),
    )
    assert Sd.seed(stmts, [_P_ERK]) == [stmts[-1]]


def test_one_statement_per_interaction():
    """Readers describe one interaction several ways; seeding each reading multiplied Fujita's
    EGF-EGFR edge into thousands of reactions. Evidence breaks the belief tie: 183 of Fujita's
    717 statements sit at belief 1.0."""
    weak, strong = _ranked((1.0, Activation(_MEK, _ERK)), (1.0, Inhibition(_MEK, _ERK)))
    weak.n_evidence, strong.n_evidence = 2, 40
    assert Sd.seed([weak, strong], [_P_ERK]) == [strong]


def test_self_action_seeds():
    """At a coarse-grained level ``AKT activates AKT`` is a feedback loop through steps the model
    does not name, not a reading error, and it is a different interaction from one on AKT alone."""
    loop, alone = _ranked(
        (1.0, Activation(_AKT, _AKT)), (1.0, Translocation(_AKT, None, "nucleus"))
    )
    assert len(Sd.seed([loop, alone], [_AKT])) == 2


def test_the_measured_state_wins_and_an_open_site_reads_as_it():
    """Fujita's seed took ``EGFR activates AKT`` (205 papers) over ``EGFR phosphorylates AKT``
    (35); an activation moves no observable reading phospho-S473, and INDRA names an open site
    ``phospho``, which none reads either, so pAKT never left zero."""
    activates, phosphorylates = _ranked(
        (1.0, Activation(_MEK, _ERK)), (1.0, Phosphorylation(_MEK, _ERK))
    )
    activates.n_evidence, phosphorylates.n_evidence = 205, 35
    (seeded,) = Sd.seed([activates, phosphorylates], [_P_ERK])
    assert isinstance(seeded, Phosphorylation)
    assert (seeded.residue, seeded.position) == ("T", "185")
    assert phosphorylates.position is None  # the corpus statement itself is untouched


def test_seed_order_is_independent_of_corpus_order():
    """Equal rank breaks on the statement hash, so a reshuffled corpus seeds the same model."""
    stmts = [Activation(_MEK, _ERK), Activation(_RAF, _ERK), Phosphorylation(_AKT, _S6)]
    forward = Sd.seed(stmts, [_P_ERK, _S6], budget=2)
    assert Sd.seed(stmts[::-1], [_P_ERK, _S6], budget=2) == forward


def test_filter_reads_gzipped_corpora_and_keeps_evidence_counts(tmp_path):
    """The corpora are pinned gzipped, and INDRA's reader drops ``n_evidence``, which is the only
    thing that ranks statements tied at belief 1.0."""
    record = Phosphorylation(_MEK, _ERK).to_json()
    record["n_evidence"] = 8966
    with gzip.open(tmp_path / "corpus.json.gz", "wt") as fh:
        json.dump([record], fh)
    (tmp_path / "entities.json").write_text(_BASE_ENTITY_MAP.format(extra=""))
    entities = Sd.load_entity_map(str(tmp_path / "entities.json"))
    (stmt,) = Sd.filter_corpus(str(tmp_path / "corpus.json.gz"), entities)
    assert stmt.n_evidence == 8966


def test_inputs_are_declared_in_the_entity_map(tmp_path):
    """An input is a fact about the experiment, declared beside the readouts; an entry without one
    is no input."""
    extra = ', "EGF": {"name": "EGF", "input": "piecewise(EGF_0, time < EGF_end, 0)"}'
    (tmp_path / "entities.json").write_text(_BASE_ENTITY_MAP.format(extra=extra))
    assert Sd.load_inputs(str(tmp_path / "entities.json")) == {
        "EGF": "piecewise(EGF_0, time < EGF_end, 0)"
    }
