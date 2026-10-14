"""Seed a model from the pinned statement corpus.

Three stages, each capped: map this problem's species to groundings, filter the corpus to
statements all of whose agents are in that map, and instantiate the survivors with INDRA's
DEFAULT policies. Defaults rather than curated entries on purpose — the seed is a floor the
agent improves on, and seeding with the best available entries would make the library's own
contribution unmeasurable against the baseline it is meant to beat. That third stage is INDRA's
``PysbAssembler().make_model()`` as it stands, so it has no function here.

The cap is not optional. Entity filtering alone leaves 83 to 1,186 statements per benchmark in
the published coverage table; instantiating them gives a model with more reactions than
measurements, which neither fits nor reads. The seed takes statements at path length one from a
readout to :data:`SEED_MAX_STATEMENTS`, and what fills that cap is the whole difficulty (see
:func:`seed`).

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
rather than asserted as a statement or built by a mechanism (see RATIONALE.md). Groundings are
compared as ``"NS:ID"`` strings.

So is what is present at the start, when the default does not fit (:func:`load_initials`). An
entry may declare its species' INITIAL amount, as a number or as an expression over parameter ids
(a shipped row, a condition-table column)::

    {"Pathogen": {"name": "Pathogen", "initial": 1},
     "SMAD2": {"name": "SMAD2", "db_refs": {"UP": "Q62432"}, "initial": "S2tot"}}

:func:`problem_sbml.to_antimony` says what the declaration replaces.
"""

from __future__ import annotations

import gzip
import json
from copy import deepcopy
from pathlib import Path

from indra.statements import (
    AddModification,
    Agent,
    Complex,
    ModCondition,
    Modification,
    Statement,
    stmts_from_json,
)

#: Ceiling on seeded statements, set against the coverage table rather than tuned.
SEED_MAX_STATEMENTS = 25


def _groundings(agent: Agent) -> set[str]:
    # ``TEXT`` is what a reader saw, not an identity (see :func:`statements.ungrounded`).
    return {f"{ns}:{ref}" for ns, ref in agent.db_refs.items() if ns != "TEXT"}


def load_entity_map(path: str) -> dict[str, Agent]:
    """Species id to the INDRA ``Agent`` it is, from the JSON file at ``path``."""
    return {k: Agent._from_json(v) for k, v in json.loads(Path(path).read_text()).items()}


def load_inputs(path: str) -> dict[str, str]:
    """Species id to the protocol formula an experiment sets it by, for each entry of the entity
    map at ``path`` that declares one."""
    entries = json.loads(Path(path).read_text())
    return {k: v["input"] for k, v in entries.items() if "input" in v}


def load_initials(path: str) -> dict[str, float | str]:
    """Species id to the initial amount its entry of the entity map at ``path`` declares (a number,
    or an expression over parameter ids), for each entry that declares one."""
    entries = json.loads(Path(path).read_text())
    return {k: v["initial"] for k, v in entries.items() if "initial" in v}


def filter_corpus(path: str, entities: dict[str, Agent]) -> list[Statement]:
    """The corpus statements at ``path`` (JSON, or gzipped JSON as the corpora are pinned) all of
    whose agents carry a grounding in ``entities``.

    Each statement keeps the corpus's ``n_evidence`` as an attribute of that name: INDRA's reader
    drops the field, and a pinned corpus lists only a sample of the evidence it counts.
    """
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as fh:
        raw = json.load(fh)
    known = set().union(*map(_groundings, entities.values()))
    kept = []
    for stmt, record in zip(stmts_from_json(raw), raw, strict=True):
        if all(_groundings(a) & known for a in stmt.real_agent_list()):
            stmt.n_evidence = record.get("n_evidence", len(stmt.evidence))
            kept.append(stmt)
    return kept


def _usable(stmt: Statement, readouts: list[Agent]) -> bool:
    """Whether ``stmt`` can stand as a seed: every role filled, no complex of more than two, and
    no site that contradicts the one a readout measures.

    Each clause names what broke a Fujita seed: 11 of 25 statements had no cause ("AKT is
    phosphorylated on some threonine"), others modified AKT at T450 or EGFR at T678 where the data
    measures S473 and Y1068, and one EGF-EGFR-AKT complex kept BioNetGen expanding for minutes —
    INDRA assembles an n-ary complex as pairwise bonds, which chain without end, while the nine
    statements beside it closed at 25 reactions. An agent acting on itself (``AKT activates
    AKT``) is deliberately NOT refused: at a coarse-grained level it is a feedback loop through
    steps the model does not name.
    """
    agents = stmt.agent_list()
    if None in agents:
        return False
    if isinstance(stmt, Complex) and len(agents) > 2:
        return False
    if isinstance(stmt, Modification):
        for readout in readouts:
            sites = {(m.residue, m.position) for m in readout.mods}
            if sites and _groundings(stmt.sub) & _groundings(readout):
                if not any(
                    stmt.residue in (None, residue) and stmt.position in (None, position)
                    for residue, position in sites
                ):
                    return False
    return True


def _measured_site(stmt: Statement, readouts: list[Agent]) -> ModCondition | None:
    """The readout modification ``stmt`` produces, if it produces one."""
    if not isinstance(stmt, AddModification):
        return None
    mod_type = stmt._get_mod_condition().mod_type
    for readout in readouts:
        if _groundings(stmt.sub) & _groundings(readout):
            for mod in readout.mods:
                if mod.mod_type == mod_type and stmt.residue in (None, mod.residue):
                    if stmt.position in (None, mod.position):
                        return mod
    return None


def _on_measured_site(stmt: Statement, readouts: list[Agent]) -> Statement:
    """``stmt`` with an unnamed site read as the measured one: ``EGFR phosphorylates AKT`` seeds
    as phosphorylation at S473. INDRA otherwise names the site ``phospho``, a state no observable
    reads, and Fujita's seeded pAKT stayed at zero. A coarse-graining assumption, made only where
    the statement leaves the site open."""
    site = _measured_site(stmt, readouts)
    if site is None or (stmt.residue, stmt.position) == (site.residue, site.position):
        return stmt
    bound = deepcopy(stmt)
    bound.residue, bound.position = site.residue, site.position
    bound._shallow_hash = bound._full_hash = None
    return bound


def seed(
    stmts: list[Statement], readouts: list[Agent], budget: int = SEED_MAX_STATEMENTS
) -> list[Statement]:
    """Up to ``budget`` usable statements touching a readout, the readouts taking turns. They are
    INDRA's knowledge only, built with INDRA's default mechanisms: what the corpus does not say —
    Fujita's EGF dosing, the dephosphorylation that lets a signal fall — is the agent's to add.

    Turns, because by belief alone the most-studied genes take the whole cap: on Fujita, AKT and
    EGFR appear in ~410 statements each and RPS6 in 56, and the measured S6 got none. Within a
    readout, a statement producing the measured state ranks first — an ``Activation`` of AKT
    moves no observable that reads phospho-S473 — then belief, then evidence count (183 of
    Fujita's 717 tie at belief 1.0, so belief alone ranks almost nothing), then hash, so a corpus
    seeds the same model on every run.
    """

    def rank(stmt: Statement) -> tuple:
        return (
            _measured_site(stmt, readouts) is None,
            -stmt.belief,
            -getattr(stmt, "n_evidence", len(stmt.evidence)),
            stmt.get_hash(),
        )

    # One statement per set of agents. Readers describe one interaction several ways — Fujita's
    # EGF and EGFR came as activation, binding, phosphorylation, inhibition and the reverse
    # activation — and seeding every reading multiplied one edge into 6,064 reactions. The
    # best-ranked reading stands for the interaction, which also drops the weakly evidenced
    # reverse of a well-evidenced edge (RPS6 phosphorylating AKT, 2 papers, against 13).
    usable, seen = [], set()
    candidates = (_on_measured_site(s, readouts) for s in stmts if _usable(s, readouts))
    for stmt in sorted(candidates, key=rank):
        pair = tuple(sorted(a.name for a in stmt.real_agent_list()))
        if pair not in seen:
            seen.add(pair)
            usable.append(stmt)
    queues = [
        [s for s in usable if any(_groundings(a) & _groundings(r) for a in s.real_agent_list())]
        for r in readouts
    ]
    chosen: list[Statement] = []
    while len(chosen) < budget and any(queues):
        for queue in queues:
            while queue and queue[0] in chosen:
                queue.pop(0)
            if queue and len(chosen) < budget:
                chosen.append(queue.pop(0))
    return chosen
