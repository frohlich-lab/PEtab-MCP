"""The entity map, and the INDRA warm start it filters a pinned corpus by.

The entity map is SUPPLIED, not derived: no corpus knows that this problem's ``A_p`` means
protein PROTA phosphorylated at S10. It is read from a path on the command line, so
benchmark-specific data stays out of this package. Each species maps to an INDRA ``Agent`` in
INDRA's own JSON, so the state rides in INDRA's own ``mods``::

    {"A_p": {"name": "PROTA", "db_refs": {"UP": "<accession>"},
             "mods": [{"mod_type": "phosphorylation", "residue": "S", "position": "10"}]}}

The same map names each species' PySB pattern (:func:`problem_sbml.to_antimony`), which is what
binds a model built from statements to the fixed observable table. It does not declare which
species the experiment sets: the problem does, by its condition targets and, for an input that
changes within a period, its ``inputs.tsv`` (:mod:`protocols`), which every route reads (see
RATIONALE.md).

The warm start (``--corpus``, :func:`warm_start`) seeds the session's first model from a pinned
statement corpus in three stages, each capped: keep the corpus statements all of whose agents the
map grounds (:func:`filter_corpus`), choose up to :data:`SEED_MAX_STATEMENTS` of those touching a
readout (:func:`seed`), and build each with INDRA's default mechanism. Defaults rather than curated
entries on purpose: the seed is a floor the agent improves on, and seeding with the best available
entries would make the library's own contribution unmeasurable against the baseline it is meant to
beat. The server commits the seed through the ``statements`` route like any other model, so it is
fitted, versioned and the agent's to edit.

The cap is not optional. Entity filtering alone leaves 83 to 1,186 statements per benchmark in
the published coverage table; instantiating them gives a model with more reactions than
measurements, which neither fits nor reads. What fills that cap is the whole difficulty (see
:func:`seed`). Groundings are compared as ``"NS:ID"`` strings.
"""

from __future__ import annotations

import difflib
import gzip
import json
from copy import deepcopy
from pathlib import Path

from indra.statements import (
    ActiveForm,
    Agent,
    Autophosphorylation,
    Complex,
    ModCondition,
    Modification,
    Statement,
    stmts_from_json,
)

from .errors import EntityMapError, ModelUpdateError, PetabMcpError, SeedError

#: Ceiling on seeded statements, set against the coverage table rather than tuned.
SEED_MAX_STATEMENTS = 25

#: The keys an entry is read by: INDRA's ``Agent`` JSON. Any other key is left to the map's other
#: readers (a benchmark's own annotation keys), except one close to these, which is a typo that
#: would otherwise be dropped without a word.
ENTRY_KEYS = (
    "name",
    "db_refs",
    "mods",
    "mutations",
    "bound_conditions",
    "activity",
    "location",
)


def _entries(path: str) -> dict[str, dict]:
    """The entity map at ``path``, refused at the first entry that would not be read as written."""
    entries = json.loads(Path(path).read_text())
    if not isinstance(entries, dict):
        raise EntityMapError(
            f"entity map {path} is a JSON {type(entries).__name__}; write an object from each "
            'species id to its INDRA agent: {"A_p": {"name": "PROTA", ...}}'
        )
    for species, entry in entries.items():
        where = f"entity map {path}, entry {species!r}"
        if not isinstance(entry, dict):
            raise EntityMapError(
                f'{where} is {entry!r}; write the INDRA agent it is: {{"name": "PROTA", ...}}'
            )
        if not entry.get("name") or not isinstance(entry["name"], str):
            raise EntityMapError(
                f'{where} needs the name of the INDRA agent it is: "name": "PROTA"'
            )
        if "input" in entry:
            raise EntityMapError(
                f"{where} declares an input; declare it in the inputs.tsv beside problem.yaml, "
                "which every route reads"
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


def _groundings(agent: Agent) -> set[str]:
    # ``TEXT`` is what a reader saw, not an identity (see :func:`statements.ungrounded`).
    return {f"{ns}:{ref}" for ns, ref in agent.db_refs.items() if ns != "TEXT"}


def load_corpus(path: str | Path) -> tuple[list[Statement], dict]:
    """The statements of the pinned corpus at ``path`` (JSON, or gzipped JSON as corpora are
    pinned: an object whose ``statements`` are INDRA's JSON), and everything else it records —
    the query, the service and the date that pinned it.

    Each statement keeps the corpus's ``n_evidence`` as an attribute of that name: INDRA's reader
    drops the field, and a pinned corpus lists only a sample of the evidence it counts. Records are
    read one at a time, so one INDRA cannot read (a type it lacks, or no object at all) is skipped
    and every count stays on its own statement.
    """
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as fh:
        corpus = json.load(fh)
    if not isinstance(corpus, dict) or not isinstance(corpus.get("statements"), list):
        raise SeedError(
            f"corpus {path} is no object with a `statements` list of INDRA statement JSON"
        )
    stmts = []
    for record in corpus["statements"]:
        if not isinstance(record, dict):
            continue
        for stmt in stmts_from_json([record]):
            stmt.n_evidence = record.get("n_evidence", len(stmt.evidence))
            stmts.append(stmt)
    return stmts, {k: v for k, v in corpus.items() if k != "statements"}


def _agents(stmt: Statement) -> list[Agent]:
    """Every agent ``stmt`` names: its roles, and the partners each must be bound to or free of."""
    found, todo = [], list(stmt.real_agent_list())
    while todo:
        agent = todo.pop()
        found.append(agent)
        todo += [bc.agent for bc in agent.bound_conditions]
    return found


def filter_corpus(stmts: list[Statement], entities: dict[str, Agent]) -> list[Statement]:
    """The statements all of whose agents, bound partners included, carry a grounding in
    ``entities``, each agent named as the map names it where its groundings point at one name:
    PySB keys a molecule on its name, so a corpus ``AKT1`` beside a map ``Akt`` would build two
    molecules, and the readout would read the one no statement touches. A partner the map does not
    ground is one no seeded statement binds, so a rule that needs it would never fire."""
    names: dict[str, set[str]] = {}
    for agent in entities.values():
        for grounding in _groundings(agent):
            names.setdefault(grounding, set()).add(agent.name)
    kept = []
    for stmt in stmts:
        if not all(_groundings(a) & names.keys() for a in _agents(stmt)):
            continue
        stmt = deepcopy(stmt)
        for agent in _agents(stmt):
            (name, *more) = set().union(*(names.get(g, set()) for g in _groundings(agent)))
            if not more and name != agent.name:
                agent.name = name
                stmt._shallow_hash = stmt._full_hash = None
        kept.append(stmt)
    return kept


def _substrate(stmt: Statement) -> Agent | None:
    """The agent whose modification ``stmt`` changes, if it changes one: an enzyme's substrate,
    or the agent that phosphorylates itself."""
    if isinstance(stmt, Modification):
        return stmt.sub
    if isinstance(stmt, Autophosphorylation):
        return stmt.enz
    return None


def _usable(stmt: Statement, readouts: list[Agent]) -> bool:
    """Whether ``stmt`` can stand as a seed: a type INDRA's default builds a rule for, every role
    filled, no complex of more than two, and a site some readout of its substrate measures.

    Each clause names what broke a Fujita seed: 11 of 25 statements had no cause ("AKT is
    phosphorylated on some threonine"), others modified AKT at T450 or EGFR at T678 where the data
    measures S473 and Y1068, and one EGF-EGFR-AKT complex kept BioNetGen expanding for minutes —
    INDRA assembles an n-ary complex as pairwise bonds, which chain without end, while the nine
    statements beside it closed at 25 reactions. ``ActiveForm`` has a default but builds no rule,
    so the statements route refuses its row. An agent acting on itself (``AKT activates AKT``) is
    deliberately NOT refused: at a coarse-grained level it is a feedback loop through steps the
    model does not name.
    """
    from .mechanisms import INDRA_DEFAULT, REGISTRY

    if isinstance(stmt, ActiveForm) or (type(stmt).__name__, INDRA_DEFAULT) not in REGISTRY:
        return False
    agents = stmt.agent_list()
    if None in agents:
        return False
    if isinstance(stmt, Complex) and len(agents) > 2:
        return False
    substrate = _substrate(stmt)
    if substrate is not None:
        sites = {
            (m.residue, m.position)
            for readout in readouts
            if _groundings(substrate) & _groundings(readout)
            for m in readout.mods
        }
        if sites and not any(
            stmt.residue in (None, residue) and stmt.position in (None, position)
            for residue, position in sites
        ):
            return False
    return True


def _measured_site(stmt: Statement, readouts: list[Agent]) -> ModCondition | None:
    """The readout modification ``stmt`` adds or removes, if it changes one."""
    substrate = _substrate(stmt)
    if substrate is None:
        return None
    # A removal's condition is the modification it removes: Dephosphorylation's is phosphorylation.
    mod_type = stmt._get_mod_condition().mod_type
    for readout in readouts:
        if _groundings(substrate) & _groundings(readout):
            for mod in readout.mods:
                if mod.mod_type == mod_type and stmt.residue in (None, mod.residue):
                    if stmt.position in (None, mod.position):
                        return mod
    return None


def _on_measured_site(stmt: Statement, readouts: list[Agent]) -> Statement:
    """``stmt`` with an unnamed site read as the measured one: ``EGFR phosphorylates AKT`` seeds
    as phosphorylation at S473, and ``PP2A dephosphorylates AKT`` as its removal. INDRA otherwise
    names the site ``phospho``, a state no observable reads. A coarse-graining assumption, made
    only where the statement leaves the site open."""
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
    readout, a statement changing the measured state ranks first — an ``Activation`` of AKT
    moves no observable that reads phospho-S473 — then belief, then evidence count (183 of
    Fujita's 717 tie at belief 1.0, so belief alone ranks almost nothing), then hash, so a corpus
    seeds the same model on every run.

    One statement stands for each set of agent names, so the number of such sets touching a
    readout caps the seed wherever it is below ``budget``: four entities make at most 12.
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


def readouts(session) -> list[Agent]:
    """The entity-map agents the problem's observables read: those a seed must touch."""
    from petab.v2.C import OBSERVABLE_FORMULA

    from .reconcile import formula_identifiers

    obs = session.problem.observable_df
    read = set().union(*(formula_identifiers(f) for f in obs[OBSERVABLE_FORMULA]))
    return [agent for species, agent in session.entities.items() if species in read]


def warm_start(session, path: str) -> tuple[list[dict], str]:
    """The ``statements`` rows the INDRA warm start commits for ``session`` from the corpus at
    ``path``, and the justification it commits them with.

    Each chosen statement is a row under INDRA's default mechanism, prefixed ``seed<rank>``. A
    row whose rules never fire from the start, or that chains complexes without end, is left out
    and the rest rebuilt, since the seed is a floor and such a row moves nothing; a model that
    still does not build — a readout no chosen statement produces — is :class:`SeedError`.
    """
    from . import operations
    from .mechanisms import INDRA_DEFAULT

    read = readouts(session)
    if not read:
        raise SeedError(
            "the entity map binds no species the observables read, so no seed can reach the data; "
            "a warm start needs the map's readouts"
        )
    stmts, pinned = load_corpus(path)
    kept = filter_corpus(stmts, session.entities)
    rows = [
        {
            "statement": {
                k: v
                for k, v in stmt.to_json().items()
                if k not in ("id", "belief", "evidence", "matches_hash")
            },
            "policy": INDRA_DEFAULT,
            "prefix": f"seed{rank}",
        }
        for rank, stmt in enumerate(seed(kept, read), 1)
    ]
    left_out: list[str] = []
    while True:
        if not rows:
            raise SeedError(
                f"corpus {path}: none of its {len(kept)} statements among the entity map's agents "
                "seeds a row that builds"
                + (f" (left out: {', '.join(left_out)})" if left_out else "")
            )
        try:
            operations.statements_to_antimony(session, rows)
            break
        except ModelUpdateError as exc:
            if not exc.rows:
                raise SeedError(
                    f"corpus {path} seeds no model this problem can score: {exc}"
                ) from exc
            left_out += exc.rows
            rows = [row for row in rows if row["prefix"] not in exc.rows]
        except PetabMcpError as exc:
            raise SeedError(f"corpus {path} seeds no model this problem can score: {exc}") from exc
    pinned_by = ", ".join(str(pinned[k]) for k in ("source", "date") if k in pinned)
    justification = (
        f"INDRA warm start: {len(rows)} statement{'s' * (len(rows) != 1)} of the pinned corpus "
        f"{Path(path).name}{f' ({pinned_by})' if pinned_by else ''}, chosen from the "
        f"{len(kept)} among the entity map's agents, the readouts taking turns, each built with "
        f"INDRA's default mechanism."
    )
    if left_out:
        justification += f" Left out, as their rules never fire or chain: {', '.join(left_out)}."
    return rows, justification
