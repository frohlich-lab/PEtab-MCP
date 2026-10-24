"""The regulator corpus (``tests/data/regulator_corpus``): every repro from the reviews of
PEtab-MCP#127 and #128, and every mechanism agents wrote in two NEMO runs, through the check of
a row's regulators (:mod:`petab_mcp.regulation`).

- ``rows.json``: rows as a model states them, with the mechanisms they use, built as
  `make_changes` builds them (:func:`problem_sbml.to_antimony`), ``time`` over an entry's
  ``times`` where it gives them, and signed again on its ``problem`` where it gives one (a
  parameter table, conditions, measurement times and the model's values: :func:`regulation.resign`
  as a commit does).
- ``mechanisms.json``: mechanisms alone (perica_egf's, sweep_indra_55max's), each a reader of
  amounts other than its roles re-expressed as a regulator it states, built on a statement of
  agents named by role with every complex its rules start from present, as other rows would make
  them.

For each entry: no crash, done within :data:`BOUND` seconds, a refusal only of a kind
:data:`PROVEN` lists, and each direction a built row records borne out by J at points of its own,
apart from the check's. ``verdicts.json`` holds each entry's verdict and the kinds its rows record,
so a change to any shows in review: ``python tests/test_regulator_corpus.py`` writes it again.
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pysb
import pytest
import sympy
from indra.statements import Agent, stmts_from_json

from petab_mcp import mechanisms as M
from petab_mcp import notation, problem_sbml, regulation
from petab_mcp.errors import PetabMcpError
from petab_mcp.schemes import placeholder
from petab_mcp.statements import indra_part, regulators, with_defaults

CORPUS = Path(__file__).parent / "data" / "regulator_corpus"
#: Seconds one entry may take.
BOUND = 60.0
#: What a refusal may be, each by what its message says: the stated <=> read contract, and a
#: regulator no rate the row has can change with (one that needs the product present apart). Any
#: other claim is recorded, never refused.
KINDS = {
    "unstated": r"which its statement does not state",
    "unread": r"does not read: the maths would not carry the claim",
    "unnamed": r"which its statement does not name|as an agent of its own",
    "unformed": r"nothing in the model forms",
    "unseen": r"does not change with it: a statement's regulator is about its own reaction",
    "needs_product": r"needs the product present, and its statement's own reaction rate holds",
}
PROVEN = set(KINDS)


def _entries() -> list[dict]:
    rows = json.loads((CORPUS / "rows.json").read_text())
    alone = json.loads((CORPUS / "mechanisms.json").read_text())
    return [{**e, "form": "rows"} for e in rows] + [{**e, "form": "alone"} for e in alone]


def _library(definitions) -> dict:
    library = dict(M.REGISTRY)
    for definition in definitions:
        library[(definition["statement_type"], definition["policy"])] = M.written(definition, library)
    return library


def _rows(entry) -> tuple[pysb.Model, list[tuple], list]:
    """``entry``'s rows built as `make_changes` builds them, and the model they built."""
    library = _library(entry.get("mechanisms", []))
    parsed = [notation.parse(r, library=library) if isinstance(r, str) else dict(r)
              for r in entry["rows"]]
    for row in parsed:
        row["statement"] = with_defaults(row["statement"])
    stmts = stmts_from_json([indra_part(row["statement"]) for row in parsed])
    rows = [(s, row["policy"], row["prefix"], row.get("parameters"), regulators(row["statement"]))
            for s, row in zip(stmts, parsed)]
    seen = {}
    record = regulation.record

    def kept(model, rows_, emissions, times=None):
        seen.update(model=model, emissions=record(model, rows_, emissions, times))
        return seen["emissions"]

    regulation.record = kept
    try:
        times = entry.get("times") and tuple(entry["times"])
        problem_sbml.to_antimony(rows, {}, library, times=times)
    finally:
        regulation.record = record
    return seen["model"], rows, seen["emissions"]


def _alone(entry) -> tuple[pysb.Model, list[tuple], list]:
    """``entry``'s mechanism on a statement of agents named by role, stating each regulator it
    reads, with every complex its rules start from present."""
    definition = entry["mechanism"]
    library = _library([definition])
    stated: dict[str, list[Agent]] = {}
    for name in entry["stated"]:
        kind = re.match(r"(inhibited|activated|modulated)_by", name).group(0)
        i = int(name.rsplit("_", 1)[1]) if name[-1].isdigit() else 0
        given = stated.setdefault(kind, [])
        given += [Agent(f"{kind[:3].upper()}{len(given) + k}") for k in range(i + 1 - len(given))]
    stmt = placeholder(definition["statement_type"])
    rows = [(stmt, definition["policy"], "r", None, stated)]
    model, emissions = M.assemble(rows, {}, library)
    problem_sbml._seed(model, list(model.monomers))
    present = {str(i.pattern) for i in model.initials}
    for rule in model.rules:
        for cp in rule.reactant_pattern.complex_patterns:
            if cp is None:
                continue
            full = pysb.ComplexPattern([
                mp.monomer(**{
                    site: mp.monomer.site_states[site][0] if site in mp.monomer.site_states else None
                    for site in mp.monomer.sites
                } | mp.site_conditions)
                for mp in cp.monomer_patterns
            ], None)
            if full.is_concrete() and str(full) not in present:
                present.add(str(full))
                amount = pysb.Parameter(f"__given{len(present)}", 1.0, _export=False)
                model.add_component(amount)
                model.add_initial(pysb.Initial(full, amount, _export=False))
    problem_sbml._expand(model)
    return model, rows, regulation.record(model, rows, emissions)


class _Problem:
    """An entry's ``problem``, as :func:`regulation.bounded` reads one."""

    def __init__(self, spec: dict):
        self.parameter_df = pd.DataFrame(spec.get("parameters") or [{"parameterId": ""}])
        self.condition_df = pd.DataFrame(spec["conditions"]) if spec.get("conditions") else None
        self.measurement_df = pd.DataFrame({"time": spec.get("times", [0.0]), "experimentId": ""})
        self.experiment_df = None
        self.model = self
        self.values = spec.get("values", {})

    def get_parameter_value(self, pid: str) -> float:
        return self.values[pid]


def _resigned(entry, rows, emissions) -> list:
    """``emissions`` with the kinds ``entry``'s ``problem`` records."""
    from dataclasses import asdict, replace

    def statement(stmt, stated) -> dict:
        return stmt.to_json() | {k: [a.to_json() for a in v] for k, v in (stated or {}).items()}

    record = {"rows": [
        {"prefix": e.prefix, "emission": asdict(e), "statement": statement(stmt, stated)}
        for (stmt, *_, stated), e in zip(rows, emissions)
    ]}
    signed, _, _ = regulation.resign(record, _Problem(entry["problem"]))
    return [replace(e, derived=row["emission"]["derived"])
            for e, row in zip(emissions, signed["rows"])]


def _verdict(entry) -> tuple[dict, tuple | None]:
    """``entry``'s verdict, and what it built where it built."""
    try:
        built = (_rows if entry["form"] == "rows" else _alone)(entry)
        if entry.get("problem"):
            built = (*built[:2], _resigned(entry, *built[1:]))
    except PetabMcpError as exc:
        text = str(exc)
        kinds = [k for k, pattern in KINDS.items() if re.search(pattern, text)]
        return {"verdict": "refused", "kind": kinds[0] if kinds else "build"}, None
    derived = {e.prefix: e.derived for e in built[2] if e.derived}
    return {"verdict": "accepted", **({"derived": derived} if derived else {})}, built


def _against(rate: regulation.NetRate, stated: str, claim: int) -> int:
    """How many of 2048 points of its own, apart from :func:`regulation.sampled`'s, show J moving
    against ``claim`` as the regulator at ``stated`` rises."""
    generator = np.random.default_rng(1)
    symbols = sorted(rate.value.free_symbols, key=str)
    n = 2048
    columns = []
    for s in symbols:
        values = 10 ** generator.uniform(-4, 4, n)
        if not s.is_positive:
            values[: n // 8] = 0.0
        columns.append(values)
    f = sympy.lambdify(symbols, rate.value, "numpy")
    against = 0
    for symbol in rate.regulators.get(stated, ()):
        if symbol not in symbols:
            continue
        k = symbols.index(symbol)
        moved = list(columns)
        moved[k] = columns[k] * (1 + 1e-5) + 1e-5
        with np.errstate(all="ignore"):
            j0 = np.asarray(f(*columns), dtype=complex) * np.ones(n)
            j1 = np.asarray(f(*moved), dtype=complex) * np.ones(n)
        real = (np.isfinite(j0) & np.isfinite(j1) & (np.abs(j0.imag) < 1e-9)
                & (np.abs(j1.imag) < 1e-9) & (np.maximum(np.abs(j0), np.abs(j1)) > 1e-250))
        step = (j1 - j0).real[real] * claim
        hits = np.flatnonzero(real)[step < -1e-9 * np.abs(j0.real[real])]
        # Read again to 50 digits: a J written as a difference rounds to the wrong sign in floats.
        against += sum(
            regulation._precise_way(rate.value, symbols, columns, moved, h) == -claim
            for h in hits[:8]
        )
    return against


_ENTRIES = _entries()
_VERDICTS = json.loads((CORPUS / "verdicts.json").read_text()) if (CORPUS / "verdicts.json").is_file() else {}


@pytest.mark.parametrize("entry", _ENTRIES, ids=[e["id"] for e in _ENTRIES])
def test_the_corpus_entry_keeps_its_verdict(entry):
    start = time.monotonic()
    verdict, built = _verdict(entry)
    assert time.monotonic() - start < BOUND
    assert verdict["verdict"] == "accepted" or verdict["kind"] in PROVEN | {"build"}
    assert verdict == _VERDICTS.get(entry["id"]), "a changed verdict: rewrite verdicts.json"
    if built is None:
        return
    model, rows, emissions = built
    for (stmt, *_, stated), emission in zip(rows, emissions):
        recorded = {
            f"{kind}_{i}": emission.derived.get(f"{kind}_{i}", kind)
            for kind, agents in (stated or {}).items() for i in range(len(agents))
        }
        claims = {s: {"inhibited_by": -1, "activated_by": 1}[k] for s, k in recorded.items()
                  if k != "modulated_by"}
        if not claims:
            continue
        rate = regulation.net_rate(model, emission.rules, stmt)
        if entry.get("problem"):
            rate = regulation.bounded(rate, _Problem(entry["problem"]))[0]
        for s, claim in claims.items():
            assert _against(rate, s, claim) == 0, f"{s}: J moves against what it records"


if __name__ == "__main__":
    verdicts = {}
    for entry in _ENTRIES:
        verdicts[entry["id"]] = _verdict(entry)[0]
        print(entry["id"], verdicts[entry["id"]], file=sys.stderr)
    (CORPUS / "verdicts.json").write_text(json.dumps(verdicts, indent=1, sort_keys=True) + "\n")
