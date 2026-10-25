"""Causal constraints (:mod:`petab_mcp.constraints`, PEtab-MCP#134): claims every model must make,
checked on a statements record's rows, their regulators as their rates record them, and indirect
ones traced through the network the rows expand to (:mod:`petab_mcp.tracing`).

Records are written by hand here, each row a statement in the notation and the kinds its rate
records (``derived``), so each test says exactly which claim it turns on; the model an indirect
constraint is traced on is built from the record's rows as the statements route builds it, each
row under the library's own mechanism for its type (a rate reading its regulators where it states
any). The server tests at the end build one through ``make_changes``.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from petab_mcp import constraints as K
from petab_mcp import mechanisms as M
from petab_mcp import problem_sbml
from petab_mcp.errors import ValidationError
from petab_mcp.operations import OperationError
from petab_mcp.references import StubReferenceResolver
from petab_mcp.server import create_server
from petab_mcp.statements import indra_part, regulators, with_defaults

YAML_PATH = Path(__file__).parent / "data" / "toy_v2" / "problem.yaml"


def _record(*rows) -> dict:
    """A statements record of ``rows``, each ``(prefix, statement, derived?, policy?)``."""
    return {
        "rows": [
            {
                "prefix": prefix,
                "statement": with_defaults(K.statement_json(text)),
                "emission": {"derived": rest[0] if rest else {}},
            }
            | ({"policy": rest[1]} if len(rest) > 1 else {})
            for prefix, text, *rest in rows
        ]
    }


def _model(record: dict) -> tuple:
    """The model ``record``'s rows build, seeded and never reduced, and its emissions: each row
    under its ``policy`` or the library's own mechanism for its type, a row stating regulators
    under one whose rate reads them."""
    from indra.statements import Modification, RemoveModification, SelfModification, stmts_from_json

    library, rows = dict(M.REGISTRY), []
    for row in record["rows"]:
        (stmt,) = stmts_from_json([indra_part(row["statement"])])
        stated = regulators(row["statement"])
        kind = type(stmt).__name__
        if row.get("policy"):
            policy = row["policy"]
        elif kind in ("PopulationTransition", "Conversion"):
            policy = "mass_action"
        elif (isinstance(stmt, Modification | RemoveModification)
              and not isinstance(stmt, SelfModification) and stmt.enz is None):
            policy = "basal"
        else:
            policy = M.INDRA_DEFAULT
        if stated:
            names = [k if len(a) == 1 else f"{k}_{i}" for k, a in stated.items() for i in range(len(a))]
            read = {"statement_type": kind, "policy": f"{policy}_read", "based_on": policy,
                    "rate": "k" + "".join(f" * (1 + {n})" for n in names), "parameters": ["k"]}
            library[(kind, read["policy"])] = M.written(read, library)
            policy = read["policy"]
        rows.append((stmt, policy, row["prefix"], None, stated))
    return problem_sbml.to_antimony(rows, {}, library, seeded=True)


def _check(lines: list[str], record: dict) -> dict[str, K.Verdict]:
    return K.check(K.parse(lines), record, lambda: _model(record))


#: A cascade, as fujita's formulation writes it.
_CASCADE = (
    ("r1", "EGF binds EGFR"),
    ("r9", "EGFR{!EGF} autophosphorylates at Y1068"),
    ("r23", "EGFR{pY1068} phosphorylates AKT1 at S473"),
    ("r56", "AKT1{pS473} phosphorylates RPS6 at S235"),
    ("r8", "RPS6 is dephosphorylated at S235"),
)


# ----------------------------------------------------------------------------------------- read
def test_a_constraint_says_directly_or_indirectly():
    """There is no default kind (#134, decision 1): a constraint without a qualifier is refused,
    shown both ways."""
    with pytest.raises(
        K.ConstraintError,
        match=r"`EGF increases AKT1, directly`.*`EGF increases "
        r"AKT1, indirectly`",
    ):
        K.parse(["c1: EGF increases AKT1"])


def test_there_is_no_signed_indirect_negative():
    """Feedback gives a chain of each sign, so `not ..., indirectly` would refuse true models and
    push the agent to delete true feedback (#134, decision 3)."""
    with pytest.raises(K.ConstraintError, match="no `not ..., indirectly`"):
        K.parse(["c1: not SOCS3 increases STAT5{pY694}, indirectly"])
    K.parse(
        ["c1: not EPO phosphorylates STAT5, directly", "c2: STAT5 does not affect EPO, indirectly"]
    )


def test_an_indirect_constraint_names_a_cause_and_states_no_regulator():
    """A chain connects a cause to an effect: a binding has neither, and a regulator is a claim
    about one row's reaction."""
    with pytest.raises(K.ConstraintError, match="names no cause and effect"):
        K.parse(["c1: A binds B, indirectly"])
    with pytest.raises(K.ConstraintError, match="regulator is a claim about one row"):
        K.parse(["c1: K phosphorylates S at S10, inhibited by X, indirectly"])


def test_activity_is_what_the_file_declares():
    """`activates` is about forms the user declares active, never inferred from the agent's rows
    (#134, decision 4): an undeclared agent's activity is refused as the file loads."""
    with pytest.raises(
        K.ConstraintError, match="activity of AKT1, and no line declares which form of it is active"
    ):
        K.parse(["c1: EGF activates AKT1, indirectly"])
    got = K.parse(["a1: AKT1{pS473} is active", "c1: EGF activates AKT1, indirectly"])
    assert [f.text for f in got.active_forms] == ["AKT1{pS473} is active"]


def test_ids_are_unique():
    with pytest.raises(K.ConstraintError, match="id 'c1' is given twice"):
        K.parse(["c1: A becomes B, directly", "c1: B becomes A, directly"])


def test_a_file_reads_one_constraint_per_line(tmp_path):
    """A line's trailing comment is its source; a line that starts with `#` is no constraint."""
    lines = tmp_path / "constraints.txt"
    lines.write_text("# the bench's claims\nc1: A becomes B, directly  # [curated] Doe 2000\n")
    (c1,) = K.read(lines).items
    assert (c1.id, c1.direct, c1.source) == ("c1", True, "[curated] Doe 2000")


# --------------------------------------------------------------------------------------- direct
def test_a_row_entails_its_reaction_however_it_is_written():
    """A row entails a constraint where its reaction takes what the constraint's takes and makes
    what it makes, driven by the constraint's subject: as its catalyst, as an activator, or by a
    conversion that is the same reaction."""
    lines = ["c1: K phosphorylates S at S10, directly"]
    for row in (
        "K phosphorylates S at S10",
        "S is phosphorylated at S10, activated by K",
        "S{-pS10} -> S{pS10} by K",
    ):
        assert _check(lines, _record(("r", row)))["c1"].rows == ("r",), row
    # Made from nothing, it takes no unphosphorylated S: not that reaction.
    made = _check(lines, _record(("r", "K increases S{pS10}")))["c1"]
    assert not made.ok and "not from what the constraint's reaction takes" in made.why


def test_a_regulator_counts_as_its_rate_records_it():
    """The constraint rests on the maths: a row stating SOCS3 as an activator of a rate that falls
    with it (#128 records it `inhibited_by`) satisfies "SOCS3 decreases", not "increases"."""
    record = _record(
        (
            "v13",
            "EpoR_JAK2{pY1007} phosphorylates STAT5 at Y694, activated by SOCS3",
            {"activated_by_0": "inhibited_by"},
        )
    )
    got = _check(
        [
            "c1: SOCS3 decreases STAT5{pY694}, directly",
            "c2: SOCS3 increases STAT5{pY694}, directly",
        ],
        record,
    )
    assert got["c1"].rows == ("v13",) and not got["c2"].ok
    assert "has SOCS3 change STAT5{pY694} the other way (as its rate records it)" in got["c2"].why


def test_an_unsigned_regulator_claims_no_direction_but_is_an_effect():
    """bachmann's CISH on v14 is recorded `modulated_by`: it satisfies no signed constraint, and
    the refusal says why, but it is still a claim that CISH changes the reaction."""
    record = _record(
        (
            "v14",
            "EpoR_JAK2{pY343} phosphorylates STAT5 at Y694, inhibited by CISH",
            {"inhibited_by_0": "modulated_by"},
        )
    )
    got = _check(
        ["c1: CISH decreases STAT5{pY694}, directly", "c2: CISH does not affect STAT5, directly"],
        record,
    )
    assert not got["c1"].ok and "recorded `modulated_by`" in got["c1"].why
    assert not got["c2"].ok and got["c2"].rows == ("v14",)


def test_an_amount_is_the_named_form_and_only_a_net_change_counts():
    """A binding moves EGFR between forms and leaves its total as it is (#134, decision 5): it
    increases EGFR{!EGF}, not EGFR."""
    got = _check(
        ["c1: EGF increases EGFR, directly", "c2: EGF increases EGFR{!EGF}, directly"],
        _record(("r1", "EGF binds EGFR")),
    )
    assert not got["c1"].ok and got["c2"].rows == ("r1",)


def test_a_constraints_regulator_must_be_on_the_row_as_recorded():
    lines = ["c1: E phosphorylates S at S10, inhibited by C, directly"]
    assert not _check(lines, _record(("r", "E phosphorylates S at S10")))["c1"].ok
    assert _check(lines, _record(("r", "E phosphorylates S at S10, inhibited by C")))["c1"].ok
    assert not _check(
        lines,
        _record(
            ("r", "E phosphorylates S at S10, inhibited by C", {"inhibited_by_0": "modulated_by"})
        ),
    )["c1"].ok


def test_agents_that_share_a_grounding_stay_two_agents():
    """INDRA's `Agent.matches` keys on grounding, so `statements._reaction` reads a conversion
    between a gene's mRNA and its protein, both grounded to one HGNC id, as making nothing; the
    constraints read agents by name."""
    record = _record(("t", "X_mRNA -> X, activated by Y"))
    for row in record["rows"]:
        for side in ("obj_from", "obj_to"):
            for agent in row["statement"][side]:
                agent["db_refs"] = {"HGNC": "1"}
    assert _check(["c1: Y increases X, directly"], record)["c1"].rows == ("t",)


def test_not_directly_and_does_not_affect():
    record = _record(*_CASCADE)
    got = _check(
        [
            "c1: not EGFR phosphorylates AKT1, directly",
            "c2: not EGF phosphorylates AKT1, directly",
            "c3: RPS6 does not affect EGF, indirectly",
            "c4: EGF does not affect RPS6, indirectly",
        ],
        record,
    )
    assert not got["c1"].ok and got["c1"].rows == ("r23",)
    assert got["c2"].ok and got["c3"].ok
    assert not got["c4"].ok and "r56" in got["c4"].rows
    assert got["c4"].why.startswith("tracing EGF moves RPS6 through rows")


# ------------------------------------------------------------------------------------- indirect
def test_tracing_carries_a_cascade_with_its_sign():
    """fujita's cascade: EGF's tag reaches RPS6{pS235} through four rows, and nothing lowers it, so
    the false "EGF decreases RPS6{pS235}" is refused saying what the trace reached."""
    got = _check(
        ["c1: EGF increases RPS6{pS235}, indirectly", "c2: EGF decreases RPS6{pS235}, indirectly"],
        _record(*_CASCADE),
    )
    assert got["c1"].ok and got["c1"].rows == ("r56",)
    assert not got["c2"].ok
    assert got["c2"].why.startswith(
        "tracing EGF through the network its rows expand to never lowers RPS6{pS235}; it reaches "
        "AKT1, EGFR, RPS6"
    )


def test_a_tag_alone_is_no_change():
    """EGF phosphorylating EGFR tags it and leaves total EGFR as it is: only the net change of a
    target's forms counts, as a readout reads it (read by its tag, fujita's false "EGF increases
    EGFR" held)."""
    got = _check(
        ["c1: EGF increases EGFR, indirectly", "c2: EGF increases EGFR{pY1068}, indirectly"],
        _record(*_CASCADE),
    )
    assert not got["c1"].ok and got["c2"].ok


def test_under_feedback_both_signs_hold():
    """Under negative feedback the network moves P each way, and which way the fitted model moves
    it is the fit's to say: both hold, and neither is reported against the other (#136)."""
    record = _record(
        ("a", "E increases P"), ("b", "P increases N"), ("c", "P is degraded, activated by N")
    )
    got = _check(["c1: E increases P, indirectly", "c2: E decreases P, indirectly"], record)
    assert got["c1"].ok and got["c2"].ok and got["c2"].rows == ("c",)


def test_a_tag_follows_the_molecule_not_the_form_a_rule_reads():
    """E phosphorylating A{pS1} at S2 tags A, and the rule A{pS1} catalyses takes the tag on,
    though E leaves the amount of A{pS1} as it is: held, where the claims chain (#134) refused it.
    Deliberately so: a false acceptance costs less than a false refusal, and the lowered side,
    A{pS1} unphosphorylated at S2, is traced on only where the phosphorylated form does not do what
    it did (#136)."""
    record = _record(
        ("r0", "K phosphorylates A at S1"),
        ("r1", "E phosphorylates A{pS1} at S2"),
        ("r2", "A{pS1} phosphorylates B at S3"),
    )
    got = _check(["c1: E increases B{pS3}, indirectly", "c2: E decreases B{pS3}, indirectly"], record)
    assert got["c1"].ok and not got["c2"].ok


def test_a_competitor_lowers_what_it_takes_goes_on_to_make():
    """Less of something is no tag, so a decrease by mass action is traced on from what the
    traced driver took away: H sequestering C lowers what free C phosphorylates, one step after X
    makes H too (isensee's Rp8 antagonist competing with cAMP for the holoenzyme)."""
    record = _record(
        ("h", "H binds C"), ("p", "C{~H} phosphorylates S at S1"), ("m", "X increases H")
    )
    got = _check(
        [
            "c1: H decreases S{pS1}, indirectly",
            "c2: X decreases S{pS1}, indirectly",
            "c3: X increases S{pS1}, indirectly",
        ],
        record,
    )
    assert got["c1"].ok and got["c1"].rows == ("p",)
    assert got["c2"].ok and got["c2"].rows == ("p",)
    assert not got["c3"].ok


def test_a_partner_taken_into_a_complex_that_keeps_its_part_lowers_nothing():
    """EGF binding free EGFR{pY1068} takes it away, but bound EGFR{pY1068} phosphorylates AKT1 as
    the free one did, so the loss is not traced on: traced through that phosphorylation, it read as
    EGF lowering AKT1{pS473} (#136)."""
    got = _check(["c1: EGF decreases AKT1{pS473}, indirectly"], _record(*_CASCADE))
    assert not got["c1"].ok


def test_a_junction_does_not_come_back_to_its_own_form():
    """R binding free A{pS1} takes it away. Traced on, less free A{pS1} is dephosphorylated less and
    then phosphorylated less: a cycle back to the form the binding only moved, which read as
    fujita's false "RPS6 decreases AKT1{pS473}". A junction pass reads no form its seeds are."""
    record = _record(
        ("k", "K phosphorylates A at S1"),
        ("d", "A{~R} is dephosphorylated at S1"),
        ("b", "A{pS1} binds R"),
    )
    got = _check(
        ["c1: R decreases A{pS1}, indirectly", "c2: R increases A{pS1, !R}, indirectly"], record
    )
    assert not got["c1"].ok and got["c2"].ok


def test_a_partner_a_bond_wildcard_hides_drives_a_release():
    """isensee releases PKA's catalytic subunit by a binding whose bound condition holds both ways
    (`stated`), so the release names cAMP only as "R bound to something": written out as the
    partner that site binds, cAMP hands its tag to the subunit it frees."""
    record = _record(
        ("holo", "R binds C"),
        ("occ", "R{!C} binds cAMP"),
        ("rel", "R{!cAMP} binds C", {}, "stated"),
    )
    got = _check(["c1: cAMP increases C{~R}, indirectly"], record)["c1"]
    assert got.ok and "rel" in got.rows


def test_activity_rises_with_a_declared_active_form():
    """`activates` reaches a declared active form, or lowers a declared inactive one; a regulator
    of a reaction a declared active form catalyses regulates its activity there, as `as_indra`
    projects it."""
    forms = ["a1: AKT1{pS473} is active", "a2: RAF1{!drug} is inactive", "a3: PDE is active"]
    record = _record(
        *_CASCADE, ("d", "drug binds RAF1"), ("pde", "PDE decreases cAMP, inhibited by IBMX")
    )
    got = _check(
        [
            *forms,
            "c1: EGF activates AKT1, indirectly",
            "c2: drug inhibits RAF1, directly",
            "c3: IBMX inhibits PDE, directly",
            "c4: EGF inhibits AKT1, indirectly",
        ],
        record,
    )
    assert got["c1"].ok and got["c2"].rows == ("d",) and got["c3"].rows == ("pde",)
    assert not got["c4"].ok
    # Traced, a regulator of a reaction the active form catalyses moves its activity too.
    traced = _check([*forms, "c5: IBMX inhibits PDE, indirectly"], record)["c5"]
    assert traced.ok and traced.rows == ("pde",)
    # Refused, it names what it is about: the declared forms, not an activity the rows never write.
    late = _check([*forms, "c5: EGF activates AKT1, directly"], record)["c5"]
    assert not late.ok and late.why.startswith("rows `r23` change the activity of AKT1")


# -------------------------------------------------------------------------------------- enforce
def test_enforce_refuses_naming_each_constraint_it_breaks():
    constraints = K.parse(
        [
            "c1: EGF increases RPS6{pS235}, indirectly",
            "c2: EGF phosphorylates RPS6 at S235, directly",
        ]
    )
    record = _record(*_CASCADE)
    with pytest.raises(OperationError) as raised:
        K.enforce(constraints, record, lambda: _model(record))
    assert raised.value.cause == "constraint_violated"
    message = str(raised.value)
    assert "breaks 1 of the session's constraints" in message and "`c2`" in message
    assert "`c1`" not in message and message.endswith("NOTHING was applied.")
    with pytest.raises(OperationError, match="not built from statements"):
        K.enforce(constraints, None, None)


def test_enforce_keeps_verdicts_and_holds_what_it_could_not_trace():
    """The verdicts go with the record. A trace that cannot run refuses nothing: the constraint
    holds, unchecked, and the result says so."""
    constraints = K.parse(["c1: EGF increases RPS6{pS235}, indirectly"])
    record = _record(*_CASCADE)
    kept, warning = K.enforce(constraints, record, lambda: _model(record))
    assert kept["checked"] == ["c1: EGF increases RPS6{pS235}, indirectly"]
    assert kept["verdicts"] == {"c1": {"ok": True, "rows": ["r56"]}} and warning is None

    def broken():
        raise RuntimeError("no BioNetGen")

    kept, warning = K.enforce(constraints, record, broken)
    assert kept["verdicts"]["c1"]["ok"] and "no BioNetGen" in kept["verdicts"]["c1"]["unchecked"]
    assert warning.startswith("`c1` (EGF increases RPS6{pS235}, indirectly) was not checked")


# --------------------------------------------------------------------------------------- server
_ENTITY_MAP = '{"A": {"name": "A"}, "B": {"name": "B"}}'


def _change(source: str, target: str, prefix: str) -> dict:
    """`make_changes`' arguments for one row, `source becomes target`, its binding given apart."""
    statement = {
        "type": "PopulationTransition",
        "source": {"name": source},
        "target": {"name": target},
    }
    return {
        "justification": f"{source} becomes {target}",
        "statements": [{"statement": statement, "prefix": prefix}],
        "bindings": {prefix: "mass_action"},
    }


def _constrained_server(tmp_path, lines: str):
    (tmp_path / "entities.json").write_text(_ENTITY_MAP)
    (tmp_path / "constraints.txt").write_text(lines)
    return create_server(
        str(YAML_PATH),
        provenance_dir=str(tmp_path / "prov"),
        resolver=StubReferenceResolver(),
        auto_fit=False,
        entity_map=str(tmp_path / "entities.json"),
        constraints=str(tmp_path / "constraints.txt"),
    )


def _call(mcp, name, args):
    return asyncio.run(mcp.call_tool(name, args)).structured_content


def test_make_changes_refuses_a_model_that_breaks_a_constraint(tmp_path):
    """The refusal names the constraint and applies nothing; a model that makes the claim commits,
    and its stored record keeps the verdicts it was checked under."""
    mcp = _constrained_server(tmp_path, "c1: A becomes B, directly\n")
    out = _call(mcp, "make_changes", _change("B", "A", "back"))
    assert out["applied"] is False and out["error_cause"] == "constraint_violated"
    assert "`c1` (A becomes B, directly)" in out["error"]
    out = _call(mcp, "make_changes", _change("A", "B", "conv"))
    assert out["applied"] is True and out["committed"] is True, out
    record = json.loads((tmp_path / "prov" / "versions" / "v0" / "statements.json").read_text())
    assert record["constraints"]["verdicts"] == {"c1": {"ok": True, "rows": ["conv"]}}


def test_make_changes_traces_an_indirect_constraint(tmp_path):
    """An indirect constraint is traced on the model the change's rows build, built again from the
    record it would commit: B made from A carries A's tag, and the verdict is kept and listed."""
    mcp = _constrained_server(tmp_path, "c1: A increases B, indirectly\n")
    out = _call(mcp, "make_changes", _change("B", "A", "back"))
    assert out["applied"] is False and out["error_cause"] == "constraint_violated"
    assert "never raises B" in out["error"]
    out = _call(mcp, "make_changes", _change("A", "B", "conv"))
    assert out["applied"] is True and out["committed"] is True, out
    (c1,) = _call(mcp, "list_constraints", {})["constraints"]
    assert (c1["ok"], c1["rows"]) == (True, ["conv"])


def test_the_agent_reads_the_constraints(tmp_path):
    """`list_constraints` lists them with what makes each hold on the current model, `make_changes`
    says what they do, and the server offers the statements route alone: a model built from
    Antimony states no claims to check."""
    mcp = _constrained_server(
        tmp_path, "a1: B{pS1} is active\nc1: A becomes B, directly  # [x] y\n"
    )
    listed = _call(mcp, "list_constraints", {})
    (c1,) = listed["constraints"]
    assert {k: c1[k] for k in ("id", "text", "source", "ok")} == {
        "id": "c1",
        "text": "A becomes B, directly",
        "source": "[x] y",
        "ok": False,
    }
    assert c1["why"] == "no row changes B"
    assert listed["active_forms"] == ["a1: B{pS1} is active"]
    assert _call(mcp, "describe_problem", {})["n_constraints"] == 1
    tools = {t.name: t for t in asyncio.run(mcp.list_tools())}
    assert (
        "This session holds CONSTRAINTS (`list_constraints`)" in tools["make_changes"].description
    )
    assert "antimony" not in tools["make_changes"].parameters["properties"]


def test_constraints_are_refused_without_the_statements_route_or_when_unreadable(tmp_path):
    (tmp_path / "constraints.txt").write_text("c1: A becomes B\n")
    with pytest.raises(ValidationError, match="constraints need an entity_map"):
        create_server(
            str(YAML_PATH),
            provenance_dir=str(tmp_path / "p1"),
            constraints=str(tmp_path / "constraints.txt"),
        )
    (tmp_path / "entities.json").write_text(_ENTITY_MAP)
    with pytest.raises(K.ConstraintError, match="says neither"):
        create_server(
            str(YAML_PATH),
            provenance_dir=str(tmp_path / "p2"),
            entity_map=str(tmp_path / "entities.json"),
            constraints=str(tmp_path / "constraints.txt"),
        )


# -------------------------------------------------------------------------------------- updates
def test_an_update_is_told_looser_or_tighter():
    """Recorded with each update for analysis (#134, decision 6): a direct claim is the stronger
    form of the same indirect one, and no chain either way the stronger form of no row's claim."""
    old = K.parse(["c1: A increases B, directly", "c2: X does not affect Y, indirectly"])
    assert K.change(old, old) == "same"
    assert K.change(old, K.parse(["c1: A increases B, indirectly"])) == "looser"
    assert (
        K.change(
            old,
            K.parse(
                [
                    "c1: A increases B, directly",
                    "c2: X does not affect Y, indirectly",
                    "c3: B increases C, directly",
                ]
            ),
        )
        == "tighter"
    )
    assert K.change(old, K.parse(["c1: B increases A, directly"])) == "both"
    with_form = K.parse(
        [
            "a1: B{pS1} is active",
            "c1: A increases B, directly",
            "c2: X does not affect Y, indirectly",
        ]
    )
    assert K.change(old, with_form) == "both"


def test_a_version_the_constraints_break_ranks_no_more(session):
    """The best version is rebuilt from the log without the versions the constraints in force break,
    so the new-best bar and the plateau stop rank only models that make the session's claims."""
    for version, score in ((0, 100.0), (1, 80.0), (2, 90.0)):
        session.provenance.append_event(
            {
                "type": "fit",
                "version": version,
                "neg_log_likelihood": score / 2,
                "criterion": session.criterion.name,
                "criterion_value": score,
            }
        )
    session.reseed_best(set())
    assert session.best.version == 1
    session.reseed_best({1})
    assert session.best.version == 2


def _updatable_server(tmp_path, lines: str):
    (tmp_path / "entities.json").write_text(_ENTITY_MAP)
    (tmp_path / "constraints.txt").write_text(lines)
    return create_server(
        str(YAML_PATH),
        provenance_dir=str(tmp_path / "prov"),
        resolver=StubReferenceResolver(),
        auto_fit=False,
        entity_map=str(tmp_path / "entities.json"),
        constraints=str(tmp_path / "constraints.txt"),
        constraint_updates="any",
    )


def test_set_constraints_exists_only_where_updates_are_on(tmp_path):
    """Updating can be turned off (#134, requirement 4): off, the default, the tool is absent."""
    for sub in ("a", "b"):
        (tmp_path / sub).mkdir()
    names = {t.name for t in asyncio.run(_constrained_server(tmp_path / "a", "").list_tools())}
    assert "set_constraints" not in names
    names = {t.name for t in asyncio.run(_updatable_server(tmp_path / "b", "").list_tools())}
    assert "set_constraints" in names
    with pytest.raises(ValidationError, match="needs constraints"):
        create_server(str(YAML_PATH), provenance_dir=str(tmp_path / "c"), constraint_updates="any")


def test_an_update_flags_the_versions_it_breaks_and_holds_the_next_change_to_it(tmp_path):
    """A version committed before the update keeps its own verdicts, but under the new set it is
    flagged, cannot be restored, and the next change must make the new claims (#134, decision 7)."""
    mcp = _updatable_server(tmp_path, "c1: A becomes B, directly\n")
    out = _call(mcp, "make_changes", _change("A", "B", "conv"))
    assert out["committed"] is True, out
    refused = _call(mcp, "set_constraints", {"constraints": ["c1: B becomes A"], "rationale": "x"})
    assert refused["applied"] is False and refused["error_cause"] == "unreadable_constraints"
    out = _call(
        mcp,
        "set_constraints",
        {"constraints": ["c1: B becomes A, directly"], "rationale": "the curator's correction"},
    )
    assert out["change"] == "both" and out["head_violates"] == ["c1"]
    assert out["violating_versions"] == {"0": ["c1"]}
    (v0,) = _call(mcp, "list_versions", {})["result"]
    assert v0["violates"] == ["c1"]
    restored = _call(mcp, "restore_version", {"version": 0})
    assert restored["ok"] is False and "breaks the session's constraints `c1`" in restored["error"]
    listed = _call(mcp, "list_constraints", {})
    assert listed["updates"] == "any"
    assert [c["text"] for c in listed["constraints"]] == ["B becomes A, directly"]
    out = _call(mcp, "make_changes", _change("A", "B", "conv2"))
    assert out["applied"] is False and out["error_cause"] == "constraint_violated"
    record = json.loads((tmp_path / "prov" / "versions" / "v0" / "statements.json").read_text())
    assert record["constraints"]["checked"] == ["c1: A becomes B, directly"]


def test_a_version_is_checked_against_a_constraint_once(tmp_path, monkeypatch):
    """A version keeps the verdicts its commit found, and one found later is remembered: tracing
    runs BioNetGen, and a version's model never changes."""
    from petab_mcp import operations

    mcp = _updatable_server(tmp_path, "c1: A increases B, indirectly\n")
    out = _call(mcp, "make_changes", _change("A", "B", "conv"))
    assert out["committed"] is True, out
    built = []
    model = operations.statements_model
    monkeypatch.setattr(
        operations, "statements_model", lambda *a: built.append(a) or model(*a)
    )
    assert _call(mcp, "list_versions", {})["result"][0].get("violates") is None
    assert built == []  # read off v0's record
    update = {"constraints": ["c1: A increases B, indirectly", "c2: B does not affect A, indirectly"]}
    out = _call(mcp, "set_constraints", {**update, "rationale": "x"})
    assert out["violating_versions"] == {} and len(built) == 1  # c2 traced on v0, once
    _call(mcp, "list_versions", {})
    _call(mcp, "list_constraints", {})
    assert len(built) == 1
