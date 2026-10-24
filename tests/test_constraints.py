"""Causal constraints (:mod:`petab_mcp.constraints`, PEtab-MCP#134): claims every model must make,
checked on a statements record's rows, their regulators as their rates record them.

Records are written by hand here, each row a statement in the notation and the kinds its rate
records (``derived``), so each test says exactly which claim it turns on; the server tests at the
end build one through ``make_changes``.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from petab_mcp import constraints as K
from petab_mcp.errors import ValidationError
from petab_mcp.operations import OperationError
from petab_mcp.references import StubReferenceResolver
from petab_mcp.server import create_server
from petab_mcp.statements import with_defaults

YAML_PATH = Path(__file__).parent / "data" / "toy_v2" / "problem.yaml"


def _record(*rows) -> dict:
    """A statements record of ``rows``, each ``(prefix, statement, derived?)``."""
    return {
        "rows": [
            {
                "prefix": prefix,
                "statement": with_defaults(K.statement_json(text)),
                "emission": {"derived": derived[0] if derived else {}},
            }
            for prefix, text, *derived in rows
        ]
    }


def _check(lines: list[str], record: dict) -> dict[str, K.Verdict]:
    return K.check(K.parse(lines), record)


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


def test_a_file_reads_as_lines_or_json(tmp_path):
    """A line's trailing comment is its source; a line that starts with `#` is no constraint.
    JSON takes lines and objects alike."""
    lines = tmp_path / "constraints.txt"
    lines.write_text("# the bench's claims\nc1: A becomes B, directly  # [curated] Doe 2000\n")
    (c1,) = K.read(lines).items
    assert (c1.id, c1.kind, c1.source) == ("c1", "direct", "[curated] Doe 2000")
    objects = tmp_path / "constraints.json"
    objects.write_text(
        json.dumps(
            [
                "a1: B{pS1} is active",
                {
                    "id": "c2",
                    "statement": {
                        "type": "IncreaseAmount",
                        "subj": {"name": "A"},
                        "obj": {"name": "B"},
                    },
                    "kind": "indirect",
                },
                {"id": "c3", "does_not_affect": ["B", "A"], "kind": "direct"},
            ]
        )
    )
    got = K.read(objects)
    assert [c.text for c in got.items] == [
        "A increases B, indirectly",
        "B does not affect A, directly",
    ]


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
    assert not got["c4"].ok and "(r56) RPS6" in got["c4"].chain


# ------------------------------------------------------------------------------------- indirect
def test_a_chain_carries_a_cascade_with_its_sign():
    """fujita's cascade: EGF raises RPS6{pS235} through four rows, and no chain lowers it, so the
    false "EGF decreases RPS6{pS235}" is refused saying the claims run the other way."""
    got = _check(
        ["c1: EGF increases RPS6{pS235}, indirectly", "c2: EGF decreases RPS6{pS235}, indirectly"],
        _record(*_CASCADE),
    )
    assert got["c1"].ok and got["c1"].opposite is None
    assert got["c1"].chain == (
        "EGF → (r1) EGFR{-pY1068, !EGF} → (r9) EGFR{pY1068} → (r23) AKT1{pS473} → (r56) RPS6{pS235}"
    )
    assert not got["c2"].ok and "only the other way" in got["c2"].why


def test_feedback_reports_the_chain_of_the_other_sign():
    """Under negative feedback a chain of each sign exists, and which way the model moves its
    object is the fit's to say: held, and the other chain reported (#134, decision 2)."""
    record = _record(
        ("a", "E increases P"), ("b", "P increases N"), ("c", "P is degraded, activated by N")
    )
    got = _check(["c1: E increases P, indirectly"], record)["c1"]
    assert got.ok and got.chain == "E → (a) P"
    assert got.opposite == "E → (a) P → (b) N ⊣ (c) P"


def test_moving_between_forms_a_reader_lumps_carries_nothing():
    """A row that only moves A between forms that each hold pS1 leaves the amount the next row
    reads as it is, so no chain runs through it (bachmann's spurious SOCS3 ⊣ Y401 → SHP1 chain ran
    through such a move in a species graph)."""
    record = _record(
        ("r1", "E phosphorylates A{pS1} at S2"), ("r2", "A{pS1} phosphorylates B at S3")
    )
    got = _check(["c1: E increases B{pS3}, indirectly"], record)["c1"]
    assert not got.ok and "no chain of claims connects them" in got.why


def test_sequestration_is_a_chains_first_step_only():
    """A binding partner lowers its partner's free form as a chain's first step: past it a chain
    would run back up a binding."""
    record = _record(
        ("h", "H binds C"), ("p", "C{~H} phosphorylates S at S1"), ("m", "X increases H")
    )
    got = _check(
        ["c1: H decreases S{pS1}, indirectly", "c2: X decreases S{pS1}, indirectly"], record
    )
    assert got["c1"].ok and got["c1"].chain == "H ⊣ (h) C{~H} → (p) S{pS1}"
    assert not got["c2"].ok


def test_a_binding_claims_coming_apart():
    """isensee releases PKA's catalytic subunit by a binding written with no forward rate: the
    complex coming apart raises each member's free form (#134, decision 11)."""
    record = _record(("occ", "R{!C} binds cAMP"), ("rel", "R{!cAMP} binds C"))
    got = _check(["c1: cAMP increases C{~R}, indirectly"], record)["c1"]
    assert got.ok and got.chain == "cAMP → (occ) R{!cAMP, !C} → (rel) C{~R}"


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
    with pytest.raises(OperationError) as raised:
        K.enforce(constraints, _record(*_CASCADE), None)
    assert raised.value.cause == "constraint_violated"
    message = str(raised.value)
    assert "breaks 1 of the session's constraints" in message and "`c2`" in message
    assert "`c1`" not in message and message.endswith("NOTHING was applied.")
    with pytest.raises(OperationError, match="not built from statements"):
        K.enforce(constraints, None, None)


def test_enforce_keeps_verdicts_and_warns_of_a_new_two_way_chain_once():
    """The verdicts go with the record, so the next change warns only of a chain of the other
    sign that is new since the model it starts from."""
    constraints = K.parse(["c1: E increases P, indirectly"])
    record = _record(
        ("a", "E increases P"), ("b", "P increases N"), ("c", "P is degraded, activated by N")
    )
    kept, warning = K.enforce(constraints, record, None)
    assert kept["checked"] == ["c1: E increases P, indirectly"]
    assert kept["verdicts"]["c1"]["opposite"] and "the fit's to say" in warning
    assert K.enforce(constraints, record, {"constraints": kept})[1] is None


# --------------------------------------------------------------------------------------- server
_ENTITY_MAP = '{"A": {"name": "A"}, "B": {"name": "B"}}'


def _row(source: str, target: str, prefix: str) -> dict:
    return {
        "statement": {
            "type": "PopulationTransition",
            "source": {"name": source},
            "target": {"name": target},
        },
        "policy": "mass_action",
        "prefix": prefix,
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
    out = _call(
        mcp,
        "make_changes",
        {"justification": "B becomes A", "statements": [_row("B", "A", "back")]},
    )
    assert out["applied"] is False and out["error_cause"] == "constraint_violated"
    assert "`c1` (A becomes B, directly)" in out["error"]
    out = _call(
        mcp,
        "make_changes",
        {"justification": "A becomes B", "statements": [_row("A", "B", "conv")]},
    )
    assert out["applied"] is True and out["committed"] is True, out
    record = json.loads((tmp_path / "prov" / "versions" / "v0" / "statements.json").read_text())
    assert record["constraints"]["verdicts"] == {"c1": {"ok": True, "rows": ["conv"]}}


def test_the_agent_reads_the_constraints(tmp_path):
    """`describe_problem` lists them, `make_changes` says what they do, and the server offers the
    statements route alone: a model built from Antimony states no claims to check."""
    mcp = _constrained_server(
        tmp_path, "a1: B{pS1} is active\nc1: A becomes B, directly  # [x] y\n"
    )
    described = _call(mcp, "describe_problem", {})
    assert described["constraints"] == [
        {"id": "c1", "text": "A becomes B, directly", "kind": "direct", "source": "[x] y"}
    ]
    assert described["active_forms"] == [{"id": "a1", "text": "B{pS1} is active"}]
    tools = {t.name: t for t in asyncio.run(mcp.list_tools())}
    assert "This session holds CONSTRAINTS" in tools["make_changes"].description
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
    out = _call(
        mcp,
        "make_changes",
        {"justification": "A becomes B", "statements": [_row("A", "B", "conv")]},
    )
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
    assert _call(mcp, "describe_problem", {})["constraints"][0]["text"] == "B becomes A, directly"
    out = _call(
        mcp,
        "make_changes",
        {"justification": "A becomes B again", "statements": [_row("A", "B", "conv2")]},
    )
    assert out["applied"] is False and out["error_cause"] == "constraint_violated"
    record = json.loads((tmp_path / "prov" / "versions" / "v0" / "statements.json").read_text())
    assert record["constraints"]["checked"] == ["c1: A becomes B, directly"]
