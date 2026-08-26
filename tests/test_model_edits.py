"""The STRUCTURED edit surface: name the change, not the characters.

Why it replaces substring patching, measured over 30 runs / 1,063 patches: 20% of all patches
targeted the derived ``const`` line (71% of all failures), and 81% of the remaining failures
quoted text that existed in neither the render nor the agent's own submission. Addressing a
reaction by ID removes both classes — an ID exists or it does not, and the server can say which.

BATCHING is not a nicety here: one call is one fit and one version, so an unbatched surface
would pay a multistart per operation and score intermediate states nobody meant to test.
"""

from __future__ import annotations

import asyncio
import shutil

import pytest

from conftest import DATA_DIR

from petab_mcp import model_edits as me
from petab_mcp.operations import OperationError
from petab_mcp.references import StubReferenceResolver

SRC = ("model demo\n"
       "  compartment cell = 1;\n"
       "  species A in cell, B in cell;\n"
       "  A = 1; B = 0;\n"
       "  k1 = 0.5; k2 = 0.1;\n"
       "  fwd: A => B; k1*A;\n"
       "  rev: B => A; k2*B;\n"
       "end\n")


# ------------------------------------------------------------------ the operations themselves
def test_a_batch_applies_in_order_and_reports_what_it_did():
    """The summary IS the diff, pre-classified. Every earlier analysis had to reconstruct this
    from re-rendered text with id/regex heuristics, and four of those reported artifacts."""
    out, summary = me.apply_model_edits(SRC, [
        {"op": "set_rate_law", "id": "fwd", "antimony": "k1*A/(Km + A)"},
        {"op": "add_reaction", "antimony": "deg: B =>; kdeg*B;"},
        {"op": "set_initial", "id": "k1", "antimony": 2.0},
    ])
    assert summary == ["set rate law of fwd", "added reaction deg", "set k1 = 2.0"]
    assert "fwd: A => B; k1*A/(Km + A);" in out
    assert "deg: B =>; kdeg*B;" in out
    assert out.strip().endswith("end")            # nothing emitted past the model body


def test_an_unknown_reaction_names_the_valid_ids():
    """The fabrication class the substring surface could not answer: an ID either exists or it
    does not, so the rejection can just say what does."""
    with pytest.raises(OperationError, match=r"no reaction 'nope'.*Existing: fwd, rev"):
        me.apply_model_edits(SRC, [{"op": "set_rate_law", "id": "nope", "antimony": "x"}])


def test_the_batch_is_all_or_nothing_and_names_the_failing_index():
    with pytest.raises(OperationError, match=r"model_edits\[1\].*NOTHING was applied"):
        me.apply_model_edits(SRC, [
            {"op": "set_rate_law", "id": "fwd", "antimony": "k1*A*A"},
            {"op": "remove_reaction", "id": "ghost"},
        ])


def test_rename_covers_the_parameter_merge_case():
    """A measured winning move: merging two rate constants by renaming one across every law it
    appears in (priced by the agent at dBIC -2.668). A bucket-only API would force that back to
    wholesale re-authoring."""
    out, summary = me.apply_model_edits(SRC, [{"op": "rename", "old": "k2", "new": "k1"}])
    assert summary == ["renamed k2 -> k1 (2 occurrences)"]
    assert "rev: B => A; k1*B;" in out and "k2" not in out


def test_set_initial_finds_a_symbol_sharing_a_line():
    """Agent-authored Antimony routinely puts several statements on one line (`A = 1; B = 0;`).
    A line-anchored match misses those and then appends a DUPLICATE assignment — which is what
    the first version of this code did."""
    out, _ = me.apply_model_edits(SRC, [{"op": "set_initial", "id": "A", "antimony": 5}])
    assert "A = 5; B = 0;" in out
    assert out.count("A = ") == 1


def test_adding_a_duplicate_reaction_id_is_refused():
    with pytest.raises(OperationError, match="already exists"):
        me.apply_model_edits(SRC, [{"op": "add_reaction", "antimony": "fwd: A => B; k1*A;"}])


def test_an_unknown_op_lists_the_real_ones():
    with pytest.raises(OperationError, match="unknown op 'frobnicate'"):
        me.apply_model_edits(SRC, [{"op": "frobnicate", "id": "fwd"}])


# ------------------------------------------------------------------ end to end through the tool
def _call(mcp, name, args):
    return asyncio.run(mcp.call_tool(name, args)).structured_content


@pytest.fixture
def tool_env(tmp_path, monkeypatch):
    monkeypatch.setenv("PETAB_DIFF_EDITS", "1")
    from petab_mcp.server import create_server

    work = tmp_path / "toy"
    work.mkdir()
    for f in DATA_DIR.iterdir():
        shutil.copy(f, work / f.name)
    mcp = create_server(str(work / "problem.yaml"), provenance_dir=str(tmp_path / "prov"),
                        build_root=str(tmp_path / "amici"),
                        resolver=StubReferenceResolver(), auto_fit=True)
    return mcp, tmp_path / "prov"


def test_one_call_one_fit_one_version_however_many_operations(tool_env):
    """The batching contract, and the reason it is not optional: an unbatched surface would pay
    a full multistart per operation."""
    mcp, prov = tool_env
    res = _call(mcp, "make_changes", {
        "justification": "a mechanism plus the rate law it forces",
        "model_edits": [
            {"op": "set_rate_law", "id": "r1", "antimony": "k1*A/(1 + A)"},
            {"op": "add_reaction", "antimony": "sink: B =>; k_sink*B;"},
        ]})
    assert res["applied"] is True and res["committed"] is True
    assert res["model_edits_applied"] == ["set rate law of r1", "added reaction sink"]
    assert len(list((prov / "versions").glob("v*"))) == 1        # ONE version for two operations


def test_operations_edit_the_source_not_the_render(tool_env):
    """No anchor, so nothing to keep in sync with the server's re-render — and the stored source
    is the caller's text plus its operations, which is what makes lineage checkable."""
    mcp, prov = tool_env
    res = _call(mcp, "make_changes", {
        "justification": "structured change",
        "model_edits": [{"op": "add_reaction", "antimony": "sink: B =>; k_sink*B;"}]})
    src = (prov / "versions" / f"v{res['version']}" / "model_source.ant").read_text()
    assert "sink: B =>; k_sink*B;" in src
    # the derived declaration the substring surface kept tripping over is simply not in play
    assert "const" not in src or "sink" in src.split("const")[0]


def test_a_bad_id_costs_one_rejection_and_no_fit(tool_env):
    mcp, prov = tool_env
    before = len(list((prov / "versions").glob("v*")))
    res = _call(mcp, "make_changes", {
        "justification": "wrong id",
        "model_edits": [{"op": "set_rate_law", "id": "not_a_reaction", "antimony": "k1*A"}]})
    assert res["applied"] is False
    assert "Existing:" in (res.get("error") or "")           # says what the valid ids are
    assert len(list((prov / "versions").glob("v*"))) == before


def test_only_one_model_route_per_call(tool_env):
    mcp, _ = tool_env
    res = _call(mcp, "make_changes", {
        "justification": "two routes",
        "antimony": "model m\nend",
        "model_edits": [{"op": "remove_reaction", "id": "r1"}]})
    assert res["applied"] is False
    assert "ONE model route" in (res.get("error") or "")


def test_a_fork_applies_operations_to_that_versions_source(tool_env):
    """`base_version` + `model_edits`: fork and change in one call, with the operations applied
    to the forked version's own source rather than to HEAD."""
    mcp, prov = tool_env
    v0 = _call(mcp, "make_changes", {
        "justification": "v0", "model_edits": [
            {"op": "add_reaction", "antimony": "sink: B =>; k_sink*B;"}]})
    v1 = _call(mcp, "make_changes", {
        "justification": "v1", "model_edits": [
            {"op": "add_reaction", "antimony": "src2: => A; k_src;"}]})
    v2 = _call(mcp, "make_changes", {
        "justification": "fork from v0", "base_version": v0["version"],
        "model_edits": [{"op": "set_rate_law", "id": "sink", "antimony": "k_sink*B*B"}]})
    assert v2["committed"] is True
    src2 = (prov / "versions" / f"v{v2['version']}" / "model_source.ant").read_text()
    assert "k_sink*B*B" in src2
    assert "src2" not in src2, "the fork picked up v1's change — it did not branch from v0"
    import json
    meta = json.loads((prov / "versions" / f"v{v2['version']}" / "meta.json").read_text())
    assert meta["parent"] == v0["version"]


# ------------------------------------------------------- regressions the review found
def test_set_initial_does_not_eat_a_reaction_that_comes_first():
    """THE corrupting bug: `A\\s*=` matches the `A =` inside `A => B`, and libantimony renders
    reactions BEFORE species initialisations — so on the standard text this rewrote the reaction
    into an assignment and destroyed it. The original test missed it only because its fixture
    happened to put the assignments above the reactions."""
    src = ("model demo\n  // Reactions:\n  fwd: A => B; k1*A;\n\n"
           "  // Species initializations:\n  A = 1;\n  B = 0;\nend\n")
    out, _ = me.apply_model_edits(src, [{"op": "set_initial", "id": "A", "antimony": 5}])
    assert "fwd: A => B; k1*A;" in out, "the reaction was destroyed"
    assert "A = 5;" in out


def test_a_reaction_with_a_trailing_comment_is_still_addressable():
    """Agent-authored Antimony carries comments. A strict end-anchor made those reactions
    invisible to every id-addressed op — an unrecoverable loop, since the rejection would say
    the id does not exist while the caller can see it."""
    src = "model demo\n  fwd: A => B; k1*A;  // forward step\nend\n"
    assert me.reaction_ids(src) == ["fwd"]
    out, _ = me.apply_model_edits(src, [{"op": "set_rate_law", "id": "fwd", "antimony": "k1*A*A"}])
    assert "k1*A*A" in out and "// forward step" in out


def test_rename_onto_an_existing_SPECIES_is_refused():
    """The distinction the guard has to make: merging two rate constants is a legitimate move
    (measured, dBIC -2.668), while merging two species turns `A => B` into `B => B` and reports
    success. Species are exactly the symbols appearing in a stoichiometry."""
    src = "model demo\n  fwd: A => B; k1*A;\nend\n"
    with pytest.raises(OperationError, match="would merge two pools"):
        me.apply_model_edits(src, [{"op": "rename", "old": "A", "new": "B"}])


def test_rename_refuses_a_new_name_that_is_not_an_identifier():
    """`new` is substituted into every line, so anything but a bare identifier can split a
    statement and unaddress the reaction it lands in. (It was also passed to `re.sub` as a
    replacement TEMPLATE, so a backslash raised re.error — a 500 rather than a rejection.)"""
    src = "model demo\n  fwd: A => B; k_old*A;\nend\n"
    for bad in (r"k\1new", "k new", "k;x", "9k"):
        with pytest.raises(OperationError, match="plain identifier"):
            me.apply_model_edits(src, [{"op": "rename", "old": "k_old", "new": bad}])


# ------------------------------------------------- contracts inherited from the deleted
# ------------------------------------------------- substring-surface test module
def test_arm_off_keeps_the_baseline_make_changes_schema(tmp_path, monkeypatch):
    """Every recorded baseline run's comparability rests on this: with the arm OFF the schema is
    byte-compatible with the pre-arm contract, and ON adds exactly the structured route."""
    from petab_mcp.server import create_server

    def _props(flag):
        monkeypatch.setenv("PETAB_DIFF_EDITS", flag)
        work = tmp_path / f"toy_{flag}"
        work.mkdir()
        for f in DATA_DIR.iterdir():
            shutil.copy(f, work / f.name)
        mcp = create_server(str(work / "problem.yaml"),
                            provenance_dir=str(tmp_path / f"prov_{flag}"),
                            build_root=str(tmp_path / f"amici_{flag}"),
                            resolver=StubReferenceResolver(), auto_fit=False)
        tool = asyncio.run(mcp.get_tool("make_changes"))
        return set(tool.parameters.get("properties", {}))

    base = {"justification", "antimony", "parameters", "references", "n_starts"}
    assert _props("0") == base
    assert _props("1") == base | {"model_edits", "base_version"}


def test_base_version_alone_is_rejected_naming_no_withheld_tool(tool_env):
    """The refusal must not point at `restore_version`, which the lead does not hold — naming an
    uncallable tool is the defection-trigger class the draft-era A/B measured."""
    mcp, _ = tool_env
    res = _call(mcp, "make_changes", {"justification": "base_version alone", "base_version": 0})
    assert res["applied"] is False
    err = res.get("error") or ""
    assert "base_version" in err and "`model_edits`" in err
    for withheld in ("restore_version", "get_draft", "edit_draft"):
        assert withheld not in err


def test_an_unknown_base_version_is_a_clean_rejection(tool_env):
    mcp, _ = tool_env
    res = _call(mcp, "make_changes", {
        "justification": "fork a version that does not exist", "base_version": 99,
        "model_edits": [{"op": "set_rate_law", "id": "r1", "antimony": "k1*A"}]})
    assert res["applied"] is False
    assert "No such version" in (res.get("error") or "")


def test_a_bare_call_names_only_arguments_the_caller_can_send(tmp_path, monkeypatch):
    """Arm-aware advice: naming an argument the schema does not carry instructs an impossible
    call, which is how a lead ends up defecting to a route it was told not to use."""
    from petab_mcp.server import create_server

    def _bare(flag):
        monkeypatch.setenv("PETAB_DIFF_EDITS", flag)
        work = tmp_path / f"bare_{flag}"
        work.mkdir()
        for f in DATA_DIR.iterdir():
            shutil.copy(f, work / f.name)
        mcp = create_server(str(work / "problem.yaml"),
                            provenance_dir=str(tmp_path / f"bprov_{flag}"),
                            build_root=str(tmp_path / f"bamici_{flag}"),
                            resolver=StubReferenceResolver(), auto_fit=False)
        res = _call(mcp, "make_changes", {"justification": "bare call"})
        assert res["applied"] is False
        return res.get("error") or ""

    on = _bare("1")
    assert "`model_edits`" in on and "`antimony`" in on
    off = _bare("0")
    assert "`model_edits`" not in off and "`antimony`" in off


def test_every_rejection_reports_a_DISTINCT_machine_readable_cause(tool_env):
    """`edit_rejects.by_cause` is the arm's central metric. It used to be derived by
    substring-matching English prose in a hand-ordered tuple, which silently put three separate
    buckets on unreachable branches in one day (a needle shared with an earlier entry steals
    every match). The server now labels its own rejections; this pins that they are distinct."""
    mcp, _ = tool_env
    whole = _call(mcp, "get_model", {})["result"]
    cases = {
        "unknown_reaction_id": {"model_edits": [{"op": "set_rate_law", "id": "nope",
                                                 "antimony": "k1*A"}]},
        "duplicate_reaction_id": {"model_edits": [{"op": "add_reaction", "antimony": "r1: A => B; k1*A;"}]},
        # (an unknown `op` never reaches the server: the pydantic Literal rejects it
        #  client-side, which is a stricter guarantee than a cause label)
        "rename_target_missing": {"model_edits": [{"op": "rename", "old": "zzz", "new": "q"}]},
        # A redundant `id` that MATCHES the statement is accepted (it states the same thing
        # twice); only a DIFFERING one is ambiguous about which name later ops address.
        "id_mismatch": {"model_edits": [{"op": "add_reaction", "id": "not_r9",
                                         "antimony": "r9: A => B; k1*A;"}]},
        "route_conflict": {"antimony": whole,
                           "model_edits": [{"op": "remove_reaction", "id": "r1"}]},
    }
    got = {}
    for expected, args in cases.items():
        res = _call(mcp, "make_changes", {"justification": f"cause {expected}", **args})
        assert res["applied"] is False, expected
        got[expected] = res.get("error_cause")
    assert got == {k: k for k in cases}, got


def test_a_rejected_batch_does_not_write_the_operations_into_the_error_log(tool_env, tmp_path):
    """`_Ctx.short` caps containers, and `archive_run` copies errors.jsonl into every run
    archive — so a rejected batch must log a COUNT, not the operations."""
    mcp, prov = tool_env
    marker = "UNIQUE_MARKER_" + "x" * 400
    res = _call(mcp, "make_changes", {
        "justification": "rejected batch",
        "model_edits": [{"op": "set_rate_law", "id": "nope", "antimony": marker}]})
    assert res["applied"] is False
    log = prov / "errors.jsonl"
    assert log.is_file(), "the rejection was not logged at all — this guard would be vacuous"
    assert marker not in log.read_text()


def test_the_matcher_handles_the_rate_laws_biology_actually_uses():
    """A regression with teeth: excluding `/` from the rate-law group (to make room for `/* */`
    comments) made every SATURATING law unmatchable — Michaelis-Menten, Hill, any division — so
    the reaction vanished from `reaction_ids` the moment the agent wrote the most common kinetics
    in the field. It surfaced as a fork failing with "no reaction 'r1'. Existing: r2" against a
    source that plainly contained r1."""
    shapes = {
        "division":      "  r1: A => B; (k1/2 + k1/2)*A;",
        "michaelis":     "  mm: S => P; Vmax*S/(Km + S);",
        "hill":          "  h: A => B; k*A^n/(K^n + A^n);",
        "no semicolon":  "  r2: B -> A; kr*B",
        "line comment":  "  r3: A => C; k3*A;  // note",
        "block comment": "  r4: C => A; k4*C; /* note */",
        "both":          "  r5: A => D; k*A/(Km+A); // saturating",
    }
    for name, line in shapes.items():
        ids = me.reaction_ids(f"model d\n{line}\nend\n")
        assert len(ids) == 1, f"{name}: {line!r} was not recognised as a reaction ({ids})"
    # and the law survives an edit intact
    src = "model d\n  mm: S => P; Vmax*S/(Km + S);\nend\n"
    out, _ = me.apply_model_edits(src, [{"op": "set_rate_law", "id": "mm",
                                         "antimony": "Vmax*S/(Km + S + S^2/Ki)"}])
    assert "mm: S => P; Vmax*S/(Km + S + S^2/Ki);" in out


def test_several_statements_on_one_line_stay_addressable():
    """Antimony allows several statements per line and agents write them that way. A line-anchored
    matcher made `fwd: ...; rev: ...;` match NOTHING, so both reactions were unaddressable and the
    error listed no valid ids to recover from — an unrecoverable loop reached by formatting."""
    src = "model m\n  fwd: A => B; k1*A; rev: B => A; k2*B;\nend\n"
    assert me.reaction_ids(src) == ["fwd", "rev"]
    # editing one leaves the other alone, on the same line
    out, _ = me.apply_model_edits(src, [{"op": "set_rate_law", "id": "rev",
                                         "antimony": "k2*B/(Km+B)"}])
    assert "fwd: A => B; k1*A;" in out and "rev: B => A; k2*B/(Km+B);" in out
    # and removing one keeps the other
    out2, _ = me.apply_model_edits(src, [{"op": "remove_reaction", "id": "fwd"}])
    assert me.reaction_ids(out2) == ["rev"]


def test_a_comment_describing_a_reaction_is_not_a_reaction():
    """`// A -> B; k*A;` is a remark. Matching statement syntax inside comments made prose about
    the model read as model structure — for the unnamed-reaction refusal, that means refusing to
    author a model whose only offence is a comment."""
    assert me.unnamed_reactions("model m\n  // A -> B; k*A;\nend") == []
    assert me.unnamed_reactions("model m\n  A -> B; k*A;\nend") == ["A -> B"]
    assert me.unnamed_reactions("model m\n  r1: A -> B; k*A;\nend") == []


def test_the_assigned_symbol_remedy_sees_symbols_sharing_a_line():
    """`A = 1; B = 0;` is one line assigning two symbols. Reporting "Assigned symbols: (none)"
    for it sends the caller back to guessing — the retry loop the list exists to end."""
    assert me.assigned_symbols("model m\n  A = 1; B = 0;\nend") == ["A", "B"]
    with pytest.raises(OperationError) as ei:
        me.apply_model_edits("model m\n  A = 1; B = 0;\nend\n",
                             [{"op": "set_initial", "id": "Aa", "antimony": "5"}])
    assert "A, B" in str(ei.value) and "Did you mean 'A'" in str(ei.value)


def test_rename_will_not_rewrite_antimony_syntax():
    """`old` is substituted across the WHOLE model, so it needs the identifier check `new` had.
    `{old: 'const'}` passed every other guard and rewrote the declaration `const k1, k2;`, which
    breaks the model on the next reconcile and reports success."""
    src = "model m\n  const k1, k2;\n  r1: A => B; k1*A;\nend\n"
    for bad, why in (({"op": "rename", "old": "const", "new": "konst"}, "keyword"),
                     ({"op": "rename", "old": "k1;x", "new": "k2"}, "not an identifier")):
        with pytest.raises(OperationError) as ei:
            me.apply_model_edits(src, [bad])
        assert "NOTHING was applied" in str(ei.value), why
    # a genuine rename still works
    out, _ = me.apply_model_edits(src, [{"op": "rename", "old": "k1", "new": "k_fwd"}])
    assert "const k_fwd, k2;" in out and "k_fwd*A" in out


def test_set_initial_on_a_reconcile_minted_init_parameter_says_which_surface_owns_it():
    """The reconcile mints `*_init` parameters that appear in get_parameters()/get_model() — the
    RENDER — while these operations apply to the SOURCE. A caller reading `EGFR_init` off the
    parameter table is not confused about its model; it is seeing a symbol the text really does
    not contain, so the error has to say where that symbol lives."""
    with pytest.raises(OperationError) as ei:
        me.apply_model_edits("model m\n  r1: EGFR => pEGFR; k*EGFR;\nend\n",
                             [{"op": "set_initial", "id": "EGFR_init", "antimony": "5"}])
    msg = str(ei.value)
    assert "parameter table" in msg and "`parameters`" in msg


def test_a_reaction_may_span_lines_or_omit_its_rate_law():
    """Both are legal Antimony and both were reported as UNNAMED by a line-bounded matcher — so
    authoring was refused for a reaction that plainly has an id, with an error naming no fix, and
    any such model that got in was unaddressable by every operation."""
    multi = "model m()\n  r1: A + B -> C;\n     k1*A*B;\nend\n"
    assert me.reaction_ids(multi) == ["r1"]
    assert me.unnamed_reactions(multi) == []
    # a law-less reaction must not swallow the assignment that follows it
    lawless = "model m()\n  r1: A -> B;\n  k1 = 0.5;\nend\n"
    assert me.reaction_ids(lawless) == ["r1"]
    assert me.assigned_symbols(lawless) == ["k1"]


def test_an_assignment_without_a_trailing_semicolon_can_still_be_set():
    """`assigned_symbols` treated the `;` as optional while the rewrite pattern required it, so
    the caller was told its id was unknown and shown that same id in the list of known ones."""
    src = "model m()\n  r1: A -> B; k1*A\n  A = 1\n  k1 = 0.5\nend\n"
    assert me.assigned_symbols(src) == ["A", "k1"]
    out, _ = me.apply_model_edits(src, [{"op": "set_initial", "id": "A", "antimony": "7"}])
    assert "A = 7;" in out and "k1 = 0.5" in out


def test_no_operation_reads_or_writes_inside_a_comment():
    """Comments are free text. Editing one and reporting success is the worst failure shape here:
    the run fits and commits a model whose initial condition never changed, while the summary says
    it did."""
    src = "model m()\n  // A = 1; the literature value\n  A = 3;\nend\n"
    out, log = me.apply_model_edits(src, [{"op": "set_initial", "id": "A", "antimony": "9"}])
    assert "// A = 1; the literature value" in out, "the comment must be untouched"
    assert "A = 9;" in out and "A = 3;" not in out, "the real assignment must be the one edited"
    # and a symbol that exists ONLY in a comment is not in the model
    with pytest.raises(OperationError, match="does not occur in the model"):
        me.apply_model_edits("model m()\n  // kcat is the turnover number\n  r1: A->B; k1*A;\nend",
                             [{"op": "rename", "old": "kcat", "new": "kturn"}])


def test_rename_refuses_a_keyword_on_EITHER_side():
    """The argument for guarding `new` is the argument for guarding `old`: both are substituted
    across the whole model. `{new: 'const'}` yields `const const;`, which fails the next reconcile
    with an opaque Antimony error and burns the batch."""
    src = "model m()\n  const k1;\n  r1: A -> B; k1*A;\n  k1 = 0.5;\nend\n"
    for bad in ({"op": "rename", "old": "const", "new": "konst"},
                {"op": "rename", "old": "k1", "new": "const"}):
        with pytest.raises(OperationError, match="Antimony syntax"):
            me.apply_model_edits(src, [bad])


def test_a_new_reaction_id_cannot_collide_with_an_existing_symbol():
    """Antimony keeps ONE symbol table, so reusing a parameter's name for a reaction is a
    redefinition — reported by libantimony as an opaque parse failure at reconcile time, far from
    the operation that caused it, instead of as a rejection with a remedy."""
    src = "model m()\n  r1: A -> B; k1*A;\n  k1 = 0.5;\nend\n"
    with pytest.raises(OperationError, match="one symbol table"):
        me.apply_model_edits(src, [{"op": "add_reaction", "antimony": "k1: B -> C; k2*B;"}])


def test_add_reaction_keeps_a_trailing_comment_and_refuses_smuggled_statements():
    """Reassembling the statement from parsed groups silently dropped whatever else was in the
    field. Carry the comment; refuse the extra statement rather than discard it."""
    src = "model m()\n  r1: A->B; k1*A;\nend\n"
    out, _ = me.apply_model_edits(src, [{"op": "add_reaction",
                                         "antimony": "r2: B -> C; k2*B; // reverse of r1"}])
    assert "// reverse of r1" in out
    with pytest.raises(OperationError, match="beyond the reaction statement"):
        me.apply_model_edits(src, [{"op": "add_reaction", "antimony": "r3: A->C; k*A; k = 5;"}])


def test_rename_refuses_time_and_the_math_builtins():
    """`rename(old='time', new='tau')` rewrote `piecewise(0, time < t0, 1)` and reported success:
    nothing binds `tau` to simulation time, so reconcile mints it as an ESTIMATED parameter and
    every gated input goes constant while the model still compiles and fits. The one code path
    that can cause that failure directly was the only place not guarding against it."""
    src = "model m\n  r1: A -> B; k1*A*piecewise(0, time < t0, 1);\nend"
    for bad in ({"op": "rename", "old": "time", "new": "tau"},
                {"op": "rename", "old": "k1", "new": "exp"},
                {"op": "rename", "old": "k1", "new": "time"}):
        with pytest.raises(OperationError) as ei:
            me.apply_model_edits(src, [bad])
        assert "NOTHING was applied" in str(ei.value)
    # a real rename still works
    out, _ = me.apply_model_edits(src, [{"op": "rename", "old": "k1", "new": "k_fwd"}])
    assert "k_fwd*A" in out and "time < t0" in out


def test_a_time_gated_rate_law_stays_addressable():
    """The law group excluded `=`, so `piecewise(0, time >= t0, 1)` made its reaction invisible to
    every operation AND falsely reported as unnamed — meaning the wholesale path rejected exactly
    the time-gated models the isensee/raimundez prompts instruct the agent to write. Same class as
    the `/` exclusion that broke every saturating law."""
    for law in ("k1*A*piecewise(0, time >= t0, 1)", "k1*A*piecewise(0, time <= t0, 1)",
                "k1*A*piecewise(0, time < t0, 1)"):
        src = f"model m\n  r1: A -> B; {law};\nend"
        assert me.reaction_ids(src) == ["r1"], law
        assert me.unnamed_reactions(src) == [], law
    # and the reason `=` was excluded still holds: a law-less reaction must not eat the next line
    lawless = "model m\n  r1: A -> B;\n  k1 = 0.5;\nend"
    assert me.reaction_ids(lawless) == ["r1"]
    assert me.assigned_symbols(lawless) == ["k1"]


def test_add_reaction_inserts_inside_the_model_and_after_a_multiline_reaction():
    """Anchoring on the last reaction's START inserted the new statement between a multi-line
    reaction's stoichiometry and its rate law — orphaning the law and reporting success. And with
    `end` on the same line as the last reaction, a line-end insert landed OUTSIDE `model ... end`."""
    multi = "model m\n  r1: A -> B;\n    k1*A;\n  A = 1;\nend"
    out, _ = me.apply_model_edits(multi, [{"op": "add_reaction", "antimony": "r2: B -> C; k2*B;"}])
    assert me.reaction_ids(out) == ["r1", "r2"], out
    assert "r1: A -> B;\n    k1*A;" in out, "r1 lost its rate law"
    inline_end = "model m\n  A = 1; r1: A -> B; k1*A; end"
    out2, _ = me.apply_model_edits(inline_end, [{"op": "add_reaction",
                                                 "antimony": "r2: B -> C; k2*B;"}])
    assert me.reaction_ids(out2) == ["r1", "r2"]
    assert out2.index("r2:") < out2.rindex("end"), "inserted outside the model block"


def test_set_initial_does_not_rewrite_an_event_assignment():
    """An event assigns the same symbol at a trigger time; rewriting THAT leaves the real initial
    value untouched and reports success. Of the remaining candidates the LAST wins, because a
    symbol assigned twice at top level takes the later value."""
    ev = "model m\n  r1: A -> B; k1*A;\n  E1: at (time > 10): A = 5;\n  A = 1;\nend"
    out, _ = me.apply_model_edits(ev, [{"op": "set_initial", "id": "A", "antimony": "7"}])
    assert "at (time > 10): A = 5;" in out, "the event was rewritten instead of the initial value"
    assert "A = 7;" in out and "A = 1;" not in out
    dup = "model m\n  r1: A -> B; k1*A;\n  A = 1;\n  A = 3;\nend"
    out2, _ = me.apply_model_edits(dup, [{"op": "set_initial", "id": "A", "antimony": "9"}])
    assert out2.count("A = 9;") == 1 and "A = 1;" in out2, "rewrote the shadowed first assignment"


def test_a_new_reaction_id_cannot_collide_with_a_rate_law_parameter():
    """`model_symbols` read only stoichiometries, so it missed rate-law-only symbols — which are
    exactly the ones the reconcile mints. `add_reaction('k_deg: ...')` against a model whose only
    `k_deg` lives in a rate law was accepted, redefining it and turning this guard's own refusal
    into an opaque libantimony error at reconcile time."""
    src = "model m\n  r1: A -> B; k_deg*A;\nend"
    assert "k_deg" in me.model_symbols(src)
    with pytest.raises(OperationError, match="rate-law parameter"):
        me.apply_model_edits(src, [{"op": "add_reaction", "antimony": "k_deg: B -> C; k2*B;"}])


def test_add_reaction_targets_the_model_block_not_a_preceding_function():
    """Antimony allows `function ... end` before the model. Anchoring on the FIRST `end` in the
    source put the new reaction inside the function body and still reported "added reaction" —
    libantimony then either resolves it in the wrong namespace or fails opaquely at reconcile."""
    src = ("function hill(x, k, n)\n  x^n/(k^n + x^n);\nend\n\n"
           "model m()\n  r1: A -> B; k1*A;\nend\n")
    out, _ = me.apply_model_edits(src, [{"op": "add_reaction",
                                         "antimony": "r2: B -> C; k2*B;"}])
    fn_body = out[out.index("function"):out.index("model")]
    assert "r2" not in fn_body, f"reaction landed in the function block:\n{out}"
    assert "r2: B -> C" in out[out.index("model"):]


def test_add_reaction_targets_the_model_block_with_no_reactions_yet():
    """Same trap on the FIRST reaction of a model, where there is no reaction to anchor on: the
    terminator search starts after the `model` header rather than at offset 0."""
    src = ("function hill(x, k, n)\n  x^n/(k^n + x^n);\nend\n\n"
           "model m()\n  A = 1;\nend\n")
    out, _ = me.apply_model_edits(src, [{"op": "add_reaction",
                                         "antimony": "r2: B -> C; k2*B;"}])
    assert "r2" not in out[out.index("function"):out.index("model")]
    assert "r2: B -> C" in out[out.index("model"):]


# ------------------------------------------------- reserved / colliding ids (third review pass)
@pytest.mark.parametrize("rid", ["at", "in", "time", "log10"])
def test_add_reaction_refuses_a_reserved_id_parsed_from_the_statement(rid):
    """`_PROTECTED_SYMBOLS` was consulted where the caller supplies `id` separately, and on
    rename — but NOT for the id parsed out of the statement, which is the documented normal form.
    So an Antimony keyword sailed through and libantimony reported `unexpected ->`, an opaque
    parse error naming nothing about the operation that caused it. `model_symbols` even asserts
    "a caller cannot name a reaction `time` anyway: that is refused as a keyword" — which held
    only when the model happened to mention `time` already."""
    with pytest.raises(OperationError) as ei:
        me.apply_model_edits(SRC, [{"op": "add_reaction", "antimony": f"{rid}: A => B; k1*A;"}])
    assert ei.value.cause == "reserved_id"
    assert rid in str(ei.value)


def test_add_reaction_still_accepts_an_ordinary_id():
    """The guard must not start refusing normal names — `at_rate` merely CONTAINS a keyword."""
    out, summary = me.apply_model_edits(
        SRC, [{"op": "add_reaction", "antimony": "at_rate: A => B; k1*A;"}])
    assert summary == ["added reaction at_rate"]
    assert "at_rate: A => B; k1*A;" in out


def test_rename_onto_an_existing_REACTION_id_is_refused():
    """The species guard reads `in_stoich`, and reaction ids never appear in a stoichiometry — so
    this collision came through the other door. `rename(k1 -> rev)` produced
    `rev: B => A; rev*B;`, and in Antimony a reaction id inside an expression denotes that
    reaction's RATE, so the law referenced itself. It committed, and the summary reported
    `renamed k1 -> rev (1 occurrences)` as a success."""
    with pytest.raises(OperationError) as ei:
        me.apply_model_edits(SRC, [{"op": "rename", "old": "k1", "new": "rev"}])
    assert ei.value.cause == "reaction_id_merge_refused"


def test_renaming_a_reaction_itself_is_still_refused_only_on_collision():
    """Renaming a reaction to a FREE name stays legal — the guard is about merging, not about
    reactions being untouchable."""
    out, summary = me.apply_model_edits(SRC, [{"op": "rename", "old": "rev", "new": "back"}])
    assert summary == ["renamed rev -> back (1 occurrences)"]
    assert "back: B => A; k2*B;" in out


def test_add_reaction_into_a_SINGLE_LINE_model_block():
    """`_model_block_start` matched to end of LINE, so on a one-line `model m() end` it consumed
    the terminator as well and the search for `end` started past the block — appending the first
    reaction outside the model. Only bites when the model declares no reactions yet, which is
    exactly the from-scratch authoring case."""
    src = "model m() end\n"
    out, summary = me.apply_model_edits(src, [
        {"op": "add_reaction", "antimony": "r1: A => B; k1*A;"}])
    assert summary == ["added reaction r1"]
    body, _, tail = out.partition("end")
    assert "r1: A => B; k1*A;" in body, f"reaction landed outside the model block: {out!r}"


def test_add_reaction_into_a_starred_model_header():
    """libantimony emits `model *name()`; the header match has to survive the star."""
    src = "model *m()\nend\n"
    out, _ = me.apply_model_edits(src, [
        {"op": "add_reaction", "antimony": "r1: A => B; k1*A;"}])
    body, _, _ = out.partition("end")
    assert "r1: A => B; k1*A;" in body, f"reaction landed outside the model block: {out!r}"


# ------------------------------------------------------------------ set_assignment (readouts)
def test_set_assignment_creates_a_rule_inside_the_model_block():
    out, summary = me.apply_model_edits(
        SRC, [{"op": "set_assignment", "id": "readout_total", "antimony": "A + B"}])
    assert summary == ["set assignment readout_total := A + B"]
    body, _, _ = out.partition("end")
    assert "readout_total := A + B;" in body


def test_set_assignment_replaces_an_existing_rule_in_place():
    """Re-defining a readout must not leave two `:=` statements — Antimony would reject the
    redefinition opaquely."""
    once, _ = me.apply_model_edits(
        SRC, [{"op": "set_assignment", "id": "readout_total", "antimony": "A + B"}])
    twice, summary = me.apply_model_edits(
        once, [{"op": "set_assignment", "id": "readout_total", "antimony": "2*B"}])
    assert summary == ["set assignment readout_total := 2*B"]
    assert twice.count(":=") == 1 and "readout_total := 2*B;" in twice


def test_set_assignment_refuses_a_species():
    """A state integrated by the ODEs cannot also carry an assignment rule — the model would be
    over-determined, and libantimony's rejection would not name the cause."""
    with pytest.raises(OperationError) as ei:
        me.apply_model_edits(SRC, [{"op": "set_assignment", "id": "A", "antimony": "2*B"}])
    assert ei.value.cause == "assign_to_species_refused"


def test_set_assignment_refuses_a_reserved_name():
    with pytest.raises(OperationError) as ei:
        me.apply_model_edits(SRC, [{"op": "set_assignment", "id": "time", "antimony": "A"}])
    assert ei.value.cause == "reserved_id"


def test_a_boundary_species_declared_out_of_line_is_not_a_dynamic_state():
    """`$` marks the boundary wherever it is written.

    Antimony round-trips a boundary species as a `species $EGF;` DECLARATION with the reaction
    naming it bare, so stripping only the inline `$EGF -> B` form counted it as a dynamic state.
    Everything built on this predicate then saw an over-determined model and refused the whole
    batch -- including the `remove_assignment` the error message recommends, locking the caller
    out of `model_edits` on exactly the models that need it.
    """
    from petab_mcp.model_edits import _stoichiometry_symbols

    out_of_line = _stoichiometry_symbols("species $EGF, B;\n  R1: EGF + B -> C; k*EGF*B\n")
    assert "EGF" not in out_of_line and {"B", "C"} <= out_of_line
    # the inline form keeps working
    assert "EGF" not in _stoichiometry_symbols("R1: $EGF -> B; k\n")
