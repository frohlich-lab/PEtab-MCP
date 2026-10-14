from __future__ import annotations

import json
import asyncio
import shutil
from pathlib import Path

import pytest

from petab_mcp.references import StubReferenceResolver
from petab_mcp.server import create_server
from conftest import with_empty_sbml

YAML_PATH = Path(__file__).parent / "data" / "toy_v2" / "problem.yaml"
DATA_DIR = YAML_PATH.parent

# Tables that must never be mutable — assert no tool can write them.
_FORBIDDEN_TOOL_SUBSTRINGS = ("measurement", "observable", "condition")
_WRITE_VERBS = ("update", "set", "add", "delete", "remove", "edit")

# The shipped toy model — structural edits in these tests are made by re-authoring this base
# (via `set_model`) with extra reaction lines spliced into `{extra}`.
_TOY_BASE_MODEL = """model conversion_reaction
  compartment compartment_ = 1;
  species A in compartment_;
  species B in compartment_;
  A = A_init;
  B = B_init;
  A_init = 1.0;
  B_init = 0.0;
  k1 = 0.5;
  k2 = 0.1;
  r1: A => B; k1 * A;
  r2: B => A; k2 * B;
{extra}end
"""


def _reauthored(extra_reactions=()):
    """The toy model Antimony re-authored with ``extra_reactions`` (Antimony lines) added."""
    body = "".join(f"  {line}\n" for line in extra_reactions)
    return _TOY_BASE_MODEL.format(extra=body)


def _server(tmp_path, auto_fit=False):
    # auto_fit defaults off so structural/versioning tests don't compile+fit on every
    # commit (and run without AMICI); the fit-tool test opts in explicitly.
    return create_server(
        str(YAML_PATH),
        provenance_dir=str(tmp_path / "prov"),
        resolver=StubReferenceResolver(),
        auto_fit=auto_fit,
    )


def _empty_model_server(tmp_path, *, sbml=True):
    """A server over a problem that ships WITHOUT a model: an empty SBML and no model.ant, or,
    with ``sbml=False``, no model file at all (``model_files: {}``)."""
    work = tmp_path / "empty"
    work.mkdir()
    for name in ("problem.yaml", "measurements.tsv", "observables.tsv", "parameters.tsv"):
        shutil.copy(DATA_DIR / name, work / name)
    if sbml:
        with_empty_sbml(work / "problem.yaml")
    else:
        (work / "problem.yaml").write_text(
            "format_version: 2.0.0\nparameter_files:\n- parameters.tsv\nmodel_files: {}\n"
            "measurement_files:\n- measurements.tsv\nobservable_files:\n- observables.tsv\n",
            encoding="utf-8",
        )
    return create_server(
        str(work / "problem.yaml"),
        provenance_dir=str(tmp_path / "prov"),
        resolver=StubReferenceResolver(),
        auto_fit=False,
    )


_REVERSIBLE_ANTIMONY = (
    "model conversion_reaction\n"
    "  compartment compartment_ = 1;\n"
    "  species A in compartment_;\n"
    "  species B in compartment_;\n"
    "  A = A_init; B = B_init;\n"
    "  A_init = 1.0; B_init = 0.0;\n"
    "  k1 = 0.25; k2 = 0.05;\n"
    "  r1: A => B; k1 * A;\n"
    "  r2: B => A; k2 * B;\n"
    "end\n"
)

# A one-reaction re-author (drops r2/k2) — a genuinely different structure from the toy model.
_ONE_REACTION_ANTIMONY = (
    "model conversion_reaction\n"
    "  compartment compartment_ = 1;\n"
    "  species A in compartment_;\n"
    "  species B in compartment_;\n"
    "  A = A_init; B = B_init;\n"
    "  A_init = 1.0; B_init = 0.0;\n"
    "  k1 = 0.25;\n"
    "  r1: A => B; k1 * A;\n"
    "end\n"
)


def _structured(result):
    return result.structured_content  # FastMCP.call_tool -> ToolResult(content, structured_content)


def test_tools_registered(tmp_path):
    mcp = _server(tmp_path)
    names = {t.name for t in asyncio.run(mcp.list_tools())}
    expected = {
        "describe_problem",
        "list_observables",
        "list_conditions",
        "list_experiments",
        "get_measurements",
        "get_model",
        "get_model_structure",
        "make_changes",
        "get_parameters",
        "get_fit_summary",
        "get_fit_diagnosis",
        "get_run_timing",
        "query_data",
        "list_data_tables",
        "get_fitting_log",
        "get_provenance",
        "list_versions",
        "get_version",
        "diff_versions",
        "restore_version",
        "get_progress",
    }
    assert expected <= names
    # `rollback_to` and `current_version` are NOT tools: restore_version IS rollback_to under the
    # name the going-back workflow uses, and describe_problem already returns current_version /
    # One operation, one tool schema.
    assert not ({"rollback_to", "current_version", "reset"} & names)
    # `validate_problem` and `sync_parameters` were dead weight: every edit already
    # runs the same lint and the same reconcile, so neither told a caller anything it
    # could act on. The underlying validation.validate_problem() is unchanged and
    # still gates every commit.
    assert not ({"validate_problem", "sync_parameters"} & names)
    # the incremental structural tools are not registered — structure is edited via make_changes
    assert not ({"add_reaction", "change_rate_law", "remove_reaction"} & names)
    # `make_changes` is the single editor AND the only thing that fits: there is no fit-free edit
    # and no standalone fit, so a change and its score cannot come apart.
    assert not ({"set_model", "change_parameters", "fit"} & names)
    assert not (
        {
            "simulate",
            "evaluate_objective",
            "optimize",
            "residual_statistics",
            "update_parameters",
            "refit",
        }
        & names
    )
    assert "update_model" not in names and "set_model_parameter" not in names
    # the batch tools were deleted (issue #81): edits no longer fit, so batching is obsolete
    assert not ({"begin_batch", "commit_batch", "abort_batch"} & names)
    # changing a parameter's estimate status in-loop is deliberately unsupported — no such tool
    assert not (
        {"estimate_parameter", "change_parameter_estimated", "fix_parameter", "pin_parameter"}
        & names
    )
    # the handcoded residual-analysis tools are gone — SQL via query_data replaces them
    assert not ({"inspect_fit", "get_simulations"} & names)


def test_no_tool_mutates_fixed_tables(tmp_path):
    names = {t.name for t in asyncio.run(_server(tmp_path).list_tools())}
    for name in names:
        writes = any(v in name for v in _WRITE_VERBS)
        touches_fixed = any(s in name for s in _FORBIDDEN_TOOL_SUBSTRINGS)
        assert not (writes and touches_fixed), f"tool {name} could mutate fixed data"


def test_describe_and_parameters(tmp_path):
    mcp = _server(tmp_path)

    async def go():
        desc = _structured(await mcp.call_tool("describe_problem", {}))
        assert desc["n_measurements"] == 6
        assert desc["free_parameters"] == ["k1", "k2"]
        assert desc["current_version"] == -1

        params = _structured(await mcp.call_tool("get_parameters", {}))
        recs = params["result"] if isinstance(params, dict) and "result" in params else params
        assert {r["parameterId"] for r in recs} == {"k1", "k2", "sigma_B", "A_init", "B_init"}

    asyncio.run(go())


def test_edit_then_fit_creates_version_via_tools(tmp_path):
    # One `make_changes` call is one edit AND its fit, snapshotting one version. auto_fit is off,
    # so the fit is skipped (no AMICI) but still commits — enough to exercise the flow and the
    # event log, which must record BOTH an edit and a fit event for the single call.
    mcp = _server(tmp_path)

    async def go():
        res = _structured(
            await mcp.call_tool(
                "make_changes",
                {
                    "antimony": _reauthored(["J: B => D; k3 * B;"]),
                    "justification": "B converts to D",
                    "references": ["PMID:1"],
                },
            )
        )
        assert res["applied"] is True and res["committed"] is True and res["version"] == 0
        assert _structured(await mcp.call_tool("describe_problem", {}))["current_version"] == 0

        versions = _structured(await mcp.call_tool("list_versions", {}))
        vlist = (
            versions["result"] if isinstance(versions, dict) and "result" in versions else versions
        )
        assert {v["version"] for v in vlist} == {0}

        prov = _structured(await mcp.call_tool("get_provenance", {}))
        plist = prov["result"] if isinstance(prov, dict) and "result" in prov else prov
        assert any(e["type"] == "edit" for e in plist)
        assert any(e["type"] == "fit" for e in plist)

    asyncio.run(go())


def test_the_result_is_the_fit_and_nothing_else(tmp_path):
    """`make_changes` reports the SCORE, flat, plus whether the change is in place. The edit's own
    bookkeeping — the reconcile report, the validation report, the parameter diff, the per-species
    initial-condition account, the echoed justification/references — is not in it: none of it is a
    score, and all of it would ride on every call into a history nothing prunes. What it dropped
    is still readable on demand (`get_parameters` annotates each row's role and estimated/fixed
    status, `get_model_structure` the reconciled model)."""
    mcp = _server(tmp_path)  # auto_fit off -> commit without fitting (no AMICI needed)

    async def go():
        res = _structured(
            await mcp.call_tool(
                "make_changes",
                {
                    "antimony": _reauthored(["J: B => D; k3 * B;"]),
                    "justification": "B converts to D",
                },
            )
        )
        assert res["applied"] is True and res["committed"] is True and res["version"] == 0
        assert not (
            {
                "reconcile",
                "validation",
                "parameter_diff",
                "initial_conditions",
                "justification",
                "references",
                "operation",
                "staged_edit",
                "fit",
            }
            & set(res)
        )
        # the fit's own keys ARE there, at the top level and named as get_fit_summary names them
        assert {"fit_status", "neg_log_likelihood", "n_estimated", "bic"} <= set(res)

    asyncio.run(go())


def test_references_optional_but_justification_required(tmp_path):
    mcp = _server(tmp_path)  # auto_fit off -> commit without fitting (no AMICI needed)

    async def go():
        # references are now OPTIONAL for model ops: an edit without any still applies
        ok = _structured(
            await mcp.call_tool(
                "make_changes",
                {
                    "antimony": _reauthored(["J: B => D; k3 * B;"]),
                    "justification": "B is degraded to D",
                },
            )
        )
        assert ok["applied"] is True

        # justification is still REQUIRED: an empty one is rejected
        bad = _structured(
            await mcp.call_tool(
                "make_changes",
                {
                    "antimony": _reauthored(["J2: D => E; k4 * D;"]),
                    "justification": "   ",
                    "references": [],
                },
            )
        )
        assert bad["applied"] is False and "error" in bad

    asyncio.run(go())


def test_rejected_operation_is_logged(tmp_path):
    mcp = _server(tmp_path)

    async def go():
        res = _structured(
            await mcp.call_tool(
                "make_changes",
                {
                    "parameters": [{"parameter_id": "A_init", "scale": "log10"}],
                    "justification": "x",
                },
            )
        )
        assert res["applied"] is False  # A_init lb=0 -> log10 invalid
        errs = _structured(await mcp.call_tool("get_errors", {}))
        elist = errs["result"] if isinstance(errs, dict) and "result" in errs else errs
        assert any(e["tool"] == "make_changes" for e in elist)

    asyncio.run(go())


def test_negative_n_starts_rejected(tmp_path):
    # n_starts is validated before the edit applies, so a bad value is a pure no-op: no edit,
    # no version.
    mcp = _server(tmp_path)

    async def go():
        res = _structured(
            await mcp.call_tool(
                "make_changes",
                {
                    "antimony": _reauthored(["J: B => D; k3 * B;"]),
                    "justification": "x",
                    "n_starts": -1,
                },
            )
        )
        assert res["applied"] is False
        assert "n_starts" in res["error"]
        cur = _structured(await mcp.call_tool("describe_problem", {}))
        assert cur["current_version"] == -1

    asyncio.run(go())


def test_a_change_that_changes_nothing_is_rejected(tmp_path):
    # `antimony` and `parameters` are both optional, but not both absent: there would be nothing
    # to score, and the call would still burn a fit.
    mcp = _server(tmp_path)

    async def go():
        res = _structured(await mcp.call_tool("make_changes", {"justification": "nothing"}))
        assert res["applied"] is False
        assert "antimony" in res["error"] and "parameters" in res["error"]
        assert _structured(await mcp.call_tool("describe_problem", {}))["current_version"] == -1

    asyncio.run(go())


def test_query_data_rejects_bad_input(tmp_path):
    mcp = _server(tmp_path)  # arg validation fires before any simulation, so no AMICI needed

    async def go():
        empty = _structured(await mcp.call_tool("query_data", {"sql": ""}))
        assert empty.get("ok") is False and "sql" in empty.get("error", "")
        bad_rows = _structured(
            await mcp.call_tool("query_data", {"sql": "SELECT 1", "max_rows": 0})
        )
        assert bad_rows.get("ok") is False and "max_rows" in bad_rows.get("error", "")

    asyncio.run(go())


def test_query_data_logs_sql_on_error(tmp_path):
    mcp = _server(tmp_path)  # querying a missing table fails; the sql must be in get_errors

    async def go():
        res = _structured(
            await mcp.call_tool("query_data", {"sql": "SELECT * FROM does_not_exist"})
        )
        assert res.get("ok") is False
        errs = _structured(await mcp.call_tool("get_errors", {}))
        elist = errs["result"] if isinstance(errs, dict) and "result" in errs else errs
        assert any(
            e.get("tool") == "query_data"
            and "does_not_exist" in str((e.get("args") or {}).get("sql", ""))
            for e in elist
        )

    asyncio.run(go())


def test_query_data_runs_on_copy(tmp_path):
    # Mutating SQL runs against a disposable copy: it succeeds and does not affect the
    # session — a fresh query still sees the full table.
    mcp = _server(tmp_path)

    async def go():
        dropped = _structured(await mcp.call_tool("query_data", {"sql": "DROP TABLE measurements"}))
        assert dropped.get("ok") is not False  # not rejected
        again = _structured(
            await mcp.call_tool("query_data", {"sql": "SELECT count(*) AS n FROM measurements"})
        )
        assert again["rows"][0]["n"] == 6

    asyncio.run(go())


def test_query_data_blocks_file_access(tmp_path):
    mcp = _server(tmp_path)  # host file access stays disabled even on the copy

    async def go():
        res = _structured(
            await mcp.call_tool(
                "query_data", {"sql": "SELECT * FROM read_csv_auto('/etc/hostname')"}
            )
        )
        assert res.get("ok") is False

    asyncio.run(go())


def test_get_measurements_raw_is_paged(tmp_path):
    # The unpaged dump measured 73k tokens on isensee and 81k on lucarelli, in ONE tool result
    # that then re-enters context every later turn — one such call has ended a run on the
    # provider's "prompt exceeds max length". Raw reads are paged; the summary is not (it is
    # already compact) and keeps its list/dict shape.
    mcp = _server(tmp_path)

    def _page(res):
        # get_measurements is annotated `list | dict`, so FastMCP wraps the payload in
        # {"result": ...} — the same unwrap test_describe_and_parameters does for get_parameters.
        out = _structured(res)
        return out["result"] if isinstance(out, dict) and "result" in out else out

    async def go():
        first = _page(await mcp.call_tool("get_measurements", {"summarize": False, "max_rows": 4}))
        assert first["row_count"] == 4
        assert first["total_rows"] == 6
        assert first["offset"] == 0
        assert first["truncated"] is True
        assert first["next_offset"] == 4

        second = _page(
            await mcp.call_tool(
                "get_measurements",
                {"summarize": False, "max_rows": 4, "offset": first["next_offset"]},
            )
        )
        assert second["row_count"] == 2
        assert second["truncated"] is False
        assert "next_offset" not in second

        # the two pages together are the whole table, each row once
        rows = first["rows"] + second["rows"]
        assert len(rows) == 6
        assert len({json.dumps(r, sort_keys=True) for r in rows}) == 6

        # a page that fits is not marked truncated
        whole = _page(
            await mcp.call_tool("get_measurements", {"summarize": False, "max_rows": 500})
        )
        assert whole["row_count"] == 6 and whole["truncated"] is False

    asyncio.run(go())


def test_list_data_tables(tmp_path):
    mcp = _server(tmp_path)  # schema is static; no simulation/AMICI needed

    async def go():
        schema = _structured(await mcp.call_tool("list_data_tables", {}))
        tables = {t["table"] for t in schema["tables"]}
        assert tables == {
            "measurements",
            "conditions",
            "experiments",
            "observables",
            "parameters",
            "simulations",
            "simulations_history",
            "optimization_starts",
            "optimization_trace",
        }

    asyncio.run(go())


def test_query_data_static_tables(tmp_path):
    # Querying the static PEtab tables works even when the simulation dataframe cannot be
    # produced (no AMICI here): query_data logs and falls through with an empty `simulations`.
    mcp = _server(tmp_path)

    async def go():
        out = _structured(
            await mcp.call_tool("query_data", {"sql": "SELECT parameterId FROM parameters"})
        )
        ids = {r["parameterId"] for r in out["rows"]}
        assert ids == {"k1", "k2", "sigma_B", "A_init", "B_init"}
        assert out["truncated"] is False

    asyncio.run(go())


def test_a_re_author_and_its_bounds_are_one_edit_and_one_version(tmp_path):
    # A structural change and the bounds it needs are ONE call and therefore ONE version — the
    # use-case the old edit/edit/fit sequence served, minus the chance of leaving either half
    # unscored. auto_fit off so the fit snapshots without AMICI.
    mcp = _server(tmp_path)

    async def go():
        res = _structured(
            await mcp.call_tool(
                "make_changes",
                {
                    "antimony": _reauthored(["J: B => D; k3 * B;"]),
                    "parameters": [{"parameter_id": "k1", "lower_bound": 1e-2, "upper_bound": 1e2}],
                    "justification": "x",
                    "references": ["PMID:1"],
                },
            )
        )
        assert res["applied"] is True and res["version"] == 0
        cur = _structured(await mcp.call_tool("describe_problem", {}))
        assert cur["current_version"] == 0
        # both halves are recorded as ONE edit, and the fit event says it scored exactly that one
        prov = _structured(await mcp.call_tool("get_provenance", {}))
        plist = prov["result"] if isinstance(prov, dict) and "result" in prov else prov
        edit_ev = [e for e in plist if e["type"] == "edit"][-1]
        assert edit_ev["operation"] == "make_changes"
        fit_ev = [e for e in plist if e["type"] == "fit"][-1]
        assert fit_ev["n_edits"] == 1

    asyncio.run(go())


def test_a_re_author_is_not_refused_when_a_model_exists(tmp_path):
    # A re-author is NOT refused on an existing model — it replaces it wholesale.
    mcp = _server(tmp_path)

    async def go():
        desc = _structured(await mcp.call_tool("describe_problem", {}))
        assert desc["model_authored"] is True
        res = _structured(
            await mcp.call_tool(
                "make_changes",
                {
                    "antimony": _ONE_REACTION_ANTIMONY,
                    "justification": "re-author",
                    "references": ["PMID:1"],
                },
            )
        )
        assert res["applied"] is True
        struct = _structured(await mcp.call_tool("get_model_structure", {}))
        assert {r["id"] for r in struct["reactions"]} == {"r1"}  # the edit took effect
        # (the fit-time de-dup no-op for an identical model is covered in test_fit.py)

    asyncio.run(go())


def test_restore_version_tool_registered_and_restores(tmp_path):
    # restore_version (= session.rollback_to) restores a prior version and re-simulates it, so it needs
    # the engine (AMICI). The engine-free restore behaviour is covered in test_dedup.
    mcp = _server(tmp_path)

    async def go():
        # score the reversible model as v0, re-author to one reaction (v1), then restore v0
        await mcp.call_tool(
            "make_changes",
            {
                "antimony": _REVERSIBLE_ANTIMONY,
                "justification": "score the reversible model",
                "references": ["PMID:1"],
            },
        )
        struct = _structured(await mcp.call_tool("get_model_structure", {}))
        assert {r["id"] for r in struct["reactions"]} == {"r1", "r2"}
        await mcp.call_tool(
            "make_changes",
            {
                "antimony": _ONE_REACTION_ANTIMONY,
                "justification": "drop r2",
                "references": ["PMID:1"],
            },
        )
        struct = _structured(await mcp.call_tool("get_model_structure", {}))
        assert {r["id"] for r in struct["reactions"]} == {"r1"}
        out = _structured(await mcp.call_tool("restore_version", {"version": 0}))
        assert out["rolled_back_to"] == 0
        struct = _structured(await mcp.call_tool("get_model_structure", {}))
        assert {r["id"] for r in struct["reactions"]} == {"r1", "r2"}

    asyncio.run(go())


@pytest.mark.parametrize("sbml", [True, False], ids=["empty_sbml", "no_model_files"])
def test_make_changes_builds_the_first_model_via_tool(tmp_path, sbml):
    mcp = _empty_model_server(tmp_path, sbml=sbml)

    async def go():
        desc = _structured(await mcp.call_tool("describe_problem", {}))
        assert desc["model_authored"] is False
        assert desc["n_reactions"] == 0

        # `antimony` builds the first model when the problem ships model-less
        res = _structured(
            await mcp.call_tool(
                "make_changes",
                {
                    "antimony": _REVERSIBLE_ANTIMONY,
                    "justification": "reversible conversion",
                    "references": ["PMID:1"],
                },
            )
        )
        assert res["applied"] is True

        desc2 = _structured(await mcp.call_tool("describe_problem", {}))
        assert desc2["model_authored"] is True  # the edit built the working model
        assert desc2["n_reactions"] == 2

    asyncio.run(go())


def test_an_unauthored_model_gets_no_parameter_diagnosis(tmp_path):
    """REGRESSION: the parameter diagnosis of a run that has authored nothing used to answer with a
    confident FIM of the SEEDED skeleton — ranked candidates, a rank-deficient FIM and merges between
    initial conditions of a model that does not exist — and paid an AMICI build (12.5 s measured on
    blasi) for it, while the same payload said "commit a model first". It is a local expansion around
    an optimum; with no frozen fit to expand around it reports `available: false` and computes
    nothing."""
    from petab_mcp import identifiability as ident_mod

    mcp = _empty_model_server(tmp_path)
    fims: list[int] = []
    real = ident_mod._compute_fim
    ident_mod._compute_fim = lambda record: fims.append(1) or real(record)

    async def go():
        diag = _structured(await mcp.call_tool("get_fit_diagnosis", {"sections": "all"}))
        assert fims == []  # nothing was computed, so nothing was compiled either
        ident = diag["identifiability"]
        assert ident["available"] is False and "no fitted optimum" in ident["note"]
        assert "fim" not in ident and "merge_candidates" not in ident
        assert diag["per_observable"] == []

    try:
        asyncio.run(go())
    finally:
        ident_mod._compute_fim = real


def test_an_unauthored_model_has_no_score_to_finish_on(tmp_path):
    """Regression: the seed simulates to constants, so get_fit_summary reported a finite nll and
    bic before anything was authored — which is the finish gate's whole admission test."""
    mcp = _empty_model_server(tmp_path)

    async def go():
        assert _structured(await mcp.call_tool("describe_problem", {}))["model_authored"] is False

        fit = _structured(await mcp.call_tool("get_fit_summary", {}))
        assert fit.get("neg_log_likelihood") is None, "the seed is not a fit"
        assert fit.get("bic") is None, "and it must not offer a score to finish on"

        # the static tables stay queryable — the simulations table is empty but TYPED
        tables = _structured(await mcp.call_tool("list_data_tables", {}))["tables"]
        by_name = {t["table"]: t for t in tables}
        assert by_name["simulations"]["n_rows"] == 0
        assert by_name["measurements"]["n_rows"] > 0

        # ...and once a mechanism IS authored, the score comes back
        await mcp.call_tool(
            "make_changes",
            {"antimony": _REVERSIBLE_ANTIMONY, "justification": "reversible conversion"},
        )
        assert _structured(await mcp.call_tool("describe_problem", {}))["model_authored"] is True

    asyncio.run(go())


def test_resources_registered(tmp_path):
    resources = asyncio.run(_server(tmp_path).list_resources())
    uris = {str(r.uri) for r in resources}
    assert any("petab://provenance" in u for u in uris)
    assert any("petab://versions" in u for u in uris)
    assert any("petab://fit/log" in u for u in uris)
    assert any("petab://simulations" in u for u in uris)
    assert any("petab://data/schema" in u for u in uris)
    assert any("model/antimony" in u for u in uris)


def test_a_tool_acts_on_the_session_routed_to_at_call_time(tmp_path):
    """REGRESSION: every tool group captured `ctx.session` when it was REGISTERED, so one
    registration could only ever serve the session it was built with. Two sessions on the same
    problem behind one `route`: a change made while it points at one is seen exactly while it
    points there, and no group may resolve a session while registering."""
    from fastmcp import FastMCP

    from petab_mcp import server as S

    def open_ctx(store):
        return S._open_ctx(
            str(YAML_PATH),
            provenance_dir=str(tmp_path / store),
            criterion=None,
            auto_fit_starts=1,
            validation_measurements=None,
            score_shipped_model=False,
            build_root=None,
            reference_mode="best_effort",
            resolver=StubReferenceResolver(),
            auto_fit=False,
        )

    a, b = open_ctx("a"), open_ctx("b")
    bound = []  # empty while registering, so a group that resolves a session there raises
    mcp = FastMCP("route")
    for name in [n for n in dir(S) if n.startswith("_register_")]:
        getattr(S, name)(mcp, lambda: bound[-1])

    async def read():
        version = _structured(await mcp.call_tool("describe_problem", {}))["current_version"]
        return version, "k3" in _structured(await mcp.call_tool("get_model", {}))["result"]

    async def go():
        bound.append(a)
        await mcp.call_tool(
            "make_changes", {"antimony": _reauthored(["J: B => D; k3 * B;"]), "justification": "x"}
        )
        assert await read() == (0, True)
        bound.append(b)
        assert await read() == (-1, False)
        bound.append(a)
        assert await read() == (0, True)

    asyncio.run(go())


def test_get_fitting_log(tmp_path):
    # No AMICI/pyPESTO needed: the pypesto.log handler is installed at session start, so
    # the tool reads a (possibly empty) log without a fit ever running. Validates the
    # read path + argument checking that the mathematician persona relies on.
    mcp = _server(tmp_path)

    async def go():
        res = _structured(await mcp.call_tool("get_fitting_log", {"tail_lines": 50}))
        assert set(res) >= {"path", "exists", "n_lines_total", "tail_lines", "log"}
        assert res["path"].endswith("pypesto.log")
        assert isinstance(res["log"], str)
        # negative tail_lines is rejected (logged as an error, returns ok=False)
        bad = _structured(await mcp.call_tool("get_fitting_log", {"tail_lines": -1}))
        assert bad.get("ok") is False

    asyncio.run(go())


def test_fit_tools(tmp_path):
    import pytest

    mcp = _server(tmp_path, auto_fit=True)

    async def go():
        # BUILD FIRST. A cold read simulates on demand only when a module for this structure is
        # already compiled -- `fitting._simulation_for_read` logs "a read never pays a build" and
        # returns nothing otherwise. This test used to skip the fit and still find six rows,
        # because `create_server` without an explicit `build_root` fell back to
        # `tempfile.gettempdir()/petab_mcp_amici`, one directory shared by every xdist worker, and
        # some OTHER test had compiled this same toy model into it. Isolating that root per worker
        # (conftest._private_tempdir_per_worker) removed the borrowed artifact and left this
        # asserting behaviour it never exercised. One commit gives it its own module and its own
        # fit -- which it needs regardless, since it goes on to assert a non-None
        # neg_log_likelihood.
        committed = _structured(
            await mcp.call_tool(
                "make_changes",
                {
                    "antimony": _reauthored([]),
                    "justification": "fit the shipped structure so the read below has a module",
                },
            )
        )
        assert committed["applied"] is True and committed["committed"] is True

        # the simulation dataframe is queryable with SQL via query_data; it is generated on
        # demand once a module exists (no re-simulation on subsequent calls).
        allrows = _structured(
            await mcp.call_tool("query_data", {"sql": "SELECT * FROM simulations"})
        )
        assert allrows["row_count"] == 6
        assert {"residual", "residual_weighted", "noise"} <= set(allrows["columns"])

        # residual breakdowns that the old inspect_fit hardcoded are now plain SQL
        by_obs = _structured(
            await mcp.call_tool(
                "query_data",
                {
                    "sql": (
                        "SELECT observable_id, count(*) AS n, "
                        "sum(residual_weighted*residual_weighted) AS chi2 "
                        "FROM simulations GROUP BY observable_id"
                    )
                },
            )
        )
        assert {r["observable_id"] for r in by_obs["rows"]} == {"obs_B"}
        assert by_obs["rows"][0]["n"] == 6 and by_obs["rows"][0]["chi2"] is not None

        by_cond = _structured(
            await mcp.call_tool(
                "query_data", {"sql": ("SELECT DISTINCT experiment_id FROM simulations")}
            )
        )
        assert {r["experiment_id"] for r in by_cond["rows"]} == {""}
        by_time = _structured(
            await mcp.call_tool("query_data", {"sql": ("SELECT DISTINCT time FROM simulations")})
        )
        assert by_time["row_count"] == 6  # six distinct timepoints

        # worst-fit points: an ORDER BY ... LIMIT instead of a dedicated tool
        worst = _structured(
            await mcp.call_tool(
                "query_data",
                {"sql": ("SELECT * FROM simulations ORDER BY abs(residual_weighted) DESC LIMIT 3")},
            )
        )
        assert worst["row_count"] == 3

        # compact metrics-only read (no records) — and the objective is now in describe
        summary = _structured(await mcp.call_tool("get_fit_summary", {}))
        assert "records" not in summary and summary["neg_log_likelihood"] is not None
        desc = _structured(await mcp.call_tool("describe_problem", {}))
        assert desc["neg_log_likelihood"] == pytest.approx(summary["neg_log_likelihood"])

        # max_rows caps the result and flags truncation
        capped = _structured(
            await mcp.call_tool("query_data", {"sql": "SELECT * FROM simulations", "max_rows": 2})
        )
        assert capped["row_count"] == 2 and capped["truncated"] is True

        # a bounds change: the result carries the unified fit block nested under `fit`, plus a
        # fitted version.
        res = _structured(
            await mcp.call_tool(
                "make_changes",
                {
                    "parameters": [{"parameter_id": "k1", "lower_bound": 1e-3, "upper_bound": 1e3}],
                    "justification": "widen bounds",
                    "n_starts": 2,
                },
            )
        )
        # v1, not v0: the setup commit at the top of this test holds v0 (it has to, so a module
        # exists for the cold read above).
        assert res["applied"] is True and res["committed"] is True and res["version"] == 1
        assert res["fit_status"] == "fit"
        assert res["neg_log_likelihood"] is not None
        assert res["improvement"]["is_new_best"] is not None
        # the compact residual-structure verdict rides on the fit result (issue #113), so the
        # agent doesn't spend a get_fit_diagnosis call to learn its model's form is incomplete —
        # but the DETAIL behind it deliberately does not, or it would ride on every fit.
        assert set(res["residual_structure"]) == {
            "flagged_observables",
            "low_power",
            "summary",
            "replicate_lack_of_fit",
        }
        assert "per_observable" not in res and "worst_points" not in res
        # `parameters_at_bound` stays HERE, on every fit result: it is the at-bound signal the lead
        # acts on (get_parameters carries none), and it costs ~25 bytes when empty.
        assert "parameters_at_bound" in res
        # ...and the tool returns the detail, in two sections, WITHOUT a second copy of the verdict.
        # The DEFAULT is the cheap residual half — the parameter half costs several kB plus a FIM,
        # and every payload asked for is re-sent in the agent's history on every later turn.
        diag = _structured(await mcp.call_tool("get_fit_diagnosis", {}))
        assert "residual_structure" not in diag and "identifiability" not in diag
        assert diag["per_observable"] and "worst_points" in diag
        both = _structured(await mcp.call_tool("get_fit_diagnosis", {"sections": "all"}))
        assert both["per_observable"] and both["identifiability"]["n_estimated"] >= 1
        params_only = _structured(
            await mcp.call_tool("get_fit_diagnosis", {"sections": "parameters"})
        )
        assert set(params_only) == {"identifiability"}
        assert params_only["identifiability"] == both["identifiability"]
        # a typo must not silently hand back half a diagnosis
        bad = _structured(await mcp.call_tool("get_fit_diagnosis", {"sections": "resid"}))
        assert bad["ok"] is False and "residuals" in bad["error"]
        # a FIM failure must not take the residual half with it: before the consolidation these
        # were two tools, so one could not break the other, and the judge's result.json block (the
        # only record of whether the replicate arm fired) rides on the residual half.
        from petab_mcp import identifiability as ident_mod

        broken = getattr(ident_mod, "identifiability")
        try:
            ident_mod.identifiability = lambda *a, **kw: (_ for _ in ()).throw(
                RuntimeError("FIM exploded")
            )
            degraded = _structured(await mcp.call_tool("get_fit_diagnosis", {"sections": "all"}))
        finally:
            ident_mod.identifiability = broken
        assert degraded["per_observable"] and degraded["identifiability"]["available"] is False
        # diff_versions now reports the fit delta between versions
        await mcp.call_tool(
            "make_changes",
            {
                "antimony": _ONE_REACTION_ANTIMONY,
                "justification": "a second scored version",
                "n_starts": 2,
            },
        )
        # The two versions this block actually cares about — the bounds change and the structure
        # committed just above — shifted up by one for the setup commit.
        d = _structured(await mcp.call_tool("diff_versions", {"from_version": 1, "to_version": 2}))
        assert "fit" in d and "neg_log_likelihood_delta" in d["fit"]

        # that fit's pyPESTO multistart is now queryable: results + per-iteration trace
        starts = _structured(
            await mcp.call_tool(
                "query_data",
                {"sql": ("SELECT count(*) AS n, min(fval) AS best FROM optimization_starts")},
            )
        )
        assert starts["rows"][0]["n"] == 2 and starts["rows"][0]["best"] is not None
        trace = _structured(
            await mcp.call_tool(
                "query_data", {"sql": ("SELECT count(*) AS n FROM optimization_trace")}
            )
        )
        assert trace["rows"][0]["n"] >= 1

    asyncio.run(go())


def test_get_parameters_filters_rows(tmp_path):
    """`parameter_ids`/`role` filter server-side. On 100+-row tables (lucarelli: 118) the
    unfiltered payload was a top-3 context cost, re-sent in the caller's history every later
    turn — and telling the model to ignore rows does not un-bill the tokens."""
    mcp = _server(tmp_path)

    def rows(payload):
        return payload["result"] if isinstance(payload, dict) and "result" in payload else payload

    async def go():
        allrows = rows(_structured(await mcp.call_tool("get_parameters", {})))
        assert len(allrows) >= 3
        some_id = allrows[0]["parameterId"]
        only = rows(
            _structured(await mcp.call_tool("get_parameters", {"parameter_ids": [some_id]}))
        )
        assert [r["parameterId"] for r in only] == [some_id]
        # pick a row with a TRUTHY role: parameters_to_records leaves role None for anything
        # reconcile cannot classify, and role=None would skip the filter and fail for reasons
        # unrelated to the code under test.
        role = next(r["role"] for r in allrows if r.get("role"))
        byrole = rows(_structured(await mcp.call_tool("get_parameters", {"role": role})))
        assert byrole and all(r["role"] == role for r in byrole)
        assert len(byrole) <= len(allrows)
        # an unknown id is an ERROR, not an empty list: [] is indistinguishable from a real
        # empty table, and the agent would act on "that parameter does not exist".
        # An unknown id RAISES. It must not return [] (indistinguishable from a real empty
        # table) and it must not return a ctx.run {ok:false} dict either: this tool declares a
        # LIST return, so FastMCP validates its structured output against an array schema and a
        # dict fails validation with a RuntimeError that KILLS THE RUN — that regression
        # cost 6 of 14 runs in the 2026-08-16 sweep before it was caught. The raise becomes a
        # ToolMessage the model reads and corrects.
        raised = None
        try:
            await mcp.call_tool("get_parameters", {"parameter_ids": ["no_such_parameter"]})
        except Exception as exc:  # fastmcp.exceptions.ToolError
            raised = exc
        assert raised is not None, "an unknown id must not return [] — that reads as 'no such row'"
        assert "unknown parameter_ids" in str(raised)
        # The message must CARRY its remedy, and that is asserted on the string a caller
        # receives — not on the helper, and not on the source. Both earlier guards were written
        # the other way and both passed while the feature was dead: the near-miss hint was
        # computed into a local and never interpolated into the error, and a unit test that
        # called the scorer directly could not see it.
        every_id = sorted(str(r["parameterId"]) for r in allrows)
        if len(every_id) <= 60:
            assert all(p in str(raised) for p in every_id), (
                "the id list is the remedy; truncating it sends the caller back to guessing"
            )
        # A token SWAP is the commonest real miss (`pEGFR_scaling` for `scaling_pEGFR`), so
        # manufacture exactly that from a genuine id and require the message to name it.
        swap = next(
            (
                (("_".join(reversed(p.split("_")))), p)
                for p in every_id
                if "_" in p and "_".join(reversed(p.split("_"))) not in every_id
            ),
            None,
        )
        if swap is not None:
            typo, meant = swap
            swapped_err = None
            try:
                await mcp.call_tool("get_parameters", {"parameter_ids": [typo]})
            except Exception as exc:
                swapped_err = exc
            assert swapped_err is not None
            assert "Did you mean" in str(swapped_err) and meant in str(swapped_err), (
                f"a swapped id must be pointed back at {meant!r}; got: {swapped_err}"
            )
        # ...and the VALID paths still return a list, which is the contract that broke.
        assert isinstance(allrows, list) and isinstance(byrole, list)

    asyncio.run(go())


def test_an_unknown_id_suggests_the_one_the_caller_meant_even_when_the_tokens_are_SWAPPED():
    """Measured on the 2026-08-17 sweep: all 18 unknown-id errors invented the right CONCEPT under
    the wrong convention, and the agent tried BOTH token orderings across runs (`pEGFR_scaling`
    5x, `scaling_pEGFR` 4x). Plain difflib fails exactly there — it compares characters in order,
    so a swap reads as a mismatch and it suggests a DIFFERENT parameter (`EGFR_init`) with
    confidence. That is worse than no hint."""
    # ONE definition of the rule, shared by every surface that reports an unknown id — a second
    # copy in server.py meant a cutoff tuned on one surface silently diverged from the other.
    from petab_mcp.model_edits import closest_symbols

    known = sorted(
        ["Akt_init", "EGFR_init", "S6_init", "k_akt_phos", "scaling_pEGFR", "scaling_pS6", "k_bind"]
    )
    assert closest_symbols("pEGFR_scaling", known)[0] == "scaling_pEGFR"  # the swap
    assert closest_symbols("pAkt_init", known)[0] == "Akt_init"  # ordinary near-miss
    assert closest_symbols("k_totally_unrelated_xyz", known) == []  # no false confidence


def test_the_unknown_id_message_carries_its_own_remedy():
    """Asserted on the MESSAGE the server builds, not on the helper beside it and not on the
    module source. Both of those weaker forms were written first and both passed while the hint
    they were guarding was assigned to a local and never interpolated."""
    from petab_mcp.server import unknown_ids_message

    small = sorted(["Akt_init", "EGFR_init", "S6_init", "scaling_pEGFR", "k_bind"])
    msg = unknown_ids_message(["pEGFR_scaling"], small)
    assert "Did you mean" in msg and "scaling_pEGFR" in msg, "the token swap must be named"
    assert all(p in msg for p in small), "a short table must be listed in full"

    # Over the cap: truncated, but never a dead end.
    big = [f"p{i:03d}_rate" for i in range(80)]
    big_msg = unknown_ids_message(["nope_xyz"], big)
    assert "(+20 more" in big_msg
    assert "call get_parameters() with no parameter_ids" in big_msg
    assert "p079_rate" not in big_msg, "the ceiling must still bind — lucarelli is 118 rows"

    # No false confidence when nothing is close.
    assert "Did you mean" not in unknown_ids_message(["totally_unrelated_xyz"], small)


def test_the_reserved_t_hint_says_use_the_builtin_time_not_a_rename():
    """`t` collides because the caller is expressing TIME, and renaming is the wrong fix: a free
    symbol has nothing binding it to simulation time, so it is estimated as a parameter and the
    time dependence silently disappears. 4 of 16 isensee runs hit this collision on 2026-08-17 and
    the one that took the old 'rename to tau' advice lost ~512 nll to it."""
    from petab_mcp.operations import _compile_fit_failure_message

    msg = _compile_fit_failure_message(KeyError("amici_t"))
    assert "built-in" in msg and "`time`" in msg
    assert "tau" in msg and "free parameter" in msg, "must say WHY renaming fails"
    # the other reserved names keep their plain rename hints
    other = _compile_fit_failure_message(KeyError("amici_k"))
    assert "k_rate" in other and "built-in" not in other
