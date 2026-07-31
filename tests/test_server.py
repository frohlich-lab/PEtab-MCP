from __future__ import annotations

import asyncio
import shutil
from pathlib import Path

import libsbml

from petab_mcp.references import StubReferenceResolver
from petab_mcp.server import create_server

YAML_PATH = Path(__file__).parent / "data" / "toy_v1" / "problem.yaml"
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


def _empty_model_server(tmp_path):
    """A server over a problem that ships WITHOUT a model (empty SBML, no model.ant)."""
    work = tmp_path / "empty"
    work.mkdir()
    for name in ("problem.yaml", "conditions.tsv", "measurements.tsv",
                 "observables.tsv", "parameters.tsv"):
        shutil.copy(DATA_DIR / name, work / name)
    doc = libsbml.SBMLDocument(3, 2)
    model = doc.createModel()
    model.setId("conversion_reaction")
    model.setMetaId("conversion_reaction")
    libsbml.writeSBMLToFile(doc, str(work / "conversion_reaction.xml"))
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
        "describe_problem", "list_observables", "list_conditions", "get_measurements",
        "get_model", "get_model_structure",
        "make_changes",
        "get_parameters",
        "get_fit_summary", "get_fit_diagnosis", "get_run_timing", "query_data", "list_data_tables",
        "get_fitting_log",
        "get_provenance", "list_versions", "get_version", "diff_versions",
        "restore_version", "get_progress", "reset",
    }
    assert expected <= names
    # `rollback_to` and `current_version` are NOT tools: restore_version IS rollback_to under the
    # name the going-back workflow uses, and describe_problem already returns current_version /
    # has_unfit_edits / n_pending_edits. One operation, one tool schema.
    assert not ({"rollback_to", "current_version"} & names)
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
    assert not ({"simulate", "evaluate_objective", "optimize", "residual_statistics",
                 "update_parameters", "refit"} & names)
    assert "update_model" not in names and "set_model_parameter" not in names
    assert "list_experiments" not in names
    # the batch tools were deleted (issue #81): edits no longer fit, so batching is obsolete
    assert not ({"begin_batch", "commit_batch", "abort_batch"} & names)
    # changing a parameter's estimate status in-loop is deliberately unsupported — no such tool
    assert not ({"estimate_parameter", "change_parameter_estimated",
                 "fix_parameter", "pin_parameter"} & names)
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
        assert desc["current_version"] == 0

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
        res = _structured(await mcp.call_tool("make_changes", {
            "antimony": _reauthored(["J: B => D; k3 * B;"]),
            "justification": "B converts to D",
            "references": ["PMID:1"],
        }))
        assert res["applied"] is True and res["committed"] is True and res["version"] == 1
        assert _structured(await mcp.call_tool("describe_problem", {}))["current_version"] == 1

        versions = _structured(await mcp.call_tool("list_versions", {}))
        vlist = versions["result"] if isinstance(versions, dict) and "result" in versions else versions
        assert {v["version"] for v in vlist} == {0, 1}

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
        res = _structured(await mcp.call_tool("make_changes", {
            "antimony": _reauthored(["J: B => D; k3 * B;"]), "justification": "B converts to D",
        }))
        assert res["applied"] is True and res["committed"] is True and res["version"] == 1
        assert not ({"reconcile", "validation", "parameter_diff", "initial_conditions",
                     "justification", "references", "operation", "pending_fit",
                     "n_pending_edits", "fit"} & set(res))
        # the fit's own keys ARE there, at the top level and named as get_fit_summary names them
        assert {"fit_status", "neg_log_likelihood", "n_estimated", "bic"} <= set(res)

    asyncio.run(go())


def test_references_optional_but_justification_required(tmp_path):
    mcp = _server(tmp_path)  # auto_fit off -> commit without fitting (no AMICI needed)

    async def go():
        # references are now OPTIONAL for model ops: an edit without any still applies
        ok = _structured(await mcp.call_tool("make_changes", {
            "antimony": _reauthored(["J: B => D; k3 * B;"]), "justification": "B is degraded to D",
        }))
        assert ok["applied"] is True

        # justification is still REQUIRED: an empty one is rejected
        bad = _structured(await mcp.call_tool("make_changes", {
            "antimony": _reauthored(["J2: D => E; k4 * D;"]), "justification": "   ",
            "references": [],
        }))
        assert bad["applied"] is False and "error" in bad

    asyncio.run(go())


def test_rejected_operation_is_logged(tmp_path):
    mcp = _server(tmp_path)

    async def go():
        res = _structured(await mcp.call_tool("make_changes", {
            "parameters": [{"parameter_id": "A_init", "scale": "log10"}], "justification": "x",
        }))
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
        res = _structured(await mcp.call_tool("make_changes", {
            "antimony": _reauthored(["J: B => D; k3 * B;"]), "justification": "x",
            "n_starts": -1,
        }))
        assert res["applied"] is False
        assert "n_starts" in res["error"]
        cur = _structured(await mcp.call_tool("describe_problem", {}))
        assert cur["current_version"] == 0 and cur["has_unfit_edits"] is False

    asyncio.run(go())


def test_a_change_that_changes_nothing_is_rejected(tmp_path):
    # `antimony` and `parameters` are both optional, but not both absent: there would be nothing
    # to score, and the call would still burn a fit.
    mcp = _server(tmp_path)

    async def go():
        res = _structured(await mcp.call_tool("make_changes", {"justification": "nothing"}))
        assert res["applied"] is False
        assert "antimony" in res["error"] and "parameters" in res["error"]
        assert _structured(await mcp.call_tool("describe_problem", {}))["current_version"] == 0

    asyncio.run(go())


def test_query_data_rejects_bad_input(tmp_path):
    mcp = _server(tmp_path)  # arg validation fires before any simulation, so no AMICI needed

    async def go():
        empty = _structured(await mcp.call_tool("query_data", {"sql": ""}))
        assert empty.get("ok") is False and "sql" in empty.get("error", "")
        bad_rows = _structured(await mcp.call_tool(
            "query_data", {"sql": "SELECT 1", "max_rows": 0}))
        assert bad_rows.get("ok") is False and "max_rows" in bad_rows.get("error", "")

    asyncio.run(go())


def test_query_data_logs_sql_on_error(tmp_path):
    mcp = _server(tmp_path)  # querying a missing table fails; the sql must be in get_errors

    async def go():
        res = _structured(await mcp.call_tool(
            "query_data", {"sql": "SELECT * FROM does_not_exist"}))
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
        dropped = _structured(await mcp.call_tool(
            "query_data", {"sql": "DROP TABLE measurements"}))
        assert dropped.get("ok") is not False  # not rejected
        again = _structured(await mcp.call_tool(
            "query_data", {"sql": "SELECT count(*) AS n FROM measurements"}))
        assert again["rows"][0]["n"] == 6

    asyncio.run(go())


def test_query_data_blocks_file_access(tmp_path):
    mcp = _server(tmp_path)  # host file access stays disabled even on the copy

    async def go():
        res = _structured(await mcp.call_tool(
            "query_data", {"sql": "SELECT * FROM read_csv_auto('/etc/hostname')"}))
        assert res.get("ok") is False

    asyncio.run(go())


def test_list_data_tables(tmp_path):
    mcp = _server(tmp_path)  # schema is static; no simulation/AMICI needed

    async def go():
        schema = _structured(await mcp.call_tool("list_data_tables", {}))
        tables = {t["table"] for t in schema["tables"]}
        assert tables == {
            "measurements", "conditions", "observables", "parameters", "simulations",
            "optimization_starts", "optimization_trace",
        }

    asyncio.run(go())


def test_query_data_static_tables(tmp_path):
    # Querying the static PEtab tables works even when the simulation dataframe cannot be
    # produced (no AMICI here): query_data logs and falls through with an empty `simulations`.
    mcp = _server(tmp_path)

    async def go():
        out = _structured(await mcp.call_tool(
            "query_data", {"sql": "SELECT parameterId FROM parameters"}))
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
        res = _structured(await mcp.call_tool("make_changes", {
            "antimony": _reauthored(["J: B => D; k3 * B;"]),
            "parameters": [{"parameter_id": "k1", "lower_bound": 1e-2, "upper_bound": 1e2}],
            "justification": "x", "references": ["PMID:1"]}))
        assert res["applied"] is True and res["version"] == 1
        cur = _structured(await mcp.call_tool("describe_problem", {}))
        assert cur["current_version"] == 1 and cur["has_unfit_edits"] is False
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
        res = _structured(await mcp.call_tool("make_changes", {
            "antimony": _ONE_REACTION_ANTIMONY, "justification": "re-author", "references": ["PMID:1"],
        }))
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
        # re-author to a one-reaction model (v1), then restore the original reversible v0
        await mcp.call_tool("make_changes", {
            "antimony": _ONE_REACTION_ANTIMONY, "justification": "drop r2", "references": ["PMID:1"]})
        struct = _structured(await mcp.call_tool("get_model_structure", {}))
        assert {r["id"] for r in struct["reactions"]} == {"r1"}
        out = _structured(await mcp.call_tool("restore_version", {"version": 0}))
        assert out["rolled_back_to"] == 0
        struct = _structured(await mcp.call_tool("get_model_structure", {}))
        assert {r["id"] for r in struct["reactions"]} == {"r1", "r2"}

    asyncio.run(go())


def test_make_changes_builds_the_first_model_via_tool(tmp_path):
    mcp = _empty_model_server(tmp_path)

    async def go():
        desc = _structured(await mcp.call_tool("describe_problem", {}))
        assert desc["model_authored"] is False
        assert desc["n_reactions"] == 0

        # `antimony` builds the first model when the problem ships model-less
        res = _structured(await mcp.call_tool("make_changes", {
            "antimony": _REVERSIBLE_ANTIMONY,
            "justification": "reversible conversion", "references": ["PMID:1"],
        }))
        assert res["applied"] is True

        desc2 = _structured(await mcp.call_tool("describe_problem", {}))
        assert desc2["model_authored"] is True  # the edit built the working model
        assert desc2["n_reactions"] == 2

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
        # the simulation dataframe is queryable with SQL via query_data; it is generated on
        # demand at cold start (no re-simulation on subsequent calls).
        allrows = _structured(await mcp.call_tool(
            "query_data", {"sql": "SELECT * FROM simulations"}))
        assert allrows["row_count"] == 6
        assert {"residual", "residual_weighted", "noise"} <= set(allrows["columns"])

        # residual breakdowns that the old inspect_fit hardcoded are now plain SQL
        by_obs = _structured(await mcp.call_tool("query_data", {"sql": (
            "SELECT observable_id, count(*) AS n, "
            "sum(residual_weighted*residual_weighted) AS chi2 "
            "FROM simulations GROUP BY observable_id")}))
        assert {r["observable_id"] for r in by_obs["rows"]} == {"obs_B"}
        assert by_obs["rows"][0]["n"] == 6 and by_obs["rows"][0]["chi2"] is not None

        by_cond = _structured(await mcp.call_tool("query_data", {"sql": (
            "SELECT DISTINCT condition_id FROM simulations")}))
        assert {r["condition_id"] for r in by_cond["rows"]} == {"c0"}
        by_time = _structured(await mcp.call_tool("query_data", {"sql": (
            "SELECT DISTINCT time FROM simulations")}))
        assert by_time["row_count"] == 6  # six distinct timepoints

        # worst-fit points: an ORDER BY ... LIMIT instead of a dedicated tool
        worst = _structured(await mcp.call_tool("query_data", {"sql": (
            "SELECT * FROM simulations ORDER BY abs(residual_weighted) DESC LIMIT 3")}))
        assert worst["row_count"] == 3

        # compact metrics-only read (no records) — and the objective is now in describe
        summary = _structured(await mcp.call_tool("get_fit_summary", {}))
        assert "records" not in summary and summary["neg_log_likelihood"] is not None
        desc = _structured(await mcp.call_tool("describe_problem", {}))
        assert desc["neg_log_likelihood"] == pytest.approx(summary["neg_log_likelihood"])

        # max_rows caps the result and flags truncation
        capped = _structured(await mcp.call_tool(
            "query_data", {"sql": "SELECT * FROM simulations", "max_rows": 2}))
        assert capped["row_count"] == 2 and capped["truncated"] is True

        # a bounds change: the result carries the unified fit block nested under `fit`, plus a
        # fitted version.
        res = _structured(await mcp.call_tool("make_changes", {
            "parameters": [{"parameter_id": "k1", "lower_bound": 1e-3, "upper_bound": 1e3}],
            "justification": "widen bounds", "n_starts": 2,
        }))
        assert res["applied"] is True and res["committed"] is True and res["version"] == 1
        assert res["fit_status"] == "fit"
        assert res["neg_log_likelihood"] is not None
        assert res["improvement"]["is_new_best"] is not None
        # the compact residual-structure verdict rides on the fit result (issue #113), so the
        # agent doesn't spend a get_fit_diagnosis call to learn its model's form is incomplete —
        # but the DETAIL behind it deliberately does not, or it would ride on every fit.
        assert set(res["residual_structure"]) == {
            "flagged_observables", "low_power", "summary"}
        assert "per_observable" not in res and "worst_points" not in res
        diag = _structured(await mcp.call_tool("get_fit_diagnosis", {}))
        assert diag["residual_structure"] == res["residual_structure"]
        assert diag["per_observable"] and "worst_points" in diag
        # diff_versions now reports the fit delta between versions
        d = _structured(await mcp.call_tool("diff_versions", {"from_version": 0, "to_version": 1}))
        assert "fit" in d and "neg_log_likelihood_delta" in d["fit"]

        # that fit's pyPESTO multistart is now queryable: results + per-iteration trace
        starts = _structured(await mcp.call_tool("query_data", {"sql": (
            "SELECT count(*) AS n, min(fval) AS best FROM optimization_starts")}))
        assert starts["rows"][0]["n"] == 2 and starts["rows"][0]["best"] is not None
        trace = _structured(await mcp.call_tool("query_data", {"sql": (
            "SELECT count(*) AS n FROM optimization_trace")}))
        assert trace["rows"][0]["n"] >= 1

    asyncio.run(go())
