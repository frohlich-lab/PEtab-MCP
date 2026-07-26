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
    return result[1]  # FastMCP.call_tool -> (content_blocks, structured_result)


def test_tools_registered(tmp_path):
    mcp = _server(tmp_path)
    names = {t.name for t in asyncio.run(mcp.list_tools())}
    expected = {
        "describe_problem", "list_observables", "list_conditions", "get_measurements",
        "get_model", "get_model_structure", "validate_problem",
        "set_model",
        "change_parameters",
        "fit",
        "get_parameters", "sync_parameters",
        "get_fit_summary", "get_fit_diagnosis", "get_run_timing", "query_data", "list_data_tables",
        "get_fitting_log",
        "get_provenance", "list_versions", "get_version", "diff_versions",
        "rollback_to", "restore_version", "current_version", "reset",
    }
    assert expected <= names
    # the incremental structural tools are not registered — structure is edited via set_model
    assert not ({"add_reaction", "change_rate_law", "remove_reaction"} & names)
    # edits are fit-free; `fit` is the single tool that fits — no other manual fit/score/tune tools
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
    # An edit is fit-free (no version); the `fit` tool snapshots the version. auto_fit is off, so
    # `fit` snapshots a (skipped) version without AMICI — enough to exercise the edit→fit→version
    # flow and the event log.
    mcp = _server(tmp_path)

    async def go():
        res = _structured(await mcp.call_tool("set_model", {
            "antimony": _reauthored(["J: B => D; k3 * B;"]),
            "justification": "B converts to D",
            "references": ["PMID:1"],
        }))
        assert res["applied"] is True and res["version"] is None and res["pending_fit"] is True
        # the edit did not create a version
        assert _structured(await mcp.call_tool("current_version", {}))["version"] == 0

        fit = _structured(await mcp.call_tool("fit", {}))
        assert fit["committed"] is True and fit["version"] == 1

        versions = _structured(await mcp.call_tool("list_versions", {}))
        vlist = versions["result"] if isinstance(versions, dict) and "result" in versions else versions
        assert {v["version"] for v in vlist} == {0, 1}

        prov = _structured(await mcp.call_tool("get_provenance", {}))
        plist = prov["result"] if isinstance(prov, dict) and "result" in prov else prov
        assert any(e["type"] == "edit" for e in plist)
        assert any(e["type"] == "fit" for e in plist)

    asyncio.run(go())


def test_initial_conditions_report_via_tool(tmp_path):
    mcp = _server(tmp_path)  # auto_fit off -> commit without fitting (no AMICI needed)

    async def go():
        # a reconciling model op's result carries the per-species initial_conditions report
        res = _structured(await mcp.call_tool("set_model", {
            "antimony": _reauthored(["J: B => D; k3 * B;"]), "justification": "B converts to D",
        }))
        ics = {r["species"]: r for r in res["initial_conditions"]}
        assert {"A", "B", "D"} <= set(ics)
        assert ics["D"]["newly_added"] is True
        assert ics["D"]["t0_source"] == "fixed" and ics["D"]["declared_t0_used"] is True

    asyncio.run(go())


def test_references_optional_but_justification_required(tmp_path):
    mcp = _server(tmp_path)  # auto_fit off -> commit without fitting (no AMICI needed)

    async def go():
        # references are now OPTIONAL for model ops: an edit without any still applies
        ok = _structured(await mcp.call_tool("set_model", {
            "antimony": _reauthored(["J: B => D; k3 * B;"]), "justification": "B is degraded to D",
        }))
        assert ok["applied"] is True

        # justification is still REQUIRED: an empty one is rejected
        bad = _structured(await mcp.call_tool("set_model", {
            "antimony": _reauthored(["J2: D => E; k4 * D;"]), "justification": "   ",
            "references": [],
        }))
        assert bad["applied"] is False and "error" in bad

    asyncio.run(go())


def test_rejected_operation_is_logged(tmp_path):
    mcp = _server(tmp_path)

    async def go():
        res = _structured(await mcp.call_tool("change_parameters", {
            "changes": [{"parameter_id": "A_init", "scale": "log10"}], "justification": "x",
        }))
        assert res["applied"] is False  # A_init lb=0 -> log10 invalid
        errs = _structured(await mcp.call_tool("get_errors", {}))
        elist = errs["result"] if isinstance(errs, dict) and "result" in errs else errs
        assert any(e["tool"] == "change_parameters" for e in elist)

    asyncio.run(go())


def test_negative_n_starts_rejected(tmp_path):
    # n_starts moved from the (now fit-free) edits to the `fit` tool; a negative value is rejected
    # there, before any optimisation, and creates no version.
    mcp = _server(tmp_path)

    async def go():
        res = _structured(await mcp.call_tool("fit", {"n_starts": -1}))
        assert res["committed"] is False
        assert "n_starts" in res["error"]
        cur = _structured(await mcp.call_tool("current_version", {}))
        assert cur["version"] == 0

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


def test_edit_then_fit_via_tools_multi_edit(tmp_path):
    # Several fit-free edits collapse into ONE version at the `fit` call (the old batch use-case,
    # now the default). auto_fit off so `fit` snapshots without AMICI.
    mcp = _server(tmp_path)

    async def go():
        e1 = _structured(await mcp.call_tool("set_model", {
            "antimony": _reauthored(["J: B => D; k3 * B;"]), "justification": "x",
            "references": ["PMID:1"]}))
        assert e1["applied"] is True
        e2 = _structured(await mcp.call_tool("change_parameters", {
            "changes": [{"parameter_id": "k1", "lower_bound": 1e-2, "upper_bound": 1e2}],
            "justification": "tighter"}))
        assert e2["applied"] is True and e2["n_pending_edits"] == 2
        cur = _structured(await mcp.call_tool("current_version", {}))
        assert cur["version"] == 0 and cur["has_unfit_edits"] is True  # no version yet
        fit = _structured(await mcp.call_tool("fit", {}))
        assert fit["version"] == 1
        # the fit event records the two edits it scored
        prov = _structured(await mcp.call_tool("get_provenance", {}))
        plist = prov["result"] if isinstance(prov, dict) and "result" in prov else prov
        fit_ev = [e for e in plist if e["type"] == "fit"][-1]
        assert fit_ev["n_edits"] == 2

    asyncio.run(go())


def test_set_model_reauthors_when_model_exists_via_tool(tmp_path):
    # set_model is NOT refused on an existing model — it re-authors it wholesale as a fit-free edit.
    mcp = _server(tmp_path)

    async def go():
        desc = _structured(await mcp.call_tool("describe_problem", {}))
        assert desc["model_initialized"] is True
        # A wholesale re-author applies as an edit (no version until fit).
        res = _structured(await mcp.call_tool("set_model", {
            "antimony": _ONE_REACTION_ANTIMONY, "justification": "re-author", "references": ["PMID:1"],
        }))
        assert res["applied"] is True and res["version"] is None
        struct = _structured(await mcp.call_tool("get_model_structure", {}))
        assert {r["id"] for r in struct["reactions"]} == {"r1"}  # the edit took effect
        # (the fit-time de-dup no-op for an identical model is covered in test_fit.py)

    asyncio.run(go())


def test_restore_version_tool_registered_and_restores(tmp_path):
    # restore_version -> rollback_to restores a prior version and re-simulates it, so it needs
    # the engine (AMICI). The engine-free restore behaviour is covered in test_dedup.
    import pytest

    pytest.importorskip("amici", reason="AMICI not installed; skipping restore_version run")
    mcp = _server(tmp_path)

    async def go():
        # re-author to a one-reaction model (v1), then restore the original reversible v0
        await mcp.call_tool("set_model", {
            "antimony": _ONE_REACTION_ANTIMONY, "justification": "drop r2", "references": ["PMID:1"]})
        struct = _structured(await mcp.call_tool("get_model_structure", {}))
        assert {r["id"] for r in struct["reactions"]} == {"r1"}
        out = _structured(await mcp.call_tool("restore_version", {"version": 0}))
        assert out["rolled_back_to"] == 0
        struct = _structured(await mcp.call_tool("get_model_structure", {}))
        assert {r["id"] for r in struct["reactions"]} == {"r1", "r2"}

    asyncio.run(go())


def test_set_model_builds_via_tool(tmp_path):
    mcp = _empty_model_server(tmp_path)

    async def go():
        desc = _structured(await mcp.call_tool("describe_problem", {}))
        assert desc["model_initialized"] is False
        assert desc["n_reactions"] == 0

        # set_model builds the first model from Antimony when the problem ships model-less
        res = _structured(await mcp.call_tool("set_model", {
            "antimony": _REVERSIBLE_ANTIMONY,
            "justification": "reversible conversion", "references": ["PMID:1"],
        }))
        assert res["applied"] is True and res["version"] is None

        desc2 = _structured(await mcp.call_tool("describe_problem", {}))
        assert desc2["model_initialized"] is True  # the edit built the working model
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

    pytest.importorskip("amici", reason="AMICI not installed; skipping fit-tool run")
    pytest.importorskip("pypesto", reason="pyPESTO not installed; skipping fit-tool run")
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

        # a fit-free edit followed by `fit`: the fit result carries the unified fit block + a
        # fitted version (n_starts now lives on `fit`, not the edit).
        edit = _structured(await mcp.call_tool("change_parameters", {
            "changes": [{"parameter_id": "k1", "lower_bound": 1e-3, "upper_bound": 1e3}],
            "justification": "widen bounds",
        }))
        assert edit["applied"] is True and edit["version"] is None
        res = _structured(await mcp.call_tool("fit", {"n_starts": 2}))
        assert res["committed"] is True
        assert res["fit_status"] == "fit"
        assert res["neg_log_likelihood"] is not None
        assert res["version"] == 1
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
