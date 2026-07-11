from __future__ import annotations

import asyncio
from pathlib import Path

from petab_mcp.references import StubReferenceResolver
from petab_mcp.server import create_server

YAML_PATH = Path(__file__).parent / "data" / "toy_v1" / "problem.yaml"

# Tables that must never be mutable — assert no tool can write them.
_FORBIDDEN_TOOL_SUBSTRINGS = ("measurement", "observable", "condition")
_WRITE_VERBS = ("update", "set", "add", "delete", "remove", "edit")


def _server(tmp_path):
    return create_server(
        str(YAML_PATH),
        provenance_dir=str(tmp_path / "prov"),
        resolver=StubReferenceResolver(),
    )


def _structured(result):
    return result[1]  # FastMCP.call_tool -> (content_blocks, structured_result)


def test_tools_registered(tmp_path):
    mcp = _server(tmp_path)
    names = {t.name for t in asyncio.run(mcp.list_tools())}
    expected = {
        "describe_problem", "list_observables", "list_conditions", "get_measurements",
        "get_model", "get_model_structure", "validate_problem",
        "add_reaction", "change_rate_law", "remove_reaction",
        "get_parameters", "update_parameters", "sync_parameters",
        "simulate", "evaluate_objective", "optimize",
        "inspect_fit", "residual_statistics",
        "get_provenance", "list_versions", "get_version", "diff_versions",
        "rollback_to", "current_version", "reset",
    }
    assert expected <= names
    assert "update_model" not in names and "set_model_parameter" not in names
    assert "list_experiments" not in names


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

        upd = _structured(await mcp.call_tool(
            "update_parameters", {"updates": [{"id": "k1", "nominalValue": 0.7}]}
        ))
        assert "k1" in upd["updated"]

    asyncio.run(go())


def test_add_reaction_and_versions_via_tools(tmp_path):
    mcp = _server(tmp_path)

    async def go():
        res = _structured(await mcp.call_tool("add_reaction", {
            "reaction": "J: B -> D; k3*B",
            "justification": "B converts to D",
            "references": ["PMID:1"],
        }))
        assert res["committed"] is True and res["version"] == 1

        versions = _structured(await mcp.call_tool("list_versions", {}))
        vlist = versions["result"] if isinstance(versions, dict) and "result" in versions else versions
        assert {v["version"] for v in vlist} == {0, 1}

        prov = _structured(await mcp.call_tool("get_provenance", {}))
        plist = prov["result"] if isinstance(prov, dict) and "result" in prov else prov
        assert any(e["type"] == "alteration" for e in plist)

    asyncio.run(go())


def test_operation_missing_references_reports_error(tmp_path):
    mcp = _server(tmp_path)

    async def go():
        res = _structured(await mcp.call_tool("add_reaction", {
            "reaction": "J: B -> D; k3*B", "justification": "x", "references": [],
        }))
        assert res["committed"] is False
        assert "error" in res

    asyncio.run(go())


def test_rejected_operation_is_logged(tmp_path):
    mcp = _server(tmp_path)

    async def go():
        res = _structured(await mcp.call_tool("change_parameter_scale", {
            "parameter_id": "A_init", "scale": "log10", "justification": "x",
        }))
        assert res["committed"] is False  # A_init lb=0 -> log10 invalid
        errs = _structured(await mcp.call_tool("get_errors", {}))
        elist = errs["result"] if isinstance(errs, dict) and "result" in errs else errs
        assert any(e["tool"] == "change_parameter_scale" for e in elist)

    asyncio.run(go())


def test_batch_tools(tmp_path):
    mcp = _server(tmp_path)

    async def go():
        await mcp.call_tool("begin_batch", {"label": "b"})
        await mcp.call_tool("add_reaction", {
            "reaction": "J: B -> D; k3*B", "justification": "x", "references": ["PMID:1"]})
        cur = _structured(await mcp.call_tool("current_version", {}))
        assert cur["batch_active"] is True and cur["version"] == 0  # no version yet
        cb = _structured(await mcp.call_tool("commit_batch", {}))
        assert cb["version"] == 1

    asyncio.run(go())


def test_resources_registered(tmp_path):
    resources = asyncio.run(_server(tmp_path).list_resources())
    uris = {str(r.uri) for r in resources}
    assert any("petab://provenance" in u for u in uris)
    assert any("petab://versions" in u for u in uris)
    assert any("petab://fit" in u for u in uris)
    assert any("model/antimony" in u for u in uris)


def test_fit_tools(tmp_path):
    import pytest

    pytest.importorskip("amici", reason="AMICI not installed; skipping fit-tool run")
    mcp = _server(tmp_path)

    async def go():
        report = _structured(await mcp.call_tool("inspect_fit", {"top_n": 3}))
        assert report["summary"]["n"] == 6
        assert report["chi2"] is not None
        assert len(report["worst_points"]) <= 3
        # residuals broken down across each cross section
        stats = report["statistics"]
        assert {o["observable_id"] for o in stats["by_observable"]} == {"obs_B"}
        assert {c["condition_id"] for c in stats["by_condition"]} == {"c0"}
        assert len(stats["by_time"]) == 6  # six distinct timepoints

        by_obs = _structured(await mcp.call_tool("residual_statistics", {"group_by": "observable"}))
        assert "by_observable" in by_obs and "by_condition" not in by_obs

        exp = _structured(await mcp.call_tool("residual_statistics", {"group_by": "experiment"}))
        assert "by_condition" in exp  # 'experiment' aliases to condition

        bad = _structured(await mcp.call_tool("residual_statistics", {"group_by": "nope"}))
        assert bad["ok"] is False

    asyncio.run(go())
