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


def test_resources_registered(tmp_path):
    resources = asyncio.run(_server(tmp_path).list_resources())
    uris = {str(r.uri) for r in resources}
    assert any("petab://provenance" in u for u in uris)
    assert any("petab://versions" in u for u in uris)
    assert any("model/antimony" in u for u in uris)
