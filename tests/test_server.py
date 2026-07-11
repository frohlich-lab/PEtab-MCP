from __future__ import annotations

import asyncio
import json

from petab_mcp.server import create_server

# Tables that must never be mutable — assert no tool can write them.
_FORBIDDEN_TOOL_SUBSTRINGS = ("measurement", "observable", "experiment", "condition")
_WRITE_VERBS = ("update", "set", "add", "delete", "remove", "edit")


def _structured(result):
    # FastMCP.call_tool returns (content_blocks, structured_result)
    return result[1]


def test_tools_registered(yaml_path):
    mcp = create_server(yaml_path)
    tools = asyncio.run(mcp.list_tools())
    names = {t.name for t in tools}
    expected = {
        "describe_problem", "list_observables", "list_conditions",
        "get_measurements", "get_model", "update_model",
        "get_model_structure", "set_model_parameter", "validate_problem",
        "get_parameters", "update_parameters", "sync_parameters",
        "simulate", "evaluate_objective", "optimize", "reset",
    }
    assert expected <= names
    assert "list_experiments" not in names  # no experiments in PEtab v1


def test_no_tool_mutates_fixed_tables(yaml_path):
    mcp = create_server(yaml_path)
    names = {t.name for t in asyncio.run(mcp.list_tools())}
    for name in names:
        writes = any(v in name for v in _WRITE_VERBS)
        touches_fixed = any(s in name for s in _FORBIDDEN_TOOL_SUBSTRINGS)
        assert not (writes and touches_fixed), f"tool {name} could mutate fixed data"


def test_describe_and_parameters(yaml_path):
    mcp = create_server(yaml_path)

    async def go():
        desc = _structured(await mcp.call_tool("describe_problem", {}))
        assert desc["n_measurements"] == 6
        assert desc["free_parameters"] == ["k1", "k2"]

        params = _structured(await mcp.call_tool("get_parameters", {}))
        # structured list results are wrapped under "result"
        recs = params["result"] if isinstance(params, dict) and "result" in params else params
        ids = {r["parameterId"] for r in recs}
        assert ids == {"k1", "k2", "sigma_B"}

        upd = _structured(await mcp.call_tool(
            "update_parameters", {"updates": [{"id": "k1", "nominalValue": 0.7}]}
        ))
        assert "k1" in upd["updated"]

    asyncio.run(go())


def test_update_model_guardrail_via_tool(yaml_path):
    mcp = create_server(yaml_path)

    async def go():
        broken = "model conversion_reaction\n species A=1; k1=0.5; r1: A=>; k1*A; end"
        out = _structured(await mcp.call_tool("update_model", {"text": broken}))
        assert out["committed"] is False
        assert out["issues"]

    asyncio.run(go())


def test_resources_registered(yaml_path):
    mcp = create_server(yaml_path)
    resources = asyncio.run(mcp.list_resources())
    uris = {str(r.uri) for r in resources}
    assert any("petab://observables" in u for u in uris)
    assert any("model/antimony" in u for u in uris)
