"""Every tool's ERROR path must satisfy its own declared output schema.

The bug this exists for: `get_parameters` declares `-> list[dict]`, so FastMCP validates its
structured output against an ARRAY schema. Returning `ctx.run`'s `{"ok": false, "error": ...}`
dict from its unknown-id branch failed that validation with
`RuntimeError: Invalid structured content ... is not of type 'array'`, which does not degrade —
it KILLS THE RUN. Six of fourteen runs died that way in the 2026-08-16 sneyd/fujita sweep and it
had to be cancelled, with 1130 unit tests green: every one of them called the python function
directly, where the declared return type is never enforced.

So this suite drives the tools THROUGH the MCP surface, on the paths most likely to produce a
non-conforming payload: bad arguments, unknown ids, and no-model state. A tool may reject
(raising ToolError is fine and is what a list-returning tool must do) — what it may not do is
return a payload that violates the schema it published.
"""

from __future__ import annotations

import asyncio
import shutil

from conftest import DATA_DIR

from petab_mcp.references import StubReferenceResolver
from petab_mcp.server import create_server

#: Argument sets aimed at each tool's rejection path. A tool absent here is still covered by the
#: no-argument sweep below.
_BAD_ARGS = {
    "get_parameters": [{"parameter_ids": ["no_such_parameter"]},
                       {"parameter_ids": ["no_such_parameter"], "role": "kinetic"}],
    "get_version": [{"version": 9999}],
    "diff_versions": [{"a": 0, "b": 9999}],
    "restore_version": [{"version": 9999}],
    "query_data": [{"sql": "SELECT * FROM no_such_table"}],
    "get_skill": [{"name": "no-such-skill"}],
    "make_changes": [{"justification": "x", "antimony": "this is not antimony"}],
    "get_measurements": [{"observable_id": "no_such_observable"}],
}


def _server(tmp_path):
    work = tmp_path / "toy"
    work.mkdir()
    for f in DATA_DIR.iterdir():
        shutil.copy(f, work / f.name)
    return create_server(str(work / "problem.yaml"),
                         provenance_dir=str(tmp_path / "prov"),
                         build_root=str(tmp_path / "amici"),
                         resolver=StubReferenceResolver(), auto_fit=False)


def _violations(mcp, calls):
    """(tool, detail) for every call whose RETURNED payload violates the tool's own published
    output schema.

    The check has to VALIDATE, not just watch for an exception: `mcp.call_tool` hands back a
    non-conforming payload without complaint — the enforcement happens client-side, in
    langchain-mcp-adapters, against the schema the tool published. A first version of this file
    only caught raised errors and passed happily when the original bug was reintroduced verbatim.
    So: call, then validate structured_content against `tool.output_schema` exactly as the
    client does. A RAISED rejection is fine (that is how a list-returning tool must refuse); a
    returned payload of the wrong shape is the bug, because it kills the run.
    """
    import jsonschema

    bad = []

    async def go():
        for name, args in calls:
            try:
                result = await mcp.call_tool(name, args)
            except Exception:  # noqa: BLE001 - a clean rejection is acceptable
                continue
            tool = await mcp.get_tool(name)
            schema = getattr(tool, "output_schema", None)
            payload = getattr(result, "structured_content", None)
            if not schema or payload is None:
                continue
            try:
                jsonschema.validate(payload, schema)
            except jsonschema.ValidationError as exc:
                bad.append((name, f"{args} -> {exc.message[:160]}"))

    asyncio.run(go())
    return bad


def test_no_tool_violates_its_own_output_schema_on_a_bad_call(tmp_path):
    mcp = _server(tmp_path)
    calls = [(name, args) for name, arglist in _BAD_ARGS.items() for args in arglist]
    bad = _violations(mcp, calls)
    assert not bad, ("tools returned payloads violating their declared output schema — this "
                     f"class of failure KILLS a run, it does not degrade: {bad}")


def test_no_tool_violates_its_own_output_schema_with_no_arguments(tmp_path):
    """The zero-argument sweep: catches a tool whose empty/no-model payload does not match its
    declared type (e.g. a list-returning read that returns a status dict before a model exists)."""
    mcp = _server(tmp_path)

    async def names():
        return [t.name for t in await mcp.list_tools()]

    bad = _violations(mcp, [(n, {}) for n in asyncio.run(names())])
    assert not bad, f"schema violations on the no-argument path: {bad}"
