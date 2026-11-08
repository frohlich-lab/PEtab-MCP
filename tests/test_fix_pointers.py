"""``create_server(fix_pointers=False)`` sends the facts and none of the text telling the agent
what to change; on, the server sends what it sent before the switch existed (gemot's
``fix_pointers`` arm)."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from fastmcp.exceptions import ToolError

from petab_mcp import identifiability as I
from petab_mcp.messages import fix, fix_or, render
from petab_mcp.references import StubReferenceResolver
from petab_mcp.server import create_server

YAML_PATH = Path(__file__).parent / "data" / "toy_v2" / "problem.yaml"
_BRACKETS = "⟦⟧⟪⟫"


def _server(tmp_path, pointers: bool):
    return create_server(
        str(YAML_PATH),
        provenance_dir=str(tmp_path / f"prov{int(pointers)}"),
        resolver=StubReferenceResolver(),
        auto_fit=False,
        fix_pointers=pointers,
    )


def test_render_keeps_one_side_and_no_bracket():
    text = "fact." + fix(" Send X" + fix(" now") + ".") + " A " + fix_or("is B", "differs") + "."
    assert render(text, True) == "fact. Send X now. A is B."
    assert render(text, False) == "fact. A differs."


@pytest.mark.parametrize("pointers", [True, False])
def test_what_the_agent_reads_first_carries_no_bracket(tmp_path, pointers):
    """Every description, schema and the instructions, as a client lists them: a bracket left in
    would be read as text, on either side."""
    mcp = _server(tmp_path, pointers)
    tools = {t.name: t for t in asyncio.run(mcp.list_tools())}
    sent = [mcp.instructions] + [f"{t.description}{t.parameters}" for t in tools.values()]
    assert not [s for s in sent if any(b in s for b in _BRACKETS)]
    diagnosis = tools["get_fit_diagnosis"].description
    assert "The expansion is local" in diagnosis  # the fact, either way
    assert ("Remedies are MODEL edits" in diagnosis) is pointers
    assert ("a `merge_hints` row's `take`" in tools["make_changes"].description) is pointers


@pytest.mark.parametrize("pointers", [True, False])
def test_a_refusal_keeps_why_and_drops_what_to_send(tmp_path, pointers):
    """The unnamed-reaction refusal: why it is refused stays, the statement to write instead goes."""
    mcp = _server(tmp_path, pointers)
    antimony = YAML_PATH.with_name("model.ant").read_text().replace("r1: ", "", 1)
    res = asyncio.run(
        mcp.call_tool("make_changes", {"antimony": antimony, "justification": "test"})
    ).structured_content
    assert res["applied"] is False and "have no id" in res["error"]
    assert ("Write 'name: A => B; k*A;'" in res["error"]) is pointers
    assert not any(b in str(res) for b in _BRACKETS)


@pytest.mark.parametrize("pointers", [True, False])
def test_a_raised_error_is_rendered_too(tmp_path, pointers):
    """A tool error leaves through every middleware: the brackets must not ride on it."""
    mcp = _server(tmp_path, pointers)

    @mcp.tool
    def boom() -> None:
        raise ToolError("refused." + fix(" Send X instead."))

    with pytest.raises(ToolError) as caught:
        asyncio.run(mcp.call_tool("boom", {}))
    assert str(caught.value) == ("refused. Send X instead." if pointers else "refused.")


def test_off_no_stored_tie_is_offered():
    class _Off:
        fix_pointers = False

    assert I.stored_merge_hints(_Off()) is None
