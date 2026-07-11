from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from petab_mcp import skills as skills_mod
from petab_mcp.references import StubReferenceResolver
from petab_mcp.server import create_server

REPO_ROOT = Path(__file__).parent.parent
SKILLS_DIR = REPO_ROOT / ".skills"
YAML_PATH = Path(__file__).parent / "data" / "toy_v1" / "problem.yaml"


def _structured(result):
    return result[1]  # FastMCP.call_tool -> (content_blocks, structured_result)


def _unwrap(result):
    return result["result"] if isinstance(result, dict) and "result" in result else result


def _server(tmp_path):
    return create_server(
        str(YAML_PATH),
        provenance_dir=str(tmp_path / "prov"),
        resolver=StubReferenceResolver(),
        skills_dir=str(SKILLS_DIR),
    )


# -- frontmatter parsing ---------------------------------------------------
def test_parse_frontmatter_simple():
    meta, body = skills_mod.parse_frontmatter(
        "---\nname: foo\ndescription: a bar\n---\n# Title\ntext\n"
    )
    assert meta == {"name": "foo", "description": "a bar"}
    assert body.startswith("# Title")


def test_parse_frontmatter_folded_block_scalar():
    text = (
        "---\n"
        "name: petab-model-building\n"
        "description: >-\n"
        "  first line\n"
        "  second line\n"
        "---\n"
        "body\n"
    )
    meta, body = skills_mod.parse_frontmatter(text)
    assert meta["name"] == "petab-model-building"
    assert meta["description"] == "first line second line"
    assert body.strip() == "body"


def test_parse_frontmatter_no_header():
    meta, body = skills_mod.parse_frontmatter("# Just markdown\n")
    assert meta == {}
    assert body == "# Just markdown\n"


# -- loading ---------------------------------------------------------------
def test_bundled_skill_loads():
    skills = skills_mod.load_skills(SKILLS_DIR)
    names = {s.name for s in skills}
    assert "petab-model-building" in names
    skill = next(s for s in skills if s.name == "petab-model-building")
    assert skill.description  # non-empty description from frontmatter
    assert "tools.md" in skill.files
    assert "SKILL.md" not in skill.files  # SKILL.md is the body, not a support file
    assert "add_reaction" in skill.text


def test_default_skills_dir_finds_repo_skills():
    found = skills_mod.default_skills_dir()
    assert found is not None
    assert (Path(found) / "petab-model-building" / "SKILL.md").is_file()


def test_load_skills_missing_dir_is_empty():
    assert skills_mod.load_skills(None) == []
    assert skills_mod.load_skills("/no/such/dir") == []


# -- MCP exposure ----------------------------------------------------------
def test_skill_tools_registered(tmp_path):
    mcp = _server(tmp_path)
    names = {t.name for t in asyncio.run(mcp.list_tools())}
    assert {"list_skills", "get_skill"} <= names


def test_skills_resource_registered(tmp_path):
    mcp = _server(tmp_path)
    uris = {str(r.uri) for r in asyncio.run(mcp.list_resources())}
    assert any("petab://skills" in u for u in uris)
    templates = {str(t.uriTemplate) for t in asyncio.run(mcp.list_resource_templates())}
    assert "petab://skills/{name}" in templates


def test_list_and_get_skill_tools(tmp_path):
    mcp = _server(tmp_path)

    async def go():
        listed = _unwrap(_structured(await mcp.call_tool("list_skills", {})))
        assert any(s["name"] == "petab-model-building" for s in listed)

        body = _unwrap(_structured(await mcp.call_tool(
            "get_skill", {"name": "petab-model-building"}
        )))
        assert "Working a PEtab problem" in body

        support = _unwrap(_structured(await mcp.call_tool(
            "get_skill", {"name": "petab-model-building", "file": "tools.md"}
        )))
        assert "tool & resource catalog" in support

    asyncio.run(go())


def test_get_skill_unknown_name_errors(tmp_path):
    mcp = _server(tmp_path)

    async def go():
        # UnknownEntityError propagates; FastMCP surfaces it as a ToolError whose
        # message carries our clear, actionable text.
        with pytest.raises(Exception, match="No skill named 'does-not-exist'"):
            await mcp.call_tool("get_skill", {"name": "does-not-exist"})
        with pytest.raises(Exception, match="no file 'nope.md'"):
            await mcp.call_tool(
                "get_skill", {"name": "petab-model-building", "file": "nope.md"}
            )

    asyncio.run(go())


def test_read_skill_resource(tmp_path):
    mcp = _server(tmp_path)

    async def go():
        contents = list(await mcp.read_resource("petab://skills/petab-model-building"))
        assert "petab-model-building" in contents[0].content

    asyncio.run(go())


def test_no_skills_when_dir_absent(tmp_path):
    mcp = create_server(
        str(YAML_PATH),
        provenance_dir=str(tmp_path / "prov"),
        resolver=StubReferenceResolver(),
        skills_dir=str(tmp_path / "empty_skills"),  # does not exist
    )
    names = {t.name for t in asyncio.run(mcp.list_tools())}
    assert "list_skills" not in names and "get_skill" not in names
