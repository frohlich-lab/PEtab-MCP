from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from petab_mcp import skills as skills_mod
from petab_mcp.references import StubReferenceResolver
from petab_mcp.server import create_server

REPO_ROOT = Path(__file__).parent.parent
SKILLS_DIR = REPO_ROOT / "src" / "petab_mcp" / "_skills"
YAML_PATH = Path(__file__).parent / "data" / "toy_v2" / "problem.yaml"


def _structured(result):
    return result.structured_content  # FastMCP.call_tool -> ToolResult(content, structured_content)


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
        "---\nname: petab-model-building\ndescription: >-\n  first line\n  second line\n---\nbody\n"
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
    assert "SKILL.md" not in skill.files  # SKILL.md is the body, not a support file
    assert "observation model" in skill.text


def test_default_skills_dir_finds_bundled_skills():
    found = skills_mod.default_skills_dir()
    assert found is not None
    assert Path(found).name == "_skills"
    assert (Path(found) / "petab-model-building" / "SKILL.md").is_file()


def test_load_skills_missing_dir_is_empty():
    assert skills_mod.load_skills(None) == []
    assert skills_mod.load_skills("/no/such/dir") == []


def test_env_override_valid_is_used(monkeypatch, tmp_path):
    custom = tmp_path / "custom"
    (custom / "mine").mkdir(parents=True)
    (custom / "mine" / "SKILL.md").write_text("---\nname: mine\n---\nbody", encoding="utf-8")
    monkeypatch.setenv("PETAB_MCP_SKILLS_DIR", str(custom))
    assert skills_mod.default_skills_dir() == custom


def test_env_override_invalid_falls_back_to_bundled(monkeypatch):
    # An env var pointing at a non-existent path must not disable the bundled skills.
    monkeypatch.setenv("PETAB_MCP_SKILLS_DIR", "/no/such/skills/dir")
    found = skills_mod.default_skills_dir()
    assert found is not None
    assert Path(found).name == "_skills"
    assert (Path(found) / "petab-model-building" / "SKILL.md").is_file()


def test_support_file_symlink_out_of_tree_excluded(tmp_path):
    # A symlinked support file pointing outside the skill tree must not be exposed.
    secret = tmp_path / "secret.txt"
    secret.write_text("TOP SECRET", encoding="utf-8")
    skill = tmp_path / "skills" / "myskill"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text("---\nname: myskill\n---\nbody", encoding="utf-8")
    (skill / "good.md").write_text("fine", encoding="utf-8")
    (skill / "evil.md").symlink_to(secret)

    skills = skills_mod.load_skills(tmp_path / "skills")
    assert len(skills) == 1
    files = skills[0].files
    assert "good.md" in files
    assert "evil.md" not in files
    assert "TOP SECRET" not in "".join(files.values())


def test_summary_resource_uses_slug():
    skill = skills_mod.load_skills(SKILLS_DIR)[0]
    assert skill.summary()["resource"] == f"petab://skills/{skill.slug}"


def test_instructions_mention_skills_only_when_present(tmp_path):
    with_skills = _server(tmp_path)
    assert "get_skill('petab-model-building')" in with_skills.instructions

    without_skills = create_server(
        str(YAML_PATH),
        provenance_dir=str(tmp_path / "prov2"),
        resolver=StubReferenceResolver(),
        skills_dir=str(tmp_path / "empty_skills"),  # does not exist
    )
    assert "list_skills" not in without_skills.instructions


def test_every_skill_the_instructions_name_is_bundled(tmp_path):
    """The instructions point at the skill holding the tools' long reference (#40); a renamed or
    dropped skill must not leave the pointer reading nothing."""
    import re

    bundled = {s.name for s in skills_mod.load_skills(SKILLS_DIR)}
    named = set(re.findall(r"get_skill\('([^']+)'\)", _server(tmp_path).instructions))
    assert "reading-fits" in named and named <= bundled, sorted(named - bundled)


def test_a_route_skill_is_registered_and_named_only_where_its_route_is_offered(tmp_path):
    """The `statements` grammar was taught by refusals alone, 4-14 per run before the first fit
    (gemot round 3, #193); a server without the route neither lists the skill nor names it. The
    text routes' `antimony-authoring` is likewise listed only where they are offered."""
    entity_map = tmp_path / "entities.json"
    entity_map.write_text("{}")
    offered = create_server(
        str(YAML_PATH),
        provenance_dir=str(tmp_path / "prov2"),
        resolver=StubReferenceResolver(),
        skills_dir=str(SKILLS_DIR),
        entity_map=str(entity_map),
    )
    for mcp, shown in ((_server(tmp_path), False), (offered, True)):
        listed = _unwrap(_structured(asyncio.run(mcp.call_tool("list_skills", {}))))
        listed = {s["name"] for s in listed}
        assert ("statements-grammar" in listed) is shown
        assert ("antimony-authoring" in listed) is not shown
        assert ("get_skill('statements-grammar')" in mcp.instructions) is shown


# -- MCP exposure ----------------------------------------------------------
def test_skill_tools_registered(tmp_path):
    mcp = _server(tmp_path)
    names = {t.name for t in asyncio.run(mcp.list_tools())}
    assert {"list_skills", "get_skill"} <= names


def test_skills_resource_registered(tmp_path):
    mcp = _server(tmp_path)
    uris = {str(r.uri) for r in asyncio.run(mcp.list_resources())}
    assert any("petab://skills" in u for u in uris)
    templates = {str(t.uri_template) for t in asyncio.run(mcp.list_resource_templates())}
    assert "petab://skills/{name}" in templates


def test_list_and_get_skill_tools(tmp_path):
    mcp = _server(tmp_path)

    async def go():
        listed = _unwrap(_structured(await mcp.call_tool("list_skills", {})))
        assert any(s["name"] == "petab-model-building" for s in listed)

        body = _unwrap(
            _structured(await mcp.call_tool("get_skill", {"name": "petab-model-building"}))
        )
        assert "PEtab semantics" in body

    asyncio.run(go())


def test_get_skill_unknown_name_errors(tmp_path):
    mcp = _server(tmp_path)

    async def go():
        # UnknownEntityError propagates; FastMCP surfaces it as a ToolError whose
        # message carries our clear, actionable text.
        with pytest.raises(Exception, match="No skill named 'does-not-exist'"):
            await mcp.call_tool("get_skill", {"name": "does-not-exist"})
        with pytest.raises(
            Exception, match="no file 'nope.md'.*without `file` returns its SKILL.md"
        ):
            await mcp.call_tool("get_skill", {"name": "petab-model-building", "file": "nope.md"})

    asyncio.run(go())


def test_get_skill_file_skill_md_is_the_skill(tmp_path):
    """`file="SKILL.md"` was refused with `Files: []` (#86): the agent asked for the one file
    every skill has, by the name `list_skills` describes it by."""
    mcp = _server(tmp_path)

    async def go():
        args = {"name": "petab-model-building"}
        plain = _unwrap(_structured(await mcp.call_tool("get_skill", args)))
        named = _unwrap(_structured(await mcp.call_tool("get_skill", {**args, "file": "SKILL.md"})))
        assert named == plain

    asyncio.run(go())


def test_read_skill_resource(tmp_path):
    mcp = _server(tmp_path)

    async def go():
        contents = (await mcp.read_resource("petab://skills/petab-model-building")).contents
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


# -- the length budget ----------------------------------------------------
def test_the_skills_stay_short():
    """They are read by an AGENT, into a history nothing prunes — so every character is re-sent
    on every subsequent turn. Length is a correctness property here, not a style preference.

    Raised 6000 -> 6500 to fit three silent-failure rules in antimony-authoring
    (time must be the built-in `time`; a reaction id must not be an Antimony keyword; never assign
    a symbol to itself). Each one costs a whole edit-and-fit cycle when it fires and gives an error
    that points somewhere else, so they earn their characters — but the budget is a real cost, so
    keep new entries at rule-plus-consequence length and keep the measured evidence out of the
    skill."""
    for skill_dir in sorted(SKILLS_DIR.iterdir()):
        md = skill_dir / "SKILL.md"
        if md.is_file():
            n = len(md.read_text())
            assert n < 6500, f"{skill_dir.name}/SKILL.md is {n} chars — too long to re-send"


def test_skills_version_moves_when_a_skill_changes(tmp_path):
    """A skill edit must be visible in the version, or the tracking is decorative."""
    from petab_mcp.skills import skills_version

    src = SKILLS_DIR
    before = skills_version(src)

    fake = tmp_path / "_skills" / "petab-model-building"
    fake.mkdir(parents=True)
    (fake / "SKILL.md").write_text(
        (src / "petab-model-building" / "SKILL.md").read_text() + "\nAn extra rule.\n"
    )
    assert skills_version(tmp_path / "_skills") != before
