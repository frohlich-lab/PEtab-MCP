"""Load agent *skills* from the bundled ``_skills/`` directory and expose them over MCP.

A *skill* is a directory that documents how to use this server's tools. It contains
a ``SKILL.md`` file with a small YAML-style frontmatter block (``name`` and
``description``) followed by a markdown body, plus any number of supporting files
(e.g. ``tools.md``) alongside it. The skills ship inside the package::

    src/petab_mcp/_skills/
      petab-model-building/
        SKILL.md
        tools.md

Because ``_skills`` lives next to this module it is found identically from a source
checkout and an installed wheel.

Skills are surfaced to MCP clients two ways:

* **tools** — ``list_skills()`` and ``get_skill(name, file?)``. Tools are consumed
  by essentially every LLM client, so this is the reliable path for the agent to
  discover and read the documentation on demand.
* **resources** — ``petab://skills`` (a JSON index) and ``petab://skills/{name}``
  (a skill's ``SKILL.md``), mirroring the server's other ``petab://`` resources.

The skills directory is resolved from (in order): an explicit ``skills_dir``
argument, the ``PETAB_MCP_SKILLS_DIR`` environment variable, and the bundled
``_skills`` folder next to this module. If none exists the server registers no skills.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from .errors import UnknownEntityError

_SKILL_FILE = "SKILL.md"


@dataclass
class Skill:
    """A single loaded skill: its metadata, ``SKILL.md`` text, and support files."""

    slug: str                       # directory name
    name: str                       # frontmatter `name` (falls back to slug)
    description: str                # frontmatter `description`
    text: str                       # full SKILL.md, as authored
    files: dict[str, str] = field(default_factory=dict)  # relative path -> content

    def matches(self, key: str) -> bool:
        return key in (self.name, self.slug)

    def summary(self) -> dict[str, object]:
        # The resource URI uses the directory `slug` (a safe path segment), not the
        # free-text frontmatter `name`, which may contain spaces or other characters
        # that are invalid in a `petab://skills/{...}` path. `_find` resolves either.
        return {
            "name": self.name,
            "slug": self.slug,
            "description": self.description,
            "files": sorted(self.files),
            "resource": f"petab://skills/{self.slug}",
        }


def _unquote(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        return value[1:-1]
    return value


def parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    """Parse a leading ``---`` frontmatter block into a dict + the remaining body.

    Supports simple ``key: value`` lines and YAML block scalars (``key: >-`` / ``|``)
    whose continuation lines are indented — enough for a ``name``/``description``
    header without pulling in a YAML dependency.
    """
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, text
    meta: dict[str, str] = {}
    end = None
    key: str | None = None
    block: list[str] | None = None
    for i in range(1, len(lines)):
        line = lines[i]
        if line.strip() == "---":
            end = i
            break
        if block is not None and (line.startswith((" ", "\t")) or not line.strip()):
            block.append(line.strip())
            continue
        if block is not None and key is not None:
            meta[key] = " ".join(s for s in block if s).strip()
            key, block = None, None
        if ":" in line:
            raw_key, raw_val = line.split(":", 1)
            key = raw_key.strip()
            val = raw_val.strip()
            if val in (">", ">-", "|", "|-"):  # block scalar: collect indented lines
                block = []
            else:
                meta[key] = _unquote(val)
                key = None
    if block is not None and key is not None:
        meta[key] = " ".join(s for s in block if s).strip()
    body = "\n".join(lines[end + 1:]) if end is not None else text
    return meta, body


def load_skills(skills_dir: str | Path | None) -> list[Skill]:
    """Load every skill directory (one containing a ``SKILL.md``) under ``skills_dir``."""
    skills: list[Skill] = []
    if not skills_dir:
        return skills
    root = Path(skills_dir)
    if not root.is_dir():
        return skills
    for child in sorted(root.iterdir()):
        skill_md = child / _SKILL_FILE
        if not child.is_dir() or not skill_md.is_file():
            continue
        text = skill_md.read_text(encoding="utf-8")
        meta, _body = parse_frontmatter(text)
        base = child.resolve()
        files: dict[str, str] = {}
        for f in sorted(child.rglob("*")):
            if not f.is_file() or f.name == _SKILL_FILE or f.is_symlink():
                continue
            # Never follow a symlink out of the skill tree: a support file's resolved
            # path must stay within the skill directory before we read/expose it.
            try:
                if not f.resolve().is_relative_to(base):
                    continue
                files[f.relative_to(child).as_posix()] = f.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError, ValueError):
                continue  # skip binary/unreadable/out-of-tree support files
        skills.append(
            Skill(
                slug=child.name,
                name=meta.get("name") or child.name,
                description=meta.get("description", ""),
                text=text,
                files=files,
            )
        )
    return skills


def default_skills_dir() -> Path | None:
    """Locate the skills directory (see module docstring).

    Prefers ``PETAB_MCP_SKILLS_DIR`` when it points at a real directory, then the
    ``_skills`` folder bundled next to this module. Returns the first that exists, or
    ``None``. An invalid env override therefore falls back to the bundled skills
    rather than disabling them.
    """
    candidates: list[Path] = []
    env = os.environ.get("PETAB_MCP_SKILLS_DIR")
    if env:
        candidates.append(Path(env))
    candidates.append(Path(__file__).resolve().parent / "_skills")
    for cand in candidates:
        if cand.is_dir():
            return cand
    return None


def register_skills(mcp, skills: list[Skill]) -> list[Skill]:
    """Register ``list_skills``/``get_skill`` tools and ``petab://skills`` resources.

    Returns the registered skills (possibly empty). Registration is skipped entirely
    when there are no skills, so the server surface is unchanged if ``_skills`` is
    absent.
    """
    if not skills:
        return skills

    by_key = {}
    for s in skills:
        by_key[s.name] = s
        by_key.setdefault(s.slug, s)

    def _find(name: str) -> Skill:
        skill = by_key.get(name)
        if skill is None:
            raise UnknownEntityError(
                f"No skill named {name!r}. Available: {sorted(s.name for s in skills)}"
            )
        return skill

    @mcp.tool()
    def list_skills() -> list[dict[str, object]]:
        """List the skills this server exposes (name, description, supporting files).

        Skills document how to use the server's tools. Fetch one with
        `get_skill(name)`; fetch a supporting file with `get_skill(name, file=...)`.
        """
        return [s.summary() for s in skills]

    @mcp.tool()
    def get_skill(name: str, file: str | None = None) -> str:
        """Return a skill's SKILL.md, or one of its supporting files.

        `name` is the skill name from `list_skills`. Pass `file` (e.g. "tools.md")
        to fetch a supporting file listed in that skill's `files`.
        """
        skill = _find(name)
        if file is None:
            return skill.text
        if file not in skill.files:
            raise UnknownEntityError(
                f"Skill {skill.name!r} has no file {file!r}. Files: {sorted(skill.files)}"
            )
        return skill.files[file]

    @mcp.resource("petab://skills")
    def _res_skills_index() -> str:
        return json.dumps([s.summary() for s in skills], indent=2)

    @mcp.resource("petab://skills/{name}")
    def _res_skill(name: str) -> str:
        return _find(name).text

    return skills
