"""A one-line notation for `statements` rows, readable and lossless for the model it builds.

INDRA's statement JSON spells every agent out — each modification a dict, each bound partner a
nested agent, every grounding repeated — so a row takes several times the characters it takes
written as::

    r1: A{pY10} phosphorylates B at S10 | two_step kf=k_on kr=k_off kc=k_cat
    r2: A{-pY10, ~C} is degraded | default kf=A_turnover
    v1: A{-mod"P1"} + B -> A{mod"P1"} by E | mass_action k=k_p1
    m1: E phosphorylates S at T20, inhibited by C | inhibited k=k1 Ki=K_i

``<prefix>: <statement> | <policy> <kind>=<binding> ... # [source] text`` — the row's prefix, its
statement, the mechanism, its parameter bindings (:func:`mechanisms.bind`), and the statement's
evidence: one ``source_api`` and text, both optional (``# Author et al. 2000``). The statement
ends with the agents that regulate its rate, each kind once: ``, inhibited by C and D``,
``, activated by A``, ``, modulated by B`` (:data:`statements.REGULATORS`).

An agent is ``NAME{tag, ...}``, ``NAME`` quoted when it is not an identifier. Tags:

* ``pY10`` modified, ``-pY10`` unmodified: the modification's abbreviation (:data:`ABBREV`)
  then its site — a residue and position, a residue alone (``pS``), a position that is not one
  (``mod"G2"``), or nothing (``p``);
* ``kinase:on`` / ``kinase:off``: an activity;
* ``@nucleus`` (``@"plasma membrane"``): a location;
* ``!B`` bound to, ``~B`` not bound to — any agent, tags and all;
* ``mut:V600E``: a mutation (``_`` for a part INDRA leaves unset).

Groundings are not written: an agent takes the ``db_refs`` the entity map gives its name.

Statements read as what they assert — ``X phosphorylates Y at S10``, ``Y is dephosphorylated at
S10``, ``X autophosphorylates at Y10``, ``A binds B binds C``, ``X increases Y`` / ``Y is
produced``, ``X decreases Y`` / ``Y is degraded``, ``A + B -> C + D by X`` (``nothing`` for an
empty side), ``A becomes B``, ``A moves from cytoplasm to nucleus``, ``X activates Y`` (``Y as
kinase`` for an activity other than ``activity``), ``X inhibits Y``. A policy INDRA implements is
written without its ``indra_`` prefix when no entry of that name exists for the type.

What the notation does not cover — another statement type, evidence beyond one source and text —
is written as INDRA's JSON;
the route takes both. :func:`render` returns ``None`` for a row it cannot write losslessly.
"""

from __future__ import annotations

import re
from typing import Any

#: Modification type -> the abbreviation its tags use. Lower case, so the residue that follows
#: (upper case) or a quoted position marks where it ends.
ABBREV = {
    "phosphorylation": "p",
    "ubiquitination": "ub",
    "sumoylation": "sumo",
    "hydroxylation": "hydroxy",
    "acetylation": "ac",
    "glycosylation": "glyco",
    "methylation": "me",
    "farnesylation": "farnesyl",
    "geranylgeranylation": "gerger",
    "palmitoylation": "palm",
    "myristoylation": "myr",
    "ribosylation": "ribo",
    "modification": "mod",
}
MODTYPE = {abbrev: mod_type for mod_type, abbrev in ABBREV.items()}
#: INDRA's activity types (``indra.statements.activity_types``), written out so the notation
#: imports nothing from INDRA.
ACTIVITIES = frozenset(
    {"activity", "kinase", "phosphatase", "catalytic", "gtpbound", "transcription", "gef", "gap"}
)

_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_.]*")
_RESIDUE_SITE = re.compile(r"([A-Z])([0-9]+)?")


class NotationError(ValueError):
    """A row the notation cannot read, with where and why."""


# ------------------------------------------------------------------------------------- render
def _name(text: str) -> str:
    return text if _IDENT.fullmatch(text) else '"' + text.replace('"', '\\"') + '"'


def _site(residue: str | None, position: str | None) -> str:
    if residue is None and position is None:
        return ""
    if residue is not None and re.fullmatch(r"[A-Z]", residue):
        if position is None:
            return residue
        if re.fullmatch(r"[0-9]+", position):
            return residue + position
    if residue is None:
        return '"' + str(position) + '"'
    raise ValueError(f"site {residue!r}/{position!r} has no lossless form")


def agent_text(agent: dict[str, Any]) -> str:
    """``agent`` (INDRA Agent JSON) in the notation. Raises ``ValueError`` for what it lacks."""
    if set(agent) - {"name", "db_refs", "mods", "bound_conditions", "location", "activity",
                     "mutations", "sbo"}:
        raise ValueError(f"agent fields {sorted(agent)}")
    tags = []
    for mod in agent.get("mods") or []:
        if mod["mod_type"] not in ABBREV:
            raise ValueError(f"modification type {mod['mod_type']!r}")
        mark = "" if mod.get("is_modified", True) else "-"
        tags.append(mark + ABBREV[mod["mod_type"]] + _site(mod.get("residue"), mod.get("position")))
    for mut in agent.get("mutations") or []:
        parts = (mut.get("residue_from"), mut.get("position"), mut.get("residue_to"))
        tags.append("mut:" + "".join("_" if p is None else str(p) for p in parts))
    if agent.get("activity"):
        act = agent["activity"]
        tags.append(f"{act['activity_type']}:{'on' if act.get('is_active', True) else 'off'}")
    if agent.get("location"):
        tags.append("@" + _name(agent["location"]))
    for bc in agent.get("bound_conditions") or []:
        tags.append(("!" if bc.get("is_bound", True) else "~") + agent_text(bc["agent"]))
    return _name(agent["name"]) + ("{" + ", ".join(tags) + "}" if tags else "")


def _verb(mod_type: str) -> tuple[str, str]:
    if mod_type == "modification":
        return "modifies", "is modified"
    stem = mod_type[: -len("ion")]  # phosphorylation -> phosphorylat
    return stem + "es", "is " + stem + "ed"


def _modtype(cls_name: str) -> str | None:
    from indra.statements import get_statement_by_name, modclass_to_modtype

    try:
        return modclass_to_modtype.get(get_statement_by_name(cls_name))
    except Exception:  # noqa: BLE001 - not a statement type
        return None


def evidence_text(st: dict[str, Any]) -> str:
    """The statement's evidence as a trailing comment: one source and text at most."""
    evidence = st.get("evidence") or []
    if not evidence:
        return ""
    if len(evidence) > 1 or set(evidence[0]) - {"source_api", "text"}:
        raise ValueError("evidence beyond one source and text")
    ev = evidence[0]
    text = (ev.get("text") or "").replace("\n", " ")
    if text != (ev.get("text") or ""):
        raise ValueError("evidence text over several lines")
    source = f"[{ev['source_api']}] " if ev.get("source_api") else ""
    if not source and text.startswith("["):
        raise ValueError("evidence text that reads as a source")
    return f" # {source}{text}".rstrip()


def statement_text(st: dict[str, Any], derived: dict[str, str] | None = None) -> str:
    """The statement part of a row, its regulators last, each one its rate records as another kind
    (``derived``, :func:`statements.signed`) marked with it, ``C (inhibits, from its rate)``,
    which :func:`parse` reads back as stated. Raises ``ValueError`` for what the notation lacks."""
    from .statements import REGULATORS

    def marked(kind: str, i: int, agent: dict[str, Any]) -> str:
        shown = (derived or {}).get(f"{kind}_{i}")
        return agent_text(agent) + (f" ({_SHOWN[shown]}, from its rate)" if shown else "")

    text = _action_text(st)
    for kind, words in REGULATORS.items():
        if st.get(kind):
            text += f", {words} " + " and ".join(
                marked(kind, i, a) for i, a in enumerate(st[kind])
            )
    return text


#: How the kind a regulator's rate records is marked in a row (:func:`statement_text`).
_SHOWN = {"inhibited_by": "inhibits", "activated_by": "activates", "modulated_by": "modulates"}


def _action_text(st: dict[str, Any]) -> str:
    t = st["type"]
    at = ""
    if st.get("residue") is not None or st.get("position") is not None:
        at = " at " + _site(st.get("residue"), st.get("position"))
    mod_type = _modtype(t)
    if mod_type and t not in ("Autophosphorylation", "Transphosphorylation"):
        active, passive = _verb(mod_type)
        if st.get("enz"):
            return f"{agent_text(st['enz'])} {active} {agent_text(st['sub'])}{at}"
        return f"{agent_text(st['sub'])} {passive}{at}"
    if t in ("Autophosphorylation", "Transphosphorylation"):
        return f"{agent_text(st['enz'])} {t.lower()[:-3]}es{at}"
    if t == "Complex":
        return " binds ".join(agent_text(a) for a in st["members"])
    if t in ("IncreaseAmount", "DecreaseAmount"):
        up = t == "IncreaseAmount"
        if st.get("subj"):
            verb = "increases" if up else "decreases"
            return f"{agent_text(st['subj'])} {verb} {agent_text(st['obj'])}"
        return f"{agent_text(st['obj'])} {'is produced' if up else 'is degraded'}"
    if t == "Conversion":
        left = " + ".join(agent_text(a) for a in st.get("obj_from") or []) or "nothing"
        right = " + ".join(agent_text(a) for a in st.get("obj_to") or []) or "nothing"
        by = f" by {agent_text(st['subj'])}" if st.get("subj") else ""
        return f"{left} -> {right}{by}"
    if t == "PopulationTransition":
        return f"{agent_text(st['source'])} becomes {agent_text(st['target'])}"
    if t == "Translocation":
        text = f"{agent_text(st['agent'])} moves"
        if st.get("from_location"):
            text += " from " + _name(st["from_location"])
        if st.get("to_location"):
            text += " to " + _name(st["to_location"])
        return text
    if t in ("Activation", "Inhibition"):
        verb = "activates" if t == "Activation" else "inhibits"
        activity = st.get("obj_activity", "activity")
        suffix = "" if activity == "activity" else f" as {activity}"
        return f"{agent_text(st['subj'])} {verb} {agent_text(st['obj'])}{suffix}"
    raise ValueError(f"statement type {t}")


def _value(value: Any) -> str:
    if isinstance(value, bool):
        raise ValueError("boolean binding")
    if isinstance(value, (int, float)):
        return repr(value)
    text = str(value)
    if re.fullmatch(r"[^\s\"|]+", text) and not _number(text):
        return text
    return '"' + text + '"'


def _number(text: str) -> bool:
    try:
        float(text)
    except ValueError:
        return False
    return True


def render(
    row: dict[str, Any], library: dict | None = None, derived: dict[str, str] | None = None
) -> str | None:
    """``row`` (``{statement, policy, prefix, parameters?}``) as one line, or ``None`` when the
    notation cannot write it losslessly (keep it as JSON). ``derived`` marks the regulators whose
    kind the row's rate proves (:func:`statement_text`)."""
    try:
        policy = row["policy"]
        short = policy.removeprefix("indra_")
        key = (row["statement"]["type"], short)
        if short != policy and (library is None or key not in library):
            policy = short
        tail = [policy]
        tail += [f"{k}={_value(v)}" for k, v in (row.get("parameters") or {}).items()]
        if set(row) - {"statement", "policy", "prefix", "parameters"}:
            return None
        if not _IDENT.fullmatch(row["prefix"]):
            return None
        line = f"{row['prefix']}: {statement_text(row['statement'], derived)} | {' '.join(tail)}"
        return line + evidence_text(row["statement"])
    except ValueError:
        return None


# -------------------------------------------------------------------------------------- parse
class _Reader:
    """A cursor over one row's text."""

    def __init__(self, text: str, groundings: dict[str, dict]):
        self.text, self.pos, self.groundings = text, 0, groundings

    def fail(self, what: str):
        raise NotationError(f"{what} at column {self.pos + 1}: {self.text!r}")

    def space(self) -> None:
        while self.pos < len(self.text) and self.text[self.pos].isspace():
            self.pos += 1

    def done(self) -> bool:
        self.space()
        return self.pos >= len(self.text)

    def peek(self, word: str) -> bool:
        self.space()
        if not self.text.startswith(word, self.pos):
            return False
        end = self.pos + len(word)
        return end >= len(self.text) or not (word[-1].isalnum() and self.text[end].isalnum())

    def take(self, word: str) -> bool:
        if self.peek(word):
            self.pos += len(word)
            return True
        return False

    def expect(self, word: str) -> None:
        if not self.take(word):
            self.fail(f"expected {word!r}")

    def name(self) -> str:
        self.space()
        if self.pos < len(self.text) and self.text[self.pos] == '"':
            end = self.pos + 1
            while end < len(self.text) and (self.text[end] != '"' or self.text[end - 1] == "\\"):
                end += 1
            if end >= len(self.text):
                self.fail("unclosed quote")
            value = self.text[self.pos + 1 : end].replace('\\"', '"')
            self.pos = end + 1
            return value
        m = _IDENT.match(self.text, self.pos)
        if not m:
            self.fail("expected a name")
        self.pos = m.end()
        return m.group(0)

    def site(self) -> tuple[str | None, str | None]:
        if self.pos < len(self.text) and self.text[self.pos] == '"':
            return None, self.name()
        m = _RESIDUE_SITE.match(self.text, self.pos)
        if m and not (m.end() < len(self.text) and self.text[m.end()].isalnum()):
            self.pos = m.end()
            return m.group(1), m.group(2)
        return None, None

    def agent(self) -> dict[str, Any]:
        name = self.name()
        agent: dict[str, Any] = {"name": name}
        if name in self.groundings:
            agent["db_refs"] = dict(self.groundings[name])
        if self.pos < len(self.text) and self.text[self.pos] == "{":
            self.pos += 1
            while True:
                self.space()
                self.tag(agent)
                self.space()
                if self.take("}"):
                    break
                self.expect(",")
        return agent

    def tag(self, agent: dict[str, Any]) -> None:
        c = self.text[self.pos : self.pos + 1]
        if c in "!~":
            self.pos += 1
            agent.setdefault("bound_conditions", []).append(
                {"agent": self.agent(), "is_bound": c == "!"}
            )
        elif c == "@":
            self.pos += 1
            agent["location"] = self.name()
        elif self.text.startswith("mut:", self.pos):
            self.pos += 4
            m = re.match(r"([A-Z_])([0-9_]+|_)([A-Z_])", self.text[self.pos :])
            if not m:
                self.fail("expected a mutation like V600E")
            self.pos += m.end()
            parts = [None if p == "_" else p for p in m.groups()]
            agent.setdefault("mutations", []).append(
                {"residue_from": parts[0], "position": parts[1], "residue_to": parts[2]}
            )
        else:
            unmodified = c == "-"
            if unmodified:
                self.pos += 1
            m = re.match(r"[a-z]+", self.text[self.pos :])
            if not m:
                self.fail("expected a tag")
            word = m.group(0)
            after = self.pos + m.end()
            if not unmodified and self.text.startswith(":", after):
                self.pos = after + 1
                state = self.name()
                if state not in ("on", "off"):
                    self.fail("an activity is :on or :off")
                if word not in ACTIVITIES:
                    # `A{mod:on}` is how a scheme writes a molecule's modification site, and
                    # read here as an activity named "mod" it built a species no rule makes.
                    self.fail(
                        f"unknown activity {word!r}: an activity is one of {sorted(ACTIVITIES)}; a "
                        "modification is a tag like pS10 (modified) or -pS10 (unmodified)"
                    )
                agent["activity"] = {"activity_type": word, "is_active": state == "on"}
                return
            if word not in MODTYPE:
                self.fail(f"unknown modification {word!r}")
            self.pos = after
            residue, position = self.site()
            mod = {"mod_type": MODTYPE[word], "residue": residue, "position": position,
                   "is_modified": not unmodified}
            agent.setdefault("mods", []).append(mod)

    def agents(self, separator: str) -> list[dict[str, Any]]:
        if self.take("nothing"):
            return []
        out = [self.agent()]
        while self.take(separator):
            out.append(self.agent())
        return out


def _active_verb(word: str) -> str | None:
    """The statement class an active modification verb names (``phosphorylates``)."""
    from indra.statements import modtype_to_modclass

    if word == "modifies":
        return "Modification"
    if word.endswith("es"):
        cls = modtype_to_modclass.get(word[:-2] + "ion")
        return cls.__name__ if cls else None
    return None


def _passive_verb(word: str) -> str | None:
    from indra.statements import modtype_to_modclass

    if word == "modified":
        return "Modification"
    if word.endswith("ed"):
        cls = modtype_to_modclass.get(word[:-2] + "ion")
        return cls.__name__ if cls else None
    return None


def _at(r: _Reader, st: dict[str, Any]) -> None:
    if r.take("at"):
        r.space()
        st["residue"], st["position"] = r.site()
        if st["residue"] is None and st["position"] is None:
            r.fail("expected a site")


def statement(r: _Reader) -> dict[str, Any]:
    start = r.pos
    head = r.text[start:]
    if " -> " in f" {head} " or head.lstrip().startswith("->"):
        left = r.agents("+")
        r.expect("->")
        right = r.agents("+")
        st = {"type": "Conversion", "subj": None, "obj_from": left, "obj_to": right}
        if r.take("by"):
            st["subj"] = r.agent()
        return st
    first = r.agent()
    r.space()
    if r.take("binds"):
        members = [first, r.agent()]
        while r.take("binds"):
            members.append(r.agent())
        return {"type": "Complex", "members": members}
    if r.take("becomes"):
        return {"type": "PopulationTransition", "source": first, "target": r.agent()}
    # `translocates` is a word agents reach for, so it reads as `moves`.
    if r.take("moves") or r.take("translocates"):
        st = {"type": "Translocation", "agent": first, "from_location": None, "to_location": None}
        if r.take("from"):
            st["from_location"] = r.name()
        if r.take("to"):
            st["to_location"] = r.name()
        return st
    if r.take("is"):
        word = r.name()
        if word in ("produced", "degraded"):
            kind = "IncreaseAmount" if word == "produced" else "DecreaseAmount"
            return {"type": kind, "subj": None, "obj": first}
        cls = _passive_verb(word)
        if cls is None:
            r.fail(f"unknown verb 'is {word}'")
        st = {"type": cls, "enz": None, "sub": first, "residue": None, "position": None}
        _at(r, st)
        return st
    word = r.name()
    if word in ("increases", "decreases"):
        kind = "IncreaseAmount" if word == "increases" else "DecreaseAmount"
        return {"type": kind, "subj": first, "obj": r.agent()}
    if word in ("activates", "inhibits"):
        st = {"type": "Activation" if word == "activates" else "Inhibition", "subj": first,
              "obj": r.agent(), "obj_activity": "activity"}
        if r.take("as"):
            st["obj_activity"] = r.name()
        return st
    if word in ("autophosphorylates", "transphosphorylates"):
        st = {"type": word[:-2].capitalize() + "ion", "enz": first, "residue": None,
              "position": None}
        _at(r, st)
        return st
    cls = _active_verb(word)
    if cls is None:
        r.fail(f"unknown verb {word!r}")
    st = {"type": cls, "enz": first, "sub": r.agent(), "residue": None, "position": None}
    _at(r, st)
    return st


def _split_tail(text: str) -> tuple[str, str]:
    """Split ``text`` at the ``|`` outside quotes and braces."""
    depth, quoted = 0, False
    for i, c in enumerate(text):
        if c == '"' and (i == 0 or text[i - 1] != "\\"):
            quoted = not quoted
        elif not quoted and c == "{":
            depth += 1
        elif not quoted and c == "}":
            depth -= 1
        elif not quoted and depth == 0 and c == "|":
            return text[:i], text[i + 1 :]
    raise NotationError(f"no '|' before the mechanism: {text!r}")


def _split_comment(tail: str) -> tuple[str, str | None]:
    """Split ``tail`` at the ``#`` outside quotes: the evidence comment, ``None`` without one."""
    quoted = False
    for i, c in enumerate(tail):
        if c == '"' and (i == 0 or tail[i - 1] != "\\"):
            quoted = not quoted
        elif c == "#" and not quoted:
            return tail[:i], tail[i + 1 :].strip()
    return tail, None


def parse(
    text: str,
    groundings: dict[str, dict] | None = None,
    library: dict | None = None,
) -> dict[str, Any]:
    """One row from ``text``: ``{statement (INDRA JSON), policy, prefix, parameters?}``.
    ``groundings`` (agent name to ``db_refs``) ground the agents; ``library`` resolves a policy
    written without its ``indra_`` prefix."""
    from .statements import REGULATORS

    groundings = groundings or {}
    m = re.match(r"\s*([A-Za-z_][A-Za-z0-9_]*)\s*:\s*", text)
    if not m:
        raise NotationError(f"a row starts with its prefix and ':': {text!r}")
    head, tail = _split_tail(text[m.end() :])
    tail, comment = _split_comment(tail)
    r = _Reader(head, groundings)
    st = statement(r)
    while r.take(","):
        kind = next((k for k, words in REGULATORS.items() if r.take(words)), None)
        if kind is None or kind in st:
            r.fail(f"expected one of {', '.join(map(repr, REGULATORS.values()))}, each once")
        st[kind] = [r.agent()]
        while True:
            # What get_model marks a recorded kind with (`statement_text`): the row states it as is.
            if r.take("("):
                if not any(r.take(f"{words}, from its rate)") for words in _SHOWN.values()):
                    r.fail("expected `(inhibits, from its rate)`, `(activates, ...)` or "
                           "`(modulates, ...)`")
            if not r.take("and"):
                break
            st[kind].append(r.agent())
    if not r.done():
        r.fail("unexpected text")
    if comment is not None:
        ev: dict[str, Any] = {}
        source = re.match(r"\[([^\]]*)\]\s?", comment)
        if source:
            ev["source_api"], comment = source.group(1), comment[source.end() :]
        if comment:
            ev["text"] = comment
        st["evidence"] = [ev]
    t = _Reader(tail, groundings)
    policy = t.name()
    if library is not None and (st["type"], policy) not in library:
        if any(p == "indra_" + policy for _, p in library):
            policy = "indra_" + policy
    row: dict[str, Any] = {"statement": st, "policy": policy, "prefix": m.group(1)}
    while not t.done():
        key = t.name()
        if t.peek(":="):
            t.fail(
                f"`{key}:=` names no regulator: the agents that regulate a row's rate are in its "
                "statement, `<statement>, inhibited by C` (activated by, modulated by), read in "
                "its mechanism's rate as `inhibited_by`"
            )
        t.expect("=")
        t.space()
        if t.text[t.pos : t.pos + 1] == '"':
            value: Any = t.name()
        else:
            end = t.pos
            while end < len(t.text) and not t.text[end].isspace():
                end += 1
            raw, t.pos = t.text[t.pos : end], end
            value = float(raw) if _number(raw) else raw
        row.setdefault("parameters", {})[key] = value
    return row


def parse_agent(text: str, groundings: dict[str, dict] | None = None) -> dict[str, Any]:
    """One agent (INDRA Agent JSON) from its notation."""
    r = _Reader(text, groundings or {})
    agent = r.agent()
    if not r.done():
        r.fail("unexpected text after the agent")
    return agent
