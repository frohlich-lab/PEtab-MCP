"""A one-line notation for `statements` rows and their bindings, readable and lossless for the
model they build.

INDRA's statement JSON spells every agent out — each modification a dict, each bound partner a
nested agent, every grounding repeated — so a row takes several times the characters it takes
written as::

    r1: A{pY10} phosphorylates B at S10
    r2: A{-pY10, ~C} is degraded
    v1: A{-mod"P1"} + B -> A{mod"P1"} by E
    m1: E phosphorylates S at T20, inhibited by C # [curated] Doe et al. 2000

``<prefix>: <statement> # [source] text`` — the row's prefix, its statement, and the statement's
evidence: one ``source_api`` and text, both optional (``# Author et al. 2000``). The statement
ends with the agents that regulate its rate, each kind once: ``, inhibited by C and D``,
``, activated by A``, ``, modulated by B`` (:data:`statements.REGULATORS`). That is the claim, in
words. The maths behind it, the mechanism and its parameter bindings (:func:`mechanisms.bind`), is
the row's binding, ``<policy> <kind>=<binding> ...``, given apart under its prefix::

    r1: two_step kf=k_on kr=k_off kc=k_cat
    m1: inhibited k=k1 Ki=K_i

so a statement is never read for its maths, nor a binding for its claim. A row written with its
binding inline, after a ``|`` (how a row was written before the two were apart), is refused saying
how it splits (:class:`InlineBinding`): one notation, not two.

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
kinase`` for an activity other than ``activity``), ``X inhibits Y``, ``X is a GEF for Y``, ``X is
a GAP for Y``. A policy INDRA implements is
written without its ``indra_`` prefix when no entry of that name exists for the type.

What the notation does not cover — another statement type, evidence beyond one source and text —
is written as INDRA's JSON, and a binding it cannot write as ``{policy, parameters}``; the route
takes both. :func:`render` and :func:`render_binding` return ``None`` for what they cannot write
losslessly.
"""

from __future__ import annotations

import json
import re
from typing import Any

from indra.assemblers.english.assembler import statement_passive_verb, statement_present_verb
from indra.statements import (
    Autophosphorylation,
    Transphosphorylation,
    activity_types,
    get_statement_by_name,
    modclass_to_modtype,
    modtype_to_modclass,
)

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

_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_.]*")
#: A row's prefix and the ``:`` after it: an identifier without dots, as a prefix names the row's
#: parameters (``<prefix>_kf``).
_ROW_PREFIX = re.compile(r"\s*([A-Za-z_][A-Za-z0-9_]*)\s*:\s*")
_RESIDUE_SITE = re.compile(r"([A-Z])([0-9]+)?")
#: A control character but tab and newline, which no row, prefix or name holds: a client's stray
#: ``\r`` (``r\r_act``) prints as the name without it.
_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")


class NotationError(ValueError):
    """A row the notation cannot read, with where and why."""


class InlineBinding(NotationError):
    """A row written with its binding after a ``|``, refused with the row and binding it splits
    into: the notation the route took before the two were given apart (module docstring). Its
    :meth:`advice` says where the binding goes for the call that sent it."""

    def __init__(self, prefix: str, statement: str, binding: str):
        self.prefix, self.statement, self.binding = prefix, statement, binding
        super().__init__(self.advice())

    def advice(self, op: str | None = None) -> str:
        """The refusal, its advice for ``statements`` or, with ``op``, for that edit."""
        return f"row {self.prefix!r} has its binding inline, after '|': " + split_advice(
            self.prefix, self.statement, self.binding, op
        )


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
    from .statements import AGENT_FIELDS

    if set(agent) - set(AGENT_FIELDS):
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


def _modtype(cls_name: str) -> str | None:
    try:
        return modclass_to_modtype.get(get_statement_by_name(cls_name))
    except Exception:  # noqa: BLE001 - not a statement type
        return None


#: The statement type each modification verb names, active (``phosphorylates``) and passive
#: (``is phosphorylated``): INDRA's English for each, read back.
_ACTIVE = {statement_present_verb(t): cls.__name__ for t, cls in modtype_to_modclass.items()}
_PASSIVE = {statement_passive_verb(t): cls.__name__ for t, cls in modtype_to_modclass.items()}
#: The same for a modification of the enzyme itself, which names no substrate.
_SELF = {
    statement_present_verb(cls.__name__.lower()): cls.__name__
    for cls in (Autophosphorylation, Transphosphorylation)
}


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
    if t in _SELF.values():
        return f"{agent_text(st['enz'])} {statement_present_verb(t.lower())}{at}"
    if _modtype(t):
        if st.get("enz"):
            verb = statement_present_verb(t.lower())
            return f"{agent_text(st['enz'])} {verb} {agent_text(st['sub'])}{at}"
        return f"{agent_text(st['sub'])} is {statement_passive_verb(t.lower())}{at}"
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
    if t in ("Gef", "Gap"):
        return f"{agent_text(st[t.lower()])} is a {t.upper()} for {agent_text(st['ras'])}"
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
    # Quoted where it holds what ends a value (space) or opens a comment (`#`), or reads as a
    # number, so :func:`parse_binding` reads it back as the string it is.
    if re.fullmatch(r"[^\s\"|#]+", text) and not _number(text):
        return text
    return '"' + text.replace('"', '\\"') + '"'


def _number(text: str) -> bool:
    try:
        float(text)
    except ValueError:
        return False
    return True


def render(row: dict[str, Any], derived: dict[str, str] | None = None) -> str | None:
    """``row`` (``{prefix, statement}``) as one line, or ``None`` when the notation cannot write it
    losslessly (keep it as JSON). ``derived`` marks the regulators whose kind the row's rate proves
    (:func:`statement_text`)."""
    try:
        if set(row) - {"statement", "prefix"} or not _ROW_PREFIX.fullmatch(row["prefix"] + ":"):
            return None
        line = f"{row['prefix']}: {statement_text(row['statement'], derived)}"
        return line + evidence_text(row["statement"])
    except ValueError:
        return None


def render_binding(
    binding: dict[str, Any], statement_type: str, library: dict | None = None
) -> str | None:
    """``binding`` (``{policy, parameters?}``, of a row whose statement is a ``statement_type``) as
    its text, ``<policy> <kind>=<binding> ...``, or ``None`` when the notation cannot write it
    losslessly (keep it as JSON). ``indra_`` is dropped where ``library`` holds no entry of the
    shorter name for the type, so :func:`parse_binding` reads it back as it was."""
    try:
        if set(binding) - {"policy", "parameters"} or not isinstance(binding.get("policy"), str):
            return None
        policy = binding["policy"]
        short = policy.removeprefix("indra_")
        if short != policy and (library is None or (statement_type, short) not in library):
            policy = short
        tail = [_name(policy)]
        tail += [f"{k}={_value(v)}" for k, v in (binding.get("parameters") or {}).items()]
        return " ".join(tail)
    except ValueError:
        return None


# -------------------------------------------------------------------------------------- parse
class _Reader:
    """A cursor over one row's text."""

    def __init__(self, text: str, groundings: dict[str, dict]):
        self.text, self.pos, self.groundings = text, 0, groundings

    def fail(self, what: str, hint: str | None = None):
        said = f"{what} at column {self.pos + 1}: {self.text!r}"
        raise NotationError(said + (f"; {hint}" if hint else ""))

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
                if word not in activity_types:
                    # `A{mod:on}` is how a scheme writes a molecule's modification site, and
                    # read here as an activity named "mod" it built a species no rule makes.
                    self.fail(
                        f"unknown activity {word!r}: an activity is one of "
                        f"{sorted(activity_types)}; a modification is a tag like pS10 (modified) "
                        "or -pS10 (unmodified)"
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


_CONVERSION = "a Conversion is written `A -> B by E`"
#: The verbs agents write for a statement the notation states otherwise (#227), each answered
#: with the form that states it.
_FORM_OF = {
    "converts": _CONVERSION,
    "converted": _CONVERSION,
    "forms": f"{_CONVERSION}, a Complex `A binds B`",
    "produces": "a production is written `E increases B`",
}


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
        if word == "a":
            kind = r.name()
            if kind not in ("GEF", "GAP"):
                r.fail(f"unknown statement 'is a {kind}'")
            r.expect("for")
            return {"type": kind.capitalize(), kind.lower(): first, "ras": r.agent()}
        if word in ("produced", "degraded"):
            kind = "IncreaseAmount" if word == "produced" else "DecreaseAmount"
            return {"type": kind, "subj": None, "obj": first}
        cls = _PASSIVE.get(word)
        if cls is None:
            r.fail(f"unknown verb 'is {word}'", _FORM_OF.get(word))
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
    if word in _SELF:
        st = {"type": _SELF[word], "enz": first, "residue": None, "position": None}
        _at(r, st)
        return st
    cls = _ACTIVE.get(word)
    if cls is None:
        r.fail(f"unknown verb {word!r}", _FORM_OF.get(word))
    st = {"type": cls, "enz": first, "sub": r.agent(), "residue": None, "position": None}
    _at(r, st)
    return st


def _bar(text: str) -> int | None:
    """Where ``text`` has a ``|`` outside quotes and braces, ``None`` where it has none."""
    depth, quoted = 0, False
    for i, c in enumerate(text):
        if c == '"' and (i == 0 or text[i - 1] != "\\"):
            quoted = not quoted
        elif not quoted and c == "{":
            depth += 1
        elif not quoted and c == "}":
            depth -= 1
        elif not quoted and depth == 0 and c == "|":
            return i
    return None


def _split_comment(text: str) -> tuple[str, str | None]:
    """Split ``text`` at the ``#`` outside quotes: the evidence comment, ``None`` without one."""
    quoted = False
    for i, c in enumerate(text):
        if c == '"' and (i == 0 or text[i - 1] != "\\"):
            quoted = not quoted
        elif c == "#" and not quoted:
            return text[:i], text[i + 1 :].strip()
    return text, None


def split_advice(prefix: str, statement: str, binding: str, op: str | None = None) -> str:
    """How a row given with its binding is written instead: the row, its statement alone, and the
    binding under its prefix in ``bindings``; or, sent in a ``statement_edits`` edit ``op``, that
    edit with its own ``binding``."""
    row = f"{prefix}: {statement}"
    if op is None:
        return (
            "a row is its statement, and its policy and parameter bindings go in `bindings` under "
            f"its prefix: `{row}` with `bindings={json.dumps({prefix: binding})}`"
        )
    return (
        "a row is its statement, and its policy and parameter bindings go in the edit's own "
        f"`binding`: `{json.dumps({'op': op, 'row': row, 'binding': binding})}`"
    )


def refuse_control(what: str, text: str) -> None:
    """Refuse ``text`` (``what`` names it) where it holds a control character, named by its escape
    and column."""
    if m := _CONTROL.search(text):
        raise NotationError(
            f"{what} {text!r} holds the control character {m.group()!r} at column {m.start()}: "
            "send it without one"
        )


def refuse_inline(text: str) -> None:
    """Refuse ``text``, a row, where it carries its binding after a ``|`` (:class:`InlineBinding`);
    whether the rest reads is :func:`parse`'s to say."""
    m = _ROW_PREFIX.match(text)
    if not m:
        return
    head, comment = _split_comment(text[m.end() :])
    if (bar := _bar(head)) is not None:
        said = head[:bar].strip() + ("" if comment is None else f" # {comment}")
        raise InlineBinding(m.group(1), said, head[bar + 1 :].strip())


def resolve_policy(policy: str, statement_type: str | None, library: dict | None) -> str:
    """``policy`` as ``library`` names it: one written without its ``indra_`` prefix is INDRA's
    where the library holds no entry of that name for the type, in a binding's text or its JSON
    alike."""
    if library is not None and (statement_type, policy) not in library:
        if any(p == "indra_" + policy for _, p in library):
            return "indra_" + policy
    return policy


def parse(text: str, groundings: dict[str, dict] | None = None) -> dict[str, Any]:
    """One row from ``text``: ``{prefix, statement (INDRA JSON)}``. ``groundings`` (agent name to
    ``db_refs``) ground the agents. A row with its binding after a ``|`` is refused saying how it
    splits (:class:`InlineBinding`)."""
    from .statements import REGULATORS

    groundings = groundings or {}
    refuse_control("row", text)
    m = _ROW_PREFIX.match(text)
    if not m:
        raise NotationError(f"a row starts with its prefix and ':': {text!r}")
    refuse_inline(text)
    head, comment = _split_comment(text[m.end() :])
    r = _Reader(head, groundings)
    st = statement(r)
    while r.take(","):
        kind = next((k for k, words in REGULATORS.items() if r.take(words)), None)
        if kind is None or kind in st:
            r.fail(
                f"expected one of {', '.join(map(repr, REGULATORS.values()))}, each once: several "
                "of one kind are one clause joined by `and` (`, inhibited by C and D`), and in a "
                "JSON row a list (`inhibited_by: [{name: 'C'}, {name: 'D'}]`)"
            )
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
    return {"prefix": m.group(1), "statement": st}


def parse_binding(
    text: str, statement_type: str | None = None, library: dict | None = None
) -> dict[str, Any]:
    """A row's binding from its text, ``<policy> <kind>=<binding> ...``: ``{policy, parameters?}``.
    ``library`` resolves a policy written without its ``indra_`` prefix, for a row whose statement
    is a ``statement_type``."""
    if not text.strip():
        raise NotationError("a binding names its policy first: `mass_action`, `two_step kf=k_on`")
    if re.match(r"\s*[A-Za-z_][A-Za-z0-9_]*\s*:(?!=)", text):
        raise NotationError(
            "a binding is `<policy> <kind>=<binding> ...`; its row's prefix is its key in "
            f"`bindings`, not written in it: {text!r}"
        )
    if _split_comment(text)[1] is not None:
        raise NotationError(
            "a binding carries no evidence: a statement's evidence is a comment on its row, "
            f"`<prefix>: <statement> # [source] text`: {text!r}"
        )
    t = _Reader(text, {})
    binding: dict[str, Any] = {"policy": resolve_policy(t.name(), statement_type, library)}
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
        binding.setdefault("parameters", {})[key] = value
    return binding


def parse_agent(text: str, groundings: dict[str, dict] | None = None) -> dict[str, Any]:
    """One agent (INDRA Agent JSON) from its notation."""
    r = _Reader(text, groundings or {})
    agent = r.agent()
    if not r.done():
        r.fail("unexpected text after the agent")
    return agent
