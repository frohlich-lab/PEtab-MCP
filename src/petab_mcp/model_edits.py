"""STRUCTURED edits to a model: say WHAT changes, not which characters to replace.

The substring-patch surface this replaces was measured over 30 sessions and 1,063 patches, and every
one of its failure classes came from addressing a structured object by its text:

* 20% of all patches targeted the ``const ...;`` declaration — a DERIVED line libantimony
  regenerates, which cannot change the model (a parameter is constant unless declared ``var``)
  and is never the text the agent last saw. It caused 71% of all patch failures.
* Of the failures that remained, 81% quoted text present in NEITHER the rendered model nor the
  agent's own submission — a plausible-looking anchor rather than a copied one. Patching turns
  authoring (which the model does well) into exact reproduction (which it does badly); no
  instruction fixes that, because the text was already in its context.
* Reaction IDs churn when a model is re-authored, so a purely additive change reads as a
  wholesale rewrite: 55% of the "coordinated rewrite" steps in an earlier analysis were nothing
  but renumbering.

Addressing a reaction by its ID removes all three at once: an ID either exists (and the server
can list the valid ones when it does not), it survives re-rendering, and the operation says what
it means, so the diff arrives pre-classified instead of being reverse-engineered afterwards.

BATCHED, for the same reason the patch list was: one call is one coherent change, one fit, one
version. A structural change is rarely one operation — adding a mechanism usually means a new
reaction plus a rate law change on the reaction it feeds — and paying a fit per operation would
cost more than the whole surface saves. Operations apply IN ORDER to an in-memory copy and the
batch is ALL-OR-NOTHING: any failure names the failing index and nothing is applied.

The edits apply to the model SOURCE — the text the caller last authored or last edited (or, on a
shipped problem, the model it was given to repair) — not to the reconciled re-render. That is
what makes a version's lineage checkable: a child's source is its parent's source plus this
call's operations.

PARSING. Everything below works on STATEMENTS found in a comment-masked copy of the whole source,
never on lines. Each earlier line-based iteration produced its own silent corruption class, and
they were all the same mistake wearing different clothes: `/` excluded from the rate law so every
saturating law became unmatchable; line anchoring so two reactions on one line were both
unaddressable; a trailing `;` required in one pattern and optional in its sibling, so a symbol was
reported unknown and listed as known in the same sentence; and comments stripped in three places
but not the fourth, so `set_initial` would rewrite a value inside a comment and report success.
Masking comments once (preserving offsets, so spans still index the original text) and scanning
statements across newlines removes the class rather than the instance.

Why not parse with libantimony, which is already a dependency: round-tripping through it returns
a RE-RENDER, and the difference between the caller's source and the render is the thing this
module exists to preserve. Reading a model through libantimony to VALIDATE ids is a fair future
addition; replacing the source with its output is not.
"""

from __future__ import annotations

import difflib
import re
from typing import Any

from .operations import OperationError

#: Ceiling on any id list in an error message. Same rule and reason as `server._ID_LIST_CAP`:
#: a 118-symbol parameter table was a measured top-3 context cost, and an unbounded list lands in
#: the caller's history for the rest of the session.
_ID_LIST_CAP = 60

#: Antimony syntax that a `rename` must never treat as a symbol, on EITHER side. Not exhaustive —
#: it is the declaration/structure vocabulary, which is what a whole-model substitution corrupts.
_ANTIMONY_KEYWORDS = frozenset(
    {
        "model",
        "end",
        "species",
        "compartment",
        "const",
        "var",
        "formula",
        "substanceOnly",
        "function",
        "is",
        "has",
        "in",
        "at",
        "if",
        "else",
        "delete",
        "import",
        "unit",
        "event",
    }
)

#: Names a `rename` must refuse on EITHER side even though they are not declaration keywords:
#: MathML/SBML constants and the function vocabulary. `time` is the one that matters — renaming it
#: is the silent failure this module warns about everywhere else: `rename(old='time', new='tau')`
#: rewrites `piecewise(0, time < t0, 1)` to use `tau`, nothing binds `tau` to simulation time, the
#: reconcile mints it as an ESTIMATED parameter, and every gated input goes constant while the
#: model still compiles, fits and commits. `rename(new='exp')` is the same class from the other
#: side: it produces `exp*A;` and shadows the function.
_PROTECTED_SYMBOLS = _ANTIMONY_KEYWORDS | frozenset(
    {
        "time",
        "pi",
        "avogadro",
        "exponentiale",
        "true",
        "false",
        "inf",
        "nan",
        "notanumber",
        "abs",
        "exp",
        "ln",
        "log",
        "log10",
        "sqrt",
        "pow",
        "root",
        "piecewise",
        "delay",
        "sin",
        "cos",
        "tan",
        "asin",
        "acos",
        "atan",
        "sinh",
        "cosh",
        "tanh",
        "ceil",
        "ceiling",
        "floor",
        "factorial",
        "and",
        "or",
        "not",
        "xor",
        "gt",
        "lt",
        "geq",
        "leq",
        "eq",
        "neq",
    }
)

#: ONE named reaction statement, found ANYWHERE in the masked source: ``id: A + B -> C; law;``.
#:
#: Not line-anchored and not line-bounded. Antimony allows several statements on one line AND one
#: statement across several lines; agents write both. Anchoring made `fwd: ...; rev: ...;` match
#: nothing at all, and line-bounding reported a perfectly legal
#: ``r1: A + B -> C;\n     k1*A*B;`` as an UNNAMED reaction — refused on authoring, with an error
#: naming no fix, for a reaction that plainly has an id.
#:
#: The id is REQUIRED — it is how every operation addresses a reaction, so an unnamed one is
#: unaddressable by construction. Unnamed reactions are legal Antimony, so they are refused where
#: they are WRITTEN (see :func:`unnamed_reactions`) rather than auto-named here, which would
#: invent ids the caller never chose and renumber them on the next re-author.
#:
#: The law may be EMPTY (`r1: A -> B;` is legal, the rate law omitted) and its terminator may be a
#: `;` or the end of a line — Antimony takes a newline as a statement terminator, and the bundled
#: antimony-authoring skill writes exactly that form (`r1: A -> B; kf*A - kr*B`).
#:
#: The law also STOPS at the start of a new named statement (`(?=\w+\s*:)`) and cannot contain a
#: `:` at all. Without that it ran straight through the next statement: `fwd: A -> B; rev: B -> A;`
#: parsed as ONE reaction whose law was `rev: B -> A`, so `rev` was unaddressable and
#: `remove_reaction("fwd")` deleted BOTH while reporting one — and `r1: A -> B; r2: C => D;` gave
#: `r2` the law `end`, swallowing the model's terminator. Rate laws are arithmetic; none contains a
#: colon, so excluding it costs nothing. It stops at a bare `end` for the same reason — a law-less
#: reaction as the last statement otherwise took the model's terminator as its rate law
#: (`rev: B -> A;` + `end` gave law `end`). `\bend\b` cannot fire inside a parameter named
#: `k_end`, since the preceding `_` is a word character.
#:
#: The law admits COMPARISONS (`>=`, `<=`, `==`, `!=`) but never a bare `=`. Both halves matter:
#: excluding `=` outright made every time-gated law unmatchable — `r1: A -> B; k*A*piecewise(0,
#: time >= t0, 1);` returned NO reaction ids and was reported as UNNAMED, so the wholesale path
#: rejected exactly the time-gated models an agent is routinely asked to write. Allowing
#: a bare `=` would let a law-less reaction swallow the assignment after it (`r1: A -> B;` +
#: `k1 = 0.5;` must stay two statements). The two-char alternative is tried FIRST so `>=` is
#: consumed as a unit rather than backtracked into.
_RXN_STMT = re.compile(
    r"(?P<id>\w+)\s*:\s*(?P<stoich>[^;:=]*?(?:->|=>)[^;:=]*?)\s*;"
    r"\s*(?P<law>(?:[<>!=]=|[^;=:])*?)\s*(?:;|$|(?=\w+\s*:)|(?=\bend\b))",
    re.MULTILINE,
)

#: A scalar assignment: ``k1 = 0.5;`` — parameters, species initials, compartment sizes. The value
#: stays on its line; the terminating `;` is optional, exactly as for a reaction.
#: `=(?![>=])` keeps this off the `A =>` of a reaction.
_ASSIGN_STMT = re.compile(
    r"(?<![\w.])(?P<id>\w+)\s*=(?![>=])\s*(?P<value>[^;\n]+?)\s*(?:;|$)", re.MULTILINE
)

#: An arrow — what makes a statement a reaction rather than an assignment.
_ARROW = re.compile(r"->|=>")

#: What an op's expression never holds and a statement does: a label or `:=` colon, a `->`
#: arrow, or a lone `=` (which also covers `=>`; `==`, `<=`, `>=`, `!=` stay comparisons).
_STATEMENT_MARK = re.compile(r":|->|(?<![<>!=])=(?!=)")

OPS = (
    "add_reaction",
    "remove_reaction",
    "set_rate_law",
    "set_initial",
    "set_assignment",
    "remove_assignment",
    "rename",
)


#: One-entry memo for :func:`mask_comments`. Every helper masks INDEPENDENTLY — a single
#: `add_reaction` masks via `_reactions`, `unnamed_reactions`, `reaction_ids`, `model_symbols`
#: (twice, once through `assigned_symbols`), `assigned_symbols`, and the leftover check — so the
#: same source was re-walked character by character in a Python list six or more times per
#: operation, all under `session.lock`, and a batch over a 53-reaction model re-walked it
#: dozens of times. The masking is a pure function of the text, so remembering the LAST result
#: collapses all of that: within one operation every call passes the identical string. One entry
#: rather than an lru_cache because model sources are large and only the current one is ever hot.
_MASK_MEMO: tuple[str, str] | None = None


def mask_comments(text: str) -> str:
    """``text`` with every comment blanked to spaces, SAME LENGTH and same newlines.

    Offsets are preserved so a match found in the mask indexes straight into the original. This is
    the single place comments are handled; every scan below runs on the mask, which is what stops
    one operation from quietly treating a comment as code (`set_initial` once rewrote the value
    inside `// A = 1; the literature value`, left the real `A = 3;` alone, and reported success).
    """
    global _MASK_MEMO
    src = text or ""
    # ONE read into a local. Reading the global twice (guard, then [1]) let another tool thread
    # swap it in between, returning the mask of a DIFFERENT source — and every caller then indexes
    # the original text with offsets from a differently-sized string, which corrupts the model
    # silently instead of raising. FastMCP dispatches each tool call on its own thread and
    # `make_changes` masks before it takes `session.lock`, so this is reachable.
    memo = _MASK_MEMO
    if memo is not None and memo[0] == src:
        return memo[1]
    out = list(src)
    i, n = 0, len(out)
    while i < n:
        if out[i] == "/" and i + 1 < n and out[i + 1] == "/":
            j = text.find("\n", i)
            j = n if j < 0 else j
            for k in range(i, j):
                out[k] = " "
            i = j
        elif out[i] == "/" and i + 1 < n and out[i + 1] == "*":
            j = text.find("*/", i + 2)
            j = n if j < 0 else j + 2
            for k in range(i, j):
                if out[k] != "\n":
                    out[k] = " "
            i = j
        else:
            i += 1
    masked = "".join(out)
    _MASK_MEMO = (src, masked)
    return masked


def _fmt_ids(ids: list[str]) -> str:
    """An id list for an error message, capped — see :data:`_ID_LIST_CAP`."""
    if not ids:
        return "(none)"
    if len(ids) <= _ID_LIST_CAP:
        return ", ".join(ids)
    return f"{', '.join(ids[:_ID_LIST_CAP])}, (+{len(ids) - _ID_LIST_CAP} more)"


def closest_symbols(wanted: str, known: list[str], n: int = 2, cutoff: float = 0.6) -> list[str]:
    """The known symbols most likely to be what ``wanted`` meant.

    Token overlap alongside character similarity, taking whichever is kinder: the commonest real
    miss is the right concept under a swapped convention (`X2_scaling` for `scaling_X2` —
    agents wrote both orderings across sessions), and character-order matching scores a swap as a
    near-total mismatch, then confidently suggests a DIFFERENT symbol instead.

    One definition, used by every surface that reports an unknown id — see
    `server.unknown_ids_message`, which calls this rather than keeping a second copy.
    """

    def score(cand: str) -> float:
        w, c = set(wanted.lower().split("_")), set(cand.lower().split("_"))
        union = w | c
        jaccard = len(w & c) / len(union) if union else 0.0
        return max(difflib.SequenceMatcher(None, wanted.lower(), cand.lower()).ratio(), jaccard)

    ranked = sorted(((score(k), k) for k in known), key=lambda t: (-t[0], t[1]))
    return [k for s, k in ranked[:n] if s >= cutoff]


def _reactions(text: str) -> list[re.Match]:
    """Every named reaction statement, in order, matched against the comment mask."""
    return list(_RXN_STMT.finditer(mask_comments(text)))


def reaction_ids(text: str) -> list[str]:
    """Every reaction id in the model, in order — what an unknown-id error should list."""
    return [m.group("id") for m in _reactions(text or "")]


def unnamed_reactions(text: str) -> list[str]:
    """Reaction statements written without an id, stripped — empty when every one is named.

    Called on the WHOLESALE path so the model arrives fully addressable: `model_edits` names
    reactions by id, and a reaction the agent never named is one it can never change without
    re-authoring the whole model, which is the cost this surface exists to avoid.

    Works by DELETING the named statements from the mask and asking what arrows are left, so a
    line holding both a named and an unnamed reaction still reports the unnamed one, and a comment
    describing a reaction reports nothing.
    """
    masked = _RXN_STMT.sub("", mask_comments(text or ""))
    out: list[str] = []
    for part in masked.split(";"):
        arrow = _ARROW.search(part)
        if not arrow:
            continue
        # Report the OFFENDING STATEMENT, not the whole chunk. Scanning the source as one string
        # means a chunk begins wherever the previous statement ended, which can be several lines
        # back — quoting `model m\n  A -> B` at the caller names the wrong thing and reads as a
        # parser confusion rather than a missing id.
        out.append(" ".join(part[part.rfind("\n", 0, arrow.start()) + 1 :].split()))
    return out


_RULE_STMT = re.compile(r"(?<![\w.])(?P<id>[A-Za-z_]\w*)\s*:=", re.MULTILINE)


def _stoichiometry_symbols(text: str) -> set[str]:
    """The DYNAMIC states in a reaction's stoichiometry.

    Two things in a stoichiometry are not states, and counting them made every guard built on
    this refuse legal models outright:

    * `$S` marks a BOUNDARY species — held by something outside the reaction network, so an
      assignment rule on it is legal SBML (libantimony loads it without complaint). The `$` may
      sit on the reaction (`$u -> B`) OR on a separate declaration (`species $u;`) with the
      reaction naming it bare. libantimony's own round-trip re-emits the `$` in BOTH places, so
      this is about HAND-AUTHORED text -- which is what the agent writes, and
      `species $u, B; R1: u + B -> C; u := 3;` loads clean. Stripping only the inline form
      counted such a species as dynamic and locked the caller out of `model_edits` on the very
      models that need it;
    * `A in cyt` names a COMPARTMENT, and a growing compartment (`cyt := V0*exp(mu*time)`) is an
      ordinary modelling pattern. A bare regex also picked up the keyword `in` itself.

    Counting either made the batch over-determination check fire on the FINAL text of every
    batch, so a model containing one locked the caller out of `model_edits` entirely — including
    the `remove_assignment` the error message recommends.
    """
    masked = mask_comments(text or "")
    # `$` marks the species boundary WHEREVER it appears, so collect the marks document-wide
    # rather than per-reaction: a `species $u;` declaration up top makes a bare `u` in a
    # reaction below non-dynamic just as surely as writing `$u` in the reaction would.
    boundary = set(re.findall(r"\$\s*([A-Za-z_]\w*)", masked))
    out: set[str] = set()
    for m in _RXN_STMT.finditer(masked):
        stoich = m.group("stoich")
        stoich = re.sub(r"(?<![\w.])in[ \t]+[A-Za-z_]\w*", " ", stoich)  # drop `in <compartment>`
        out |= set(re.findall(r"[A-Za-z_]\w*", stoich))
    return out - boundary


def rule_target_symbols(text: str) -> set[str]:
    """Symbols carrying an assignment rule (`id := ...`) in this source."""
    masked = mask_comments(text or "")
    return {m.group("id") for m in _RULE_STMT.finditer(masked)}


def self_assignments(text: str) -> list[str]:
    """Symbols whose assignment rule reads the symbol itself (`x := 2*x`), in order."""
    masked = mask_comments(text or "")
    rules = re.finditer(r"(?<![\w.])(?P<id>[A-Za-z_]\w*)\s*:=\s*(?P<expr>[^;\n]*)", masked)
    return [
        m["id"] for m in rules if re.search(rf"(?<![\w.]){re.escape(m['id'])}(?!\w)", m["expr"])
    ]


def assigned_symbols(text: str) -> list[str]:
    """Every symbol given a value — the remedy list `set_initial` owes a caller whose id missed."""
    masked = mask_comments(text or "")
    spans = [(m.start(), m.end()) for m in _RXN_STMT.finditer(masked)]
    out: set[str] = set()
    for m in _ASSIGN_STMT.finditer(masked):
        if any(s <= m.start() < e for s, e in spans):
            continue  # an `=` inside a reaction statement is that reaction's business
        out.add(m.group("id"))
    return sorted(out)


def model_symbols(text: str) -> set[str]:
    """Every name the model already uses — reaction ids, stoichiometry species, assigned symbols.

    Antimony keeps ONE symbol table, so a new reaction id colliding with a parameter or species is
    a redefinition the reconcile rejects with an opaque parse error, far from the operation that
    caused it.
    """
    masked = mask_comments(text or "")
    names: set[str] = set(assigned_symbols(text or ""))
    for m in _RXN_STMT.finditer(masked):
        names.add(m.group("id"))
        # BOTH sides. Rate-law-only symbols are precisely the ones the reconcile MINTS (measured at
        # 19 per session), so a guard that reads only the stoichiometry misses its commonest case:
        # `add_reaction('k_deg: B -> C; k2*B;')` against a model whose only `k_deg` is in a rate
        # law was accepted, redefining it as a reaction id and turning the refusal this function
        # exists to give into an opaque libantimony error at reconcile time.
        names |= set(re.findall(r"[A-Za-z_]\w*", m.group("stoich")))
        # NOT minus _PROTECTED_SYMBOLS. Filtering the law side would drop a parameter a model
        # legitimately named `log`/`pow`/`root`/`delay`, and losing it here loses the collision
        # guard for exactly that name — the opaque libantimony redefinition this function exists to
        # pre-empt. A caller cannot name a reaction `time` anyway: that is refused as a keyword.
        names |= set(re.findall(r"[A-Za-z_]\w*", m.group("law")))
    return names


def _find_reaction(text: str, rid: str) -> re.Match | None:
    for m in _reactions(text):
        if m.group("id") == rid:
            return m
    return None


def _unknown_reaction(where: str, rid: str, text: str) -> OperationError:
    ids = reaction_ids(text)
    close = closest_symbols(rid, ids)
    hint = f" Did you mean {' or '.join(repr(c) for c in close)}?" if close else ""
    return OperationError(
        f"{where}: no reaction {rid!r}.{hint} Existing: {_fmt_ids(ids)}. NOTHING was applied.",
        cause="unknown_reaction_id",
    )


def _reject_semicolons(where: str, **fields: str) -> None:
    """A `;` inside a field would terminate the statement early, so the text this writes stops
    matching `_RXN_STMT` and the reaction becomes permanently unaddressable — created
    successfully, then invisible to every later operation."""
    for name, value in fields.items():
        if ";" in (value or ""):
            raise OperationError(
                f"{where}: `{name}` must not contain ';' — one operation writes one statement, "
                "and a stray ';' would split it and make the reaction unaddressable. "
                "NOTHING was applied."
            )


def _line_bounds(text: str, pos: int) -> tuple[int, int]:
    """``(start, end)`` offsets of the line containing ``pos``, end excluding the newline."""
    start = text.rfind("\n", 0, pos) + 1
    end = text.find("\n", pos)
    return start, (len(text) if end < 0 else end)


def _model_block_start(masked: str) -> int:
    """Offset just past the ``model <name>(...)`` header, or 0 when there is no explicit block.

    Only used when the model declares NO reactions yet, where there is no reaction to anchor a
    forward search on. Antimony permits `function ... end` blocks before the model, so starting
    the terminator search at 0 would find the FUNCTION's `end` and insert the first reaction
    inside it. 0 is the right fallback for the common bare-body model (no `model` keyword at
    all), which has exactly one block anyway.
    """
    # Match the HEADER only -- `model`, an optional `*`, the name, and an optional argument list.
    # `[^\n]*` ran to end of LINE instead, which is the same thing for the usual multi-line model
    # but wrong for a single-line one: on `model m() end` it consumed the `end` too, so the
    # terminator search below started PAST the block and the first reaction was appended outside
    # it (or found the next block's `end`).
    m = re.search(
        r"(?<![\w.])model(?![\w.])[ \t]*\*?[ \t]*(?:[A-Za-z_]\w*)?[ \t]*"
        r"(?:\([^)]*\))?",
        masked,
    )
    return m.end() if m else 0


def _op_add_reaction(text, op, where):
    # ONE Antimony statement, written as it would appear in the model. Decomposing it into
    # id/stoichiometry/rate_law fields made the caller learn a bespoke format for a language it
    # already writes (and the skill already teaches), and cost ~125 tokens of schema on EVERY
    # turn. The id is parsed back out because it is how every later operation addresses this
    # reaction — which is also why it is required rather than optional.
    stmt = (op.get("antimony") or "").strip()
    if not stmt:
        raise OperationError(
            f"{where}: add_reaction needs `antimony` — one named reaction statement, e.g. "
            "'transfer: X1 => X2; k*u*X1;'. NOTHING was applied."
        )
    found = _reactions(stmt)
    if (
        not found
        and re.fullmatch(r"[A-Za-z_]\w*", str(op.get("id") or "").strip())
        and str(op.get("id")).strip() not in _PROTECTED_SYMBOLS
        and unnamed_reactions(stmt)
    ):
        # The caller supplied the id separately and wrote the statement without it. Refusing here
        # would be the same turn-burning pedantry the redundant-id rejection was removed for --
        # and the schema now says "repeating the same id is accepted", which invites exactly this
        # shape. The id is unambiguous (there is only one), so assemble it and carry on.
        # Guarded on the SAME id pattern `rename` uses (an SBML SId, not \w+, which would admit
        # '3oxo' and fail later inside libantimony), and on the protected-symbol list. Built into
        # a LOCAL and adopted only if it actually parses: committing it to `stmt` first makes
        # every downstream error quote a string the caller never wrote -- e.g. an antimony
        # missing its rate law would be reported as "cannot read 'r1: A => B' ... it has no id",
        # which is both wrong and self-contradictory.
        candidate = f"{str(op['id']).strip()}: {stmt.lstrip()}"
        if _reactions(candidate):
            stmt = candidate
            found = _reactions(stmt)
    if not found:
        why = (
            "it needs a rate law after the stoichiometry, e.g. 'transfer: X1 => X2; k*u*X1;'"
            if str(op.get("id") or "").strip() and unnamed_reactions(stmt)
            else "it has no id — name it, e.g. 'transfer: X1 => X2; k*u*X1;', because "
            "the id is how you will address this reaction later"
            if unnamed_reactions(stmt)
            else "expected 'id: <stoichiometry>; <rate law>;' with -> (reversible) or => "
            "(irreversible), e.g. 'transfer: X1 => X2; k*u*X1;'"
        )
        raise OperationError(
            f"{where}: cannot read {stmt!r} as a reaction — {why}. NOTHING was applied.",
            cause="malformed_reaction",
        )
    if len(found) > 1:
        raise OperationError(
            f"{where}: `antimony` holds {len(found)} reactions "
            f"({_fmt_ids([m.group('id') for m in found])}) — one operation writes one statement, "
            "so send one add_reaction per reaction. NOTHING was applied.",
            cause="malformed_reaction",
        )
    m = found[0]
    # A redundant `id` used to be REJECTED outright ("takes its id from the statement — drop
    # `id`"), which is interface pedantry agent traces paid for: 12 rejections in one set of
    # sessions, and in one of them the burned turn was the FIRST attempt at a delay chain — the
    # single most valuable mechanism for that problem. A matching
    # id states the caller's intent twice, consistently; only a MISMATCH is ambiguous (which name
    # would later operations address?), so only a mismatch is refused.
    given = str(op.get("id") or "").strip()
    if given and given != m.group("id"):
        raise OperationError(
            f"{where}: `id` is {given!r} but the statement names the reaction "
            f"{m.group('id')!r} — the statement's name is the one the model gets, so either "
            "drop `id` or make them agree. NOTHING was applied.",
            cause="id_mismatch",
        )
    # Anything BEYOND the reaction (a second assignment, say) would be dropped on the floor by the
    # reassembly below, so refuse rather than silently discard. A trailing comment is kept.
    leftover = (mask_comments(stmt)[: m.start()] + mask_comments(stmt)[m.end() :]).strip(" \t\n;")
    if leftover:
        raise OperationError(
            f"{where}: `antimony` has content beyond the reaction statement ({leftover!r}) — one "
            "operation writes one statement. Send the rest as its own operation. "
            "NOTHING was applied.",
            cause="malformed_reaction",
        )
    comment = ""
    cmatch = re.search(r"(?://|/\*).*$", stmt, re.S)
    if cmatch and cmatch.start() >= m.end():
        comment = f"  {cmatch.group(0).strip()}"
    rid, stoich, law = m.group("id"), m.group("stoich").strip(), m.group("law").strip()
    # Reserved names, checked HERE and not only on the auto-id path above. `_PROTECTED_SYMBOLS`
    # was consulted where the caller supplies `id` separately, and on rename -- but not for the
    # id parsed out of the statement, which is the documented normal form. So
    # `add_reaction('at: -> X; k*C;')` wrote an Antimony KEYWORD as a reaction id and surfaced as
    # `unexpected ->` from libantimony, an opaque parse error far from the operation that caused
    # it. `time` was the same: `model_symbols` asserts "a caller cannot name a reaction `time`
    # anyway: that is refused as a keyword", which was only true if the model already mentioned
    # it somewhere.
    if rid in _PROTECTED_SYMBOLS:
        kind = "an Antimony keyword" if rid in _ANTIMONY_KEYWORDS else "a built-in symbol"
        raise OperationError(
            f"{where}: {rid!r} is {kind}, so it cannot be a reaction id — libantimony would "
            "reject the model with a parse error that does not mention this reaction. Pick "
            "another id. NOTHING was applied.",
            cause="reserved_id",
        )
    # ONE symbol table: a reaction id that is already a parameter or species is a redefinition,
    # and libantimony reports it as an opaque parse failure at reconcile time rather than as this
    # module's own rejection with a remedy.
    if rid in reaction_ids(text):
        # A duplicate REACTION id has a better remedy than "pick another name", so keep it
        # distinct from the symbol-table collision below.
        raise OperationError(
            f"{where}: reaction {rid!r} already exists — use set_rate_law to "
            "change it, or pick another id. NOTHING was applied.",
            cause="duplicate_reaction_id",
        )
    if rid in rule_target_symbols(text):
        # `model_symbols` cannot see rule targets (`assigned_symbols` matches `=`, not `:=`), so
        # this collision slipped through -- and libantimony does NOT reject it: it reinterprets
        # the rule as the new reaction's KINETIC LAW, silently discarding both the law the caller
        # wrote and the assignment rule.
        raise OperationError(
            f"{where}: {rid!r} already carries an assignment rule in this model. Antimony keeps "
            "one symbol table and would read the rule as this reaction's rate law, discarding "
            "both. Pick another reaction id (or remove_assignment first). NOTHING was applied.",
            cause="id_collision",
        )
    if rid in model_symbols(text):
        masked_all = mask_comments(text)
        in_stoich, in_law = set(), set()
        # `_m`, not `m`: `m` is the match for the statement being ADDED, and rebinding it here
        # left this correct only by accident (rid/stoich/law were read earlier and the block always
        # raises). Any code added after this loop would read a match from the wrong text.
        for _m in _RXN_STMT.finditer(masked_all):
            in_stoich |= set(re.findall(r"[A-Za-z_]\w*", _m.group("stoich")))
            in_law |= set(re.findall(r"[A-Za-z_]\w*", _m.group("law")))
        kind = (
            "an assigned symbol"
            if rid in assigned_symbols(text)
            else "a species"
            if rid in in_stoich
            else "a rate-law parameter"
            if rid in in_law
            else "already in use"
        )
        raise OperationError(
            f"{where}: {rid!r} is already {kind} in this model, and Antimony keeps one symbol "
            "table — reusing the name is a redefinition. Pick another reaction id. "
            "NOTHING was applied.",
            cause="id_collision",
        )
    body = f"{rid}: {stoich}; {law};" if law else f"{rid}: {stoich};"
    masked = mask_comments(text)
    existing = _reactions(text)
    # The terminator of the block the reaction belongs in — NOT simply the first `end` in the
    # source. Antimony allows blocks before the model (`function f(x) ... end`, or a second model),
    # and anchoring on the first `end` inserted the new reaction into THAT block while reporting
    # "added reaction r2": libantimony then either puts it in the wrong namespace or fails
    # opaquely at reconcile. Search forward from the anchor instead, so the match is the `end`
    # that closes the block actually being edited.
    #
    # From `last.end()` and not from the end of its LINE, so an `end` sharing the last reaction's
    # line (`A = 1; r1: A -> B; k1*A; end`) is still found — that case needs the insert-before-end
    # fallback below, and searching from the line end would skip past it to the next block's `end`
    # (or find none) and silently append outside the model.
    search_from = existing[-1].end() if existing else _model_block_start(masked)
    end_tok = re.compile(r"(?<![\w.])end(?![\w.])").search(masked, search_from)
    if existing:
        last = existing[-1]
        # END of the last reaction's match, not its START: a reaction may span lines
        # (`r1: A -> B;\n   k1*A;`), and anchoring on the start inserted the new statement
        # BETWEEN the stoichiometry and its rate law — orphaning the law, leaving `k1*A;` as a
        # bare statement, and reporting success.
        line_start, line_end = _line_bounds(text, last.end())
        # ...but never past the model's `end`. With `end` on the same line as the last reaction
        # (`A = 1; r1: A -> B; k1*A; end`) the line-end insert lands OUTSIDE `model ... end`,
        # where libantimony puts it in another namespace or fails to parse — reported as success
        # either way. Fall back to inserting before `end` when that happens.
        if end_tok is None or end_tok.start() >= line_end:
            indent = re.match(r"[ \t]*", text[line_start:]).group(0) or "  "
            return text[:line_end] + f"\n{indent}{body}{comment}" + text[
                line_end:
            ], f"added reaction {rid}"
    if end_tok:
        at = end_tok.start()
        # Keep `end` where it is; put the statement on its own line just before it.
        pre = text[:at].rstrip(" \t")
        return pre + f"\n  {body}{comment}\n" + text[at:], f"added reaction {rid}"
    sep = "" if text.endswith("\n") or not text else "\n"
    return text + f"{sep}  {body}{comment}\n", f"added reaction {rid}"


def _op_remove_reaction(text, op, where):
    rid = (op.get("id") or "").strip()
    m = _find_reaction(text, rid)
    if m is None:
        raise _unknown_reaction(where, rid, text)
    # Cut the STATEMENT, not the line: a line may carry another reaction the caller did not ask to
    # remove. Drop the whole line only once nothing but whitespace is left on it.
    cut = text[: m.start()] + text[m.end() :]
    line_start, line_end = _line_bounds(cut, min(m.start(), max(len(cut) - 1, 0)))
    if not cut[line_start:line_end].strip():
        cut = cut[:line_start] + cut[min(line_end + 1, len(cut)) :]
    return cut, f"removed reaction {rid}"


def _op_set_rate_law(text, op, where):
    rid = (op.get("id") or "").strip()
    law = (op.get("antimony") or "").strip().rstrip(";")
    if not law:
        raise OperationError(
            f"{where}: set_rate_law needs `antimony` — the new rate law, e.g. "
            "'k*u*X1'. NOTHING was applied."
        )
    if _STATEMENT_MARK.search(law):
        raise OperationError(
            f"{where}: set_rate_law's `antimony` is the rate expression only, e.g. 'k*u*X1', "
            "not a reaction statement. It cannot change reactants or products: for that, use "
            "remove_reaction, then add_reaction with the new statement. NOTHING was applied.",
            cause="statement_not_expression",
        )
    _reject_semicolons(where, antimony=law)
    m = _find_reaction(text, rid)
    if m is None:
        raise _unknown_reaction(where, rid, text)
    stmt = f"{rid}: {m.group('stoich').strip()}; {law};"
    tail = text[m.end() :]
    # The law match can end immediately before the next statement (the `(?=\w+\s*:)` stop), and
    # the space that separated them was consumed by the pattern's leading `\s*`. Put one back so
    # the source the caller reads is `...; k1*A; rev: ...` rather than `...; k1*A;rev: ...`.
    if tail[:1] not in ("", " ", "\t", "\n", ";"):
        tail = " " + tail
    return text[: m.start()] + stmt + tail, f"set rate law of {rid}"


def _op_set_initial(text, op, where):
    sid = (op.get("id") or "").strip()
    value = op.get("antimony")
    if isinstance(value, str):
        value = value.strip().rstrip(";")
    if not sid or value is None or value == "":
        raise OperationError(
            f"{where}: set_initial needs `id` and `antimony` — the new value, "
            "a number like '5' (estimated from there; exactly 0 or 1 stays "
            "fixed) or a symbol like 'A_init'. NOTHING was applied."
        )
    _reject_semicolons(where, antimony=str(value))
    # Found on the MASK, rewritten in the original: the terminating `;` is optional here exactly as
    # it is in `assigned_symbols`, because a pattern that required it reported a symbol as unknown
    # and listed it as known in the same sentence.
    masked = mask_comments(text)
    rxn_spans = [(m.start(), m.end()) for m in _RXN_STMT.finditer(masked)]
    pat = re.compile(rf"(?<![\w.]){re.escape(sid)}\s*=(?![>=])\s*[^;\n]+?\s*(?:;|$)", re.MULTILINE)

    # An EVENT assigns the same symbol at a trigger time (`E1: at (time > 10): A = 5;`), and
    # rewriting THAT leaves the real initial value untouched while reporting success — the exact
    # class of silent wrong result the comment masking removed. Skip event statements, and of the
    # remaining candidates take the LAST: a symbol assigned twice at top level takes the later
    # value, so rewriting the first would be overwritten by the second.
    def _in_event(pos: int) -> bool:
        # Back to the true STATEMENT start — `;` only, never the nearest newline. An event may
        # span lines (`E1: at (time > 10):\n  A = 5,\n  B = 3;`), and stopping at the newline made
        # the slice for `B` just "  B", missing the trigger entirely, so the event assignment was
        # rewritten and reported as the initial value. The trigger may also be parenless
        # (`E1: at time > 10: A = 5;`), so `\(` cannot be required.
        stmt_start = masked.rfind(";", 0, pos) + 1
        return re.search(r"(?<![\w.])at(?![\w.])", masked[stmt_start:pos]) is not None

    candidates = [
        m
        for m in pat.finditer(masked)
        if not any(s <= m.start() < e for s, e in rxn_spans) and not _in_event(m.start())
    ]
    if candidates:
        m = candidates[-1]
        return text[: m.start()] + f"{sid} = {value};" + text[m.end() :], f"set {sid} = {value}"
    # NOT assigned anywhere. Creating the assignment silently would mint a parameter the caller
    # never asked for: the reconcile gives it a row, the fit estimates it, BIC pays for it, and
    # the summary says "set". A typo'd symbol must fail like every other unknown id does.
    known = assigned_symbols(text)
    close = closest_symbols(sid, known)
    hint = f" Did you mean {' or '.join(repr(c) for c in close)}?" if close else ""
    # The `*_init` case is worth naming explicitly: the reconcile MINTS an initial-value parameter
    # for every species that needs one, and those show up in get_model()/get_parameters() — the
    # render — while these operations apply to the SOURCE. A caller reading `X1_init` off a
    # parameter table is not confused about its model; it is seeing a symbol this text genuinely
    # does not contain, so say which surface owns it.
    minted = (
        ""
        if not sid.endswith("_init")
        else f" Note {sid!r} looks like an initial-value parameter the reconcile mints from the "
        "model: those live in the parameter table, not in the model text — set its bounds "
        "with `parameters`, or set the species' value here using the species' own name."
    )
    raise OperationError(
        f"{where}: nothing named {sid!r} is assigned in the model, so there is no initial value "
        f"to set.{hint} Assigned symbols: {_fmt_ids(known)}.{minted} To introduce a new quantity, "
        "add the reaction that uses it — the reconcile mints its parameter row. "
        "NOTHING was applied.",
        cause="unknown_symbol",
    )


def _op_rename(text, op, where):
    old, new = (op.get("old") or "").strip(), (op.get("new") or "").strip()
    if not old or not new:
        raise OperationError(f"{where}: rename needs `old` and `new`. NOTHING was applied.")
    # BOTH sides, both checks. `new` is substituted into every statement, so a non-identifier can
    # split one and unaddress the reaction it lands in; `old` is the SEARCH side and matches across
    # the whole model, so a keyword or a fragment renames things the caller never meant.
    # `{old: 'const'}` rewrote the declaration `const k1, k2;` and reported success; symmetrically
    # `{new: 'const'}` produces `const const;`, which fails the next reconcile with an opaque
    # Antimony error and burns the whole batch.
    for field, value in (("old", old), ("new", new)):
        if not re.fullmatch(r"[A-Za-z_]\w*", value):
            raise OperationError(
                f"{where}: `{field}` must be a plain identifier (letters, digits, underscore; not "
                f"starting with a digit), not {value!r} — rename substitutes across every rate "
                "law and declaration. NOTHING was applied.",
                cause="invalid_identifier",
            )
        if value in _PROTECTED_SYMBOLS:
            why = (
                "simulation time — renaming it detaches every gate from the clock while the "
                "model still compiles, so the inputs go constant silently"
                if value == "time"
                else "Antimony syntax or a built-in function, not a symbol of yours"
            )
            raise OperationError(
                f"{where}: {value!r} is {why}. Using it as `{field}` would rewrite the model's "
                "own vocabulary. Name the quantity you mean. NOTHING was applied.",
                cause="invalid_identifier",
            )
    # Counted and substituted on the MASK: a symbol occurring only in a comment is not in the
    # model, and renaming it would report `renamed x -> y (1 occurrences)` for a call that changed
    # no model text at all, then pay a full multistart to come back `no_change`.
    masked = mask_comments(text)
    pat = re.compile(rf"(?<![\w.]){re.escape(old)}\b")
    spans = [m.span() for m in pat.finditer(masked)]
    if not spans:
        raise OperationError(
            f"{where}: symbol {old!r} does not occur in the model. NOTHING was applied.",
            cause="rename_target_missing",
        )
    # A collision MERGES two symbols, and whether that is legitimate depends on WHAT they are:
    #   * two PARAMETERS -> a real move. Measured: a winning session merged two rate constants this
    #     way and priced it itself at dBIC -2.668. Allowed.
    #   * a SPECIES onto another species -> structural corruption reported as success, because
    #     `A => B` silently becomes `B => B`. Refused.
    # Species are exactly the symbols appearing in a STOICHIOMETRY; parameters appear only in rate
    # laws and assignments, so the reaction statements tell the two apart.
    in_stoich: set[str] = set()
    for m in _RXN_STMT.finditer(masked):
        in_stoich |= set(re.findall(r"[A-Za-z_]\w*", m.group("stoich")))
    collides = re.search(rf"(?<![\w.]){re.escape(new)}\b", masked)
    # A REACTION id is the same collision arriving through the other door, and `in_stoich` cannot
    # see it: reaction ids never appear in a stoichiometry. So `rename(k1 -> r1)` against
    # `r1: A -> B; k1*A;` passed both tests and produced `r1: A -> B; r1*A;` — in Antimony a
    # reaction id in an expression denotes that reaction's RATE, so the law now references itself.
    # It committed, and the summary reported `renamed k1 -> r1 (1 occurrences)` as a success.
    targets = rule_target_symbols(text)
    if collides and (old in targets or new in targets):
        raise OperationError(
            f"{where}: {new!r} already carries an assignment rule, so this rename would merge a "
            "symbol into a rule TARGET — every use of the old name would start reading a "
            "rule-computed value (a rate law reading a measurement readout instead of its own "
            "kinetic parameter, and the model still compiles). Pick another name, or "
            "remove_assignment first. NOTHING was applied.",
            cause="rule_target_merge_refused",
        )
    rxn_ids = reaction_ids(text)
    if collides and (old in rxn_ids or new in rxn_ids):
        raise OperationError(
            f"{where}: {new!r} already exists and one of these is a REACTION id, so this rename "
            "would make a rate law reference a reaction — Antimony reads a reaction id in an "
            "expression as that reaction's rate, so the law would refer to itself. Rename the "
            "reaction with its own operation, or pick another name. NOTHING was applied.",
            cause="reaction_id_merge_refused",
        )
    if collides and (old in in_stoich or new in in_stoich):
        raise OperationError(
            f"{where}: {new!r} already exists and one of these is a SPECIES, so this rename "
            f"would merge two pools — `X => {new}` collapsing into `{new} => {new}`. Change the "
            "stoichiometry with set_rate_law/add_reaction/remove_reaction if that is the intent. "
            "NOTHING was applied.",
            cause="species_merge_refused",
        )
    out = []
    prev = 0
    for start, end in spans:
        out.append(text[prev:start])
        out.append(new)
        prev = end
    out.append(text[prev:])
    return "".join(out), f"renamed {old} -> {new} ({len(spans)} occurrences)"


def _op_set_assignment(text, op, where):
    """Write or replace an ASSIGNMENT RULE — ``id := expr;``.

    The op behind measurement-mapping (``readout_*``) symbols: a blinded problem ships each
    observable formula referencing ``readout_<observableId>`` instead of the reference model's own
    species combination, and the caller must say what that readout IS in terms of its model —
    which in Antimony/SBML is an assignment rule, not an initial value (``set_initial`` writes
    ``id = value``, a t=0 statement; this writes ``id := expr``, which holds at ALL times).
    General-purpose beyond that: any derived/algebraic quantity is the same statement.
    """
    sid = (op.get("id") or "").strip()
    expr = op.get("antimony")
    if isinstance(expr, str):
        expr = expr.strip().rstrip(";")
    if not sid or not re.fullmatch(r"[A-Za-z_]\w*", sid) or expr is None or expr == "":
        raise OperationError(
            f"{where}: set_assignment needs `id` (the symbol to define) and `antimony` (the "
            "expression it equals at all times), e.g. "
            "{op: 'set_assignment', id: 'readout_001', antimony: 'X2 + X3'}. "
            "NOTHING was applied."
        )
    expr = str(expr)
    # A whole `id := expr` (or `id = expr`) statement: keep its right-hand side if it assigns `id`.
    lhs = re.match(r"([A-Za-z_]\w*)\s*:?=(?![=>])", expr)
    if lhs and lhs[1] != sid:
        raise OperationError(
            f"{where}: `antimony` assigns {lhs[1]!r}, but `id` is {sid!r}. `antimony` is the "
            "right-hand side only, e.g. `X2 + X3`; the symbol goes in `id`. NOTHING was applied.",
            cause="id_mismatch",
        )
    if lhs:
        expr = expr[lhs.end() :].strip()
    if not expr or _STATEMENT_MARK.search(expr):
        raise OperationError(
            f"{where}: `antimony` is the right-hand side only, e.g. `X2 + X3`; the symbol goes in "
            "`id`. NOTHING was applied.",
            cause="statement_not_expression",
        )
    _reject_semicolons(where, antimony=expr)
    if sid in _PROTECTED_SYMBOLS:
        raise OperationError(
            f"{where}: {sid!r} is a reserved Antimony/SBML name and cannot be assigned. "
            "NOTHING was applied.",
            cause="reserved_id",
        )
    masked = mask_comments(text)
    rxn_spans = [(m.start(), m.end()) for m in _RXN_STMT.finditer(masked)]
    # A STATE cannot also be assigned: a species produced/consumed by a reaction with an
    # assignment rule on top is an over-determined model libantimony rejects opaquely.
    in_stoich = _stoichiometry_symbols(text)
    if sid in reaction_ids(text):
        # Antimony keeps ONE symbol table and reads a reaction id in an expression as that
        # reaction's rate, so assigning to one is a redefinition -- the same collision
        # `_op_add_reaction` refuses, arriving through the other door.
        raise OperationError(
            f"{where}: {sid!r} is a REACTION id in this model, and Antimony keeps one symbol "
            "table — assigning to it is a redefinition. Pick another symbol. NOTHING was "
            "applied.",
            cause="id_collision",
        )
    if sid in in_stoich:
        # The remedy has to name the way OUT, not just the refusal: when the blocked symbol is a
        # readout the caller wired into a reaction, "assign a different symbol" is unfollowable —
        # the observable references THAT name and no other.
        raise OperationError(
            f"{where}: {sid!r} is a SPECIES in this model's reactions — a state integrated by the "
            "ODEs cannot also carry an assignment rule (the model would be over-determined). "
            f"If {sid!r} is meant to be a measured READOUT, it must not be a reaction "
            "participant: remove it from the stoichiometry (remove_reaction / set_rate_law, or "
            "re-author) and assign it instead. If it is meant to be a state, leave it alone and "
            "assign a different symbol. NOTHING was applied.",
            cause="assign_to_species_refused",
        )
    pat = re.compile(rf"(?<![\w.]){re.escape(sid)}\s*:=\s*[^;\n]+?\s*(?:;|$)", re.MULTILINE)
    hits = [m for m in pat.finditer(masked) if not any(s <= m.start() < e for s, e in rxn_spans)]
    if hits:
        m = hits[-1]
        return (
            text[: m.start()] + f"{sid} := {expr};" + text[m.end() :],
            f"set assignment {sid} := {expr}",
        )
    # New rule: insert before the model block's `end` (after the last reaction when there is one,
    # so the rule reads in context; the block header otherwise).
    search_from = rxn_spans[-1][1] if rxn_spans else _model_block_start(masked)
    end_tok = re.compile(r"(?<![\w.])end(?![\w.])").search(masked, search_from)
    if end_tok is None:
        # No `end`: a bare-body model. Append, with the same separator guard `_op_add_reaction`
        # uses -- without it the rule is glued onto a previous unterminated statement
        # (`B = 2` + `  readout_x := ...` on one line) and libantimony fails to parse, reported
        # by this op as a success.
        sep = "" if text.endswith("\n") or not text else "\n"
        return text + f"{sep}  {sid} := {expr};\n", f"set assignment {sid} := {expr}"
    return (
        text[: end_tok.start()] + f"  {sid} := {expr};\n" + text[end_tok.start() :],
        f"set assignment {sid} := {expr}",
    )


def _op_remove_assignment(text, op, where):
    """Delete an assignment rule — the inverse of `set_assignment`.

    Needed because nothing else can undo one: `sync_parameter_table` strips rule targets from the
    parameter table and `prune_orphans` deliberately never removes a rule target ("a rule's TARGET
    is assigned by that rule, so it is never an orphan"), so a rule created with a mistyped id
    would otherwise survive every later edit and could only be cleared by re-authoring the whole
    model — the cost this operation surface exists to remove. It is also the only way out for a
    symbol the caller now needs as a dynamic STATE: set_assignment refuses a species, add_reaction
    would over-determine it, so the rule has to go first.
    """
    sid = (op.get("id") or "").strip()
    if not sid:
        raise OperationError(
            f"{where}: remove_assignment needs `id` — the assigned symbol. NOTHING was applied."
        )
    masked = mask_comments(text)
    pat = re.compile(rf"(?<![\w.]){re.escape(sid)}\s*:=\s*[^;\n]+?\s*(?:;|$)", re.MULTILINE)
    hits = list(pat.finditer(masked))
    if not hits:
        have = sorted(rule_target_symbols(text))
        listed = f" Assignment rules here: {', '.join(have)}." if have else " There are none."
        raise OperationError(
            f"{where}: no assignment rule for {sid!r}.{listed} NOTHING was applied.",
            cause="unknown_symbol",
        )
    m = hits[-1]
    cut = text[: m.start()] + text[m.end() :]
    line_start, line_end = _line_bounds(cut, min(m.start(), max(len(cut) - 1, 0)))
    if not cut[line_start:line_end].strip():
        cut = cut[:line_start] + cut[min(line_end + 1, len(cut)) :]
    return cut, f"removed assignment {sid}"


_HANDLERS = {
    "add_reaction": _op_add_reaction,
    "remove_reaction": _op_remove_reaction,
    "set_rate_law": _op_set_rate_law,
    "set_initial": _op_set_initial,
    "set_assignment": _op_set_assignment,
    "remove_assignment": _op_remove_assignment,
    "rename": _op_rename,
}


def apply_model_edits(
    source: str, edits: "list[dict[str, Any]]", *, where: str = "make_changes"
) -> tuple[str, list[str]]:
    """Apply a batch of structured operations to the model SOURCE.

    Returns ``(new_source, summary)`` — the summary is one line per operation, which is also the
    pre-classified diff (add / remove / rate law / initial / rename) that an analysis would
    otherwise have to reconstruct from text.
    """
    if not isinstance(edits, (list, tuple)) or not edits:
        raise OperationError(
            f"{where}: `model_edits` must be a non-empty list of operations ({', '.join(OPS)})."
        )
    text = source or ""
    summary: list[str] = []
    for i, op in enumerate(edits):
        label = f"{where}: model_edits[{i}]"
        if not isinstance(op, dict):
            raise OperationError(f"{label} is not an object. NOTHING was applied.")
        name = (op.get("op") or "").strip()
        handler = _HANDLERS.get(name)
        if handler is None:
            raise OperationError(
                f"{label}: unknown op {name!r}; use one of {', '.join(OPS)}. NOTHING was applied.",
                cause="unknown_op",
            )
        text, msg = handler(text, op, label)
        summary.append(msg)
    # BATCH-level, because the per-op guards see only the text as it stood when they ran: a
    # `set_assignment` followed by an `add_reaction` that puts the same symbol in a stoichiometry
    # is accepted one op at a time and yields an over-determined model — a state integrated by the
    # ODEs that also carries an assignment rule — which libantimony rejects with a message naming
    # neither op. Checked once at the end, where the whole batch is visible.
    over_determined = rule_target_symbols(text) & _stoichiometry_symbols(text)
    if over_determined:
        names = ", ".join(repr(s) for s in sorted(over_determined))
        raise OperationError(
            f"{where}: {names} would be BOTH a reaction participant and an assignment-rule "
            "target, which is an over-determined model. A measured readout must not appear in a "
            "stoichiometry; a dynamic state must not carry a rule (use remove_assignment). "
            "NOTHING was applied.",
            cause="assign_to_species_refused",
        )
    return text, summary
