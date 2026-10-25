"""Causal tracing: which forms of a model's agents a subject moves, and which way, read off the
network BioNetGen expands from the model with every molecule tagged (PEtab-MCP#136), as the
tracing of Fröhlich et al. (MSB 2023, doi:10.15252/msb.202210988) follows a signal through tag
sites its rules copy.

Every monomer gets a site ``__tag``: ``o`` untraced, ``p`` there because the subject rose (more
of it), ``n`` less of it. The subject's own molecules are tagged ``p`` by a rule, wherever the
network forms them. Each rule keeps its untagged form and gets one variant per driver and sign:

- a molecule on the rule's left (a catalyst, a substrate, a binding partner) tagged ``s`` hands
  ``s`` to each molecule the rule changes the state of or makes. Never through a binding alone:
  each molecule keeps its own tag there, since a tag passed on at a binding ran back up two-step
  mechanisms (RPS6 tagged EGF);
- a partner a bond wildcard hides (INDRA writes "E bound to L" ``E(l=ANY)``) is written out as
  each molecule that site binds in the rules. Its own bond is untouched, so it is context: it
  hands its tag to what the rule changes, makes or frees, a release included (cAMP freeing PKA's
  catalytic subunit; L's receptor autophosphorylating);
- a regulator a rate reads by an observable, which no pattern holds, joins the variant's pattern
  and hands ``s`` times the kind its row records (:attr:`mechanisms.Emission.derived`, as
  :func:`regulation.resign` signed it): ``inhibited_by`` flips it. ``modulated_by`` claims no
  direction, so its variant exists only where any effect counts (``unsigned``).

Which tagged species and reactions exist is then what BioNetGen's matching allows, never a
hand-written reading of the rows. A form is RAISED where a variant driven at ``p`` makes more of
it, net over its forms, or one driven at ``n`` takes less of it away; LOWERED the other way round.
Only the net change counts, never a tag on its own: EGF phosphorylating EGFR tags it and leaves
total EGFR as it is (read by its tag, "EGF increases EGFR" held). A driver taken away by its own
reaction is its own decay, not a decrease (more pS6, more of it dephosphorylated).

Less of something is no tag on any molecule, so a decrease by mass action (a competitor taking a
partner, an antagonist occupying a site) stops where it happens. One more pass traces on from
what a traced driver consumed, seeded with the driver's tag and read the other way round: X
raises I, I takes W, W raises Y reads as X lowers Y. It runs without the rule that took W, whose
flux is the driver's own and traced already, and without each rule that what W became takes as W
did, which goes on regardless (EGF binding free EGFR{pY1068} takes no kinase away: the bound one
phosphorylates AKT1 as the free one did). It does not read a target its seeds are forms of: what
comes back to that form from them is their own cycle, and the step that took them was read in the
first pass (less free pAKT, dephosphorylated and phosphorylated again, read as fujita's false
"RPS6 decreases AKT1{pS473}"). On the formulations #136 measured, that pass held isensee's Rp8
antagonist and boehm's STAT5B competition and held nothing false; a second pass held nothing
more and added a wrong answer, so there is one.

A tag follows a molecule, not the form a later rule reads: E phosphorylating A{pS1} at S2 tags
A, and a rule A{pS1} catalyses takes the tag on, though E leaves the amount of A{pS1} as it is.
Deliberately left so: a false acceptance costs the agent less than a false refusal.

The model traced is the one the rows build as written, seeded as the build seeds it before its
reduction (:func:`problem_sbml.to_antimony`), never reduced: the reduction takes molecules apart
into blocks that are no longer the statements' agents. The subject is tagged by a rule, so
BioNetGen expands the model once per network, the untagged part with the tagged.
"""

from __future__ import annotations

import re
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import pysb
from pysb import ANY, WILD
from pysb.bng import generate_equations
from pysb.pattern import get_bonds_in_pattern, match_complex_pattern

SITE = "__tag"
_STATES = ["o", "p", "n"]
_FLIP = {"p": "n", "n": "p"}
#: A recorded regulator kind's sign; ``modulated_by`` claims none.
_SIGN = {"activated_by": 1, "inhibited_by": -1, "modulated_by": None}


@dataclass
class Moved:
    """Which ways a trace moves a form, and the rows whose reactions move it each way (``1``
    raised, ``-1`` lowered)."""

    raised: bool = False
    lowered: bool = False
    rows: dict[int, list[str]] = field(default_factory=lambda: {1: [], -1: []})

    def add(self, way: int, rows) -> None:
        if way > 0:
            self.raised = True
        else:
            self.lowered = True
        self.rows[way] = list(dict.fromkeys([*self.rows[way], *rows]))


@dataclass
class Trace:
    """What tracing a subject found: each target's :class:`Moved`, the agents the trace reached,
    and the networks' sizes and times."""

    moved: dict[Any, Moved]
    reached: list[str]
    sizes: list[dict[str, Any]]


def trace(model: pysb.Model, emissions, derived: dict[str, dict], subject, targets: dict,
          *, unsigned: bool = False) -> Trace:
    """Trace ``subject`` (an INDRA agent) through ``model``, the rows' model as written and seeded,
    for each of ``targets`` (key -> ``(agent, active, any_form)``: an active form's activity also
    moves with a regulator of a reaction it catalyses, and ``any_form`` counts any change of any
    of its forms, a modification included, not the net). ``emissions`` say which rules each row built and
    ``derived`` (prefix -> kinds) what each row's regulators record. ``unsigned``: a
    ``modulated_by`` regulator moves its reaction both ways. Raises what BioNetGen raises."""
    tagger = _Tagger(model, emissions, derived, unsigned)
    m, meta = tagger.build(subject=subject)
    sizes = [_expand(m)]
    moved = {key: Moved() for key in targets}
    _read(m, meta, targets, moved, tagger.row_of, way=1)
    reached = sorted({mp.monomer.name for sp in m.species for mp in sp.monomer_patterns
                      if mp.site_conditions.get(SITE) != "o"})
    # A junction is traced on, one network per rule that consumed one, without that rule and
    # without each rule what it became takes as it did: there, the products carry on its part.
    takes: dict[int, set[str]] = {}
    for rx in m.reactions:
        for rule in _rules(rx):
            if base := meta[rule][3] if rule in meta else tagger.base_of.get(rule):
                for i in rx["reactants"]:
                    takes.setdefault(i, set()).add(base)
    groups: dict[str, tuple[dict, set[str]]] = {}
    for i, s, base, products in _consumed(m, meta):
        seeds, blocked = groups.setdefault(base, ({}, {base}))
        seeds[(str(m.species[i]), s)] = (m.species[i], s)
        blocked |= takes.get(i, set()) & set().union(*(takes.get(j, set()) for j in products))
    for seeds, blocked in groups.values():
        mm, mt = tagger.build(seeds=list(seeds.values()), blocked=blocked)
        sizes.append(_expand(mm))
        _read(mm, mt, targets, moved, tagger.row_of, way=-1,
              seeds=range(len(model.initials), len(mm.initials)))
    return Trace(moved, reached, sizes)


def _expand(m: pysb.Model) -> dict[str, Any]:
    from .problem_sbml import MAX_COMPLEX

    t0 = time.monotonic()
    generate_equations(m, max_agg=MAX_COMPLEX)
    return {"species": len(m.species), "reactions": len(m.reactions),
            "seconds": round(time.monotonic() - t0, 2)}


class _Tagger:
    def __init__(self, model: pysb.Model, emissions, derived: dict[str, dict], unsigned: bool):
        self.src = model
        self.emissions = list(emissions)
        self.derived = derived
        self.unsigned = unsigned
        self.prefix_of = {rule: e.prefix for e in self.emissions for rule in e.rules}
        #: Each tagged rule's row, and each untagged rule's direction, by name: the same names in
        #: every pass.
        self.row_of: dict[str, str] = {}
        self.base_of: dict[str, str] = {}
        self.partners = _bond_map(model)

    # ------------------------------------------------------------------------------ conversion
    def _mp(self, mp, tag=None, **sites):
        sc = {**mp.site_conditions, **sites}
        if tag is not None:
            sc[SITE] = tag
        return pysb.MonomerPattern(self.mon[mp.monomer.name], sc, mp.compartment)

    def _cp(self, cp, tags=None):
        tags = tags or {}
        return pysb.ComplexPattern(
            [self._mp(mp, tags.get(i)) for i, mp in enumerate(cp.monomer_patterns)],
            cp.compartment, match_once=cp.match_once,
        )

    @staticmethod
    def _rp(cps):
        return pysb.ReactionPattern([c for c in cps if c is not None])

    def _expr(self, e):
        return e.xreplace({a: self.comp[a.name]
                           for a in e.atoms(pysb.Parameter, pysb.Expression, pysb.Observable)})

    # ----------------------------------------------------------------------------------- build
    def build(self, subject=None, seeds=None, blocked=()) -> tuple[pysb.Model, dict]:
        """The tagged model, the subject's molecules tagged by a rule, or each of ``seeds``
        (``(species, tag)`` of an earlier pass) present with every molecule tagged; the rules
        ``blocked`` taken by untraced molecules alone. Returns it and each variant's
        ``(rule, tag it hands on, driver monomer or None, direction)``."""
        from .mechanisms import agent_pattern

        src = self.src
        m = pysb.Model(name="traced", _export=False)
        self.mon, self.comp = {}, {}
        for mon in src.monomers:
            new = pysb.Monomer(mon.name, [*mon.sites, SITE],
                               {**{k: list(v) for k, v in mon.site_states.items()}, SITE: _STATES},
                               _export=False)
            m.add_component(new)
            self.mon[mon.name] = new
        for p in src.parameters:
            self.comp[p.name] = pysb.Parameter(p.name, p.value, _export=False)
            m.add_component(self.comp[p.name])
        for o in src.observables:
            self.comp[o.name] = pysb.Observable(
                o.name, self._rp([self._cp(c) for c in o.reaction_pattern.complex_patterns]),
                o.match, _export=False,
            )
            m.add_component(self.comp[o.name])
        pending = list(src.expressions)
        while pending:  # in the order they read each other
            ready = [e for e in pending if all(
                a.name in self.comp
                for a in e.expr.atoms(pysb.Parameter, pysb.Expression, pysb.Observable))]
            if not ready:
                raise ValueError(f"expressions read each other in a cycle: {pending}")
            for e in ready:
                self.comp[e.name] = pysb.Expression(e.name, self._expr(e.expr), _export=False)
                m.add_component(self.comp[e.name])
            pending = [e for e in pending if e not in ready]
        for ic in src.initials:
            cp = ic.pattern
            m.add_initial(pysb.Initial(self._cp(cp, dict.fromkeys(range(len(cp.monomer_patterns)), "o")),
                                       self.comp[ic.value.name], _export=False))
        one = pysb.Parameter("__traced", 1.0, _export=False)
        m.add_component(one)
        meta: dict[str, tuple] = {}
        if subject is not None:
            # The subject's own molecule (its pattern's first; the rest are partners it is bound
            # to), wherever the network forms it.
            first, *partners = agent_pattern(m, subject)

            def subject_tagged(tag):
                return pysb.ComplexPattern([self._mp(first, tag), *partners], None)

            m.add_component(pysb.Rule("__trace", pysb.RuleExpression(
                self._rp([subject_tagged("o")]), self._rp([subject_tagged("p")]), False),
                one, _export=False))
        seeded = {str(ic.pattern) for ic in m.initials}
        for sp, t in seeds or ():
            cp = self._cp(sp, dict.fromkeys(range(len(sp.monomer_patterns)), t))
            if str(cp) not in seeded:
                m.add_initial(pysb.Initial(cp, one, _export=False))
                seeded.add(str(cp))
        regulators = self._regulators()
        n = 0
        for rule in src.rules:
            directions = [(rule.reactant_pattern, rule.product_pattern, rule.rate_forward, "f")]
            if rule.is_reversible:
                directions.append((rule.product_pattern, rule.reactant_pattern, rule.rate_reverse, "r"))
            for lhs, rhs, rate, d in directions:
                base = f"{rule.name}__{d}"
                rate_new = self.comp[rate.name]
                L = [c for c in lhs.complex_patterns if c is not None]
                R = [c for c in rhs.complex_patterns if c is not None]
                pairs = _pairs(L, R)
                inv = {v: k for k, v in pairs.items()}
                made = _made(L, R, pairs)

                def add(name, left, right, about=None):
                    m.add_component(pysb.Rule(name, pysb.RuleExpression(
                        self._rp(left), self._rp(right), False), rate_new, _export=False))
                    if rule.name in self.prefix_of:
                        self.row_of[name] = self.prefix_of[rule.name]
                    if about is not None:
                        meta[name] = (rule.name, *about, base)

                def untraced(cps):
                    return [self._cp(c, dict.fromkeys(range(len(c.monomer_patterns)), "o"))
                            for c in cps]

                def before(moved, but=None):
                    """Left tags: BioNetGen sets a state only where the reactant states one, so
                    a molecule a variant tags is untraced before it (every form exists
                    untraced, so each driver reaches it)."""
                    tags = {k: {} for k in range(len(L))}
                    for pc, pm in moved:
                        if (pc, pm) in inv and inv[(pc, pm)] != but:
                            lc, lm = inv[(pc, pm)]
                            tags[lc][lm] = "o"
                    return tags

                self.base_of[base] = base
                if base in blocked:
                    add(base, untraced(L), untraced(R))
                    continue
                created = {(cj, mj) for cj, c in enumerate(R) for mj in range(len(c.monomer_patterns))
                           if (cj, mj) not in inv}
                add(base, [self._cp(c) for c in L],
                    [self._cp(c, {mj: "o" for cj, mj in created if cj == k}) for k, c in enumerate(R)])
                for ci, c in enumerate(L):
                    for mi, mp in enumerate(c.monomer_patterns):
                        counterpart = pairs.get((ci, mi))
                        for s in ("p", "n"):
                            n += 1
                            tags = before(made, but=(ci, mi))
                            tags[ci][mi] = s
                            right = []
                            for k, rc in enumerate(R):
                                rt = {pm: s for pc, pm in made if pc == k}
                                if counterpart and counterpart[0] == k:
                                    rt[counterpart[1]] = s
                                right.append(self._cp(rc, rt))
                            add(f"{base}__t{n}", [self._cp(x, tags[k]) for k, x in enumerate(L)],
                                right, (s, mp.monomer.name))
                        if counterpart is None:
                            continue
                        cj, mj = counterpart
                        mq = R[cj].monomer_patterns[mj]
                        for site, lv in mp.site_conditions.items():
                            hl, hr = _hidden(lv), _hidden(mq.site_conditions.get(site))
                            if hl is None or hr is None or hl[0] is not hr[0]:
                                continue
                            b = 1 + max([0, *get_bonds_in_pattern(L[ci]), *get_bonds_in_pattern(R[cj])])
                            moved = set(made) | _freed(L, R, pairs)
                            for pmon, psite in sorted(self.partners.get((mp.monomer.name, site), ())):
                                for s in ("p", "n"):
                                    n += 1
                                    partner = pysb.MonomerPattern(self.mon[pmon], {psite: b, SITE: s}, None)
                                    tags = before(moved)
                                    left = [self._cp(x, tags[k]) for k, x in enumerate(L)]
                                    left[ci] = _joined(left[ci], mi, site, hl[1], b, partner)
                                    right = [self._cp(x, {pm: s for pc, pm in moved if pc == k})
                                             for k, x in enumerate(R)]
                                    right[cj] = _joined(right[cj], mj, site, hr[1], b, partner)
                                    add(f"{base}__t{n}", left, right, (s, pmon))
                for obs, k in regulators.get(rule.name, ()):
                    rcp = obs.reaction_pattern.complex_patterns[0]
                    for s in ("p", "n"):
                        for out in _handed(s, k, self.unsigned):
                            n += 1
                            tags = before(made)
                            add(f"{base}__t{n}",
                                [*(self._cp(x, tags[kk]) for kk, x in enumerate(L)), self._cp(rcp, {0: s})],
                                [*(self._cp(x, {pm: out for pc, pm in made if pc == kk})
                                   for kk, x in enumerate(R)), self._cp(rcp, {0: s})],
                                (out, None))
        return m, meta

    def _regulators(self) -> dict[str, list[tuple[pysb.Observable, int | None]]]:
        """Rule name -> each observable its rate reads by a regulator's name, with the sign its
        row records for that regulator."""
        from .regulation import _observables, position
        from .statements import REGULATOR_NAME

        named = re.compile(rf"_({REGULATOR_NAME.pattern})$")
        out: dict[str, list] = {}
        for e in self.emissions:
            derived = self.derived.get(e.prefix) or {}
            for name in e.rules:
                rule = self.src.rules[name]
                for obs in _observables(rule.rate_forward) | _observables(rule.rate_reverse):
                    if said := named.search(obs.name):
                        kind = re.sub(r"_\d+$", "", said.group(1))
                        out.setdefault(name, []).append(
                            (obs, _SIGN[derived.get(position(said.group(1)), kind)]))
        return out


def _handed(s: str, k: int | None, unsigned: bool) -> list[str]:
    if k is None:
        return ["p", "n"] if unsigned else []
    return [s if k > 0 else _FLIP[s]]


# ------------------------------------------------------------------------------------ patterns
def _flat(cps):
    return [(ci, mi, mp) for ci, c in enumerate(cps) for mi, mp in enumerate(c.monomer_patterns)]


def _pairs(L, R) -> dict[tuple[int, int], tuple[int, int]]:
    """Left molecule -> right molecule, by type in order, as BioNetGen maps them."""
    used, pairs = set(), {}
    for ci, mi, mp in _flat(L):
        for cj, mj, mq in _flat(R):
            if (cj, mj) not in used and mq.monomer.name == mp.monomer.name:
                pairs[(ci, mi)] = (cj, mj)
                used.add((cj, mj))
                break
    return pairs


def _made(L, R, pairs) -> list[tuple[int, int]]:
    """Right molecules the rule makes or changes the state of (a modification, an activity, a
    location): what a tag is set on, as the paper's channel site is set on the MEK a RAF
    phosphorylates."""
    inv = {v: k for k, v in pairs.items()}
    out = []
    for cj, mj, mq in _flat(R):
        if (cj, mj) not in inv:
            out.append((cj, mj))
        elif _states(L[inv[(cj, mj)][0]].monomer_patterns[inv[(cj, mj)][1]]) != _states(mq):
            out.append((cj, mj))
    return out


def _freed(L, R, pairs) -> set[tuple[int, int]]:
    """Right molecules the rule unbinds at a site bound on the left."""
    out = set()
    for (ci, mi), (cj, mj) in pairs.items():
        right = R[cj].monomer_patterns[mj].site_conditions
        for site, v in L[ci].monomer_patterns[mi].site_conditions.items():
            if _bonds(v) and not _bonds(right.get(site)) and _hidden(right.get(site)) is None:
                out.add((cj, mj))
    return out


def _states(mp) -> dict[str, str]:
    """A molecule's internal states, bonds left out."""
    out = {}
    for k, v in mp.site_conditions.items():
        if isinstance(v, str):
            out[k] = v
        elif isinstance(v, tuple) and len(v) == 2 and isinstance(v[0], str):
            out[k] = v[0]
    return out


def _bonds(v) -> list[int]:
    if isinstance(v, int) and not isinstance(v, bool):
        return [v]
    if isinstance(v, tuple) and len(v) == 2 and isinstance(v[1], int):
        return [v[1]]
    if isinstance(v, list):
        return [b for b in v if isinstance(b, int)]
    return []


def _hidden(v):
    """``(wildcard, state)`` for a bond wildcard (``ANY``, ``WILD``, or a state with one), else
    ``None``."""
    if v is ANY or v is WILD:
        return v, None
    if isinstance(v, tuple) and len(v) == 2 and isinstance(v[0], str) and (v[1] is ANY or v[1] is WILD):
        return v[1], v[0]
    return None


def _joined(cp, mi: int, site: str, state, bond: int, partner) -> pysb.ComplexPattern:
    """``cp`` with molecule ``mi``'s ``site`` bound (``bond``, keeping its ``state``) to
    ``partner``, joined to it."""
    mps = list(cp.monomer_patterns)
    mp = mps[mi]
    mps[mi] = pysb.MonomerPattern(
        mp.monomer, {**mp.site_conditions, site: bond if state is None else (state, bond)},
        mp.compartment,
    )
    return pysb.ComplexPattern([*mps, partner], cp.compartment, match_once=cp.match_once)


def _bond_map(model: pysb.Model) -> dict[tuple[str, str], set[tuple[str, str]]]:
    """(monomer, site) -> each (monomer, site) it binds, from the bonds the rules and the seeds
    write: every bond the network forms is one of those."""
    cps = [c for r in model.rules for rp in (r.reactant_pattern, r.product_pattern)
           for c in rp.complex_patterns if c is not None]
    cps += [ic.pattern for ic in model.initials]
    out: dict[tuple[str, str], set] = {}
    for cp in cps:
        ends: dict[int, list] = {}
        for mp in cp.monomer_patterns:
            for site, v in mp.site_conditions.items():
                for b in _bonds(v):
                    ends.setdefault(b, []).append((mp.monomer.name, site))
        for pair in ends.values():
            if len(pair) == 2:
                a, c = pair
                out.setdefault(a, set()).add(c)
                out.setdefault(c, set()).add(a)
    return out


# ------------------------------------------------------------------------------------- reading
def _rules(rx) -> list[str]:
    return list(rx["rule"]) if isinstance(rx["rule"], (list, tuple)) else [rx["rule"]]


def _read(m, meta, targets, moved, row_of, way: int, seeds=()) -> None:
    """Add to ``moved`` the ways ``m``'s network moves each target: by the net change of its forms
    in each traced reaction, ``way`` -1 for a junction pass (tag ``p`` there is less of it). A tag
    is not read on its own: EGF phosphorylating EGFR tags it and leaves total EGFR as it is. A
    junction pass whose ``seeds`` (species indices: BioNetGen lists a model's seeds first, in their
    order) hold a target does not read it: what
    comes back to that form from them is their own cycle (less free pAKT, dephosphorylated, then
    phosphorylated again), and the junction's own step was read in the pass that found it."""
    from .mechanisms import agent_pattern

    for key, (agent, active, any_form) in targets.items():
        pattern = pysb.ComplexPattern(agent_pattern(m, agent), None)
        holders = {i for i, sp in enumerate(m.species) if match_complex_pattern(pattern, sp)}
        if any(i in holders for i in seeds):
            continue
        for rx in m.reactions:
            rows = [row_of[r] for r in _rules(rx) if r in row_of]
            taken, made = Counter(rx["reactants"]), Counter(rx["products"])
            changed = any(i in holders for i in (taken - made) + (made - taken))
            for rule in _rules(rx):
                if rule not in meta:
                    continue
                _, s, driver, _ = meta[rule]
                sign = 1 if s == "p" else -1
                if any_form:
                    if changed:  # any form changed is an effect, a modification included
                        moved[key].add(sign * way, rows)
                    continue
                # Net over the target's forms: a phosphorylation leaves total S as it is, and a form
                # a driver makes by binding or release, which carries no tag of its own, counts.
                net = (sum(1 for i in rx["products"] if i in holders)
                       - sum(1 for i in rx["reactants"] if i in holders))
                own_decay = any(
                    mp.monomer.name == driver and mp.site_conditions.get(SITE) == s
                    for i in rx["reactants"] if i in holders for mp in m.species[i].monomer_patterns
                )
                if net > 0 or (net < 0 and not own_decay):
                    moved[key].add(sign * way * (1 if net > 0 else -1), rows)
                if active and driver is None and any(
                    i in holders and i in rx["products"] for i in rx["reactants"]
                ):
                    # A regulator of a reaction the active form catalyses moves its activity.
                    moved[key].add(sign * way, rows)


def _consumed(m, meta) -> list[tuple[int, str, str, list[int]]]:
    """Species a traced driver takes away that hold no traced molecule: ``(species, the driver's
    tag, the rule direction that took it, what that reaction made of it)``, a catalyst not made."""
    out = {}
    for rx in m.reactions:
        for rule in _rules(rx):
            if rule not in meta:
                continue
            _, s, driver, base = meta[rule]
            made = Counter(rx["products"]) - Counter(rx["reactants"])
            for i in Counter(rx["reactants"]) - Counter(rx["products"]):
                if all(mp.site_conditions.get(SITE) == "o" for mp in m.species[i].monomer_patterns):
                    found = out.setdefault((i, s, base), [])
                    found += [j for j in made if j not in found]
    return [(i, s, base, products) for (i, s, base), products in out.items()]
