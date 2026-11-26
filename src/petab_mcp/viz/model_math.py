"""A stored model's mathematics, read off its Antimony: species, reactions, ODEs and observables,
and the graphs the Model tab draws at each level of detail.

The model is read through libsbml (``SbmlModel.from_antimony``, as :func:`petab_mcp.model.version_text`
reads a stored version) and its math through ``libsbml.formulaToL3String`` into sympy. A formula
sympy cannot read (an SBML function it has no name for, an id that is a Python keyword) becomes a
symbol standing for it, so one odd rate law leaves the rest of the ODEs intact.

A model built from statements keeps what it was built from in its names: PySB names each species
by its pattern (``A(b=1) ._br_B(phospho='u', a=1)``) and each reaction by the rules that make it,
and a row's :class:`~petab_mcp.mechanisms.Emission` lists its rules. So a species maps to the agents
in it, a reaction to its rules and a rule to its row, and the graph can be drawn at any of those
levels (:func:`levels`) without building anything.

Every node has a ``key`` that names the same thing in another version, so two versions' graphs can
be compared: a species pattern or id, a reaction's reactants and products, a row's prefix.
"""

from __future__ import annotations

import keyword
import math
import re
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
import sympy

#: SBML L3 infix functions sympy names otherwise, and its constants.
_FUNCTIONS: dict[str, Any] = {
    "ln": sympy.log,
    "log10": lambda x: sympy.log(x, 10),
    "log": lambda *a: sympy.log(a[-1], a[0]) if len(a) == 2 else sympy.log(a[0], 10),
    "pow": sympy.Pow,
    "power": sympy.Pow,
    "sqrt": sympy.sqrt,
    "root": lambda n, x=None: sympy.sqrt(n) if x is None else x ** (1 / n),
    "abs": sympy.Abs,
    "exp": sympy.exp,
    "exponentiale": sympy.E,
    "pi": sympy.pi,
    "true": sympy.true,
    "false": sympy.false,
    "piecewise": lambda *a: sympy.Piecewise(
        *[(a[i], a[i + 1]) for i in range(0, len(a) - 1, 2)],
        *([(a[-1], True)] if len(a) % 2 else []),
    ),
    "min": sympy.Min,
    "max": sympy.Max,
    "floor": sympy.floor,
    "ceil": sympy.ceiling,
    "time": sympy.Symbol("t"),
}

#: The edge kinds of the influence and statement levels, in the order a legend lists them:
#: ``role`` joins a statement to the agents it names, subject in and object out.
EDGE_KINDS = (
    "activates",
    "inhibits",
    "modulates",
    "modifies",
    "converts",
    "binds",
    "role",
    "reads",
)


@dataclass(frozen=True)
class Species:
    id: str
    label: str
    #: The agents (PySB monomers) in it; the species' own id where the model has no agents.
    agents: tuple[str, ...]
    key: str
    #: A boundary or constant species: no ODE.
    held: bool


@dataclass(frozen=True)
class Reaction:
    id: str
    key: str
    reactants: dict[str, float]
    products: dict[str, float]
    #: Species the rate reads besides its reactants, through assignment rules too.
    modifiers: tuple[str, ...]
    rate: sympy.Expr
    #: The PySB rules that make it (a statements model), from its name.
    rules: tuple[str, ...]


@dataclass(frozen=True)
class Observable:
    id: str
    formula: sympy.Expr
    noise: sympy.Expr | None
    distribution: str
    #: The species its formula reads, through assignment rules too.
    reads: tuple[str, ...]


@dataclass
class ModelMath:
    species: dict[str, Species]
    reactions: list[Reaction]
    assignments: dict[str, sympy.Expr]
    odes: dict[str, sympy.Expr]
    initial: dict[str, sympy.Expr]
    observables: dict[str, Observable]
    #: The rows of a model built from statements (``statements.json``), else empty, and each
    #: row's binding by prefix.
    rows: list[dict[str, Any]] = field(default_factory=list)
    bindings: dict[str, Any] = field(default_factory=dict)

    def symbol(self, sid: str) -> sympy.Symbol:
        return sympy.Symbol(sid)


def parse(text: str, names: dict[str, sympy.Symbol]) -> sympy.Expr:
    """``text``, SBML L3 infix or a PEtab formula, as sympy over ``names``; a symbol naming the
    text where sympy cannot read it."""
    src = text.replace("&&", "&").replace("||", "|")
    local = {**_FUNCTIONS, **{k: v for k, v in names.items() if not keyword.iskeyword(k)}}
    try:
        return sympy.sympify(src, locals=local)
    except Exception:  # noqa: BLE001 - anything sympy cannot read; see the module docstring
        return sympy.Symbol(f"\\text{{{text}}}")


def pattern_label(name: str) -> tuple[str, tuple[str, ...]]:
    """A PySB species name as the graph prints it, and its agents: bonds and free sites dropped,
    states kept, a complex's members joined by ``:``. ``A(b=1) ._br_B(phospho='u', a=1)`` is
    ``A:B(phospho=u)``, its agents ``(A, B)``."""
    labels, agents = [], []
    for part in name.split("._br_"):
        part = part.strip()
        m = re.fullmatch(r"(\w+)\((.*)\)", part)
        if not m:
            labels.append(part)
            agents.append(part)
            continue
        monomer, sites = m.groups()
        agents.append(monomer)
        states = [f"{s}={v}" for s, v in re.findall(r"(\w+)=\(?\s*'([^']*)'", sites)]
        labels.append(f"{monomer}({', '.join(states)})" if states else monomer)
    return ":".join(labels), tuple(agents)


def _number(value: float) -> int | float:
    """A stoichiometry as an int where it is one, so an ODE reads ``-A*k`` and not ``-1.0*A*k``."""
    return int(value) if float(value).is_integer() else value


def _is_pattern(sid: str, name: str) -> bool:
    return sid.startswith("__s") and "(" in name


def _expand(expr: sympy.Expr, assignments: dict[sympy.Symbol, sympy.Expr]) -> sympy.Expr:
    """``expr`` with the assignment rules it reads substituted, transitively (rules are acyclic in
    SBML; ten rounds bound a malformed model)."""
    for _ in range(10):
        hit = expr.free_symbols & assignments.keys()
        if not hit:
            break
        expr = expr.subs({s: assignments[s] for s in hit})
    return expr


def from_antimony(
    text: str,
    observable_df: pd.DataFrame | None = None,
    statements: dict[str, Any] | None = None,
) -> ModelMath:
    """The mathematics of the model ``text`` (Antimony), its observables from the problem's
    ``observable_df`` and, for a model built from statements, its ``statements`` record."""
    import libsbml
    from petab.v2.models.sbml_model import SbmlModel

    sm = SbmlModel.from_antimony(text).sbml_model
    ids = [e.getId() for e in sm.getListOfAllElements() if e.isSetId()]
    names = {i: sympy.Symbol(i) for i in ids}

    def read(math_ast) -> sympy.Expr:
        if math_ast is None:
            return sympy.nan
        return parse(libsbml.formulaToL3String(math_ast), names)

    species: dict[str, Species] = {}
    for sp in sm.getListOfSpecies():
        sid, name = sp.getId(), sp.getName() or sp.getId()
        if _is_pattern(sid, name):
            label, agents = pattern_label(name)
            key = name
        else:
            label, agents, key = sid, (sid,), sid
        species[sid] = Species(
            sid, label, agents, key, held=bool(sp.getBoundaryCondition() or sp.getConstant())
        )

    assignments = {
        sympy.Symbol(r.getVariable()): read(r.getMath())
        for r in sm.getListOfRules()
        if r.isAssignment()
    }
    rate_rules = {r.getVariable(): read(r.getMath()) for r in sm.getListOfRules() if r.isRate()}
    species_symbols = {sympy.Symbol(s) for s in species}

    def reads(expr: sympy.Expr) -> tuple[str, ...]:
        return tuple(
            sorted(str(s) for s in _expand(expr, assignments).free_symbols & species_symbols)
        )

    reactions = []
    for rx in sm.getListOfReactions():
        reactants = {r.getSpecies(): _number(r.getStoichiometry()) for r in rx.getListOfReactants()}
        products = {r.getSpecies(): _number(r.getStoichiometry()) for r in rx.getListOfProducts()}
        rate = (
            read(rx.getKineticLaw().getMath())
            if rx.isSetKineticLaw()
            else sympy.Symbol(f"v_{rx.getId()}")
        )
        declared = {m.getSpecies() for m in rx.getListOfModifiers()}
        modifiers = tuple(sorted((set(reads(rate)) | declared) - set(reactants)))
        rules = (
            tuple(r.strip() for r in (rx.getName() or "").split(" + ") if r.strip())
            if (statements)
            else ()
        )

        def side(stoich: dict[str, float]) -> str:
            return "+".join(sorted(species[s].key for s in stoich if s in species))

        key = f"{side(reactants)}>{side(products)}" + (f"|{'+'.join(rules)}" if rules else "")
        reactions.append(Reaction(rx.getId(), key, reactants, products, modifiers, rate, rules))

    # Each reaction's terms, collected by the species they change: walking every species past
    # every reaction multiplied mostly zeros, 96 s of Froehlich_CellSystems2018's 1396 x 2686.
    terms: dict[str, list[sympy.Expr]] = {}
    for rx in reactions:
        for sid in {*rx.reactants, *rx.products}:
            net = rx.products.get(sid, 0) - rx.reactants.get(sid, 0)
            if net:
                terms.setdefault(sid, []).append(net * rx.rate)
    odes: dict[str, sympy.Expr] = {}
    for sid, sp in species.items():
        if sp.held or sympy.Symbol(sid) in assignments:
            continue
        flux = sympy.Add(*terms.get(sid, ()))
        sbml = sm.getSpecies(sid)
        compartment = sm.getCompartment(sbml.getCompartment())
        if (
            not sbml.getHasOnlySubstanceUnits()
            and compartment is not None
            and not (compartment.getConstant() and compartment.getSize() == 1)
        ):
            flux = flux / sympy.Symbol(compartment.getId())
        odes[sid] = rate_rules.get(sid, flux)
    for var, expr in rate_rules.items():
        odes.setdefault(var, expr)

    initial: dict[str, sympy.Expr] = {}
    for sp in sm.getListOfSpecies():
        ia = sm.getInitialAssignment(sp.getId())
        if ia is not None:
            initial[sp.getId()] = read(ia.getMath())
        elif sp.isSetInitialAmount():
            initial[sp.getId()] = sympy.Float(sp.getInitialAmount(), 6)
        elif sp.isSetInitialConcentration():
            initial[sp.getId()] = sympy.Float(sp.getInitialConcentration(), 6)

    observables: dict[str, Observable] = {}
    if observable_df is not None:
        known = {**names, **{str(s): s for s in assignments}}
        for oid, row in observable_df.iterrows():
            formula = parse(str(row.get("observableFormula", "")), known)
            raw_noise = row.get("noiseFormula")
            noise = (
                parse(str(raw_noise), known)
                if isinstance(raw_noise, str)
                or (isinstance(raw_noise, (int, float)) and not math.isnan(raw_noise))
                else None
            )
            distribution = row.get("noiseDistribution")
            observables[str(oid)] = Observable(
                str(oid),
                formula,
                noise,
                distribution if isinstance(distribution, str) else "normal",
                reads(formula),
            )

    return ModelMath(
        species=species,
        reactions=reactions,
        assignments={str(k): v for k, v in assignments.items()},
        odes=odes,
        initial=initial,
        observables=observables,
        rows=list((statements or {}).get("rows") or []),
        bindings=dict((statements or {}).get("bindings") or {}),
    )


# -- signs --------------------------------------------------------------------------------------
def influence_sign(ode: sympy.Expr, x: sympy.Symbol) -> str | None:
    """How ``x`` moves ``ode`` with every name positive: ``activates`` where the derivative is
    positive for all of them, ``inhibits`` where negative, ``modulates`` where sympy cannot tell,
    ``None`` where ``ode`` does not read ``x``."""
    if x not in ode.free_symbols:
        return None
    derivative = sympy.diff(ode, x)
    if derivative == 0:
        return None
    positive = {
        s: sympy.Symbol(s.name, positive=True)
        for s in derivative.free_symbols
        if isinstance(s, sympy.Symbol)
    }
    d = derivative.subs(positive)
    if d.is_positive:
        return "activates"
    if d.is_negative:
        return "inhibits"
    return "modulates"


#: A statement type's subject roles (what acts) and object roles (what is acted on).
_SUBJECT_ROLES = ("enz", "subj", "source", "obj_from")
_OBJECT_ROLES = ("sub", "obj", "target", "obj_to", "agent")
_ACTIVATING = {"Activation", "IncreaseAmount", "GtpActivation", "Gef"}
_INHIBITING = {"Inhibition", "DecreaseAmount", "Gap"}
_CONVERTING = {"Conversion", "PopulationTransition", "Translocation"}
_REGULATOR_KINDS = {
    "inhibited_by": "inhibits",
    "activated_by": "activates",
    "modulated_by": "modulates",
}


def _agent_names(value: Any) -> list[str]:
    if isinstance(value, dict) and value.get("name"):
        return [str(value["name"])]
    if isinstance(value, list):
        return [n for v in value for n in _agent_names(v)]
    return []


def row_edges(row: dict[str, Any]) -> list[tuple[str, str, str]]:
    """The edges a statements row draws between agents, ``(source, target, kind)``: what its type
    says the subject does to the object, a complex's members bound to each other, and each
    regulator to the object as its rate records it (``emission.derived``)."""
    stmt = row.get("statement") or {}
    kind_of = stmt.get("type", "")
    subjects = [n for r in _SUBJECT_ROLES for n in _agent_names(stmt.get(r))]
    objects = [n for r in _OBJECT_ROLES for n in _agent_names(stmt.get(r))]
    if kind_of in _ACTIVATING:
        kind = "activates"
    elif kind_of in _INHIBITING:
        kind = "inhibits"
    elif kind_of in _CONVERTING:
        kind = "converts"
    else:
        kind = "modifies"
    out = [(s, o, kind) for s in subjects for o in objects if s != o]
    members = _agent_names(stmt.get("members"))
    out += [(a, b, "binds") for i, a in enumerate(members) for b in members[i + 1 :]]
    derived = (row.get("emission") or {}).get("derived") or {}
    targets = objects or members
    for regulator_kind, edge in _REGULATOR_KINDS.items():
        for i, name in enumerate(_agent_names(stmt.get(regulator_kind))):
            recorded = _REGULATOR_KINDS.get(derived.get(f"{regulator_kind}_{i}", ""), edge)
            out += [(name, t, recorded) for t in targets if t != name]
    return out


def row_agents(row: dict[str, Any]) -> list[str]:
    stmt = row.get("statement") or {}
    found = [n for k, v in stmt.items() if k not in ("evidence", "type") for n in _agent_names(v)]
    return list(dict.fromkeys(found))


def row_text(row: dict[str, Any]) -> str:
    """A row as the notation writes it (:func:`petab_mcp.notation.render`), or as its type and
    agents where the ``statements`` extra (INDRA) is not installed."""
    derived = (row.get("emission") or {}).get("derived")
    try:
        from .. import notation

        text = notation.render(
            {"prefix": row.get("prefix"), "statement": row["statement"]}, derived
        )
    except Exception:  # noqa: BLE001 - INDRA missing, or a row the notation cannot write
        text = None
    if text:
        return text
    stmt = row.get("statement") or {}
    return f"{row.get('prefix')}: {stmt.get('type')}({', '.join(row_agents(row))})"


def row_english(row: dict[str, Any]) -> str | None:
    try:
        from .. import statements

        return statements.english(row["statement"], (row.get("emission") or {}).get("derived"))
    except Exception:  # noqa: BLE001 - INDRA missing, or a statement its assembler refuses
        return None


# -- the graph at each level --------------------------------------------------------------------
#: The levels the Model tab draws, coarse to fine.
LEVELS = ("influence", "statements", "reactions")


@dataclass
class Graph:
    """Nodes and edges of one level. ``nodes`` has ``key``, ``label``, ``kind`` (agent, species,
    row, reaction, observable) and ``members`` (keys of the finest-level nodes it stands for);
    ``edges`` has ``source``, ``target`` (node keys) and ``kind``."""

    nodes: pd.DataFrame
    edges: pd.DataFrame


def _graph(nodes: list[dict], edges: list[tuple[str, str, str]]) -> Graph:
    node_df = pd.DataFrame(nodes, columns=["key", "label", "kind", "members", "info"])
    node_df = node_df.drop_duplicates("key").reset_index(drop=True)
    edge_df = pd.DataFrame(edges, columns=["source", "target", "kind"]).drop_duplicates()
    known = set(node_df["key"])
    edge_df = edge_df[edge_df["source"].isin(known) & edge_df["target"].isin(known)]
    return Graph(node_df, edge_df.reset_index(drop=True))


def _observable_nodes(math_: ModelMath, owner) -> tuple[list[dict], list[tuple[str, str, str]]]:
    """Each observable as a node reading what ``owner(species id)`` names at this level."""
    nodes, edges = [], []
    for oid, obs in math_.observables.items():
        key = f"o:{oid}"
        nodes.append(
            {
                "key": key,
                "label": oid,
                "kind": "observable",
                "members": (key,),
                "info": sympy.sstr(obs.formula),
            }
        )
        edges += [(owner(s), key, "reads") for s in obs.reads]
    return nodes, edges


def reaction_graph(math_: ModelMath) -> Graph:
    """Species and reactions, each reaction joined to what it consumes, makes and reads."""
    nodes: list[dict] = []
    edges: list[tuple[str, str, str]] = []
    for sp in math_.species.values():
        nodes.append(
            {
                "key": f"s:{sp.key}",
                "label": sp.label,
                "kind": "species",
                "members": (f"s:{sp.key}",),
                "info": sp.id,
            }
        )
    for rx in math_.reactions:
        key = f"r:{rx.key}"
        nodes.append(
            {
                "key": key,
                "label": rx.id,
                "kind": "reaction",
                "members": (key,),
                "info": sympy.sstr(rx.rate),
            }
        )
        edges += [(f"s:{math_.species[s].key}", key, "consumes") for s in rx.reactants]
        edges += [(key, f"s:{math_.species[s].key}", "produces") for s in rx.products]
        edges += [(f"s:{math_.species[s].key}", key, "modulates") for s in rx.modifiers]
    obs_nodes, obs_edges = _observable_nodes(math_, lambda s: f"s:{math_.species[s].key}")
    return _graph(nodes + obs_nodes, edges + obs_edges)


def _agents_of(math_: ModelMath) -> dict[str, list[str]]:
    """Each agent's species keys."""
    out: dict[str, list[str]] = {}
    for sp in math_.species.values():
        for agent in sp.agents:
            out.setdefault(agent, []).append(f"s:{sp.key}")
    return out


def _rule_reactions(math_: ModelMath) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for rx in math_.reactions:
        for rule in rx.rules:
            out.setdefault(rule, []).append(f"r:{rx.key}")
    return out


def _row_members(math_: ModelMath, row: dict[str, Any]) -> tuple[str, ...]:
    """The reaction keys a row's rules make; a reduced rule keeps its name as a prefix."""
    by_rule = _rule_reactions(math_)
    rules = (row.get("emission") or {}).get("rules") or []
    found = [
        k
        for rule in rules
        for name, keys in by_rule.items()
        if name == rule or name.startswith(f"{rule}_")
        for k in keys
    ]
    return tuple(dict.fromkeys(found))


def statement_graph(math_: ModelMath) -> Graph:
    """Agents and the rows that state something of them, each row joined to its agents by the part
    they play (subject in, object out, regulator signed)."""
    agents = _agents_of(math_)
    nodes = [
        {"key": f"a:{a}", "label": a, "kind": "agent", "members": tuple(keys), "info": a}
        for a, keys in agents.items()
    ]
    edges: list[tuple[str, str, str]] = []
    for row in math_.rows:
        key = f"p:{row.get('prefix')}"
        nodes.append(
            {
                "key": key,
                "label": str(row.get("prefix")),
                "kind": "row",
                "members": _row_members(math_, row),
                "info": row_text(row),
            }
        )
        stmt = row.get("statement") or {}
        for role, value in stmt.items():
            for name in _agent_names(value):
                agent = f"a:{name}"
                nodes.append(
                    {
                        "key": agent,
                        "label": name,
                        "kind": "agent",
                        "members": tuple(agents.get(name, ())),
                        "info": name,
                    }
                )
                if role in _REGULATOR_KINDS:
                    edges.append((agent, key, _REGULATOR_KINDS[role]))
                elif role in _OBJECT_ROLES or role == "members":
                    edges.append((key, agent, "role"))
                else:
                    edges.append((agent, key, "role"))
    obs_nodes, obs_edges = _observable_nodes(math_, lambda s: f"a:{math_.species[s].agents[0]}")
    return _graph(nodes + obs_nodes, edges + obs_edges)


def influence_graph(math_: ModelMath, max_pairs: int = 1500) -> Graph:
    """Signed influences. From the rows where the model was built from statements
    (:func:`row_edges`); else between species, from the sign of each ODE's derivative in each
    species it reads (:func:`influence_sign`), self-influence left out. Past ``max_pairs`` pairs
    the derivative is not taken and an edge is ``modulates``."""
    if math_.rows:
        agents = _agents_of(math_)
        nodes = [
            {"key": f"a:{a}", "label": a, "kind": "agent", "members": tuple(keys), "info": a}
            for a, keys in agents.items()
        ]
        edges = []
        for row in math_.rows:
            for name in row_agents(row):
                nodes.append(
                    {
                        "key": f"a:{name}",
                        "label": name,
                        "kind": "agent",
                        "members": tuple(agents.get(name, ())),
                        "info": name,
                    }
                )
            edges += [(f"a:{s}", f"a:{t}", k) for s, t, k in row_edges(row)]
        obs_nodes, obs_edges = _observable_nodes(math_, lambda s: f"a:{math_.species[s].agents[0]}")
        return _graph(nodes + obs_nodes, edges + obs_edges)

    expanded = {sympy.Symbol(k): v for k, v in math_.assignments.items()}
    nodes = [
        {
            "key": f"s:{sp.key}",
            "label": sp.label,
            "kind": "species",
            "members": (f"s:{sp.key}",),
            "info": sp.id,
        }
        for sp in math_.species.values()
    ]
    edges = []
    pairs = 0
    for sid, ode in math_.odes.items():
        if sid not in math_.species:
            continue
        rhs = _expand(ode, expanded)
        for other in sorted(str(s) for s in rhs.free_symbols if str(s) in math_.species):
            if other == sid:
                continue
            pairs += 1
            sign = influence_sign(rhs, sympy.Symbol(other)) if pairs <= max_pairs else "modulates"
            if sign:
                edges.append((f"s:{math_.species[other].key}", f"s:{math_.species[sid].key}", sign))
    obs_nodes, obs_edges = _observable_nodes(math_, lambda s: f"s:{math_.species[s].key}")
    return _graph(nodes + obs_nodes, edges + obs_edges)


def levels(math_: ModelMath) -> dict[str, Graph]:
    """The graph at each level the model has: ``statements`` only for a model built from them."""
    out = {"influence": influence_graph(math_), "reactions": reaction_graph(math_)}
    if math_.rows:
        out["statements"] = statement_graph(math_)
    return {k: out[k] for k in LEVELS if k in out}


# -- layout -------------------------------------------------------------------------------------
def layout(graphs: list[Graph], seed: int = 0) -> dict[str, tuple[float, float]]:
    """Positions for the finest-level nodes of ``graphs`` (each a :func:`reaction_graph`, of one
    version or two), each connected component laid out on its own and the components packed in
    rows, so a component never sits on top of another."""
    import networkx as nx

    g = nx.Graph()
    for graph in graphs:
        g.add_nodes_from(graph.nodes["key"])
        g.add_edges_from(zip(graph.edges["source"], graph.edges["target"]))
    components = sorted(nx.connected_components(g), key=len, reverse=True)
    out: dict[str, tuple[float, float]] = {}
    x0, y0, row_height, row_width = 0.0, 0.0, 0.0, 0.0
    width_cap = max(4.0, math.sqrt(len(g)) * 1.2)
    for nodes in components:
        sub = g.subgraph(nodes)
        if len(sub) == 1:
            pos = {next(iter(nodes)): (0.0, 0.0)}
        elif len(sub) <= 250:
            pos = nx.kamada_kawai_layout(sub)
        else:
            # Froehlich_CellSystems2018's 3942 nodes took 52 s at 100 iterations, 13 s at 30.
            pos = nx.spring_layout(sub, seed=seed, iterations=30 if len(sub) > 1000 else 100)
        xs = [p[0] for p in pos.values()]
        ys = [p[1] for p in pos.values()]
        scale = max(1.0, math.sqrt(len(sub))) / max(max(xs) - min(xs), max(ys) - min(ys), 1e-9)
        if len(sub) == 1:
            scale = 1.0
        w = (max(xs) - min(xs)) * scale + 1.0
        h = (max(ys) - min(ys)) * scale + 1.0
        if row_width and row_width + w > width_cap:
            x0, y0, row_width, row_height = 0.0, y0 - row_height, 0.0, 0.0
        for k, (x, y) in pos.items():
            out[k] = (x0 + row_width + (x - min(xs)) * scale, y0 - (y - min(ys)) * scale)
        row_width += w
        row_height = max(row_height, h)
    return out


def place(graph: Graph, finest: dict[str, tuple[float, float]]) -> pd.DataFrame:
    """``graph``'s nodes with ``x``, ``y``: each at the centroid of the finest-level nodes it stands
    for, so zooming in splits a node where it stood. A node standing for none (an agent only a
    regulator names) sits at the centroid of its neighbours, else at the origin."""
    df = graph.nodes.copy()
    xy = []
    for members in df["members"]:
        pts = [finest[m] for m in members if m in finest]
        xy.append(tuple(sum(c) / len(pts) for c in zip(*pts)) if pts else None)
    placed = dict(zip(df["key"], xy))
    for i, key in enumerate(df["key"]):
        if xy[i] is None:
            near = [
                placed[n]
                for n in list(graph.edges.loc[graph.edges["source"] == key, "target"])
                + list(graph.edges.loc[graph.edges["target"] == key, "source"])
                if placed.get(n) is not None
            ]
            xy[i] = tuple(sum(c) / len(near) for c in zip(*near)) if near else (0.0, 0.0)
    df["x"] = [p[0] for p in xy]
    df["y"] = [p[1] for p in xy]
    return df


# -- equations ----------------------------------------------------------------------------------
def _text(text: str) -> str:
    """``text`` as LaTeX prints it verbatim: a name, not a product of symbols."""
    escaped = text.replace("\\", "").replace("_", r"\_").replace("{", r"\{").replace("}", r"\}")
    return r"\text{" + escaped + "}"


def _latex_name(name: str) -> str:
    return _text(name)


def _amount(label: str) -> str:
    """A species' amount: its name in brackets, so a product of two reads as two."""
    return rf"[{_text(label)}]"


#: LaTeX characters past which an equation is split a term per line.
LINE = 70


def equations(
    math_: ModelMath,
    species: list[str] | None = None,
    reactions: list[str] | None = None,
    observables: list[str] | None = None,
    values: dict[str, float] | None = None,
    limit: int = 40,
) -> list[str]:
    """LaTeX lines for the selection: each species' ODE (or assignment) and initial value, each
    reaction's rate, each observable's formula and noise. With ``values``, parameters are shown at
    them. ``None`` for every argument is the whole model."""
    everything = species is None and reactions is None and observables is None
    subs = {sympy.Symbol(k): sympy.Float(v, 4) for k, v in (values or {}).items()}
    names = {sympy.Symbol(sid): _amount(sp.label) for sid, sp in math_.species.items()}

    def tex(expr: sympy.Expr) -> str:
        expr = expr.subs(subs)
        # sympy reads every underscore as a subscript, so `reaction_9_k1` would print as
        # reaction with a subscript 9k1: a name with more than one is printed as written.
        verbatim = {
            s: _text(s.name)
            for s in expr.free_symbols
            if isinstance(s, sympy.Symbol) and s.name.count("_") > 1 and s not in names
        }
        # An explicit dot: a model's names are words, and `cyt*k1aBasic` juxtaposed read as one.
        return sympy.latex(expr, symbol_names={**names, **verbatim}, mul_symbol="dot")

    def rhs(left: str, expr: sympy.Expr, relation: str = "=") -> list[str]:
        """``left relation expr``, a long one a term per aligned line, so an ODE reads down the
        pane rather than off its edge. A factor outside a sum (``Cell*(a - b)``, a compartment
        times a reversible rate) is multiplied in first, one level deep, so its terms split too."""
        whole = tex(expr)
        terms = [
            t
            for term in sympy.Add.make_args(expr)
            for t in sympy.Add.make_args(sympy.expand_mul(term, deep=False))
        ]
        if len(terms) < 2 or len(whole) <= LINE:
            return [rf"{left} &{relation} {whole}"]
        out = []
        for i, term in enumerate(terms):
            negative = term.could_extract_minus_sign()
            body = tex(-term if negative else term)
            sign = "- " if negative else ("+ " if i else "")
            out.append((rf"{left} &{relation} " if i == 0 else r"&\quad ") + sign + body)
        return out

    def name(sid: str) -> str:
        return _amount(math_.species[sid].label) if sid in math_.species else _latex_name(sid)

    lines = []
    for n, sid in enumerate(list(math_.odes) if everything else species or []):
        if len(lines) >= limit:  # each line is LaTeX to render: stop once the page is full
            lines.append(rf"&\text{{... and {len(math_.odes) - n} more: click a node for its own}}")
            return lines
        if sid in math_.odes:
            lines += rhs(rf"\frac{{d\,{name(sid)}}}{{dt}}", math_.odes[sid])
        elif sid in math_.assignments:
            lines += rhs(name(sid), math_.assignments[sid], ":=")
        elif sid in math_.species:
            lines.append(rf"{name(sid)} &\ \text{{held}}")
        if sid in math_.initial and not everything:
            lines.append(rf"{name(sid)}(0) &= {tex(math_.initial[sid])}")
    by_id = {rx.id: rx for rx in math_.reactions}
    for rid in reactions or []:
        rx = by_id[rid]
        left = " + ".join(name(s) for s in rx.reactants) or r"\emptyset"
        right = " + ".join(name(s) for s in rx.products) or r"\emptyset"
        lines.append(rf"{_latex_name(rid)}: &\ {left} \to {right}")
        lines.append(rf"v &= {tex(rx.rate)}")
    for oid in list(math_.observables) if everything else observables or []:
        obs = math_.observables[oid]
        lines.append(rf"{_latex_name(oid)} &= {tex(obs.formula)}")
        if obs.noise is not None:
            lines.append(
                rf"\sigma_{{{_latex_name(oid)}}} &= {tex(obs.noise)}"
                rf"\quad \text{{({obs.distribution})}}"
            )
    if len(lines) > limit:
        lines = lines[:limit] + [rf"&\text{{... and {len(lines) - limit} more}}"]
    return lines
