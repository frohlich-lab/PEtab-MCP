"""A row's regulators, checked on the row's net rate in the model the rows build
(:mod:`petab_mcp.regulation`).

Each case builds rows through :func:`problem_sbml.to_antimony`, as `make_changes` does: J is read
off BioNetGen's expansion of every row, so a scheme is checked with the row's own agents, the
states other rows give them and its bindings, never a placeholder.
"""

from __future__ import annotations

import time

import pytest
import sympy
from indra.statements import stmts_from_json

from petab_mcp import mechanisms as M
from petab_mcp import notation, problem_sbml, regulation
from petab_mcp.errors import MechanismError, PetabMcpError
from petab_mcp.statements import _REACTIONS, indra_part, regulators, with_defaults


def _build(rows: list, *definitions: dict):
    """``rows`` (the notation, or INDRA JSON rows) built with ``definitions`` written: the
    emissions. Raises what `make_changes` refuses with."""
    library = dict(M.REGISTRY)
    for definition in definitions:
        library[(definition["statement_type"], definition["policy"])] = M.written(definition, library)
    parsed = [notation.parse(r, library=library) if isinstance(r, str) else dict(r) for r in rows]
    for row in parsed:
        row["statement"] = with_defaults(row["statement"])
    stmts = stmts_from_json([indra_part(row["statement"]) for row in parsed])
    _, emissions = problem_sbml.to_antimony(
        [(s, row["policy"], row["prefix"], row.get("parameters"), regulators(row["statement"]))
         for s, row in zip(stmts, parsed)],
        {}, library,
    )
    return emissions


def _scheme(stmt_type: str, scheme: str, *parameters: str, policy: str = "written") -> dict:
    return {"statement_type": stmt_type, "policy": policy, "scheme": scheme,
            "parameters": list(parameters)}


def _rate(rate: str, *parameters: str, stmt_type="Phosphorylation", based_on="indra_default"):
    return {"statement_type": stmt_type, "policy": "written", "based_on": based_on, "rate": rate,
            "parameters": list(parameters)}


_PHOSPHO = "p: MEK phosphorylates ERK at T202, {regulator} | written"


# ------------------------------------------------------------------------------- written rates
@pytest.mark.parametrize(
    "rate, regulator, match",
    [
        # An inhibitor the rate rises with: the claim and the maths disagree outright.
        ("k*inhibited_by/(Ki + inhibited_by)", "inhibited by D",
         r"`inhibited by D`, which claims it inhibits the reaction, so its net rate must fall as D "
         r"rises; it rises instead: state it `activated by D`"),
        ("k*Ki/(Ki + activated_by)", "activated by D",
         r"must rise as D rises; it falls instead: state it `inhibited by D`"),
        # Sneyd's Ca: the rate rises or falls with it depending on the constants.
        ("(k + Ki*inhibited_by)/(1 + inhibited_by)", "inhibited by D",
         r"cannot be shown for every value of its names .*: state it `modulated by D`, which "
         r"claims no direction"),
        # A switch's jump is no derivative: the rate is not shown to rise with the drug.
        ("k*piecewise(1, activated_by > 0, 0)", "activated by D",
         r"nor across a piecewise switching on it\): state it `modulated by D`"),
    ],
)
def test_a_rate_must_move_as_its_regulators_claim(rate, regulator, match):
    """PEtab-MCP#122: the statement says which way a regulator moves the reaction; a rate that
    moves it otherwise, or that cannot be shown to move it so, is refused."""
    names = sorted({"k", "Ki"} & set(M._RATE_TOKENS.findall(rate)))
    with pytest.raises(MechanismError, match=match):
        _build([_PHOSPHO.format(regulator=regulator)], _rate(rate, *names))


@pytest.mark.parametrize(
    "rate, regulator",
    [
        ("k * Ki / (Ki + inhibited_by)", "inhibited by D"),
        ("k/(i*inhibited_by^6/(inhibited_by^6 + Ki^6) + 1)", "inhibited by D"),  # alkan's repression
        ("k*activated_by^6/(activated_by^6 + Ki^6)", "activated by D"),  # alkan's driven death
        ("k*(1 - inhibited_by)", "inhibited by D"),  # perelson's ritonavir gate
        ("k*piecewise(1, modulated_by > 0, 0)", "modulated by D"),  # a switch claims no direction
        ("(k + Ki*modulated_by)/(1 + modulated_by)", "modulated by D"),
    ],
)
def test_a_rate_moving_as_its_regulators_claim_builds(rate, regulator):
    names = sorted({"k", "Ki", "i"} & set(M._RATE_TOKENS.findall(rate)))
    _build([_PHOSPHO.format(regulator=regulator)], _rate(rate, *names))


def test_a_regulator_nothing_forms_is_refused_naming_what_to_add():
    """A rate reading PTEN phosphorylated where nothing phosphorylates PTEN does not change with
    it: the model holds no such PTEN."""
    with pytest.raises(MechanismError, match=r"nothing in the model forms PTEN\{pS380\}, so the "
                       r"rate does not change with it. Add the statement that forms it"):
        _build([_PHOSPHO.format(regulator="inhibited by PTEN{pS380}")],
               _rate("k/(1 + inhibited_by)", "k"))


# ------------------------------------------------------------------------------- schemes
#: Michaelis and Menten's two steps, each rate open to a regulator.
_TWO_STEPS = (
    "enz(sub=None) + sub(enz=None, mod='off') | enz(sub=1) % sub(enz=1, mod='off') @ {kf}, {kr}\n"
    "enz(sub=1) % sub(enz=1, mod='off') >> enz(sub=None) + sub(enz=None, mod='on') @ {kc}"
)
_SLOWED = "/(1 + amount(inhibited_by)/Ki)"
_SPED = "*(1 + amount(inhibited_by)/Ki)"


def _two_steps(**rates):
    rates = {"kf": "kf", "kr": "kr", "kc": "kc"} | rates
    scheme = _scheme("Phosphorylation", _TWO_STEPS.format(**rates), "kf", "kr", "kc", "Ki")
    return _build([_PHOSPHO.format(regulator="inhibited by I")], scheme)


@pytest.mark.parametrize(
    "rates",
    [
        {"kc": "kc" + _SLOWED},  # the catalytic step slowed
        {"kr": "kr" + _SPED},  # unbinding sped: less complex to turn over
        {"kf": "kf" + _SLOWED, "kr": "kr" + _SLOWED, "kc": "kc" + _SLOWED},  # every step slowed
    ],
)
def test_an_inhibitor_of_a_scheme_of_several_steps_lowers_its_net_rate(rates):
    """PEtab-MCP#125: whether a step's rate speeds the reaction depends on where the step sits,
    so a scheme is checked on its net rate, its intermediates steady and its products absent."""
    _two_steps(**rates)


def test_a_regulator_that_moves_the_net_rate_the_other_way_is_refused():
    """Slowing unbinding leaves more complex to turn over: it speeds the reaction, whatever its
    own rate does."""
    with pytest.raises(MechanismError, match=r"it rises instead: state it `activated by I`"):
        _two_steps(kr="kr" + _SLOWED)


def test_a_binding_line_written_dissociation_first_is_the_same_scheme():
    """review of #128, netrate t2: `ES | E + S` read as written held the complex as a reactant, so
    an inhibitor of binding read as not moving J at all."""
    scheme = (
        "enz(sub=1) % sub(enz=1, mod='off') | enz(sub=None) + sub(enz=None, mod='off') @ kr, "
        "kf" + _SLOWED + "\n"
        "enz(sub=1) % sub(enz=1, mod='off') >> enz(sub=None) + sub(enz=None, mod='on') @ kc"
    )
    _build([_PHOSPHO.format(regulator="inhibited by I")],
           _scheme("Phosphorylation", scheme, "kf", "kr", "kc", "Ki"))


def test_a_step_into_a_dead_end_inhibits():
    """No rule on a step's own rate gets this one: a step pointing forward, parking the enzyme in a
    form that makes nothing, slows the reaction as it speeds."""
    dead_end = (
        "enz(sub=None, conf='open') + sub(enz=None, mod='off') | "
        "enz(sub=1, conf='open') % sub(enz=1, mod='off') @ kf, kr\n"
        "enz(sub=1) % sub(enz=1, mod='off') >> enz(sub=None) + sub(enz=None, mod='on') @ kc\n"
        "enz(sub=None, conf='open') | enz(sub=None, conf='shut') @ kd*(1 + amount(inhibited_by)/Ki), kb"
    )
    _build([_PHOSPHO.format(regulator="inhibited by I")],
           _scheme("Phosphorylation", dead_end, "kf", "kr", "kc", "kd", "kb", "Ki"))


def test_a_row_without_one_net_rate_states_its_regulators_without_a_direction():
    """A scheme that also destroys its substrate is two reactions at unrelated rates: no one net
    rate, so its regulators claim no direction."""
    branched = (
        "enz() + sub(mod='off') >> enz() + sub(mod='on') @ kc/(1 + amount({kind}_by)/Ki)\n"
        "enz() + sub(mod='off') >> enz() @ kd"
    )
    with pytest.raises(MechanismError, match=r"at rates no constant ratio relates: two reactions"
                       r".*State each `modulated by`"):
        _build([_PHOSPHO.format(regulator="inhibited by I")],
               _scheme("Phosphorylation", branched.format(kind="inhibited"), "kc", "kd", "Ki"))
    _build([_PHOSPHO.format(regulator="modulated by I")],
           _scheme("Phosphorylation", branched.format(kind="modulated"), "kc", "kd", "Ki"))


#: A GTPase cycle: a GEF turns obj_from into obj_to, a GAP turns it back.
_CYCLE = (
    "obj_from_0() >> obj_to_0() @ kf*amount(activated_by_0){allo}\n"
    "obj_to_0() >> obj_from_0() @ kgap{basal}"
)


def _cycle(allo="", basal="", stated="activated by SOS"):
    parameters = ["kf", "kgap"] + (["allo"] if allo else []) + (["kb"] if basal else [])
    kind = stated.split(" by")[0]
    scheme = _CYCLE.format(allo=allo, basal=basal).replace("activated_by_0", f"{kind}_by_0")
    return _build([f'c: RAS{{-mod"GTP"}} -> RAS{{mod"GTP"}}, {stated} | written'],
                  _scheme("Conversion", scheme, *parameters))


def test_a_cycle_back_to_its_substrate_is_checked_on_its_forward_step():
    """perica_egf's RAS cycle (`ras_gef`, run 61230791_0) was refused as making nothing: its step
    back makes obj_from again, so obj_from read as an intermediate and nothing as a reactant. It is
    the conversion's input, which the statement says; with obj_to absent the step back is silent."""
    _cycle()
    # ras_gef3: three steps, the GEF sped by RAS-GTP and a basal exchange beside it.
    _cycle(allo="*(1 + allo*amount(activated_by_1))", basal="\nobj_from_0() >> obj_to_0() @ kb",
           stated='activated by SOS and RAS{mod"GTP"}')
    with pytest.raises(MechanismError, match=r"it rises instead: state it `activated by SOS`"):
        _cycle(stated="inhibited by SOS")


def test_a_scheme_gated_on_a_state_other_rows_give_is_checked_on_the_real_row():
    """Eight one-rule perica_egf schemes (`mek_act_braf`, `egfr_act2`, ...) write a site in one
    state (`enz(gtpbound='active')`) whose other states other rows give: a placeholder statement
    could not build them, the row as built does."""
    gef = {"statement": {"type": "Gef", "gef": {"name": "SOS"}, "ras": {"name": "KRAS"}},
           "policy": "indra_default", "prefix": "gef"}

    def gated(rate):
        scheme = _scheme("Phosphorylation", "enz(gtpbound='active') + sub(mod='off') >> "
                         f"enz(gtpbound='active') + sub(mod='on') @ {rate}", "kf")
        return _build([gef, "p: KRAS phosphorylates RAF at S338, activated by X | written"], scheme)

    gated("kf*amount(activated_by)")
    with pytest.raises(MechanismError, match=r"must rise as X rises; it falls"):
        gated("kf/(1 + amount(activated_by))")


#: A binding written as two one-way lines (isensee's `camp_release*`), and its two sides.
_BOUND = "members_0(members_1=1) % members_1(members_0=1)"
_FREE = "members_0(members_1=None) + members_1(members_0=None)"


def _binding(on, off, stated="inhibited by X"):
    return _build([f"b: A binds B, {stated} | written"],
                  _scheme("Complex", f"{_FREE} >> {_BOUND} @ {on}\n{_BOUND} >> {_FREE} @ {off}",
                          "kf", "kr", "K"))


#: What a binding's regulator of its way back alone (sweep-55max a1, a2) is refused with, and the
#: Conversion it is told to write instead.
_DISSOCIATION = (
    r"reads `inhibited by X` only in `A % B >> A \+ B`, the dissociation, and its statement's own "
    r"reaction rate, its products absent, does not change with it: a statement's regulator is "
    r"about its own reaction\. State it on a row of that step instead, `A\{!B\} -> A \+ "
    r"B, activated by X`: a Conversion whose scheme breaks the bond and reads it, "
)
_APART = _scheme(
    "Conversion",
    "obj_from_0(obj_to_1=1) % obj_to_1(obj_to_0=1) >> obj_to_0(obj_to_1=None) + "
    "obj_to_1(obj_to_0=None) @ kr*(1 + amount(activated_by)/K)",
    "kr", "K", policy="apart",
)


def test_a_binding_written_as_two_lines_has_a_net_rate():
    """sweep-55max: the members, made by the way back and used by the way on, read as
    intermediates, so a binding as two lines had no J where the same as one `|` line had one."""
    _binding("kf/(1 + amount(inhibited_by)/K)", "kr*K")  # a4


def test_a_regulator_of_the_way_back_alone_names_the_dissociation_and_its_row_builds():
    """With its complex absent a binding does not come apart, so what speeds only its way back
    (a2) leaves J as it is: the agent regulates the dissociation, a statement of its own. The row
    the refusal names builds as written."""
    with pytest.raises(MechanismError, match=_DISSOCIATION):
        _binding("kf*K", "kr*(1 + amount(inhibited_by)/K)")
    _build(["b: A binds B | indra_default", "u: A{!B} -> A + B, activated by X | apart"], _APART)


def test_a_reversible_step_is_read_both_ways_on_its_own_row():
    """With its product absent a step does not run back, so what slows only its way back claims
    nothing a regulator of the reaction does (sweep-55max a1)."""
    def binding(kf, kr):
        return _build(["b: A binds B, inhibited by X | written"],
                      _scheme("Complex", f"{_FREE} | {_BOUND} @ {kf}, {kr}", "kf", "kr", "Ki"))

    binding("kf" + _SLOWED, "kr" + _SLOWED)
    with pytest.raises(MechanismError, match=_DISSOCIATION):
        binding("kf", "kr" + _SPED)


def test_a_regulator_of_what_the_reaction_makes_names_its_degradation_and_its_row_builds():
    """sweep-55max c1: a repressor that speeds mRNA decay does not change the rate of synthesis.
    Signed on mRNA's steady state it would read as an inhibitor of `TF increases mRNA`, and
    `as_indra` would put it on synthesis; it is the degradation's, a DecreaseAmount row, which
    builds as the refusal writes it."""
    with pytest.raises(MechanismError, match=(
        r"reads `inhibited by Rep` only in `mRNA >> None`, mRNA's degradation, and its statement's "
        r"own reaction rate, its products absent, does not change with it: .* State it on a row "
        r"of that step instead, `mRNA is degraded, activated by Rep`: a DecreaseAmount "
        r"whose mechanism reads it, a written `rate` `kd\*\(1 \+ activated_by/K\)`"
    )):
        _build(["s: TF increases mRNA, inhibited by Rep | written"],
               _scheme("IncreaseAmount", "subj() >> subj() + obj() @ ks\n"
                       "obj() >> None @ kd*(1 + amount(inhibited_by)/K) delete_molecules",
                       "ks", "kd", "K"))
    _build(["s: TF increases mRNA | indra_default", "d: mRNA is degraded, activated by Rep | written"],
           _rate("kd*(1 + activated_by/K)", "kd", "K", stmt_type="DecreaseAmount"))


@pytest.mark.parametrize(
    "degradation",
    [
        "kd*(1 + amount(activated_by)/K)",  # b2
        "kd*amount(obj)*(1 + amount(activated_by)/K)",  # b3: second order in obj, itself
    ],
)
def test_turnover_with_basal_synthesis_is_signed_on_its_degradation(degradation):
    """sweep-55max: basal synthesis makes obj, so obj did not count as the reactant a
    DecreaseAmount removes, and the scheme made nothing (b2), or read as two intermediates meeting
    (b3, where obj meets itself). The statement says obj is what it removes."""
    _build(["d: mRNA is degraded, activated by X | written"],
           _scheme("DecreaseAmount", f"obj() >> None @ {degradation}\nNone >> obj() @ ks",
                   "kd", "ks", "K"))


@pytest.mark.parametrize(
    "stmt_type, row, scheme, refused",
    [
        # The conversion written backwards: X speeds its way back, which is A becoming B.
        ("Conversion", "c: A -> B, activated by X",
         "obj_to() | obj_from() @ kr, kf*amount(activated_by)", None),
    ],
)
def test_a_reversible_line_is_read_in_its_statements_direction(stmt_type, row, scheme, refused):
    """A `|` line written against its statement's direction was read left to right: signed
    backwards, or refused as not moving the reaction at all."""
    definition = _scheme(stmt_type, scheme, *sorted(set(M._RATE_TOKENS.findall(scheme)) & {
        "ks", "kd", "kf", "kr"}))
    if refused is None:
        _build([f"{row} | written"], definition)
        return
    with pytest.raises(MechanismError, match=refused):
        _build([f"{row} | written"], definition)


def test_two_names_for_one_regulator_are_one_amount():
    """`amount(inhibited_by)` on one step and `amount(inhibited_by_0)` on another read the same
    agent: one amount in J, not two read apart, one of which J alone does not move."""
    scheme = _TWO_STEPS.format(kf="kf/(1 + amount(inhibited_by)/Ki)", kr="kr",
                               kc="kc/(1 + amount(inhibited_by_0)/Ki)")
    _build([_PHOSPHO.format(regulator="inhibited by I")],
           _scheme("Phosphorylation", scheme, "kf", "kr", "kc", "Ki"))


def test_a_large_rate_is_signed_in_bounded_time():
    """sweep-55max's lucarelli gexpr13 did not return from `written` in 20 minutes: an unused
    placeholder role read as an intermediate, and the linearity test expanded the whole rate in
    it. Now it is signed factor by factor."""
    rate = (
        "kd*(1 + a34*amount(modulated_by_0)/(H34 + amount(modulated_by_0)) + "
        "a24*amount(modulated_by_1)/(H24 + amount(modulated_by_1)))*"
        "(1 - rho2*amount(inhibited_by)/(H2 + amount(inhibited_by)))/"
        "(1 + rho3*amount(modulated_by_2)/(H3 + amount(modulated_by_2)) + "
        "rhoD*(amount(modulated_by_1) + amount(modulated_by_0))/"
        "(HD + amount(modulated_by_1) + amount(modulated_by_0)) + rL)"
    )
    start = time.monotonic()
    _build(["g: TF increases SKI, inhibited by S2, modulated by S3 and S4 and S5 | written"],
           _scheme("IncreaseAmount", f"None >> obj() @ {rate}\nobj() >> None @ kd",
                   *"kd a34 a24 rho2 rho3 rhoD rL H2 H3 H24 H34 HD".split()))
    assert time.monotonic() - start < 60


# ------------------------------------------------------------------------------- every type
def _types():
    from petab_mcp.schemes import ABSTRACT

    return sorted({t for t, _ in M.REGISTRY} - ABSTRACT - {"ActiveForm"})


@pytest.mark.parametrize("stmt_type", _types())
def test_every_type_has_a_net_rate_for_a_forward_step_and_its_way_back(stmt_type):
    """review of #128: Activation, Inhibition, Gef, Gap and the self-modifications had no reaction
    in the statement's table, so a scheme of their forward step and a way back read as
    intermediates meeting, an honest activator was refused, and one acting off the statement's
    reaction escaped. Every type a row may use names what it consumes and makes; each builds its
    library entry's forward step, sped by an activator, with a basal way back."""
    from indra.statements import Agent, Complex, get_statement_by_name

    from petab_mcp.schemes import describe, placeholder

    assert any(cls in _REACTIONS for cls in get_statement_by_name(stmt_type).__mro__)
    base = "indra_default" if (stmt_type, "indra_default") in M.REGISTRY else "mass_action"
    forward = describe(stmt_type, base, M.REGISTRY).splitlines()[0]
    sides, rates = forward.split(" @ ")
    before, after = sides.split(" | " if " | " in sides else " >> ")
    rate = rates.split(", ")[0]
    scheme = f"{before} >> {after} @ {rate}*amount(activated_by)\n{after} >> {before} @ kback"
    definition = _scheme(stmt_type, scheme, rate, "kback")
    stmt = placeholder(stmt_type)
    library = dict(M.REGISTRY)
    library[(stmt_type, "written")] = M.written(definition, library)
    rows = [(stmt, "written", "r", None, {"activated_by": [Agent("X")]})]
    if stmt_type == "Transphosphorylation":  # its enzyme acts on the partner it is bound to
        rows.append((Complex([Agent("enz"), Agent("enz")]), "indra_default", "dimer"))
    model, emissions = M.assemble(rows, {}, library)
    problem_sbml._seed(model, list(model.monomers))
    for i, cp in enumerate(_reactants_in_full(model)):
        import pysb

        amount = pysb.Parameter(f"__given{i}", 1.0, _export=False)
        model.add_component(amount)
        model.add_initial(pysb.Initial(cp, amount, _export=False))
    problem_sbml._expand(model)
    rate_ = regulation.net_rate(model, emissions[0].rules, stmt)
    assert regulation.direction(rate_, "activated_by_0") == 1


def _reactants_in_full(model):
    """Each rule's reactant complexes, every site they leave unstated in its default: what starts,
    so a type whose forward step needs a complex (a transphosphorylation's dimer) fires."""
    import pysb

    seen = {}
    for rule in model.rules:
        for cp in rule.reactant_pattern.complex_patterns:
            if cp is None:
                continue
            full = pysb.ComplexPattern([
                mp.monomer(**{
                    site: mp.monomer.site_states[site][0] if site in mp.monomer.site_states else None
                    for site in mp.monomer.sites
                } | mp.site_conditions)
                for mp in cp.monomer_patterns
            ], None)
            if full.is_concrete():
                seen.setdefault(str(full), full)
    initial = {str(i.pattern) for i in model.initials}
    return [cp for key, cp in seen.items() if key not in initial]


# ------------------------------------------------------------------------------- never a crash
@pytest.mark.parametrize(
    "rows, definition",
    [
        # review of #128, t9: a kinase active only while bound (ANY), its catalysis inhibited.
        (["l: MEK binds L | indra_default",
          "p: MEK phosphorylates ERK at T202, inhibited by I | written"],
         _scheme("Phosphorylation",
                 "enz(sub=None, l=ANY) + sub(enz=None, mod='off') | "
                 "enz(sub=1, l=ANY) % sub(enz=1, mod='off') @ kf, kr\n"
                 "enz(sub=1, l=ANY) % sub(enz=1, mod='off') >> enz(sub=None, l=ANY) + "
                 "sub(enz=None, mod='on') @ kc/(1 + amount(inhibited_by))", "kf", "kr", "kc")),
        # t10: WILD in a two-step scheme's reactant.
        (["p: MEK phosphorylates ERK at T202, activated by X | written"],
         _scheme("Phosphorylation",
                 "enz(lig=WILD) + sub(mod='off') >> enz(lig=WILD) + sub(mod='on') @ "
                 "kc*amount(activated_by)\nsub(mod='on') >> sub(mod='off') @ kd", "kc", "kd")),
    ],
)
def test_a_pattern_no_placeholder_could_start_is_checked_without_a_crash(rows, definition):
    """review of #128: `ANY` and `WILD` made a placeholder's start not concrete, and the check
    crashed with PySB's "Pattern not concrete". The rows build, or are refused with a reason."""
    try:
        _build(rows, definition)
    except PetabMcpError:
        pass


def test_the_one_reaction_test_reads_names_apart():
    """review of #128: three points at which every name moved together read `exp(a - b)` as one
    value, so two reactions at unrelated rates passed as one."""
    a, b = sympy.symbols("a b", positive=True)
    assert not regulation._constant(sympy.exp(a), sympy.exp(b))
    assert regulation._constant(2 * a + 2 * b, a + b)
    # A piecewise branch where both are 0 says nothing against it.
    t = sympy.Symbol("time", nonnegative=True)
    switched = sympy.Piecewise((0, t < a), (b, True))
    assert regulation._constant(-switched, switched)


# ------------------------------------------------------------------------------- the prover
_FUZZ_NAMES = ["k", "K", "a", "n", "H", "rho", "t1"]


def _fuzz_rate(draw, depth=0) -> str:
    """A random rate over two regulators, a handful of positive names and `time` (the review of
    #128's fuzzer, 4,800 rates of which found no direction proven wrongly)."""
    if depth > 3 or draw.random() < 0.25:
        c = draw.random()
        if c < 0.35:
            return draw.choice(["inhibited_by", "inhibited_by", "activated_by"])
        if c < 0.8:
            return draw.choice(_FUZZ_NAMES)
        return "time" if c < 0.9 else draw.choice(["2", "0.5", "3"])
    a, b, c = _fuzz_rate(draw, depth + 1), _fuzz_rate(draw, depth + 1), draw.random()
    if c < 0.25:
        return f"({a} + {b})"
    if c < 0.38:
        return f"({a} - {b})"
    if c < 0.58:
        return f"{a}*{b}"
    if c < 0.73:
        return f"{a}/({b})"
    if c < 0.80:
        return f"({a})^{draw.choice(['2', '3', 'n', '0.5', '-1', 'a'])}"
    if c < 0.85:
        return f"exp({a})" if draw.random() < 0.5 else f"exp(-{a})"
    if c < 0.89:
        return f"log({a})"
    if c < 0.92:
        return f"sqrt({a})"
    return f"piecewise({a}, time < t1, {b})"


def test_a_proven_direction_holds_at_every_point_of_random_rates():
    """No direction :func:`mechanisms._direction` proves for a rate is contradicted by the rate's
    own slope at points across its names' ranges, regulators and `time` 0 among them."""
    import random

    import numpy as np

    draw, generator = random.Random(0), np.random.default_rng(0)
    checked = 0
    for _ in range(400):
        text = _fuzz_rate(draw)
        try:
            rate = M.assumed(M.parse_rate(text))
        except MechanismError:
            continue
        for name in ("inhibited_by", "activated_by"):
            symbol = sympy.Symbol(name, nonnegative=True)
            if symbol not in rate.free_symbols:
                continue
            proven = M._direction(rate, symbol)
            if proven not in (1, -1, 0):
                continue
            symbols = sorted(rate.free_symbols, key=str)
            slope = sympy.lambdify(symbols, sympy.diff(rate, symbol), "numpy")
            value = sympy.lambdify(symbols, rate, "numpy")
            columns = []
            for s in symbols:
                values = 10 ** generator.uniform(-3, 3, 2000)
                if not s.is_positive:
                    values[:100] = 0.0
                columns.append(values)
            with np.errstate(all="ignore"):
                d = np.asarray(slope(*columns), dtype=complex) * np.ones(2000)
                v = np.asarray(value(*[c.astype(complex) for c in columns]), dtype=complex) * np.ones(2000)
            real = (np.isfinite(d) & np.isfinite(v) & (np.abs(d.imag) < 1e-12 * np.maximum(1, np.abs(d.real)))
                    & (np.abs(v.imag) < 1e-12 * np.maximum(1, np.abs(v.real))))
            d = d.real[real]
            against = (d < -1e-9) if proven == 1 else (d > 1e-9) if proven == -1 else (np.abs(d) > 1e-9)
            assert not against.any(), f"{text}: {name} proven {proven}"
            checked += 1
    assert checked > 50


def test_a_regulator_of_synthesis_is_a_synthesis_rows_not_a_degradations():
    """review of #128, netrate #1: a DecreaseAmount's J counted synthesis, so X speeding
    `None >> obj` read as an inhibitor of degradation, though the degradation flux rises with X.
    Its way back is silent as a product's is, and X belongs on the synthesis row, which builds as
    the refusal writes it; the mirror of c1."""
    for scheme in ("obj() >> None @ kd\nNone >> obj() @ ks*(1 + amount(activated_by)/K)",
                   "None | obj() @ ks*(1 + amount(activated_by)/K), kd"):
        with pytest.raises(MechanismError, match=(
            r"reads `activated by X` only in `None >> mRNA`, mRNA's synthesis, .* State it on a row "
            r"of that step instead, `mRNA is produced, activated by X`: an IncreaseAmount"
        )):
            _build(["d: mRNA is degraded, activated by X | written"],
                   _scheme("DecreaseAmount", scheme, "kd", "ks", "K"))
    _build(["d: mRNA is degraded | indra_default", "s: mRNA is produced, activated by X | written"],
           _rate("ks*(1 + activated_by/K)", "ks", "K", stmt_type="IncreaseAmount"))


@pytest.mark.parametrize("scheme", [
    "obj_from_0() + obj_to_0() >> obj_to_0() + obj_to_0() @ ka*amount(activated_by)",
    "obj_from_0() >> obj_to_0() @ kb\n"
    "obj_from_0() + obj_to_0() >> obj_to_0() + obj_to_0() @ ka*amount(activated_by)",
])
def test_a_product_that_makes_itself_is_refused_alike_alone_or_beside_a_basal_step(scheme):
    """review of #128, netrate #4: with its products absent an autocatalytic step does not run,
    so J does not change with what speeds it, alone (J is 0) or beside a basal step; the two read
    alike. Whether products are held at 0 is the user's call to make."""
    with pytest.raises(MechanismError, match=r"reads `activated by X` only in `A \+ B >> B \+ B`"):
        _build(["c: A -> B, activated by X | written"],
               _scheme("Conversion", scheme, *sorted({"ka", "kb"} & set(M._RATE_TOKENS.findall(scheme)))))


def test_a_rate_of_several_steps_is_signed_through_their_coefficients():
    """review of #128, netrate #8: three saturating regulators on the catalytic step of two made J
    too large to sign whole, though the one-step twin signs all three. Signed by the chain rule
    through each step's coefficient, on J over the coefficients (King and Altman's form)."""
    factor = ("*(1 + a*amount(activated_by)/(Ha + amount(activated_by)))"
              "/(1 + r*amount(inhibited_by)/(Hi + amount(inhibited_by)))")
    _build([_PHOSPHO.format(regulator="activated by A, inhibited by I")],
           _scheme("Phosphorylation", _TWO_STEPS.format(kf="kf", kr="kr", kc="kc" + factor),
                   "kf", "kr", "kc", "a", "r", "Ha", "Hi"))


def test_an_enzyme_stated_as_its_own_inhibitor_is_one_amount():
    """review of #128, netrate #3: `E phosphorylates S, inhibited by E` with `kc/(1 + E/K)` gave
    the regulator a molecule of its own; the flux, kc*E*S/(1 + E/K), rises with E."""
    with pytest.raises(MechanismError, match=r"it rises instead: state it `activated by MEK`"):
        _build([_PHOSPHO.format(regulator="inhibited by MEK")], _rate("kc/(1 + inhibited_by/K)", "kc", "K"))


# ------------------------------------------------------------------------------- products held at 0
_RAS = 'c1: RAS{-mod"GTP"} -> RAS{mod"GTP"}'
_RAS_FED = f'{_RAS}, activated by SOS and RAS{{mod"GTP"}} | fed'
_GAP = 'c2: RAS{mod"GTP"} -> RAS{-mod"GTP"} | mass_action'
_FEEDBACK = "obj_from_0() >> obj_to_0() @ kb + kf*amount(activated_by_0)*amount(activated_by_1)"
_SELF = "obj_from_0() >> obj_to_0() @ ka*amount(activated_by_0)*amount(activated_by_1)"
_GENE = "subj() >> subj() + obj() @ ks + ka*amount(activated_by_0)*amount(activated_by_1)"


def _fed(stmt_type: str, scheme: str, *parameters: str) -> dict:
    return _scheme(stmt_type, scheme, *parameters, policy="fed")


@pytest.mark.parametrize(
    "rows, mechanism, stated, back, followed",
    [
        # review of #127/#128, refusals/t1: Ras-SOS feedback, SOS working through RAS-GTP, the
        # GAP step on the row.
        ([f"{_RAS}, activated by SOS | written"],
         _scheme("Conversion", "obj_from_0() >> obj_to_0() @ kb + kf*amount(activated_by)*"
                 "amount(obj_to_0())\nobj_to_0() >> obj_from_0() @ kgap", "kb", "kf", "kgap"),
         'RAS{-mod"GTP"} -> RAS{mod"GTP"}, activated by SOS and RAS{mod"GTP"}',
         'RAS{mod"GTP"} -> RAS{-mod"GTP"}',
         [([_RAS_FED], _fed("Conversion", f"{_FEEDBACK}\nobj_to_0() >> obj_from_0() @ kgap",
                            "kb", "kf", "kgap")),
          ([_RAS_FED, _GAP], _fed("Conversion", _FEEDBACK, "kb", "kf"))]),
        # The same step written alone.
        ([f"{_RAS}, activated by SOS | written"],
         _scheme("Conversion", "obj_from_0() >> obj_to_0() @ kb + kf*amount(activated_by)*"
                 "amount(obj_to_0())", "kb", "kf"),
         'RAS{-mod"GTP"} -> RAS{mod"GTP"}, activated by SOS and RAS{mod"GTP"}',
         'RAS{mod"GTP"} -> RAS{-mod"GTP"}',
         [([_RAS_FED], _fed("Conversion", _FEEDBACK, "kb", "kf")),
          ([_RAS_FED, _GAP], _fed("Conversion", _FEEDBACK, "kb", "kf"))]),
        # refusals/t1/autoreg: a co-activator working only with the gene product present.
        (["g: TF increases G, activated by X | written"],
         _scheme("IncreaseAmount", "subj() >> subj() + obj() @ ks + ka*amount(activated_by)*"
                 "amount(obj())\nobj() >> None @ kd delete_molecules", "ks", "ka", "kd"),
         "TF increases G, activated by X and G", "G is degraded",
         [(["g: TF increases G, activated by X and G | fed"],
           _fed("IncreaseAmount", f"{_GENE}\nobj() >> None @ kd delete_molecules", "ks", "ka", "kd")),
          (["g: TF increases G, activated by X and G | fed", "d: G is degraded | indra_default"],
           _fed("IncreaseAmount", _GENE, "ks", "ka"))]),
        # netrate/t6: the product makes itself, beside a basal step (a term of the one rule's
        # rate, as the refusal says: two rows of one reaction would be one reaction) or alone.
        (["c: A -> B, activated by X | written"],
         _scheme("Conversion", "obj_from_0() >> obj_to_0() @ kb\nobj_from_0() + obj_to_0() >> "
                 "obj_to_0() + obj_to_0() @ ka*amount(activated_by)", "ka", "kb"),
         "A -> B, activated by X and B", "B -> A",
         [(["c: A -> B, activated by X and B | fed"],
           _fed("Conversion", f"obj_from_0() >> obj_to_0() @ kb\n{_SELF}", "ka", "kb")),
          (["c: A -> B, activated by X and B | fed"],
           _fed("Conversion", "obj_from_0() >> obj_to_0() @ kb + ka*amount(activated_by_0)*"
                "amount(activated_by_1)", "ka", "kb"))]),
        (["c: A -> B, activated by X | written"],
         _scheme("Conversion", "obj_from_0() + obj_to_0() >> obj_to_0() + obj_to_0() @ "
                 "ka*amount(activated_by)", "ka"),
         "A -> B, activated by X and B", "B -> A",
         [(["c: A -> B, activated by X and B | fed"], _fed("Conversion", _SELF, "ka"))] * 2),
    ],
    ids=["ras_sos", "ras_sos_one_rule", "autoregulation", "autocatalysis_basal", "autocatalysis"],
)
def test_a_regulator_that_needs_the_product_is_told_both_rows_and_each_builds(
    rows, mechanism, stated, back, followed
):
    """Products stay at 0 in J (the user's decision, 2026-10-24), so a regulator that acts only
    with the product present (feedback, autoregulation, autocatalysis) does not move it. The
    refusal names the two rows that state it, written for the row, and each builds as written:
    the product stated as a regulator too; and that step alone on its own row so, its way back a
    statement of its own. The step alone without the product stated still reads it at 0."""
    with pytest.raises(MechanismError) as refused:
        _build(rows, mechanism)
    said = str(refused.value)
    assert f"State the product as a regulator too, `{stated}`" in said
    assert "Or put that step on a row of its own so, a one-rule mechanism" in said
    assert f"`{back}`" in said
    for rewrite, definition in followed:
        _build(rewrite, definition)
