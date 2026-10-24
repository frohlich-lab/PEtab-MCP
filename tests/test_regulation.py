"""A row's regulators, checked on the row's net rate in the model the rows build
(:mod:`petab_mcp.regulation`).

Each case builds rows through :func:`problem_sbml.to_antimony`, as `make_changes` does: J is read
off BioNetGen's expansion of every row, so a scheme is checked with the row's own agents, the
states other rows give them and its bindings, never a placeholder.
"""

from __future__ import annotations

import json
import time

import pandas as pd
import pysb
import pytest
import sympy
from conftest import rows_with_bindings
from indra.statements import stmts_from_json

from petab_mcp import mechanisms as M
from petab_mcp import problem_sbml, regulation
from petab_mcp.errors import MechanismError
from petab_mcp.provenance import binding_of
from petab_mcp.statements import _REACTIONS, indra_part, regulators, with_defaults


def _library(definitions) -> dict:
    library = dict(M.REGISTRY)
    for definition in definitions:
        library[(definition["statement_type"], definition["policy"])] = M.written(definition, library)
    return library


def _build(rows: list, *definitions: dict, times=None):
    """``rows`` (each a row and its binding) built with ``definitions`` written, ``time`` over
    ``times``: the emissions. Raises what `make_changes` refuses with."""
    library = _library(definitions)
    parsed = rows_with_bindings(rows, library)
    for row in parsed:
        row["statement"] = with_defaults(row["statement"])
    stmts = stmts_from_json([indra_part(row["statement"]) for row in parsed])
    _, emissions = problem_sbml.to_antimony(
        [(s, row["policy"], row["prefix"], row.get("parameters"), regulators(row["statement"]))
         for s, row in zip(stmts, parsed)],
        {}, library, times=times,
    )
    return emissions


def _scheme(stmt_type: str, scheme: str, *parameters: str, policy: str = "written") -> dict:
    return {"statement_type": stmt_type, "policy": policy, "scheme": scheme,
            "parameters": list(parameters)}


def _rate(rate: str, *parameters: str, stmt_type="Phosphorylation", based_on="indra_default"):
    return {"statement_type": stmt_type, "policy": "written", "based_on": based_on, "rate": rate,
            "parameters": list(parameters)}


_PHOSPHO = "p: MEK phosphorylates ERK at T202, {regulator}"


def _phospho(regulator: str) -> tuple[str, str]:
    """MEK phosphorylating ERK, ``regulator`` stated, bound to the mechanism written as `written`."""
    return _PHOSPHO.format(regulator=regulator), "written"


# ------------------------------------------------------------------------------- written rates
@pytest.mark.parametrize(
    "rate, regulator, derived",
    [
        # Upgraded: a modulator the rate proves a direction for.
        ("k * Ki / (Ki + modulated_by)", "modulated by D", {"modulated_by_0": "inhibited_by"}),
        ("k * modulated_by", "modulated by D", {"modulated_by_0": "activated_by"}),
        # Reversed: an inhibitor the rate rises with, an activator it falls with.
        ("k*inhibited_by/(Ki + inhibited_by)", "inhibited by D", {"inhibited_by_0": "activated_by"}),
        ("k*Ki/(Ki + activated_by)", "activated by D", {"activated_by_0": "inhibited_by"}),
        # Downgraded: sneyd's Ca, which way it moves the rate depending on the constants, and a
        # switch, whose jump is no derivative.
        ("(k + Ki*inhibited_by)/(1 + inhibited_by)", "inhibited by D",
         {"inhibited_by_0": "modulated_by"}),
        ("k*piecewise(1, activated_by > 0, 0)", "activated by D", {"activated_by_0": "modulated_by"}),
        # Left as stated.
        ("k * Ki / (Ki + inhibited_by)", "inhibited by D", {}),
        ("k/(i*inhibited_by^6/(inhibited_by^6 + Ki^6) + 1)", "inhibited by D", {}),  # alkan
        ("k*activated_by^6/(activated_by^6 + Ki^6)", "activated by D", {}),  # alkan's driven death
        # Negative past D = 1, a rate law with no direction there (the second review of #128).
        ("k*(1 - inhibited_by)", "inhibited by D", {"inhibited_by_0": "modulated_by"}),
        ("k*piecewise(1, modulated_by > 0, 0)", "modulated by D", {}),
        ("(k + Ki*modulated_by)/(1 + modulated_by)", "modulated by D", {}),
    ],
)
def test_a_regulator_is_recorded_as_the_kind_its_net_rate_shows(rate, regulator, derived):
    """The statement keeps what it states, and the row records beside it the kind its net rate
    shows where that differs: a modulator signed, a claim downgraded or reversed, never refused."""
    names = sorted({"k", "Ki", "i"} & set(M._RATE_TOKENS.findall(rate)))
    (emission,) = _build([_phospho(regulator)], _rate(rate, *names))
    assert emission.derived == derived


def test_a_regulator_nothing_forms_is_refused_naming_what_to_add():
    """A rate reading PTEN phosphorylated where nothing phosphorylates PTEN does not change with
    it: the model holds no such PTEN."""
    with pytest.raises(MechanismError, match=r"nothing in the model forms PTEN\{pS380\}, so the "
                       r"rate does not change with it. Add the statement that forms it"):
        _build([_phospho("inhibited by PTEN{pS380}")],
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
    return _build([_phospho("inhibited by I")], scheme)


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


def test_a_regulator_that_moves_the_net_rate_the_other_way_is_recorded_so():
    """Slowing unbinding leaves more complex to turn over: it speeds the reaction, whatever its
    own rate does, so the inhibitor the statement claims is recorded as an activator."""
    (emission,) = _two_steps(kr="kr" + _SLOWED)
    assert emission.derived == {"inhibited_by_0": "activated_by"}


def test_a_modulator_of_a_scheme_takes_the_direction_of_its_net_rate():
    """The net rate decides, as for a stated regulator: slowing unbinding activates."""
    for rates, kind in (({"kc": "kc" + _SLOWED}, "inhibited_by"),
                        ({"kr": "kr" + _SLOWED}, "activated_by")):
        scheme = _TWO_STEPS.format(**{"kf": "kf", "kr": "kr", "kc": "kc"} | rates)
        (emission,) = _build([_phospho("modulated by I")], _scheme(
            "Phosphorylation", scheme.replace("inhibited_by", "modulated_by"),
            "kf", "kr", "kc", "Ki"))
        assert emission.derived == {"modulated_by_0": kind}


def test_a_binding_line_written_dissociation_first_is_the_same_scheme():
    """review of #128, netrate t2: `ES | E + S` read as written held the complex as a reactant, so
    an inhibitor of binding read as not moving J at all."""
    scheme = (
        "enz(sub=1) % sub(enz=1, mod='off') | enz(sub=None) + sub(enz=None, mod='off') @ kr, "
        "kf" + _SLOWED + "\n"
        "enz(sub=1) % sub(enz=1, mod='off') >> enz(sub=None) + sub(enz=None, mod='on') @ kc"
    )
    _build([_phospho("inhibited by I")],
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
    _build([_phospho("inhibited by I")],
           _scheme("Phosphorylation", dead_end, "kf", "kr", "kc", "kd", "kb", "Ki"))


def test_a_row_without_one_net_rate_states_its_regulators_without_a_direction():
    """A scheme that also destroys its substrate is two reactions at unrelated rates: no one net
    rate, so its regulators claim no direction."""
    branched = (
        "enz() + sub(mod='off') >> enz() + sub(mod='on') @ kc/(1 + amount({kind}_by)/Ki)\n"
        "enz() + sub(mod='off') >> enz() @ kd"
    )
    (inhibited,) = _build(
        [_phospho("inhibited by I")],
        _scheme("Phosphorylation", branched.format(kind="inhibited"), "kc", "kd", "Ki"),
    )
    assert inhibited.derived == {"inhibited_by_0": "modulated_by"}
    assert "two reactions, with no one net rate" in inhibited.rate["none"]
    (modulated,) = _build(
        [_phospho("modulated by I")],
        _scheme("Phosphorylation", branched.format(kind="modulated"), "kc", "kd", "Ki"),
    )
    assert modulated.derived == {}


#: A GTPase cycle: a GEF turns obj_from into obj_to, a GAP turns it back.
_CYCLE = (
    "obj_from_0() >> obj_to_0() @ kf*amount(activated_by_0){allo}\n"
    "obj_to_0() >> obj_from_0() @ kgap{basal}"
)


def _cycle(allo="", basal="", stated="activated by SOS"):
    parameters = ["kf", "kgap"] + (["allo"] if allo else []) + (["kb"] if basal else [])
    kind = stated.split(" by")[0]
    scheme = _CYCLE.format(allo=allo, basal=basal).replace("activated_by_0", f"{kind}_by_0")
    return _build([(f'c: RAS{{-mod"GTP"}} -> RAS{{mod"GTP"}}, {stated}', "written")],
                  _scheme("Conversion", scheme, *parameters))


def test_a_cycle_back_to_its_substrate_is_checked_on_its_forward_step():
    """perica_egf's RAS cycle (`ras_gef`, run 61230791_0) was refused as making nothing: its step
    back makes obj_from again, so obj_from read as an intermediate and nothing as a reactant. It is
    the conversion's input, which the statement says; with obj_to absent the step back is silent."""
    _cycle()
    # ras_gef3: three steps, the GEF sped by RAS-GTP and a basal exchange beside it.
    _cycle(allo="*(1 + allo*amount(activated_by_1))", basal="\nobj_from_0() >> obj_to_0() @ kb",
           stated='activated by SOS and RAS{mod"GTP"}')
    (reversed_,) = _cycle(stated="inhibited by SOS")
    assert reversed_.derived == {"inhibited_by_0": "activated_by"}


def test_a_scheme_gated_on_a_state_other_rows_give_is_checked_on_the_real_row():
    """Eight one-rule perica_egf schemes (`mek_act_braf`, `egfr_act2`, ...) write a site in one
    state (`enz(gtpbound='active')`) whose other states other rows give: a placeholder statement
    could not build them, the row as built does."""
    gef = ({"statement": {"type": "Gef", "gef": {"name": "SOS"}, "ras": {"name": "KRAS"}},
            "prefix": "gef"}, {"policy": "indra_default"})

    def gated(rate):
        scheme = _scheme("Phosphorylation", "enz(gtpbound='active') + sub(mod='off') >> "
                         f"enz(gtpbound='active') + sub(mod='on') @ {rate}", "kf")
        return _build([gef, ("p: KRAS phosphorylates RAF at S338, activated by X", "written")], scheme)

    assert gated("kf*amount(activated_by)")[1].derived == {}
    assert gated("kf/(1 + amount(activated_by))")[1].derived == {"activated_by_0": "inhibited_by"}


#: A binding written as two one-way lines (isensee's `camp_release*`), and its two sides.
_BOUND = "members_0(members_1=1) % members_1(members_0=1)"
_FREE = "members_0(members_1=None) + members_1(members_0=None)"


def _binding(on, off, stated="inhibited by X"):
    return _build([(f"b: A binds B, {stated}", "written")],
                  _scheme("Complex", f"{_FREE} >> {_BOUND} @ {on}\n{_BOUND} >> {_FREE} @ {off}",
                          "kf", "kr", "K"))


#: What a binding's regulator of its way back alone (sweep-55max a1, a2) is refused with, and the
#: Conversion it is told to write instead.
_DISSOCIATION = (
    r"reads `inhibited by X` only in `A % B >> A \+ B`, the dissociation, and its statement's own "
    r"reaction rate, its products absent, does not change with it: a statement's regulator is "
    r"about its own reaction\. State it on a row of that step instead, `A\{!B\} -> A \+ "
    r"B, activated by X`, bound to a Conversion scheme that breaks the bond and reads it, "
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
    _build([("b: A binds B", "indra_default"), ("u: A{!B} -> A + B, activated by X", "apart")], _APART)


def test_a_reversible_step_is_read_both_ways_on_its_own_row():
    """With its product absent a step does not run back, so what slows only its way back claims
    nothing a regulator of the reaction does (sweep-55max a1)."""
    def binding(kf, kr):
        return _build([("b: A binds B, inhibited by X", "written")],
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
        r"of that step instead, `mRNA is degraded, activated by Rep`, bound to a DecreaseAmount "
        r"mechanism that reads it, a written `rate` `kd\*\(1 \+ activated_by/K\)`"
    )):
        _build([("s: TF increases mRNA, inhibited by Rep", "written")],
               _scheme("IncreaseAmount", "subj() >> subj() + obj() @ ks\n"
                       "obj() >> None @ kd*(1 + amount(inhibited_by)/K) delete_molecules",
                       "ks", "kd", "K"))
    _build([("s: TF increases mRNA", "indra_default"), ("d: mRNA is degraded, activated by Rep", "written")],
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
    _build([("d: mRNA is degraded, activated by X", "written")],
           _scheme("DecreaseAmount", f"obj() >> None @ {degradation}\nNone >> obj() @ ks",
                   "kd", "ks", "K"))


@pytest.mark.parametrize(
    "stmt_type, row, scheme, derived",
    [
        # The conversion written backwards: X speeds its way back, which is A becoming B.
        ("Conversion", "c: A -> B, activated by X",
         "obj_to() | obj_from() @ kr, kf*amount(activated_by)", {}),
    ],
)
def test_a_reversible_line_is_read_in_its_statements_direction(stmt_type, row, scheme, derived):
    """A `|` line written against its statement's direction was read left to right: signed
    backwards, or refused as not moving the reaction at all."""
    definition = _scheme(stmt_type, scheme, *sorted(set(M._RATE_TOKENS.findall(scheme)) & {
        "ks", "kd", "kf", "kr"}))
    (emission,) = _build([(row, "written")], definition)
    assert emission.derived == derived


def test_two_names_for_one_regulator_are_one_amount():
    """`amount(inhibited_by)` on one step and `amount(inhibited_by_0)` on another read the same
    agent: one amount in J, not two read apart, one of which J alone does not move."""
    scheme = _TWO_STEPS.format(kf="kf/(1 + amount(inhibited_by)/Ki)", kr="kr",
                               kc="kc/(1 + amount(inhibited_by_0)/Ki)")
    _build([_phospho("inhibited by I")],
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
    _build([("g: TF increases SKI, inhibited by S2, modulated by S3 and S4 and S5", "written")],
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
        ([("l: MEK binds L", "indra_default"),
          ("p: MEK phosphorylates ERK at T202, inhibited by I", "written")],
         _scheme("Phosphorylation",
                 "enz(sub=None, l=ANY) + sub(enz=None, mod='off') | "
                 "enz(sub=1, l=ANY) % sub(enz=1, mod='off') @ kf, kr\n"
                 "enz(sub=1, l=ANY) % sub(enz=1, mod='off') >> enz(sub=None, l=ANY) + "
                 "sub(enz=None, mod='on') @ kc/(1 + amount(inhibited_by))", "kf", "kr", "kc")),
        # t10: WILD in a two-step scheme's reactant.
        ([("p: MEK phosphorylates ERK at T202, activated by X", "written")],
         _scheme("Phosphorylation",
                 "enz(lig=WILD) + sub(mod='off') >> enz(lig=WILD) + sub(mod='on') @ "
                 "kc*amount(activated_by)\nsub(mod='on') >> sub(mod='off') @ kd", "kc", "kd")),
    ],
)
def test_a_pattern_no_placeholder_could_start_is_checked_without_a_crash(rows, definition):
    """review of #128: `ANY` and `WILD` made a placeholder's start not concrete, and the check
    crashed with PySB's "Pattern not concrete". Each regulator reads as stated."""
    assert _build(rows, definition)[-1].derived == {}


def test_a_bug_reading_the_net_rate_is_recorded_as_a_failure(monkeypatch):
    """review of #127: a bug of the check read as "no one net rate ... rewrite its rate", a reason
    to weaken the claim. It claims no direction, never a crash, is counted, and says it is the
    server's."""
    from petab_mcp.operations import regulator_warning

    def broken(*_):
        raise ZeroDivisionError("a bug")

    monkeypatch.setattr(regulation, "net_rate", broken)
    record = _record([_phospho("inhibited by I")],
                     _rate("k/(1 + inhibited_by)", "k"))
    signed, failed, _ = regulation.resign(record, _Problem([_parameter("p_k", 1e-3, 1e3)]))
    assert failed == ["p"] and signed["regulator_check_failures"] == 1
    assert signed["rows"][0]["emission"]["derived"] == {"inhibited_by_0": "modulated_by"}
    said = regulator_warning(signed, None, (), failed)
    assert said == ("`p`: the server failed checking its regulators (a bug of its own, logged), so "
                    "they claim no direction until a later edit checks them; its mechanism needs no "
                    "change for this.")


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


# ------------------------------------------------------------------------------- one amount each
@pytest.mark.parametrize(
    "row, rate, derived",
    [
        # review of #128, b_overlap: the regulator is the reactant; J = k*naive/(K + naive) rises.
        ("r: naive becomes effector, inhibited by naive", "k/(K + inhibited_by)",
         {"inhibited_by_0": "activated_by"}),
        # rewrite t6: one agent stated twice, inhibitor and activator: its one amount moves J both
        # ways, so neither claim is shown.
        ("r: A becomes B, inhibited by X, activated by X",
         "k*K/(K + inhibited_by)*(1 + activated_by/Ka)",
         {"inhibited_by_0": "modulated_by", "activated_by_0": "modulated_by"}),
        # rewrite t6: one agent twice, a bell.
        ("r: A becomes B, modulated by X and X", "k*modulated_by_0/(K + modulated_by_1)^2", {}),
        # prover alias: two names of the one agent stated, (1 + D)/(1 + 2D) falls.
        ("r: A becomes B, inhibited by D", "k*(1 + inhibited_by_0)/(1 + 2*inhibited_by)", {}),
    ],
)
def test_one_agent_is_one_amount_however_it_is_named(row, rate, derived):
    names = sorted({"k", "K", "Ka"} & set(M._RATE_TOKENS.findall(rate)))
    (emission,) = _build([(row, "written")],
                         _rate(rate, *names, stmt_type="PopulationTransition", based_on="mass_action"))
    assert emission.derived == derived


def test_overlapping_patterns_of_one_agent_claim_no_direction():
    """review of #128, b_overlap: D phosphorylated and D in all its forms share a species, so
    neither moves J alone, whatever either rate does on its own."""
    emissions = _build(
        [("k: K phosphorylates D at S10", "indra_default"),
         ("r: A becomes B, activated by D{pS10}, inhibited by D", "two")],
        _rate("k*(1 + activated_by)/(1 + 2*inhibited_by)", "k", stmt_type="PopulationTransition",
              based_on="mass_action") | {"policy": "two"},
    )
    assert emissions[1].derived == {"activated_by_0": "modulated_by",
                                    "inhibited_by_0": "modulated_by"}


def test_a_large_rate_signs_its_modulators_in_bounded_time():
    """gexpr13's three modulators each get their sign, factor by factor, in bounded time; one the
    rate does not provably move one way stays unsigned."""
    rate = (
        "kd*(1 + a34*amount(modulated_by_0)/(H34 + amount(modulated_by_0)) + "
        "a24*amount(modulated_by_1)/(H24 + amount(modulated_by_1))){inhibition}/"
        "(1 + rho3*amount(modulated_by_2)/(H3 + amount(modulated_by_2)) + "
        "rhoD*(amount(modulated_by_1) + amount(modulated_by_0))/"
        "(HD + amount(modulated_by_1) + amount(modulated_by_0)) + rL)"
    )
    for inhibition, derived in [
        # 1 - rho2*X/(H2 + X) turns negative for rho2 > 1, and J with it: no regulator of it is
        # given a direction, the stated inhibitor included.
        ("*(1 - rho2*amount(inhibited_by)/(H2 + amount(inhibited_by)))",
         {"inhibited_by_0": "modulated_by"}),
        # Kept positive, J falls with modulated_by_2, only in the denominator; modulated_by_0 and
        # _1, in the numerator and the denominator both, stay unsigned.
        ("/(1 + rho2*amount(inhibited_by)/(H2 + amount(inhibited_by)))",
         {"modulated_by_2": "inhibited_by"}),
    ]:
        start = time.monotonic()
        (emission,) = _build(
            [("g: TF increases SKI, inhibited by S2, modulated by S3 and S4 and S5", "written")],
            _scheme("IncreaseAmount",
                    f"None >> obj() @ {rate.format(inhibition=inhibition)}\nobj() >> None @ kd",
                    *"kd a34 a24 rho2 rho3 rhoD rL H2 H3 H24 H34 HD".split()),
        )
        assert time.monotonic() - start < 60
        assert emission.derived == derived


# ------------------------------------------------------------------------------- a version's bounds
def _record(rows, *definitions) -> dict:
    """``rows`` built, as a version's statements record keeps them."""
    from dataclasses import asdict

    parsed = rows_with_bindings(rows, _library(definitions))
    return {"rows": [{"prefix": row["prefix"], "statement": with_defaults(row["statement"]),
                      "emission": asdict(e)} for row, e in zip(parsed, _build(rows, *definitions))],
            "bindings": {row["prefix"]: binding_of(row) for row in parsed}}


class _Problem:
    """A problem's tables and its model's values (``values``), as :func:`regulation.bounded`
    reads them."""

    def __init__(self, parameters: list[dict], conditions=None, times=None, values=None):
        import pandas as pd

        self.parameter_df = pd.DataFrame(parameters).set_index("parameterId")
        self.condition_df = pd.DataFrame(conditions) if conditions else None
        self.measurement_df = pd.DataFrame({"time": times, "experimentId": ""}) if times else None
        self.experiment_df = None
        self.model = _Model(values or {})


class _Model:
    """A problem's model, as :func:`regulation.bounded` reads a condition input's own value."""

    def __init__(self, values: dict[str, float]):
        self.values = values

    def get_parameter_value(self, pid: str) -> float:
        return self.values[pid]


def _parameter(pid, low, high, estimate=1, nominal=1.0, scale="lin"):
    return {"parameterId": pid, "lowerBound": low, "upperBound": high, "nominalValue": nominal,
            "estimate": estimate, "parameterScale": scale}


@pytest.mark.parametrize(
    "a, derived",
    [
        (_parameter("conv_a", 0.0, 10.0), {}),  # never negative: activates
        (_parameter("conv_a", 1e-3, 10.0, scale="log10"), {}),
        # Always negative: J falls with C, and below 0 past C = 1/|a|, so no direction.
        (_parameter("conv_a", -10.0, -1e-3), {"activated_by_0": "modulated_by"}),
        (_parameter("conv_a", -1.0, 1.0), {"activated_by_0": "modulated_by"}),  # either sign
        (_parameter("conv_a", -1.0, 1.0, estimate=0, nominal=-0.5),
         {"activated_by_0": "modulated_by"}),
    ],
)
def test_a_regulator_is_signed_on_its_versions_bounds(a, derived):
    """review of #128: every name but a regulator was taken positive, so `k*(1 + a*C)` read
    `activated by C` whatever `a`'s bounds; a fit that drives `a` negative inverts the claim. The
    version's own table decides: a fixed parameter by its value, an estimated one by its bounds."""
    record = _record([("conv: A becomes B, activated by C", "lin")],
                     _rate("k*(1 + a*activated_by)", "k", "a", stmt_type="PopulationTransition",
                           based_on="mass_action") | {"policy": "lin"})
    problem = _Problem([_parameter("conv_k", 1e-3, 1e3, scale="log10"), a])
    signed, failed, _ = regulation.resign(record, problem)
    assert failed == []
    assert signed["rows"][0]["emission"]["derived"] == derived


def test_a_condition_input_and_time_are_signed_on_their_ranges():
    """A condition input ranges over the values its conditions set and the model's own, `time`
    over the problem's experiments: `k*(1 + (dose - 1)*C)` activates for doses of at least 1, not
    for 0.5."""
    definition = _rate("k*(1 + d*activated_by)", "k", "d", stmt_type="PopulationTransition",
                       based_on="mass_action") | {"policy": "dosed"}
    record = _record([("conv: A becomes B, activated by C", 'dosed d="dose - 1"')], definition)
    for doses, derived in (([1.0, 2.0, 4.0], {}), ([0.5, 2.0], {"activated_by_0": "modulated_by"})):
        problem = _Problem(
            [_parameter("conv_k", 1e-3, 1e3, scale="log10")],
            [{"conditionId": f"c{i}", "targetId": "dose", "targetValue": d} for i, d in enumerate(doses)],
            times=[0.0, 10.0], values={"dose": 1.0},
        )
        signed, _, _ = regulation.resign(record, problem)
        assert signed["rows"][0]["emission"]["derived"] == derived


def test_a_stored_net_rate_reads_back_as_it_was_built():
    record = _record([_phospho("inhibited by I")],
                     _scheme("Phosphorylation", _TWO_STEPS.format(kf="kf", kr="kr", kc="kc" + _SLOWED),
                             "kf", "kr", "kc", "Ki"))
    rate = regulation.restored(record["rows"][0]["emission"]["rate"])
    assert regulation.direction(rate, "inhibited_by_0") == -1
    assert set(rate.parameters.values()) == {"p_kf", "p_kr", "p_kc", "p_Ki"}


def test_a_net_rate_that_does_not_read_back_claims_no_direction_and_is_counted():
    """A sign that breaks is no crash: the regulator is recorded as modulated, the row counted."""
    record = _record([("conv: A becomes B, inhibited by C", "inh")],
                     _rate("k*K/(K + inhibited_by)", "k", "K", stmt_type="PopulationTransition",
                           based_on="mass_action") | {"policy": "inh"})
    readable = record["rows"][0]["emission"]["rate"]["J"]
    record["rows"][0]["emission"]["rate"]["J"] = "p0*("
    signed, failed, _ = regulation.resign(record, _Problem([_parameter("conv_k", 1e-3, 1e3)]))
    assert failed == ["conv"]
    assert signed["rows"][0]["emission"]["derived"] == {"inhibited_by_0": "modulated_by"}
    assert signed["regulator_check_failures"] == 1
    # The count is the version's own: signed again cleanly, it is gone.
    signed["rows"][0]["emission"]["rate"]["J"] = readable
    again, failed, _ = regulation.resign(signed, _Problem([_parameter("conv_k", 1e-3, 1e3)]))
    assert failed == [] and "regulator_check_failures" not in again
    assert again["rows"][0]["emission"]["derived"] == {}


def test_a_row_built_before_its_net_rate_was_kept_keeps_its_kinds():
    """review of #128: a record from before this PR has no stored net rate, and its first
    parameters-only commit after a resume downgraded every signed claim to modulated, telling the
    agent to rewrite rates nothing had read. Such a row keeps what it had, and is named as
    unsigned."""
    record = _record([("conv: A becomes B, inhibited by C", "inh")],
                     _rate("k*K/(K + inhibited_by)", "k", "K", stmt_type="PopulationTransition",
                           based_on="mass_action") | {"policy": "inh"})
    record["rows"][0]["emission"]["rate"] = {}
    record["rows"][0]["emission"]["derived"] = {"inhibited_by_0": "activated_by"}  # as it was
    signed, failed, unsigned = regulation.resign(record, _Problem([_parameter("conv_k", 1e-3, 1e3)]))
    assert (failed, unsigned) == ([], ["conv"])
    assert signed["rows"][0]["emission"]["derived"] == {"inhibited_by_0": "activated_by"}
    from petab_mcp.operations import regulator_warning

    assert "`conv`: built before regulators were signed on a version's bounds, so they keep the " \
           "kinds they had" in regulator_warning(signed, record, unsigned)


def test_a_condition_input_is_signed_on_its_conditions_whatever_a_parameter_row_says():
    """review of #128: with `dose` both fixed at 2 in the parameter table and set to 0.5 by a
    condition, the table won, and `k*(1 + (dose - 1)*C)` read `activated by C` though condition
    c0 inhibits. PEtab forbids the overlap; the conditions win anyway."""
    definition = _rate("k*(1 + d*activated_by)", "k", "d", stmt_type="PopulationTransition",
                       based_on="mass_action") | {"policy": "dosed"}
    record = _record([("conv: A becomes B, activated by C", 'dosed d="dose - 1"')], definition)
    problem = _Problem(
        [_parameter("conv_k", 1e-3, 1e3, scale="log10"),
         _parameter("dose", 2.0, 2.0, estimate=0, nominal=2.0)],
        [{"conditionId": f"c{i}", "targetId": "dose", "targetValue": d} for i, d in enumerate((0.5, 2.0))],
    )
    signed, _, _ = regulation.resign(record, problem)
    assert signed["rows"][0]["emission"]["derived"] == {"activated_by_0": "modulated_by"}


def test_the_warning_says_why_a_row_has_no_net_rate_and_when_a_kind_is_back():
    """review of #128: a row of two reactions was told, as a sneyd-like rate is, that its rate
    cannot be shown to move one way and to rewrite it; and a bound edit that brought back the kind
    stated said nothing."""
    from petab_mcp.operations import regulator_warning

    branched = _scheme("Phosphorylation", "enz() + sub(mod='off') >> enz() + sub(mod='on') @ "
                       "kc/(1 + amount(inhibited_by)/Ki)\nenz() + sub(mod='off') >> enz() @ kd",
                       "kc", "kd", "Ki")
    record = _record([_phospho("inhibited by I")], branched)
    said = regulator_warning(record, None)
    assert said.startswith("`p`: you stated `inhibited by I`, but its mechanism has no one net "
                           "rate to show a direction on (it changes ")
    assert "two reactions, with no one net rate), so it reads as `modulated by I`" in said
    assert "a mechanism with no one net rate (two reactions, say) is one statement for each" in said
    signed = _record([("conv: A becomes B, activated by C", "lin")],
                     _rate("k*(1 + a*activated_by)", "k", "a", stmt_type="PopulationTransition",
                           based_on="mass_action") | {"policy": "lin"})
    earlier = json.loads(json.dumps(signed))
    earlier["rows"][0]["emission"]["derived"] = {"activated_by_0": "modulated_by"}
    assert regulator_warning(signed, earlier) == "`conv`: `activated by C` reads as you stated it again."


@pytest.mark.parametrize("b, derived", [("-2", {}), ("2", {"inhibited_by_0": "activated_by"})])
def test_a_pinned_constant_is_signed_as_its_value(b, derived):
    """review of #128, a_e2e_pin: `k*exp(b*C)` with `b=-2` was signed with `b` positive, so a
    correct `inhibited by C` was recorded as `activated by C`. The row's binding is its value."""
    (emission,) = _build([("conv: A becomes B, inhibited by C", f"loglin b={b}")],
                         _rate("k*exp(b*inhibited_by)", "k", "b", stmt_type="PopulationTransition",
                               based_on="mass_action") | {"policy": "loglin"})
    assert emission.derived == derived


def test_a_regulator_of_synthesis_is_a_synthesis_rows_not_a_degradations():
    """review of #128, netrate #1: a DecreaseAmount's J counted synthesis, so X speeding
    `None >> obj` read as an inhibitor of degradation, though the degradation flux rises with X.
    Its way back is silent as a product's is, and X belongs on the synthesis row, which builds as
    the refusal writes it; the mirror of c1."""
    for scheme in ("obj() >> None @ kd\nNone >> obj() @ ks*(1 + amount(activated_by)/K)",
                   "None | obj() @ ks*(1 + amount(activated_by)/K), kd"):
        with pytest.raises(MechanismError, match=(
            r"reads `activated by X` only in `None >> mRNA`, mRNA's synthesis, .* State it on a row "
            r"of that step instead, `mRNA is produced, activated by X`, bound to an "
            r"IncreaseAmount"
        )):
            _build([("d: mRNA is degraded, activated by X", "written")],
                   _scheme("DecreaseAmount", scheme, "kd", "ks", "K"))
    _build([("d: mRNA is degraded", "indra_default"), ("s: mRNA is produced, activated by X", "written")],
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
        _build([("c: A -> B, activated by X", "written")],
               _scheme("Conversion", scheme, *sorted({"ka", "kb"} & set(M._RATE_TOKENS.findall(scheme)))))


def test_a_rate_of_several_steps_is_signed_through_their_coefficients():
    """review of #128, netrate #8: three saturating regulators on the catalytic step of two made J
    too large to sign whole, though the one-step twin signs all three. Signed by the chain rule
    through each step's coefficient, on J over the coefficients (King and Altman's form)."""
    factor = ("*(1 + a*amount(activated_by)/(Ha + amount(activated_by)))"
              "/(1 + r*amount(inhibited_by)/(Hi + amount(inhibited_by)))")
    (emission,) = _build([_phospho("activated by A, inhibited by I")], _scheme(
        "Phosphorylation", _TWO_STEPS.format(kf="kf", kr="kr", kc="kc" + factor),
        "kf", "kr", "kc", "a", "r", "Ha", "Hi"))
    assert emission.derived == {}


def test_an_enzyme_stated_as_its_own_inhibitor_is_one_amount():
    """review of #128, netrate #3: `E phosphorylates S, inhibited by E` with `kc/(1 + E/K)` gave
    the regulator a molecule of its own; the flux, kc*E*S/(1 + E/K), rises with E."""
    (emission,) = _build([_phospho("inhibited by MEK")],
                         _rate("kc/(1 + inhibited_by/K)", "kc", "K"))
    assert emission.derived == {"inhibited_by_0": "activated_by"}


@pytest.mark.parametrize("row", ["c: A + E -> B + E, {} X", "c: L + R -> L + R{{pS10}}, {} X"])
def test_an_agent_a_reaction_gives_back_as_it_was_is_a_catalyst(row):
    """review of #128 (second pass), t_catalyst: every agent a conversion gives was a product,
    so the `E` of `A + E -> B + E` was held at 0, J was 0 and the row refused as needing the
    product. E is given back as it was: a catalyst, its amount held, and J = k*X*A*E."""
    scheme = _scheme("Conversion", "obj_from_0() + obj_from_1() >> obj_to_0() + obj_to_1() @ "
                     "k*amount({})", "k")
    (emission,) = _build([(row.format("activated by"), "written")],
                         {**scheme, "scheme": scheme["scheme"].format("activated_by")})
    assert emission.derived == {}
    (emission,) = _build([(row.format("inhibited by"), "written")],
                         {**scheme, "scheme": scheme["scheme"].format("inhibited_by")})
    assert emission.derived == {"inhibited_by_0": "activated_by"}


# ------------------------------------------------------------------------------- own forms
#: Three steps: binding, the catalytic step slowed by the regulator, and release.
_THREE_STEPS = (
    "enz(sub=None) + sub(enz=None, mod='off') | enz(sub=1) % sub(enz=1, mod='off') @ kf, kr\n"
    "enz(sub=1) % sub(enz=1, mod='off') >> enz(sub=1) % sub(enz=1, mod='on') @ "
    "kc/(1 + amount({kind})/Ki)\n"
    "enz(sub=1) % sub(enz=1, mod='on') >> enz(sub=None) + sub(enz=None, mod='on') @ kp"
)


@pytest.mark.parametrize("kind", ["activated_by", "inhibited_by"])
def test_a_regulator_reading_a_form_the_row_solves_for_claims_no_direction(kind):
    """review of #127, blocking 1: `ERK{pT202}` matches the complex the catalytic step makes, which
    was held as a regulator's amount, so J was read off a constant: the false `activated by` was
    kept, the true `inhibited by` reversed. No J reads it, and the record says why."""
    regulator = f"{kind.replace('_', ' ')} ERK{{pT202}}"
    (emission,) = _build([_phospho(regulator)],
                         _scheme("Phosphorylation", _THREE_STEPS.format(kind=kind),
                                 "kf", "kr", "kc", "kp", "Ki"))
    assert emission.derived == {f"{kind}_0": "modulated_by"}
    assert ("reads ERK(T202='p', mek=1) % MEK(erk=1), a form this row makes and uses on its way"
            in emission.rate["none"])
    assert "(`~MEK`, not bound to MEK)" in emission.rate["none"]


def test_a_regulator_reading_every_form_of_the_enzyme_reads_its_total():
    """review of #127, blocking 1: `MEK` on MEK's own two steps matches it free and bound, which
    the steady state conserves: it reads their total. Free MEK alone is solved for: no J reads
    it."""
    def two_steps(regulator):
        kind = regulator.split(" ")[0] + "_by"
        steps = _TWO_STEPS.format(kf="kf", kr="kr", kc=f"kc/(1 + amount({kind})/Ki)")
        (emission,) = _build([_phospho(regulator)],
                             _scheme("Phosphorylation", steps, "kf", "kr", "kc", "Ki"))
        return emission

    assert two_steps("activated by MEK").derived == {}
    assert two_steps("inhibited by MEK").derived == {"inhibited_by_0": "activated_by"}
    free = two_steps("inhibited by MEK{~ERK}")
    assert free.derived == {"inhibited_by_0": "modulated_by"}
    assert "reads MEK(erk=None), a form this row makes" in free.rate["none"]


def _w(stmt_type: str, policy: str) -> dict:
    return {"statement_type": stmt_type, "policy": policy, "rate": "k/(1 + inhibited_by/K)",
            "parameters": ["k", "K"]}


@pytest.mark.parametrize(
    "rows, definition",
    [
        ([("m1: MEK phosphorylates ERK at Y187", "indra_default"),
          ("p: MEK phosphorylates ERK at T185, inhibited by U", "w")], _w("Phosphorylation", "w")),
        ([("m1: MEK phosphorylates ERK at Y187", "indra_default"),
          ("p: MEK phosphorylates ERK at T185, inhibited by U", "written")],
         _scheme("Phosphorylation", _TWO_STEPS.format(kf="kf", kr="kr", kc="kc" + _SLOWED),
                 "kf", "kr", "kc", "Ki")),
        ([("b: ERK binds DUSP", "indra_default"),
          ("p: MEK phosphorylates ERK at T185, inhibited by U", "w")], _w("Phosphorylation", "w")),
        ([("k: K phosphorylates A at S10", "indra_default"), ("c: A -> B, inhibited by U", "w")],
         _w("Conversion", "w")),
        ([("k: K phosphorylates A at S10", "indra_default"), ("d: A is degraded, inhibited by U", "w")],
         _w("DecreaseAmount", "w")),
        ([("k: K phosphorylates A at S10", "indra_default"),
          ("k2: K phosphorylates A at Y20", "indra_default"),
          ("d: P dephosphorylates A at S10, inhibited by U", "w")], _w("Dephosphorylation", "w")),
    ],
)
def test_an_agent_in_states_other_rows_give_is_one_lump(rows, definition):
    """review of #127, blocking 2: each species of a product (ERK at T185 with Y187 either way,
    bound to DUSP or not) read as a reaction of its own, so every signed regulator of an agent
    another row gives a site or a partner had no one net rate, so claimed no direction."""
    assert _build(rows, definition)[-1].derived == {}
    flipped = _build([(row.replace("inhibited by", "activated by"), b) for row, b in rows], {
        key: value.replace("inhibited_by", "activated_by") if key in ("rate", "scheme") else value
        for key, value in definition.items()
    })
    assert flipped[-1].derived == {"activated_by_0": "inhibited_by"}


def test_a_long_chain_of_enzyme_forms_is_signed_by_the_chain_rule():
    """review of #127: an inhibitor of the catalytic step, past twelve reversible forms of the
    enzyme, was not shown. J over its steps' coefficients is of degree one in each, as in every
    network of single steps, and is signed so."""
    n = 13
    last = f"c{n - 1}"
    lines = [f"enz(c='c{i}', sub=None) | enz(c='c{i + 1}', sub=None) @ ka{i}, kb{i}"
             for i in range(n - 1)]
    lines.append(_TWO_STEPS.format(kf="kf", kr="kr", kc="kc" + _SLOWED)
                 .replace("enz(sub=", f"enz(c='{last}', sub="))
    names = [f"k{x}{i}" for x in "ab" for i in range(n - 1)]
    scheme = _scheme("Phosphorylation", "\n".join(lines), *names, "kf", "kr", "kc", "Ki")
    (emission,) = _build([_phospho("inhibited by I")], scheme)
    assert emission.derived == {}


_SITES = ["Y1068", "Y1173", "Y992", "Y1045"]


def test_forms_apart_only_in_sites_the_row_never_reads_are_one():
    """review of #128 (second pass), t_sites: EGFR with four sites other rows set made 32 enzyme
    forms of one two-step row, past the bound, so `inhibited by I` had no J; its flux falls with I.
    Forms that differ only in sites the row's rules, rates and statement never read are lumped:
    two forms, as written."""
    rows = [(f"k{i}: SRC phosphorylates EGFR at {site}", "indra_default")
            for i, site in enumerate(_SITES)]
    scheme = _scheme("Phosphorylation", _TWO_STEPS.format(kf="kf", kr="kr", kc="kc" + _SLOWED),
                     "kf", "kr", "kc", "Ki")
    row = "p: EGFR phosphorylates PLCG at Y783, inhibited by I"
    assert _build([*rows, (row, "written")], scheme)[-1].derived == {}
    flipped = _build([*rows, (row.replace("inhibited", "activated"), "written")],
                     {**scheme, "scheme": scheme["scheme"].replace("inhibited_by", "activated_by")})
    assert flipped[-1].derived == {"activated_by_0": "inhibited_by"}
    # A row that reads a site keeps it apart: EGFR phosphorylated at Y1068 is a regulator's.
    read = {**scheme, "scheme": scheme["scheme"].replace("enz(sub=", "enz(Y1068='u', sub=")}
    assert _build([*rows, (row, "written")], read)[-1].derived == {}


def test_a_tiny_net_rate_is_no_zero():
    """review of #128 (second pass), t_vanish: eight enzyme forms, four inhibitors on every step,
    put J near 1e-14 at the zero test's points, below its absolute 1e-9, so the row "made nothing
    from its reactants alone". Zero is now relative to the terms J sums."""
    slowed = "".join(f"/(1 + amount(inhibited_by_{j})/Ki{j})" for j in range(4))
    sped = "".join(f"*(1 + amount(inhibited_by_{j})/Ki{j})" for j in range(4))
    lines = [f"enz(c='c{i}', sub=None) | enz(c='c{i + 1}', sub=None) @ ka{i}{slowed}, kb{i}{sped}"
             for i in range(7)]
    lines.append(_TWO_STEPS.format(kf="kf", kr="kr", kc="kc" + slowed)
                 .replace("enz(sub=", "enz(c='c7', sub="))
    names = [f"k{x}{i}" for x in "ab" for i in range(7)]
    _build([_phospho("inhibited by I0 and I1 and I2 and I3")],
           _scheme("Phosphorylation", "\n".join(lines), *names, "kf", "kr", "kc",
                   *(f"Ki{j}" for j in range(4))))
    a, b = sympy.symbols("a b", positive=True)
    assert not M._vanishes(1e-20 * a * b / (1 + a))
    assert M._vanishes(sympy.exp(sympy.log(a + 1)) - a - 1 + 0 * b)


@pytest.mark.parametrize("times, refused", [((0.0, 10.0), False), ((-5.0, 10.0), True)])
def test_time_ranges_over_the_problems_experiments(times, refused):
    """review of #128 (second pass), e2e_time: `time` was never negative, though an experiment may
    start at -5. `k*C*time/(tau + time)` moves with C as `time` does: activated from 0 on, not
    shown where time can be negative."""
    rate = _rate("k*activated_by*time/(tau + time)", "k", "tau", stmt_type="PopulationTransition",
                 based_on="mass_action")
    # As deep in a long run: sympy's cache cleared, a copied model's `time` was a new symbol that
    # BioNetGen's export refused (CI on 28d1d26).
    sympy.core.cache.clear_cache()
    (emission,) = _build([("conv: A becomes B, activated by C", "written")], rate, times=times)
    assert emission.derived == ({"activated_by_0": "modulated_by"} if refused else {})


def test_a_binding_keeps_its_comparison():
    """review of #128 (second pass): `g="piecewise(0, ko > 0, 1)"` raised a TypeError out of
    `bind`, a parameter being positive to sympy from its seeded value, which also decided the
    comparison where it did not raise. The comparison stays as written."""
    ko = pysb.Parameter("ko", 1.0, _export=False)
    kept = M.over(sympy.sympify("piecewise(2, ko > 0, 1)", locals={
        "ko": sympy.Symbol("ko"), "piecewise": M._RATE_FUNCTIONS["piecewise"]}), {"ko": ko})
    assert kept.atoms(sympy.core.relational.Relational) and ko in kept.free_symbols
    _build([("conv: A becomes B, activated by C", 'written g="piecewise(2, ko > 0, 1)"')],
           _rate("k*(1 + g*activated_by)", "k", "g", stmt_type="PopulationTransition",
                 based_on="mass_action"))


# ------------------------------------------------------------------------------- products held at 0
_RAS = 'c1: RAS{-mod"GTP"} -> RAS{mod"GTP"}'
_RAS_FED = (f'{_RAS}, activated by SOS and RAS{{mod"GTP"}}', "fed")
_GAP = ('c2: RAS{mod"GTP"} -> RAS{-mod"GTP"}', "mass_action")
_FEEDBACK = "obj_from_0() >> obj_to_0() @ kb + kf*amount(activated_by_0)*amount(activated_by_1)"
_SELF = "obj_from_0() >> obj_to_0() @ ka*amount(activated_by_0)*amount(activated_by_1)"
_GENE = "subj() >> subj() + obj() @ ks + ka*amount(activated_by_0)*amount(activated_by_1)"


def _fed(stmt_type: str, scheme: str, *parameters: str) -> dict:
    return _scheme(stmt_type, scheme, *parameters, policy="fed")


def _flux(rows: list, *definitions: dict, same: dict | None = None) -> dict[str, float]:
    """Each species' rate of change in the model ``rows`` build, its regulators unchecked, at one
    state: every species and parameter at a value of its name (``same`` naming, for a parameter,
    the one whose value it takes)."""
    import zlib

    seen = []
    real = regulation.record
    regulation.record = lambda model, rows_, emissions, *_: seen.append(model) or emissions
    try:
        _build(rows, *definitions)
    finally:
        regulation.record = real
    (model,) = seen

    def value(name: str) -> float:
        return 0.3 + zlib.crc32(name.encode()) % 97 / 40

    point = {sympy.Symbol(f"__s{i}"): value(str(sp)) for i, sp in enumerate(model.species)}
    point |= {p: value((same or {}).get(p.name, p.name)) for p in model.parameters}
    out = {}
    for i, ode in enumerate(model.odes):
        while found := ode.atoms(pysb.Expression) | ode.atoms(pysb.Observable):
            ode = ode.xreplace({x: x.expand_expr() if isinstance(x, pysb.Expression)
                                else x.expand_obs() for x in found})
        out[str(model.species[i])] = float(ode.xreplace(point))
    return out


_SOS_FED = (
    "gef() + ras(gtpbound='inactive') >> gef() + ras(gtpbound='active') @ "
    "kb + k*amount(activated_by_0)*amount(activated_by_1)"
)
_TWO_PRODUCTS = "obj_from_0() >> obj_to_0() + obj_to_1() @ "


@pytest.mark.parametrize(
    "rows, mechanism, stated, back, followed",
    [
        # review of #127/#128, refusals/t1: Ras-SOS feedback, SOS working through RAS-GTP, the
        # GAP step on the row.
        ([(f"{_RAS}, activated by SOS", "written")],
         _scheme("Conversion", "obj_from_0() >> obj_to_0() @ kb + kf*amount(activated_by)*"
                 "amount(obj_to_0())\nobj_to_0() >> obj_from_0() @ kgap", "kb", "kf", "kgap"),
         'RAS{-mod"GTP"} -> RAS{mod"GTP"}, activated by SOS and RAS{mod"GTP"}',
         'RAS{mod"GTP"} -> RAS{-mod"GTP"}',
         [([_RAS_FED], _fed("Conversion", f"{_FEEDBACK}\nobj_to_0() >> obj_from_0() @ kgap",
                            "kb", "kf", "kgap"), {}),
          ([_RAS_FED, _GAP], _fed("Conversion", _FEEDBACK, "kb", "kf"), {"c2_k": "c1_kgap"})]),
        # The same step written alone: it has no way back to put on a row.
        ([(f"{_RAS}, activated by SOS", "written")],
         _scheme("Conversion", "obj_from_0() >> obj_to_0() @ kb + kf*amount(activated_by)*"
                 "amount(obj_to_0())", "kb", "kf"),
         'RAS{-mod"GTP"} -> RAS{mod"GTP"}, activated by SOS and RAS{mod"GTP"}',
         'RAS{mod"GTP"} -> RAS{-mod"GTP"}',
         [([_RAS_FED], _fed("Conversion", _FEEDBACK, "kb", "kf"), {})] * 2),
        # refusals/t1/autoreg: a co-activator working only with the gene product present.
        ([("g: TF increases G, activated by X", "written")],
         _scheme("IncreaseAmount", "subj() >> subj() + obj() @ ks + ka*amount(activated_by)*"
                 "amount(obj())\nobj() >> None @ kd delete_molecules", "ks", "ka", "kd"),
         "TF increases G, activated by X and G", "G is degraded",
         [([("g: TF increases G, activated by X and G", "fed")],
           _fed("IncreaseAmount", f"{_GENE}\nobj() >> None @ kd delete_molecules", "ks", "ka",
                "kd"), {}),
          ([("g: TF increases G, activated by X and G", "fed"), ("d: G is degraded", "indra_default")],
           _fed("IncreaseAmount", _GENE, "ks", "ka"), {"d_kf": "g_kd"})]),
        # netrate/t6: the product makes itself, beside a basal step (a term of the one rule's
        # rate, as the refusal says: two rows of one reaction would be one reaction) or alone.
        ([("c: A -> B, activated by X", "written")],
         _scheme("Conversion", "obj_from_0() >> obj_to_0() @ kb\nobj_from_0() + obj_to_0() >> "
                 "obj_to_0() + obj_to_0() @ ka*amount(activated_by)", "ka", "kb"),
         "A -> B, activated by X and B", "B -> A",
         [([("c: A -> B, activated by X and B", "fed")],
           _fed("Conversion", f"obj_from_0() >> obj_to_0() @ kb\n{_SELF}", "ka", "kb"), {}),
          ([("c: A -> B, activated by X and B", "fed")],
           _fed("Conversion", "obj_from_0() >> obj_to_0() @ kb + ka*amount(activated_by_0)*"
                "amount(activated_by_1)", "ka", "kb"), {})]),
        ([("c: A -> B, activated by X", "written")],
         _scheme("Conversion", "obj_from_0() + obj_to_0() >> obj_to_0() + obj_to_0() @ "
                 "ka*amount(activated_by)", "ka"),
         "A -> B, activated by X and B", "B -> A",
         [([("c: A -> B, activated by X and B", "fed")], _fed("Conversion", _SELF, "ka"), {})] * 2),
        # review of #128 (second pass), t_gef: SOS's feedback through KRAS-GTP crashed writing
        # the advice, a Gef row the notation could not write.
        ([("gef: SOS is a GEF for KRAS, activated by X", "written")],
         _scheme("Gef", "gef() + ras(gtpbound='inactive') >> gef() + ras(gtpbound='active') @ "
                 "kb + k*amount(activated_by)*amount(ras(gtpbound='active'))", "kb", "k"),
         "SOS is a GEF for KRAS, activated by X and KRAS{gtpbound:on}", None,
         [([("gef: SOS is a GEF for KRAS, activated by X and KRAS{gtpbound:on}", "fed")],
           _fed("Gef", _SOS_FED, "kb", "k"), {})] * 2),
        # t_multiproduct: the feedback is through C, the second product; the advice named B.
        ([("c: A -> B + C, activated by X", "written")],
         _scheme("Conversion", f"{_TWO_PRODUCTS}kb\nobj_from_0() + obj_to_1() >> obj_to_0() + "
                 "obj_to_1() + obj_to_1() @ ka*amount(activated_by)", "ka", "kb"),
         "A -> B + C, activated by X and C", "B + C -> A",
         [([("c: A -> B + C, activated by X and C", "fed")],
           _fed("Conversion", f"{_TWO_PRODUCTS}kb\n{_TWO_PRODUCTS}ka*amount(activated_by_0)*"
                "amount(activated_by_1)", "ka", "kb"), {}),
          ([("c: A -> B + C, activated by X and C", "fed")],
           _fed("Conversion", f"{_TWO_PRODUCTS}kb + ka*amount(activated_by_0)*"
                "amount(activated_by_1)", "ka", "kb"), {})]),
    ],
    ids=["ras_sos", "ras_sos_one_rule", "autoregulation", "autocatalysis_basal", "autocatalysis",
         "gef", "second_product"],
)
def test_a_regulator_that_needs_the_product_is_told_both_rows_and_each_builds(
    rows, mechanism, stated, back, followed
):
    """Products stay at 0 in J (the user's decision, 2026-10-24), so a regulator that acts only
    with the product present (feedback, autoregulation, autocatalysis) does not move it. The
    refusal names the two rows that state it, written for the row, and each builds as written
    and keeps the model: every species changes as it did. The product named is the one the step
    reads. The step alone without the product stated still reads it at 0."""
    with pytest.raises(MechanismError) as refused:
        _build(rows, mechanism)
    said = str(refused.value)
    assert f"State the product as a regulator too, `{stated}`" in said
    assert "Or put that step on a row of its own so, bound to a one-rule mechanism" in said
    assert back is None or f"`{back}`" in said
    before = _flux(rows, mechanism)
    for rewrite, definition, same in followed:
        # Each regulator recorded as stated, the product's own an activator, and the model kept.
        assert all(not emission.derived for emission in _build(rewrite, definition))
        assert _flux(rewrite, definition, same=same) == pytest.approx(before, rel=1e-12)


# ------------------------------------------------------------------------------- second review
def _signed(rows, definition, problem) -> dict:
    record = _record(rows, definition)
    signed, failed, _ = regulation.resign(record, problem)
    assert failed == []
    return signed["rows"][-1]["emission"]


def test_a_condition_input_is_signed_on_the_models_own_value_too():
    """review of #128 (second pass), e2e_dose: `dose` set to 2 by the only condition was signed
    over {2}, so `k*(1 + (dose - 1)*C)` read `activated by C`; the model's own 0.1 holds wherever
    no period sets it (a pre-equilibration with no condition, a measurement in no experiment),
    and there J falls with C."""
    dosed = _rate("k*(1 + d*activated_by)", "k", "d", stmt_type="PopulationTransition",
                  based_on="mass_action") | {"policy": "dosed"}
    rows = [("conv: A becomes B, activated by C", 'dosed d="dose - 1"')]
    for default, derived in ((0.1, {"activated_by_0": "modulated_by"}), (3.0, {})):
        problem = _Problem([_parameter("conv_k", 1e-3, 1e3, scale="log10")],
                           [{"conditionId": "c1", "targetId": "dose", "targetValue": 2.0}],
                           [0.0, 10.0], values={"dose": default})
        assert _signed(rows, dosed, problem)["derived"] == derived


def test_time_is_signed_over_the_problems_experiments():
    """review of #128 (second pass), e2e_time: `time` ran from 0 to the last measurement, though
    an experiment starting at -5 reads `k*C*time/(tau + time)` at negative times."""
    onset = _rate("k*activated_by*time/(tau + time)", "k", "tau", stmt_type="PopulationTransition",
                  based_on="mass_action") | {"policy": "onset"}
    rows = [("conv: A becomes B, activated by C", "onset")]
    table = [_parameter("conv_k", 1e-3, 1e3, scale="log10"), _parameter("conv_tau", 1.0, 10.0)]
    assert _signed(rows, onset, _Problem(table, times=[0.0, 10.0]))["derived"] == {}
    early = _Problem(table, times=[-4.0, 10.0])
    early.experiment_df = pd.DataFrame({"experimentId": ["e1"], "time": [-5.0],
                                        "conditionId": [None]})
    early.measurement_df["experimentId"] = "e1"
    assert _signed(rows, onset, early)["derived"] == {"activated_by_0": "modulated_by"}


@pytest.mark.parametrize("rate, binding, table", [
    ("k*(1 - e)*activated_by", "", [_parameter("conv_e", 0.0, 1.0)]),
    ("k*(1 + a*activated_by)", ' a="x - y"',
     [_parameter("x", 2.0, 10.0), _parameter("y", 0.0, 1.0)]),
])
def test_a_bounded_parameter_keeps_both_ends(rate, binding, table):
    """review of #128 (second pass), b1_fraction: a bounded parameter was its lower bound plus a
    name never negative, with no upper end, so `1 - e` over `e` in [0, 1] and `x - y` over x in
    [2, 10], y in [0, 1] were of either sign."""
    names = sorted({"k", "e", "a"} & set(M._RATE_TOKENS.findall(rate)))
    definition = _rate(rate, *names, stmt_type="PopulationTransition",
                       based_on="mass_action") | {"policy": "lin"}
    problem = _Problem([_parameter("conv_k", 1e-3, 1e3, scale="log10"), *table])
    assert _signed([("conv: A becomes B, activated by C", f"lin{binding}")], definition, problem)[
        "derived"] == {}


_TURNOVER = ("subj() >> subj() + obj() @ ks*(1 + amount({kind})/K)\n"
             "obj() >> None @ kd*(1 + amount({kind})/K) delete_molecules")


@pytest.mark.parametrize("kind", ["modulated_by", "activated_by"])
def test_a_regulator_also_taking_the_product_away_is_not_signed_up(kind):
    """review of #128 (second pass), sweep case33/gexpr5: M speeds both the synthesis and the
    product's degradation; J, its product absent, rises with M, but the product's steady state
    falls. A `modulated by M` is not signed up; one stated with a direction keeps J's, and both
    record the step."""
    scheme = _scheme("IncreaseAmount", _TURNOVER.format(kind=kind), "ks", "kd", "K")
    word = kind.replace("_", " ")
    (emission,) = _build([(f"g: TF increases G, {word} M", "written")], scheme)
    assert emission.derived == {}
    assert emission.rate["consumes"] == {f"{kind}_0": ["`G >> None`"]}
    # Synthesis alone reads it: signed up.
    alone = _scheme("IncreaseAmount", _TURNOVER.format(kind=kind).replace(
        f"kd*(1 + amount({kind})/K)", "kd"), "ks", "kd", "K")
    (emission,) = _build([(f"g: TF increases G, {word} M", "written")], alone)
    upgraded = {"modulated_by_0": "activated_by"} if kind == "modulated_by" else {}
    assert emission.derived == upgraded
    assert "consumes" not in emission.rate


@pytest.mark.parametrize("rate", ["k*(modulated_by - 1)", "k/(modulated_by - 1)"])
def test_a_rate_that_can_go_negative_or_pass_a_pole_is_not_signed(rate):
    """review of #128 (second pass), sweep multi108: J rose with M2 on each side of a pole, and was
    signed so, though J(M2=0.5) = -2 and J(1.5) = +2. A J not shown never negative and finite
    claims no direction."""
    definition = _rate(rate, "k", stmt_type="PopulationTransition", based_on="mass_action")
    (emission,) = _build([("conv: A becomes B, modulated by M", "written")], definition)
    assert emission.derived == {}
    assert emission.rate["ill_posed"]


def test_the_warning_names_a_negative_rate_law_and_a_step_taking_the_product_away():
    """review of #128 (second pass): each of the two goes to the agent, the step by name."""
    from petab_mcp.operations import regulator_warning

    record = _record([("conv: A becomes B, activated by M", "neg")],
                     _rate("k*(activated_by - 1)", "k", stmt_type="PopulationTransition",
                           based_on="mass_action") | {"policy": "neg"})
    said = regulator_warning(record, None)
    assert said.startswith("`conv`: its rate law can go negative, or pass a pole, for values of "
                           "its names within their ranges")
    assert "(`activated by M` reads as `modulated by M`)" in said
    assert "`k/(1 + X/K)`, not `k*(1 - X)`" in said
    record = _record([("g: TF increases G, modulated by M", "written")],
                     _scheme("IncreaseAmount", _TURNOVER.format(kind="modulated_by"),
                             "ks", "kd", "K"))
    assert regulator_warning(record, None) == (
        "`g`: `modulated by M` also acts on `G >> None`, a step that takes the product away, which "
        "the reaction's rate (the product absent) does not see, so what it does to the product's "
        "amount is not shown: it stays `modulated by`."
    )
    assert regulator_warning(record, record) is None  # said once, for the row new or changed
