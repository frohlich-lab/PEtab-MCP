"""The SBML a set of statements builds for one PEtab problem: readout names, starting amounts,
inputs."""

from __future__ import annotations

import libsbml
import pytest
from indra.statements import Agent, BoundCondition, ModCondition, Phosphorylation, Translocation
from petab.v1.models.sbml_model import SbmlModel

from petab_mcp import mechanisms as M
from petab_mcp import problem_sbml as PS
from petab_mcp.errors import ModelUpdateError
from petab_mcp.mechanisms import INDRA_DEFAULT
from petab_mcp.protocols import Input
from petab_mcp.statements import PopulationTransition

_PHOSPHO = Phosphorylation(Agent("MEK", db_refs={"HGNC": "6840"}), Agent("ERK"), "T", "185")
_TRANSITION = PopulationTransition(Agent("naive"), Agent("effector"))


def test_readouts_name_the_species_the_observables_read():
    """The fixed observable table reads species ids, so the export must define them. A readout
    that is one species names that species, modified forms included: left as a rule over it,
    every species looked disconnected from the data to the edit's reachability warning."""
    p_erk = Agent._from_json(
        {
            "name": "ERK",
            "mods": [{"mod_type": "phosphorylation", "residue": "T", "position": "185"}],
        }
    )
    antimony, _ = PS.to_antimony(
        [(_PHOSPHO, INDRA_DEFAULT, "r1"), (_TRANSITION, "mass_action", "r2")],
        {"naive": Agent("naive"), "pERK": p_erk},
    )
    sbml = SbmlModel.from_antimony(antimony).sbml_model
    assert sbml.getSpecies("naive").getName() == "naive()"
    assert sbml.getSpecies("pERK").getName() == "ERK(T185='p')"
    assert not any(sbml.getAssignmentRuleByVariable(s) for s in ("naive", "pERK"))


def test_an_input_is_held_to_its_protocol():
    """A dose is a fact about the experiment: its species sits under the protocol inputs.tsv
    declares, with no starting amount of its own, and every condition target the protocol names
    exists for the table to set — a warm start was refused for a condition column it had nowhere
    to put. The entity map need not name it: an empty one is enough."""
    ramp = "piecewise(EGF_0 + EGF_rate*time, time < EGF_end, 0)"
    antimony, _ = PS.to_antimony(
        [(Phosphorylation(Agent("EGF"), Agent("AKT")), INDRA_DEFAULT, "r1")],
        {},
        inputs=(Input("EGF", ramp, ("EGF_0", "EGF_rate", "EGF_end")),),
    )
    sbml = SbmlModel.from_antimony(antimony).sbml_model
    assert sbml.getSpecies("EGF").getBoundaryCondition()
    assert sbml.getAssignmentRuleByVariable("EGF") is not None
    assert sbml.getInitialAssignmentBySymbol("EGF") is None
    assert {"EGF_0", "EGF_rate", "EGF_end"} <= {p.getId() for p in sbml.getListOfParameters()}


def test_a_species_the_conditions_set_is_held_without_a_rule():
    """Conditions that set the species `Fsk` itself say it is an input: no file declares it a
    second time, and a rule `Fsk := Fsk` would read itself."""
    antimony, _ = PS.to_antimony(
        [(Phosphorylation(Agent("forskolin"), Agent("AKT")), INDRA_DEFAULT, "r1")],
        {"Fsk": Agent("forskolin")},
        targets=frozenset({"Fsk"}),
    )
    sbml = SbmlModel.from_antimony(antimony).sbml_model
    assert sbml.getSpecies("Fsk").getBoundaryCondition()
    assert sbml.getAssignmentRuleByVariable("Fsk") is None
    assert sbml.getInitialAssignmentBySymbol("Fsk") is None


def test_chained_dimers_are_refused_before_the_expansion():
    """Two kinds each dimerising with itself and with the other chain without end;
    refused only once BioNetGen had built every complex up to the cap, the refusal took 19 s on
    an idle machine and 251 s on a loaded one. The dimers its advice writes still build."""
    import time

    from indra.statements import Complex

    a, b = "STAT5A", "STAT5B"
    entities = {a: Agent(a), b: Agent(b)}
    start = time.perf_counter()
    with pytest.raises(ModelUpdateError, match=r"rows \['aa', 'ab', 'bb'\] bind .* into chains"):
        PS.to_antimony(
            [
                (Complex([Agent(a), Agent(a)]), INDRA_DEFAULT, "aa"),
                (Complex([Agent(a), Agent(b)]), INDRA_DEFAULT, "ab"),
                (Complex([Agent(b), Agent(b)]), INDRA_DEFAULT, "bb"),
            ],
            entities,
        )
    assert time.perf_counter() - start < 10

    def free(name, of):
        return Agent(name, bound_conditions=[BoundCondition(Agent(of), False)])

    rows = [
        (Complex([free(a, b), free(a, b)]), INDRA_DEFAULT, "aa"),
        (Complex([free(a, a), free(b, b)]), INDRA_DEFAULT, "ab"),
        (Complex([free(b, a), free(b, a)]), INDRA_DEFAULT, "bb"),
    ]
    assert "STAT5A" in PS.to_antimony(rows, entities)[0]


def test_what_is_made_starts_empty_and_what_is_modified_starts_present():
    """Effectors exist only once naive cells convert; a kinase that is only ever modified must be
    there at the start. The modified one's amount is `<monomer>_init`, not INDRA's
    `<monomer>_0`, which skipped EGF outright where the stimulus column is `EGF_0`."""
    antimony, _ = PS.to_antimony(
        [(_TRANSITION, "mass_action", "r1"), (_PHOSPHO, INDRA_DEFAULT, "r2")],
        {"effector": Agent("effector")},
    )
    sbml = SbmlModel.from_antimony(antimony).sbml_model
    assert sbml.getInitialAssignmentBySymbol("effector") is None
    present = {ia.getMath().getName() for ia in sbml.getListOfInitialAssignments()}
    assert present == {"naive_init", "MEK_init", "ERK_init"}


def test_a_species_two_readouts_read_is_named_once():
    """Renamed once per readout, the species took the second name and left the first defined by
    nothing, so what read the first read a parameter fitted in place of the species."""
    antimony, _ = PS.to_antimony(
        [(_TRANSITION, "mass_action", "r1")],
        {"naive": Agent("naive"), "naive_total": Agent("naive")},
    )
    sbml = SbmlModel.from_antimony(antimony).sbml_model
    assert sbml.getSpecies("naive") is not None
    assert sbml.getAssignmentRuleByVariable("naive_total").getMath().getName() == "naive"


def test_a_blinded_readout_stays_a_rule():
    """A `readout_*` symbol is a measurement mapping, which the lint accepts only as an assignment
    rule; renamed to the species it reads, the whole statement model was rejected."""
    antimony, _ = PS.to_antimony(
        [(_TRANSITION, "mass_action", "r1")], {"readout_naive": Agent("naive")}
    )
    sbml = SbmlModel.from_antimony(antimony).sbml_model
    assert sbml.getAssignmentRuleByVariable("readout_naive") is not None


def test_a_rate_may_read_time():
    """PySB lists `time` in a rate as a modifier species, which no species is, so every model
    whose written rate read `time` was unreadable SBML."""
    library = dict(M.REGISTRY)
    library[("PopulationTransition", "waning")] = M.written(
        {
            "statement_type": "PopulationTransition",
            "policy": "waning",
            "based_on": "mass_action",
            "rate": "k*exp(-time)",
            "parameters": ["k"],
        },
        library,
    )
    antimony, _ = PS.to_antimony([(_TRANSITION, "waning", "r1")], {}, library)
    assert "r1_rate := r1_k/exp(time);" in antimony
    sbml = SbmlModel.from_antimony(antimony).sbml_model
    assert [r.getNumModifiers() for r in sbml.getListOfReactions()] == [0]


def test_a_rate_may_switch_with_piecewise():
    """A rate that steps on an estimated day (Oliveira's transmission) or holds a term only while
    a dose is applied (Alkan's SN-38) had no written form, and was held as an input instead, a
    modelling claim the problem does not make (PEtabAgentBench#6). Both reach the SBML as one
    piecewise, through BioNetGen's `if`."""
    library = dict(M.REGISTRY)
    library[("PopulationTransition", "stepped")] = M.written(
        {
            "statement_type": "PopulationTransition",
            "policy": "stepped",
            "based_on": "mass_action",
            "rate": "piecewise(k_0, time < t_1, k_1)",
            "parameters": ["k_0", "k_1", "t_1"],
        },
        library,
    )
    library[("PopulationTransition", "while_dosed")] = M.written(
        {
            "statement_type": "PopulationTransition",
            "policy": "while_dosed",
            "based_on": "mass_action",
            "rate": "k*piecewise(1, drug > 0, 0)",
            "parameters": ["k"],
            "modifiers": ["drug"],
        },
        library,
    )
    rows = [
        (_TRANSITION, "stepped", "r1"),
        (PopulationTransition(Agent("A"), Agent("B")), "while_dosed", "r2", None, {"drug": Agent("D")}),
    ]
    antimony, _ = PS.to_antimony(rows, {}, library)
    sbml = SbmlModel.from_antimony(antimony).sbml_model
    for rate in ("r1_rate", "r2_rate"):
        math = sbml.getAssignmentRuleByVariable(rate).getMath()
        assert "piecewise" in libsbml.formulaToL3String(math), rate


def test_a_piecewise_condition_must_compare():
    """A condition that is no comparison, or a comparison outside `piecewise`, is refused with how
    to write one, rather than read as sympy's truth value of a symbol."""
    for rate in ("k*piecewise(1, drug, 0)", "k*(time < t_1)"):
        with pytest.raises(M.MechanismError, match="compare only in a piecewise condition"):
            M.written(
                {
                    "statement_type": "PopulationTransition",
                    "policy": "bad",
                    "based_on": "mass_action",
                    "rate": rate,
                    "parameters": ["k", "t_1"] if "t_1" in rate else ["k"],
                    "modifiers": ["drug"] if "drug" in rate else [],
                },
                dict(M.REGISTRY),
            )


def test_reaction_ids_never_collide_with_the_models_own():
    """PySB numbers reactions `r0`, `r1`... whatever else bears the name: an agent's shared rate
    `r8` made the model unreadable SBML. Here the readout `r0` takes it."""
    antimony, _ = PS.to_antimony([(_TRANSITION, "mass_action", "conv")], {"r0": Agent("naive")})
    sbml = SbmlModel.from_antimony(antimony).sbml_model
    assert sbml.getSpecies("r0").getName() == "naive()"
    assert [r.getId() for r in sbml.getListOfReactions()] == ["r_0"]
    antimony, _ = PS.to_antimony([(_TRANSITION, "mass_action", "conv")], {})
    assert "r0: __s0 => __s1; conv_k*__s0;" in antimony


def test_a_start_location_does_not_depend_on_row_order():
    """Without a cytoplasm among its locations, a monomer started in the first location INDRA met,
    so swapping two rows moved where it started."""
    from indra.statements import Translocation

    x = Agent("X")
    rows = [
        (Translocation(x, "nucleus", "mitochondria"), INDRA_DEFAULT, "r1"),
        (Translocation(x, "mitochondria", "nucleus"), INDRA_DEFAULT, "r2"),
    ]

    def started(rows):
        sbml = SbmlModel.from_antimony(PS.to_antimony(rows, {})[0]).sbml_model
        return [sbml.getSpecies(a.getSymbol()).getName() for a in sbml.getListOfInitialAssignments()]

    assert started(rows) == started(rows[::-1]) == ["X(loc='mitochondria')"]


def test_what_the_start_cannot_reach_starts_present():
    """Populations that only interconvert are each made by a rule, but nothing at the start makes
    them; with every made monomer started empty, as a diff of each rule's names had it, the model
    stayed zero. A made monomer the start does reach still starts empty."""
    antimony, _ = PS.to_antimony(
        [
            (_TRANSITION, "mass_action", "r1"),
            (PopulationTransition(Agent("A"), Agent("B")), "mass_action", "r2"),
            (PopulationTransition(Agent("B"), Agent("A")), "mass_action", "r3"),
        ],
        {},
    )
    sbml = SbmlModel.from_antimony(antimony).sbml_model
    present = {ia.getMath().getName() for ia in sbml.getListOfInitialAssignments()}
    assert present == {"naive_init", "A_init", "B_init"}


def test_a_complex_nothing_forms_is_refused():
    """The start holds free monomers only, so a kinase active only while bound to a partner no
    statement binds it to never acts; built anyway, its rate constant was one no fit could move."""
    raf = Agent("RAF", bound_conditions=[BoundCondition(Agent("RAS"))])
    with pytest.raises(ModelUpdateError, match=r"rows \['r1'\].*readouts \['RAF_RAS'\]"):
        PS.to_antimony(
            [(Phosphorylation(raf, Agent("MEK")), INDRA_DEFAULT, "r1")], {"RAF_RAS": raf}
        )


def test_a_rule_that_never_fires_points_to_starts():
    """The refusal said everything starts free, which `starts` made untrue, and never named it as
    the way to have the missing form present at t=0."""
    raf = Agent("RAF", bound_conditions=[BoundCondition(Agent("RAS"))])
    with pytest.raises(ModelUpdateError, match="declare it in `starts`"):
        PS.to_antimony([(Phosphorylation(raf, Agent("MEK")), INDRA_DEFAULT, "r1")], {})


# -- declared starts (make_changes' `starts`) ----------------------------------------
def _ia(sbml, sid):
    ia = sbml.getInitialAssignmentBySymbol(sid)
    return None if ia is None else libsbml.formulaToL3String(ia.getMath())


def test_a_declared_unit_pool_replaces_the_estimated_seed():
    """A population that starts at 1 is a unit normalisation, which reconcile fixes; seeded at an
    estimated `<monomer>_init` instead, each cost one k for nothing."""
    antimony, _ = PS.to_antimony(
        [(_PHOSPHO, INDRA_DEFAULT, "r1")],
        {"MEK": Agent("MEK", db_refs={"HGNC": "6840"})},
        starts=[("pERK", Agent("ERK"), 1)],
    )
    sbml = SbmlModel.from_antimony(antimony).sbml_model
    params = {p.getId(): p.getValue() for p in sbml.getListOfParameters()}
    assert params["pERK_init"] == 1.0 and "ERK_init" not in params
    assert "MEK_init" in params  # an undeclared monomer is still seeded as before


def test_a_made_monomer_can_be_declared_present():
    """Effectors start empty because naive cells make them; a population present at the start
    despite being made must say so, and then it is present."""
    antimony, _ = PS.to_antimony(
        [(_TRANSITION, "mass_action", "r1")],
        {"effector": Agent("effector")},
        starts=[("effector", Agent("effector"), 5)],
    )
    sbml = SbmlModel.from_antimony(antimony).sbml_model
    assert _ia(sbml, "effector") == "effector_init"
    assert sbml.getParameter("effector_init").getValue() == 5.0


def test_a_declared_start_is_the_state_the_agent_writes():
    """A species may start modified, or in each of several states: the start is the Agent's own
    pattern, every unstated site in its default state."""
    p_erk = Agent._from_json(
        {"name": "ERK", "mods": [{"mod_type": "phosphorylation", "residue": "T", "position": "185"}]}
    )
    antimony, _ = PS.to_antimony(
        [(_PHOSPHO, INDRA_DEFAULT, "r1")],
        {"pERK": p_erk},
        starts=[("pERK", p_erk, 0.5), ("ERK", Agent("ERK"), 2)],
    )
    sbml = SbmlModel.from_antimony(antimony).sbml_model
    names = {sp.getId(): sp.getName() for sp in sbml.getListOfSpecies()}
    assert names["pERK"] == "ERK(T185='p')"
    assert _ia(sbml, "pERK") == "pERK_init"
    assert {_ia(sbml, sid) for sid in names} >= {"pERK_init", "ERK_init"}


def test_a_declared_complex_starts_bound():
    """A receptor-kinase complex that exists before any rule runs; the partner is built on the
    sites INDRA names after each other."""
    raf_ras = Agent("RAF", bound_conditions=[BoundCondition(Agent("RAS"))])
    antimony, _ = PS.to_antimony(
        [(Phosphorylation(raf_ras, Agent("MEK")), INDRA_DEFAULT, "r1")],
        {"RAF_RAS": raf_ras},
        starts=[("RAF_RAS", raf_ras, 3)],
    )
    sbml = SbmlModel.from_antimony(antimony).sbml_model
    assert sbml.getSpecies("RAF_RAS").getName().count("._br_") == 1
    assert _ia(sbml, "RAF_RAS") == "RAF_RAS_init"


def test_a_declared_expression_names_parameters_the_table_fills():
    """A start at a condition column or at a shipped total: the names become parameters for the
    tables to fill, the placeholder goes."""
    antimony, _ = PS.to_antimony(
        [(_TRANSITION, "mass_action", "r1")],
        {"naive": Agent("naive")},
        starts=[("naive", Agent("naive"), "pool*fold")],
    )
    sbml = SbmlModel.from_antimony(antimony).sbml_model
    assert _ia(sbml, "naive") == "pool * fold"
    ids = {p.getId() for p in sbml.getListOfParameters()}
    assert {"pool", "fold"} <= ids and "naive_init" not in ids


def test_two_starts_of_one_species_are_refused_naming_both():
    """An unmodified site written out is its default state, so `ERK{-pT185}` starts what `ERK`
    starts; PySB's "Duplicate species" named neither the other start nor the species."""
    u_erk = Agent("ERK", mods=[ModCondition("phosphorylation", "T", "185", False)])
    with pytest.raises(
        ModelUpdateError, match=r"starts 'S0' and 'S1' both start ERK\(T185='u'\)"
    ):
        PS.to_antimony(
            [(_PHOSPHO, INDRA_DEFAULT, "r1")], {},
            starts=[("S0", Agent("ERK"), 1), ("S1", u_erk, 2)],
        )


def test_an_input_declares_no_start():
    """An input is held to its protocol; a start for it would be overwritten at t=0."""
    egf = Agent("EGF")
    with pytest.raises(ModelUpdateError, match="is an input"):
        PS.to_antimony(
            [(Phosphorylation(egf, Agent("AKT")), INDRA_DEFAULT, "r1")],
            {"EGF": egf},
            inputs=(Input("EGF", "piecewise(EGF_0, time < 5, 0)", ("EGF_0",)),),
            starts=[("EGF", egf, 1)],
        )


def test_a_declared_empty_start_has_no_parameter():
    """A substrate every condition sets: an `_init` behind it, even a zero one, was an estimated
    row no condition left any effect to."""
    antimony, _ = PS.to_antimony(
        [(_TRANSITION, "mass_action", "r1")], {"naive": Agent("naive")},
        starts=[("naive", Agent("naive"), 0)],
    )
    sbml = SbmlModel.from_antimony(antimony).sbml_model
    assert _ia(sbml, "naive") is None and sbml.getParameter("naive_init") is None
    assert sbml.getSpecies("naive").getInitialAmount() == 0.0


def test_what_rules_make_from_nothing_is_noted_as_starting_empty():
    """A kind made by zero-order synthesis started at 0 with nothing said. A start declared at 0
    says it, and drops the note."""
    from indra.statements import IncreaseAmount

    rows = [(IncreaseAmount(None, Agent("EGFR")), INDRA_DEFAULT, "r1")]
    notes = []
    PS.to_antimony(rows, {}, notes=notes)
    assert len(notes) == 1 and "EGFR" in notes[0] and "`starts`" in notes[0]
    notes = []
    PS.to_antimony(rows, {}, starts=[("EGFR", Agent("EGFR"), 0)], notes=notes)
    assert notes == []


def test_a_start_location_chosen_among_several_is_noted():
    """With no cytoplasm among its locations, where a monomer started went unsaid; then each
    such monomer repeated the whole note."""
    from indra.statements import Translocation

    x, y = Agent("X"), Agent("Y")
    notes = []
    PS.to_antimony(
        [
            (Translocation(x, "nucleus", "mitochondria"), INDRA_DEFAULT, "r1"),
            (Translocation(x, "mitochondria", "nucleus"), INDRA_DEFAULT, "r2"),
            (Translocation(y, "mitochondria", "nucleus"), INDRA_DEFAULT, "r3"),
        ],
        {},
        notes=notes,
    )
    assert len(notes) == 1
    assert (
        ": X in mitochondria of ['mitochondria', 'nucleus']; "
        "Y in mitochondria of ['mitochondria', 'nucleus']."
    ) in notes[0]


def test_indra_padded_rules_build():
    """INDRA pads one-step Conversion and subject-driven amount rules with None for Kappa; the
    trace read every side as a complex and every such statement failed to build."""
    from indra.statements import Conversion, DecreaseAmount

    antimony, _ = PS.to_antimony(
        [
            (Conversion(None, [Agent("A")], [Agent("B")]), INDRA_DEFAULT, "r1"),
            (DecreaseAmount(Agent("C"), Agent("B")), INDRA_DEFAULT, "r2"),
        ],
        {"B": Agent("B"), "C": Agent("C")},
    )
    assert SbmlModel.from_antimony(antimony).sbml_model.getNumReactions() == 2


def test_complexes_that_grow_without_bound_are_refused():
    """Homo- and heterodimers of two kinds chain without end; BioNetGen expanded such a network
    until the client timed out."""
    from indra.statements import Complex

    rows = [
        (Complex([Agent("STAT5A"), Agent("STAT5A")]), "stated", "aa"),
        (Complex([Agent("STAT5A"), Agent("STAT5B")]), "stated", "ab"),
        (Complex([Agent("STAT5B"), Agent("STAT5B")]), "stated", "bb"),
    ]
    with pytest.raises(ModelUpdateError, match="grow without bound"):
        PS.to_antimony(rows, {"STAT5A": Agent("STAT5A"), "STAT5B": Agent("STAT5B")})


def test_trimers_assembled_from_free_partners_are_not_refused():
    """Lucarelli's three SMAD trimers, each assembled in one step from partners free of all
    others: their products close a cycle of bonds, but no rule binds a molecule already bound, so
    each complex is one trimer. Refused as chains that grow without bound (PEtabAgentBench#23).
    Linked by a dimer that binds whatever else its partners hold, they do chain, and still are."""
    from indra.statements import Complex

    def free(name, *of):
        return Agent(name, bound_conditions=[BoundCondition(Agent(o), False) for o in of])

    s2, s3, s4 = (
        lambda: free("SMAD2", "SMAD3", "SMAD4"),
        lambda: free("SMAD3", "SMAD2", "SMAD3", "SMAD4"),
        lambda: free("SMAD4", "SMAD2", "SMAD3", "SMAD4"),
    )
    rows = [
        (Complex([s2(), s3(), s3()]), "assembly", "k233"),
        (Complex([s2(), s3(), s4()]), "assembly", "k234"),
        (Complex([s2(), s4(), s4()]), "assembly", "k244"),
    ]
    antimony, _ = PS.to_antimony(rows, {"SMAD2": Agent("SMAD2")})
    assert SbmlModel.from_antimony(antimony).sbml_model.getNumReactions() == 3

    trimer = Complex([free("A", "B", "C"), free("B", "A", "C"), free("C", "A", "B")])
    linked = [(trimer, "assembly", "abc"), (Complex([Agent("B"), Agent("C")]), INDRA_DEFAULT, "bc")]
    with pytest.raises(ModelUpdateError, match=r"rows \['abc', 'bc'\] bind .* into chains"):
        PS.to_antimony(linked, {"A": Agent("A")})


def test_the_dimers_the_refusal_advises_are_not_refused():
    """The same three dimers, each partner free of its others where it binds: dimers only."""
    from indra.statements import Complex

    def free(name, of):
        return Agent(name, bound_conditions=[BoundCondition(Agent(of), False)])

    a, b = "STAT5A", "STAT5B"
    rows = [
        (Complex([free(a, b), free(a, b)]), "stated", "aa"),
        (Complex([free(a, a), free(b, b)]), "stated", "ab"),
        (Complex([free(b, a), free(b, a)]), "stated", "bb"),
    ]
    antimony, _ = PS.to_antimony(rows, {a: Agent(a), b: Agent(b)})
    assert "STAT5A" in antimony and "STAT5B" in antimony


# -- readout formulas (make_changes' `readouts` over `amounts`) ----------------------
def test_a_readout_formula_reads_other_entries_and_parameters():
    """A readout may be a ratio of two patterns scaled by a shipped constant; one pattern's
    amount cannot say it."""
    p_erk = Agent._from_json(
        {"name": "ERK", "mods": [{"mod_type": "phosphorylation", "residue": "T", "position": "185"}]}
    )
    antimony, _ = PS.to_antimony(
        [(_PHOSPHO, INDRA_DEFAULT, "r1")],
        {},
        formulas={"readout_ratio": "scale * pERK / tERK"},
        amounts={"pERK": p_erk, "tERK": Agent("ERK")},
    )
    sbml = SbmlModel.from_antimony(antimony).sbml_model
    rule = sbml.getAssignmentRuleByVariable("readout_ratio")
    assert libsbml.formulaToL3String(rule.getMath()) == "scale * pERK / tERK"
    assert sbml.getParameter("scale") is not None


def test_a_readout_formula_takes_pythons_power():
    """`(a*b*c)**3` was refused as no formula, libSBML's parser having only `^`; read as `^`, it
    binds and associates as Python's `**`."""
    p_erk = Agent._from_json(
        {"name": "ERK", "mods": [{"mod_type": "phosphorylation", "residue": "T", "position": "185"}]}
    )
    antimony, _ = PS.to_antimony(
        [(_PHOSPHO, INDRA_DEFAULT, "r1")],
        {},
        formulas={"readout_x": "-(pERK/tERK)**3 + tERK**2**-1"},
        amounts={"pERK": p_erk, "tERK": Agent("ERK")},
    )
    sbml = SbmlModel.from_antimony(antimony).sbml_model
    rule = sbml.getAssignmentRuleByVariable("readout_x")
    assert libsbml.formulaToL3String(rule.getMath()) == "-(pERK / tERK)^3 + tERK^(2^-1)"


def test_a_readout_formula_reads_an_agent_only_through_amounts():
    """Naming an agent in a formula without declaring it in `amounts` made a parameter of that
    name, so the readout measured a fitted constant instead of the species (`readout_x := X`,
    `X` estimated at its bound)."""
    with pytest.raises(ModelUpdateError, match="only through `amounts`"):
        PS.to_antimony(
            [(_PHOSPHO, INDRA_DEFAULT, "r1")],
            {},
            formulas={"readout_total": "scale * ERK"},
        )


def test_a_readout_formula_names_a_new_symbol():
    """A readout's id that is also a species or an entity-map id would define one symbol twice."""
    with pytest.raises(ModelUpdateError, match="also names something else"):
        PS.to_antimony(
            [(_TRANSITION, "mass_action", "r1")],
            {"naive": Agent("naive")},
            formulas={"naive": "2 * naive"},
        )


def test_a_readout_formula_never_reads_the_readout():
    """`readout_x := readout_x * X` made a parameter of the readout's own name before the rule's,
    and libSBML failed on the duplicate id ("id must be unique")."""
    with pytest.raises(ModelUpdateError, match="'readout_total' = .* reads itself"):
        PS.to_antimony(
            [(_PHOSPHO, INDRA_DEFAULT, "r1")],
            {},
            formulas={"readout_total": "readout_total * tERK"},
            amounts={"tERK": Agent("ERK")},
        )


# -- entries written from the statement's own agents ---------------------------------
def _reactions(rows, entities):
    antimony, _ = PS.to_antimony(rows, entities)
    sbml = SbmlModel.from_antimony(antimony).sbml_model
    names = {sp.getId(): sp.getName() for sp in sbml.getListOfSpecies()}
    return [
        (
            sorted(names[r.getSpecies()] for r in rx.getListOfReactants()),
            sorted(names[p.getSpecies()] for p in rx.getListOfProducts()),
        )
        for rx in sbml.getListOfReactions()
    ]


def test_a_conversion_makes_its_products_in_the_states_they_write():
    """Two reactants into one stated product, a phosphorylation-and-dimerisation step, which
    INDRA's Conversion (one reactant, base-state products) could not build."""
    from indra.statements import Conversion

    p_dimer = Agent._from_json(
        {"name": "D", "mods": [{"mod_type": "phosphorylation", "residue": "Y", "position": "1"}]}
    )
    (reaction,) = _reactions(
        [(Conversion(None, [Agent("A"), Agent("A")], [p_dimer]), "mass_action", "r1")],
        {"A": Agent("A"), "pD": p_dimer},
    )
    assert reaction == (["A()", "A()"], ["D(Y1='p')"])


def test_a_release_frees_the_partner_it_was_bound_to():
    """An internalised ligand is degraded and its receptor recycles; INDRA's degradation
    deletes the whole complex, receptor included."""
    from indra.statements import Complex, DecreaseAmount

    bound = Agent("L", bound_conditions=[BoundCondition(Agent("R"))])
    reactions = _reactions(
        [
            (Complex([Agent("L"), Agent("R")]), INDRA_DEFAULT, "bind"),
            (DecreaseAmount(None, bound), "release", "deg"),
        ],
        {"R": Agent("R"), "L": Agent("L")},
    )
    released = [r for r in reactions if r[1] == ["R(l=None)"] and len(r[0]) == 1]
    assert released and "._br_" in released[0][0][0]


def test_a_population_renews_itself():
    """A population expands from itself; INDRA reads "X increases X" as nothing."""
    from indra.statements import IncreaseAmount

    reactions = _reactions(
        [(IncreaseAmount(None, Agent("E")), "self_renewal", "r1")], {"E": Agent("E")}
    )
    assert (["E()"], ["E()", "E()"]) in reactions


def test_a_modification_goes_on_without_an_enzyme():
    """Basal acetylations and phosphorylations have no named enzyme; `basal` existed only for
    removal."""
    from indra.statements import Acetylation

    reactions = _reactions(
        [(Acetylation(None, Agent("H4"), "K", "5"), "basal", "r1")], {"H4": Agent("H4")}
    )
    assert reactions == [(["H4(K5='n')"], ["H4(K5='y')"])]  # INDRA's acetylation states


def _p(name, residue="S", position="1", modified=True, **kwargs):
    return Agent._from_json(
        {
            "name": name,
            "mods": [
                {
                    "mod_type": "phosphorylation",
                    "residue": residue,
                    "position": position,
                    "is_modified": modified,
                }
            ],
            **kwargs,
        }
    )


def test_a_stated_complex_comes_apart_only_as_stated():
    """An internalised complex must not dissociate at the surface rate: the reverse reaction
    keeps the member's stated phospho-state, where INDRA's dissociates any complex."""
    from indra.statements import Complex

    reactions = _reactions(
        [
            (Phosphorylation(None, Agent("A"), "S", "1"), "basal", "p"),
            (Complex([_p("A"), Agent("B")]), "stated", "bind"),
        ],
        {"A": Agent("A"), "B": Agent("B")},
    )
    bindings = [r for r in reactions if len(r[0]) == 2 or len(r[1]) == 2]
    assert len(bindings) == 1  # only phospho-A binds, and only that complex dissociates
    assert any("S1='p'" in s for s in bindings[0][0] + bindings[0][1])


def test_an_assembly_joins_repeated_members_in_one_step():
    """An A + 2 B trimer forms in one irreversible step; INDRA's multi_way has one site per
    partner type and cannot hold the second B."""
    from indra.statements import Complex

    (reaction,) = _reactions(
        [(Complex([Agent("A"), Agent("B"), Agent("B")]), "assembly", "tri")],
        {"A": Agent("A"), "B": Agent("B")},
    )
    reactants, (product,) = reaction
    assert sorted(m.split("(")[0] for m in reactants) == ["A", "B", "B"]
    assert product.count("._br_") == 2


def test_a_nondepleting_ligand_occupies_without_being_used_up():
    """A ligand in excess: occupying its receptor leaves the free pool as it was, and vacating
    deletes only the bound copy."""
    from indra.statements import Complex

    reactions = _reactions(
        [(Complex([Agent("R"), Agent("L")]), "nondepleting", "occ")],
        {"R": Agent("R"), "L": Agent("L")},
    )
    occupy = next(r for r in reactions if len(r[0]) == 2)
    assert "L(r=None)" in occupy[0] and "L(r=None)" in occupy[1]
    vacate = next(r for r in reactions if len(r[0]) == 1)
    assert vacate[1] == ["R(l=None)"]


def test_a_processive_step_sets_every_listed_site():
    """A substrate goes from unmodified to modified at both of its sites in one step."""
    sub = Agent._from_json(
        {
            "name": "S",
            "mods": [
                {"mod_type": "phosphorylation", "residue": "S", "position": "1", "is_modified": False},
                {"mod_type": "phosphorylation", "residue": "S", "position": "2", "is_modified": False},
            ],
        }
    )
    reactions = _reactions(
        [(Phosphorylation(Agent("K"), sub, "S", "1"), "processive", "pp")],
        {"S": Agent("S"), "K": Agent("K")},
    )
    (reaction,) = reactions
    assert any("S1='p'" in s and "S2='p'" in s for s in reaction[1])


def test_a_dissociative_removal_dissolves_the_complexes_it_was_in():
    """A phosphatase releases a complex's members as it dephosphorylates the one it binds."""
    from indra.statements import Complex, Dephosphorylation

    reactions = _reactions(
        [
            (Phosphorylation(None, Agent("A"), "S", "1"), "basal", "p"),
            (Complex([_p("A"), Agent("B")]), "assembly", "dimer"),
            (Dephosphorylation(None, Agent("A"), "S", "1"), "dissociative", "dp"),
        ],
        {"A": Agent("A"), "B": Agent("B")},
    )
    released = [r for r in reactions if len(r[0]) == 1 and len(r[1]) == 2]
    assert released and any("S1='u'" in s for s in released[0][1])


def test_a_shared_rate_named_by_a_start_is_not_a_unit_pool():
    """A receptor total that is both a rate constant several rows share and the receptor's start;
    minted at 1.0 by the binding, reconcile read it as a unit normalisation and fixed it."""
    antimony, _ = PS.to_antimony(
        [(_TRANSITION, "mass_action", "r1", {"k": "total"})],
        {"naive": Agent("naive")},
        starts=[("naive", Agent("naive"), "total")],
    )
    sbml = SbmlModel.from_antimony(antimony).sbml_model
    assert sbml.getParameter("total").getValue() != 1.0


def test_a_protein_is_made_in_the_state_it_is_written():
    """A protein made in the state (here a location) it is measured in; INDRA's IncreaseAmount
    makes the base state."""
    from indra.statements import IncreaseAmount

    nuclear = Agent("P", location="nucleus")
    reactions = _reactions(
        [
            (IncreaseAmount(None, nuclear), "mass_action", "syn"),
            (Translocation(Agent("P"), "nucleus", "cytoplasm"), INDRA_DEFAULT, "out"),
        ],
        {"P": Agent("P")},
    )
    assert ([], ["P(loc='nucleus')"]) in reactions


def test_a_written_rate_leaves_no_unread_rules():
    """The amounts a written rate reads are inlined into it; the observable rules PySB also
    exports for them were read by nothing."""
    library = dict(M.REGISTRY)
    library[("PopulationTransition", "sat")] = M.written(
        {
            "statement_type": "PopulationTransition",
            "policy": "sat",
            "based_on": "mass_action",
            "rate": "vmax / (K + source)",
            "parameters": ["vmax", "K"],
        },
        library,
    )
    antimony, _ = PS.to_antimony([(_TRANSITION, "sat", "r1")], {"naive": Agent("naive")}, library)
    assert "__obs" not in antimony
