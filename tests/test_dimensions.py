"""Dimension inference from a model's math (:mod:`petab_mcp.dimensions`), which keeps the merge
price from pairing a rate with a concentration.

The first three tests are the ties gemot#94 found taken in scored models: a channel-closing rate
tied to a Ca²⁺ midpoint (sneyd), a feedback threshold tied to a dephosphorylation rate (fujita),
and a drug EC50 tied to a carrying capacity (alkan). The rest pin what keeps the filter from
dropping ties that are fine.
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np

from petab_mcp import dimensions as D

_BASE_MODEL = "model m\n{extra}\nend\n"


def _dims(extra: str, inputs=(), entries=()):
    from petab.v2.models.sbml_model import SbmlModel

    sm = SbmlModel.from_antimony(_BASE_MODEL.format(extra=extra), model_id="m").sbml_model
    return D.infer(sm, inputs=inputs, condition_entries=entries)


def test_a_closing_rate_is_not_a_calcium_midpoint():
    """sneyd: `k_close == K_Cd` was in a scored model. One is a first-order rate, the other is added
    to the Ca²⁺ input in a Hill denominator, so it is a concentration of Ca²⁺."""
    dims = _dims(
        "species O = 1; species C = 0; Ca = 0.4; k_close = 1; k_open = 1; K_Cd = 1; n = 2;\n"
        "r1: O => C; k_close*O;\n"
        "r2: C => O; k_open*Ca^n/(K_Cd^n + Ca^n)*C;",
        inputs={"Ca"},
    )
    assert D.describe(dims["k_close"]) == "time^-1"
    assert D.describe(dims["K_Cd"]) == "Ca"
    assert D.differ(dims["k_close"], dims["K_Cd"]) is True
    # Two first-order rates for different processes share a dimension: whether they are one
    # quantity is the agent's biology call, not this filter's.
    assert D.differ(dims["k_close"], dims["k_open"]) is False


def test_a_feedback_threshold_is_not_a_rate():
    """fujita: `akt_p_Kfb == s6k_dp_k` was taken. The threshold divides a species in a `1 + x/K`
    inhibition term, which only says anything once a summed number is read as dimensionless."""
    dims = _dims(
        "species S = 1; species pS = 0; species pAkt = 1; k_p = 1; k_dp = 1; K_fb = 1;\n"
        "r1: S => pS; k_p*pAkt*S/(1 + pS/K_fb);\n"
        "r2: pS => S; k_dp*pS;"
    )
    assert D.differ(dims["K_fb"], dims["k_dp"]) is True
    assert D.differ(dims["K_fb"], dims["S"]) is False  # pS's pool, which S => pS makes S's
    # a second-order rate constant is not a first-order one either
    assert D.differ(dims["k_p"], dims["k_dp"]) is True


def test_a_drug_ec50_is_not_a_carrying_capacity():
    """alkan: `K3 = Kcarry`. A cell count and a drug level no reaction or sum connects stay
    different generators, so a constant of each differs."""
    dims = _dims(
        "species N = 1; species Dead = 0; SN38 = 1; r = 1; Kcarry = 100; K3 = 1; kdeath = 1;\n"
        "g: => N; r*N*(1 - N/Kcarry);\n"
        "d: N => Dead; kdeath*SN38/(K3 + SN38)*N;",
        inputs={"SN38"},
    )
    assert D.differ(dims["K3"], dims["Kcarry"]) is True
    assert D.differ(dims["r"], dims["kdeath"]) is False


def test_an_input_only_ever_multiplied_is_a_number():
    """A knockdown multiplier or a cell-line switch scales a rate; given a dimension of its own it
    would split the scaled rate from the same rate unscaled, a tie that is perfectly writable
    (Bruno's `kc4_multiplier`, Raimundez's cell-line indicators)."""
    dims = _dims(
        "species A = 1; species B = 0; kd = 1; k1 = 1; k2 = 1;\n"
        "r1: A => B; k1*kd*A;\n"
        "r2: B => A; k2*B;",
        inputs={"kd"},
    )
    assert D.differ(dims["k1"], dims["k2"]) is False


def test_what_the_math_leaves_open_is_never_a_mismatch():
    """Fail OPEN: a product of two parameters pins neither, so both stay candidates, while a
    difference that holds whatever is undetermined is still one."""
    dims = _dims(
        "species A = 1; species B = 0; k1 = 1; k2 = 1; k = 1; K = 1;\n"
        "r1: A => B; k1*k2*A;\n"
        "r2: B => A; k*B/(K + B);"
    )
    assert any(c.startswith(D.UNDETERMINED) for c in dims["k1"])
    assert D.differ(dims["k1"], dims["k"]) is None
    assert D.differ(dims["k1"], dims["k2"]) is None
    assert D.differ(None, dims["k"]) is None  # an id the model does not have
    assert D.differ(dims["k"], dims["K"]) is True


def test_one_inconsistent_rate_law_costs_only_its_own_information():
    """`A => B; A` has no rate constant: read literally it makes time a number, or, posted
    participant by participant, a species another species per time. Either would corrupt every
    other parameter's dimension; dropped, the rest stays exact."""
    dims = _dims(
        "species A = 1; species B = 0; species C = 0; k = 1; ka = 1;\n"
        "r1: A => B; A;\n"
        "r2: B => C; k*B;\n"
        "r3: => A; ka + 1e-6;"
    )
    assert D.describe(dims["k"]) == "time^-1"
    # zero-order synthesis is an amount per time; the offset summed onto it is not read as making
    # the flux dimensionless
    per_time = {c: v for c, v in dims["A"].items()} | {D.TIME: Fraction(-1)}
    assert D.differ(dims["ka"], per_time) is False


def test_time_and_transcendental_arguments_pin_rates():
    """`exp(-k*time)` makes `k` a rate with no reaction saying so; a delay compared with time is a
    time."""
    dims = _dims(
        "species A = 1; k = 1; t_on = 1; k_deg = 1;\n"
        "r1: A => ; k_deg*A*exp(-k*time)*piecewise(1, time > t_on, 0);"
    )
    assert D.describe(dims["k"]) == "time^-1"
    assert D.describe(dims["t_on"]) == "time"
    assert D.differ(dims["k"], dims["k_deg"]) is False


def test_a_condition_table_entry_gives_its_target_its_value():
    """lucarelli ships totals the condition table assigns to initial amounts by name: the total is
    that amount, though no model math mentions it."""
    dims = _dims(
        "species S2 = init_S2; init_S2 = 1; k = 1;\nr1: S2 => ; k*S2;",
        inputs={"init_S2"},
        entries=[("init_S2", "S2tot")],
    )
    assert D.differ(dims["S2tot"], dims["S2"]) is False
    assert D.differ(dims["S2tot"], dims["k"]) is True


def test_the_merge_report_drops_a_dimension_mismatch_and_says_so():
    """The filter beside role and scale: a refused pair is counted by reason, a pair it cannot place
    is still priced, and a call without the table says the filter was off."""
    from petab_mcp.identifiability import merge_candidates

    names = ["k_close", "K_Cd", "k_open", "x"]
    dims = {
        "k_close": {D.TIME: Fraction(-1)},
        "K_Cd": {"Ca": Fraction(1)},
        "k_open": {D.TIME: Fraction(-1)},
        "x": {D.UNDETERMINED + "x": Fraction(1)},
    }
    rep = merge_candidates(np.eye(4), names, [1.0] * 4, dimensions=dims)
    priced = {tuple(t["parameters"]) for t in rep["ties"]}
    assert ("k_close", "K_Cd") not in priced and ("K_Cd", "k_open") not in priced
    assert ("k_close", "k_open") in priced
    assert {("k_close", "x"), ("K_Cd", "x"), ("k_open", "x")} <= priced  # undetermined: kept
    assert rep["n_ties_skipped"] == {"dimension_mismatch": 2}
    assert "dimension" not in rep.get("filters_off", [])

    off = merge_candidates(np.eye(4), names, [1.0] * 4)
    assert off["n_ties_priced"] == 6 and "dimension" in off["filters_off"]
