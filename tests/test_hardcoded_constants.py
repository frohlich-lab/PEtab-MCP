"""Hardcoded constants are priced into ``k`` (``petab_mcp.literals``, issue #180).

Every complexity-penalised score charged ``k = n_estimated``, so a value written straight into a
rate law cost nothing however it was obtained: one agent's model had 21 reactions, 10 species and
``k = 4`` — its rate laws read ``0.000878``, ``2.69e-5``, ``1.333``… — and scored best of 29
sessions. Two halves are pinned here: the RULE (which literals count, on libsbml ASTs built the
way the server builds them, from Antimony) and the ASSEMBLY (every place ``k`` is derived reads the
same addend, from the fit payload to the stored version's bic). A caller that derives ``k`` on its
own pins that derivation against this one.
"""

from __future__ import annotations

import math

import antimony
import libsbml
import pytest


from petab_mcp import fitting
from petab_mcp import literals as L


def _sbml_model(ant: str):
    antimony.clearPreviousLoads()
    assert antimony.loadAntimonyString(ant) >= 0, antimony.getLastError()
    doc = libsbml.readSBMLFromString(antimony.getSBMLString(antimony.getMainModuleName()))
    return doc.getModel()


def _rate_law(law: str) -> str:
    return f"model m\n  species A, B;\n  A = 5; B = 0;\n  kf = 0.5;\n  R1: A -> B; {law};\nend\n"


@pytest.mark.parametrize(
    "law, priced",
    [
        # the literals of the motivating model: fitted values, frozen in
        ("0.000878*A", ["0.000878"]),
        ("2.69e-5*A", ["2.69e-5"]),
        ("kf*A^1.333", ["1.333"]),
        ("kf*A/(0.0305 + A)", ["0.0305"]),
        # published-looking numbers, integer or not
        ("68190*kf", ["68190"]),
        ("kf*A/(12.5 + A)", ["12.5"]),
        # STRUCTURAL_VALUES stay free: zero, the identities, the oligomer multiplicities and their
        # reciprocals, however spelled
        ("0*A + 1*B", []),
        ("2*kf*A", []),
        ("kf*A^2/(1 + A^2)", []),
        ("kf*A^3", []),
        ("kf*A^4", []),
        ("6*kf*A", []),
        ("0.5*kf*A - 0.25*kf*B", []),  # 1/2 and 1/4 by value
        ("kf*A*(1/3) + kf*B*(1/6)", []),
        ("kf*A/6 - kf*B/4", []),
        # percent is free only as a readout's factor (see the readout test): in a rate it is a rate
        ("100*kf*A", ["100"]),
        ("kf*A/100", ["100"]),
        # ...and nothing else is. A unit or scale factor is a magnitude like any other: name it, or
        # pay for it. These were all free under the old three-significant-digit threshold.
        ("1000000*kf*A", ["1000000"]),
        ("60*kf*A + 3600*kf*B", ["60", "3600"]),
        ("5*kf*A", ["5"]),  # 5 is not an oligomer size and not in the set
        ("5e-5*A", ["5e-5"]),
        # exact rational comparison: a rounded third is not a third
        ("kf*A*0.333", ["0.333"]),
    ],
)
def test_only_the_structural_multiplicities_are_free(law, priced):
    """The free set is enumerated (`literals.STRUCTURAL_VALUES`), not a threshold, so there is no
    rounding move onto it: the nearest free value to a fitted rate constant is `0`, which deletes
    the term. Under the old three-digit rule `0.878` could be written `0.88` for nothing, and
    `1000000`/`60`/`3600` were free for having few digits rather than for being structural.

    Membership is by exact rational VALUE, so `0.5` IS `1/2` and free while `0.333` is not `1/3`.
    """
    assert L.hardcoded_constants(_sbml_model(_rate_law(law))) == priced


def test_the_free_set_is_closed_under_sign_and_reciprocal():
    """Both directions of every multiplicity, because `2*x`, `x/2`, `-2*x` and `-x/2` are the same
    structural statement written four ways — and a rule that charged only some of them would push
    an author toward the spelling that happens to be free."""
    from fractions import Fraction

    for n in (1, 2, 3, 4, 6):
        for v in (Fraction(n), -Fraction(n), Fraction(1, n), -Fraction(1, n)):
            assert v in L.STRUCTURAL_VALUES, v
    assert Fraction(0) in L.STRUCTURAL_VALUES
    # 5 and its reciprocal are deliberately absent: no oligomer is a pentamer here, and every
    # value admitted is a value a fitted number could be rounded onto.
    assert Fraction(5) not in L.STRUCTURAL_VALUES
    assert Fraction(1, 5) not in L.STRUCTURAL_VALUES
    assert Fraction(100) not in L.STRUCTURAL_VALUES


@pytest.mark.parametrize(
    "law, priced",
    [
        # issue #338: a Hill switch of order 8 written as a product of eight factors was free
        ("kf*A*A*A*A*A*A*A*A/(K^8 + A*A*A*A*A*A*A*A)", ["A^8"]),
        ("kf*A^4*A^4", ["A^8"]),
        ("kf*(A^4)^2", ["A^8"]),
        ("kf*(A*A)^4", ["A^8"]),
        ("kf*A^(4 + 4)", ["A^8"]),
        ("kf*A*A*A*A*A*A", []),  # a hexamer is still structural
        # free literals multiplied, added or passed through exp into a value that is not free
        ("2*2*2*kf*A", ["2 * 2 * 2"]),
        ("kf*A/2/2/2", ["0.125"]),
        ("3*0.5*kf*A", ["3 * 0.5"]),
        ("(6 - 1)*kf*A", ["6 - 1"]),
        ("exp(-4)*A", ["exp(-4)"]),
        ("kf*A + kf*A + kf*A + kf*A + kf*A", ["5*(kf * A)"]),
        # ...while a value already charged as written is not charged again for what it makes
        ("kf*A^0.7*A^0.7", ["0.7"]),
        ("10^-3*kf*A", ["10"]),
        ("kf*A/5 + 5*kf*B", ["5"]),
        ("0.5*0.5*kf*A", []),
        ("2*pi*kf*A", []),
    ],
)
def test_free_literals_do_not_combine_into_a_priced_value(law, priced):
    """Issue #338: `x^8` written `x*x*...*x` carried no literal, so the exponent a Hill switch
    charges was free. Normalising repeated products to powers closes that; the same holds for
    any other way of composing free values into one that is not free."""
    sm = _sbml_model(
        f"model m\n  species A, B;\n  A = 5; B = 0;\n  kf = 0.5; K = 1;\n  R1: A -> B; {law};\nend\n"
    )
    assert L.hardcoded_constants(sm) == priced


def test_percent_is_free_only_as_a_whole_readouts_factor():
    """Issue #338: the exempt factor 100 froze a bound-limited rate for free (`100*LR*P3`). It
    stays free where nothing can absorb it, a factor of a whole assignment rule nothing in the
    dynamics reads, through a readout chain (`readout_frac := frac`) as the benchmark wires it."""
    readout = (
        "model m\n  species A, B;\n  A = 5; B = 0;\n  kf = 0.5;\n  R1: A -> B; kf*A;\n"
        "  frac := {frac};\n  readout_frac := frac;\nend\n"
    )
    assert L.hardcoded_constants(_sbml_model(readout.format(frac="(1 - A/(A + B))*100"))) == []
    assert L.hardcoded_constants(_sbml_model(readout.format(frac="50*A/(A + B)"))) == []
    # not a factor of the whole readout
    assert L.hardcoded_constants(_sbml_model(readout.format(frac="100 - 100*A/(A + B)"))) == ["100"]
    # a rule the dynamics read is a rate, not a readout
    feeds = "model m\n  species A, B;\n  A = 5; B = 0;\n  kr := 100*A;\n  R1: A -> B; kr;\nend\n"
    assert L.hardcoded_constants(_sbml_model(feeds)) == ["100"]


def test_two_rates_sharing_a_value_are_one_constant_as_their_merge_is():
    """Issue #338 named two unrelated rates sharing `50` as a gap; it is the price of the merge
    that dominates it. Estimating the shared value is one parameter, so the literal pair costs
    what that merge costs and fits no better. A sign does not make it a second constant."""
    sm = _sbml_model(
        "model m\n  species A, B, C;\n  A = 5; B = 0; C = 0;\n"
        "  R1: A -> B; 50*A;\n  R2: B -> C; B/(50 + B) - (-50)*C;\nend\n"
    )
    assert L.hardcoded_constants(sm) == ["50"]


def test_each_distinct_value_is_charged_once_however_written():
    """`1.333` appears in three of the motivating model's rate laws as ONE shared exponent — one
    degree of freedom, one parameter. Two spellings of the same value are the same constant."""
    ant = (
        "model m\n  species A, B, C;\n  A = 5; B = 0; C = 0;\n  kf = 0.5;\n"
        "  R1: A -> B; kf*A^1.333;\n"
        "  R2: B -> C; kf*B^1.333 + 2.69e-5*B;\n"
        "  R3: C -> A; 0.0000269*C + kf*C^1.333;\nend\n"
    )
    assert L.hardcoded_constants(_sbml_model(ant)) == ["2.69e-5", "1.333"]


def test_every_math_container_is_scanned_but_species_initials_and_parameter_values_are_not():
    """A literal is priced wherever the model puts math — an assignment rule (the readout route),
    an initial assignment, a function body, an event — because each is a place a fitted value can
    be frozen. A species initial value is NOT scanned: the reconcile turns a nonzero, non-unit one
    into an ESTIMATED `<sid>_init` row already, so scanning it would double-charge. A parameter's
    model value is overridden by the table, so it is not scanned either. `2.5` in the assignment
    IS charged — two significant digits bought nothing once the free set became a fixed list."""
    ant = (
        "function hill(y)\n  y^1.333/(0.0305^1.333 + y^1.333)\nend\n"
        "model m\n  species A, B;\n  A = 5.678; B = 0;\n  kf = 0.000878;\n  scale = 0.5;\n"
        "  R1: A -> B; kf*hill(A);\n"
        "  readout_B := 0.0345*B;\n"
        "  scale = 2.5*kf;\n"
        "  E1: at (time > 10.5): A = 1.234;\nend\n"
    )
    sm = _sbml_model(ant)
    assert L.hardcoded_constants(sm) == ["0.0305", "0.0345", "1.234", "1.333", "2.5", "10.5"]


def test_complexity_k_adds_the_constants_to_the_estimated_rows():
    assert (
        fitting.complexity_k({"n_estimated": 4, "hardcoded_constants": ["0.000878", "1.333"]}) == 6
    )
    # a payload from before the field existed charges none; one without n_estimated has no k at all
    assert fitting.complexity_k({"n_estimated": 4}) == 4
    assert fitting.complexity_k({"hardcoded_constants": ["1.333"]}) is None
    assert fitting.complexity_k(None) is None


def test_the_fit_result_block_prices_the_constants_and_names_them():
    """The block's bic must charge the literals AND say so: a `k` above `n_estimated` with nothing
    to explain it reads as a bug, and the agent cannot act on a number it is not shown."""
    fitsim = {
        "status": "fit",
        "neg_log_likelihood": -50.0,
        "chi2": 18.0,
        "n_data_points": 100,
        "n_estimated": 4,
        "hardcoded_constants": ["0.000878", "2.69e-5", "1.333"],
        "fitted_parameters": {"a": 1.0, "b": 2.0, "c": 3.0, "d": 4.0},
    }
    block = fitting.fit_result_block(fitsim)
    assert block["n_estimated"] == 4 and block["k"] == 7
    assert block["hardcoded_constants"] == ["0.000878", "2.69e-5", "1.333"]
    assert block["bic"] == pytest.approx(7 * math.log(100) + 2 * -50.0)
    # ...and the list rides only when there is something to explain: it is re-sent every turn
    clean = fitting.fit_result_block({**fitsim, "hardcoded_constants": []})
    assert clean["k"] == 4 and "hardcoded_constants" not in clean


def test_a_new_best_is_judged_on_the_priced_bic(session):
    """`is_new_best` tracks BIC, so freezing four fitted values must not register as an
    improvement over estimating them: same nll, same k, same bic."""
    estimated = {
        "status": "fit",
        "neg_log_likelihood": -50.0,
        "chi2": 18.0,
        "n_data_points": 100,
        "n_estimated": 8,
        "hardcoded_constants": [],
    }
    frozen = {
        **estimated,
        "n_estimated": 4,
        "hardcoded_constants": ["0.000878", "2.69e-5", "1.333", "0.0305"],
    }
    a = fitting.compute_improvement(session, nllh=-50.0, objective_before=None, fitsim=estimated)
    b = fitting.compute_improvement(session, nllh=-50.0, objective_before=None, fitsim=frozen)
    assert a["bic"] == pytest.approx(b["bic"])


def test_the_streak_checkpoint_and_the_stored_version_use_the_same_k(session):
    """The early-stop checkpoint (`record_fit_progress`) and the version meta a caller scans
    (`_snapshot_version`) must price the literals too, or a frozen model would become the
    best-BIC checkpoint a caller restores while the fit block said otherwise."""
    fitsim = {
        "status": "fit",
        "neg_log_likelihood": -50.0,
        "chi2": 18.0,
        "n_data_points": 100,
        "n_estimated": 4,
        "hardcoded_constants": ["0.000878", "1.333"],
        "fitted_parameters": {"k1": 0.5, "k2": 0.1},
    }
    expected = 6 * math.log(100) + 2 * -50.0
    session.record_fit_progress(0, fitsim, {"is_new_best": True})
    # Checkpoint.score is the session's criterion — bic under the default the session resolves to.
    assert session.best.score == pytest.approx(expected)
    v = session._snapshot_version(meta={"neg_log_likelihood": -50.0}, sim=fitsim)
    assert session.provenance.get_meta(v)["bic"] == pytest.approx(expected)


_FROZEN_TOY = """model conversion_reaction
  compartment compartment_ = 1;
  species A in compartment_;
  species B in compartment_;
  A = A_init;
  B = B_init;
  A_init = 1.0;
  B_init = 0.0;
  k1 = 0.5;
  r1: A => B; k1 * A;
  r2: B => A; 0.0987 * B;
end
"""


def test_a_fit_carries_the_constants_through_every_score(fit_session):
    """The fit path proper: the payload `fit_and_simulate` stores, the fit block the agent reads,
    the improvement it is ranked by, the stored version's meta and `get_fit_summary` all charge
    the frozen `0.0987`. This is the assembly that a green unit suite cannot vouch for."""
    from petab_mcp import operations as O

    sess = fit_session
    O.set_model(sess, _FROZEN_TOY, justification="freeze k2 at its fitted value")
    res = sess.fit(n_starts=2)
    assert res["committed"] is True
    sim = sess.simulation
    assert sim["hardcoded_constants"] == ["0.0987"]
    k = sim["n_estimated"] + 1
    assert res["k"] == k and res["hardcoded_constants"] == ["0.0987"]
    expected = k * math.log(sim["n_data_points"]) + 2 * sim["neg_log_likelihood"]
    assert res["bic"] == pytest.approx(expected)
    assert sess.provenance.get_meta(res["version"])["bic"] == pytest.approx(expected)
    summary = fitting.fit_summary(sess, sess.engine)
    assert summary["k"] == k and summary["bic"] == pytest.approx(expected)


@pytest.mark.parametrize("model, constants", [(None, []), (_FROZEN_TOY, ["0.0987"])])
def test_tic_charges_the_constants_through_k_eff(fit_session, model, constants):
    """Issue #10: TIC charges `k_eff`, and the fit counted it against `n_estimated` alone, so under
    `--criterion tic` the frozen `0.0987` cost nothing. A literal has no coordinate in the FIM, so it
    reaches `k_eff` through the `k − m` addend: one unit each on top of the same trace, two
    deviance units of TIC, and nothing at all on the shipped toy, which has no literal."""
    from petab_mcp import operations as O
    from petab_mcp.identifiability import effective_parameter_count

    sess = fit_session
    if model is not None:
        O.set_model(sess, model, justification="freeze k2 at its fitted value")
    res = sess.fit(n_starts=2)
    assert res["committed"] is True
    sim = sess.simulation
    assert sim["hardcoded_constants"] == constants
    rec = sess.fit_record
    uncharged = effective_parameter_count(rec.pypesto_problem, rec.x_free, k=sim["n_estimated"])
    assert uncharged is not None
    assert sim["k_eff"] == pytest.approx(uncharged + len(constants))
    expected = 2 * sim["neg_log_likelihood"] + 2 * sim["k_eff"]
    assert res["tic"] == pytest.approx(expected)
    assert sess.provenance.get_meta(res["version"])["tic"] == pytest.approx(expected)
