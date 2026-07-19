from __future__ import annotations

import pytest

from petab_mcp import operations as O
from petab_mcp import parameters as P
from petab_mcp.errors import UnknownEntityError, ValidationError
from petab_mcp.operations import OperationError, parse_reaction_line


def _ids(session):
    return {r["parameterId"] for r in P.parameters_to_records(session.problem)}


def _species(session):
    sm = session.model.sbml_model
    return {sm.getSpecies(i).getId() for i in range(sm.getNumSpecies())}


# -- parsing ---------------------------------------------------------------
def test_parse_reaction_line():
    spec = parse_reaction_line("J: 2 A + B -> C; k1*A*B")
    assert spec["id"] == "J"
    assert spec["reactants"] == [("A", 2.0), ("B", 1.0)]
    assert spec["products"] == [("C", 1.0)]
    assert spec["rate_law"] == "k1*A*B"


def test_parse_reaction_line_degradation():
    spec = parse_reaction_line("A => ; k_deg*A")
    assert spec["reactants"] == [("A", 1.0)]
    assert spec["products"] == []


def test_parse_reaction_line_no_arrow():
    with pytest.raises(OperationError):
        parse_reaction_line("A + B; k*A")


# -- enforcement -----------------------------------------------------------
def test_requires_justification(session):
    with pytest.raises(OperationError):
        O.add_reaction(session, "J: B -> D; k3*B", justification="  ", references=["PMID:1"])


def test_references_optional_for_model_ops(session):
    # references are optional for model ops now (justification still required). This edit
    # adds species D, which appears in no observable -> a parsimony warning.
    res = O.add_reaction(session, "J: B -> D; k3*B", justification="B is degraded to D")
    assert res["committed"] is True
    assert res["references"] == []
    assert any("unobserved species" in w for w in res["warnings"])


def test_no_parsimony_warning_when_reusing_species(session):
    # a repair reusing existing species introduces no new species -> no parsimony warning
    # (a sigma-bound-tightening warning may still appear; that's a separate concern)
    res = O.add_reaction(session, "J: B -> A; k2*B", justification="reverse conversion")
    assert res["committed"] is True
    assert not any("unobserved species" in w for w in res["warnings"])


# -- add / change / remove -------------------------------------------------
def test_add_reaction_auto_adds_species_and_params(session):
    res = O.add_reaction(
        session, "J: B -> D; k3*B", justification="B converts to D", references=["PMID:1"]
    )
    assert res["committed"] is True
    assert "D" in _species(session)
    assert {"D_init", "k3"} <= _ids(session)  # new IC param + rate constant added to table
    assert set(res["reconcile"]["parameters"]["added"]) == {"D_init", "k3"}


def test_ic_parameter_fixed_by_default(session):
    # A new species' initial-condition parameter is FIXED (estimate 0) unless the scenario opts in.
    O.add_reaction(session, "J: B -> D; k3*B", justification="B converts to D")
    assert int(session.problem.parameter_df.loc["D_init", "estimate"]) == 0


def test_ic_parameter_estimated_when_opted_in(session):
    # With estimate_initial_conditions on, the IC is estimated: non-negative, linear scale.
    session.estimate_initial_conditions = True
    O.add_reaction(session, "J: B -> D; k3*B", justification="B converts to D")
    row = session.problem.parameter_df.loc["D_init"]
    assert int(row["estimate"]) == 1
    assert row["parameterScale"] == "lin"
    assert float(row["lowerBound"]) == 0.0


def test_change_rate_law(session):
    O.add_reaction(session, "J: B -> D; k3*B", justification="x", references=["PMID:1"])
    res = O.change_rate_law(
        session, "J", "k3*B*B", justification="second order", references=["PMID:2"]
    )
    assert res["committed"] is True
    rxn = session.model.sbml_model.getReaction("J")
    assert "B * B" in rxn.getKineticLaw().getFormula().replace("^", " ") or "B^2" in rxn.getKineticLaw().getFormula()


def test_remove_reaction_prunes_orphans(session):
    O.add_reaction(session, "J: B -> D; k3*B", justification="x", references=["PMID:1"])
    res = O.remove_reaction(session, "J", justification="retracted", references=["PMID:3"])
    assert res["committed"] is True
    assert "D" not in _species(session)
    assert not ({"D_init", "k3"} & _ids(session))  # orphans pruned from the table


def test_remove_unknown_reaction(session):
    with pytest.raises(UnknownEntityError):
        O.remove_reaction(session, "nope", justification="x", references=["PMID:1"])


# -- parameter operations --------------------------------------------------
# -- initial-condition report (issue #57) ----------------------------------
def test_commit_result_includes_initial_conditions(session):
    # Every reconciling model op reports each species' IC status; a new species is flagged.
    res = O.add_reaction(session, "J: B -> D; k3*B", justification="B converts to D")
    ics = {r["species"]: r for r in res["initial_conditions"]}
    assert {"A", "B", "D"} <= set(ics)
    assert ics["D"]["newly_added"] is True
    assert ics["D"]["t0_source"] == "fixed" and ics["D"]["declared_t0_used"] is True


def test_no_parameter_estimate_or_fix_operation():
    # Changing a parameter's estimate status in-loop is deliberately unsupported — no such op.
    assert not hasattr(O, "estimate_parameter")
    assert not hasattr(O, "change_parameter_estimated")
    assert not hasattr(O, "fix_parameter")


def test_change_parameter_scale(session):
    res = O.change_parameter_scale(session, "sigma_B", "log10", justification="spans magnitudes")
    assert res["committed"] is True
    assert res["references"] == []  # references optional for parameter operations
    recs = {r["parameterId"]: r for r in P.parameters_to_records(session.problem)}
    assert recs["sigma_B"]["parameterScale"] == "log10"


def test_change_parameter_scale_requires_justification(session):
    with pytest.raises(OperationError):
        O.change_parameter_scale(session, "sigma_B", "log10", justification="")


def test_change_parameter_scale_invalid_is_reverted(session):
    # A_init has lowerBound 0 -> log10 is invalid; the table must be unchanged.
    before = {r["parameterId"]: r for r in P.parameters_to_records(session.problem)}
    with pytest.raises(OperationError):
        O.change_parameter_scale(session, "A_init", "log10", justification="x")
    after = {r["parameterId"]: r for r in P.parameters_to_records(session.problem)}
    assert after["A_init"] == before["A_init"]


def test_change_parameter_bounds(session):
    res = O.change_parameter_bounds(session, "k1", 1e-3, 1e3, justification="physiological range")
    assert res["committed"] is True
    recs = {r["parameterId"]: r for r in P.parameters_to_records(session.problem)}
    assert float(recs["k1"]["lowerBound"]) == 1e-3
    assert float(recs["k1"]["upperBound"]) == 1e3


def test_change_parameter_bounds_bad_range_reverts(session):
    with pytest.raises(Exception):
        O.change_parameter_bounds(session, "k1", 10.0, 1.0, justification="x")
    recs = {r["parameterId"]: r for r in P.parameters_to_records(session.problem)}
    assert float(recs["k1"]["upperBound"]) != 1.0  # not applied


def test_compile_fit_failure_message_reserved_name():
    # A codegen clash on an AMICI-reserved symbol (KeyError 'amici_h') must yield a rename
    # hint that names the offending symbol — NOT the generic "ill-posed ODE" message.
    msg = O._compile_fit_failure_message(KeyError("amici_h"))
    assert "'h'" in msg and "reserves" in msg and "rename" in msg.lower()
    assert "ill-posed" not in msg


def test_compile_fit_failure_message_generic_is_ill_posed():
    # A genuine solver failure keeps the ill-posed guidance.
    msg = O._compile_fit_failure_message(RuntimeError("CVODES failed: too much accuracy"))
    assert "ill-posed" in msg


def test_compile_fit_failure_message_no_false_positive_on_internals():
    # 'amici_model' is an internal identifier, not a user symbol clash -> ill-posed fallback.
    msg = O._compile_fit_failure_message(RuntimeError("could not build amici_model in /tmp"))
    assert "ill-posed" in msg
    assert "reserves" not in msg


def test_operation_records_version_and_provenance(session):
    before = session.provenance.current_version
    O.add_reaction(session, "J: B -> D; k3*B", justification="x", references=["PMID:1"])
    assert session.provenance.current_version == before + 1
    events = session.provenance.recent_events()
    alt = [e for e in events if e["type"] == "alteration"][-1]
    assert alt["operation"] == "add_reaction"
    assert alt["justification"] == "x"
    assert alt["references"][0]["input"] == "PMID:1"
