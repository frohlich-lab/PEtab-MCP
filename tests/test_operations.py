from __future__ import annotations

import pytest

from petab_mcp import operations as O
from petab_mcp import parameters as P
from petab_mcp.operations import OperationError

# The shipped toy model (tests/data/toy_v1/model.ant), used as the base for every re-author.
# Structural edits are made by authoring this base with the change applied via ``set_model`` —
# ``{extra}`` is where extra reaction lines are spliced in.
_BASE_MODEL = """model conversion_reaction
  compartment compartment_ = 1;
  species A in compartment_;
  species B in compartment_;
  A = A_init;
  B = B_init;
  A_init = 1.0;
  B_init = 0.0;
  k1 = 0.5;
  k2 = 0.1;
  r1: A => B; k1 * A;
  r2: B => A; k2 * B;
{extra}end
"""


def _set_model(session, extra=(), *, justification="test", **kwargs):
    """Re-author the base model with ``extra`` reaction lines spliced in (each on its own line)."""
    antimony = _BASE_MODEL.format(extra="".join(f"  {line}\n" for line in extra))
    return O.set_model(session, antimony, justification=justification, **kwargs)


def _ids(session):
    return {r["parameterId"] for r in P.parameters_to_records(session.problem)}


def _species(session):
    sm = session.model.sbml_model
    return {sm.getSpecies(i).getId() for i in range(sm.getNumSpecies())}


# -- enforcement -----------------------------------------------------------
def test_requires_justification(session):
    with pytest.raises(OperationError):
        O.set_model(session, _BASE_MODEL.format(extra=""), justification="  ", references=["PMID:1"])


def test_references_optional_for_model_ops(session):
    # references are optional for model ops now (justification still required). This edit
    # adds species D, which appears in no observable -> a parsimony warning.
    res = _set_model(session, ["J: B => D; k3 * B;"], justification="B is degraded to D")
    assert res["committed"] is True
    assert res["references"] == []
    assert any("unobserved species" in w for w in res["warnings"])


def test_no_parsimony_warning_when_reusing_species(session):
    # a repair reusing existing species introduces no new species -> no parsimony warning
    # (a sigma-bound-tightening warning may still appear; that's a separate concern)
    res = _set_model(session, ["J: A => B; k3 * A;"], justification="extra forward conversion")
    assert res["committed"] is True
    assert not any("unobserved species" in w for w in res["warnings"])


def test_no_parsimony_warning_when_new_species_feeds_observable(session):
    # C is unobserved but PRODUCES the observed species B (C -> B), so the data constrains it
    # through the network — a legitimate upstream addition, not a degenerate sink. The warning
    # must not fire, or it trains the agent to ignore the one warning type that ever does.
    res = _set_model(session, ["J: C => B; k3 * C;"], justification="C is converted to B")
    assert res["committed"] is True
    assert "C" in _species(session)
    assert not any("unobserved species" in w for w in res["warnings"])


def test_parsimony_warning_still_fires_for_disconnected_sink(session):
    # D is a pure downstream sink of the observed species B (B -> D): it feeds no observable and
    # no input sets it, so it stays flagged as a genuinely disconnected hidden state.
    res = _set_model(session, ["J: B => D; k3 * B;"], justification="B drains to D")
    assert res["committed"] is True
    assert any("unobserved species" in w for w in res["warnings"])


# -- re-author: reconcile of new/removed structure -------------------------
def test_reauthor_auto_adds_species_and_params(session):
    res = _set_model(session, ["J: B => D; k3 * B;"], justification="B converts to D",
                     references=["PMID:1"])
    assert res["committed"] is True
    assert "D" in _species(session)
    assert {"D_init", "k3"} <= _ids(session)  # new IC param + rate constant added to table
    assert set(res["reconcile"]["parameters"]["added"]) == {"D_init", "k3"}


def test_ic_parameter_fixed_by_default(session):
    # A new species' initial-condition parameter is FIXED (estimate 0) unless the scenario opts in.
    _set_model(session, ["J: B => D; k3 * B;"], justification="B converts to D")
    assert int(session.problem.parameter_df.loc["D_init", "estimate"]) == 0


def test_empty_pool_ic_stays_fixed_even_when_opted_in(session):
    # A newly-added species starts EMPTY (t=0 == 0). Even with estimate_initial_conditions on, an
    # empty pool's IC stays FIXED at 0 — estimate_ics estimates only genuinely-seeded (nonzero)
    # species (see reconcile.sync_parameter_table + test_reconcile). A modeller seeds a pool by
    # giving it a nonzero initial; a species declared to start empty has no t=0 value to fit.
    session.estimate_initial_conditions = True
    _set_model(session, ["J: B => D; k3 * B;"], justification="B converts to D")
    row = session.problem.parameter_df.loc["D_init"]
    assert int(row["estimate"]) == 0
    assert float(row["nominalValue"]) == 0.0


def test_reauthor_change_rate_law(session):
    # Re-authoring the same reaction id with a new rate law replaces its kinetic law.
    _set_model(session, ["J: B => D; k3 * B;"], justification="x", references=["PMID:1"])
    res = _set_model(session, ["J: B => D; k3 * B * B;"], justification="second order",
                     references=["PMID:2"])
    assert res["committed"] is True
    rxn = session.model.sbml_model.getReaction("J")
    formula = rxn.getKineticLaw().getFormula()
    assert "B * B" in formula.replace("^", " ") or "B^2" in formula


def test_reauthor_without_reaction_prunes_orphans(session):
    # Author a reaction that introduces D, then re-author WITHOUT it: D and its params are pruned.
    _set_model(session, ["J: B => D; k3 * B;"], justification="x", references=["PMID:1"])
    res = _set_model(session, [], justification="retract the sink", references=["PMID:3"])
    assert res["committed"] is True
    assert "D" not in _species(session)
    assert not ({"D_init", "k3"} & _ids(session))  # orphans pruned from the table


# -- parameter operations --------------------------------------------------
# -- initial-condition report (issue #57) ----------------------------------
def test_commit_result_includes_initial_conditions(session):
    # Every reconciling model op reports each species' IC status; a new species is flagged.
    res = _set_model(session, ["J: B => D; k3 * B;"], justification="B converts to D")
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


# -- repeated-widening backstop (issue #56) --------------------------------
def _add_version(session, *, upper, nllh, kind="alteration"):
    """Snapshot a real version whose stored parameter table sets k1's UPPER bound to ``upper``
    and whose meta records ``neg_log_likelihood`` (``None`` => a simulate-only commit). The
    guard reconstructs widening history from these stored tables, so this mirrors exactly how a
    real commit / batch / rollback lands on disk."""
    from petab.v1.C import UPPER_BOUND

    df = session.problem.parameter_df.copy(deep=True)
    df.loc["k1", UPPER_BOUND] = upper
    meta = {} if nllh is None else {"neg_log_likelihood": nllh}
    session.provenance.snapshot_version(
        session.problem.model.to_antimony(), df, kind=kind, meta=meta)


def _pin_k1_at_upper(session, upper):
    """Point the live table + last fit at ``upper`` so k1 reads as re-pinned at its upper bound."""
    from petab.v1.C import UPPER_BOUND

    session.problem.parameter_df.loc["k1", UPPER_BOUND] = upper
    session.simulation = {"fitted_parameters": {"k1": float(upper)}}


def test_reject_third_widening_of_repinning_parameter(session):
    # Two prior widenings (100 -> 1e3 -> 1e4), still pinned, negligible last gain (ΔBIC=0.2) ->
    # the third widening is refused.
    _add_version(session, upper=1e3, nllh=100.0)
    _add_version(session, upper=1e4, nllh=99.9)
    _pin_k1_at_upper(session, 1e4)
    with pytest.raises(OperationError, match="Refusing to widen"):
        O._reject_repeated_widening(session, session.problem, "k1", 1e-4, 1e5)


def test_third_widening_rejected_through_full_operation_and_reverts(session):
    # Same, but through the public op: it must raise without applying the new bound.
    from petab.v1.C import UPPER_BOUND

    _add_version(session, upper=1e3, nllh=100.0)
    _add_version(session, upper=1e4, nllh=99.9)
    _pin_k1_at_upper(session, 1e4)
    with pytest.raises(OperationError):
        O.change_parameter_bounds(session, "k1", 1e-4, 1e5, justification="widen again")
    assert float(session.problem.parameter_df.loc["k1", UPPER_BOUND]) == 1e4  # unchanged


def test_second_widening_is_allowed(session):
    # Only one prior widening -> the next one is fine (widen at most once is skill guidance;
    # the hard backstop only blocks the THIRD).
    _add_version(session, upper=1e3, nllh=100.0)
    _pin_k1_at_upper(session, 1e3)
    O._reject_repeated_widening(session, session.problem, "k1", 1e-4, 1e5)  # no raise


def test_widening_allowed_when_not_at_bound(session):
    # Two prior widenings but the parameter now sits mid-range -> not the pathology; allowed.
    _add_version(session, upper=1e3, nllh=100.0)
    _add_version(session, upper=1e4, nllh=99.9)
    session.problem.parameter_df.loc["k1", "upperBound"] = 1e4
    session.simulation = {"fitted_parameters": {"k1": 1.0}}  # comfortably inside the box
    O._reject_repeated_widening(session, session.problem, "k1", 1e-4, 1e5)  # no raise


def test_widening_allowed_when_last_gain_is_meaningful(session):
    # The last widening bought a real BIC improvement (ΔBIC = 2*(100-80) = 40) -> keep going.
    _add_version(session, upper=1e3, nllh=100.0)
    _add_version(session, upper=1e4, nllh=80.0)
    _pin_k1_at_upper(session, 1e4)
    O._reject_repeated_widening(session, session.problem, "k1", 1e-4, 1e5)  # no raise


def test_widening_allowed_when_last_gain_unknown(session):
    # A simulate-only commit before the last widening leaves its BIC gain undeterminable. The
    # backstop is defined by a NEGLIGIBLE gain, so an unknown gain must NOT trigger it.
    _add_version(session, upper=1e3, nllh=None)   # no neg_log_likelihood recorded
    _add_version(session, upper=1e4, nllh=99.9)
    _pin_k1_at_upper(session, 1e4)
    O._reject_repeated_widening(session, session.problem, "k1", 1e-4, 1e5)  # no raise


def test_narrowing_a_repinning_parameter_is_never_blocked(session):
    # Moving the pinned bound INWARD is not a widening -> never guarded.
    _add_version(session, upper=1e3, nllh=100.0)
    _add_version(session, upper=1e4, nllh=99.9)
    _pin_k1_at_upper(session, 1e4)
    O._reject_repeated_widening(session, session.problem, "k1", 1e-4, 5e3)  # no raise


def test_rollback_resets_widening_count(session):
    # Two widenings then a rollback that restores the smaller box (an inward move) -> the run of
    # widenings resets, so the next widening is a fresh first, not a blocked third.
    _add_version(session, upper=1e3, nllh=100.0)
    _add_version(session, upper=1e4, nllh=99.9)
    _add_version(session, upper=1e3, nllh=100.0, kind="rollback")  # restores the smaller box
    _pin_k1_at_upper(session, 1e3)
    O._reject_repeated_widening(session, session.problem, "k1", 1e-4, 1e4)  # no raise


def test_widenings_inside_committed_batches_are_counted(session):
    # A widening committed via a batch lands as a normal version snapshot, so it counts toward
    # the backstop exactly like a single-op widening.
    _add_version(session, upper=1e3, nllh=100.0, kind="batch")
    _add_version(session, upper=1e4, nllh=99.9, kind="batch")
    _pin_k1_at_upper(session, 1e4)
    with pytest.raises(OperationError, match="Refusing to widen"):
        O._reject_repeated_widening(session, session.problem, "k1", 1e-4, 1e5)


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
    _set_model(session, ["J: B => D; k3 * B;"], justification="x", references=["PMID:1"])
    assert session.provenance.current_version == before + 1
    events = session.provenance.recent_events()
    alt = [e for e in events if e["type"] == "alteration"][-1]
    assert alt["operation"] == "set_model"
    assert alt["justification"] == "x"
    assert alt["references"][0]["input"] == "PMID:1"
