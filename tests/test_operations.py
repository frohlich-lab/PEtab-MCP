from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from petab_mcp import operations as O
from petab_mcp import parameters as P
from petab_mcp.errors import PetabMcpError
from petab_mcp.operations import OperationError
from petab_mcp.references import StubReferenceResolver
from petab_mcp.session import PetabSession

# The shipped toy model (tests/data/toy_v2/model.ant), used as the base for every re-author.
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
        O.set_model(
            session, _BASE_MODEL.format(extra=""), justification="  ", references=["PMID:1"]
        )


def test_commit_change_refuses_a_no_op_edit(session):
    """`_commit_change` with no callback must raise, not record an empty edit.

    Neither real caller can reach this (set_model always passes model_build, change_parameters
    always passes param_edit), so the guard is unreachable today and was briefly removed as dead
    code. It is not: without it, both callbacks None falls straight through to the recording
    step, applying nothing but still appending an `edit` event to the append-only provenance log
    step, applying nothing but still appending an `edit` event to the append-only provenance log.
    A no-op entry in an audit trail is worse than a crash, so the contract the docstring states is
    enforced, and the log must stay untouched.
    """
    before_events = len(session.provenance.recent_events(limit=1000))
    before_version = session.provenance.current_version

    with pytest.raises(ValueError, match="at least one callback"):
        O._commit_change(session, "bogus_op", {}, "a justification", None)

    assert len(session.provenance.recent_events(limit=1000)) == before_events
    assert session.provenance.current_version == before_version


def test_references_optional_for_model_ops(session):
    # references are optional for model ops now (justification still required). This edit
    # adds species D, which appears in no observable -> a parsimony warning.
    res = _set_model(session, ["J: B => D; k3 * B;"], justification="B is degraded to D")
    assert res["applied"] is True
    assert res["references"] == []
    assert any("unobserved species" in w for w in res["warnings"])


def test_no_parsimony_warning_when_reusing_species(session):
    # a repair reusing existing species introduces no new species -> no parsimony warning
    # (a sigma-bound-tightening warning may still appear; that's a separate concern)
    res = _set_model(session, ["J: A => B; k3 * A;"], justification="extra forward conversion")
    assert res["applied"] is True
    assert not any("unobserved species" in w for w in res["warnings"])


def test_no_parsimony_warning_when_new_species_feeds_observable(session):
    # C is unobserved but PRODUCES the observed species B (C -> B), so the data constrains it
    # through the network — a legitimate upstream addition, not a degenerate sink. The warning
    # must not fire, or it trains the agent to ignore the one warning type that ever does.
    res = _set_model(session, ["J: C => B; k3 * C;"], justification="C is converted to B")
    assert res["applied"] is True
    assert "C" in _species(session)
    assert not any("unobserved species" in w for w in res["warnings"])


def test_parsimony_warning_still_fires_for_disconnected_sink(session):
    # D is a pure downstream sink of the observed species B (B -> D): it feeds no observable and
    # no input sets it, so it stays flagged as a genuinely disconnected hidden state.
    res = _set_model(session, ["J: B => D; k3 * B;"], justification="B drains to D")
    assert res["applied"] is True
    assert any("unobserved species" in w for w in res["warnings"])


def _blinded_session(tmp_path):
    """The toy problem, blinded: its observable reads ``readout_B``, a measurement mapping the
    model must define by an assignment rule."""
    work = tmp_path / "blinded"
    shutil.copytree(Path(__file__).parent / "data" / "toy_v1", work)
    observables = work / "observables.tsv"
    observables.write_text(observables.read_text().replace("\tB\t", "\treadout_B\t"))
    return PetabSession(
        str(work / "problem.yaml"),
        provenance_dir=str(tmp_path / "prov"),
        resolver=StubReferenceResolver(),
    )


def test_a_readout_rule_carries_what_it_reads_into_the_observed_set(tmp_path):
    # On a blinded problem the formula names only `readout_B`, never a species. Read off the
    # formula alone, every added species was reported as feeding no observable — all ten, in one
    # model. C feeds B, which the readout reads; the sink D is still flagged.
    session = _blinded_session(tmp_path)
    res = _set_model(session, ["readout_B := B;", "J: C => B; k3 * C;"], justification="C feeds B")
    assert res["applied"] is True
    assert not any("unobserved species" in w for w in res["warnings"])
    res = _set_model(session, ["readout_B := B;", "J: B => D; k3 * B;"], justification="B to D")
    assert any("unobserved species ['D']" in w for w in res["warnings"])


# -- re-author: reconcile of new/removed structure -------------------------
def test_reauthor_auto_adds_species_and_params(session):
    res = _set_model(
        session, ["J: B => D; k3 * B;"], justification="B converts to D", references=["PMID:1"]
    )
    assert res["applied"] is True
    assert "D" in _species(session)
    # D is created EMPTY (no declared initial), so it gets NO `D_init` — an unset initial is an
    # empty pool, and minting a zero-nominal parameter for it is the artifact the IC convention
    # removes. Only the rate constant is added.
    assert "k3" in _ids(session)
    assert "D_init" not in _ids(session)
    assert set(res["reconcile"]["parameters"]["added"]) == {"k3"}


def test_empty_pool_ic_is_not_parametrised(session):
    # A species created EMPTY (t0 == 0, whether written as `D = 0;` or left unwritten) gets NO
    # `<sid>_init` at all. It used to get one FIXED at 0: a row nothing could ever use (the draft
    # TSV carries only bounds/scale, and `parameters=` sets bounds/scale), which regenerated into
    # the draft on every commit and, being zero-nominal, could not live on log10. ICs are
    # otherwise ESTIMATED by default.
    _set_model(session, ["J: B => D; k3 * B;"], justification="B converts to D")
    assert "D_init" not in session.problem.parameter_df.index
    assert session.model.sbml_model.getParameter("D_init") is None


def test_seeded_ic_estimated_by_default(session):
    # A newly-authored species given a positive initial is ESTIMATED (always-on): the declared
    # value is only a guess and must never be silently frozen (issue #90).
    _set_model(
        session,
        ["J: B => D; k3 * B;", "D = 5.0;"],
        justification="B converts to D, D seeded at a guess",
    )
    row = session.problem.parameter_df.loc["D_init"]
    assert int(row["estimate"]) == 1
    assert abs(float(row["nominalValue"]) - 5.0) < 1e-9


def test_reauthor_change_rate_law(session):
    # Re-authoring the same reaction id with a new rate law replaces its kinetic law.
    _set_model(session, ["J: B => D; k3 * B;"], justification="x", references=["PMID:1"])
    res = _set_model(
        session, ["J: B => D; k3 * B * B;"], justification="second order", references=["PMID:2"]
    )
    assert res["applied"] is True
    rxn = session.model.sbml_model.getReaction("J")
    formula = rxn.getKineticLaw().getFormula()
    assert "B * B" in formula.replace("^", " ") or "B^2" in formula


def test_reauthor_without_reaction_prunes_orphans(session):
    # Author a reaction that introduces D, then re-author WITHOUT it: D and its params are pruned.
    _set_model(session, ["J: B => D; k3 * B;"], justification="x", references=["PMID:1"])
    res = _set_model(session, [], justification="retract the sink", references=["PMID:3"])
    assert res["applied"] is True
    assert "D" not in _species(session)
    assert not ({"D_init", "k3"} & _ids(session))  # orphans pruned from the table


# -- parameter operations --------------------------------------------------
# -- initial-condition report (issue #57) ----------------------------------
def test_commit_result_includes_initial_conditions(session):
    # Every reconciling model op reports each species' IC status; a new species is flagged.
    res = _set_model(session, ["J: B => D; k3 * B;"], justification="B converts to D")
    ics = {r["species"]: r for r in res["initial_conditions"]}
    assert {"A", "B", "D"} <= set(ics)
    # D is an empty pool carried as a literal: no IC parameter, so nothing was "added" for it,
    # but the declared t=0 still governs the simulated state and must be reported as such.
    assert ics["D"]["ic_parameter"] is None and ics["D"]["newly_added"] is False
    assert ics["D"]["declared_t0"] == 0.0
    assert ics["D"]["t0_source"] == "fixed" and ics["D"]["declared_t0_used"] is True
    assert "costs nothing in k" in ics["D"]["note"]


def test_no_parameter_estimate_or_fix_operation():
    # Changing a parameter's estimate status in-loop is deliberately unsupported — no such op.
    assert not hasattr(O, "estimate_parameter")
    assert not hasattr(O, "change_parameter_estimated")
    assert not hasattr(O, "fix_parameter")


def test_change_parameters_scale(session):
    res = O.change_parameters(
        session, [{"parameter_id": "sigma_B", "scale": "log10"}], justification="spans magnitudes"
    )
    assert res["applied"] is True
    assert res["operation"] == "make_changes"
    assert res["references"] == []  # references optional for parameter operations
    recs = {r["parameterId"]: r for r in P.parameters_to_records(session.problem)}
    assert recs["sigma_B"]["parameterScale"] == "log10"


def test_change_parameters_requires_justification(session):
    with pytest.raises(OperationError):
        O.change_parameters(
            session, [{"parameter_id": "sigma_B", "scale": "log10"}], justification=""
        )


def test_change_parameters_empty_list_rejected(session):
    with pytest.raises(OperationError, match="non-empty list"):
        O.change_parameters(session, [], justification="nothing to do")


def test_change_parameters_invalid_scale_is_reverted(session):
    # A_init has lowerBound 0 -> log10 is invalid; the table must be unchanged.
    before = {r["parameterId"]: r for r in P.parameters_to_records(session.problem)}
    with pytest.raises(OperationError):
        O.change_parameters(
            session, [{"parameter_id": "A_init", "scale": "log10"}], justification="x"
        )
    after = {r["parameterId"]: r for r in P.parameters_to_records(session.problem)}
    assert after["A_init"] == before["A_init"]


def test_change_parameters_bounds(session):
    res = O.change_parameters(
        session,
        [{"parameter_id": "k1", "lower_bound": 1e-3, "upper_bound": 1e3}],
        justification="physiological range",
    )
    assert res["applied"] is True
    recs = {r["parameterId"]: r for r in P.parameters_to_records(session.problem)}
    assert float(recs["k1"]["lowerBound"]) == 1e-3
    assert float(recs["k1"]["upperBound"]) == 1e3


def test_change_parameters_one_bound_keeps_the_other(session):
    """A one-sided bound edit was refused with "needs BOTH" (9x in the indra sweep, #86); it keeps
    the bound the table holds on the other side."""
    df = session.problem.parameter_df
    upper, lower = float(df.loc["k1", "upperBound"]), float(df.loc["k2", "lowerBound"])
    res = O.change_parameters(
        session,
        [{"parameter_id": "k1", "lower_bound": 1e-3}, {"parameter_id": "k2", "upper_bound": 10.0}],
        justification="one side each",
    )
    assert res["applied"] is True
    recs = {r["parameterId"]: r for r in P.parameters_to_records(session.problem)}
    assert (float(recs["k1"]["lowerBound"]), float(recs["k1"]["upperBound"])) == (1e-3, upper)
    assert (float(recs["k2"]["lowerBound"]), float(recs["k2"]["upperBound"])) == (lower, 10.0)


def test_a_one_sided_bound_past_the_kept_one_names_the_kept_one(session):
    """The kept bound is one the caller never sent, so a crossing must say where it came from
    rather than report update_parameters' zero-width-box text about it."""
    with pytest.raises(OperationError, match="keeps its current lowerBound 0.0001.*give both"):
        O.change_parameters(
            session, [{"parameter_id": "k1", "upper_bound": 1e-5}], justification="too low"
        )


def test_a_forks_one_sided_bound_keeps_the_forked_versions_other_bound(session):
    """A `base_version` fork filled the other bound from HEAD, handing the fork the abandoned
    branch's widening that restoring the version is there to undo."""
    stored = session.problem.parameter_df.copy()
    stored.loc["k1", "upperBound"] = 10.0  # the forked version's box; HEAD's upper is 100
    entry = {"parameter_id": "k1", "lower_bound": 1e-3, "upper_bound": None, "scale": None}
    O._composed_parameter_edit(session, [entry], stored)(session.problem)
    df = session.problem.parameter_df
    assert (float(df.loc["k1", "lowerBound"]), float(df.loc["k1", "upperBound"])) == (1e-3, 10.0)


def test_change_parameters_names_an_assignment_rule_symbol(session):
    """An analyst bounding `d16_kf_pinned`, a statements binding's rule, got the generic unknown-id
    text on the edit path while get_parameters explained it (#86)."""
    _set_model(session, ["r3: A => ; k3_bound * A;", "k3_bound := 2 * k1;"])
    with pytest.raises(PetabMcpError, match=r"assignment rule of the model.*it reads k1\."):
        O.change_parameters(
            session, [{"parameter_id": "k3_bound", "lower_bound": 1e-3}], justification="x"
        )


def test_change_parameters_names_a_noise_placeholder(session):
    """`noiseParameter1_<obs>` got "Unknown parameter id ... Known ids: [...]" (17x, #86), and the
    agent kept trying to estimate a value the measurement table sets."""
    odf = session.problem.observable_df.copy()
    odf["noiseFormula"] = "noiseParameter1_obs_B"
    odf["noisePlaceholders"] = "noiseParameter1_obs_B"
    session.problem = session.problem.replace(observable_df=odf)
    with pytest.raises(
        PetabMcpError, match="noise placeholder of observable 'obs_B'.*noiseParameters"
    ):
        O.change_parameters(
            session,
            [{"parameter_id": "noiseParameter1_obs_B", "lower_bound": 1e-3, "upper_bound": 1.0}],
            justification="estimate the noise",
        )


def test_change_parameters_batch_applies_bounds_and_scale_atomically(session):
    # One call retunes several parameters at once — the point of the batched editor.
    res = O.change_parameters(
        session,
        [
            {"parameter_id": "k1", "lower_bound": 1e-3, "upper_bound": 1e3},
            {"parameter_id": "sigma_B", "scale": "log10"},
        ],
        justification="coherent retuning",
    )
    assert res["applied"] is True
    recs = {r["parameterId"]: r for r in P.parameters_to_records(session.problem)}
    assert float(recs["k1"]["lowerBound"]) == 1e-3 and float(recs["k1"]["upperBound"]) == 1e3
    assert recs["sigma_B"]["parameterScale"] == "log10"


def test_change_parameters_single_entry_bounds_and_scale(session):
    # ONE entry carrying BOTH a bound change and a scale change (the merged-entry path that
    # a `changes_by_id` payload produces): bounds must apply AND the positive-box check must run
    # against the new scale+bounds. k1 = 0.5 with lb 1e-3 is a valid log10 box.
    res = O.change_parameters(
        session,
        [
            {"parameter_id": "k1", "lower_bound": 1e-3, "upper_bound": 1e3, "scale": "log10"},
        ],
        justification="retune k1 box and scale together",
    )
    assert res["applied"] is True
    recs = {r["parameterId"]: r for r in P.parameters_to_records(session.problem)}
    assert float(recs["k1"]["lowerBound"]) == 1e-3 and float(recs["k1"]["upperBound"]) == 1e3
    assert recs["k1"]["parameterScale"] == "log10"


def test_change_parameters_rejects_unknown_entry_key(session):
    # A misspelled optional key must be REJECTED, not silently dropped (a `scal` typo would
    # otherwise leave the scale unchanged while the call reports success).
    with pytest.raises(OperationError, match="unknown key"):
        O.change_parameters(
            session,
            [{"parameter_id": "k1", "lower_bound": 1e-3, "upper_bound": 1e3, "scal": "log10"}],
            justification="typo'd scale key",
        )


def test_change_parameters_duplicate_id_rejected(session):
    with pytest.raises(OperationError, match="more than one entry"):
        O.change_parameters(
            session,
            [
                {"parameter_id": "k1", "scale": "log10"},
                {"parameter_id": "k1", "lower_bound": 1e-3, "upper_bound": 1e3},
            ],
            justification="combine these",
        )


def test_change_parameters_bad_range_reverts(session):
    with pytest.raises(Exception):
        O.change_parameters(
            session,
            [{"parameter_id": "k1", "lower_bound": 10.0, "upper_bound": 1.0}],
            justification="x",
        )
    recs = {r["parameterId"]: r for r in P.parameters_to_records(session.problem)}
    assert float(recs["k1"]["upperBound"]) != 1.0  # not applied


# -- repeated-widening backstop (issue #56) --------------------------------
def _add_version(session, *, upper, nllh, kind="fit"):
    """Snapshot a real version whose stored parameter table sets k1's UPPER bound to ``upper``
    and whose meta records ``neg_log_likelihood`` (``None`` => a simulate-only fit). The
    guard reconstructs widening history from these stored tables, so this mirrors exactly how a
    real fit / rollback lands on disk."""
    from petab.v2.C import UPPER_BOUND

    df = session.problem.parameter_df.copy(deep=True)
    df.loc["k1", UPPER_BOUND] = upper
    meta = {} if nllh is None else {"neg_log_likelihood": nllh}
    session.provenance.snapshot_version(
        session.problem.model.to_antimony(), df, kind=kind, meta=meta
    )


def _pin_k1_at_upper(session, upper):
    """Point the live table + last fit at ``upper`` so k1 reads as re-pinned at its upper bound."""
    from petab.v2.C import UPPER_BOUND

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
    from petab.v2.C import UPPER_BOUND

    _add_version(session, upper=1e3, nllh=100.0)
    _add_version(session, upper=1e4, nllh=99.9)
    _pin_k1_at_upper(session, 1e4)
    with pytest.raises(OperationError):
        O.change_parameters(
            session,
            [{"parameter_id": "k1", "lower_bound": 1e-4, "upper_bound": 1e5}],
            justification="widen again",
        )
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
    _add_version(session, upper=1e3, nllh=None)  # no neg_log_likelihood recorded
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


def test_widenings_across_fitted_versions_are_counted(session):
    # A widening that reaches a version via any fit snapshot counts toward the backstop exactly
    # like any other; the guard reads stored parameter tables regardless of the version's kind.
    _add_version(session, upper=1e3, nllh=100.0, kind="fit")
    _add_version(session, upper=1e4, nllh=99.9, kind="fit")
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


def test_edit_records_event_but_no_version(session):
    # An edit is fit-free: it logs an `edit` event and marks the change pending a fit, but does
    # NOT snapshot a version (versions are fit snapshots — issue #81).
    before = session.provenance.current_version
    _set_model(session, ["J: B => D; k3 * B;"], justification="x", references=["PMID:1"])
    assert session.provenance.current_version == before  # edit does not snapshot a version
    events = session.provenance.recent_events()
    edit = [e for e in events if e["type"] == "edit"][-1]
    assert edit["operation"] == "make_changes"
    assert edit["justification"] == "x"
    assert edit["references"][0]["input"] == "PMID:1"


def test_a_rejected_edit_never_touches_the_live_problem(session):
    """The edit builds a CANDIDATE and swaps it in only once everything passes, so a rejection
    leaves the live problem object — not merely its contents — identical."""
    before_problem = session.problem
    before_df = session.problem.parameter_df
    before_model = session.problem.model
    before_rows = session.problem.parameter_df.to_dict()

    with pytest.raises(PetabMcpError):
        O.set_model(session, "model m\n  species Z;\n  Z = 1;\nend", justification="no dynamics")

    assert session.problem is before_problem, "the live problem object was replaced"
    assert session.problem.parameter_df is before_df, "the live parameter table was replaced"
    assert session.problem.model is before_model, "the live model was replaced"
    assert session.problem.parameter_df.to_dict() == before_rows


def test_a_successful_edit_swaps_in_a_new_problem_object(session):
    """The complement: a change that passes replaces the reference rather than mutating in place,
    which is what makes the snapshot in `working_state` a valid undo."""
    before_problem = session.problem
    res = O.change_parameters(
        session,
        [{"parameter_id": "k1", "lower_bound": 1e-3, "upper_bound": 1e3}],
        justification="widen",
    )
    assert res["applied"] is True
    assert session.problem is not before_problem
    # nothing is shared with the problem it replaced — a deep copy, so no in-place edit anywhere
    # in the candidate can reach the object `working_state` snapshots for the undo
    assert session.problem.parameter_df is not before_problem.parameter_df
    assert session.problem.model is not before_problem.model


def test_a_failure_after_the_edit_applies_still_leaves_the_live_problem_alone(session, monkeypatch):
    """The candidate is swapped in only once nothing left can raise.

    The provenance write is the last thing that can fail, and `make_changes` returns on
    `applied: false` without reverting — so a swap before it would leave the working model
    carrying a change the caller was told was rejected."""
    before_problem = session.problem

    def _boom(_event):
        raise OSError("provenance write failed")

    monkeypatch.setattr(session.provenance, "append_event", _boom)
    with pytest.raises(OSError):
        O.change_parameters(
            session,
            [{"parameter_id": "k1", "lower_bound": 1e-3, "upper_bound": 1e3}],
            justification="widen",
        )

    assert session.problem is before_problem


def test_a_parameters_only_edit_that_touched_the_model_is_refused(session, monkeypatch):
    """Two things: the live model must be UNTOUCHED (the candidate is a deep copy, so an
    in-place SBML edit cannot reach it), and the edit must be refused (a parameters-only call
    skips the reconcile, so a model change through it would reach a version unchecked).

    The refusal alone is not enough — it fires after `param_edit` has run, so with a shared model
    the mutation had already landed in the working problem."""
    real = O._parameter_edit

    def _also_mutates_the_model(sess, normalized):
        inner = real(sess, normalized)

        def edit(problem):
            out = inner(problem)
            problem.model.sbml_model.getSpecies(0).setId("mutated_in_place")
            return out

        return edit

    before_ids = [
        session.problem.model.sbml_model.getSpecies(i).getId()
        for i in range(session.problem.model.sbml_model.getNumSpecies())
    ]

    monkeypatch.setattr(O, "_parameter_edit", _also_mutates_the_model)
    with pytest.raises(PetabMcpError, match="changed the MODEL"):
        O.change_parameters(
            session,
            [{"parameter_id": "k1", "lower_bound": 1e-3, "upper_bound": 1e3}],
            justification="widen",
        )

    after_ids = [
        session.problem.model.sbml_model.getSpecies(i).getId()
        for i in range(session.problem.model.sbml_model.getNumSpecies())
    ]
    assert after_ids == before_ids, "the rejected edit reached the live model"


def test_the_sigma_widening_warning_reads_an_upgraded_log10_sigma_in_its_own_units():
    """The cap is an sd on the log scale the noise is additive on; an upgraded log10 observable's
    sd is log(10)*sigma, so its sigma reaches the cap ln 10 times sooner than its raw value says."""
    import math

    import pandas as pd

    from petab_mcp import reconcile as R
    from petab_mcp.problem import WorkingProblem

    measurements = pd.DataFrame(
        {
            "observableId": ["o"] * 5,
            "experimentId": [""] * 5,
            "time": [0.0, 1.0, 2.0, 3.0, 4.0],
            "measurement": [1e2, 1e3, 1e4, 1e5, 1e6],
            "noiseParameters": ["sd_o"] * 5,
        }
    )
    # What the upgrade (`problem._from_v1`) makes of a v1 log10 observable.
    observables = pd.DataFrame(
        {
            "observableId": ["o"],
            "observableFormula": ["x"],
            "noiseFormula": ["log(10) * (noiseParameter1_o)"],
            "noiseDistribution": ["log-normal"],
            "noisePlaceholders": ["noiseParameter1_o"],
        }
    ).set_index("observableId")
    parameters = pd.DataFrame(
        {"parameterId": ["sd_o"], "lowerBound": [1e-3], "upperBound": [1e-2], "estimate": [True]}
    ).assign(nominalValue=0.1)
    problem = WorkingProblem(
        observable_df=observables, measurement_df=measurements, parameter_df=parameters
    )
    cap = R.sigma_upper_bound_cap(problem, "sd_o") / math.log(10)  # in log10 units
    assert O._warn_sigma_bound_widening(problem, "sd_o", 1.2 * cap) is not None
    assert O._warn_sigma_bound_widening(problem, "sd_o", 0.8 * cap) is None


def test_rejection_summary_names_the_reasons():
    from petab_mcp.operations import _rejection_summary

    out = _rejection_summary(
        [
            "'RAF_total' ships FIXED (estimate=0) in this problem — a known constant",
            "Extraneous parameter(s) in parameter table: {'pEGFR_Akt', 'pAkt_S6'}",
        ]
    )
    assert "resulting problem is invalid" in out  # bucket substring preserved
    assert "RAF_total" in out and "pAkt_S6" in out  # the actionable detail is IN the message
    assert "more in `issues`" not in out  # nothing elided at 2 issues


def test_rejection_summary_caps_and_says_what_it_elided():
    from petab_mcp.operations import _rejection_summary

    out = _rejection_summary([f"issue number {i}" for i in range(7)])
    assert "issue number 0" in out and "issue number 2" in out
    assert "issue number 3" not in out  # capped at 3
    assert "(+4 more in `issues`)" in out  # a truncated view never looks complete


def test_rejection_summary_collapses_multiline_libsbml_noise():
    from petab_mcp.operations import _rejection_summary

    long_sbml = (
        "libSBML Error (SBML component consistency): The value of a 'symbol' field in any "
        "<initialAssignment>\n   definition cannot also appear as the value of a "
        "'variable' field.\n   " + "x" * 400
    )
    out = _rejection_summary([long_sbml])
    assert "\n" not in out  # one line, not a wall of text
    assert len(out) < 420  # per-issue truncation actually applied


def test_rejection_summary_falls_back_when_there_is_no_error_level_issue():
    from petab_mcp.operations import _rejection_summary

    assert _rejection_summary([]) == "the resulting problem is invalid."
