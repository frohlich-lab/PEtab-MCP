"""The commit-time nesting check (``petab_mcp.nesting``).

A committed variant whose estimated-parameter count ROSE while its nll got WORSE cannot be
nested in its parent — the ML nll of a nested model cannot rise when a parameter is added — so
the comparison the agent is about to reason from is invalid. The regression case is real: a
session whose v3 -> v4 edit meant to ADD a ``c2`` factor to ``r_dbc_a`` instead OVERWROTE
the ``c1`` already in that rate law. v4 gained a parameter, lost 3.984 nll, and the agent read
the worse score as "the model is complete" and finished — on its only structure-adding
experiment.
"""

from __future__ import annotations

from petab_mcp.nesting import nesting_advisory, valid_added_parameter_experiment

# --- the real v3/v4 case ------------------------------------------------------------------
# nll values are verbatim from that session's versions/v{3,4}/meta.json, the estimated ids from
# its parameters.tsv under placeholder names. Structural k is 6 -> 7 (a1/a2/a3/a4/c3/c1 plus, in
# v4, c2); the ESTIMATED count the check compares is 7 -> 8, since `sigma` is free in both.
_V3_NLL = -987.832003765968
_V4_NLL = -983.848189476257  # 3.983814289711 WORSE
_V3_IDS = ["a3", "a4", "a1", "a2", "c3", "c1", "sigma"]
_V4_IDS = [*_V3_IDS, "c2"]

# The rate laws that mention c1, from each version's model.ant under the same placeholder names
# (9 references in v3, 8 in v4 — `r_dbc_a` lost its c1 to the new c2).
_V3_MODEL = """\
  r_ab_c: AB -> ABC; c1*a4*AB - d*ABC;
  r_ac_b: AC -> ABC; c1*a3*AC - d*ABC;
  r_bc_a: BC -> ABC; c1*a2*BC - d*ABC;
  r_dab_c: DAB -> DABC; c1*a4*DAB - d*DABC;
  r_dac_b: DAC -> DABC; c1*a3*DAC - d*DABC;
  r_dbc_a: DBC -> DABC; c1*a2*DBC - d*DABC;
  r_abc_d: ABC -> DABC; c3*c1*a1*ABC - d*DABC;
  c1 = 1;
  const d, a1, a2, a3, a4, c3, c1;
"""

_V4_MODEL = """\
  r_ab_c: AB -> ABC; c1*c2*a4*AB - d*ABC;
  r_ac_b: AC -> ABC; c1*a3*AC - d*ABC;
  r_bc_a: BC -> ABC; c1*c2*a2*BC - d*ABC;
  r_dab_c: DAB -> DABC; c1*c2*a4*DAB - d*DABC;
  r_dac_b: DAC -> DABC; c1*a3*DAC - d*DABC;
  r_dbc_a: DBC -> DABC; c2*a2*DBC - d*DABC;
  r_abc_d: ABC -> DABC; c3*c1*c2*a1*ABC - d*DABC;
  c1 = 1;
  c2 = 1;
  const d, a1, a2, a3, a4, c3, c1, c2;
"""


def _v3(**over):
    return {
        "version": 3,
        "n_estimated": 7,
        "neg_log_likelihood": _V3_NLL,
        "estimated_ids": _V3_IDS,
        "model": _V3_MODEL,
        **over,
    }


def _v4(**over):
    return {
        "version": 4,
        "n_estimated": 8,
        "neg_log_likelihood": _V4_NLL,
        "estimated_ids": _V4_IDS,
        "model": _V4_MODEL,
        **over,
    }


def test_an_overwritten_factor_flags_the_displaced_parameter():
    msg = nesting_advisory(_v3(), _v4())
    assert msg is not None
    # names the parent it is invalid against, the k rise, and how much worse the nll got
    assert "v3" in msg and "7->8" in msg and "3.9838" in msg
    # ...and names WHICH parameter was displaced: c1 stayed ESTIMATED but lost one of its
    # nine rate-law references, so only the reference-count diff can find it.
    assert "c1 now appears in 8 places, was 9" in msg
    # both causes are on the table — the optimizer is the other way this can happen
    assert "not additive" in msg and "under-converged" in msg
    # and it says what the agent must NOT conclude from it
    assert "does NOT show the added structure fails to help" in msg


def test_an_overwritten_factor_is_not_a_valid_added_parameter_experiment():
    assert valid_added_parameter_experiment(_v3(), _v4()) is False


def test_parameter_dropped_from_the_table_is_named_outright():
    # The unambiguous shape: c1 is present in the parent and ABSENT from the child, so the
    # estimated-id diff alone identifies it — no reference counting needed.
    child = _v4(
        estimated_ids=[i for i in _V4_IDS if i != "c1"],
        n_estimated=8,
        model=_V4_MODEL.replace("c1*", "").replace("c1", ""),
    )
    # keep k rising: the child swapped c1 for two new factors
    child["estimated_ids"] = [*child["estimated_ids"], "c4"]
    child["n_estimated"] = len(child["estimated_ids"])
    msg = nesting_advisory(_v3(), child)
    assert msg is not None and "c1 no longer estimated" in msg


def test_no_advisory_when_the_added_parameter_actually_helps():
    # v2 -> v3 in the same session: k rose AND the nll improved by 150.7. Nothing to say.
    v2 = _v3(
        version=2,
        n_estimated=6,
        neg_log_likelihood=-837.1807567142856,
        estimated_ids=[i for i in _V3_IDS if i != "c1"],
    )
    assert nesting_advisory(v2, _v3()) is None
    assert valid_added_parameter_experiment(v2, _v3()) is True


def test_no_advisory_when_k_did_not_rise():
    # A SIMPLER model fitting worse is ordinary and expected — the invariant only binds one way.
    assert nesting_advisory(_v4(), _v3()) is None
    assert valid_added_parameter_experiment(_v4(), _v3()) is False


def test_no_advisory_for_a_noise_floor_worsening():
    # k rose but the nll is worse by 1e-9 — the same optimum re-reported, not a violation.
    assert nesting_advisory(_v3(), _v4(neg_log_likelihood=_V3_NLL + 1e-9)) is None


def test_no_advisory_when_the_parent_was_never_fitted():
    # v0/initial and legacy versions carry no nll: no comparison, so no claim about one.
    assert nesting_advisory(_v3(neg_log_likelihood=None), _v4()) is None
    assert nesting_advisory(_v3(n_estimated=None), _v4()) is None
    assert nesting_advisory(_v3(neg_log_likelihood=float("nan")), _v4()) is None


def test_cause_left_open_when_no_parameter_can_be_named():
    # Nothing dropped and no reference count fell (the models are unavailable): still advise, but
    # point at diff_versions instead of guessing a culprit.
    msg = nesting_advisory(_v3(model=None), _v4(model=None))
    assert msg is not None
    assert "diff_versions" in msg and "under-converged" in msg


# --- the session wiring: read the parent from the version store ----------------------------
def _snapshot_parent(session, nll):
    """Snapshot the current working state as a fitted parent version carrying ``nll``."""
    return session._snapshot_version(meta={"neg_log_likelihood": nll}, sim=None)


def test_session_flags_a_worse_fit_that_freed_another_parameter(session):
    """The commit path reads the parent's nll (meta.json), its estimated ids (parameters.tsv)
    and its Antimony (model.ant) straight out of the version store, so the check works on the
    real lineage rather than on whatever the caller passes."""
    from petab.v2.C import ESTIMATE

    _snapshot_parent(session, -100.0)
    session.problem.parameter_df.loc["sigma_B", ESTIMATE] = True  # k: 2 -> 3
    advisory = session._check_nesting_vs_parent(
        {"neg_log_likelihood": -90.0}, session.model.to_antimony()
    )
    assert advisory is not None and "2->3" in advisory and "v" in advisory
    # ...and it is remembered for a client's finish gate, which reads it through fit_summary.
    assert session.invalid_comparison == advisory


def test_session_silent_on_an_ordinary_improving_commit(session):
    from petab.v2.C import ESTIMATE

    _snapshot_parent(session, -100.0)
    session.problem.parameter_df.loc["sigma_B", ESTIMATE] = True
    assert (
        session._check_nesting_vs_parent(
            {"neg_log_likelihood": -140.0}, session.model.to_antimony()
        )
        is None
    )
    assert session.invalid_comparison is None


def test_a_valid_added_parameter_commit_clears_the_outstanding_flag(session):
    """An agent redoing the experiment properly must stop being told it has none."""
    from petab.v2.C import ESTIMATE

    _snapshot_parent(session, -100.0)
    session.problem.parameter_df.loc["sigma_B", ESTIMATE] = True
    assert (
        session._check_nesting_vs_parent({"neg_log_likelihood": -90.0}, session.model.to_antimony())
        is not None
    )
    assert session.invalid_comparison is not None
    # a second, genuinely additive experiment: k rises again and the nll improves
    _snapshot_parent(session, -90.0)
    session.problem.parameter_df.loc["A_init", ESTIMATE] = True
    assert (
        session._check_nesting_vs_parent({"neg_log_likelihood": -95.0}, session.model.to_antimony())
        is None
    )
    assert session.invalid_comparison is None


def test_no_parent_version_yields_no_advisory(session):
    # v0 exists but was never fitted (no nll in its meta) -> nothing to compare against.
    from petab.v2.C import ESTIMATE

    session.problem.parameter_df.loc["sigma_B", ESTIMATE] = True
    assert (
        session._check_nesting_vs_parent({"neg_log_likelihood": -90.0}, session.model.to_antimony())
        is None
    )


def test_fit_summary_carries_the_outstanding_flag_for_the_finish_gate(session):
    """`fit_summary` is what a client's finish gate polls; the key must be ABSENT (not False /
    None) when there is nothing outstanding, so an omitted key reads as "no invalid comparison"."""
    from petab_mcp import fitting

    session.simulation = {
        "status": "fit",
        "neg_log_likelihood": -100.0,
        "chi2": 1.0,
        "n_data_points": 10,
        "fitted_parameters": {},
        "records": [{"observable_id": "obs_B"}],
    }
    assert "invalid_comparison" not in fitting.fit_summary(session, engine=None)
    session.invalid_comparison = "INVALID COMPARISON vs v3: ..."
    assert (
        fitting.fit_summary(session, engine=None)["invalid_comparison"]
        == "INVALID COMPARISON vs v3: ..."
    )


def test_the_outstanding_flag_survives_a_durable_resume(session, tmp_path):
    """The flag is what a client's finish gate reads, and the botched variant is usually rolled back
    immediately — so a process restart must not be a way to launder it away. Rebuilt from the
    persisted fit events, not from the per-commit advisories: a valid additive experiment clears
    the flag without writing an advisory of its own."""
    from petab.v2.C import ESTIMATE
    from petab_mcp.references import StubReferenceResolver
    from petab_mcp.session import PetabSession

    _snapshot_parent(session, -100.0)
    session.problem.parameter_df.loc["sigma_B", ESTIMATE] = True
    advisory = session._check_nesting_vs_parent(
        {"neg_log_likelihood": -90.0}, session.model.to_antimony()
    )
    assert advisory is not None
    # what session.fit records on every commit
    session.provenance.append_event(
        {"type": "fit", "version": 1, "invalid_comparison": session.invalid_comparison}
    )

    resumed = PetabSession(
        session.yaml_path,
        provenance_dir=str(session.provenance.root),
        resolver=StubReferenceResolver(),
    )
    assert resumed.invalid_comparison == advisory

    # ...and a later commit that cleared it stays cleared across the restart
    session.provenance.append_event({"type": "fit", "version": 2, "invalid_comparison": None})
    again = PetabSession(
        session.yaml_path,
        provenance_dir=str(session.provenance.root),
        resolver=StubReferenceResolver(),
    )
    assert again.invalid_comparison is None


def test_the_model_is_only_read_when_the_advisory_fires():
    """The session hands the parent's Antimony as a lazy callable — it lives on disk, and the
    overwhelming majority of commits improve the nll and never need it to name a culprit."""
    reads = []

    def _lazy(text):
        def read():
            reads.append(text)
            return text

        return read

    # clean comparison: the model is never resolved
    v2 = _v3(
        version=2,
        n_estimated=6,
        neg_log_likelihood=-837.1807567142856,
        estimated_ids=[i for i in _V3_IDS if i != "c1"],
        model=_lazy(_V3_MODEL),
    )
    assert nesting_advisory(v2, _v3(model=_lazy(_V3_MODEL))) is None
    assert reads == []

    # firing comparison: resolved, and still names the displaced parameter
    msg = nesting_advisory(_v3(model=_lazy(_V3_MODEL)), _v4(model=_lazy(_V4_MODEL)))
    assert "c1 now appears in 8 places, was 9" in msg
    assert len(reads) == 2


def test_advisory_names_the_parent_generically_without_a_version():
    msg = nesting_advisory(_v3(version=None), _v4())
    assert msg is not None and "vNone" not in msg and "vs its parent" in msg


def test_a_new_best_supersedes_the_outstanding_flag(session):
    """A session that pivots to SIMPLIFYING after a botched experiment never raises k again, so the
    added-parameter clearing path alone would pin the note to every later finish. A later new best
    is what the finish would actually rest on, so it supersedes the stale comparison."""
    from petab.v2.C import ESTIMATE

    _snapshot_parent(session, -100.0)
    session.problem.parameter_df.loc["sigma_B", ESTIMATE] = True
    assert (
        session._check_nesting_vs_parent({"neg_log_likelihood": -90.0}, session.model.to_antimony())
        is not None
    )
    assert session.invalid_comparison is not None

    # a later commit that does NOT add parameters, but is a new best
    _snapshot_parent(session, -90.0)
    assert (
        session._check_nesting_vs_parent(
            {"neg_log_likelihood": -120.0}, session.model.to_antimony(), is_new_best=True
        )
        is None
    )
    assert session.invalid_comparison is None
