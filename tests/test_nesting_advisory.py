"""The commit-time nesting check (``petab_mcp.nesting``).

A committed variant whose estimated-parameter count ROSE while its nll got WORSE cannot be
nested in its parent — the ML nll of a nested model cannot rise when a parameter is added — so
the comparison the agent is about to reason from is invalid. The regression case is real: blasi
run ``80ccf20efcbc020dc2becda41d14c62e``, where the v3 -> v4 edit meant to ADD a ``c816`` factor
to ``r5_12_16_8`` instead OVERWROTE the ``c81216`` already in that rate law. v4 gained a
parameter, lost 3.984 nll, and the panel read the worse score as "the model is complete" and
finished — on its only structure-adding experiment.
"""

from __future__ import annotations

from petab_mcp.nesting import nesting_advisory, valid_added_parameter_experiment

# --- the real v3/v4 case ------------------------------------------------------------------
# nll and estimated ids are verbatim from the archived run's versions/v{3,4}/meta.json +
# parameters.tsv. Structural k is 6 -> 7 (a5/a8/a12/a16/c512/c81216 plus, in v4, c816); the
# ESTIMATED count the check compares is 7 -> 8, since `sigma` is free in both.
_V3_NLL = -987.832003765968
_V4_NLL = -983.848189476257  # 3.983814289711 WORSE
_V3_IDS = ["a12", "a16", "a5", "a8", "c512", "c81216", "sigma"]
_V4_IDS = [*_V3_IDS, "c816"]

# The rate laws that mention c81216, verbatim from each version's model.ant (9 references in v3,
# 8 in v4 — `r5_12_16_8` lost its c81216 to the new c816).
_V3_MODEL = """\
  r8_12_16: x_k8k12 -> x_k8k12k16; c81216*a16*x_k8k12 - d*x_k8k12k16;
  r8_16_12: x_k8k16 -> x_k8k12k16; c81216*a12*x_k8k16 - d*x_k8k12k16;
  r12_16_8: x_k12k16 -> x_k8k12k16; c81216*a8*x_k12k16 - d*x_k8k12k16;
  r5_8_12_16: x_k5k8k12 -> x_4ac; c81216*a16*x_k5k8k12 - d*x_4ac;
  r5_8_16_12: x_k5k8k16 -> x_4ac; c81216*a12*x_k5k8k16 - d*x_4ac;
  r5_12_16_8: x_k5k12k16 -> x_4ac; c81216*a8*x_k5k12k16 - d*x_4ac;
  r8_12_16_5: x_k8k12k16 -> x_4ac; c512*c81216*a5*x_k8k12k16 - d*x_4ac;
  c81216 = 1;
  const d, a5, a8, a12, a16, c512, c81216;
"""

_V4_MODEL = """\
  r8_12_16: x_k8k12 -> x_k8k12k16; c81216*c816*a16*x_k8k12 - d*x_k8k12k16;
  r8_16_12: x_k8k16 -> x_k8k12k16; c81216*a12*x_k8k16 - d*x_k8k12k16;
  r12_16_8: x_k12k16 -> x_k8k12k16; c81216*c816*a8*x_k12k16 - d*x_k8k12k16;
  r5_8_12_16: x_k5k8k12 -> x_4ac; c81216*c816*a16*x_k5k8k12 - d*x_4ac;
  r5_8_16_12: x_k5k8k16 -> x_4ac; c81216*a12*x_k5k8k16 - d*x_4ac;
  r5_12_16_8: x_k5k12k16 -> x_4ac; c816*a8*x_k5k12k16 - d*x_4ac;
  r8_12_16_5: x_k8k12k16 -> x_4ac; c512*c81216*c816*a5*x_k8k12k16 - d*x_4ac;
  c81216 = 1;
  c816 = 1;
  const d, a5, a8, a12, a16, c512, c81216, c816;
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


def test_blasi_v3_to_v4_flags_the_displaced_parameter():
    msg = nesting_advisory(_v3(), _v4())
    assert msg is not None
    # names the parent it is invalid against, the k rise, and how much worse the nll got
    assert "v3" in msg and "7->8" in msg and "3.9838" in msg
    # ...and names WHICH parameter was displaced: c81216 stayed ESTIMATED but lost one of its
    # nine rate-law references, so only the reference-count diff can find it.
    assert "c81216 now appears in 8 places, was 9" in msg
    # both causes are on the table — the optimizer is the other way this can happen
    assert "not additive" in msg and "under-converged" in msg
    # and it says what the agent must NOT conclude from it
    assert "does NOT show the added structure fails to help" in msg


def test_blasi_v3_to_v4_is_not_a_valid_added_parameter_experiment():
    assert valid_added_parameter_experiment(_v3(), _v4()) is False


def test_parameter_dropped_from_the_table_is_named_outright():
    # The unambiguous shape: c81216 is present in the parent and ABSENT from the child, so the
    # estimated-id diff alone identifies it — no reference counting needed.
    child = _v4(
        estimated_ids=[i for i in _V4_IDS if i != "c81216"],
        n_estimated=8,
        model=_V4_MODEL.replace("c81216*", "").replace("c81216", ""),
    )
    # keep k rising: the child swapped c81216 for two new factors
    child["estimated_ids"] = [*child["estimated_ids"], "c8"]
    child["n_estimated"] = len(child["estimated_ids"])
    msg = nesting_advisory(_v3(), child)
    assert msg is not None and "c81216 no longer estimated" in msg


def test_no_advisory_when_the_added_parameter_actually_helps():
    # v2 -> v3 in the same run: k rose AND the nll improved by 150.7. Nothing to say.
    v2 = _v3(
        version=2,
        n_estimated=6,
        neg_log_likelihood=-837.1807567142856,
        estimated_ids=[i for i in _V3_IDS if i != "c81216"],
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
    # ...and it is remembered for the finish gate, which reads it through fit_summary.
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
    """The lead redoing the experiment properly must stop being told it has none."""
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
    """`fit_summary` is what the harness finish gate polls; the key must be ABSENT (not False /
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
    """The flag is what the finish gate reads, and the botched variant is usually rolled back
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
        estimated_ids=[i for i in _V3_IDS if i != "c81216"],
        model=_lazy(_V3_MODEL),
    )
    assert nesting_advisory(v2, _v3(model=_lazy(_V3_MODEL))) is None
    assert reads == []

    # firing comparison: resolved, and still names the displaced parameter
    msg = nesting_advisory(_v3(model=_lazy(_V3_MODEL)), _v4(model=_lazy(_V4_MODEL)))
    assert "c81216 now appears in 8 places, was 9" in msg
    assert len(reads) == 2


def test_advisory_names_the_parent_generically_without_a_version():
    msg = nesting_advisory(_v3(version=None), _v4())
    assert msg is not None and "vNone" not in msg and "vs its parent" in msg


def test_a_new_best_supersedes_the_outstanding_flag(session):
    """A run that pivots to SIMPLIFYING after a botched experiment never raises k again, so the
    added-parameter clearing path alone would pin the note to every remaining finish ARM. A later
    new best is what the finish would actually rest on, so it supersedes the stale comparison."""
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
