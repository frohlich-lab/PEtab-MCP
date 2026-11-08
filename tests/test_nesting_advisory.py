"""The commit-time nesting check (``petab_mcp.nesting``).

A committed variant whose estimated-parameter count ROSE while its nll got WORSE is an INVALID
COMPARISON only when it is nested in its parent: the edit only added reactions and rules, every
parent parameter is still estimated, and every parent fitted value lies inside the child's bounds.
Otherwise the line names why it is not nested and gives no re-fit advice. #263: in sweep 63372750
the advisory fired on removed reactions, rewritten rate laws and narrowed bounds alike, and agents
re-fitted each with 200-250 starts for up to 99 minutes, to the same nll.
"""

from __future__ import annotations

from petab_mcp.nesting import nesting_advisory, valid_added_parameter_experiment
from petab_mcp.progress_signals import render

# --- a lun-shaped parent and its additive child ---------------------------------------------
# Canonical libantimony layout: rules and reactions each on a line of their own.
_PARENT_MODEL = """\
  // Assignment Rules:
  Act := kact*ERK;

  // Reactions:
  R_on: R -> RL; kon*R*L;
  R_off: RL -> R; koff*RL;

  // Variable initializations:
  kon = 100;
  koff = 0;
"""
_ADDITIVE_MODEL = _PARENT_MODEL.replace(
    "  R_off: RL -> R; koff*RL;\n",
    "  R_off: RL -> R; koff*RL;\n  R_int: RL -> RLi; kint*RL;\n",
).replace("  koff = 0;\n", "  koff = 0;\n  kint = 1;\n")
_PARENT_IDS = ["kact", "koff", "kon"]
_CHILD_IDS = [*_PARENT_IDS, "kint"]


def _parent(**over):
    return {
        "version": 12,
        "n_estimated": len(_PARENT_IDS),
        "neg_log_likelihood": -50.0,
        "estimated_ids": _PARENT_IDS,
        "values": {"kact": 1.0, "koff": 0.0, "kon": 100.0},
        "model": _PARENT_MODEL,
        **over,
    }


def _child(**over):
    return {
        "n_estimated": len(_CHILD_IDS),
        "neg_log_likelihood": -45.0,  # 5 WORSE
        "estimated_ids": _CHILD_IDS,
        "bounds": {pid: (0.0, 1000.0) for pid in _CHILD_IDS},
        "model": _ADDITIVE_MODEL,
        **over,
    }


def _not_nested(parent, child) -> str:
    """The not-nested line, asserting it is one: no INVALID COMPARISON and no re-fit advice."""
    found = nesting_advisory(parent, child)
    assert found is not None
    line, invalid = render(found[0], True), found[1]
    assert invalid is False
    assert "INVALID COMPARISON" not in line and "more starts" not in line
    return line


def test_a_purely_additive_edit_that_fits_worse_is_an_invalid_comparison():
    # The one case the advisory exists for: R_int added, nothing else touched, the parent's
    # values inside the box. A nested child cannot fit worse, so the fit under-converged.
    line, invalid = nesting_advisory(_parent(), _child())
    line = render(line, True)
    assert invalid is True
    assert line.startswith("INVALID COMPARISON vs v12") and "3->4" in line and "5" in line
    assert "more starts" in line and "does NOT show the added structure fails to help" in line


def test_a_removed_reaction_is_not_nested():
    # lun 14 turn 9: R_off removed; the agent re-fitted with 250 starts for 61 min, same nll.
    child = _child(model=_ADDITIVE_MODEL.replace("  R_off: RL -> R; koff*RL;\n", ""))
    line = _not_nested(_parent(), child)
    assert line.startswith("vs v12, not nested: reaction R_off removed.")
    assert "3->4" in line and "add only that structure to v12" in line


def test_a_rewritten_rate_law_is_not_nested():
    # lun 12 turns 16-17: two rate laws rewritten; re-fitted with 250 starts for 99 min.
    child = _child(model=_ADDITIVE_MODEL.replace("kon*R*L;", "kon*R*L/(1 + Act);"))
    assert "not nested: reaction R_on rewritten." in _not_nested(_parent(), child)


def test_a_rewritten_rule_is_not_nested():
    child = _child(model=_ADDITIVE_MODEL.replace("Act := kact*ERK;", "Act := kact*ERK*RSK;"))
    assert "not nested: rule Act rewritten." in _not_nested(_parent(), child)


def test_a_rate_law_extended_by_a_factor_is_rewritten():
    # The case the advisory was first written for: v4 meant to add a c2 factor to r_dbc_a's rate
    # law and overwrote its c1 instead. Multiplying by c2 elsewhere rewrites a rate law too, and
    # the diff cannot see whether the box holds c2's neutral value, so it is not nested.
    parent = """\
  r_ab_c: AB -> ABC; c1*a4*AB - d*ABC;
  r_dbc_a: DBC -> DABC; c1*a2*DBC - d*DABC;
"""
    child = """\
  r_ab_c: AB -> ABC; c1*c2*a4*AB - d*ABC;
  r_dbc_a: DBC -> DABC; c2*a2*DBC - d*DABC;
"""
    line = _not_nested(_parent(model=parent), _child(model=child))
    assert "not nested: reaction r_ab_c rewritten." in line


def test_a_parameter_no_longer_estimated_is_not_nested():
    # koff fixed while two new parameters keep k rising: the child no longer contains the parent.
    ids = ["kact", "kon", "kint", "kint2"]
    child = _child(estimated_ids=ids, n_estimated=len(ids))
    assert "not nested: koff no longer estimated." in _not_nested(_parent(), child)


def test_an_upper_bound_below_the_parents_value_is_not_nested():
    # lun 14 turns 11-12: a reaction added, but the parent's 100 cut to at most 5.
    child = _child(bounds={**_child()["bounds"], "kon": (1e-3, 5.0)})
    line = _not_nested(_parent(), child)
    assert "not nested: kon's upper bound 5 excludes v12's fitted 100." in line


def test_a_lower_bound_above_the_parents_value_is_not_nested():
    # fontaine 9 turns 29-30: new lower bounds of 0.01 excluded the parent's 0.
    child = _child(bounds={**_child()["bounds"], "koff": (0.01, 10.0)})
    line = _not_nested(_parent(), child)
    assert "not nested: koff's lower bound 0.01 excludes v12's fitted 0." in line


def test_a_parent_value_on_the_childs_bound_is_inside_it():
    child = _child(bounds={**_child()["bounds"], "kon": (1e-3, 100.0 * (1 - 1e-12))})
    assert nesting_advisory(_parent(), child)[1] is True


def test_the_first_reason_found_is_named():
    # Structure, then estimation, then bounds.
    ids = ["kact", "kon", "kint", "kint2"]
    dropped_and_narrowed = _child(
        estimated_ids=ids, n_estimated=len(ids), bounds={pid: (1e-3, 5.0) for pid in ids}
    )
    assert "not nested: koff no longer estimated." in _not_nested(_parent(), dropped_and_narrowed)
    removed_too = {
        **dropped_and_narrowed,
        "model": _ADDITIVE_MODEL.replace("  R_off: RL -> R; koff*RL;\n", ""),
    }
    assert "not nested: reaction R_off removed." in _not_nested(_parent(), removed_too)


def test_a_moved_or_renumbered_reaction_is_kept():
    # A rule-built model renumbers its reactions when one is added; the text is what counts.
    parent = "  r0: A -> B; k1*A;\n  r1: B -> C; k2*B;\n"
    child = "  r0: X -> A; k0*X;\n  r1: A -> B; k1*A;\n  r2: B -> C; k2*B;\n"
    assert nesting_advisory(_parent(model=parent), _child(model=child))[1] is True


def test_no_advisory_when_the_added_parameter_actually_helps():
    child = _child(neg_log_likelihood=-80.0)
    assert nesting_advisory(_parent(), child) is None
    assert valid_added_parameter_experiment(_parent(), child) is True


def test_no_advisory_when_k_did_not_rise():
    # A SIMPLER model fitting worse is ordinary and expected — the invariant only binds one way.
    simpler = _parent(n_estimated=2, estimated_ids=["kact", "kon"], neg_log_likelihood=-45.0)
    assert nesting_advisory(_parent(), simpler) is None
    assert valid_added_parameter_experiment(_parent(), simpler) is False


def test_no_advisory_for_a_noise_floor_worsening():
    # k rose but the nll is worse by 1e-9 — the same optimum re-reported, not a violation.
    assert nesting_advisory(_parent(), _child(neg_log_likelihood=-50.0 + 1e-9)) is None


def test_no_advisory_when_the_parent_was_never_fitted():
    # v0/initial versions carry no nll: no comparison, so no claim about one.
    assert nesting_advisory(_parent(neg_log_likelihood=None), _child()) is None
    assert nesting_advisory(_parent(n_estimated=None), _child()) is None
    assert nesting_advisory(_parent(neg_log_likelihood=float("nan")), _child()) is None


def test_no_advisory_without_both_models():
    # Nestedness cannot be read without the diff, so no claim either way.
    assert nesting_advisory(_parent(model=None), _child()) is None
    assert nesting_advisory(_parent(), _child(model=lambda: "")) is None


def test_the_model_is_only_read_once_k_rose_and_the_nll_got_worse():
    """The session hands the parent's Antimony as a lazy callable — it lives on disk, and the
    overwhelming majority of commits improve the nll and never need it."""
    reads = []

    def _lazy(text):
        def read():
            reads.append(text)
            return text

        return read

    improving = _child(neg_log_likelihood=-80.0, model=_lazy(_ADDITIVE_MODEL))
    assert nesting_advisory(_parent(model=_lazy(_PARENT_MODEL)), improving) is None
    assert reads == []

    worse = _child(model=_lazy(_ADDITIVE_MODEL))
    assert nesting_advisory(_parent(model=_lazy(_PARENT_MODEL)), worse)[1] is True
    assert len(reads) == 2


def test_advisory_names_the_parent_generically_without_a_version():
    line, _ = nesting_advisory(_parent(version=None), _child())
    assert "vNone" not in line and "vs its parent" in line


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


def test_session_leaves_the_flag_unset_on_a_not_nested_commit(session):
    """A client's finish gate repeats the re-fit advice for whatever `invalid_comparison` holds,
    so a not-nested child must not set it: the child's k1 box excludes the parent's fitted 0.5."""
    from petab.v2.C import ESTIMATE, UPPER_BOUND

    _snapshot_parent(session, -100.0)
    session.problem.parameter_df.loc["sigma_B", ESTIMATE] = True
    session.problem.parameter_df.loc["k1", UPPER_BOUND] = 0.1
    advisory = session._check_nesting_vs_parent(
        {"neg_log_likelihood": -90.0}, session.model.to_antimony()
    )
    assert advisory is not None and "not nested: k1's upper bound 0.1 excludes" in advisory
    assert session.invalid_comparison is None


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
