"""Structure-level de-duplication — issue #59.

Covers the pure pieces (structure fingerprint, estimation-setup signature, the `decide`
state machine) without AMICI. Since issue #81 relocated the de-dup TRIGGER from ``set_model``
(a now fit-free edit) to the ``fit`` tool, the end-to-end skip/warn/terminate behaviour is
exercised against a real fit in ``test_fit.py``; here we only check that a re-author is a plain
fit-free edit (no dedup at edit time) and that ``restore_version`` still goes back cheaply.
"""

from __future__ import annotations

import pytest

from petab_mcp import dedup as D
from petab_mcp import operations as O
from petab_mcp import parameters as P
from petab.v2.C import ESTIMATE, NOMINAL_VALUE, UPPER_BOUND
from petab.v2.models.sbml_model import SbmlModel

# Reversible A<->B (the toy structure) and a one-reaction variant that drops r2/k2.
REVERSIBLE = """model conversion_reaction
  compartment compartment_ = 1;
  species A in compartment_;
  species B in compartment_;
  A = A_init;
  B = B_init;
  A_init = 1.0;
  B_init = 0.0;
  k1 = 0.25;
  k2 = 0.05;
  r1: A => B; k1 * A;
  r2: B => A; k2 * B;
end
"""

# Same STRUCTURE as REVERSIBLE, only the starting guesses differ (the fit ignores them).
REVERSIBLE_OTHER_VALUES = REVERSIBLE.replace("k1 = 0.25", "k1 = 3.0").replace(
    "k2 = 0.05", "k2 = 9.9"
)

ONE_REACTION = """model conversion_reaction
  compartment compartment_ = 1;
  species A in compartment_;
  species B in compartment_;
  A = A_init;
  B = B_init;
  A_init = 1.0;
  B_init = 0.0;
  k1 = 0.25;
  r1: A => B; k1 * A;
end
"""


def _sm(antimony: str):
    return SbmlModel.from_antimony(antimony, model_id=None).sbml_model


# -- structure_fingerprint --------------------------------------------------
def test_structure_fingerprint_ignores_nominal_values(session):
    """Two authorings of the same structure with different starting guesses fingerprint the same —
    the property that lets a re-author be recognised despite the fit overwriting nominal values."""
    df = session.problem.parameter_df
    a = P.structure_fingerprint(_sm(REVERSIBLE), df)
    b = P.structure_fingerprint(_sm(REVERSIBLE_OTHER_VALUES), df)
    assert a == b


def test_structure_fingerprint_changes_with_structure(session):
    df = session.problem.parameter_df
    assert P.structure_fingerprint(_sm(REVERSIBLE), df) != P.structure_fingerprint(
        _sm(ONE_REACTION), df
    )


def test_structure_fingerprint_changes_with_estimated_set(session):
    """Fixing a parameter (estimate 1 -> 0) is a structural change for fingerprint purposes."""
    df = session.problem.parameter_df.copy(deep=True)
    sm = _sm(REVERSIBLE)
    base = P.structure_fingerprint(sm, df)
    df.loc["k2", ESTIMATE] = False
    assert P.structure_fingerprint(sm, df) != base


def _with(*lines: str, model: str = REVERSIBLE) -> str:
    return model.replace("end\n", "".join(f"  {line}\n" for line in lines) + "end\n")


_FUNCTION = "function f(x)\n  x / (1 + x)\nend\n"

# Pairs of models whose ONLY difference is one equation the reactions do not show: each changes
# the trajectory or what is measured, so neither may pass as the other's duplicate.
_UNSEEN_BY_THE_REACTIONS = {
    # every rate on the statements route: a rule the rate law multiplies
    "assignment rule": (
        _with("k_eff := k1;").replace("k1 * A", "k_eff * A"),
        _with("k_eff := k1 / (1 + A);").replace("k1 * A", "k_eff * A"),
    ),
    "readout": (_with("readout_B := B;"), _with("readout_B := A + B;")),
    "rate rule": (_with("C' = k1 * A;", "C = 0;"), _with("C' = k2 * B;", "C = 0;")),
    "initial assignment": (REVERSIBLE, REVERSIBLE.replace("A = A_init;", "A = 2 * A_init;")),
    "event": (_with("E1: at time > 5: A = 0;"), _with("E1: at time > 6: A = 0;")),
    "boundary species": (REVERSIBLE, REVERSIBLE.replace("r1: A => B", "r1: $A => B")),
    "function body": (
        _FUNCTION + REVERSIBLE.replace("k1 * A", "k1 * f(A)"),
        _FUNCTION.replace("1 + x", "2 + x") + REVERSIBLE.replace("k1 * A", "k1 * f(A)"),
    ),
}


@pytest.mark.parametrize("pair", _UNSEEN_BY_THE_REACTIONS.values(), ids=_UNSEEN_BY_THE_REACTIONS)
def test_structure_fingerprint_sees_every_equation(session, pair):
    """REGRESSION: hashing the reactions alone called a changed rate or readout a duplicate, so
    the change was refused and never fitted."""
    df = session.problem.parameter_df
    a, b = (P.structure_fingerprint(_sm(m), df) for m in pair)
    assert a != b


@pytest.mark.parametrize("pair", _UNSEEN_BY_THE_REACTIONS.values(), ids=_UNSEEN_BY_THE_REACTIONS)
def test_structure_fingerprint_survives_a_render_round_trip(session, pair):
    """A version is stored as its Antimony render and restored by parsing it back; a fingerprint
    that moved on that round trip would call a restored model a different structure."""
    df = session.problem.parameter_df
    for antimony in pair:
        model = SbmlModel.from_antimony(antimony, model_id=None)
        again = SbmlModel.from_antimony(model.to_antimony(), model_id=None)
        assert P.structure_fingerprint(model.sbml_model, df) == P.structure_fingerprint(
            again.sbml_model, df
        )


def test_structure_fingerprint_ignores_starting_values_inside_rules(session):
    """The added equations keep the fingerprint's blindness to values: a rule re-authored under
    other starting guesses is still the same structure."""
    df = session.problem.parameter_df
    ruled = _with("k_eff := k1 / (1 + A);").replace("k1 * A", "k_eff * A")
    other = ruled.replace("k1 = 0.25", "k1 = 3.0")
    assert P.structure_fingerprint(_sm(ruled), df) == P.structure_fingerprint(_sm(other), df)


# -- estimation_setup_signature ---------------------------------------------
def test_setup_signature_ignores_estimated_nominal_but_not_bounds(session):
    df = session.problem.parameter_df.copy(deep=True)
    base = D.estimation_setup_signature(df)
    # Changing an ESTIMATED parameter's starting value must not change the signature...
    df2 = df.copy(deep=True)
    df2.loc["k1", NOMINAL_VALUE] = float(df2.loc["k1", NOMINAL_VALUE]) * 7 + 1
    assert D.estimation_setup_signature(df2) == base
    # ...but changing its bounds must.
    df3 = df.copy(deep=True)
    df3.loc["k1", UPPER_BOUND] = float(df3.loc["k1", UPPER_BOUND]) * 10
    assert D.estimation_setup_signature(df3) != base


# -- reauthor_key: identical re-author identity (structure + setup) ---------
def test_reauthor_key_distinguishes_different_bounds(session):
    """The loop-guard key changes when bounds change, so re-authoring the same structure with
    different bounds is NOT counted as an identical re-author (addresses PR #70 review)."""
    df = session.problem.parameter_df
    sh = P.structure_fingerprint(_sm(REVERSIBLE), df)
    df2 = df.copy(deep=True)
    df2.loc["k1", UPPER_BOUND] = float(df2.loc["k1", UPPER_BOUND]) * 10
    assert D.reauthor_key(sh, df) != D.reauthor_key(sh, df2)
    # ...but it is stable for the same structure + same setup (an identical re-author).
    assert D.reauthor_key(sh, df) == D.reauthor_key(sh, df.copy(deep=True))


# -- decide() (pure) --------------------------------------------------------
SETUP, NEW_SETUP = ("bounds",), ("wider bounds",)


def _reg(*entries, setup=SETUP, fitted=True):
    """Build a registry from (version, structure_hash, score) tuples, each fitted with 2 starts
    under ``setup``."""
    return [
        {
            "version": v,
            "structure_hash": s,
            "setup": setup,
            "fitted": fitted,
            "n_starts": 2,
            "neg_log_likelihood": None,
            "score": b,
        }
        for v, s, b in entries
    ]


def _decide(registry, *, head_version, setup=SETUP, starts=2, prior_attempts=1):
    """``decide`` for a candidate of structure ``X``."""
    return D.decide(
        structure_hash="X",
        setup=setup,
        starts=starts,
        head_version=head_version,
        registry=registry,
        prior_attempts=prior_attempts,
    )


def test_decide_proceeds_on_fresh_structure():
    d = _decide(_reg((0, "head", 1.0)), head_version=0, prior_attempts=0)
    assert d["action"] == "proceed"
    assert d["warnings"] == []


def test_decide_skips_identical_head():
    d = _decide(_reg((3, "X", 1.0)), head_version=3)
    assert d["action"] == "skip"
    assert d["duplicate_of"] == 3
    assert "kept and scored" in d["message"]


def test_decide_no_skip_when_setup_differs():
    # Same structure as HEAD but a bound changed — a real edit, not a no-op.
    d = _decide(_reg((3, "X", 1.0)), setup=NEW_SETUP, head_version=3)
    assert d["action"] == "proceed"


def test_decide_no_skip_against_an_unfitted_version():
    """A simulate-only version records a neg_log_likelihood too; standing in for a real fit would
    return its un-optimized metrics."""
    d = _decide(_reg((3, "X", 1.0), fitted=False), head_version=3)
    assert d["action"] == "proceed"


def test_decide_terminates_after_repeated_reauthors():
    d = _decide(_reg((5, "Y", 1.0)), head_version=5, prior_attempts=D.MAX_IDENTICAL_REAUTHORS)
    assert d["action"] == "terminate"
    assert "kept and scored" in d["message"]


def test_decide_refuses_an_earlier_version_with_the_same_setup():
    """REGRESSION: an earlier version's exact model, re-committed, was fitted again as a new
    version with only a warning, though the version it repeats already stands for it."""
    d = _decide(_reg((1, "X", 1.0), (2, "Y", 1.0)), head_version=2)
    assert d["action"] == "refuse" and d["duplicate_of"] == 1
    assert "identical to v1" in d["message"] and "base_version=1" in d["message"]


def test_decide_warns_on_revisiting_a_structure_under_a_new_setup():
    d = _decide(_reg((1, "X", 1.0), (2, "Y", 1.0)), setup=NEW_SETUP, head_version=2)
    assert d["action"] == "proceed"
    assert any("version(s) [1]" in w for w in d["warnings"])


def test_decide_refits_an_earlier_version_to_search_harder():
    d = _decide(_reg((1, "X", 1.0), (2, "Y", 1.0)), starts=4, head_version=2)
    assert d["action"] == "proceed"
    assert any("identical to v1" in w and "against its 2" in w for w in d["warnings"])


def test_decide_warns_on_worse_family():
    # Structure X's best bic (20) is far worse than the global best (1, at v2).
    d = _decide(_reg((1, "X", 20.0), setup=NEW_SETUP) + _reg((2, "Y", 1.0)), head_version=2)
    assert d["action"] == "proceed"
    assert any("WORSE" in w for w in d["warnings"])


def test_decide_no_worse_warning_within_margin():
    d = _decide(_reg((1, "X", 3.0), setup=NEW_SETUP) + _reg((2, "Y", 1.0)), head_version=2)
    assert not any("WORSE" in w for w in d["warnings"])


def test_decide_handles_numpy_bic_scalars():
    """Scores stored as NumPy scalars must still drive the worse-family warning — an
    isinstance(float) filter would silently drop numpy.float64 (addresses PR #70 review)."""
    import numpy as np

    reg = [
        {
            "version": 1,
            "structure_hash": "X",
            "problem_hash": None,
            "neg_log_likelihood": None,
            "score": np.float64(20.0),
        },
        {
            "version": 2,
            "structure_hash": "Y",
            "problem_hash": None,
            "neg_log_likelihood": None,
            "score": np.float64(1.0),
        },
    ]
    d = _decide(reg, head_version=2)
    assert any("WORSE" in w for w in d["warnings"])


# -- integration: an edit no longer dedups (that moved to `fit`, issue #81) --
def test_set_model_reauthor_is_a_fit_free_edit(session):
    """Re-authoring the current structure is now a plain fit-free edit — it APPLIES (no dedup,
    no fit, no version at edit time); the de-dup that skips an identical model happens at fit."""
    assert session.provenance.current_version == -1
    res = O.set_model(session, REVERSIBLE_OTHER_VALUES, justification="same structure again")
    assert res["applied"] is True
    assert res["version"] is None
    # no version is snapshotted by an edit
    assert session.provenance.current_version == -1


def test_restore_version_goes_back_without_dedup(session, reauthor_toy):
    """restore_version is the cheap way back — it restores a prior model and is not a re-author."""
    reauthor_toy(session)  # v0: the reversible structure, scored
    O.set_model(session, ONE_REACTION, justification="drop r2")
    assert {
        session.model.sbml_model.getReaction(i).getId()
        for i in range(session.model.sbml_model.getNumReactions())
    } == {"r1"}
    out = session.restore_version(0)
    assert out["rolled_back_to"] == 0
    # the reversible structure (r1 + r2) is back
    assert {
        session.model.sbml_model.getReaction(i).getId()
        for i in range(session.model.sbml_model.getNumReactions())
    } == {"r1", "r2"}
