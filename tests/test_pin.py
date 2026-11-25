"""``make_changes``' ``pin`` (issue #338): fix one member of a flat direction and move the rest so
the fit is unchanged, verified on the fit in place, and undone by the next model edit."""

from __future__ import annotations

import math

import pytest

from petab_mcp import operations as O

# k1 and kx enter only as their product, so the data pin k1*kx and leave the split free.
_SYMMETRIC = """model conversion_reaction
  compartment compartment_ = 1;
  species A in compartment_;
  species B in compartment_;
  A = A_init;
  B = B_init;
  A_init = 1.0;
  B_init = 0.0;
  k1 = 0.5;
  kx = 2;
  k2 = 0.1;
  r1: A => B; k1 * kx * A;
  r2: B => A; k2 * B;
end
"""


def _row(sess, pid):
    return sess.problem.parameter_df.loc[pid]


def test_a_pin_fixes_one_member_and_keeps_the_fit_until_the_model_changes(fit_session):
    """Agents fixed one member of a flat direction and rescaled the rest by hand, and lost basins
    doing it (alkan 70/71). A pin does it exactly or not at all, costs one fewer parameter at the
    same nll, and holds only for the structure it was verified on."""
    sess = fit_session
    O.set_model(sess, _SYMMETRIC, justification="split rate")
    first = sess.fit(n_starts=2)
    product = float(_row(sess, "k1")["nominalValue"]) * float(_row(sess, "kx")["nominalValue"])

    # an identifiable parameter is not pinned: the data set it
    with pytest.raises(O.OperationError, match="no flat direction"):
        O.change_parameters(sess, [{"parameter_id": "k2", "pin": 0.1}], justification="pin k2")

    edit = O.change_parameters(sess, [{"parameter_id": "kx", "pin": 1.0}], justification="pin kx")
    assert any("'kx' is pinned at 1, and ['k1'] moved" in w for w in edit["warnings"])
    assert not bool(_row(sess, "kx")["estimate"]) and float(_row(sess, "kx")["nominalValue"]) == 1
    assert float(_row(sess, "k1")["nominalValue"]) == pytest.approx(product, rel=1e-6)
    assert sess.pins == {"kx": 1.0}

    pinned = sess.fit(n_starts=2)
    assert pinned["committed"] is True
    assert pinned["n_estimated"] == first["n_estimated"] - 1
    assert pinned["neg_log_likelihood"] == pytest.approx(first["neg_log_likelihood"], abs=1e-3)
    assert sess.provenance.get_meta(pinned["version"])["pins"] == {"kx": 1.0}

    # a model edit estimates it again, from its pinned value, and says so
    edit = O.set_model(sess, _SYMMETRIC.replace("k2 * B", "k2 * B * B"), justification="dimer")
    assert bool(_row(sess, "kx")["estimate"]) and sess.pins == {}
    assert any("kx" in w and "estimated again" in w for w in edit["warnings"])

    # a restore brings the version's pins back with its table
    sess.apply_version(pinned["version"])
    assert sess.pins == {"kx": 1.0} and not bool(_row(sess, "kx")["estimate"])


@pytest.mark.parametrize(
    "parameters, cause",
    [
        ([{"parameter_id": "k1", "pin": 1.0, "upper_bound": 10.0}], "malformed_pin"),
        ([{"parameter_id": "k1", "pin": math.inf}], "malformed_pin"),
        ([{"parameter_id": "k1", "pin": 1.0}, {"parameter_id": "k2", "pin": 1.0}], "malformed_pin"),
    ],
)
def test_a_pin_is_one_finite_value_alone_in_its_entry(session, parameters, cause):
    with pytest.raises(O.OperationError) as err:
        O.change_parameters(session, parameters, justification="pin")
    assert err.value.cause == cause


def test_a_pin_is_refused_beside_a_model_route(session):
    """The pin is verified on the structure that was fitted, which a model route replaces."""
    with pytest.raises(O.OperationError) as err:
        O.set_model(session, _SYMMETRIC, "pin", parameters=[{"parameter_id": "k1", "pin": 1.0}])
    assert err.value.cause == "pin_with_model"


def test_a_pin_without_a_fit_in_this_process_is_refused(session):
    with pytest.raises(O.OperationError, match="refit, then pin"):
        O.change_parameters(session, [{"parameter_id": "k1", "pin": 1.0}], justification="pin")
