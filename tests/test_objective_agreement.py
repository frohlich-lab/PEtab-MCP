"""The fit and the re-simulation must agree about the likelihood at the fitted point.

This is the tripwire for a specific, silent failure: the optimizer and the objective evaluation
each ending up with their OWN AMICI model or solver. `fitting.estimation_problem` unified them
for exactly this reason — while they were separate, a crauste fit converged at nll 266.62 while
re-simulating its optimum failed to integrate past t=22 and reported NaN.

`_reconcile_objective` already computes the comparison on every fit and records a disagreement as
`objective_note`. Nothing asserted on it, so a re-split would have been invisible: both numbers
are individually plausible, and the reported nll would silently become whichever branch won.

These tests exist to be run against any change that touches how the problem reaches the fit.
"""

from __future__ import annotations

import pytest

from petab_mcp import operations as O

# Structural variants, so agreement is checked over more than one model shape: an extra species
# on a linear path, a second product, and a reversible branch.
_VARIANTS = {
    "degradation": ["r3: B => ; k3 * B;"],
    "extra_species": [
        "r3: B => C; k3 * B;",
        "  species C in compartment_;",
        "  C = C_init;",
        "  C_init = 0.0;",
    ],
    "reversible_branch": [
        "r3: A => C; k3 * A;",
        "r4: C => A; k4 * C;",
        "  species C in compartment_;",
        "  C = C_init;",
        "  C_init = 0.0;",
    ],
}


def _assert_agrees(res, label):
    """The fit reported a score, and the two ways of computing it agreed."""
    assert res["neg_log_likelihood"] is not None, f"{label}: no objective at all"
    assert res.get("objective_note") is None, (
        f"{label}: the optimizer and the re-simulation disagree about the likelihood at the "
        f"fitted point — {res['objective_note']}"
    )


def test_the_fit_and_the_resimulation_agree_on_the_toy(fit_session):
    _assert_agrees(fit_session.fit(n_starts=2), "toy")


@pytest.mark.parametrize("name", sorted(_VARIANTS))
def test_the_fit_and_the_resimulation_agree_after_a_reauthor(fit_session, reauthor_toy, name):
    res = reauthor_toy(fit_session, _VARIANTS[name], justification=f"variant {name}")
    _assert_agrees(res, name)


def test_the_payload_carries_both_numbers_so_a_gap_stays_visible(fit_session):
    """The optimizer's value is authoritative, but the re-simulated one is kept beside it —
    resolving the two silently is what made the crauste gap invisible."""
    fit_session.fit(n_starts=2)
    sim = fit_session.simulation
    assert sim["neg_log_likelihood"] is not None
    assert "neg_log_likelihood_simulated" in sim
    assert sim["neg_log_likelihood_simulated"] == pytest.approx(sim["neg_log_likelihood"], rel=1e-6)


def test_a_bounds_only_refit_still_agrees(fit_session):
    """The path a bounds edit takes: same structure, a rebuilt pyPESTO problem, warm starts from
    the previous optima. A problem rebuilt from the wrong table would show up here."""
    fit_session.fit(n_starts=2)
    O.change_parameters(
        fit_session,
        [{"parameter_id": "k1", "lower_bound": 1e-4, "upper_bound": 1e4}],
        justification="widen k1",
    )
    _assert_agrees(fit_session.fit(n_starts=2), "bounds-only refit")
