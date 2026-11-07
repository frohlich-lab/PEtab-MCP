"""The fit and the re-simulation must agree about the likelihood at the fitted point.

This is the tripwire for a specific, silent failure: the optimizer and the objective evaluation
each ending up with their OWN AMICI model or solver. `fitting.estimation_problem` unified them
for exactly this reason — while they were separate, a fit converged at nll 266.62 while
re-simulating its optimum failed to integrate past t=22 and reported NaN.

`_reconcile_objective` computes the comparison on every fit and leaves a fit that disagrees
unscored, saying why in `objective_note` and the call's `messages`. Nothing asserted on it, so a
re-split would have been invisible: both numbers are individually plausible.

These tests exist to be run against any change that touches how the problem reaches the fit.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from petab_mcp import fitting
from petab_mcp import operations as O
from petab_mcp.engine import AmiciEngine
from petab_mcp.references import StubReferenceResolver
from petab_mcp.server import create_server

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


def _assert_agrees(session, label):
    """The fit was scored, and the two ways of computing it agreed."""
    sim = session.simulation
    assert sim["objective_note"] is None, (
        f"{label}: the optimizer and the re-simulation disagree about the likelihood at the "
        f"fitted point — {sim['objective_note']}"
    )
    assert sim["neg_log_likelihood"] is not None, f"{label}: no objective at all"


def test_the_fit_and_the_resimulation_agree_on_the_toy(fit_session):
    fit_session.fit(n_starts=2)
    _assert_agrees(fit_session, "toy")


@pytest.mark.parametrize("name", sorted(_VARIANTS))
def test_the_fit_and_the_resimulation_agree_after_a_reauthor(fit_session, reauthor_toy, name):
    reauthor_toy(fit_session, _VARIANTS[name], justification=f"variant {name}")
    _assert_agrees(fit_session, name)


# C relaxes to 1 at k_slow during the pre-equilibration, and turns into the measured B once dosed.
_SLOW_PREEQUILIBRATION = """model conversion_reaction
  compartment compartment_ = 1;
  species C in compartment_;
  species B in compartment_;
  C = 0;
  B = B_init;
  dose = 0;
  k_slow = 1e-5;
  k1 = 0.5;
  k2 = 0.1;
  r0: => C; k_slow * (1 - C);
  r1: C => B; dose * k1 * C;
  r2: B => ; k2 * B;
end
"""


def test_a_fit_is_re_scored_with_its_gradient(dosed_session, amici_build_root):
    """#265: scored without the gradient, the steady state of a slow mode stops short of the one
    the optimizer's sensitivities integrated to (C 0.92 against 1.0 here), and the fit's own point
    came out 6% worse than its optimizer said. Re-scored with the gradient, the two agree."""
    O.set_model(
        dosed_session,
        _SLOW_PREEQUILIBRATION,
        justification="a slow pre-equilibration",
        parameters=[{"parameter_id": "k_slow", "lower_bound": 1e-5, "upper_bound": 2e-5}],
    )
    dosed_session.engine = AmiciEngine(dosed_session, build_root=str(amici_build_root))
    dosed_session.auto_fit = True
    dosed_session.fit(n_starts=2)
    _assert_agrees(dosed_session, "slow pre-equilibration")
    sim = dosed_session.simulation
    assert sim["neg_log_likelihood_simulated"] == pytest.approx(sim["neg_log_likelihood"], rel=1e-6)
    without = dosed_session.engine.residuals(solve_inner=True)["neg_log_likelihood"]
    assert fitting._disagree(sim["neg_log_likelihood"], without)


def test_a_fit_the_resimulation_does_not_reproduce_is_committed_unscored(tmp_path, monkeypatch):
    """#265: an optimizer value nothing reproduces is neither ranked nor compared, and the agent
    reads why in `messages`."""
    run_fit = fitting._run_fit

    def _off_by_50(*args, **kwargs):
        fit = run_fit(*args, **kwargs)
        return {**fit, "neg_log_likelihood": fit["neg_log_likelihood"] - 50.0}

    monkeypatch.setattr(fitting, "_run_fit", _off_by_50)
    mcp = create_server(
        str(Path(__file__).parent / "data" / "toy_v2" / "problem.yaml"),
        provenance_dir=str(tmp_path / "prov"),
        resolver=StubReferenceResolver(),
        auto_fit=True,
    )
    res = asyncio.run(
        mcp.call_tool(
            "make_changes",
            {
                "parameters": [{"parameter_id": "k1", "lower_bound": 1e-4, "upper_bound": 50.0}],
                "justification": "refit",
            },
        )
    ).structured_content
    assert res["committed"] is True
    notes = [m for m in res["messages"] if m["source"] == "fit" and "not scored" in m["message"]]
    assert len(notes) == 1, res["messages"]
    assert "objective_note" not in res
    assert res["neg_log_likelihood"] is None and res["chi2"] is None
    assert res["criterion_value"] is None and res["chi2_gof_pvalue"] is None
    assert "best_version" not in res["progress"]


def test_the_payload_carries_both_numbers_so_a_gap_stays_visible(fit_session):
    """The optimizer's value is authoritative, but the re-simulated one is kept beside it —
    resolving the two silently is what made that gap invisible."""
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
    fit_session.fit(n_starts=2)
    _assert_agrees(fit_session, "bounds-only refit")
