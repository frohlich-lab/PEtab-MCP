"""Unit tests for :func:`petab_mcp.fitting._reconcile_objective`.

Two numbers claim to be "the" neg-log-likelihood of a fitted model: the objective pyPESTO
minimised, and the one obtained by re-simulating at the written-back parameters. They are the
same quantity at the same point, but they are produced by different code paths — the
re-simulation goes through ``simulate_petab`` with AMICI's default solver settings rather than
the ones pyPESTO's objective was built with — so they can disagree, and at a stiff optimum the
re-simulation can fail outright while the fit itself converged.

Regression for the crauste sweep run where exactly that happened: pyPESTO reported
``best value: 266.62`` over 50 starts ("Converged according to fval difference"), the
re-simulation failed to integrate past t=22, and the harness recorded the fit's likelihood as
NaN — which then poisoned the best-BIC checkpoint (see test_early_stop.py) and cost the run
its score. This is pure bookkeeping over two payloads, so no AMICI/pyPESTO fit is needed.
"""

from __future__ import annotations

import logging

import pytest

from petab_mcp import fitting


class _Session:
    """Just enough session for _reconcile_objective (it only logs)."""

    def __init__(self):
        self.logger = logging.getLogger("test_objective_reconciliation")


@pytest.fixture
def sess():
    return _Session()


def _sim(nll, *, missing=0, expected=21):
    return {"neg_log_likelihood": nll, "n_missing_simulations": missing,
            "n_expected_points": expected}


def test_optimizer_value_wins_when_resimulation_fails(sess):
    """The lost-run case: a converged fit whose re-simulation could not be evaluated."""
    nllh, note = fitting._reconcile_objective(
        sess, {"neg_log_likelihood": 266.6246503480759}, _sim(float("nan"), missing=5))
    assert nllh == pytest.approx(266.6246503480759)
    assert note is not None
    assert "5 of 21" in note
    # The score must now be a real, comparable BIC rather than NaN.
    bic = fitting.goodness_of_fit(nllh, None, 10, 21)["bic"]
    assert bic == pytest.approx(563.69, abs=0.01)


def test_agreeing_values_produce_no_note(sess):
    nllh, note = fitting._reconcile_objective(
        sess, {"neg_log_likelihood": 250.0}, _sim(250.0 + 1e-9))
    assert nllh == pytest.approx(250.0)
    assert note is None


def test_material_disagreement_is_surfaced_not_hidden(sess):
    """Both finite but materially different: keep the optimizer's, and say so."""
    nllh, note = fitting._reconcile_objective(
        sess, {"neg_log_likelihood": 250.0}, _sim(311.0))
    assert nllh == pytest.approx(250.0)
    assert note is not None and "disagrees" in note


def test_simulate_only_pass_uses_the_simulated_value(sess):
    """No fit ran (rollback restore / n_starts=0), so there is no optimizer value to prefer."""
    nllh, note = fitting._reconcile_objective(sess, None, _sim(263.28))
    assert nllh == pytest.approx(263.28)
    assert note is None


def test_no_finite_objective_anywhere_stays_unscored(sess):
    """A genuinely failed fit must NOT be rescued into a fake number."""
    nllh, note = fitting._reconcile_objective(
        sess, {"neg_log_likelihood": float("nan")}, _sim(float("nan"), missing=21))
    assert nllh is None
    assert note is not None and "no finite objective" in note


def test_optimizer_nan_but_resimulation_fine(sess):
    """Symmetric case: trust the value that exists rather than the one that doesn't."""
    nllh, note = fitting._reconcile_objective(
        sess, {"neg_log_likelihood": float("inf")}, _sim(263.28))
    assert nllh == pytest.approx(263.28)


def test_missing_pre_fit_objective_does_not_abort_the_fit(sess):
    """A non-finite objective at the PRE-FIT nominal values must not stop the fit.

    `_run_fit` reads `objective_before` from `evaluate_objective`, which reports a non-finite
    likelihood as a missing score (None). Coercing that with `float(...)` raised a TypeError
    that `session.fit` wrapped as "the model passed validation but could not be compiled/fit ...
    ill-posed for estimation" — refusing to fit an unstable-but-fittable model and telling the
    agent the wrong thing about it. compute_improvement must simply carry the gap through.
    """
    fs = {"status": "fit", "neg_log_likelihood": 266.62, "chi2": 100.0,
          "n_data_points": 21, "n_estimated": 10}
    imp = fitting.compute_improvement(sess, nllh=266.62, objective_before=None, fitsim=fs)
    assert imp["self_improvement"] is None      # unknown, not an error
    assert imp["bic"] is not None               # the fit is still scored
    assert imp["is_new_best"] is True
