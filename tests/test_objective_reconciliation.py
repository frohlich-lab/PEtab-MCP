"""Unit tests for :func:`petab_mcp.fitting._reconcile_objective`.

Two numbers claim to be "the" neg-log-likelihood of a fitted model: the objective pyPESTO
minimised, and the one obtained by re-simulating at the written-back parameters. The
re-simulation evaluates the same objective the same way, gradient included, so they agree, and a
fit is scored only where they do. Where they do not, it is committed unscored, and its note says
why: scoring it on the optimizer's value ranked lun's v3 on a number its re-simulation put 18%
higher, and its agent refit it for 116 min into the same gap (#265). This is pure bookkeeping over
two payloads, so no AMICI/pyPESTO fit is needed.
"""

from __future__ import annotations

import logging

import pytest

from petab_mcp import fitting


class _Session:
    """Just enough session for _reconcile_objective (which only logs) and compute_improvement,
    which reads the best-BIC checkpoint. Spelled out rather than left to a getattr default in
    production code: a fake that omits state the real session always has lets a typo pass."""

    def __init__(self):
        from types import SimpleNamespace

        from petab_mcp.criteria import Criterion

        self.logger = logging.getLogger("test_objective_reconciliation")
        self.problem = SimpleNamespace(experiment_df=None, model=None)  # no pre-equilibration
        self.best = None
        self.criterion = Criterion()


@pytest.fixture
def sess():
    return _Session()


def _sim(nll, *, missing=0, expected=21):
    return {
        "neg_log_likelihood": nll,
        "n_missing_simulations": missing,
        "n_expected_points": expected,
    }


def test_agreeing_values_score_the_optimizers(sess):
    nllh, note = fitting._reconcile_objective(
        sess, {"neg_log_likelihood": 250.0}, _sim(250.0 + 1e-9)
    )
    assert nllh == 250.0
    assert note is None


def test_a_persistent_gap_leaves_the_fit_unscored(sess):
    """lun 10 v3's numbers: neither value is the score, and the note names both."""
    nllh, note = fitting._reconcile_objective(
        sess, {"neg_log_likelihood": 2426.61}, _sim(2857.15, expected=2700)
    )
    assert nllh is None
    assert "not scored" in note and "2426.61" in note and "2857.15" in note
    imp = fitting.compute_improvement(
        sess, nllh, fitsim={"neg_log_likelihood": nllh, "n_data_points": 2700, "n_estimated": 10}
    )
    assert imp["criterion_value"] is None and imp["is_new_best"] is False


def test_a_resimulation_with_no_value_leaves_the_fit_unscored(sess):
    """Not the optimizer's value: it can count only the points that simulated (pyPESTO#1794)."""
    nllh, note = fitting._reconcile_objective(
        sess, {"neg_log_likelihood": 266.62}, _sim(float("nan"), missing=5)
    )
    assert nllh is None
    assert "not scored" in note and "5 of 21" in note


@pytest.mark.parametrize("nll_sim", [float("nan"), 263.28])
def test_an_optimizer_with_no_finite_objective_leaves_the_fit_unscored(sess, nll_sim):
    nllh, note = fitting._reconcile_objective(
        sess, {"neg_log_likelihood": float("inf")}, _sim(nll_sim, missing=21)
    )
    assert nllh is None
    assert "no finite objective" in note


def test_simulate_only_pass_uses_the_simulated_value(sess):
    """No fit ran (rollback restore / n_starts=0), so there is no optimizer value to compare."""
    nllh, note = fitting._reconcile_objective(sess, None, _sim(263.28))
    assert nllh == pytest.approx(263.28)
    assert note is None


def test_missing_pre_fit_objective_does_not_abort_the_fit(sess):
    """A non-finite objective at the PRE-FIT nominal values must not stop the fit.

    `_run_fit` reads `objective_before` from `evaluate_objective`, which reports a non-finite
    likelihood as a missing score (None). Coercing that with `float(...)` raised a TypeError
    that `session.fit` wrapped as "the model passed validation but could not be compiled/fit ...
    ill-posed for estimation" — refusing to fit an unstable-but-fittable model and telling the
    agent the wrong thing about it. compute_improvement must simply carry the gap through.
    """
    fs = {
        "status": "fit",
        "neg_log_likelihood": 266.62,
        "chi2": 100.0,
        "n_data_points": 21,
        "n_estimated": 10,
    }
    imp = fitting.compute_improvement(sess, nllh=266.62, objective_before=None, fitsim=fs)
    assert imp["self_improvement"] is None  # unknown, not an error
    assert imp["bic"] is not None  # the fit is still scored
    assert imp["is_new_best"] is True
