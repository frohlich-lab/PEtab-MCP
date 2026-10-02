"""The estimation-scale reparameterisation (`petab_mcp.scaling`), on an analytic objective.

No AMICI: a least-squares objective with known derivatives stands in for the v2 objective, so the
chain rule is checked exactly rather than through an integrator's tolerance.
"""

from __future__ import annotations

import math

import numpy as np
import pypesto
import pytest
from pypesto.C import FVAL, GRAD, HESS, MODE_FUN, MODE_RES, RES, SRES

from petab_mcp.scaling import to_estimation_scale

#: Residuals r(p) = A p - b of three parameters, linear in p: the FIM is exactly A^T A.
_A = np.array([[1.0, 2.0, 0.0], [0.0, 1.0, 3.0], [1.0, 0.0, 1.0], [2.0, 1.0, 1.0]])
_B = np.array([1.0, 2.0, 3.0, 4.0])


def _linear_problem():
    """A pyPESTO problem over (k, s, c) in LINEAR coordinates, ``c`` fixed at 2."""

    def res(p):
        return _A @ p - _B

    objective = pypesto.Objective(
        fun=lambda p: 0.5 * float(res(p) @ res(p)),
        grad=lambda p: _A.T @ res(p),
        hess=lambda p: _A.T @ _A,
        res=res,
        sres=lambda p: _A,
    )
    return pypesto.Problem(
        objective=objective,
        lb=[1e-3, 1e-2, 0.5],
        ub=[1e2, 1e1, 5.0],
        x_fixed_indices=[2],
        x_fixed_vals=[2.0],
        x_names=["k", "s", "c"],
    )


@pytest.fixture
def scaled():
    return to_estimation_scale(_linear_problem(), {"k": "log10", "s": "log", "c": "log10"})


def test_bounds_fixed_values_and_names_move_to_the_estimation_scale(scaled):
    assert scaled.x_scales == ["log10", "log", "log10"]
    assert scaled.lb == pytest.approx([-3.0, math.log(1e-2)])
    assert scaled.ub == pytest.approx([2.0, math.log(1e1)])
    assert scaled.x_fixed_vals == pytest.approx([math.log10(2.0)])
    assert scaled.x_names == ["k", "s", "c"]


def test_the_objective_is_evaluated_at_the_linear_point(scaled):
    lin = _linear_problem()
    p = np.array([0.5, 3.0])
    x = np.array([math.log10(0.5), math.log(3.0)])
    assert scaled.objective(x) == pytest.approx(lin.objective(p))


def test_derivatives_follow_the_chain_rule(scaled):
    """grad = J g, sres = S J, and the Gauss-Newton FIM J H J — what AMICI's own pscale produced
    under v1, where sensitivities were taken with respect to the scaled parameters directly."""
    k, s = 0.5, 3.0
    x = np.array([math.log10(k), math.log(s)])
    J = np.diag([k * math.log(10.0), s])
    p_full = np.array([k, s, 2.0])
    r = _A @ p_full - _B
    g = (_A.T @ r)[:2]
    H = (_A.T @ _A)[:2, :2]

    out = scaled.objective(x, sensi_orders=(0, 1, 2), mode=MODE_FUN, return_dict=True)
    assert out[FVAL] == pytest.approx(0.5 * float(r @ r))
    assert out[GRAD] == pytest.approx(J @ g)
    assert out[HESS] == pytest.approx(J @ H @ J)
    res = scaled.objective(x, sensi_orders=(0, 1), mode=MODE_RES, return_dict=True)
    assert res[RES] == pytest.approx(r)
    assert res[SRES] == pytest.approx(_A[:, :2] @ J)


def test_the_gradient_matches_finite_differences_in_estimation_coordinates(scaled):
    x = np.array([math.log10(0.5), math.log(3.0)])
    eps = 1e-6
    fd = [
        (scaled.objective(x + eps * e) - scaled.objective(x - eps * e)) / (2 * eps)
        for e in np.eye(2)
    ]
    assert scaled.objective(x, sensi_orders=(1,)) == pytest.approx(fd, rel=1e-6)


def test_start_points_are_drawn_from_the_scaled_box(scaled):
    """Uniform in log10 is what v1's default ``parameterScaleUniform`` meant. Uniform in linear
    space would put nearly every start for a [1e-3, 1e2] rate above 1."""
    starts = scaled.startpoint_method(n_starts=400, problem=scaled)
    k = 10.0 ** starts[:, 0]
    assert np.mean(k < 1.0) > 0.4
