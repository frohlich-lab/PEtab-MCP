"""The estimation scale (`petab_mcp.scaling`) on the toy: the pyPESTO problem moves onto each
parameter's scale, and AMICI's PEtab v2 simulator reads the scaled values and returns derivatives
with respect to them. Checked against the same problem built linear, by the chain rule.
"""

from __future__ import annotations

import math
import pickle
from pathlib import Path

import numpy as np
import pypesto.petab
import pytest
from pypesto.C import FVAL, GRAD, HESS, MODE_FUN, MODE_RES, RES, SRES

from petab_mcp.problem import WorkingProblem
from petab_mcp.scaling import to_estimation_scale

TOY = Path(__file__).parent / "data" / "toy_v2" / "problem.yaml"
#: Both log flavours, and the FIXED sigma_B on log10, whose value the simulator has to unscale.
SCALES = {"k1": "log10", "k2": "log", "sigma_B": "log10"}
K1, K2 = 0.5, 0.1
X = np.array([math.log10(K1), math.log(K2)])
#: d lin / d scaled for the free parameters (k1, k2).
J = np.array([K1 * math.log(10.0), K2])


@pytest.fixture(scope="module")
def problems(tmp_path_factory):
    """The toy's pyPESTO problem, linear and on SCALES, from one compiled module."""
    importer = pypesto.petab.PetabImporter(
        WorkingProblem.from_yaml(TOY).to_petab(),
        output_folder=str(tmp_path_factory.mktemp("amici") / "toy_scaling"),
        model_name="toy_scaling",
    )
    linear = importer.create_problem(verbose=False)
    return linear, to_estimation_scale(importer.create_problem(verbose=False), SCALES)


def test_bounds_and_fixed_values_move_to_the_estimation_scale(problems):
    linear, scaled = problems
    assert scaled.x_names == linear.x_names == ["k1", "k2", "sigma_B", "A_init", "B_init"]
    assert scaled.x_scales == ["log10", "log", "log10", "lin", "lin"]
    assert scaled.lb == pytest.approx([-4.0, math.log(1e-4)])
    assert scaled.ub == pytest.approx([2.0, math.log(1e2)])
    assert scaled.x_fixed_vals == pytest.approx([-1.0, 1.0, 0.0])


def test_the_simulator_takes_scaled_values_and_returns_scaled_derivatives(problems):
    """Same value and residuals; gradient ``J g``, FIM ``J H J`` and residual sensitivities
    ``S J`` of the linear problem's -- what AMICI's ``pscale`` produced under v1."""
    linear, scaled = problems
    x_lin = np.array([K1, K2])
    ref = linear.objective(x_lin, sensi_orders=(0, 1, 2), mode=MODE_FUN, return_dict=True)
    out = scaled.objective(X, sensi_orders=(0, 1, 2), mode=MODE_FUN, return_dict=True)
    assert out[FVAL] == pytest.approx(ref[FVAL], rel=1e-10)
    assert out[GRAD] == pytest.approx(J * ref[GRAD], rel=1e-6)
    assert out[HESS] == pytest.approx(np.outer(J, J) * ref[HESS], rel=1e-6)
    ref = linear.objective(x_lin, sensi_orders=(0, 1), mode=MODE_RES, return_dict=True)
    out = scaled.objective(X, sensi_orders=(0, 1), mode=MODE_RES, return_dict=True)
    assert out[RES] == pytest.approx(ref[RES], rel=1e-10, abs=1e-12)
    assert out[SRES] == pytest.approx(ref[SRES] * J, rel=1e-6, abs=1e-12)


def test_the_scales_survive_pickling(problems):
    """Multistart pickles the objective into its workers, which would read log values as linear
    if the scaled experiment manager stayed behind."""
    _, scaled = problems
    clone = pickle.loads(pickle.dumps(scaled.objective))
    assert clone(X) == pytest.approx(scaled.objective(X), rel=1e-12)


def test_start_points_are_drawn_from_the_scaled_box(problems):
    """Uniform in log10 is what v1's default ``parameterScaleUniform`` meant. Uniform in linear
    space would put nearly every start for a [1e-4, 1e2] rate above 1."""
    _, scaled = problems
    starts = scaled.get_startpoints(400)
    assert np.mean(10.0 ** starts[:, 0] < 1.0) > 0.4
