"""Estimate each parameter on its ``parameterScale`` although PEtab v2 has none.

PEtab v2 dropped ``parameterScale`` from the format: the scale a parameter is optimised on is the
optimiser's business, not the problem's. Neither consumer picked it up — AMICI's v2 simulator takes
linear parameters only (``ExperimentManager``: "TODO: support for pscale?") and pyPESTO's v2
importer builds every problem with ``x_scales = lin``. A rate constant boxed in [1e-4, 1e2] would
then be searched linearly, with uniform start points that almost all land above 1; that is not the
same fit, and it is not the fit the scores here were calibrated on. So the server keeps the column
(:data:`problem.PARAMETER_SCALE`) and reparameterises the objective pyPESTO builds.

:func:`to_estimation_scale` turns a linear pyPESTO problem into the same problem in estimation
coordinates: bounds, fixed values and start-point sampling move to the scaled box, and
:class:`ScaledObjective` maps each point back to linear before the AMICI objective sees it. The
derivatives follow by the chain rule with ``J = diag(d lin / d scaled)``: the gradient is
``J g``, residual sensitivities ``S J``, and the Hessian slot ``J H J`` — exact for the
Gauss-Newton FIM every AMICI objective here returns, which has no curvature term of its own to
carry. That is the same matrix AMICI's own pscale produced for v1, where sensitivities were taken
with respect to the scaled parameters directly.

Inner (hierarchically solved) parameters never reach this objective: they are not in its vector,
and pyPESTO solves them in linear space whatever their scale.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np
from pypesto.C import GRAD, HESS, SRES
from pypesto.objective import ObjectiveBase

from .problem import LIN, LOG, LOG10, scale, unscale


def _dlin_dscaled(x_lin: np.ndarray, scales: Sequence[str]) -> np.ndarray:
    """``d lin / d scaled`` per coordinate, at the LINEAR point ``x_lin``."""
    out = np.ones(len(scales))
    for i, sc in enumerate(scales):
        if sc == LOG10:
            out[i] = x_lin[i] * math.log(10.0)
        elif sc == LOG:
            out[i] = x_lin[i]
    return out


class ScaledObjective(ObjectiveBase):
    """A linear-parameter pyPESTO objective, called in estimation coordinates.

    Everything that is not the call itself — the AMICI model and solver, the hierarchical
    calculator, ``x_ids`` — is read through to the wrapped objective, so code written against an
    AMICI objective keeps working on this one.
    """

    def __init__(
        self, objective: ObjectiveBase, scales: Sequence[str], x_names: Sequence[str]
    ) -> None:
        # `x_names` FULL, not `objective.x_names`: once the wrapped objective has been through a
        # problem, that property returns the free names only.
        super().__init__(x_names=list(x_names))
        self.objective = objective
        self.scales = tuple(str(s) for s in scales)

    def __getattr__(self, name: str) -> Any:
        # Only reached when ordinary lookup fails. `objective` itself is looked up in __dict__ so
        # an object that is half-unpickled (no `objective` yet) raises instead of recursing.
        inner = self.__dict__.get("objective")
        if inner is None or name.startswith("__"):
            raise AttributeError(name)
        return getattr(inner, name)

    def initialize(self) -> None:
        super().initialize()
        self.objective.initialize()

    def check_mode(self, mode) -> bool:
        return self.objective.check_mode(mode)

    def check_sensi_orders(self, sensi_orders, mode) -> bool:
        return self.objective.check_sensi_orders(sensi_orders, mode)

    def get_config(self) -> dict:
        return {**self.objective.get_config(), "scales": list(self.scales)}

    def to_linear(self, x: np.ndarray) -> np.ndarray:
        """A FULL estimation-coordinate vector as the linear one the wrapped objective takes."""
        return np.array([unscale(v, s) for v, s in zip(np.asarray(x, dtype=float), self.scales)])

    def update_from_problem(self, dim_full, x_free_indices, x_fixed_indices, x_fixed_vals) -> None:
        super().update_from_problem(
            dim_full=dim_full,
            x_free_indices=x_free_indices,
            x_fixed_indices=x_fixed_indices,
            x_fixed_vals=x_fixed_vals,
        )
        # The wrapped objective is called with full vectors (below), so its own pre/post-processing
        # never runs; this tells it which parameters are free, which its calculator checks
        # sensitivities against.
        self.objective.update_from_problem(
            dim_full=dim_full,
            x_free_indices=x_free_indices,
            x_fixed_indices=x_fixed_indices,
            x_fixed_vals=[
                float(unscale(v, self.scales[i])) for i, v in zip(x_fixed_indices, x_fixed_vals)
            ],
        )

    def call_unprocessed(self, x, sensi_orders, mode, return_dict, **kwargs) -> dict:
        x_lin = self.to_linear(x)
        result = self.objective.call_unprocessed(
            x=x_lin, sensi_orders=sensi_orders, mode=mode, return_dict=return_dict, **kwargs
        )
        if all(s == LIN for s in self.scales):
            return result
        d = _dlin_dscaled(x_lin, self.scales)
        if result.get(GRAD) is not None:
            result[GRAD] = np.asarray(result[GRAD]) * d
        if result.get(HESS) is not None:
            result[HESS] = np.asarray(result[HESS]) * np.outer(d, d)
        if result.get(SRES) is not None:
            result[SRES] = np.asarray(result[SRES]) * d
        return result


def to_estimation_scale(problem, scales: dict[str, str]):
    """``problem`` (a linear pyPESTO problem) in estimation coordinates, per ``scales``
    (``{parameter id: lin|log|log10}``; an id not in it stays linear).

    A :class:`pypesto.hierarchical.HierarchicalProblem` stays one, with its inner parameters as
    they were. Start points are drawn uniformly from the SCALED box, which is what v1's default
    ``parameterScaleUniform`` initialisation prior meant.
    """
    from pypesto import Problem
    from pypesto.problem import HierarchicalProblem

    sc = [str(scales.get(str(name), LIN)) for name in problem.x_names]
    kwargs = dict(
        objective=ScaledObjective(problem.objective, sc, problem.x_names),
        lb=[float(scale(v, s)) for v, s in zip(problem.lb_full, sc)],
        ub=[float(scale(v, s)) for v, s in zip(problem.ub_full, sc)],
        dim_full=problem.dim_full,
        x_fixed_indices=list(problem.x_fixed_indices),
        x_fixed_vals=[
            float(scale(v, sc[i])) for i, v in zip(problem.x_fixed_indices, problem.x_fixed_vals)
        ],
        x_names=list(problem.x_names),
        x_scales=sc,
        copy_objective=False,
    )
    if isinstance(problem, HierarchicalProblem):
        return HierarchicalProblem(
            inner_x_names=problem.inner_x_names,
            inner_lb=problem.inner_lb,
            inner_ub=problem.inner_ub,
            **kwargs,
        )
    return Problem(**kwargs)
