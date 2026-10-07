"""Estimate each parameter on its ``parameterScale`` although PEtab v2 has none.

PEtab v2 dropped ``parameterScale`` from the format: the scale a parameter is optimised on is the
optimiser's business, not the problem's. pyPESTO's v2 importer builds every problem linear, with
``x_scales = lin`` and linear bounds (ICB-DCM/pyPESTO#1781). A rate constant boxed in [1e-4, 1e2]
would then be searched linearly, with uniform start points that almost all land above 1; that is
not the same fit, and it is not the fit the scores here were calibrated on. So the server keeps the
column (:data:`problem.PARAMETER_SCALE`) and moves the pyPESTO problem onto it.

AMICI's v2 simulator takes the scales itself (``ExperimentManager(parameter_scales=...)``,
AMICI-dev/AMICI#3312): it reads every problem parameter on its scale and returns ``sllh``,
``sres`` and the FIM with respect to the scaled values, as AMICI's ``pscale`` did under v1. So
:func:`to_estimation_scale` moves bounds, fixed values and start points to the scaled box and gives
the objective's simulator an experiment manager with the scales; there is no chain rule here.

Inner (hierarchically solved) parameters stay linear: they are not in the outer vector, and
pyPESTO passes them to the simulator in linear space.
"""

from __future__ import annotations

from .problem import LIN, scale


def to_estimation_scale(problem, scales: dict[str, str]):
    """``problem`` (a linear pyPESTO problem from the PEtab v2 importer) in estimation
    coordinates, per ``scales`` (``{parameter id: lin|log|log10}``; an id not in it stays linear).

    The returned problem takes over ``problem``'s objective, whose simulator now reads scaled
    values, so ``problem`` itself must not be evaluated afterwards. A
    :class:`pypesto.hierarchical.HierarchicalProblem` stays one, with its inner parameters as they
    were. Start points are drawn uniformly from the SCALED box, which is what v1's default
    ``parameterScaleUniform`` initialisation prior meant.
    """
    from amici.sim.sundials.petab import ExperimentManager
    from pypesto import Problem
    from pypesto.problem import HierarchicalProblem

    sc = [str(scales.get(str(name), LIN)) for name in problem.x_names]
    # pyPESTO builds the simulator without scales; the experiment manager is the only part of it
    # that depends on them, and every holder of the simulator (the objective, its calculator, a
    # hierarchical collector) sees the swap.
    simulator = problem.objective._petab_simulator
    simulator._exp_man = ExperimentManager(
        simulator.model,
        simulator.exp_man.petab_problem,
        parameter_scales=dict(zip(problem.x_names, sc)),
    )
    kwargs = dict(
        objective=problem.objective,
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
