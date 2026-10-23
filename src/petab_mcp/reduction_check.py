"""Whether the reduced model a statements version simulates agrees with the model its rows build
unreduced, checked once, on the model a run delivers, by whoever runs the session.

:mod:`reduction` rewrites the rules before BioNetGen expands them, and its tests hold it to
trajectory equality on small models. A run's model is none of those, so the delivered model is
checked as it stands. Its rows are built again with no reduction, the expansion they would have
had, and reconciled against the same tables. Both models are then evaluated at the version's fitted
parameter table, and the observables at every measured point and the nll must agree. Both are
integrated at :data:`INTEGRATION_RTOL`, a hundred times tighter than a fit's AMICI default, so the
integration error of two different but equal ODE systems sits far below the :data:`RTOL` they are
compared at. Neither side solves inner parameters (noise sigmas, scalings): both read the table's
values, so the two likelihoods are the same function of the same point.

The unreduced model can be large (perica_egf's first model expands to 1,464 reactions, a minute or
more to compile), so this is not part of a fit. It is the tool ``verify_reduction``, which, like the
held-out score, is for whoever runs the session and never the agent's
(``server._register_reduction_tools``). Its result goes to the HEAD version's
:data:`provenance.REDUCTION_CHECK`, which no agent-facing read opens.

A mismatch is reported, never repaired. ``status`` is ``reduction_mismatch``, with the unreduced
model's nll beside the reduced one's: the unreduced nll is the likelihood of the agent's model at
its fitted point. The version keeps the score its fit computed, so the discrepancy stays visible
in both places instead of being overwritten in one.
"""

from __future__ import annotations

import copy
import time
from typing import Any

#: The relative difference at or under which the two models agree: on the observables at each
#: measured point (against :data:`FLOOR` of the observable's largest value, for a point near 0),
#: and on the nll (against 1 for an nll near 0).
RTOL = 1e-6
#: A point's difference is relative to at least this fraction of its observable's largest value.
FLOOR = 1e-3
#: How tightly both models are integrated, relative and absolute, and how many steps either may
#: take to get there.
INTEGRATION_RTOL = 1e-10
INTEGRATION_ATOL = 1e-12
MAX_STEPS = 10**6

#: ``status`` values: the two models agree; they do not; nothing was reduced, so there is nothing
#: to compare; the model was not built from statements; a side could not be integrated.
STATUS_OK = "ok"
STATUS_MISMATCH = "reduction_mismatch"
STATUS_NOT_REDUCED = "not_reduced"
STATUS_NOT_STATEMENTS = "not_statements"
STATUS_NON_FINITE = "non_finite"


def verify(session, engine) -> dict[str, Any]:
    """The session's CURRENT model against its unreduced expansion, at its fitted table, written
    to the HEAD version's :data:`provenance.REDUCTION_CHECK` (the reply's ``file``). Held under
    ``session.lock``, so the model, its rows and its table are one snapshot."""
    with session.lock:
        t0 = time.perf_counter()
        out = _verify(session, engine)
        out["seconds"] = round(time.perf_counter() - t0, 3)
        head = session.provenance.current_version
        if head >= 0:
            out["version"] = head
            out["file"] = str(session.provenance.write_reduction_check(head, out))
        if out["status"] == STATUS_MISMATCH:
            session.logger.error("verify_reduction: the reduced model disagrees: %s", out)
        return out


def _verify(session, engine) -> dict[str, Any]:
    record = session.model_statements
    if not record:
        return {"status": STATUS_NOT_STATEMENTS}
    out: dict[str, Any] = {
        "network": record.get("network"),
        "rtol": RTOL,
        "integration_rtol": INTEGRATION_RTOL,
    }
    antimony = _unreduced_antimony(session, record)
    # Not the network's counts, which a rewrite can leave as they were (one catalysed step is
    # one reaction either way): the model itself, which is the reduced one only if nothing was
    # rewritten.
    if antimony == session.model_source:
        return {**out, "status": STATUS_NOT_REDUCED}
    full = _reconciled(session, antimony)
    extra = sorted(
        set(map(str, full.parameter_df.index)) ^ set(map(str, session.problem.parameter_df.index))
    )
    if extra:
        return {
            **out,
            "status": STATUS_MISMATCH,
            "reason": f"the two models have different parameters: {extra}",
        }
    full.parameter_df = session.problem.parameter_df.copy()
    nll, simulated = _evaluate(session.problem, engine)
    full_nll, full_simulated = _evaluate(full, engine)
    out["nll"] = {"reduced": nll, "unreduced": full_nll}
    if nll is None or full_nll is None:
        both = nll is None and full_nll is None
        return {
            **out,
            "status": STATUS_NON_FINITE if both else STATUS_MISMATCH,
            "reason": "neither model integrates at the fitted point"
            if both
            else "only one of the two models integrates at the fitted point",
        }
    out |= _compare(simulated, full_simulated)
    out["nll_difference"] = abs(nll - full_nll) / max(1.0, abs(full_nll))
    agree = out["nll_difference"] <= RTOL and out["max_relative_difference"] <= RTOL
    return {**out, "status": STATUS_OK if agree else STATUS_MISMATCH}


def _unreduced_antimony(session, record) -> str:
    """The Antimony the version's rows build with no reduction."""
    from . import operations

    rows, starts, readouts, amounts, _ = operations.apply_statement_edits(record, [])
    antimony, *_ = operations.statements_to_antimony(
        session, rows, None, starts, readouts, amounts, reduce=False
    )
    return antimony


def _reconciled(session, antimony: str):
    """The session's tables with ``antimony`` for their model, reconciled as
    :func:`operations.set_model` reconciles every model."""
    from petab.v1.models.sbml_model import SbmlModel

    from . import reconcile as R

    problem = copy.deepcopy(session.problem)
    model = SbmlModel.from_antimony(antimony, model_id=None)
    model.rel_path = getattr(session.problem.model, "rel_path", None) or "model.xml"
    problem.model = model
    R.reconcile(problem, shipped=session.shipped_parameter_df)
    return problem


def _evaluate(problem, engine) -> tuple[float | None, list[dict[str, Any]]]:
    """``problem``'s nll at its table's values and its simulation at each measured point, as the
    engine evaluates a model (:meth:`engine.AmiciEngine._objective_evaluate`, without the inner
    solve), integrated at :data:`INTEGRATION_RTOL`. Builds the module if it is not there yet."""
    from pypesto.C import FVAL, RDATAS

    from . import fitting
    from .engine import AmiciEngine
    from .formatting import finite_float

    flat = fitting.flattened(problem) if fitting.needs_flattening(problem) else None
    kwargs = engine.pypesto_importer_kwargs(problem.module_fingerprint())
    pp = fitting.pypesto_problem(flat or problem, kwargs, validate=False)
    fitting._apply_amici_solver_defaults(pp)
    solver = pp.objective.amici_solver
    solver.set_relative_tolerance(INTEGRATION_RTOL)
    solver.set_absolute_tolerance(INTEGRATION_ATOL)
    solver.set_max_steps(MAX_STEPS)
    x = fitting.problem_x_from_nominal(pp, problem.parameter_df)
    ret = pp.objective(x, sensi_orders=(0,), return_dict=True)
    records = AmiciEngine._records_from_rdatas(problem, pp.objective, ret[RDATAS], flat)
    return finite_float(ret[FVAL]), records


def _compare(reduced: list[dict[str, Any]], unreduced: list[dict[str, Any]]) -> dict[str, Any]:
    """The largest relative difference between two models' simulations of the same measured
    points, and the point it is at."""
    keys = ("observable_id", "experiment_id", "time")
    scale: dict[str, float] = {}
    for r in unreduced:
        scale[r["observable_id"]] = max(
            scale.get(r["observable_id"], 0.0), abs(r["simulation"] or 0.0)
        )
    worst, at = 0.0, None
    for a, b in zip(reduced, unreduced, strict=True):
        if [a[k] for k in keys] != [b[k] for k in keys]:
            raise ValueError(f"verify_reduction: the two models' points differ: {a} and {b}")
        if (a["simulation"] is None) != (b["simulation"] is None):
            return {"max_relative_difference": float("inf"), "worst": {k: a[k] for k in keys}}
        if a["simulation"] is None:
            continue
        x, y = a["simulation"], b["simulation"]
        denominator = max(abs(x), abs(y), FLOOR * scale[a["observable_id"]])
        difference = abs(x - y) / denominator if denominator else 0.0
        if difference >= worst:
            worst, at = difference, {**dict((k, a[k]) for k in keys), "reduced": x, "unreduced": y}
    return {"max_relative_difference": worst, "worst": at, "n_points": len(reduced)}
