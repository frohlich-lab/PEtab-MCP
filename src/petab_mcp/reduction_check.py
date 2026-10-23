"""Whether the reduced model a statements version simulates agrees with the model its rows build
unreduced, checked once, on the model a run delivers, by whoever runs the session.

:mod:`reduction` rewrites the rules before BioNetGen expands them, and its tests hold it to
trajectory equality on small models. A run's model is none of those, so the delivered model is
checked as it stands. Its rows are built again with no reduction, the expansion they would have
had, and reconciled against the same tables. Both models are then evaluated at the version's fitted
parameter table, and the observables at every measured point and the nll must agree. Neither side
solves inner parameters (noise sigmas, scalings): both read the table's values, so the two
likelihoods are the same function of the same point.

Two different but equal ODE systems agree only to their integration error, and a pre-equilibrated
model's start carries an error of its own: a steady state is accepted once its rate of change is
small, which leaves a slow mode off by that tolerance over its rate. On perica_egf's r2_v2 that put
the two models 1.3e-4 apart at t=0 with every tolerance at 1e-10 but the steady state's (verify-98,
NEMO). So both models are integrated at :data:`TIGHT`, the steady state's tolerances included, and
the unreduced one again at :data:`LOOSE`, a hundred times looser: its own change between the two is
the noise of its integration. A difference is a mismatch only beyond :data:`SPREAD` times that noise
plus :data:`RTOL` of the value, so neither a stiff steady state nor a model the tight level cannot
resolve reads as a reduction error, while a wrong rate, which moves the model and not its noise,
still does.

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
from typing import Any, NamedTuple


class Tolerances(NamedTuple):
    """What AMICI integrates to: relative and absolute, in time and at a pre-equilibration's
    steady state."""

    rtol: float
    atol: float
    rtol_steady_state: float
    atol_steady_state: float


#: Both models are integrated here, and the unreduced one also at :data:`LOOSE`.
TIGHT = Tolerances(1e-12, 1e-14, 1e-14, 1e-18)
LOOSE = Tolerances(*(100 * t for t in TIGHT))
#: The steps either may take, in time or to a steady state, so a tolerance it cannot meet ends.
MAX_STEPS = 10**6
#: A mismatch is a difference beyond this many times the unreduced model's own change from
#: :data:`TIGHT` to :data:`LOOSE`, plus :data:`RTOL` of the value: on the observables at each
#: measured point, and on the nll.
SPREAD = 10
#: The relative difference always allowed: a point's against its size, at least :data:`FLOOR` of
#: its observable's largest value and :data:`ATOL` / :data:`RTOL`; the nll's against at least 1.
RTOL = 1e-6
FLOOR = 1e-3
#: The absolute difference always allowed: ten times what the looser integration resolves.
#: Without it a model whose observable stays near 1e-15 at the point it is checked at, as a seeded
#: toy's does, mismatched on rounding.
ATOL = 10 * LOOSE.atol

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
        "tolerances": {"tight": TIGHT._asdict(), "loose": LOOSE._asdict()},
        "spread_factor": SPREAD,
        "rtol": RTOL,
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
    nll, simulated = _evaluator(session.problem, engine)(TIGHT)
    unreduced = _evaluator(full, engine)
    full_nll, full_simulated = unreduced(TIGHT)
    loose_nll, loose_simulated = unreduced(LOOSE)
    out["nll"] = {"reduced": nll, "unreduced": full_nll, "unreduced_loose": loose_nll}
    if nll is None or full_nll is None:
        both = nll is None and full_nll is None
        return {
            **out,
            "status": STATUS_NON_FINITE if both else STATUS_MISMATCH,
            "reason": "neither model integrates at the fitted point"
            if both
            else "only one of the two models integrates at the fitted point",
        }
    # A looser integration that fails leaves no noise to allow for: the rule is then RTOL alone.
    out |= _compare(simulated, full_simulated, loose_simulated if loose_nll is not None else None)
    base = max(1.0, abs(full_nll))
    spread = abs(full_nll - loose_nll) if loose_nll is not None else 0.0
    out["nll_difference"] = abs(nll - full_nll) / base
    out["nll_spread"] = spread / base
    out["nll_margin"] = abs(nll - full_nll) / (SPREAD * spread + RTOL * base)
    agree = out["margin"] <= 1 and out["nll_margin"] <= 1
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


def _evaluator(problem, engine):
    """A function of :class:`Tolerances` giving ``problem``'s nll at its table's values and its
    simulation at each measured point, as the engine evaluates a model
    (:meth:`engine.AmiciEngine._objective_evaluate`, without the inner solve). Builds the module
    if it is not there yet, once for every level."""
    from pypesto.C import FVAL, RDATAS

    from . import fitting
    from .engine import AmiciEngine
    from .formatting import finite_float

    flat = fitting.flattened(problem) if fitting.needs_flattening(problem) else None
    kwargs = engine.pypesto_importer_kwargs(problem.module_fingerprint())
    pp = fitting.pypesto_problem(flat or problem, kwargs, validate=False)
    fitting._apply_amici_solver_defaults(pp)
    x = fitting.problem_x_from_nominal(pp, problem.parameter_df)

    def at(level: Tolerances) -> tuple[float | None, list[dict[str, Any]]]:
        solver = pp.objective.amici_solver
        solver.set_relative_tolerance(level.rtol)
        solver.set_absolute_tolerance(level.atol)
        solver.set_relative_tolerance_steady_state(level.rtol_steady_state)
        solver.set_absolute_tolerance_steady_state(level.atol_steady_state)
        solver.set_max_steps(MAX_STEPS)
        ret = pp.objective(x, sensi_orders=(0,), return_dict=True)
        records = AmiciEngine._records_from_rdatas(problem, pp.objective, ret[RDATAS], flat)
        return finite_float(ret[FVAL]), records

    return at


def _compare(reduced, unreduced, loose=None) -> dict[str, Any]:
    """How far apart two models' simulations of the same measured points are: the largest
    relative difference, the unreduced model's own largest relative change at ``loose``
    tolerances, and the largest ``margin``, each point's difference over what it is allowed
    (``SPREAD`` times that change plus ``RTOL`` of its size), with the point it is at."""
    keys = ("observable_id", "experiment_id", "time")
    scale: dict[str, float] = {}
    for r in unreduced:
        scale[r["observable_id"]] = max(
            scale.get(r["observable_id"], 0.0), abs(r["simulation"] or 0.0)
        )
    out = {"max_relative_difference": 0.0, "max_relative_spread": 0.0, "margin": 0.0}
    out |= {"worst": None, "n_points": len(reduced)}
    for i, (a, b) in enumerate(zip(reduced, unreduced, strict=True)):
        if [a[k] for k in keys] != [b[k] for k in keys]:
            raise ValueError(f"verify_reduction: the two models' points differ: {a} and {b}")
        if (a["simulation"] is None) != (b["simulation"] is None):
            inf = float("inf")
            return {
                **out,
                "max_relative_difference": inf,
                "margin": inf,
                "worst": {k: a[k] for k in keys},
            }
        if a["simulation"] is None:
            continue
        x, y = a["simulation"], b["simulation"]
        z = loose[i]["simulation"] if loose is not None else None
        spread = abs(y - z) if z is not None else 0.0
        base = max(abs(x), abs(y), FLOOR * scale[a["observable_id"]], ATOL / RTOL)
        margin = abs(x - y) / (SPREAD * spread + RTOL * base)
        out["max_relative_difference"] = max(out["max_relative_difference"], abs(x - y) / base)
        out["max_relative_spread"] = max(out["max_relative_spread"], spread / base)
        if margin >= out["margin"]:
            out["margin"] = margin
            out["worst"] = {k: a[k] for k in keys} | {
                "reduced": x,
                "unreduced": y,
                "unreduced_loose": z,
                "allowed": SPREAD * spread + RTOL * base,
            }
    return out
