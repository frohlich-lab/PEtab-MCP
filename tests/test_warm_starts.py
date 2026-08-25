"""Warm multistart starts (``fitting.warm_start_guesses``).

Half of every multistart re-enters the PREVIOUS fit's optima. Start 0 is the model's current
parameter values — the parent's BEST optimum, written back by ``_run_fit``. For a NESTED edit (a
new parameter introduced at a neutral value) that point reproduces the parent's likelihood
exactly, so the fit cannot converge worse than its parent unless the added structure genuinely
does not help. Without it, "adding this term made the fit worse" is ambiguous between a real
answer and a multistart that never found the parent's basin — the ambiguity
``petab_mcp.nesting`` has to warn about.

The remaining warm starts are the parent multistart's OTHER converged optima (``previous_optima``),
remapped onto the child's parameter vector by NAME: a shared parameter is carried across through
linear (the two problems can scale it differently), and anything the parent did not have keeps
its neutral nominal value, which is how a lower-dimensional parent point expands into the child's
space. Everything is projected into the child's bounds before use, and deduplicated after, so a
tightened box cannot spend two warm starts on one feasible point. Jitter around start 0 fills the
budget only when the parent cannot supply enough distinct optima.

The other half stays freshly sampled: a new parameter can open local optima none of the parent's
basins contain.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from petab_mcp import fitting


class _FakeProblem:
    """The pyPESTO ``Problem`` surface ``warm_start_guesses`` reads: names/scales for the full
    vector, ``x_free_indices`` to locate the free coordinates, and free-space ``lb``/``ub``."""

    def __init__(self, names, scales, free, lb, ub, fixed_vals=()):
        self.x_names = list(names)
        self.x_scales = list(scales)
        self.x_free_indices = list(free)
        self.x_fixed_indices = [i for i in range(len(names)) if i not in free]
        self.x_fixed_vals = list(fixed_vals)
        self.lb = np.asarray(lb, dtype=float)
        self.ub = np.asarray(ub, dtype=float)


def _problem():
    # k1/k2 free on log10, d fixed — the shape a real PEtab problem has after reconciliation.
    return _FakeProblem(
        names=["k1", "k2", "d"], scales=["log10", "log10", "lin"],
        free=[0, 1], lb=[-4.0, -4.0], ub=[2.0, 2.0], fixed_vals=[1.0])


def _pdf(k1=0.5, k2=0.1):
    return pd.DataFrame(
        [{"parameterId": "k1", "nominalValue": k1},
         {"parameterId": "k2", "nominalValue": k2},
         {"parameterId": "d", "nominalValue": 1.0}]).set_index("parameterId")


def test_half_the_starts_are_warm_and_the_first_is_the_nominal_point():
    g = fitting.warm_start_guesses(_problem(), _pdf(), n_starts=10)
    assert g.shape == (5, 3)  # WARM_START_FRACTION of 10, full-dimensional vectors
    # start 0 is the un-jittered current optimum, on each parameter's ESTIMATION scale
    assert g[0, 0] == pytest.approx(np.log10(0.5))
    assert g[0, 1] == pytest.approx(np.log10(0.1))
    # the fixed coordinate is carried through untouched (pyPESTO re-inserts it either way)
    assert g[0, 2] == pytest.approx(1.0)


def test_the_remaining_warm_starts_scatter_around_it():
    g = fitting.warm_start_guesses(_problem(), _pdf(), n_starts=10)
    # all distinct — running the same optimization five times would waste half the budget
    assert len({tuple(row) for row in g}) == 5
    # ...but local: within a few jitter widths of the parent optimum, not a fresh uniform draw
    spread = fitting.WARM_START_JITTER * (2.0 - -4.0)
    assert np.abs(g[1:, :2] - g[0, :2]).max() < 5 * spread


def test_jitter_is_clipped_to_the_box():
    # nominal sitting hard against the upper bound: every jittered start must stay inside it
    g = fitting.warm_start_guesses(_problem(), _pdf(k1=100.0, k2=100.0), n_starts=40)
    assert (g[:, :2] <= 2.0 + 1e-12).all() and (g[:, :2] >= -4.0 - 1e-12).all()


def test_a_nominal_point_outside_a_tightened_bound_is_pulled_back_in():
    # A stored optimum can fall outside a bound the agent narrowed since. pyPESTO would only warn
    # and then optimize from an infeasible point, so start 0 is clipped like the rest.
    g = fitting.warm_start_guesses(_problem(), _pdf(k1=1e6), n_starts=4)
    assert g[0, 0] == pytest.approx(2.0)


def test_deterministic_across_calls():
    a = fitting.warm_start_guesses(_problem(), _pdf(), n_starts=10)
    b = fitting.warm_start_guesses(_problem(), _pdf(), n_starts=10)
    assert np.array_equal(a, b)


def test_no_guesses_when_there_is_no_room_for_one():
    # n_starts=1 leaves no half to warm — the single start stays a fresh sample.
    assert fitting.warm_start_guesses(_problem(), _pdf(), n_starts=1).shape == (0, 3)


def test_no_guesses_when_nothing_is_free():
    p = _FakeProblem(names=["k1"], scales=["log10"], free=[], lb=[], ub=[], fixed_vals=[0.5])
    assert fitting.warm_start_guesses(p, _pdf(), n_starts=10).shape == (0, 1)


def test_no_guesses_from_a_non_finite_nominal_point():
    # A NaN/absent nominal leaves no point to start from; falling back to a pure fresh sample is
    # the honest behaviour, not a guess built around a NaN.
    assert fitting.warm_start_guesses(
        _problem(), _pdf(k1=float("nan")), n_starts=10).shape == (0, 3)


# --- against the real pyPESTO problem ------------------------------------------------------
def test_pypesto_accepts_the_guesses_and_takes_them_as_start_points(session, engine):
    """The contract that actually matters: pyPESTO must accept the array as ``x_guesses_full``
    and fill the FIRST n_warm start points from it (see pypesto.startpoint.base), so start 0 is
    the parent optimum and the rest of the multistart is sampled."""
    problem, _ = fitting.estimation_problem(session, engine)
    n_starts = 8
    guesses = fitting.warm_start_guesses(problem, session.problem.parameter_df, n_starts)
    problem.x_guesses_full = guesses
    assert problem.x_guesses.shape == (4, problem.dim)  # reduced to the free coordinates
    # the guess equals the nominal point pyPESTO's objective would be evaluated at
    assert problem.x_guesses[0] == pytest.approx(
        fitting.problem_x_from_nominal(problem, session.problem.parameter_df))
    xs = problem.startpoint_method(n_starts=n_starts, problem=problem)
    assert xs.shape == (n_starts, problem.dim)
    assert xs[: len(guesses)] == pytest.approx(problem.x_guesses)


def test_a_nested_refit_cannot_come_out_worse_than_its_parent(fit_session):
    """The property the whole mechanism exists for, end to end.

    Freeing a parameter that was FIXED at its nominal value is a strictly nested extension: the
    child reproduces the parent exactly at that value, so its ML nll cannot be worse. The warm
    start puts start 0 on precisely that point, which is what turns "cannot be worse in theory"
    into "is not worse in practice" — an under-converged multistart that never found the parent's
    basin is how a nested variant ends up looking like evidence against the added structure (see
    petab_mcp.nesting). Uses a parameter-table change rather than a model edit so the fixture's
    compiled AMICI module is reused.
    """
    from petab.v1.C import ESTIMATE

    parent = fit_session.fit()
    parent_nll = parent["neg_log_likelihood"]
    assert parent["fit_status"] == "fit" and parent_nll is not None

    fit_session.problem.parameter_df.loc["sigma_B", ESTIMATE] = 1  # nested: sigma was fixed at 0.1
    child = fit_session.fit()
    assert child["n_estimated"] > parent["n_estimated"]
    assert child["neg_log_likelihood"] <= parent_nll + 1e-6
    # ...so the commit-time nesting check has nothing to complain about
    assert "nesting_advisory" not in child and fit_session.invalid_comparison is None


def test_fit_still_runs_when_the_nominal_point_has_no_finite_objective(fit_session, monkeypatch):
    """An unstable model at its pre-fit nominal values has no point worth starting from, so
    `_run_fit` bypasses `warm_start_guesses` and hands pyPESTO an empty guess array built inline.
    That array must still be dim_full-shaped: pyPESTO rejects any other width, which would abort
    the fit on exactly the fragile model where a crash costs most."""
    from petab_mcp import fitting

    real = fit_session.engine.evaluate_objective
    monkeypatch.setattr(
        fit_session.engine, "evaluate_objective",
        lambda: {**real(), "neg_log_likelihood": float("nan")})

    out = fitting.fit_and_simulate(fit_session, fit_session.engine, do_fit=True, n_starts=2)
    assert out["status"] == "fit" and out["objective_before"] is None
    # every start was freshly sampled — no guesses were left on the cached problem
    problem, _ = fitting.estimation_problem(fit_session, fit_session.engine)
    assert problem.x_guesses_full.shape == (0, len(problem.x_names))


# --- reusing the previous fit's optima ------------------------------------------------------
def _nested_problem():
    """The child of `_problem()`: k1/k2 kept, a new free `c` added (the nested-edit shape)."""
    return _FakeProblem(
        names=["k1", "k2", "c"], scales=["log10", "log10", "log10"],
        free=[0, 1, 2], lb=[-4.0, -4.0, -4.0], ub=[2.0, 2.0, 2.0])


def _nested_pdf(k1=0.5, k2=0.1, c=1.0):
    return pd.DataFrame(
        [{"parameterId": "k1", "nominalValue": k1},
         {"parameterId": "k2", "nominalValue": k2},
         {"parameterId": "c", "nominalValue": c}]).set_index("parameterId")


def _prev(*points, scales=("log10", "log10"), names=("k1", "k2")):
    """A `_optimization_payload`-shaped previous fit whose starts are already best-first."""
    return {"x_names": list(names), "x_scales": list(scales),
            "starts": [{"start": i, "fval": -100.0 + i, "x": list(p)}
                       for i, p in enumerate(points)]}


def test_a_new_parameter_gets_all_THREE_neutral_candidates_on_the_parent_optimum():
    """Which value neutralises a new parameter cannot be inferred from the symbol: a GATE is
    neutral at its lower bound, a pass-through RATE at its upper bound, and unit scale is the
    common declared convention. So all three are offered, and every one of them sits on the
    PARENT'S OWN optimum — that is what makes one of them reproduce the parent exactly.

    Deriving them from x0 is the whole point. Writing the candidates onto the rows that carry the
    parent's 2nd/3rd-best basins instead would pair the upper-bound candidate with a different
    basin, so nothing would reproduce the parent and an upper-bound-neutral edit (an RNA delay
    chain) could still come back worse and be misread as 'the mechanism does not help'.
    """
    prev = _prev([np.log10(0.5), np.log10(0.1)],      # the best optimum (= the nominal point)
                 [np.log10(9.0), np.log10(0.02)],     # a second local optimum
                 [np.log10(0.3), np.log10(7.0)])      # a third
    # n_starts=12 (n_warm=6) so all three candidates fit: they are capped at half the warm
    # budget, and at n_warm=4 only two would be generated.
    g = fitting.warm_start_guesses(_nested_problem(), _nested_pdf(), 12, optimization=prev)
    assert g.shape == (6, 3)
    # rows 0-2 are the three candidates, ALL on the parent's optimum in the shared coordinates
    for row in g[:3]:
        assert row[:2] == pytest.approx([np.log10(0.5), np.log10(0.1)])
    # UNIT first (the declared convention and the likeliest neutralisation), then lb, then ub
    assert [round(float(v), 6) for v in g[:3, 2]] == [0.0, -4.0, 2.0]
    # the parent's other basins follow, still carrying a neutral `c`
    assert g[3, :2] == pytest.approx([np.log10(9.0), np.log10(0.02)])


def test_the_neutral_candidates_collapse_when_the_edit_adds_no_parameter():
    """No new parameter means the three candidates ARE x0, so dedup leaves the previous
    behaviour untouched: start 0 is the nominal point and the rest are the other basins."""
    prev = _prev([np.log10(0.5), np.log10(0.1)],
                 [np.log10(9.0), np.log10(0.02)],
                 [np.log10(0.3), np.log10(7.0)])
    g = fitting.warm_start_guesses(_problem(), _pdf(), 8, optimization=prev)
    # `_problem()` carries a third, FIXED coordinate `d`; only the free ones are asserted
    assert g[0, :2] == pytest.approx([np.log10(0.5), np.log10(0.1)])
    assert g[1, :2] == pytest.approx([np.log10(9.0), np.log10(0.02)])
    assert g[2, :2] == pytest.approx([np.log10(0.3), np.log10(7.0)])


def test_an_infinite_bound_never_reaches_the_optimizer_as_a_start():
    """A log10 parameter with lowerBound 0 scales to -inf. Clipping cannot repair it (-inf IS
    the bound), so that candidate has to fall back to unit scale rather than handing pyPESTO a
    non-finite start coordinate."""
    prob = _FakeProblem(names=["k1", "k2", "c"], scales=["log10", "log10", "log10"],
                        free=[0, 1, 2], lb=[-4.0, -4.0, -np.inf], ub=[2.0, 2.0, np.inf])
    # n_starts=12: at 6 the candidate cap yields only the `unit` row, which is finite anyway,
    # so the infinite-bound fallback this test names would never execute.
    prev = _prev([np.log10(0.5), np.log10(0.1)])
    g = fitting.warm_start_guesses(prob, _nested_pdf(), 12, optimization=prev)
    assert np.all(np.isfinite(g)), g


def test_previous_optima_are_converted_across_a_changed_scale():
    """A hierarchical problem forces its sigmas to a linear scale, so the two fits can express
    the same parameter differently — carrying the raw coordinate across would be silent nonsense.
    """
    prev = _prev([9.0, np.log10(0.02)], scales=("lin", "log10"))  # k1 was LINEAR last time
    g = fitting.warm_start_guesses(_nested_problem(), _nested_pdf(), 12, optimization=prev)
    assert g[0, 0] == pytest.approx(np.log10(0.5))  # the candidates all sit on the nominal point
    # the previous optimum's k1=9.0 (linear) arrives as log10(9.0) on this problem's scale —
    # carrying the raw 9.0 across would place the start 9 decades out instead of ~0.95. It now
    # follows the three neutral candidates for the newly-added `c`.
    assert g[3, 0] == pytest.approx(np.log10(9.0))
    assert g[3, 1] == pytest.approx(np.log10(0.02))  # the already-log10 coordinate is unchanged


def test_duplicate_previous_optima_do_not_eat_the_warm_budget():
    # A multistart that converged everything to one point supplies ONE basin, not n_warm copies.
    same = [np.log10(0.5), np.log10(0.1)]
    g = fitting.warm_start_guesses(
        _nested_problem(), _nested_pdf(), 8, optimization=_prev(same, same, same))
    assert g.shape == (4, 3)
    assert len({tuple(row) for row in g}) == 4  # the repeats were replaced by jittered starts


def test_failed_previous_starts_are_not_treated_as_basins():
    # n_starts=12 (n_warm=6), not 4: at n_warm=2 the whole warm budget is neutral candidates and
    # `previous_optima` is never consulted, so this would pass with the NaN filter deleted.
    prev = _prev([np.log10(0.5), np.log10(0.1)], [np.log10(9.0), np.log10(0.02)])
    prev["starts"][1]["fval"] = float("nan")  # the start did not converge
    g = fitting.warm_start_guesses(_nested_problem(), _nested_pdf(), 12, optimization=prev)
    assert not any(np.allclose(row[:2], [np.log10(9.0), np.log10(0.02)]) for row in g)


def test_a_previous_fit_without_scales_falls_back_to_jitter():
    """Remapping without the scales would mix log10 and linear coordinates — worse than not
    reusing the optima at all. Legacy payloads (written before x_scales) take the old path."""
    prev = _prev([np.log10(9.0), np.log10(0.02)])
    prev.pop("x_scales")
    # n_starts=12 for the same reason as above: the guard has to be REACHED to be tested.
    g = fitting.warm_start_guesses(_nested_problem(), _nested_pdf(), 12, optimization=prev)
    assert g.shape == (6, 3)
    assert not any(np.allclose(row[:2], [np.log10(9.0), np.log10(0.02)]) for row in g)


def test_no_previous_fit_still_warm_starts_from_the_nominal_point():
    # The run's FIRST fit has no previous optima; the nominal point plus jitter is still better
    # than nothing, and keeps the warm/fresh split intact.
    g = fitting.warm_start_guesses(_nested_problem(), _nested_pdf(), 8, optimization=None)
    assert g.shape == (4, 3)
    assert g[0, :2] == pytest.approx([np.log10(0.5), np.log10(0.1)])


def test_a_real_fit_records_the_columns_the_next_fit_needs(fit_session):
    """`previous_optima` reads positional `x` vectors, so a fit must label them — and a version
    snapshot must persist those labels, or a rollback restores optima nothing can read."""
    fit_session.fit()
    opt = fit_session.optimization
    assert opt["x_names"] and opt["x_scales"]
    assert len(opt["x_scales"]) == len(opt["x_names"])

    stored = fit_session.provenance.get_version(fit_session.provenance.current_version)
    assert stored["optimization"]["x_names"] == opt["x_names"]
    assert stored["optimization"]["x_scales"] == opt["x_scales"]

    # ...and those optima map onto the next (nested) problem's vector
    problem, _ = fitting.estimation_problem(fit_session, fit_session.engine)
    x0 = fitting.problem_x_full_from_nominal(problem, fit_session.problem.parameter_df)
    points = fitting.previous_optima(problem, opt, x0)
    assert points and len(points[0]) == len(problem.x_names)


def test_previous_optima_are_projected_into_the_current_bounds():
    """The parent's optimum can sit outside the child's box — the agent tightens bounds between
    fits, and a rescaled coordinate can land anywhere. pyPESTO would only warn and then optimize
    from an infeasible point, so every mapped optimum is clipped like start 0.

    The clip happens BEFORE the dedup key is taken, so two parent optima that differ only outside
    the box collapse to one start rather than burning two on the same feasible point."""
    prev = _prev([np.log10(1e6), np.log10(1e-9)],   # both coordinates far outside [-4, 2]
                 [np.log10(1e9), np.log10(1e-12)])  # ...and so are these, to the SAME corner
    # n_starts=12 so BOTH mapped optima are in scope past the candidate rows — at n_warm=4 the
    # second one is never reached and "not spent twice" would be vacuously true.
    g = fitting.warm_start_guesses(_nested_problem(), _nested_pdf(), 12, optimization=prev)
    assert g.shape == (6, 3)
    assert (g[:, :3] <= 2.0 + 1e-12).all() and (g[:, :3] >= -4.0 - 1e-12).all()
    corner = [np.allclose(row[:2], [2.0, -4.0]) for row in g]
    # the first out-of-box optimum is pulled to the corner...
    assert any(corner)
    # ...and the second, which clips to that SAME corner, is not spent a second time
    assert sum(corner) == 1
