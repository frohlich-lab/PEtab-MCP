"""The effective parameter count ``k_eff = tr(J·H⁻¹)`` — how many parameters the data see — and
the plumbing that persists it with the version while keeping it off every result the agent reads.

The statistic is exercised on a fake built objective (residuals + sensitivities in residual mode,
an AMICI model whose sigma-residual flag it toggles), so each property is pinned by construction
rather than against the implementation's own arithmetic: the ``k_eff = m`` identity for unit
residuals, a redundant column counting nothing, high-leverage misfit inflating it, the
per-condition row folding, the ρ²-weighted sigma information, and the failure modes.
"""

from __future__ import annotations

import numpy as np
import pytest

from petab_mcp import fitting
from petab_mcp.identifiability import effective_parameter_count


class _Model:
    def __init__(self):
        self.sigma_residuals = False
        self.flags_seen: list[bool] = []

    def get_add_sigma_residuals(self):
        return self.sigma_residuals

    def set_add_sigma_residuals(self, flag):
        self.sigma_residuals = bool(flag)


class _Objective:
    """A built objective as `effective_parameter_count` sees it: residuals + sensitivities in
    residual mode, and an AMICI model whose sigma-residual flag it toggles. `blocks` are the
    per-condition row counts pyPESTO reports through `rdatas`."""

    def __init__(self, res, sres, blocks=None, *, fail_res=False):
        self.amici_model = _Model()
        self.res, self.sres = res, sres
        self.blocks = blocks or [len(res)]
        self.fail_res = fail_res

    def __call__(self, x, sensi_orders=(0,), mode="mode_fun", return_dict=False):
        assert mode == "mode_res", "only the residual mode is evaluated — one call, not two"
        self.amici_model.flags_seen.append(self.amici_model.sigma_residuals)
        if self.fail_res:
            raise RuntimeError("no residual mode here")
        return {"res": self.res, "sres": self.sres,
                "rdatas": [{"res": [0.0] * b} for b in self.blocks]}


class _Problem:
    def __init__(self, objective):
        self.objective = objective


class _Relative:
    """pyPESTO's relative inner calculator, reduced to the one method the hierarchical path drives:
    solve the inner problem at x, evaluate AMICI with the solution filled in, hand back residuals
    with a column for EVERY optimisation parameter."""

    def __init__(self, res, sres_full, blocks):
        self.res, self.sres_full, self.blocks = res, sres_full, blocks
        self.calls: list[dict] = []

    def call_amici_twice(self, **kw):
        self.calls.append(kw)
        assert kw["mode"] == "mode_res" and kw["sensi_orders"] == (0, 1)
        return ({"res": self.res, "sres": self.sres_full,
                 "rdatas": [{"res": [0.0] * b} for b in self.blocks]}, {"sigma": 0.1})


class _Collector:
    def __init__(self, relative):
        self.inner_calculators = [relative]


class _Pre:
    def __init__(self, fixed):
        self.fixed = fixed

    def preprocess(self, x):
        return np.concatenate([np.asarray(x, dtype=float), self.fixed])


class _HierarchicalObjective:
    """What pyPESTO's hierarchical AmiciObjective exposes: the collector as `calculator`, the
    pre-processor that expands the free vector, and the AMICI handles the calculator is handed."""

    def __init__(self, relative, n_fixed):
        self.amici_model = _Model()
        self.calculator = _Collector(relative)
        self.pre_post_processor = _Pre(np.zeros(n_fixed))
        self.amici_solver, self.edatas, self.n_threads, self.fim_for_hess = "solver", "edatas", 1, True
        self.x_ids = ["k1", "k2", "A_init"]
        self.parameter_mapping = "mapping"

    def par_arr_to_dct(self, x):
        return dict(zip(self.x_ids, x))

    def __call__(self, *a, **kw):
        raise RuntimeError("pyPESTO's direct hierarchical residual path — never fills res/sres")


class _HierarchicalProblem:
    def __init__(self, objective, x_free_indices):
        self.objective = objective
        self.x_free_indices = x_free_indices


def _k_eff(X, res, *, k=None, blocks=None, fail_res=False):
    """`X` and `res` are the DATA rows; each condition block gets a zero sigma half appended, as
    AMICI lays a fixed-sigma problem out."""
    X = np.asarray(X, dtype=float)
    res = np.asarray(res, dtype=float)
    blocks = blocks or [len(res)]
    rows_res, rows_sres, start = [], [], 0
    for b in blocks:
        rows_res += [res[start:start + b], np.zeros(b)]
        rows_sres += [X[start:start + b], np.zeros((b, X.shape[1]))]
        start += b
    obj = _Objective(np.concatenate(rows_res), np.vstack(rows_sres),
                     [2 * b for b in blocks], fail_res=fail_res)
    return effective_parameter_count(_Problem(obj), np.zeros(X.shape[1]), k=k), obj


# -- the statistic ---------------------------------------------------------
def test_a_correctly_specified_fit_has_as_many_effective_parameters_as_it_has_parameters():
    """The identity that makes the count equal k when the model is right: unit-variance
    residuals on any design give tr(J·H⁻¹) = tr(I) = m."""
    X = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, -1.0], [2.0, 0.5]]
    k_eff, _ = _k_eff(X, [1, -1, 1, -1, 1])
    assert k_eff == pytest.approx(2.0)


def test_a_parameter_the_data_cannot_see_counts_nothing():
    """Two identical columns: the data determine one direction, and the undetermined one is
    dropped from H⁻¹ rather than blowing up, so a redundant parameter does not count — k_eff well
    below k."""
    X = [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [1.0, 1.0]]
    k_eff, _ = _k_eff(X, [1, 1, -1, -1])
    assert k_eff == pytest.approx(1.0)


def test_misfit_on_a_high_leverage_point_inflates_the_effective_count():
    """k_eff = Σ rᵢ² hᵢᵢ: the same total squared residual costs more when it sits where the model
    is sensitive. Here one parameter, four equal-leverage points (h = 1/4): four unit residuals
    give 1, one residual of 3 gives 9/4."""
    X = [[1.0]] * 4
    assert _k_eff(X, [1, 1, 1, 1])[0] == pytest.approx(1.0)
    assert _k_eff(X, [3, 0, 0, 0])[0] == pytest.approx(2.25)


def test_the_sigma_rows_are_folded_onto_their_own_data_rows():
    """AMICI lays a condition out as the data half followed by the sigma half, index-aligned. Each
    sigma row here exactly cancels its partner's res·sres, so the right pairing gives k_eff = 0
    and any other pairing (adjacent rows, or the wrong condition) gives something positive."""
    d = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    res = np.array([1.0, 2.0, 3.0, -1.0, -2.0, -3.0, 1.0, 1.0, -1.0, -1.0])
    sres = np.vstack([d, d, [[1.0, 2.0]], [[3.0, 4.0]], [[1.0, 2.0]], [[3.0, 4.0]]])
    obj = _Objective(res, sres, blocks=[6, 4])
    assert effective_parameter_count(_Problem(obj), np.zeros(2)) == pytest.approx(0.0)


def test_an_estimated_sigma_carries_its_full_fisher_information():
    """One parameter, ln σ itself, on n unit residuals. Its score per point is ``r² − 1`` and its
    Fisher information ``2`` per point, and the ρ²-weighted sigma rows put that 2 in H — which is
    what building H from the rows buys over the objective's own ``hess`` (docs/RATIONALE.md)."""
    n, rho = 4, 3.0
    r = np.sqrt([1.5, 1.5, 0.5, 0.5]) * [1, -1, 1, -1]   # Σ r² = n, as at the MLE of a shared sigma
    res = np.concatenate([r, np.full(n, rho)])
    sres = np.concatenate([-r, np.full(n, 1.0 / rho)])[:, None]   # ∂r/∂lnσ = −r, ∂ρ/∂lnσ = 1/ρ
    obj = _Objective(res, sres, blocks=[2 * n])
    scores = -(r ** 2) + 1.0                          # r·∂r + ρ·∂ρ per point
    H = float((r ** 2).sum() + n)                     # Σ r² + Σ ρ²·(1/ρ)² = 2n
    assert H == pytest.approx(2 * n)
    assert effective_parameter_count(_Problem(obj), np.zeros(1)) == pytest.approx(
        float((scores ** 2).sum()) / H)


def test_inner_solved_parameters_are_charged_one_unit_each():
    """A hierarchical fit's sigmas/scalings have no coordinate in the outer problem, so `k`
    counting them exceeds the FIM's dimension; the difference is added at one unit each."""
    X = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, -1.0]]
    assert _k_eff(X, [1, -1, 1, -1], k=5)[0] == pytest.approx(2.0 + 3)
    assert _k_eff(X, [1, -1, 1, -1], k=2)[0] == pytest.approx(2.0)


def test_the_sigma_residual_flag_is_on_for_the_evaluation_and_restored_after_it():
    """pyPESTO refuses residual sensitivities under an estimated sigma without the flag, and
    leaving it on would change every later objective call the optimizer makes."""
    _, obj = _k_eff([[1.0], [1.0]], [1, 1])
    assert obj.amici_model.flags_seen == [True]
    assert obj.amici_model.sigma_residuals is False
    k_eff, obj = _k_eff([[1.0], [1.0]], [1, 1], fail_res=True)
    assert k_eff is None, "an unevaluable objective is 'no count', never a fallback to k"
    assert obj.amici_model.sigma_residuals is False, "restored even when the evaluation raised"


def test_mismatched_row_counts_are_no_count_rather_than_a_wrong_one():
    """The pairing rests on the blocks summing to the residual vector and each having a sigma
    half; anything else means the layout is not the one assumed. An objective reporting NO
    conditions at all is one of those: it sums to an empty residual vector, so the parity check
    passes vacuously and only the emptiness itself refuses it."""
    obj = _Objective(np.ones(4), np.ones((4, 1)), blocks=[3])
    assert effective_parameter_count(_Problem(obj), np.zeros(1)) is None
    empty = _Objective(np.zeros(0), np.zeros((0, 1)), blocks=[])
    assert effective_parameter_count(_Problem(empty), np.zeros(1)) is None


# -- where it rides --------------------------------------------------------
def _fitsim(nll: float, *, n: int = 40, k: int = 3, k_eff: float | None = 2.5) -> dict:
    return {"status": "fit", "neg_log_likelihood": nll, "chi2": float(n),
            "n_data_points": n, "n_estimated": k, "k_eff": k_eff, "fitted_parameters": {},
            "records": [{"observable_id": "obs", "measurement": 1.0, "simulation": 1.0}]}


def test_no_result_the_agent_reads_carries_the_count():
    """The count is on the fit PAYLOAD only: neither the block every fit-bearing tool result is
    built from nor the score table `get_fit_summary` shares with it copies it
    (docs/RATIONALE.md)."""
    fs = _fitsim(100.0)
    assert fs["k_eff"] == 2.5
    assert "k_eff" not in fitting.fit_result_block(fs, improvement=None)
    assert "k_eff" not in fitting.goodness_of_fit(100.0, 40.0, 3, 40)


def test_a_snapshot_keeps_the_count_out_of_the_meta_and_in_the_stored_simulation(session):
    """`list_versions` returns the meta, so it stays out; the stored simulation is what the harness
    judge reads back through `get_version`. Exercised through `_snapshot_version` itself."""
    version = session._snapshot_version(meta={}, sim=_fitsim(100.0))
    assert "k_eff" not in session.provenance.get_meta(version)
    assert session.provenance.get_version(version)["simulation"]["k_eff"] == 2.5


def test_the_stored_simulation_persists_the_count():
    """A later process (a resume, a restore) cannot recompute it without a build, so the stored
    blob has to carry it — and carry `None` faithfully for a fit that produced none."""
    assert fitting.build_sim_payload_for_version(_fitsim(100.0))["k_eff"] == 2.5
    assert fitting.build_sim_payload_for_version(_fitsim(100.0, k_eff=None))["k_eff"] is None


def test_a_hierarchical_objective_is_evaluated_through_call_amici_twice():
    """pyPESTO's hierarchical objective raises in residual mode on its own path, so the relative
    calculator's `call_amici_twice` is driven directly and its full-width `sres` cut to the free
    columns; the inner parameter, not an optimisation coordinate there, is charged its unit."""
    X = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, -1.0]])
    res = np.concatenate([[1, -1, 1, -1], np.zeros(4)])
    # a third column for the fixed `A_init`, which the post-processing must drop
    sres_full = np.vstack([np.column_stack([X, np.full(4, 7.0)]), np.zeros((4, 3))])
    rel = _Relative(res, sres_full, blocks=[8])
    obj = _HierarchicalObjective(rel, n_fixed=1)
    problem = _HierarchicalProblem(obj, x_free_indices=[0, 1])

    k_eff = effective_parameter_count(problem, np.zeros(2), k=3)
    assert k_eff == pytest.approx(2.0 + 1), "outer trace over the two free columns, plus one inner"
    call = rel.calls[0]
    assert call["x_dct"] == {"k1": 0.0, "k2": 0.0, "A_init": 0.0}, "the free point, expanded"
    assert (call["amici_model"], call["edatas"], call["parameter_mapping"]) == (
        obj.amici_model, "edatas", "mapping"), "the collector's own arguments, passed through"
    assert obj.amici_model.flags_seen == [] and obj.amici_model.sigma_residuals is False
